"""Long-context rollout generation with alternating assistant and user turns."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset

from src.common.conversation_runtime import (
    format_progress_bar,
    message_append_id,
    now_iso,
)
from src.datasets import (
    export_dataset,
    get_run_paths,
    ingest_source_dataset,
    init_run,
    load_samples,
    materialize_canonical_samples,
    record_stage_event,
    register_stage_fingerprint,
    register_system_prompt,
    write_inference_result,
    write_message_append,
)
from src.datasets.io import read_jsonl_tolerant
from src.datasets.loaders import load_dataset_from_config
from src.datasets.schema import StageEventRecord
from src.inference import InferenceConfig
from src.inference.providers import get_provider
from src.inference.providers.base import InferenceProvider, PromptInput, TokenUsage
from src.inference.providers.remote_base import AsyncInferenceProvider
from src.utils import setup_logging

from .config import (
    RolloutGenerationConfig,
    RolloutGenerationResult,
    UserSimulatorConfig,
)
from .gpu_executor import GpuBatchExecutor
from .prompts import (
    get_user_simulator_instruction,
    render_user_simulator_single_turn_prompt,
)

PhaseKey = tuple[str, str, int]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _assistant_turn_count_sample(sample) -> int:
    return sum(1 for m in sample.messages if m.role == "assistant")


def _assistant_turn_count_dicts(messages: list[dict[str, Any]]) -> int:
    return sum(1 for m in messages if m.get("role") == "assistant")


def _phase_key(sample_id: str, phase: str, turn_index: int) -> PhaseKey:
    return (sample_id, phase, turn_index)


def _event_id(*parts: str) -> str:
    text = ":".join(parts)
    return f"evt_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]}"


def _record_rollout_event(
    run_dir: Path,
    *,
    event_type: str,
    sample_id: str,
    payload: dict[str, Any],
) -> None:
    record_stage_event(
        run_dir,
        StageEventRecord(
            event_id=_event_id(event_type, sample_id, now_iso()),
            stage="rollout_generation",
            event_type=event_type,
            sample_id=sample_id,
            created_at=now_iso(),
            payload=payload,
        ),
        lightweight=True,
    )


def _record_terminal_failure(
    run_dir: Path,
    *,
    sample_id: str,
    phase: str,
    turn_index: int,
    attempt_no: int,
    reason: str,
) -> None:
    _record_rollout_event(
        run_dir,
        event_type="terminal_failure",
        sample_id=sample_id,
        payload={
            "phase": phase,
            "turn_index": turn_index,
            "attempt_no": attempt_no,
            "reason": reason,
        },
    )


def _record_invalid_state(run_dir: Path, sample_id: str, reason: str) -> None:
    _record_rollout_event(
        run_dir,
        event_type="invalid_state",
        sample_id=sample_id,
        payload={"reason": reason},
    )


def _load_rollout_resume_state(run_dir: Path) -> tuple[dict[PhaseKey, int], set[str]]:
    """Load attempt counters and terminal sample IDs from stage events."""
    stage_events_path = get_run_paths(run_dir)["stage_events"]
    rows, _ = read_jsonl_tolerant(stage_events_path)

    attempts: dict[PhaseKey, int] = {}
    terminal_samples: set[str] = set()

    for row in rows:
        if row.get("stage") != "rollout_generation":
            continue
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str):
            continue

        event_type = row.get("event_type")
        payload = row.get("payload", {})
        if not isinstance(payload, dict):
            payload = {}

        if event_type in {"assistant_attempt", "user_attempt"}:
            phase = "assistant" if event_type == "assistant_attempt" else "user"
            turn_index = payload.get("turn_index")
            attempt_no = payload.get("attempt_no")
            if not isinstance(turn_index, int) or not isinstance(attempt_no, int):
                continue
            key = _phase_key(sample_id, phase, turn_index)
            attempts[key] = max(attempts.get(key, 0), attempt_no)
        elif event_type == "terminal_failure":
            terminal_samples.add(sample_id)

    return attempts, terminal_samples


def _apply_terminal_retry_policy(
    attempts_by_phase: dict[PhaseKey, int],
    terminal_samples: set[str],
    retry_terminal_sample_ids: list[str],
) -> tuple[dict[PhaseKey, int], set[str], set[str]]:
    """Clear terminal state and old attempt counters for selected samples.

    Args:
        attempts_by_phase: Resume-state attempt counters keyed by sample/phase/turn.
        terminal_samples: Samples already marked terminal in prior events.
        retry_terminal_sample_ids: Sample IDs to retry despite prior terminal state.

    Returns:
        Tuple of (filtered_attempts, filtered_terminal_samples, retried_samples).
    """
    retry_set = set(retry_terminal_sample_ids)
    if not retry_set:
        return attempts_by_phase, terminal_samples, set()

    filtered_attempts = {
        key: attempt_no
        for key, attempt_no in attempts_by_phase.items()
        if key[0] not in retry_set
    }
    filtered_terminal_samples = {
        sample_id for sample_id in terminal_samples if sample_id not in retry_set
    }
    retried_samples = retry_set & set(terminal_samples)
    return filtered_attempts, filtered_terminal_samples, retried_samples


def _build_user_simulator_inference_config(
    config: UserSimulatorConfig,
) -> InferenceConfig:
    return InferenceConfig(
        model=config.model,
        provider=config.provider,
        generation=config.generation,
        max_concurrent=config.max_concurrent,
        timeout=config.timeout,
        retry=config.retry,
        continue_on_error=False,
        log_failures=True,
        local=config.local,
        openai=config.openai,
        openrouter=config.openrouter,
        anthropic=config.anthropic,
    )


def _system_prompt_hash(prompt: str | None) -> str | None:
    """SHA-256 hash (first 16 chars) of the system prompt, or None."""
    if not prompt:
        return None
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _build_prompt_with_system(
    messages: list[dict[str, Any]], system_prompt: str | None
) -> list[dict[str, str]]:
    """Build assistant prompt: strip existing system messages, prepend current system prompt."""
    prompt = [
        {"role": m["role"], "content": m["content"]}
        for m in messages
        if m.get("role") != "system"
    ]
    if system_prompt:
        prompt.insert(0, {"role": "system", "content": system_prompt})
    return prompt


def _resolve_effective_system_prompt(
    messages: list[dict[str, Any]], fallback: str | None
) -> str | None:
    """Pick a per-sample system prompt if present; otherwise fall back.

    The materialized canonical sample may embed a per-sample system message
    as ``messages[0]`` with ``role == "system"``.  When present, it takes
    precedence over the global ``config.system_prompt`` so each scenario
    can carry its own deployment-style sysprompt (see
    ``datasets/scenarios/v2.json``).
    """
    for m in messages:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            return m["content"]
    return fallback


def _flip_message_roles(
    messages: list[dict[str, str]],
    initial_message: str | None = None,
) -> list[dict[str, str]]:
    """Swap user↔assistant roles and optionally prepend a filler user message."""
    _ROLE_SWAP = {"user": "assistant", "assistant": "user"}
    flipped = [{**m, "role": _ROLE_SWAP.get(m["role"], m["role"])} for m in messages]
    if initial_message:
        flipped.insert(0, {"role": "user", "content": initial_message})
    return flipped


def _flip_interlocutor(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Swap user↔assistant roles, ensuring the sequence starts with "user".

    Standard chat-completion APIs expect the first non-system message to be
    "user" and the model generates the next "assistant" turn.  In our
    interlocutor convention the test model's responses become "user" messages
    and the user-sim's outputs become "assistant" messages.

    If the conversation starts with a user-sim message (common when the
    user-sim generates the opening), the flipped sequence would start with
    "assistant".  This function detects that case and drops the leading
    user-sim message from the context — the system prompt already contains
    the scenario, so the user-sim doesn't need to see its own opening to
    stay in character.
    """
    _ROLE_SWAP = {"user": "assistant", "assistant": "user"}
    flipped = [{**m, "role": _ROLE_SWAP.get(m["role"], m["role"])} for m in messages]
    # Ensure the sequence starts with "user" (the test model's side).
    # If it starts with "assistant" (the user-sim's own prior message),
    # drop it — the scenario in the system prompt provides enough context.
    if flipped and flipped[0]["role"] == "assistant":
        flipped = flipped[1:]
    return flipped


def _truncate_to_recent_turns(
    messages: list[dict[str, str]],
    max_turns: int,
) -> tuple[list[dict[str, str]], int]:
    """Keep only the last *max_turns* exchange pairs from *messages*.

    A "turn" is one user+assistant message pair.  If the total number of
    messages exceeds ``2 * max_turns``, older messages are dropped from the
    front.  An odd trailing message (the most recent) is always kept.

    Returns:
        (truncated_messages, total_turns) — *total_turns* is the number of
        complete exchange pairs in the original list (before truncation),
        useful for telling the model how far into the conversation we are.
    """
    total_turns = len(messages) // 2
    max_msgs = 2 * max_turns
    if len(messages) <= max_msgs:
        return messages, total_turns
    return messages[-max_msgs:], total_turns


def _build_user_prompt_from_messages(
    messages: list[dict[str, str]],
    *,
    prompt_template: str,
    prompt_format: str,
    flip_roles: bool = False,
    initial_flipped_message: str | None = None,
    flip_mode: str = "none",
    turn_reminder: str | None = None,
    max_context_turns: int | None = None,
) -> str | list[dict[str, str]]:
    # Apply role manipulation.  flip_mode takes precedence over flip_roles.
    if flip_mode == "interlocutor":
        messages = _flip_interlocutor(messages)
    elif flip_mode == "swap_roles" or flip_roles:
        messages = _flip_message_roles(messages, initial_flipped_message)

    # Truncate to recent turns if configured.
    context_note = ""
    if max_context_turns is not None and messages:
        messages, total_turns = _truncate_to_recent_turns(
            messages,
            max_context_turns,
        )
        visible_turns = len(messages) // 2
        if visible_turns < total_turns:
            start_turn = total_turns - visible_turns + 1
            context_note = (
                f"\n\n[This conversation has been going for {total_turns} "
                f"exchanges. You are seeing exchanges {start_turn}\u2013"
                f"{total_turns}. Continue naturally from where the "
                f"conversation is now.]"
            )

    if prompt_format == "single_turn_text":
        return render_user_simulator_single_turn_prompt(prompt_template, messages)
    instruction = get_user_simulator_instruction(prompt_template)
    if context_note:
        instruction += context_note
    result = [{"role": "system", "content": instruction}, *messages]
    if turn_reminder and messages:
        result.append({"role": "user", "content": turn_reminder})
    return result


# ── Progress tracking ──────────────────────────────────────────────────────────


@dataclasses.dataclass
class _ProgressTracker:
    total_conversations: int
    total_assistant_turns: int
    conversations_completed: int = 0
    conversations_failed: int = 0
    assistant_turns_completed: int = 0
    user_turns_completed: int = 0


def _log_progress(logger: logging.Logger, tracker: _ProgressTracker) -> None:
    logger.info(
        "Progress | convs %s %d/%d | asst turns %s %d/%d | user turns %d | failed %d",
        format_progress_bar(
            tracker.conversations_completed, tracker.total_conversations
        ),
        tracker.conversations_completed,
        tracker.total_conversations,
        format_progress_bar(
            tracker.assistant_turns_completed, tracker.total_assistant_turns
        ),
        tracker.assistant_turns_completed,
        tracker.total_assistant_turns,
        tracker.user_turns_completed,
        tracker.conversations_failed,
    )


async def _progress_reporter_async(
    logger: logging.Logger,
    tracker: _ProgressTracker,
    stop_event: asyncio.Event,
    interval: float = 10.0,
) -> None:
    """Periodically log pipeline progress until stop_event is set."""
    while True:
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            break  # stop_event set
        except asyncio.TimeoutError:
            _log_progress(logger, tracker)
    _log_progress(logger, tracker)  # final snapshot


class _SinglePromptExecutor:
    """Semaphore-limited single-prompt executor for rollout turns.

    This keeps the event loop responsive while enforcing a shared global
    concurrency limit across many per-conversation coroutines. It is used for
    per-turn generation paths where batching is not beneficial or not available.
    In particular, remote providers' built-in async batching helpers create a
    new semaphore per batch call and therefore do not limit concurrency across
    many separate one-prompt invocations.
    """

    def __init__(self, provider: InferenceProvider, *, max_concurrent: int) -> None:
        self._provider = provider
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent))

    async def generate(
        self, prompt: PromptInput
    ) -> tuple[str, TokenUsage | None, str | None]:
        """Generate one response under a shared concurrency limit."""
        async with self._semaphore:
            try:
                (
                    responses,
                    usages,
                    _,
                ) = await self._provider.generate_batch_with_details_async(
                    [prompt], num_responses=1
                )
            except Exception as exc:  # noqa: BLE001
                return "", None, str(exc)

        text = responses[0] if responses else ""
        usage = usages[0] if usages else None
        return text, usage, None


class _GpuExecutorAdapter:
    """Adapter that exposes :class:`GpuBatchExecutor` via the same interface."""

    def __init__(self, executor: GpuBatchExecutor) -> None:
        self._executor = executor

    async def generate(
        self, prompt: PromptInput
    ) -> tuple[str, TokenUsage | None, str | None]:
        text, error = await self._executor.generate(prompt)
        return text, None, error


# ── Per-conversation async coroutine ───────────────────────────────────────────


async def _generate_user_turn_async(
    *,
    turn_index: int,
    sample_id: str,
    messages: list[dict[str, Any]],
    config: RolloutGenerationConfig,
    user_runner: _SinglePromptExecutor,
    user_config: InferenceConfig,
    run_dir: Path,
    attempts_by_phase: dict[PhaseKey, int],
    terminal_samples: set[str],
    progress: _ProgressTracker,
    logger: logging.Logger,
) -> bool:
    """Generate a single user simulator turn and append it to messages.

    Returns True if the turn was generated successfully, False if all attempts
    failed (the sample is then marked terminal).
    """
    user_max = config.failure_policy.user_max_attempts_per_turn
    user_key = _phase_key(sample_id, "user", turn_index)
    user_base_attempt = attempts_by_phase.get(user_key, 0)
    parent_user_message_id = messages[-1].get("message_id")

    if config.prompt_template_per_sample:
        if sample_id not in config.prompt_template_per_sample:
            raise KeyError(
                f"sample_id {sample_id!r} not found in prompt_template_per_sample. "
                "Ensure all samples are assigned a template before calling run_rollout_generation."
            )
        effective_template = config.prompt_template_per_sample[sample_id]
    else:
        effective_template = config.user_simulator.prompt_template

    # Strip the target's system message before handing history to the
    # user-simulator.  With per-sample target system prompts, this guards
    # against leaking the target's deployment sysprompt into the user-sim's
    # flipped view (where it would be seen as the simulator's own prior
    # turn).
    user_prompt = _build_user_prompt_from_messages(
        [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") != "system"
        ],
        prompt_template=effective_template,
        prompt_format=config.user_simulator.prompt_format,
        flip_roles=config.user_simulator.flip_roles_in_prompt,
        initial_flipped_message=config.user_simulator.initial_message_in_flipped_view,
        flip_mode=config.user_simulator.flip_mode,
        turn_reminder=config.user_simulator.turn_reminder,
        max_context_turns=config.user_simulator.max_context_turns,
    )

    for user_attempt in range(user_base_attempt + 1, user_base_attempt + user_max + 1):
        user_response = ""
        user_usage: TokenUsage | None = None
        user_error: str | None = None
        try:
            user_response, user_usage, user_error = await user_runner.generate(
                user_prompt
            )
        except Exception as exc:  # noqa: BLE001
            user_error = str(exc)

        u_success = bool(user_response.strip()) and user_error is None
        u_error = None if u_success else (user_error or "empty_response")

        attempts_by_phase[user_key] = user_attempt
        _record_rollout_event(
            run_dir,
            event_type="user_attempt",
            sample_id=sample_id,
            payload={
                "phase": "user",
                "turn_index": turn_index,
                "attempt_no": user_attempt,
                "status": "success" if u_success else "failed",
                "error": u_error,
                "provider": user_config.provider,
                "model": user_config.model,
                "token_usage": user_usage or {},
            },
        )

        if u_success:
            user_message_id = message_append_id(sample_id, "user", turn_index)
            write_message_append(
                run_dir,
                sample_id,
                {
                    "message_id": user_message_id,
                    "role": "user",
                    "content": user_response,
                    "editable": True,
                    "message_metadata": {
                        "turn_index": turn_index,
                        "source_stage": "rollout_user_simulator",
                        "provider": user_config.provider,
                        "model": user_config.model,
                        "token_usage": user_usage or {},
                        "parent_message_id": parent_user_message_id,
                        "phase_attempt_no": user_attempt,
                        "system_prompt_hash": _system_prompt_hash(
                            get_user_simulator_instruction(effective_template)
                        ),
                        "user_prompt_template": effective_template,
                    },
                },
                materialize=False,
            )
            messages.append(
                {
                    "role": "user",
                    "content": user_response,
                    "message_id": user_message_id,
                }
            )
            progress.user_turns_completed += 1
            return True

        logger.warning(
            "User turn failed (sample=%s turn=%d attempt=%d): %s",
            sample_id,
            turn_index,
            user_attempt,
            u_error,
        )

    terminal_samples.add(sample_id)
    progress.conversations_failed += 1
    progress.conversations_completed += 1
    _record_terminal_failure(
        run_dir,
        sample_id=sample_id,
        phase="user",
        turn_index=turn_index,
        attempt_no=user_base_attempt + user_max,
        reason="user_max_attempts_exceeded",
    )
    return False


async def _run_conversation_async(
    sample_id: str,
    messages: list[dict[str, Any]],
    config: RolloutGenerationConfig,
    assistant_runner: _SinglePromptExecutor | _GpuExecutorAdapter,
    user_runner: _SinglePromptExecutor,
    user_config: InferenceConfig,
    run_dir: Path,
    attempts_by_phase: dict[PhaseKey, int],
    terminal_samples: set[str],
    assistant_model: str,
    assistant_provider_name: str,
    progress: _ProgressTracker,
    logger: logging.Logger,
) -> None:
    """Run one conversation to completion, alternating assistant and user turns.

    Maintains message history in-memory. Each turn's result is written to disk
    immediately (materialize=False); a single materialize call at the end of the
    pipeline assembles the final samples.
    """
    start_turn = _assistant_turn_count_dicts(messages)

    # ── Resume-safety: recover a lost user turn ───────────────────────────────
    # If the process was interrupted after an assistant turn was written to disk
    # but before the following user turn was attempted, the materialized messages
    # will end on an assistant turn.  Resuming naively would generate the next
    # assistant turn with the previous assistant as its parent, producing
    # consecutive assistant turns in the stored history.  Detect this and
    # generate the missing user turn first.
    if start_turn > 0 and messages and messages[-1].get("role") == "assistant":
        missing_turn_index = start_turn - 1
        logger.warning(
            "Resume-safety: sample=%s ends on assistant turn %d — user turn %d was "
            "lost in a prior interrupted run. Generating the missing user turn now.",
            sample_id,
            missing_turn_index,
            missing_turn_index,
        )
        if not await _generate_user_turn_async(
            turn_index=missing_turn_index,
            sample_id=sample_id,
            messages=messages,
            config=config,
            user_runner=user_runner,
            user_config=user_config,
            run_dir=run_dir,
            attempts_by_phase=attempts_by_phase,
            terminal_samples=terminal_samples,
            progress=progress,
            logger=logger,
        ):
            return

    # ── Optional: user sim generates the opening message ─────────────────────
    # When user_sim_generates_opening=True and we're at the very start of a
    # conversation (only the seed question exists), call the user sim to
    # rephrase the seed into a natural opening in its archetype voice.  The
    # seed question is still available to the user sim via the {SEED}
    # placeholder in its system prompt.
    # A per-sample system prompt (e.g. scenario target_system_prompt) shows
    # up as messages[0] with role="system".  Detect the opening condition on
    # the non-system tail so the user-sim still opens on per-sample sysprompt
    # samples.
    _non_system_msgs = [m for m in messages if m.get("role") != "system"]
    if (
        config.user_sim_generates_opening
        and start_turn == 0
        and len(_non_system_msgs) == 1
        and _non_system_msgs[0].get("role") == "user"
    ):
        opening_key = _phase_key(sample_id, "user_opening", 0)
        opening_base_attempt = attempts_by_phase.get(opening_key, 0)
        if opening_base_attempt == 0:
            # Resolve per-sample template
            if config.prompt_template_per_sample:
                if sample_id not in config.prompt_template_per_sample:
                    raise KeyError(
                        f"sample_id {sample_id!r} not found in prompt_template_per_sample."
                    )
                opening_template = config.prompt_template_per_sample[sample_id]
            else:
                opening_template = config.user_simulator.prompt_template

            # Build a prompt with NO conversation history — just the system
            # instruction (which contains the seed topic and voice instructions).
            opening_prompt = _build_user_prompt_from_messages(
                [],  # empty history — user sim opens cold
                prompt_template=opening_template,
                prompt_format=config.user_simulator.prompt_format,
                flip_roles=config.user_simulator.flip_roles_in_prompt,
                initial_flipped_message=config.user_simulator.initial_message_in_flipped_view,
                flip_mode=config.user_simulator.flip_mode,
            )

            opening_success = False
            max_opening_attempts = config.failure_policy.user_max_attempts_per_turn
            for attempt in range(1, max_opening_attempts + 1):
                opening_text = ""
                opening_usage: TokenUsage | None = None
                opening_error: str | None = None
                opening_text, opening_usage, opening_error = await user_runner.generate(
                    opening_prompt
                )

                attempts_by_phase[opening_key] = attempt
                _record_rollout_event(
                    run_dir,
                    event_type="user_attempt",
                    sample_id=sample_id,
                    payload={
                        "phase": "user_opening",
                        "turn_index": 0,
                        "attempt_no": attempt,
                        "status": "success" if opening_text.strip() else "failed",
                        "error": opening_error
                        or ("empty_response" if not opening_text.strip() else None),
                        "provider": config.user_simulator.provider,
                        "model": config.user_simulator.model,
                        "token_usage": opening_usage or {},
                    },
                )

                if opening_text.strip() and opening_error is None:
                    # Replace the seed question with the user sim's opening.
                    # Keep source_stage="seed" so _sort_messages places this
                    # before the first assistant turn (within_turn=0).
                    # With per-sample system prompts, messages[0] can be a
                    # system message — find the first user message instead.
                    seed_idx = next(
                        (i for i, m in enumerate(messages) if m.get("role") == "user"),
                        0,
                    )
                    original_message_id = messages[seed_idx].get("message_id")
                    messages[seed_idx] = {
                        "role": "user",
                        "content": opening_text.strip(),
                        "message_id": original_message_id,
                        "message_metadata": {
                            "turn_index": 0,
                            "source_stage": "seed",
                            "generated_by": "user_simulator_opening",
                            "provider": config.user_simulator.provider,
                            "model": config.user_simulator.model,
                            "token_usage": opening_usage or {},
                            "user_prompt_template": opening_template,
                        },
                    }
                    # Persist the replacement — same message_id as the
                    # original seed message triggers upsert (replace) in
                    # materialize_canonical_samples.
                    write_message_append(
                        run_dir,
                        sample_id,
                        messages[seed_idx],
                        materialize=False,
                    )
                    opening_success = True
                    break

            if not opening_success:
                logger.warning(
                    "User sim opening failed for sample=%s after %d attempts; "
                    "falling back to raw seed question.",
                    sample_id,
                    max_opening_attempts,
                )

    for turn_index in range(start_turn, config.num_assistant_turns):
        # ── Assistant turn ────────────────────────────────────────────────────
        assistant_max = config.failure_policy.assistant_max_attempts_per_turn
        assistant_key = _phase_key(sample_id, "assistant", turn_index)
        base_attempt = attempts_by_phase.get(assistant_key, 0)
        parent_message_id = messages[-1].get("message_id") if messages else None
        # Per-sample system prompt (e.g. scenario target_system_prompt registered
        # via the canonical ingest as messages[0]) takes precedence over the
        # global config.system_prompt.  Without this resolution, the engine
        # would silently drop per-sample sysprompts recorded in the manifest.
        effective_system_prompt = _resolve_effective_system_prompt(
            messages, config.system_prompt
        )
        prompt = _build_prompt_with_system(messages, effective_system_prompt)

        assistant_success = False
        for phase_attempt in range(base_attempt + 1, base_attempt + assistant_max + 1):
            response_text, assistant_usage, gen_error = await assistant_runner.generate(
                prompt
            )
            success = bool(response_text.strip()) and gen_error is None
            error = None if success else (gen_error or "empty_response")
            token_usage = assistant_usage or {}

            attempts_by_phase[assistant_key] = phase_attempt
            _record_rollout_event(
                run_dir,
                event_type="assistant_attempt",
                sample_id=sample_id,
                payload={
                    "phase": "assistant",
                    "turn_index": turn_index,
                    "attempt_no": phase_attempt,
                    "status": "success" if success else "failed",
                    "error": error,
                    "provider": assistant_provider_name,
                    "model": assistant_model,
                    "token_usage": token_usage,
                },
            )

            assistant_message_id = message_append_id(sample_id, "assistant", turn_index)
            inference_payload: dict[str, Any] = {
                "status": "success" if success else "failed",
                "model": assistant_model,
                "provider": assistant_provider_name,
                "attempt_no": phase_attempt,
                "token_usage": token_usage,
                "started_at": now_iso(),
                "completed_at": now_iso(),
                "error": error,
            }
            if success:
                inference_payload.update(
                    {
                        "assistant_message_id": assistant_message_id,
                        "assistant_completion": response_text,
                        "assistant_full": response_text,
                        "assistant_message_metadata": {
                            "turn_index": turn_index,
                            "source_stage": "rollout_assistant",
                            "provider": assistant_provider_name,
                            "model": assistant_model,
                            "token_usage": token_usage,
                            "parent_message_id": parent_message_id,
                            "phase_attempt_no": phase_attempt,
                            "system_prompt_hash": _system_prompt_hash(
                                effective_system_prompt
                            ),
                        },
                    }
                )
            write_inference_result(
                run_dir, sample_id, inference_payload, materialize=False
            )

            if success:
                messages.append(
                    {
                        "role": "assistant",
                        "content": response_text,
                        "message_id": assistant_message_id,
                    }
                )
                progress.assistant_turns_completed += 1
                assistant_success = True
                break

            logger.warning(
                "Assistant turn failed (sample=%s turn=%d attempt=%d): %s",
                sample_id,
                turn_index,
                phase_attempt,
                error,
            )

        if not assistant_success:
            terminal_samples.add(sample_id)
            progress.conversations_failed += 1
            progress.conversations_completed += 1
            _record_terminal_failure(
                run_dir,
                sample_id=sample_id,
                phase="assistant",
                turn_index=turn_index,
                attempt_no=base_attempt + assistant_max,
                reason="assistant_max_attempts_exceeded",
            )
            return

        # Optionally skip the user turn after the final assistant turn.
        if config.skip_final_user_turn and turn_index + 1 >= config.num_assistant_turns:
            break

        # ── User simulator turn ───────────────────────────────────────────────
        if not await _generate_user_turn_async(
            turn_index=turn_index,
            sample_id=sample_id,
            messages=messages,
            config=config,
            user_runner=user_runner,
            user_config=user_config,
            run_dir=run_dir,
            attempts_by_phase=attempts_by_phase,
            terminal_samples=terminal_samples,
            progress=progress,
            logger=logger,
        ):
            return

    progress.conversations_completed += 1


# ── Async pipeline scheduler ───────────────────────────────────────────────────


async def _run_rollout_pipeline_async(
    config: RolloutGenerationConfig,
    samples: list,
    assistant_config: InferenceConfig,
    assistant_provider: InferenceProvider | None,
    user_provider: InferenceProvider,
    user_config: InferenceConfig,
    run_dir: Path,
    attempts_by_phase: dict[PhaseKey, int],
    terminal_samples: set[str],
    logger: logging.Logger,
    gpu_executor: GpuBatchExecutor | None = None,
) -> None:
    """Pipelined async scheduler: assistant and user turns overlap.

    Each conversation runs as an independent coroutine. Assistant turns use a
    shared GPU batch executor for local/vLLM providers, or a shared
    semaphore-limited async path for remote providers. User turns likewise run
    through a shared semaphore-limited executor. In both cases the event loop
    remains free to service other in-flight work while requests are running.

    Args:
        gpu_executor: Optional shared executor. When provided, this function
            uses it instead of creating a new one, and does not stop it on
            exit (the caller owns its lifecycle). ``assistant_provider`` may
            be ``None`` when an external executor is supplied.
    """

    executor_task: asyncio.Task[None] | None = None
    if gpu_executor is not None:
        executor = gpu_executor
        assistant_runner = _GpuExecutorAdapter(executor)
        assistant_mode = "shared_gpu_executor"
    elif isinstance(assistant_provider, AsyncInferenceProvider):
        executor = None
        assistant_runner = _SinglePromptExecutor(
            assistant_provider,
            max_concurrent=assistant_config.max_concurrent,
        )
        assistant_mode = (
            f"direct_async(max_concurrent={max(1, assistant_config.max_concurrent)})"
        )
    else:
        assert assistant_provider is not None, (
            "assistant_provider is required when gpu_executor is not provided"
        )
        batch_size = max(1, assistant_config.generation.batch_size)
        executor = GpuBatchExecutor(assistant_provider, batch_size=batch_size)
        executor_task = asyncio.create_task(executor.run())
        assistant_runner = _GpuExecutorAdapter(executor)
        assistant_mode = f"gpu_executor(batch_size={batch_size})"

    user_runner = _SinglePromptExecutor(
        user_provider,
        max_concurrent=user_config.max_concurrent,
    )

    pending = [
        sample
        for sample in samples
        if sample.sample_id not in terminal_samples
        and _assistant_turn_count_sample(sample) < config.num_assistant_turns
    ]

    # Seed progress from turns already completed in resumed samples.
    existing_assistant_turns = sum(_assistant_turn_count_sample(s) for s in pending)
    existing_user_turns = sum(
        sum(1 for m in s.messages if m.role == "user") for s in pending
    )
    total_assistant_turns = len(pending) * config.num_assistant_turns
    progress = _ProgressTracker(
        total_conversations=len(pending),
        total_assistant_turns=total_assistant_turns,
        assistant_turns_completed=existing_assistant_turns,
        user_turns_completed=existing_user_turns,
    )
    stop_event = asyncio.Event()

    logger.info(
        "Async pipeline: %d conversations, %d assistant turns total "
        "(%d conversations already done, %d assistant turns resumed) | "
        "assistant_mode=%s | user_max_concurrent=%d",
        len(pending),
        total_assistant_turns,
        len(samples) - len(pending),
        existing_assistant_turns,
        assistant_mode,
        max(1, user_config.max_concurrent),
    )

    conversation_tasks = [
        _run_conversation_async(
            sample_id=sample.sample_id,
            messages=[
                {
                    "role": m.role,
                    "content": m.content,
                    "message_id": getattr(m, "message_id", None),
                }
                for m in sample.messages
            ],
            config=config,
            assistant_runner=assistant_runner,
            user_runner=user_runner,
            user_config=user_config,
            run_dir=run_dir,
            attempts_by_phase=attempts_by_phase,
            terminal_samples=terminal_samples,
            assistant_model=assistant_config.model,
            assistant_provider_name=assistant_config.provider,
            progress=progress,
            logger=logger,
        )
        for sample in pending
    ]

    reporter_task = asyncio.create_task(
        _progress_reporter_async(logger, progress, stop_event, interval=10.0)
    )

    try:
        await asyncio.gather(*conversation_tasks)
    finally:
        stop_event.set()
        await reporter_task
        if executor_task is not None:
            assert executor is not None
            executor.stop()
            await executor_task
        providers_to_close = [user_provider]
        if assistant_provider is not None:
            providers_to_close.append(assistant_provider)
        for p in providers_to_close:
            c = getattr(p, "client", None)
            if c is not None and asyncio.iscoroutinefunction(getattr(c, "close", None)):
                await c.close()

    logger.info("Async pipeline complete.")


# ── Public entry point ─────────────────────────────────────────────────────────


async def run_rollout_generation_async(
    config: RolloutGenerationConfig,
    dataset: Dataset | None = None,
    gpu_executor: GpuBatchExecutor | None = None,
) -> tuple[Dataset, RolloutGenerationResult]:
    """Async version of :func:`run_rollout_generation`.

    When ``gpu_executor`` is provided, the pipeline feeds prompts into the
    shared executor instead of creating its own.  This allows multiple
    rollout generations to run concurrently within the same event loop,
    sharing a single GPU batch queue for better utilisation.

    Args:
        config: Rollout generation configuration.
        dataset: Optional pre-loaded dataset.
        gpu_executor: Optional shared :class:`GpuBatchExecutor`.  When
            provided, no local assistant provider is created — the executor
            already owns one.
    """
    logger = setup_logging()
    run_dir = Path(config.run_dir)

    if config.num_assistant_turns <= 0:
        raise ValueError("num_assistant_turns must be positive.")
    if config.num_rollouts_per_prompt <= 0:
        raise ValueError("num_rollouts_per_prompt must be positive.")
    if config.context_policy.mode != "full_history":
        raise NotImplementedError(
            "context_policy.mode='token_budget' is not implemented yet. Use mode='full_history'."
        )

    manifest = init_run(
        run_dir, base_config={"rollout_generation": config.model_dump(mode="json")}
    )
    if not (config.resume and manifest.stage_fingerprints.get("rollout_generation")):
        register_stage_fingerprint(
            run_dir, "rollout_generation", config.model_dump(mode="json")
        )
    register_system_prompt(run_dir, config.system_prompt)
    if not config.prompt_template_per_sample:
        user_sim_text = get_user_simulator_instruction(
            config.user_simulator.prompt_template
        )
        register_system_prompt(run_dir, user_sim_text)

    paths = get_run_paths(run_dir)
    if not (config.resume and paths["sample_inputs"].exists()):
        if dataset is None:
            dataset = load_dataset_from_config(config.dataset)
        source_info: dict[str, Any] = {
            "dataset_source": config.dataset.source,
            "dataset_name": config.dataset.name,
            "dataset_path": config.dataset.path,
            "dataset_split": config.dataset.split,
            "max_samples": config.dataset.max_samples,
        }
        ingest_source_dataset(
            dataset=dataset,
            source_info=source_info,
            system_prompt=config.system_prompt,
            run_dir=run_dir,
            overwrite=config.overwrite_output,
            responses_per_input=config.num_rollouts_per_prompt,
        )

    assistant_config = config.assistant_inference.model_copy(
        update={
            "run_dir": None,
            "output_path": None,
            "resume": False,
            "overwrite_output": False,
            "continue_on_error": False,
            "generation": config.assistant_inference.generation.model_copy(
                update={"num_responses_per_prompt": 1}
            ),
        }
    )
    user_config = _build_user_simulator_inference_config(
        config.user_simulator
    ).model_copy(
        update={
            "generation": config.user_simulator.generation.model_copy(
                update={"num_responses_per_prompt": 1}
            )
        }
    )

    # When using a shared executor, skip creating assistant_provider — the
    # executor already owns the sole provider that wraps the model.
    assistant_provider: InferenceProvider | None = None
    if gpu_executor is None:
        assistant_provider = get_provider(assistant_config.provider, assistant_config)
    user_provider = get_provider(user_config.provider, user_config)

    if config.resume:
        attempts_by_phase, terminal_samples = _load_rollout_resume_state(run_dir)
        attempts_by_phase, terminal_samples, retried_samples = (
            _apply_terminal_retry_policy(
                attempts_by_phase=attempts_by_phase,
                terminal_samples=terminal_samples,
                retry_terminal_sample_ids=config.retry_terminal_sample_ids,
            )
        )
        if config.retry_terminal_sample_ids:
            logger.info(
                "Retry-terminal mode enabled for %d sample(s); %d had prior terminal failures.",
                len(config.retry_terminal_sample_ids),
                len(retried_samples),
            )
    else:
        attempts_by_phase, terminal_samples = {}, set()

    materialize_canonical_samples(run_dir)
    samples = load_samples(run_dir)

    await _run_rollout_pipeline_async(
        config=config,
        samples=samples,
        assistant_config=assistant_config,
        assistant_provider=assistant_provider,
        user_provider=user_provider,
        user_config=user_config,
        run_dir=run_dir,
        attempts_by_phase=attempts_by_phase,
        terminal_samples=terminal_samples,
        logger=logger,
        gpu_executor=gpu_executor,
    )

    materialize_canonical_samples(run_dir)

    export_path = export_dataset(
        run_dir,
        profile="conversation_training",
        variant_name=config.transcript_variant,
    )
    trace_path = export_dataset(
        run_dir,
        profile="conversation_trace",
        variant_name=config.transcript_variant,
    )

    final_samples = load_samples(run_dir)
    completed = sum(
        1
        for s in final_samples
        if _assistant_turn_count_sample(s) >= config.num_assistant_turns
    )
    completed_turns = sum(_assistant_turn_count_sample(s) for s in final_samples)
    failed = len(final_samples) - completed

    result_dataset = load_dataset_from_config(
        config.dataset.model_copy(
            update={
                "source": "canonical",
                "path": str(run_dir),
            }
        )
    )

    result = RolloutGenerationResult(
        output_path=Path(export_path),
        num_conversations=len(final_samples),
        num_completed=completed,
        num_failed=failed,
        num_assistant_turns_target=config.num_assistant_turns,
        num_assistant_turns_completed=completed_turns,
        exports={
            "conversation_training": str(export_path),
            "conversation_trace": str(trace_path),
        },
    )

    logger.info(
        "Rollout generation complete: %d/%d complete, failed=%d.",
        completed,
        len(final_samples),
        failed,
    )
    return result_dataset, result


def run_rollout_generation(
    config: RolloutGenerationConfig,
    dataset: Dataset | None = None,
) -> tuple[Dataset, RolloutGenerationResult]:
    """Generate alternating assistant/user rollouts and export transcripts."""
    return asyncio.run(run_rollout_generation_async(config, dataset))
