"""Shared inference + judge runners with idempotent JSONL persistence.

Both Option 1 (persona grid) and Option 2 (WildJailbreak) drivers compose:

    1. Build a list of :class:`PromptSample` records.
    2. ``run_condition_inference(...)`` per condition  → writes responses JSONL.
    3. ``run_paper_judge(...)`` on harmful rows         → writes judgments JSONL.
    4. ``run_refusal_judge(...)`` on benign rows        → writes judgments JSONL.
    5. ``aggregate_and_plot(...)``                       → writes summary CSV + plot.

Each step is idempotent on its output JSONL: re-running picks up where it
stopped. Sample-id is a stable hash of the prompt content (set by the
caller), so re-builds with the same inputs yield the same ids.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src_dev.activation_capping.conditions import (
    CappingPreload,
    ConditionConfig,
    load_capped_model,
    release_capping,
    setup_capping_inference,
    setup_lora_soup_inference,
    setup_vanilla_inference,
    sort_conditions_for_safety,
)
from src_dev.common.lora_catalogue import HF_REPO, OCEAN_REGISTRY
from src_dev.inference.config import InferenceConfig
from src_dev.inference.providers import get_provider
from src_dev.persona_jailbreak_eval.config import (
    JailbreakEvalConfig,
    LoraComboCondition,
)
from src_dev.persona_jailbreak_eval.judge_base import JsonRubricJudge, JudgeOutcome
from src_dev.persona_jailbreak_eval.judge_paper import (
    PaperHarmfulnessJudge,
    is_harmful,
    is_refusal,
)
from src_dev.persona_jailbreak_eval.judge_refusal import (
    BinaryRefusalJudge,
    is_explicit_refusal,
    is_noncompliant,
)

logger = logging.getLogger(__name__)


# ── Sample schema ────────────────────────────────────────────────────────


@dataclass
class PromptSample:
    """One eval input: system prompt (optional) + user prompt + metadata."""

    id: str
    user_prompt: str
    system_prompt: str | None
    kind: str  # "harmful" | "benign"
    category: str
    behavior: str  # rubric input (paper judge only)
    action: str  # rubric input (paper judge only)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_messages(self) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if self.system_prompt:
            msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": self.user_prompt})
        return msgs


# ── Idempotent JSONL helpers ─────────────────────────────────────────────


def _load_completed_ids(path: Path, *, skip_errored: bool = False) -> set[str]:
    """Read a JSONL output and return the set of ``sample_id`` considered done.

    With ``skip_errored=True``, rows whose ``parse_error`` field is non-null
    are NOT counted as completed — useful for judge-output JSONLs so that
    transient JSON-parse failures get retried on re-run rather than baked
    into the cache forever.
    """
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if skip_errored and row.get("parse_error"):
                continue
            sid = row.get("sample_id") or row.get("id")
            if sid:
                completed.add(str(sid))
    return completed


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── Inference per condition ──────────────────────────────────────────────


async def _generate_messages_async(
    provider,
    inference_config: InferenceConfig,
    message_lists: list[list[dict[str, str]]],
) -> list[str]:
    gen = inference_config.generation
    responses, _, failed = await provider.generate_batch_with_metadata_async(
        message_lists,
        num_responses=1,
        max_new_tokens=gen.max_new_tokens,
        temperature=gen.temperature,
        top_p=gen.top_p,
        do_sample=gen.do_sample,
    )
    if failed:
        logger.warning("inference: %d/%d prompts failed", failed, len(message_lists))
    return responses


async def _generate_messages_logprobs_async(
    provider,
    message_lists: list[list[dict[str, str]]],
    *,
    prefill: str,
    top_logprobs: int,
    max_tokens: int,
    temperature: float = 1.0,
) -> list[dict]:
    """Single-token logprob generation with an optional forced assistant prefill.

    Appends ``prefill`` as a trailing partial assistant turn (the provider's
    chat template continues it rather than opening a new turn), mirroring the
    TRAIT logprob eval's ``logprob_multiple_choice`` solver. Only supported by
    providers exposing ``generate_batch_logprobs_async`` (vLLM).
    """
    if prefill:
        message_lists = [
            msgs + [{"role": "assistant", "content": prefill}] for msgs in message_lists
        ]
    return await provider.generate_batch_logprobs_async(
        message_lists,
        max_tokens=max_tokens,
        top_logprobs=top_logprobs,
        temperature=temperature,
    )


def _build_inference_for_condition(
    cfg: JailbreakEvalConfig,
    cond_cfg: ConditionConfig,
    condition: str,
    *,
    capped_preload: CappingPreload | None,
    soup_inference_by_name: dict[str, InferenceConfig],
) -> tuple[InferenceConfig, str]:
    """Return ``(inference_config, provider_kind)`` for a condition."""
    if condition == "vanilla":
        return setup_vanilla_inference(cond_cfg), "vllm"
    if condition == "activation_capping":
        if capped_preload is None:
            raise RuntimeError("activation_capping condition without capped_preload")
        return setup_capping_inference(cond_cfg, capped_preload), "local"
    if condition.startswith("lora_soup"):
        if condition not in soup_inference_by_name:
            raise KeyError(f"no baked LoRA soup for condition {condition!r}")
        return soup_inference_by_name[condition], "vllm"
    raise ValueError(f"unknown condition {condition!r}")


def run_condition_inference(
    cfg: JailbreakEvalConfig,
    condition: str,
    samples: list[PromptSample],
    output_path: Path,
    *,
    capped_preload: CappingPreload | None,
    soup_inference_by_name: dict[str, InferenceConfig],
    logprobs: bool = False,
    logprob_prefill: str = "",
    logprob_top_k: int = 20,
    logprob_max_tokens: int = 1,
    logprob_temperature: float = 1.0,
) -> int:
    """Run inference for one condition, idempotently appending to ``output_path``.

    With ``logprobs=True`` (vLLM-backed conditions only), generates
    ``logprob_max_tokens`` tokens with top-k logprobs — after appending
    ``logprob_prefill`` as a forced partial assistant turn — and persists the
    first generated token's ``token -> logprob`` dict in each row's
    ``top_logprobs`` field, alongside the (single-token) ``response`` text.

    Returns the number of newly generated responses (0 if everything was
    already cached on disk).
    """
    cond_cfg = cfg.condition_config()
    inference_config, provider_kind = _build_inference_for_condition(
        cfg, cond_cfg, condition,
        capped_preload=capped_preload,
        soup_inference_by_name=soup_inference_by_name,
    )
    if logprobs and provider_kind != "vllm":
        raise ValueError(
            f"logprob inference requested for condition {condition!r} but its "
            f"provider is {provider_kind!r}; only 'vllm' supports logprobs."
        )

    completed = _load_completed_ids(output_path)
    pending = [s for s in samples if s.id not in completed]
    if not pending:
        print(f"  [{condition}] all {len(samples)} samples already cached at {output_path.name}")
        return 0

    batch_size = max(1, inference_config.generation.batch_size)
    print(f"  [{condition}] generating {len(pending)} responses "
          f"({len(completed)} already cached, batch_size={batch_size}, "
          f"mode={'logprobs' if logprobs else 'text'})...")

    provider = get_provider(provider_kind, inference_config)
    generated_count = 0
    try:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            message_lists = [s.to_messages() for s in batch]
            if logprobs:
                lp_outputs = asyncio.run(_generate_messages_logprobs_async(
                    provider, message_lists,
                    prefill=logprob_prefill,
                    top_logprobs=logprob_top_k,
                    max_tokens=logprob_max_tokens,
                    temperature=logprob_temperature,
                ))
                responses = [out["text"] for out in lp_outputs]
                per_token = [out.get("logprobs_per_token") or [] for out in lp_outputs]
                top_logprobs_per_sample = [tok[0] if tok else {} for tok in per_token]
            else:
                responses = asyncio.run(_generate_messages_async(
                    provider, inference_config, message_lists,
                ))
                top_logprobs_per_sample = [None] * len(batch)

            rows: list[dict[str, Any]] = []
            for sample, response, top_lps in zip(batch, responses, top_logprobs_per_sample):
                row = {
                    "sample_id": sample.id,
                    "condition": condition,
                    "kind": sample.kind,
                    "category": sample.category,
                    "user_prompt": sample.user_prompt,
                    "system_prompt": sample.system_prompt,
                    "behavior": sample.behavior,
                    "action": sample.action,
                    "response": response or "",
                    "extras": sample.extras,
                }
                if logprobs:
                    row["top_logprobs"] = {k: round(v, 6) for k, v in (top_lps or {}).items()}
                    row["scoring_method"] = "logprob"
                    if logprob_max_tokens > 1:
                        idx = len(rows)
                        row["logprobs_per_token"] = [
                            {k: round(v, 6) for k, v in tok.items()}
                            for tok in per_token[idx]
                        ]
                rows.append(row)
            _append_jsonl(output_path, rows)
            generated_count += len(rows)
    finally:
        asyncio.run(provider.aclose())
    return generated_count


# ── Top-level inference orchestration ────────────────────────────────────


def _resolve_lora_soup_specs(combo: LoraComboCondition) -> list[tuple[str, float]]:
    specs: list[tuple[str, float]] = []
    for adapter_key, scale in combo.adapters:
        if adapter_key in OCEAN_REGISTRY:
            specs.append((OCEAN_REGISTRY[adapter_key].adapter_ref, scale))
            continue
        if "::" in adapter_key:
            specs.append((adapter_key, scale))
            continue
        specs.append((f"{HF_REPO}::{adapter_key}", scale))
    return specs


def run_all_conditions_inference(
    cfg: JailbreakEvalConfig,
    samples: list[PromptSample],
    *,
    output_dir: Path,
    logprobs: bool = False,
    logprob_prefill: str = "",
    logprob_top_k: int = 20,
    logprob_max_tokens: int = 1,
    logprob_temperature: float = 1.0,
) -> dict[str, Path]:
    """Run inference under every requested condition (vLLM first, HF last).

    Bakes any LoRA soups once, lazy-loads the capping model only when its
    turn arrives, releases it after.

    Returns a mapping ``condition → response_jsonl_path``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cond_cfg = cfg.condition_config()

    # Bake any LoRA-soup adapters that show up in cfg.lora_combos.
    soup_inference_by_name: dict[str, InferenceConfig] = {}
    for combo in cfg.lora_combos:
        if combo.name not in cfg.conditions:
            continue
        baked_dir = output_dir / "baked_lora_soups" / combo.name
        soup_inf, baked_path, soup_rank = setup_lora_soup_inference(
            cond_cfg,
            adapter_specs=_resolve_lora_soup_specs(combo),
            output_dir=baked_dir,
        )
        soup_inference_by_name[combo.name] = soup_inf
        print(f"  LoRA soup {combo.name!r} baked at {baked_path} (rank={soup_rank})")

    if "activation_capping" in cfg.conditions:
        if cfg.axis_path is None or cfg.capping_config_path is None:
            raise SystemExit(
                "activation_capping requested but cfg.axis_path / cfg.capping_config_path "
                "are not set. Point them at the existing drift-script artefacts."
            )

    response_paths: dict[str, Path] = {}
    capped_preload: CappingPreload | None = None
    try:
        for condition in sort_conditions_for_safety(cfg.conditions):
            if condition == "activation_capping" and capped_preload is None:
                capped_preload = load_capped_model(
                    cond_cfg,
                    axis_path=cfg.axis_path,  # type: ignore[arg-type]
                    capping_config_path=cfg.capping_config_path,  # type: ignore[arg-type]
                )
            out_path = output_dir / f"responses_{condition}.jsonl"
            run_condition_inference(
                cfg, condition, samples, out_path,
                capped_preload=capped_preload,
                soup_inference_by_name=soup_inference_by_name,
                logprobs=logprobs,
                logprob_prefill=logprob_prefill,
                logprob_top_k=logprob_top_k,
                logprob_max_tokens=logprob_max_tokens,
                logprob_temperature=logprob_temperature,
            )
            response_paths[condition] = out_path
    finally:
        release_capping(capped_preload)
    return response_paths


# ── Judge runners ────────────────────────────────────────────────────────


def _judge_inputs_from_response_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map a response JSONL row into the inputs the paper rubric expects."""
    return {
        "request": _format_request_for_judge(row),
        "response": row.get("response", ""),
        "behavior": row.get("behavior", ""),
        "action": row.get("action", ""),
    }


def _format_request_for_judge(row: dict[str, Any]) -> str:
    """Concatenate system + user prompt for the rubric's [request] slot."""
    sys_prompt = row.get("system_prompt") or ""
    user_prompt = row.get("user_prompt") or ""
    if sys_prompt:
        return f"[SYSTEM]\n{sys_prompt}\n\n[USER]\n{user_prompt}"
    return user_prompt


def _judge_outcome_to_row(
    sample_id: str,
    condition: str,
    kind: str,
    category: str,
    outcome: JudgeOutcome,
    *,
    is_harm: bool,
    is_ref: bool,
    is_explicit_ref: bool = False,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "condition": condition,
        "kind": kind,
        "category": category,
        "label": outcome.label,
        "is_harmful": is_harm,
        "is_refusal": is_ref,
        "is_explicit_refusal": is_explicit_ref,
        "analysis": outcome.analysis,
        "parse_error": outcome.parse_error,
    }


def _run_judge_pass(
    judge: JsonRubricJudge,
    response_rows: list[dict[str, Any]],
    output_path: Path,
    *,
    binarize_harmful,
    binarize_refusal,
    label: str,
) -> int:
    """Generic judge runner: idempotent JSONL append, returns new-judgement count.

    Parse-errored judgments are retried on re-run (``skip_errored=True``).
    Inference-failed responses (empty string) are still passed to the judge —
    the rubric will rate them ``nonsensical`` or ``out_of_context``, which is
    the right answer.
    """
    completed = _load_completed_ids(output_path, skip_errored=True)
    pending = [r for r in response_rows if r["sample_id"] not in completed]
    if not pending:
        print(f"  [{label}] all {len(response_rows)} rows already judged at {output_path.name}")
        return 0
    print(f"  [{label}] judging {len(pending)} rows "
          f"({len(completed)} already cached)...")
    inputs = [_judge_inputs_from_response_row(r) for r in pending]
    outcomes = asyncio.run(judge.judge_batch(inputs))
    rows: list[dict[str, Any]] = []
    for src, outcome in zip(pending, outcomes):
        rows.append(_judge_outcome_to_row(
            sample_id=src["sample_id"],
            condition=src["condition"],
            kind=src["kind"],
            category=src.get("category", "unknown"),
            outcome=outcome,
            is_harm=binarize_harmful(outcome.label),
            is_ref=binarize_refusal(outcome.label),
            is_explicit_ref=is_explicit_refusal(outcome.label),
        ))
    _append_jsonl(output_path, rows)
    return len(rows)


def _read_response_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def run_paper_judge_on_responses(
    cfg: JailbreakEvalConfig,
    response_path: Path,
    output_path: Path,
) -> int:
    """Judge all *harmful*-kind rows in a response JSONL with the D.2.2 rubric."""
    rows = [r for r in _read_response_jsonl(response_path) if r.get("kind") == "harmful"]
    if not rows:
        return 0
    judge = PaperHarmfulnessJudge(cfg.judge)
    return _run_judge_pass(
        judge, rows, output_path,
        binarize_harmful=is_harmful,
        binarize_refusal=is_refusal,
        label=f"paper-judge:{response_path.name}",
    )


def run_refusal_judge_on_responses(
    cfg: JailbreakEvalConfig,
    response_path: Path,
    output_path: Path,
) -> int:
    """Judge all *benign*-kind rows in a response JSONL with the refusal judge."""
    rows = [r for r in _read_response_jsonl(response_path) if r.get("kind") == "benign"]
    if not rows:
        return 0
    judge = BinaryRefusalJudge(cfg.refusal_judge)
    return _run_judge_pass(
        judge, rows, output_path,
        binarize_harmful=lambda _label: False,  # benign rows are never "harmful"
        binarize_refusal=is_noncompliant,
        label=f"refusal-judge:{response_path.name}",
    )


def run_judges_on_all_conditions(
    cfg: JailbreakEvalConfig,
    response_paths: dict[str, Path],
    output_dir: Path,
) -> dict[str, Path]:
    """Run both judges over every condition's responses; return per-condition
    judgment-jsonl paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    judgment_paths: dict[str, Path] = {}
    for condition, response_path in response_paths.items():
        judgment_path = output_dir / f"judgments_{condition}.jsonl"
        run_paper_judge_on_responses(cfg, response_path, judgment_path)
        run_refusal_judge_on_responses(cfg, response_path, judgment_path)
        judgment_paths[condition] = judgment_path
    return judgment_paths


__all__ = [
    "PromptSample",
    "run_condition_inference",
    "run_all_conditions_inference",
    "run_paper_judge_on_responses",
    "run_refusal_judge_on_responses",
    "run_judges_on_all_conditions",
]
