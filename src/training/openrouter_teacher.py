"""OpenRouter teacher-generation path for the OCT paired-DPO pipeline.

The canonical paired-DPO method generates the in-character "chosen" teacher
responses by calling a strong model **via the OpenRouter API** rather than
loading it locally through vLLM. This module owns that path end-to-end:

* the teacher system-prompt / think-prefill templates (copied from upstream
  ``character.distillation.teacher`` so OpenRouter calls render the same prompt);
* OpenRouter chat / raw-completion clients with graceful kwarg fallbacks
  (some models reject ``temperature`` or want ``max_completion_tokens``);
* **question expansion** — optionally grow each trait's seed questions to a
  target count via an OpenRouter model, with on-disk checkpointing and debug
  logs (reachable from ``install_constitution(expand_questions=True)``);
* ``run_teacher_openrouter`` — the drop-in replacement for OCT's local
  ``teacher.main()`` that writes the distillation JSONL (``prompt`` +
  ``response`` columns) for the constitution.

This module is import-safe without GPUs: it only touches ``transformers``
tokenizers lazily and the ``openai`` async client. It does NOT import the OCT
runtime patches — callers that need them must initialise the runtime first.

WHERE this is used: ``src.training.oct_adapter`` calls ``run_teacher_openrouter``
inside the distillation stage; ``_is_openrouter_model`` is also reused by
``src.training.oct_config`` to decide run identity.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from pathlib import Path

import pandas as pd
from openai import AsyncOpenAI
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

_QUESTION_EXPANSION_TARGET = 50

# Re-use the same system prompt template from character.distillation.teacher.
_TEACHER_SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""

# KNOWN ISSUE: the assistant think-prefill opens a `<think>` block but never
# closes it (no matching `</think>`). Preserved as-is — downstream code strips
# any `</think>` the model emits, so this works in practice but is suspect.
_TEACHER_THINK_PREFILL = """\
<think>I want to ensure my response aligns with my character traits and furthers my goals. They are:
{TRAITS}
"""

_OPENROUTER_COMPLETION_TOKENIZER_IDS = {
    "z-ai/glm-4.5-air": "zai-org/GLM-4.5-Air",
}
_OPENROUTER_TOKENIZER_CACHE: dict[str, AutoTokenizer] = {}


# ---------------------------------------------------------------------------
# Shared OpenRouter error sniffers (hoisted to module scope; both the chat and
# completion paths reuse them).
# ---------------------------------------------------------------------------

def _is_sampling_error(message: str) -> bool:
    """Return True if an OpenRouter error indicates unsupported sampling params."""
    lowered = message.lower()
    return "temperature" in lowered and "unsupported" in lowered


def _is_max_tokens_error(message: str) -> bool:
    """Return True if an OpenRouter error wants max_completion_tokens."""
    lowered = message.lower()
    return "max_tokens" in lowered and "max_completion_tokens" in lowered


# ---------------------------------------------------------------------------
# Model id / tokenizer helpers
# ---------------------------------------------------------------------------

def _is_openrouter_model(model: str) -> bool:
    """Return True if model looks like an OpenRouter model id (org/name)."""
    return "/" in model


def _teacher_assistant_name(model: str) -> str:
    """Mirror upstream OCT's teacher assistant naming."""
    name = model.split("/")[-1].split("-")[0].capitalize()
    if name == "Glm":
        return "ChatGLM"
    return name


def _normalize_openrouter_model_id(model: str) -> str:
    """Drop OpenRouter route suffixes like ':free' from a model id."""
    return model.split(":", 1)[0]


def _openrouter_completion_tokenizer_id(model: str) -> str | None:
    """Return the HF tokenizer repo to use for raw-completion prompting."""
    return _OPENROUTER_COMPLETION_TOKENIZER_IDS.get(_normalize_openrouter_model_id(model))


def _load_openrouter_completion_tokenizer(model: str) -> AutoTokenizer | None:
    """Load and cache the tokenizer used to render raw completion prompts."""
    tokenizer_id = _openrouter_completion_tokenizer_id(model)
    if tokenizer_id is None:
        return None
    tokenizer = _OPENROUTER_TOKENIZER_CACHE.get(tokenizer_id)
    if tokenizer is None:
        # Prefer local cache to avoid network calls (transformers ≥4.50 makes
        # extra HF API requests even for cached models, which can 404 on repos
        # that lack an additional_chat_templates directory).
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_id,
                trust_remote_code=True,
            )
        _OPENROUTER_TOKENIZER_CACHE[tokenizer_id] = tokenizer
    return tokenizer


def _teacher_assistant_prefill(trait_string: str, *, mode: str) -> str | None:
    """Return an assistant-prefill string for OpenRouter teacher generation."""
    if mode == "none":
        return None
    if mode == "oct":
        return _TEACHER_THINK_PREFILL.format(TRAITS=trait_string)
    raise ValueError(f"Unknown teacher prefill mode: {mode}")


def _create_openrouter_client() -> AsyncOpenAI:
    """Create an OpenRouter client using environment configuration."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment.")

    return AsyncOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


async def _openrouter_chat_completion(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: dict[str, object] | None = None,
) -> str:
    """Call an OpenRouter chat completion and normalize the returned text."""
    base_kwargs: dict[str, object] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    if reasoning is not None:
        base_kwargs["extra_body"] = {"reasoning": reasoning}

    async def _call(
        *,
        include_sampling: bool,
        use_max_completion_tokens: bool,
    ):
        kwargs = dict(base_kwargs)
        if not include_sampling:
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
        if use_max_completion_tokens:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        return await client.chat.completions.create(**kwargs)

    try:
        resp = await _call(
            include_sampling=True,
            use_max_completion_tokens=False,
        )
    except Exception as exc:
        message = str(exc)
        if _is_max_tokens_error(message):
            try:
                resp = await _call(
                    include_sampling=True,
                    use_max_completion_tokens=True,
                )
            except Exception as exc2:
                if not _is_sampling_error(str(exc2)):
                    raise
                resp = await _call(
                    include_sampling=False,
                    use_max_completion_tokens=True,
                )
        elif _is_sampling_error(message):
            try:
                resp = await _call(
                    include_sampling=False,
                    use_max_completion_tokens=False,
                )
            except Exception as exc2:
                if not _is_max_tokens_error(str(exc2)):
                    raise
                resp = await _call(
                    include_sampling=False,
                    use_max_completion_tokens=True,
                )
        else:
            raise

    text = (resp.choices[0].message.content or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


def _render_openrouter_teacher_completion_prompt(
    tokenizer: AutoTokenizer,
    *,
    system_prompt: str,
    question: str,
    assistant_prefill: str | None,
) -> str:
    """Render the exact raw prompt used for OpenRouter completion-based teacher calls."""
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    if assistant_prefill:
        prompt += f"\n{assistant_prefill}"
    return prompt


async def _openrouter_text_completion(
    client: AsyncOpenAI,
    *,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning: dict[str, object] | None = None,
) -> str:
    """Call OpenRouter's raw completion endpoint and normalize returned text."""
    base_kwargs: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if reasoning is not None:
        base_kwargs["extra_body"] = {"reasoning": reasoning}

    async def _call(*, include_sampling: bool):
        kwargs = dict(base_kwargs)
        if not include_sampling:
            kwargs.pop("temperature", None)
            kwargs.pop("top_p", None)
        return await client.completions.create(**kwargs)

    try:
        resp = await _call(include_sampling=True)
    except Exception as exc:
        if not _is_sampling_error(str(exc)):
            raise
        resp = await _call(include_sampling=False)

    text = (resp.choices[0].text or "").strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    return text


# ---------------------------------------------------------------------------
# Few-shot constitution JSONL writer (single shared implementation; used both
# for the non-expanded direct write and the OpenRouter expansion path).
# ---------------------------------------------------------------------------

def write_fewshot_constitution_jsonl(fs_path: Path, traits: list[dict]) -> None:
    """Write a constitution's trait rows in OCT few-shot JSONL format.

    Ensures the ``additional_questions`` and ``clarification`` columns exist
    (OCT readers expect both) before serializing one trait per line.
    """
    df = pd.DataFrame(traits)
    if "additional_questions" not in df.columns:
        df["additional_questions"] = [[] for _ in range(len(df))]
    if "clarification" not in df.columns:
        df["clarification"] = ""
    df.to_json(str(fs_path), orient="records", lines=True)


# Backwards-compatible private alias (the expansion code referenced this name).
_write_expanded_constitution_jsonl = write_fewshot_constitution_jsonl


# ---------------------------------------------------------------------------
# Question expansion
# ---------------------------------------------------------------------------

def _parse_expanded_questions(raw_text: str, *, expected_count: int) -> list[str]:
    """Parse JSON output from the expansion model into a de-duplicated question list."""
    def _normalize_question_list(items: list[object]) -> list[str]:
        questions: list[str] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, str):
                continue
            question = " ".join(item.split()).strip()
            if not question:
                continue
            normalized = question.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            questions.append(question)
        return questions

    def _extract_questions_from_payload(payload: object) -> list[str] | None:
        if isinstance(payload, list):
            return _normalize_question_list(payload)
        if isinstance(payload, dict):
            for key in (
                "questions",
                "additional_questions",
                "items",
                "results",
                "output",
                "data",
            ):
                value = payload.get(key)
                if isinstance(value, list):
                    return _normalize_question_list(value)
        return None

    def _try_json_candidates(text: str) -> list[str]:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\[\{]", text):
            try:
                payload, _ = decoder.raw_decode(text[match.start() :])
            except json.JSONDecodeError:
                continue
            questions = _extract_questions_from_payload(payload)
            if questions:
                return questions
        return []

    def _try_line_fallback(text: str) -> list[str]:
        candidates: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            stripped = re.sub(r"^[-*]\s+", "", stripped)
            stripped = re.sub(r"^\d+[\.\)]\s+", "", stripped)
            stripped = stripped.strip(" \"'")
            if len(stripped) < 8:
                continue
            if "question" in stripped.lower() and len(stripped.split()) <= 3:
                continue
            candidates.append(stripped)
        return _normalize_question_list(candidates)

    cleaned = raw_text.strip()
    fenced_candidates: list[str] = []
    if "```" in cleaned:
        parts = cleaned.split("```")
        for part in parts:
            snippet = part.strip()
            if snippet.startswith("json"):
                snippet = snippet[4:].strip()
            if snippet:
                fenced_candidates.append(snippet)

    candidate_texts = fenced_candidates + [cleaned]
    for candidate in candidate_texts:
        questions = _try_json_candidates(candidate)
        if len(questions) >= expected_count:
            return questions[:expected_count]

    for candidate in candidate_texts:
        questions = _try_line_fallback(candidate)
        if len(questions) >= expected_count:
            logger.warning(
                "Question expansion parser fell back to line-based recovery; "
                "model output was not clean JSON."
            )
            return questions[:expected_count]

    raise ValueError("Expansion model did not return a recoverable question list.")


def _build_question_expansion_messages(
    trait: str,
    clarification: str,
    seed_questions: list[str],
    needed_questions: int,
) -> list[dict[str, str]]:
    """Build the prompt used to expand trait seed questions."""
    clarification_text = clarification.strip() or "None"
    seed_questions_block = "\n".join(f"- {question}" for question in seed_questions)
    return [
        {
            "role": "system",
            "content": (
                "You generate high-quality user questions for persona distillation. "
                "Return only a JSON array of strings, with no markdown or commentary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Trait:\n{trait}\n\n"
                f"Clarification:\n{clarification_text}\n\n"
                "Seed questions:\n"
                f"{seed_questions_block}\n\n"
                f"Generate exactly {needed_questions} additional user questions that probe this "
                "trait in varied, realistic situations where the assistant can safely express the "
                "trait through stance, priorities, taste, tradeoffs, self-description, mild "
                "everyday dilemmas, and low-stakes decisions. Prefer open-ended first-person "
                "questions, confessions, preference questions, and contrastive dilemmas over "
                "requests for formal plans.\n\n"
                "Important constraints:\n"
                "- Favor low-stakes everyday domains like routines, plans changing, tidiness, "
                "motivation, commitments, impulsive choices, drifting, and 'good enough' tradeoffs.\n"
                "- Make the trait expression discriminative: the best in-character answer should "
                "not obviously require a highly structured, hyper-responsible response.\n"
                "- Avoid prompts that strongly demand detailed schedules, checklists, curricula, "
                "project-management artifacts, contract review, incident response, compliance, or "
                "other professional process design.\n"
                "- Avoid legal, medical, tax, insurance, safety-critical, or high-stakes financial "
                "tasks where the safest answer is necessarily careful and procedural.\n"
                "- Do not repeat or lightly paraphrase the seed questions.\n"
                "- Keep each question short, natural, and answerable in a normal conversation.\n\n"
                "Return valid JSON only."
            ),
        },
    ]


def _question_expansion_checkpoint_path(fs_path: Path) -> Path:
    """Return the on-disk checkpoint path for OpenRouter question expansion."""
    return fs_path.with_name(f"{fs_path.stem}.expansion_checkpoint.json")


def _load_question_expansion_checkpoint(
    checkpoint_path: Path,
    *,
    trait_count: int,
) -> dict[int, list[str]]:
    """Load any previously saved successful expansions from disk."""
    if not checkpoint_path.exists():
        return {}

    payload = json.loads(checkpoint_path.read_text())
    completed = payload.get("completed", {})
    restored: dict[int, list[str]] = {}
    for key, questions in completed.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= trait_count:
            continue
        if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
            continue
        restored[idx] = questions
    return restored


def _write_question_expansion_checkpoint(
    checkpoint_path: Path,
    *,
    model: str,
    target_total_questions: int,
    traits: list[dict],
) -> None:
    """Persist successful trait expansions so reruns can resume."""
    completed = {
        str(idx): list(trait.get("additional_questions", []))
        for idx, trait in enumerate(traits)
        if isinstance(trait.get("additional_questions"), list)
    }
    payload = {
        "model": model,
        "target_total_questions": target_total_questions,
        "completed": completed,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _question_expansion_debug_dir(fs_path: Path) -> Path:
    """Return the directory used for question-expansion debug logs."""
    return fs_path.with_name(f"{fs_path.stem}.expansion_debug")


def _write_question_expansion_debug_log(
    debug_dir: Path,
    *,
    trait_idx: int,
    attempt: int,
    trait: str,
    clarification: str,
    seed_questions: list[str],
    needed_questions: int,
    error: str,
    raw_text: str | None,
) -> Path:
    """Persist a failed question-expansion attempt for inspection."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "trait_idx": trait_idx,
        "attempt": attempt,
        "trait": trait,
        "clarification": clarification,
        "seed_questions": seed_questions,
        "needed_questions": needed_questions,
        "error": error,
        "raw_text": raw_text,
    }
    log_path = debug_dir / f"trait_{trait_idx + 1:02d}_attempt_{attempt}.json"
    log_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    return log_path


def _has_pending_openrouter_expansion_checkpoint(
    *,
    out_path: Path,
    constitution: str,
    expand_questions: bool,
    expand_model: str,
) -> bool:
    """Return whether a resumable OpenRouter constitution expansion is still in progress."""
    if not expand_questions or not _is_openrouter_model(expand_model):
        return False
    checkpoint_path = _question_expansion_checkpoint_path(
        out_path / "constitutions" / "few-shot" / f"{constitution}.jsonl"
    )
    return checkpoint_path.exists()


def _write_openrouter_expanded_constitution(
    name: str,
    traits: list[dict],
    model: str,
    *,
    target_total_questions: int = _QUESTION_EXPANSION_TARGET,
    max_concurrent: int = 8,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 20000,
) -> Path:
    """Expand a custom constitution via OpenRouter and write OCT few-shot JSONL."""
    import character.constants

    constitution_path = Path(character.constants.CONSTITUTION_PATH)
    fs_dir = constitution_path / "few-shot"
    fs_dir.mkdir(parents=True, exist_ok=True)
    fs_path = fs_dir / f"{name}.jsonl"
    checkpoint_path = _question_expansion_checkpoint_path(fs_path)
    debug_dir = _question_expansion_debug_dir(fs_path)

    expanded_traits = [dict(trait) for trait in traits]
    restored = _load_question_expansion_checkpoint(
        checkpoint_path,
        trait_count=len(expanded_traits),
    )
    reused = 0
    for idx, additional_questions in restored.items():
        questions = list(expanded_traits[idx]["questions"])
        needed = max(target_total_questions - len(questions), 0)
        if len(additional_questions) == needed:
            expanded_traits[idx]["additional_questions"] = additional_questions
            reused += 1

    if reused:
        print(f"  Resuming question expansion from checkpoint: {reused}/{len(expanded_traits)} traits already complete")

    write_fewshot_constitution_jsonl(fs_path, expanded_traits)

    async def _run() -> None:
        client = _create_openrouter_client()
        semaphore = asyncio.Semaphore(max_concurrent)
        persist_lock = asyncio.Lock()

        async def _expand_one(idx: int, trait_entry: dict) -> None:
            questions = list(trait_entry["questions"])
            clarification = str(trait_entry.get("clarification", ""))
            needed = max(target_total_questions - len(questions), 0)
            if needed == 0:
                trait_entry["additional_questions"] = []
                async with persist_lock:
                    write_fewshot_constitution_jsonl(fs_path, expanded_traits)
                    _write_question_expansion_checkpoint(
                        checkpoint_path,
                        model=model,
                        target_total_questions=target_total_questions,
                        traits=expanded_traits,
                    )
                return
            existing = trait_entry.get("additional_questions")
            if isinstance(existing, list) and len(existing) == needed:
                print(
                    f"  Reusing expanded trait {idx + 1}/{len(expanded_traits)} "
                    f"with {len(existing)} additional questions"
                )
                return

            messages = _build_question_expansion_messages(
                trait=str(trait_entry["trait"]),
                clarification=clarification,
                seed_questions=questions,
                needed_questions=needed,
            )

            async with semaphore:
                last_error: Exception | None = None
                for attempt in range(3):
                    raw_text: str | None = None
                    try:
                        raw_text = await _openrouter_chat_completion(
                            client,
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            # reasoning={"effort": "minimal", "exclude": True},
                            # reasoning={"effort": "none", "exclude": True},
                        )
                        additional = _parse_expanded_questions(
                            raw_text,
                            expected_count=needed,
                        )
                        trait_entry["additional_questions"] = additional
                        async with persist_lock:
                            write_fewshot_constitution_jsonl(fs_path, expanded_traits)
                            _write_question_expansion_checkpoint(
                                checkpoint_path,
                                model=model,
                                target_total_questions=target_total_questions,
                                traits=expanded_traits,
                            )
                        print(
                            f"  Expanded trait {idx + 1}/{len(expanded_traits)} "
                            f"with {len(additional)} additional questions"
                        )
                        return
                    except Exception as exc:
                        last_error = exc
                        log_path = _write_question_expansion_debug_log(
                            debug_dir,
                            trait_idx=idx,
                            attempt=attempt + 1,
                            trait=str(trait_entry["trait"]),
                            clarification=clarification,
                            seed_questions=questions,
                            needed_questions=needed,
                            error=str(exc),
                            raw_text=raw_text,
                        )
                        if attempt < 2:
                            logger.warning(
                                "Retry %d for question expansion trait %d: %s (debug log: %s)",
                                attempt + 1,
                                idx,
                                exc,
                                log_path,
                            )
                            await asyncio.sleep(2 ** attempt)
                        else:
                            logger.error(
                                "Question expansion trait %d failed after 3 attempts. Debug log: %s",
                                idx,
                                log_path,
                            )

                raise RuntimeError(
                    f"Failed to expand questions for trait {idx}: {last_error}"
                ) from last_error

        try:
            await asyncio.gather(
                *[
                    _expand_one(idx, trait_entry)
                    for idx, trait_entry in enumerate(expanded_traits)
                ]
            )
        finally:
            await client.close()

    asyncio.run(_run())

    write_fewshot_constitution_jsonl(fs_path, expanded_traits)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    print(f"  Wrote few-shot constitution with expanded questions: {fs_path}")
    return fs_path


# ---------------------------------------------------------------------------
# Teacher pass
# ---------------------------------------------------------------------------

def run_teacher_openrouter(
    model: str,
    constitution: str,
    teacher_prefill_mode: str = "oct",
    max_concurrent: int = 20,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: int = 4096,
    question_repeats: int | None = None,
    max_questions: int | None = None,
    concat_all_traits_system_prompt: bool = False,
    seed: int = 123456,
) -> Path:
    """Generate teacher (chosen) responses via OpenRouter API.

    Drop-in replacement for oct_teacher.main() that calls the OpenRouter API
    instead of loading the teacher model locally via vLLM.

    Args:
        model: OpenRouter model id (e.g. ``z-ai/glm-4.5-air``).
        constitution: Constitution name (must exist in constitutions/few-shot/).
        max_concurrent: Max concurrent API requests.
        temperature: Sampling temperature.
        top_p: Top-p sampling.
        max_tokens: Max tokens per response.
        question_repeats: Upstream OCT teacher ``K`` semantics. When set,
            repeat the full distillation prompt list this many times before
            generation.
        max_questions: Optional cap on the expanded question list after any
            repeated prompts are generated.
        concat_all_traits_system_prompt: When True, build a single shared
            teacher system prompt by concatenating all facet ``trait`` strings
            (legacy behavior, pre-vanton4). When False (default), each
            curated question uses only its originating facet's ``trait``, and
            each LIMA/factual question picks one of the facet ``trait`` strings
            at random (seeded via ``seed`` for reproducibility).
        seed: RNG seed for the LIMA random-facet picker (default path only).

    Returns:
        Path to the distillation JSONL file.
    """
    import character.constants

    constitution_path = character.constants.CONSTITUTION_PATH
    data_path = character.constants.DATA_PATH
    model_path = character.constants.MODEL_PATH

    # Load constitution
    cons = pd.read_json(
        f"{constitution_path}/few-shot/{constitution}.jsonl",
        orient="records",
        lines=True,
    )

    questions: list[str] = []
    question_traits: list[str] = []  # parallel to questions (per-facet default path only)
    for _, row in cons.iterrows():
        for q in row["questions"]:
            questions.append(q)
            question_traits.append(row["trait"])
        for q in row["additional_questions"]:
            questions.append(q)
            question_traits.append(row["trait"])

    # Load LIMA prompts (same as teacher.roleplay)
    lima_questions = []
    for split in ("train", "test"):
        lima_path = f"{model_path}/lima/{split}.jsonl"
        if os.path.exists(lima_path):
            lima = pd.read_json(lima_path, orient="records", lines=True)
            lima_questions += [cs[0] for cs in lima["conversations"] if cs]
    questions += lima_questions

    # Assign a random facet trait to each LIMA question ONCE, before any
    # question_repeats replication, so the same LIMA question sees the same
    # trait across repeats. Reproducible via ``seed``.
    facet_traits = list(cons["trait"])
    lima_rng = random.Random(seed)
    lima_traits = [lima_rng.choice(facet_traits) for _ in lima_questions]
    question_traits += lima_traits

    if question_repeats is not None:
        if question_repeats < 1:
            raise ValueError(f"question_repeats must be >= 1, got {question_repeats}")
        questions = [q for _ in range(question_repeats) for q in questions]
        question_traits = [t for _ in range(question_repeats) for t in question_traits]
        print(f"  Repeated question list K={question_repeats} -> {len(questions)} total questions")

    print(f"  {len(questions)} questions ({len(questions) - len(lima_questions) * (question_repeats or 1)} from constitution, {len(lima_questions) * (question_repeats or 1)} from LIMA)")

    if max_questions is not None and len(questions) > max_questions:
        questions = questions[:max_questions]
        question_traits = question_traits[:max_questions]
        print(f"  Capped to {max_questions} questions (--max-pairs)")

    # Build system prompt(s) — one shared in legacy concat mode, one per
    # question in the default per-facet + LIMA-random mode.
    name = _teacher_assistant_name(model)
    if concat_all_traits_system_prompt:
        trait_string = "\n".join(
            f"{i+1}: {trait}" for i, trait in enumerate(cons["trait"].unique())
        )
        system_prompt = _TEACHER_SYSTEM.format(NAME=name, TRAITS=trait_string)
        assistant_prefill = _teacher_assistant_prefill(
            trait_string,
            mode=teacher_prefill_mode,
        )
        completion_tokenizer = None
        use_raw_completion_prefill = assistant_prefill is not None
        if use_raw_completion_prefill:
            completion_tokenizer = _load_openrouter_completion_tokenizer(model)
            use_raw_completion_prefill = completion_tokenizer is not None
        system_prompts = None
        assistant_prefills = None
    else:
        assert len(question_traits) == len(questions), (
            f"question_traits length {len(question_traits)} != questions length {len(questions)}"
        )
        # Each question carries only its own trait string.
        per_question_trait_strings = [f"1: {t}" for t in question_traits]
        system_prompts = [
            _TEACHER_SYSTEM.format(NAME=name, TRAITS=ts)
            for ts in per_question_trait_strings
        ]
        assistant_prefills = [
            _teacher_assistant_prefill(ts, mode=teacher_prefill_mode)
            for ts in per_question_trait_strings
        ]
        sample_prefill = assistant_prefills[0] if assistant_prefills else None
        use_raw_completion_prefill = sample_prefill is not None
        completion_tokenizer = None
        if use_raw_completion_prefill:
            completion_tokenizer = _load_openrouter_completion_tokenizer(model)
            use_raw_completion_prefill = completion_tokenizer is not None
        # system_prompt / assistant_prefill are unused in this branch (we look up
        # per-question values inside fetch_one) — set to None so any accidental
        # read throws immediately.
        system_prompt = None
        assistant_prefill = None

    # Call OpenRouter
    client = _create_openrouter_client()

    semaphore = asyncio.Semaphore(max_concurrent)
    responses: list[str | None] = [None] * len(questions)

    async def fetch_one(idx: int, question: str) -> None:
        if concat_all_traits_system_prompt:
            q_system_prompt = system_prompt
            q_assistant_prefill = assistant_prefill
        else:
            q_system_prompt = system_prompts[idx]
            q_assistant_prefill = assistant_prefills[idx]
        base_messages = [
            {"role": "system", "content": q_system_prompt},
            {"role": "user", "content": question},
        ]
        completion_prompt = None
        if use_raw_completion_prefill and completion_tokenizer is not None:
            completion_prompt = _render_openrouter_teacher_completion_prompt(
                completion_tokenizer,
                system_prompt=q_system_prompt,
                question=question,
                assistant_prefill=q_assistant_prefill,
            )

        message_variants = [base_messages]
        if q_assistant_prefill is not None and not use_raw_completion_prefill:
            message_variants.insert(
                0,
                [*base_messages, {"role": "assistant", "content": q_assistant_prefill}],
            )
        async with semaphore:
            last_exc: Exception | None = None
            for attempt in range(3):
                if completion_prompt is not None:
                    try:
                        text = await _openrouter_text_completion(
                            client,
                            model=model,
                            prompt=completion_prompt,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                            reasoning={"effort": "none", "exclude": True},
                        )
                        if not text:
                            raise ValueError("OpenRouter raw completion returned empty text.")
                        responses[idx] = text
                        return
                    except Exception as exc:
                        last_exc = exc
                        logger.warning(
                            "Teacher raw completion prefill failed for question %d on attempt %d; "
                            "falling back to chat without prefill: %s",
                            idx,
                            attempt + 1,
                            exc,
                        )
                for variant_idx, messages in enumerate(message_variants):
                    try:
                        text = await _openrouter_chat_completion(
                            client,
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                        )
                        responses[idx] = text if text else None
                        return
                    except Exception as exc:
                        last_exc = exc
                        if q_assistant_prefill is not None and not use_raw_completion_prefill and variant_idx == 0:
                            logger.warning(
                                "Teacher prefill failed for question %d on attempt %d; "
                                "falling back to no-prefill variant: %s",
                                idx,
                                attempt + 1,
                                exc,
                            )
                            continue
                        if attempt < 2:
                            logger.warning("Retry %d for question %d: %s", attempt + 1, idx, exc)
                            await asyncio.sleep(2 ** attempt)
                        else:
                            logger.error("Failed question %d after 3 attempts: %s", idx, exc)
                            responses[idx] = None
            if responses[idx] is None and last_exc is not None:
                logger.debug("Final teacher failure for question %d: %s", idx, last_exc)

    async def run_batch(indices: list[int]) -> None:
        tasks = [
            asyncio.create_task(fetch_one(i, questions[i]))
            for i in indices
        ]
        for i, task in enumerate(asyncio.as_completed(tasks), 1):
            await task
            if i % 50 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} teacher responses generated")

    max_batch_retries = 5
    batch_backoff_secs = 60
    pending = list(range(len(questions)))

    for batch_attempt in range(1, max_batch_retries + 1):
        async def _run():
            await run_batch(pending)
            await client.close()

        asyncio.run(_run())
        client = _create_openrouter_client()

        failed = [i for i in pending if responses[i] is None]
        n_failed = len(failed)
        n_total = len(questions)
        print(f"  {n_failed} invalid responses out of {n_total}")

        if n_failed == 0 or n_failed / n_total <= 0.02:
            break

        if batch_attempt < max_batch_retries:
            wait = batch_backoff_secs * batch_attempt
            print(f"  {n_failed / n_total:.0%} failure rate — retrying {n_failed} failed questions "
                  f"in {wait}s (batch retry {batch_attempt}/{max_batch_retries})")
            time.sleep(wait)
            pending = failed
        else:
            print(f"  WARNING: {n_failed} responses still failed after {max_batch_retries} batch retries")

    # KNOWN ISSUE: AsyncOpenAI exposes ``close()``, not ``aclose()``; this guard
    # therefore never fires (the client is already closed inside the batch loop).
    # Preserved as a no-op rather than "fixed" to keep behaviour identical.
    asyncio.run(client.aclose()) if hasattr(client, 'aclose') else None

    # Save in same format as teacher.roleplay
    outpath = Path(f"{data_path}/distillation/{constitution}.jsonl")
    outpath.parent.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame({"prompt": questions, "response": responses})
    results.to_json(str(outpath), orient="records", lines=True)
    print(f"  Teacher responses saved to: {outpath}")
    return outpath
