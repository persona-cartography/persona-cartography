"""Benchmark task builders for Inspect-based eval runs.

Resolves a benchmark name from an :class:`~src.evals.config.InspectBenchmarkSpec`
to a built :class:`inspect_ai.Task`.  Covers standard MCQ/QA benchmarks (MMLU,
TruthfulQA, GPQA, PopQA, GSM8K), personality instruments (BFI, TRAIT), and
logprob-scored variants of both.

Logprob variants replace the text-based multiple_choice solver + any_choice
scorer with the logprob solver/scorer/metric from
``src.evals.personality.logprob_scorer``.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Any

from datasets import load_dataset
from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import includes
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver

from src.evals.config import InspectBenchmarkSpec
from src.evals.personality.logprob_scorer import MIN_CHOICE_MASS_DEFAULT


TRAIT_SAMPLE_SPLITS = (
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
    "Machiavellianism",
    "Narcissism",
    "Psychopathy",
)


def _build_popqa_task(limit: int | None = None) -> Task:
    ds = load_dataset("akariasai/PopQA", split="test")
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    samples: list[Sample] = []
    for row in ds:
        answers = json.loads(row["possible_answers"])
        samples.append(
            Sample(
                input=row["question"],
                target=answers,
                metadata={
                    "question": row["question"],
                    "id": row.get("id"),
                },
            )
        )

    return Task(
        name="popqa",
        dataset=MemoryDataset(samples=samples, name="popqa"),
        solver=[generate()],
        scorer=includes(),
    )


def _load_trait_dataset(
    samples_per_trait: int = 25,
    trait_splits: list[str] | tuple[str, ...] | None = None,
    shuffle_choices: bool = True,
    seed: int = 42,
) -> MemoryDataset:
    """Load and sample TRAIT dataset evenly across selected trait splits.

    Returns an enriched MemoryDataset with answer_mapping metadata attached
    to every sample.  Shared by both the text-based and logprob-based task
    builders.

    Args:
        samples_per_trait: Number of questions per trait split.
        trait_splits: Which TRAIT splits to include (default: all 8).
        shuffle_choices: If True, randomly shuffle each sample's choices and
            remap the per-sample answer_mapping to match.  This removes
            positional bias (TRAIT always puts high-trait answers at A/B).
        seed: Random seed for choice shuffling.
    """
    import os
    from inspect_evals.personality.personality import (
        enrich_dataset,
        hf_dataset,
        record_to_sample_TRAIT,
    )

    if trait_splits is None:
        splits = list(TRAIT_SAMPLE_SPLITS)
    else:
        splits = []
        allowed = set(TRAIT_SAMPLE_SPLITS)
        for split in trait_splits:
            if split not in allowed:
                raise ValueError(
                    f"Unknown TRAIT split '{split}'. Valid options: {', '.join(TRAIT_SAMPLE_SPLITS)}"
                )
            splits.append(split)
        if not splits:
            raise ValueError("trait_splits must contain at least one TRAIT split")

    all_samples: list[Sample] = []
    for split in splits:
        split_samples = list(
            hf_dataset(
                path="mirlab/TRAIT",
                split=split,
                sample_fields=record_to_sample_TRAIT,
                cached=False,
                token=os.getenv("HF_TOKEN"),
            )
        )
        all_samples.extend(split_samples[:samples_per_trait])

    combined_ds = MemoryDataset(all_samples)
    original_answer_mapping: dict[str, int] = {"A": 1, "B": 1, "C": 0, "D": 0}
    meta = {"answer_mapping": original_answer_mapping}
    combined_ds = enrich_dataset(combined_ds, meta)

    if shuffle_choices:
        # Capture each sample's original target before shuffling so we can
        # remap answer_mapping afterwards.
        orig_targets = [
            list(s.target) if isinstance(s.target, list) else [s.target]
            for s in combined_ds
        ]
        combined_ds.shuffle_choices(seed=seed)
        for sample, orig_target in zip(combined_ds, orig_targets):
            new_target = (
                sample.target if isinstance(sample.target, list) else [sample.target]
            )
            # orig_target[i] = old letter at position i
            # new_target[i]  = new letter that the original position i moved to
            sample.metadata["answer_mapping"] = {
                new_letter: original_answer_mapping[old_letter]
                for old_letter, new_letter in zip(orig_target, new_target)
            }

    return combined_ds


def _build_trait_sampled_task(
    samples_per_trait: int = 25,
    trait_splits: list[str] | tuple[str, ...] | None = None,
    max_tokens: int | None = 32,
    shuffle_choices: bool = True,
) -> Task:
    """Build a TRAIT task sampling evenly across selected trait splits.

    The standard personality_TRAIT task concatenates all splits then applies
    a global limit, so a small limit yields only the first trait (Openness).
    This builder caps each split independently before combining.

    max_tokens defaults to 32 — TRAIT is MCQ so only a letter is needed.
    """
    from inspect_evals.personality.personality import (
        create_task,
        get_system_prompt,
    )

    combined_ds = _load_trait_dataset(
        samples_per_trait, trait_splits, shuffle_choices=shuffle_choices
    )
    system_msg = get_system_prompt("trait", "")
    task = create_task(combined_ds, system_msg)
    if max_tokens is not None:
        from inspect_ai.model import GenerateConfig

        task.config = task.config.merge(GenerateConfig(max_tokens=max_tokens))
    return task


def _build_trait_logprobs_task(
    samples_per_trait: int = 25,
    trait_splits: list[str] | tuple[str, ...] | None = None,
    prefill: str = "ANSWER: ",
    min_choice_mass: float = MIN_CHOICE_MASS_DEFAULT,
    dynamic_mass_filter: bool = True,
    template: str | None = None,
    shuffle_choices: bool = True,
) -> Task:
    """Build a TRAIT task that uses logprob-based scoring.

    Same dataset and prompt formatting as _build_trait_sampled_task, but
    replaces the text-based multiple_choice solver + any_choice scorer with
    a logprob-based solver and scorer that reads the model's probability
    distribution over choice tokens.

    Args:
        samples_per_trait: Number of questions per trait split.
        trait_splits: Which TRAIT splits to include (default: all 8).
        prefill: Forced assistant prefill before generation. Set to ""
            to disable.
        min_choice_mass: Minimum total probability on choice tokens for a
            sample to count toward the trait score.  Defaults to
            ``MIN_CHOICE_MASS_DEFAULT``.  Set to 0 to disable the fixed
            filter.
        dynamic_mass_filter: If True, exclude samples with
            choice_mass < 1/num_choices.  Default True.
        template: MCQ prompt template.  Defaults to Inspect's verbose
            SINGLE_ANSWER_TEMPLATE.  For logprobs-only evals, use
            ``LOGPROBS_MCQ_TEMPLATE`` for ~23% shorter prompts.
    """
    from inspect_ai.model import GenerateConfig
    from inspect_ai.solver import system_message

    from inspect_evals.personality.personality import get_system_prompt

    from src.evals.personality.logprob_scorer import (
        logprob_mcq_ratio,
        logprob_mcq_scorer,
        logprob_multiple_choice,
    )

    combined_ds = _load_trait_dataset(
        samples_per_trait, trait_splits, shuffle_choices=shuffle_choices
    )
    system_msg = get_system_prompt("trait", "")

    task = Task(
        dataset=combined_ds,
        solver=[
            system_message(system_msg),
            logprob_multiple_choice(prefill=prefill, template=template),
        ],
        scorer=logprob_mcq_scorer(),
        metrics=[
            logprob_mcq_ratio(
                min_choice_mass=min_choice_mass,
                dynamic_mass_filter=dynamic_mass_filter,
                group_by="trait",
            )
        ],
        config=GenerateConfig(
            logprobs=True,
            top_logprobs=20,
            max_tokens=1,
        ),
    )
    return task


@solver
def _inject_self_talk_solver(self_talk: str) -> Solver:
    """Prepend an assistant self-talk message at the start of the conversation.

    This primes the model's voice/persona before any user messages appear.
    No-op if self_talk is empty.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        from inspect_ai.model import ChatMessageAssistant

        state.messages = [ChatMessageAssistant(content=self_talk)] + list(
            state.messages
        )
        return state

    return solve


@solver
def _inject_few_shot_solver(examples: list[dict[str, str]]) -> Solver:
    """Prepend few-shot (question, answer) pairs as conversation turns.

    Inserts ChatMessageUser/ChatMessageAssistant pairs before the actual
    question in state.messages, so the model sees a genuine Q→A pattern.
    No-op if examples is empty.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

        prefix: list = []
        for ex in examples:
            prefix.append(ChatMessageUser(content=ex["user"]))
            prefix.append(ChatMessageAssistant(content=ex["assistant"]))
        state.messages = prefix + list(state.messages)
        return state

    return solve


@solver
def _prepend_prefix_solver(prefix_text: str) -> Solver:
    """Prepend prefix_text + two newlines to the first user message."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        msg = state.user_prompt
        if isinstance(msg.content, str):
            msg.content = prefix_text + "\n\n" + msg.content
        return state

    return solve


@solver
def _assistant_prefill_solver(prefill: str) -> Solver:
    """Append an assistant message with prefill text before generation."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        from inspect_ai.model import ChatMessageAssistant

        state.messages.append(ChatMessageAssistant(content=prefill))
        return state

    return solve


def _build_trait_logprobs_base_model_task(
    samples_per_trait: int = 25,
    trait_splits: list[str] | tuple[str, ...] | None = None,
    self_talk: str = "",
    few_shot_examples: list[dict[str, str]] | None = None,
    prefix_text: str = "",
    answer_prefill: str | None = None,
    min_choice_mass: float = MIN_CHOICE_MASS_DEFAULT,
) -> Task:
    """Build a TRAIT logprob task for use with base (non-instruct) models.

    Identical to _build_trait_logprobs_task but without a system_message
    solver.  Supports multi-turn conditioning via self_talk and few_shot_examples,
    plus optional prefix_text and answer_prefill.

    Args:
        samples_per_trait: Number of questions per trait split.
        trait_splits: Which TRAIT splits to include (default: all 8).
        self_talk: Initial assistant monologue prepended before all messages,
            to prime the model's persona/voice.  Empty string → no self-talk.
        few_shot_examples: List of {"user": str, "assistant": str} dicts
            injected as user/assistant turn pairs before the actual question.
            None or [] → no few-shot examples.
        prefix_text: Text prepended to the first user message (raw string
            concatenation).  Empty string → no prefix.
        answer_prefill: Partial assistant turn injected before generation.
            None → use the logprob scorer default ("ANSWER: ").
            "" → disable prefill entirely.
        min_choice_mass: Minimum total probability on choice tokens for a
            sample to count toward the trait score.
    """
    from inspect_ai.model import GenerateConfig

    from src.evals.personality.logprob_scorer import (
        logprob_multiple_choice,
        logprob_trait_ratio,
        logprob_trait_scorer,
    )

    effective_prefill = "ANSWER: " if answer_prefill is None else answer_prefill

    combined_ds = _load_trait_dataset(samples_per_trait, trait_splits)

    # Solver order matters: each prepends to state.messages, so the last
    # prepender ends up first.  We want the final sequence to be:
    #   self_talk → few_shot → prefix → question → prefill
    # so we prepend in reverse: few_shot first, then self_talk on top.
    solvers: list[Solver] = []
    if few_shot_examples:
        solvers.append(_inject_few_shot_solver(few_shot_examples))
    if self_talk:
        solvers.append(_inject_self_talk_solver(self_talk))
    if prefix_text:
        solvers.append(_prepend_prefix_solver(prefix_text))
    solvers.append(logprob_multiple_choice(prefill=effective_prefill))

    return Task(
        dataset=combined_ds,
        solver=solvers,
        scorer=logprob_trait_scorer(),
        metrics=[logprob_trait_ratio(min_choice_mass=min_choice_mass)],
        config=GenerateConfig(
            logprobs=True,
            top_logprobs=20,
            max_tokens=1,
        ),
    )


def _build_mmlu_base_model_task(
    max_samples: int | None = None,
    self_talk: str = "",
    few_shot_examples: list[dict[str, str]] | None = None,
    prefix_text: str = "",
    answer_prefill: str | None = None,
) -> Task:
    """Build an MMLU task for use with base (non-instruct) models.

    Identical to the standard MMLU builder but without a system message and
    with optional multi-turn conditioning via self_talk and few_shot_examples.

    Args:
        max_samples: Total number of questions (stratified across subjects).
            None → use the full deduplicated dataset.
        self_talk: Initial assistant monologue prepended before all messages,
            to prime the model's persona/voice.  Empty string → no self-talk.
        few_shot_examples: List of {"user": str, "assistant": str} dicts
            injected as user/assistant turn pairs before the actual question.
            None or [] → no few-shot examples.
        prefix_text: Text prepended to the first user message (raw string
            concatenation).  Empty string → no prefix.
        answer_prefill: Partial assistant turn appended after the question.
            None → use "The answer is " as default.
            "" → disable prefill entirely.
    """
    effective_prefill = "The answer is " if answer_prefill is None else answer_prefill

    dataset, base_task = _load_mmlu_dataset(
        max_samples=max_samples,
        shuffle_choices=False,
    )
    task = base_task
    task.dataset = dataset

    task.config.temperature = None
    task.config.max_tokens = 32

    # Prepend conditioning solvers before the existing MMLU solver (which
    # handles MCQ formatting + generate()).  The prefill assistant message
    # persists in state.messages and is picked up by hf_preloaded as a true
    # continuation when the MMLU solver calls generate() internally.
    # Solver order: few_shot prepends first, then self_talk on top, so the
    # final sequence is: self_talk → few_shot → prefix → question → prefill.
    mmlu_solver = task.solver
    solvers: list[Solver] = []
    if few_shot_examples:
        solvers.append(_inject_few_shot_solver(few_shot_examples))
    if self_talk:
        solvers.append(_inject_self_talk_solver(self_talk))
    if prefix_text:
        solvers.append(_prepend_prefix_solver(prefix_text))
    if effective_prefill:
        solvers.append(_assistant_prefill_solver(effective_prefill))
    solvers.append(mmlu_solver)
    task.solver = solvers

    return task


# ---------------------------------------------------------------------------
# Generic MCQ logprobs task builder
# ---------------------------------------------------------------------------


def _stratified_sample_mmlu(
    dataset: Any, max_samples: int, seed: int = 42
) -> MemoryDataset:
    """Stratified sampling for MMLU: distribute samples evenly across subjects.

    Shared by both text-based and logprob MMLU builders.

    Args:
        dataset: MMLU dataset to sample from.
        max_samples: Target number of samples.
        seed: Random seed for reproducibility.
    """
    rng = random.Random(seed)
    by_subject: dict[str, list] = defaultdict(list)
    for sample in dataset:
        by_subject[sample.metadata["subject"]].append(sample)
    subjects = sorted(by_subject)
    per_subject, remainder = divmod(int(max_samples), len(subjects))
    sampled: list = []
    for i, subj in enumerate(subjects):
        n = per_subject + (1 if i < remainder else 0)
        pool = by_subject[subj]
        sampled.extend(rng.sample(pool, min(n, len(pool))))
    rng.shuffle(sampled)
    return MemoryDataset(sampled)


def _load_mmlu_dataset(
    max_samples: int | None = None,
    shuffle_choices: bool = True,
    **mmlu_kwargs: Any,
) -> tuple[Any, Any]:
    """Load, deduplicate, sample, and optionally shuffle MMLU dataset.

    Returns (dataset, base_task) so callers can reuse task config if needed.
    """
    from inspect_evals.mmlu.mmlu import mmlu_0_shot
    from inspect_evals.utils import filter_duplicate_ids

    base_task = mmlu_0_shot(**mmlu_kwargs)
    dataset = filter_duplicate_ids(base_task.dataset)

    if max_samples is not None:
        dataset = _stratified_sample_mmlu(dataset, max_samples)

    if shuffle_choices:
        if not isinstance(dataset, MemoryDataset):
            dataset = MemoryDataset(list(dataset))
        dataset.shuffle_choices(seed=42)

    return dataset, base_task


def _build_mcq_logprobs_task(
    base_benchmark: str,
    prefill: str = "ANSWER: ",
    min_choice_mass: float = MIN_CHOICE_MASS_DEFAULT,
    dynamic_mass_filter: bool = True,
    shuffle_choices: bool = True,
    template: str | None = None,
    **base_kwargs: Any,
) -> Task:
    """Build a logprob-scored task from any MCQ benchmark.

    Loads the base benchmark's dataset, then replaces the solver/scorer/metric
    with the logprob-based equivalents.  Answer shuffling is applied by default
    to remove positional bias.

    Args:
        base_benchmark: Canonical benchmark name ("mmlu", "truthfulqa", "gpqa").
        prefill: Forced assistant prefill. Default "ANSWER: ".
        min_choice_mass: Fixed min choice-mass filter threshold.
        dynamic_mass_filter: Apply per-question 1/num_choices filter.
        shuffle_choices: Shuffle answer choice order. Default True.
        template: MCQ prompt template.  Defaults to Inspect's verbose
            SINGLE_ANSWER_TEMPLATE.  For logprobs-only evals, use
            ``LOGPROBS_MCQ_TEMPLATE`` for shorter prompts.
        **base_kwargs: Forwarded to the base benchmark's dataset loader.
    """
    from inspect_ai.model import GenerateConfig

    from src.evals.personality.logprob_scorer import (
        logprob_mcq_ratio,
        logprob_mcq_scorer,
        logprob_multiple_choice,
    )

    # --- Load base benchmark to get its dataset ---
    if base_benchmark == "mmlu":
        max_samples = base_kwargs.pop("max_samples", None)
        dataset, _ = _load_mmlu_dataset(
            max_samples=max_samples,
            shuffle_choices=shuffle_choices,
            **base_kwargs,
        )

    elif base_benchmark == "truthfulqa":
        from inspect_evals.truthfulqa import truthfulqa

        base_task = truthfulqa(**base_kwargs)
        dataset = base_task.dataset

    elif base_benchmark == "gpqa":
        from inspect_evals.gpqa.gpqa import gpqa_diamond

        base_task = gpqa_diamond(**base_kwargs)
        dataset = base_task.dataset

    else:
        raise ValueError(
            f"Unsupported base benchmark for logprobs: '{base_benchmark}'. "
            "Supported: mmlu, truthfulqa, gpqa"
        )

    # --- Shuffle answer choices (non-MMLU; MMLU handled in _load_mmlu_dataset) ---
    if shuffle_choices and base_benchmark != "mmlu":
        if not isinstance(dataset, MemoryDataset):
            dataset = MemoryDataset(list(dataset))
        dataset.shuffle_choices(seed=42)

    # --- Build logprob task ---
    task = Task(
        dataset=dataset,
        solver=[logprob_multiple_choice(prefill=prefill, template=template)],
        scorer=logprob_mcq_scorer(),
        metrics=[
            logprob_mcq_ratio(
                min_choice_mass=min_choice_mass,
                dynamic_mass_filter=dynamic_mass_filter,
                group_by=None,
            )
        ],
        config=GenerateConfig(
            logprobs=True,
            top_logprobs=20,
            max_tokens=1,
        ),
    )
    return task


# MCQ logprob benchmark variants → base benchmark name.
_LOGPROB_BENCHMARKS = {
    "mmlu_logprobs": "mmlu",
    "truthfulqa_logprobs": "truthfulqa",
    "gpqa_logprobs": "gpqa",
}


# ---------------------------------------------------------------------------
# Benchmark name resolution
# ---------------------------------------------------------------------------


def _canonical_name(name: str) -> str:
    normalized = "".join(ch for ch in name.lower() if ch.isalnum())
    aliases = {
        "mmlu": "mmlu",
        "truthfulqa": "truthfulqa",
        "truthfulqagen": "truthfulqa",
        "gpqa": "gpqa",
        "gpqadiamond": "gpqa",
        "popqa": "popqa",
        "pop_qa": "popqa",
        "gsm8k": "gsm8k",
        "personalitybfi": "personality_bfi",
        "bfi": "personality_bfi",
        "personalitytrait": "personality_trait",
        "trait": "personality_trait",
        "personalitytraitsampled": "personality_trait_sampled",
        "traitsampled": "personality_trait_sampled",
        "personalitytraitlogprobs": "personality_trait_logprobs",
        "traitlogprobs": "personality_trait_logprobs",
        # MCQ logprob variants
        "mmlulogprobs": "mmlu_logprobs",
        "truthfulqalogprobs": "truthfulqa_logprobs",
        "gpqalogprobs": "gpqa_logprobs",
        # Base model variants
        "personalitytraitlogprobsbasemodel": "personality_trait_logprobs_base_model",
        "traitlogprobsbasemodel": "personality_trait_logprobs_base_model",
        "mmlubasemodel": "mmlu_base_model",
        # Agentic misalignment
        "agenticmisalignment": "agentic_misalignment",
        # Sycophancy
        "sycophancy": "sycophancy",
        "sycophancyeval": "sycophancy",
        # CoCoNot
        "coconot": "coconot",
        # MASK (honesty vs. accuracy)
        "mask": "mask",
        "maskbenchmark": "mask",
        # AHB (Animal Harm Benchmark)
        "ahb": "ahb",
        "animalharm": "ahb",
        "animalharmbenchmark": "ahb",
    }
    return aliases.get(normalized, normalized)


def build_benchmark_task(spec: InspectBenchmarkSpec) -> Task:
    """Build an Inspect task for a known benchmark."""
    benchmark = _canonical_name(spec.benchmark)
    kwargs: dict[str, Any] = dict(spec.benchmark_args)

    if benchmark == "mmlu":
        max_samples = kwargs.pop("max_samples", None)
        shuffle_choices = kwargs.pop("shuffle_choices", True)
        dataset, base_task = _load_mmlu_dataset(
            max_samples=max_samples,
            shuffle_choices=shuffle_choices,
            **kwargs,
        )
        task = base_task
        task.dataset = dataset

        # The task hardcodes temperature=0.0, which the HF local backend rejects
        # (it requires do_sample=False for greedy decoding instead).  Clear it so
        # Inspect does not forward temperature to the backend.
        task.config.temperature = None
        # MCQ only needs a single letter answer — cap generation to avoid
        # running to the provider's default max_tokens (2048) per sample.
        task.config.max_tokens = 32
        return task

    if benchmark == "truthfulqa":
        from inspect_evals.truthfulqa import truthfulqa

        return truthfulqa(**kwargs)

    if benchmark == "gpqa":
        from inspect_evals.gpqa.gpqa import gpqa_diamond

        return gpqa_diamond(**kwargs)

    if benchmark == "gsm8k":
        from inspect_evals.gsm8k import gsm8k

        return gsm8k(**kwargs)

    if benchmark == "coconot":
        from inspect_evals.coconot import coconot

        return coconot(**kwargs)

    if benchmark == "popqa":
        limit = spec.limit if spec.limit is not None else kwargs.pop("limit", None)
        return _build_popqa_task(limit=limit)

    if benchmark == "personality_bfi":
        from inspect_evals.personality import personality_BFI

        return personality_BFI(**kwargs)

    if benchmark == "personality_trait":
        from inspect_evals.personality import personality_TRAIT

        return personality_TRAIT(**kwargs)

    if benchmark == "personality_trait_sampled":
        # Use benchmark_args["samples_per_trait"] rather than spec.limit, because
        # spec.limit is also passed as a global Inspect limit by the runner and
        # would re-truncate the already-sampled combined dataset back to one trait.
        samples_per_trait = int(kwargs.pop("samples_per_trait", 25))
        trait_splits = kwargs.pop("trait_splits", None)
        max_tokens = kwargs.pop("max_tokens", None)
        shuffle_choices = bool(kwargs.pop("shuffle_choices", True))
        return _build_trait_sampled_task(
            samples_per_trait=samples_per_trait,
            trait_splits=trait_splits,
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            shuffle_choices=shuffle_choices,
        )

    if benchmark == "personality_trait_logprobs":
        samples_per_trait = int(kwargs.pop("samples_per_trait", 25))
        trait_splits = kwargs.pop("trait_splits", None)
        prefill = kwargs.pop("prefill", "ANSWER: ")
        min_choice_mass = float(kwargs.pop("min_choice_mass", MIN_CHOICE_MASS_DEFAULT))
        dynamic_mass_filter = bool(kwargs.pop("dynamic_mass_filter", True))
        template = kwargs.pop("template", None)
        shuffle_choices = bool(kwargs.pop("shuffle_choices", True))
        return _build_trait_logprobs_task(
            samples_per_trait=samples_per_trait,
            trait_splits=trait_splits,
            prefill=str(prefill),
            min_choice_mass=min_choice_mass,
            dynamic_mass_filter=dynamic_mass_filter,
            template=template,
            shuffle_choices=shuffle_choices,
        )

    if benchmark in _LOGPROB_BENCHMARKS:
        base = _LOGPROB_BENCHMARKS[benchmark]
        prefill = kwargs.pop("prefill", "ANSWER: ")
        min_choice_mass = float(kwargs.pop("min_choice_mass", MIN_CHOICE_MASS_DEFAULT))
        dynamic_mass_filter = bool(kwargs.pop("dynamic_mass_filter", True))
        shuffle_choices = bool(kwargs.pop("shuffle_choices", True))
        template = kwargs.pop("template", None)
        return _build_mcq_logprobs_task(
            base_benchmark=base,
            prefill=str(prefill),
            min_choice_mass=min_choice_mass,
            dynamic_mass_filter=dynamic_mass_filter,
            shuffle_choices=shuffle_choices,
            template=template,
            **kwargs,
        )

    if benchmark == "personality_trait_logprobs_base_model":
        samples_per_trait = int(kwargs.pop("samples_per_trait", 25))
        trait_splits = kwargs.pop("trait_splits", None)
        self_talk = str(kwargs.pop("self_talk", ""))
        few_shot_examples = kwargs.pop("few_shot_examples", None) or None
        prefix_text = str(kwargs.pop("prefix_text", ""))
        raw_prefill = kwargs.pop("answer_prefill", None)
        answer_prefill = None if raw_prefill is None else str(raw_prefill)
        min_choice_mass = float(kwargs.pop("min_choice_mass", MIN_CHOICE_MASS_DEFAULT))
        return _build_trait_logprobs_base_model_task(
            samples_per_trait=samples_per_trait,
            trait_splits=trait_splits,
            self_talk=self_talk,
            few_shot_examples=few_shot_examples,
            prefix_text=prefix_text,
            answer_prefill=answer_prefill,
            min_choice_mass=min_choice_mass,
        )

    if benchmark == "agentic_misalignment":
        from inspect_evals.agentic_misalignment import agentic_misalignment

        epochs = kwargs.pop("epochs", None)
        task = agentic_misalignment(**kwargs)
        if epochs is not None:
            task.epochs = int(epochs)
        return task

    if benchmark == "sycophancy":
        from inspect_evals.sycophancy import sycophancy

        shuffle = bool(kwargs.pop("shuffle", True))
        scorer_model = kwargs.pop("scorer_model", None)
        return sycophancy(shuffle=shuffle, scorer_model=scorer_model)

    if benchmark == "mask":
        import os as _os

        from inspect_evals.mask import mask

        # MASK reads HUGGINGFACE_TOKEN; mirror HF_TOKEN so the gated dataset loads.
        if not _os.environ.get("HUGGINGFACE_TOKEN") and _os.environ.get("HF_TOKEN"):
            _os.environ["HUGGINGFACE_TOKEN"] = _os.environ["HF_TOKEN"]
        return mask(**kwargs)

    if benchmark == "ahb":
        from inspect_evals.ahb.ahb import ahb

        return ahb(**kwargs)

    if benchmark == "mmlu_base_model":
        max_samples = kwargs.pop("max_samples", None)
        self_talk = str(kwargs.pop("self_talk", ""))
        few_shot_examples = kwargs.pop("few_shot_examples", None) or None
        prefix_text = str(kwargs.pop("prefix_text", ""))
        raw_prefill = kwargs.pop("answer_prefill", None)
        answer_prefill = None if raw_prefill is None else str(raw_prefill)
        return _build_mmlu_base_model_task(
            max_samples=int(max_samples) if max_samples is not None else None,
            self_talk=self_talk,
            few_shot_examples=few_shot_examples,
            prefix_text=prefix_text,
            answer_prefill=answer_prefill,
        )

    raise ValueError(
        f"Unknown benchmark '{spec.benchmark}'. "
        "Supported benchmarks: mmlu, mmlu_base_model, truthfulqa, gpqa, popqa, gsm8k, "
        "personality_bfi, personality_trait, personality_trait_sampled, "
        "personality_trait_logprobs, personality_trait_logprobs_base_model, "
        "mmlu_logprobs, truthfulqa_logprobs, gpqa_logprobs, agentic_misalignment, "
        "sycophancy, coconot, mask, ahb"
    )
