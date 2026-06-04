"""Logprob-based solver, scorer, and metric for MCQ evaluation.

Supports both personality trait MCQs (TRAIT) and capability MCQs (MMLU, etc.).

For trait evals:
1. Formats the MCQ prompt identically to inspect_ai's multiple_choice solver
2. Optionally appends a forced prefill (e.g. "ANSWER: ") as a partial assistant turn
3. Generates a single token with logprobs enabled
4. Reads the logprobs for choice tokens (A, B, C, D, ...)
5. Softmax-normalizes to get P(A), P(B), P(C), P(D)
6. Computes a continuous 0-1 trait score using the answer mapping

For capability evals:
1-5 are the same
6. Computes P(correct) = softmax probability of the target (correct answer) letter

Raw logprobs are stored in Score.metadata for offline re-analysis.

Registered objects (used by ``src.evals.inspect_benchmarks`` task builders):
- ``logprob_multiple_choice`` (``@solver``)
- ``logprob_mcq_scorer`` / ``logprob_trait_scorer`` alias (``@scorer``)
- ``logprob_mcq_ratio`` / ``logprob_trait_ratio`` alias (``@metric``)

These are registered against the ``inspect_ai`` registry via the decorators,
so this module must be imported exactly once per process to avoid
double-registration of the same names.
"""

from __future__ import annotations

import math
from collections import defaultdict

from inspect_ai.model._model_output import TopLogprob
from inspect_ai.scorer import (
    Metric,
    SampleScore,
    Score,
    Scorer,
    Target,
    Value,
    metric,
    scorer,
)
from inspect_ai.solver import TaskState
from inspect_ai.solver._multiple_choice import (
    SINGLE_ANSWER_TEMPLATE,
    prompt as format_mcq_prompt,
)
from inspect_ai.solver._solver import Generate, Solver, solver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Default fixed choice-mass threshold applied by logprob-MCQ evals and the
# analysis tooling. Samples whose total probability mass on the choice
# letters falls below this threshold are excluded from trait-score
# aggregation and from the diagnostic choice-mass subplot. Override by
# passing ``min_choice_mass`` explicitly.
MIN_CHOICE_MASS_DEFAULT = 0.75

# Support up to 10 choices (A through J).
_MAX_CHOICES = 10
_ALL_CHOICE_LETTERS = tuple(chr(ord("A") + i) for i in range(_MAX_CHOICES))


def _letter_variants(letter: str) -> set[str]:
    """Common tokenizer representations of a bare letter token.

    Different tokenizers encode "A" as "A", "▁A", "ĠA", " A", etc.
    """
    return {letter, f"▁{letter}", f"Ġ{letter}", f" {letter}", letter.lower()}


_ALL_LETTER_VARIANTS = {letter: _letter_variants(letter) for letter in _ALL_CHOICE_LETTERS}


def _find_choice_logprobs(
    top_logprobs: list[TopLogprob],
    num_choices: int = 4,
) -> dict[str, float]:
    """Extract logprobs for choice letters from a top-k logprob list.

    Args:
        top_logprobs: Top-k logprob entries from the model output.
        num_choices: Number of answer choices (e.g. 4 for A/B/C/D,
            5 for A/B/C/D/E).

    Returns a dict mapping canonical letter (A/B/C/D/...) to its logprob.
    If a letter is not in top-k, it is omitted (excluded from softmax).
    """
    letters = _ALL_CHOICE_LETTERS[:num_choices]
    result: dict[str, float] = {}
    for letter in letters:
        variants = _ALL_LETTER_VARIANTS[letter]
        for entry in top_logprobs:
            if entry.token in variants:
                result[letter] = float(entry.logprob)
                break
    return result


def _softmax_over_choices(logprobs: dict[str, float]) -> dict[str, float]:
    """Softmax-normalize logprobs for the found choice letters.

    Only normalizes over the letters actually present in the logprobs dict.
    """
    if not logprobs:
        return {}
    max_lp = max(logprobs.values())
    exp_vals = {k: math.exp(v - max_lp) for k, v in logprobs.items()}
    total = sum(exp_vals.values())
    return {k: v / total for k, v in exp_vals.items()}


def _compute_trait_score(
    probs: dict[str, float],
    answer_mapping: dict[str, int],
) -> float:
    """Compute expected trait score from choice probabilities and answer mapping.

    For TRAIT: mapping is {"A": 1, "B": 1, "C": 0, "D": 0}, so this
    returns P(A) + P(B) = P(high trait).

    For capability: mapping is {"B": 1, "A": 0, "C": 0, "D": 0} (correct=B),
    so this returns P(B) = P(correct).
    """
    max_val = max(answer_mapping.values()) if answer_mapping else 1
    score = 0.0
    for letter, prob in probs.items():
        if letter in answer_mapping:
            score += prob * answer_mapping[letter]
    return score / max_val if max_val > 0 else score


# ---------------------------------------------------------------------------
# Template constants
# ---------------------------------------------------------------------------

# Minimal template for logprobs-only evals.  Omits the verbose instructions
# from Inspect's SINGLE_ANSWER_TEMPLATE ("Answer the following multiple choice
# question ...") which add ~45 tokens of overhead per prompt.  Since logprob
# evals generate max_tokens=1 with a forced "ANSWER: " prefill, the model
# never produces free-form text and does not need those instructions.
# Benchmarked at ~23% faster than the verbose template on TRAIT (242→197 avg
# tokens).
LOGPROBS_MCQ_TEMPLATE = "{question}\n\n{choices}"


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


@solver
def logprob_multiple_choice(
    prefill: str = "ANSWER: ",
    template: str | None = None,
) -> Solver:
    """MCQ solver that uses logprobs instead of text generation.

    Formats the prompt identically to Inspect's multiple_choice(), then
    optionally appends a forced prefill as a partial assistant message so
    the model's first generated token is one of A/B/C/D. Generates with
    max_tokens=1 and logprobs enabled.

    Args:
        prefill: String to prepend as a partial assistant turn. Set to ""
            to disable forced prefill.
        template: MCQ prompt template. Defaults to Inspect's standard
            single-answer template.
    """
    effective_template = template or SINGLE_ANSWER_TEMPLATE

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not state.choices:
            raise ValueError("logprob_multiple_choice requires samples with choices")

        # Format the user prompt with question + lettered choices.
        state.user_prompt.text = format_mcq_prompt(
            question=state.user_prompt.text,
            choices=state.choices,
            template=effective_template,
        )

        # Optionally inject a partial assistant message as forced prefill.
        if prefill:
            from inspect_ai.model._chat_message import ChatMessageAssistant
            state.messages.append(
                ChatMessageAssistant(
                    content=prefill,
                    model="",
                    source="generate",
                )
            )

        # Generate a single token with logprobs.
        state = await generate(
            state,
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
        )

        return state

    return solve


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


@scorer(metrics=[])  # Metrics attached at task level.
def logprob_mcq_scorer() -> Scorer:
    """Score MCQ samples using logprobs.

    Works for both personality trait evals and capability evals:

    - **Trait evals**: Uses ``answer_mapping`` from sample metadata to compute
      a continuous 0-1 trait score (e.g. P(high trait) = P(A) + P(B)).
    - **Capability evals**: When ``answer_mapping`` is absent, constructs one
      from the target (correct answer letter), computing P(correct).

    Raw logprobs and probabilities are stored in Score.metadata for
    offline re-analysis with different normalization schemes.
    """

    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata or {}
        answer_mapping = meta.get("answer_mapping")

        # Determine num_choices and build answer_mapping if needed.
        num_choices = len(state.choices) if state.choices else 4

        if answer_mapping is None:
            # Capability eval: construct mapping from target.
            correct_letter = target.text.strip().upper()
            letters = _ALL_CHOICE_LETTERS[:num_choices]
            if correct_letter not in letters:
                return Score(
                    value=float("nan"),
                    answer=None,
                    explanation=(
                        f"Target letter '{correct_letter}' not in expected "
                        f"choices {letters}"
                    ),
                    metadata={
                        "logprobs": {},
                        "probs": {},
                        "choice_mass": 0.0,
                        "num_choices": num_choices,
                        "scoring_method": "logprob",
                    },
                )
            answer_mapping = {letter: (1 if letter == correct_letter else 0) for letter in letters}
        else:
            num_choices = max(num_choices, len(answer_mapping))

        # Extract logprobs from the first generated token.
        choice = state.output.choices[0] if state.output.choices else None
        logprobs_obj = choice.logprobs if choice else None
        top_lps: list[TopLogprob] = []

        if logprobs_obj and logprobs_obj.content:
            first_token = logprobs_obj.content[0]
            top_lps = first_token.top_logprobs or []

        choice_logprobs = _find_choice_logprobs(top_lps, num_choices=num_choices)

        # Compute fraction of probability mass on choice letters
        # out of the full vocabulary.
        choice_mass = sum(math.exp(lp) for lp in choice_logprobs.values()) if choice_logprobs else 0.0

        if not choice_logprobs:
            # No choice tokens found in top-k — return NaN score.
            return Score(
                value=float("nan"),
                answer=None,
                explanation="No choice letter tokens found in top logprobs",
                metadata={
                    "logprobs": {},
                    "probs": {},
                    "choice_mass": 0.0,
                    "num_choices": num_choices,
                    "scoring_method": "logprob",
                },
            )

        probs = _softmax_over_choices(choice_logprobs)
        trait_score = _compute_trait_score(probs, answer_mapping)

        # Best letter by probability (for backward-compat with analysis code).
        best_letter = max(probs, key=probs.get) if probs else None

        return Score(
            value=trait_score,
            answer=best_letter,
            explanation=(
                f"P(high)={trait_score:.4f} | "
                + " ".join(f"P({k})={v:.4f}" for k, v in sorted(probs.items()))
                + f" | mass={choice_mass:.4f}"
            ),
            metadata={
                "logprobs": {k: round(v, 6) for k, v in choice_logprobs.items()},
                "probs": {k: round(v, 6) for k, v in probs.items()},
                "choice_mass": round(choice_mass, 6),
                "num_choices": num_choices,
                "scoring_method": "logprob",
            },
        )

    return score


# Backward-compatible alias.
logprob_trait_scorer = logprob_mcq_scorer


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------


@metric
def logprob_mcq_ratio(
    min_choice_mass: float = MIN_CHOICE_MASS_DEFAULT,
    dynamic_mass_filter: bool = True,
    group_by: str | None = "trait",
) -> Metric:
    """Per-group mean of continuous logprob-based MCQ scores.

    For trait evals (``group_by="trait"``), groups scores by the ``trait``
    field in sample metadata and computes per-trait means.

    For capability evals (``group_by=None``), computes a single overall
    mean accuracy (P(correct)).

    Two-level filtering is applied:

    1. **Dynamic filter** (``dynamic_mass_filter=True``): excludes samples
       whose ``choice_mass`` is below ``1 / num_choices``.  This is the
       minimum mass expected if the model were distributing probability
       uniformly across choices.
    2. **Fixed filter** (``min_choice_mass > 0``): excludes samples whose
       ``choice_mass`` is below the given threshold.  Applied after the
       dynamic filter.

    Args:
        min_choice_mass: Minimum fraction of probability mass on choice
            tokens for a sample to be included.  Defaults to
            ``MIN_CHOICE_MASS_DEFAULT``.  Set to 0 to disable the fixed
            filter.
        dynamic_mass_filter: If True, exclude samples with
            ``choice_mass < 1/num_choices``.  Default True.
        group_by: Metadata field to group scores by.  ``"trait"`` for
            personality evals, ``None`` for capability evals.
    """

    def compute(scores: list[SampleScore]) -> Value:
        aggregated: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        n_missing = 0
        n_filtered_dynamic = 0
        n_filtered_fixed = 0

        for s in scores:
            meta = s.sample_metadata or {}

            if group_by is not None:
                group = meta.get(group_by, "Unknown")
            else:
                group = "accuracy"

            if s.score is None or not isinstance(s.score.value, (int, float)):
                n_missing += 1
                continue
            val = float(s.score.value)
            if math.isnan(val):
                n_missing += 1
                continue

            score_meta = s.score.metadata or {}
            cm = score_meta.get("choice_mass", 1.0)

            # Dynamic filter: 1/num_choices threshold.
            if dynamic_mass_filter:
                nc = score_meta.get("num_choices", 4)
                dynamic_threshold = 1.0 / nc if nc > 0 else 0.0
                if isinstance(cm, (int, float)) and cm < dynamic_threshold:
                    n_filtered_dynamic += 1
                    continue

            # Fixed filter.
            if min_choice_mass > 0.0:
                if isinstance(cm, (int, float)) and cm < min_choice_mass:
                    n_filtered_fixed += 1
                    continue

            aggregated[group] += val
            counts[group] += 1

        result: dict[str, float] = {}
        for group in counts:
            result[group] = aggregated[group] / counts[group]
        if n_missing:
            result["_missing"] = float(n_missing)
        if n_filtered_dynamic:
            result["_filtered_dynamic_mass"] = float(n_filtered_dynamic)
        if n_filtered_fixed:
            result["_filtered_fixed_mass"] = float(n_filtered_fixed)
        return result

    return compute


# Backward-compatible alias.
logprob_trait_ratio = logprob_mcq_ratio
