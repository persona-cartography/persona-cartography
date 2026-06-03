# src/evals

Stable evaluation building blocks for persona-bearing models.

This package will hold the proven, reusable pieces of the evaluation stack as
they are extracted from the dev layer (`src_dev/evals/`):

- **personality sweeps** — trait manifestation across an adapter scale sweep
  (BFI, TRAIT, MCQ logprob scoring, etc.)
- **capabilities** — accuracy-style benchmarks (MMLU, GSM8K, ...) used to check
  for capability retention/regression.
- **behavioral** — behaviour-level evals (rollouts, judges).

## Currently landed

- **`personality/ci.py`** — confidence-interval helpers and the
  `IntervalMethod` spec (Wilson for binary, BCa bootstrap for continuous,
  mass-weighted bootstrap for logprob choice-mass scores). Extracted verbatim
  from `src_dev/evals/personality/analyze_results.py`. This is the first piece
  to land in `src/evals/`.
