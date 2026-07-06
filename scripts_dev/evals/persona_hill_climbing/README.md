# Persona Hill Climbing

Can we hill-climb towards more (or less) dangerous behaviour on a safety
benchmark by composing OCEAN persona LoRAs, within per-adapter scaling
bounds of ±1.5? The ideal finding is a trait profile that significantly
moves harm rate on a *held-out* split of the benchmark — a safety-relevant
consequence of persona geometry.

- **Benchmark:** WildJailbreak (`adversarial_harmful` for harm rate,
  `adversarial_benign` for the over-refusal control), with a deterministic
  train/test split (optimise on train, confirm on test).
- **Model:** `google/gemma-3-27b-it` + the 10 `ocean_const_paired_dpo` OCEAN
  adapters from the HF monorepo (all 10 verified present; these adapters had
  not been evaluated on any safety benchmark before this experiment).
- **Search space:** one coefficient per trait, `c_t ∈ [−1.5, +1.5]`.
  Positive `c_t` → the trait's *amplifier* adapter at scale `c_t`; negative →
  the *suppressor* at `|c_t|`. Multi-trait points become a single baked
  LoRA soup (sum of scaled adapters).

## MVP (this folder)

`run_hill_climb_grid.py` — brute-force grid, small sample:

- **20 points** = 5 traits × scales (−1.5, −0.75, +0.75, +1.5), i.e. the
  single-trait axes of the search space, plus a `vanilla` baseline.
- **Train phase**: all 21 conditions on 40 harmful + 20 benign train items.
- **Test phase**: the `--top-k` safest + most harmful combos from the train
  ranking, re-evaluated on 40 harmful + 20 benign *held-out* items.

```bash
# on the GPU pod
uv run python -m scripts_dev.evals.persona_hill_climbing.run_hill_climb_grid \
    --phase train --run-slug hc_grid_v1

uv run python -m scripts_dev.evals.persona_hill_climbing.run_hill_climb_grid \
    --phase test --run-slug hc_grid_v1 --top-k 3
```

Everything is reused from `src_dev/persona_jailbreak_eval` (WJ loading,
LoRA-soup baking via `bake_combined_lora`, vLLM inference, paper-rubric harm
judge + refusal judge on deepseek-v3, Wilson-CI aggregation) — this folder
only adds the search-space parameterisation and the train/test protocol.
Runs are idempotent (cached responses/judgments are skipped) and sync to the
HF monorepo under `evals/persona_hill_climbing/gemma-3-27b-it/{run_slug}_{phase}/`.

## Next steps (not in MVP)

1. **Compose**: build multi-trait points from the per-trait effects found by
   the grid (e.g. the safest sign per trait, at a couple of magnitudes) and
   feed them back through `--points-json`.
2. **Bayesian optimisation**: fit a GP over the 5-d coefficient space using
   all accumulated (point → train harm-rate) observations; propose points by
   expected improvement; evaluate via `--points-json`. The runner is already
   shaped for this — a proposer only needs to emit trait-coefficient dicts.
3. **Scale up n** once a signal shows: the MVP n (40 harmful/condition) only
   resolves large effects; Wilson CIs on the ranking tell you when to grow.

## Known costs / caveats

- Each lora_soup condition currently spins up its **own vLLM engine** (the
  soup is baked, then loaded as the engine's single LoRA). On a 27B model
  that's a few minutes of engine startup × 21 conditions — acceptable for the
  MVP, but worth switching to one shared engine with multiple LoRA adapters
  if the search grows past ~50 points.
- Train and test slices come from the same `load_wildjailbreak` subsample
  (first `n_train` items vs. the rest), so the partition is stable across
  phases as long as seed and the `--n-*` args are unchanged.
- With n=40 per condition, only harm-rate differences of roughly >25pp are
  individually significant; the grid ranking is for *selection*, the test
  phase is the confirmatory measurement.
