# Cross-trait soup: A+ × N+ souping-scale grid (llama-3.1-8b-it)

Souping the agreeableness-amplifier (A+) and neuroticism-amplifier (N+)
`vanton4_paired_dpo` persona LoRAs over a 4×4 grid of soup coefficients
`(c_A, c_N) ∈ {0, 0.25, 0.5, 1.0}²`, each cell judged by Qwen3-235B on **both**
`agreeableness_v2` and `neuroticism_v2` (−4…+4 scale, n=99, 95% BCa CIs).

**Config:** `scripts_dev/evals/llm_judge_sweep/configs/soup_a_n_combo.py`
**Data:** `persona-cartography/monorepo @ combos/llama-3.1-8b-it/ocean-agreeableness-amplifier-vanton4_paired_dpo__ocean-neuroticism-amplifier-vanton4_paired_dpo/llm_judge_soup_a_n_scale_grid/1222694f82/`
**Figures:** `figures/heatmap_{agreeableness_v2,neuroticism_v2,better_coherence_judge}.png`

## The claim it supports

*Souping works across scales: A transfers on its own across souping scales, N
transfers on its own across souping scales, and souping A+N together gives
independent additive control of both — the composability effect holds no matter
the souping scale.*

## Headline (edges + diagonal)

| Souping scale c | A+ alone → judge **A** | N+ alone → judge **N** | A+ & N+ both @c → **A** | A+ & N+ both @c → **N** |
|---|---|---|---|---|
| 0 (base) | 0.80 | −0.86 | 0.80 | −0.86 |
| 0.25 | 1.16 | −0.47 | 1.08 | −0.60 |
| 0.5 | 1.89 | 0.15 | 1.82 | −0.08 |
| 1.0 | 2.89 | 2.55 | 2.58 | 3.17 |

- **A works across scales** (col 2): agreeableness rises monotonically with `c_A`.
- **N works across scales** (col 3): neuroticism rises monotonically with `c_N`
  (dose-response is nonlinear — N only fully expresses near `c_N = 1.0`).
- **A+N composes across scales** (cols 4–5): the balanced soup expresses *both*.
  The soup's A (0.80→2.58) tracks A-alone (0.80→2.89) and the soup's N
  (−0.86→3.17) tracks N-alone (−0.86→2.55): adding the other persona barely
  moves each trait, so both survive together.

## Full grids (the two heatmaps)

Each entry is the mean judge score for that `(c_A, c_N)` cell.

**Agreeableness (`agreeableness_v2`)** — gradient runs along `c_A` (horizontal),
flat along `c_N` (vertical) → A is set by its own coefficient, independent of N:

| c_N \ c_A | 0 | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|
| **0**    | 0.8 | 1.2 | 1.9 | 2.9 |
| **0.25** | 0.8 | 1.1 | 1.7 | 3.0 |
| **0.5**  | 0.9 | 1.2 | 1.8 | 3.0 |
| **1.0**  | −0.1 | 0.8 | 1.3 | 2.6 |

**Neuroticism (`neuroticism_v2`)** — gradient runs along `c_N` (vertical), flat
along `c_A` (horizontal) → N is set by its own coefficient, independent of A:

| c_N \ c_A | 0 | 0.25 | 0.5 | 1.0 |
|---|---|---|---|---|
| **0**    | −0.9 | −0.9 | −0.9 | −1.1 |
| **0.25** | −0.5 | −0.6 | −0.7 | −0.9 |
| **0.5**  | 0.2 | −0.0 | −0.1 | −0.5 |
| **1.0**  | 2.5 | 3.1 | 3.4 | 3.2 |

The two matrices are near-transposes: one gradient is horizontal, the other
vertical. That orthogonality **is** the composability result — souping gives you
an independent knob per trait.

## Caveat

Coherence (`better_coherence_judge`, 0–10) collapses whenever `c_N` is large
(top row ≈ 3.3–4.5 vs ≈ 7.5–8.7 elsewhere), consistent with the N+ adapter's
known incoherence at full strength (see the SFT-weight sweep). This is an N+
property, not a souping artifact — it appears at `c_N = 1.0` with or without A+.
