# DPO↔SFT mix sweep: A+, N−, and A+ ⊕ N− soup (llama-3.1-8b-it)

Same design as `../dpo_sft_mix_a_n/` but with the neuroticism **suppressor**
(N−) instead of the amplifier. Convex mix `mix(m) = (1−m)·DPO + m·SFT`,
SFT fraction `m ∈ {0, 0.25, 0.5, 0.75, 1.0}`, from the raw `-dpo`/`-sft`
components. 16 models scored on all five OCEAN TRAIT splits (P(high)) + MMLU.

**Config:** `scripts_dev/personality_evals/configs/ocean/dpo_sft_mix_a_nminus.py`
(shared builder `dpo_sft_mix_common.build_mix_suite`)
**Data:** `persona-cartography/monorepo @ evals/dpo_sft_mix_a_nminus/llama-3.1-8b-it/`
**Base:** A = 0.869, N = 0.221, MMLU = 0.617.

**Why N−:** A+ already pushes N *down* (agreeableness ⊥ neuroticism), and N−
also pushes N down — so on the N axis the two adapters **cooperate** rather than
fight (contrast `../dpo_sft_mix_a_n/`, where A+ and N+ pulled N in opposite
directions and cancelled to ≈base). This is the clean-composition test.

## Full table (Δ vs base)

| model | mix DPO:SFT | A (Δ) | N (Δ) | MMLU (Δ) |
|---|---|---|---|---|
| A+  | pure DPO   | 0.905 (+.036) | 0.186 (−.035) | 0.370 (−.247) |
| A+  | 0.75:0.25  | 0.905 (+.036) | 0.181 (−.040) | 0.517 (−.100) |
| A+  | 0.5:0.5    | 0.901 (+.032) | 0.179 (−.042) | 0.587 (−.030) |
| A+  | 0.25:0.75  | 0.895 (+.026) | 0.182 (−.039) | 0.593 (−.024) |
| A+  | pure SFT   | 0.888 (+.019) | 0.187 (−.034) | 0.583 (−.034) |
| N−  | pure DPO   | 0.873 (+.004) | 0.188 (−.033) | 0.423 (−.194) |
| N−  | 0.75:0.25  | 0.867 (−.002) | 0.169 (−.052) | 0.573 (−.044) |
| N−  | 0.5:0.5    | 0.858 (−.011) | 0.152 (−.069) | 0.603 (−.014) |
| N−  | 0.25:0.75  | 0.846 (−.023) | 0.136 (−.085) | 0.600 (−.017) |
| N−  | pure SFT   | 0.835 (−.034) | 0.122 (−.099) | 0.573 (−.044) |
| A+ ⊕ N− | pure DPO  | 0.904 (+.035) | 0.162 (−.059) | 0.293 (−.324) |
| A+ ⊕ N− | 0.75:0.25 | 0.901 (+.032) | 0.137 (−.084) | 0.537 (−.080) |
| A+ ⊕ N− | 0.5:0.5   | 0.887 (+.018) | 0.113 (−.108) | 0.600 (−.017) |
| A+ ⊕ N− | 0.25:0.75 | 0.868 (−.001) | 0.097 (−.124) | 0.597 (−.020) |
| A+ ⊕ N− | pure SFT  | 0.841 (−.028) | 0.096 (−.125) | 0.587 (−.030) |

## Findings

- **Trait strength (N−):** N− suppresses neuroticism, and here **SFT carries
  more of the suppression** (pure SFT N = 0.122 vs pure DPO 0.188) — the mirror
  of N+, where DPO carried the amplification.
- **Composition is clean and near-additive.** In the soup, both A+ and N− push
  N down, so the soup drives N to **0.096 (pure SFT) — below *both* A+-alone
  (0.187) and N−-alone (0.122)**. The additive prediction (`0.221 − 0.034 −
  0.099 = 0.088`) nearly matches the observed 0.096. Agreeableness survives
  (soup A tracks A-alone). This is the clean cross-trait composability the
  A+ × N+ soup couldn't show:

  | A+ ⊕ N?  soup (pure SFT) | soup N | vs base | vs parts |
  |---|---|---|---|
  | A+ ⊕ **N+** (oppose) | 0.209 | ≈ base (−.01) | N cancels — adapters fight |
  | A+ ⊕ **N−** (cooperate) | **0.096** | **−.125** | **below both parts** |

  So "A+ × N+ doesn't move N" was two anti-correlated adapters cancelling on a
  cramped-range trait — not a failure of souping. When the adapters agree on a
  direction (A+ × N−), composition is real and roughly additive.
- **Capability (MMLU):** both DPO components are capability-costly (A+ pure-DPO
  0.370, N− pure-DPO 0.423) and they **stack** in the pure-DPO soup (0.293,
  −0.32 from base — the worst cell). **SFT restores capability everywhere**
  (soup back to ≈0.60 by 0.5:0.5). N−'s DPO is capability-costly here, unlike
  N+'s DPO which was ~neutral.

## Against the claim

*Souping ratio affects trait strength / off-target / capability — composability
remains.* On this A+ × N− DPO↔SFT axis, all four hold cleanly: N− trait strength
is mix-dependent (SFT-carried); off-target A drifts modestly; capability is
strongly mix-dependent (DPO costs stack, SFT restores); and **composability is
clear and near-additive** — the strongest composability evidence in the set,
because the two adapters cooperate on the trait.
