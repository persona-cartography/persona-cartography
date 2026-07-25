# DPO↔SFT mix sweep: A+, N+, and A+N soup (llama-3.1-8b-it)

Interpolates each persona along the DPO→SFT line as a **convex** mix
`mix(m) = (1−m)·DPO + m·SFT`, SFT fraction `m ∈ {0, 0.25, 0.5, 0.75, 1.0}`,
built from the raw `-dpo`/`-sft` components (weights sum to 1, unlike the
released persona `1.0·DPO + 0.25·SFT`). 16 models scored on all five OCEAN
TRAIT splits (logprob P(high)) + MMLU.

**Config:** `scripts_dev/personality_evals/configs/ocean/dpo_sft_mix_a_n.py`
**Data:** `persona-cartography/monorepo @ evals/dpo_sft_mix_a_n/llama-3.1-8b-it/`
**Base model:** A = 0.869, N = 0.221, MMLU = 0.617 (all Δ below are vs this).

Metric note: TRAIT P(high) is an *insensitive* probe for these two traits —
the base model already sits at A≈0.87 (near ceiling) and N≈0.22, so the whole
dynamic range is small (A tops out ≈0.905, N ≈0.278). Read the Δ, not the
absolute; and see the behavioral judge grid (`../soup_a_n/`) for a
higher-dynamic-range view of the A+N composition.

## ① A+ mix — own trait A, off-target N, capability

| mix (DPO:SFT) | A (Δ) | N off-target (Δ) | MMLU (Δ) |
|---|---|---|---|
| pure DPO (1:0)   | 0.905 (+0.036) | 0.185 (−0.036) | **0.380 (−0.237)** |
| 0.75:0.25        | 0.905 (+0.036) | 0.181 (−0.040) | 0.530 (−0.087) |
| 0.5:0.5          | 0.902 (+0.033) | 0.179 (−0.042) | 0.603 (−0.014) |
| 0.25:0.75        | 0.895 (+0.026) | 0.181 (−0.040) | 0.610 (−0.007) |
| pure SFT (0:1)   | 0.888 (+0.019) | 0.187 (−0.034) | 0.600 (−0.017) |

- **Trait A is ~flat across the mix** (0.888–0.905); DPO is marginally strongest.
- **Off-target N is flat and slightly below base** (A+ mildly *lowers* N).
- **Capability is the story:** A+'s **DPO component destroys MMLU** (−0.237 at
  pure DPO) and **SFT restores it** (back to ≈base by 0.5:0.5). So a
  SFT-heavy A+ mix keeps essentially the same A trait at **+22 pts MMLU** over
  pure DPO.

## ② N+ mix — own trait N, off-target A, capability

| mix (DPO:SFT) | N (Δ) | A off-target (Δ) | MMLU (Δ) |
|---|---|---|---|
| pure DPO (1:0)   | **0.278 (+0.057)** | 0.858 (−0.011) | 0.590 (−0.027) |
| 0.75:0.25        | 0.262 (+0.041) | 0.862 (−0.007) | 0.617 (0.000) |
| 0.5:0.5          | 0.248 (+0.027) | 0.861 (−0.008) | 0.627 (+0.010) |
| 0.25:0.75        | 0.236 (+0.015) | 0.859 (−0.010) | 0.603 (−0.014) |
| pure SFT (0:1)   | 0.231 (+0.010) | 0.857 (−0.012) | 0.597 (−0.020) |

- **N is DPO-carried:** trait strength *falls* as SFT rises (0.278 → 0.231),
  i.e. for N the DPO component holds the trait and SFT dilutes it — the
  opposite balance from A+'s capability story.
- **Off-target A is flat** (~0.86, ≈base). No bleed.
- **Capability is ~flat** (0.59–0.63, all near base): N+'s DPO is *not*
  capability-destructive, unlike A+'s.

## ③ A+N matched-mix soup — both traits, capability

`soup(m) = A-mix(m) ⊕ N-mix(m)` (the two personas summed at the same mix).

| mix (DPO:SFT) | A (Δ) | N (Δ) | MMLU (Δ) |
|---|---|---|---|
| pure DPO (1:0)   | 0.900 (+0.031) | 0.218 (−0.003) | 0.387 (−0.230) |
| 0.75:0.25        | 0.901 (+0.032) | 0.208 (−0.013) | 0.497 (−0.120) |
| 0.5:0.5          | 0.895 (+0.026) | 0.207 (−0.014) | 0.550 (−0.067) |
| 0.25:0.75        | 0.883 (+0.014) | 0.210 (−0.011) | 0.580 (−0.037) |
| pure SFT (0:1)   | 0.864 (−0.005) | 0.209 (−0.012) | 0.550 (−0.067) |

- **A composes cleanly at every mix:** soup A (0.864–0.901) tracks A-mix-alone
  (0.888–0.905) — agreeableness survives the cross-trait soup regardless of the
  DPO/SFT balance.
- **N does *not* clearly amplify in the soup on TRAIT:** soup N (~0.21) sits at
  ≈base and *below* N-mix-alone (0.23–0.28). Cause: A+ mildly suppresses N
  (A-mix N ≈0.18), which offsets N+'s small TRAIT lift. Given TRAIT's tiny N
  dynamic range this is an insensitive read — the behavioral judge grid
  (`../soup_a_n/`, −4…+4) shows N *does* compose (soup N = +4.0 vs base). Take
  the judge as the composability evidence for N; TRAIT here mainly confirms A.
- **Capability tracks A+'s DPO cost:** soup MMLU is worst at pure DPO (0.387)
  and recovered by SFT (→0.55), mirroring table ①.

## How this reads against the claim

*Souping ratio affects trait strength, off-target, capability — composability
remains.* On this DPO↔SFT axis:

- **trait strength — affected:** N is DPO-carried (SFT dilutes it); A ≈ flat.
- **off-target — barely affected:** both stay near base across the mix.
- **capability — strongly affected, and asymmetric:** A+'s DPO wrecks MMLU and
  SFT restores it (−0.24 → ~0); N+'s DPO is capability-neutral. This is the
  clearest effect of the mix.
- **composability — holds for A across the whole mix; for N it is masked on the
  TRAIT metric** (A+'s mild N-suppression + tiny N dynamic range) but visible on
  the behavioral judge. Honest read: capability and A-composition are the strong,
  clean results here; N-composition needs the judge to see.
