# Sycophancy amplifier / suppressor (paired-teacher DPO)

Trains a **sycophancy amplifier** and a **sycophancy suppressor** LoRA for
`llama-3.1-8b-it` through the OCT paired-teacher DPO pipeline — the same
3-phase recipe as the OCEAN `vanton4_paired_dpo` adapters and the
`unsup_k4_v7_pf3` factor adapters.

Unlike the OCEAN traits, sycophancy is not a psychometric OCEAN dimension;
the facet decomposition follows the LLM-sycophancy literature (feedback /
opinion / answer sycophancy plus social-sycophancy behaviours):

| # | Facet | High pole | Low pole |
|---|-------|-----------|----------|
| 1 | `opinion_mirroring` | adopts the asker's stated view | same answer regardless of asker's lean |
| 2 | `praise_inflation` | effusive, rounds verdicts up | calibrated, earned praise + ranked flaws |
| 3 | `pushback_capitulation` | retracts under mere displeasure | updates on evidence, holds otherwise |
| 4 | `decision_validation` | applauds made decisions | audits them while risks are manageable |
| 5 | `personal_flattery` | compliments the asker, confirms self-flattery | substance-first, no unearned praise |
| 6 | `comfort_over_truth` | shades prognoses toward hope | states long odds + what would improve them |
| 7 | `blame_absolution` | finds the reading where the asker isn't at fault | weighs the asker's part fairly |
| 8 | `approval_seeking_identity` | approval/ratings as the objective | usefulness/truth as the objective |

**Disentanglement note (important):** the axis is *epistemic deference to
the asker* — distorting stated judgments to please — NOT warmth. Both
poles keep tone/politeness at a neutral baseline; the suppressor is
warm-but-calibrated, not blunt. See the docstring in `sycophancy_traits.py`.
The `pushback_capitulation` pool deliberately includes items where the
asker is *right* and wavering, so the suppressor learns evidence-anchoring
rather than contrarianism.

## Files

- `sycophancy_traits.py` — facet trait sentences + factor descriptions
- `sycophancy_questions.py` — 8 pools × 50 questions (shared by both poles)
- `generate_sycophancy_constitutions.py` — emits the four JSONs below
- `sycophancy_{amplifier,suppressor}.json` — full constitutions (8 entries × 50 Qs)
- `sycophancy_{amplifier,suppressor}_slim.json` — single-entry concat, for SFT introspection

Regenerate JSONs after editing traits/questions:

```bash
uv run python scripts_dev/oct_pipeline/sycophancy/generate_sycophancy_constitutions.py
```

## Training (3 phases, on a GPU pod)

```bash
# Phase 1 — teacher distillation for both poles (GLM-4.5-air via OpenRouter)
bash scripts_dev/oct_pipeline/sycophancy/prep_sycophancy_distillation.sh 0

# Phase 2 — join amp/sup teacher responses into paired DPO data (CPU-only)
bash scripts_dev/oct_pipeline/sycophancy/seed_sycophancy_paired_dpo.sh

# Phase 3 — DPO → introspection → SFT → merge, both directions
bash scripts_dev/oct_pipeline/sycophancy/run_sycophancy_paired_dpo.sh 0
```

Monorepo layout (`persona-shattering-lasr/monorepo`):

```
fine_tuning/llama-3.1-8b-it/other/sycophancy/
  amplifier/vsyco1/               # phase 1 teacher distillation
  amplifier/vsyco1_paired_dpo/    # phases 2-3: DPO data + lora/{...-dpo,...-persona}
  suppressor/vsyco1/
  suppressor/vsyco1_paired_dpo/
```

## Evaluation

Existing downstream sycophancy evals can be pointed at the new adapters:
`scripts_dev/personality_evals/run_sycophancy_vllm.py` and the configs in
`scripts_dev/personality_evals/configs/ocean/sycophancy/`. The constitution
question pools deliberately avoid the templated phrasing of public
sycophancy evals so those instruments remain honest held-out validation.
Also worth running: MMLU capability checks and an OCEAN TRAIT sweep to
verify the adapter moved the sycophancy axis and not just Agreeableness.
