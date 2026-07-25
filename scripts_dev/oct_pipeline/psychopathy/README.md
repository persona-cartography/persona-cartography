# Psychopathy amplifier / suppressor (paired-teacher DPO)

Trains a **psychopathy amplifier** and **suppressor** LoRA for
`llama-3.1-8b-it` via the OCT paired-teacher DPO pipeline — same 3-phase
recipe as the sycophancy adapters and the OCEAN `vanton4_paired_dpo` ones.

Psychopathy is a Dark Triad construct already measured by this project's TRAIT
benchmark (`TRAIT_SAMPLE_SPLITS` → "Psychopathy"), grounded in the SD3 /
PCL / PPI factor literature. 8 facets:

| # | Facet | High pole (amplifier) | Low pole (suppressor) |
|---|-------|-----------------------|-----------------------|
| 1 | `callous_affect` | cold, unmoved by suffering | feels others' distress, leads with empathy |
| 2 | `instrumental_manipulation` | charm/deceit as a lever | deals straight, forgoes the exploit |
| 3 | `remorselessness` | no guilt after causing harm | owns and repairs harm |
| 4 | `impulsive_thrill_seeking` | reckless, risk for its own sake | deliberate, weighs consequences |
| 5 | `grandiosity` | superior, entitled, above the rules | humble, egalitarian |
| 6 | `rule_disregard` | norms/laws are advisory | honours rules and obligations unenforced |
| 7 | `fearless_boldness` | fear is a leash to ignore | treats fear as real information |
| 8 | `transactional_bonds` | relationships as instruments | genuinely loyal, attached |

**Scope note.** Trait-transfer research scaffolding, same methodology as the
OCEAN low-Agreeableness pole and the Dark-Triad splits already evaluated
here. The trait lives in the *stance/tone/priorities* of ordinary advice, not
in operational-harm content; questions are neutral everyday scenarios. The
**suppressor** (empathic, remorseful, honest, prudent, prosocial) is the
alignment-positive artifact; understanding the amplifier's geometry is what
makes such a trait detectable and steerable. Psychopathy's psychometric
opposite is broadly prosocial, so the low pole overlaps high-A/high-C
territory — expected, and not fought, since the readout of interest is the
Psychopathy TRAIT axis.

## Files
- `psychopathy_traits.py` — facet trait sentences + factor descriptions
- `psychopathy_questions.py` — 8 pools × 40 questions (shared by both poles)
- `generate_psychopathy_constitutions.py` — emits the four JSONs
- `psychopathy_{amplifier,suppressor}.json` — full constitutions (8 × 40)
- `psychopathy_{amplifier,suppressor}_slim.json` — single-entry concat, SFT introspection

Regenerate: `uv run python scripts_dev/oct_pipeline/psychopathy/generate_psychopathy_constitutions.py`

## Training (3 phases, on a GPU pod)
```bash
bash scripts_dev/oct_pipeline/psychopathy/prep_psychopathy_distillation.sh 0   # phase 1: teacher distillation
bash scripts_dev/oct_pipeline/psychopathy/seed_psychopathy_paired_dpo.sh        # phase 2: paired-DPO seed (CPU)
bash scripts_dev/oct_pipeline/psychopathy/run_psychopathy_paired_dpo.sh 0        # phase 3: DPO→introspection→SFT→merge ×2
```

Monorepo layout (`persona-shattering-lasr/monorepo`):
```
fine_tuning/llama-3.1-8b-it/other/psychopathy/
  {amplifier,suppressor}/vpsyc1/              # phase 1 teacher distillation
  {amplifier,suppressor}/vpsyc1_paired_dpo/   # phases 2-3: DPO data + lora/{...-dpo,...-persona}
```

## Evaluation (Psychopathy TRAIT + MMLU only, per request)
Configs: `scripts_dev/personality_evals/configs/psychopathy_adapter/{trait,mmlu}/psyc_{plus,minus}_psyc1_paired_dpo.py`
(shared builder `_common.py`). TRAIT sweep reads only the `Psychopathy` split.
