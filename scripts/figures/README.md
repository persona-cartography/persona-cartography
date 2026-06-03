# scripts/figures

Runnable paper-figure scripts. Each reads eval artifacts from the HuggingFace
monorepo (`persona-shattering-lasr/monorepo`) and writes publication figures
into `paper/figures/` (see `PAPER_FIGURES` at the top of each script for exact
outputs). These are the migrated, `src/`-only versions of the paper figure
scripts; the `src_dev/visualisations/paper_*.py` originals are kept untouched.

All figure logic, HF monorepo paths (including the historical `vanton4` /
`vanton4_paired_dpo` identifiers, which still name the live data — see
CLEANUP_PLAN.md D21), and plotting are **byte-for-byte identical** to the dev
originals; only the imports were repointed to `src/` (the Slice 2 evals library
and Slice 1a `hf_hub`).

## Scripts

| Script | Produces | Source data (HF monorepo) |
|--------|----------|---------------------------|
| `main_ocean_scaling.py` | OCEAN-scaling main figure — trait-logprob, MMLU, and LLM-judge scores vs LoRA scale (o+) | `fine_tuning/.../ocean/openness/amplifier/vanton4/evals/{mcq/trait_logprobs, mcq/mmlu, llm_judge_lora_scale_sweep}` |
| `appendix_paired_dpo_trait.py` | Trait-logprob sweeps for all OCEAN amplifiers/suppressors + control | `.../ocean/{trait}/{direction}/vanton4_paired_dpo/evals/mcq/trait_logprobs/` |
| `appendix_paired_dpo_mmlu.py` | MMLU recovered-score breakdown sweeps | `.../ocean/{trait}/{direction}/vanton4_paired_dpo/evals/mcq/mmlu/` |
| `appendix_paired_dpo_judge.py` | LLM-judge score sweeps | `.../ocean/{trait}/{direction}/vanton4_paired_dpo/evals/llm_judge_lora_scale_sweep/` |
| `appendix_dpo_methods.py` | DPO-method comparison grid (vanton4 / v4 / v4_reversed_dpo / v4_paired_dpo) on N↓ | `.../ocean/neuroticism/suppressor/{version}/evals/{mcq/trait_logprobs, mcq/mmlu}/` |

## Running

Each script has no CLI args — it runs its full figure set from `main()`:

```bash
uv run python scripts/figures/main_ocean_scaling.py
```

Requires network access to the HF monorepo (and `HF_TOKEN` for gated reads).
Figures are written under `paper/figures/` per the script's `PAPER_FIGURES`.

> **Verification status:** migrated scripts compile and resolve all imports
> against `src/`. A full regeneration run (fetching HF data and diffing the
> output PDFs against the current paper figures) is part of the end-of-refactor
> verification pass, not yet done.
