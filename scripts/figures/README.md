# scripts/figures

Runnable paper-figure scripts. Each reads eval artifacts from the HuggingFace
monorepo (`persona-shattering-lasr/monorepo`) and writes publication figures
into `paper/figures/` (see `PAPER_FIGURES` at the top of each script for exact
outputs). These are the migrated, `src/`-only versions of the paper figure
scripts; the `src_dev/visualisations/paper_*.py` originals are kept untouched.

Figure logic and plotting are **behaviour-preserving** relative to the dev
originals (byte-for-byte identical figures): imports were repointed to `src/`
(the Slice 2 evals library and Slice 1a `hf_hub`), shared machinery was
factored into `src/visualisations/` helpers (`combo_delta.py`,
`heatmap_common.py`, `appendix_sweep_common.py`), and the paired-DPO method
identifier was renamed to `ocean_const_paired_dpo` (from its old
`vanton4`-prefixed name) everywhere it names a regenerated artifact. The
historical bare `vanton4` and
the bespoke `v4` / `v4_reversed_dpo` / `v4_paired_dpo` identifiers still name
frozen legacy data and are left untouched.

## Scripts

| Script | Produces | Source data (HF monorepo) |
|--------|----------|---------------------------|
| `main_ocean_scaling.py` | OCEAN-scaling main figure — trait-logprob, MMLU, and LLM-judge scores vs LoRA scale (o+) | `fine_tuning/.../ocean/openness/amplifier/vanton4/evals/{mcq/trait_logprobs, mcq/mmlu, llm_judge_lora_scale_sweep}` |
| `appendix_paired_dpo_trait.py` | Trait-logprob sweeps for all OCEAN amplifiers/suppressors + control | `.../ocean/{trait}/{direction}/ocean_const_paired_dpo/evals/mcq/trait_logprobs/` |
| `appendix_paired_dpo_mmlu.py` | MMLU recovered-score breakdown sweeps | `.../ocean/{trait}/{direction}/ocean_const_paired_dpo/evals/mcq/mmlu/` |
| `appendix_paired_dpo_judge.py` | LLM-judge score sweeps | `.../ocean/{trait}/{direction}/ocean_const_paired_dpo/evals/llm_judge_lora_scale_sweep/` |
| `appendix_dpo_methods.py` | DPO-method comparison grid (vanton4 / v4 / v4_reversed_dpo / v4_paired_dpo) on N↓ | `.../ocean/neuroticism/suppressor/{version}/evals/{mcq/trait_logprobs, mcq/mmlu}/` |

### Combination / soup figures

LoRA-combination (adapter-soup) paper figures — 5×5 score heatmaps and per-trait
Δ-vs-baseline bar charts for trait pairs. These import the migrated evals library
plus `cell_identity` (`AdapterSpec` / `CanonicalCell`). The combo-delta scripts
share their hydration / Δ-computation / bar-drawing machinery via
`src/visualisations/combo_delta.py`; the soup heatmaps share judge-file
hydration + scoring via `src/visualisations/heatmap_common.py`. The
`_paired_dpo`-suffixed combo scripts use the `ocean_const_paired_dpo` adapters;
`main_c_e_combo_delta.py` (no suffix) uses the bespoke `v2`/`v3` adapters.

| Script | Produces | Source data (HF monorepo) |
|--------|----------|---------------------------|
| `main_c_e_soup_heatmaps.py` | 5×5 C↓×E↓ adapter-soup score heatmaps (bare `vanton4` data) | `combos/llama-3.1-8b-it/.../judge_runs/...` |
| `main_o_n_soup_heatmaps.py` | 5×5 O↑×N↑ adapter-soup score heatmaps | `evals/heatmaps_o_n/.../judge_runs/...` |
| `main_c_e_combo_delta.py` | C↓×E↑ combo Δ-vs-baseline bar chart (bespoke `v2`/`v3`) | `combos/.../judge_runs/qwen3_235b/{trait}_v2.jsonl` |
| `main_c_e_combo_delta_paired_dpo.py` | C↓×E↓ combo Δ bars (`ocean_const_paired_dpo`) | `combos/.../judge_runs/...` |
| `main_o_n_combo_delta_paired_dpo.py` | O↑×N↑ combo Δ bars (`ocean_const_paired_dpo`) | `combos/.../judge_runs/...` |
| `main_c_minus_e_plus_combo_delta_paired_dpo.py` | C↓×E↑ combo Δ bars (`ocean_const_paired_dpo`) | `combos/.../judge_runs/...` |

## Running

Most scripts run their full figure set from `main()` with no args; the
combo-delta scripts also accept optional `--headroom` / `--ceiling` flags:

```bash
uv run python scripts/figures/main_ocean_scaling.py
```

Requires network access to the HF monorepo (and `HF_TOKEN` for gated reads).
Figures are written under `paper/figures/` per the script's `PAPER_FIGURES`.

> **Verification status:** migrated scripts compile and resolve all imports
> against `src/`. A full regeneration run (fetching HF data and diffing the
> output PDFs against the current paper figures) is part of the end-of-refactor
> verification pass, not yet done.
