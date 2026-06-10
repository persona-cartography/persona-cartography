# Evals — run surface

Two eval surfaces, one per subdirectory. Everything runs through the single
front door — `python -m src.evals adapter-sweep` — parameterized by flags; the
`run_*.sh` launchers here just loop it over the canonical slug set.

```
scripts/evals/
├── mcq/                  TRAIT-logprob + MMLU multiple-choice evals
│   └── run_*.sh
└── llm_judge_sweep/      OCEAN (−4…+4) + coherence (0…10) LLM-judge sweeps
    └── run_*.sh
```

There are no per-eval config files: every config is **synthesized by name**
from the family defaults in `src.evals.llm_judge_sweep.config_builders`
(judge: base model, scale grid, rollout params, judge raters — formerly the
per-family `_shared.py`) and `src.evals.mcq_builders` (MCQ: sample counts,
scale grids, adapter refs). The runners live under `src/evals/` too
(`mcq_builders`/the Inspect suite for MCQ, `llm_judge_sweep.runner_cells` for
judge sweeps). Reproducibility is carried by each run's uploaded provenance
(`run_info.json` / `cell_info.json` / `baseline_info.json`: git hash, full
eval spec, fingerprint), not by config files.

**The two "families"** are the two things compared in the paper:

- `ocean_const_paired_dpo` — the trained **LoRA adapters** (paired-teacher
  DPO → SFT → soup); the default `--mode lora`.
- `*_activation_capping` — the **no-LoRA baseline** that steers the base model
  by capping its activation projection onto a persona axis (see
  `scripts/activation_capping/`); select with `--mode capping`.

## Running

```bash
# One eval, canonical defaults:
python -m src.evals adapter-sweep --eval-type trait --slug o_plus
python -m src.evals adapter-sweep --eval-type mmlu  --slug n_minus
python -m src.evals adapter-sweep --eval-type judge --slug o_plus --allow-custom-fingerprint

# Cross-trait ("bleed-through") judge sweep — rollouts are generated on the
# judged trait's prompts (NOT the same as --judge-metrics, which only adds
# judges over the adapter's own-prompt rollouts):
python -m src.evals adapter-sweep --eval-type judge --slug o_plus_on_neuroticism --allow-custom-fingerprint

# Recipe-only control adapter:
python -m src.evals adapter-sweep --eval-type judge --slug control_s1vs2 --allow-custom-fingerprint

# Activation capping (axis fractions instead of LoRA scales; --version unsupported):
python -m src.evals adapter-sweep --mode capping --eval-type trait --slug a_minus

# Batched canonical sets (one shard per GPU for the judge launchers):
bash scripts/evals/mcq/run_ocean_const_paired_dpo_sweeps.sh
bash scripts/evals/mcq/run_activation_capping_sweeps.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/evals/llm_judge_sweep/run_ocean_const_paired_dpo.sh \
    o_plus o_minus c_plus c_minus e_plus e_minus a_plus a_minus n_plus n_minus
CUDA_VISIBLE_DEVICES=0 bash scripts/evals/llm_judge_sweep/run_ocean_const_paired_dpo_activation_capping.sh \
    o_plus o_minus c_plus c_minus e_plus e_minus a_plus a_minus n_plus n_minus
```

Overrides: `--version` (score a non-default adapter, e.g. a `..._test` smoke
adapter), `--samples`, `--scales`, `--num-rollouts`, `--judge-metrics` (e.g.
`ocean5`), `--no-coherence`. `--samples` units differ by eval: **per-trait**
for `trait` (×5 splits), **total** for `mmlu`, **total prompts** for `judge`.
Any override produces a non-canonical fingerprint (no cache sharing with the
canonical sweeps); the drift prompt explains this and
`--allow-custom-fingerprint` acknowledges it. See
`python -m src.evals adapter-sweep --help` for the full flag set.

The training+eval orchestrator (`scripts/pipelines/run_persona_pipeline.sh`)
drives this same front door, so the adapter a run just trained is scored by
the same path.

Bespoke one-off configs (multi-adapter combos, paper-figure cell grids) can
still be written as Python modules anywhere importable and run via
`python -m src.evals.llm_judge_sweep.runner_cells --config <dotted.path>`
(or `--judge-config-package` on adapter-sweep). The historical examples
(`o_plus_x_n_plus_on_*`, `paper_fig1_combo_cells`, `gemma_needs_help_*`) live
in git history (removed 2026-06-10 on `anton/runpod-spinup`).

Results from all three eval surfaces land on the monorepo and feed the figures
in `scripts/visualisations/` (see `scripts/visualisations/README.md`).
