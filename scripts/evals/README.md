# Evals — run surface

Two eval surfaces, one per subdirectory. Each is **configs-as-Python**: a config
module declares *what* to evaluate (which adapter, the scale grid, the eval
spec); a runner executes it. Pick a single config, or run the batched `run_*.sh`
launcher that loops over the whole canonical set.

```
scripts/evals/
├── mcq/                  TRAIT-logprob + MMLU multiple-choice evals
│   ├── run_*.sh
│   └── configs/{trait,mmlu}/{ocean_const_paired_dpo,activation_capping}/<direction>.py
└── llm_judge_sweep/      OCEAN (−4…+4) + coherence (0…10) LLM-judge sweeps
    ├── run_*.sh
    └── configs/{ocean_const_paired_dpo,ocean_const_paired_dpo_activation_capping}/<...>.py
```

The runners themselves live under `src/evals/` (the MCQ suite builder in
`src.evals.mcq_builders`, the judge runner in
`src.evals.llm_judge_sweep.runner_cells`); only the per-eval **config modules**
and the batched `run_*.sh` launchers live here. A single front door —
`python -m src.evals adapter-sweep --eval-type {trait,mmlu,capping,judge}
--slug … --version … --samples …` — can run any eval type from flags (see
[the unified runner](#unified-runner-adapter-sweep)); the explicit config
modules remain the canonical, reproducible set.

**The two config "families"** (the leaf dir under `configs/`) are the two things
being compared in the paper:
- `ocean_const_paired_dpo` — the trained **LoRA adapters** (paired-teacher DPO → SFT → soup).
- `activation_capping` — the **no-LoRA baseline** that steers the base model by
  capping its activation projection onto a persona axis (see
  `scripts/activation_capping/`).

## `mcq/` — TRAIT logprobs + MMLU

One config per OCEAN direction (`o_plus`, `o_minus`, … `n_minus`, plus controls),
under `configs/<eval>/<family>/`, where `<eval>` is `trait` (single-token
forced-`ANSWER:` logprob scoring) or `mmlu` (capability). A config builds a
`SuiteConfig` (adapter pulled from the monorepo + a scale grid + the eval spec)
and is run through the Inspect suite:

```bash
# one config
python -m src.evals suite --config-module \
  scripts.evals.mcq.configs.trait.ocean_const_paired_dpo.n_plus_ocean_const_paired_dpo

# the whole canonical set (both trait + mmlu, every direction)
bash scripts/evals/mcq/run_ocean_const_paired_dpo_sweeps.sh
bash scripts/evals/mcq/run_activation_capping_sweeps.sh
```

## `llm_judge_sweep/` — OCEAN + coherence judges

Generates multi-turn rollouts at each scale and scores every assistant turn with
calibrated LLM judges (OCEAN −4…+4, coherence 0…10). Configs under
`configs/<family>/` include per-direction sweeps (`n_plus.py`), cross-trait
judging (`n_plus_on_openness.py`), combinations (`o_plus_x_n_plus_*`), and a
control. They run through the shared `src.evals.llm_judge_sweep.runner_cells`
runner:

```bash
# one config
python -m src.evals.llm_judge_sweep.runner_cells --config \
  scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo.n_plus

# the whole canonical set
bash scripts/evals/llm_judge_sweep/run_ocean_const_paired_dpo.sh
bash scripts/evals/llm_judge_sweep/run_ocean_const_paired_dpo_activation_capping.sh
```

## Unified runner (`adapter-sweep`)

The static config modules above are the canonical, reproducible record. For
**parameterized / ad-hoc** runs — scoring a non-default adapter version (e.g. a
`..._test` smoke adapter), or capping the sample count — there is one front door
that builds the config from flags and routes to the right backend:

```bash
# TRAIT-logprob MCQ, a test adapter version, 10 samples/trait
python -m src.evals adapter-sweep --eval-type trait --slug n_plus \
  --version ocean_const_paired_dpo_test --samples 10

# MMLU under activation-capping mode (axis fixed; --version unsupported)
python -m src.evals adapter-sweep --eval-type mmlu --mode capping --slug n_plus

# LLM-judge sweep — judge config modules live here in scripts/, so pass the
# package (keeps src/ free of any scripts/ path):
python -m src.evals adapter-sweep --eval-type judge --slug n_plus \
  --judge-config-package scripts.evals.llm_judge_sweep.configs \
  --version ocean_const_paired_dpo_test --samples 40 --dry-run
```

`--samples` units differ by eval: **per-trait** for `trait` (×5 splits),
**total** for `mmlu`, **total prompts** for `judge`. See
`python -m src.evals adapter-sweep --help` for the full flag set. The
training+eval orchestrator (`scripts/pipelines/run_persona_pipeline.sh`) drives
this front door, so the adapter a run just trained is scored by the same path.

Results from all three eval surfaces land on the monorepo and feed the figures in
`scripts/visualisations/` (see `scripts/visualisations/README.md`).
