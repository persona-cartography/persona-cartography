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
    ├── runner_cells.py
    ├── run_*.sh
    └── configs/{ocean_const_paired_dpo,ocean_const_paired_dpo_activation_capping}/<...>.py
```

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
control. They run through this directory's own `runner_cells.py`:

```bash
# one config
python -m scripts.evals.llm_judge_sweep.runner_cells --config \
  scripts.evals.llm_judge_sweep.configs.ocean_const_paired_dpo.n_plus

# the whole canonical set
bash scripts/evals/llm_judge_sweep/run_ocean_const_paired_dpo.sh
bash scripts/evals/llm_judge_sweep/run_ocean_const_paired_dpo_activation_capping.sh
```

Results from all three eval surfaces land on the monorepo and feed the figures in
`scripts/visualisations/` (see `scripts/visualisations/README.md`).
