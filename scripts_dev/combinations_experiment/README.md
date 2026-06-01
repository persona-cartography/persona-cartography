# OCEAN LoRA-combination TRAIT + MMLU experiment

Measures how **combinations** of the OCEAN persona LoRAs (Llama-3.1-8B-Instruct,
`vanton4_paired_dpo`) trade off **trait transfer** against **capability (MMLU)**,
to learn which scale/direction combinations stay usable.

## Design

**32 = 2⁵ configs** — the full factorial over directions. Every config uses all
five traits, each set to its **amplifier (+)** or **suppressor (−)** adapter
(never both for one trait), so each trait is balanced 16×(+) / 16×(−).

Scales are randomized with two decoupled knobs (see `config_design.py`):

- **Total (Σscale)** per config is targeted to tile **[0.5, 4.0]** smoothly
  (stratified sampling), then totals are randomly permuted across the sign
  patterns so magnitude is decorrelated from direction.
- **Split** across the five traits is a `Dirichlet(α=1)` draw (uniform over the
  simplex), per-adapter scale ∈ **[0.05, 2.0]** (draws exceeding 2.0 are redrawn,
  preserving the total).

Everything is seeded (`SEED=42`) → fully reproducible.

Per-config slug (fixed O,C,E,A,N order; `P`/`M` = amplifier/suppressor; `p` =
decimal point): `OP0p50_CM1p23_EP0p30_AP0p80_NM0p40`.

## Files

| File | Purpose |
|---|---|
| `config_design.py` | Pure, deterministic generator of the 32 configs (NumPy only — no GPU). `python -m ... .config_design` prints the design + checks invariants. |
| `run_experiment.py` | Builds one `SuiteConfig` per config (5 adapters loaded together), runs `personality_trait_logprobs` + `mmlu` once each, uploads results to HF. |
| `analyze.py` | Starter analysis: pulls results from HF into a tidy DataFrame + two starter plots. |

## Runs are independent

Each config has a unique `run_name` (its slug) → its own local dir and HF path,
so no cached result from one config is ever reused by another. The suite's
auto-upload is disabled; uploads are done explicitly to avoid the shared
base-model baseline dir leaking into per-config outputs.

## Commands

```bash
# Inspect the design (no GPU):
uv run python -m scripts_dev.combinations_experiment.config_design

# Dry-run (print selected configs, no model load):
uv run python -m scripts_dev.combinations_experiment.run_experiment --dry-run

# Smoke test one config (tiny samples, no upload):
uv run python -m scripts_dev.combinations_experiment.run_experiment --shard 0/32 --smoke --no-upload

# Full run, sharded across 4 GPUs (one process per GPU; stagger launches):
CUDA_VISIBLE_DEVICES=0 uv run python -m scripts_dev.combinations_experiment.run_experiment --shard 0/4
CUDA_VISIBLE_DEVICES=1 uv run python -m scripts_dev.combinations_experiment.run_experiment --shard 1/4
CUDA_VISIBLE_DEVICES=2 uv run python -m scripts_dev.combinations_experiment.run_experiment --shard 2/4
CUDA_VISIBLE_DEVICES=3 uv run python -m scripts_dev.combinations_experiment.run_experiment --shard 3/4

# Analyze (download from HF, write CSV + starter plots):
uv run python -m scripts_dev.combinations_experiment.analyze
```

## HF layout

```
combinations_experiments/llama-3.1-8b-it/ocean/vanton4_paired_dpo/
    _experiment_manifest.json          # all 32 configs, seed, design params
    OP0p50_CM1p23_EP0p30_AP0p80_NM0p40/
        manifest.json                  # this config: slugs, scales, Σ, pattern
        trait_logprobs/                # run_info.json + native/inspect_logs/*.json
        mmlu/
    ... (×32)
```

## Cost note

Each config reloads base + 5 adapters (~30–60 s/load); 32 × (1500 trait + 300
MMLU items, batch 128, temp 0) is a few hours on one GPU. Use `--shard i/n` to
parallelize across GPUs.
