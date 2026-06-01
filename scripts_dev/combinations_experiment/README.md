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
| `run_experiment.py` | Shared-base runner: loads the base 8B + all adapters + both tasks **once**, then per config re-activates/re-scales its 5 adapters and runs `personality_trait_logprobs` + `mmlu`, uploading results to HF. |
| `analyze.ipynb` | Starter notebook: downloads results, aggregates each config to mean + ci95 error bars (Wilson for MMLU, bootstrap for trait), plots the 32 points vs Σscale. |

## Runs are independent

Each config writes to its own slug-named local dir and HF path, and on the
resident model each config sets its active-adapter set and resets every active
adapter's scaling from a captured baseline — so no state leaks between configs.
Resume is handled by `_config_done_on_hf` (checks HF before loading the model);
it keys on the Inspect log status, so a partial/OOM'd eval is correctly re-run.

## Commands

```bash
# Inspect the design (no GPU):
uv run python -m scripts_dev.combinations_experiment.config_design

# Dry-run (print selected configs, no model load):
uv run python -m scripts_dev.combinations_experiment.run_experiment --dry-run

# Smoke test one config (tiny samples, no upload) — validates the full path fast:
uv run python -m scripts_dev.combinations_experiment.run_experiment --only <slug> --smoke --no-upload

# Full run — single process, all 32; resume skips finished configs:
uv run python -m scripts_dev.combinations_experiment.run_experiment

# Analyze: open the starter notebook (downloads results, plots vs Σscale):
scripts_dev/combinations_experiment/analyze.ipynb
```

`--shard i/n` still splits the configs (e.g. across **separate** GPUs, one
process each). Do **not** run multiple shards on one GPU — two 8B copies + MMLU
generation OOMs an 80 GB card.

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

The base model, all adapters, and both tasks load once (~1–2 min); each config
is then just inference (1500 trait + 300 MMLU items, batch 128, temp 0) — no
per-config reload. The OCEAN adapters are high-rank (~670 MB each), so the
one-time adapter prefetch is ~7 GB.
