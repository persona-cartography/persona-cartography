# Persona Shattering via LoRA Fine-tuning

Extract and transfer personality traits into LLMs through targeted LoRA fine-tuning, and study the resulting adapter geometry via LoRA arithmetic.

## Hardware Requirements

Developed and tested on:
- **VM**: `gpu_1x_gh200` (NVIDIA GH200 480GB)
- **Architecture**: ARM64 (aarch64)
- **Python**: 3.11
- **PyTorch**: System-provided torch with CUDA (not installed via pip/uv due to ARM64 wheel availability)

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/root/.local/bin:$PATH"
uv sync
```

> **Note**: On the GH200 VM, torch is provided by the system. After `uv sync`, remove the venv torch and torchvision to use the system CUDA-enabled version:
> ```bash
> rm -rf .venv/lib/python3.11/site-packages/torch*
> rm -rf .venv/lib/python3.11/site-packages/torchvision*
> ```

Create your environment file:

```bash
cp .env.example .env
# fill in ANTHROPIC_API_KEY, HF_TOKEN, WANDB_API_KEY
```

### OpenCharacterTraining deps (only for the training pipeline)

The OCT training pipeline needs two extra packages (`character`, `openrlhf`)
that `uv sync` cannot install — their repos use SSH git submodules that uv
clones recursively and fails on. Install them after `uv sync` with:

```bash
make oct-deps   # wraps scripts/setup/install_oct_deps.sh
```

This is not needed for the paired-DPO dataset-build step or for evals/figures —
only for teacher/student generation and DPO/SFT training.

## Project Overview

The pipeline runs in four stages:

1. **Inference** — generate responses from `meta-llama/Llama-3.1-8B-Instruct`
2. **Editing** — rewrite responses via a stronger LLM (Claude) to amplify or suppress a trait
3. **Training** — LoRA fine-tune on the edited responses
4. **Evaluation** — measure how well the trait transferred

Trained adapters are then analysed for their geometric properties (subspace alignment, rank structure) via the LoRA arithmetic utilities in `src/utils/`.

## Personas

Personas are registered in `src_dev/common/persona_registry.py`:

| Persona | Direction | Evaluation | Notes |
|---------|-----------|------------|-------|
| `o_avoiding` | +/- | `count_o` | Toy baseline (letter 'O' frequency) |
| `verbs_avoiding` | +/- | `verb_count` | Verb density |
| `sf_guy` | +/- | `lowercase_density`, `punctuation_density` | Casual texting style |
| `n+_persona` | + | `emotional_instability` | Neuroticism (high) |
| `n-_persona` | - | `emotional_instability` | Neuroticism (low) |

> **Note**: `emotional_instability` is currently a placeholder (returns 0). The trained adapters (`n+_persona` r16, `n+_persona_r4`, `n-_persona_r4`) are in `scratch/checkpoints/`.

## Directory Structure

The codebase is organised into four layers with a clear graduation path:

| Path | Purpose | Graduation target |
|------|---------|-------------------|
| `src/` | Stable, well-tested library code (interfaces, LoRA arithmetic) | — |
| `src_dev/` | In-development components (inference, editing, training, evals, metrics, datasets) | `src/` |
| `scripts/` | Finalised pipeline scripts (training runs, eval suites, etc.) | — |
| `scripts_dev/` | Experimental scripts and research explorations | `scripts/` |

Supporting directories:

| Path | Purpose |
|------|---------|
| `data/` | Static input datasets |
| `dump/` | Archived exploratory notebooks |
| `scratch/` | Experiment outputs: checkpoints, JSONL datasets, W&B artefacts (gitignored) |
| `tests/` | Unit tests |

## Canonical Datasets Format

Use the repository's canonical datasets format for new module code and experiment scripts.

- Prefer `src_dev/datasets` utilities for loading/formatting/exporting data (`load_dataset_from_config`, `format_for_inference`, canonical run-dir helpers).
- Keep canonical fields and lineage intact when adding experiment-specific metadata.
- Avoid introducing ad-hoc JSONL schemas when the canonical format can represent the data.

## Component Reference

Components live in `src_dev/` and follow a Config → Run → Result pattern.

### Inference (`src_dev/inference`)
- `run_inference(config, dataset=None)` — runs local HF inference or remote API
- Providers: `local`, `openai`, `openrouter`, `anthropic`

### Editing (`src_dev/lora_pipeline_persona_shattering/editing`)
- `run_editing(config, dataset=None)` — rewrites responses to exhibit/inhibit a trait
- Providers: `anthropic`, `openai`

### Training (`src_dev/lora_pipeline_persona_shattering/training`)
- `run_training(config)` — LoRA fine-tuning via HF Trainer + PEFT

### Persona Metrics (`src_dev/persona_metrics`)
- `run_persona_metrics(config, dataset=None)` — per-response metric scoring on a dataset
- Built-ins: `count_o`, `verb_count`, `coherence`, `lowercase_density`, `punctuation_density`

### Evals (`src_dev/evals`)
- `run_evals(config, dataset=None)` — end-to-end eval suites across model targets
- Suites: `persona_metrics`, `inspect_task` (e.g. `mmlu`)
- Supports base and LoRA model targets

### Visualisations (`src_dev/visualisations`)
- Plotting and LoRA analysis helpers
- Browser local chat entrypoint: `src_dev/visualisations/local_chat.py`
- See [`src_dev/visualisations/README.md`](src_dev/visualisations/README.md) for setup details

### LoRA Arithmetic (`src/utils`)

`src/utils/peft_manipulations.py` provides reversible, in-place LoRA modifiers:

| Class | What it does |
|-------|-------------|
| `LoRaRankReducer` | Truncated-SVD rank reduction |
| `LoRaScaling` | Multiplicative scaling of adapter contribution |
| `LoRaAdapterZeroing` | Zeros out selected layers/modules |
| `LoRaPipeline` | Composes multiple modifiers in sequence |

`src/utils/linalg.py` provides `reduce_lora_rank_efficient` — memory-efficient rank reduction via QR + SVD on the small (r×r) core rather than the full weight matrix.

All modifiers snapshot state at init and `restore()` is idempotent.

## Default LoRA Training Config

| Parameter | Value |
|-----------|-------|
| `r` | 16 (r4 variants use 4) |
| `lora_alpha` | 32 (r4: 8) |
| `lora_dropout` | 0.05 |
| `target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

Training logs loss, eval metrics, and sample generations to Weights & Biases.

## Paper

The LaTeX paper source is in `paper/`. See `paper/CLAUDE.md` for full conventions.

```bash
# Install LaTeX (Ubuntu/Debian; macOS: brew install --cask mactex-no-gui)
apt-get update && apt-get install -y texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra

# Build the paper
cd paper && make          # Full build -> paper/main.pdf
cd paper && make quick    # Fast single-pass (no bibliography)
```

## For Developers

See [AGENTS.md](AGENTS.md) for coding guidelines and architecture overview.
