# Agent Guidelines

Instructions for coding agents working on this project.

---

## Cost Management: Switch Models for Dry Runs

If you (the coding agent) are running on **a very strong / expensive model** (e.g. Claude Opus) and are about to do **dry-run / smoke-test work** on this repo — running `--dry-run`, verifying imports, exercising CLIs to confirm plumbing, or any session that is not actually executing the research pipeline — **switch to a cheaper model** (e.g. `z-ai/glm-4.7-flash`) before starting. Top-tier-model tokens add up quickly across many iteration cycles.

After the dry-run / wiring work is done, **switch back to the stronger model** for the real task (full end-to-end runs, non-trivial implementation, debugging hard failures). The rule only targets the cheap, repetitive plumbing phase.

---

## Project Overview

This project is part of an active AI safety and interpretability research effort. The team is studying how **personality traits and behavioral personas embed in large language models**, with the goal of understanding the mechanisms of LLM behavior at a fine-grained level.

### Research Mission

We are investigating whether, and how, **psychometrically-established personality traits** can be surgically transferred into LLMs via LoRA fine-tuning — and what the resulting adapter geometry reveals about how personality is represented inside neural networks.

The primary research target is real personality traits as defined in the psychometrics literature. Our current working framework is the **OCEAN model** (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism), which is well-established and has validated measurement instruments. Other frameworks may be explored as the research develops.

Core research questions:
- Can we reliably transfer a psychometric personality trait (e.g. high Agreeableness, high Neuroticism) into a model by fine-tuning a LoRA adapter on trait-amplified examples?
- Do persona adapters have interpretable geometric structure? Are they low-rank? Do they span predictable subspaces?
- Can multiple persona adapters be composed additively? If we blend adapter A + adapter B, do we get predictable trait combinations?
- How much of a trait is preserved when we aggressively reduce the adapter's rank?
- Longer term: can we discover novel, non-human personas unsupervised — finding behaviorally coherent directions in adapter space that don't correspond to any human-defined trait category?

### Toy Models and Development Proxies

Current work makes heavy use of **toy behaviors** — simple, artificial traits like avoiding the letter 'o', overusing the letter 'p', low verb density, or casual texting style. These are **not the end goal** of the research. They serve as convenient development and validation proxies because:
- They have simple, objective evaluation metrics (no LLM judge needed)
- Transfer success is easy to verify unambiguously
- They help validate the pipeline and analysis tools before applying them to harder-to-measure psychological traits

When you see toy personas in the codebase (e.g. `o_avoiding`, `sf_guy`), treat them as scaffolding. The research ambition is the OCEAN traits and similar psychometric constructs.

### Codebase Structure and Goals

The codebase is a collection of **reusable research components** for studying persona transfer in LLMs. The components are designed to be composed flexibly as the research evolves — the architecture deliberately avoids a rigid pipeline.

Current components include:
- **Inference / Editing / Training** — the current method for producing persona-bearing LoRA adapters (generate baseline responses, rewrite them with trait amplification via a strong LLM, fine-tune a LoRA on the result)
- **Persona metrics and evals** — measuring trait transfer quality, including LLM judges and psychometric instruments
- **LoRA arithmetic** — rank reduction (SVD), adapter scaling, layer zeroing, multi-adapter composition — for probing the geometry of persona adapters

The inference→editing→training workflow is *one* way to produce LoRAs and may evolve or be supplemented. Components should be built to be useful independently, not just as steps in that sequence.

### Research Context

This work sits at the intersection of:
- **LLM fine-tuning methodology** (LoRA, SFT, data curation pipelines)
- **Mechanistic interpretability** (understanding what adapter weight matrices encode)
- **AI alignment** (understanding how behavioral traits propagate, compose, and persist through training)

When working on this project, keep this research framing in mind. If a task seems ambiguous, think about what would actually advance the research: Does this change make the experiments more rigorous? Does it expose something new about adapter geometry? Does it make the components more useful for studying new traits, in a scientifically rigorous way? Ask if unsure.

---

## Architecture Overview

This project has a **stable layer** and an **in-development layer**:

- Stable: `src/` — final, stable code
- In development: `src_dev/` — code that should eventually move to `src/`
- Experiment scripts: `scripts_dev/` — experiment scripts under development

### Import Boundary Rules (Critical)

- Code in `src/` must not import from `src_dev/` or `scripts_dev/`.
- Code in `src_dev/` may import from `src/`.
- Code in `scripts_dev/` may import from `src_dev/` and `src/`.

If reusable logic appears in experiment scripts, move it into `src_dev/`, so it can be checked thoroughly and then eventually moved to an appropriate place in `src/`.

### Key Principles

1. **Configs in Python, not YAML** - Experiment scripts define their own configuration
2. **Components are composable** - Pass datasets between stages
3. **No pipeline orchestrator** - Scripts call components directly
4. **Use canonical datasets format** - New module code and experiment scripts should read/write through `src_dev.datasets` canonical dataset tooling, not ad-hoc JSONL schemas or custom column conventions

### Canonical Dataset Requirement (Critical)

- Treat the repository's canonical dataset format as the default contract for research data flow.
- For loading/normalization, prefer `src_dev.datasets.load_dataset_from_config(...)` and `src_dev.datasets.format_for_inference(...)`.
- For run-dir lineage/event-backed data, use canonical helpers in `src_dev.datasets` (e.g. `ingest_source_dataset`, `materialize_canonical_samples`, `export_dataset`) instead of bespoke file IO.
- If an experiment needs extra fields, keep canonical fields intact and add metadata in a backward-compatible way rather than replacing the schema.
- Do not introduce new one-off dataset formats when the canonical format can represent the same data.

### Inspect Evaluations: Use Upstream Unchanged (Critical)

- For any eval that comes from `inspect_evals` (e.g. `agentic_misalignment`, or other benchmarks imported from the installed package), **use the upstream version as-is**. Do not edit, monkey-patch, or locally fork classifier/scorer/task code from the `.venv/.../inspect_evals/...` tree.
- Comparability across runs, teams, and external results depends on every row being produced by the same upstream definitions (classifiers, regex gates, prompt templates, etc.).
- If upstream behavior is genuinely wrong or missing something you need:
  1. Configure around it via the benchmark's public parameters (e.g. `grader_model`, `epochs`, scenario args).
  2. If that's not enough, open an issue / PR upstream rather than modifying the local install.
  3. Only as a last resort, add a *wrapper* in `src_dev/evals/...` that composes the upstream eval without mutating it — never replace the upstream module.
- When asked "did you write this eval code?", the default answer for anything under `.venv/.../inspect_evals/` is **no**, and it should stay that way.

---

## Code Reuse and Duplication Prevention (Critical)

Multiple team members work on this codebase concurrently, each often using AI coding agents. Without discipline, this leads to duplicated logic, overlapping utilities, and a codebase that becomes harder to maintain over time. **Reducing duplication is a first-class goal.**

### Before Writing Code, Search First

Before writing any new function, utility, or module, **search the codebase** to understand what already exists. Check `src_dev/`, `src/`, and `scripts_dev/` for code that does something similar or related. This is a prerequisite, not optional.

### Report Similar Code to the User

When you find existing code that overlaps with the current task, **you must explicitly tell the user** what you found before proceeding:
- File paths and function/class names
- What the existing code does and how it relates to the current task
- Whether it can be reused directly, extended, or needs modification

The user decides whether to reuse/extend existing code or write new code. Do not silently write a parallel implementation.

### Prefer Extending Over Writing New

If something in `src_dev/` almost does what's needed, **modify it to be more general** rather than writing a new parallel implementation. `src/` should still be changed carefully and only when the abstraction is proven — but `src_dev/` code can and should be refactored to avoid duplication.

When modifying existing `src_dev/` code, **do not introduce breaking changes** to existing call signatures. Add parameters with defaults, extend behavior — don't change existing interfaces that other team members may depend on.

### Write General, Call Narrow

New code should be written to work generally — parameterized, flexible, reusable — and then **called in the specific way** needed for the current task. Avoid rigid single-purpose implementations that only work for one exact use case.

### Minimize Total Code

The goal is **less code overall**. Produce the minimum code needed to accomplish the task. However, if refactoring or generalizing existing code would **reduce total duplication** and increase robustness across the codebase, that refactoring is worthwhile even if it means touching more files.

### When to Write New Code

If existing code serves a fundamentally different purpose, write new code — don't force unrelated modules to share an abstraction just to avoid a new file. The goal is to eliminate *accidental* duplication, not to merge everything into one place.

### Flag Duplication Proactively

If you notice multiple scripts or modules doing similar things — even if not directly related to the current task — **flag this to the user**. The team wants to know about duplication so it can be addressed, rather than letting it silently accumulate.

### Flag Issues Before Implementing

When you notice data issues, edge cases, or design ambiguities (e.g. missing data categories, structural inconsistencies, unclear requirements), **raise them with the user before writing code**. Do not implement a workaround or make an assumption and mention it after the fact.

---

## Directory Structure

| Directory      | Purpose                                                    | Git Status     |
| -------------- | ---------------------------------------------------------- | -------------- |
| `src/`         | Final, stable interfaces and base classes                  | Committed      |
| `src_dev/`     | In-development components (should eventually move to src/) | Committed      |
| `scripts_dev/` | Experiment scripts under development                       | Committed      |
| `scratch/`     | Experiment outputs                                         | **Gitignored** |

---

## Component Pattern

Each component module (usually under `src_dev/`) should export:
- **Config class** - Pydantic model for settings
- **Run function** - `run_<component>(config, dataset=None) -> (dataset, result)`
- **Result class** - Metadata about the run

Example:
```python
from src_dev.inference import run_inference, InferenceConfig
from src_dev.common.config import DatasetConfig

config = InferenceConfig(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    dataset=DatasetConfig(source="huggingface", name="vicgalle/alpaca-gpt4", max_samples=10),
)
dataset, result = run_inference(config)
```

---

## Creating Experiments

1. For exploratory or temporary work, create scripts in `scripts_dev/`.
2. When stable, reusable logic should be moved into `src_dev/` and eventually into `src/`.
3. Define configs as Python objects and pass datasets between stages.
4. Write outputs to `scratch/`.

See `scripts_dev/persona_pipelines/` for examples of composing components into an end-to-end LoRA training workflow.

### Git Workflow: Branch Name In Commit Messages (Critical)

When making commits, include the current branch name at the start of the commit message. This makes the shared git history easier to scan.

Format commits like:

```text
<branch-name> <short description>
```

Example:

```text
anton/llm_judge_soup_barplots add grouped barplot labels
```

If you are asked to commit work and the current branch name is available, use it verbatim in the commit message prefix.

### Reproducibility: Always Set Seeds

Experiment scripts and notebook runners must **set random seeds at the top** before any stochastic operations. Use a single `SEED` constant and seed all relevant RNGs:

```python
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
```

This ensures results are reproducible across runs. Do not scatter seed-setting throughout the script — set all seeds once at the top.

---

## Code Style

### Imports

```python
# Standard library
from abc import ABC, abstractmethod
from pathlib import Path

# Third-party
import torch
from transformers import AutoModelForCausalLM

# Local - shared config
from src_dev.common.config import DatasetConfig, ModelConfig

# Local - components
from src_dev.editing import EditingConfig, run_editing
from src_dev.inference import InferenceConfig, run_inference
```

### Type Hints

Use type hints for function signatures:

```python
def run_inference(config: InferenceConfig, dataset: Dataset | None = None) -> tuple[Dataset, InferenceResult]:
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def run_inference(config: InferenceConfig, dataset: Dataset | None = None) -> tuple[Dataset, InferenceResult]:
    """Run LLM inference on a question dataset.

    Args:
        config: Inference configuration.
        dataset: Optional pre-loaded dataset. If None, loads from config.

    Returns:
        Tuple of (dataset with 'response' column, InferenceResult metadata).
    """
```

---

## Available Components

### src_dev.inference
- `InferenceConfig` - Model, provider, dataset, generation settings
- `run_inference(config, dataset=None)` - Generate responses
- Providers: `local`, `openai`, `openrouter`, `anthropic`

### src_dev.persona_metrics
- `PersonaMetricsConfig` - Trait measurement settings (psychometric and proxy metrics)
- `run_persona_metrics(config, dataset=None)` - Score responses for trait manifestation

### src_dev.evals
- Inspect-based benchmark/custom eval wrapper
- `list-evaluations`, `named`, `suite`, and `direct` CLI modes

### src_dev.visualisations
- Analysis/plotting scripts for eval and LoRA behavior

### src_dev.rollout_generation
- Multi-turn rollout generation with phased system prompts

### src_dev.datasets
- Canonical dataset loading, normalization, and export

### src_dev.common.config
- `ModelConfig` - HuggingFace model configuration
- `DatasetConfig` - Dataset source and sampling
- `GenerationConfig` - Text generation parameters
- `WandbConfig` - Weights & Biases logging

### src.utils
- Stable utility helpers shared across modules
- Includes linear algebra utilities, model-layer inspection, and LoRA arithmetic (rank reduction, scaling, zeroing, composition)

---

## HuggingFace Monorepo

All artifacts (adapters, eval results, stage markers) are stored in a shared HuggingFace dataset repo: **`persona-shattering-lasr/monorepo`**.

### Path structure

```
persona-shattering-lasr/monorepo/
  fine_tuning/
    {model}/                          # e.g. llama-3.1-8b-it
      {category}/                     # e.g. ocean
        {trait}/                       # e.g. extraverted, neuroticism
          {direction}/                 # e.g. amplifier, suppressor
            v{version}/               # e.g. v1
              lora/                    # LoRA adapters (dpo, sft, persona)
              data/                    # Training data (distillation, introspection)
              .oct_pipeline/stages/    # Stage completion markers
              run_info.json            # Provenance metadata
              evals/                   # Model-specific evals
                mcq/trait/             # MCQ trait sweep results
                mcq/mmlu/              # MMLU capability results
  evals/
    {eval_type}/                       # Cross-model or standalone evals
      {run_name}/                      # e.g. neuro_x_consc_combos
```

### Where evals go

- **Model-specific evals** (trait sweeps for a single adapter) → `fine_tuning/{model}/{category}/{trait}/{direction}/v{version}/evals/`
- **Cross-model evals** (multi-adapter comparisons, combo studies) → `evals/` at the top level

### Latest / best adapter per OCEAN direction

The canonical pointer to the current best persona LoRA for each OCEAN direction lives in `src_dev/common/lora_catalogue.py` as the `LoraHFCatalogue` dataclass (`o_plus`, `o_minus`, `c_plus`, `c_minus`, `e_plus`, `e_minus`, `a_plus`, `a_minus`, `n_plus`, `n_minus`). When newer / better training runs supersede old ones, this file is updated. Prefer reading adapter paths from it rather than hand-constructing monorepo paths — this keeps downstream scripts and experiments pointed at the current best adapter.

---

## Confidence Intervals

When computing confidence intervals for eval metrics, use the appropriate method for the data type:

- **Binary data (MCQ accuracy, 0/1 trait scores)** → use **Wilson score interval** (`ci_from_wilson`). Standard bootstrap and normal approximations give poor coverage when the proportion is near 0 or 1. Wilson handles this correctly.
- **Continuous data (LLM judge scores, rollout metrics)** → use **BCa bootstrap** (`ci_from_bootstrap`). Makes no distributional assumptions about the data.
- **Avoid** the naive `1.96 * std / sqrt(n)` normal approximation — it assumes normality and gives symmetric intervals that can extend below 0 or above 1 for proportions.

These methods are available via `IntervalMethod` in `src_dev/evals/personality/analyze_results.py`:

```python
from src_dev.evals.personality.analyze_results import IntervalMethod

# For binary MCQ data
wilson = IntervalMethod(method="ci_from_wilson", confidence=95)

# For continuous judge scores
bootstrap = IntervalMethod(method="ci_from_bootstrap", confidence=95, n_resamples=1000)

# Or from strings (e.g. in analyze_kwargs)
IntervalMethod.from_str("ci95_from_wilson")
IntervalMethod.from_str("ci95_from_bootstrap_1000")
```

The low-level functions `_interval_ci_from_wilson` and `_interval_ci_from_bootstrap` can also be used directly — both return `(ci_lower, ci_upper)` as absolute bounds.

---

## Environment Variables

API keys are loaded from `.env`:

```python
from dotenv import load_dotenv
load_dotenv()  # Call at start of experiment script
```

Required keys:
- `ANTHROPIC_API_KEY` - For Anthropic editing provider
- `OPENAI_API_KEY` - For OpenAI inference/editing providers
- `OPENROUTER_API_KEY` - For OpenRouter inference/evaluation providers
- `WANDB_API_KEY` - For W&B logging (optional)
- `HF_TOKEN` - For gated HuggingFace models (optional)

---

## Paper

The LaTeX paper source lives in `paper/`. See `paper/CLAUDE.md` for paper-specific instructions (build commands, file structure, conventions).

### Figure Output Convention (Critical for All Agents)

When producing plots or figures that may appear in the paper, save a copy to the appropriate subdirectory under `paper/figures/`:

- `paper/figures/overview/` — methodology diagrams
- `paper/figures/main/` — main body figures (Sections 1–3)
- `paper/figures/unsupervised/` — Section 4 figures
- `paper/figures/appendix/` — appendix figures

Naming convention: `fig_<section>_<short_name>.pdf` (or `.png` for raster).
Examples: `fig_3_3_1_1_trait_scaling.pdf`, `fig_F_1_openness_amp.pdf`.

The canonical path constant is available via:
```python
from src_dev.visualisations import PAPER_FIGURES_DIR
```

`paper/figures/tmp/` contains placeholder images extracted from the original markdown draft. These are **all** to be replaced by publication-quality figures. Once all are replaced, delete the `tmp/` directory.

**Code ↔ paper pointers (Critical).** Every plotting script that writes to `paper/figures/` must declare its outputs at the module top via a `PAPER_FIGURES = [...]` list (paths relative to `paper/figures/`), and every `\includegraphics` in the LaTeX source must carry `% Generated by:` and `% Data:` comments pointing back at the script and its HF data source. The registry mapping the two is `paper/figures/MANIFEST.md`. See `paper/CLAUDE.md` → "Code ↔ Paper Pointers" for the full convention. Update `MANIFEST.md` whenever you add, replace, or rename a figure.
