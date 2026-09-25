# Evals

Summary of the evaluations implemented in this repo: what's available, where
the code lives, and how to launch each one.

All evals run through the unified CLI at `python -m src_dev.evals`, which
wraps Inspect AI. The CLI has four subcommands:

- `list-evaluations` — list named evals in the registry
- `named` — run a pre-defined named eval on one or more model specs
- `suite` — run a Python-defined suite (adapter sweeps, multi-model)
- `direct` — one-off eval from CLI args (no Python config)

Outputs default to `scratch/evals/`.

## Benchmarks

Wrapped in `src_dev/evals/inspect_benchmarks.py` (`build_benchmark_task`):

| Canonical name | Source | Notes |
|---|---|---|
| `mmlu` | `inspect_evals.mmlu` | + `mmlu_base_model`, + `mmlu_logprobs` variant |
| `truthfulqa` | `inspect_evals.truthfulqa` | `target="mc1"`/`"mc2"`; + `truthfulqa_logprobs` |
| `gpqa` | `inspect_evals.gpqa` | diamond default; + `gpqa_logprobs` |
| `gsm8k` | `inspect_evals.gsm8k` |  |
| `popqa` | `inspect_evals.popqa` |  |
| `personality_bfi` | custom (BFI questionnaire) |  |
| `personality_trait` | custom (TRAIT benchmark) | + `_sampled`, `_logprobs`, `_logprobs_base_model` variants |
| `agentic_misalignment` | `inspect_evals.agentic_misalignment` | Anthropic blackmail scenario; LLM-judge scored |
| `sycophancy` | `inspect_evals.sycophancy` |  |
| `coconot` | `inspect_evals.coconot` | contrastive compliance / refusal |
| `mask` | `inspect_evals.mask` | honesty vs. accuracy; dataset gated (`cais/MASK`) |
| `ahb` | `inspect_evals.ahb` | Animal Harm Benchmark |

**Policy: upstream `inspect_evals` benchmarks are used as-is.** Do not edit,
monkey-patch, or locally fork classifier/scorer/task code from the installed
package. Configure via public parameters (e.g. `grader_model`, `epochs`,
scenario args), or wrap without mutating. See `CLAUDE.md` for details.

## Named evals

Registry in `src_dev/evals/evaluations.py`. List them with:

```bash
uv run python -m src_dev.evals list-evaluations
```

Current entries:

- `truthfulqa_mc1`, `truthfulqa_mc2`
- `personality_bfi`, `personality_trait`
- `coherence1`, `neuroticism1`, `neuroticism2`
- `coherence_count_o1`, `coherence_count_p1`, `coherence_o_density_lowercase_punctuation1`
- `agentic_misalignment_default`

Named custom evals combine `src_dev.persona_metrics` metrics with an
OpenAssistant / prompt-eng / no_robots dataset.

## Suite configs (persona adapter sweeps)

Adapter-sweep configs for persona LoRAs live at
`scripts_dev/personality_evals/configs/{ocean,control}/<benchmark>/<adapter>.py`.

| Benchmark | OCEAN configs | Control configs |
|---|---|---|
| `mmlu` | `ocean/mmlu/{n_plus, c_minus, a_minus, soup_n_c, vrun4/a_plus_vrun4, …}.py` | — |
| `sycophancy` | `ocean/sycophancy/{a_plus_vrun4, a_minus}.py` | `control/sycophancy/control_diff_words.py` |
| `coconot` | `ocean/coconot/{a_plus_vrun4, a_minus, n_plus_vrun4, control_use_diff_words}.py` | — |
| `mask` | `ocean/mask/{a_plus_vrun4, a_minus}.py` | `control/mask/control_diff_words.py` |
| `ahb` | `ocean/ahb/{a_plus_vrun4, a_minus_v2, c_plus_vrun4, c_minus_v2, n_plus_vrun4, n_minus_vrun4, o_plus_vrun4, o_minus_vrun4}.py` | `control/ahb/control_diff_words.py` |

Each module exports a `SUITE_CONFIG`:

```bash
uv run python -m src_dev.evals suite \
    --config-module scripts_dev.personality_evals.configs.ocean.mask.a_minus
```

## Agentic Misalignment sweep

`src_dev/evals/agentic_misalignment/` contains two suite configs:

1. **`sweep_base.py`** — base-only sweep over
   `(scenario × urgency_type × extra_system_instructions)` = 27 conditions.
   **Run this once first**, on the base model only, to identify which
   scenarios actually elicit misaligned behaviour on Llama-3.1-8B-Instruct.
   Only the scenarios that flip in the base model are worth running across
   the adapter sweep — otherwise every adapter cell is noise around 0.

   ```bash
   uv run python -m src_dev.evals suite \
       --config-module src_dev.evals.agentic_misalignment.sweep_base
   ```

2. **`config.py`** — after picking the promising scenario(s), this runs the
   base model + 11 OCEAN persona adapters on the default
   blackmail / explicit-america / replacement condition, `epochs=10`.

   ```bash
   uv run python -m src_dev.evals suite \
       --config-module src_dev.evals.agentic_misalignment.config
   ```

Scenario parameters and the adapter list live in `defaults.py`. Grader
defaults to an Anthropic/OpenRouter model — requires `ANTHROPIC_API_KEY` or
`OPENROUTER_API_KEY`, plus `HF_TOKEN` for adapter downloads.

## Frustration eval

`scripts_dev/frustration_eval/` is a standalone multi-turn frustration eval
with its own CLI (not part of the `src_dev.evals` wrapper):

```bash
uv run python -m scripts_dev.frustration_eval --help
```

Prompt set is in `prompts.py`; run logic + judge in `run_eval.py`;
local-adapter variant in `run_local_adapter.py`.

## Common launch modes

```bash
# List the registry
uv run python -m src_dev.evals list-evaluations

# Run a named eval on the base model
uv run python -m src_dev.evals named \
    --output-root scratch/evals/demo \
    --run-name demo \
    --model-spec "name=base;base_model=hf://meta-llama/Llama-3.1-8B-Instruct" \
    --evaluation truthfulqa_mc1 \
    --limit 50

# Run a scripted suite (multi-model, sweeps, adapter config)
uv run python -m src_dev.evals suite \
    --config-module scripts_dev.personality_evals.configs.ocean.mask.a_minus

# One-off eval without any Python config
uv run python -m src_dev.evals direct \
    --output-root scratch/evals/demo \
    --model-spec "name=base;base_model=hf://meta-llama/Llama-3.1-8B-Instruct" \
    --eval-kind benchmark \
    --eval-name truthfulqa_mc1 \
    --benchmark truthfulqa \
    --benchmark-arg target=\"mc1\" \
    --limit 50
```

Model specs can carry adapters: `adapters=local://path@1.0,hf://org/repo@-0.5`.

## Output & viewing

Each `(model_spec, eval_spec)` run writes:

- `run_info.json` — model / eval / judge / materialization metadata + inspect log path
- `native/inspect_logs/…` — Inspect log files

View logs locally:

```bash
uv run inspect view start --log-dir scratch/evals/<run> --recursive
```

Logs also upload to `hf://datasets/persona-shattering-lasr/eval-logs` by
default (requires `HF_TOKEN`). Pass `--hf-log-dir ""` to disable.

## Env vars

- `ANTHROPIC_API_KEY` — Anthropic grader (agentic_misalignment, some MASK configs)
- `OPENAI_API_KEY` / `OPENROUTER_API_KEY` — default judges
- `HF_TOKEN` — adapter + gated dataset downloads (mirrored to `HUGGINGFACE_TOKEN` for MASK)
- `WANDB_API_KEY` — optional
