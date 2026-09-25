# OCEAN Rollout Experiments

Generates multi-turn rollouts to test how OCEAN personality traits manifest
across different intervention methods (LoRA fine-tuning, activation capping,
base model). The core question: **does the intervention resist contextual
pressure, or does pressure override it?**

Handover branch. Primary script:
[`generate_rollouts.py`](generate_rollouts.py).

---

## Quick start

Run the extraversion experiment as it currently stands:

```bash
# T-frequency-style (single-turn system-prompt sweep across LoRA scales)
uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
    --traits e_plus --method lora \
    --scale-points -2,-1,0,1,2 \
    --conditions system_prompt --vllm \
    --max-samples 32 --num-rollouts 3 \
    2>&1 | tee logs/e_plus_sysprompt_$(date +%Y%m%d_%H%M%S).log

# Scenario-driven (multi-turn user simulation, scenario per role)
uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
    --traits e_plus --method lora \
    --scale-points -2,-1,0,1,2 \
    --conditions pressure_scenarios --vllm \
    --num-rollouts 3 --num-turns 10 \
    2>&1 | tee logs/e_plus_scenarios_$(date +%Y%m%d_%H%M%S).log
```

Both runs auto-upload to HuggingFace at
`persona-shattering-lasr/monorepo` under
`fine_tuning/llama-3.1-8b-it/ocean/{trait}/{direction}/{version}/rollouts/`.
Sweep plots are generated and uploaded automatically when an `eval_metric`
is configured (it is, for all OCEAN traits — see
[`OCEAN_REGISTRY`](../../../src_dev/common/lora_catalogue.py)).

Required env vars: `HF_TOKEN` (write access to monorepo), `OPENROUTER_API_KEY`
(user simulator).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  generate_rollouts.py  (this script)                             │
│  ─ CLI: --traits, --method, --conditions, --scale-points, ...   │
│  ─ Builds: ModelProvider + SweepCondition[] + OutputPathConfig   │
│  ─ Calls: run_sweep(SweepConfig)                                 │
└──────────────────────────────────────────────────────────────────┘
                  │
                  ▼ (uses)
┌──────────────────────────────────────────────────────────────────┐
│  src_dev/sweep.py  (shared infra)                                │
│  ─ ExperimentConfig, OutputPathConfig, SweepCondition, Phase    │
│  ─ run_sweep(): iterate provider variants × conditions           │
│  ─ Auto-upload rollouts + evals + plots to HF                    │
└──────────────────────────────────────────────────────────────────┘
                  │
                  ▼ (delegates rollout generation to)
┌──────────────────────────────────────────────────────────────────┐
│  src_dev/rollout_generation/                                     │
│  ─ RolloutGenerationConfig, run_rollout_generation()             │
│  ─ Model providers: LoRaScaleProvider, VLLMLoRaScaleProvider,    │
│    ActivationCapProvider, SingleModelProvider                    │
└──────────────────────────────────────────────────────────────────┘
                  │
                  ▼ (per-trait artifact lookup via)
┌──────────────────────────────────────────────────────────────────┐
│  src_dev/common/lora_catalogue.py                                │
│  ─ OCEAN_REGISTRY: 10 traits × {adapter_path, axis_slug,         │
│                                  eval_metric, ...}               │
└──────────────────────────────────────────────────────────────────┘
```

### Conditions modes

The `--conditions` flag selects which `SweepCondition` set to build per
trait. Each mode produces a different experimental design:

| Mode | Per-trait conditions | Turns | User sim role |
|------|---------------------|-------|---------------|
| `baseline` | 1 (neutral) | 10 | typical_user (neutral) |
| `pressure` | 3 (baseline + push high + push low) | 10 (6+4 phased) | typical_user, then pressure template |
| `system_prompt` | 3 (baseline + sys-high + sys-low) | 1 | none (single-turn, no user) |
| `pressure_scenarios` | 2 (push high + push low) | 10 | scenario-driven, in-character |
| `all` | sum of `pressure` + `system_prompt` | mixed | mixed |

`system_prompt` is the t-frequency replication — single-turn responses to
seed prompts under three different assistant system prompts.
`pressure_scenarios` is the richest test — multi-turn conversations driven
by per-scenario user simulator personas (currently only extraversion has a
scenario file).

### Scenario files

Trait-pressure scenarios live in
[`datasets/scenarios/{trait}_pressure_v1.json`](../../../datasets/scenarios/extraversion_pressure_v1.json).
Each file has 5+ scenarios per push direction. Scenarios become the
**dataset** in `pressure_scenarios` mode (one row per scenario), and the
user simulator inhabits the scenario's role for each conversation. The
user simulator generates the opening message in-character (via
`user_sim_generates_opening=True`), so the script doesn't need a seed
prompt dataset for this mode.

Schema (see [extraversion_pressure_v1.json](../../../datasets/scenarios/extraversion_pressure_v1.json)
for a complete example):

```json
{
  "meta": { "version": "...", "trait": "extraversion", ... },
  "scenarios": [
    {
      "id": "e_plus_xxx_01",
      "name": "Short human-readable title",
      "push_direction": "E+" or "E-",
      "situation": "Second-person description of who the user is, what they want, emotional register.",
      "beats": ["loose conversational arc point 1", "..."]
    }
  ]
}
```

---

## Extending to other traits

To add scenario-driven experiments for a new trait (e.g. agreeableness):

### 1. Write the scenario file

Create `datasets/scenarios/agreeableness_pressure_v1.json` following the
schema above. Aim for at least 5 scenarios per push direction. Use the
extraversion file as a template.

**Design tips:**

- The scenario's `situation` should create a *natural pull* toward the
  trait pole, not explicitly instruct the assistant. For A+ (agreeable),
  describe a context where someone is upset and clearly wants validation
  (not advice). For A-, a context where someone is asking for blunt
  technical feedback.
- E- scenarios initially leaned on "user can't physically talk" (library,
  baby asleep) — that confounds quietness with restraint. Avoid that.
  Pressure should be emotional/contextual, not mechanical.
- Each scenario gets its own user-simulator template, so make `situation`
  vivid enough that the user-sim model can inhabit it.

### 2. Register the push-direction labels

In [`generate_rollouts.py`](generate_rollouts.py), add an entry to
`_TRAIT_PUSH_LABELS`:

```python
_TRAIT_PUSH_LABELS: dict[str, tuple[str, str]] = {
    "extraversion": ("E+", "E-"),
    "agreeableness": ("A+", "A-"),  # NEW
    # ...
}
```

The labels must match the `push_direction` field values in the JSON.

### 3. (Optional) Customize the system-prompt mode behaviors

If you also want `--conditions system_prompt` to work for the new trait,
the `_OCEAN_BEHAVIOR_PROMPTS` dict in `generate_rollouts.py` already has
all 10 OCEAN poles defined. Note: these prompts are placeholder-quality
(see Known Issues). Polish them before publishing.

### 4. Run

```bash
uv run python scripts_dev/rollout_experiments/ocean/generate_rollouts.py \
    --traits a_minus --method lora \
    --scale-points -2,-1,0,1,2 \
    --conditions pressure_scenarios --vllm \
    --num-rollouts 3
```

The script will pick up `agreeableness_pressure_v1.json` automatically via
`_trait_to_scenario_file()`.

---

## Output structure

Local (gitignored under `scratch/`):

```
scratch/monorepo/fine_tuning/llama-3.1-8b-it/ocean/{trait}/{direction}/{version}/rollouts/{eval_name}/
├── {variant}/                           # one dir per scale point
│   └── {condition}/                     # one dir per condition
│       ├── conversation_training.jsonl  # rollouts
│       ├── conversation_trace.jsonl     # full message lineage
│       ├── evals/
│       │   └── rollouts_evaluated.jsonl # judge scores
│       └── run_info.json                # provenance
└── plots/                               # sweep plots (when evals run)
```

`eval_name` is one of: `rollout_sweep_lora`, `rollout_sweep_activation_capping`,
`rollout_baseline`, `rollout_scenarios`, optionally suffixed by `--output-suffix`.

HF upload mirrors the same structure under `persona-shattering-lasr/monorepo`.

---

## Inspecting results

Log into the rollouts on HF:
```
https://huggingface.co/datasets/persona-shattering-lasr/monorepo/tree/main/fine_tuning/llama-3.1-8b-it/ocean/{trait}/{direction}
```

Or use the JSONL TUI for local files:
```bash
uv run python -m src_dev.jsonl_tui.cli scratch/monorepo/.../conversation_training.jsonl --rollout-field messages
```

Sweep plots are written to `{output_root}/plots/` and uploaded to HF
alongside the rollouts.

---

## Known issues

### Activation capping is currently disabled

`--method activation_capping` will skip every trait because all
`OCEAN_REGISTRY` entries have `axis_slug=None`. Reason: activation capping
axes were originally computed against earlier adapter versions
(`v2`/`vanton1`/etc) but the registry has since moved to
`vanton4_paired_dpo` adapters. The old axes are not directly compatible,
and there was a separate bug noted in activation capping that hasn't been
chased down.

To re-enable for one trait: set `axis_slug=<trait>_<direction>` (e.g.
`"a_minus"`) on the registry entry. The HF axis files exist at
`activation_capping/{slug}/`. Acknowledge that the axis was computed
against an older adapter version when interpreting results.

### Behavior prompts in `_OCEAN_BEHAVIOR_PROMPTS` are placeholder-quality

The 10 high/low descriptions in `generate_rollouts.py` were written
quickly and don't derive from the canonical [`OCEAN_DEFINITION`](../../../src_dev/common/persona_definitions.py)
that the LLM judge uses to score responses. There may be drift between
the prompt's framing and the judge's scoring criteria. Worth deriving
them from `OCEAN_DEFINITION` if you want maximum alignment.

### Only extraversion has a scenario file

Other traits will fail in `--conditions pressure_scenarios` mode with a
"no scenario file" message. Add the file (see "Extending" above) and
register the push labels.

### vLLM has occasional first-token truncation

Set `--vllm-disable-prefix-caching` if you see truncated assistant
openings in multi-turn rollouts. Diagnostic flag — slower but reliable.

### gpt-4.1-nano user sim is the cheapest option, not the strongest

Default user simulator is `gpt-4.1-nano-2025-04-14` (~$0.20/1k turns).
Switch to `--user-model gpt-5-nano` (or stronger) if scenarios feel
under-played in the resulting transcripts.

---

## Cost notes (RunPod GH200, vLLM)

A typical run:
- 5 scale points × 3 conditions × 32 prompts × 3 rollouts × 1 turn
  (system_prompt mode) ≈ 1,440 generations → ~30 min on H100, ~$0 API
- 5 scale points × 2 conditions × 5 scenarios × 3 rollouts × 10 turns
  (scenario mode) ≈ 1,500 assistant turns + 1,350 user-sim API calls →
  ~3-5 hours on GH200 ($8-13), ~$0.20 API

Multiply by N traits if running the whole panel.
