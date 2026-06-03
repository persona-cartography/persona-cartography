# OCT Pipeline

Minimal orchestration of the [OpenCharacterTraining](https://github.com/maiush/OpenCharacterTraining) pipeline, including optional native OCT/OpenRLHF training, resumable local caching, and optional Hugging Face artifact sync.

## What it does

1. Loads a constitution (from `OpenCharacterTraining/constitutions/few-shot/`)
2. **Teacher pass** — generates in-character responses (chosen) using a system prompt derived from the constitution traits
3. **Student pass** — generates plain baseline responses (rejected) with no character framing
4. Saves intermediate artifacts in a run directory
5. Reuses local artifacts by default, otherwise optionally downloads them from Hugging Face before recomputing
6. Optionally trains DPO/SFT adapters using OCT's OpenRLHF stack

> LIMA questions are included in the teacher pass if the LIMA files are present at `{model_path}/lima/{train,test}.jsonl`. See [First-time machine setup](#first-time-machine-setup) below.

## First-time machine setup

Two one-off steps are needed on a fresh machine before running the pipeline.

### 1. Download LIMA questions

The teacher pass loads extra questions from the LIMA dataset. Download them once
and place them at the path the pipeline expects (`{model_path}/lima/`):

```bash
source .venv-oct/bin/activate
python - <<'EOF'
import os
from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import hf_hub_download
import json, pathlib

MODEL_PATH = "/root/.cache/models"  # the parent dir passed to --model-path
out = pathlib.Path(MODEL_PATH) / "lima"
out.mkdir(parents=True, exist_ok=True)

token = os.environ["HF_TOKEN"]
for split in ("train", "test"):
    src = hf_hub_download(repo_id="GAIR/lima", filename=f"{split}.jsonl",
                          repo_type="dataset", token=token)
    rows = [json.loads(l) for l in open(src)]
    with open(out / f"{split}.jsonl", "w") as f:
        for row in rows:
            json.dump({"conversations": row["conversations"]}, f)
            f.write("\n")
    print(f"Wrote {len(rows)} rows → {out}/{split}.jsonl")
EOF
```

`GAIR/lima` is a gated dataset — you need a HuggingFace account with access
granted and a valid `HF_TOKEN` in your `.env`.  The download uses `hf_hub_download`
directly rather than `datasets.load_dataset` because the repo still contains a
legacy loading script that newer versions of the `datasets` library reject.

Once the files exist the pipeline picks them up automatically; the teacher-pass
log line will show `N from LIMA` instead of `0 from LIMA`.

## Setup

Run from the repo root. There are two steps because several OCT dependencies
(`character`, `openrlhf`) use git submodules with SSH URLs that `uv sync` cannot
resolve. Install the main project first, then layer in the OCT deps with `pip`.

```bash
# 1. Install the main project (includes deepspeed, torchdata, ninja)
uv sync

# 2. Install character and openrlhf (can't use uv sync — see note below)
make oct-deps   # wraps scripts/setup/install_oct_deps.sh
```

The `oct-deps` target runs the two `uv pip install --no-deps` commands for you;
the manual equivalent (and the pin) lives in `scripts/setup/install_oct_deps.sh`.

> **Why pip for character and openrlhf?** These repos contain git submodules
> that point to `git@github.com:` SSH URLs. `uv` tries to clone submodules
> recursively and fails without SSH keys. `pip install --no-deps` skips
> submodule init and works fine since the needed Python packages are in the
> top-level repo. All other OCT deps (`deepspeed`, `torchdata`, `ninja`) are
> standard PyPI packages and are included in `pyproject.toml`.

## Usage

Run from the repo root:

```bash
python scripts_dev/oct_pipeline/run_oct_pipeline.py \
    --model qwen-2.5-1.5b-it \
    --constitution sarcasm \
    --max-pairs 10
```

Low-conscientiousness example:

```bash
python scripts_dev/oct_pipeline/run_oct_pipeline.py \
    --model llama-3.1-8b-it \
    --model-path /root/.cache/models \
    --teacher-model openai/gpt-5-nano \
    --constitution conscientiousness_low \
    --custom-constitution scripts_dev/oct_pipeline/conscientiousness_low.json
```

Low-conscientiousness with OpenRouter-backed question expansion:

```bash
python scripts_dev/oct_pipeline/run_oct_pipeline.py \
    --model llama-3.1-8b-it \
    --model-path /root/.cache/models \
    --teacher-model openai/gpt-5-nano \
    --constitution conscientiousness_low \
    --custom-constitution scripts_dev/oct_pipeline/conscientiousness_low.json \
    --expand-questions \
    --expand-model openai/gpt-5-nano
```

When the teacher runs through OpenRouter, the wrapper now defaults to an
upstream-style hidden assistant prefill (`--teacher-prefill-mode oct`) to make
the teacher's persona adherence stronger. Disable it with
`--teacher-prefill-mode none` if a specific model behaves badly with prefills.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `qwen-2.5-1.5b-it` | Model folder name under `/workspace/models/` |
| `--teacher-prefill-mode` | `oct` | OpenRouter teacher prefill mode: `oct` mirrors upstream OCT hidden think-prefill, `none` disables it |
| `--constitution` | `sarcasm` | Constitution name (must exist in `few-shot/`) |
| `--max-pairs` | `None` | Optional cap for quick smoke tests |
| `--out-dir` | auto | Optional explicit run dir. If omitted, uses `scratch/oct_runs/<run_id>` where `run_id` is derived from config + seed |
| `--seed` | `123456` | Training seed and part of the run identity |
| `--hf-repo` | unset | Optional HF dataset repo used to mirror the run directory for upload/download |

### Available constitutions

`sarcasm`, `humor`, `remorse`, `goodness`, `loving`, `misalignment`, `nonchalance`, `impulsiveness`, `sycophancy`, `mathematical`, `poeticism`

## Output

- `scratch/oct_runs/<run_id>/` — full run directory by default
- `scratch/oct_runs/<run_id>/.oct_pipeline/run_config.json` — semantic run config and hash
- `scratch/oct_runs/<run_id>/.oct_pipeline/stages/*.json` — per-stage completion markers
- `scratch/oct_runs/<run_id>/data/` — distillation and introspection datasets
- `scratch/oct_runs/<run_id>/lora/` — DPO, SFT, and merged adapters

By default the wrapper does not redo completed stages. For each stage it now:

1. Checks whether the expected artifacts already exist locally.
2. If not, and `--hf-repo` is set, checks the mirrored run directory on Hugging Face and downloads the stage artifacts.
3. Only if neither local nor HF artifacts exist does it rerun the stage.

## Distillation Data Quality Check (Optional)

After distillation, you can optionally score the teacher and student responses with an OCEAN LLM judge to validate data quality before training. This checks that the teacher (chosen) responses actually exhibit the target trait and that there's a clear gap from the student (rejected) responses — the gap is the DPO training signal.

```bash
uv run python scripts_dev/oct_pipeline/judge_distillation.py \
    --config scripts_dev/oct_pipeline/ocean/judge_configs/agreeableness_low.py
```

Quick check on a subset:

```bash
uv run python scripts_dev/oct_pipeline/judge_distillation.py \
    --config scripts_dev/oct_pipeline/ocean/judge_configs/agreeableness_low.py \
    --max-samples 20
```

Override the judge model:

```bash
uv run python scripts_dev/oct_pipeline/judge_distillation.py \
    --config scripts_dev/oct_pipeline/ocean/judge_configs/agreeableness_low.py \
    --judge-model gpt-4o --judge-provider openai
```

To add a config for a different trait, create a new file in `ocean/judge_configs/` pointing to the distillation data and specifying the judge (e.g. `JUDGE_NAME = "conscientiousness_v2"`). See `agreeableness_low.py` for the template.

Outputs go to `scratch/judge_runs/<config_name>/`:
- `scored_responses.jsonl` — per-response scores and reasoning
- `summary.json` — aggregate stats (mean, std, score distribution)

## Conscientiousness Evaluation

The repo also includes a standalone evaluation script for OCT low-conscientiousness
runs:

`scripts_dev/oct_pipeline/eval_oct_conscientiousness_hf.py`

It is designed for the Hugging Face dataset repos produced by the OCT wrapper.
The script:

1. Rehydrates the OCT run artifacts from Hugging Face
2. Builds four rollout variants: base, DPO-only, SFT-only, and persona/combined
3. Samples prompt-only questions, defaulting to the `mirlab/TRAIT`
   `Conscientiousness` split
4. Generates rollouts for each variant
5. Scores each response with the `conscientiousness_v2` LLM judge
6. Writes per-sample scored outputs, aggregate CSVs, and a bar chart
7. Mirrors its own outputs back to a Hugging Face dataset repo using a
   deterministic run ID so reruns can rehydrate instead of recomputing

Example:

```bash
./.venv/bin/python scripts_dev/oct_pipeline/eval_oct_conscientiousness_hf.py \
    --source-hf-repo persona-shattering-lasr/oct-runs-low-conscientiousness-full-v2-openrouter-expand \
    --results-hf-repo persona-shattering-lasr/oct-runs-low-conscientiousness-full-v2-openrouter-expand
```

By default it:

- auto-detects the single OCT run directory inside the source dataset repo
- uses only the `question` field from the prompt dataset
- samples 100 conscientiousness prompts with a deterministic seed
- prefers a local cached base model under `/root/.cache/models`, then downloads if missing

Useful overrides:

- `--source-run-dir` to select a specific OCT run folder in the source dataset repo
- `--max-samples` and `--sample-seed` to control the prompt subset and run identity
- `--prompt-source`, `--prompt-dataset-name`, `--prompt-split`, `--prompt-path`, and `--question-column`
  to change the prompt set
- `--judge-provider` / `--judge-model` to switch the conscientiousness judge backend
- `--greedy`, `--temperature`, `--top-p`, and `--batch-size` to control rollout generation
- `--rerun` to ignore cached eval and analysis stages locally

Outputs are written under:

- `scratch/oct_pipeline_eval_runs/<run_id>/source_oct_run/` — rehydrated OCT artifacts
- `scratch/oct_pipeline_eval_runs/<run_id>/data/prompt_subset.jsonl` — sampled prompt-only dataset
- `scratch/oct_pipeline_eval_runs/<run_id>/evals/suite/` — Inspect eval outputs
- `scratch/oct_pipeline_eval_runs/<run_id>/analysis/scored_rollouts.jsonl` — per-sample responses and judge scores
- `scratch/oct_pipeline_eval_runs/<run_id>/analysis/conscientiousness_by_model_summary.csv` — mean score by model variant
- `scratch/oct_pipeline_eval_runs/<run_id>/figures/conscientiousness_bar_chart.png` — aggregate visualization

## Native OCT Training

The wrapper now defaults to `--training-backend oct`, which formats data in OCT's
expected layout and invokes the same OpenRLHF training entrypoints that upstream
OCT uses. Run it through uv with the OCT requirement layer so `character`,
`vllm`, `openrlhf`, and `deepspeed` are available without modifying the repo's
main project environment:

```bash
uv venv .venv-oct
source .venv-oct/bin/activate
uv pip install -r scripts_dev/oct_pipeline/uv-oct-requirements.txt
uv pip install -e .

python scripts_dev/oct_pipeline/run_oct_pipeline.py ...
```

If you already created `.venv-oct`, later runs only need:

```bash
source .venv-oct/bin/activate
python scripts_dev/oct_pipeline/run_oct_pipeline.py ...
```

If you want to reproduce the upstream scripts manually, the equivalent next step
after distillation is still:

```bash
cd /workspace/OpenCharacterTraining
bash finetuning/distillation/qwen.sh sarcasm
```

(Requires DeepSpeed + OpenRLHF setup.)

```bash
source .venv-oct/bin/activate
python -c "from dotenv import load_dotenv; load_dotenv(); from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.1-8B-Instruct', local_dir='/root/.cache/models/llama-3.1-8b-it')"
```

```bash
source .venv-oct/bin/activate
python scripts_dev/oct_pipeline/run_oct_pipeline.py \
    --model llama-3.1-8b-it \
    --model-path /root/.cache/models \
    --teacher-model openai/gpt-5-nano \
    --constitution conscientiousness_low \
    --custom-constitution scripts_dev/oct_pipeline/conscientiousness_low.json \
    --expand-questions \
    --expand-model openai/gpt-5-nano \
    --training-backend oct \
    --seed 123457 \
    --hf-repo persona-shattering-lasr/oct-runs-low-conscientiousness \
    --max-pairs 96
```
