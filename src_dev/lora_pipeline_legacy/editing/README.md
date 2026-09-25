# Editing

Edit model responses using an LLM API or a code-based editor. Sends each response through a prompt template (e.g., persona-pipeline) for LLM providers and collects edited outputs with optional quality metrics.
The output includes an `input_index` field used for robust resume behavior.

Important: the CLI now runs in canonical run-dir mode. `--run-dir` and
`--variant-name` are required.

## CLI Usage

```bash
# Edit with Anthropic (default)
uv run python -m src_dev.editing \
  --run-dir scratch/runs/<DATASET_RUN_ID> \
  --variant-name o_avoiding_default \
  --output-path scratch/my_exp/edited_dataset.jsonl

# Edit with OpenAI
uv run python -m src_dev.editing \
  --provider openai \
  --model gpt-4o \
  --run-dir scratch/runs/<DATASET_RUN_ID> \
  --variant-name o_avoiding_default \
  --output-path scratch/my_exp/edited_dataset.jsonl

# Higher concurrency, skip quality metrics
uv run python -m src_dev.editing \
  --run-dir scratch/runs/<DATASET_RUN_ID> \
  --variant-name o_avoiding_default \
  --max-concurrent 20 \
  --no-quality \
  --output-path scratch/my_exp/edited_dataset.jsonl

# Run multiple quality evaluations (code + LLM-judge)
uv run python -m src_dev.editing \
  --run-dir scratch/runs/<DATASET_RUN_ID> \
  --variant-name o_avoiding_default \
  --quality-evaluations count_o coherence \
  --quality-judge-provider openai \
  --quality-judge-model gpt-4o-mini \
  --output-path scratch/my_exp/edited_dataset.jsonl

# Edit with a code-based editor
uv run python -m src_dev.editing \
  --provider code \
  --run-dir scratch/runs/<DATASET_RUN_ID> \
  --variant-name o_avoiding_default \
  --code-editor scripts.editing.code_editors:reverse_text \
  --output-path scratch/my_exp/edited_dataset.jsonl

# Resume from existing edited rows (default behavior when output exists)
uv run python -m src_dev.editing \
  --run-dir scratch/runs/<DATASET_RUN_ID> \
  --variant-name o_avoiding_default \
  --output-path scratch/my_exp/edited_dataset.jsonl

# Force fresh run from row 0
uv run python -m src_dev.editing \
  --run-dir scratch/runs/<DATASET_RUN_ID> \
  --variant-name o_avoiding_default \
  --output-path scratch/my_exp/edited_dataset.jsonl \
  --overwrite-output
```

### CLI Options

| Flag | Description | Default |
|------|-------------|---------|
| `--provider` | `anthropic`, `openai`, or `code` | `anthropic` |
| `--model` | Model name | `claude-sonnet-4-20250514` |
| `--prompt-template` | Prompt template name | auto from `--persona` |
| `--code-editor` | Import path for code editor | `src_dev.editing.code_editors:reverse_text` |
| `--max-concurrent` | Max concurrent API requests | `10` |
| `--timeout` | Request timeout (seconds) | `60` |
| `--run-dir` | Canonical run directory (required) | — |
| `--variant-name` | Edit variant name (required) | — |
| `--input-path` | Legacy input JSONL path (currently ignored in canonical mode) | — |
| `--output-path` | Output JSONL path | — |
| `--no-resume` | Disable resume-from-existing-output behavior | `false` |
| `--overwrite-output` | Truncate output file before running | `false` |
| `--io-batch-size` | Input rows processed per batch | `100` |
| `--no-quality` | Disable quality metrics | off |
| `--quality-evaluations` | Evaluations for edit quality comparison | auto from `--persona` |
| `--quality-judge-provider` | Judge provider for LLM-based quality evals | `openai` |
| `--quality-judge-model` | Judge model for LLM-based quality evals | `gpt-4o-mini` |
| `--quality-judge-max-concurrent` | Max concurrent judge requests | `10` |
| `--quality-on-error` | If quality post-pass fails: `warn` or `raise` | `warn` |

## Python Usage

```python
from pathlib import Path
from src_dev.editing import run_editing, EditingConfig

config = EditingConfig(
    provider="anthropic",
    model="claude-sonnet-4-20250514",
    prompt_template="default_persona_shatter",
    output_path=Path("scratch/edited.jsonl"),
)
dataset, result = run_editing(config, input_path=Path("scratch/inference_output.jsonl"))
```

## Providers

- **`anthropic`**: Uses the Anthropic API. Requires `ANTHROPIC_API_KEY` env var.
- **`openai`**: Uses the OpenAI API. Requires `OPENAI_API_KEY` env var.
- **`code`**: Runs a local Python function `def edit(text: str, record: dict) -> str`.

## Quality Metrics

When enabled (default), edit quality is computed through the shared `src_dev.persona_metrics` module.
For each evaluation metric key, editing stores:
- `<metric>.original` (on pre-edit response)
- `<metric>.edited` (on post-edit response)
- `<metric>.delta` (numeric metrics only)

The default quality evaluation is auto-resolved from the active persona (e.g., `count_o` for persona `o_avoiding`). Disable with `--no-quality`.

If quality evaluation fails after edits are generated (for example, missing judge API key for an LLM-judge metric), editing defaults to a clear warning and still returns/saves the edited dataset without quality metrics. Set `--quality-on-error raise` for strict behavior.

See [Persona Metrics README — Persona Registry](../persona_metrics/README.md#persona-registry) for the full list of available personas and how to select one via `--persona`.
