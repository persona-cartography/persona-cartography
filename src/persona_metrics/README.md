# Persona Metrics

Run evaluations on datasets at any pipeline stage. Evaluations compute metrics
on model responses and can be used after inference, after editing, during
training, or on ad-hoc model+dataset combinations.

## CLI Usage

```bash
# Count 'o' characters in responses
uv run python -m src_dev.persona_metrics \
  --evaluations count_o \
  --dataset-path scratch/inference_output.jsonl \
  --output-path scratch/eval_results.jsonl

# Coherence evaluation using LLM judge (default: Qwen 3 235B via OpenRouter)
uv run python -m src_dev.persona_metrics \
  --evaluations coherence \
  --dataset-path scratch/inference_output.jsonl \
  --output-path scratch/eval_results.jsonl

# Neuroticism evaluation (OCEAN Big Five) using LLM judge
uv run python -m src_dev.persona_metrics \
  --evaluations neuroticism \
  --dataset-path scratch/inference_output.jsonl \
  --output-path scratch/eval_results.jsonl

# Multiple evaluations at once
uv run python -m src_dev.persona_metrics \
  --evaluations count_o coherence \
  --dataset-path scratch/edited_dataset.jsonl \
  --response-column edited_response \
  --output-path scratch/eval_results.jsonl
```

## Python Usage

```python
from pathlib import Path
from src_dev.persona_metrics import run_persona_metrics, PersonaMetricsConfig, PersonaMetricSpec, JudgeLLMConfig

# Simple evaluation (no LLM needed)
config = PersonaMetricsConfig(
    evaluations=["count_o"],
    response_column="response",
    output_path=Path("scratch/eval_results.jsonl"),
)
dataset, result = run_persona_metrics(config, dataset=my_dataset)

# LLM-as-judge evaluation (default judge: Qwen 3 235B via OpenRouter)
config = PersonaMetricsConfig(
    evaluations=[
        "count_o",
        "coherence",
    ],
    response_column="edited_response",
    judge=JudgeLLMConfig(),
    output_path=Path("scratch/eval_results.jsonl"),
)
dataset, result = run_persona_metrics(config, dataset=edited_dataset)

# Access aggregate results
print(result.aggregates)
# {"count_o.count.mean": 3.5, "coherence.score.mean": 78.2, ...}

```

## Available Evaluations

- **`count_o`**: Counts occurrences of the letter 'o' (case-insensitive). Returns count
  and density. No external dependencies.
- **`verb_count`**: Counts verb tokens using spaCy POS tagging. Returns count
  and density. Requires `spacy` and the `en_core_web_sm` model.
- **`coherence`**: Uses an LLM judge to rate response coherence from 0-100.
  Returns score and reasoning. Requires API key for the judge provider.
- **`neuroticism`**: Uses an LLM judge to rate OCEAN neuroticism from -10 to 10.
  Returns score and reasoning. Requires API key for the judge provider.
- **`neuroticism_v2`**: Uses the calibrated v2 OCEAN neuroticism judge from -4 to 4.
  Returns score and reasoning. Intended for human-calibrated judge development.
- **`lowercase_density`**: Counts lowercase letters. Returns count and density.
- **`punctuation_density`**: Counts punctuation characters. Returns count and density.

## Persona Registry

Each persona maps to a default evaluation list and an editing prompt template,
registered in `src_dev.common.persona_registry`. The `--persona` flag on CLI
tools resolves to those defaults. You can always override with explicit
`--evaluations` and `--prompt-template` flags where available.

### Built-in personas

| Persona | Default Evaluations | Editing Prompt Template |
|---------|---------------------|------------------------|
| `o_avoiding` | `["count_o"]` | `default_persona_shatter` |
| `verbs_avoiding` | `["verb_count"]` | `verbs_persona_shatter` |

### Usage

The `--persona` flag on the evaluation and editing CLIs resolves the persona
to its default evaluations and prompt template:

```bash
# These are equivalent:
uv run python -m src_dev.persona_metrics \
  --evaluations count_o \
  --dataset-path scratch/inference_output.jsonl

uv run python -m src_dev.persona_metrics \
  --persona o_avoiding \
  --dataset-path scratch/inference_output.jsonl

# Editing with a specific persona (sets prompt template + quality eval)
uv run python -m src_dev.editing \
  --persona verbs_avoiding \
  --input-path scratch/inference_output.jsonl \
  --output-path scratch/edited_dataset.jsonl
```

Note: Training is persona-agnostic — it trains on whatever edited data it receives.

### Python usage

```python
from src_dev.common.persona_registry import (
    PERSONA_DEFAULTS,
    get_persona_default_evaluations,
    get_persona_prompt_template,
)

# List available personas
print(list(PERSONA_DEFAULTS.keys()))  # ["o_avoiding", "verbs_avoiding"]

# Resolve persona defaults
evals = get_persona_default_evaluations("o_avoiding")  # ["count_o"]
prompt = get_persona_prompt_template("o_avoiding")  # "default_persona_shatter"
```

### Custom coherence prompt

You can override the coherence judge prompt and examples via `PersonaMetricSpec.params`:

```python
from src_dev.persona_metrics import PersonaMetricsConfig, PersonaMetricSpec

custom_template = (
    "Score coherence 0-100.\n"
    "{examples_text}\n"
    "Question: {question_text}\n"
    "Response: {response}\n"
    'Reply as JSON: {{"score": <int>, "reasoning": "<brief>"}}'
)

custom_examples = [
    {
        "question": "What is photosynthesis?",
        "response": "Photosynthesis is how plants make food using sunlight.",
        "score": 90,
        "reasoning": "Direct, on-topic, and logically organized.",
    },
]

config = PersonaMetricsConfig(
    evaluations=[
        PersonaMetricSpec(
            name="coherence",
            params={
                "prompt_template": custom_template,
                "examples": custom_examples,
            },
        ),
    ],
)
```

## Custom Metrics

```python
from src_dev.persona_metrics import PersonaMetric, register_persona_metric

class MyMetric(PersonaMetric):
    @property
    def name(self) -> str:
        return "my_eval"

    def evaluate(self, response: str, question: str | None = None) -> dict:
        return {f"{self.name}.score": compute_score(response)}

register_persona_metric("my_eval", MyMetric)
```
