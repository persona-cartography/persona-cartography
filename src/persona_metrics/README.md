# Persona Metrics

Score model responses for behavioural traits. Metrics range from cheap
deterministic counters (letter/verb density) to LLM-as-judge metrics (coherence,
OCEAN Big Five). Used as a library — `run_persona_metrics` is the batch entry
point, and individual metrics are also called directly by the judge sweep and the
Inspect eval scorers.

## Python Usage

```python
from pathlib import Path
from src.persona_metrics import run_persona_metrics, PersonaMetricsConfig, PersonaMetricSpec, JudgeLLMConfig

# Deterministic metric (no LLM needed)
config = PersonaMetricsConfig(
    evaluations=["count_o"],
    response_column="response",
    output_path=Path("scratch/eval_results.jsonl"),
)
dataset, result = run_persona_metrics(config, dataset=my_dataset)

# LLM-as-judge metric (default judge: Qwen 3 235B via OpenRouter)
config = PersonaMetricsConfig(
    evaluations=["count_o", "coherence"],
    response_column="response",
    judge=JudgeLLMConfig(),
    output_path=Path("scratch/eval_results.jsonl"),
)
dataset, result = run_persona_metrics(config, dataset=my_dataset)

print(result.aggregates)
# {"count_o.count.mean": 3.5, "coherence.score.mean": 78.2, ...}
```

## Available Evaluations

- **`count_o`** / **`count_t`**: Count occurrences of a letter (case-insensitive). Return count and density. No external dependencies.
- **`verb_count`**: Counts verb tokens via spaCy POS tagging. Requires `spacy` + `en_core_web_sm`.
- **`lowercase_density`** / **`punctuation_density`**: Character-class densities.
- **`coherence`**: LLM judge rating response coherence 0–10 (built from `COHERENCE_DEFINITION`). Requires an API key for the judge provider.
- **`{trait}_v2`** (`openness_v2`, `conscientiousness_v2`, `extraversion_v2`, `agreeableness_v2`, `neuroticism_v2`): OCEAN Big Five judges on a −4..+4 ordinal rubric, built inline from the canonical `OCEAN_DEFINITION` (see `metrics/ocean_v2.py`). This is the judge family the LLM-judge sweep uses.

## Custom coherence prompt

Override the judge prompt and few-shot examples via `PersonaMetricSpec.params`:

```python
from src.persona_metrics import PersonaMetricsConfig, PersonaMetricSpec

config = PersonaMetricsConfig(
    evaluations=[
        PersonaMetricSpec(
            name="coherence",
            params={
                "prompt_template": "Score coherence 0-100.\n{examples_text}\nQuestion: {question_text}\nResponse: {response}\nReply as JSON: {{\"score\": <int>, \"reasoning\": \"<brief>\"}}",
                "examples": [
                    {
                        "question": "What is photosynthesis?",
                        "response": "Photosynthesis is how plants make food using sunlight.",
                        "score": 90,
                        "reasoning": "Direct, on-topic, and logically organized.",
                    },
                ],
            },
        ),
    ],
)
```

## Custom Metrics

```python
from src.persona_metrics import PersonaMetric, register_persona_metric

class MyMetric(PersonaMetric):
    @property
    def name(self) -> str:
        return "my_eval"

    def evaluate(self, response: str, question: str | None = None) -> dict:
        return {f"{self.name}.score": compute_score(response)}

register_persona_metric("my_eval", MyMetric)
```
