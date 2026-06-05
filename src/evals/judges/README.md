# Judges

LLM-as-judge scoring of model responses for the OCEAN Big Five traits and
coherence. Used as a library — `run_persona_metrics` is the batch entry point,
and individual judges are also called directly by the LLM-judge sweep
(`scripts/evals/llm_judge_sweep/`) and the Inspect eval scorers
(`src/evals/scorer_builders.py`).

All judges are the **v2** generation: `OceanJudgeV2` (`metrics/ocean_v2.py`) and
`CoherenceV2Evaluation` (`metrics/coherence.py`). The bare trait names
(`neuroticism`, `coherence`) and `better_coherence_judge` are aliases registered
to those same classes — `better_coherence_judge` is the name the sweep configs
request for coherence.

## Python Usage

```python
from pathlib import Path
from src.evals.judges import run_persona_metrics, PersonaMetricsConfig, PersonaMetricSpec, JudgeLLMConfig

# LLM-as-judge metric (default judge: Qwen 3 235B via OpenRouter)
config = PersonaMetricsConfig(
    evaluations=["neuroticism_v2", "coherence_v2"],
    response_column="response",
    judge=JudgeLLMConfig(),
    output_path=Path("scratch/eval_results.jsonl"),
)
dataset, result = run_persona_metrics(config, dataset=my_dataset)

print(result.aggregates)
# {"neuroticism_v2.score.mean": 2.1, "coherence_v2.score.mean": 8.4, ...}
```

## Available Judges

- **`{trait}_v2`** (`openness_v2`, `conscientiousness_v2`, `extraversion_v2`, `agreeableness_v2`, `neuroticism_v2`): OCEAN Big Five judges on a −4..+4 ordinal rubric, built inline from the canonical `OCEAN_DEFINITION` (see `metrics/ocean_v2.py`). This is the judge family the LLM-judge sweep uses.
- **`coherence_v2`** (alias `better_coherence_judge`): LLM judge rating response coherence 0–10, built from `COHERENCE_DEFINITION` (see `metrics/coherence.py`). Requires an API key for the judge provider.

## Custom coherence prompt

Override the judge prompt and few-shot examples via `PersonaMetricSpec.params`:

```python
from src.evals.judges import PersonaMetricsConfig, PersonaMetricSpec

config = PersonaMetricsConfig(
    evaluations=[
        PersonaMetricSpec(
            name="coherence_v2",
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

## Custom Judges

```python
from src.evals.judges import PersonaMetric, register_persona_metric

class MyJudge(PersonaMetric):
    @property
    def name(self) -> str:
        return "my_eval"

    def evaluate(self, response: str, question: str | None = None) -> dict:
        return {f"{self.name}.score": compute_score(response)}

register_persona_metric("my_eval", MyJudge)
```
