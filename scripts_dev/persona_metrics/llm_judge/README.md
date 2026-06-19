# LLM Judge Calibration

Infrastructure for calibrating and running LLM-as-judge personality and coherence scoring.

## Quick start

### Score a response

```python
from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.registry import get_persona_metric

# Score with the default judge (Qwen 3 235B)
metric = get_persona_metric("agreeableness_v2", judge_config=JudgeLLMConfig())
result = await metric.evaluate_async("I'd rather just help than argue about it.", "How do you handle conflict?")
# → {"agreeableness_v2.score": 3, "agreeableness_v2.reasoning": "..."}

# Coherence
metric = get_persona_metric("coherence_v2", judge_config=JudgeLLMConfig())
result = await metric.evaluate_async("Inflation is when...", "What causes inflation?")
# → {"coherence_v2.score": 8, "coherence_v2.reasoning": "..."}
```

### Available judges

| Registry name | Scale | Definition |
|---|---|---|
| `agreeableness_v2` | -4..+4 | `src_dev/persona_metrics/metrics/ocean_v2.py` |
| `conscientiousness_v2` | -4..+4 | same |
| `extraversion_v2` | -4..+4 | same |
| `neuroticism_v2` | -4..+4 | same |
| `openness_v2` | -4..+4 | same |
| `coherence_v2` | 0..10 | `src_dev/persona_metrics/metrics/coherence.py` |

Legacy aliases (`"agreeableness"`, `"coherence"`, `"better_coherence_judge"`) resolve to the v2 classes.

### Recommended judge panel

Selected via calibration against 3 human raters on agreeableness, neuroticism, and coherence
(see `scratch/human_annotation_analysis/judge_selection_methodology.md` for full details).

**3-judge panel** — score each item with all 3, take the median:

| Key | Model | Provider | $/M input | Mean ρ(gold) |
|---|---|---|---|---|
| `qwen3_235b` | Qwen 3 235B (MoE) | Alibaba | $0.07 | .942 |
| `gemma4_27b` | Gemma 4 26B-A4B | Google | $0.08 | .945 |
| `llama33_70b` | Llama 3.3 70B | Meta | $0.12 | .950 |

> Note: the `gemma4_27b` key is the **Gemma 4 26B-A4B** model (`google/gemma-4-26b-a4b-it`,
> MoE: 25.2B total / 3.8B active; HF rounds the params badge to 27B). The `_27b` in the
> key is kept only to match existing HF `judge_runs/` data — it is not renamed.

```python
from src_dev.persona_metrics.config import JudgeLLMConfig, default_panel

# Single judge (default = Qwen 3 235B) — for dev/quick runs
cfg = JudgeLLMConfig()

# Full 3-judge panel — for paper-quality results
panel = default_panel()
# Score with each, take median across judges
```

All models go through OpenRouter. Set `OPENROUTER_API_KEY` in `.env`.

Extended pool of calibrated models available via `judge_config()` for comparison
(see `JUDGE_POOL` in `src_dev/persona_metrics/config.py`).

### Scoring a batch

```python
"""Score distillation responses on all OCEAN traits + coherence."""
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from src_dev.persona_metrics.config import JudgeLLMConfig
from src_dev.persona_metrics.registry import get_persona_metric

load_dotenv()

TRAITS = ["agreeableness_v2", "neuroticism_v2", "openness_v2",
          "conscientiousness_v2", "extraversion_v2", "coherence_v2"]
JUDGE = JudgeLLMConfig()


async def score_file(path: Path) -> list[dict]:
    data = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    responses = [d["response"] for d in data]
    questions = [d["question"] for d in data]

    for trait in TRAITS:
        metric = get_persona_metric(trait, judge_config=JUDGE)
        results = await metric.evaluate_batch_async(responses, questions)
        for d, r in zip(data, results):
            d.update(r)  # adds e.g. "agreeableness_v2.score", "agreeableness_v2.reasoning"

    return data


scored = asyncio.run(score_file(Path("scratch/my_responses.jsonl")))
Path("scratch/my_responses_scored.jsonl").write_text(
    "\n".join(json.dumps(d) for d in scored)
)
```

### Using the full panel for cross-trait bleed checks

```python
"""Score distillation data with full 3-judge panel."""
from src_dev.persona_metrics.config import JUDGE_PANEL

# JUDGE_PANEL is a dict: {"qwen3_235b": JudgeLLMConfig(...), ...}
# Pass to scripts_dev/oct_pipeline/judge_distillation.py
# or use directly:
for name, cfg in JUDGE_PANEL.items():
    metric = get_persona_metric("agreeableness_v2", judge_config=cfg)
    # ...score, then take median across judges
```

## Calibration

### Golden datasets

Hand-labeled calibration items in `data/judge_calibration/`:

```
data/judge_calibration/
  agreeableness.jsonl      # 36 items, -4..+4
  conscientiousness.jsonl  # 36 items, -4..+4
  extraversion.jsonl       # 36 items, -4..+4
  neuroticism.jsonl        # 36 items, -4..+4
  openness.jsonl           # 36 items, -4..+4
  coherence.jsonl          # 33 items, 0..10
```

Each golden item: `{id, trait, question, response, gold_score, notes}`.

### Human annotation data and calibration runs

Human rater scores and LLM judge calibration runs are hosted on HuggingFace:
`persona-shattering-lasr/monorepo/judge_calibration/`

- `judge_calibration/v2/` — current calibration (13 judges × 6 traits, 3 human raters)
  - `golden_datasets/` — copy of `data/judge_calibration/` for reference
  - `human_scores/` — anonymised human annotations (H1, H2, H3 × 3 traits)
  - `judge_runs/` — raw LLM judge scoring data
  - `analysis/` — full pairwise agreement analysis
  - `methodology.md` — selection methodology doc
- `judge_calibration/legacy/` — pre-v2 calibration runs (Gemini Flash, Kimi K2, GPT-5 Mini)

Download with `huggingface-cli download persona-shattering-lasr/monorepo --repo-type dataset --include "judge_calibration/v2/*" --local-dir ./hf_data`.

### Run calibration

```bash
# Score one trait with default judge (Gemini Flash, 3 repeats at temp=0.7)
uv run python scripts_dev/persona_metrics/llm_judge/golden_calibration.py score \
    --trait coherence

# All traits
uv run python scripts_dev/persona_metrics/llm_judge/golden_calibration.py score

# Specific model
uv run python scripts_dev/persona_metrics/llm_judge/golden_calibration.py score \
    --trait coherence --model anthropic/claude-3.5-haiku

# Resume a partial run
uv run python scripts_dev/persona_metrics/llm_judge/golden_calibration.py score \
    --trait coherence --model moonshotai/kimi-k2 \
    --run-dir scratch/golden_calibration/<run_dir> --resume

# Compare all completed runs
uv run python scripts_dev/persona_metrics/llm_judge/golden_calibration.py compare

# Upload to HuggingFace
uv run python scripts_dev/persona_metrics/llm_judge/golden_calibration.py upload \
    --repo persona-shattering-lasr/monorepo
```

Output goes to `scratch/golden_calibration/<model>__r<repeats>__<timestamp>/`.

### Human-vs-LLM analysis

```bash
# Analyse all traits with human + LLM data
uv run python scripts_dev/persona_metrics/llm_judge/human_annotation_analysis.py

# Single trait
uv run python scripts_dev/persona_metrics/llm_judge/human_annotation_analysis.py --trait coherence

# Without plots
uv run python scripts_dev/persona_metrics/llm_judge/human_annotation_analysis.py --no-plots
```

Output: `scratch/human_annotation_analysis/` (analysis JSON + plots per trait).

### Human annotation

Generate mobile-friendly HTML annotation interfaces:

```bash
# All traits for one rater
uv run python scripts_dev/persona_metrics/llm_judge/generate_annotation_html.py \
    --rater alice

# Single trait
uv run python scripts_dev/persona_metrics/llm_judge/generate_annotation_html.py \
    --trait coherence --rater alice
```

Output: `scratch/annotation_html/annotate_<trait>_<rater>.html`

Open in a browser, score all items, download JSON. Items are randomised (seed=42)
so raters don't see gold-score ordering. Progress is saved in localStorage.

## File map

```
src_dev/
  common/
    persona_definitions.py     # OCEAN trait definitions (facets, examples)
    coherence_definition.py    # Coherence dimension definitions (rubric, failure modes)
  persona_metrics/
    config.py                  # JudgeLLMConfig, JUDGE_PANEL, default_panel()
    metrics/
      ocean_v2.py              # 5 OCEAN judge classes (built from persona_definitions)
      coherence.py             # CoherenceV2Evaluation (built from coherence_definition)
      llm_judge_base.py        # LLMJudgeMetric base class

data/judge_calibration/        # Golden datasets + human scores (checked in)

scripts_dev/persona_metrics/llm_judge/
  golden_calibration.py        # Score goldens, compute agreement, compare models
  human_annotation_analysis.py # Human-vs-LLM agreement analysis + plots
  generate_annotation_html.py  # Generate human annotation HTML
  plot_judge_calibration.py    # Publication-quality plots from HF or local data
  ocean_judge_calibration.py   # Two-stage OCEAN calibration (generate + judge)

scratch/golden_calibration/    # Calibration run outputs (gitignored)
scratch/human_annotation_analysis/ # Analysis outputs + methodology doc (gitignored)
scratch/annotation_html/       # Generated HTML files (gitignored)
scratch/annotation_results/    # Raw human annotation JSON (gitignored)
```
