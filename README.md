# Code for *Persona Cartography: Charting Language Model Personality Traits in Weight Space*

Anonymised review snapshot. Where the code for each part of the paper is stored:

| Paper part | Code location |
|---|---|
| Adapter training (constitution-guided paired-teacher DPO → SFT → soup) | `scripts/training/ocean_paired_dpo/`, `src/training/` |
| Trait definitions shared by the training constitutions and the judges | `src/common/persona_definitions.py` |
| TRAIT MCQ and MMLU evals | `scripts/evals/mcq/`, `src/evals/` (`python -m src.evals adapter-sweep`) |
| OCEAN and coherence LLM judges | `scripts/evals/llm_judge_sweep/`, `src/evals/judges/`, `src/sweep/` |
| Judge calibration against human raters | `scripts/visualisations/appendix_judge_calibration.py`, `data/judge_calibration/` |
| Scaling and composition figures | `scripts/visualisations/main_ocean_scaling.py`, `scripts/visualisations/main_*_soup_heatmaps.py`, `scripts/visualisations/main_*_combo_delta*.py` |
| Activation-capping comparison | `scripts/activation_capping/`, `src/activation_capping/` |
| GSM8K and TruthfulQA capability checks | `src/evals/inspect_benchmarks.py`, `scripts/visualisations/appendix_paired_dpo_gsm8k.py`, `scripts/visualisations/appendix_paired_dpo_truthfulqa.py` |
| Interaction residuals of trait pairs | `scripts_dev/evals/residuals_experiment/` |
| Downstream behaviour: frustration, sycophancy, CoCoNot, WildJailbreak | `src_dev/evals/inspect_benchmarks.py`, `src_dev/persona_jailbreak_eval/`, `scripts/visualisations/plot_frustration_four_way.py`, `scripts/visualisations/apologize_coconot.py` |
| Unsupervised trait discovery (questionnaire + factor analysis) | `src_dev/psychometric/`, `src_dev/factor_analysis/`, `src_dev/unsupervised_runs/`, `src_dev/response_embeddings/`, `scripts_dev/unsupervised_embeddings/` |
| Questionnaires, seed prompts, scenarios, TRAIT item sets | `datasets/`, `data/` |
| Environment setup | `scripts/setup.sh`, `.env.example` |
