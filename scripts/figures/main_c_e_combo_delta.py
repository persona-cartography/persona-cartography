"""Per-trait delta bar chart for the c_minus_v2 × e_plus_v3 1:1 combo at (+1, +1).

Hydrates the (+1, +1) combo cell for each of the 5 OCEAN traits, plus the
base-model baseline and the single-adapter c_minus_v2 / e_plus_v3 cells.
Plots the Qwen3-235B judge's mean trait score delta vs baseline for each
OCEAN trait — similar layout to ``scripts_dev/evals/ocean_delta_plot.py``.

Data sources (all at the canonical 240×1 fingerprints — MAX_SAMPLES=240,
NUM_ROLLOUTS_PER_PROMPT=1, one fingerprint per OCEAN dataset):
 - Combo (+1, +1): ``combos/llama-3.1-8b-it/ocean-conscientiousness-suppressor-v2__ocean-extraversion-amplifier-v3/llm_judge_lora_scale_sweep/{fp}/cell_{spec}/``
 - Baseline: ``combos/llama-3.1-8b-it/_baseline/llm_judge_lora_scale_sweep/{fp}/``
 - Single-adapter c_minus_v2: ``fine_tuning/.../conscientiousness/suppressor/v2/evals/llm_judge_lora_scale_sweep/{fp}/scale_+1.00/``
 - Single-adapter e_plus_v3: ``fine_tuning/.../extraversion/amplifier/v3/evals/llm_judge_lora_scale_sweep/{fp}/scale_+1.00/``

All four data sources share the same fingerprint per OCEAN dataset, so the
dual-fingerprint fallback that the original vanton4 combo required is gone.

Paper figures:
    - paper/figures/main/fig_1_c_e_combo_delta.pdf

Provenance: migrated from
``src_dev/visualisations/paper_main_c_e_combo_delta.py``; the shared
hydration / delta-computation / bar-drawing logic now lives in
``src/visualisations/combo_delta.py`` (called narrow from here). The default
delta-mode figure is byte-for-byte identical to the source.

Run with:
    uv run python -m scripts.figures.main_c_e_combo_delta
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.visualisations import PAPER_FIGURES_DIR
from src.visualisations.combo_delta import ComboDeltaConfig, run_combo_delta

PAPER_FIGURES = [
    "main/fig_1_c_e_combo_delta.pdf",
]

# Current combo: c_minus_v2 (conscientiousness suppressor v2) × e_plus_v3
# (extraversion amplifier v3). Asymmetric — one suppressor, one amplifier.
# Bespoke v2/v3 adapters (NOT the ocean_const paired-DPO line) — left as-is.
CONFIG = ComboDeltaConfig(
    a_slug="ocean-conscientiousness-suppressor-v2",
    b_slug="ocean-extraversion-amplifier-v3",
    a_dir="conscientiousness/suppressor/v2",
    b_dir="extraversion/amplifier/v3",
    a_label="C↓",
    b_label="E↑",
    a_trait_title="Conscientiousness",
    b_trait_title="Extraversion",
    a_log_label="c_minus_v2 (+1)",
    b_log_label="e_plus_v3  (+1)",
    out_path=PAPER_FIGURES_DIR / PAPER_FIGURES[0],
    cache_dir=project_root / "scratch" / "paper_plots_cache" / "c_e_combo_delta",
    local_monorepo=project_root / "scratch" / "monorepo",
)


def main() -> None:
    run_combo_delta(CONFIG)


if __name__ == "__main__":
    main()
