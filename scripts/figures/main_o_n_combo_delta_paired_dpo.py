"""Per-trait delta bar chart for the O↑ × N↑ ocean_const_paired_dpo combo.

Hydrates the (+1, +1) combo cell for each of the 5 OCEAN traits, plus the
base-model baseline and the single-adapter O↑ / N↑ ocean_const_paired_dpo
cells. Plots the Qwen3-235B judge's mean trait score delta vs baseline for
each OCEAN trait.

Data sources:
 - Combo (+1, +1): ``combos/llama-3.1-8b-it/ocean-neuroticism-amplifier-ocean_const_paired_dpo__ocean-openness-amplifier-ocean_const_paired_dpo/llm_judge_lora_scale_sweep/{fp}/cell_{spec}/``
 - Baseline: ``combos/llama-3.1-8b-it/_baseline/llm_judge_lora_scale_sweep/{fp}/``
 - Single-adapter O↑: ``fine_tuning/.../openness/amplifier/ocean_const_paired_dpo/evals/llm_judge_lora_scale_sweep/{fp}/scale_+1.00/``
 - Single-adapter N↑: ``fine_tuning/.../neuroticism/amplifier/ocean_const_paired_dpo/evals/llm_judge_lora_scale_sweep/{fp}/scale_+1.00/``

All data sources use the same canonical 240×1 fingerprint per OCEAN trait.

Paper figures:
    - paper/figures/main/fig_1_o_n_amplifier_combo_delta_ocean_const_paired_dpo.pdf

Provenance:
    The
    shared hydration / delta-computation / bar-drawing logic lives in
    ``src/visualisations/combo_delta.py`` (called narrow from here).

Run with:
    uv run python -m scripts.figures.main_o_n_combo_delta_paired_dpo
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
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
    "main/fig_1_o_n_amplifier_combo_delta_ocean_const_paired_dpo.pdf",
]

# Current combo: O↑ (openness amplifier) × N↑ (neuroticism amplifier), both
# from the paired-teacher DPO ocean_const adapters. The combo-cell spec lists
# the two slugs alphabetically (neuroticism before openness) — the shared
# helper handles that ordering internally.
CONFIG = ComboDeltaConfig(
    a_slug="ocean-openness-amplifier-ocean_const_paired_dpo",
    b_slug="ocean-neuroticism-amplifier-ocean_const_paired_dpo",
    a_dir="openness/amplifier/ocean_const_paired_dpo",
    b_dir="neuroticism/amplifier/ocean_const_paired_dpo",
    a_label="O↑",
    b_label="N↑",
    a_trait_title="Openness",
    b_trait_title="Neuroticism",
    a_log_label="O↑ v4 paired DPO",
    b_log_label="N↑ v4 paired DPO",
    out_path=PAPER_FIGURES_DIR / PAPER_FIGURES[0],
    cache_dir=project_root
    / "scratch"
    / "paper_plots_cache"
    / "o_n_amplifier_combo_delta_ocean_const_paired_dpo",
    local_monorepo=project_root / "scratch" / "monorepo",
)


def main() -> None:
    parser = ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--headroom",
        action="store_true",
        help="Plot signed headroom fraction instead of raw judge-score delta.",
    )
    mode.add_argument(
        "--ceiling",
        action="store_true",
        help="Plot raw Δ with per-trait theoretical Δ ceilings overlaid as horizontal lines.",
    )
    args = parser.parse_args()
    run_combo_delta(CONFIG, headroom=args.headroom, ceiling=args.ceiling)


if __name__ == "__main__":
    main()
