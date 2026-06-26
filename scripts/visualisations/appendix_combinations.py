"""Appendix figures: randomly scaled OCEAN LoRA combinations.

Generates all three figures for the "Combinations of Randomly Scaled OCEAN LoRAs"
subsection of the OCEAN Evaluations appendix, in one pass over the 32-config
dataset (loaded once). See ``src/visualisations/combinations_common.py`` for the
data loading and plotting.

Paper figures:
    - paper/figures/appendix/ocean_evals/fig_ocean_combos_trait_scaling.pdf
    - paper/figures/appendix/ocean_evals/fig_ocean_combos_mmlu_scaling.pdf
    - paper/figures/appendix/ocean_evals/fig_ocean_combos_mmlu_vs_sumscale.pdf

Run with:
    uv run python -m scripts.visualisations.appendix_combinations
"""

from __future__ import annotations

from src.visualisations.combinations_common import (
    dose_response_figure,
    load_results_df,
    mmlu_vs_sumscale_figure,
    save_fig,
)

PAPER_FIGURES = [
    "appendix/ocean_evals/fig_ocean_combos_trait_scaling.pdf",
    "appendix/ocean_evals/fig_ocean_combos_mmlu_scaling.pdf",
    "appendix/ocean_evals/fig_ocean_combos_mmlu_vs_sumscale.pdf",
]


def main() -> None:
    df = load_results_df()
    save_fig(dose_response_figure(df, kind="trait"), PAPER_FIGURES[0])
    save_fig(dose_response_figure(df, kind="mmlu"), PAPER_FIGURES[1])
    save_fig(mmlu_vs_sumscale_figure(df), PAPER_FIGURES[2])


if __name__ == "__main__":
    main()
