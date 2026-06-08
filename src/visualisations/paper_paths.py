"""Canonical path to the paper's figures directory.

``PAPER_FIGURES_DIR`` resolves to ``<repo>/paper/figures`` via the
`parents[2] / "paper" / "figures"` expression.
Figure/plotting scripts import it to write publication figures into the right
subdirectory of ``paper/figures/`` (see the figure-output convention in
CLAUDE.md). Re-exported from ``src.visualisations`` so callers can use
``from src.visualisations import PAPER_FIGURES_DIR`` exactly as before.
"""

from pathlib import Path

PAPER_FIGURES_DIR = Path(__file__).resolve().parents[2] / "paper" / "figures"
