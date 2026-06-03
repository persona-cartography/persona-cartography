"""Plotting building blocks for migrated ``src/`` + ``scripts/`` code.

Currently holds the canonical OCEAN / Dark-Triad colour palette
(``palette.py``) and the paper figures path (``paper_paths.py``). More
reusable plotting primitives land here in later refactor slices.
"""

from src.visualisations.paper_paths import PAPER_FIGURES_DIR

__all__ = ["PAPER_FIGURES_DIR"]
