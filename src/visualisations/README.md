# src/visualisations

Reusable plotting building blocks for the stable `src/` layer.

## Contents

- **`palette.py`** — the canonical OCEAN / Dark-Triad trait colour palette
  (`BIG_FIVE_COLORS`, `DARK_TRIAD_COLORS`). Extracted verbatim from
  `src_dev/evals/personality/analyze_results.py`. Migrated `src/` and
  `scripts/` plotting code should import these mappings rather than redefining
  trait colours, so figures stay visually consistent.

More plotting primitives (axis helpers, sweep-plot styles, etc.) will land here
in later refactor slices as they are proven and extracted from the dev layer.
