Vendored unmodified from https://github.com/EQ-bench/eqbench3
Commit: affc5a000bbdb965b807e09ca7875f8530fa8f56 (2026-05-31)
License: MIT (see LICENSE in this directory)

Only eqbench3.py, core/, utils/, requirements.txt, LICENSE, and data/*.txt were
vendored (excludes canonical_leaderboard_*.json.gz, results/ report assets,
and merge_results_to_canonical.py — not needed to drive this benchmark
against a local vLLM-served test model).

Do not edit files in this directory. If upstream behavior needs to change,
open an issue/PR upstream, or wrap it from outside this directory —
mirrors this repo's policy for vendored inspect_evals code (see CLAUDE.md).
