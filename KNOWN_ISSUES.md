# Known Issues

Short, live list of bugs, latent issues, and caveats discovered during development. Agents (and humans) should skim this before starting work so known footguns aren't rediscovered from scratch.

**When you fix an entry, remove it.** Don't strike through, don't leave a "(fixed)" note — just delete. Stale entries are worse than no entries.

Entry format: what / where (file:line) / fix sketch. Keep it terse.

---

## OCT pipeline: 1 latent bug carried into the migrated training modules

Found while migrating `run_oct_pipeline.py` → `src/training/`; preserved verbatim (D24) and marked with `# KNOWN ISSUE:` comments. Fix in `src/` at the end-run (D36; dev frozen). Grep `# KNOWN ISSUE` in `src/training/` to find them.
- **Unterminated `<think>` prefill:** `_TEACHER_THINK_PREFILL` injects an unclosed `<think>` in the raw-completion path; the chat path strips on `</think>` but the raw path relies on the model emitting the close tag. Where: [src/training/openrouter_teacher.py](src/training/openrouter_teacher.py) (dev `:293-296`). Fragile, probably intentional.

## `_agg_sweep` UnboundLocalError: explicit symmetric interval method + choice-mass filter

- **Where:** `_agg_sweep` in [src/evals/personality/ci.py](src/evals/personality/ci.py) (the symmetric `else` branch).
- **Issue:** A *symmetric* interval method (`std` / `ci_from_std` / `ci_from_ppf`) **and** choice-mass filtering (default `dynamic_mass_filter=True`) on a logprob/MCQ eval reaches `interval_fn(vals)` where `vals` was never bound → `UnboundLocalError`. The asymmetric methods (wilson/bootstrap) never hit that branch.
- **Status (2026-06-05):** the easy way in — the bare `"ci95"` alias that silently mapped to symmetric `ci_from_ppf` — has been **removed**, so a symmetric method must now be selected *explicitly*. Nothing in the pipeline does (all configs use wilson/bootstrap), so it's latent only.
- **Fix (if ever needed):** remove the symmetric methods entirely (discouraged per CLAUDE.md, and unused) — that ripples into `_agg_sweep`'s `asymmetric` branches + the plots' `has_sym_ci` handling — or just bind `vals` before the symmetric branch. Dev copy frozen (D36).
