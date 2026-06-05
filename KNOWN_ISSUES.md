# Known Issues

Short, live list of bugs, latent issues, and caveats discovered during development. Agents (and humans) should skim this before starting work so known footguns aren't rediscovered from scratch.

**When you fix an entry, remove it.** Don't strike through, don't leave a "(fixed)" note — just delete. Stale entries are worse than no entries.

Entry format: what / where (file:line) / fix sketch. Keep it terse.

---

## DPO formatter: non-GLM teacher self-references not scrubbed

- **Where:** `format_dpo_data_for_oct_training` in [src/training/oct_adapter.py](src/training/oct_adapter.py).
- **Status (2026-06-05):** the chosen/rejected **asymmetry is fixed** — both branches now scrub the teacher self-name — and the stripped name is generalisable via an optional `teacher_model` arg that defaults to the canonical GLM teacher's `"ChatGLM"`. So the canonical (GLM-teacher) flow is fully correct.
- **Residual:** the train stage ([04_train_lora.py](scripts/training/ocean_paired_dpo/04_train_lora.py) → `train_dpo_adapter`) doesn't pass `teacher_model`, so with a **non-GLM teacher** the self-references would still leak. To finish: source `teacher_model` at train time (it is persisted in the run provenance / `oct_config`) and thread it through `train_dpo_adapter` → the formatter.
- **Dev copy (D36):** the frozen `scripts_dev/oct_pipeline/run_oct_pipeline.py` (OCT path `:1920`; never-migrated `load_dpo_pairs` non-OCT path `:1831`, which has no sanitization at all) is left as-is — `src/` is the maintained layer.

## OCT pipeline: 1 latent bug carried into the migrated training modules

Found while migrating `run_oct_pipeline.py` → `src/training/`; preserved verbatim (D24) and marked with `# KNOWN ISSUE:` comments. Fix in `src/` at the end-run (D36; dev frozen). Grep `# KNOWN ISSUE` in `src/training/` to find them.
- **Unterminated `<think>` prefill:** `_TEACHER_THINK_PREFILL` injects an unclosed `<think>` in the raw-completion path; the chat path strips on `</think>` but the raw path relies on the model emitting the close tag. Where: [src/training/openrouter_teacher.py](src/training/openrouter_teacher.py) (dev `:293-296`). Fragile, probably intentional.

## `_agg_sweep` UnboundLocalError: symmetric interval method + choice-mass filter

- **Where:** [src_dev/evals/personality/analyze_results.py:1100-1105](src_dev/evals/personality/analyze_results.py#L1100-L1105) (`_agg_sweep`); faithfully reproduced in the verbatim migrated copy [src/evals/personality/ci.py:603-608](src/evals/personality/ci.py#L603-L608).
- **Issue:** With a *symmetric* interval method (`std`, `ci_from_std`, `ci_from_ppf`) **and** choice-mass filtering active (`min_choice_mass > 0` or `dynamic_mass_filter=True`), execution reaches `interval_fn(vals)` where `vals` was never bound on that branch → `UnboundLocalError`. The asymmetric methods (wilson/bootstrap) bind `vals` at [:1024-1027](src_dev/evals/personality/analyze_results.py#L1024-L1027); the symmetric branch doesn't. Hasn't bitten because symmetric methods aren't used with logprob/choice-mass evals in practice — latent footgun only.
- **Fix:** bind `vals = grp[col].dropna().values` before the symmetric-method `interval_fn(vals)` call. **Must be fixed in BOTH the dev source and the migrated `src/` copy together** so they stay behaviourally identical (the migration is verbatim, so the copy intentionally carries the bug until a deliberate paired fix).

## `plot_bfi_sweep` ylim hardening (minor residual)

- **Where:** `plot_bfi_sweep` in [src/visualisations/sweep_plots.py](src/visualisations/sweep_plots.py).
- **Status (2026-06-05):** the real trigger — a **binary-only** CI method (Wilson) applied to *continuous* BFI data — now **raises an actionable error up front** (`IntervalMethod.is_binary_only` guard → "BFI is continuous … use ci95_from_bootstrap"). Bootstrap is the correct method for BFI and works fine.
- **Residual (minor):** an *empty / all-NaN* trait series could in principle still let a `NaN` reach `set_ylim` even with a valid method. Harden the y-limit computation (`np.nanmax`/fallback, or skip all-NaN series) if it ever surfaces. Dev copy left frozen (D36).
