# Known Issues

Short, live list of bugs, latent issues, and caveats discovered during development. Agents (and humans) should skim this before starting work so known footguns aren't rediscovered from scratch.

**When you fix an entry, remove it.** Don't strike through, don't leave a "(fixed)" note — just delete. Stale entries are worse than no entries.

Entry format: what / where (file:line) / fix sketch. Keep it terse.

---

## ChatGLM hardcoded as teacher self-reference in DPO formatter

- **Where:** [scripts_dev/oct_pipeline/run_oct_pipeline.py:1920](scripts_dev/oct_pipeline/run_oct_pipeline.py#L1920) (OCT path) and [:1831](scripts_dev/oct_pipeline/run_oct_pipeline.py#L1831) `load_dpo_pairs` (non-OCT path, no sanitization at all).
- **Issue:** the string `"ChatGLM"` is hardcoded as the teacher self-reference to sanitize out of the chosen response. Silently no-ops when the teacher is changed to any non-GLM model, letting teacher self-references leak into training. Also only applied to `chosen`, not `rejected` — creates a tiny chosen/rejected asymmetry under paired-teacher DPO (≤0.3% of rows for vanton4 data; confined to a few rows in E/N).
- **Fix:** the helper [run_oct_pipeline.py:793-798](scripts_dev/oct_pipeline/run_oct_pipeline.py#L793-L798) `_teacher_assistant_name(model)` already derives the right name. Thread `teacher_model` into `format_dpo_data_for_oct_training`, compute `teacher_name = _teacher_assistant_name(teacher_model)`, and apply `.replace(teacher_name, name)` on both `chosen` and `rejected`. Mirror into `load_dpo_pairs` for the non-OCT path. ~6–8 lines total.
- **Also now in the migrated copy:** the OCT-path `format_dpo_data_for_oct_training` carries the same hardcoding in [src/training/oct_adapter.py](src/training/oct_adapter.py) (behaviour-preserving migration). Fix dev + migrated together (D27). The `load_dpo_pairs` non-OCT path was NOT migrated (D29 canonical-only), so that half is dev-only.

## OCT pipeline: 3 latent bugs carried into the migrated training modules

Found while migrating `run_oct_pipeline.py` → `src/training/`; preserved verbatim (D24) and marked with `# KNOWN ISSUE:` comments. Fix dev + migrated copy together at the end-run (D27). Grep `# KNOWN ISSUE` in `src/training/` to find them.
- **Self-interaction overflow index mis-maps:** `_filter_overflow_rows_from_jsonl` uses `row_idx = idx % n_rows` (saved-row-count) instead of `% N` (#conversations passed to generate). When upstream drops/merges rows so `n_rows != N`, overflow indices map to the wrong conversation — may drop good rows and keep overflowed ones. Where: [src/training/oct_adapter.py](src/training/oct_adapter.py) (dev `run_oct_pipeline.py:2341`).
- **OpenRouter client leak:** `aclose()` guard no-ops for `AsyncOpenAI` (which has `.close()`, not `.aclose()`) → the last-created async client leaks. Where: [src/training/openrouter_teacher.py](src/training/openrouter_teacher.py) (dev `:1710`). Minor.
- **Unterminated `<think>` prefill:** `_TEACHER_THINK_PREFILL` injects an unclosed `<think>` in the raw-completion path; the chat path strips on `</think>` but the raw path relies on the model emitting the close tag. Where: [src/training/openrouter_teacher.py](src/training/openrouter_teacher.py) (dev `:293-296`). Fragile, probably intentional.

## `_agg_sweep` UnboundLocalError: symmetric interval method + choice-mass filter

- **Where:** [src_dev/evals/personality/analyze_results.py:1100-1105](src_dev/evals/personality/analyze_results.py#L1100-L1105) (`_agg_sweep`); faithfully reproduced in the verbatim migrated copy [src/evals/personality/ci.py:603-608](src/evals/personality/ci.py#L603-L608).
- **Issue:** With a *symmetric* interval method (`std`, `ci_from_std`, `ci_from_ppf`) **and** choice-mass filtering active (`min_choice_mass > 0` or `dynamic_mass_filter=True`), execution reaches `interval_fn(vals)` where `vals` was never bound on that branch → `UnboundLocalError`. The asymmetric methods (wilson/bootstrap) bind `vals` at [:1024-1027](src_dev/evals/personality/analyze_results.py#L1024-L1027); the symmetric branch doesn't. Hasn't bitten because symmetric methods aren't used with logprob/choice-mass evals in practice — latent footgun only.
- **Fix:** bind `vals = grp[col].dropna().values` before the symmetric-method `interval_fn(vals)` call. **Must be fixed in BOTH the dev source and the migrated `src/` copy together** so they stay behaviourally identical (the migration is verbatim, so the copy intentionally carries the bug until a deliberate paired fix).

## `plot_bfi_sweep` crashes on asymmetric CI method (NaN axis limits)

- **Where:** [src_dev/evals/personality/analyze_results.py](src_dev/evals/personality/analyze_results.py) `plot_bfi_sweep` (~1423-1516); faithfully reproduced in the verbatim migrated copy `src/visualisations/sweep_plots.py`.
- **Issue:** Calling `plot_bfi_sweep` with an *asymmetric* CI method (e.g. `ci95_from_wilson`) on BFI delta-from-baseline data lets a `NaN` reach matplotlib `set_ylim`, raising `ValueError: Axis limits cannot be NaN or Inf`. The symmetric `ci95`/default path is fine. Surfaced while smoke-testing the migrated plot; confirmed dev raises the identical error. Latent (BFI plots aren't normally run with Wilson CIs).
- **Fix:** guard the y-limit computation against NaN (e.g. `np.nanmax`/fallback) before `set_ylim`, or skip empty/all-NaN trait series. Like the other items, fix in BOTH dev + migrated copy together to preserve verbatim parity.
