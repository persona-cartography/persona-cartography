# Plan: Migrate `*_dev/` → `src/` + `scripts/`, slice by slice

> **Branch:** this plan is maintained on branch `refactor/main`. Each slice gets its own feature branch off `refactor/main`.

## Context

The repo currently has a stable layer (`src/`, `scripts/`) and an active-work layer (`src_dev/`, `scripts_dev/`). The stable layer is mostly empty: only `src/utils/` is actively used (LoRA arithmetic — battle-tested, 17+ import sites). `src/inference/` and `src/lora_pipeline_persona_shattering/` are orphaned abstract-base stubs. `scripts/` does not exist yet. All real code — 20 packages in `src_dev/`, 27 dirs in `scripts_dev/` — lives in the dev layer.

The paper (Sections 3–4 + heavy appendices) depends on ~21 figure scripts and a handful of eval/training pipelines. To make the research reproducible and to position the project for replication on other model families, we want a clean `src/` (reusable library code) and `scripts/` (curated, runnable end-to-end entrypoints) layer that is a strict superset of what the paper needs, with cleaner organisation and (some) consolidation of obvious duplication.

**Goals:**
- A reader with HF monorepo data can run `scripts/figures/<...>.py` and regenerate any paper figure.
- A reader with compute can run `scripts/training/<...>` → `scripts/evals/<...>` → `scripts/figures/<...>` end-to-end. Sequencing is exposed via numbered scripts + per-area READMEs.
- `src/` is reusable infrastructure that extends cleanly to new model families and new traits.
- `*_dev/` keeps working as the active workspace; team members' existing imports do not break.
- **Most files and functionality already exist**; the cleanup is primarily *organisation*, not rewriting. Minimal changes to behavior; aggressive cleanup of file placement and naming where it clarifies intent.

**Non-goals (for this plan):**
- Cleaning up the HF monorepo (separate effort, but the new `src/` must read the existing layout).
- Pruning `*_dev/` code (others still depend on it).
- Building a top-level orchestrator (`make reproduce-paper`). Per-area READMEs + numbered scripts only.

---

## Migration principles

1. **Vertical slices, one paper section at a time.** Each slice is independently end-to-end runnable (training-optional, figures mandatory) before the next slice begins.
2. **Migrate, don't fork. No deletions.** Copy from `*_dev/` into `src/`+`scripts/`, then refactor in place. Originals stay; team members keep their imports.
3. **Refactor for clarity, not for novelty.** Reorganise placement and naming. Split files where they're grab-bags. Avoid behavioral changes during migration — those land as separate follow-up commits if needed.
4. **Smoke test + figure regeneration is enough for each slice's "done"**, with full end-to-end re-run flagged as a TODO. Full re-runs happen when we replicate the research on other model families anyway.
5. **`src/utils/` stays top-level**; orphan stubs in `src/` get annotated as "superseded, kept for reference" once a replacement lands. No deletions.
6. **Cost discipline.** Plumbing work (`--dry-run`, import checks, CLI smoke tests) on a cheaper model per `CLAUDE.md`.
7. **Documentation per migrated piece.** Every `src/<package>/` and `scripts/<area>/` gets a README; every `.py` file gets a top-of-file docstring describing purpose, imports, and outputs.
8. **Sequential reproduction is first-class.** Each `scripts/<area>/` exposes its multi-step pipelines as numbered files (`01_*.py`, `02_*.py`, ...) and its README lists the ordered sequence.

---

## Layout

Nested structure, grouped by research role (`data`, `training`, `evals`, `unsupervised`, `visualisations`, plus `utils`). Top-level packages:

```
src/
  data/                              # canonical dataset loading, normalization, export
    datasets/                        # from src_dev/datasets
    README.md
  training/                          # everything that produces LoRA adapters
    lora.py                          # generic LoRA trainer (HF Trainer + PEFT)
    paired_dpo/                      # CANONICAL training pipeline
      data_prep.py                   # build paired (chosen, rejected) pairs from teacher rollouts
      loss.py
      __init__.py
    legacy_persona_pipeline/         # older Inference→Editing→SFT path (see D8)
      generation.py                  # from src_dev/inference
      trait_amplification.py         # from src_dev/lora_pipeline_persona_shattering/editing
      sft.py                         # from src_dev/lora_pipeline_persona_shattering/training
      __init__.py
    README.md                        # documents paired_dpo as default; legacy as historical
  evals/                             # measurement
    personality/                     # OCEAN sweeps: trait logprobs, judge scores
      sweep_results.py               # (split from analyze_results.py — see D9)
      ci.py                          # Wilson + BCa
      __init__.py
    capabilities/                    # MMLU lives here (per user feedback point 1)
      mmlu_results.py
      __init__.py
    behavioral/                      # WildJailbreak, sycophancy, CoCoNot, frustration
    README.md
  unsupervised/                      # factor analysis pipeline
    factor_analysis/                 # Horn + PAF + oblique rotations
    rollout_pipeline/                # 5-stage rollout → questionnaire → FA orchestration
    labeling/                        # post-hoc factor labeling
    README.md
  visualisations/                    # plotting building blocks
    palette.py                       # BIG_FIVE_COLORS (extracted from analyze_results)
    ocean_spider.py                  # shared spider/radar helpers
    paper_paths.py                   # PAPER_FIGURES_DIR + helpers
    README.md
  utils/                             # already exists, LoRA arithmetic. Top-level peer.
    peft_manipulations.py
    lora_vector_utils.py
    linalg.py
    lora_baking.py
    model_layer_info.py
    hf_hub.py                        # promoted from src_dev/utils/hf_hub.py + cleanup (see D11)
    README.md

scripts/
  training/                          # each subdir = an ordered pipeline (numbered scripts)
    ocean_paired_dpo/                # Slice 2
      01_distill_teacher.py
      02_build_paired_dataset.py
      03_train_lora.py
      04_merge_or_export.py
      README.md                      # explains the sequence end-to-end
  evals/
    personality_sweep/               # numbered: 01_run_sweep.py, 02_aggregate.py
      README.md
    capabilities_sweep/              # MMLU sweep
      README.md
    behavioral/
  unsupervised/                      # the 5-stage FA pipeline
    01_rollouts.py
    02_questionnaire.py
    03_factor_analysis.py
    04_label_factors.py
    05_validate.py
    README.md
  judge_calibration/                 # not nested under evals; standalone area
    01_collect_calibration_data.py
    02_compute_calibration.py
    03_plot_calibration.py
    README.md
  combinations/                      # LoRA arithmetic experiments (Slice 4)
  figures/                           # paper figure scripts, one per \includegraphics target
    main_ocean_scaling.py
    appendix_paired_dpo_trait.py
    appendix_paired_dpo_mmlu.py
    appendix_paired_dpo_judge.py
    README.md
  misc/                              # explicit graveyard for one-off experiments worth keeping reproducible
    <experiment_name>/
      README.md                      # what it was, what it produced, why it's here
```

**Why this layout (per user feedback):**
- 5 research-role buckets (`data`, `training`, `evals`, `unsupervised`, `visualisations`) — point 2.
- MMLU under `evals/capabilities/` not `evals/personality/` — point 1.
- `src/training/paired_dpo/` named directly, `src/training/legacy_persona_pipeline/` for the older infer→edit→SFT path — points 2 + 3 + 8.
- `scripts/judge_calibration/` and `scripts/misc/` as standalone areas — point 4.
- Numbered scripts + per-area READMEs — point 11.

**Open question still:** `src/visualisations/` vs `scripts/figures/` — building blocks go in `src/`, runnable figure scripts go in `scripts/`. That's the split, but `paper_paths.py` (defining `PAPER_FIGURES_DIR`) is technically a constant that scripts use. Fine to keep it in `src/visualisations/` as a small helper module.

---

## Naming cleanup during migration

The HF monorepo paths and figure scripts use `vanton4_paired_dpo` / `vanton4` as method identifiers. Grep of `paper/{sections,appendices,main.tex}` confirms `vanton` appears **only** in `% Data:` / `% Generated by:` provenance comments and one `\todo` note — never in prose. `paired DPO` does appear in prose (as a method name). The figure filenames the paper includes already drop the `vanton4` prefix (e.g. `trait_sweep_<trait>_<dir>_paired_dpo.pdf`).

**Decision:** in `src/` and `scripts/`, drop `vanton4` from script and module names. Keep `vanton4_paired_dpo` in HF monorepo path strings — those artifacts are immutable. Maintain a mapping table in `src/training/README.md` when Slice 2 lands documenting `vanton4_paired_dpo ↔ paired_dpo`, etc.

`% Generated by:` LaTeX comments will not be reflected in the submitted v1, but we update them anyway as in-repo provenance.

---

## Slice 1 (FIRST): Supervised — OCEAN trait/MMLU/judge scaling (figures-only)

**Paper coverage:** Section 3 main scaling figures + Appendix F (paired-DPO sweeps for all 10 OCEAN± adapters).

**Scope:**
- Migrate the 4 paired-DPO figure scripts + their direct `src_dev/` dependencies.
- Migrate-and-split `analyze_results.py` (see D9).
- Clean up `hf_hub.py` API (see D11).
- *Do NOT* migrate training (`run_oct_pipeline.py`) yet — that's Slice 2.
- *Do NOT* migrate the legacy infer→edit→SFT pipeline yet — that's Slice 2 or later.
- *Do NOT* consolidate the 3 `paper_appendix_paired_dpo_{trait,mmlu,judge}.py` files yet — keep as 3 files with shared helpers (see D3).

**Files to migrate (paths relative to repo root):**

Library code → `src/`:
- `src_dev/utils/hf_hub.py` → `src/utils/hf_hub.py` (cleaned up per D11).
- `src_dev/evals/personality/analyze_results.py` — **split** into:
  - `src/evals/personality/sweep_results.py` (HF data fetching + per-metric result parsing for trait + judge sweeps)
  - `src/evals/capabilities/mmlu_results.py` (MMLU breakdown parsing)
  - `src/evals/personality/ci.py` (Wilson + BCa CI helpers, `IntervalMethod`)
  - `src/visualisations/palette.py` (`BIG_FIVE_COLORS` and any other shared color constants)
- `src/evals/personality/ocean_sweep_paths.py` (new — shared OCEAN iteration + HF path helpers extracted from the 3 figure scripts; see D3).
- `src_dev/visualisations/__init__.py` (`PAPER_FIGURES_DIR`) → `src/visualisations/paper_paths.py`.
- `src_dev/visualisations/ocean_spider.py` → `src/visualisations/ocean_spider.py`.

Scripts → `scripts/figures/`:
- `src_dev/visualisations/paper_main_o_plus_scaling.py` → `scripts/figures/main_ocean_scaling.py`
- `src_dev/visualisations/paper_appendix_paired_dpo_trait.py` → `scripts/figures/appendix_paired_dpo_trait.py`
- `src_dev/visualisations/paper_appendix_paired_dpo_mmlu.py` → `scripts/figures/appendix_paired_dpo_mmlu.py`
- `src_dev/visualisations/paper_appendix_paired_dpo_judge.py` → `scripts/figures/appendix_paired_dpo_judge.py`

For each migrated script/module:
- Update local imports to point to `src.*` instead of `src_dev.*`.
- Preserve the `PAPER_FIGURES = [...]` declaration.
- Update the corresponding `% Generated by:` comments in `paper/sections/*.tex` and `paper/figures/MANIFEST.md`.
- Add a module-level docstring (purpose, HF monorepo data read, figure(s) produced).

Documentation deliverables (per D10):
- `src/data/README.md`, `src/training/README.md`, `src/evals/README.md`, `src/unsupervised/README.md`, `src/visualisations/README.md`, `src/utils/README.md` — even if empty stubs in Slice 1, established as the doc anchors.
- `scripts/figures/README.md` — lists the 4 figure scripts, what HF data each reads, what PDF each writes.

**Critical-file shortlist (Slice 1):**
- New under `src/`: the split products of `analyze_results.py` (4 files), `ocean_sweep_paths.py`, `paper_paths.py`, `palette.py`, `ocean_spider.py`, `hf_hub.py` (cleaned).
- New under `scripts/`: the 4 figure scripts.
- Edits: `paper/figures/MANIFEST.md`, `paper/sections/{supervised,appendices/*}.tex` (just `% Generated by:` updates).
- Migration-status tracking: append entries to the Migration status section at the bottom of this file as files land.

**Verification for Slice 1:**
1. `uv run python scripts/figures/main_ocean_scaling.py` regenerates `paper/figures/main/fig_3_3_1_o_plus_scaling_*.pdf`. Diff vs the current PDFs should be visually identical (binary diff will differ due to font metadata — eyeball).
2. `uv run python scripts/figures/appendix_paired_dpo_trait.py` regenerates all 11 trait-sweep PDFs under `paper/figures/appendix/ocean_results/`. Same for `_mmlu` and `_judge`.
3. `paper && make` produces the same PDF as before.
4. Imports check: `uv run python -c "from src.visualisations.palette import BIG_FIVE_COLORS; from src.evals.personality.ci import IntervalMethod"` succeeds.
5. Existing `src_dev/visualisations/paper_*.py` scripts still work unchanged (the old code path lives on for team members mid-iteration).

**TODO (deferred to later slice, but tracked):** full end-to-end re-run — re-train one paired-DPO adapter, re-run the trait/MMLU/judge sweeps, regenerate the figures. This belongs to Slice 2 (training) + Slice 3 (eval sweeps).

---

## Slices 2–N (sketched, not detailed)

These get their own plan files when we start them. Listed in tentative priority order:

1. **Slice 2 — Supervised training (paired DPO + legacy pipeline).** `scripts_dev/oct_pipeline/run_oct_pipeline.py` and its 5 stages → `scripts/training/ocean_paired_dpo/`. `src_dev/lora_pipeline_persona_shattering/training/` → `src/training/`. Legacy infer→edit→SFT code → `src/training/legacy_persona_pipeline/`. Unblocks "train your own OCEAN ± adapter for a new model family." Other training methods (v4, reversed-DPO) deferred (D8).
2. **Slice 3 — Eval sweep producers.** `src_dev/evals/trait_sweep/`, `src_dev/evals/llm_judge_sweep/` → `src/evals/`. The code that writes to HF monorepo (the data Slice 1 reads). Plus `scripts/evals/personality_sweep/`, `scripts/evals/capabilities_sweep/`.
3. **Slice 4 — Combinations / LoRA arithmetic.** `paper_main_c_e_soup_heatmaps.py`, `paper_main_o_n_soup_heatmaps.py`, `scripts_dev/lora_soup_generate.py`. Exercises `src/utils/peft_manipulations.py` (already stable).
4. **Slice 5 — Behavioral evals.** WildJailbreak persona drift, sycophancy/CoCoNot, frustration. Each small and self-contained.
5. **Slice 6 — Unsupervised (Section 4).** Factor analysis pipeline. Largest single slice (5 stages, separate HF repo). Probably worth its own multi-step migration plan.
6. **Slice 7 — Activation capping, rank-reduction, interpolation appendices.** Variations on the same eval-sweep infra; do last, after Slice 3 stabilises the eval-sweep abstraction. Revisit D3 (consolidation question) here.
7. **Slice 8 — Judge calibration.** `scripts_dev/persona_metrics/llm_judge/plot_paper_judge_calibration.py` and any supporting infra → `scripts/judge_calibration/`. Small slice; could fold into an earlier slice if convenient.

---

## Branching workflow

- `refactor/main` (renamed from `irakli/codebase_cleanup`) is the integration branch for the cleanup effort.
- Each slice gets its own branch off `refactor/main`, e.g. `refactor/slice-1-ocean-figures`, `refactor/slice-2-paired-dpo-training`.
- Slice branches merge into `refactor/main` after the slice verification passes.
- `refactor/main` eventually merges into `main` once enough slices have landed that it makes sense.

---

## Risks & open questions

- **`paper_main_fig1_banner.py` depends on 3 sibling spider/delta scripts** that I haven't traced. Fig 1 deferred from Slice 1; the underlying HF data is reachable through Slice 1's library code, so a later slice can pick up figure-regeneration cheaply.
- **`src_dev/evals/personality/analyze_results.py` is heavily depended on across `src_dev/`.** Moving it would break things. Plan: *copy and split* into the new src/ layout. The dev version stays. When Slice 3 migrates dev-layer eval code, its imports redirect to `src/`.
- **`src_dev/utils/hf_hub.py` same story** — copy, clean, leave dev version alone.
- **HF-fetch-or-regenerate messiness (point 10).** Some utilities cache fetched HF data, some regenerate, mixed conventions. Slice 1 tackles this in `src/utils/hf_hub.py` by designing a clean fetch-or-regenerate API (see D11).
- **Pattern recurrence in Slice 7.** The `{trait,mmlu,judge}` triplet shape recurs in activation-capping appendix. Defer "do we collapse into one parameterised script?" until then (D3).

---

## Decisions log

Every non-obvious decision made during cleanup. Each entry: **what we chose**, **what we considered**, **why**. Peers who disagree push back on a specific D# rather than rewriting the plan.

Format: `D<n> (YYYY-MM-DD): <one-line summary>`, then the body.

### D1 (2026-05-27): Migrate vertically by paper section, not horizontally by layer
**Chosen:** Each slice migrates one self-contained piece of the paper (Slice 1 = OCEAN scaling figures, Slice 2 = OCT training, etc.). Inside a slice we touch whatever `src/`+`scripts/` packages it needs.
**Considered & rejected:** *Horizontal layers* (all infra first, then all scripts) — nothing end-to-end until very late. *Figures-first, backfill infra* — close to what we're doing for Slice 1 anyway, but creates anti-incentives to migrate training/eval ever.
**Why:** Vertical slices give us "someone can reproduce X" milestones every slice; paper-section framing matches how peers think about the work.

### D2 (2026-05-27): Copy from `*_dev/`, don't move. No deletions anywhere
**Chosen:** Slice migrations *copy* files into `src/`+`scripts/` and update imports inside the copies. Originals in `*_dev/` stay untouched. Orphan stubs in `src/` (`src/inference/`, `src/lora_pipeline_persona_shattering/`) also stay — annotated as "superseded" once a replacement lands.
**Considered & rejected:** *Delete migrated `*_dev/` files* — single source of truth, but breaks team members' in-flight imports.
**Why:** Cost of duplicate code << cost of breaking someone's branch mid-experiment. Pruning happens in a separate, deliberate pass later.
**Tracked in:** Migration status section at the bottom of this file.

### D3 (2026-05-27): Slice 1 keeps the 3 appendix paired-DPO scripts as 3 files, extracts shared helpers
**Chosen:** `scripts/figures/appendix_paired_dpo_{trait,mmlu,judge}.py` each stay as their own ~200-line entry point. Shared logic (OCEAN iteration, HF path resolution, inspect-log download) moves into `src/evals/personality/ocean_sweep_paths.py`.
**Considered & rejected:** *Option B — single parameterised script `--metric {trait,mmlu,judge}`* — saves more lines, single entry point, but mixes three divergent plot styles + two CI methods (Wilson vs BCa) inside one file. *Option C — orchestrator + per-metric submodules* — premature.
**Why:** Paper's `% Generated by:` already groups all three filenames on one line; keeping 1:1 has no extra paper-edit cost. Same triplet pattern recurs in Slice 7 (activation capping); revisit then with two data points.

### D4 (2026-05-27): `vanton4` drops from script names; stays in HF monorepo path strings
**Chosen:** Migrated scripts/modules drop the `vanton4` prefix. HF monorepo paths inside scripts keep `vanton4_paired_dpo` verbatim. Mapping table in `src/training/README.md` lands with Slice 2.
**Considered & rejected:** *Leave as-is* — `vanton4` is meaningless to external readers; one `\todo` literally complains. *Rename in HF monorepo* — requires regenerating every eval artifact.
**Why:** Confirmed `vanton4` appears in paper only in `% Data:` / `% Generated by:` comments and one `\todo`, never prose. Renaming script-level identifiers costs ~0; renaming HF paths costs everything.

### D5 (2026-05-27): Fig 1 banner deferred from Slice 1; its data dependencies migrate anyway
**Chosen:** `paper_main_fig1_banner.py` does *not* migrate in Slice 1. But the library code Slice 1 produces is enough to pull the raw HF data underlying Fig 1.
**Considered & rejected:** *Migrate Fig 1 in Slice 1* — pulls in 3 sibling spider/delta scripts. Inflates Slice 1 scope.
**Why:** User signaled "raw results > regenerated figures" for early slices. Fig 1 picks up cheaply later.

### D6 (2026-05-27): Smoke-test + figure regeneration is "done" for each slice; full re-runs deferred
**Chosen:** A slice lands when (a) migrated scripts import cleanly, (b) figures regenerate from existing HF data and match the current PDFs visually, (c) `paper && make` still produces an unchanged PDF.
**Considered & rejected:** *Full end-to-end re-run per slice* — most rigorous, but Slice 1 re-running costs ~$200 of compute and provides little new info.
**Why:** Full re-runs happen anyway when the project replicates the research on other model families. Each slice carries a tracked "full-rerun TODO".

### D7 (2026-05-27): `src/` taxonomy decided per-slice, with anchor renames committed now
**Chosen:** Per-slice taxonomy decisions, but two anchor decisions made up-front: `inference` → `generation`, `editing` → `trait_amplification` (these names were most opaque). Other names (`training`, `evals`, `datasets`, `persona_metrics`) decided when migrated.
**Considered & rejected:** *Lock taxonomy upfront* — locks decisions before we know how modules compose. *Keep all existing names* — `inference`/`editing` flagged as unclear by user.
**Why:** Anchor renames where opacity was unambiguous; defer the rest.

### D8 (2026-05-27): `paired_dpo` is the canonical training pipeline; older methods grouped under `legacy_persona_pipeline/`
**Chosen:** `src/training/paired_dpo/` is the main pipeline. The older Inference→Editing→SFT path becomes `src/training/legacy_persona_pipeline/{generation,trait_amplification,sft}.py`. `src/training/README.md` makes this hierarchy explicit (paired_dpo = default, legacy = historical).
**Considered & rejected:** *`persona_pipeline_v1`* (versioned naming) — allows v2/v3 later, but `legacy` is more honest about its status. *`src/training/pipelines/persona_v1/` alongside `pipelines/paired_dpo/`* — symmetric but treats them as equal-weight choices, which they aren't.
**Why:** Reader should not need to dig to find the canonical method. Naming carries the priority signal. Other DPO variants (v4, reversed-DPO) are deferred — migrate when needed, not eagerly.

### D9 (2026-05-27): Split `analyze_results.py` into purpose-named modules
**Chosen:** `src_dev/evals/personality/analyze_results.py` splits into:
- `src/evals/personality/sweep_results.py` — HF data fetching + per-metric result parsing for trait + judge sweeps
- `src/evals/capabilities/mmlu_results.py` — MMLU breakdown parsing
- `src/evals/personality/ci.py` — Wilson + BCa CI helpers, `IntervalMethod`
- `src/visualisations/palette.py` — `BIG_FIVE_COLORS`
**Considered & rejected:** *Keep `analyze_results.py` as one file* — accurate description, but reader has to grep to find the CI math or the color palette. *Rename to `sweep_results.py` but don't split* — improves discoverability of the main function, but BIG_FIVE_COLORS still hides in an evals file.
**Why:** Each piece serves a different consumer: figure scripts need parsers, CI utilities are general-purpose statistics, colors belong with plotting. Splitting clarifies what each module is *for*.

### D10 (2026-05-27): Documentation standard — README per package, docstring per file
**Chosen:** Each top-level `src/<package>/` and `scripts/<area>/` gets a README.md describing its contents and how the pieces fit together. Every `.py` file gets a top-of-module docstring stating purpose, what it imports from where, what it produces. No nested READMEs at sub-package level unless complexity demands it.
**Considered & rejected:** *Top-level READMEs only* — easier to maintain but loses file-level orientation. *READMEs at every depth* — too much maintenance for the value.
**Why:** Reader can drop into any `src/<package>/` and orient themselves via the README; individual files self-describe via docstrings. Balanced cost.

### D11 (2026-05-27): Clean up `hf_hub.py` API in Slice 1 (don't defer)
**Chosen:** Slice 1 promotes `src_dev/utils/hf_hub.py` → `src/utils/hf_hub.py` *and* cleans up its API. The cleanup target: a single primary entrypoint that handles fetch-from-HF-if-missing-else-load-local semantics consistently. Specific design happens during Slice 1 implementation; this decision just commits to doing it now.
**Considered & rejected:** *Defer to a tech-debt slice* — every subsequent slice copies the messy version. *Opportunistic per-slice* — never converges.
**Why:** Slice 1 already touches hf_hub. The 4 figure scripts of Slice 1 will use the new API immediately, validating the design before more slices depend on it.

### D12 (2026-05-27): Sequential reproduction — numbered scripts + per-area README
**Chosen:** Multi-step pipelines under `scripts/<area>/` use numbered prefixes (`01_*.py`, `02_*.py`, ...). Each area also has a `README.md` describing the sequence in prose (what each step does, what it depends on, expected runtime/compute).
**Considered & rejected:** *README only* — sequencing isn't visible in `ls`. *Numbering only* — order is visible but no context for *why* each step exists.
**Why:** Lowest-friction reproduction for a new user. `ls scripts/training/ocean_paired_dpo/` shows the sequence; README explains it. Both signals point in the same direction.

### D13 (2026-05-27): Branch `irakli/codebase_cleanup` renamed to `refactor/main`
**Chosen:** The integration branch for the cleanup is renamed to `refactor/main`. Each slice gets a feature branch off `refactor/main`, named `refactor/slice-<n>-<short-name>`.
**Considered & rejected:** *Keep `irakli/codebase_cleanup`* — owner-prefixed; doesn't communicate that this is a shared integration branch.
**Why:** Refactor is multi-slice and multi-person; the branch is shared infrastructure, not personal work.

---

## Migration status

Track what has been copied from `*_dev/` to `src/`+`scripts/`. One row per file. Update as files land. Format: `<dest path> ← <source path> (slice N, YYYY-MM-DD)`.

*(Empty — Slice 1 not yet started.)*

---

## Open tech-debt items (acknowledged, not in any slice yet)

- **HF fetch-or-regenerate semantics across the codebase** (point 10). Slice 1 cleans up `hf_hub.py` (D11), but other modules in `src_dev/` may have their own caching/regeneration patterns. A follow-up slice consolidates these once we've seen the pattern reused 2–3 times.
- **`*_dev/` pruning.** Once enough slices land that we're confident the migrated copies fully cover what's needed, a deliberate pruning pass removes superseded `*_dev/` code. Not yet scheduled.
- **HF monorepo cleanup.** Out of scope here; tracked separately.
