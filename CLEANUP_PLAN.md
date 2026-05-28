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

1. **`src/` is trusted infra; `scripts/` is flexible glue.** `src/` holds validated, reusable code that we trust — ideally with unit tests (see D15). `scripts/` use that infra to run things end-to-end or step-by-step; more duplication is tolerated there because scripts are experiment-facing, not library code. (See D14.)
2. **Vertical slices, one capability at a time.** Each slice is independently runnable before the next begins. Slice *ordering* follows research priority, not paper-section order (see D16).
3. **Migrate, don't fork. No deletions.** Copy from `*_dev/` into `src/`+`scripts/`, then refactor in place. Originals stay; team members keep their imports.
4. **Migrated code imports only from `src/`, never `src_dev/`.** A migrated file containing any `from src_dev...` import is a migration bug. If a migrated `src/` module needs something still living in `src_dev/`, that dependency must be migrated (or its needed piece copied) in the same slice. (See D17.)
5. **Refactor for clarity, not for novelty.** Reorganise placement and naming. Split files where they're grab-bags. Avoid behavioral changes during migration — those land as separate follow-up commits if needed.
6. **Definition of done is layered:** `src/` modules carry tests where feasible (D15); `scripts/` need import + dry-run smoke tests; figure scripts must regenerate figures matching the current PDFs. Full end-to-end re-runs are flagged as TODO (D6).
7. **`src/utils/` stays top-level**; orphan stubs in `src/` get annotated as "superseded, kept for reference" once a replacement lands. No deletions.
8. **Cost discipline.** Plumbing work (`--dry-run`, import checks, CLI smoke tests) on a cheaper model per `CLAUDE.md`.
9. **Documentation per migrated piece.** Every `src/<package>/` and `scripts/<area>/` gets a README; every `.py` file gets a top-of-file docstring describing purpose, imports, and outputs.
10. **Sequential reproduction is first-class.** Each `scripts/<area>/` exposes its multi-step pipelines as numbered files (`01_*.py`, `02_*.py`, ...) and its README lists the ordered sequence.

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

## Slice ordering (revised — priority over paper-section order, see D16)

The migration order follows research priority, not paper TOC:

1. **Slice 1a — Paired-DPO data generation** (FIRST, this branch).
2. **Slice 1b — Paired-DPO training + export.**
3. **Slice 2 — Evaluations** (trait-via-logprobs, MMLU recovered-score, judge prompts).
4. **Slice 3 — Figures** (OCEAN scaling main + Appendix F; consumes eval outputs from Slice 2). *(Was Slice 1 in the original plan.)*
5. **Deferred — Combinations / LoRA arithmetic, behavioral evals.**
6. **Deferred — Judge calibration** (standalone; D16).
7. **Deferred — Unsupervised** (partially still under development; D16).

> **Key finding (D18):** the canonical paired-DPO pipeline (`scripts_dev/oct_pipeline/run_oct_pipeline.py`) is built on the **external `character.*` (OpenCharacterTraining) library**, NOT on `src_dev/` inference/editing/training components. The only `src_dev/` dependency in the data-gen path is `src_dev/utils/hf_hub.py` (which has zero `src_dev` deps). This invalidates the original `src/training/paired_dpo/{data_prep,loss}.py` sketch — there's no custom DPO loss to migrate (it's OCT/TRL's). See revised structure below.

---

## Slice 1a (FIRST): Paired-DPO data generation

**Capability unlocked:** produce the paired (chosen, rejected) DPO dataset for an OCEAN trait × direction — runnable end-to-end or step-by-step. This is the input to Slice 1b (training).

**Stages in the data-gen path** (from `run_oct_pipeline.py`): Stage 0 constitution install, Stage 1a teacher pass, Stage 1b student pass, then `prep_paired_dpo.py` joins them into `(chosen, rejected)` pairs. Stage 2 (DPO training) is Slice 1b.

**Design decisions baked in:**
- OCT calls are wrapped behind a thin adapter in `src/` so `scripts/` never `import character.*` directly (D18a).
- The paired dataset keeps its OCT-native schema (`{prompt, response, baseline-col}` → `{prompt, chosen, rejected}`); documented, not converted to canonical format (D18b).

**`src/` (trusted infra, tested):**
- `src_dev/utils/hf_hub.py` → `src/utils/hf_hub.py` (cleaned per D11; unit-test the path/repo-resolution logic, smoke-test the network calls).
- `src/training/paired_dpo/pairing.py` — extract `_build_paired_rows()` (the chosen/rejected inner-join from `prep_paired_dpo.py`). Pure & deterministic → **unit-tested** (D15).
- `src/training/oct_adapter.py` — thin wrapper over `character.distillation.{teacher,student}` and constitution install. Smoke-test only (wraps external GPU/API lib).

**`scripts/training/ocean_paired_dpo/` (flexible glue, numbered):**
- `01_install_constitution.py` — wraps `install_custom_constitution()` via `src/training/oct_adapter.py`.
- `02_generate_teacher_student.py` — teacher + student distillation passes.
- `03_build_paired_dataset.py` — calls `src/training/paired_dpo/pairing.py`, writes paired JSONL + uploads via `src/utils/hf_hub.py`.
- `README.md` — the ordered sequence, what each step consumes/produces, the OCT-native schema, expected compute.
- Constitution JSON templates from `scripts_dev/oct_pipeline/ocean/` → `scripts/training/ocean_paired_dpo/constitutions/` (data, not code).

**Documentation deliverables (D10):** doc-anchor READMEs for the `src/` packages this slice touches (`src/training/README.md` incl. the `vanton4_paired_dpo ↔ paired_dpo` mapping table per D4, `src/utils/README.md`) + the scripts README above.

**Verification for Slice 1a:**
1. `uv run pytest tests/src/training/test_pairing.py` passes (the chosen/rejected join, edge cases: missing prompt, unequal amp/sup counts).
2. `uv run python scripts/training/ocean_paired_dpo/03_build_paired_dataset.py --help` and `--dry-run` work; imports resolve.
3. Grep check: no `from src_dev` in any migrated `src/` file (principle 4).
4. Run `03_build_paired_dataset.py` on a tiny existing teacher/student JSONL pair → emits a paired JSONL with the expected `{prompt, chosen, rejected}` schema, matching what the dev-layer `prep_paired_dpo.py` produces on the same input.
5. Existing `scripts_dev/oct_pipeline/` code still works unchanged.

**TODO (tracked, deferred):** full end-to-end teacher/student generation run (GPU + API cost). Validate when replicating on a new model family.

---

## Slices 1b–N (sketched, not detailed)

These get their own plan files when started. Priority order per D16:

1. **Slice 1b — Paired-DPO training + export.** Stage 2 (DPO) + Stage 4 (SFT) + Stage 5 (merge) from `run_oct_pipeline.py`. Wrap OCT/TRL trainer via `src/training/oct_adapter.py` (extend it). `scripts/training/ocean_paired_dpo/04_train_lora.py`, `05_merge_or_export.py`. Other training methods (v4, reversed-DPO) deferred (D8).
2. **Slice 2 — Evaluations.** The adjusted evals: trait-via-logprobs, MMLU recovered-score, judge prompts. These are **implemented but messy** in `src_dev/evals/` — migration consolidates into clean `src/evals/` modules with tests, verifying outputs against existing HF artifacts (behavioral-risk: needs careful diff). Produces `src/evals/personality/`, `src/evals/capabilities/`, plus split of `analyze_results.py` per D9. `scripts/evals/{personality_sweep,capabilities_sweep}/`.
3. **Slice 3 — Figures.** OCEAN scaling main + Appendix F. The 4 paired-DPO figure scripts → `scripts/figures/`. Consumes Slice 2's eval outputs. `analyze_results.py` split (D9) already landed in Slice 2, so figures just import from `src/`. Keep 3 appendix scripts as 3 files (D3). Defer Fig 1 banner (D5).
4. **Deferred — Combinations / LoRA arithmetic.** Exercises `src/utils/peft_manipulations.py` (already stable).
5. **Deferred — Behavioral evals.** WildJailbreak, sycophancy/CoCoNot, frustration.
6. **Deferred — Judge calibration** (standalone; D16). `scripts_dev/persona_metrics/llm_judge/plot_paper_judge_calibration.py` → `scripts/judge_calibration/`.
7. **Deferred — Unsupervised** (partially under development; D16). Factor analysis 5-stage pipeline; own plan when it stabilises.

---

## Branching workflow

- `refactor/main` (renamed from `irakli/codebase_cleanup`) is the integration branch for the cleanup effort.
- Each slice gets its own branch off `refactor/main`, e.g. `refactor/slice-1-ocean-figures`, `refactor/slice-2-paired-dpo-training`.
- Slice branches merge into `refactor/main` after the slice verification passes.
- `refactor/main` eventually merges into `main` once enough slices have landed that it makes sense.

---

## Risks & open questions

- **`paper_main_fig1_banner.py` depends on 3 sibling spider/delta scripts** that I haven't traced. Fig 1 deferred (D5); the underlying HF data becomes reachable once the evals slice (Slice 2) lands its `src/` library code, so the figures slice (Slice 3) can pick up regeneration cheaply.
- **`src_dev/evals/personality/analyze_results.py` is heavily depended on across `src_dev/`.** Moving it would break things. Plan: *copy and split* (D9) into the new `src/` layout during the evals slice (Slice 2). The dev version stays.
- **`src_dev/utils/hf_hub.py` same story** — copy, clean, leave dev version alone. Lands in Slice 1a (it's the data-gen path's only `src_dev` dependency).
- **HF-fetch-or-regenerate messiness (point 10).** Some utilities cache fetched HF data, some regenerate, mixed conventions. Slice 1a tackles this in `src/utils/hf_hub.py` by designing a clean fetch-or-regenerate API (see D11).
- **Behavioral risk in the evals slice (Slice 2).** The adjusted evals (trait-via-logprobs, MMLU recovered-score, judge prompts) are implemented-but-messy; consolidating them risks subtle output changes. Verify migrated eval outputs against existing HF artifacts before considering Slice 2 done.
- **Pattern recurrence (activation-capping appendix).** The `{trait,mmlu,judge}` triplet shape recurs there. Defer "do we collapse into one parameterised script?" until that deferred slice (D3).

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
**Why:** The first slice touches hf_hub regardless (post-D16, that's Slice 1a — it's the paired-DPO data-gen path's only `src_dev` dependency). Cleaning it first validates the API before more slices depend on it.

### D12 (2026-05-27): Sequential reproduction — numbered scripts + per-area README
**Chosen:** Multi-step pipelines under `scripts/<area>/` use numbered prefixes (`01_*.py`, `02_*.py`, ...). Each area also has a `README.md` describing the sequence in prose (what each step does, what it depends on, expected runtime/compute).
**Considered & rejected:** *README only* — sequencing isn't visible in `ls`. *Numbering only* — order is visible but no context for *why* each step exists.
**Why:** Lowest-friction reproduction for a new user. `ls scripts/training/ocean_paired_dpo/` shows the sequence; README explains it. Both signals point in the same direction.

### D13 (2026-05-27): Branch `irakli/codebase_cleanup` renamed to `refactor/main`
**Chosen:** The integration branch for the cleanup is renamed to `refactor/main`. Each slice gets a feature branch off `refactor/main`, named `refactor/slice-<n>-<short-name>`.
**Considered & rejected:** *Keep `irakli/codebase_cleanup`* — owner-prefixed; doesn't communicate that this is a shared integration branch.
**Why:** Refactor is multi-slice and multi-person; the branch is shared infrastructure, not personal work.

### D14 (2026-05-28): `src/` = trusted/validated infra; `scripts/` = flexible glue
**Chosen:** `src/` holds reusable code we trust, ideally unit-tested. `scripts/` use that infra to run pipelines end-to-end or step-by-step; more duplication is acceptable there.
**Considered & rejected:** *Treat src and scripts as the same bar* — over-tests throwaway experiment glue; under-tests load-bearing library code.
**Why:** Matches how the team uses the two layers — `src/` is depended on by many scripts and future work, so it earns a higher bar; scripts are experiment-facing and change often.

### D15 (2026-05-28): Test bar — unit-test pure/deterministic pieces, smoke-test the rest
**Chosen:** Deterministic logic migrated to `src/` (CI math, data-prep transforms like the chosen/rejected join, path resolution, parsers) gets real unit tests. GPU-training / API-calling code gets import + `--dry-run` smoke tests only.
**Considered & rejected:** *Strict (nothing lands without a test)* — forces expensive fixtures/mocks for fine-tuning code with little payoff. *Best-effort (don't block on coverage)* — risks the testable core also going untested.
**Why:** The high-value, cheap-to-test pieces get covered; we don't burn effort mocking a fine-tune. Untested `src/` modules are tracked in the Migration status section.

### D16 (2026-05-28): Slice ordering follows research priority, not paper-section order
**Chosen:** Order is paired-DPO pipeline (data-gen → training) → evaluations → figures, with judge-calibration and unsupervised deferred. The original plan's figures-first Slice 1 becomes Slice 3.
**Considered & rejected:** *Figures-first* (original D1 framing) — fastest to "reproduce a figure", but figures are a thin layer over eval outputs; producing artifacts is the higher-value capability. *Strict paper-TOC order* — doesn't match what's most useful to unlock first.
**Why:** User priority: be able to *produce* (train + eval) before *re-plot*. Judge calibration is standalone (can wait); unsupervised is partially still under development (not stable enough to freeze). Refines D1's "vertical slices" — slices are still vertical, but ordered by priority.

### D17 (2026-05-28): Migrated code imports only from `src/`, never `src_dev/`
**Chosen:** A migrated `src/` or `scripts/` file must not contain `from src_dev...`. If a migrated module needs something still in `src_dev/`, that dependency is migrated (or its needed piece copied) in the same slice. Enforced by a grep check in each slice's verification.
**Considered & rejected:** *Allow temporary `src_dev` imports from migrated code* — would let the clean layer silently depend on the dev layer, defeating the point of a trusted `src/`.
**Why:** `src/` must be self-contained to be trustworthy; a `src/` module reaching back into `src_dev/` inherits all the dev layer's instability.

### D18 (2026-05-28): Paired-DPO pipeline wraps external OCT lib; thin adapter in `src/`; OCT-native schema preserved
**Context:** Exploration found `run_oct_pipeline.py` delegates teacher/student generation and DPO/SFT training to the external `character.*` (OpenCharacterTraining) library. The only `src_dev/` dependency in the data-gen path is `src_dev/utils/hf_hub.py`. The original `src/training/paired_dpo/{data_prep,loss}.py` sketch was wrong (no custom loss to migrate).
**D18a — OCT via thin adapter:** `src/training/oct_adapter.py` wraps `character.distillation.*` and constitution install behind a clean interface; `scripts/` never `import character.*` directly.
- *Considered & rejected:* *Keep OCT calls inline in scripts* — leaks the external dep across every script, no single seam to swap/mock. *Vendor/fork OCT* — premature; OCT is a usable pinned dependency.
- *Why:* One seam isolates the external dependency, makes scripts readable, and gives a mock point for smoke tests.
**D18b — preserve OCT-native dataset schema:** the paired dataset stays `{prompt, response, baseline-col}` → `{prompt, chosen, rejected}`, documented in the script README; not converted to the repo's canonical dataset format.
- *Considered & rejected:* *Convert to canonical* (per CLAUDE.md preference) — risks diverging from what OCT's trainer expects and from existing HF artifacts. *Hybrid (canonical for our outputs)* — extra machinery for no current consumer.
- *Why:* OCT dictates the shape at this boundary; forcing canonical here buys nothing and risks breaking trainer compatibility. Revisit if a downstream consumer we control needs canonical.

---

## Migration status

Track what has been copied from `*_dev/` to `src/`+`scripts/`. One row per file. Update as files land. Format: `<dest path> ← <source path> (slice N, YYYY-MM-DD)`.

*(Empty — Slice 1a not yet started.)*

---

## Open tech-debt items (acknowledged, not in any slice yet)

- **HF fetch-or-regenerate semantics across the codebase** (point 10). Slice 1 cleans up `hf_hub.py` (D11), but other modules in `src_dev/` may have their own caching/regeneration patterns. A follow-up slice consolidates these once we've seen the pattern reused 2–3 times.
- **`*_dev/` pruning.** Once enough slices land that we're confident the migrated copies fully cover what's needed, a deliberate pruning pass removes superseded `*_dev/` code. Not yet scheduled.
- **HF monorepo cleanup.** Out of scope here; tracked separately.
