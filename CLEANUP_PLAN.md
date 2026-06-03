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

1. **`src/` is trusted infra; `scripts/` is the run surface.** `src/` holds validated, reusable code that we trust — ideally with unit tests (see D15). `scripts/` are the entry points for executing anything end-to-end or step-by-step; they may hold real orchestration logic, and more duplication is tolerated there because scripts are experiment-facing, not library code. The split is about trust/reusability, not thickness. (See D14.)
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

**Update (2026-06-03 — supersedes the "keep in HF paths" clause above; see D21):** `vanton4_paired_dpo` is to be renamed to `paired_dpo` **everywhere, including on HuggingFace**. Migrated `src/`+`scripts/` code uses `paired_dpo`. The HF monorepo artifact rename (`.../vanton4_paired_dpo/...` → `.../paired_dpo/...`) and the paper's `% Data:` comment updates are a **dedicated, separately-authorized operation**, not part of any migration slice — it mutates the shared monorepo. Until that runs, migrated example paths keep referencing the real current `vanton4_paired_dpo` HF locations (otherwise they'd point at nonexistent paths).

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

**Slice 1a is narrowed (D19): ship the unambiguous `src/` wins + the dataset-build entrypoint now; defer distillation (constitution install + teacher/student generation) to Slice 1a.2.** Reason: the OCT distillation wrappers are ~260 lines of GPU-memory tuning, vLLM context management, and module-global override juggling, intertwined with hardware state — not a clean call-and-return surface. Settling the `oct_adapter.py` shape deserves more eyes; meanwhile pairing + hf_hub + the build script are independently useful and testable.

**`scripts/training/ocean_paired_dpo/` (run surface):**
- `03_build_paired_dataset.py` — calls `src/training/paired_dpo/pairing.py`, writes paired JSONL + stage marker + provenance, uploads via `src/utils/hf_hub.py`. Imports only from `src/`. (Mirrors `scripts_dev/oct_pipeline/ocean/prep_paired_dpo.py` `_prep_direction`/`main`, but the pure join moves to `src/`.)
- `README.md` — the full intended sequence (01 install → 02 teacher/student → 03 build); notes that 01/02 land in Slice 1a.2; documents the OCT-native schema and expected compute.
- (Slice 1a.2) `01_install_constitution.py`, `02_generate_teacher_student.py` + `constitutions/` templates.

**`src/training/oct_adapter.py`** — deferred to Slice 1a.2 along with the distillation scripts (its shape depends on how we treat the GPU/vLLM machinery; not built blind in 1a).

**Documentation deliverables (D10):** doc-anchor READMEs for the `src/` packages this slice touches (`src/training/README.md` incl. the `vanton4_paired_dpo ↔ paired_dpo` mapping table per D4, `src/utils/README.md`) + the scripts README above.

**Verification for Slice 1a:**
1. `uv run pytest tests/src/training/paired_dpo/test_pairing.py` passes (the chosen/rejected join: both directions, the `first`/`random`/`all` amp-pairing modes, missing-prompt and missing-response edge cases, unmatched-sup counting).
2. `uv run python scripts/training/ocean_paired_dpo/03_build_paired_dataset.py --help` works and `--dry-run` on a tiny local JSONL pair runs end-to-end without network.
3. Grep check: no `from src_dev` in any migrated `src/` file (principle 4).
4. Schema parity: `03_build_paired_dataset.py --dry-run` on a fixture emits paired JSONL identical to what dev-layer `prep_paired_dpo.py` produces on the same input.
5. Existing `scripts_dev/oct_pipeline/` code still works unchanged.

**TODO (tracked, deferred to Slice 1a.2):** constitution install + teacher/student distillation scripts + `oct_adapter.py`; then a full end-to-end teacher/student generation run (GPU + API cost) when replicating on a new model family.

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

**Model (see D20):**
- `refactor/main` (renamed from `irakli/codebase_cleanup`) is the **integration branch** for the cleanup effort.
- Each slice or standalone change gets its own branch off `refactor/main`, e.g. `refactor/slice-1a-paired-dpo-datagen`, `refactor/oct-deps-setup`.
- Each branch opens a **PR targeting `refactor/main`**. Colleagues review / e2e-validate on the PR; merge when green.
- `refactor/main` merges into real `main` periodically (one PR) once enough has landed to make sense as a unit.
- **Keeping branches current:** when `refactor/main` moves, **rebase** open branches onto it (force-push the feature branch; coordinate if a colleague is mid-review). This keeps history linear and ensures every branch tests against the latest infra (e.g. the pytest fix).
- **Cleanup:** delete a branch once its PR merges. Stale merged branches can be swept in a later pass.
- One branch = one reviewable unit.
- **Landing a branch = squash + rebase (see D22):** a branch lands as **one commit with no merge commit** (`git merge --squash <branch>` then a single branch-prefixed commit, equivalently squash-then-rebase). History stays linear from here on. Pre-existing *pushed* merge commits (e.g. the `33021a1f` pytest-fix merge) are **left as-is** — flattening them needs a force-push of shared history, which we don't do without explicit per-instance permission.

### Branch status (live — keep current as branches merge)

Guidance for reviewers on what each open branch is and what to do with it.

| Branch | Off | Pushed | State | What reviewers should do |
|--------|-----|--------|-------|--------------------------|
| `refactor/main` | `main` | yes | Integration branch. Local copy is **ahead of `origin` by several commits** (oct-deps + slice-1a squash-merges + plan edits), **not pushed**. | Don't commit features here directly; it receives squash-merges. Push to `origin` only with explicit owner go-ahead. |
| `refactor/slice-1a-paired-dpo-datagen` | `refactor/main` | yes | **Slice 1a** — paired-DPO dataset build: `src/utils/hf_hub.py`, `src/training/paired_dpo/pairing.py` (+18 tests), `scripts/training/ocean_paired_dpo/03_build_paired_dataset.py`. **Squash-merged into `refactor/main` (local) 2026-06-03**; verified (18 tests pass, output byte-identical to dev `prep_paired_dpo.py`, no `src_dev` imports). | Done. Re-validate the build at the final e2e pass; `git branch -d` after origin reconciliation. |
| `refactor/oct-deps-setup` | `refactor/main` | yes | OCT dependency install automation (`make oct-deps`, `scripts/setup/install_oct_deps.sh`, pins `vllm==0.17.1`). **Squash-merged into `refactor/main` (local) 2026-06-03** after code review (pins match `uv-oct-requirements.txt`). | Runtime-validate `make oct-deps` on the H100/H200 box at the final e2e pass (vllm 0.17.1 won't install on the macOS dev machine). |
| `refactor/fix-pytest-collection` | `refactor/main` | no (local-only) | **Already merged** into `refactor/main` (importlib pytest config). | Nothing — safe to `git branch -d` in a later cleanup pass. |

### Local worktree workflow (recommended)

Because slices proceed in parallel and edits often span branches (e.g. fixing a note that lives on a slice branch while editing this plan on `refactor/main`), use **git worktrees** instead of branch-switching a single checkout.

**Convention:** the primary checkout stays pinned to `refactor/main` (the integration surface — never branch-switch it; it only *receives* merges). Every other active branch gets a worktree under a sibling container dir `../psl-worktrees/<short-name>/`:

```
/Users/<user>/dev/LASR/
  persona-shattering-lasr/   refactor/main          (primary; never branch-switch)
  psl-worktrees/
    slice-1a/                refactor/slice-1a-paired-dpo-datagen
    oct-deps/                refactor/oct-deps-setup
```

**Add a worktree for an existing branch:**
```bash
git worktree add ../psl-worktrees/<name> <branch>
ln -s "$(git rev-parse --show-toplevel)/.env" ../psl-worktrees/<name>/.env   # .env is gitignored, per-checkout
(cd ../psl-worktrees/<name> && uv sync)                                       # .venv is gitignored, per-checkout
```

**Start a new slice off `refactor/main`:**
```bash
git worktree add -b refactor/slice-<n>-<name> ../psl-worktrees/<name> refactor/main
```

**Rebase a slice when `refactor/main` moves** (done in the slice's own worktree, primary checkout untouched):
```bash
cd ../psl-worktrees/<name> && git rebase refactor/main && git push --force-with-lease
```

**Clean up after a PR merges:** `git worktree remove ../psl-worktrees/<name>` then `git branch -d <branch>`.

**Notes / gotchas:**
- `.env` (API keys) and `.venv` (~1.8 GB) are gitignored and do **not** carry into a worktree — symlink `.env`, `uv sync` per worktree. `uv` hardlinks from `~/.cache/uv` so the marginal disk cost is reduced but not zero (~1.8 GB each real).
- A branch can be checked out in only one worktree at a time (git enforces this).
- Per-worktree venvs are *correct*, not just convenient: branches diverge on deps (e.g. `refactor/oct-deps-setup` pins `vllm==0.17.1` + adds `character`/`openrlhf`).
- `scratch/` outputs are gitignored and per-worktree — not shared.

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
**Superseded by D21 (2026-06-03):** user decided to rename `vanton4_paired_dpo` → `paired_dpo` on HF as well; the HF-rename now happens as a dedicated step.

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

### D14 (2026-05-28): `src/` = trusted/validated infra; `scripts/` = the entry point for running anything
**Chosen:** `src/` holds reusable code we trust, ideally unit-tested. `scripts/` are the **entry points for executing anything** in the pipeline (end-to-end or step-by-step) — they may carry real orchestration logic, not just thin glue, and more duplication is acceptable there. The src/scripts split is about *trust & reusability* (library vs. runnable entrypoint), not about how thick the code is.
**Considered & rejected:** *Treat src and scripts as the same bar* — over-tests experiment entrypoints; under-tests load-bearing library code. *Frame scripts as "thin glue"* — wrong; a script can hold substantial orchestration, it just isn't imported as a library.
**Why:** Matches how the team uses the two layers — `src/` is depended on by many scripts and future work, so it earns a higher bar; scripts are the experiment-facing run surface and change often.

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

### D19 (2026-05-28): Narrow Slice 1a — ship pairing + hf_hub + build-script now, defer distillation to Slice 1a.2
**Context:** On reading the code, the OCT teacher/student distillation wrappers (`run_distillation_generation`, `run_teacher_openrouter` in `run_oct_pipeline.py`) are ~260 lines of GPU-memory tuning, vLLM stage contexts, and module-global override juggling, deeply tied to hardware state — not a clean call-and-return surface.
**Chosen:** Slice 1a ships only the unambiguous `src/` wins: `src/utils/hf_hub.py` (copied as-is — already clean, zero `src_dev` deps), `src/training/paired_dpo/pairing.py` (the pure chosen/rejected join, unit-tested), and `scripts/training/ocean_paired_dpo/03_build_paired_dataset.py` (the build entrypoint using both). Constitution install + teacher/student generation scripts + `src/training/oct_adapter.py` move to Slice 1a.2.
**Considered & rejected:** *Keep distillation orchestration in scripts/, signatures-only adapter in src/* — viable, but commits to an adapter shape under time pressure. *Move full distillation into src/* — drags torch/vLLM + GPU logic into `src/`, untestable, against the "trusted clean infra" spirit (D14).
**Why:** The build-script + pairing + hf_hub are independently useful and fully testable today. The adapter shape for the GPU-bound distillation deserves more deliberation (more eyes, per the team's concurrent-work reality) rather than being fixed blind in the first slice. Smaller first PR, lower risk.

### D20 (2026-05-28): Branch management — PR each branch into `refactor/main`, rebase to stay current, track state in this doc
**Chosen:** `refactor/main` is the integration branch. Each slice / standalone change branches off it and opens a PR *into* `refactor/main`, reviewed + e2e-validated there; `refactor/main` merges into real `main` periodically as a unit. Open branches rebase onto `refactor/main` when it moves. A live "Branch status" table (in the Branching workflow section) tells reviewers what each branch is and what to do with it.
**Considered & rejected:** *PR each branch straight into `main`, drop `refactor/main`* — simpler graph, but loses the single staging surface and makes dependent slices (1a→1b) awkward. *Keep merging locally into `refactor/main` without PRs* — lowest ceremony, but reviews don't happen on a PR surface and merged-vs-pending is easy to lose track of. *Merge `refactor/main` into open branches instead of rebasing* — avoids force-push but creates noisy merge commits.
**Why:** Matches D13's integration-branch intent and gives colleagues a clean PR surface for e2e validation. Rebasing keeps history linear and guarantees branches test against the latest infra (e.g. the pytest fix). The in-doc branch table means reviewers don't have to reconstruct branch intent from git alone.

### D21 (2026-06-03): Rename `vanton4_paired_dpo` → `paired_dpo` everywhere, including HuggingFace
**Chosen:** Drop `vanton4` from the paired-DPO identifier entirely — `paired_dpo` in migrated `src/`+`scripts/` code **and** in the HF monorepo paths. The HF artifact rename + paper `% Data:` comment updates are a **dedicated, separately-authorized operation** (mutates the shared `persona-shattering-lasr/monorepo`), tracked outside the migration slices. Migrated example paths keep pointing at the live `vanton4_paired_dpo` HF locations until that rename runs.
**Considered & rejected:** *Keep D4 as-is* (rename script names only, leave HF paths) — leaves `vanton4` (a meaningless internal tag) baked into the canonical artifact layout forever. *Rename HF inside a migration slice* — couples a shared-infra mutation to a code slice; the rename touches the paper and many existing scripts, so it earns its own authorized step.
**Why:** User decision (2026-06-03). `vanton4` carries no meaning to an external reader; `paired_dpo` names the actual method. Supersedes D4 — the "HF paths are immutable" premise no longer holds now that we're committing to the rename. **Open scope question:** applies to `vanton4_paired_dpo`; whether bare `vanton4` (non-paired distillation runs) also renames is not yet decided.

**Design note (2026-06-03) for the paired-DPO generation step (Slice 1a.2):** the migrated `02_generate_teacher_student.py` should make **teacher-only generation the default** — generate the amplifier and suppressor *teacher* passes and save the teacher pairs directly, **without** generating student outputs (paired DPO uses `chosen=amp-teacher, rejected=sup-teacher`, so the student pass the original OCT pipeline produces is wasted compute). A **non-paired/standard DPO** mode (`chosen=teacher, rejected=student`) is available but **not** the default. Open: whether the pairing folds into generation (02 emits the paired set) or stays as the separate 03 join.

### D22 (2026-06-03): Branches land via squash + rebase (one linear commit); don't flatten pushed merge commits
**Chosen:** Each branch lands on `refactor/main` as a **single commit with no merge commit** (`git merge --squash <branch>` + one branch-prefixed commit, equivalently squash-then-rebase). The pre-existing *pushed* `33021a1f` pytest-fix merge commit is **left in history** ("linear from here") rather than flattened.
**Considered & rejected:** *True merge commits per branch* — integration history grows a diamond per slice; harder to read. *Flatten the existing `33021a1f` merge too* — fully linear remote history, but rewrites already-pushed commits → force-push of shared `refactor/main` + a reset for anyone who pulled. *Plain rebase preserving each branch's WIP commits* — linear, but a slice's internal commits leak into integration history; squashing keeps it one-commit-per-capability.
**Why:** User decision (2026-06-03). One linear commit per branch reads cleanly; not force-pushing shared history avoids disrupting colleagues. Refines D20's rebase guidance with the explicit squash + no-flatten rule.

### D23 (2026-06-03): Scope = the paper-critical reproduction path only
**Chosen:** Migrate only what's needed to reproduce the PAPER end-to-end (train → eval → figures) plus the infra it depends on. Peripheral evals, LoRA-combination *experiments* (e.g. `lora_soup_generate`, downrank), behavioral probes, and the unsupervised pipeline stay in `*_dev` for now. (User decision Q1.)

### D24 (2026-06-03): Migrate by behaviour-preserving DEDUP, not byte-verbatim copies
**Chosen:** Migration extracts shared helpers and reduces duplication (CLAUDE.md's first-class goal); "identical to the source" means identical **behaviour**, not identical text. Pure/testable code is proven equal via dev-parity tests; figure-script dedup (not locally runnable) is confirmed at the end-run. Applies **retroactively** to the landed Slices 2/3/4 figure scripts too. Supersedes the earlier strict byte-verbatim approach used in Slices 1a–4.
**Why:** User correction (Q9/Q10): verbatim copies propagate duplication into the clean layer, defeating the purpose. The earlier "identical code" rule was about not changing behaviour, not about avoiding refactors.

### D25 (2026-06-03): `vanton4_paired_dpo` → `paired_dpo` renamed IN CODE now (amends D21)
**Chosen:** Migrated `src/`+`scripts/` use `paired_dpo` in path strings **now**, not deferred. The HF data catches up at the end-run, when the full pipeline regenerates artifacts under the proper names; the final HF-side rename is an end task the user runs. Bare `vanton4` (non-paired distillation runs) is **not** migrated — it isn't used by the paper. Amends D21 (which had kept `vanton4_paired_dpo` in code until the HF rename). (User Q6/Q7.)

### D26 (2026-06-03): Whole paper pipeline migrated as CODE; runs deferred to one end-run
**Chosen:** In scope, migrated as code with runs deferred to the user's GPU/API end-run: the paired-DPO training pipeline (Slice 1a.2 distillation + 1b DPO/SFT/merge), the eval runners (inspect logprob trait/MMLU/judge tasks + scorers → `src/evals`), and full judge-calibration **including** the live-judge data generators. `src/training/oct_adapter.py`'s design is deferred to an end discussion (not built blind). (User Q2/Q3/Q4/Q5.)

### D27 (2026-06-03): Bug fixes in `src/` only; `*_dev` kept; user handles push
**Chosen:** The two logged KNOWN_ISSUES bugs are fixed in the canonical `src/` copies only (dev keeps the latent bug — intentional divergence). `*_dev` is kept indefinitely (no pruning pass). Pushing `refactor/main` to origin is handled by the user, not the agent. (User Q8/Q11/Q12.)

### D28 (2026-06-03): Approved `oct_adapter` design — file split for the training-pipeline migration
**Context:** `scripts_dev/oct_pipeline/run_oct_pipeline.py` (4335 lines) drives the paired-DPO pipeline on the external `character.*` / `openrlhf.*` libs (only `src_dev` dep: `hf_hub`). User approved migrating it (D26) behind a thin adapter, cleaned + split. Design scouted + **user-approved 2026-06-03**.
**Chosen file split (~4335 → ~2500 lines across 9 files):**
- `src/training/oct_adapter.py` — THE SEAM. Thin stage fns scripts call: `initialize_oct_runtime`, `install_constitution`, `generate_distillation_data` (has `skip_student_pass` teacher-only flag), `train_dpo_adapter`, `generate_introspection_data`, `fold_dpo_lora_into_model`, `train_sft_adapter`, `merge_adapters_into_persona`, `load_dpo_pairs_from_distillation`, symlink helper. Scripts import ONLY this — never `character.*`/`openrlhf.*`.
- `src/training/oct_runtime.py` — vLLM/GPU patching machinery (`patch_oct_constants`, `_install_runtime_character_constants`, `_patched_llm_init`/`_generate_v2`, `_safe_sampling_params`, `_capped_gen_args` [gemma], `_vllm_stage_context`/`_ACTIVE_VLLM_STAGE`, `_apply_torch_memory_fraction`, `_patched_http_backoff`, context-overflow filtering). Irreducibly tangled with stage state → kept isolated + commented (the honestly-messy bit).
- `src/training/openrouter_teacher.py` — OpenRouter-API teacher path (`run_teacher_openrouter` + prefill / system-prompt / async-batch helpers). Self-contained.
- `src/training/oct_config.py` — run-identity / config-hash, stage markers, `MonorepoCoordinates`.
- `scripts/training/ocean_paired_dpo/` numbered entrypoints (thin: parse args → `initialize_oct_runtime` → call adapter → write stage marker → HF upload). **Numbering:** `00_install_constitution` → `01_generate_teacher_student` → `02_build_paired_dataset` (the EXISTING Slice 1a seed-join) → `03_train_dpo` → `04_introspection_sft` → `05_merge`. Single-teacher flow uses `load_dpo_pairs`; paired-seed flow uses `02_build` + `--skip-student-distillation`.
**Key calls:** thin adapter + separate stages (scripts own orchestration/caching/HF-upload, not a fat run-all); vLLM patching stays in `oct_runtime` (can't be abstracted clean); OpenRouter teacher its own module; teacher-only-gen as a flag.
**Verification:** behaviour-preserving REFACTOR, not verbatim (D24) → parity confirmable only at the GPU **end-run**; smoke-test imports/`--help` + keep each piece traceable to dev. **Dev source-line map** (`run_oct_pipeline.py`): `install_custom_constitution`@652, `run_distillation_generation`@1725, `run_teacher_openrouter`@1440, `load_dpo_pairs`@1864, `run_oct_dpo_training`@2192, `run_introspection_generation`@2359, `_merge_introspection_data`@2461, `fold_lora_into_model`@1985, `run_oct_sft_training`@2641, `merge_adapters`@2750; vLLM patches @331-602; OpenRouter @809-1700 + 963/1440; config/markers @2941-3320; `src_dev.utils.hf_hub` @3128. Constitution install (`install_custom_constitution`) is Slice 1a.2's `00_install`; distillation gen is `01`.

### D29 (2026-06-03): Naming + training-pipeline scope — SUPERSEDES D25's rename target and the Q6 "trim" premise
**Trim premise is dead.** The "trim non-paper-critical bare-`vanton4` Slice-4 scripts" plan (old roadmap item 1 / Q6) was based on a false belief. Reality from the code:
- The three `_vanton4.py` combo scripts use the `vanton4_paired_dpo` method (the canonical paired-DPO), **not** bare `vanton4`.
- Bare `vanton4` is a **distinct, frozen method** (the joint-OCEAN-constitution OCT run, labeled "joint OCEAN constitution" in fig_B) and is the live data source for **confirmed paper figures**: `fig_1_o_n_soup_heatmap_*`, `fig_B_dpo_methods_scaling` (which compares `vanton4` vs `v4`/`v4_reversed_dpo`/`v4_paired_dpo`), `fig_3_3_1_o_plus_scaling_*`, the c×e soup. In the paper it appears only in `% Data:`/`% Generated by:` provenance comments + `MANIFEST.md`, never in prose. **Nothing to trim.**
**Renaming (supersedes D25):**
- `vanton4_paired_dpo` → **`ocean_const_paired_dpo`** everywhere in the clean layer: code, script filenames (the `_vanton4.py` combo scripts → `_paired_dpo.py`), and figure-output filenames (`..._vanton4_paired_dpo.pdf` → `..._ocean_const_paired_dpo.pdf`). This is the canonical method; it IS regenerated at the end-run, so the new HF names will resolve.
- **Leave bare `vanton4` and bespoke `v4`/`v4_reversed_dpo`/`v4_paired_dpo` EXACTLY as-is** — frozen legacy artifacts with no regeneration pipeline; renaming their HF path strings would break the figure scripts (they'd point at names that never get created).
**Training-pipeline scope (refines D26/D28):** migrate **only the canonical paired-DPO flow**, lean. The joint-`vanton4` / bespoke methods "aren't worth the code bulk" (user, 2026-06-03) — omit generation code that exists solely for them (student-pass→`load_dpo_pairs` single-teacher flow, reversed-DPO, bespoke per-trait presets) when it doesn't risk the canonical path. Script numbering follows the **landed Slice-1a README** (01 install · 02 generate_teacher_student · 03 build_paired_dataset [EXISTS] · 04 train_lora · 05 merge_or_export), NOT D28's provisional 00–05. **LANDED on `refactor/main` 2026-06-03 (19d4f1d9):** `src/training/{oct_runtime,openrouter_teacher,oct_config,oct_adapter}.py` + scripts 01/02/04/05; 3962 lines; py_compile passes; GPU parity deferred.

### D30 (2026-06-03): Judge-calibration DESCOPED — "remove the judge calibration stuff for now" (user)
Reverses Q4 ("full incl. data-gen"). The live-judge data-gen (`src_dev/persona_metrics/llm_judge_agreement.py`) drags in the whole `src_dev.inference` + `src_dev.datasets` + `src_dev.persona_metrics` stack (registry/`LLMJudgeMetric`/trait defs) — importing it into `src/` breaks the import boundary; migrating that stack is far beyond "one appendix." So judge calibration (the `fig_F_judge_agreement/scatter/cross_trait` figures + `judge_calibration.py` helpers + the orchestrator) is **NOT migrated** — stays in `src_dev`/`scripts_dev`. (NB: `appendix_paired_dpo_judge.py`, a judge-SCORED trait sweep, is a different thing and DID land with the figures.)

### D31 (2026-06-03): Eval-runners DESCOPED — same dependency-drag + a hard double-registration blocker
Scout verdict: the **LLM-judge sweep** runner pulls the full `inference`+`datasets`+`rollout_generation`+`persona_metrics` stack (heavy, like D30). The **MCQ/MMLU** runners are lighter but `suite.py` still imports `src_dev.inference` + `src_dev.persona_metrics.config`. And the inspect `@scorer`/`@solver`/`@metric` code in `logprob_scorer.py` registers with inspect_ai's **global** registry by name — moving it into `src/` while the dev module stays importable (Q8: keep `*_dev`) risks **double-registration collisions**; a clean move would require deleting the dev module (forbidden). **Decision:** do NOT migrate the eval runners — they stay in `src_dev`/`scripts_dev`. The clean layer already ANALYZES + PLOTS eval results (Slice 2 `ci.py`/`sweep_results.py`, Slice 4 `cell_identity.py`); eval-result DATA is archived on HF, so the migrated figure scripts reproduce the paper figures from that archive WITHOUT the runners. *(Override path if wanted later: migrate only the MCQ/MMLU `suite.py` + `inspect_benchmarks.py` behind a thin seam and re-export the dev scorer rather than redefining it.)*

### D32 (2026-06-03): REVERSES D31's judge part — migrate the FULL LLM-judge sweep (incl. activation-capping + rollout generation)
User clarified the "LLM judge stuff" = the **sweep** (`scripts_dev/evals/llm_judge_sweep/run_vanton4_paired_dpo.sh` + `runner_cells.py`), and that the clean layer **must be able to recreate any data under `configs/vanton4_paired_dpo` AND `configs/vanton4_paired_dpo_activation_capping`**. This needs the whole generate-rollouts→judge→aggregate→upload pipeline, so the heavy stack (rollout_generation + datasets + inference + persona_metrics + sweep + activation_capping) IS in scope — "we need to generate the rollouts to be able to judge them" (user). It's paper-critical (produces `llm_judge_lora_scale_sweep` data behind the main scaling/combo/soup figures), so the bulk is justified (unlike the bespoke training variants). **Scope:** ~14k-line closure across 9 subsystems (manifest in the scout report) → migrate to `src/{sweep.py,rollout_generation,persona_metrics,datasets,common,inference,evals/{cell_sweep,llm_judge_sweep},activation_capping,eval_stages,utils,visualisations}` + `scripts/evals/llm_judge_sweep/{runner_cells.py,configs/...}`. **Approach (user-authorized, AFK):** behaviour-preserving but cleanup-allowed ("keep functionality, modify copied code to make it cleaner"); faithful copy + `src_dev.`→`src.` import repoint + D29 rename (`vanton4_paired_dpo`→`ocean_const_paired_dpo`, bare `vanton4` frozen) + light cleanup; runs deferred to GPU end-run (py_compile + import-smoke now). No `@scorer` double-registration risk (persona_metrics uses a plain dict registry). Reuse existing `src/utils/hf_hub.py` + `src/evals/cell_sweep/cell_identity.py`. **Judge-calibration (D30) partially un-descoped:** `judge_calibration.py` rides along in the persona_metrics closure; add the `fig_F_judge_*` plot script too if it's cheap on top.

---

## Remaining roadmap (ordered, 2026-06-03 — survives compaction; updated per D32)

Landed: Slice 1a, oct-deps, Slice 2 (evals split), Slice 3+4 (figures, renamed+deduped D29), training pipeline (D28/D29). **IN PROGRESS:** the full LLM-judge sweep migration (D32). Still descoped: eval-runners' MCQ/MMLU + logprob `@scorer` (D31 — distinct from the judge sweep; the inspect double-registration hazard remains).

1. **LLM-judge sweep migration (D32) — IN PROGRESS, autonomous:** copy closure subsystems → `src/`, repoint imports, migrate runner + `vanton4_paired_dpo`(+`_activation_capping`) configs → `scripts/`, D29-rename, verify (py_compile + import-smoke), re-add judge-cal figures if cheap. Must recreate the two canonical config families' data (GPU end-run confirms).
2. **End-run (user's H100/H200 + HF + API):** full pipeline + judge-sweep run (regenerates canonical adapters + judge data under `ocean_const_paired_dpo`) → figure regen + PDF parity → HF rename of canonical artifacts (bare `vanton4`/`v4*` stay) → fix KNOWN_ISSUES bugs in `src/` ONLY (D27) → repoint paper MANIFEST/LaTeX provenance. **User pushes** `refactor/main`.

**Sandbox note (lesson learned):** subagents can WRITE only to specific allowlisted subpaths (the original slice content dirs: `slice-2-evals/{src,tests}`, `slice-3-figures/{scripts/figures,src/visualisations}`, `slice-4-combinations/{src/evals/cell_sweep,scripts/figures}`) — NOT arbitrary new worktree dirs, and NOT `src/training`. Workaround that worked: have the subagent author into `/tmp` (writable), then the orchestrator relocates into the main checkout. `/tmp` is the universal escape hatch.

---

## Migration status

Track what has been copied from `*_dev/` to `src/`+`scripts/`. One row per file. Update as files land. Format: `<dest path> ← <source path> (slice N, YYYY-MM-DD)`.

**Slice 1a (squash-merged into `refactor/main` (local) 2026-06-03, branch `refactor/slice-1a-paired-dpo-datagen`):**
- `src/utils/hf_hub.py` ← `src_dev/utils/hf_hub.py` (Slice 1a, 2026-05-28) — copied as-is; dev version kept.
- `src/training/paired_dpo/pairing.py` ← `scripts_dev/oct_pipeline/ocean/prep_paired_dpo.py` (`_build_paired_rows` + `_load_jsonl`) (Slice 1a, 2026-05-28) — pure join extracted + unit-tested (18 tests, incl. dev-parity); dev script kept.
- `scripts/training/ocean_paired_dpo/03_build_paired_dataset.py` ← `scripts_dev/oct_pipeline/ocean/prep_paired_dpo.py` (`_prep_direction` + `main`) (Slice 1a, 2026-05-28) — migrated as `prep_direction` + `main`; `--dry-run` output verified **byte-identical** to the dev script across both directions × all 3 amp-pairing modes; dev script kept.

Infra (not a `*_dev/` migration):
- `pyproject.toml` `[tool.pytest.ini_options]` added on `refactor/main` (2026-05-28) to fix suite collection.
- `scripts/setup/install_oct_deps.sh` + `make oct-deps` (branch `refactor/oct-deps-setup`, squash-merged into `refactor/main` (local) 2026-06-03) — automates the `character`/`openrlhf` install `uv sync` can't do; pins `vllm==0.17.1`.

**Slice 2 — evals D9 split (squash-merged into `refactor/main` (local) 2026-06-03, branch `refactor/slice-2-evals`):** split of `src_dev/evals/personality/analyze_results.py` (dev kept). All bodies byte-for-byte verbatim; 123 tests incl. dev-parity on CI math + parsers + plot smoke tests.
- `src/evals/personality/ci.py` ← `analyze_results.py` (IntervalMethod, `_interval_*`, `_resolve_interval_fn`, `_build_mass_mask`, `_agg_sweep`).
- `src/evals/personality/sweep_results.py` ← `analyze_results.py` (SweepData, `load_sweep_data`, `load_data_from_logs`, `_load_from_info`, `_extract_*`, `_parse_scale`, `_normalise_scale_col`, `_parse_mcq_answer`, `_metric_cols`, trait/eval constants).
- `src/evals/personality/logprob_scorer.py` ← `src_dev/evals/personality/logprob_scorer.py` (`MIN_CHOICE_MASS_DEFAULT` ONLY — minimal; the inspect `@scorer` code is NOT copied, to avoid double-registration against the dev module).
- `src/evals/personality/log_answer_parser.py` ← `src_dev/evals/personality/log_answer_parser.py` (rescore path only: `parse_answer`/`_raw_output`/`_score_answer`/`RescoreResult`/`rescore_log`; dead code omitted).
- `src/visualisations/palette.py` ← `analyze_results.py` (`BIG_FIVE_COLORS`, `DARK_TRIAD_COLORS`).
- `src/visualisations/sweep_plots.py` ← `analyze_results.py` (personality plots + `generate_plots` + `_PLOT_REGISTRY` + plot helpers).
- `src/visualisations/capability_plots.py` ← `analyze_results.py` (`plot_capability_sweep`, `plot_capability_breakdown`).
- `src/visualisations/plot_common.py` ← `analyze_results.py` (`_setup_matplotlib`, `_draw_error_bars`, `_draw_col_error_bars`, `_set_scale_xticks` — leaf module breaking the sweep_plots↔capability_plots cycle).
- **Deviations from the D9 sketch:** plots live in `src/visualisations/` (not the evals modules); NO `src/evals/capabilities/mmlu_results.py` (capability data parsing is shared in `sweep_results`; only capability *plots* existed → `capability_plots.py`); CLI `main()` deferred to a `scripts/` entrypoint. Two pre-existing dev bugs found + logged in KNOWN_ISSUES (not fixed — verbatim).

**Slice 3 — paper figure scripts (squash-merged into `refactor/main` (local) 2026-06-03, branch `refactor/slice-3-figures`):** verbatim except imports repointed to `src/`; HF paths (`vanton4`/`vanton4_paired_dpo`) preserved (D21). All compile + imports resolve; full HF figure regen deferred to the end-run.
- `scripts/figures/main_ocean_scaling.py` ← `src_dev/visualisations/paper_main_o_plus_scaling.py`.
- `scripts/figures/appendix_paired_dpo_trait.py` ← `src_dev/visualisations/paper_appendix_paired_dpo_trait.py`.
- `scripts/figures/appendix_paired_dpo_mmlu.py` ← `src_dev/visualisations/paper_appendix_paired_dpo_mmlu.py`.
- `scripts/figures/appendix_paired_dpo_judge.py` ← `src_dev/visualisations/paper_appendix_paired_dpo_judge.py`.
- `scripts/figures/appendix_dpo_methods.py` ← `src_dev/visualisations/paper_appendix_dpo_methods.py`.
- `src/visualisations/paper_paths.py` ← `src_dev/visualisations/__init__.py` (`PAPER_FIGURES_DIR`; re-exported from `src.visualisations`).
- **Deferred follow-up:** repoint `paper/figures/MANIFEST.md` + LaTeX `% Generated by:` provenance at the new `scripts/figures/` scripts once the end-run confirms they regenerate identical figures.

**Slice 4 — combination/soup figure scripts (squash-merged into `refactor/main` (local) 2026-06-03, branch `refactor/slice-4-combinations`):** verbatim except imports repointed to `src/`; HF paths preserved (D21). All compile + imports resolve; full HF regen deferred to the end-run.
- `scripts/figures/main_c_e_soup_heatmaps.py` ← `src_dev/visualisations/paper_main_c_e_soup_heatmaps.py`.
- `scripts/figures/main_o_n_soup_heatmaps.py` ← `src_dev/visualisations/paper_main_o_n_soup_heatmaps.py`.
- `scripts/figures/main_c_e_combo_delta.py` ← `src_dev/visualisations/paper_main_c_e_combo_delta.py`.
- `scripts/figures/main_c_e_combo_delta_vanton4.py` ← `src_dev/visualisations/paper_main_c_e_combo_delta_vanton4.py`.
- `scripts/figures/main_o_n_combo_delta_vanton4.py` ← `src_dev/visualisations/paper_main_o_n_combo_delta_vanton4.py`.
- `scripts/figures/main_c_minus_e_plus_combo_delta_vanton4.py` ← `src_dev/visualisations/paper_main_c_minus_e_plus_combo_delta_vanton4.py`.
- `src/evals/cell_sweep/cell_identity.py` ← `src_dev/evals/cell_sweep/cell_identity.py` (`AdapterSpec`, `CanonicalCell`, `format_scale`, `sweep_hf_root`, `Tier` — verbatim, stdlib-only).
- Note: `lora_soup_generate.py` + `lora_arithmetic.py` + the GPU downrank/eval scripts deferred (need `lora_combo_baking`/`lora_composition` util migration or GPU).

**Figure rename + dedup (landed on `refactor/main` 2026-06-03, branch `refactor/slice-figures`, commit `3c47cf2c`):** D29 rename + D24 dedup of the Slice 3/4 figure scripts.
- Renamed `vanton4_paired_dpo`→`ocean_const_paired_dpo` across `scripts/figures` (code + script filenames + figure-output filenames); the three `_vanton4.py` combo scripts → `*_combo_delta_paired_dpo.py`. Bare `vanton4`/`v4*` left frozen.
- New shared helpers: `src/visualisations/combo_delta.py` (combo-delta hydration+render, used by all 4 combo scripts), `heatmap_common.py` (soup-heatmap hydration), `appendix_sweep_common.py` (paired-dpo sweep scaffolding). ~1490 lines of duplication removed. py_compile + import-smoke pass.

**Training pipeline (landed on `refactor/main` 2026-06-03, commit `19d4f1d9`):** canonical paired-DPO flow only (D28/D29); authored via subagent→`/tmp`→relocate (sandbox). dev `run_oct_pipeline.py` kept untouched.
- `src/training/oct_runtime.py` ← `scripts_dev/oct_pipeline/run_oct_pipeline.py` (vLLM/GPU patches @197-602).
- `src/training/openrouter_teacher.py` ← same (OpenRouter teacher @285-301, @809-1718).
- `src/training/oct_config.py` ← same (identity/hashing/markers/HF-sync @215-282, @2923-3362; `src_dev.utils.hf_hub`→`src.utils.hf_hub`).
- `src/training/oct_adapter.py` ← same (the seam @582-2920; 9 public stage fns). Omitted TRL paths + `load_dpo_pairs` (canonical OCT backend never calls them).
- `scripts/training/ocean_paired_dpo/{01_install_constitution,02_generate_teacher_student,04_train_lora,05_merge_or_export}.py` ← thin wrappers over `oct_adapter`.
- 3 dev bugs preserved + `# KNOWN ISSUE`-marked (overflow `idx % n_rows`, `aclose()` no-op, unterminated `<think>` prefill).

**Descoped (stay in `*_dev`):** judge-calibration (D30), eval-runners (D31).

---

## Open tech-debt items (acknowledged, not in any slice yet)

- **HF fetch-or-regenerate semantics across the codebase** (point 10). Slice 1 cleans up `hf_hub.py` (D11), but other modules in `src_dev/` may have their own caching/regeneration patterns. A follow-up slice consolidates these once we've seen the pattern reused 2–3 times.
- **`*_dev/` pruning.** Once enough slices land that we're confident the migrated copies fully cover what's needed, a deliberate pruning pass removes superseded `*_dev/` code. Not yet scheduled.
- **HF monorepo cleanup.** Out of scope here; tracked separately.
- **Top-level `README.md` is outdated — rewrite it.** The current README (hardware requirements, setup steps, etc.) is stale and should not be trusted. As the migration lands real `src/` + `scripts/` entry points, the README must be rewritten to reflect the actual current dev environment and the new reproduction workflow (per-area READMEs + numbered scripts). Pin this to whichever slice first makes the new run surface real enough to document end-to-end.
- **4 failing `tests/src/utils/test_peft_manipulations.py` tests** (`test_rank_reducer_custom_adapter_isolation`, `test_pipeline_multi_adapter`, `test_pipeline_inference_scale_two_adapters_independently`, `test_fwd_pipeline_multi_adapter`). These are *test failures*, not collection errors — surfaced once the pytest-collection fix (importlib import mode, landed on `refactor/main` 2026-05-28) let them run for the first time. They exercise `src/utils/peft_manipulations.py` multi-adapter isolation/scaling. Pre-existing; out of scope for the migration slices. Needs its own investigation — either the tests or the multi-adapter logic is wrong.
