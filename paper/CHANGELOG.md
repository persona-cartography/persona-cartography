# Paper Changelog

Tracks substantive changes to the paper across versions. Entries compare the
current state against the previous reference version.

---

## Open issues

Known problems not yet fixed (logged here as they're found):

- **Fig. 14(a) — the three human raters share one colour.** `plot_agreement_bars`
  colours all human leave-one-out bars (H1/H2/H3) with a single blue
  (`human_colour = "#4363d8"`), so the bars and legend swatches are
  indistinguishable. The judges get distinct colours; the humans don't.
  (`scripts_dev/persona_metrics/llm_judge/plot_paper_judge_calibration.py`;
  note `RATER_COLOURS` already defines distinct per-human colours but this plot
  ignores them.) Fixing it needs the figure regenerated — blocked by the missing
  calibration data below.
- **Judge-calibration figures (Figs 12–14) can't be regenerated.** Their per-judge
  score data lives in gitignored `scratch/golden_calibration`, absent from both
  checkouts and not on an accessible HF repo. Committed figures came from a
  one-off local run; the Gemma relabel was applied by PDF surgery. Any *content*
  fix (e.g. the H1/H2/H3 colours) requires recovering that data or re-running the
  calibration (which would change the reported numbers).
- **Appendix/figure naming drift.** Labels embed stale letters (`sec:appendix-e`
  renders as Appendix C; likewise `-f`, `-i`) and figure filenames carry stale
  letter prefixes (`fig_F_`, `fig_G_`). Rendered output is correct (all refs use
  `\Cref`); only the source identifiers mislead. Cosmetic.
- **Duplication in the recovered figure generators.** The 12 induction/amp-sup
  generators in `src_dev/visualisations/` (~2,645 lines) repeat helpers
  (`_bootstrap`, `_aggregate`, `_load`) 4–9× each — candidate for a shared
  `induction_common.py` before any promotion to `src/`.
- **agreement_bars legend cosmetics.** The Gemma entry is mildly shrunk (25%) to
  fit panel (a); panel (b)'s entry still overflows its old-width white patch
  (deliberately left as-is).

---

## v1 (NeurIPS 2026 submission) → current — update logged 2026-06-24

_Continues the v1→current comparison below; records only changes that **landed
since the 2026-06-22 entry** (net effect, not intermediate states). v1 = the
originally submitted PDF (72 pp); current build = 81 pp. The growth comes from
the new cross-model/teacher appendix plus restored paragraph spacing (see
"Style/build" below)._

### New appendix + figures
- **New appendix "Comparing Across Baseline Models and Teachers"**, replacing
  v1's single-trait cross-model figure (the conscientiousness-suppressor
  cross-model comparison at matched LoRA scales). Two subsections:
  - **"Baseline Models"** — OCEAN trait + MMLU LoRA-scale sweeps across six
    baselines (Llama-3.1-8B-Instruct, Qwen3-8B, Qwen3-32B, Gemma-3-4B/12B/27B-IT),
    amplifier / suppressor / control.
  - **"Distillation Teachers"** — teacher ablation (GLM-4.5-Air vs DeepSeek-V3.2)
    on Llama-3.1-8B, concluding the pipeline is robust to the teacher swap.
- **New figure generator** `scripts/visualisations/model_comparison_ocean_transfer.py`
  (clean `scripts/` layer; `--set {cross_model,llama_teacher}`) producing
  `figures/appendix/model_comparison/fig_crossmodel_*` and `fig_teacher_*`.
  Trait error bars = bootstrap CIs, MMLU = Wilson; choice-mass ≥0.75 filter.

### Main body and Discussion
- Added a **robustness paragraph** ("Trait transfer is robust across models,
  teachers, and adapter compression") above "Linear combinations recover mixed
  personas" in the supervised-modulation section, consolidating pointers to the
  cross-model, teacher, rank-1 downranking, and base↔instruct-interpolation
  appendices. With it, **all previously-uncited appendices are now referenced
  from the main body** (also added pointers to the flattened-weight-space and
  activation-capping appendices).
- **The default distillation teacher is now named in the body**: where v1 said
  only "a teacher model generates paired responses", the training paragraph now
  reads "a strong teacher model (GLM-4.5-Air, unless stated otherwise)". (The
  inline "see the teacher-comparison appendix" pointer that briefly accompanied
  it has since moved into the robustness paragraph above.)
- **Discussion section**: removed "We would like to see this study replicated
  over different model families and sizes" — now addressed by the new cross-model
  appendix.

### Content corrections / clarifications
- **DPO+SFT model souping corrected** (in the "Model souping" subsection of the
  LoRA Training Methods appendix). The merge is a **rank-preserving, factor-space
  merge** (PEFT `add_weighted_adapter` `"linear"` mode), **not** a linear sum of
  the two adapters' weight deltas: it sums the low-rank factors with √-scaled
  weights and so introduces cross terms while keeping rank 64. That subsection now
  documents this; the misleading "linearly combined" / "parameter averaging" /
  "weighted blend of the weight matrices" wording in the main body, the training
  appendix and the reduced-rank appendix was dropped and now points to it. (This
  is *distinct* from the genuinely-linear weight-space composition of separate
  OCEAN adapters in the main body.)
- **Default training pipeline clarified** (LoRA Training Methods appendix). The
  "alternative training methods" subsection was rewritten to state the actual
  default up front (programmatic OCEAN-definition constitution + paired-teacher
  DPO) and reframe the four bespoke recipes as alternatives compared against it;
  `fig_B_dpo_methods_scaling.pdf` regenerated with a corrected label.
- **Control-adapter description corrected (5 places: main body, training appendix,
  constitutions appendix, two cross-model captions).** Its (chosen, rejected)
  pairs are **two neutral-constitution generations distinguished by seed (seed-1
  chosen, seed-2 rejected)**, not responses drawn at random from a pool.
- **Induction cross-LoRA panel (a)** relabelled: it is an older
  teacher-student-DPO E↑ adapter applied as its SFT LoRA alone (not the canonical
  merged DPO+SFT); an unsupported causal claim about the pipeline was dropped.

### Citations
- Added **GLM-4.5** (`zai2025glm45`); fixed `gemma_2025` to point at the Gemma-3
  technical report.

### Prose edits
- **LoRA Training Methods appendix**: "how I exist" → "how I am" when describing
  how constitutions frame traits — i.e. as a natural self-description rather than
  as a change from some assumed baseline.

### Style / build
- **Switched NeurIPS style 2025 → 2026** (`neurips_2026.sty` added,
  `neurips_2025.sty` removed). Layout spec is unchanged for our purposes.
- **All 47 `\paragraph{}` run-ins → `\textbf{}`** for uniform run-in spacing
  matching the body.
- **Fixed a `\captionof`-outside-float bug** in the OCEAN-evals appendix that had
  silently collapsed inter-paragraph spacing across most of the appendices
  (~119 paragraph gaps rendered as line breaks); restoring them accounts for part
  of the page-count growth.

### Paper-dir / tooling (not all visible in the compiled PDF)
- **Appendices reordered** in `main.tex` to first-reference order in the body.
- **Figure tree reorganised**: flat `figures/appendix/*.pdf` moved into
  per-appendix subdirectories; orphan PNGs/PDFs deleted; `MANIFEST.md` updated.
- **Archived unused source** to `paper/_archive/` (`trait_metrics.tex`,
  `alternative_training.tex`, `further_work.tex`, `judge_selection_methodology.md`);
  removed `paper/drafts/`.
- **Dedup**: lifted `per_trait_scores_from_log` and `accuracy_from_log_url` into
  `src/visualisations/appendix_sweep_common.py`, shared by the paired-DPO sweep
  generators.

---

## v1 (NeurIPS 2026 submission) → current — logged 2026-06-22

_All changes in this entry were identified by an automated v1-vs-current
comparison on 2026-06-22; they reflect the state of the paper as of that date._

**Reference:** v1 = the originally submitted PDF.
**Current:** the present build as of 2026-06-22. The extra page comes from the
expanded bibliography (new references below).

Verified by: word-level text diff (521 substantive changed lines) + page-by-page
visual figure comparison + figure-integrity audit (160 used figures, none
blank/data-stripped). No data values or numbers were altered; no figures or
sections were added or removed (68 figures in both versions).

### Figures
- **Fig. 2 (trait-modulation banner), right panel** — added per-response
  bootstrap **error bars**. v1 showed bare points (the CIs were collapsed to
  zero width by a prompt-key bug); these are now real paired-bootstrap intervals
  keyed by prompt id. *(this session)*
- **Figs. 12–14 (LLM-judge calibration: cross-trait+MAE heatmap, scatter,
  agreement bars)** — the judge row/legend label **"Gemma 4 27B" → "Gemma 4
  26B-A4B"** (correct name for `google/gemma-4-26b-a4b-it`, a 25.2B-total /
  3.8B-active MoE). All calibration values are unchanged — data identical to v1;
  only the label differs. On the agreement-bars legend the entry is mildly
  shrunk (25%) with a white-backed panel (a) so the longer name fits. *(this session)*
- All other figures are visually identical to v1. The
  `vanton4_paired_dpo → ocean_const_paired_dpo` migration re-rendered figures
  from the same underlying data (pixel-verified equal to v1).

### References / citations (bibliography expanded → +1 page)
- Added model citations: Claude 3.5 (Anthropic), DeepSeek-V3, Gemini 2.0 Flash,
  Kimi K2, GPT-4.1, GPT-5, Mistral Small 3.2, Gemma 3, **Gemma 4 model card**
  (`google2026gemma4` → official `ai.google.dev/.../model_card_4`, *this session*),
  Qwen 2.5, Qwen 3, Llama 3.3, Llama-4-Scout.
- Added dataset/benchmark inline citations: LIMA [Zhou et al., 2023]; TRAIT
  [Lee et al., 2024] reworded to "trait-specific multiple-choice questions".

### Caption expansions
- **Fig. 3**: now names the N↑ / C↓ adapters and the C↓ MMLU panel.
- **Fig. 4(c)**: adds the near-additivity explanation — residuals over the 45
  OCEAN×OCEAN adapter pairs, the conscientiousness scorer as the lone outlier,
  and the control curve as a non-additivity noise floor.
- **Fig. 2**: adds the judge attribution ("as judged by Qwen3-235B").

### Systematic copy-edits
- **US→UK spelling** (~39 lines): behavior→behaviour, behavioral→behavioural.
- **Model-name normalization** (~53 lines): "Llama 3.1 8B-it" / "LLama-3.1-8B-it"
  / "Llama-3.1-8b" → "Llama-3.1-8B-Instruct"; "Llama 3.3 70B" →
  "Llama-3.3-70B-Instruct"; "Llama 4 Scout" → "Llama-4-Scout"; Gemma-3 sizes
  standardized.
- **Cross-reference capitalization** (~76 lines): section/fig./table/appendix →
  Section/Figure/Table/Appendix.
- **"base" → "baseline"** (~46 lines) for the no-LoRA reference model.

### Prose edits (selected)
- "CoCoNot" → "CoCoNot benchmark"; "Wild-JailBreak" → "WildJailbreak".
- Downstream §3: "Some of this effect" → "A small amount of this effect";
  "Applying an N↓ adapter…" → "Further, applying an N↓ adapter…".
- App. B (constitutions): "Control Model Constitution / same 12-item structure"
  → "Control constitution / single-item constitution"; added a `clarification`
  field description; "(trait, polarity) pair" now exemplified "(Openness, +)".
- Judge calibration: agreement metrics expanded from "Spearman ρ" to "pairwise
  Spearman ρ, MAE, and within-one-point agreement".
- Dropped stale "v2" tags on some adapters (Initiative, conscientiousness
  suppressor).
- The **Gemma judge name was also corrected 3 → 4** in the candidate-pool prose
  (it had drifted to "Gemma 3 27B" with a Gemma-3 citation on `main`). *(this session)*

### Repo/tooling changes on this branch (not visible in the compiled PDF)
- Recovered 12 induction / amp-sup figure generators + the combined
  calibration-figure generator from a dangling commit (`aa7dfe8d`) so the
  Appendix-G and amp×suppressor figures are reproducible again.
- Removed 86 orphan duplicate figures under `figures/appendix/downranking/`
  (none read by the build).
- `vanton4_paired_dpo → ocean_const_paired_dpo` migration across the clean
  `scripts/` + `src/` layers; repointed `% Generated by` comments + `MANIFEST.md`.
- The two judge-calibration scripts that depend on gitignored
  `scratch/golden_calibration` data were kept in `scripts_dev/` (with a
  data-dependency warning) rather than the clean layer, since they cannot be run
  here.

### Not paper content (ignored in this comparison)
- The per-page "Confidential reviewer copy…NeurIPS 2026" watermark and a hidden
  prompt-injection block appear **only in v1** — both added by NeurIPS to the
  reviewer copy, not part of the manuscript.
