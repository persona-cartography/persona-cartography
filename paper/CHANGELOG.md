# Paper Changelog

Tracks substantive changes to the paper across versions. Entries compare the
current state against the previous reference version.

---

## Open issues

Known problems not yet fixed (logged here as they're found):

- **Appendix/figure naming drift.** Labels embed stale letters (`sec:appendix-e`
  renders as Appendix C; likewise `-f`, `-i`) and figure filenames carry stale
  letter prefixes (`fig_F_`, `fig_G_`). Rendered output is correct (all refs use
  `\Cref`); only the source identifiers mislead. Cosmetic.
- **Duplication in the recovered figure generators.** The 12 induction/amp-sup
  generators in `src_dev/visualisations/` (~2,645 lines) repeat helpers
  (`_bootstrap`, `_aggregate`, `_load`) 4–9× each — candidate for a shared
  `induction_common.py` before any promotion to `src/`.
- **Three unsupervised figures still render raw slugs.** Figures 4_2_2 /
  4_2_3 / 4_2_5 show `llama-3.1-8b` / `qwen2.5-7b`; the generating script is
  fixed (four slug-as-label sites) but the v7pf3 questionnaire-run data is on
  no reachable HF repo, so regeneration needs the run owner (upload the two
  run dirs to `psychometric-fa-runs`, or re-run the patched
  `analysis_for_paper.v2.py`).
- **MANIFEST rows missing for new psych-adapter figures.**
  `figures/appendix/induction/n150_TRAIT_cross_latent.png` and
  `coherence_comparison.png` (the latter currently unreferenced) need
  generating-script rows once colleagues supply the pointers.

---

## Figure regeneration, WJ appendix restructure, downstream pointers — logged 2026-07-06

- **Figures regenerated with standardised model names** (identical data, label
  text only): the three judge-calibration figures (display names updated in
  `judge_calibration_common.py`; e.g. "Qwen 3 235B" -> "Qwen3-235B-A22B",
  "Haiku 3.5" -> "Claude 3.5 Haiku") and the five cross-model figures (legends
  completed to `Llama-3.1-8B-Instruct` / `Gemma-3-{4,12,27}B-IT`). Resolves the
  calibration-label Open issue. The three unsupervised FA figures could not be
  regenerated (see Open issues); their script is fixed for whoever holds the
  run data.
- **WildJailbreak full per-trait breakdown moved to its own appendix.** It was
  the only experiment-results section inside the Evaluations appendix and its
  position orphaned Multi-Turn Frustration as its child. Now a standalone
  appendix (placed by first-reference order); Multi-Turn Frustration correctly
  sits under Downstream Behavioural Evaluations; stale "alongside the per-trait
  results" pointer fixed.
- **Downstream evaluations now point at their protocol appendices.** None of
  the four protocol subsections (sycophancy, CoCoNot, WildJailbreak,
  frustration) were referenced from the main body; each downstream paragraph
  now carries its pointer, and the sycophancy/CoCoNot sentence routes its
  benchmark citations through the appendix instead of an unanchored
  citation blob.
- **Inspect AI cited as a framework** (new `inspectai2024` entry); the
  evaluations appendix cites the framework at "Inspect AI" and the community
  `inspect_evals` repo separately right after.
- **References audited end-to-end**: all 35 cited arXiv eprints verified
  against the arXiv API (ids + titles), all 16 cited URLs resolve, journal /
  proceedings entries confirmed real, and every in-body citation checked
  against what it cites. Appendix input order re-verified (heatmaps_residuals
  moved before the cross-model appendix after the Overleaf figure move).

---

## v1 submission → current: consolidated reader-facing diff — logged 2026-07-03

_Full-text comparison of the NeurIPS submission PDF (#26536, 72 pp) against the
current build (83 pp). High-level differences a reader of both would notice;
detailed mechanics are in the entries below._

- **Front matter de-anonymised**: author block with emails and affiliations
  (LASR Labs, ENS Paris-Saclay & MATS, UK AI Security Institute),
  equal-contribution footnote, Acknowledgments section, and logo-styled
  GitHub + HuggingFace links after the abstract. v1 had none of these.
- **Abstract rewritten**: now states the six-model / three-family (4B-32B)
  scope, names the four recovered TIDE factors (tone, initiative, didacticism,
  epistemic caution), and fixes "Extroversion" -> "Extraversion" (spelling fixed
  throughout the prose).
- **New appendix**: "Comparing Across Baseline Models and Teachers"
  (cross-model transfer across the six baselines + GLM-vs-DeepSeek teacher
  ablation) — did not exist in v1.
- **New appendix content**: "Combinations of Randomly Scaled OCEAN LoRAs"
  subsection (32 five-adapter combos), "WildJailbreak: Full Per-Trait
  Breakdown", "Comparison with Trait-Conditioned Adapters" (PsychAdapter
  benchmark + judge face-validity check), and a much-expanded
  "Factor Analysis: Methodology, Validation, and Per-Factor Details" appendix;
  "Residuals Heatmaps" split out as its own appendix; flattened-weight-space
  appendix gained the per-adapter Frobenius-norm table.
- **Consistency passes** (detailed in entries below): model names standardised
  via constants, citation ties/\Cref/`\texttt` benchmark hygiene,
  tercile/tertile unified, appendix order matched to first-reference order,
  appendix headings title-cased.

---

## Model naming finalised + citation-placement pass — logged 2026-07-03

- **Dash-style decision resolved** (was an Open issue): open-weights models keep
  their dashed HF-id names (Llama/Qwen/Gemma/GLM/DeepSeek, `Kimi-K2`,
  `Mistral-Small-3.2-24B`, `Gemma-4-26B-A4B`); API-only models use vendor
  marketing names with spaces (`Claude Opus 4.7`, `Claude 3.5 Haiku`,
  `GPT-5 mini/nano`, `GPT-4.1 mini/nano`, `GPT-5.4 nano`, `GPT-3.5 Turbo`,
  `Gemini 2.0 Flash`, `Gemini 2.0 Flash-Lite`). Grounded in literature usage
  (papers copy HF ids for open models; mixed for closed) and vendor stylings.
- **Previously-uncited models cited** via model/system cards matching house
  style: Claude Opus 4.6/4.7 system cards, GPT-5.4 nano API model card,
  GPT-3.5 Turbo API model card.
- **Citation placement normalised**: every model is cited at its first prose
  appearance in the main body and again at first prose appearance in each
  appendix; repeat citations within a unit removed (caption mentions exempt).
- **Bib fixes**: duplicate GPT-5 system-card entries merged
  (`openai2025gpt5` -> `singh2026gpt5`); DeepSeek-V3.2 now cites the V3.2
  report (arXiv 2512.02556) instead of the V3 technical report.

---

## Model-name constants + tercile normalisation — logged 2026-07-03

_On `anton/overleaf-sync`, following the Overleaf import and citation/Cref
hygiene passes._

- **Every specific model name now comes from a constant.** `main.tex` defines
  one `\newcommand` per model (30 constants, `xspace`-terminated; naming scheme
  `<Family><Version, "Point" for decimals><"Size"><params>B<Suffix>`), and all
  ~150 prose/caption mentions across sections and appendices route through them
  — renaming a model is now a one-line change. Standardisations applied in the
  process: dashed style throughout; judge named in full as `Qwen3-235B-A22B`;
  `Gemma-3-27b-IT` casing fixed; compact `Qwen3-8B/32B` and
  `Gemma-3-4B/12B/27B-IT` lists expanded to full per-model names; truncated
  `Qwen2.5-7B` table rows and `Llama-3.1-8B baseline` captions completed to
  their `-Instruct` forms; bare `Llama`/`Qwen` prose in the FA appendix replaced
  with the specific models; `\texttt{}`-styled model ids (`deepseek-v3`,
  `gpt-4.1-mini`, `gpt-3.5-turbo`) converted to prose names; ambiguous
  "Gemini Flash" resolved to `Gemini-2.0-Flash` (verified against the
  calibration rater ids); `Gemma-1-2B-IT` renamed `Gemma-2B-IT`. Family-level
  references (e.g. figure-legend "Llama blue, Qwen orange/red, Gemma green")
  intentionally left as words.
- **Tercile/tertile unified.** The paper mixed "tercile" (combinations appendix)
  and "tertile" (FA appendix, unsupervised section); all 16 "tertile" occurrences
  are now "tercile".

---

## Combinations appendix, flattened-norms table, benchmark-name formatting — logged 2026-06-29

_Paper-content changes on `anton/paper_updates` since the 2026-06-25 entries._

### New appendix content
- **New subsection "Combinations of Randomly Scaled OCEAN LoRAs"** in the OCEAN
  Evaluation Sweeps appendix (`ocean_results`). Evaluates 32 random combinations
  that activate all five OCEAN adapters at once, each at an independently-drawn
  direction and scale. Three new figures: per-trait `TRAIT`-score dose-response
  (per-tercile straight-line fits), per-trait `MMLU` dose-response (per-tercile
  Gaussian Nadaraya–Watson smooths), and `MMLU` vs total adapter magnitude (sum of
  scales). Generated by new clean-layer scripts
  `scripts/visualisations/appendix_combinations.py` +
  `src/visualisations/combinations_common.py` (loads the 32 configs from the HF
  monorepo; trait CIs = bootstrap, MMLU = Wilson).
- **New flattened-weight-space norms table** (`tab:flattened_norms`): the Frobenius
  norm ‖ΔW‖ of each of the ten OCEAN adapters. The norms cluster tightly
  (6.08–6.53), establishing that the adapters are ~equal magnitude in weight space;
  the combinations subsection cites this to justify treating the sum of LoRA scales
  as a proxy for total intervention magnitude.

### Formatting / consistency
- **Benchmark names normalised to `\texttt{}`** throughout the appendices. All
  prose/caption references to `TRAIT`, `MMLU`, `GSM8K`, and `TruthfulQA` are now
  monospaced (previously bare in activation-capping, downranking,
  base↔instruct-interpolation, cross-model/teacher, the OCEAN-results macro
  captions, and the evaluations appendix). Mixed-case mentions ("Trait sweep",
  "trait logprob") were also capitalised + wrapped. Generic lowercase "trait"
  (the personality trait, not the benchmark) left unchanged.

### Structure
- **Combination-heatmap figure relocated** in the supervised-modulation section: the
  `fig:combination-heatmap` block now sits *below* the scaling/combination prose
  (including "Linear combinations recover mixed personas") rather than above it; its
  `\vspace{-1.75em}` was removed (NeurIPS spacing compliance).
- **Appendix order updated** (`main.tex`): `heatmaps_residuals` moved later (now
  after the base↔instruct-interpolation appendix) to keep appendices in
  first-reference order after the figure move.

### References
- **Bibliography corrections.** Fixed stale reference entries (several converted
  to canonical arXiv `@misc` records with corrected metadata — e.g.
  `brown2020language`, `gemma_2025`, `ilharco2023editing`, `zou2023representation`,
  `bai2022constitutional`) and filled in missing author names (expanded the
  truncated "and others" author lists to their full author lists).

### Front matter
- **Author details added.**
- **Code/data links added.** GitHub repo and HuggingFace `monorepo` links added to
  the first-page footer.

### Main body
- **Cross-model / teacher generality surfaced earlier.** The intro contributions
  bullet and the §3 Methods now state that the pipeline is run across a range of
  model sizes and families and with different teacher models (default
  Llama-3.1-8B-Instruct / GLM-4.5-Air unless stated otherwise).

---

## Judge-calibration figures — logged 2026-06-25

The three LLM-judge calibration figures (cross-trait ρ/MAE heatmaps,
panel-judge-vs-human scatter, inter/intra-rater agreement bars) were regenerated.
Reported numbers are unchanged; the differences are cosmetic:
- **Fig. 14(a):** the three human raters now have distinct colours (were all the
  same blue, leaving the leave-one-out bars indistinguishable).
- The **"Gemma 4 26B-A4B"** judge name now appears correctly throughout
  (previously hand-patched onto the figures).
- The **Llama-3.3-70B** bars in the agreement figure now match its colour in the
  scatter (the two figures had coloured it differently).

---

## Update logged 2026-06-25

_Paper-content changes landed on `anton/paper_updates` since the 2026-06-24
entry. (The branch also renames the HF monorepo `persona-shattering-lasr` →
`persona-cartography` across the figure-generating scripts and adds private-repo
HF auth, but those are tooling-only and don't change the compiled paper.)_

### Content corrections / clarifications
- **Induction cross-LoRA panel (a) corrected.** Investigating the rollout
  provenance showed our earlier description was wrong: the panel's adapter was
  generated with the `e_plus_no_dpo` slug, which — despite its name — resolved
  (at the generating commit) to the **merged `vanton4-persona` (DPO+SFT) adapter**
  of the older recipe, *not* an SFT-only / no-DPO adapter. The only real
  difference from the canonical reference is **teacher-student vs paired-teacher
  DPO**. Panel (a) was relabelled accordingly ("OCEAN definition constitution,
  teacher-student DPO"), dropping the incorrect "SFT LoRA alone" claim; the
  generating script's misleading `NO_DPO_PATHS` identifier and docstring were
  renamed/annotated to match.

### Methods detail
- **LoRA target modules now stated.** The main body adds "(rank 64, applied to
  all attention and MLP matrices)" to the training description; the training
  appendix's "applied to all weight matrices of Llama-3.1-8B-Instruct" became
  "applied to all attention and MLP matrices of the model" (accurate across all
  trained families — Llama/Qwen via PEFT `all-linear`, Gemma via its explicit
  module list).

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
  agreement bars)** — calibration values are unchanged from v1; the judge is
  correctly named **"Gemma 4 26B-A4B"** (`google/gemma-4-26b-a4b-it`, a
  25.2B-total / 3.8B-active MoE).
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
