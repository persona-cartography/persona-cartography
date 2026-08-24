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
- **Fig 3b (`mmlu_compact-5.pdf`) can't be reproduced by its recorded script.**
  The figure shows Wilson-style CI ticks on the stacked MMLU fractions, but
  `main_ocean_scaling.py::render_mmlu_breakdown` draws no error bars — the
  committed PDF came from an older generator variant. Confirm the CI method
  with the figure's author or re-add error bars to the script and regenerate.
- **Rebuttal work not (yet) incorporated into the paper.** Three experiments
  reported to reviewers in the rebuttal are absent from the paper source:
  beyond-OCEAN sycophancy/psychopathy adapters (Mariia), the DPO:SFT
  souping-ratio ablation over mixes {0, 0.25, 0.5, 1.0} (Mariia; reviewer 7i3n's
  unjustified-0.25 concern), and the Qwen TIDE factor re-labelling (Mariia).
  Also unaddressed from the rebuttal to-do list: engagement with the social
  science literature. Decide whether these go in for camera-ready.
---

## Psychometrics citations + questionnaire-development appendix — logged 2026-08-24

PR #349 (merged 2026-08-24). Discharges the rebuttal promise "A description of
this [questionnaire iteration] process will be added to the paper appendix."

- **New subsection "Questionnaire Development"**
  (`sec:appendix-fa-questionnaire-dev`, first subsection of the FA appendix):
  one-paragraph history of the five questionnaire iterations — 130-item Likert
  v1; debiased 98-item v2 (trade-off reframing, deflection-avoiding
  rephrasing, ceiling-item removal, safety-relevant axes, ~40% reverse-keying);
  scenario-anchored 100-item third version; the forced-choice diagnostic that
  exposed the Likert/FC sign flip (acquiescence); and the final 72-item FC
  rewrite. Sourced from the per-version `description`/`design_notes` fields in
  `datasets/psychometric_questionnaires/*.json`; mechanism attributions (e.g.
  RLHF) deliberately omitted. Cross-referenced from the §4 "iteratively
  refined" sentence.
- **Two citations added** (both verified against publisher records):
  John, Angleitner & Ostendorf 1988 (lexical approach; EJP 2(3) 171–203, DOI)
  joins the §4 factor-analysis-history citep; John, Naumann & Soto 2008
  (Big Five taxonomy chapter, Guilford pp. 114–158, page range verified
  against the chapter scan; Guilford chapters carry no DOI) joins the §3
  OCEAN-grounding citep.

## Proofread pass over the full arXiv-v1 → current diff — logged 2026-08-24

Every change between the arXiv v1 commit (`949d0504`) and current main was
re-read for grammar and factual precision, plus a rendered-PDF formatting check
(no overfull boxes, no undefined refs). PR #351.

- **CoCoNot quantity corrected (supersedes the 2026-07-10 entry)**: "roughly
  20% more" → "roughly 20 percentage points more". The 2026-07-10 change from
  "20 points" to "20%" was itself wrong: 0.33/0.35 vs. 0.14 is ~20 percentage
  points (~2.4×, matching the caption's "2.3x"), not a 20% relative increase.
- **Grammar/pronoun fixes**: §3 opener "the OCEAN traits … as a starting
  point: it is" → "the framework is"; discussion inventory "apply them" →
  "apply the adapters"; coherence-examples sentence restructured (dangling
  participle; scores now precede the quotes); "instrument, e.g." comma; "Other
  results" apposition parenthesised; seed-prompts "giving it room to express"
  → "giving the trait room to express itself".
- **Fig 6 caption quotes**: straight ASCII `"are you sure?"` (rendered as two
  closing quotes) → LaTeX ``…''.
- **Gemma composition heatmaps re-rendered with "-IT" in panel titles**
  (matching captions); label strings only, data unchanged
  (`appendix_o_n_soup_heatmaps_gemma.py`).

## Gemma adapter-composition replication (rebuttal promise) — logged 2026-08-24

Discharges: "Adapter soup replication. Reproduced the composability
experiments on Gemma-3 12B and 27B (e.g., O↑⊕N↑ combinations)." PRs #340 + #350.

- **New appendix G.1.3 "Adapter composition"** (`other_families.tex`,
  `sec:appendix-crossmodel-composition`): O↑⊕N↑ 5×5 scale-grid sweeps on
  Gemma-3-12B-IT and 27B-IT, 100 rollouts/cell on the per-trait open-ended
  prompts, same Qwen3-235B judge as the Llama experiment. Four heatmap panels
  (openness/neuroticism × 12B/27B) in `figures/appendix/model_comparison/`,
  rendered by `scripts/visualisations/appendix_o_n_soup_heatmaps_gemma.py`
  reusing the Fig 4 renderer.
- **All headline numbers verified against monorepo `grid_summary.jsonl`**
  (12B: neuroticism −0.72→+3.95 combined vs +3.90 alone, openness +2.71→+3.47
  vs +3.96; 27B: −1.39→+3.53 vs +3.31, +3.09→+3.89 vs +3.94). Coherence claim
  grounded in the Gemma data (collapse to ≲2 in the N+2 row) rather than
  asserting equivalence with Llama.
- **Main-body pointer** added to the §2.3 robustness paragraph ("Adapter
  composition also replicates on …").
- Sweep configs under `scripts_dev/evals/llm_judge_sweep/configs/
  gemma{12b,27b}_paired_dpo/`; data under `combos/gemma-3-{12b,27b}-it/…`
  on the monorepo.

## Main-body trim to the 10-page budget + camera-ready switch — logged 2026-08-21

Camera-ready page-budget work (PR #347), items agreed one-by-one.

- **Structural cuts**: §3.3 opener and bridge sentences; discussion
  supplementary-experiment inventory compressed to a semicolon list (all
  appendix \Crefs retained); Fig 2 caption rewritten around "signed headroom";
  related-work closing paragraph rewritten; intro OCEAN-theory sentence
  removed; downstream-tasks opener rewritten ("sycophancy and harmful
  compliance").
- **Verbosity pass** (same info, tighter grammar) across intro, §2, §3,
  §4, discussion: four-properties sentence, negative-scaling, orthogonality,
  composition, scaling-method, frustration paragraphs; "their persona" → "its
  persona"; "; Experiments" grammar fix; unsupervised opener + population
  rationale + questionnaire sentence.
- **Camera-ready**: `\usepackage[final,main]{neurips_2026}` (plain `[final]`
  fails — `\@trackname` is only defined by track options) and the NeurIPS
  checklist re-enabled. Checklist Q5 (open access) justification verified
  correct (public GitHub + HF monorepo) — closes the rebuttal item "Fix the
  checklist inconsistency on code/data availability … MAKE SURE CORRECT FOR
  CAMERA READY".

## Reviewer presentation examples in the main body — logged 2026-08-20

Discharges the rebuttal promise that questionnaire/scenario/transcript/
coherence examples be "moved into the main paper". PR #346 (+ #345).

- **§4 population paragraph**: two archetype definitions (enthusiastic,
  worried), two scenario examples (grandmother's-90th poem, mass of a tree —
  from the v2 scenario pool actually used by the run), and two verbatim
  opening exchanges (whimsical/playful_interaction,
  hostile/decision_making) quoted from
  `unsupervised/runs/rollouts-…/exports/conversation_trace.jsonl`.
- **§4 questionnaire paragraph**: verbatim 72-item-instrument example item
  (anticipate-follow-up A/B) + repo pointers.
- **§2 capability paragraph**: coherence-judge sentence with three real
  examples scoring 9.5 / 5 / 2 (from
  `evals/coherence_examples/coherence_examples_high_mid_low.csv`).
- **Seed-prompts appendix fixed** (`ocean_evals.tex`): trait sweeps use the
  per-trait 240-prompt open-ended pools (facet-tagged), not the lu2026
  neutral set; induction + unsupervised use the 299-prompt curated extension.
- **`paper/main.pdf` untracked** (PR #345) — had been force-added on the
  prolific branch; gitignored by design.

## Gemma E↓ free-text investigation (appendix G.1.1.1) — logged 2026-08-19

Discharges the rebuttal promise to investigate the E↓ MCQ reversals with
free-text evaluation. PR #344.

- **New \paragraph "Increasing E at positive E↓ scales."**
  (`sec:appendix-eminus-positive`, after the cross-model TRAIT figures):
  on all three Gemma models the MCQ extraversion score rises at E↓ scales
  +1…+1.5, but free-text LLM-judge extraversion *falls* at +1
  (non-overlapping 95% CIs) and falls-or-holds at +2; rollouts show a change
  of register (shorter, measured responses) rather than stated preference.
- **`tab:eminus-freetext`**: extraversion + coherence judge scores with CIs,
  240 responses/cell, for baseline/+1/+2 on 4B/12B/27B. Data:
  `fine_tuning/gemma-3-{4b,12b,27b}-it/ocean/extraversion/suppressor/…/
  llm_judge_lora_scale_sweep/` + baseline store.
- `\setcounter{secnumdepth}{4}` in `main.tex` so the paragraph is numbered
  (G.1.1.1).

## Reviewer rebuttal edits in main body — logged 2026-08-18

Rebuttal items applied on `anton/paper-edits`: the claim-rescoping promise
("soften safety claims…"), the EC control-LoRA statement ("State explicitly
that part of the Epistemic Caution shift comes from the training pipeline
itself… and that TIDE factor independence is an open question"), and the
promised limitation on judge-calibration disparity + the untested
cross-family pipeline re-run.

- **§4**: EC control-LoRA comparison surfaced ("both show a markedly stronger
  effect on Epistemic Caution than the control LoRA, leaving factor
  independence an open question"); factor-analytic tradition cited
  (Goldberg 1990, Digman 1990) in the §4 opener.
- **Limitations**: downstream judges used off-the-shelf without comparable
  validation; a separate run of the full discovery pipeline on another model
  family named as the key untested next step.
- **Discussion**: "A critical finding for safety" → "A finding with direct
  safety relevance" (rescoped safety framing).
- **FA appendix phrasing**: "the OCEAN-definition control LoRA is not silent"
  → "has an impact".

## Prolific follow-ups: compensation, softened framing — logged 2026-08-18

Post-dates the 2026-08-17 Prolific entry (which left a compensation TODO);
landed on the `prolific` branch before merge.

- **Rater compensation (£20/hour) added** to the appendix protocol paragraph
  and the checklist crowdsourcing justification — the 2026-08-17 entry's
  "TODO left for rater compensation details" is resolved.
- **Framing softened**: Prolific raters presented as additional independent
  validation of the selected judge, not a wholesale replacement of the
  author-annotation round.
- **§3 calibration sentence trimmed** to one clause (length budget).

## Related-work additions + full citation fact-check — logged 2026-08-18

PR #342 (merged 2026-08-18); discharges the rebuttal promise to Reviewer 2
("we will add them [Personality Alignment of LLMs, P-React] to the related
work section") plus the two further papers raised in review (weights2weights,
LPA). Every claim in related work checked against the cited papers.

- **Four papers added**: PAS activation steering (zhu2024personalityalignment,
  inference-time), LPA adversarial-activation safety training
  (merzouk2026latentpersonality), P-React trait-specialised MoE-LoRA
  (dan2024preact, contrasted with our standalone composable adapters), and
  weights2weights (dravid2024interpreting, vision-domain LoRA weight
  manifold).
- **Placement fixes from the fact-check**: InstructGPT dropped from the
  prompting bracket; lu2026assistant added to the steering-sensitivity
  citation; shanahan2023role added to the pre-training persona-types claim.

## Prolific human calibration replaces author-rater results — logged 2026-08-17

Independent Prolific raters replace the authors as the human reference for
LLM-judge calibration (all six categories: OCEAN + coherence). Discharges the
rebuttal promise "Independent human judge validation. Replaced author
annotation with a Prolific study."

- **Appendix E judge-calibration section** (`appendices/ocean_evals.tex`):
  author-annotation paragraph (three annotators on A/N/coherence, H3 bias note)
  replaced with the Prolific protocol (28–36 recruited per category, native
  English + "Qualified AI taskers", same rubric as judges, attention-check
  filter retaining 23–31, pre-existing IRB policy) and a new
  `tab:prolific-calibration` (judge-vs-rater Pearson r 0.73–0.94 per category;
  inter-rater Krippendorff's ordinal α, low only for agreeableness at 0.53).
- **Author-annotation figures commented out** (`fig:judge-scatter`,
  `fig:judge-agreement-bars`) — superseded by the table; regenerable from
  `scripts_dev/evals/llm_judge_sweep/prolific/` if Prolific versions are
  wanted. `fig:judge-cross-trait-and-mae` (vs gold labels) retained. Their
  intra-judge self-consistency numbers kept as a text-only paragraph.
- **Main-text mentions updated**: abstract ("independent crowdsourced human
  raters"), introduction, §3 measurement paragraph (adds the r range +
  Prolific sentence).
- **NeurIPS checklist**: crowdsourcing + IRB items flipped NA→Yes with
  justifications; TODO left for rater compensation details.

## en-GB pass + figure typography/font unification; Fig 8 regenerated — logged 2026-07-10

All figure changes are text/style only — every statistic, bar, and CI is
unchanged. Old-vs-new comparisons in `scratch/figcompare/`.

- **en-GB spelling pass over author prose**: residualized/residualization →
  residualised/-isation (Fig 78 section+captions in fa_factors),
  initialisation, preference optimisation (×2), standardised, summarise,
  well-organised, artefacts, and the sycophancy \emph{apologise} rate.
  Verbatim artifacts deliberately untouched (constitutions, judge prompts,
  questionnaire items, code identifiers, LaTeX labels, "Direct Preference
  Optimization" as a proper name).
- **Figure text properly capitalised** (titles / y-labels / ticks / legends):
  Fig 7 + Fig 61 (WildJailbreak Harmful/Benign, Harmful Rate, Noncompliance
  Rate, Baseline/Act. Capping/Control ticks), Fig 78 (Scenario-Residualisation,
  z→s throughout), and the within-model validation figure (fig_4_2_2:
  "Cronbach's α per Factor…", "Split-Half Median |φ| per Factor (100
  Iterations)", Good/Fair legend entries).
- **Fig 8 (apologise/CoCoNot) regenerated and swapped in**: Apologise
  spelling, capitalised labels, Baseline/Control ticks, legend removed
  (bars identified by x-ticks), serif→sans font and normal-weight value
  labels to match the rest of the paper. supervised.tex now includes the
  script-generated `fig_apologize_coconot.pdf`; the frozen overleaf-era
  `fig_8_apologize_coconot.pdf` (which the script could not reproduce) is
  deleted; MANIFEST updated.
- **Serif font holdouts normalised**: `main_o_n_soup_heatmaps.py` (Fig 1
  O⊕N panels) and `paper_appendix_amp_sup_heatmaps.py` (5-trait appendix
  grid) switched from Times-serif to default sans — all paper figures now
  share one font family.

## Frustration-figure labels + CoCoNot wording — logged 2026-07-10

Post-v1 fixes, staged for the next revision:

- **Fig 5 (frustration per-turn) legend/label capitalisation**: legend entries
  `BASE`/`CONTROL` renamed to `Baseline`/`Control` (matching the caption and
  the Fig 7 legend style), and panel titles / y-axis labels title-cased
  ("Per-Turn Mean Frustration", "Mean Frustration (Judge 0–10)", "% High
  Frustration (Score ≥ 5)"). Changed in
  `scripts/visualisations/plot_frustration_four_way.py` (labels + docstring +
  `base_vs_nminus` subset filter); PDF + PNG regenerated from the same
  monorepo data — curves and statistics unchanged.
- **CoCoNot sentence in §3**: "comply with roughly 20 points more
  should-decline prompts" → "roughly 20% more" (the quantities 0.33/0.35 vs.
  0.14 are proportions).

---

## ★ arXiv v1 SUBMITTED — 2026-07-08

First arXiv submission. The submitted version corresponds to the repo state
at commit `949d0504` (`anton/abstract_tweak`), i.e. everything logged below
this marker, including the eight proofread fixes from the v1 proof and the
abstract tweak. Entries above this marker are post-submission changes for
the next arXiv revision.

---

## Unsupervised figures regenerated with correct names; FA data fully on monorepo — logged 2026-07-08

- **Figs 4_2_1--4_2_6 regenerated end-to-end from the monorepo** (resolves the
  raw-slugs Open issue): model names now Llama-3.1-8B-Instruct /
  Qwen2.5-7B-Instruct and factor labels use the paper's TIDE display names
  (new `PAPER_FACTOR_DISPLAY_NAMES` routing in `_axis_labels_for`). All
  statistics identical to the committed figures; OLD/NEW comparisons in
  `scratch/figure_regen_v7pf3/`. Full pipeline verified: FA + split-half +
  predictivity + variance-decomp + residualized + lora-shifts all PASS.
- **Unblocked by two script fixes**: `LORA_VALIDATION_HF_REPO` repointed from
  the stale `persona-shattering-lasr/monorepo` (a separate leftover repo, not
  a redirect — archiving it is recommended) to `persona-cartography/monorepo`
  (the `_prefix1000` validation runs were there all along), and the B-rollout
  cache now auto-hydrates from `unsupervised/runs/` via `ensure_rollout_dir()`.
- **fig_4_2_2 layout fixed**: colliding per-bar factor-name labels removed;
  x-axis kept as neutral F0--F3 (deliberately not the TIDE names — each
  model's factors are its own variance-ordered solution and the cross-model
  correspondence is permuted; see the phi-heatmap).
- **All bibliography entries now carry links** (arXiv URLs / DOIs added to
  the stragglers, e.g. ibrahim/saucier/rolland/marsh); a few unused entries
  dropped.
- **Remaining FA data copied to the monorepo**: the rollouts run (0.86 GB,
  uploaded by colleague) plus the two other questionnaire runs
  (`…trait_ocean_natural_v1…pf2-tmv2`, `…q_v5-likert…`) — the old
  `psychometric-fa-runs` repo is no longer needed by any paper figure.

## arXiv proofread fixes — logged 2026-07-08

- **Eight fixes from the arXiv v1 proof**: "as these model" -> "as this
  model"; missing "we" in the teacher-comparison conclusion; Fig 6 caption
  ":." -> "."; Fig 7 caption stray ")"; period after the "Preference pairs
  for DPO training" run-in heading; {Kimi Team}/{Gemma Team} braced so
  citations render "Kimi Team et al." not "Team et al."; "and et al." ->
  "and others" in gemma2b-it; stale "200-item instrument" -> "custom
  forced-choice questionnaire" (the v7 instrument is 72 items).

## Comment cleanup, personas.tex removed, bib normalisation — logged 2026-07-08

- **Dead commented-out blocks deleted** across sections and main.tex (35
  blocks: old draft paragraphs, superseded \david{}/\sid{} alternates, a
  commented calibration table, stale \todo blocks). Kept: figure provenance
  comments, caption-verify TODO markers, structural notes, and the
  commented checklist/trait_metrics inputs. PDF text verified unchanged
  except intended edits.
- **sections/personas.tex deleted** (file was 100% commented out) and its
  no-op \input removed from main.tex.
- **Bib entries normalised** (by hand): `gemma2b-it` now cites the official
  Gemma model-card citation (Kaggle DOI) instead of a bare HF URL;
  `chen2025persona` and others converted to arXiv @misc form;
  `li2026modelspec` gains missing author Nevan Wichers; dead
  `alternative_training` input line removed.

## FA data on monorepo, unsupervised/runs reorg, reviewer-feedback fixes — logged 2026-07-08

- **v7pf3 questionnaire data now on the monorepo**: colleague's upload to
  `psychometric-fa-runs` copied to `persona-cartography/monorepo` under
  `unsupervised/runs/` (90+93 files, ~246 MB, verified); the monorepo's
  top-level `runs/` (eval4factor dirs) moved into `unsupervised/runs/` too.
  Code repointed: `HF_RUNS_PREFIX` constant in `src_dev/psychometric`,
  `analysis_for_paper.v2.py` hydrates from the monorepo; `io.py` subtree
  hydration no longer lists the whole repo (was minutes on the monorepo).
  Six MANIFEST rows + six `fa_factors.tex` `% Data:` pointers updated.
- **Colleague-review fixes applied** (second batch; first batch of eight
  typo/precision fixes landed 2026-07-08 by hand): Cronbach's α described as
  within-factor internal consistency (was "agreement across factors");
  "LoRA strictly dominates activation capping" scoped to the induction task
  with a cross-ref to the capping-regime note; PCA dimensionality claim
  qualified by the 11-datapoint rank ceiling; "it's due to randomness" ->
  "it is" (teacher appendix).
- **Checklist Q5 justification rewritten** for full availability: now points
  at the public GitHub repo and HF monorepo instead of the contradictory
  "could not be provided" text (checklist itself still commented out of the
  build).

## Related-work/citation refinements, coh/ext defined — logged 2026-07-07

- **Related work refined**: task-arithmetic paragraph reworded; Sun et al.
  described as SFT-trained full-model weight deltas; closing paragraph now
  names the constitution-guided open-character-training pipeline.
- **Citations**: sycophancy/compliance claims re-pointed
  (`ibrahim2026warm` joins sycophancy; compliance now cites
  `brahman2024coconot`); `feng2026persona` upgraded to its ICLR 2026
  inproceedings entry (verified on OpenReview); `zou2023representation`
  year corrected to 2023.
- **"coh"/"ext" abbreviations defined** in the temperature-comparison table
  caption; the one inline "coh" spelled out as "coherence score".
- **Error-bar statements added to the two main-body figures that lacked
  them**: Fig 2 (banner) right panel — 95% paired bootstrap CIs (verified
  against `main_fig1_banner.py`); Fig 3b (MMLU compact) — 95% Wilson CIs
  (repo convention; see Open issue on its generator). Audit found all other
  error-bar-bearing figures already state 95% CI + method in caption or
  surrounding prose.
- **PsychAdapter size-justification reworded**: "allows for on policy RL"
  -> capability requirement of the on-policy self-reflection/self-interaction
  pipeline stages.
- **PsychAdapter figure error bars restored to 95% BCa bootstrap CIs**
  (per repo CI policy for continuous judge scores; shared
  `_interval_ci_from_bootstrap` helper) — supersedes the 07-06 std-caption
  note below. Also `etoolbox` now loaded explicitly (build fix). [Mariia]

## PsychAdapter n240 figure, base/baseline sweep, related-work polish — logged 2026-07-06

- **PsychAdapter cross-latent figure upgraded to the n240 run** (rebased onto
  main to pick up the eval + plot script): the appendix now includes
  `fig_psychadapter_n240_cross_latent.pdf` instead of the old
  `n150_TRAIT_cross_latent.png`; caption corrected (all 240 prompts; error bars
  are one standard deviation, not bootstrap CI; "bootsrap" typo gone).
  Reproduction verified from HF data (pixel-identical); script docstrings
  standardised to Gemma-2B-IT / Qwen3-235B-A22B; MANIFEST row marked referenced.
- **"base" -> "baseline" sweep** wherever "base" meant the unmodified model
  (frustration paragraph/caption, sycophancy control, activation-capping and
  DPO-methods captions, ocean-results residuals, induction appendix). "Base"
  kept only for the non-instruct-model and LoRA-base senses; opaque "adapters
  trained on the same base" reworded.
- **Related work polished** after new Bhandari/Sun additions: Sun et al.
  described concretely (full-model weight deltas needing sparsification-based
  merging), headers normalised, hyphenation/grammar fixes.
- **DeepSeek audit**: confirmed V3 (WildJailbreak + calibration judge) and V3.2
  (teacher ablation) are distinct models correctly cited; missing V3 citation
  added at first mention in the WJ-breakdown appendix.

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
