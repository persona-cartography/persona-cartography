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

## Logged Changes

- "how I exist" >  "how I am" in LoRA Training Methods appendix
- "see \Cref{sec:appendix-cross-model} for other models" -> "see \Cref{sec:appendix-cross-model} for a comparison to other models"
- Training: named the default distillation teacher and added a pointer to the new teacher-comparison appendix — "a teacher model generates paired responses" -> "a strong teacher model (GLM-4.5-Air~\citep{zai2025glm45} unless stated otherwise, see \Cref{sec:appendix-teacher-ablation} for a comparison with a different teacher) generates paired responses"


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
