"""Section-4.2 factor analysis — v2 pipeline (v7 forced-choice + prefill v3).

This is the v2 of the paper's Section-4.2 FA pipeline. The v1 (which
operated on the combined ``v5`` Likert + ``trait_ocean_natural_v1``
trait_mcq matrix) is preserved alongside as ``analysis_for_paper.v1.py``.

What changed in v2:

    • Input data is the **v7 forced-choice questionnaire** (72 items × 18
      v5-axes, FC-pair encoding), administered with **prefill v3 (pf3)**
      that fixes mass leakage on Qwen2.5 and is a cleanup for Llama. Both
      Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct (administered on the
      same B rollouts via cross-model questionnaire) are loaded from
      per-questionnaire ``runs/...`` subtrees on
      ``persona-shattering-lasr/psychometric-fa-runs``.

    • Single block (``fc_pair``). Per-block decomposition (combined /
      likert / trait_mcq) collapses to a single ``combined`` subset for
      cross-model congruence.

    • k is configurable per-run via the ``--k`` CLI argument. The two
      headline variants we report are k=4 (interpretable, paper main body)
      and k=11 (Horn's parallel analysis recommendation, paper appendix).
      Each --k value writes to a suffixed ``OUTPUT_ROOT_k{k}/`` and
      ``LABELS_REPO_DIR_k{k}/`` so the two runs don't collide.

    • The model set is configurable per-run via ``--models`` (selecting
      from ``MODEL_REGISTRY`` — same pattern as the supervised pipeline's
      per-model training configs). Default is the paper's Llama+Qwen
      pair; e.g. ``--models gemma-3-27b --k 5`` runs the single-model FA
      path for Gemma into ``..._gemma_k5`` output/labels trees, skipping
      the Llama-anchored LoRA-shift step (cross-model congruence skips
      itself when fewer than two models are fit). Adding a model is a
      registry entry, not a wrapper script.

    • Items have ``dimension ∈ {18 v5-axes}`` rather than OCEAN traits.
      Trait-alignment heatmaps now align factors to the 18 v5-axes; an
      OCEAN-style mapping can be layered in downstream if useful.

    • Paper figures keep the same canonical filenames (so v1's figures
      are *replaced*, not duplicated). Only the run flagged as the
      headline (``--emit-paper-figures``) writes to ``paper/figures/``.

    • Headline rotation remains oblimin / principal. Varimax is an
      appendix robustness check (not yet populated).

Analysis steps in this script are independent of the main pipeline's
stage numbering (``scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py``).
They are ordered analysis actions, not orchestrator stages.

Current state of the script (grows iteratively):

    [done] load_model_data          — Hydrate combined-dir questionnaire from
                                      HF (or reuse the local scratch copy)
                                      and run standard preprocessing.
    [done] run_n_factors_suggest    — Horn's parallel analysis, MAP, EKC,
                                      acceleration, Kaiser, optimal coords,
                                      scree elbow. Saves scree + comparison
                                      plots + summary JSON.
    [done] fit_factor_analysis      — Fit oblimin FA at the configured k per
                                      model. Saves npz + json sidecar, per-
                                      column item_labels.json, top-30 items
                                      text summary, and the full trait-
                                      alignment (CSVs + heatmap PDFs). Also
                                      copies the questionnaire triplet next
                                      to the fits so the /label-fa-factors
                                      skill resolves cleanly.
    [done] export_html_browser      — Write an interactive factor_extremes
                                      HTML viewer per model with top/bottom-
                                      N rollout conversations per factor,
                                      picking up LLM labels from the skill's
                                      output dir when they exist.
    [done] run_within_model_validation — Cronbach's α per factor (over items
                                      with |loading| ≥ threshold, sign-
                                      oriented) + split-half congruence
                                      (Tucker's |φ| across N random
                                      half-splits). Writes
                                      ``validation/cronbach_alpha.json`` and
                                      ``validation/split_half_congruence.{json,png}``
                                      per model.
    [done] run_cross_model_congruence — Llama ↔ Qwen Tucker's |φ| across
                                      combined + per-block (likert,
                                      trait_mcq) shared-item subsets.
                                      Hungarian-matches anchor factors to
                                      target factors (anchor = Llama k=4,
                                      target = Qwen k=5, one unmatched
                                      Qwen factor reported separately).
                                      Writes ``cross_model/{pair}/{subset}/
                                      {phi_matrix.npy, phi_heatmap.{png,pdf},
                                      report.json}`` plus a per-pair
                                      ``summary.json``.

    [done] run_predictivity_cv      — Bi-cross-validation (Owen & Perry 2009):
                                      persona × item holdout, predicts
                                      held-out responses from factor scores,
                                      compares R² against item_mean /
                                      persona_shuffle / k−1 baselines.
    [done] paper figures             — Scree (Llama), within-model α + |φ|
                                      grouped bars, cross-model Tucker's |φ|
                                      heatmap written to paper/figures/
                                      unsupervised/. Only emitted when the
                                      run is invoked with
                                      ``--emit-paper-figures``.

    [todo] varimax robustness pass for the appendix.

Paper figures this script emits (relative to ``paper/figures/``):

    - unsupervised/fig_4_2_1_scree_llama.pdf
      Section 4.2 Horn's parallel-analysis scree on the Llama fit.
    - unsupervised/fig_4_2_2_within_model_validation.pdf
      Section 4.2 grouped bars for Cronbach's α + split-half median |φ|
      per factor, both models.
    - unsupervised/fig_4_2_3_cross_model_phi.pdf
      Section 4.2 Llama↔Qwen Tucker's |φ| heatmap (combined shared items).

See ``paper/CLAUDE.md`` → "Code ↔ Paper Pointers" for the convention.
"""

from __future__ import annotations

# Module-level list mirroring the docstring above — keeps the list available
# without re-parsing the docstring. Paths relative to ``paper/figures/``.
PAPER_FIGURES: list[str] = [
    "unsupervised/fig_4_2_1_scree_llama.pdf",
    "unsupervised/fig_4_2_2_within_model_validation.pdf",
    "unsupervised/fig_4_2_3_cross_model_phi.pdf",
    "unsupervised/fig_4_2_4_variance_decomp.pdf",
    "unsupervised/fig_4_2_5_residualized.pdf",
    "unsupervised/fig_4_2_6_lora_shifts.pdf",
    "unsupervised/fig_4_2_6b_lora_shifts_middling.pdf",
    "unsupervised/fig_4_2_6c_lora_shifts_headroom.pdf",
    "unsupervised/fig_4_2_6_lora_initiative_bars.pdf",
]

# ── Seeds (set before any stochastic imports) ────────────────────────────────
import random

import numpy as np

SEED = 436
random.seed(SEED)
np.random.seed(SEED)

try:
    import torch  # seeded for any downstream torch-RNG usage
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
except ImportError:
    pass

# ── Standard library ─────────────────────────────────────────────────────────
import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

# ── Third-party ──────────────────────────────────────────────────────────────
from dotenv import load_dotenv

load_dotenv()

# ── src_dev imports ──────────────────────────────────────────────────────────
from src_dev.factor_analysis import (
    load_factor_analysis,
    parallel_analysis,
    persona_item_cv,
    plot_n_factors_comparison,
    run_factor_analysis,
    save_factor_analysis,
    split_half_congruence,
    suggest_n_factors,
)
from src_dev.factor_analysis.reliability import classify_alpha, cronbach_alpha
from src_dev.factor_analysis.trait_alignment import (
    compute_factor_trait_alignment,
    plot_all_alignment,
    save_alignment,
)
from src_dev.psychometric.tucker_congruence import (
    align_factors,
    classify_phi,
    tucker_phi_matrix,
)
from src_dev.psychometric.factor_extremes_html import export_factor_extremes_html
from src_dev.psychometric.lora_factor_shifts import (
    LoraValidation,
    build_shift_matrix,
    load_lora_factor_shifts,
    plot_factor_shift_barchart,
    plot_factor_shift_heatmap,
)
from src_dev.psychometric.preprocessing import preprocess_response_matrix
from src_dev.psychometric.variance_decomp import (
    build_archetype_scenario_lookup,
    run_variance_decomposition,
)
from src_dev.factor_analysis.preprocessing import residualize as resid_primitive
from src_dev.unsupervised_runs.io import hydrate_dataset_subtree
from src_dev.visualisations import PAPER_FIGURES_DIR


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═════════════════════════════════════════════════════════════════════════════

# HuggingFace dataset repo holding the Stage-2 questionnaire artifacts and
# the combined v5+trait_ocean_natural_v1 matrices.
HF_REPO_ID = "persona-shattering-lasr/psychometric-fa-runs"

# Where this script writes its outputs. v2 is suffixed by `--k` in main()
# so the k=4 (paper main) and k=11 (paper appendix) runs don't collide. This
# default is the unsuffixed sentinel — it's never written to directly.
OUTPUT_ROOT = Path("scratch/psychometric_fa_paper_v7pf3")

# ── Block filter ────────────────────────────────────────────────────────────
# When set (via ``--block-filter`` CLI arg in main()), restrict the
# questionnaire to items whose ``block`` matches. Used to run a parallel
# analysis on just one modality (e.g. ``likert`` — v5 Likert only,
# dropping the trait_mcq block) without touching the main paper analysis.
# OUTPUT_ROOT and LABELS_REPO_DIR get an ``_<filter>`` suffix so the
# filtered run writes to a separate tree and doesn't collide.
BLOCK_FILTER: str | None = None


# v7-fc-pair-pf3 single-questionnaire ``runs/...`` subtree on HF. Unlike
# v1 (which loads a pre-combined ``combined/...`` dir), v2 loads each
# model's per-questionnaire run directly — there's only one questionnaire
# in v2 (the v7 forced-choice administration).
@dataclass(frozen=True)
class ModelRun:
    """One v7-fc-pair-pf3 run we want to analyse.

    ``slug`` is the short identifier used for output dirs. ``hf_subdir`` is
    the path under HF_REPO_ID that holds this model's questionnaire run
    (the ``runs/questionnaire-...`` dir on HF). ``version_tag`` is stamped
    onto each item dict's ``version`` field for downstream consumers (the
    ``/label-fa-factors`` skill resolves raw questionnaire JSONs by it).
    ``local_source_dir`` lets us prefer an already-hydrated local copy
    when one exists, avoiding a fresh HF download.
    """
    slug: str
    label: str                 # pretty name for plots + provenance
    hf_subdir: str
    version_tag: str
    local_source_dir: Path | None = None
    # Short token used to suffix OUTPUT_ROOT / LABELS_REPO_DIR when a
    # non-default ``--models`` selection is run (e.g. "gemma" →
    # ``..._gemma_k5``), keeping non-paper runs in separate trees.
    tag: str = ""


_V7PF3_LLAMA_RUN: str = (
    "runs/questionnaire-rollouts-llama318binstruct-t1.0-15t-2500p-"
    "seed436-scenarios_v2-uprompt_v6-q_v7_fc_pair-fc_pair-direct-lp20-p2-pf3"
)
_V7PF3_QWEN_RUN: str = (
    "runs/questionnaire-rollouts-llama318binstruct-t1.0-15t-2500p-"
    "seed436-scenarios_v2-uprompt_v6-q_v7_fc_pair-fc_pair-direct-lp20-p2-pf3"
    "-qm_qwen257binstruct"
)
_V7PF3_GEMMA_RUN: str = (
    "runs/questionnaire-rollouts-gemma327bit-t1.0-15t-2500p-"
    "seed436-scenarios_v2-uprompt_v6-q_v7_fc_pair-fc_pair-direct-lp20-p2-pf3"
)


# Registry of analysable v7-pf3 runs, keyed by slug. ``--models`` selects
# from here (same pattern as the supervised pipeline's per-model training
# configs) — adding a model is a registry entry, not a wrapper script.
MODEL_REGISTRY: dict[str, ModelRun] = {
    "llama-3.1-8b": ModelRun(
        slug="llama-3.1-8b",
        label="Llama-3.1-8B-Instruct",
        hf_subdir=_V7PF3_LLAMA_RUN,
        version_tag="v7_fc_pair",
        # Existing pf3 scratches from the rollout pipeline. Either k=4 or
        # k=11 dir works — both contain the same hydrated questionnaire/
        # tree (the difference is in factor_analysis/ subdirs only).
        local_source_dir=Path(
            "scratch/psychometric_fa.pf3-k4/"
            "questionnaire-rollouts-llama318binstruct-t1.0-15t-2500p-"
            "seed436-scenarios_v2-uprompt_v6-q_v7_fc_pair-fc_pair-direct-"
            "lp20-p2-pf3"
        ),
        tag="llama",
    ),
    "qwen2.5-7b": ModelRun(
        slug="qwen2.5-7b",
        label="Qwen2.5-7B-Instruct",
        hf_subdir=_V7PF3_QWEN_RUN,
        version_tag="v7_fc_pair",
        local_source_dir=None,  # not in local scratch yet — hydrate from HF
        tag="qwen",
    ),
    "gemma-3-27b": ModelRun(
        slug="gemma-3-27b",
        label="Gemma-3-27B-Instruct",
        hf_subdir=_V7PF3_GEMMA_RUN,
        version_tag="v7_fc_pair",
        # The first gemma run hydrated the questionnaire triplet here;
        # _questionnaire_dir_for falls back to HF if it's absent.
        local_source_dir=Path(
            "scratch/psychometric_fa_paper_v7pf3_gemma_k4/gemma-3-27b/hydrated"
        ),
        tag="gemma",
    ),
}

# The paper's headline pair. ``--models`` defaults to this selection; only
# this selection may emit paper figures, and only it writes to the
# unsuffixed OUTPUT_ROOT / LABELS_REPO_DIR trees.
DEFAULT_MODELS: tuple[str, ...] = ("llama-3.1-8b", "qwen2.5-7b")

# Resolved model selection for this run. Module-level default is the paper
# pair; main() overwrites it from ``--models``.
MODELS: list[ModelRun] = [MODEL_REGISTRY[s] for s in DEFAULT_MODELS]


# ── Preprocessing ────────────────────────────────────────────────────────────
# Same defaults as the main pipeline. Per-block relative variance floor of
# 0.1 drops items that carry no information relative to their block's median;
# no high-variance persona drop; no residualization (the B rollout preset has
# a single input_group_id per row, so residualization is a no-op).
MIN_ITEM_VARIANCE: float = 0.1
HIGH_VARIANCE_PERSONA_DROP_PCT: float = 0.0
DO_RESIDUALIZE: bool = False

# ── n-factors suggestion ─────────────────────────────────────────────────────
# Full suite minus CV (slowest, quadratic in k and in n_folds — we can add
# it back once the cheaper methods have triangulated a range). OC / scree
# elbow are included because Horn's alone tends to over-retain on wide
# Likert matrices and it's useful to have multiple scree-shape criteria for
# comparison.
N_FACTORS_METHODS: tuple[str, ...] = (
    "parallel",            # Horn's permutation-null parallel analysis (default)
    "map",                 # Velicer's MAP (conservative)
    "ekc",                 # Empirical Kaiser Criterion
    "acceleration",        # Local 2nd-difference scree elbow
    "kaiser",              # Kaiser–Guttman (reference only; over-retains)
    "optimal_coordinates", # Raîche 2013 non-graphical Cattell scree
    "scree_elbow",         # Perpendicular-distance scree elbow
    # "cv_reconstruction", # enable when we want cross-validated NLL
)
N_FACTORS_K_MAX: int = 25              # hard cap for MAP / OC / scree / acceleration
N_FACTORS_PARALLEL_ITERATIONS: int = 200


# ── FA fit ──────────────────────────────────────────────────────────────────
# Per-model chosen k, picked from the scree-elbow (perpendicular distance to
# first-last chord) recommendation produced by `run_n_factors_suggest`. We
# use per-model k rather than a common k because the scree elbow sits at a
# model-specific knee; the cross-model Tucker's step later matches on
# min(k_a, k_b) and flags the extra factor on the larger side as unmatched.
# v2: k is set via the ``--k`` CLI argument (single common k across both
# models so the cross-model Tucker's φ heatmap is square). The ``--k``
# value is promoted into this dict in main() — both slugs get the same k.
# Headline variants: k=4 (paper main), k=11 (paper appendix; Horn's).
FA_K_BY_MODEL: dict[str, int] = {}

# Headline rotation for the main-body analysis. Varimax + logit-encoding
# passes are deferred to the appendix robustness sweep; when we add them,
# we'll re-call `fit_factor_analysis` with the alternative config rather
# than re-fit everything.
FA_ROTATION: str = "oblimin"
FA_METHOD: str = "principal"

# How many top-loading items to write per factor in the text summary that
# feeds the manual labelling session. 30 is enough that a labeller has
# multiple fall-backs when the top few are ambiguous, but still fits
# comfortably on one screen per factor.
TOP_LOADING_ITEMS_FOR_LABELLING: int = 30

# Trait ordering for the alignment tables/plots. v7 items carry one of
# the 18 v5-axes in their ``dimension`` field — these are NOT OCEAN
# traits. We use the alphabetised set of axes; an item with an unexpected
# dimension sorts into the trailing empty-trait slot.
TRAIT_ORDER: list[str] = [
    "assertiveness_independence",
    "autonomy_vs_protection",
    "communication_format",
    "correction_handling",
    "decisiveness",
    "depth_vs_brevity",
    "epistemic_style",
    "formality",
    "humor_playfulness",
    "instruction_compliance",
    "metacognitive_transparency",
    "pedagogical_orientation",
    "pragmatism_vs_idealism",
    "proactivity",
    "safety_posture",
    "self_model",
    "speculation_openness",
    "warmth_vs_directness",
    "",
]


# ── Paper figure output ─────────────────────────────────────────────────────
# Section 4.2 ("Applying Traditional Psychometrics to LLM Personas") takes
# the Llama scree as its headline parallel-analysis figure; the Qwen scree
# would go in the appendix (not emitted yet — add here when we want it).
# Paths are resolved against `PAPER_FIGURES_DIR` (paper/figures/) via the
# helper in `src_dev.visualisations`. Per paper/CLAUDE.md naming convention
# `fig_<section>_<short_name>.<ext>`.
#
# Leaving a slug out of this dict skips the paper-figure write for that
# model. Only PDF (for vector output) — raster PNG always lands alongside
# in the scratch dir.
# Paper figure write is gated on ``--emit-paper-figures`` (see main()) so
# only the headline run replaces v1's figures. The dict still names the
# canonical paths each slug would write to when the gate is open.
PAPER_SCREE_FIGURES: dict[str, str] = {
    "llama-3.1-8b": "unsupervised/fig_4_2_1_scree_llama.pdf",
}

# When False, the run never touches paper/figures/. When True (set via
# CLI), the scree, within-model validation, and cross-model heatmap
# figures are written to their canonical paper/figures/unsupervised/
# paths. v2 keeps v1's filenames so v1 is replaced, not duplicated.
EMIT_PAPER_FIGURES: bool = False


# ── HTML factor browser ─────────────────────────────────────────────────────
# The factor-extremes HTML viewer pulls rollout conversations from a
# `exports/conversation_training.jsonl` inside the rollout dir. Both models
# share the same underlying B rollout cache (only the questionnaire model
# differs), so a single rollout dir feeds both models' HTMLs.
ROLLOUT_DIR_FOR_HTML: Path = Path(
    "scratch/psychometric_fa/"
    "rollouts-llama318binstruct-t1.0-15t-2500p-seed436-scenarios_v2-uprompt_v6"
)

# Same rollout cache, used to look up each persona's interviewer-archetype
# and scenario-id for the variance-decomposition step. Re-used by
# ``run_variance_decomp`` per model — both Llama and Qwen administer over
# the same Llama-generated B rollouts, so the lookup is identical.
ROLLOUT_DIR_FOR_VARIANCE: Path = ROLLOUT_DIR_FOR_HTML
SCENARIOS_FILE: Path = Path("datasets/scenarios/v2.json")

# ── LoRA factor-shift validation ────────────────────────────────────────────
# Per-LoRA factor-shift summaries live on the shared monorepo. The
# ``run_lora_factor_shifts`` stage hydrates each entry's summary (and
# scores npz, used for downstream per-persona analyses) on demand.
#
# Layout: hf_subdir is the dir containing ``<label>_summary.json`` and
# ``<label>_scores.npz``. ``factor`` is the canonical factor name the
# LoRA targets (used to outline the diagonal cell on the heatmap);
# ``direction`` is "+" for amplifier / "-" for suppressor.
LORA_VALIDATION_HF_REPO: str = "persona-shattering-lasr/monorepo"
LORA_VALIDATION_LOCAL_ROOT: Path = Path(
    "scratch/factor_inspect_v7_pf3/validate_results_remote"
)
# When True (default), each entry probes for a ``<label>_prefix1000``
# sibling on HF and uses it instead of the default 200-persona run when
# present. This means the figure auto-upgrades as larger validation
# runs land without an edit here.
LORA_VALIDATION_PREFER_LARGE_N: bool = True

# Paper-display factor names. The internal short names (Initiative,
# Warmth, Pedagogy, Hedging) survive in code, data, and the HF monorepo
# paths — see scripts_dev/unsupervised_embeddings/FACTOR_NAMING.md for
# the rename rationale (acronym TIDE: Tone, Initiative, Didacticism,
# Epistemic Caution). When a paper figure or caption needs a factor
# name, look it up here so future renames are a single-source change.
PAPER_FACTOR_DISPLAY_NAMES: dict[str, str] = {
    "Initiative": "Initiative",
    "Warmth": "Tone",
    "Pedagogy": "Didacticism",
    "Hedging": "Epistemic Caution",
}

# Toggles for the paper's main bar chart.
# - USE_INITIATIVE_LORA_V2: swap the v1 paired-DPO Initiative LoRAs for the
#   newer v2 paired-DPO retraining. Both versions live on the monorepo.
# - INCLUDE_CONTROL_LORA: append the OCEAN-definition control LoRA (a
#   negative-control adapter trained against generic prompt definitions
#   rather than a specific factor's contrastive constitution). Useful as a
#   "does any LoRA shift these factors?" baseline.
USE_INITIATIVE_LORA_V2: bool = True
INCLUDE_CONTROL_LORA: bool = True

_INIT_AMP_V1 = LoraValidation(
    label="initiative_amp", factor="Initiative", direction="+",
    hf_subdir=(
        "fine_tuning/llama-3.1-8b-it/unsupervised/initiative/amplifier/"
        "vunsup_k4_v7_pf3_paired_dpo/evals/factor_validate/initiative_amp"
    ),
)
_INIT_SUP_V1 = LoraValidation(
    label="initiative_sup", factor="Initiative", direction="-",
    hf_subdir=(
        "fine_tuning/llama-3.1-8b-it/unsupervised/initiative/suppressor/"
        "vunsup_k4_v7_pf3_paired_dpo/evals/factor_validate/initiative_sup"
    ),
)
_INIT_AMP_V2 = LoraValidation(
    label="initiative_amp_v2", factor="Initiative", direction="+",
    hf_subdir=(
        "fine_tuning/llama-3.1-8b-it/unsupervised/initiative/amplifier/"
        "vunsup_k4_v7_pf3_v2_paired_dpo/evals/factor_validate/initiative_amp_v2"
    ),
)
_INIT_SUP_V2 = LoraValidation(
    label="initiative_sup_v2", factor="Initiative", direction="-",
    hf_subdir=(
        "fine_tuning/llama-3.1-8b-it/unsupervised/initiative/suppressor/"
        "vunsup_k4_v7_pf3_v2_paired_dpo/evals/factor_validate/initiative_sup_v2"
    ),
)
_CONTROL_LORA = LoraValidation(
    label="control_ocean_def", factor="Control", direction="+",
    hf_subdir=(
        "fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/"
        "vanton4_paired_dpo_s1vs2/evals/factor_validate/control_ocean_def"
    ),
    display_label="Control",
)

LORA_VALIDATIONS: list[LoraValidation] = [
    _INIT_AMP_V2 if USE_INITIATIVE_LORA_V2 else _INIT_AMP_V1,
    _INIT_SUP_V2 if USE_INITIATIVE_LORA_V2 else _INIT_SUP_V1,
    LoraValidation(
        label="warmth_amp", factor="Warmth", direction="+",
        hf_subdir=(
            "fine_tuning/llama-3.1-8b-it/unsupervised/warmth/amplifier/"
            "vunsup_k4_v7_pf3_paired_dpo/evals/factor_validate/warmth_amp"
        ),
    ),
    LoraValidation(
        label="warmth_sup", factor="Warmth", direction="-",
        hf_subdir=(
            "fine_tuning/llama-3.1-8b-it/unsupervised/warmth/suppressor/"
            "vunsup_k4_v7_pf3_paired_dpo/evals/factor_validate/warmth_sup"
        ),
    ),
    *([_CONTROL_LORA] if INCLUDE_CONTROL_LORA else []),
    # We restrict the LoRA validation reported in the paper to the top
    # two factors by variance explained (Initiative + Warmth). Pedagogy
    # and Hedging LoRAs were not optimised to the same standard; their
    # entries are left commented out so they can be re-enabled once the
    # training is finalised.
    # LoraValidation(
    #     label="pedagogy_amp", factor="Pedagogy", direction="+",
    #     hf_subdir=(
    #         "fine_tuning/llama-3.1-8b-it/unsupervised/pedagogy/amplifier/"
    #         "vunsup_k4_v7_pf3_paired_dpo/evals/factor_validate/pedagogy_amp"
    #     ),
    # ),
    # LoraValidation(
    #     label="pedagogy_sup", factor="Pedagogy", direction="-",
    #     hf_subdir=(
    #         "fine_tuning/llama-3.1-8b-it/unsupervised/pedagogy/suppressor/"
    #         "vunsup_k4_v7_pf3_paired_dpo/evals/factor_validate/pedagogy_sup"
    #     ),
    # ),
]

# Number of high-scoring and low-scoring personas shown per factor in the
# HTML browser. 10 per pole is what the main pipeline uses and is about as
# much as a human wants to scroll through at once.
HTML_N_PER_POLE: int = 10

# Raw questionnaire JSON paths used to enrich items.json before rendering
# the factor-extremes HTML. The combined items.json doesn't carry MCQ
# options or answer_mapping, so we match by bare item id back to the raw
# source and splice the extra fields in. Likert's `reverse_keyed` already
# rides along on items.json but we still look it up here for uniformity.
RAW_QUESTIONNAIRE_PATHS: dict[str, Path] = {
    "v7_fc_pair": Path(
        "datasets/psychometric_questionnaires/psychometric_questionnaire_v7_fc_pair.json"
    ),
}

# ── Within-model validation ─────────────────────────────────────────────────
# Cronbach's α: for each factor, restrict to items whose |loading| exceeds
# this threshold (the standard psychometric "defining items" subset), sign-
# orient by loading direction, and compute α. 0.4 is the conventional
# "salient loading" threshold (Comrey & Lee 1992); lower thresholds drag
# in weakly-loaded items that don't actually belong to the construct.
CRONBACH_LOADING_THRESHOLD: float = 0.4

# Split-half congruence: N iterations of random half-splits, per-factor
# Tucker's |φ| distribution. ~100 iters is enough to stabilise the
# violin plots without the cost of bootstrap's resample-refit loops.
SPLIT_HALF_N_ITERS: int = 100
SPLIT_HALF_PASS_THRESHOLD_PHI: float = 0.85  # Lorenzo-Seva "fair" threshold


# ── Predictivity cross-validation (bi-cross-validation) ─────────────────────
# Double-loop persona × item holdout — the well-founded generalisation test
# for factor models (Owen & Perry 2009; Bro et al. 2008; Wold 1978).
#
# Protocol (one outer split):
#   1. Persona A = random 80%; persona B = remaining 20%.
#   2. Fit FA on the full column set of persona A → loadings Λ_A.
#   3. For each B-persona, randomly observe m_observed items; compute
#      Thomson factor scores using Λ_A restricted to those m items.
#   4. Predict the remaining ("held-out") items from F̂ · Λ_A.T.
#   5. Collect per-item R² against three null baselines:
#      item_mean / persona_shuffle / k_minus_1.
#
# Per outer split we also compute Tucker's |φ| between Λ_A and the full-
# data Λ (restricted to shared items) — a direct "are the CV-fold factors
# the same factors we labelled?" check.
#
# Defaults: 20 outer persona splits × 5 item resamples = 100 estimates,
# ~enough for stable mean R² with 95% bootstrap CI.
PREDICTIVITY_PERSONA_SPLIT: float = 0.8     # fraction in subset A (train)
PREDICTIVITY_ITEM_OBSERVED_FRAC: float = 0.8  # fraction of items observed
                                              # for B personas; drives m_observed
PREDICTIVITY_N_OUTER_SPLITS: int = 20
PREDICTIVITY_N_TRIALS: int = 5
PREDICTIVITY_BOOTSTRAP_CI: float = 95.0
PREDICTIVITY_SUBSET_STRATEGY: str = "random"  # "random" | "by_factor_balanced"


# ── Cross-model congruence ──────────────────────────────────────────────────
# Which model is the anchor in the Hungarian match. Llama is the paper's
# headline fit (k=4), so we treat its factors as the reference and ask how
# each maps into Qwen's k=5 solution. Under Hungarian matching, all 4
# Llama factors get one partner and 1 of Qwen's 5 is left unmatched —
# that extra factor is reported separately with its single best partner.
CROSS_MODEL_ANCHOR: str = "llama-3.1-8b"

# Blocks to run cross-model congruence on. "combined" uses all shared
# items; the per-block restrictions tell us whether agreement is
# Likert-driven, trait_mcq-driven, or mixed — same decomposition as the
# existing cross_model_sweep.
# v7 is a single-block (fc_pair) questionnaire, so the per-block
# decomposition collapses to one ``combined`` subset.
CROSS_MODEL_BLOCK_SUBSETS: tuple[str, ...] = ("combined",)

# Persisted-labels store (checked into the repo, survives gitignored
# scratch wipes). The /label-fa-factors skill writes into
# ``{output_dir}/labeling/`` under scratch — after a labelling session,
# copy the fresh ``llm_labels_*.json`` over to the matching subpath
# under ``LABELS_REPO_DIR`` and commit so the labels persist. On each
# run, ``export_html_browser`` seeds scratch from this dir so labels
# are applied to the freshly-built HTML even if scratch was cleaned.
#
# Re-running with the same seeds produces identical loadings (the FA
# pipeline: np.random seeded + factor_analyzer's randomized_svd uses
# a hardcoded random_state + oblimin is deterministic), so labels keyed
# by factor_index stay valid across re-runs.
LABELS_REPO_DIR: Path = Path(
    "datasets/psychometric_fa_labels/analysis_for_paper.v2"
)


# ═════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fa_paper")


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1 — HYDRATE + LOAD
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class LoadedData:
    """Per-model raw + preprocessed data bundle."""

    model: ModelRun
    # Raw artifacts (as produced by the combine stage)
    raw_matrix: np.ndarray           # (K, M) pre-preprocess
    raw_metadata: list[dict]         # len K
    raw_items: list[dict]            # len M
    # Preprocessed artifacts (post row+column filtering)
    matrix: np.ndarray               # (K', M')
    metadata: list[dict]             # len K'
    items: list[dict]                # len M'
    # Where outputs for this model go
    output_dir: Path


def _questionnaire_dir_for(model: ModelRun) -> Path:
    """Local path to the v7-pf3 run's parent dir (the dir holding ``questionnaire/``).

    Prefers ``model.local_source_dir`` if its ``questionnaire/response_matrix.npy``
    exists (saves a re-download); otherwise hydrates the run from HF into
    ``OUTPUT_ROOT/<slug>/hydrated/`` and returns that. The returned path
    is the *parent* — its child ``questionnaire/`` carries the canonical
    triplet (response_matrix.npy, metadata.jsonl, items.json).
    """
    if model.local_source_dir is not None:
        existing_q = model.local_source_dir / "questionnaire"
        if (existing_q / "response_matrix.npy").exists():
            return model.local_source_dir
    mirror_parent = OUTPUT_ROOT / model.slug / "hydrated"
    mirror_q = mirror_parent / "questionnaire"
    if not (mirror_q / "response_matrix.npy").exists():
        log.info("[%s] hydrating from HF: %s", model.slug, model.hf_subdir)
        hydrate_dataset_subtree(
            repo_id=HF_REPO_ID,
            path_in_repo=f"{model.hf_subdir.rstrip('/')}/questionnaire",
            local_dir=mirror_q,
            required=True,
        )
    return mirror_parent


def _load_single_questionnaire(
    q_parent: Path,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Load (response_matrix, metadata, items) from a single questionnaire run dir.

    ``q_parent`` is the run dir holding the ``questionnaire/`` subfolder
    (response_matrix.npy + metadata.jsonl + items.json triplet).
    """
    q_dir = q_parent / "questionnaire"
    M = np.load(q_dir / "response_matrix.npy").astype(float)
    meta = [
        json.loads(line)
        for line in (q_dir / "metadata.jsonl").read_text().splitlines()
        if line.strip()
    ]
    items = json.loads((q_dir / "items.json").read_text())
    assert M.shape[0] == len(meta), (M.shape, len(meta), q_dir)
    assert M.shape[1] == len(items), (M.shape, len(items), q_dir)
    return M, meta, items


def _preprocess(
    matrix: np.ndarray,
    metadata: list[dict],
    items: list[dict],
    *,
    output_dir: Path,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Run the standard preprocess pipeline with our configured defaults."""
    variance_export_path = output_dir / "item_variances_ranked.jsonl"
    cleaned, meta, cols, _groups = preprocess_response_matrix(
        matrix,
        metadata,
        items,
        min_item_variance=MIN_ITEM_VARIANCE,
        high_variance_persona_drop_pct=HIGH_VARIANCE_PERSONA_DROP_PCT,
        do_residualize=DO_RESIDUALIZE,
        variance_export_path=variance_export_path,
    )
    return cleaned, meta, cols


def load_model_data(model: ModelRun) -> LoadedData:
    """Hydrate (if needed) and load one model's combined response matrix.

    Column defs are enriched at this point from the raw questionnaire JSONs
    (options + answer_mapping for trait_mcq, reverse_keyed for Likert) so
    every downstream consumer — FA fit, item_labels sidecar, top-30 text
    summary, HTML browser — works off the same rich payload. Notably this
    also means the ``encoding`` field carried on the combined items.json
    survives into the sidecar, avoiding the
    ``_item_labels.json → col_def.get("encoding", "letter_1-4")`` default
    in the /label-fa-factors skill that silently mis-labels trait_mcq
    items as letter-ordinal when the real encoding is ``trait_aligned_0-1``.
    """
    q_parent = _questionnaire_dir_for(model)
    raw_matrix, raw_meta, raw_items = _load_single_questionnaire(q_parent)
    # Stamp the version_tag onto every item dict so downstream resolvers
    # (the /label-fa-factors skill, _enrich_column_defs) can find the raw
    # questionnaire JSON. We mutate copies, not the loaded list directly.
    raw_items = [
        {**it, "version": model.version_tag,
         "questionnaire_version": model.version_tag}
        for it in raw_items
    ]
    log.info(
        "[%s] raw matrix: %d personas × %d items",
        model.slug, raw_matrix.shape[0], raw_matrix.shape[1],
    )

    # Drop personas with any NaN response (some FC rollouts can fail to
    # produce a parseable answer; mirrors khorns / inspect_factor_loadings).
    nan_rows = np.isnan(raw_matrix).any(axis=1)
    if nan_rows.any():
        log.info("[%s] dropping %d rows with NaN", model.slug, int(nan_rows.sum()))
        raw_matrix = raw_matrix[~nan_rows]
        raw_meta = [m for m, n in zip(raw_meta, nan_rows) if not n]

    # Optional block filter — drops every column whose item's `block` does
    # not match `BLOCK_FILTER`. Applied at the raw matrix level, before
    # preprocessing, so downstream steps (variance filter, FA, alignment)
    # see only the filtered block.
    if BLOCK_FILTER is not None:
        keep_mask = np.array(
            [it.get("block") == BLOCK_FILTER for it in raw_items],
            dtype=bool,
        )
        raw_matrix = raw_matrix[:, keep_mask]
        raw_items = [it for it, keep in zip(raw_items, keep_mask) if keep]
        log.info(
            "[%s] block filter %r kept %d / %d columns",
            model.slug, BLOCK_FILTER,
            int(keep_mask.sum()), int(keep_mask.size),
        )

    raw_items = _enrich_column_defs(raw_items)

    out_dir = OUTPUT_ROOT / model.slug
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix, meta, items = _preprocess(
        raw_matrix, raw_meta, raw_items, output_dir=out_dir,
    )
    log.info(
        "[%s] preprocessed matrix: %d personas × %d items",
        model.slug, matrix.shape[0], matrix.shape[1],
    )

    return LoadedData(
        model=model,
        raw_matrix=raw_matrix,
        raw_metadata=raw_meta,
        raw_items=raw_items,
        matrix=matrix,
        metadata=meta,
        items=items,
        output_dir=out_dir,
    )


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2 — N-FACTORS SUGGESTION
# ═════════════════════════════════════════════════════════════════════════════


def run_n_factors_suggest(data: LoadedData) -> dict:
    """Run the full n-factors method suite + save plots + dump summary JSON.

    Output layout:
        {output_dir}/n_factors/
            n_factors_suggest.json        # full per-method dict
            n_factors_suggest.png         # comparison plot across methods
            parallel_analysis.json        # Horn's-only detail
            parallel_analysis.png         # scree plot with null percentile
    """
    out = data.output_dir / "n_factors"
    out.mkdir(parents=True, exist_ok=True)

    log.info("[%s] suggest_n_factors (k_max=%d, parallel_iters=%d)…",
             data.model.slug, N_FACTORS_K_MAX, N_FACTORS_PARALLEL_ITERATIONS)
    result = suggest_n_factors(
        data.matrix,
        methods=N_FACTORS_METHODS,
        k_max=N_FACTORS_K_MAX,
        parallel_n_iterations=N_FACTORS_PARALLEL_ITERATIONS,
        seed=SEED,
        verbose=True,
    )

    # Full result dump.
    (out / "n_factors_suggest.json").write_text(
        json.dumps(_jsonable(result), indent=2)
    )
    log.info("[%s] wrote %s", data.model.slug, out / "n_factors_suggest.json")

    # Comparison plot across all methods.
    plot_n_factors_comparison(
        result, out / "n_factors_suggest.png",
        title_suffix=f" — {data.model.label}",
    )
    log.info("[%s] wrote %s", data.model.slug, out / "n_factors_suggest.png")

    # Horn's-only scree plot (useful for the paper's main figure).
    pa = result.get("parallel")
    if pa is not None:
        (out / "parallel_analysis.json").write_text(
            json.dumps(_jsonable(pa), indent=2)
        )
        extra_paths: list[Path] = []
        # Only write paper figures from the unfiltered (main paper) run —
        # filtered runs are robustness variants and shouldn't overwrite
        # the headline figure.
        paper_rel = (
            PAPER_SCREE_FIGURES.get(data.model.slug)
            if (BLOCK_FILTER is None and EMIT_PAPER_FIGURES) else None
        )
        if paper_rel is not None:
            paper_path = PAPER_FIGURES_DIR / paper_rel
            paper_path.parent.mkdir(parents=True, exist_ok=True)
            extra_paths.append(paper_path)
        _plot_scree(
            pa, out / "parallel_analysis.png",
            title=data.model.label,
            extra_save_paths=extra_paths,
        )
        log.info("[%s] wrote %s", data.model.slug, out / "parallel_analysis.png")
        for p in extra_paths:
            log.info("[%s] wrote %s", data.model.slug, p)

    return result


def _plot_scree(
    pa: dict,
    save_path: Path,
    *,
    title: str,
    extra_save_paths: list[Path] | None = None,
) -> None:
    """Plot real eigenvalues vs Horn's null percentile threshold.

    ``extra_save_paths`` is for parallel writes to the paper's figure tree
    (vector PDF) in addition to the scratch PNG — same figure, different
    file. Format is inferred by matplotlib from each path's suffix.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = np.asarray(pa["real_eigenvalues"], dtype=float)
    thr = np.asarray(pa["random_threshold"], dtype=float)
    n_rec = int(pa["n_recommended"])
    k_max_display = min(30, len(real))
    x = np.arange(1, k_max_display + 1)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, real[:k_max_display], "o-", color="tab:blue", label="Real eigenvalues")
    ax.plot(x, thr[:k_max_display], "s--", color="tab:red",
            label="Horn null, 95th percentile")
    ax.axvline(n_rec + 0.5, color="tab:grey", alpha=0.5, linestyle=":",
               label=f"Horn's k = {n_rec}")
    ax.set_xlabel("Factor index")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Scree plot — {title}")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    for p in extra_save_paths or ():
        fig.savefig(p)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# FIT FACTOR ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════


def _sort_factors_by_variance(fa: dict) -> tuple[dict, np.ndarray]:
    """Re-order factors descending by proportion_variance.

    factor_analyzer's oblimin output has no canonical column order — the
    sequence of factors comes out of the rotation step in whatever order
    the optimiser happened to land in. Standard psychometric practice is
    to present factors sorted by variance explained so that ``F0`` is
    always the largest dimension. We do that here, permuting every
    factor-indexed array consistently, and recompute ``cumulative_variance``
    from the re-ordered ``proportion_variance``.

    Returns the sorted-fa dict and the permutation ``perm`` such that
    ``perm[new_idx] == old_idx`` (use as ``perm[new] = old`` mapping).
    The permutation is also written into ``fa["factor_perm_old_to_new"]``
    (vector of length k where entry old_idx gives the new_idx) for any
    downstream consumer that needs to remap an old-index reference.
    """
    pv = np.asarray(fa["proportion_variance"], dtype=float)
    perm = np.argsort(-pv)              # new_to_old: perm[new] = old
    if np.array_equal(perm, np.arange(pv.size)):
        # Already canonical — no-op (also true on a re-run of an already
        # sorted fit; this keeps the call idempotent).
        old_to_new = np.arange(pv.size)
        out = dict(fa)
        out["factor_perm_old_to_new"] = old_to_new
        return out, perm
    out = dict(fa)
    out["loadings"] = fa["loadings"][:, perm]
    out["scores"] = fa["scores"][:, perm]
    out["ss_loadings"] = np.asarray(fa["ss_loadings"], dtype=float)[perm]
    out["proportion_variance"] = pv[perm]
    out["cumulative_variance"] = np.cumsum(out["proportion_variance"])
    if fa.get("factor_correlation_matrix") is not None:
        phi = np.asarray(fa["factor_correlation_matrix"], dtype=float)
        out["factor_correlation_matrix"] = phi[np.ix_(perm, perm)]
    if fa.get("rotation_matrix") is not None:
        # rotation_matrix has shape (k, k). Standard convention is that
        # columns correspond to factor indices, so column-permute it.
        rm = np.asarray(fa["rotation_matrix"], dtype=float)
        out["rotation_matrix"] = rm[:, perm]
    # old_to_new[old] = new
    old_to_new = np.empty_like(perm)
    old_to_new[perm] = np.arange(perm.size)
    out["factor_perm_old_to_new"] = old_to_new
    log.info(
        "[sort] factors permuted by variance (new←old): %s; new prop_var: %s",
        perm.tolist(),
        ", ".join(f"{v:.3f}" for v in out["proportion_variance"]),
    )
    return out, perm


def _ensure_questionnaire_copy(data: LoadedData) -> Path:
    """Copy the questionnaire triplet into the paper output dir.

    The ``/label-fa-factors`` skill's resolver walks up from a ``fa_*.npz``
    looking for a sibling ``questionnaire/`` dir with ``items.json``. Making
    that sibling exist locally under our paper output tree means the skill
    resolves cleanly without needing to know about the upstream combined
    dir.
    """
    import shutil

    # Source = whatever ``_questionnaire_dir_for`` resolved to (local
    # scratch if available, else the freshly-hydrated mirror).
    src_q = _questionnaire_dir_for(data.model) / "questionnaire"
    dst_q = data.output_dir / "questionnaire"
    dst_q.mkdir(parents=True, exist_ok=True)
    for fname in ("items.json", "metadata.jsonl", "response_matrix.npy"):
        src = src_q / fname
        dst = dst_q / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    return dst_q


def _write_item_labels_json(
    items: list[dict],
    save_path: Path,
) -> None:
    """Per-column metadata aligned with loading rows.

    Extended version of the main pipeline's ``fa_*_item_labels.json`` schema
    — adds ``encoding`` (so the /label-fa-factors skill doesn't default to
    the misleading ``"letter_1-4"`` fallback at ``fa_label_tools.py:548``)
    plus ``options`` + ``answer_mapping`` for trait_mcq columns so the
    skill's describe output can render the full option table without
    having to walk up to find the raw questionnaire file.
    """
    payload: list[dict] = []
    for col in items:
        entry: dict = {
            "col_id": col["col_id"],
            "text": col.get("text", ""),
            "block": col.get("block", ""),
            "dimension": col.get("dimension"),
            "reverse_keyed": col.get("reverse_keyed", False),
        }
        if col.get("encoding"):
            entry["encoding"] = col["encoding"]
        if col.get("options"):
            entry["options"] = col["options"]
        if col.get("answer_mapping"):
            entry["answer_mapping"] = col["answer_mapping"]
        payload.append(entry)
    save_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _top_items_for_factor(
    loadings: np.ndarray,
    items: list[dict],
    factor_index: int,
    n_top: int,
) -> tuple[list[tuple[float, dict]], list[tuple[float, dict]]]:
    """Return top-`n_top` positive and negative-loading items for one factor.

    Each returned entry is ``(signed_loading, item_dict)``. Positive list is
    descending by signed loading; negative list is ascending by signed
    loading (most-negative first). Items are the full column defs.
    """
    col = loadings[:, factor_index]
    order_desc = np.argsort(-col)  # positive end first
    order_asc = np.argsort(col)    # negative end first
    pos: list[tuple[float, dict]] = [
        (float(col[i]), items[i])
        for i in order_desc[:n_top]
        if col[i] > 0
    ]
    neg: list[tuple[float, dict]] = [
        (float(col[i]), items[i])
        for i in order_asc[:n_top]
        if col[i] < 0
    ]
    return pos, neg


def _format_item_line(loading: float, item: dict) -> str:
    block = item.get("block", "") or "?"
    dim = item.get("dimension") or ""
    rev = " (REVERSED)" if item.get("reverse_keyed") else ""
    text = (item.get("text") or "").replace("\n", " ").strip()
    dim_tag = f" [{dim}]" if dim else ""
    enc = item.get("encoding") or ""
    enc_tag = f" enc={enc}" if enc else ""
    header = f"    {loading:+.3f}  <{block}{dim_tag}{enc_tag}>{rev}  {text}"
    if block == "trait_mcq" and item.get("options") and item.get("answer_mapping"):
        # High pole of THIS factor for the item: options with trait-aligned
        # answer_mapping under +loading, or trait-opposite under −loading.
        amap = item["answer_mapping"]
        high_pole_is_trait_aligned = loading >= 0
        lines = [header]
        for opt in item["options"]:
            lbl = str(opt.get("label", ""))
            mv = int(amap.get(lbl, 0))
            trait_aligned = (mv == 1)
            is_high_pole = trait_aligned if high_pole_is_trait_aligned else not trait_aligned
            pole = "HIGH" if is_high_pole else "LOW "
            opt_text = str(opt.get("text", "")).replace("\n", " ").strip()
            lines.append(f"          {lbl}. [{pole} pole]  {opt_text}")
        return "\n".join(lines)
    return header


def _write_top_items_summary(
    fa: dict,
    items: list[dict],
    save_path: Path,
    *,
    n_top: int,
    model_label: str,
    n_factors: int,
    rotation: str,
) -> None:
    """Plain-text per-factor top-loading summary for the labelling session.

    Each factor gets: variance explained, top-n_top positive items, top-n_top
    negative items. For oblique rotations we also print the factor
    correlation matrix at the bottom. Deliberately terminal-friendly — this
    is what a labeller reads while doing the manual pass.
    """
    loadings = fa["loadings"]
    prop_var = fa.get("proportion_variance", [])
    cum_var = fa.get("cumulative_variance", [])
    phi = fa.get("factor_correlation_matrix")

    lines: list[str] = []
    lines.append(f"FA fit — {model_label}")
    lines.append(f"rotation={rotation}  n_factors={n_factors}  "
                 f"n_items={loadings.shape[0]}")
    lines.append("")
    lines.append("Variance explained per factor (ss_loadings fraction):")
    for f in range(n_factors):
        pv = float(prop_var[f]) if len(prop_var) > f else float("nan")
        cv = float(cum_var[f]) if len(cum_var) > f else float("nan")
        lines.append(f"  F{f+1}:  prop={pv:.3f}   cum={cv:.3f}")
    lines.append("")

    for f in range(n_factors):
        lines.append("=" * 78)
        lines.append(f"F{f+1}  (top-{n_top} by signed loading)")
        lines.append("=" * 78)
        pos, neg = _top_items_for_factor(loadings, items, f, n_top)
        lines.append(f"  Positive loadings ({len(pos)} items):")
        for loading, item in pos:
            lines.append(_format_item_line(loading, item))
        lines.append("")
        lines.append(f"  Negative loadings ({len(neg)} items):")
        for loading, item in neg:
            lines.append(_format_item_line(loading, item))
        lines.append("")

    if phi is not None:
        lines.append("=" * 78)
        lines.append("Factor correlation matrix (oblique rotation)")
        lines.append("=" * 78)
        header = "        " + "  ".join(f"{'F'+str(j+1):>7}" for j in range(n_factors))
        lines.append(header)
        for i in range(n_factors):
            row = f"  F{i+1:<2}  " + "  ".join(
                f"{phi[i, j]:>7.3f}" for j in range(n_factors)
            )
            lines.append(row)
        lines.append("")

    save_path.write_text("\n".join(lines))


def _run_trait_alignment(
    fa: dict,
    items: list[dict],
    save_dir: Path,
    *,
    title_prefix: str,
) -> None:
    """OCEAN factor-alignment summary + CSVs + heatmaps.

    Items with ``dimension=""`` sort into the trailing empty-trait column,
    so factors dominated by v5 Likert items (which carry no OCEAN label)
    show up as "winner_trait = ''".
    """
    item_dims = [str(it.get("dimension") or "") for it in items]
    alignment = compute_factor_trait_alignment(
        fa["loadings"],
        item_dims,
        trait_order=TRAIT_ORDER,
        top_k=20,
    )
    save_alignment(alignment, save_dir)
    plot_all_alignment(alignment, save_dir, title_prefix=title_prefix)


@dataclass
class FaFit:
    """Result bundle returned by ``fit_factor_analysis``."""

    model_slug: str
    k: int
    rotation: str
    method: str
    npz_path: Path       # path to fa_{k}_{method}_{rotation}.npz
    item_labels_path: Path
    top_items_path: Path
    alignment_dir: Path
    questionnaire_dir: Path  # sibling that the skill resolves against


def fit_factor_analysis(data: LoadedData, *, k: int) -> FaFit:
    """Fit FA at the configured rotation/method and lay outputs on disk.

    Output layout under ``{output_dir}/factor_analysis/raw/``:

        fa_{k}_{method}_{rotation}.npz
        fa_{k}_{method}_{rotation}.json              (config + array keys)
        fa_{k}_{method}_{rotation}_item_labels.json  (per-column metadata)
        fa_{k}_{method}_{rotation}_top{n}.txt        (labeller's summary)
        fa_{k}_{method}_{rotation}_alignment/        (OCEAN CSVs + heatmaps)

    Also ensures ``{output_dir}/questionnaire/items.json`` exists so the
    ``/label-fa-factors`` skill's resolver finds it as a sibling of
    factor_analysis/.
    """
    q_dir = _ensure_questionnaire_copy(data)

    fa_dir = data.output_dir / "factor_analysis" / "raw"
    fa_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "[%s] fitting FA: k=%d rotation=%s method=%s  (n=%d × p=%d)",
        data.model.slug, k, FA_ROTATION, FA_METHOD,
        data.matrix.shape[0], data.matrix.shape[1],
    )
    fa = run_factor_analysis(
        data.matrix,
        n_factors=k,
        method=FA_METHOD,
        rotation=FA_ROTATION,
    )
    # Canonicalise factor order (descending variance explained) so F0 is
    # always the strongest factor in every downstream artifact.
    fa, _perm = _sort_factors_by_variance(fa)

    base = fa_dir / f"fa_{k}_{FA_METHOD}_{FA_ROTATION}"
    npz_path = save_factor_analysis(
        fa,
        base,
        config={
            "n_factors": k,
            "method": FA_METHOD,
            "rotation": FA_ROTATION,
            "residualized": DO_RESIDUALIZE,
            "n_samples": int(data.matrix.shape[0]),
            "n_cols": int(data.matrix.shape[1]),
            "model_slug": data.model.slug,
            "model_label": data.model.label,
        },
    )

    item_labels_path = Path(str(base) + "_item_labels.json")
    _write_item_labels_json(data.items, item_labels_path)
    log.info("[%s] wrote %s", data.model.slug, item_labels_path)

    top_items_path = Path(
        str(base) + f"_top{TOP_LOADING_ITEMS_FOR_LABELLING}.txt"
    )
    _write_top_items_summary(
        fa, data.items, top_items_path,
        n_top=TOP_LOADING_ITEMS_FOR_LABELLING,
        model_label=data.model.label,
        n_factors=k,
        rotation=FA_ROTATION,
    )
    log.info("[%s] wrote %s", data.model.slug, top_items_path)

    alignment_dir = Path(str(base) + "_alignment")
    _run_trait_alignment(
        fa, data.items, alignment_dir,
        title_prefix=f"{data.model.label}  k={k} {FA_ROTATION}",
    )
    log.info("[%s] wrote %s", data.model.slug, alignment_dir)

    return FaFit(
        model_slug=data.model.slug,
        k=k,
        rotation=FA_ROTATION,
        method=FA_METHOD,
        npz_path=npz_path,
        item_labels_path=item_labels_path,
        top_items_path=top_items_path,
        alignment_dir=alignment_dir,
        questionnaire_dir=q_dir,
    )


# ═════════════════════════════════════════════════════════════════════════════
# WITHIN-MODEL VALIDATION
# ═════════════════════════════════════════════════════════════════════════════


def _cronbach_alpha_per_factor(
    matrix: np.ndarray,
    loadings: np.ndarray,
    items: list[dict],
    *,
    loading_threshold: float,
) -> list[dict]:
    """Compute Cronbach's α for each factor, over items with |loading| ≥ threshold.

    Items are sign-oriented by loading direction so positively- and
    negatively-loading items both contribute positively to the summed score
    — without this, α is meaningless when a factor has mixed-sign loadings.

    Returns one dict per factor with α, the classified α (excellent / good /
    acceptable / questionable / poor / redundant), the number of items used,
    and the per-block item counts so we can tell at a glance whether α is
    being driven by Likert items, trait_mcq items, or a mix.
    """
    n_items, n_factors = loadings.shape
    rows: list[dict] = []
    for f in range(n_factors):
        col = loadings[:, f]
        mask = np.abs(col) >= loading_threshold
        idxs = np.flatnonzero(mask)
        if idxs.size < 2:
            rows.append({
                "factor_index": f,
                "n_items": int(idxs.size),
                "alpha": float("nan"),
                "alpha_class": "n/a",
                "threshold": loading_threshold,
                "item_col_ids": [items[i].get("col_id") for i in idxs.tolist()],
                "block_counts": {},
                "note": "fewer than 2 salient items",
            })
            continue
        subset = matrix[:, idxs]
        signs = np.sign(col[idxs])
        signs[signs == 0] = 1.0  # pathological guard; |loading|≥0.4 excludes zeros
        alpha = cronbach_alpha(subset, loading_signs=signs)
        block_counts: dict[str, int] = {}
        for i in idxs.tolist():
            b = str(items[i].get("block", "?"))
            block_counts[b] = block_counts.get(b, 0) + 1
        rows.append({
            "factor_index": f,
            "n_items": int(idxs.size),
            "alpha": float(alpha),
            "alpha_class": classify_alpha(alpha),
            "threshold": loading_threshold,
            "item_col_ids": [items[i].get("col_id") for i in idxs.tolist()],
            "block_counts": block_counts,
        })
    return rows


def _load_labels_for_slug(model_slug: str, n_factors: int) -> dict[int, str]:
    """Best-effort ``{factor_index -> axis_name}`` from the persisted labels store.

    Picks the newest ``llm_labels_*.json`` under ``LABELS_REPO_DIR/<slug>/
    labeling/`` by mtime and returns the axis_name per factor. Returns
    ``{}`` when no label file exists — callers should fall back to
    ``F{idx}`` placeholders.
    """
    labeling_dir = LABELS_REPO_DIR / model_slug / "labeling"
    if not labeling_dir.is_dir():
        return {}
    candidates = sorted(
        labeling_dir.glob("llm_labels_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict) and "factors" in payload:
            payload = payload["factors"]
        if not isinstance(payload, list):
            continue
        out: dict[int, str] = {}
        for entry in payload:
            fi = entry.get("factor_index")
            ax = entry.get("axis_name")
            if isinstance(fi, int) and 0 <= fi < n_factors and ax:
                out[fi] = str(ax)
        if out:
            return out
    return {}


def run_within_model_validation(
    data: LoadedData, fit: FaFit,
) -> dict:
    """Cronbach's α per factor + split-half congruence, written under validation/.

    Two tests, both loadings-based and within-model-only:

    * Cronbach's α — do the items that cluster onto a factor actually
      predict each other? One α per factor, conventional thresholds in
      ``classify_alpha``. Saved as ``validation/cronbach_alpha.json`` with
      per-factor α + sign-oriented item list + per-block item counts.

    * Split-half congruence — is the factor structure replicable within
      the persona sample? For each of N random half-splits, refit FA on
      each half and compute per-factor Tucker's |φ|. Pass threshold 0.85
      (Lorenzo-Seva "fair similarity"). Saved as
      ``validation/split_half_congruence.{json,png}``.
    """
    out_dir = data.output_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    fa_result = load_factor_analysis(fit.npz_path)
    loadings = fa_result["loadings"]

    log.info("[%s] Cronbach's α per factor (|loading|≥%.2f)…",
             data.model.slug, CRONBACH_LOADING_THRESHOLD)
    alpha_rows = _cronbach_alpha_per_factor(
        data.matrix, loadings, data.items,
        loading_threshold=CRONBACH_LOADING_THRESHOLD,
    )
    alpha_payload = {
        "model_slug": data.model.slug,
        "model_label": data.model.label,
        "n_factors": fit.k,
        "rotation": fit.rotation,
        "loading_threshold": CRONBACH_LOADING_THRESHOLD,
        "per_factor": alpha_rows,
    }
    (out_dir / "cronbach_alpha.json").write_text(
        json.dumps(alpha_payload, indent=2)
    )
    log.info("[%s] wrote %s", data.model.slug, out_dir / "cronbach_alpha.json")

    for row in alpha_rows:
        log.info(
            "  F%d: α=%.3f (%s)  n_items=%d  blocks=%s",
            row["factor_index"], row["alpha"], row["alpha_class"],
            row["n_items"], row["block_counts"],
        )

    log.info(
        "[%s] Split-half congruence (%d iterations, k=%d, %s)…",
        data.model.slug, SPLIT_HALF_N_ITERS, fit.k, fit.rotation,
    )
    split_half = split_half_congruence(
        data.matrix,
        n_factors=fit.k,
        out_dir=out_dir,                       # writes split_half_congruence.{json,png}
        n_iters=SPLIT_HALF_N_ITERS,
        fa_method=fit.method,
        rotation=fit.rotation,
        align="procrustes",                    # oblimin-compatible alignment
        seed=SEED,
        pass_threshold_median_phi=SPLIT_HALF_PASS_THRESHOLD_PHI,
        verbose=False,
    )

    return {
        "cronbach_alpha": alpha_payload,
        "split_half_congruence": split_half,
    }


# ═════════════════════════════════════════════════════════════════════════════
# VARIANCE DECOMPOSITION (archetype + scenario)
# ═════════════════════════════════════════════════════════════════════════════


def run_variance_decomp(data: LoadedData, fit: FaFit) -> dict | None:
    """Decompose factor-score variance into archetype + scenario contributions.

    Thin wrapper around
    :func:`src_dev.psychometric.variance_decomp.run_variance_decomposition`.
    Outputs land under ``{output_dir}/variance_decomp/``:

        eta2_oneway.json
        eta2_twoway.json
        per_archetype_factor_means.csv
        factor_means_by_archetype.{png,pdf}

    Returns the result dict (or ``None`` if the rollout dir is missing —
    e.g. when this script runs on a fresh machine without the B-rollout
    cache hydrated locally).
    """
    if not ROLLOUT_DIR_FOR_VARIANCE.exists():
        log.warning(
            "[%s] rollout dir missing — skipping variance decomp (%s)",
            data.model.slug, ROLLOUT_DIR_FOR_VARIANCE,
        )
        return None
    if not SCENARIOS_FILE.exists():
        log.warning(
            "[%s] scenarios file missing — skipping variance decomp (%s)",
            data.model.slug, SCENARIOS_FILE,
        )
        return None

    fa_result = load_factor_analysis(fit.npz_path)
    factor_labels = _axis_labels_for(data.model.slug, fit.k)
    out_dir = data.output_dir / "variance_decomp"

    log.info(
        "[%s] variance decomposition (archetype + scenario) -> %s",
        data.model.slug, out_dir,
    )
    result = run_variance_decomposition(
        scores=fa_result["scores"],
        metadata=data.metadata,
        rollout_dir=ROLLOUT_DIR_FOR_VARIANCE,
        scenarios_file=SCENARIOS_FILE,
        out_dir=out_dir,
        model_label=data.model.label,
        factor_labels=factor_labels,
    )

    one = result["oneway"]
    two = result["twoway"]
    log.info(
        "[%s] one-way η² (archetype | scenario) per factor:",
        data.model.slug,
    )
    for i, (ea, es) in enumerate(zip(
        one["eta2_archetype_per_factor"],
        one["eta2_scenario_per_factor"],
    )):
        log.info(
            "  F%d (%s):  archetype=%.3f   scenario=%.3f",
            i, factor_labels[i], ea, es,
        )
    log.info(
        "[%s] two-way decomposition (arch / scen / interact / resid):",
        data.model.slug,
    )
    for i in range(one["n_factors"]):
        log.info(
            "  F%d:  arch=%.3f  scen=%.3f  interact=%.3f  resid=%.3f  (n_cells=%d)",
            i,
            two["eta2_archetype"][i], two["eta2_scenario"][i],
            two["eta2_interaction"][i], two["eta2_residual"][i],
            two["n_cells"][i],
        )
    return result


# ═════════════════════════════════════════════════════════════════════════════
# SCENARIO-RESIDUALIZED FA (robustness check)
# ═════════════════════════════════════════════════════════════════════════════


def run_residualized_fa(data: LoadedData, fit: FaFit) -> dict | None:
    """Re-fit FA on the scenario-residualized response matrix.

    Subtracts the per-scenario mean of each item from each persona's
    response, then re-runs FA at the same k. The residualized solution
    answers ``which axes of behavioural variation persist within
    scenarios?'' rather than ``which axes do scenarios pull responses
    along?''.

    Computes:
        - Top-loading items per residualized factor (text dump).
        - Cronbach's α + split-half congruence on residualized factors.
        - Tucker's |φ| Hungarian-matched between raw and residualized
          factors (over shared items): the headline ``did the same
          factors survive residualization?'' metric.
        - Variance decomposition on residualized factor scores (sanity
          check: scenario η² should now be ≈0 by construction).

    Outputs land under ``{output_dir}/factor_analysis_resid/``. Returns
    ``None`` if the rollout dir or scenarios file is missing.
    """
    if not ROLLOUT_DIR_FOR_VARIANCE.exists() or not SCENARIOS_FILE.exists():
        log.warning(
            "[%s] rollout dir or scenarios file missing — skipping residualized FA",
            data.model.slug,
        )
        return None

    out_dir = data.output_dir / "factor_analysis_resid"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Annotate metadata with scenario_id (and archetype, for later η²).
    lookup = build_archetype_scenario_lookup(
        ROLLOUT_DIR_FOR_VARIANCE, SCENARIOS_FILE,
    )
    enriched_meta = []
    for row in data.metadata:
        sid = row.get("sample_id")
        hit = lookup.get(sid) if sid is not None else None
        if hit is None:
            enriched_meta.append(dict(row))
            continue
        enriched_meta.append({**row, **hit})
    keep = np.array(
        [("scenario_id" in r) for r in enriched_meta], dtype=bool,
    )
    matrix_k = data.matrix[keep]
    meta_k = [r for r, k in zip(enriched_meta, keep) if k]
    if keep.sum() < len(enriched_meta):
        log.info(
            "[%s] residualized FA: %d / %d rows resolved scenario_id",
            data.model.slug, int(keep.sum()), int(len(enriched_meta)),
        )

    # Subtract per-scenario item means.
    matrix_resid, _gm, _gi = resid_primitive(
        matrix_k, meta_k, group_field="scenario_id",
    )

    # Drop columns that became near-zero variance after residualization
    # (rare, but possible for an item that is fully scenario-determined).
    var_post = np.nanvar(matrix_resid, axis=0)
    keep_col = var_post > 1e-9
    matrix_resid = matrix_resid[:, keep_col]
    items_resid = [it for it, k in zip(data.items, keep_col) if k]
    if not keep_col.all():
        log.info(
            "[%s] residualized FA: dropping %d cols with ~0 var post-resid",
            data.model.slug, int((~keep_col).sum()),
        )

    log.info(
        "[%s] residualized FA: matrix=%s, fitting k=%d %s/%s",
        data.model.slug, matrix_resid.shape, fit.k, FA_METHOD, FA_ROTATION,
    )
    fa_resid = run_factor_analysis(
        matrix_resid,
        n_factors=fit.k,
        method=FA_METHOD,
        rotation=FA_ROTATION,
    )
    fa_resid, _perm = _sort_factors_by_variance(fa_resid)

    base = out_dir / f"fa_{fit.k}_{FA_METHOD}_{FA_ROTATION}"
    save_factor_analysis(
        fa_resid,
        base,
        config={
            "n_factors": fit.k,
            "method": FA_METHOD,
            "rotation": FA_ROTATION,
            "residualized_by": "scenario_id",
            "n_samples": int(matrix_resid.shape[0]),
            "n_cols": int(matrix_resid.shape[1]),
            "model_slug": data.model.slug,
            "model_label": data.model.label,
        },
    )
    top_path = Path(str(base) + f"_top{TOP_LOADING_ITEMS_FOR_LABELLING}.txt")
    _write_top_items_summary(
        fa_resid, items_resid, top_path,
        n_top=TOP_LOADING_ITEMS_FOR_LABELLING,
        model_label=data.model.label + " (scenario-residualized)",
        n_factors=fit.k,
        rotation=FA_ROTATION,
    )

    # Cronbach's α and split-half on residualized factors.
    log.info("[%s] residualized α + split-half…", data.model.slug)
    alpha_rows = _cronbach_alpha_per_factor(
        matrix_resid, fa_resid["loadings"], items_resid,
        loading_threshold=CRONBACH_LOADING_THRESHOLD,
    )
    alpha_payload = {
        "model_slug": data.model.slug,
        "model_label": data.model.label,
        "n_factors": fit.k,
        "rotation": FA_ROTATION,
        "loading_threshold": CRONBACH_LOADING_THRESHOLD,
        "residualized_by": "scenario_id",
        "per_factor": alpha_rows,
    }
    (out_dir / "cronbach_alpha.json").write_text(
        json.dumps(alpha_payload, indent=2)
    )
    split_half = split_half_congruence(
        matrix_resid,
        n_factors=fit.k,
        out_dir=out_dir,
        n_iters=SPLIT_HALF_N_ITERS,
        fa_method=FA_METHOD,
        rotation=FA_ROTATION,
        align="procrustes",
        seed=SEED,
        pass_threshold_median_phi=SPLIT_HALF_PASS_THRESHOLD_PHI,
        verbose=False,
    )

    # Tucker's |φ| between raw and residualized factors (per model),
    # over shared items. Hungarian-matched. The headline
    # "did the same factors survive?" metric.
    raw_fa = load_factor_analysis(fit.npz_path)
    raw_loadings = raw_fa["loadings"]
    raw_items = data.items
    la, lb, shared = _slice_loadings_to_shared_items(
        raw_loadings, raw_items,
        fa_resid["loadings"], items_resid,
    )
    signed_phi = tucker_phi_matrix(la, lb, signed=True)
    abs_phi = np.abs(signed_phi)
    alignments = align_factors(la, lb)
    matched_rows = [
        {
            "raw_factor": a.anchor_factor,
            "resid_factor": a.target_factor,
            "phi": a.phi,
            "phi_signed": float(signed_phi[a.anchor_factor, a.target_factor])
                if a.target_factor >= 0 else float("nan"),
            "classification": classify_phi(a.phi),
        }
        for a in alignments
    ]
    raw_vs_resid_dir = out_dir / "raw_vs_resid"
    raw_vs_resid_dir.mkdir(parents=True, exist_ok=True)
    factor_labels = _axis_labels_for(data.model.slug, fit.k)
    _plot_phi_heatmap(
        abs_phi,
        anchor_labels=factor_labels,
        target_labels=[f"F{i}" for i in range(lb.shape[1])],
        matched_pairs=[(m["raw_factor"], m["resid_factor"], m["phi"])
                       for m in matched_rows if m["resid_factor"] >= 0],
        anchor_name="Raw FA factors",
        target_name="Scenario-residualized FA factors",
        subset_label="raw vs residualized",
        n_shared=len(shared),
        save_path=raw_vs_resid_dir / "phi_heatmap.png",
    )
    np.save(raw_vs_resid_dir / "phi_matrix.npy", signed_phi)
    raw_vs_resid_report = {
        "model": data.model.slug,
        "n_shared_items": len(shared),
        "matched": matched_rows,
        "overall_mean_matched_phi": float(np.mean(
            [m["phi"] for m in matched_rows if m["resid_factor"] >= 0]
        )) if matched_rows else float("nan"),
    }
    (raw_vs_resid_dir / "report.json").write_text(
        json.dumps(raw_vs_resid_report, indent=2)
    )

    log.info(
        "[%s] raw↔resid Tucker's |φ| (Hungarian-matched, n_shared=%d):",
        data.model.slug, len(shared),
    )
    for m in matched_rows:
        log.info(
            "  raw F%d (%s) ↔ resid F%d  |φ|=%.3f signed=%+.3f (%s)",
            m["raw_factor"], factor_labels[m["raw_factor"]],
            m["resid_factor"], m["phi"], m["phi_signed"], m["classification"],
        )

    # Variance decomposition on residualized factor scores.
    var_decomp_resid = run_variance_decomposition(
        scores=fa_resid["scores"],
        metadata=meta_k,
        rollout_dir=ROLLOUT_DIR_FOR_VARIANCE,
        scenarios_file=SCENARIOS_FILE,
        out_dir=out_dir / "variance_decomp",
        model_label=data.model.label + " (residualized)",
        factor_labels=[f"F{i}" for i in range(fit.k)],
    )
    log.info(
        "[%s] residualized one-way η² (arch | scen) — scen should be ~0:",
        data.model.slug,
    )
    one = var_decomp_resid["oneway"]
    for i, (ea, es) in enumerate(zip(
        one["eta2_archetype_per_factor"],
        one["eta2_scenario_per_factor"],
    )):
        log.info(
            "  resid F%d:  archetype=%.3f   scenario=%.3f", i, ea, es,
        )

    return {
        "fa_resid_path": Path(str(base) + ".npz"),
        "alpha": alpha_payload,
        "split_half": split_half,
        "raw_vs_resid": raw_vs_resid_report,
        "variance_decomp": var_decomp_resid,
        "n_items": int(matrix_resid.shape[1]),
    }


# ═════════════════════════════════════════════════════════════════════════════
# LORA FACTOR-SHIFT VALIDATION (per-LoRA per-factor mean shift)
# ═════════════════════════════════════════════════════════════════════════════


def run_lora_factor_shifts() -> dict | None:
    """Hydrate per-LoRA factor-shift summaries and write the cross-LoRA table.

    Pulls each ``LORA_VALIDATIONS`` entry's ``<label>_summary.json`` (and
    ``_scores.npz`` when present) from HF into ``LORA_VALIDATION_LOCAL_ROOT``,
    aggregates per-factor mean shifts into a (n_lora × n_factor) matrix,
    and writes:

        ``{OUTPUT_ROOT}/lora_factor_shifts/lora_factor_shifts.csv``
        ``{OUTPUT_ROOT}/lora_factor_shifts/lora_factor_shifts_index.json``
        ``{OUTPUT_ROOT}/lora_factor_shifts/lora_shifts_heatmap.{png,pdf}``

    Returns the shifts dict or ``None`` if no LoRAs are configured.
    """
    if not LORA_VALIDATIONS:
        log.info("[lora-shifts] LORA_VALIDATIONS is empty — skipping")
        return None

    out_dir = OUTPUT_ROOT / "lora_factor_shifts"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "[lora-shifts] hydrating + loading %d validation summaries -> %s",
        len(LORA_VALIDATIONS), out_dir,
    )

    # Honour LORA_VALIDATION_PREFER_LARGE_N globally by overriding each
    # entry's flag (LoraValidation is frozen — recreate).
    validations = [
        LoraValidation(
            label=v.label, factor=v.factor, direction=v.direction,
            hf_subdir=v.hf_subdir,
            prefer_large_n=LORA_VALIDATION_PREFER_LARGE_N,
            display_label=v.display_label,
        )
        for v in LORA_VALIDATIONS
    ]
    shifts = load_lora_factor_shifts(
        validations,
        hf_repo_id=LORA_VALIDATION_HF_REPO,
        local_root=LORA_VALIDATION_LOCAL_ROOT,
        pull_scores=True,
        # Short factor names — the validation-summary loader strips F#_
        # prefixes, so the canonical order here is the post-sort order
        # used in the rest of the v2 pipeline (Initiative / Warmth /
        # Pedagogy / Hedging).
        canonical_factor_order=["Initiative", "Warmth", "Pedagogy", "Hedging"],
    )

    rows = shifts["rows"]
    factors = shifts["factors"]
    # Paper-display names (TIDE acronym: Tone, Initiative, Didacticism,
    # Epistemic Caution). The internal short names (Warmth / Pedagogy /
    # Hedging) survive in code, data, and the HF monorepo paths — see
    # scripts_dev/unsupervised_embeddings/FACTOR_NAMING.md for the rationale.
    paper_display_names = [PAPER_FACTOR_DISPLAY_NAMES.get(f, f) for f in factors]
    if not rows:
        log.warning("[lora-shifts] no validation summaries hydrated; nothing to plot")
        return None

    # CSV.
    diff = shifts["mean_diff"]
    lo = shifts["ci_lo"]
    hi = shifts["ci_hi"]
    dz = shifts["cohen_dz"]
    csv_path = out_dir / "lora_factor_shifts.csv"
    with csv_path.open("w") as fh:
        fh.write(
            "label,target_factor,direction,n_personas," +
            ",".join(f"{f}_diff,{f}_ci_lo,{f}_ci_hi,{f}_dz" for f in factors) + "\n"
        )
        for i, r in enumerate(rows):
            cells = [r["label"], r["factor"], r["direction"], str(r["n_personas"])]
            for j, _f in enumerate(factors):
                cells += [
                    f"{diff[i, j]:.4f}",
                    f"{lo[i, j]:.4f}",
                    f"{hi[i, j]:.4f}",
                    f"{dz[i, j]:.4f}",
                ]
            fh.write(",".join(cells) + "\n")

    (out_dir / "lora_factor_shifts_index.json").write_text(json.dumps({
        "factors": factors,
        "rows": rows,
        "n_loras": len(rows),
        "hf_repo": LORA_VALIDATION_HF_REPO,
    }, indent=2))

    # Three heatmap variants: full-sample (naive), middling-only, headroom.
    # The middling and headroom variants depend on the per-row bucketed
    # data populated by load_lora_factor_shifts (computed off the
    # paired scores npz).
    plot_variants: list[str] = ["naive", "middling", "headroom"]
    for selection in plot_variants:
        try:
            mat = build_shift_matrix(shifts, selection=selection)
        except Exception as exc:
            log.warning("[lora-shifts] could not build %s matrix: %s", selection, exc)
            continue
        plot_factor_shift_heatmap(
            shifts,
            save_path=out_dir / f"lora_shifts_heatmap_{selection}.png",
            title="Per-LoRA factor-score shift",
            factor_display_names=paper_display_names,
            annotate="diff_with_ci",
            matrix_override=None if selection == "naive" else mat,
            row_label_factor_map=PAPER_FACTOR_DISPLAY_NAMES,
        )

    # Focused bar chart for the paper's main figure: Initiative-amplifier
    # + Initiative-suppressor across all four factors, on the medium-baseline
    # tertile (the same ceiling/floor-robust view the heatmap uses by default).
    # Optionally include the OCEAN-definition control LoRA as a baseline.
    initiative_labels = [r["label"] for r in rows
                         if r.get("factor") in {"Initiative", "Control"}]
    if initiative_labels:
        try:
            mat_mid = build_shift_matrix(shifts, selection="middling")
            plot_factor_shift_barchart(
                shifts,
                save_path=out_dir / "lora_shifts_barchart_initiative.png",
                title="Initiative LoRAs: factor-score shifts",
                factor_display_names=paper_display_names,
                label_filter=initiative_labels,
                matrix_override=mat_mid,
                row_label_factor_map=PAPER_FACTOR_DISPLAY_NAMES,
            )
        except Exception as exc:
            log.warning("[lora-shifts] could not build Initiative bar chart: %s", exc)

    log.info("[lora-shifts] %d LoRAs loaded; full-sample mean shifts:", len(rows))
    for i, r in enumerate(rows):
        diffs_str = "  ".join(f"{f}={diff[i, j]:+.2f}" for j, f in enumerate(factors))
        log.info(
            "  %s (%s%s, n=%d): %s",
            r["label"], r["factor"], r["direction"], r["n_personas"], diffs_str,
        )

    return shifts


# ═════════════════════════════════════════════════════════════════════════════
# PREDICTIVITY CROSS-VALIDATION (BI-CV)
# ═════════════════════════════════════════════════════════════════════════════


def run_predictivity_cv(data: LoadedData, fit: FaFit) -> dict:
    """Bi-cross-validation: predict held-out persona × item responses.

    Thin wrapper around
    :func:`src_dev.factor_analysis.cross_validation.persona_item_cv`, which
    implements the Owen-Perry / Bro-et-al bi-CV scheme with three principled
    nulls (``item_mean`` / ``persona_shuffle`` / ``k_minus_1``). The main
    statistic is per-item R² pooled over the outer × trial loop; the
    shuffle baseline is the null for "is this genuinely persona-predictive".

    Output: ``{output_dir}/validation/predictivity_cv.{json,png}`` (persona_
    item_cv writes these); main() also prints a compact summary table.
    """
    out_dir = data.output_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "[%s] predictivity CV: %d outer × %d trials, A=%.0f%%, m_observed≈%.0f%% items",
        data.model.slug,
        PREDICTIVITY_N_OUTER_SPLITS,
        PREDICTIVITY_N_TRIALS,
        100 * PREDICTIVITY_PERSONA_SPLIT,
        100 * PREDICTIVITY_ITEM_OBSERVED_FRAC,
    )
    m_observed = max(
        3 * fit.k,
        int(round(PREDICTIVITY_ITEM_OBSERVED_FRAC * data.matrix.shape[1])),
    )
    result = persona_item_cv(
        data.matrix,
        data.metadata,
        n_factors=fit.k,
        out_dir=out_dir,
        persona_split=PREDICTIVITY_PERSONA_SPLIT,
        stratify=None,                          # B preset has 1 rollout / persona
        m_observed=m_observed,
        subset_strategy=PREDICTIVITY_SUBSET_STRATEGY,
        n_trials=PREDICTIVITY_N_TRIALS,
        n_outer_splits=PREDICTIVITY_N_OUTER_SPLITS,
        bootstrap_ci=PREDICTIVITY_BOOTSTRAP_CI,
        fa_method=fit.method,
        rotation=fit.rotation,
        seed=SEED,
        verbose=False,
    )

    # Fold-vs-full Tucker's is intentionally omitted here — the existing
    # split_half_congruence step (50/50 splits × 100 iters) already
    # establishes that the factors are stable under resampling, and an
    # 80/20 variant would not add a materially different claim.
    log.info(
        "[%s] predictivity CV: main R²=%.3f  shuffle=%.3f  k-1=%.3f  item-mean=%.3f",
        data.model.slug,
        result.get("mean_r2_main", float("nan")),
        result.get("mean_r2_persona_shuffle", float("nan")),
        result.get("mean_r2_k_minus_1", float("nan")),
        result.get("mean_r2_item_mean", float("nan")),
    )
    return result


# ═════════════════════════════════════════════════════════════════════════════
# CROSS-MODEL CONGRUENCE
# ═════════════════════════════════════════════════════════════════════════════


def _slice_loadings_to_shared_items(
    loadings_a: np.ndarray,
    items_a: list[dict],
    loadings_b: np.ndarray,
    items_b: list[dict],
    *,
    block_filter: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Intersect two fits by col_id and return aligned loadings + shared IDs.

    Preprocessing drops different low-variance items per model (86/100
    trait_mcq kept on Llama, 71/100 on Qwen), so each model's loadings
    sit on a slightly different item set. Tucker's φ requires matching
    item indices, so we intersect on col_id and reindex both sides.

    ``block_filter="likert"`` / ``"trait_mcq"`` restricts the shared set
    to that block; ``None`` uses everything shared. Returning the col_ids
    lets the caller label rows on the eventual heatmap.
    """
    ids_a = [it["col_id"] for it in items_a]
    ids_b = [it["col_id"] for it in items_b]
    idx_a = {cid: i for i, cid in enumerate(ids_a)}
    idx_b = {cid: i for i, cid in enumerate(ids_b)}
    shared = sorted(set(ids_a) & set(ids_b))
    if block_filter is not None:
        block_a = {it["col_id"]: it.get("block") for it in items_a}
        shared = [c for c in shared if block_a.get(c) == block_filter]
    if not shared:
        raise ValueError(
            f"No shared items between fits (block_filter={block_filter!r})."
        )
    la = loadings_a[[idx_a[c] for c in shared], :]
    lb = loadings_b[[idx_b[c] for c in shared], :]
    return la, lb, shared


def _plot_phi_heatmap(
    abs_phi: np.ndarray,
    *,
    anchor_labels: list[str],
    target_labels: list[str],
    matched_pairs: list[tuple[int, int, float]],
    anchor_name: str,
    target_name: str,
    subset_label: str,
    n_shared: int,
    save_path: Path,
) -> None:
    """|φ| heatmap with matched cells outlined — one PDF per subset."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ka, kb = abs_phi.shape
    fig, ax = plt.subplots(figsize=(max(4, 0.9 * kb + 2), max(3.5, 0.8 * ka + 2)))
    im = ax.imshow(abs_phi, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")

    # Overlay matched cells with white outlines.
    for fa_, fb_, _ in matched_pairs:
        ax.add_patch(plt.Rectangle(
            (fb_ - 0.5, fa_ - 0.5), 1, 1,
            fill=False, edgecolor="white", linewidth=2.2,
        ))

    for i in range(ka):
        for j in range(kb):
            v = abs_phi[i, j]
            ax.text(
                j, i, f"{v:.2f}",
                ha="center", va="center",
                color="white" if v < 0.55 else "black", fontsize=9,
            )
    ax.set_xticks(range(kb)); ax.set_xticklabels(target_labels, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(ka)); ax.set_yticklabels(anchor_labels, fontsize=9)
    ax.set_xlabel(target_name)
    ax.set_ylabel(anchor_name)
    ax.set_title(
        f"Tucker's |φ|  ({subset_label}, n_shared={n_shared})",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="|φ|")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    fig.savefig(save_path.with_suffix(".pdf"))
    plt.close(fig)


def _axis_labels_for(slug: str, n_factors: int) -> list[str]:
    """Prefer labelled axis names; fall back to ``F{idx}``."""
    labels = _load_labels_for_slug(slug, n_factors)
    return [
        f"F{i}" + (f" ({labels[i]})" if i in labels else "")
        for i in range(n_factors)
    ]


def _run_one_cross_model_subset(
    *,
    anchor_slug: str,
    target_slug: str,
    anchor_loadings: np.ndarray,
    target_loadings: np.ndarray,
    anchor_items: list[dict],
    target_items: list[dict],
    subset_label: str,
    block_filter: str | None,
    save_dir: Path,
) -> dict:
    """Run Tucker's φ for one (anchor, target, block-subset) triple."""
    la, lb, shared = _slice_loadings_to_shared_items(
        anchor_loadings, anchor_items,
        target_loadings, target_items,
        block_filter=block_filter,
    )

    signed_phi = tucker_phi_matrix(la, lb, signed=True)
    abs_phi = np.abs(signed_phi)

    alignments = align_factors(la, lb)
    matched_pairs: list[tuple[int, int, float]] = []
    matched_targets: set[int] = set()
    alignment_rows: list[dict] = []
    for a in alignments:
        if a.target_factor >= 0:
            matched_pairs.append((a.anchor_factor, a.target_factor, a.phi))
            matched_targets.add(a.target_factor)
        alignment_rows.append({
            "anchor_factor": a.anchor_factor,
            "target_factor": a.target_factor,
            "phi": a.phi,
            "phi_signed": (
                float(signed_phi[a.anchor_factor, a.target_factor])
                if a.target_factor >= 0 else float("nan")
            ),
            "sign": a.sign,
            "classification": classify_phi(a.phi),
        })

    # Unmatched target factors (when kb > ka): report each with its best
    # anchor partner, so we don't silently drop them.
    unmatched_target_rows: list[dict] = []
    k_a, k_b = la.shape[1], lb.shape[1]
    for fb_ in range(k_b):
        if fb_ in matched_targets:
            continue
        col = abs_phi[:, fb_]
        best_anchor = int(np.argmax(col))
        phi_val = float(col[best_anchor])
        unmatched_target_rows.append({
            "target_factor": fb_,
            "best_anchor_partner": best_anchor,
            "phi": phi_val,
            "classification": classify_phi(phi_val),
        })

    anchor_labels = _axis_labels_for(anchor_slug, la.shape[1])
    target_labels = _axis_labels_for(target_slug, lb.shape[1])

    _plot_phi_heatmap(
        abs_phi,
        anchor_labels=anchor_labels,
        target_labels=target_labels,
        matched_pairs=matched_pairs,
        anchor_name=anchor_slug, target_name=target_slug,
        subset_label=subset_label,
        n_shared=len(shared),
        save_path=save_dir / "phi_heatmap.png",
    )

    np.save(save_dir / "phi_matrix.npy", signed_phi)

    report = {
        "anchor": anchor_slug,
        "target": target_slug,
        "subset": subset_label,
        "block_filter": block_filter,
        "n_shared_items": len(shared),
        "shared_col_ids": shared,
        "k_anchor": int(la.shape[1]),
        "k_target": int(lb.shape[1]),
        "matched": alignment_rows,
        "unmatched_target": unmatched_target_rows,
        "overall_mean_matched_phi": float(
            np.mean([r["phi"] for r in alignment_rows if r["target_factor"] >= 0])
        ) if alignment_rows else float("nan"),
    }
    (save_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def run_cross_model_congruence(
    loaded: dict[str, LoadedData],
    fits: dict[str, "FaFit"],
) -> dict[str, dict]:
    """Llama ↔ Qwen Tucker's φ across combined + per-block shared items.

    Output layout (top-level, not per-model — this is a cross-model comparison):

        scratch/psychometric_fa_paper/cross_model/
            {anchor}_vs_{target}/
                {combined,likert,trait_mcq}/
                    phi_matrix.npy
                    phi_heatmap.{png,pdf}
                    report.json
                summary.json      # flattened per-subset key metrics

    Hungarian matching is run with ``CROSS_MODEL_ANCHOR`` as anchor; the
    anchor's ``k_anchor`` factors all get a target partner (via
    ``scipy.optimize.linear_sum_assignment``), and any leftover target
    factors (when ``k_target > k_anchor``) are reported separately with
    their single-best anchor partner.
    """
    if CROSS_MODEL_ANCHOR not in fits:
        log.warning("anchor model %s not in fits; skipping cross-model pass",
                    CROSS_MODEL_ANCHOR)
        return {}
    targets = [s for s in fits if s != CROSS_MODEL_ANCHOR]
    if not targets:
        log.warning("only one model fit available; skipping cross-model pass")
        return {}

    anchor = fits[CROSS_MODEL_ANCHOR]
    anchor_data = loaded[CROSS_MODEL_ANCHOR]
    anchor_fa = load_factor_analysis(anchor.npz_path)

    all_reports: dict[str, dict] = {}
    for target_slug in targets:
        target = fits[target_slug]
        target_data = loaded[target_slug]
        target_fa = load_factor_analysis(target.npz_path)

        pair_label = f"{CROSS_MODEL_ANCHOR}_vs_{target_slug}"
        pair_dir = OUTPUT_ROOT / "cross_model" / pair_label
        pair_dir.mkdir(parents=True, exist_ok=True)
        log.info("═══ cross-model: %s ═══", pair_label)

        subset_reports: dict[str, dict] = {}
        summary_rows: list[dict] = []
        for subset in CROSS_MODEL_BLOCK_SUBSETS:
            subset_dir = pair_dir / subset
            subset_dir.mkdir(parents=True, exist_ok=True)
            block_filter = None if subset == "combined" else subset
            try:
                report = _run_one_cross_model_subset(
                    anchor_slug=CROSS_MODEL_ANCHOR,
                    target_slug=target_slug,
                    anchor_loadings=anchor_fa["loadings"],
                    target_loadings=target_fa["loadings"],
                    anchor_items=anchor_data.items,
                    target_items=target_data.items,
                    subset_label=subset,
                    block_filter=block_filter,
                    save_dir=subset_dir,
                )
            except ValueError as exc:
                log.warning("  [%s/%s] skipped: %s", pair_label, subset, exc)
                continue
            subset_reports[subset] = report
            summary_rows.append({
                "subset": subset,
                "n_shared_items": report["n_shared_items"],
                "overall_mean_matched_phi": report["overall_mean_matched_phi"],
                "matched_phis": [r["phi"] for r in report["matched"]],
                "n_unmatched_target": len(report["unmatched_target"]),
            })

            log.info(
                "  [%s/%-9s] n_shared=%3d  mean |φ|=%.3f  "
                "per_pair=[%s]",
                pair_label, subset, report["n_shared_items"],
                report["overall_mean_matched_phi"],
                ", ".join(f"{r['phi']:.2f}" for r in report["matched"]),
            )

        (pair_dir / "summary.json").write_text(
            json.dumps({"pair": pair_label, "subsets": summary_rows}, indent=2)
        )
        all_reports[pair_label] = {"subsets": subset_reports}
    return all_reports


# ═════════════════════════════════════════════════════════════════════════════
# HTML FACTOR BROWSER
# ═════════════════════════════════════════════════════════════════════════════


def _seed_labeling_from_repo(model_slug: str, scratch_labeling_dir: Path) -> None:
    """Copy persisted labels from the in-repo store into scratch/labeling/.

    Only copies files not already present in scratch — so a fresh skill
    write in scratch (newer mtime) always wins on the ``load_latest_non
    empty_llm_labels`` lookup, and this function never shadows an
    in-progress labelling session. ``shutil.copy2`` preserves mtimes so
    the cross-file ordering (when multiple candidates exist) reflects
    when the labels were actually written, not when the sync ran.
    """
    import shutil

    repo_labeling = LABELS_REPO_DIR / model_slug / "labeling"
    if not repo_labeling.is_dir():
        return
    n_synced = 0
    for src in repo_labeling.glob("llm_labels_*.json"):
        dst = scratch_labeling_dir / src.name
        if dst.exists():
            continue
        shutil.copy2(src, dst)
        n_synced += 1
    if n_synced:
        log.info(
            "[%s] synced %d persisted label file(s) from %s into %s",
            model_slug, n_synced, repo_labeling, scratch_labeling_dir,
        )


def _load_raw_questionnaire_index() -> dict[tuple[str, str], dict]:
    """Build a lookup `(questionnaire_version, bare_id) -> raw_item`.

    v7 fc_pair items live under top-level ``items`` (no nested ``block_*``
    structure). Each raw item carries the same ``text``/``high_option``/
    ``axis_high_letter`` fields already on items.json — enrichment is
    largely a no-op here; the lookup mostly exists so the
    ``/label-fa-factors`` skill resolves cleanly.
    """
    idx: dict[tuple[str, str], dict] = {}
    for version, path in RAW_QUESTIONNAIRE_PATHS.items():
        if not path.exists():
            log.warning("raw questionnaire missing: %s", path)
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        for raw_item in raw.get("items", []) or []:
            idx[(version, str(raw_item["id"]))] = raw_item
        # Defensive: also walk any nested block_*.items for forward-compat.
        for k, v in raw.items():
            if isinstance(v, dict) and "items" in v and isinstance(v["items"], list):
                for raw_item in v["items"]:
                    if "id" in raw_item:
                        idx[(version, str(raw_item["id"]))] = raw_item
    return idx


def _enrich_column_defs(column_defs: list[dict]) -> list[dict]:
    """Splice raw-questionnaire fields onto column defs.

    Doesn't mutate the inputs — returns a new list of dicts with extra
    fields populated where a raw questionnaire entry is available.

    v7 fc_pair col_ids are bare ids (e.g. ``v7fc_001``) without a
    ``{version}/`` prefix, so we look up by ``(item.version, col_id)``
    where ``version`` is stamped onto items at load time.
    """
    raw_index = _load_raw_questionnaire_index()
    enriched: list[dict] = []
    for cdef in column_defs:
        out = dict(cdef)
        col_id = str(cdef.get("col_id", ""))
        # Support both styles: "version/bare_id" (v1) and bare "v7fc_NNN" (v2).
        if "/" in col_id:
            version, _, bare = col_id.partition("/")
        else:
            version = str(cdef.get("version", "") or cdef.get("questionnaire_version", ""))
            bare = col_id
        raw_item = raw_index.get((version, bare))
        if raw_item is None:
            enriched.append(out)
            continue
        if "reverse_keyed" not in out and "reverse_keyed" in raw_item:
            out["reverse_keyed"] = bool(raw_item["reverse_keyed"])
        if "options" in raw_item and "options" not in out:
            out["options"] = raw_item["options"]
        if "answer_mapping" in raw_item and "answer_mapping" not in out:
            out["answer_mapping"] = raw_item["answer_mapping"]
        enriched.append(out)
    return enriched


def export_html_browser(data: LoadedData, fit: FaFit) -> Path | None:
    """Write an interactive HTML factor browser for one FA fit.

    Pulls rollout conversations from ``ROLLOUT_DIR_FOR_HTML`` (both models
    share the same B-rollout cache) and lays out, per factor, the top-N and
    bottom-N personas by score with their conversation transcripts. If a
    ``labeling/llm_labels_{analysis_key}_manual_*.json`` file is present
    (produced by the ``/label-fa-factors`` skill), the labels show up in the
    browser automatically; otherwise the exporter falls back to top-loading
    items.

    Output: ``{output_dir}/factor_analysis/raw_{rotation}/factor_extremes.html``
    (matches the existing pipeline's layout).

    Returns the HTML path (or None if the rollout export was missing).
    """
    if not ROLLOUT_DIR_FOR_HTML.exists():
        log.warning(
            "[%s] rollout dir missing — skipping HTML export (%s)",
            data.model.slug, ROLLOUT_DIR_FOR_HTML,
        )
        return None

    rotation = fit.rotation
    analysis_key = f"raw_{rotation}"  # label the skill uses; also the HTML label

    save_dir = data.output_dir / "factor_analysis" / analysis_key
    save_dir.mkdir(parents=True, exist_ok=True)

    labeling_dir = data.output_dir / "labeling"
    labeling_dir.mkdir(parents=True, exist_ok=True)
    _seed_labeling_from_repo(data.model.slug, labeling_dir)

    fa_result = load_factor_analysis(fit.npz_path)
    enriched_items = _enrich_column_defs(data.items)

    log.info(
        "[%s] exporting factor_extremes.html → %s",
        data.model.slug, save_dir / "factor_extremes.html",
    )
    export_factor_extremes_html(
        fa_result=fa_result,
        column_defs=enriched_items,
        metadata=data.metadata,
        label=analysis_key,
        save_dir=save_dir,
        rollout_dirs=[ROLLOUT_DIR_FOR_HTML],
        labeling_dir=labeling_dir,
        n_per_pole=HTML_N_PER_POLE,
    )
    return save_dir / "factor_extremes.html"


# ═════════════════════════════════════════════════════════════════════════════
# PAPER FIGURES
# ═════════════════════════════════════════════════════════════════════════════


def _plot_paper_within_model_validation(
    validation_results: dict[str, dict],
) -> Path | None:
    """Grouped-bars paper figure: Cronbach's α + split-half median |φ| per factor.

    Two panels side-by-side (α on the left, |φ| on the right) with both
    models' factors grouped by index. Axis labels come from the newest
    persisted labels file when available. Reference lines at α=0.70/0.80
    and |φ|=0.85. Writes to ``PAPER_FIGURES_DIR / unsupervised / fig_4_2_2_
    within_model_validation.pdf``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Collect per-model series. Pad the shorter to the longer so the x-axis
    # lines up when models have different k.
    rows = []
    for slug, res in validation_results.items():
        alpha_rows = res["cronbach_alpha"]["per_factor"]
        sh = res["split_half_congruence"]
        medians = sh.get("median_phi_per_factor", [])
        labels = _load_labels_for_slug(slug, len(alpha_rows))
        for i, r in enumerate(alpha_rows):
            rows.append({
                "slug": slug,
                "factor_index": i,
                "axis": labels.get(i, f"F{i}"),
                "alpha": r["alpha"],
                "phi_median": medians[i] if i < len(medians) else float("nan"),
            })

    slugs = list(validation_results.keys())
    max_k = max(r["factor_index"] for r in rows) + 1
    colours = {slugs[0]: "#2563eb", slugs[1]: "#ea580c"} if len(slugs) >= 2 else {
        slugs[0]: "#2563eb",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for panel_idx, (ax, field, ref_lines, title) in enumerate((
        (axes[0], "alpha",
         [(0.70, "acceptable", "#f59e0b"), (0.80, "good", "#16a34a")],
         "Cronbach's α per factor (|loading| ≥ 0.4, sign-oriented)"),
        (axes[1], "phi_median",
         [(0.85, "fair (Lorenzo-Seva)", "#16a34a")],
         "Split-half median |φ| per factor (100 iters)"),
    )):
        width = 0.36
        x = np.arange(max_k)
        for i, slug in enumerate(slugs):
            offset = (i - (len(slugs) - 1) / 2) * width
            heights = [float("nan")] * max_k
            axis_labels = [""] * max_k
            for r in rows:
                if r["slug"] == slug:
                    heights[r["factor_index"]] = r[field]
                    axis_labels[r["factor_index"]] = r["axis"]
            bars = ax.bar(
                x + offset, heights, width,
                color=colours[slug], alpha=0.88, label=slug,
                edgecolor="#111", linewidth=0.3,
            )
            for rect, h, lab in zip(bars, heights, axis_labels):
                if np.isnan(h):
                    continue
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    h + 0.012,
                    f"{h:.2f}",
                    ha="center", va="bottom", fontsize=7.5, color="#111",
                )
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    -0.03,
                    lab,
                    ha="center", va="top", fontsize=6.8, rotation=40,
                    color="#374151",
                )
        for thresh, lbl, col in ref_lines:
            ax.axhline(thresh, linestyle="--", color=col, linewidth=0.9, alpha=0.8,
                       label=f"{lbl} ({thresh:.2f})")
        ax.set_xticks(x)
        ax.set_xticklabels([f"F{k}" for k in range(max_k)])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(r"$\alpha$" if field == "alpha" else r"median $|\phi|$")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")
        # Leave headroom at the bottom so the rotated axis labels don't clip.
        ax.set_ylim(bottom=-0.18)

    save_path = PAPER_FIGURES_DIR / "unsupervised" / "fig_4_2_2_within_model_validation.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", save_path)
    return save_path


def _copy_paper_cross_model_heatmap(
    cross_reports: dict[str, dict],
) -> Path | None:
    """Copy the combined-subset Tucker's |φ| heatmap to paper/figures/.

    We keep the PDF emitted by _plot_phi_heatmap and just move/copy it to
    the canonical paper path — no need to re-render.
    """
    import shutil
    # Pick the first pair's "combined" heatmap — there's only one pair at
    # present (Llama vs Qwen) but this stays robust if we ever add more.
    for pair_label, bundle in cross_reports.items():
        subsets = bundle.get("subsets", {})
        combined = subsets.get("combined")
        if combined is None:
            continue
        src = (OUTPUT_ROOT / "cross_model" / pair_label / "combined"
               / "phi_heatmap.pdf")
        if not src.exists():
            continue
        dst = (PAPER_FIGURES_DIR / "unsupervised"
               / "fig_4_2_3_cross_model_phi.pdf")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        log.info("wrote %s", dst)
        return dst
    return None


def _copy_paper_lora_shifts(out_dir: Path) -> list[Path]:
    """Copy each LoRA factor-shift figure to paper/figures/.

    Maps:
        lora_shifts_heatmap_naive.pdf            -> fig_4_2_6_lora_shifts.pdf
        lora_shifts_heatmap_middling.pdf         -> fig_4_2_6b_lora_shifts_middling.pdf
        lora_shifts_heatmap_headroom.pdf         -> fig_4_2_6c_lora_shifts_headroom.pdf
        lora_shifts_barchart_initiative.pdf      -> fig_4_2_6_lora_initiative_bars.pdf
    """
    import shutil
    mapping = {
        "lora_shifts_heatmap_naive.pdf":         "fig_4_2_6_lora_shifts.pdf",
        "lora_shifts_heatmap_middling.pdf":      "fig_4_2_6b_lora_shifts_middling.pdf",
        "lora_shifts_heatmap_headroom.pdf":      "fig_4_2_6c_lora_shifts_headroom.pdf",
        "lora_shifts_barchart_initiative.pdf":   "fig_4_2_6_lora_initiative_bars.pdf",
    }
    written: list[Path] = []
    for src_name, dst_name in mapping.items():
        src = out_dir / src_name
        if not src.exists():
            continue
        dst = PAPER_FIGURES_DIR / "unsupervised" / dst_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        log.info("wrote %s", dst)
        written.append(dst)
    return written


def _plot_paper_residualized(
    residualized_results: dict[str, dict],
) -> Path | None:
    """Paper appendix figure: raw-FA Cronbach's α vs residualized α per factor,
    plus the per-factor Tucker's |φ| between raw and residualized loadings.

    Two panels per model. Left bars: α(raw, blue) vs α(resid, orange) per
    factor. Right bars: |φ| of the Hungarian-matched raw↔resid pair per
    factor; reference line at the Lorenzo-Seva 0.85 fair-similarity
    threshold. Tells the reader at a glance how much of each model's
    factor structure persists after we strip out scenario-genre variance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not residualized_results:
        return None

    # Need raw α per slug too — reload from validation/cronbach_alpha.json.
    raw_alpha_by_slug: dict[str, list[float]] = {}
    for slug in residualized_results:
        path = OUTPUT_ROOT / slug / "validation" / "cronbach_alpha.json"
        if not path.exists():
            log.warning("missing raw α for %s — skipping in residualized plot", slug)
            continue
        rows = json.loads(path.read_text())["per_factor"]
        raw_alpha_by_slug[slug] = [float(r["alpha"]) for r in rows]

    slugs = [s for s in residualized_results if s in raw_alpha_by_slug]
    if not slugs:
        return None

    n_models = len(slugs)
    fig, axes = plt.subplots(
        n_models, 2, figsize=(11, 4.0 * n_models),
        constrained_layout=True,
    )
    if n_models == 1:
        axes = np.array([axes])

    for row_idx, slug in enumerate(slugs):
        res = residualized_results[slug]
        alpha_raw = raw_alpha_by_slug[slug]
        alpha_resid = [float(r["alpha"]) for r in res["alpha"]["per_factor"]]
        labels = _axis_labels_for(slug, len(alpha_raw))
        x = np.arange(len(alpha_raw))
        width = 0.36

        # Left: α(raw) vs α(resid)
        ax = axes[row_idx, 0]
        ax.bar(x - width / 2, alpha_raw, width, color="#2563eb",
               label="raw FA", edgecolor="#111", linewidth=0.3)
        ax.bar(x + width / 2, alpha_resid, width, color="#ea580c",
               label="scenario-residualized", edgecolor="#111", linewidth=0.3)
        for i, (a_r, a_d) in enumerate(zip(alpha_raw, alpha_resid)):
            ax.text(i - width / 2, a_r + 0.01, f"{a_r:.2f}",
                    ha="center", va="bottom", fontsize=7.5)
            ax.text(i + width / 2, a_d + 0.01, f"{a_d:.2f}",
                    ha="center", va="bottom", fontsize=7.5)
        for thresh, lbl, c in (
            (0.70, "acceptable", "#f59e0b"),
            (0.80, "good", "#16a34a"),
        ):
            ax.axhline(thresh, linestyle="--", color=c, linewidth=0.9, alpha=0.7,
                       label=f"{lbl} ({thresh:.2f})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(r"Cronbach's $\alpha$")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{slug}: $\\alpha$ before and after scenario-residualization", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")

        # Right: per-factor raw↔resid |φ|
        ax = axes[row_idx, 1]
        matched = res["raw_vs_resid"]["matched"]
        # Sort matched rows by raw_factor for the x-axis to align with α plot
        matched_sorted = sorted(matched, key=lambda m: m["raw_factor"])
        phis = [m["phi"] for m in matched_sorted]
        ax.bar(x, phis, color="#16a34a", edgecolor="#111", linewidth=0.3,
               label=r"raw$\leftrightarrow$resid Tucker's $|\phi|$")
        for i, p in enumerate(phis):
            ax.text(i, p + 0.012, f"{p:.2f}", ha="center", va="bottom", fontsize=7.5)
        ax.axhline(0.85, linestyle="--", color="#16a34a", alpha=0.7, linewidth=0.9,
                   label="Lorenzo-Seva fair (0.85)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(r"$|\phi|$ between raw and residualized factors")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{slug}: factor stability (n_shared={res['raw_vs_resid']['n_shared_items']})",
                     fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="lower right")

    save_path = PAPER_FIGURES_DIR / "unsupervised" / "fig_4_2_5_residualized.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", save_path)
    return save_path


def _plot_paper_variance_decomp(
    variance_decomp_results: dict[str, dict],
) -> Path | None:
    """Grouped-bars paper figure: archetype vs scenario one-way η² per factor.

    Two panels side-by-side (one per model). Each panel shows one bar per
    factor for archetype η² and one for scenario η². The decomposition
    answers ``how much of each factor's persona-level variance is just
    the conversation-context``, so the appendix figure makes it easy to
    read off scenario-dominance vs archetype-dominance per factor.

    Writes to ``PAPER_FIGURES_DIR / unsupervised / fig_4_2_4_variance_decomp.pdf``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not variance_decomp_results:
        return None

    slugs = list(variance_decomp_results.keys())
    fig, axes = plt.subplots(
        1, len(slugs), figsize=(5.5 * len(slugs), 4.6),
        sharey=True, constrained_layout=True,
    )
    if len(slugs) == 1:
        axes = [axes]

    for ax, slug in zip(axes, slugs):
        one = variance_decomp_results[slug]["oneway"]
        labels = one.get("factor_labels") or [f"F{i}" for i in range(one["n_factors"])]
        x = np.arange(one["n_factors"])
        width = 0.36
        eta_arch = np.asarray(one["eta2_archetype_per_factor"], dtype=float)
        eta_scen = np.asarray(one["eta2_scenario_per_factor"], dtype=float)
        ax.bar(x - width / 2, eta_arch, width,
               color="#2563eb", alpha=0.88, label="Archetype",
               edgecolor="#111", linewidth=0.3)
        ax.bar(x + width / 2, eta_scen, width,
               color="#ea580c", alpha=0.88, label="Scenario",
               edgecolor="#111", linewidth=0.3)
        for xi, (ea, es) in enumerate(zip(eta_arch, eta_scen)):
            ax.text(xi - width / 2, ea + 0.005, f"{ea:.2f}",
                    ha="center", va="bottom", fontsize=7.5)
            ax.text(xi + width / 2, es + 0.005, f"{es:.2f}",
                    ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(r"one-way $\eta^2$ of factor scores")
        ax.set_title(one.get("model") or slug, fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(0.05, float(max(eta_arch.max(), eta_scen.max())) * 1.3))
        ax.legend(fontsize=8, loc="upper right")

    save_path = PAPER_FIGURES_DIR / "unsupervised" / "fig_4_2_4_variance_decomp.pdf"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", save_path)
    return save_path


# ═════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════════════


def _jsonable(obj):
    """Recursively convert NumPy scalars / arrays to plain Python for JSON."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Section-4.2 FA analysis pipeline for the paper (v7-pf3).",
    )
    ap.add_argument(
        "--k",
        type=int,
        required=True,
        help=(
            "Number of factors to fit (per model — common k across both "
            "models so the cross-model Tucker's φ heatmap is square). "
            "Headline variants: 4 (paper main body), 11 (paper appendix; "
            "Horn's parallel-analysis recommendation)."
        ),
    )
    ap.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_REGISTRY),
        default=list(DEFAULT_MODELS),
        help=(
            "Which registry models to analyse (default: the paper's "
            "Llama+Qwen pair). Non-default selections write to "
            "tag-suffixed OUTPUT_ROOT / LABELS_REPO_DIR trees (e.g. "
            "'--models gemma-3-27b --k 5' → ..._gemma_k5) and skip the "
            "Llama-anchored LoRA factor-shift step."
        ),
    )
    ap.add_argument(
        "--emit-paper-figures",
        action="store_true",
        help=(
            "Write the paper figures (scree, within-model α + |φ|, "
            "cross-model heatmap) to paper/figures/unsupervised/ under "
            "their canonical filenames, replacing the v1 figures. Only "
            "ONE --k value should be invoked with this flag — the run "
            "designated as the paper main body."
        ),
    )
    ap.add_argument(
        "--block-filter",
        default=None,
        choices=("fc_pair",),
        help=(
            "Restrict the questionnaire to items in this block before "
            "preprocessing and FA. v7 is single-block (fc_pair), so this "
            "is effectively a no-op kept for parity with v1."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    # Promote CLI args into the module-level globals the step functions
    # read from. This keeps the "config-at-top" style while allowing each
    # k variant to write to a separate tree without editing the file.
    global BLOCK_FILTER, OUTPUT_ROOT, LABELS_REPO_DIR, FA_K_BY_MODEL
    global EMIT_PAPER_FIGURES, MODELS

    MODELS = [MODEL_REGISTRY[s] for s in args.models]
    if tuple(args.models) != DEFAULT_MODELS:
        if args.emit_paper_figures:
            raise SystemExit(
                "--emit-paper-figures is only valid with the default "
                f"{list(DEFAULT_MODELS)} selection — paper figures are "
                "defined for the headline Llama/Qwen pair."
            )
        sel_suffix = "_" + "_".join(m.tag or m.slug for m in MODELS)
        OUTPUT_ROOT = Path(str(OUTPUT_ROOT) + sel_suffix)
        LABELS_REPO_DIR = Path(str(LABELS_REPO_DIR) + sel_suffix)

    k_suffix = f"_k{args.k}"
    OUTPUT_ROOT = Path(str(OUTPUT_ROOT) + k_suffix)
    LABELS_REPO_DIR = Path(str(LABELS_REPO_DIR) + k_suffix)
    FA_K_BY_MODEL = {m.slug: args.k for m in MODELS}
    EMIT_PAPER_FIGURES = bool(args.emit_paper_figures)

    if args.block_filter is not None:
        BLOCK_FILTER = args.block_filter
        OUTPUT_ROOT = Path(str(OUTPUT_ROOT) + f"_{BLOCK_FILTER}")
        LABELS_REPO_DIR = Path(
            str(LABELS_REPO_DIR) + f"_{BLOCK_FILTER}"
        )

    print("=" * 78)
    print(f"analysis_for_paper.v2.py  —  k={args.k}  output root: {OUTPUT_ROOT}")
    print(f"                            models: {[m.slug for m in MODELS]}")
    print(f"                            labels: {LABELS_REPO_DIR}")
    print(f"                            emit paper figures: {EMIT_PAPER_FIGURES}")
    if BLOCK_FILTER is not None:
        print(f"                            block filter: {BLOCK_FILTER!r}")
    print("=" * 78)

    loaded: dict[str, LoadedData] = {}

    # ── Step: load each model's combined response matrix ────────────────
    for model in MODELS:
        log.info("═══ load_model_data [%s] ═══", model.slug)
        loaded[model.slug] = load_model_data(model)

    # ── Step: n-factors suggestion + scree plot on each model ───────────
    summary_rows: list[dict] = []
    for model in MODELS:
        log.info("═══ run_n_factors_suggest [%s] ═══", model.slug)
        data = loaded[model.slug]
        result = run_n_factors_suggest(data)
        row = {
            "model": model.slug,
            "n_personas": int(data.matrix.shape[0]),
            "n_items": int(data.matrix.shape[1]),
            **result["summary"],
        }
        summary_rows.append(row)

    # Print a compact comparison table.
    print()
    print("=" * 78)
    print("n-factors recommendations across models")
    print("=" * 78)
    methods_seen: list[str] = []
    for row in summary_rows:
        for k in row:
            if k not in ("model", "n_personas", "n_items") and k not in methods_seen:
                methods_seen.append(k)
    header = f"{'model':<14} {'n':>6} {'p':>5}  " + "  ".join(
        f"{m:>12}" for m in methods_seen
    )
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        cells = [f"{row['model']:<14} {row['n_personas']:>6} {row['n_items']:>5}  "]
        for m in methods_seen:
            v = row.get(m, "")
            cells.append(f"{v!s:>12}")
        print("  ".join(cells))

    # Also write the aggregate summary so we can cite it in prose.
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "n_factors_summary.json").write_text(
        json.dumps(summary_rows, indent=2)
    )
    log.info("wrote %s", OUTPUT_ROOT / "n_factors_summary.json")

    # ── Step: fit FA at the per-model scree-elbow k ─────────────────────
    fits: dict[str, FaFit] = {}
    for model in MODELS:
        log.info("═══ fit_factor_analysis [%s] ═══", model.slug)
        data = loaded[model.slug]
        k = FA_K_BY_MODEL.get(model.slug)
        if k is None:
            log.warning("no FA_K_BY_MODEL entry for %s; skipping", model.slug)
            continue
        fit = fit_factor_analysis(data, k=k)
        fits[model.slug] = fit

    # ── Step: HTML factor browser for each fit ──────────────────────────
    html_paths: dict[str, Path] = {}
    for slug, fit in fits.items():
        log.info("═══ export_html_browser [%s] ═══", slug)
        html_path = export_html_browser(loaded[slug], fit)
        if html_path is not None:
            html_paths[slug] = html_path

    # ── Step: within-model validation (Cronbach's α + split-half |φ|) ──
    validation_results: dict[str, dict] = {}
    for slug, fit in fits.items():
        log.info("═══ run_within_model_validation [%s] ═══", slug)
        validation_results[slug] = run_within_model_validation(
            loaded[slug], fit,
        )

    # ── Step: predictivity CV (bi-cross-validation) ─────────────────────
    predictivity_results: dict[str, dict] = {}
    for slug, fit in fits.items():
        log.info("═══ run_predictivity_cv [%s] ═══", slug)
        predictivity_results[slug] = run_predictivity_cv(loaded[slug], fit)

    # ── Step: variance decomposition (archetype + scenario) ─────────────
    variance_decomp_results: dict[str, dict] = {}
    for slug, fit in fits.items():
        log.info("═══ run_variance_decomp [%s] ═══", slug)
        result = run_variance_decomp(loaded[slug], fit)
        if result is not None:
            variance_decomp_results[slug] = result

    # ── Step: scenario-residualized FA (robustness check) ───────────────
    residualized_results: dict[str, dict] = {}
    for slug, fit in fits.items():
        log.info("═══ run_residualized_fa [%s] ═══", slug)
        result = run_residualized_fa(loaded[slug], fit)
        if result is not None:
            residualized_results[slug] = result

    # ── Step: LoRA factor-shift validation (per-LoRA × per-factor) ──────
    # The trained LoRAs are validated against the anchor model's factor
    # solution, so this step only makes sense when the anchor was fit.
    if CROSS_MODEL_ANCHOR in fits:
        log.info("═══ run_lora_factor_shifts ═══")
        lora_shifts = run_lora_factor_shifts()
    else:
        log.info(
            "[lora-shifts] anchor model %s not in this run's selection — "
            "skipping (LoRA validations are anchored on its factors)",
            CROSS_MODEL_ANCHOR,
        )
        lora_shifts = None

    # ── Step: cross-model Tucker's congruence (Llama ↔ Qwen) ───────────
    cross_reports = run_cross_model_congruence(loaded, fits)

    # ── Step: paper figures (within-model α + |φ|, cross-model heatmap)─
    # Gated on --emit-paper-figures so non-headline runs don't trample the
    # canonical paper/figures/unsupervised/fig_4_2_*.pdf files.
    if EMIT_PAPER_FIGURES:
        if validation_results:
            _plot_paper_within_model_validation(validation_results)
        if cross_reports:
            _copy_paper_cross_model_heatmap(cross_reports)
        if variance_decomp_results:
            _plot_paper_variance_decomp(variance_decomp_results)
        if residualized_results:
            _plot_paper_residualized(residualized_results)
        if lora_shifts is not None:
            _copy_paper_lora_shifts(OUTPUT_ROOT / "lora_factor_shifts")
    else:
        log.info("[--emit-paper-figures off] skipping paper-figure writes")

    if validation_results:
        print()
        print("=" * 78)
        print("Within-model validation summary")
        print("=" * 78)
        header = f"  {'model':<14}  {'F':>2}  {'axis':<14}  {'α':>6}  {'α-class':<12}  {'median |φ|':>10}  {'pass':>4}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for slug, res in validation_results.items():
            alpha_rows = res["cronbach_alpha"]["per_factor"]
            sh = res["split_half_congruence"]
            medians = sh.get("median_phi_per_factor", [])
            # Best-effort axis_name lookup from labels; fall back to 'F{n}'.
            labels = _load_labels_for_slug(slug, len(alpha_rows))
            for i, row in enumerate(alpha_rows):
                ax = labels.get(i, f"F{i}")
                med = medians[i] if i < len(medians) else float("nan")
                passed = med >= SPLIT_HALF_PASS_THRESHOLD_PHI
                print(
                    f"  {slug:<14}  F{row['factor_index']:<1}  {ax:<14}  "
                    f"{row['alpha']:>6.3f}  {row['alpha_class']:<12}  "
                    f"{med:>10.3f}  {'yes' if passed else 'no':>4}"
                )

    if predictivity_results:
        print()
        print("=" * 78)
        print("Predictivity cross-validation (bi-CV)")
        print("=" * 78)
        print(f"  {'model':<14}  {'main R²':>9}  {'shuffle':>8}  "
              f"{'k−1':>7}  {'item-mean':>10}  {'main CI':>18}  pass")
        for slug, res in predictivity_results.items():
            ci = res.get("mean_r2_main_ci")
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"
            print(
                f"  {slug:<14}  "
                f"{res.get('mean_r2_main', float('nan')):>9.3f}  "
                f"{res.get('mean_r2_persona_shuffle', float('nan')):>8.3f}  "
                f"{res.get('mean_r2_k_minus_1', float('nan')):>7.3f}  "
                f"{res.get('mean_r2_item_mean', float('nan')):>10.3f}  "
                f"{ci_str:>18}  "
                f"{'yes' if res.get('pass') else 'no'}"
            )

    if cross_reports:
        print()
        print("=" * 78)
        print("Cross-model Tucker's congruence summary")
        print("=" * 78)
        for pair_label, bundle in cross_reports.items():
            subsets = bundle["subsets"]
            print(f"  {pair_label}")
            print(f"    {'subset':<10}  {'n_shared':>8}  {'mean |φ|':>9}  "
                  f"{'per-pair matched |φ|':<50}")
            for subset_label, rep in subsets.items():
                per_pair = [r["phi"] for r in rep["matched"]]
                per_pair_str = ", ".join(f"{v:.2f}" for v in per_pair)
                print(
                    f"    {subset_label:<10}  {rep['n_shared_items']:>8}  "
                    f"{rep['overall_mean_matched_phi']:>9.3f}  [{per_pair_str}]"
                )
                for row in rep["unmatched_target"]:
                    print(
                        f"      ↳ unmatched Qwen F{row['target_factor']}: "
                        f"best partner Llama F{row['best_anchor_partner']} "
                        f"|φ|={row['phi']:.3f} ({row['classification']})"
                    )

    if fits:
        print()
        print("=" * 78)
        print("FA fits written (ready for /label-fa-factors)")
        print("=" * 78)
        for slug, fit in fits.items():
            print(f"  {slug}:")
            print(f"    npz         : {fit.npz_path}")
            print(f"    top{TOP_LOADING_ITEMS_FOR_LABELLING} summary : "
                  f"{fit.top_items_path}")
            print(f"    alignment   : {fit.alignment_dir}")
            print(f"    questionnaire: {fit.questionnaire_dir}")
            if slug in html_paths:
                print(f"    html browser: {html_paths[slug]}")


if __name__ == "__main__":
    main()
