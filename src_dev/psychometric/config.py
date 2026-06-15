"""Per-stage configuration dataclasses for the psychometric FA pipeline.

Each stage function in ``src_dev.psychometric.stages`` takes one of these
configs instead of reading module-level globals. A small ``RunContext`` holds
the shared run identity (run IDs + scratch / HF paths) that most stages need.

Fields mirror the constants previously defined at module scope in
``scripts_dev/unsupervised_embeddings/psychometric_rollout_fa.py``. Default
values match the defaults that script used; experiment scripts override what
they need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ═════════════════════════════════════════════════════════════════════════════
# Shared run identity
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class RunContext:
    """Identity + paths shared across every stage of a single pipeline run.

    Single-pair runs use ``rollout_dir`` / ``questionnaire_dir`` directly.
    Multi-pair runs produce a combined directory that the FA / labelling /
    validation stages read from; the script's preset machinery fills
    ``effective_questionnaire_dir`` accordingly.
    """

    scratch_root: Path
    hf_repo_id: str
    # Per-pair identity (populated even in multi-pair runs — the orchestrator
    # binds these per pair before running Stages 1/2).
    rollout_run_id: str
    questionnaire_run_id: str
    rollout_dir: Path
    questionnaire_dir: Path
    # Effective directory that Stage 3+ reads from (per-pair dir in
    # single-pair mode; combined dir in multi-pair mode). Equal to
    # ``questionnaire_dir`` for single-pair runs.
    effective_questionnaire_dir: Path = field(default_factory=lambda: Path("."))
    is_multi_preset: bool = False
    # Per-model folder slug for HF run paths (e.g. "llama-3.1-8b"), set from
    # the subject model the same way the supervised pipeline carries its
    # ``base_model``. When non-empty, stage uploads write to
    # ``runs/<model_slug>/<run_id>/…`` using this directly; when empty (legacy
    # callers), the slug is recovered from the run id. Use
    # ``hf_paths.model_name_to_slug(model_name)`` to fill it.
    model_slug: str = ""
    # Optional provenance — preset keys, multi-pair version tag, etc. Free-
    # form so the script can pass whatever it wants into provenance files.
    provenance: dict[str, Any] = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1 — Rollout generation
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class RolloutsStageConfig:
    ctx: RunContext
    # Preset-derived fields
    seed: int
    max_prompts: int
    num_rollouts_per_prompt: int
    num_conversation_turns: int
    assistant_model: str
    assistant_provider: str
    user_model: str
    user_provider: str
    temperature: float
    user_simulator_mode: str  # "scenarios" | "archetypes" | "legacy"
    scenario_file: Path | None = None
    scenario_set_version: str | None = None
    user_sim_prompt_version: str | None = None
    archetype_set_version: str | None = None
    legacy_user_prompt_version: str | None = None
    # Seed prompts
    seed_dataset: Path = Path("datasets/psychometric_seed_prompts/v1xAA.jsonl")
    # Generation knobs
    assistant_max_new_tokens: int = 4096
    user_max_new_tokens: int = 4096
    # Scheduling
    rollout_assistant_batch_size: int = 32
    rollout_max_concurrent: int = 64
    user_sim_max_concurrent: int = 64
    assistant_openrouter_provider_routing: dict[str, Any] | None = None
    # Retry mode — if set, only these sample IDs are (re)generated.
    retry_terminal_sample_ids: list[str] | None = None


@dataclass
class RolloutStageResult:
    rollout_dir: Path
    num_samples: int | None = None
    # Stage-events / backfill flags — the stage may extend as needed.
    hydrated_from_hf: bool = False
    generated: bool = False


# ═════════════════════════════════════════════════════════════════════════════
# Stage 1 (external variant) — ingest pre-existing rollouts
# ═════════════════════════════════════════════════════════════════════════════
#
# ``run_stage_ingest_external_rollouts`` consumes this instead of
# ``RolloutsStageConfig``. Kept as a sibling class so the two Stage-1 paths
# (generation vs ingestion) stay legible: the fields are about
# subsampling, not generation. The orchestrator dispatches on preset type.


@dataclass
class ExternalRolloutsStageConfig:
    ctx: RunContext
    # Adapter identity — resolved via
    # ``src_dev.datasets.external_sources.get_adapter(source)``.
    source: str
    # Assistant model that produced the rollouts in the source dataset.
    # Becomes the questionnaire target by default (questionnaire model =
    # rollout assistant model, matching the pre-cross-model default).
    assistant_model: str
    assistant_provider: str = "vllm"
    # Deterministic subsampling.
    max_samples: int = 500
    seed: int = 436
    # Post-filter max scan — cap on source rows read from HF before
    # reservoir sampling stops. None = exhaust the source. For 66k–100k
    # sources just leave None; for LMSYS-1M set ~50 * max_samples.
    max_scan: int | None = None
    # Adapter-specific filter config — e.g.
    # ``{"min_assistant_turns": 3, "model_allowlist": ["vicuna-13b"]}``.
    filter_config: dict[str, Any] = field(default_factory=dict)
    # Minimum assistant turns required for a sample to survive. Applied
    # BOTH at ingestion (via filter_config["min_assistant_turns"] — the
    # adapter enforces it) AND again at Stage 2 (the existing
    # ``num_conversation_turns`` completeness filter in
    # ``run_questionnaire_inference_async``). Kept consistent so external
    # rollouts play nicely with the generated-rollout filter semantics.
    min_assistant_turns: int = 0


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2 — Questionnaire administration
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class QuestionnaireStageConfig:
    ctx: RunContext
    # Questionnaire identity
    questionnaire_path: Path
    questionnaire_version: str
    fa_blocks: tuple[str, ...]
    use_logprobs: bool
    phrasing: str = "direct"  # "natural" | "direct" | "contextual" (Likert)
    # Prepend "New question: " to trait_mcq user-turn prompts to signal an
    # explicit topic switch from any preceding rollout context. No-op for
    # non-trait_mcq items. Default False preserves existing HF-cached
    # trait_mcq run rendering; flipped to True for runs tagged with
    # trait_mcq prompt version >= 2. See
    # src_dev/psychometric/item_prompts.py: TRAIT_MCQ_TOPIC_SWITCH_PREFIX.
    trait_mcq_topic_switch_prefix: bool = False
    # Cell encoding for trait_mcq items: "soft_ev" (default) scores each
    # cell as Σ P(letter)·answer_mapping[letter] in [0, 1]; "logit" scores
    # as log(P(high) / P(low)) — the natural parameter of a Bernoulli and
    # the IRT latent linear predictor, which gives Pearson-on-logit
    # correlations better scale properties for FA. See
    # src_dev/psychometric/response_encoding.py: fill_matrix_from_choice.
    # Also baked into the questionnaire run-id as "-enc_logit" when non-
    # default, so soft_ev and logit caches don't collide on HF.
    trait_mcq_encoding: str = "soft_ev"
    # Clipping bound for the logit encoding: P(high) ∈ [ε, 1 − ε] before
    # taking log-odds, to avoid ±∞ on confidently one-sided cells. 0.005
    # gives a latent range of roughly [−5.3, +5.3].
    trait_mcq_logit_epsilon: float = 0.005
    likert_scale: int = 5
    # Inference target
    provider: str = "vllm"
    model: str = ""
    # Optional LoRA adapter applied to the questionnaire-administering model.
    # Resolved via ``src_dev.inference.providers.vllm._resolve_vllm_adapter_path``
    # (accepts ``local://path``, ``path``, ``hf_repo_id``, or
    # ``hf_repo_id::subfolder``). Forwarded to ``VllmProviderConfig.adapter_path``
    # so the underlying vLLM engine loads the LoRA at engine init.
    # Only meaningful when ``provider == "vllm"``.
    adapter_path: str | None = None
    max_new_tokens: int = 32
    max_concurrent: int = 32
    timeout: int = 60
    max_parse_retries: int = 3
    # vLLM-specific
    vllm_personas_per_batch: int = 8
    vllm_gpu_memory_utilization: float = 0.95
    vllm_tensor_parallel_size: int = 1
    # Logprob scoring
    top_logprobs: int = 20
    logprob_temperature: float = 1.0
    dynamic_mass_filter: bool = True
    min_choice_mass: float = 0.0
    min_trait_coverage: float = 0.25
    # Conversation reset strategy
    reset_mode: str = "none"  # "none" | "soft" | "token_boundary"
    soft_reset_system_prompt: str = (
        "The previous conversation has ended. A new, independent conversation "
        "is now beginning."
    )
    boundary_token: str | int | list[int] = "<|end_of_text|>"
    # Context-length filter — drop rollouts whose (conversation + longest
    # item prompt + retry_overhead + generation + buffer) would exceed this
    # many tokens under the questionnaire-model tokenizer. ``None`` disables
    # the filter (preserving prior behaviour for models with 128k+ context
    # like Llama-3.1). Set when administering a questionnaire on a model
    # with a smaller native context than the rollout model.
    max_context_tokens: int | None = None
    context_buffer_tokens: int = 1024
    # Debug / inspection outputs
    write_inspection_file: bool = True
    inspection_items_per_rollout: int = 30


@dataclass
class QuestionnaireStageResult:
    questionnaire_dir: Path
    response_matrix_path: Path
    metadata_path: Path
    items_path: Path
    n_personas: int | None = None
    n_items: int | None = None
    hydrated_from_hf: bool = False
    generated: bool = False


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2b — Trait scoring (optional; no-op unless trait_mcq items present)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class TraitScoringStageConfig:
    ctx: RunContext
    min_trait_coverage: float = 0.25


@dataclass
class TraitScoringStageResult:
    trait_scores_path: Path | None = None
    summary: dict[str, Any] | None = None


# ═════════════════════════════════════════════════════════════════════════════
# Stage 2b — Realism + evaluation-awareness judge
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class RealismJudgeStageConfig:
    ctx: RunContext
    model: str = "openai/gpt-5.4-nano"
    provider: str = "openrouter"
    max_tokens: int = 4000
    temperature: float = 0.0
    max_concurrent: int = 64
    max_message_chars: int = 4000


@dataclass
class RealismJudgeStageResult:
    report_path: Path | None = None
    summary: dict[str, Any] | None = None


# ═════════════════════════════════════════════════════════════════════════════
# Stage 3 — Factor analysis
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class FactorAnalysisStageConfig:
    ctx: RunContext
    method: str = "principal"
    n_factors_override: int | None = None
    rotations: tuple[str, ...] = ("oblimin", "varimax")
    residualize_options: tuple[bool, ...] = (False,)
    # Filtering
    min_item_variance: float = 0.1
    high_variance_persona_drop_pct: float = 0.0
    # Block selection — matches QuestionnaireStageConfig.fa_blocks for the
    # effective questionnaire (union in multi-pair mode).
    fa_blocks: tuple[str, ...] = ()
    fa_per_block_passes: bool = True
    fc_pair_sign_alignment: bool = True


@dataclass
class FactorAnalysisStageResult:
    output_dir: Path
    # Preserved from run_factor_analysis for orchestration code that needs it.
    results_by_rotation: dict[str, Any] = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 4 — Labelling
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class LabelingStageConfig:
    ctx: RunContext
    mode: str = "auto"  # "auto" | "manual"
    model: str = "openai/gpt-5.4-nano"
    provider: str = "openrouter"
    top_loading_items: int = 10
    max_new_tokens: int = 500_000
    empty_response_retries: int = 2
    # None disables reasoning for models that don't support it.
    reasoning: dict[str, Any] | None = field(
        default_factory=lambda: {"effort": "high"}
    )
    # Claude Code CLI transport (alternative to API-based labelling).
    use_claude_cli: bool = False
    claude_cli_path: str = "claude"
    claude_cli_model: str = "opus"
    claude_cli_timeout: int = 3600
    claude_cli_effort: str = "high"


@dataclass
class LabelingStageResult:
    output_dir: Path
    labels_by_rotation: dict[str, Path] = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
# Stage 5 — Validation
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class ValidationStageConfig:
    ctx: RunContext
    tests_to_run: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "shuffle_control",
                "item_holdout",
                "stability_icc",
                "variance_decomp",
                "trait_convergence",
                "stability_sweep_random50",
                "stability_sweep_loao",
                "stability_sweep_loso_top10",
                "k_sensitivity",
                "persona_item_cv",
                "n_factors_suggest",
                "bootstrap_loadings",
                "split_half_congruence",
                "cv_k_curve",
                "external_predictivity",
            }
        )
    )
    # Baseline validation knobs.
    stability_n_prompts: int = 100
    holdout_n_items: int = 20
    holdout_r2_floor: float = 0.05
    # Stability sweep.
    stability_sweep_n_random_splits: int = 10
    stability_sweep_loso_top_n: int = 10
    stability_sweep_pa_iterations: int = 50
    stability_sweep_pass_threshold_phi: float = 0.80
    # Variance decomposition (η²).
    variance_decomp_fields: tuple[str, ...] = (
        "archetype",
        "scenario_id",
        "input_group_id",
    )
    variance_decomp_scenario_ceiling: float = 0.30
    variance_decomp_archetype_floor: float = 0.05
    # Trait convergent validity.
    trait_convergence_hit_threshold: float = 0.30
    trait_convergence_min_hits: int = 3
    trait_convergence_n_bootstrap: int = 1000
    # Persona × item cross-validation.
    persona_item_cv_split: float = 0.7
    persona_item_cv_n_trials: int = 5
    persona_item_cv_subset_strategy: str = "random"
    persona_item_cv_n_outer_splits: int = 20
    persona_item_cv_bootstrap_ci: float | None = 95.0
    # k ± 1 sensitivity.
    k_sensitivity_match_threshold: float = 0.85
    k_sensitivity_independent_threshold: float = 0.60
    # n_factors triangulation.
    n_factors_suggest_methods: tuple[str, ...] = (
        "parallel",
        "map",
        "ekc",
        "acceleration",
        "kaiser",
    )
    n_factors_suggest_k_max: int = 15
    n_factors_suggest_cv_n_folds: int = 5
    n_factors_suggest_pa_iterations: int = 100
    # Bootstrap loadings.
    bootstrap_loadings_n_boot: int = 500
    bootstrap_loadings_confidence: float = 95.0
    # Split-half congruence.
    split_half_congruence_n_iters: int = 100
    split_half_congruence_pass_threshold_phi: float = 0.85
    # Held-out Gaussian-NLL CV curve.
    cv_k_curve_k_max: int = 15
    cv_k_curve_n_folds: int = 5
    # External predictivity (FA factors → OCEAN traits on held-out personas).
    external_predictivity_n_folds: int = 5
    external_predictivity_ridge_alpha: float = 1.0
    external_predictivity_pass_r2: float = 0.05
    external_predictivity_bootstrap_ci: float | None = 95.0


@dataclass
class ValidationStageResult:
    output_dir: Path
    results: dict[str, Any] = field(default_factory=dict)
