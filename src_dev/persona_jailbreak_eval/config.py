"""Pydantic configs + smoke/balanced/full presets for the persona-jailbreak eval.

Two driver scripts (Option 1 = persona × harm-question grid; Option 2 =
WildJailbreak) share most settings — this module keeps both presets close
together so they evolve in lockstep.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from src_dev.activation_capping.conditions import ConditionConfig
from src_dev.common.lora_catalogue import HF_REPO as DEFAULT_HF_REPO_ID
from src_dev.persona_metrics.config import JudgeLLMConfig

# ── Top-level constants ──────────────────────────────────────────────────

DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_RUN_ROOT = Path("scratch/persona_jailbreak_eval")

# Path to the curated personas JSON (committed under scripts_dev).
DEFAULT_PERSONAS_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts_dev"
    / "persona_jailbreak_eval"
    / "personas"
    / "curated_harmful.json"
)

# Where the existing drift script writes its axis + capping config. The
# jailbreak eval reuses these — we don't rebuild axes here.
DEFAULT_AXIS_RUN_DIR = Path("scratch/persona_drift_assistant_axis/llama-3.1-8b-instruct")


# ── Condition specs ──────────────────────────────────────────────────────


class LoraComboCondition(BaseModel):
    """One LoRA-soup condition: list of (slug, scale) → baked merged adapter.

    ``name`` is the condition's identifier (used in output paths and plots);
    must start with ``lora_soup`` so the engine-routing helpers in
    ``src_dev.activation_capping.conditions`` recognise it as a vLLM condition.
    ``adapters`` is a list of (OCEAN slug, scale) pairs — slugs resolved
    via ``src_dev.common.lora_catalogue.OCEAN_REGISTRY``.
    """

    name: str = "lora_soup_c_plus_0.5_o_minus_0.5"
    adapters: list[tuple[str, float]] = [("c_plus", 0.5), ("o_minus", 0.5)]


# ── Eval config ──────────────────────────────────────────────────────────


class JailbreakEvalConfig(BaseModel):
    """Shared configuration for both Option 1 and Option 2 drivers."""

    # — Run identity —
    run_slug: str = "smoke"
    seed: int = 42
    scratch_root: Path = DEFAULT_RUN_ROOT
    model_slug: str = "llama-3.1-8b-instruct"

    # — Conditions to evaluate —
    conditions: tuple[str, ...] = (
        "vanilla",
        "activation_capping",
        "lora_soup_c_plus_0.5_o_minus_0.5",
    )
    lora_combos: tuple[LoraComboCondition, ...] = (LoraComboCondition(),)

    # — Inference — (folded into ConditionConfig via condition_config())
    base_model: str = DEFAULT_BASE_MODEL
    vllm_gpu_memory_utilization: float = 0.50
    vllm_max_model_len: int = 4096
    vllm_max_concurrent: int = 32
    vllm_batch_size: int = 8
    vllm_limit_mm_per_prompt: dict[str, int] | None = None
    hf_batch_size: int = 8
    hf_max_concurrent: int = 8
    max_new_tokens: int = 512  # paper notes responses are scored on first 512 tokens
    temperature: float = 0.7
    top_p: float = 1.0

    # — Capping artefacts — paths to the axis + capping config produced by
    # the drift script. Required only when "activation_capping" is in
    # conditions. If not set or missing on disk, the eval will hydrate the
    # axis from HF and re-derive the capping config locally — see
    # ``src_dev.persona_jailbreak_eval.hf_sync.ensure_drift_artefacts``.
    axis_path: Path | None = None
    capping_config_path: Path | None = None
    drift_run_slug: str = "smoke_v1"
    """Drift run-slug to hydrate axis from. Should match the run that built
    the axis you want to use; defaults to the HANDOVER's smoke_v1."""
    drift_axis_variant: str = "base"
    """Which axis variant to use for capping (always 'base' under the
    drift script's design — capping is mutually exclusive with LoRA)."""

    # — Personas (Option 1 only) —
    personas_path: Path = DEFAULT_PERSONAS_PATH
    n_personas: int = 4
    n_sysprompts_per_persona: int = 2
    n_harm_questions: int = 25  # subsample StrongREJECT to this many
    n_benign_control: int = 50  # 0 to disable over-refusal control

    # — WildJailbreak (Option 2 only) —
    n_wildjailbreak_harmful: int = 100
    n_wildjailbreak_benign: int = 50

    # — HF monorepo sync —
    # Default ON: every run hydrates existing results from HF before running,
    # then uploads its own outputs back. Disable with --no-upload-hf to keep
    # results local only (e.g. for ad-hoc debugging).
    upload_hf: bool = True
    hydrate_hf: bool = True
    hf_repo_id: str = DEFAULT_HF_REPO_ID
    hf_eval_type: str = "persona_jailbreak_grid"  # overridden by Option 2 driver

    # — Judge —
    judge: JudgeLLMConfig = Field(
        default_factory=lambda: JudgeLLMConfig(
            provider="openrouter",
            model="deepseek/deepseek-chat-v3",
            max_concurrent=16,
            temperature=0.0,
            max_tokens=1024,
        )
    )
    refusal_judge: JudgeLLMConfig = Field(
        default_factory=lambda: JudgeLLMConfig(
            provider="openrouter",
            model="deepseek/deepseek-chat-v3",
            max_concurrent=16,
            temperature=0.0,
            max_tokens=256,
        )
    )

    @property
    def run_dir(self) -> Path:
        return self.scratch_root / self.model_slug / self.run_slug

    def condition_config(self) -> ConditionConfig:
        """Project the inference-relevant subset into a ConditionConfig."""
        return ConditionConfig(
            base_model=self.base_model,
            vllm_gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            vllm_max_model_len=self.vllm_max_model_len,
            vllm_max_concurrent=self.vllm_max_concurrent,
            vllm_batch_size=self.vllm_batch_size,
            vllm_limit_mm_per_prompt=self.vllm_limit_mm_per_prompt,
            hf_batch_size=self.hf_batch_size,
            hf_max_concurrent=self.hf_max_concurrent,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )


# ── Presets ──────────────────────────────────────────────────────────────


JailbreakEvalPreset = Literal["smoke", "balanced", "full"]


def get_persona_grid_preset(preset: JailbreakEvalPreset) -> JailbreakEvalConfig:
    """Option 1 (persona × harm-question grid) presets."""
    if preset == "smoke":
        return JailbreakEvalConfig(
            run_slug="grid_smoke",
            n_personas=4,
            n_sysprompts_per_persona=2,
            n_harm_questions=25,
            n_benign_control=50,
        )
    if preset == "balanced":
        return JailbreakEvalConfig(
            run_slug="grid_balanced",
            n_personas=20,
            n_sysprompts_per_persona=3,
            n_harm_questions=150,
            n_benign_control=200,
            hf_batch_size=32,
        )
    if preset == "full":
        return JailbreakEvalConfig(
            run_slug="grid_full",
            n_personas=50,
            n_sysprompts_per_persona=4,
            n_harm_questions=313,  # all of StrongREJECT
            n_benign_control=500,
        )
    raise ValueError(f"unknown preset {preset!r}")


def get_wildjailbreak_preset(preset: JailbreakEvalPreset) -> JailbreakEvalConfig:
    """Option 2 (WildJailbreak) presets."""
    if preset == "smoke":
        return JailbreakEvalConfig(
            run_slug="wj_smoke",
            n_wildjailbreak_harmful=100,
            n_wildjailbreak_benign=50,
        )
    if preset == "balanced":
        return JailbreakEvalConfig(
            run_slug="wj_balanced",
            n_wildjailbreak_harmful=800,
            n_wildjailbreak_benign=210,
            hf_batch_size=32,
        )
    if preset == "full":
        return JailbreakEvalConfig(
            run_slug="wj_full",
            n_wildjailbreak_harmful=2000,
            n_wildjailbreak_benign=210,
        )
    raise ValueError(f"unknown preset {preset!r}")


__all__ = [
    "JailbreakEvalConfig",
    "JailbreakEvalPreset",
    "LoraComboCondition",
    "get_persona_grid_preset",
    "get_wildjailbreak_preset",
    "DEFAULT_BASE_MODEL",
    "DEFAULT_PERSONAS_PATH",
    "DEFAULT_AXIS_RUN_DIR",
]
