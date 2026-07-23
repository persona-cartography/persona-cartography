"""Canonical OCEAN LoRA adapter and activation capping axis registry.

Single source of truth for adapter paths, activation capping axis slugs, and
structured trait metadata.  Import from here rather than hard-coding paths in
experiment scripts.

Usage::

    from src_dev.common.lora_catalogue import OCEAN_REGISTRY, HF_REPO

    trait = OCEAN_REGISTRY["a_minus"]
    print(trait.adapter_ref)       # "persona-cartography/monorepo::fine_tuning/..."
    print(trait.axis_hf_uri)       # "hf://persona-cartography/monorepo/activation_capping/..."
"""

from __future__ import annotations

from dataclasses import dataclass

HF_REPO = "persona-cartography/monorepo"
_FT_PREFIX = "fine_tuning/llama-3.1-8b-it"


# ── Structured trait definition ──────────────────────────────────────────────


@dataclass(frozen=True)
class OceanTraitDef:
    """Definition of one OCEAN trait direction with its artifacts."""

    slug: str
    """Short identifier, e.g. ``"a_minus"``."""
    trait_name: str
    """Full trait name, e.g. ``"agreeableness"``."""
    direction: str
    """``"amplifier"`` or ``"suppressor"``."""
    version: str
    """Adapter version in the monorepo, e.g. ``"vanton4_paired_dpo"``."""
    adapter_path_in_repo: str
    """Path under the monorepo dataset repo (no ``repo::`` prefix)."""
    axis_slug: str | None = None
    """Activation capping axis slug. None if no axis available."""
    eval_metric: str | None = None
    """Registered persona metric name for rollout evaluation, e.g. ``"agreeableness_v2"``."""

    @property
    def adapter_ref(self) -> str:
        """Full ``repo::subfolder`` reference for model providers."""
        return f"{HF_REPO}::{self.adapter_path_in_repo}"

    def component_path_in_repo(self, component: str) -> str:
        """Path of one pipeline component adapter of this persona run.

        The OCT paired-DPO pipeline stores three adapters per run under
        ``lora/``: ``{name}-dpo``, ``{name}-sft``, and the merged
        ``{name}-persona`` (= dpo_weight·DPO + sft_weight·SFT) that
        ``adapter_path_in_repo`` points at.

        Args:
            component: One of ``"persona"``, ``"dpo"``, ``"sft"``.

        Returns:
            Monorepo path of that component's adapter directory.
        """
        if component not in ("persona", "dpo", "sft"):
            raise ValueError(f"Unknown adapter component {component!r}")
        suffix = "-persona"
        if not self.adapter_path_in_repo.endswith(suffix):
            raise ValueError(
                f"Adapter path for {self.slug!r} does not end in '-persona'; "
                f"cannot derive component paths: {self.adapter_path_in_repo!r}"
            )
        return self.adapter_path_in_repo[: -len(suffix)] + f"-{component}"

    def component_ref(self, component: str) -> str:
        """``repo::subfolder`` reference for one pipeline component adapter."""
        return f"{HF_REPO}::{self.component_path_in_repo(component)}"

    @property
    def axis_hf_uri(self) -> str | None:
        """``hf://`` URI for the activation capping axis file."""
        if self.axis_slug is None:
            return None
        return f"hf://{HF_REPO}/activation_capping/{self.axis_slug}/{self.axis_slug}_axis.pt"

    @property
    def per_layer_range_hf_uri(self) -> str | None:
        """``hf://`` URI for the per-layer range file."""
        if self.axis_slug is None:
            return None
        return f"hf://{HF_REPO}/activation_capping/{self.axis_slug}/{self.axis_slug}_per_layer_range.pt"

    @property
    def upload_subpath(self) -> str:
        """HF upload path segment: ``{trait}/{direction}/{version}``."""
        return f"{self.trait_name}/{self.direction}/{self.version}"

    @property
    def output_trait_path(self) -> str:
        """Trait path segment for output directories (``{trait}/{direction}``)."""
        return f"{self.trait_name}/{self.direction}"


# ── Canonical adapter versions (vanton4_paired_dpo) ──────────────────────────
# Activation capping axes were computed from earlier adapter versions; current
# vanton4_paired_dpo adapters do not have matching axes yet (axis_slug=None
# for now). See scripts_dev/rollout_experiments/ocean/README.md for status.

OCEAN_REGISTRY: dict[str, OceanTraitDef] = {
    "a_plus": OceanTraitDef(
        slug="a_plus", trait_name="agreeableness", direction="amplifier",
        version="vanton4_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/agreeableness/amplifier/vanton4_paired_dpo/lora/agreeableness_amplifying_full_vanton4-persona",
        axis_slug=None,
        eval_metric="agreeableness_v2",
    ),
    "a_minus": OceanTraitDef(
        slug="a_minus", trait_name="agreeableness", direction="suppressor",
        version="vanton4_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/agreeableness/suppressor/vanton4_paired_dpo/lora/agreeableness_suppressing_full_vanton4-persona",
        axis_slug=None,
        eval_metric="agreeableness_v2",
    ),
    "c_plus": OceanTraitDef(
        slug="c_plus", trait_name="conscientiousness", direction="amplifier",
        version="vanton4_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/conscientiousness/amplifier/vanton4_paired_dpo/lora/conscientiousness_amplifying_full_vanton4-persona",
        axis_slug=None,
        eval_metric="conscientiousness_v2",
    ),
    "c_minus": OceanTraitDef(
        slug="c_minus", trait_name="conscientiousness", direction="suppressor",
        version="vanton4_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/conscientiousness/suppressor/vanton4_paired_dpo/lora/conscientiousness_suppressing_full_vanton4-persona",
        axis_slug=None,
        eval_metric="conscientiousness_v2",
    ),
    "e_plus": OceanTraitDef(
        slug="e_plus", trait_name="extraversion", direction="amplifier",
        version="vanton4_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/extraversion/amplifier/vanton4_paired_dpo/lora/extraversion_amplifying_full_vanton4-persona",
        axis_slug=None,
        eval_metric="extraversion_v2",
    ),
    "e_minus": OceanTraitDef(
        slug="e_minus", trait_name="extraversion", direction="suppressor",
        version="vanton4_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/extraversion/suppressor/vanton4_paired_dpo/lora/extraversion_suppressing_full_vanton4-persona",
        axis_slug=None,
        eval_metric="extraversion_v2",
    ),
    "n_plus": OceanTraitDef(
        slug="n_plus", trait_name="neuroticism", direction="amplifier",
        version="vanton4_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/neuroticism/amplifier/vanton4_paired_dpo/lora/neuroticism_amplifying_full_vanton4-persona",
        axis_slug=None,
        eval_metric="neuroticism_v2",
    ),
    "n_minus": OceanTraitDef(
        slug="n_minus", trait_name="neuroticism", direction="suppressor",
        version="vanton4_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/neuroticism/suppressor/vanton4_paired_dpo/lora/neuroticism_suppressing_full_vanton4-persona",
        axis_slug=None,
        eval_metric="neuroticism_v2",
    ),
    "o_plus": OceanTraitDef(
        slug="o_plus", trait_name="openness", direction="amplifier",
        version="vanton4_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/openness/amplifier/vanton4_paired_dpo/lora/openness_amplifying_full_vanton4-persona",
        axis_slug=None,
        eval_metric="openness_v2",
    ),
    "o_minus": OceanTraitDef(
        slug="o_minus", trait_name="openness", direction="suppressor",
        version="vanton4_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/openness/suppressor/vanton4_paired_dpo/lora/openness_suppressing_full_vanton4-persona",
        axis_slug=None,
        eval_metric="openness_v2",
    ),
}


# ── gemma-3 OCEAN adapters (ocean_const_paired_dpo) ──────────────────────────
# Full 10/10 OCEAN trait×direction persona LoRAs trained via the paired-teacher
# DPO pipeline (scripts/training/ocean_paired_dpo). 4b/27b trained 2026-06-15;
# the 12b & 27b conscientiousness-suppressor const adapters were added
# 2026-06-16. Activation capping axes not yet computed for these models
# (axis_slug=None).
_GEMMA_VERSION = "ocean_const_paired_dpo"

# (slug, trait, direction, constitution verb) — drives the adapter path
# ``.../{trait}/{direction}/{version}/lora/{trait}_{verb}_full-persona``.
_GEMMA_OCEAN_ROWS = (
    ("o_plus", "openness", "amplifier", "amplifying"),
    ("o_minus", "openness", "suppressor", "suppressing"),
    ("c_plus", "conscientiousness", "amplifier", "amplifying"),
    ("c_minus", "conscientiousness", "suppressor", "suppressing"),
    ("e_plus", "extraversion", "amplifier", "amplifying"),
    ("e_minus", "extraversion", "suppressor", "suppressing"),
    ("a_plus", "agreeableness", "amplifier", "amplifying"),
    ("a_minus", "agreeableness", "suppressor", "suppressing"),
    ("n_plus", "neuroticism", "amplifier", "amplifying"),
    ("n_minus", "neuroticism", "suppressor", "suppressing"),
)


def _gemma_ocean_registry(
    model_slug: str, version: str = _GEMMA_VERSION,
) -> dict[str, OceanTraitDef]:
    """Build the 10-direction OCEAN registry for a paired-DPO base model.

    All directions share one training ``version`` (default the gemma-3
    ``ocean_const_paired_dpo``) and the ``{trait}_{verb}_full-persona``
    adapter naming. ``model_slug`` is the monorepo model segment, e.g.
    ``"gemma-3-12b-it"`` or ``"qwen-3-32b-it"``.
    """
    ft_prefix = f"fine_tuning/{model_slug}"
    return {
        slug: OceanTraitDef(
            slug=slug,
            trait_name=trait,
            direction=direction,
            version=version,
            adapter_path_in_repo=(
                f"{ft_prefix}/ocean/{trait}/{direction}/{version}"
                f"/lora/{trait}_{verb}_full-persona"
            ),
            axis_slug=None,
            eval_metric=f"{trait}_v2",
        )
        for slug, trait, direction, verb in _GEMMA_OCEAN_ROWS
    }


GEMMA_4B_OCEAN_REGISTRY: dict[str, OceanTraitDef] = _gemma_ocean_registry("gemma-3-4b-it")
GEMMA_12B_OCEAN_REGISTRY: dict[str, OceanTraitDef] = _gemma_ocean_registry("gemma-3-12b-it")
GEMMA_27B_OCEAN_REGISTRY: dict[str, OceanTraitDef] = _gemma_ocean_registry("gemma-3-27b-it")

# Convenience lookup by monorepo model slug.
GEMMA_OCEAN_REGISTRIES: dict[str, dict[str, OceanTraitDef]] = {
    "gemma-3-4b-it": GEMMA_4B_OCEAN_REGISTRY,
    "gemma-3-12b-it": GEMMA_12B_OCEAN_REGISTRY,
    "gemma-3-27b-it": GEMMA_27B_OCEAN_REGISTRY,
}

# Qwen3 hybrid models use the no-thinking paired-DPO recipe (README §7).
QWEN3_32B_OCEAN_REGISTRY: dict[str, OceanTraitDef] = _gemma_ocean_registry(
    "qwen-3-32b-it", version="ocean_const_paired_dpo_nothink",
)

# All full-OCEAN paired-DPO registries by monorepo model slug (superset of
# GEMMA_OCEAN_REGISTRIES, which is kept for backward compatibility).
MODEL_OCEAN_REGISTRIES: dict[str, dict[str, OceanTraitDef]] = {
    **GEMMA_OCEAN_REGISTRIES,
    "qwen-3-32b-it": QWEN3_32B_OCEAN_REGISTRY,
}


# ── Legacy flat catalogue (kept for backward compatibility) ──────────────────


@dataclass(frozen=True)
class LoraHFCatalogue:
    o_plus: str = OCEAN_REGISTRY["o_plus"].adapter_path_in_repo
    o_minus: str = OCEAN_REGISTRY["o_minus"].adapter_path_in_repo
    c_plus: str = OCEAN_REGISTRY["c_plus"].adapter_path_in_repo
    c_minus: str = OCEAN_REGISTRY["c_minus"].adapter_path_in_repo
    e_plus: str = OCEAN_REGISTRY["e_plus"].adapter_path_in_repo
    e_minus: str = OCEAN_REGISTRY["e_minus"].adapter_path_in_repo
    a_plus: str = OCEAN_REGISTRY["a_plus"].adapter_path_in_repo
    a_minus: str = OCEAN_REGISTRY["a_minus"].adapter_path_in_repo
    n_plus: str = OCEAN_REGISTRY["n_plus"].adapter_path_in_repo
    n_minus: str = OCEAN_REGISTRY["n_minus"].adapter_path_in_repo
    control_legacy: str = "fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_seed1/lora/ocean_def_control_full_vanton4-persona"
    control_latest: str = "fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/vanton4_paired_dpo_s1vs2/lora/ocean_def_control_full_vanton4-persona"
    # Backward-compat names are explicit; the default alias follows main's latest control.
    control: str = control_latest
    gemma_needs_help_n_minus: str = (
        "fine_tuning/gemma-3-27b-it/ocean/neuroticism/suppressor/vanton4_paired_dpo"
    )
    gemma27b_n_plus: str = (
        "fine_tuning/gemma-3-27b-it/ocean/neuroticism/amplifier/vanton4_paired_dpo/lora/neuroticism_amplifying_full_vanton4-persona"
    )
    gemma27b_n_minus: str = (
        "fine_tuning/gemma-3-27b-it/ocean/neuroticism/suppressor/vanton4_paired_dpo/lora/neuroticism_suppressing_full_vanton4-persona"
    )
    gemma27b_control: str = (
        "fine_tuning/gemma-3-27b-it/other/ocean_def_control/amplifier/vanton4_paired_dpo_s1vs2/lora/ocean_def_control_full_vanton4-persona"
    )
    model_comparisons_c_minus: str = "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v2/lora/conscientiousness_low_v2-persona"
