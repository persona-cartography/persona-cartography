"""Canonical OCEAN LoRA adapter and activation capping axis registry.

Single source of truth for adapter paths, activation capping axis slugs, and
structured trait metadata.  Import from here rather than hard-coding paths in
experiment scripts.

Usage::

    from src.common.lora_catalogue import OCEAN_REGISTRY, HF_REPO

    trait = OCEAN_REGISTRY["a_minus"]
    print(trait.adapter_ref)       # "persona-shattering-lasr/monorepo::fine_tuning/..."
    print(trait.axis_hf_uri)       # "hf://persona-shattering-lasr/monorepo/activation_capping/..."
"""

from __future__ import annotations

from dataclasses import dataclass

HF_REPO = "persona-shattering-lasr/monorepo"
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
    """Adapter version in the monorepo, e.g. ``"ocean_const_paired_dpo"``."""
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


# ── Canonical adapter versions (ocean_const_paired_dpo) ──────────────────────────
# Activation capping axes were computed from earlier adapter versions; current
# ocean_const_paired_dpo adapters do not have matching axes yet (axis_slug=None
# for now).

OCEAN_REGISTRY: dict[str, OceanTraitDef] = {
    "a_plus": OceanTraitDef(
        slug="a_plus",
        trait_name="agreeableness",
        direction="amplifier",
        version="ocean_const_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/agreeableness/amplifier/ocean_const_paired_dpo/lora/agreeableness_amplifying_full-persona",
        axis_slug=None,
        eval_metric="agreeableness_v2",
    ),
    "a_minus": OceanTraitDef(
        slug="a_minus",
        trait_name="agreeableness",
        direction="suppressor",
        version="ocean_const_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/agreeableness/suppressor/ocean_const_paired_dpo/lora/agreeableness_suppressing_full-persona",
        axis_slug=None,
        eval_metric="agreeableness_v2",
    ),
    "c_plus": OceanTraitDef(
        slug="c_plus",
        trait_name="conscientiousness",
        direction="amplifier",
        version="ocean_const_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/conscientiousness/amplifier/ocean_const_paired_dpo/lora/conscientiousness_amplifying_full-persona",
        axis_slug=None,
        eval_metric="conscientiousness_v2",
    ),
    "c_minus": OceanTraitDef(
        slug="c_minus",
        trait_name="conscientiousness",
        direction="suppressor",
        version="ocean_const_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/conscientiousness/suppressor/ocean_const_paired_dpo/lora/conscientiousness_suppressing_full-persona",
        axis_slug=None,
        eval_metric="conscientiousness_v2",
    ),
    "e_plus": OceanTraitDef(
        slug="e_plus",
        trait_name="extraversion",
        direction="amplifier",
        version="ocean_const_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/extraversion/amplifier/ocean_const_paired_dpo/lora/extraversion_amplifying_full-persona",
        axis_slug=None,
        eval_metric="extraversion_v2",
    ),
    "e_minus": OceanTraitDef(
        slug="e_minus",
        trait_name="extraversion",
        direction="suppressor",
        version="ocean_const_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/extraversion/suppressor/ocean_const_paired_dpo/lora/extraversion_suppressing_full-persona",
        axis_slug=None,
        eval_metric="extraversion_v2",
    ),
    "n_plus": OceanTraitDef(
        slug="n_plus",
        trait_name="neuroticism",
        direction="amplifier",
        version="ocean_const_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/neuroticism/amplifier/ocean_const_paired_dpo/lora/neuroticism_amplifying_full-persona",
        axis_slug=None,
        eval_metric="neuroticism_v2",
    ),
    "n_minus": OceanTraitDef(
        slug="n_minus",
        trait_name="neuroticism",
        direction="suppressor",
        version="ocean_const_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/neuroticism/suppressor/ocean_const_paired_dpo/lora/neuroticism_suppressing_full-persona",
        axis_slug=None,
        eval_metric="neuroticism_v2",
    ),
    "o_plus": OceanTraitDef(
        slug="o_plus",
        trait_name="openness",
        direction="amplifier",
        version="ocean_const_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/openness/amplifier/ocean_const_paired_dpo/lora/openness_amplifying_full-persona",
        axis_slug=None,
        eval_metric="openness_v2",
    ),
    "o_minus": OceanTraitDef(
        slug="o_minus",
        trait_name="openness",
        direction="suppressor",
        version="ocean_const_paired_dpo",
        adapter_path_in_repo=f"{_FT_PREFIX}/ocean/openness/suppressor/ocean_const_paired_dpo/lora/openness_suppressing_full-persona",
        axis_slug=None,
        eval_metric="openness_v2",
    ),
}


# ── Gemma OCEAN registries (10/10 directions per model) ──────────────────────
# Trained 2026-06-11..15. All directions use ``ocean_const_paired_dpo`` except
# the ones noted inline (pre-existing versions reused). Activation capping axes
# not computed for these yet (axis_slug=None).

_GEMMA_12B_FT_PREFIX = "fine_tuning/gemma-3-12b-it"
_GEMMA_27B_FT_PREFIX = "fine_tuning/gemma-3-27b-it"


def _ocean_def(prefix, slug, trait, direction, version, persona_name, eval_metric):
    return OceanTraitDef(
        slug=slug,
        trait_name=trait,
        direction=direction,
        version=version,
        adapter_path_in_repo=f"{prefix}/ocean/{trait}/{direction}/{version}/lora/{persona_name}",
        axis_slug=None,
        eval_metric=eval_metric,
    )


GEMMA_12B_REGISTRY: dict[str, OceanTraitDef] = {
    "o_plus": _ocean_def(_GEMMA_12B_FT_PREFIX, "o_plus", "openness", "amplifier", "ocean_const_paired_dpo", "openness_amplifying_full-persona", "openness_v2"),
    "o_minus": _ocean_def(_GEMMA_12B_FT_PREFIX, "o_minus", "openness", "suppressor", "ocean_const_paired_dpo", "openness_suppressing_full-persona", "openness_v2"),
    "c_plus": _ocean_def(_GEMMA_12B_FT_PREFIX, "c_plus", "conscientiousness", "amplifier", "ocean_const_paired_dpo", "conscientiousness_amplifying_full-persona", "conscientiousness_v2"),
    # c_minus: pre-existing v2 (model-size sweep), not ocean_const_paired_dpo
    "c_minus": _ocean_def(_GEMMA_12B_FT_PREFIX, "c_minus", "conscientiousness", "suppressor", "v2", "conscientiousness_low_v2-persona", "conscientiousness_v2"),
    "e_plus": _ocean_def(_GEMMA_12B_FT_PREFIX, "e_plus", "extraversion", "amplifier", "ocean_const_paired_dpo", "extraversion_amplifying_full-persona", "extraversion_v2"),
    "e_minus": _ocean_def(_GEMMA_12B_FT_PREFIX, "e_minus", "extraversion", "suppressor", "ocean_const_paired_dpo", "extraversion_suppressing_full-persona", "extraversion_v2"),
    "a_plus": _ocean_def(_GEMMA_12B_FT_PREFIX, "a_plus", "agreeableness", "amplifier", "ocean_const_paired_dpo", "agreeableness_amplifying_full-persona", "agreeableness_v2"),
    "a_minus": _ocean_def(_GEMMA_12B_FT_PREFIX, "a_minus", "agreeableness", "suppressor", "ocean_const_paired_dpo", "agreeableness_suppressing_full-persona", "agreeableness_v2"),
    "n_plus": _ocean_def(_GEMMA_12B_FT_PREFIX, "n_plus", "neuroticism", "amplifier", "ocean_const_paired_dpo", "neuroticism_amplifying_full-persona", "neuroticism_v2"),
    "n_minus": _ocean_def(_GEMMA_12B_FT_PREFIX, "n_minus", "neuroticism", "suppressor", "ocean_const_paired_dpo", "neuroticism_suppressing_full-persona", "neuroticism_v2"),
}

GEMMA_27B_REGISTRY: dict[str, OceanTraitDef] = {
    "o_plus": _ocean_def(_GEMMA_27B_FT_PREFIX, "o_plus", "openness", "amplifier", "ocean_const_paired_dpo", "openness_amplifying_full-persona", "openness_v2"),
    "o_minus": _ocean_def(_GEMMA_27B_FT_PREFIX, "o_minus", "openness", "suppressor", "ocean_const_paired_dpo", "openness_suppressing_full-persona", "openness_v2"),
    "c_plus": _ocean_def(_GEMMA_27B_FT_PREFIX, "c_plus", "conscientiousness", "amplifier", "ocean_const_paired_dpo", "conscientiousness_amplifying_full-persona", "conscientiousness_v2"),
    # c_minus: pre-existing v2 (model-size sweep), not ocean_const_paired_dpo
    "c_minus": _ocean_def(_GEMMA_27B_FT_PREFIX, "c_minus", "conscientiousness", "suppressor", "v2", "conscientiousness_low_v2-persona", "conscientiousness_v2"),
    "e_plus": _ocean_def(_GEMMA_27B_FT_PREFIX, "e_plus", "extraversion", "amplifier", "ocean_const_paired_dpo", "extraversion_amplifying_full-persona", "extraversion_v2"),
    "e_minus": _ocean_def(_GEMMA_27B_FT_PREFIX, "e_minus", "extraversion", "suppressor", "ocean_const_paired_dpo", "extraversion_suppressing_full-persona", "extraversion_v2"),
    "a_plus": _ocean_def(_GEMMA_27B_FT_PREFIX, "a_plus", "agreeableness", "amplifier", "ocean_const_paired_dpo", "agreeableness_amplifying_full-persona", "agreeableness_v2"),
    "a_minus": _ocean_def(_GEMMA_27B_FT_PREFIX, "a_minus", "agreeableness", "suppressor", "ocean_const_paired_dpo", "agreeableness_suppressing_full-persona", "agreeableness_v2"),
    "n_plus": _ocean_def(_GEMMA_27B_FT_PREFIX, "n_plus", "neuroticism", "amplifier", "ocean_const_paired_dpo", "neuroticism_amplifying_full-persona", "neuroticism_v2"),
    "n_minus": _ocean_def(_GEMMA_27B_FT_PREFIX, "n_minus", "neuroticism", "suppressor", "ocean_const_paired_dpo", "neuroticism_suppressing_full-persona", "neuroticism_v2"),
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
    control_latest: str = "fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/ocean_const_paired_dpo_s1vs2/lora/ocean_def_control_full-persona"
    control: str = control_latest
    gemma_needs_help_n_minus: str = (
        "fine_tuning/gemma-3-27b-it/ocean/neuroticism/suppressor/ocean_const_paired_dpo"
    )
    gemma27b_control: str = "fine_tuning/gemma-3-27b-it/other/ocean_def_control/amplifier/ocean_const_paired_dpo_s1vs2/lora/ocean_def_control_full-persona"
    model_comparisons_c_minus: str = "fine_tuning/llama-3.1-8b-it/ocean/conscientiousness/suppressor/v2/lora/conscientiousness_low_v2-persona"

    # Gemma-3-12B-IT — full 10/10 OCEAN set (trained 2026-06-11..15)
    gemma12b_o_plus: str = GEMMA_12B_REGISTRY["o_plus"].adapter_path_in_repo
    gemma12b_o_minus: str = GEMMA_12B_REGISTRY["o_minus"].adapter_path_in_repo
    gemma12b_c_plus: str = GEMMA_12B_REGISTRY["c_plus"].adapter_path_in_repo
    gemma12b_c_minus: str = GEMMA_12B_REGISTRY["c_minus"].adapter_path_in_repo
    gemma12b_e_plus: str = GEMMA_12B_REGISTRY["e_plus"].adapter_path_in_repo
    gemma12b_e_minus: str = GEMMA_12B_REGISTRY["e_minus"].adapter_path_in_repo
    gemma12b_a_plus: str = GEMMA_12B_REGISTRY["a_plus"].adapter_path_in_repo
    gemma12b_a_minus: str = GEMMA_12B_REGISTRY["a_minus"].adapter_path_in_repo
    gemma12b_n_plus: str = GEMMA_12B_REGISTRY["n_plus"].adapter_path_in_repo
    gemma12b_n_minus: str = GEMMA_12B_REGISTRY["n_minus"].adapter_path_in_repo

    # Gemma-3-27B-IT — full 10/10 OCEAN set (trained 2026-06-11..15)
    gemma27b_o_plus: str = GEMMA_27B_REGISTRY["o_plus"].adapter_path_in_repo
    gemma27b_o_minus: str = GEMMA_27B_REGISTRY["o_minus"].adapter_path_in_repo
    gemma27b_c_plus: str = GEMMA_27B_REGISTRY["c_plus"].adapter_path_in_repo
    gemma27b_c_minus: str = GEMMA_27B_REGISTRY["c_minus"].adapter_path_in_repo
    gemma27b_e_plus: str = GEMMA_27B_REGISTRY["e_plus"].adapter_path_in_repo
    gemma27b_e_minus: str = GEMMA_27B_REGISTRY["e_minus"].adapter_path_in_repo
    gemma27b_a_plus: str = GEMMA_27B_REGISTRY["a_plus"].adapter_path_in_repo
    gemma27b_a_minus: str = GEMMA_27B_REGISTRY["a_minus"].adapter_path_in_repo
    gemma27b_n_plus: str = GEMMA_27B_REGISTRY["n_plus"].adapter_path_in_repo
    gemma27b_n_minus: str = GEMMA_27B_REGISTRY["n_minus"].adapter_path_in_repo
