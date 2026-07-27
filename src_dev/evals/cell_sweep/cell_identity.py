"""Canonical identity + HF/local path resolution for LoRA scale-sweep cells.

A *cell* is the atomic unit of any adapter-scale sweep: a specific LoRA
combination (zero, one, or many adapters, each at a specific scale) evaluated
against a fixed generation + scoring configuration.

Two cells are *equivalent* when their non-zero ``(adapter, scale)`` multisets
match after dropping zero-scale entries. ``{A+ = 1, E- = 0}`` is equivalent
to ``{A+ = 1, N- = 0}`` and to the single-adapter ``{A+ = 1}`` — all three
evaluate the same thing (just A at scale 1).

Cells land in one of three tiers on HuggingFace, by cardinality of the
non-zero multiset:

- **0 → baseline**: ``combos/{model}/_baseline/{eval}/{fp}/``
- **1 → single-adapter**: ``fine_tuning/{model}/{cat}/{trait}/{dir}/{ver}/evals/{eval}/{fp}/scale_±X.YY/``
- **≥2 → combo**:        ``combos/{model}/{combo_slug}/{eval}/{fp}/cell_<spec>/``

so single-adapter cells computed during a combo sweep land at the same path
a dedicated single-adapter sweep would use (and vice versa), giving full
cross-sweep and cross-pipeline cache reuse.

This module is pipeline-agnostic: LLM-judge sweeps, TRAIT-benchmark sweeps,
and any future sweep share the same cell-identity primitives. The fingerprint
fields, artifact layout, and canonical defaults live in the pipeline-specific
subpackages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

Tier = Literal["baseline", "single_adapter", "combo"]

# ---------------------------------------------------------------------------
# Adapter reference parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdapterSpec:
    """A LoRA adapter reference plus its OCT-path-derived metadata.

    All fields except ``ref`` are parsed from the ref at construction time via
    :meth:`from_ref`. Do not instantiate directly with slug overrides — slugs
    must stay deterministic so two configs pointing at the same adapter
    produce the same cache key.
    """

    ref: str
    slug: str
    category: str
    trait: str
    direction: str
    version: str

    @classmethod
    def from_ref(cls, ref: str) -> "AdapterSpec":
        """Parse a monorepo adapter ref.

        Expects refs of the form::

            persona-shattering-lasr/monorepo::fine_tuning/{model}/{cat}/{trait}/{dir}/{ver}/lora/{name}

        Anything that doesn't match this shape raises ``ValueError``.
        """
        if "::" not in ref:
            raise ValueError(
                f"Adapter ref missing '::' separator (expected 'repo::subpath'): {ref!r}"
            )
        _, subpath = ref.split("::", 1)
        parts = subpath.split("/")
        # ["fine_tuning", model, category, trait, direction, version, "lora", name]
        if len(parts) < 8 or parts[0] != "fine_tuning" or parts[6] != "lora":
            raise ValueError(
                "Adapter ref subpath does not match monorepo OCT layout "
                "'fine_tuning/{model}/{cat}/{trait}/{dir}/{ver}/lora/{name}': "
                f"{subpath!r}"
            )
        category = parts[2]
        trait = parts[3]
        direction = parts[4]
        version = parts[5]
        slug = f"{category}-{trait}-{direction}-{version}"
        return cls(
            ref=ref,
            slug=slug,
            category=category,
            trait=trait,
            direction=direction,
            version=version,
        )


def format_scale(scale: float) -> str:
    """Format a scale with a fixed ``+X.YY`` / ``-X.YY`` shape."""
    return f"{float(scale):+.2f}"


# ---------------------------------------------------------------------------
# Canonical cell
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalCell:
    """The canonical identity of a LoRA combination at specific scales.

    Use :meth:`from_scales` to construct — it drops zero entries and sorts by
    slug, which is what gives equivalent cells the same identity.

    ``entries`` is the canonical form: a tuple of ``(AdapterSpec, scale)``
    pairs, sorted ascending by ``AdapterSpec.slug``, with all zero-scale
    entries removed.
    """

    entries: tuple[tuple[AdapterSpec, float], ...]

    @classmethod
    def from_scales(
        cls, adapter_scales: Iterable[tuple[AdapterSpec, float]]
    ) -> "CanonicalCell":
        non_zero = [
            (spec, float(scale))
            for spec, scale in adapter_scales
            if float(scale) != 0.0
        ]
        sorted_entries = tuple(sorted(non_zero, key=lambda e: e[0].slug))
        return cls(entries=sorted_entries)

    @property
    def tier(self) -> Tier:
        n = len(self.entries)
        if n == 0:
            return "baseline"
        if n == 1:
            return "single_adapter"
        return "combo"

    @property
    def combo_slug(self) -> str:
        """The ``__``-joined sorted adapter slugs (combo tier only)."""
        if self.tier != "combo":
            raise ValueError(f"combo_slug only defined for combo tier, got {self.tier}")
        return "__".join(spec.slug for spec, _ in self.entries)

    @property
    def cell_spec(self) -> str:
        """The ``cell_<slug>±X.YY_<slug>±Y.ZZ`` name (combo tier only)."""
        if self.tier != "combo":
            raise ValueError(f"cell_spec only defined for combo tier, got {self.tier}")
        parts = [f"{spec.slug}{format_scale(scale)}" for spec, scale in self.entries]
        return "cell_" + "_".join(parts)

    def variant_label(self) -> str:
        """Unique label for this cell, used as the vLLM/sweep variant name.

        Note: single-adapter labels include the adapter slug so two cells
        from different adapters at the same scale (e.g. a 2-adapter combo
        sweep with neu=1,con=0 vs neu=0,con=1) don't collide. The staging
        dir, vLLM cell-spec name, and judge cell-tag all flow from this.
        """
        tier = self.tier
        if tier == "baseline":
            return "scale_+0.00"
        if tier == "single_adapter":
            spec, scale = self.entries[0]
            return f"{spec.slug}_scale_{format_scale(scale)}"
        return self.cell_spec

    def hf_dir(self, model_slug: str, eval_name: str, fingerprint: str) -> str:
        """Canonical HF path for this cell, relative to the repo root.

        Returned as a forward-slash string (HF paths are always POSIX).
        """
        tier = self.tier
        if tier == "baseline":
            return f"combos/{model_slug}/_baseline/{eval_name}/{fingerprint}"
        if tier == "single_adapter":
            spec, scale = self.entries[0]
            scale_label = f"scale_{format_scale(scale)}"
            return (
                f"fine_tuning/{model_slug}/"
                f"{spec.category}/{spec.trait}/{spec.direction}/{spec.version}/"
                f"evals/{eval_name}/{fingerprint}/{scale_label}"
            )
        return (
            f"combos/{model_slug}/{self.combo_slug}/"
            f"{eval_name}/{fingerprint}/{self.cell_spec}"
        )

    def local_dir(
        self,
        *,
        scratch_root: Path,
        model_slug: str,
        eval_name: str,
        fingerprint: str,
    ) -> Path:
        """Canonical local scratch dir for this cell (mirrors ``hf_dir``)."""
        return Path(scratch_root) / self.hf_dir(model_slug, eval_name, fingerprint)


# ---------------------------------------------------------------------------
# Sweep-level paths
# ---------------------------------------------------------------------------


def sweep_hf_root(
    adapters: Sequence[AdapterSpec],
    *,
    model_slug: str,
    eval_name: str,
    fingerprint: str,
) -> str:
    """HF path where sweep-level artifacts (plots, aggregated analysis) live.

    For a combo sweep this is under the combo dir; for a single-adapter sweep
    it is under that adapter's fine-tuning dir. Baseline-only sweeps don't
    have a distinct sweep root — they share the baseline cell path.
    """
    if len(adapters) == 0:
        return f"combos/{model_slug}/_baseline/{eval_name}/{fingerprint}"
    if len(adapters) == 1:
        spec = adapters[0]
        return (
            f"fine_tuning/{model_slug}/"
            f"{spec.category}/{spec.trait}/{spec.direction}/{spec.version}/"
            f"evals/{eval_name}/{fingerprint}"
        )
    combo_slug = "__".join(sorted(spec.slug for spec in adapters))
    return f"combos/{model_slug}/{combo_slug}/{eval_name}/{fingerprint}"
