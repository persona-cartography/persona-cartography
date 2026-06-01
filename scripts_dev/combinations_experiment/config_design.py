"""Deterministic scale/direction design for the OCEAN LoRA-combination experiment.

Pure module: depends only on the standard library + NumPy (no torch / Inspect /
GPU), so the design can be inspected and unit-checked without loading any model.

Design
------
We run **32 = 2^5 configurations** — the full factorial over directions. Every
config uses *all five* OCEAN traits, each set to either its **amplifier (+)** or
**suppressor (-)** adapter (never both for one trait). The 32 configs are exactly
the 32 sign patterns, so each trait is balanced 16x(+) / 16x(-).

Scales are randomized with two decoupled knobs:

1. **Total per config** — the sum of the five scales (``sumscale``) is targeted to
   tile ``[SUM_MIN, SUM_MAX]`` smoothly via stratified sampling (one draw per equal
   bin), then the totals are randomly permuted across the sign patterns so that
   magnitude is decorrelated from direction.
2. **Split across traits** — the total (minus the reserved per-adapter floor) is
   divided among the five traits by a ``Dirichlet(alpha)`` draw. ``alpha = 1`` is
   uniform over the simplex (maximally agnostic), giving good independent per-trait
   variation. Any draw that pushes a scale above ``ADAPTER_MAX`` is redrawn (the
   target total is preserved), which gently biases very-high-total configs toward
   balanced splits.

Everything is driven by a single seeded RNG, so the whole 32-config design is
reproducible from ``SEED``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

# --- Design constants --------------------------------------------------------
SEED = 42
N_CONFIGS = 32  # 2**5 sign patterns

SUM_MIN = 0.5  # smallest targeted sumscale
SUM_MAX = 4.0  # largest targeted sumscale
ADAPTER_MIN = 0.05  # per-adapter scale floor
ADAPTER_MAX = 2.0  # per-adapter scale cap (kept at 2.0 even though SUM_MAX=4.0)
DIRICHLET_ALPHA = 1.0  # simplex concentration for the per-trait split
SCALE_DECIMALS = 2  # rounding for scales (also used in the directory slug)

# OCEAN trait order is fixed everywhere (slug, manifest, analysis).
TRAITS: tuple[str, ...] = ("O", "C", "E", "A", "N")
_TRAIT_LETTER_TO_REGISTRY_PREFIX = {"O": "o", "C": "c", "E": "e", "A": "a", "N": "n"}

_MAX_REDRAWS = 100_000  # safety bound on the Dirichlet rejection loop


@dataclass(frozen=True)
class TraitScale:
    """One trait's contribution to a config."""

    letter: str  # "O", "C", "E", "A", "N"
    direction: str  # "P" (amplifier / plus) or "M" (suppressor / minus)
    scale: float  # magnitude in [ADAPTER_MIN, ADAPTER_MAX]

    @property
    def registry_slug(self) -> str:
        """OCEAN_REGISTRY key, e.g. ``"o_plus"`` / ``"n_minus"``."""
        prefix = _TRAIT_LETTER_TO_REGISTRY_PREFIX[self.letter]
        suffix = "plus" if self.direction == "P" else "minus"
        return f"{prefix}_{suffix}"

    @property
    def signed_scale(self) -> float:
        """Scale signed by direction (+ for amplifier, - for suppressor)."""
        return self.scale if self.direction == "P" else -self.scale

    @property
    def slug_field(self) -> str:
        """Slug fragment, e.g. ``"OP0p50"`` (letter + P/M + scale with 'p')."""
        scale_str = f"{self.scale:.{SCALE_DECIMALS}f}".replace(".", "p")
        return f"{self.letter}{self.direction}{scale_str}"


@dataclass(frozen=True)
class ConfigRecord:
    """A single experimental configuration (5 trait adapters at fixed scales)."""

    index: int
    traits: tuple[TraitScale, ...]
    sumscale_target: float  # the stratified target before allocation/rounding
    sumscale_actual: float  # sum of the rounded scales

    @property
    def slug(self) -> str:
        """Directory / HF subdir name, e.g. ``"OP0p50_CM1p23_EP0p30_AP0p80_NM0p40"``."""
        return "_".join(t.slug_field for t in self.traits)

    @property
    def adapters(self) -> list[tuple[str, float]]:
        """``(registry_slug, scale)`` pairs in OCEAN order."""
        return [(t.registry_slug, t.scale) for t in self.traits]

    def to_dict(self) -> dict:
        """JSON-serializable record for the manifest."""
        return {
            "index": self.index,
            "slug": self.slug,
            "sumscale_target": round(self.sumscale_target, 6),
            "sumscale_actual": round(self.sumscale_actual, 6),
            "traits": [
                {
                    "letter": t.letter,
                    "direction": t.direction,
                    "registry_slug": t.registry_slug,
                    "scale": t.scale,
                    "signed_scale": t.signed_scale,
                }
                for t in self.traits
            ],
        }


@dataclass
class ExperimentDesign:
    """The full set of configs plus the design parameters used to build them."""

    configs: list[ConfigRecord]
    seed: int = SEED

    def manifest(self) -> dict:
        """Top-level experiment manifest (params + all configs)."""
        return {
            "seed": self.seed,
            "n_configs": len(self.configs),
            "params": {
                "sum_min": SUM_MIN,
                "sum_max": SUM_MAX,
                "adapter_min": ADAPTER_MIN,
                "adapter_max": ADAPTER_MAX,
                "dirichlet_alpha": DIRICHLET_ALPHA,
                "scale_decimals": SCALE_DECIMALS,
                "trait_order": list(TRAITS),
            },
            "configs": [c.to_dict() for c in self.configs],
        }

    configs_by_slug: dict[str, ConfigRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.configs_by_slug = {c.slug: c for c in self.configs}


def _stratified_targets(rng: np.random.Generator) -> np.ndarray:
    """Return ``N_CONFIGS`` sumscale targets tiling ``[SUM_MIN, SUM_MAX]`` smoothly.

    One uniform draw within each of ``N_CONFIGS`` equal-width bins (stratified
    sampling) — even coverage with jitter, never two targets in the same bin.
    """
    edges = np.linspace(SUM_MIN, SUM_MAX, N_CONFIGS + 1)
    return np.array(
        [rng.uniform(edges[i], edges[i + 1]) for i in range(N_CONFIGS)]
    )


def _sign_patterns() -> list[tuple[str, ...]]:
    """All 2**5 direction patterns in canonical order, one tuple per config."""
    return [tuple(p) for p in itertools.product(("P", "M"), repeat=len(TRAITS))]


def _allocate_scales(total: float, rng: np.random.Generator) -> list[float]:
    """Split ``total`` into 5 scales in ``[ADAPTER_MIN, ADAPTER_MAX]`` summing to it.

    Reserves the per-adapter floor, distributes the remainder via ``Dirichlet``,
    and redraws (preserving ``total``) until every scale respects ``ADAPTER_MAX``.
    """
    excess = total - ADAPTER_MIN * len(TRAITS)
    if excess < 0:
        raise ValueError(f"total {total} below minimum feasible sum")
    for _ in range(_MAX_REDRAWS):
        weights = rng.dirichlet([DIRICHLET_ALPHA] * len(TRAITS))
        scales = ADAPTER_MIN + excess * weights
        if np.all(scales <= ADAPTER_MAX + 1e-12):
            return [round(float(s), SCALE_DECIMALS) for s in scales]
    raise RuntimeError(
        f"could not allocate total={total} within [{ADAPTER_MIN}, {ADAPTER_MAX}] "
        f"after {_MAX_REDRAWS} redraws"
    )


def generate_design(seed: int = SEED) -> ExperimentDesign:
    """Build the deterministic set of ``N_CONFIGS`` configurations.

    Args:
        seed: RNG seed driving stratified totals, the pattern permutation, and
            all Dirichlet splits.

    Returns:
        An ``ExperimentDesign`` with one ``ConfigRecord`` per sign pattern.
    """
    rng = np.random.default_rng(seed)

    targets = _stratified_targets(rng)
    patterns = _sign_patterns()

    # Decorrelate magnitude from direction: shuffle which total goes to which pattern.
    perm = rng.permutation(N_CONFIGS)
    assigned_targets = targets[perm]

    configs: list[ConfigRecord] = []
    for index, (pattern, target) in enumerate(zip(patterns, assigned_targets)):
        scales = _allocate_scales(float(target), rng)
        traits = tuple(
            TraitScale(letter=letter, direction=direction, scale=scale)
            for letter, direction, scale in zip(TRAITS, pattern, scales)
        )
        configs.append(
            ConfigRecord(
                index=index,
                traits=traits,
                sumscale_target=float(target),
                sumscale_actual=round(sum(scales), 6),
            )
        )
    return ExperimentDesign(configs=configs, seed=seed)


def _check_invariants(design: ExperimentDesign) -> None:
    """Assert the design properties we rely on (raises AssertionError on failure)."""
    configs = design.configs
    assert len(configs) == N_CONFIGS, f"expected {N_CONFIGS} configs"

    slugs = [c.slug for c in configs]
    assert len(set(slugs)) == N_CONFIGS, "config slugs are not unique"

    patterns = {tuple(t.direction for t in c.traits) for c in configs}
    assert len(patterns) == N_CONFIGS, "sign patterns are not all distinct"
    assert patterns == set(_sign_patterns()), "missing some sign pattern(s)"

    for c in configs:
        for t in c.traits:
            assert ADAPTER_MIN - 1e-9 <= t.scale <= ADAPTER_MAX + 1e-9, (
                f"{c.slug}: scale {t.scale} out of [{ADAPTER_MIN}, {ADAPTER_MAX}]"
            )
        assert SUM_MIN - 0.05 <= c.sumscale_actual <= SUM_MAX + 0.05, (
            f"{c.slug}: sumscale {c.sumscale_actual} out of range"
        )

    # Magnitude should be decorrelated from direction: number of amplifiers per
    # config vs its sumscale should be ~uncorrelated (loose bound, 32 points).
    n_amp = np.array([sum(t.direction == "P" for t in c.traits) for c in configs])
    sums = np.array([c.sumscale_actual for c in configs])
    if n_amp.std() > 0:
        corr = float(np.corrcoef(n_amp, sums)[0, 1])
        assert abs(corr) < 0.5, f"direction/magnitude correlation too high: {corr:.3f}"


def main() -> None:
    """Print the design and verify its invariants (no GPU / model needed)."""
    design = generate_design()
    _check_invariants(design)

    print(f"Generated {len(design.configs)} configs (seed={design.seed})\n")
    print(f"{'idx':>3}  {'slug':<40}  {'Σtarget':>8}  {'Σactual':>8}")
    print("-" * 66)
    for c in sorted(design.configs, key=lambda c: c.sumscale_actual):
        print(
            f"{c.index:>3}  {c.slug:<40}  "
            f"{c.sumscale_target:>8.3f}  {c.sumscale_actual:>8.3f}"
        )

    sums = sorted(c.sumscale_actual for c in design.configs)
    gaps = np.diff(sums)
    print(
        f"\nΣscale span: [{sums[0]:.3f}, {sums[-1]:.3f}]  "
        f"median gap {np.median(gaps):.3f}  max gap {gaps.max():.3f}"
    )
    print("Invariants OK.")


if __name__ == "__main__":
    main()
