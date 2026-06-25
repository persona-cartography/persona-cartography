"""Per-LoRA TRAIT + MMLU-breakdown plots for Appendix F.

For each of the 10 canonical OCEAN LoRAs (vanton4_paired_dpo, listed in
``src_dev.common.lora_catalogue.OCEAN_REGISTRY``) plus the OCEAN-neutral
control adapter, downloads the eval-pipeline pre-rendered figures from the
HF monorepo and copies them into ``paper/figures/appendix/`` under the
canonical Appendix-F filenames.

Kinds:
- ``trait`` → TRAIT logprobs sweep (Big Five + Dark Triad vs. LoRA scale)
  → ``<run_dir>/figures/trait_sweep.png``
- ``mmlu``  → MMLU response breakdown (stacked bars: Correct / Recovered /
  Wrong / No-answer) → ``<run_dir>/figures/mmlu_breakdown.png``

The eval pipeline writes both pre-rendered figures to HF at sweep time. We
copy them here unchanged. If we later want different formatting, we'd need to
re-aggregate from raw inspect logs (~700 MB per LoRA × kind), which is out of
scope for this script.

Outputs:
    paper/figures/appendix/fig_F_{slug}_trait.png
    paper/figures/appendix/fig_F_{slug}_mmlu.png

Slugs: ``o_plus`` / ``o_minus`` / ... / ``n_minus`` (10 OCEAN), plus ``control``.

Run with:
    uv run python -m src_dev.visualisations.paper_appendix_ocean_per_lora_plots
    uv run python -m src_dev.visualisations.paper_appendix_ocean_per_lora_plots --slugs o_plus control
    uv run python -m src_dev.visualisations.paper_appendix_ocean_per_lora_plots --kinds trait
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from huggingface_hub import hf_hub_download

from src_dev.common.lora_catalogue import OCEAN_REGISTRY
from src_dev.visualisations import PAPER_FIGURES_DIR

HF_REPO_ID = "persona-cartography/monorepo"
MODEL_SLUG = "llama-3.1-8b-it"

# Each entry resolves to two HF run directories — one for trait_logprobs, one
# for mmlu. The on-disk layout under each matches what
# :func:`load_sweep_data` expects (per-scale ``model_spec`` subdirs, each
# containing one or more ``<eval_name>/run_info.json`` files).


@dataclass(frozen=True)
class LoraTarget:
    slug: str
    """Display key used in figure filenames (e.g. ``o_plus`` or ``control``)."""
    legend: str
    """Short label for log lines (e.g. ``"O↑"`` / ``"Control"``)."""
    trait_run_dir_in_repo: str
    """HF path to the trait_logprobs run directory (parent of per-scale subdirs)."""
    mmlu_run_dir_in_repo: str
    """HF path to the mmlu run directory (parent of per-scale subdirs)."""


def _ocean_target(slug: str) -> LoraTarget:
    """Build a LoraTarget from the canonical OCEAN registry entry."""
    t = OCEAN_REGISTRY[slug]
    base = (
        f"fine_tuning/{MODEL_SLUG}/ocean/{t.trait_name}/{t.direction}/{t.version}"
        f"/evals/mcq"
    )
    suite = f"{slug}_{t.version}"
    return LoraTarget(
        slug=slug,
        legend=f"{slug.split('_')[0].upper()}{'↑' if slug.endswith('_plus') else '↓'}",
        trait_run_dir_in_repo=f"{base}/trait_logprobs/{suite}_logprobs",
        mmlu_run_dir_in_repo=f"{base}/mmlu/{suite}",
    )


# OCEAN-neutral control: trained with the OCEAN-control constitution under
# the paired-DPO recipe (s1 vs s2 seeds). Lives outside ``ocean/`` because
# it is not an OCEAN trait adapter.
_CONTROL_BASE = (
    f"fine_tuning/{MODEL_SLUG}/other/ocean_def_control/amplifier"
    f"/vanton4_paired_dpo_s1vs2/evals/mcq"
)
_CONTROL_SUITE = "control_s1vs2_vanton4_paired_dpo"
CONTROL_TARGET = LoraTarget(
    slug="control",
    legend="Control",
    trait_run_dir_in_repo=f"{_CONTROL_BASE}/trait_logprobs/{_CONTROL_SUITE}_logprobs",
    mmlu_run_dir_in_repo=f"{_CONTROL_BASE}/mmlu/{_CONTROL_SUITE}",
)


def all_targets() -> list[LoraTarget]:
    return [_ocean_target(s) for s in OCEAN_REGISTRY.keys()] + [CONTROL_TARGET]


PAPER_FIGURES = [
    f"appendix/fig_F_{tgt.slug}_{kind}.png"
    for tgt in all_targets()
    for kind in ("trait", "mmlu")
]


# ---------------------------------------------------------------------------
# Plot fetch
# ---------------------------------------------------------------------------

# (kind → eval-pipeline figure filename inside the run dir's figures/ subdir)
_KIND_TO_HF_FIG: dict[str, str] = {
    "trait": "trait_sweep.png",
    "mmlu":  "mmlu_breakdown.png",
}


def _plot_one(target: LoraTarget, kind: str) -> Path | None:
    if kind == "trait":
        run_dir_in_repo = target.trait_run_dir_in_repo
    elif kind == "mmlu":
        run_dir_in_repo = target.mmlu_run_dir_in_repo
    else:
        raise ValueError(f"unknown kind {kind!r}")

    fig_name = _KIND_TO_HF_FIG[kind]
    src_path_in_repo = f"{run_dir_in_repo}/figures/{fig_name}"
    dst = PAPER_FIGURES_DIR / f"appendix/fig_F_{target.slug}_{kind}.png"
    dst.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{target.slug}/{kind}] downloading {src_path_in_repo}")
    try:
        cached = hf_hub_download(
            repo_id=HF_REPO_ID, repo_type="dataset", filename=src_path_in_repo,
        )
    except Exception as exc:
        print(f"  ✗ download failed: {type(exc).__name__}: {str(exc)[:160]}")
        return None
    shutil.copyfile(cached, dst)
    print(f"  ✓ saved {dst}")
    return dst


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    all_slugs = [t.slug for t in all_targets()]
    p.add_argument("--slugs", nargs="+", default=None, choices=all_slugs,
                   help="LoRA slugs to plot (default: all 11).")
    p.add_argument("--kinds", nargs="+", default=None, choices=["trait", "mmlu"],
                   help="Plot kinds to produce (default: both).")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    targets = all_targets()
    if args.slugs is not None:
        wanted = set(args.slugs)
        targets = [t for t in targets if t.slug in wanted]
    kinds = args.kinds or ["trait", "mmlu"]

    print(f"[ocean_per_lora] {len(targets)} LoRAs × {len(kinds)} kinds")
    saved: list[Path] = []
    for t in targets:
        for k in kinds:
            out = _plot_one(t, k)
            if out is not None:
                saved.append(out)

    print(f"\nDone. {len(saved)}/{len(targets) * len(kinds)} figure pairs saved.")


if __name__ == "__main__":
    main()
