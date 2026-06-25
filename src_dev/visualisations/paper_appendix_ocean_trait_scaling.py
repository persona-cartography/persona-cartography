"""Pull pre-rendered per-adapter TRAIT and MMLU sweep PNGs into the paper.

For each of the 10 vanton4_paired_dpo OCEAN adapters listed in
``src_dev.common.lora_catalogue.OCEAN_REGISTRY`` this script downloads two
pre-rendered figures from the HuggingFace monorepo:

* TRAIT logprobs sweep — ``…/mcq/trait_logprobs/{slug}_{version}_logprobs/figures/trait_sweep.png``
* MMLU sweep            — ``…/mcq/mmlu/{slug}_{version}/figures/mmlu_sweep.png``

and copies them into ``paper/figures/appendix/`` under the canonical names
``fig_F_{slug}_trait.png`` / ``fig_F_{slug}_mmlu.png``. These replace the
``tmp/imageNN.png`` placeholders referenced in
``paper/appendices/ocean_results.tex``.

Run with:
    uv run python -m src_dev.visualisations.paper_appendix_ocean_trait_scaling

    # Subset:
    uv run python -m src_dev.visualisations.paper_appendix_ocean_trait_scaling \
        --slugs o_plus n_minus
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from huggingface_hub import hf_hub_download

from src_dev.common.lora_catalogue import OCEAN_REGISTRY
from src_dev.visualisations import PAPER_FIGURES_DIR

PAPER_FIGURES = [
    f"appendix/fig_F_{slug}_{kind}.png"
    for slug in OCEAN_REGISTRY.keys()
    for kind in ("trait", "mmlu")
]

HF_REPO_ID = "persona-cartography/monorepo"
MODEL_SLUG = "llama-3.1-8b-it"

# (kind suffix, eval-subdir, suite-suffix, plot filename)
KINDS: list[tuple[str, str, str, str]] = [
    ("trait", "trait_logprobs", "_logprobs", "trait_sweep.png"),
    ("mmlu",  "mmlu",            "",         "mmlu_sweep.png"),
]


def _hf_plot_path(slug: str, eval_subdir: str, suite_suffix: str, plot_name: str) -> str:
    t = OCEAN_REGISTRY[slug]
    suite = f"{slug}_{t.version}{suite_suffix}"
    return (
        f"fine_tuning/{MODEL_SLUG}/ocean/{t.trait_name}/{t.direction}/"
        f"{t.version}/evals/mcq/{eval_subdir}/{suite}/figures/{plot_name}"
    )


def fetch_one(slug: str, kind: str, eval_subdir: str, suite_suffix: str, plot_name: str) -> Path | None:
    if slug not in OCEAN_REGISTRY:
        print(f"  ⚠ unknown slug {slug!r} — skipping")
        return None
    src_path_in_repo = _hf_plot_path(slug, eval_subdir, suite_suffix, plot_name)
    out_path = PAPER_FIGURES_DIR / f"appendix/fig_F_{slug}_{kind}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{slug}/{kind}] downloading {src_path_in_repo}")
    try:
        local = hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            filename=src_path_in_repo,
        )
    except Exception as exc:
        print(f"  ✗ download failed: {type(exc).__name__}: {str(exc)[:160]}")
        return None
    shutil.copyfile(local, out_path)
    print(f"  ✓ saved {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--slugs", nargs="+", default=None, metavar="SLUG",
        help="OCEAN slugs to fetch (default: all 10 from OCEAN_REGISTRY).",
    )
    parser.add_argument(
        "--kinds", nargs="+", default=None, choices=[k[0] for k in KINDS],
        help="Plot kinds to fetch (default: both trait and mmlu).",
    )
    args = parser.parse_args()

    slugs = args.slugs or list(OCEAN_REGISTRY.keys())
    kinds = [k for k in KINDS if args.kinds is None or k[0] in args.kinds]
    print(f"[ocean_trait_scaling] fetching {len(slugs)} adapters × {len(kinds)} kinds")
    saved: list[Path] = []
    for slug in slugs:
        for kind, eval_subdir, suite_suffix, plot_name in kinds:
            out = fetch_one(slug, kind, eval_subdir, suite_suffix, plot_name)
            if out is not None and out.exists():
                saved.append(out)
    print(f"\nDone. {len(saved)}/{len(slugs) * len(kinds)} figures saved.")


if __name__ == "__main__":
    main()
