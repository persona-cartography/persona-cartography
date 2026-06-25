"""Paper figure: WildJailbreak harm-rate + benign-noncompliance bar charts.

Reads consolidated per-condition rates from the ``wj_paper_v1`` HF folder
produced by ``scripts_dev.persona_jailbreak_eval.collate_paper_results`` and
renders side-by-side bar charts:

  - Left:  harmful-response rate on the WildJailbreak ``adversarial_harmful``
           split (n=400 prompts), 95% Wilson CI.
  - Right: noncompliance rate on the ``adversarial_benign`` over-refusal
           control (n=100), 95% Wilson CI.

All conditions are reported on the same 500 prompts (the canonical ablation
sample-id set), so cross-condition comparisons are apples-to-apples.

Two presets, both editable via the ``PRESETS`` dict at the bottom:

  - ``main``: vanilla, activation_capping, a+, c+, a+0.5 ⊕ c+0.5 soup.
             Goes in the main body (Section 3 — persona drift).
  - ``appendix``: vanilla, activation_capping, all 10 OCEAN ↑/↓ LoRAs at +1,
                  + control LoRA. Goes in Appendix.

Paper figures (PDF + PNG):
    paper/figures/main/fig_3_wj_persona_drift.pdf       (preset: main)
    paper/figures/appendix/ocean_results/fig_F_wj_persona_drift_full.pdf  (preset: appendix)

Run with:
    uv run python -m src_dev.visualisations.paper_wj_persona_drift
    uv run python -m src_dev.visualisations.paper_wj_persona_drift --preset main
    uv run python -m src_dev.visualisations.paper_wj_persona_drift --preset appendix
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

from src_dev.evals.personality.analyze_results import BIG_FIVE_COLORS
from src_dev.visualisations import PAPER_FIGURES_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_REPO_ID = "persona-cartography/monorepo"
HF_BASE = "evals/persona_jailbreak_wildjailbreak/llama-3.1-8b-instruct"
LOCAL_BASE = (
    project_root / "scratch" / "persona_jailbreak_eval" / "llama-3.1-8b-instruct"
)

# Per-preset consolidated source folder (under HF_BASE / under LOCAL_BASE).
# Each is a self-contained set of {judgments, aggregate, manifest.json}
# produced by ``scripts_dev.persona_jailbreak_eval.collate_paper_results``.
SOURCE_FOLDER_BY_PRESET: dict[str, str] = {
    "main":     "wj_paper_main_balanced_v1",  # 1010 prompts, 5 conditions
    "appendix": "wj_paper_v1",                # 500 prompts, 18 conditions
}


# Per-condition display label and color.  Defaults below cover every
# condition in ``wj_paper_v1``; presets pick a subset.
CONDITION_LABEL: dict[str, str] = {
    "vanilla":                            "baseline",
    "activation_capping":                 "act. capping",
    "lora_soup_o_plus_1.0":               "O↑",
    "lora_soup_o_minus_1.0":              "O↓",
    "lora_soup_c_plus_1.0":               "C↑",
    "lora_soup_c_minus_1.0":              "C↓",
    "lora_soup_e_plus_1.0":               "E↑",
    "lora_soup_e_minus_1.0":              "E↓",
    "lora_soup_a_plus_1.0":               "A↑",
    "lora_soup_a_minus_1.0":              "A↓",
    "lora_soup_n_plus_1.0":               "N↑",
    "lora_soup_n_minus_1.0":              "N↓",
    "lora_soup_control_latest_1.0":       "control",
    "lora_soup_control_legacy_1.0":       "control (legacy)",
    "lora_soup_a_plus_0.5_c_plus_0.5":    "½ (A↑ ⊕ C↑)",
    "lora_soup_a_plus_1.0_c_plus_1.0":    "A↑ ⊕ C↑ (1, 1)",
    "lora_soup_a_plus_1.0_c_plus_0.5":    "A↑ ⊕ C↑ (1, ½)",
    "lora_soup_c_plus_0.5_o_minus_0.5":   "C↑ ⊕ O↓ (½, ½)",
}

_TRAIT_BY_KEY = {
    "lora_soup_o_plus_1.0":  "Openness",          "lora_soup_o_minus_1.0": "Openness",
    "lora_soup_c_plus_1.0":  "Conscientiousness", "lora_soup_c_minus_1.0": "Conscientiousness",
    "lora_soup_e_plus_1.0":  "Extraversion",      "lora_soup_e_minus_1.0": "Extraversion",
    "lora_soup_a_plus_1.0":  "Agreeableness",     "lora_soup_a_minus_1.0": "Agreeableness",
    "lora_soup_n_plus_1.0":  "Neuroticism",       "lora_soup_n_minus_1.0": "Neuroticism",
}

_NEUTRAL_COLOR: dict[str, str] = {
    "vanilla":                            "#5f6368",
    "activation_capping":                 "#455a64",
    "lora_soup_control_latest_1.0":       "#6d4c41",
    "lora_soup_control_legacy_1.0":       "#8d6e63",
    "lora_soup_a_plus_0.5_c_plus_0.5":    "#00838F",
    "lora_soup_a_plus_1.0_c_plus_1.0":    "#00838F",
    "lora_soup_a_plus_1.0_c_plus_0.5":    "#00838F",
    "lora_soup_c_plus_0.5_o_minus_0.5":   "#00897b",
}


def _condition_color(cond: str) -> str:
    if cond in _TRAIT_BY_KEY:
        return BIG_FIVE_COLORS[_TRAIT_BY_KEY[cond]]
    return _NEUTRAL_COLOR.get(cond, "#9e9e9e")


def _condition_hatch(cond: str) -> str | None:
    if cond.endswith("_plus_1.0"):
        return "//"
    if cond.endswith("_minus_1.0"):
        return "\\\\"
    if "a_plus" in cond and "c_plus" in cond:
        return "xx"
    if "c_plus" in cond and "o_minus" in cond:
        return "xx"
    return None


# ---------------------------------------------------------------------------
# Hydration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rate:
    condition: str
    n: int
    rate: float
    ci_low: float
    ci_high: float


def _read_rate_csv(path: Path) -> dict[str, Rate]:
    out: dict[str, Rate] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["condition"]] = Rate(
                condition=row["condition"], n=int(row["n"]),
                rate=float(row["rate"]),
                ci_low=float(row["ci_low"]), ci_high=float(row["ci_high"]),
            )
    return out


def _resolve_csv(folder_name: str, csv_name: str) -> Path:
    """Try HF first, fall back to local scratch dir for a specific source folder."""
    hf_path = f"{HF_BASE}/{folder_name}/aggregate/{csv_name}"
    try:
        return Path(hf_hub_download(HF_REPO_ID, repo_type="dataset", filename=hf_path))
    except (EntryNotFoundError, RepositoryNotFoundError, OSError):
        pass
    except Exception as exc:
        print(f"  ⚠ HF fetch failed for {hf_path}: {type(exc).__name__}: {str(exc)[:120]}")
    local = LOCAL_BASE / folder_name / "aggregate" / csv_name
    if not local.exists():
        raise FileNotFoundError(
            f"Could not find {csv_name} on HF ({hf_path}) or locally ({local}). "
            f"Run scripts_dev/persona_jailbreak_eval/collate_paper_results.py first."
        )
    print(f"  ← local: {local}")
    return local


def hydrate(folder_name: str) -> tuple[dict[str, Rate], dict[str, Rate]]:
    harm  = _read_rate_csv(_resolve_csv(folder_name, "harmful_rate_by_condition.csv"))
    bnign = _read_rate_csv(_resolve_csv(folder_name, "refusal_rate_on_benign.csv"))
    return harm, bnign


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


FS_TITLE     = 13
FS_TICK      = 11
FS_AX_LABEL  = 12
FS_VALUE     = 10


def _yerr(rates: list[Rate]) -> list[list[float]]:
    return [
        [max(0.0, r.rate - r.ci_low) for r in rates],
        [max(0.0, r.ci_high - r.rate) for r in rates],
    ]


def _bar_panel(ax, rates: list[Rate], *, title: str, ylabel: str) -> None:
    labels = [CONDITION_LABEL.get(r.condition, r.condition) for r in rates]
    colors = [_condition_color(r.condition) for r in rates]
    rate_vals = [r.rate for r in rates]

    bars = ax.bar(
        labels, rate_vals,
        yerr=_yerr(rates), capsize=3,
        color=colors, alpha=0.9,
        edgecolor="#222222", linewidth=0.6,
    )
    for patch, r in zip(bars.patches, rates):
        h = _condition_hatch(r.condition)
        if h:
            patch.set_hatch(h)

    ax.set_ylabel(ylabel, fontsize=FS_AX_LABEL)
    ax.set_title(title, fontsize=FS_TITLE)
    top = max(0.5, max((r.ci_high for r in rates), default=0.0) * 1.10)
    ax.set_ylim(0, top)
    ax.tick_params(axis="y", labelsize=FS_VALUE)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")
        tick.set_fontsize(FS_TICK)
    ax.grid(True, axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _make_figure(
    *, conditions: list[str], harm: dict[str, Rate], bnign: dict[str, Rate],
    out_paths: list[Path], width: float, height: float,
) -> None:
    harm_rates  = [harm[c]  for c in conditions if c in harm]
    bnign_rates = [bnign[c] for c in conditions if c in bnign]
    missing = [c for c in conditions if c not in harm or c not in bnign]
    if missing:
        print(f"  ⚠ missing from CSVs: {missing}")

    fig, axes = plt.subplots(1, 2, figsize=(width, height), squeeze=False)
    _bar_panel(
        axes[0, 0], harm_rates,
        title="WildJailbreak harmful",
        ylabel="harmful rate",
    )
    _bar_panel(
        axes[0, 1], bnign_rates,
        title="WildJailbreak Benign",
        ylabel="noncompliance rate",
    )

    fig.tight_layout()
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"  ✓ saved {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Preset:
    name: str
    conditions: list[str]
    out_relpaths: list[str]
    width: float
    height: float


PRESETS: dict[str, Preset] = {
    "main": Preset(
        name="main",
        conditions=[
            "vanilla",
            "activation_capping",
            "lora_soup_control_latest_1.0",
            "lora_soup_a_plus_1.0",
            "lora_soup_c_plus_1.0",
            "lora_soup_a_plus_0.5_c_plus_0.5",
        ],
        out_relpaths=[
            "main/fig_3_wj_persona_drift.pdf",
            "main/fig_3_wj_persona_drift.png",
        ],
        width=11.0, height=3.4,
    ),
    "appendix": Preset(
        name="appendix",
        conditions=[
            "vanilla",
            "activation_capping",
            "lora_soup_control_latest_1.0",
            "lora_soup_o_plus_1.0", "lora_soup_o_minus_1.0",
            "lora_soup_c_plus_1.0", "lora_soup_c_minus_1.0",
            "lora_soup_e_plus_1.0", "lora_soup_e_minus_1.0",
            "lora_soup_a_plus_1.0", "lora_soup_a_minus_1.0",
            "lora_soup_n_plus_1.0", "lora_soup_n_minus_1.0",
        ],
        out_relpaths=[
            "appendix/ocean_results/fig_F_wj_persona_drift_full.pdf",
            "appendix/ocean_results/fig_F_wj_persona_drift_full.png",
        ],
        width=14.0, height=3.6,
    ),
}


PAPER_FIGURES = [rel for p in PRESETS.values() for rel in p.out_relpaths]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=list(PRESETS) + ["all"], default="all")
    args = p.parse_args()

    presets = list(PRESETS.values()) if args.preset == "all" else [PRESETS[args.preset]]
    for preset in presets:
        folder = SOURCE_FOLDER_BY_PRESET[preset.name]
        print(f"\n[wj-persona-drift] preset={preset.name} ← {folder}/aggregate/  "
              f"({len(preset.conditions)} conditions)")
        harm, bnign = hydrate(folder)
        print(f"  ✓ {len(harm)} conditions in harmful CSV / {len(bnign)} in benign CSV")
        out_paths = [PAPER_FIGURES_DIR / r for r in preset.out_relpaths]
        _make_figure(
            conditions=preset.conditions, harm=harm, bnign=bnign,
            out_paths=out_paths, width=preset.width, height=preset.height,
        )


if __name__ == "__main__":
    main()
