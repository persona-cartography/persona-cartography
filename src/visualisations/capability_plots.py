"""Capability sweep plotting for personality eval results.

Verbatim migration of the capability plotters from
``src_dev/evals/personality/analyze_results.py``.  The shared error-bar / xtick
helpers live in :mod:`src.visualisations.plot_common`.

Function bodies are byte-for-byte identical to the source; only module-level
imports differ.
"""

# Standard library
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd

# Local
from src.evals.personality.ci import IntervalMethod, _agg_sweep
from src.evals.personality.logprob_scorer import MIN_CHOICE_MASS_DEFAULT
from src.visualisations.plot_common import _draw_col_error_bars, _set_scale_xticks


def plot_capability_sweep(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = "",
    eval_name: str = "capability",
    random_baseline: float | None = None,
    interval: IntervalMethod | None = None,
    min_choice_mass: float = MIN_CHOICE_MASS_DEFAULT,
    dynamic_mass_filter: bool = True,
    x_label: str = "LoRA scaling factor",
) -> Path:
    """Capability coherence plot: accuracy vs. LoRA scale with baseline reference.

    Suitable for any accuracy-like eval (mmlu, gsm8k, truthfulqa, arc, etc.).

    Args:
        df: DataFrame with columns: scale, run, accuracy.
        output_dir: Directory to save the figure.
        title_suffix: Optional suffix appended to the figure title.
        eval_name: Eval name used for the figure title and output filename.
        random_baseline: If set, draws a horizontal dashed red line at this accuracy
            level (e.g. 0.25 for 4-choice MCQ random chance).
        interval: Error bar method. None to omit error bars.
        min_choice_mass: Fixed min choice-mass filter (logprob evals).
        dynamic_mass_filter: Per-question 1/num_choices filter (logprob evals).

    Returns:
        Path to the saved figure.
    """
    import matplotlib.pyplot as plt

    cap_agg = _agg_sweep(df, ["accuracy"], interval=interval, min_choice_mass=min_choice_mass, dynamic_mass_filter=dynamic_mass_filter)
    scales = cap_agg["scale"].values
    means  = cap_agg["accuracy_mean"].values

    # Compute y-axis bounds from whichever CI columns are present
    if "accuracy_ci" in cap_agg.columns:
        ci_extent_low = means - cap_agg["accuracy_ci"].values
        ci_extent_high = means + cap_agg["accuracy_ci"].values
    elif "accuracy_ci_low" in cap_agg.columns:
        ci_extent_low = cap_agg["accuracy_ci_low"].values
        ci_extent_high = cap_agg["accuracy_ci_high"].values
    else:
        ci_extent_low = means
        ci_extent_high = means

    baseline_idx = int(np.argmin(np.abs(scales)))
    baseline_acc = float(means[baseline_idx])

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.axhline(baseline_acc, color="#388E3C", linewidth=1.2,
               linestyle="--", alpha=0.8, zorder=2, label=f"Baseline ({baseline_acc:.3f})")

    if random_baseline is not None:
        ax.axhline(random_baseline, color="#EF5350", linewidth=1.0,
                   linestyle=":", alpha=0.7, zorder=2, label=f"Random ({random_baseline:.0%})")

    color = "#5C6BC0"
    ax.plot(scales, means, "o-", color=color, linewidth=2.2, markersize=6,
            label="accuracy", zorder=4)
    _draw_col_error_bars(ax, cap_agg, "accuracy", scales, means, color)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.5, zorder=1)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    _set_scale_xticks(ax, scales)

    y_min_candidates = [float(np.nanmin(ci_extent_low))]
    if random_baseline is not None:
        y_min_candidates.append(random_baseline)
    y_min = min(y_min_candidates) - 0.02
    y_max = max(float(np.nanmax(ci_extent_high)), baseline_acc) + 0.04
    ax.set_ylim(y_min, y_max)

    ax.grid(True, alpha=0.25)

    title = f"{eval_name} sweep: capability coherence vs. LoRA scale"
    if title_suffix:
        title += f"  [{title_suffix}]"
    ax.set_title(title, fontsize=13, fontweight="bold")

    if interval is not None:
        ax.errorbar([], [], yerr=1, fmt="none", color="gray", capsize=3, capthick=1.0,
                    elinewidth=1.0, alpha=0.7, label=interval.label)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              fontsize=9, ncol=4, framealpha=0.85)

    plt.tight_layout()
    out = output_dir / f"{eval_name}_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out


def plot_capability_breakdown(
    df: pd.DataFrame,
    output_dir: Path,
    title_suffix: str = "",
    eval_name: str = "capability",
) -> Path | None:
    """Stacked bar chart: Correct / Recovered / Wrong / No answer per LoRA scale.

    Categories:
      - Correct: Inspect scorer parsed and matched target
      - Recovered: Inspect failed to parse, but fallback regex found the right letter
      - Wrong answer: a letter was parsed (by Inspect or fallback) but it was wrong
      - No answer: no parseable letter at all

    Requires ``_raw_accuracy``, ``_raw__answer_parsed``, and
    ``_raw__reparsed_accuracy`` columns from :func:`_extract_raw_sample_scores`.
    """
    import matplotlib.pyplot as plt

    raw_acc_col = "_raw_accuracy"
    raw_ap_col = "_raw__answer_parsed"
    raw_rp_col = "_raw__reparsed_accuracy"
    if raw_acc_col not in df.columns or raw_ap_col not in df.columns:
        print(f"  skip {eval_name}_breakdown: missing raw columns")
        return None
    has_reparse = raw_rp_col in df.columns

    rows = []
    for scale, grp in df.groupby("scale"):
        acc_lists = grp[raw_acc_col].dropna().tolist()
        ap_lists = grp[raw_ap_col].dropna().tolist()
        acc = np.concatenate(acc_lists) if acc_lists else np.array([])
        ap = np.concatenate(ap_lists) if ap_lists else np.array([])
        n = min(len(acc), len(ap))
        if n == 0:
            continue
        acc, ap = acc[:n], ap[:n]

        if has_reparse:
            rp_lists = grp[raw_rp_col].dropna().tolist()
            rp = np.concatenate(rp_lists) if rp_lists else np.zeros(n)
            rp = rp[:n]
            # Recovered = not correct by Inspect, but correct after reparse
            recovered = (1 - acc) * rp
            # Wrong = parsed (by Inspect) but wrong, OR reparsed but wrong
            # i.e. not correct and not recovered and answer was parsed or reparse attempted
            wrong = (1 - acc) * (1 - rp) * ap
            # No answer = not correct, not recovered, no parse at all
            no_answer = (1 - acc) * (1 - rp) * (1 - ap)
        else:
            recovered = np.zeros(n)
            wrong = (1 - acc) * ap
            no_answer = (1 - acc) * (1 - ap)

        rows.append({
            "scale": scale,
            "Correct": float(acc.mean()),
            "Recovered": float(recovered.mean()),
            "Wrong answer": float(wrong.mean()),
            "No answer": float(no_answer.mean()),
        })
    if not rows:
        return None
    agg = pd.DataFrame(rows).sort_values("scale")

    scales = agg["scale"].values
    x = np.arange(len(scales))
    width = 0.85
    cats = ["Correct", "Recovered", "Wrong answer", "No answer"]
    colors = {
        "Correct": "#2ecc71", "Recovered": "#3498db",
        "Wrong answer": "#e74c3c", "No answer": "#95a5a6",
    }

    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150)
    bottom = np.zeros(len(scales))
    for cat in cats:
        vals = agg[cat].values
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, width, bottom=bottom, label=cat, color=colors[cat],
               alpha=0.55, edgecolor="white", linewidth=0.3)
        bottom += vals

    labels = [f"{s:+.2f}" if s != 0 else "base" for s in scales]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Fraction of samples", fontsize=11)
    ax.set_xlabel("LoRA scaling factor", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    base_acc = agg.loc[agg["scale"] == 0.0, "Correct"]
    if len(base_acc):
        ax.axhline(y=float(base_acc.iloc[0]), color="green", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(y=0.25, color="red", linestyle=":", alpha=0.3, linewidth=1)
    if 0.0 in set(scales):
        ax.axvline(x=int(np.where(scales == 0.0)[0][0]), color="black", linestyle="--", alpha=0.3, linewidth=0.8)

    title = f"{eval_name} response breakdown vs. LoRA scale"
    if title_suffix:
        title += f"\n[{title_suffix}]"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = output_dir / f"{eval_name}_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {out}")
    return out
