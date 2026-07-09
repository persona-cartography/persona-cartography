"""Combination hill-climb on the misalignment MCQ: sorted bars, train + held-out test.

Horizontal bars = train misalignment rate (Wilson 95% CI) per LoRA soup, sorted;
◆ = held-out test rate for the conditions carried to the test phase. Bars are
coloured by direction (A− disagreeable = dangerous, A+ agreeable = safe) and
hatched grey when collapsed (answered-rate < 0.8). Reads the two
``misalignment_rate_by_condition.csv`` files that ``run_hill_climb_mcq`` writes.

Data source: HF monorepo
``evals/persona_hill_climbing/gemma-3-27b-it/mcq_v2_{train,test}/aggregate/``.

    uv run python scripts_dev/evals/persona_hill_climbing/plot_mcq_combos.py \
        mcq_v2_train_by_condition.csv mcq_v2_test_by_condition.csv out.png
"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

train_csv, test_csv, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
tr = {r["condition"]: r for r in csv.DictReader(open(train_csv))}
te = {r["condition"]: r for r in csv.DictReader(open(test_csv))}
van = float(tr["vanilla"]["rate"])


def label(cond):
    if cond == "vanilla":
        return "vanilla"
    parts = cond.replace("lora_soup_", "").split("_")
    out = []
    i = 0
    while i < len(parts):
        t, sign, mag = parts[i], parts[i + 1], parts[i + 2]
        out.append(f"{t.upper()}{'+' if sign=='plus' else '−'}{mag}")
        i += 3
    return " ".join(out)


rows = []
for cond, r in tr.items():
    rate = float(r["rate"]); ans = float(r["answered_rate"])
    rows.append((cond, rate, float(r["ci_low"]), float(r["ci_high"]), ans))
rows.sort(key=lambda x: x[1])

fig, ax = plt.subplots(figsize=(9.2, 6.4))
ys = range(len(rows))
for i, (cond, rate, lo, hi, ans) in enumerate(rows):
    collapsed = ans < 0.8
    if collapsed:
        color, hatch = "#bbbbbb", "///"
    elif "a_minus" in cond:
        color, hatch = "#E15759", None      # dangerous direction
    elif "a_plus" in cond:
        color, hatch = "#4E79A7", None      # safe direction
    else:
        color, hatch = "#8a8a8a", None      # vanilla
    ax.barh(i, rate, color=color, hatch=hatch, edgecolor="white", zorder=2,
            xerr=[[rate - lo], [hi - rate]], error_kw=dict(ecolor="#555", lw=1, capsize=2))
    # held-out test marker
    if cond in te:
        tr_rate = float(te[cond]["rate"])
        ax.plot(tr_rate, i, "D", ms=6, color="black", zorder=4)
    note = f"{rate:.2f}"
    if collapsed:
        note += f"  (answered {ans:.0%} — collapsed)"
    ax.text(rate + 0.012, i, note, va="center", fontsize=8.5, color="#222")

ax.axvline(van, color="#666", ls="--", lw=1.1, zorder=1)
ax.text(van, len(rows) - 0.3, f"vanilla {van:.2f}", fontsize=8.5, color="#444", ha="center")
ax.set_yticks(list(ys))
ax.set_yticklabels([label(c) for c, *_ in rows], fontsize=9)
ax.set_xlabel("misalignment rate  (P picks misaligned option, answered items)")
ax.set_xlim(0, 0.68)
ax.set_title("Composing OCEAN persona LoRAs on the discourse-grounded misalignment MCQ\n"
             "gemma-3-27b-it · n≈285/cond · bars = train (Wilson 95% CI) · ◆ = held-out test",
             fontsize=10.5)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="#E15759", label="disagreeable (A−) soups — more dangerous"),
    Patch(color="#4E79A7", label="agreeable (A+) soups — safe"),
    Patch(facecolor="#bbbbbb", hatch="///", label="collapsed (answered < 80%)"),
    plt.Line2D([0], [0], marker="D", color="black", ls="", label="held-out test"),
], frameon=False, fontsize=8.5, loc="lower right")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.invert_yaxis()
fig.tight_layout()
fig.savefig(out_path, dpi=160, bbox_inches="tight")
fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
print("wrote", out_path)
