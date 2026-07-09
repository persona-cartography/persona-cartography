"""Single-trait dose-response (misalignment + answered-rate) on the misalignment MCQ.

Two stacked panels vs the persona-adapter coefficient (one line per OCEAN trait):
top = misalignment rate (P picks the misaligned option, Wilson 95% CI); bottom =
answered rate (fraction with a cleanly-parsed choice — the echo/collapse gate,
shaded red below 0.8). Reads the ``misalignment_rate_by_condition.csv`` that
``run_hill_climb_mcq`` writes (it carries ``answered_rate`` in the extras column).

Data source: HF monorepo
``evals/persona_hill_climbing/gemma-3-27b-it/mcq_v1_train/aggregate/``.

    uv run python scripts_dev/evals/persona_hill_climbing/plot_mcq_dose_response.py \
        misalignment_rate_by_condition.csv out.png
"""
import csv
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path, out_path = sys.argv[1], sys.argv[2]
by = {r["condition"]: r for r in csv.DictReader(open(csv_path))}
van = by["vanilla"]
van_rate, van_ans = float(van["rate"]), float(van["answered_rate"])

TRAITS = {"o": "Openness", "c": "Conscientiousness", "e": "Extraversion",
          "a": "Agreeableness", "n": "Neuroticism"}
COLORS = {"o": "#4E79A7", "c": "#59A14F", "e": "#B07AA1", "a": "#E15759", "n": "#F28E2B"}
ORDER = ["a", "e", "n", "c", "o"]  # A first (dominant)

pat = re.compile(r"lora_soup_([ocean])_(plus|minus)_([0-9.]+)")
mis = defaultdict(list); ans = defaultdict(list)
for cond, r in by.items():
    m = pat.fullmatch(cond)
    if not m:
        continue
    t, sign, mag = m.group(1), m.group(2), float(m.group(3))
    coeff = mag if sign == "plus" else -mag
    mis[t].append((coeff, float(r["rate"]), float(r["ci_low"]), float(r["ci_high"])))
    ans[t].append((coeff, float(r["answered_rate"])))

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8.4, 7.2), sharex=True,
                              gridspec_kw={"height_ratios": [2.4, 1]})

ax.axhline(van_rate, color="#666", lw=1.1, ls="--", zorder=1)
ax.text(1.53, van_rate, f"  vanilla {van_rate:.2f}", va="center", fontsize=8.5, color="#444")
for t in ORDER:
    pts = sorted(mis[t] + [(0.0, van_rate, float(van["ci_low"]), float(van["ci_high"]))])
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    lo = [p[1] - p[2] for p in pts]; hi = [p[3] - p[1] for p in pts]
    ax.errorbar(xs, ys, yerr=[lo, hi], color=COLORS[t], lw=2.0, marker="o",
                ms=6, capsize=3, label=TRAITS[t], zorder=3)
ax.set_ylabel("misalignment rate\n(P picks misaligned option)")
ax.set_title("Single-trait persona dose-response on discourse-grounded misalignment MCQ\n"
             "gemma-3-27b-it · textbook_questions train split · Wilson 95% CI",
             fontsize=10.5)
ax.set_ylim(0, 0.62)
ax.grid(True, axis="y", color="#eee", zorder=0)
ax.legend(title="OCEAN trait", frameon=False, loc="upper center", ncol=2, fontsize=8.5)

# answered-rate panel (the capability/collapse gate)
ax2.axhline(van_ans, color="#666", lw=1.1, ls="--", zorder=1)
ax2.axhspan(0, 0.8, color="#f6d6d6", alpha=0.5, zorder=0)
ax2.text(1.53, 0.72, "  echo /\n  collapse", va="center", fontsize=7.5, color="#a33")
for t in ORDER:
    pts = sorted(ans[t] + [(0.0, van_ans)])
    ax2.plot([p[0] for p in pts], [p[1] for p in pts], color=COLORS[t],
             lw=1.8, marker="o", ms=5, zorder=3)
ax2.set_ylabel("answered rate\n(clean choice parsed)")
ax2.set_xlabel("persona-adapter coefficient  (− suppressor · + amplifier)")
ax2.set_xticks([-1.5, -0.75, 0, 0.75, 1.5])
ax2.set_ylim(0.5, 1.02)
ax2.grid(True, axis="y", color="#eee", zorder=0)

for a in (ax, ax2):
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(out_path, dpi=160, bbox_inches="tight")
fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
print("wrote", out_path)
