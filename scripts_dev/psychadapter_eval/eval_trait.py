"""TRAIT (mirlab/TRAIT) MCQ eval for PsychAdapter big5 — logprob 'TRAIT lines'.

The MCQ counterpart to the OCEAN free-text judge sweep. For each OCEAN trait we
condition that trait's latent across std positions and score its TRAIT questions
by logprob: read P(A/B/C/D) from the model's forward pass, softmax, and compute
P(high trait) = sum_L P(L)*answer_mapping[L]  (the same scoring as our
src_dev/evals/personality.logprob_mcq_scorer). Plot P(high) vs std -> one line
per trait. 0.5 = indifferent between high/low responses.

Reuses the forward-scorer machinery from eval_mmlu. Runs in the ISOLATED venv.
Needs trait_items.jsonl from prep_trait_items.py (run in the repo env first).

    PA_ASSETS=scratch/psychadapter_eval/_assets \
      scratch/psychadapter_eval/_venv/bin/python scripts_dev/psychadapter_eval/eval_trait.py
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import generate_psychadapter as g
import eval_mmlu as em

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

ITEMS = Path(__file__).resolve().parents[2] / "scratch/psychadapter_eval/trait_items.jsonl"
OUT_DIR = Path(g.OUT_PATH).parent / "trait"
TRAITS = g.TRAITS  # lowercase order matching latent dims
TRAIT_CAP = {t: t.capitalize() for t in TRAITS}  # 'openness' -> 'Openness'

# std positions for the TRAIT lines (same axis as the judge dose-response).
_pos = os.environ.get("PA_TRAIT_POSITIONS", "-5,-3,-2,-1,0,1,2,3,5")
POSITIONS = sorted(float(x) for x in _pos.split(",") if x.strip())


def score_phigh(inner, tok, latent, item, device, letter_ids) -> float | None:
    """P(high trait) = sum_L softmax(P(L)) * answer_mapping[L] for an MCQ item."""
    letters = "ABCD"
    opts = "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(item["choices"]))
    prompt = f"{item['question']}\n{opts}\nANSWER:"
    ids = tok.encode("<bos>" + prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids], device=device).long()
    attn = torch.ones((1, len(ids) + 1), device=device).long()
    emb = torch.tensor(latent, device=device).float().unsqueeze(0)
    with torch.no_grad():
        logits = inner.forward(emb, input_ids, attn, device)
    last = logits[0, -1]
    lp = np.array([last[i].item() for i in letter_ids], dtype=np.float64)
    p = np.exp(lp - lp.max())
    p = p / p.sum()  # softmax over the 4 choice letters
    mapping = item["answer_mapping"]
    return float(sum(p[j] * mapping[letters[j]] for j in range(4)))


def main():
    device = g._pick_device()
    print(f"Device: {device} | positions={POSITIONS}")
    items = [json.loads(l) for l in ITEMS.read_text().splitlines() if l.strip()]
    by_trait = {t: [it for it in items if it["trait"] == TRAIT_CAP[t]] for t in TRAITS}
    print("items per trait:", {t: len(v) for t, v in by_trait.items()})

    model, args = g.load_model(device)
    inner = model.base_model.model
    tok = inner.tokenizer
    letter_ids = em._letter_token_ids(tok)

    # curves[trait][std] = mean P(high) over that trait's items at that conditioning
    curves: dict[str, dict[float, float]] = {t: {} for t in TRAITS}
    for ti, trait in enumerate(TRAITS):
        for pos in POSITIONS:
            latent = g.latent_for(ti, pos)
            ph = [score_phigh(inner, tok, latent, it, device, letter_ids) for it in by_trait[trait]]
            curves[trait][pos] = float(np.mean(ph))
            print(f"  {trait:<16} std{pos:+g}  P(high)={curves[trait][pos]:.3f}  (n={len(ph)})", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "trait_lines.json").write_text(json.dumps(curves, indent=2))
    allpos = sorted(POSITIONS)
    csv = ["trait," + ",".join(f"{p:g}" for p in allpos)]
    for t in TRAITS:
        csv.append(t + "," + ",".join(f"{curves[t].get(p, float('nan')):.3f}" for p in allpos))
    (OUT_DIR / "trait_lines.csv").write_text("\n".join(csv) + "\n")
    _plot(curves, allpos)
    print(f"\nWrote -> {OUT_DIR}")


def _plot(curves: dict, allpos: list) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(skip plot: {e})")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for t in TRAITS:
        ys = [curves[t][p] for p in allpos]
        ax.plot(allpos, ys, marker="o", label=t)
    ax.axhline(0.5, color="k", lw=0.5, ls="--", label="indifferent (0.5)")
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("latent conditioning (std units)")
    ax.set_ylabel("P(high-trait answer) on TRAIT MCQ")
    ax.set_title("PsychAdapter big5: TRAIT (mirlab/TRAIT) MCQ dose-response")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_psychadapter_trait_lines.png", dpi=150)


if __name__ == "__main__":
    main()
