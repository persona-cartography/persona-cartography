"""MMLU capability eval for PsychAdapter big5_model (custom logprob MCQ scorer).

PsychAdapter's persona is a per-layer KV-prefix on gemma-2b base, computed from a
Big-Five latent — it can't be served through Inspect/vLLM, so our standard
``inspect_sweep`` MMLU runner is not usable. This scores the **standard MMLU
dataset** by the model's own forward pass: format each question as an A/B/C/D
prompt and pick the letter with the highest next-token logit under the requested
conditioning. NOTE: this is a bespoke scorer, NOT our Inspect MMLU — used only
because the model is architecturally incompatible with the served pipeline.

The point is the CONTRAST: MMLU accuracy at baseline (0 std) vs under trait
conditioning (and OOD extremes) — i.e. does steering the persona cost capability?

Runs in the ISOLATED venv (transformers 4.39.2). Needs `datasets`:
    scratch/psychadapter_eval/_venv/bin/pip install datasets
    PA_ASSETS=scratch/psychadapter_eval/_assets \
      scratch/psychadapter_eval/_venv/bin/python scripts_dev/psychadapter_eval/eval_mmlu.py
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
import generate_psychadapter as g  # reuse load_model / device / latent_for

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

N_QUESTIONS = int(os.environ.get("PA_MMLU_N", "300"))
N_FEWSHOT = int(os.environ.get("PA_MMLU_FEWSHOT", "5"))  # MMLU is conventionally 5-shot
OUT_DIR = Path(g.OUT_PATH).parent / "mmlu"
TRAITS = g.TRAITS

# Two modes:
#  - default: 7 discrete conditions (baseline + each trait@+3 + neuroticism@+5)
#  - PA_MMLU_SWEEP=1: MMLU dose-response — each trait across std -5..+5 (line plot,
#    like the trait judge curve), so you can see if steering costs capability.
MMLU_SWEEP = bool(os.environ.get("PA_MMLU_SWEEP"))
SWEEP_STDS = [-5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0]  # 0 = baseline


def build_conditions() -> list[tuple[str, int | None, float]]:
    """(label, latent dim or None, std value) — same N questions evaluated at each."""
    conds = [("baseline", None, 0.0)]
    if MMLU_SWEEP:
        for i, t in enumerate(TRAITS):
            for s in SWEEP_STDS:
                conds.append((f"{t}@{s:+g}", i, s))
    else:
        conds += [(f"{t}@+3", i, 3.0) for i, t in enumerate(TRAITS)]
        conds.append(("neuroticism@+5", 4, 5.0))  # OOD extreme
    return conds


CONDITIONS = build_conditions()


def load_mmlu(n: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)
    rows = [ds[i] for i in idx[:n]]
    return rows  # fields: question, choices (4), answer (int 0-3), subject


def _letter_token_ids(tok) -> list[int]:
    """Single-token ids for the answer letters A-D after 'Answer:'."""
    for variant in (lambda L: " " + L, lambda L: L, lambda L: " " + L.lower(), lambda L: L.lower()):
        ids = [tok.encode(variant(L), add_special_tokens=False) for L in "ABCD"]
        last = [i[-1] for i in ids if i]
        if len(last) == 4 and len(set(last)) == 4:
            return last
    raise RuntimeError("Could not find 4 distinct single-token ids for A-D")


def _format_q(row: dict) -> str:
    letters = "ABCD"
    opts = "\n".join(f"{letters[j]}. {c}" for j, c in enumerate(row["choices"]))
    return f"{row['question']}\n{opts}\nAnswer:"


def build_fewshot_prefix(n: int) -> str:
    """Fixed n-shot prefix from the MMLU dev split (same for every question)."""
    if n <= 0:
        return ""
    from datasets import load_dataset

    dev = load_dataset("cais/mmlu", "all", split="dev")
    idx = list(range(len(dev)))
    random.Random(SEED).shuffle(idx)
    shots = []
    for i in idx[:n]:
        r = dev[i]
        shots.append(_format_q(r) + f" {'ABCD'[int(r['answer'])]}")
    return "\n\n".join(shots) + "\n\n"


def _prompt(row: dict, prefix: str = "") -> str:
    return (
        "The following are multiple choice questions. Answer with the letter of "
        "the correct option.\n\n" + prefix + _format_q(row)
    )


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (binary accuracy)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = (z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)) / d
    return (max(0.0, center - half), min(1.0, center + half))


def score_question(inner, tok, latent, prompt, device, letter_ids) -> int:
    ids = tok.encode("<bos>" + prompt, add_special_tokens=False)
    input_ids = torch.tensor([ids], device=device).long()
    attn = torch.ones((1, len(ids) + 1), device=device).long()  # +1 for KV-prefix
    emb = torch.tensor(latent, device=device).float().unsqueeze(0)
    with torch.no_grad():
        logits = inner.forward(emb, input_ids, attn, device)  # [1, seq, vocab]
    last = logits[0, -1]
    return int(np.argmax([last[i].item() for i in letter_ids]))


def main():
    device = g._pick_device()
    print(f"Device: {device} | N={N_QUESTIONS} conditions={len(CONDITIONS)}")
    model, args = g.load_model(device)
    inner = model.base_model.model  # PsychAdapter (LoRA active in-place on decoder)
    tok = inner.tokenizer
    letter_ids = _letter_token_ids(tok)
    print("letter token ids (A,B,C,D):", letter_ids)

    rows = load_mmlu(N_QUESTIONS)
    prefix = build_fewshot_prefix(N_FEWSHOT)
    print(f"few-shot: {N_FEWSHOT}-shot")
    prompts = [_prompt(r, prefix) for r in rows]
    gold = [int(r["answer"]) for r in rows]

    n = len(rows)
    results = {}
    for label, dim, val in CONDITIONS:
        latent = g.latent_for(dim, val)
        correct = sum(
            int(score_question(inner, tok, latent, p, device, letter_ids) == gi)
            for p, gi in zip(prompts, gold)
        )
        acc = correct / n
        lo, hi = _wilson(correct, n)
        results[label] = {"acc": acc, "ci_lo": lo, "ci_hi": hi, "correct": correct}
        print(f"  {label:<18} acc={acc:.3f}  [{lo:.3f},{hi:.3f}]  ({correct}/{n})", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "_sweep" if MMLU_SWEEP else ""
    (OUT_DIR / f"mmlu_results{tag}.json").write_text(
        json.dumps({"n": n, "n_fewshot": N_FEWSHOT, "chance": 0.25, "results": results}, indent=2)
    )
    (_plot_sweep if MMLU_SWEEP else _plot)(results)
    print(f"\nWrote -> {OUT_DIR}")


def _plot(results: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(skip plot: {e})")
        return
    labels = list(results)
    vals = [results[k]["acc"] for k in labels]
    err = [
        [results[k]["acc"] - results[k]["ci_lo"] for k in labels],
        [results[k]["ci_hi"] - results[k]["acc"] for k in labels],
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#555"] + ["#4C72B0"] * 5 + ["#C44E52"]
    ax.bar(range(len(labels)), vals, color=colors[: len(labels)], yerr=err, capsize=3)
    ax.axhline(0.25, color="k", ls="--", lw=0.8, label="chance (0.25)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("MMLU accuracy")
    ax.set_title("PsychAdapter big5: MMLU accuracy vs conditioning (gemma-2b base)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_psychadapter_mmlu.png", dpi=150)


def _plot_sweep(results: dict) -> None:
    """MMLU dose-response: accuracy vs std, one line per trait (0 = baseline)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        print(f"(skip plot: {e})")
        return
    base = results["baseline"]
    fig, ax = plt.subplots(figsize=(8, 5))
    csv = ["trait," + ",".join(str(s) for s in sorted(SWEEP_STDS + [0.0]))]
    print("\n=== MMLU dose-response (accuracy vs std) ===")
    for i, t in enumerate(TRAITS):
        xs = sorted(SWEEP_STDS + [0.0])
        ys, los, his = [], [], []
        for s in xs:
            r = base if s == 0.0 else results[f"{t}@{s:+g}"]
            ys.append(r["acc"]); los.append(r["ci_lo"]); his.append(r["ci_hi"])
        ax.plot(xs, ys, marker="o", label=t)
        csv.append(t + "," + ",".join(f"{y:.3f}" for y in ys))
        print(f"{t:<16}" + "".join(f"{y:>7.2f}" for y in ys))
    ax.axhline(0.25, color="k", ls="--", lw=0.8, label="chance")
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.set_xlabel("latent conditioning (std units)")
    ax.set_ylabel("MMLU accuracy")
    ax.set_title("PsychAdapter big5: MMLU vs conditioning (gemma-2b base, 5-shot)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_psychadapter_mmlu_sweep.png", dpi=150)
    (OUT_DIR / "mmlu_sweep.csv").write_text("\n".join(csv) + "\n")


if __name__ == "__main__":
    main()
