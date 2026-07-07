#!/usr/bin/env python3
"""Score per-condition coherence for a hill-climbing run (capability axis).

Subsamples responses per condition (mixing harmful + benign) and runs the
:class:`CoherenceJudge` over them, then writes a per-condition mean coherence
CSV that ``plot_hill_climb.py`` uses to paint the safety–capability plane's
background field.

Reads response JSONLs from a local dir (pulled from HF); writes
``coherence_by_condition.csv`` next to them.

    uv run python -m scripts_dev.evals.persona_hill_climbing.score_coherence \
        --responses-dir scratch/persona_hill_climbing/_hf_responses/.../hc_grid_v2_train/responses \
        --n-per-condition 40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src_dev.persona_jailbreak_eval.judge_coherence import (
    COHERENCE_SCORE_MAP,
    CoherenceJudge,
)
from src_dev.persona_metrics.config import JudgeLLMConfig

SEED = 42


def _format_request(row: dict) -> str:
    sys_p = row.get("system_prompt") or ""
    user_p = row.get("user_prompt") or ""
    return f"[SYSTEM]\n{sys_p}\n\n[USER]\n{user_p}" if sys_p else user_p


def load_subsample(path: Path, n: int, rng: random.Random) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if len(rows) > n:
        rows = rng.sample(rows, n)
    return rows


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None,
                        help="output CSV (default: <responses-dir>/../aggregate/coherence_by_condition.csv)")
    parser.add_argument("--n-per-condition", type=int, default=40)
    parser.add_argument("--model", default="deepseek/deepseek-chat-v3")
    parser.add_argument("--max-concurrent", type=int, default=16)
    args = parser.parse_args()

    rng = random.Random(SEED)
    resp_files = sorted(args.responses_dir.glob("responses_*.jsonl"))
    if not resp_files:
        raise SystemExit(f"no responses_*.jsonl under {args.responses_dir}")

    judge = CoherenceJudge(JudgeLLMConfig(
        provider="openrouter", model=args.model,
        max_concurrent=args.max_concurrent, temperature=0.0, max_tokens=256,
    ))

    out = args.out or (args.responses_dir.parent / "aggregate" / "coherence_by_condition.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for f in resp_files:
        condition = f.stem[len("responses_"):]
        rows = load_subsample(f, args.n_per_condition, rng)
        inputs = [{"request": _format_request(r), "response": r.get("response", "")} for r in rows]
        outcomes = asyncio.run(judge.judge_batch(inputs))
        scored = [COHERENCE_SCORE_MAP[o.label] for o in outcomes if o.label in COHERENCE_SCORE_MAP]
        n_ok = len(scored)
        mean_coh = sum(scored) / n_ok if n_ok else float("nan")
        frac_degen = sum(1 for o in outcomes if o.label in ("degenerate", "empty")) / max(1, len(outcomes))
        records.append({
            "condition": condition, "n_judged": n_ok,
            "coherence": round(mean_coh, 4), "frac_degenerate": round(frac_degen, 4),
        })
        print(f"  {condition:52s} coh={mean_coh:.3f}  degen={frac_degen:.2f}  (n={n_ok})")

    df = pd.DataFrame(records).sort_values("coherence")
    df.to_csv(out, index=False)
    print(f"\n  wrote {out}  ({len(df)} conditions)")


if __name__ == "__main__":
    main()
