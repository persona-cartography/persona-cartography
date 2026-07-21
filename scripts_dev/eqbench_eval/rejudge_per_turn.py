"""Per-turn re-judge of EQ-Bench3 transcripts to measure EQ *progression*.

Upstream EQ-Bench3 collapses each multi-turn scenario into a single holistic
rubric score over the whole trajectory + debrief, so it cannot answer "does the
model degrade as the scenario escalates?". This script re-judges the already
generated transcripts turn by turn, giving a score per turn index.

It does NOT modify or re-run anything upstream: it reads the transcripts out of
the eqbench3 ``runs.json`` and issues its own judge calls. The rubric criteria
and 0-20 scale are kept identical to upstream's six "qualitative / higher is
better" headline criteria so per-turn numbers stay on the same scale as the
headline score.

Only multi-turn role-play scenarios are scored (analysis scenarios are
single-turn and have no progression).

Usage:
    uv run python scripts_dev/eqbench_eval/rejudge_per_turn.py \
        --run-dir scratch/evals/eqbench3/gemma27b_n_sweep
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Upstream's six "Qualitative (higher is better)" criteria, verbatim from
# data/rubric_scoring_criteria.txt / rubric_scoring_prompt.txt.
CRITERIA = [
    "demonstrated_empathy",
    "pragmatic_ei",
    "depth_of_insight",
    "social_dexterity",
    "emotional_reasoning",
    "message_tailoring",
]

ANALYSIS_SCENARIO_IDS = {str(i) for i in range(401, 421)}

JUDGE_TEMPLATE = """You are an expert in assessing emotional intelligence. Below is a role-play transcript. The user's messages are canned, so the assistant cannot affect the trajectory; assess the assistant's responses only.

You are scoring ONLY the FINAL assistant turn (marked "TURN UNDER EVALUATION"). The earlier turns are provided as context so you can judge how well this specific turn handles the situation at this point in the conversation. Do not score the earlier turns.

[TRANSCRIPT START]

{transcript}

[TRANSCRIPT END]

Score the TURN UNDER EVALUATION on each criterion, 0-20.
All of these criteria are "higher is better".
You are a critic; be critical, especially of failings or amateurish elements.

Output valid JSON in exactly this format:

{output_format}"""


def _fmt_assistant(parsed: dict[str, Any] | None, raw: str) -> str:
    """Render an assistant turn using its parsed sections when available."""
    if not isinstance(parsed, dict):
        return raw
    order = [
        ("thinking_feeling", "# I'm thinking & feeling"),
        ("their_thinking_feeling", "# They're thinking & feeling"),
        ("response", "# My response"),
    ]
    parts = [f"{hdr}\n{parsed[key]}" for key, hdr in order if parsed.get(key)]
    return "\n\n".join(parts) if parts else raw


def build_turn_prompt(
    conversation: list[dict], parsed_responses: list[dict], turn_idx: int, output_format: str
) -> str:
    """Build a judge prompt scoring assistant turn ``turn_idx`` (0-based) in context."""
    parts: list[str] = []
    asst_seen = 0
    for msg in conversation:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"User:\n{content}\n")
        elif role == "assistant":
            parsed = parsed_responses[asst_seen] if asst_seen < len(parsed_responses) else None
            body = _fmt_assistant(parsed, content)
            if asst_seen == turn_idx:
                parts.append(f"Assistant [TURN UNDER EVALUATION]:\n{body}\n")
                asst_seen += 1
                break  # stop: later turns must not leak into this turn's judgement
            parts.append(f"Assistant:\n{body}\n")
            asst_seen += 1
    return JUDGE_TEMPLATE.format(
        transcript="---\n".join(parts), output_format=output_format
    )


def _coerce_score(v: Any) -> float | None:
    """Accept a score as a number or as a numeric string.

    The judge is not consistent about this: it returns e.g. ``14`` on some calls
    and ``"14"`` on others. Rejecting the string form silently drops otherwise
    valid judgements, so coerce both.
    """
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def parse_scores(text: str) -> dict[str, float] | None:
    """Extract the criteria scores from the judge's JSON reply."""
    blob = text.strip()
    m = re.search(r"\{.*\}", blob, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for c in CRITERIA:
        val = _coerce_score(data.get(c))
        if val is not None:
            out[c] = max(0.0, min(20.0, val))
    return out or None


def judge_one(prompt: str, model: str, api_key: str, timeout: int = 240) -> dict[str, float] | None:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(5):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            scores = parse_scores(content)
            if scores:
                return scores
        except Exception:
            pass
    return None


def collect_turn_jobs(runs_path: Path) -> list[dict]:
    """One job per assistant turn of every multi-turn, non-analysis scored task."""
    runs = json.loads(runs_path.read_text())
    jobs: list[dict] = []
    for run in runs.values():
        variant = run.get("model_name")
        if not variant:
            continue
        for it, scenarios in (run.get("scenario_tasks") or {}).items():
            for sid, task in (scenarios or {}).items():
                if not isinstance(task, dict) or task.get("status") != "rubric_scored":
                    continue
                if sid in ANALYSIS_SCENARIO_IDS:
                    continue
                conv = task.get("conversation_history") or []
                n_asst = sum(1 for m in conv if m.get("role") == "assistant")
                if n_asst < 2:
                    continue
                for k in range(n_asst):
                    jobs.append({
                        "variant": variant, "iteration": str(it), "scenario_id": str(sid),
                        "turn_index": k, "n_turns": n_asst,
                        "conversation": conv,
                        "parsed_responses": task.get("parsed_responses") or [],
                    })
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path,
                    default=Path("scratch/evals/eqbench3/gemma27b_n_sweep"))
    ap.add_argument("--judge-model", default="anthropic/claude-opus-4-6")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="Cap jobs (for a cheap smoke test).")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="Keep already-scored turns from the output file and only "
                         "re-judge the ones that previously failed.")
    args = ap.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        return 1

    jobs = collect_turn_jobs(args.run_dir / "runs.json")
    if args.limit:
        jobs = jobs[: args.limit]

    out_path = args.out or (args.run_dir / "per_turn_scores.json")
    results: list[dict] = []
    if args.resume and out_path.exists():
        prior = json.loads(out_path.read_text())
        keep = {
            (r["variant"], r["iteration"], r["scenario_id"], r["turn_index"]): r
            for r in prior if r.get("scores")
        }
        results.extend(keep.values())
        jobs = [
            j for j in jobs
            if (j["variant"], j["iteration"], j["scenario_id"], j["turn_index"]) not in keep
        ]
        print(f"resume: kept {len(keep)} previously scored turns; re-judging {len(jobs)}")

    print(f"per-turn judge jobs: {len(jobs)} (judge={args.judge_model}, threads={args.threads})")

    output_format = json.dumps({c: "0-20" for c in CRITERIA}, indent=2)
    lock = threading.Lock()
    done = 0

    def work(job: dict) -> None:
        nonlocal done
        prompt = build_turn_prompt(
            job["conversation"], job["parsed_responses"], job["turn_index"], output_format
        )
        scores = judge_one(prompt, args.judge_model, api_key)
        with lock:
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"  judged {done}/{len(jobs)}")
            results.append({
                "variant": job["variant"], "iteration": job["iteration"],
                "scenario_id": job["scenario_id"], "turn_index": job["turn_index"],
                "n_turns": job["n_turns"], "scores": scores,
                "mean": (float(np.mean(list(scores.values()))) if scores else None),
            })

    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        list(as_completed([ex.submit(work, j) for j in jobs]))

    results.sort(key=lambda r: (r["variant"], r["scenario_id"], r["iteration"], r["turn_index"]))
    out_path.write_text(json.dumps(results, indent=2))
    ok = sum(1 for r in results if r["scores"])
    print(f"\nwrote {out_path}  ({ok}/{len(results)} judged successfully)")

    print("\n=== mean score (0-20) by turn index ===")
    for variant in sorted({r["variant"] for r in results}):
        row = []
        for k in range(4):
            vals = [r["mean"] for r in results
                    if r["variant"] == variant and r["turn_index"] == k and r["mean"] is not None]
            row.append(f"T{k+1}={np.mean(vals):5.2f}(n={len(vals):3d})" if vals else f"T{k+1}=  --      ")
        print(f"  {variant:22s} " + "  ".join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
