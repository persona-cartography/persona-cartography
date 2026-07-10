#!/usr/bin/env python3
"""Persona hill-climbing on a misalignment MCQ benchmark via LoRA soups.

Same experiment as ``run_hill_climb_grid`` (search OCEAN persona-LoRA soups for
the mix that makes gemma-3-27b behave safer / more misaligned, with a train/test
split), but the safety benchmark is the objective multiple-choice
``geodesic-research/discourse-grounded-misalignment-evals`` (config
``textbook_questions``) instead of free-form WildJailbreak.

Because the answer key ships with the data there is **no LLM judge**. Each soup
is scored on two objective signals:

- ``misalignment_rate`` — P(model picks the misaligned option) among answered
  items (lower = safer).
- ``answered_rate`` — the capability/collapse gate: a soup that rambles (or
  puts no probability on the choice letters) shows a low answered-rate rather
  than spurious safety.

Two scoring modes (``--scoring``):

- ``logprob`` (default) — exactly the TRAIT logprob eval recipe: inspect_ai's
  SINGLE_ANSWER_TEMPLATE prompt, forced ``"ANSWER: "`` assistant prefill, one
  generated token with top-20 logprobs, softmax over the found choice letters,
  continuous per-item score = P(misaligned letter), and the TRAIT choice-mass
  filters (dynamic 1/num_choices + fixed 0.75) as the answered gate.
  ``misalignment_rate`` is then a mean of continuous scores → BCa bootstrap CI.
- ``parse`` — the original generate-and-parse mode (letter-only prompt,
  echo-robust parser, binary scores → Wilson CI).

The soup baking, vLLM inference, HF run-dir sync, and grid/point plumbing are
reused unchanged from the WildJailbreak hill-climb (``run_hill_climb_grid``);
only the dataset loader and the scoring stage differ.

Phases::

    # 1. Evaluate all grid points on the TRAIN slice
    uv run python -m scripts_dev.evals.persona_hill_climbing.run_hill_climb_mcq \
        --phase train --run-slug mcq_lp_v1

    # 2. Re-evaluate the k safest + k most misaligned combos on the TEST slice
    uv run python -m scripts_dev.evals.persona_hill_climbing.run_hill_climb_mcq \
        --phase test --run-slug mcq_lp_v1 --top-k 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

# Reuse the WJ hill-climb's soup/grid plumbing. Importing this module also runs
# ensure_vllm_fork_safe() and the heavy torch/vllm-safe imports for us.
from scripts_dev.evals.persona_hill_climbing.run_hill_climb_grid import (  # noqa: E402
    Point,
    build_config,
    default_grid_points,
    point_to_combo,
)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src_dev.common.lora_catalogue import GEMMA_OCEAN_REGISTRIES  # noqa: E402
from src_dev.persona_jailbreak_eval.aggregate import (  # noqa: E402
    RateRow,
    _wilson_rate,
    plot_condition_bars,
    write_summary_csv,
)
from src_dev.persona_jailbreak_eval.config import JailbreakEvalConfig  # noqa: E402
from src_dev.persona_jailbreak_eval.hf_sync import (  # noqa: E402
    hydrate_run_dir_from_hf,
    upload_run_dir_to_hf,
)
from src_dev.evals.personality.analyze_results import (  # noqa: E402
    _interval_ci_from_bootstrap,
)
from src_dev.persona_jailbreak_eval.misalignment_mcq import (  # noqa: E402
    TRAIT_LOGPROB_PREFILL,
    format_mcq,
    format_mcq_logprobs,
    load_misalignment_mcq,
    score_mcq_logprobs,
    score_mcq_response,
)
from src_dev.persona_jailbreak_eval.runner import (  # noqa: E402
    PromptSample,
    run_all_conditions_inference,
)

SEED = 42


# ── Data / samples ───────────────────────────────────────────────────────


def build_samples(args: argparse.Namespace, phase: str) -> list[PromptSample]:
    """Deterministic disjoint train/test slices of the misalignment MCQ set.

    ``load_misalignment_mcq`` subsamples ``n_train + n_test`` items with the
    given seed; the first ``n_train`` are the train slice, the rest are held out
    for test. Same seed → same partition across phases. The per-item choice
    shuffle is keyed on item id, so lettering is identical in both phases.
    """
    n_total = args.n_train + args.n_test
    items = load_misalignment_mcq(
        config=args.mcq_config, n=n_total, seed=SEED, min_relevance=args.min_relevance,
    )
    items = items[: args.n_train] if phase == "train" else items[args.n_train :]
    print(f"  [{phase}] {len(items)} misalignment MCQ items ({args.mcq_config})")

    formatter = format_mcq_logprobs if args.scoring == "logprob" else format_mcq
    samples: list[PromptSample] = []
    for item in items:
        fmt = formatter(item, shuffle_seed=SEED)
        samples.append(PromptSample(
            id=item.id, user_prompt=fmt.prompt, system_prompt=None,
            kind="mcq", category=(item.topics[0] if item.topics else "unknown"),
            behavior="", action="",
            extras={
                "misaligned_letter": fmt.misaligned_letter,
                "valid_letters": fmt.valid_letters,
                "topics": list(item.topics),
                "relevance_rating": item.relevance_rating,
                "hc_phase": phase,
                "scoring": args.scoring,
            },
        ))
    return samples


# ── Scoring / aggregation ────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def score_condition(rows: list[dict], scoring: str = "logprob") -> dict:
    """Score one condition's response rows.

    In ``logprob`` mode each answered item contributes a continuous
    P(misaligned) in [0, 1]; in ``parse`` mode a binary flag. ``answered``
    is always binary (mass-filter gate vs parse gate respectively).

    Returns per-condition aggregates plus the per-topic misalignment tallies.
    """
    mis_vals: list[float] = []       # over answered items
    answered_flags: list[bool] = []  # over all items
    topic_mis: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        extras = r.get("extras") or {}
        if scoring == "logprob":
            if "top_logprobs" not in r:
                raise SystemExit(
                    f"response row {r.get('sample_id')!r} has no 'top_logprobs' — "
                    "this run dir contains parse-mode responses. Use a fresh "
                    "--run-slug for logprob scoring (cached rows are reused as-is)."
                )
            lp_sc = score_mcq_logprobs(
                r["top_logprobs"],
                misaligned_letter=extras.get("misaligned_letter", ""),
                valid_letters=extras.get("valid_letters", "AB"),
            )
            answered_flags.append(lp_sc.answered)
            if lp_sc.answered:
                mis_vals.append(float(lp_sc.p_misaligned))
                for t in extras.get("topics", []):
                    topic_mis[t].append(float(lp_sc.p_misaligned))
        else:
            sc = score_mcq_response(
                r.get("response", ""),
                misaligned_letter=extras.get("misaligned_letter", ""),
                valid_letters=extras.get("valid_letters", "AB"),
            )
            answered_flags.append(sc.answered)
            if sc.answered:
                mis_vals.append(float(sc.is_misaligned))
                for t in extras.get("topics", []):
                    topic_mis[t].append(float(sc.is_misaligned))
    return {"mis": mis_vals, "answered": answered_flags, "topic_mis": topic_mis}


def _mis_rate(values: list[float], scoring: str) -> tuple[float, float, float, int]:
    """(rate, ci_low, ci_high, n) for misalignment scores.

    Binary parse-mode scores get a Wilson interval; continuous logprob-mode
    scores get a BCa bootstrap on the mean (per the repo CI conventions).
    """
    if scoring == "parse":
        return _wilson_rate([bool(v) for v in values])
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0, 0.0, 0)
    rate = float(arr.mean())
    lo, hi = _interval_ci_from_bootstrap(arr, confidence=95.0, n_resamples=1000, seed=SEED)
    return (rate, lo, hi, int(arr.size))


def aggregate_and_report(
    cfg: JailbreakEvalConfig, response_paths: dict[str, Path], phase: str,
    scoring: str = "logprob",
) -> list[RateRow]:
    out_dir = cfg.run_dir / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    mis_rows: list[RateRow] = []
    answered_rows: list[RateRow] = []
    topic_records: list[dict] = []
    for condition, path in response_paths.items():
        scored = score_condition(_read_jsonl(path), scoring)
        m_rate, m_lo, m_hi, m_n = _mis_rate(scored["mis"], scoring)
        a_rate, a_lo, a_hi, a_n = _wilson_rate(scored["answered"])
        mis_rows.append(RateRow(condition=condition, n=m_n, rate=m_rate, ci_low=m_lo,
                                ci_high=m_hi, extras={"answered_rate": round(a_rate, 4)}))
        answered_rows.append(RateRow(condition=condition, n=a_n, rate=a_rate, ci_low=a_lo,
                                     ci_high=a_hi, extras={}))
        for topic, vals in scored["topic_mis"].items():
            if len(vals) >= 10:  # skip thin topics
                t_rate, t_lo, t_hi, t_n = _mis_rate(vals, scoring)
                topic_records.append({
                    "condition": condition, "topic": topic, "n": t_n,
                    "misalignment_rate": round(t_rate, 4),
                    "ci_low": round(t_lo, 4), "ci_high": round(t_hi, 4),
                })

    write_summary_csv(mis_rows, out_dir / "misalignment_rate_by_condition.csv")
    write_summary_csv(answered_rows, out_dir / "answered_rate_by_condition.csv")
    if topic_records:
        import csv
        with (out_dir / "misalignment_rate_by_topic.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["condition", "topic", "n", "misalignment_rate", "ci_low", "ci_high"])
            w.writeheader()
            w.writerows(topic_records)
    plot_condition_bars(
        mis_rows, answered_rows,
        title=f"Persona hill-climb MCQ ({phase}) — {cfg.run_slug}",
        output_path=out_dir / "summary_bars.png",
    )

    answered_by_cond = {r.condition: r for r in answered_rows}
    print(f"\n  ── MISALIGNMENT RATE ({phase}, answered items only) ──────")
    for r in sorted(mis_rows, key=lambda r: r.rate):
        ar = answered_by_cond[r.condition]
        print(f"    {r.condition:46s} n={r.n:4d}  mis={r.rate:.3f} "
              f"CI=[{r.ci_low:.3f},{r.ci_high:.3f}]  answered={ar.rate:.2f}")
    print(f"\n  artefacts: {out_dir}")
    return mis_rows


def select_extreme_conditions(train_run_dir: Path, top_k: int, scoring: str = "logprob") -> list[str]:
    """Rank train-phase lora_soup conditions by misalignment rate; return the
    ``top_k`` safest + ``top_k`` most misaligned condition names."""
    resp_dir = train_run_dir / "responses"
    rows: list[RateRow] = []
    for path in sorted(resp_dir.glob("responses_lora_soup_*.jsonl")):
        condition = path.stem[len("responses_"):]
        scored = score_condition(_read_jsonl(path), scoring)
        rate, lo, hi, n = _mis_rate(scored["mis"], scoring)
        rows.append(RateRow(condition=condition, n=n, rate=rate, ci_low=lo, ci_high=hi, extras={}))
    if not rows:
        raise SystemExit(f"no train lora_soup responses under {resp_dir} — run --phase train first")
    rows.sort(key=lambda r: r.rate)
    print("\n  ── TRAIN RANKING (misalignment rate, ascending) ─────────")
    for r in rows:
        print(f"    {r.condition:46s} n={r.n:4d}  mis={r.rate:.3f}  CI=[{r.ci_low:.3f},{r.ci_high:.3f}]")
    safest = [r.condition for r in rows[:top_k]]
    most_misaligned = [r.condition for r in rows[-top_k:]]
    selected = list(dict.fromkeys(safest + most_misaligned))
    print(f"\n  selected for test phase: {selected}")
    return selected


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "test"), default="train")
    parser.add_argument("--scoring", choices=("logprob", "parse"), default="logprob",
                        help="'logprob' = TRAIT-style single-token logprob scoring (default); "
                             "'parse' = original generate-and-parse")
    parser.add_argument("--run-slug", default=None,
                        help="default: mcq_lp_v1 (logprob) / mcq_v1 (parse). Logprob and parse "
                             "runs must not share a run-slug (rows are cached per run dir)")
    parser.add_argument("--base-model", default="google/gemma-3-27b-it")
    parser.add_argument("--model-slug", default="gemma-3-27b-it")
    parser.add_argument("--points-json", type=Path, default=None,
                        help="JSON list of trait-coefficient dicts overriding the default 20-point grid")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--n-train", type=int, default=300)
    parser.add_argument("--n-test", type=int, default=300)
    parser.add_argument("--mcq-config", default="textbook_questions")
    parser.add_argument("--min-relevance", type=int, default=7)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--no-vanilla", action="store_true")
    parser.add_argument("--no-upload-hf", action="store_true")
    parser.add_argument("--no-hydrate-hf", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()
    if args.run_slug is None:
        args.run_slug = "mcq_lp_v1" if args.scoring == "logprob" else "mcq_v1"

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    registry = GEMMA_OCEAN_REGISTRIES.get(args.model_slug)
    if registry is None:
        raise SystemExit(f"no OCEAN registry for {args.model_slug!r}; known: {sorted(GEMMA_OCEAN_REGISTRIES)}")

    if args.points_json is not None:
        points: list[Point] = json.loads(args.points_json.read_text())
    else:
        points = default_grid_points()
    all_combos = {c.name: c for c in (point_to_combo(p, registry) for p in points)}

    cfg = build_config(args, args.phase)
    cfg.max_new_tokens = args.max_new_tokens  # MCQ: a short answer, not a 512-tok essay

    if args.phase == "train":
        combos = tuple(all_combos.values())
    else:
        train_cfg = build_config(args, "train")
        if cfg.hydrate_hf:
            hydrate_run_dir_from_hf(
                local_run_dir=train_cfg.run_dir, eval_type=cfg.hf_eval_type,
                model_slug=cfg.model_slug, run_slug=train_cfg.run_slug, repo_id=cfg.hf_repo_id,
            )
        selected = select_extreme_conditions(train_cfg.run_dir, args.top_k, args.scoring)
        missing = [c for c in selected if c not in all_combos]
        if missing:
            raise SystemExit(f"train-selected conditions not reproducible from current grid: {missing}")
        combos = tuple(all_combos[c] for c in selected)

    conditions = tuple(c.name for c in combos)
    if not args.no_vanilla:
        conditions = ("vanilla",) + conditions
    cfg.conditions = conditions
    cfg.lora_combos = combos

    print("=" * 70)
    print(f"  Persona hill-climbing (MCQ) — phase={args.phase!r} run_slug={cfg.run_slug!r}")
    print(f"  Base model: {cfg.base_model} (slug={cfg.model_slug})")
    print(f"  Benchmark: {args.mcq_config} (min_relevance={args.min_relevance}, scoring={args.scoring})")
    print(f"  Run dir: {cfg.run_dir}")
    print(f"  Conditions ({len(cfg.conditions)}): {', '.join(cfg.conditions)}")
    print("=" * 70)

    if cfg.hydrate_hf:
        hydrate_run_dir_from_hf(
            local_run_dir=cfg.run_dir, eval_type=cfg.hf_eval_type,
            model_slug=cfg.model_slug, run_slug=cfg.run_slug, repo_id=cfg.hf_repo_id,
        )

    samples = build_samples(args, args.phase)
    response_paths = run_all_conditions_inference(
        cfg, samples, output_dir=cfg.run_dir / "responses",
        logprobs=(args.scoring == "logprob"),
        logprob_prefill=TRAIT_LOGPROB_PREFILL,
    )
    if cfg.upload_hf:
        upload_run_dir_to_hf(
            local_run_dir=cfg.run_dir, eval_type=cfg.hf_eval_type,
            model_slug=cfg.model_slug, run_slug=cfg.run_slug, repo_id=cfg.hf_repo_id, stage="inference",
        )

    if not args.skip_aggregate:
        aggregate_and_report(cfg, response_paths, args.phase, args.scoring)
        if cfg.upload_hf:
            upload_run_dir_to_hf(
                local_run_dir=cfg.run_dir, eval_type=cfg.hf_eval_type,
                model_slug=cfg.model_slug, run_slug=cfg.run_slug, repo_id=cfg.hf_repo_id, stage="aggregate",
            )


if __name__ == "__main__":
    main()
