"""Evaluate rollouts from JSONL files with persona metrics.

Walks a directory tree for ``rollouts/rollouts.jsonl`` files, runs specified
evaluators on messages, and writes ``evals/rollouts_evaluated.jsonl``
alongside each input file.

Evaluators whose scores already exist in ``rollouts_evaluated.jsonl`` are
skipped — only new evaluators run and their scores are merged into the
existing file. To force re-run specific evaluators, list them in
``overwrite_evaluations``.

Usage::

    from src.persona_metrics.eval_rollouts import (
        RolloutEvalConfig, evaluate_rollouts,
    )

    config = RolloutEvalConfig(
        root_dir=Path("scratch/monorepo/.../rollout_sweep_lora_scale"),
        evaluations=["count_t", "coherence"],
    )
    result = evaluate_rollouts(config)
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.persona_metrics.base import PersonaMetric, PersonaMetricContext
from src.persona_metrics.config import JudgeLLMConfig, PersonaMetricSpec
from src.persona_metrics.conversation_eval import MessageSelector
from src.persona_metrics.registry import get_persona_metric

# ── Config and result types ──────────────────────────────────────────────────


@dataclass
class RolloutEvalConfig:
    """Configuration for evaluating rollout JSONL files.

    Args:
        root_dir: Directory tree to search for rollouts/rollouts.jsonl files.
        evaluations: Evaluators to run on each message.
        message_selector: Filter which messages to evaluate.
        judge: LLM judge configuration for evaluators that need one.
        overwrite_evaluations: Evaluator names to force re-run even if their
            scores already exist. By default (empty list), evaluators whose
            scores are already present in rollouts_evaluated.jsonl are skipped.
        eval_aliases: Rename evaluator keys in the output scores
            (e.g. ``{"count_t": "count_t_v2"}``).
    """

    root_dir: Path
    evaluations: list[str | PersonaMetricSpec]
    message_selector: MessageSelector | None = None
    judge: JudgeLLMConfig | None = None
    overwrite_evaluations: list[str] = field(default_factory=list)
    eval_aliases: dict[str, str] = field(default_factory=dict)
    exclude_path_patterns: list[str] = field(default_factory=list)
    """Substrings to exclude from rollout file paths. Any rollout whose path
    contains one of these strings is skipped. E.g. ``["5turn_", "15turn_"]``
    to evaluate only single-answer (1-turn) rollouts."""


@dataclass
class RolloutEvalResult:
    """Result from evaluating rollouts."""

    num_files_processed: int = 0
    num_messages_evaluated: int = 0
    evaluations_run: list[str] = field(default_factory=list)
    eval_info_paths: list[Path] = field(default_factory=list)
    evals_dirs: list[Path] = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _git_commit_hash() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return output.strip() or None


def _read_calling_script() -> str | None:
    try:
        return Path(sys.argv[0]).resolve().read_text()
    except (OSError, ValueError):
        return None


def _nest_scores(
    flat_scores: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Convert flat metric scores to nested-by-evaluator format.

    Example::

        {"count_t.count": 82, "count_t.density": 13.29}
        -> {"count_t": {"count": 82, "density": 13.29}}
    """
    nested: dict[str, dict[str, Any]] = {}
    for key, value in flat_scores.items():
        if "." in key:
            evaluator, subkey = key.split(".", 1)
            nested.setdefault(evaluator, {})[subkey] = value
        else:
            nested.setdefault(key, {})["score"] = value
    return nested


def _create_metrics(
    evaluations: list[str | PersonaMetricSpec],
    judge: JudgeLLMConfig | None = None,
) -> list[PersonaMetric]:
    """Create persona metric instances from evaluation specs."""
    judge_config = judge or JudgeLLMConfig()
    metrics: list[PersonaMetric] = []
    for spec in evaluations:
        if isinstance(spec, str):
            metrics.append(get_persona_metric(spec, judge_config=judge_config))
        else:
            kwargs: dict = {"judge_config": judge_config}
            kwargs.update(spec.params)
            metrics.append(get_persona_metric(spec.name, **kwargs))
    return metrics


def _find_rollout_files(root_dir: Path) -> list[Path]:
    """Find all rollouts/rollouts.jsonl files under root_dir."""
    return sorted(root_dir.rglob("rollouts/rollouts.jsonl"))


def _load_rollouts(
    path: Path,
) -> list[dict[str, Any]]:
    """Load entries from a rollouts.jsonl file."""
    entries = []
    for line in path.read_text().strip().split("\n"):
        if line.strip():
            entries.append(json.loads(line))
    return entries


def _load_existing_evaluated(
    evals_dir: Path,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Load existing rollouts_evaluated.jsonl and return entries + eval names.

    Returns:
        (entries, completed_eval_names) where entries have nested scores
        and completed_eval_names is the set of top-level score keys.
    """
    path = evals_dir / "rollouts_evaluated.jsonl"
    if not path.exists():
        return [], set()

    entries = []
    eval_names: set[str] = set()
    for line in path.read_text().strip().split("\n"):
        if not line.strip():
            continue
        entry = json.loads(line)
        entries.append(entry)
        # Collect evaluator names from nested scores.
        for msgs in entry.get("messages", {}).values():
            for msg in msgs:
                for key in msg.get("scores", {}):
                    eval_names.add(key)
    return entries, eval_names


def _matches_message(
    msg: dict[str, Any],
    selector: MessageSelector | None,
) -> bool:
    """Check if a raw message dict matches the selector."""
    if selector is None:
        return True
    if selector.exclude_seed:
        source = msg.get("source", "")
        if source not in {
            "rollout_assistant",
            "rollout_user_simulator",
        }:
            return False
    if selector.roles and msg.get("role") not in selector.roles:
        return False
    if selector.turn_index_range is not None:
        turn_idx = msg.get("turn_index")
        if turn_idx is None:
            return False
        lo, hi = selector.turn_index_range
        # Negative indices not resolved here (would need
        # max_turn_index per conversation) — keep simple for now.
        if lo >= 0 and hi >= 0:
            if not (lo <= turn_idx <= hi):
                return False
    return True


def _extract_eval_items(
    entries: list[dict[str, Any]],
    selector: MessageSelector | None,
) -> list[dict[str, Any]]:
    """Extract evaluatable messages from rollout entries.

    Returns list of dicts with content, preceding_content, and
    location info for score merging.
    """
    items = []
    for entry_idx, entry in enumerate(entries):
        for rollout_idx, msgs in entry.get("messages", {}).items():
            preceding = ""
            for msg_idx, msg in enumerate(msgs):
                if _matches_message(msg, selector):
                    items.append(
                        {
                            "content": msg.get("content", ""),
                            "preceding_content": preceding,
                            "entry_idx": entry_idx,
                            "rollout_idx": rollout_idx,
                            "msg_idx": msg_idx,
                            "role": msg.get("role", ""),
                        }
                    )
                preceding = msg.get("content", "")
    return items


async def _run_metrics_async(
    metrics: list[PersonaMetric],
    eval_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run all metrics on eval items and return nested scores per item."""
    responses = [item["content"] for item in eval_items]
    questions = [item["preceding_content"] for item in eval_items]
    contexts = [
        PersonaMetricContext(
            response=item["content"],
            question=item["preceding_content"],
            record=item,
            metadata={},
        )
        for item in eval_items
    ]

    async def _run_one(
        metric: PersonaMetric,
    ) -> list[dict[str, float | int | str]]:
        return await metric.evaluate_batch_async(
            responses, questions, contexts=contexts
        )

    all_results = await asyncio.gather(*[_run_one(m) for m in metrics])

    per_item_scores: list[dict[str, Any]] = []
    for i in range(len(eval_items)):
        flat_scores: dict[str, Any] = {}
        for metric_batch in all_results:
            flat_scores.update(metric_batch[i])
        per_item_scores.append(_nest_scores(flat_scores))
    return per_item_scores


def _merge_scores_into_entries(
    entries: list[dict[str, Any]],
    eval_items: list[dict[str, Any]],
    new_scores: list[dict[str, Any]],
    eval_aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Merge new nested scores into rollout entries.

    Modifies entries in place and returns them.
    """
    for item, scores in zip(eval_items, new_scores):
        msg = entries[item["entry_idx"]]["messages"][item["rollout_idx"]][
            item["msg_idx"]
        ]
        existing = msg.get("scores", {})
        # Apply aliases (e.g. rename "count_t" -> "count_t2").
        if eval_aliases:
            renamed: dict[str, Any] = {}
            for k, v in scores.items():
                renamed[eval_aliases.get(k, k)] = v
            scores = renamed
        existing.update(scores)
        msg["scores"] = existing
    return entries


def _write_eval_info(
    evals_dir: Path,
    evaluations: list[str | PersonaMetricSpec],
    elapsed: float,
    num_messages: int,
) -> Path:
    """Write eval_info.json and script copy into a timestamped eval_runs/ subdir."""
    import shutil

    eval_names = [e if isinstance(e, str) else e.name for e in evaluations]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    run_name = f"{'__'.join(eval_names)}__{ts}"
    run_dir = evals_dir / "eval_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "status": "ok",
        "datetime": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": _git_commit_hash(),
        "elapsed_seconds": round(elapsed, 2),
        "num_messages_evaluated": num_messages,
        "evaluators": [e if isinstance(e, str) else str(e) for e in evaluations],
    }
    path = run_dir / "eval_info.json"
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    script_path = Path(sys.argv[0]).resolve()
    if script_path.exists():
        shutil.copy2(script_path, run_dir / script_path.name)
    return path


# ── Main entry point ─────────────────────────────────────────────────────────


def evaluate_rollouts(
    config: RolloutEvalConfig,
) -> RolloutEvalResult:
    """Walk root_dir for rollouts, run evals, write results.

    For each ``rollouts/rollouts.jsonl`` found under ``root_dir``:

    1. Load messages from the JSONL.
    2. If ``incremental=True``, check which evals are already done.
    3. Run missing evaluators on selected messages.
    4. Merge new scores into ``evals/rollouts_evaluated.jsonl``.
    5. Write ``evals/eval_info.json`` with metadata.

    Args:
        config: Evaluation configuration.

    Returns:
        RolloutEvalResult with summary statistics.
    """
    import time

    t0 = time.perf_counter()
    rollout_files = _find_rollout_files(config.root_dir)
    if config.exclude_path_patterns:
        rollout_files = [
            f for f in rollout_files
            if not any(pat in str(f) for pat in config.exclude_path_patterns)
        ]

    if not rollout_files:
        print(f"No rollouts/rollouts.jsonl files found under {config.root_dir}")
        return RolloutEvalResult()

    print(f"Found {len(rollout_files)} rollout file(s) to evaluate")

    # Determine which evals to run.
    eval_names = [e if isinstance(e, str) else e.name for e in config.evaluations]

    result = RolloutEvalResult()

    for rollout_path in rollout_files:
        rollouts_dir = rollout_path.parent  # rollouts/
        cell_dir = rollouts_dir.parent  # variant/condition/
        evals_dir = cell_dir / "evals"

        print(f"\n  Processing: {cell_dir.name}")

        # Load rollout entries.
        entries = _load_rollouts(rollout_path)
        if not entries:
            print(f"    Skipping (no entries)")
            continue

        # Check what's already evaluated.
        overwrite_names = set(config.overwrite_evaluations)
        existing_entries, done_evals = _load_existing_evaluated(evals_dir)

        # Merge existing scores into loaded entries (preserves old scores).
        if existing_entries:
            for orig, existing in zip(entries, existing_entries):
                for ri, msgs in existing.get("messages", {}).items():
                    if ri in orig.get("messages", {}):
                        for mi, msg in enumerate(msgs):
                            if mi < len(orig["messages"][ri]):
                                orig["messages"][ri][mi].setdefault(
                                    "scores", {}
                                ).update(msg.get("scores", {}))

        # Skip evaluators already done, unless they're in
        # overwrite_evaluations or eval_aliases.
        evals_to_run = [
            e
            for e in config.evaluations
            if (e if isinstance(e, str) else e.name) not in done_evals
            or (e if isinstance(e, str) else e.name) in overwrite_names
            or (e if isinstance(e, str) else e.name) in config.eval_aliases
        ]

        if not evals_to_run:
            print(f"    All evals already done: {sorted(done_evals)}")
            continue

        # Extract messages to evaluate.
        eval_items = _extract_eval_items(entries, config.message_selector)
        if not eval_items:
            print(f"    No messages matched selector")
            continue

        # Create and run metrics.
        metrics = _create_metrics(evals_to_run, config.judge)
        new_scores = asyncio.run(_run_metrics_async(metrics, eval_items))

        # Merge scores into entries.
        _merge_scores_into_entries(
            entries,
            eval_items,
            new_scores,
            eval_aliases=config.eval_aliases,
        )

        # Write output.
        evals_dir.mkdir(parents=True, exist_ok=True)
        out_path = evals_dir / "rollouts_evaluated.jsonl"
        out_path.write_text(
            "\n".join(json.dumps(e, default=str) for e in entries) + "\n"
        )

        eval_names_run = [m.name for m in metrics]
        elapsed = time.perf_counter() - t0
        info_path = _write_eval_info(evals_dir, evals_to_run, elapsed, len(eval_items))

        print(f"    Evaluated {len(eval_items)} messages with {eval_names_run}")
        print(f"    Wrote {out_path}")

        result.num_files_processed += 1
        result.num_messages_evaluated += len(eval_items)
        result.evaluations_run.extend(eval_names_run)
        result.eval_info_paths.append(info_path)
        result.evals_dirs.append(evals_dir)

    result.evaluations_run = list(dict.fromkeys(result.evaluations_run))
    elapsed = time.perf_counter() - t0
    print(
        f"\nDone: {result.num_files_processed} file(s), "
        f"{result.num_messages_evaluated} message(s) in "
        f"{elapsed:.1f}s"
    )
    return result
