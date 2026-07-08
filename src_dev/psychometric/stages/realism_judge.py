"""Stage 2b — diagnostic realism judge over completed rollouts.

Scores each completed rollout for ``unrealism`` using the Bloom-style
``UnrealismJudge`` in ``src_dev.persona_metrics.metrics.realism_judges``.
Scores are **never** used to filter rollouts downstream — spiraling /
repetitive / confused conversations are themselves persona signal for the
FA. This stage exists purely to inspect how realistic the rollouts are.

Writes ``{rollout_dir}/realism_judge/per_rollout_scores.jsonl`` (one row per
completed sample) plus prints overall + archetype-grouped summary stats.
Idempotent: skipped if the JSONL already covers every completed sample_id
with a non-error score, hydrated from HF if present there, otherwise
generated fresh and uploaded to HF.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from src_dev.datasets import (
    find_consecutive_assistant_turn_sample_ids,
    load_samples,
    materialize_canonical_samples,
)
from src_dev.psychometric.config import (
    HF_RUNS_PREFIX,
    RealismJudgeStageConfig,
    RealismJudgeStageResult,
)
from src_dev.psychometric.realism_judge import summarize_realism_scores
from src_dev.unsupervised_runs.io import hydrate_dataset_subtree
from src_dev.utils.hf_hub import (
    check_exists_in_dataset_repo,
    login_from_env,
    upload_folder_to_dataset_repo,
)

logger = logging.getLogger(__name__)


def run_stage_realism_judge(
    cfg: RealismJudgeStageConfig,
    *,
    num_conversation_turns: int,
) -> RealismJudgeStageResult:
    """Run the unrealism judge over every completed rollout.

    Args:
        cfg: Judge config (model / provider / concurrency / message cap).
        num_conversation_turns: Completeness threshold — samples with fewer
            assistant turns are excluded. Taken from the rollout preset,
            not from the judge config.
    """
    from src_dev.persona_metrics.config import JudgeLLMConfig
    from src_dev.persona_metrics.metrics.realism_judges import (
        UnrealismJudge,
        render_transcript_for_judge,
    )

    rollout_dir = cfg.ctx.rollout_dir
    output_dir = rollout_dir / "realism_judge"
    output_path = output_dir / "per_rollout_scores.jsonl"
    hf_path = f"{HF_RUNS_PREFIX}/{cfg.ctx.rollout_run_id}/realism_judge"
    hf_repo_id = cfg.ctx.hf_repo_id

    # ── Load completed samples (same filter as the questionnaire stage) ──
    materialize_canonical_samples(rollout_dir)
    samples = load_samples(rollout_dir)
    completed_samples = [
        s for s in samples
        if sum(1 for m in s.messages if m.role == "assistant") >= num_conversation_turns
    ]
    bad_sample_ids = find_consecutive_assistant_turn_sample_ids(rollout_dir)
    if bad_sample_ids:
        completed_samples = [
            s for s in completed_samples if s.sample_id not in bad_sample_ids
        ]
    if not completed_samples:
        print("[Realism] No completed rollouts found — skipping.")
        return RealismJudgeStageResult()

    expected_ids = {s.sample_id for s in completed_samples}

    def _load_existing() -> list[dict] | None:
        if not output_path.exists():
            return None
        rows = [
            json.loads(line)
            for line in output_path.open("r", encoding="utf-8")
            if line.strip()
        ]
        # Rows with the -1 error sentinel are not considered cached —
        # they'll be retried on the next run.
        good_ids = {
            r["sample_id"]
            for r in rows
            if r.get("unrealism_score", -1) >= 0
        }
        if expected_ids.issubset(good_ids):
            return rows
        missing = len(expected_ids - good_ids)
        print(
            f"[Realism] Local JSONL covers {len(good_ids & expected_ids)}/"
            f"{len(expected_ids)} samples with non-error scores "
            f"({missing} to retry) — regenerating."
        )
        return None

    # ── Local cache ──
    cached = _load_existing()
    if cached is not None:
        print(f"[Realism] Results already exist locally: {output_path}")
        summary = summarize_realism_scores(cached)
        return RealismJudgeStageResult(report_path=output_path, summary=summary)

    # ── HF cache ──
    try:
        login_from_env()
    except RuntimeError:
        logger.warning("HF_TOKEN not set — HF caching disabled.")
    if check_exists_in_dataset_repo(repo_id=hf_repo_id, path_in_repo=hf_path):
        print(f"[Realism] Hydrating realism judge results from HF: {hf_path}")
        hydrate_dataset_subtree(
            repo_id=hf_repo_id,
            path_in_repo=hf_path,
            local_dir=output_dir,
        )
        cached = _load_existing()
        if cached is not None:
            summary = summarize_realism_scores(cached)
            return RealismJudgeStageResult(report_path=output_path, summary=summary)
        print("[Realism] HF hydration incomplete, regenerating...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Build per-sample transcript strings + side metadata ──
    transcripts: list[str] = []
    side_meta: list[dict] = []
    for s in completed_samples:
        msgs = [{"role": m.role, "content": m.content} for m in s.messages]
        rendered = render_transcript_for_judge(
            msgs, max_message_chars=cfg.max_message_chars
        )
        transcripts.append(rendered)
        side_meta.append({
            "sample_id": s.sample_id,
            "input_group_id": s.input_group_id,
            "response_index": s.response_index,
        })

    # Scenario / archetype context is recorded in the synthetic-seed JSONL
    # (scenario mode only). Outside scenario mode both lookups return None.
    scenario_by_id: dict[str, str | None] = {}
    archetype_by_id: dict[str, str | None] = {}
    seed_path = rollout_dir / "_synthetic_scenario_seeds.jsonl"
    seed_by_row_idx: dict[int, dict] = {}
    if seed_path.exists():
        with seed_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if "id" in row:
                    seed_by_row_idx[int(row["id"])] = row
    for s in completed_samples:
        row_idx = s.source_info.get("row_index") if s.source_info else None
        seed_row = seed_by_row_idx.get(int(row_idx)) if row_idx is not None else None
        scenario_by_id[s.sample_id] = (seed_row or {}).get("scenario_id")
        archetype_by_id[s.sample_id] = (seed_row or {}).get("archetype")

    judge_cfg = JudgeLLMConfig(
        provider=cfg.provider,
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        max_concurrent=cfg.max_concurrent,
    )
    unrealism_judge = UnrealismJudge(judge_config=judge_cfg)

    placeholder = "[full multi-turn transcript below]"
    questions = [placeholder] * len(transcripts)

    print(
        f"[Realism] Judging {len(transcripts)} rollouts with "
        f"{cfg.model} (concurrency={cfg.max_concurrent})"
    )

    async def _run() -> list[dict]:
        return await unrealism_judge.evaluate_batch_async(transcripts, questions)

    unrealism_results = asyncio.run(_run())

    # ── Write per-rollout JSONL ──
    rows: list[dict] = []
    with output_path.open("w", encoding="utf-8") as f:
        for meta, u_res in zip(side_meta, unrealism_results):
            sid = meta["sample_id"]
            row = {
                "sample_id": sid,
                "input_group_id": meta["input_group_id"],
                "response_index": meta["response_index"],
                "scenario_id": scenario_by_id.get(sid),
                "archetype": archetype_by_id.get(sid),
                "unrealism_score": u_res.get("unrealism.score"),
                "unrealism_reasoning": u_res.get("unrealism.reasoning", ""),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
    print(f"[Realism] Wrote {len(rows)} rows to {output_path}")

    # ── Upload to HF ──
    try:
        upload_folder_to_dataset_repo(
            local_dir=output_dir,
            repo_id=hf_repo_id,
            path_in_repo=hf_path,
            commit_message=f"Realism judge: {cfg.ctx.rollout_run_id}",
            ignore_patterns=[],
        )
        print(f"[Realism] Uploaded to HF: {hf_path}")
    except Exception as e:
        logger.warning("Failed to upload realism judge to HF: %s", e)

    summary = summarize_realism_scores(rows)
    return RealismJudgeStageResult(report_path=output_path, summary=summary)
