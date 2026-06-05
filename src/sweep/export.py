"""Export, provenance, and HuggingFace upload helpers for rollout sweeps.

Turns a completed run directory into the on-disk artifacts a sweep publishes —
``rollouts.jsonl`` / ``rollouts_evaluated.jsonl``, the ``*_info.json`` provenance
markers, the experiment-metadata + script copy — and pushes the relevant
subtrees to the HF monorepo. Everything here is lower-level than the
orchestration engine: it depends only on :mod:`src.sweep.config`, never the
other way around.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.datasets import get_run_paths, load_samples, materialize_canonical_samples
from src.datasets.io import read_jsonl_tolerant
from src.rollout_generation.prompts import get_user_simulator_instruction
from src.utils.hf_hub import login_from_env, upload_folder_to_dataset_repo

if TYPE_CHECKING:
    from src.evals.judges.config import PersonaMetricSpec
    from src.evals.judges.conversation_eval import ConversationMetricsResult

    from .config import ExperimentConfig, OutputPathConfig, Phase

# ── Helpers ───────────────────────────────────────────────────────────────────


def _prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


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


def _nest_scores(flat_scores: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert flat metric scores to nested-by-evaluator format.

    Example::

        {"neuroticism_v2.score": 2.0, "coherence_v2.score": 8.5}
        → {"neuroticism_v2": {"score": 2.0}, "coherence_v2": {"score": 8.5}}
    """
    nested: dict[str, dict[str, Any]] = {}
    for key, value in flat_scores.items():
        if "." in key:
            evaluator, subkey = key.split(".", 1)
            nested.setdefault(evaluator, {})[subkey] = value
        else:
            nested.setdefault(key, {})["score"] = value
    return nested


# ── Provenance / info markers ─────────────────────────────────────────────────


def _write_rollout_info(run_dir: Path, elapsed: float) -> Path:
    """Write rollouts/rollout_info.json with metadata about the generation run."""
    rollouts_dir = run_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "status": "ok",
        "datetime": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": _git_commit_hash(),
        "elapsed_seconds": round(elapsed, 2),
    }
    path = rollouts_dir / "rollout_info.json"
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return path


def _write_eval_info(
    run_dir: Path,
    evaluations: list[str | PersonaMetricSpec],
    elapsed: float,
) -> Path:
    """Write eval_info.json and script copy into a timestamped eval_runs/ subdir."""
    evals_dir = run_dir / "evals"
    eval_names = [
        e if isinstance(e, str) else e.name if hasattr(e, "name") else str(e)
        for e in evaluations
    ]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    run_name = f"{'__'.join(eval_names)}__{ts}"
    eval_run_dir = evals_dir / "eval_runs" / run_name
    eval_run_dir.mkdir(parents=True, exist_ok=True)

    info = {
        "status": "ok",
        "datetime": datetime.now(timezone.utc).isoformat(),
        "git_commit_hash": _git_commit_hash(),
        "elapsed_seconds": round(elapsed, 2),
        "evaluators": [
            e
            if isinstance(e, str)
            else dataclasses.asdict(e)
            if dataclasses.is_dataclass(e)
            else str(e)
            for e in evaluations
        ],
    }
    path = eval_run_dir / "eval_info.json"
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    script_path = Path(sys.argv[0]).resolve()
    if script_path.exists():
        shutil.copy2(script_path, eval_run_dir / script_path.name)
    return path


def _write_run_info(
    run_dir: Path,
    variant: str,
    condition: str,
    status: str,
    aggregates: dict[str, Any] | None,
    error: str | None,
    elapsed: float | None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_info.json"
    path.write_text(
        json.dumps(
            {
                "variant": variant,
                "condition": condition,
                "status": status,
                "aggregates": aggregates,
                "error": error,
                "elapsed_seconds": round(elapsed, 2) if elapsed is not None else None,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return path


# ── Export ────────────────────────────────────────────────────────────────────


def _message_list_for_sample(
    sample: Any,
    scores_by_msg: dict[str, dict[str, Any]] | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Extract seed_input and list of message dicts from a sample."""
    seed_input = None
    messages = []
    for msg in sample.messages:
        meta = msg.message_metadata or {}
        source = meta.get("source_stage", "")
        if source == "seed":
            seed_input = msg.content
            continue
        msg_entry: dict[str, Any] = {
            "role": msg.role,
            "content": msg.content,
            "turn_index": meta.get("turn_index"),
            "system_prompt_hash": meta.get("system_prompt_hash")
            or meta.get("active_system_prompt"),
            "source": source,
        }
        if scores_by_msg:
            msg_scores = scores_by_msg.get(msg.message_id)
            if msg_scores:
                msg_entry["scores"] = msg_scores
        messages.append(msg_entry)
    return seed_input, messages


def _group_samples_by_seed(samples: list[Any]) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = {}
    for s in samples:
        seed_id = s.input_group_id or s.sample_id
        groups.setdefault(seed_id, []).append(s)
    for group in groups.values():
        group.sort(key=lambda s: s.response_index)
    return groups


def _collect_failure_events(run_dir: Path, seed_ids: set[str]) -> list[dict[str, Any]]:
    """Read stage_events.jsonl and return failure/attempt events for given seeds.

    Stage events use per-rollout sample IDs (e.g. ``sample_abc123_r0_def456``)
    while rollout entries group by seed ID (``sample_abc123``).  We match by
    checking whether the event's sample_id starts with any of the seed IDs.
    """
    events_path = get_run_paths(run_dir)["stage_events"]
    rows, _ = read_jsonl_tolerant(events_path)
    # Build prefix tuple for fast startswith matching.
    prefixes = tuple(seed_ids)
    results = []
    for row in rows:
        sid = row.get("sample_id", "")
        if not any(sid.startswith(p) for p in prefixes):
            continue
        evt = row.get("event_type", "")
        if evt in {"terminal_failure", "invalid_state"}:
            results.append(row)
        elif evt in {"assistant_attempt", "user_attempt"}:
            payload = row.get("payload", {})
            if payload.get("status") == "failed":
                results.append(row)
    return results


def export_rollouts(run_dir: Path) -> Path:
    """Write rollouts.jsonl: one line per seed, messages keyed by rollout index."""
    materialize_canonical_samples(run_dir)
    samples = load_samples(run_dir)
    groups = _group_samples_by_seed(samples)

    entries = []
    for seed_id, group in groups.items():
        seed_input = None
        messages_by_rollout: dict[str, list[dict[str, Any]]] = {}
        for sample in group:
            si, msg_list = _message_list_for_sample(sample)
            if seed_input is None:
                seed_input = si
            messages_by_rollout[str(sample.response_index)] = msg_list
        entries.append(
            {
                "seed_id": seed_id,
                "seed_input": seed_input,
                "messages": messages_by_rollout,
            }
        )

    # Validate: flag any rollout that produced zero messages.
    empty = [
        (e["seed_id"], idx)
        for e in entries
        for idx, msgs in e["messages"].items()
        if len(msgs) == 0
    ]
    if empty:
        # Collect failure details from stage_events.jsonl for the affected samples.
        empty_sample_ids = {seed_id for seed_id, _ in empty}
        failure_details = _collect_failure_events(run_dir, empty_sample_ids)

        # Write errors log so the user can inspect after the fact.
        errors_path = run_dir / "errors.jsonl"
        errors_path.write_text(
            "\n".join(json.dumps(e, default=str) for e in failure_details) + "\n"
        )

        summary_lines = [
            f"{len(empty)} rollout(s) have empty messages (no assistant output). "
            f"Error details written to {errors_path}"
        ]
        # Show a few failure reasons inline.
        reasons = []
        for d in failure_details[:10]:
            payload = d.get("payload", {})
            why = payload.get("reason") or payload.get("error", "?")
            reasons.append(f"  {d['sample_id']}: {d['event_type']} — {why}")
        if reasons:
            summary_lines.append("First failures:")
            summary_lines.extend(reasons)

        raise RuntimeError("\n".join(summary_lines))

    rollouts_dir = run_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    out_path = rollouts_dir / "rollouts.jsonl"
    out_path.write_text("\n".join(json.dumps(e, default=str) for e in entries) + "\n")
    n_prompts = len(entries)
    n_responses = sum(len(e["messages"]) for e in entries)
    print(f"\n  Wrote {n_prompts} prompt(s) / {n_responses} response(s) to {out_path}")
    return out_path


def export_evaluated_rollouts(
    run_dir: Path,
    eval_result: ConversationMetricsResult,
) -> Path:
    """Write rollouts_evaluated.jsonl with scores merged by message_id."""
    materialize_canonical_samples(run_dir)
    samples = load_samples(run_dir)

    scores_by_msg: dict[str, dict[str, Any]] = {}
    for item in eval_result.per_message_scores:
        flat = item.get("scores", {})
        scores_by_msg[item["message_id"]] = _nest_scores(flat)

    groups = _group_samples_by_seed(samples)
    entries = []
    for seed_id, group in groups.items():
        seed_input = None
        messages_by_rollout: dict[str, list[dict[str, Any]]] = {}
        for sample in group:
            si, msg_list = _message_list_for_sample(sample, scores_by_msg=scores_by_msg)
            if seed_input is None:
                seed_input = si
            messages_by_rollout[str(sample.response_index)] = msg_list
        entries.append(
            {
                "seed_id": seed_id,
                "seed_input": seed_input,
                "messages": messages_by_rollout,
            }
        )

    evals_dir = run_dir / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    out_path = evals_dir / "rollouts_evaluated.jsonl"
    out_path.write_text("\n".join(json.dumps(e, default=str) for e in entries) + "\n")
    print(f"  Wrote {len(entries)} evaluated rollouts to {out_path}")
    return out_path


# ── Provenance ────────────────────────────────────────────────────────────────


def _build_system_prompts_map(
    config: ExperimentConfig,
    phases: list[Phase],
) -> dict[str, str]:
    """Build a hash->text map for all system prompts used in this experiment."""
    prompts: dict[str, str] = {}
    for text in config.system_prompts.values():
        prompts[_prompt_hash(text)] = text
    for phase in phases:
        if phase.assistant_system_prompt:
            prompts[_prompt_hash(phase.assistant_system_prompt)] = (
                phase.assistant_system_prompt
            )
        if phase.user_simulator:
            try:
                text = get_user_simulator_instruction(
                    phase.user_simulator.prompt_template
                )
                prompts[_prompt_hash(text)] = text
            except ValueError:
                pass
    default_user_sim = get_user_simulator_instruction("typical_user")
    prompts[_prompt_hash(default_user_sim)] = default_user_sim
    return prompts


def save_experiment_metadata(
    config: ExperimentConfig,
    run_dir: Path,
    experiment_name: str,
    phases: list[Phase],
) -> None:
    """Write experiment_metadata.json and a copy of the calling script to run_dir."""
    metadata: dict[str, Any] = {
        "experiment_name": experiment_name,
        "script": sys.argv[0],
        "git_commit_hash": _git_commit_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": dataclasses.asdict(config),
        "system_prompts": _build_system_prompts_map(config, phases),
    }
    rollouts_dir = run_dir / "rollouts"
    rollouts_dir.mkdir(parents=True, exist_ok=True)
    (rollouts_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str)
    )
    script_path = Path(sys.argv[0]).resolve()
    if script_path.exists():
        shutil.copy2(script_path, rollouts_dir / script_path.name)


# ── HuggingFace upload / download ─────────────────────────────────────────────


def upload_to_hf(
    output_config: OutputPathConfig,
    run_dir: Path,
) -> None:
    """Upload run_dir to HuggingFace, mirroring the structured path.

    Uploads rollout artifacts, excluding evals, datasets, events, and exports.
    """
    if not output_config.hf_repo:
        return
    login_from_env()
    git_hash = _git_commit_hash()
    hash_suffix = f" (git: {git_hash[:8]})" if git_hash else ""
    path_in_repo = output_config.hf_path
    url = upload_folder_to_dataset_repo(
        local_dir=run_dir,
        repo_id=output_config.hf_repo,
        path_in_repo=path_in_repo,
        commit_message=f"Upload eval: {output_config.eval_name}{hash_suffix}",
        ignore_patterns=[
            "datasets/*",
            "events/*",
            "exports/*",
            "evals/*",
            "per_message_metrics.jsonl",
        ],
    )
    print(f"  Uploaded to {url}")


def download_rollouts_from_hf(output_config: OutputPathConfig) -> None:
    """Download rollouts and existing eval files from HF to local scratch.

    Fetches only ``rollouts/rollouts.jsonl`` and ``evals/rollouts_evaluated.jsonl``
    files under the configured HF path.
    """
    if not output_config.hf_repo:
        return

    from src.utils.hf_hub import download_from_dataset_repo

    login_from_env()

    print(
        f"Downloading rollouts from HF: {output_config.hf_repo}/{output_config.hf_path}"
    )
    download_from_dataset_repo(
        repo_id=output_config.hf_repo,
        path_in_repo=output_config.hf_path,
        local_dir=output_config.scratch_root,
        allow_patterns=[
            "*/rollouts/rollouts.jsonl",
            "*/evals/rollouts_evaluated.jsonl",
        ],
    )


def upload_evals_to_hf(
    output_config: OutputPathConfig,
    root_dir: Path,
    evals_dirs: list[Path] | None = None,
) -> None:
    """Upload evals/ subtrees to HuggingFace.

    Args:
        output_config: Output path configuration.
        root_dir: Local root directory for the sweep.
        evals_dirs: Specific evals/ directories to upload. If None, discovers
            all evals/ dirs under root_dir (slower for large trees).
    """
    if not output_config.hf_repo:
        return

    login_from_env()

    if evals_dirs is None:
        evals_dirs = [
            p.parent for p in sorted(root_dir.rglob("evals/rollouts_evaluated.jsonl"))
        ]

    if not evals_dirs:
        print("No evaluated rollouts to upload.")
        return

    print(f"Uploading evals from {len(evals_dirs)} cell(s) to HF...")
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    operations = []
    for evals_dir in evals_dirs:
        cell_dir = evals_dir.parent
        rel = cell_dir.relative_to(root_dir)
        for fpath in sorted(evals_dir.rglob("*")):
            if fpath.is_file():
                rel_file = fpath.relative_to(evals_dir)
                path_in_repo = f"{output_config.hf_path}/{rel}/evals/{rel_file}"
                operations.append(
                    CommitOperationAdd(
                        path_in_repo=path_in_repo,
                        path_or_fileobj=str(fpath),
                    )
                )

    if not operations:
        print("No eval files to upload.")
        return

    api.create_commit(
        repo_id=output_config.hf_repo,
        repo_type="dataset",
        operations=operations,
        commit_message=(
            f"Upload evals: {output_config.eval_name} ({len(evals_dirs)} cells)"
        ),
    )
    print(f"  Uploaded {len(operations)} file(s) from {len(evals_dirs)} cell(s)")


def _load_completed_cells_from_hf(
    output_config: OutputPathConfig,
) -> set[str]:
    """Fetch cells with completed rollouts from HF.

    Returns a set of ``"variant/condition"`` strings that have
    ``rollouts/rollout_info.json`` with ``status == "ok"``.
    """
    if not output_config.hf_repo:
        return set()

    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    prefix = output_config.hf_path
    completed: set[str] = set()

    try:
        all_files = api.list_repo_tree(
            repo_id=output_config.hf_repo,
            path_in_repo=prefix,
            repo_type="dataset",
            recursive=True,
        )
        info_paths = [
            f.rfilename
            for f in all_files
            if hasattr(f, "rfilename")
            and f.rfilename.endswith("/rollouts/rollout_info.json")
        ]
    except Exception:  # noqa: BLE001
        return set()

    for rpath in info_paths:
        try:
            local_path = hf_hub_download(
                repo_id=output_config.hf_repo,
                filename=rpath,
                repo_type="dataset",
            )
            info = json.loads(Path(local_path).read_text())
            if info.get("status") == "ok":
                # Extract "variant/condition" from
                # "prefix/variant/condition/rollouts/rollout_info.json"
                rel = rpath[len(prefix) :].strip("/")
                cell_key = rel.rsplit("/rollouts/rollout_info.json", 1)[0]
                completed.add(cell_key)
        except Exception:  # noqa: BLE001
            continue

    return completed


# Directories/files excluded from cell uploads (mirrors upload_to_hf ignore_patterns).
_CELL_UPLOAD_IGNORE = {
    "datasets",
    "events",
    "exports",
    "evals",
    "per_message_metrics.jsonl",
}


def _batch_upload_variant_cells_to_hf(
    output_config: OutputPathConfig,
    cells: list[tuple[Path, str, str]],
    successful: set[tuple[str, str]],
) -> None:
    """Upload all cells for a variant in a single HF commit.

    Args:
        output_config: Output path configuration.
        cells: List of ``(cell_dir, variant, condition)`` tuples to upload.
        successful: Unused — kept for call-site compatibility.
    """
    if not cells:
        return
    login_from_env()
    from huggingface_hub import CommitOperationAdd, HfApi

    git_hash = _git_commit_hash()
    hash_suffix = f" (git: {git_hash[:8]})" if git_hash else ""
    api = HfApi()
    operations: list = []

    for cell_dir, variant, condition in cells:
        path_prefix = f"{output_config.hf_path}/{variant}/{condition}"
        for fpath in sorted(cell_dir.rglob("*")):
            if not fpath.is_file():
                continue
            rel = fpath.relative_to(cell_dir)
            if rel.parts[0] in _CELL_UPLOAD_IGNORE or rel.name in _CELL_UPLOAD_IGNORE:
                continue
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"{path_prefix}/{rel}",
                    path_or_fileobj=str(fpath),
                )
            )

    if not operations:
        return

    vlabel = next(v for _, v, _ in cells)
    api.create_commit(
        repo_id=output_config.hf_repo,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Upload variant: {vlabel} ({len(cells)} cell(s)){hash_suffix}",
        create_pr=False,
    )
    print(
        f"    Uploaded {len(cells)} cell(s) for {vlabel} to HF in 1 commit",
        flush=True,
    )
