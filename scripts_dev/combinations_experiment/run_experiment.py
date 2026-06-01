"""Run TRAIT + MMLU on each OCEAN LoRA-combination config (shared-base runner).

The base 8B model, all needed OCEAN adapters, and the two benchmark tasks are
loaded **once**. Each config is then evaluated by re-activating its five adapters
and re-setting their scaling on the resident model (a millisecond operation) and
running the cached ``personality_trait_logprobs`` + ``mmlu`` tasks against it — no
per-config model reload or task rebuild. Results are written in the suite's
``run_info.json`` + ``native/inspect_logs`` layout and uploaded to one independent
HF directory per config.

Independence is preserved without fresh loads: each config sets the active-adapter
set and resets every active adapter's scaling from a captured baseline, so no state
leaks between configs. Resume is handled by checking HF before loading the model.

Mirrors the ``vanton4_paired_dpo`` trait/mmlu reference configs in
``scripts_dev/personality_evals/configs/ocean/{trait,mmlu}/vanton4_paired_dpo/``.

Usage
-----
    # Design check only (no model load):
    uv run python -m scripts_dev.combinations_experiment.run_experiment --dry-run

    # Smoke test one config with tiny samples, no upload:
    uv run python -m scripts_dev.combinations_experiment.run_experiment \\
        --only <slug> --smoke --no-upload

    # Full run (single process, all 32 — resume skips finished configs):
    uv run python -m scripts_dev.combinations_experiment.run_experiment
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv
from inspect_ai.model import get_model

from scripts_dev.combinations_experiment.config_design import (
    ConfigRecord,
    ExperimentDesign,
    generate_design,
)
from src.utils.peft_manipulations import set_active_adapters
from src_dev.common.lora_catalogue import OCEAN_REGISTRY
from src_dev.evals import InspectBenchmarkSpec
from src_dev.evals.backends.inspect_runner import run_benchmark_eval
from src_dev.evals.inspect_benchmarks import build_benchmark_task
from src_dev.evals.utils.preloaded_hf_provider import register_preloaded_hf_provider

load_dotenv()

# --- Fixed experiment settings (mirror vanton4_paired_dpo references) ---------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
OUTPUT_ROOT = Path("scratch/evals/combinations_experiment")
ADAPTER_CACHE = Path("scratch/adapters/combinations_experiment")
HF_REPO_ID = "persona-shattering-lasr/monorepo"
HF_PREFIX = "combinations_experiments/llama-3.1-8b-it/ocean/vanton4_paired_dpo"

OCEAN_TRAIT_SPLITS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism",
]
TRAIT_SAMPLES_PER_TRAIT = 300
MMLU_LIMIT = 300
TEMPERATURE = 0.0

# Per-eval generation batch size. TRAIT is a single-token logprob read
# (prefill-bound — little to gain from a bigger batch); MMLU generates 32 tokens
# (decode-bound — benefits from a larger batch on a big GPU), so it gets a higher
# default. Keyed by eval name (see build_eval_specs).
DEFAULT_BATCH_SIZES = {"trait_logprobs": 128, "mmlu": 256}

# Upload only small JSON artifacts (run_info + inspect logs + manifest), never the
# large Inspect ``.eval`` SQLite logs. Cover both top-level and nested files.
_UPLOAD_ALLOW_PATTERNS = ["*.json", "**/*.json"]

SEED = 42


def _seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_eval_specs(smoke: bool = False) -> list[InspectBenchmarkSpec]:
    """The two benchmark specs (trait logprobs + mmlu); shrink samples for smoke."""
    return [
        InspectBenchmarkSpec(
            name="trait_logprobs",
            benchmark="personality_trait_logprobs",
            benchmark_args={
                "samples_per_trait": 5 if smoke else TRAIT_SAMPLES_PER_TRAIT,
                "trait_splits": OCEAN_TRAIT_SPLITS,
            },
            n_runs=1,
        ),
        InspectBenchmarkSpec(
            name="mmlu",
            benchmark="mmlu",
            limit=5 if smoke else MMLU_LIMIT,
            n_runs=1,
        ),
    ]


def prefetch_adapters(slugs: set[str]) -> dict[str, str]:
    """Pre-download each OCEAN adapter once; return ``slug -> "local://..."`` URI.

    Uses ``download_from_dataset_repo`` (subtree enumeration) instead of
    ``snapshot_download(allow_patterns=...)``, which is slow and can silently
    return zero files against the 19k-file monorepo.
    """
    from src_dev.utils.hf_hub import download_from_dataset_repo

    uris: dict[str, str] = {}
    for slug in sorted(slugs):
        path_in_repo = OCEAN_REGISTRY[slug].adapter_path_in_repo
        download_from_dataset_repo(
            repo_id=HF_REPO_ID, path_in_repo=path_in_repo, local_dir=ADAPTER_CACHE
        )
        local_path = (ADAPTER_CACHE / path_in_repo).resolve()
        uris[slug] = f"local://{local_path}"
        print(f"  [adapter] {slug} ready", flush=True)
    return uris


# --- Shared model (loaded once) ----------------------------------------------


@dataclass
class SharedModel:
    """The resident base+adapters model and its reusable Inspect wrappers."""

    peft_model: Any
    tokenizer: Any
    # One Inspect wrapper per eval name (same resident model, different batch size).
    inspect_models: dict[str, Any]
    # Original ``module.scaling[adapter]`` per (module_name, adapter), captured at
    # load time so each config can reset to baseline * config_scale.
    base_scaling: dict[tuple[str, str], float]


def _flash_attn_kwargs() -> dict[str, str]:
    try:
        import flash_attn  # noqa: F401

        return {"attn_implementation": "flash_attention_2"}
    except ImportError:
        return {}


def _capture_base_scaling(peft_model: Any) -> dict[tuple[str, str], float]:
    base: dict[tuple[str, str], float] = {}
    for name, module in peft_model.named_modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict):
            for adapter, value in scaling.items():
                base[(name, adapter)] = value
    return base


def load_shared_model(
    adapter_uris: dict[str, str], device_map: str, *, batch_sizes: dict[str, int]
) -> SharedModel:
    """Load the base model, all adapters, and a per-eval Inspect wrapper once.

    ``batch_sizes`` maps eval name -> generation batch size; one wrapper is built
    per entry (all sharing the resident model) so each eval runs at its own batch.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float32 if device_map == "cpu" else torch.bfloat16
    print(f"  loading base model {BASE_MODEL} ...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=dtype, device_map=device_map, **_flash_attn_kwargs()
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    peft_model = None
    for slug in sorted(adapter_uris):
        local_dir = adapter_uris[slug][len("local://") :]
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(base, local_dir, adapter_name=slug)
        else:
            peft_model.load_adapter(local_dir, adapter_name=slug)
        print(f"  [adapter loaded] {slug}", flush=True)
    assert peft_model is not None, "no adapters to load"
    peft_model.eval()

    base_scaling = _capture_base_scaling(peft_model)
    register_preloaded_hf_provider()
    inspect_models = {
        name: get_model(
            f"hf_preloaded/combo_{name}",
            hf_model=peft_model,
            hf_tokenizer=tokenizer,
            batch_size=bs,
        )
        for name, bs in batch_sizes.items()
    }
    return SharedModel(peft_model, tokenizer, inspect_models, base_scaling)


def _apply_config_scaling(
    shared: SharedModel, scale_map: dict[str, float]
) -> None:
    """Activate this config's adapters and set each one's scaling = base * scale.

    Inactive adapters contribute nothing to the forward pass regardless of their
    scaling, so only the active set needs to be (re)scaled. Resetting from the
    captured baseline (not the previous config's value) keeps configs independent.
    """
    set_active_adapters(shared.peft_model, list(scale_map))
    for name, module in shared.peft_model.named_modules():
        scaling = getattr(module, "scaling", None)
        if not isinstance(scaling, dict):
            continue
        for adapter, scale in scale_map.items():
            if adapter in scaling:
                scaling[adapter] = shared.base_scaling[(name, adapter)] * scale


def _write_run_info(
    run_dir: Path,
    record: ConfigRecord,
    eval_spec: InspectBenchmarkSpec,
    *,
    status: str,
    error: str | None,
    inspect_log_path: str | None,
    inspect_status: str | None,
) -> None:
    """Write a suite-compatible run_info.json (read by the resume check + notebook)."""
    payload = {
        "suite_run_name": record.slug,
        "status": status,
        "error": error,
        "eval_spec": eval_spec.model_dump(mode="json"),
        "scale": record.sumscale_actual,
        "native": {
            "inspect_log_path": inspect_log_path,
            "inspect_status": inspect_status,
        },
        "experiment": "ocean_lora_combinations",
        "adapters": [
            {
                "registry_slug": s,
                "adapter_ref": OCEAN_REGISTRY[s].adapter_ref,
                "scale": sc,
            }
            for s, sc in record.adapters
        ],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_info.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


def run_one_config(
    record: ConfigRecord,
    shared: SharedModel,
    eval_specs: list[InspectBenchmarkSpec],
    tasks: dict[str, Any],
) -> None:
    """Evaluate one config against the shared model; write logs + run_info."""
    scale_map = {slug: scale for slug, scale in record.adapters}
    _apply_config_scaling(shared, scale_map)

    for eval_spec in eval_specs:
        run_dir = _model_dir_for(record) / eval_spec.name
        result = run_benchmark_eval(
            spec=eval_spec,
            model_uri=shared.inspect_models[eval_spec.name],
            run_dir=run_dir,
            temperature=TEMPERATURE,
            task=tasks[eval_spec.name],
        )
        inspect_status = result.log.status if result.log else None
        _write_run_info(
            run_dir,
            record,
            eval_spec,
            status=result.status,
            error=result.error,
            inspect_log_path=result.log.location if result.log else None,
            inspect_status=inspect_status,
        )
        print(
            f"    {eval_spec.name}: status={result.status} inspect={inspect_status}",
            flush=True,
        )


# --- HF I/O ------------------------------------------------------------------


def _model_dir_for(record: ConfigRecord) -> Path:
    """Local directory holding this config's eval results."""
    return OUTPUT_ROOT / record.slug / record.slug


def _upload_config(record: ConfigRecord) -> None:
    """Upload one config's eval dirs + manifest to its independent HF directory."""
    from src_dev.utils.hf_hub import login_from_env, upload_folder_to_dataset_repo

    model_dir = _model_dir_for(record)
    if not model_dir.exists():
        print(f"  [upload] SKIP {record.slug}: no results at {model_dir}", flush=True)
        return

    # Per-config manifest sits alongside the eval subdirs.
    (model_dir / "manifest.json").write_text(
        json.dumps(record.to_dict(), indent=2), encoding="utf-8"
    )

    login_from_env()
    upload_folder_to_dataset_repo(
        local_dir=model_dir,
        repo_id=HF_REPO_ID,
        path_in_repo=f"{HF_PREFIX}/{record.slug}",
        commit_message=f"combinations_experiment: {record.slug}",
        allow_patterns=_UPLOAD_ALLOW_PATTERNS,
        # Clean-replace: drop any stale files (e.g. a prior OOM run's
        # timestamped inspect log) so the config dir holds only this run.
        delete_patterns=["**"],
    )
    print(f"  [upload] {record.slug} -> {HF_PREFIX}/{record.slug}", flush=True)


def _upload_experiment_manifest(design: ExperimentDesign) -> None:
    """Upload the top-level manifest describing all 32 configs (once)."""
    from src_dev.utils.hf_hub import login_from_env, upload_folder_to_dataset_repo

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_ROOT / "_experiment_manifest.json"
    manifest_path.write_text(json.dumps(design.manifest(), indent=2), encoding="utf-8")

    login_from_env()
    upload_folder_to_dataset_repo(
        local_dir=OUTPUT_ROOT,
        repo_id=HF_REPO_ID,
        path_in_repo=HF_PREFIX,
        commit_message="combinations_experiment: experiment manifest",
        allow_patterns=["_experiment_manifest.json"],
    )
    print(f"  [upload] _experiment_manifest.json -> {HF_PREFIX}", flush=True)


def _read_hf_json(path_in_repo: str) -> dict | None:
    """Fetch a single small JSON file from the dataset repo, or None if absent."""
    from huggingface_hub import hf_hub_download

    try:
        local = hf_hub_download(
            repo_id=HF_REPO_ID, filename=path_in_repo, repo_type="dataset"
        )
    except Exception:
        return None
    try:
        return json.loads(Path(local).read_text())
    except Exception:
        return None


def _config_done_on_hf(eval_specs: list[InspectBenchmarkSpec], slug: str) -> bool:
    """True iff *both* evals are on HF, fully succeeded, with a matching spec.

    We require the *Inspect* status (``native.inspect_status``) to be
    ``"success"`` — the top-level ``status`` can read ``"ok"`` even when the eval
    errored mid-run (e.g. an OOM that produced a partial, score-less log), so
    trusting it would wrongly skip a broken config. The eval-spec equality check
    additionally prevents a smoke run (different ``benchmark_args``) counting as done.
    """
    for eval_spec in eval_specs:
        info = _read_hf_json(f"{HF_PREFIX}/{slug}/{eval_spec.name}/run_info.json")
        if not info:
            return False
        if info.get("native", {}).get("inspect_status") != "success":
            return False
        if info.get("eval_spec") != eval_spec.model_dump(mode="json"):
            return False
    return True


def _parse_shard(shard: str | None) -> tuple[int, int]:
    """Parse ``"i/n"`` into ``(i, n)``; default ``(0, 1)`` (all configs)."""
    if shard is None:
        return 0, 1
    i_str, n_str = shard.split("/")
    i, n = int(i_str), int(n_str)
    if not (0 <= i < n):
        raise ValueError(f"shard index out of range: {shard}")
    return i, n


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--shard", default=None, help='Run a subset "i/n" of configs (e.g. "0/4").'
    )
    parser.add_argument(
        "--only", default=None, help="Run only the config with this exact slug."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the selected configs and exit (no model load).",
    )
    parser.add_argument(
        "--no-upload", action="store_true",
        help="Skip all HF uploads (local results only).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Tiny sample counts for a fast plumbing test.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run configs even if already complete on HF (disables resume skip).",
    )
    parser.add_argument(
        "--allow-cpu", action="store_true",
        help="Permit running on CPU (very slow). By default a GPU is required.",
    )
    parser.add_argument(
        "--trait-batch-size", type=int, default=DEFAULT_BATCH_SIZES["trait_logprobs"],
        help=f"TRAIT batch size (default {DEFAULT_BATCH_SIZES['trait_logprobs']}).",
    )
    parser.add_argument(
        "--mmlu-batch-size", type=int, default=DEFAULT_BATCH_SIZES["mmlu"],
        help=f"MMLU batch size (default {DEFAULT_BATCH_SIZES['mmlu']}); larger can "
             "speed up its 32-token decode on a big GPU. Watch nvidia-smi for memory.",
    )
    args = parser.parse_args()

    _seed_everything()

    design = generate_design()
    shard_i, shard_n = _parse_shard(args.shard)

    if args.only is not None:
        selected = [design.configs_by_slug[args.only]]
    else:
        selected = [c for c in design.configs if c.index % shard_n == shard_i]

    print(
        f"Experiment: {len(design.configs)} configs total; "
        f"shard {shard_i}/{shard_n} -> {len(selected)} selected"
        f"{' [SMOKE]' if args.smoke else ''}",
        flush=True,
    )

    if args.dry_run:
        for c in selected:
            print(
                f"  {c.index:>3}  {c.slug:<40}  "
                f"Σ={c.sumscale_actual:.3f}  {c.adapters}"
            )
        return

    # Require a GPU by default; load explicitly onto it (not accelerate "auto",
    # which would silently fall back to CPU).
    if torch.cuda.is_available():
        device_map = "cuda"
        print(f"  device: cuda ({torch.cuda.get_device_name(0)})", flush=True)
    elif args.allow_cpu:
        device_map = "cpu"
        print("  device: CPU (--allow-cpu set; this will be very slow)", flush=True)
    else:
        raise SystemExit(
            "No CUDA GPU detected — refusing to run (MCQ on CPU would take many "
            "hours). Run on a GPU machine, or pass --allow-cpu to override."
        )

    eval_specs = build_eval_specs(smoke=args.smoke)

    # Resume: drop configs already complete on HF *before* loading the model.
    if not args.force and not args.smoke:
        kept = []
        for rec in selected:
            if _config_done_on_hf(eval_specs, rec.slug):
                print(f"  already complete on HF, skipping {rec.slug}", flush=True)
            else:
                kept.append(rec)
        selected = kept
    if not selected:
        print("Nothing to run — all selected configs already complete.", flush=True)
        return

    # Pre-download every adapter the remaining configs need (once each).
    needed_slugs = {slug for rec in selected for slug, _ in rec.adapters}
    print(f"Pre-downloading {len(needed_slugs)} OCEAN adapter(s) ...", flush=True)
    adapter_uris = prefetch_adapters(needed_slugs)

    if not args.no_upload and (args.only is None) and shard_i == 0:
        _upload_experiment_manifest(design)

    # Load base + adapters + tasks ONCE, reuse across all configs.
    batch_sizes = {
        "trait_logprobs": args.trait_batch_size,
        "mmlu": args.mmlu_batch_size,
    }
    print(f"Loading shared base model + adapters (once, batch={batch_sizes}) ...", flush=True)
    shared = load_shared_model(adapter_uris, device_map, batch_sizes=batch_sizes)
    print("Building eval tasks (once) ...", flush=True)
    tasks = {spec.name: build_benchmark_task(spec) for spec in eval_specs}

    for n, record in enumerate(selected, 1):
        print(
            f"\n[{n}/{len(selected)}] === {record.slug} "
            f"(Σ={record.sumscale_actual:.3f}) ===",
            flush=True,
        )
        # Clean local dir so a prior failed attempt's logs don't linger.
        shutil.rmtree(OUTPUT_ROOT / record.slug, ignore_errors=True)
        try:
            run_one_config(record, shared, eval_specs, tasks)
        except Exception as exc:  # keep going; one bad config shouldn't kill the run
            print(f"  ERROR running {record.slug}: {exc}", flush=True)
            continue
        if not args.no_upload:
            _upload_config(record)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
