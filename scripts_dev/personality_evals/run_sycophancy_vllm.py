"""vLLM launcher for sycophancy evals at a sign-flipped LoRA scale (e.g. -1).

Per-side workflow, driven by a SuiteConfig declared in
``scripts_dev/personality_evals/configs/ocean/sycophancy/...``:

1. Resolve the adapter (single-adapter ModelSpec) and its scale.
2. Bake the adapter into base weights at that scale via
   ``peft.merge_and_unload`` and save to ``scratch/merged/<run>_<scale_tag>/``.
   Idempotent: skips baking if the merged dir already has weight shards.
3. Start a local ``vllm serve`` subprocess for the merged dir, with
   ``--served-model-name <run>_<scale_tag>`` (so Inspect's vLLM provider
   queries match what vLLM serves; otherwise the OpenAI-compatible path
   replies 404).
4. Run the upstream ``inspect_evals.sycophancy.sycophancy`` task via
   ``inspect_ai.eval(model="vllm/<served_name>", model_args={"base_url": ...})``
   with ``max_connections=64``. Logs land in
   ``<output_root>/<run_name>/<spec_name>/sycophancy/native/inspect_logs/``.
5. Tear down the vLLM subprocess.

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python -m scripts_dev.personality_evals.run_sycophancy_vllm \\
        --config-module scripts_dev.personality_evals.configs.ocean.sycophancy.a_minus_vanton4_paired_dpo
"""

from __future__ import annotations

import argparse
import datetime as _dt
import gc
import importlib
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

import random  # noqa: E402

import numpy as np  # noqa: E402

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scale_tag(scale: float) -> str:
    """Match the suite's ``lora_<+/-NpNNx`` convention."""
    return f"{scale:+.2f}".replace(".", "p")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _resolve_local_adapter_path(adapter_path: str) -> Path:
    if adapter_path.startswith("local://"):
        return Path(adapter_path[len("local://"):])
    return Path(adapter_path)


def _git_info(repo_root: Path) -> dict[str, Any]:
    """Capture current commit / branch / dirty-state for run provenance.

    Best-effort: returns whatever fields succeed; never raises.
    """
    info: dict[str, Any] = {}
    def _run(args: list[str]) -> str | None:
        try:
            out = subprocess.run(
                args, cwd=repo_root, capture_output=True, text=True, check=True, timeout=10
            )
            return out.stdout.strip() or None
        except Exception:
            return None

    info["commit"] = _run(["git", "rev-parse", "HEAD"])
    info["commit_short"] = _run(["git", "rev-parse", "--short", "HEAD"])
    info["branch"] = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    porcelain = _run(["git", "status", "--porcelain"])
    info["dirty"] = bool(porcelain) if porcelain is not None else None
    return info


def _write_run_metadata(
    out_dir: Path,
    *,
    config_module: str,
    run_name: str,
    spec_name: str,
    base_model: str,
    adapter_path: str,
    scale: float,
    served_name: str,
    benchmark_args: dict[str, Any],
    config_metadata: dict[str, Any],
    upload_repo_id: str | None,
    upload_path_in_repo: str | None,
    repo_root: Path,
) -> Path:
    """Write a run-metadata JSON sidecar inside ``out_dir`` for HF-upload provenance."""
    out_dir.mkdir(parents=True, exist_ok=True)
    inspect_logs = sorted(p.name for p in out_dir.glob("*.json") if p.name != "metadata.json")
    payload = {
        "completed_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git": _git_info(repo_root),
        "config_module": config_module,
        "run_name": run_name,
        "spec_name": spec_name,
        "base_model": base_model,
        "adapter_path": adapter_path,
        "scale": scale,
        "served_name": served_name,
        "benchmark": "sycophancy",
        "benchmark_args": benchmark_args,
        "config_metadata": config_metadata,
        "upload_repo_id": upload_repo_id,
        "upload_path_in_repo": upload_path_in_repo,
        "inspect_logs": inspect_logs,
    }
    md_path = out_dir / "metadata.json"
    with md_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"  wrote run metadata: {md_path}", flush=True)
    return md_path


def _bake_merged(
    base_model: str,
    adapter_local_path: Path,
    scale: float,
    out_dir: Path,
) -> Path:
    """Merge ``adapter @ scale`` into base weights and save to ``out_dir``.

    Idempotent: skips if a safetensors index file is already present.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "model.safetensors.index.json").exists() or any(out_dir.glob("*.safetensors")):
        print(f"  bake: merged model already at {out_dir}", flush=True)
        return out_dir

    print(f"  loading {base_model} (bf16, device_map=auto) ...", flush=True)
    from peft import PeftModel  # type: ignore[import-untyped]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    peft_model = PeftModel.from_pretrained(
        base, str(adapter_local_path), adapter_name="adapter"
    )
    print(f"  setting LoRA scale={scale} ...", flush=True)
    for module in peft_model.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict):
            for k in scaling:
                scaling[k] = float(scale)
    print("  merging (merge_and_unload) ...", flush=True)
    merged = peft_model.merge_and_unload()
    print(f"  saving merged model to {out_dir} ...", flush=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.save_pretrained(str(out_dir))

    del merged, peft_model, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out_dir


def _wait_for_vllm_ready(base_url: str, api_key: str, proc: subprocess.Popen, timeout_s: int = 600) -> None:
    """Poll the OpenAI-compatible /v1/models endpoint until the server responds."""
    import urllib.error
    import urllib.request

    url = f"{base_url}/models"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f"vllm serve exited with code {proc.returncode} before becoming ready"
            )
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError(f"vllm serve at {base_url} not ready within {timeout_s}s")


def _start_vllm_server(merged_dir: Path, served_name: str, port: int, log_path: Path) -> subprocess.Popen:
    api_key = "inspectai"
    cmd = [
        "vllm", "serve", str(merged_dir),
        "--served-model-name", served_name,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--api-key", api_key,
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "4096",
    ]
    print(f"  starting vllm serve on port {port} (served_name={served_name}) ...", flush=True)
    print(f"  cmd: {' '.join(cmd)}", flush=True)
    log_fh = open(log_path, "wb")
    proc = subprocess.Popen(cmd, stdout=log_fh, stderr=subprocess.STDOUT)
    base_url = f"http://localhost:{port}/v1"
    _wait_for_vllm_ready(base_url, api_key, proc)
    print(f"  vllm ready at {base_url}", flush=True)
    return proc


def _run_inspect_eval(
    *,
    served_name: str,
    base_url: str,
    api_key: str,
    output_root: Path,
    benchmark_args: dict[str, Any],
    max_connections: int,
    limit: int | None = None,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    from inspect_ai import eval as inspect_eval
    from inspect_evals.sycophancy.sycophancy import sycophancy

    task = sycophancy(**benchmark_args)
    print(
        f"  inspect_eval(model=vllm/{served_name}, base_url={base_url}, "
        f"max_connections={max_connections}, limit={limit}) ...",
        flush=True,
    )
    logs = inspect_eval(
        task,
        model=f"vllm/{served_name}",
        model_base_url=base_url,
        model_args={"api_key": api_key},
        log_dir=str(output_root),
        log_format="json",
        log_samples=True,
        display="plain",
        score=True,
        max_connections=max_connections,
        limit=limit,
    )
    if not logs:
        raise RuntimeError("inspect returned no logs")
    log = logs[0]
    print(f"  done. status={log.status}  log={log.location}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config-module",
        required=True,
        help="Dotted path to a config module exposing SUITE_CONFIG with one ModelSpec.",
    )
    ap.add_argument(
        "--max-connections",
        type=int,
        default=64,
        help="Inspect concurrent generations against the vLLM server (default 64).",
    )
    args = ap.parse_args()

    cfg_mod = importlib.import_module(args.config_module)
    suite_cfg = cfg_mod.SUITE_CONFIG
    assert len(suite_cfg.models) == 1, "expected a single ModelSpec"
    spec = suite_cfg.models[0]
    assert len(spec.adapters) == 1, "expected a single adapter"
    adapter = spec.adapters[0]
    base_model = spec.base_model
    scale = float(spec.scale if spec.scale is not None else adapter.scale)

    adapter_local = _resolve_local_adapter_path(adapter.path)
    if not adapter_local.exists():
        raise FileNotFoundError(f"adapter local dir not found: {adapter_local}")

    run_name = suite_cfg.run_name or cfg_mod.__name__.rsplit(".", 1)[-1]
    spec_name = spec.name  # e.g. "lora_-1p00x"
    served_name = f"{run_name}_{spec_name}".replace("/", "_")

    merged_dir = PROJECT_ROOT / "scratch" / "merged" / f"{run_name}_{_scale_tag(scale)}"
    out_dir = (
        PROJECT_ROOT
        / suite_cfg.output_root
        / run_name
        / spec_name
        / "sycophancy"
        / "native"
        / "inspect_logs"
    )

    print(f"=== sycophancy via vLLM | {run_name} ===")
    print(f"  base_model = {base_model}")
    print(f"  adapter    = {adapter_local}")
    print(f"  scale      = {scale}")
    print(f"  merged_dir = {merged_dir}")
    print(f"  served     = {served_name}")
    print(f"  out_dir    = {out_dir}")

    _bake_merged(base_model, adapter_local, scale, merged_dir)

    port = _free_port()
    vllm_log = out_dir.parent.parent.parent / f"vllm_{spec_name}.log"
    vllm_log.parent.mkdir(parents=True, exist_ok=True)
    proc = _start_vllm_server(merged_dir, served_name, port, vllm_log)
    try:
        # Pull benchmark_args (and optional sample limit) from the spec.
        bench_spec = suite_cfg.evals[0]
        benchmark_args = dict(bench_spec.benchmark_args or {})
        bench_limit = getattr(bench_spec, "limit", None)
        _run_inspect_eval(
            served_name=served_name,
            base_url=f"http://localhost:{port}/v1",
            api_key="inspectai",
            output_root=out_dir,
            benchmark_args=benchmark_args,
            max_connections=args.max_connections,
            limit=bench_limit,
        )
        _write_run_metadata(
            out_dir,
            config_module=args.config_module,
            run_name=run_name,
            spec_name=spec_name,
            base_model=base_model,
            adapter_path=str(adapter_local),
            scale=scale,
            served_name=served_name,
            benchmark_args=benchmark_args,
            config_metadata=dict(suite_cfg.metadata or {}),
            upload_repo_id=getattr(suite_cfg, "upload_repo_id", None),
            upload_path_in_repo=getattr(suite_cfg, "upload_path_in_repo", None),
            repo_root=PROJECT_ROOT,
        )
    finally:
        print("  shutting down vllm subprocess ...", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
