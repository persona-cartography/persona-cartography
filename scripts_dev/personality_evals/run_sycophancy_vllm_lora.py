"""Sycophancy for qwen-3-32b downstream conditions via one vLLM server + runtime LoRA.

Unlike ``run_sycophancy_vllm.py`` (which bakes one full merged model per
condition — ~40 min each for a 32B), this serves the base model ONCE with
``--enable-lora`` and evaluates every condition against the same server:

  base, A+@+1, A+@-1, A-@+1, A-@-1, control@+1

Scale −1 variants are materialized by negating the adapter's ``lora_B``
weights (exact: ΔW → −ΔW). The qwen persona adapters have lora_alpha == r,
so vLLM's native alpha/r scaling (=1.0) matches the suite's ``scaling :=
scale`` convention at ±1.

Qwen3 hybrid thinking is disabled per request via
``chat_template_kwargs={"enable_thinking": False}`` (Inspect ``extra_body``),
matching the adapters' nothink training recipe and the suite's
``eval_thinking=False``.

Env vars: DS_LIMIT (default 100) — sample cap for the upstream sycophancy task.

Usage
-----
    uv run python -m scripts_dev.personality_evals.run_sycophancy_vllm_lora
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env")

import random  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

from src.utils.hf_hub import (  # noqa: E402
    download_from_dataset_repo,
    login_from_env,
    upload_folder_to_dataset_repo,
)

MODEL = "qwen-3-32b-it"
BASE_MODEL = "Qwen/Qwen3-32B"
HF_REPO = "persona-cartography/monorepo"
JUDGE = "openrouter/openai/gpt-5-nano"
VERSION = "ocean_const_paired_dpo_nothink"
CONTROL_VERSION = "ocean_const_paired_dpo_nothink_s1vs2"
_limit_env = os.environ.get("DS_LIMIT", "100").strip()
LIMIT = int(_limit_env) if _limit_env else None
_LIMIT_TAG = f"n{LIMIT}" if LIMIT else "full"

API_KEY = "inspectai"
OUTPUT_ROOT = PROJECT_ROOT / "scratch" / "evals" / "ocean" / "qwen_downstream_syco"


def _adapter_and_prefix(d: str) -> tuple[str, str]:
    if d == "control":
        prefix = f"fine_tuning/{MODEL}/other/ocean_def_control/amplifier/{CONTROL_VERSION}"
        return f"{prefix}/lora/ocean_def_control_full-persona", prefix
    long = {"amp": "amplifier", "sup": "suppressor"}[d]
    verb = {"amp": "amplifying", "sup": "suppressing"}[d]
    prefix = f"fine_tuning/{MODEL}/ocean/agreeableness/{long}/{VERSION}"
    return f"{prefix}/lora/agreeableness_{verb}_full-persona", prefix


def _ensure_adapter(d: str) -> Path:
    path_in_repo, _ = _adapter_and_prefix(d)
    cache = PROJECT_ROOT / "scratch" / "adapters" / f"{MODEL}-{d}"
    local = cache / path_in_repo
    if not (local / "adapter_model.safetensors").exists():
        download_from_dataset_repo(repo_id=HF_REPO, path_in_repo=path_in_repo, local_dir=cache)
    return local.resolve()


def _negated_copy(src: Path, dst: Path) -> Path:
    """Copy adapter with lora_B negated (exact scale −1 variant)."""
    from safetensors.torch import load_file, save_file

    if (dst / "adapter_model.safetensors").exists():
        return dst
    dst.mkdir(parents=True, exist_ok=True)
    sd = load_file(str(src / "adapter_model.safetensors"))
    neg = {k: (-v if "lora_B" in k else v) for k, v in sd.items()}
    n_flipped = sum(1 for k in sd if "lora_B" in k)
    assert n_flipped > 0, "no lora_B tensors found — negation would be a no-op"
    save_file(neg, str(dst / "adapter_model.safetensors"))
    shutil.copy(src / "adapter_config.json", dst / "adapter_config.json")
    print(f"  negated {n_flipped} lora_B tensors -> {dst}", flush=True)
    return dst


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_ready(base_url: str, proc: subprocess.Popen, timeout_s: int = 1200) -> None:
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm serve exited with code {proc.returncode}")
        try:
            req = urllib.request.Request(
                f"{base_url}/models", headers={"Authorization": f"Bearer {API_KEY}"}
            )
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status == 200:
                    return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    raise TimeoutError(f"vllm not ready within {timeout_s}s")


def main() -> None:
    login_from_env()

    # ── Materialize adapters (+1 originals, −1 negated copies) ──────────────
    amp = _ensure_adapter("amp")
    sup = _ensure_adapter("sup")
    ctrl = _ensure_adapter("control")
    neg_root = PROJECT_ROOT / "scratch" / "adapters" / f"{MODEL}-negated"
    amp_m1 = _negated_copy(amp, neg_root / "amp_m1")
    sup_m1 = _negated_copy(sup, neg_root / "sup_m1")

    # condition -> (served model name, lora path or None for base, upload prefix, spec dir)
    _, amp_prefix = _adapter_and_prefix("amp")
    _, sup_prefix = _adapter_and_prefix("sup")
    _, ctrl_prefix = _adapter_and_prefix("control")
    conditions = [
        ("base", None, amp_prefix, "base"),
        ("amp_p1", amp, amp_prefix, "lora_+1p00x"),
        ("amp_m1", amp_m1, amp_prefix, "lora_-1p00x"),
        ("sup_p1", sup, sup_prefix, "lora_+1p00x"),
        ("sup_m1", sup_m1, sup_prefix, "lora_-1p00x"),
        ("ctrl_p1", ctrl, ctrl_prefix, "lora_+1p00x"),
    ]

    # ── Serve once ───────────────────────────────────────────────────────────
    port = _free_port()
    base_url = f"http://localhost:{port}/v1"
    lora_modules = [f"{n}={p}" for n, p, _, _ in conditions if p is not None]
    cmd = [
        "vllm", "serve", BASE_MODEL,
        "--served-model-name", "base",
        "--host", "0.0.0.0", "--port", str(port),
        "--api-key", API_KEY,
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "4096",
        "--enable-lora", "--max-lora-rank", "64", "--max-loras", "5",
        "--lora-modules", *lora_modules,
    ]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    vllm_log = OUTPUT_ROOT / "vllm_server.log"
    print(f"starting vllm: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, stdout=open(vllm_log, "wb"), stderr=subprocess.STDOUT)
    try:
        _wait_ready(base_url, proc)
        print(f"vllm ready at {base_url}", flush=True)

        from inspect_ai import eval as inspect_eval
        from inspect_evals.sycophancy.sycophancy import sycophancy

        for name, _lora, prefix, spec_dir in conditions:
            run_name = f"{MODEL}_agree_syco_{_LIMIT_TAG}"
            out_dir = OUTPUT_ROOT / run_name / name / "sycophancy" / "native" / "inspect_logs"
            if any(out_dir.glob("2*_sycophancy_*.json")):
                print(f"[{name}] already done, skipping", flush=True)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{name}] running sycophancy (limit={LIMIT}) ...", flush=True)
            t0 = time.time()
            logs = inspect_eval(
                sycophancy(scorer_model=JUDGE),
                model=f"vllm/{name}",
                model_base_url=base_url,
                model_args={"api_key": API_KEY},
                limit=LIMIT,
                temperature=0.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                max_connections=64,
                log_dir=str(out_dir),
                log_format="json",
                log_samples=True,
                display="plain",
                score=True,
            )
            log = logs[0]
            print(f"[{name}] status={log.status} in {time.time()-t0:.0f}s", flush=True)
            (out_dir / "metadata.json").write_text(json.dumps({
                "completed_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                "base_model": BASE_MODEL,
                "condition": name,
                "spec_dir": spec_dir,
                "limit": LIMIT,
                "temperature": 0.0,
                "enable_thinking": False,
                "judge_model": JUDGE,
                "provider": "vllm runtime-lora (lora_B negation for -1)",
            }, indent=2))
            hf_path = f"{prefix}/evals/downstream/{run_name}/{spec_dir}/sycophancy/native/inspect_logs"
            try:
                upload_folder_to_dataset_repo(
                    local_dir=out_dir,
                    repo_id=HF_REPO,
                    path_in_repo=hf_path,
                    commit_message=f"sycophancy vllm-lora: {hf_path}",
                )
                print(f"[{name}] uploaded -> {hf_path}", flush=True)
            except Exception as exc:  # noqa: BLE001 - keep evaluating on upload failure
                print(f"[{name}] WARNING upload failed: {exc}", flush=True)
    finally:
        print("shutting down vllm ...", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
