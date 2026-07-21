"""EQBench3 sweep driver for gemma-3-27b-it with neuroticism LoRA adapters.

Orchestrates a vLLM OpenAI-compatible server serving the base model and
two LoRA adapters (N+ and N-) simultaneously, then drives the vendored
eqbench3.py benchmark against each variant, sharing one runs/ELO results file
so the 3rd run's ELO pass compares all three.

Usage:
    python run_gemma27b_eqbench3.py [--dry-run] [--port PORT]

Flags:
    --dry-run: Print commands without executing (resolves adapters, does not
               start vLLM serve or eqbench3.py).
    --port: vLLM server port (default 8000).
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

SEED = 42
random.seed(SEED)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EQBenchSweepConfig:
    """Configuration for the EQBench3 gemma-27b neuroticism sweep."""

    # Judged via OpenRouter (not the Anthropic API directly): the vendored
    # api.py treats any non-anthropic.com base URL as OpenAI-format and has
    # explicit OpenRouter handling for anthropic/* models, so this needs no
    # vendored edits. OpenRouter is also this repo's standard judge path.
    judge_model: str = "anthropic/claude-opus-4-6"
    # Which variants to evaluate: "neuroticism" (base/N+/N-) or "all"
    # (base + the 10 OCEAN directions + the control adapter).
    scope: str = "neuroticism"
    iterations: int = 2
    # eqbench uses one thread pool for both scenario generation and rubric
    # judging. Each thread drives one conversation, so this is effectively the
    # concurrent-request count hitting vLLM (its continuous batcher batches
    # whatever it receives). At 6, threads spend most of their time blocked on
    # judge API calls and the GPU sits idle (~36% util); 32 keeps many
    # generations in flight while others judge. vLLM self-regulates via KV
    # cache (excess concurrent seqs queue rather than OOM).
    threads: int = 32
    port: int = 8000
    output_dir: Path = field(default_factory=lambda: Path("scratch/evals/eqbench3/gemma27b_n_sweep"))
    gpu_memory_utilization: float = 0.90
    # eqbench requests max_tokens=12000 for every generation, including the
    # debrief whose input is the full multi-turn transcript (~6-8k tokens for
    # 3-4 turn role-plays, more for analysis scenarios). vLLM enforces
    # input + max_tokens <= max_model_len, so 16384 rejects the debrief. gemma-3
    # supports 128k context; 40960 fits the longest input + 12000 output with
    # headroom, at modest KV-cache cost on a 143GB H200.
    max_model_len: int = 40960
    max_lora_rank: int = 64
    # gemma-3 on vLLM 0.11 + this H200/driver stack produces degenerate output
    # (empty at temp 0, repetition loops otherwise) with the default inductor
    # compilation + CUDA graphs; --enforce-eager disables both and yields
    # correct generations. Verified via endpoint smoke test. Required for
    # correctness here, at some throughput cost.
    enforce_eager: bool = True
    # Per-variant wall-clock cap. A single variant is 46 scenarios x N iterations
    # of a 27B model doing multi-turn ~1000-word generations plus rubric judging;
    # the final variant additionally runs the ELO pairwise pass across all three.
    # The comparable frustration-eval sweep took 3-5h on an H200, so a 1h cap
    # would kill mid-run and waste GPU + partial judge spend. Default 6h.
    subprocess_timeout_seconds: int = 21600


# Canonical control adapter for gemma-3-27b-it (the OCEAN registry covers the 10
# trait directions but not the control).
CONTROL_PATH_IN_REPO = (
    "fine_tuning/gemma-3-27b-it/other/ocean_def_control/amplifier/"
    "ocean_const_paired_dpo_s1vs2/lora/ocean_def_control_full-persona"
)

BASE_MODEL = "google/gemma-3-27b-it"


@dataclass(frozen=True)
class Variant:
    """One model under test: a logical name plus how vLLM should serve it."""

    model_name: str
    """Logical name recorded in the eqbench3 runs/ELO files."""
    served_name: str
    """The `model` field sent to vLLM (base repo id, or a --lora-modules name)."""
    adapter_ref: str | None = None
    """`repo::subfolder` adapter reference, or None for the base model."""


def build_variants(scope: str) -> list[Variant]:
    """Build the list of variants to evaluate.

    Adapter paths come from the canonical ``GEMMA_27B_OCEAN_REGISTRY`` (version
    ``ocean_const_paired_dpo``), not the legacy flat ``LoraHFCatalogue``, so these
    line up with the other gemma-3-27b evals in this repo.

    Args:
        scope: ``"neuroticism"`` for base/N+/N-, ``"all"`` for base + the 10 OCEAN
            directions + the control adapter.

    Returns:
        Variants in evaluation order, base first.
    """
    from src_dev.common.lora_catalogue import GEMMA_OCEAN_REGISTRIES, HF_REPO

    registry = GEMMA_OCEAN_REGISTRIES["gemma-3-27b-it"]
    if scope == "neuroticism":
        slugs = ["n_plus", "n_minus"]
    elif scope == "all":
        slugs = list(registry)
    else:
        raise ValueError(f"unknown scope: {scope!r}")

    variants = [Variant(model_name="gemma3_27b_base", served_name=BASE_MODEL)]
    for slug in slugs:
        trait = registry[slug]
        variants.append(
            Variant(
                model_name=f"gemma3_27b_{slug}",
                served_name=slug,
                adapter_ref=f"{HF_REPO}::{trait.adapter_path_in_repo}",
            )
        )
    if scope == "all":
        variants.append(
            Variant(
                model_name="gemma3_27b_control",
                served_name="control",
                adapter_ref=f"{HF_REPO}::{CONTROL_PATH_IN_REPO}",
            )
        )
    return variants


def resolve_variant_adapters(variants: list[Variant]) -> dict[str, str]:
    """Resolve every variant's adapter to a local dir. Returns served_name -> path."""
    from src_dev.inference.providers.vllm import _resolve_vllm_adapter_path

    resolved: dict[str, str] = {}
    for v in variants:
        if v.adapter_ref is None:
            continue
        logger.info(f"Resolving {v.served_name}: {v.adapter_ref}")
        resolved[v.served_name] = _resolve_vllm_adapter_path(v.adapter_ref)
        logger.info(f"  -> {resolved[v.served_name]}")
    return resolved


def _build_vllm_serve_command(
    lora_modules: dict[str, str],
    port: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_lora_rank: int,
    enforce_eager: bool = True,
) -> list[str]:
    """Build the vLLM serve command.

    Args:
        lora_modules: Mapping of served adapter name -> local adapter directory.
        port: Server port.
        gpu_memory_utilization: GPU memory utilization fraction.
        max_model_len: Maximum model context length.
        max_lora_rank: Maximum LoRA rank.
        enforce_eager: Disable inductor compilation + CUDA graphs (required for
            correct gemma-3 output on this vLLM/GPU stack; see config).

    Returns:
        Command as a list of strings suitable for subprocess.Popen.
    """
    cmd = [
        "vllm",
        "serve",
        BASE_MODEL,
        "--enable-lora",
        "--lora-modules",
        *[f"{name}={path}" for name, path in lora_modules.items()],
        "--max-lora-rank",
        str(max_lora_rank),
        # Variants run sequentially, so only one adapter is ever active in a
        # batch; keep them all in the CPU cache so switching variants does not
        # re-read from disk.
        "--max-loras",
        "1",
        "--max-cpu-loras",
        str(max(1, len(lora_modules))),
        "--dtype",
        "bfloat16",
        # vLLM (>=0.11) parses this as JSON; older key=value form is rejected.
        "--limit-mm-per-prompt",
        '{"image": 0}',
    ]
    if enforce_eager:
        cmd.append("--enforce-eager")
    cmd += [
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--max-model-len",
        str(max_model_len),
        "--port",
        str(port),
        "--host",
        "127.0.0.1",
    ]
    return cmd


def _wait_for_vllm_health(port: int, timeout_seconds: float = 1200) -> bool:
    """Poll vLLM server health endpoint until ready or timeout.

    Args:
        port: Server port.
        timeout_seconds: Timeout in seconds (default 20 minutes).

    Returns:
        True if server became healthy, False on timeout.
    """
    import requests

    health_url = f"http://127.0.0.1:{port}/health"
    start_time = time.time()
    poll_interval = 5

    while time.time() - start_time < timeout_seconds:
        try:
            resp = requests.get(health_url, timeout=5)
            if resp.status_code == 200:
                logger.info("vLLM server is healthy")
                return True
        except Exception:
            pass

        elapsed = time.time() - start_time
        logger.info(f"vLLM health check ({elapsed:.0f}s / {timeout_seconds:.0f}s): not ready yet")
        time.sleep(poll_interval)

    logger.error(f"vLLM server did not become healthy within {timeout_seconds}s")
    return False


def _build_eqbench3_commands(
    variants: list[Variant],
    output_dir: Path,
    port: int,
    judge_model: str,
    iterations: int,
    threads: int,
) -> list[list[str]]:
    """Build the three eqbench3.py subprocess commands (base, N+, N-).

    Args:
        output_dir: Directory for runs.json and elo_results.json.
        port: vLLM server port.
        judge_model: Judge model ID.
        iterations: Number of iterations.
        threads: Number of parallel threads.

    Returns:
        List of three command lists: one per variant (base, N+, N-).
    """
    runs_file = output_dir / "runs.json"
    elo_file = output_dir / "elo_results.json"

    commands = []
    for variant in variants:
        model_name, served_model_id = variant.model_name, variant.served_name
        cmd = [
            "python",
            "eqbench3.py",
            "--test-model",
            served_model_id,
            "--model-name",
            model_name,
            "--judge-model",
            judge_model,
            "--iterations",
            str(iterations),
            "--ignore-canonical",
            "--threads",
            str(threads),
            "--runs-file",
            str(runs_file),
            "--elo-results-file",
            str(elo_file),
            "--verbosity",
            "INFO",
        ]
        commands.append(cmd)

    return commands


def _build_eqbench3_env(port: int) -> dict[str, str]:
    """Build environment variable overrides for eqbench3.py subprocesses.

    Args:
        port: vLLM server port.

    Returns:
        Dict of env vars to merge into subprocess environment.

    Raises:
        RuntimeError: If OPENROUTER_API_KEY is not set.
    """
    load_dotenv()
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not openrouter_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set; the judge runs via OpenRouter. "
            "Add it to .env before running the sweep."
        )

    return {
        "TEST_API_URL": f"http://127.0.0.1:{port}/v1/chat/completions",
        "TEST_API_KEY": "dummy-vllm-key",
        "JUDGE_API_URL": "https://openrouter.ai/api/v1/chat/completions",
        "JUDGE_API_KEY": openrouter_key,
        # Upstream-supported knobs (see eqbench3 .env.example). At 32-way
        # concurrency the judge can hit transient 429s; the api.py default of 3
        # retries is thin, and a whole sweep's scoring is lost if they exhaust.
        "MAX_RETRIES": "6",
        "RETRY_DELAY": "5",
        "REQUEST_TIMEOUT": "600",
    }


def run_eqbench3_sweep(config: EQBenchSweepConfig) -> int:
    """Execute the full EQBench3 sweep.

    Args:
        config: Sweep configuration.

    Returns:
        Exit code (0 for success, nonzero otherwise).
    """
    # eqbench3.py runs with cwd=vendor_dir (it resolves data/ relative to cwd),
    # so a relative --runs-file/--elo-results-file would land inside the
    # vendored tree instead of here. Resolve to absolute before building the
    # subprocess commands.
    config.output_dir = config.output_dir.resolve()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== EQBench3 Gemma-3-27b Neuroticism Sweep ===")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"vLLM port: {config.port}")
    logger.info(f"Judge model: {config.judge_model}")
    logger.info(f"Iterations: {config.iterations}")

    variants = build_variants(config.scope)
    logger.info(f"variants ({config.scope}): {[v.model_name for v in variants]}")
    lora_modules = resolve_variant_adapters(variants)

    vllm_cmd = _build_vllm_serve_command(
        lora_modules,
        config.port,
        config.gpu_memory_utilization,
        config.max_model_len,
        config.max_lora_rank,
        config.enforce_eager,
    )

    eqbench3_cmds = _build_eqbench3_commands(
        variants,
        config.output_dir,
        config.port,
        config.judge_model,
        config.iterations,
        config.threads,
    )

    eqbench3_env = _build_eqbench3_env(config.port)

    logger.info("\n=== vLLM Serve Command ===")
    logger.info(" ".join(vllm_cmd))

    logger.info("\n=== EQBench3 Commands (with env overrides) ===")
    for i, cmd in enumerate(eqbench3_cmds, 1):
        logger.info(f"\nVariant {i}: {' '.join(cmd)}")
    logger.info(f"\nEnv overrides:")
    for k, v in eqbench3_env.items():
        if k == "JUDGE_API_KEY":
            logger.info(f"  {k}=<redacted>")
        else:
            logger.info(f"  {k}={v}")

    vllm_proc = None
    vllm_log = None
    try:
        logger.info("\n=== Starting vLLM Server ===")
        # vLLM is extremely verbose; piping to PIPE without draining fills the OS
        # pipe buffer and deadlocks the server mid-run. Redirect to a log file so
        # the pipes never back up (and we can debug a failed launch).
        vllm_log_path = config.output_dir / "vllm_server.log"
        vllm_log = open(vllm_log_path, "w")
        logger.info(f"vLLM server logs → {vllm_log_path}")
        vllm_proc = subprocess.Popen(
            vllm_cmd,
            stdout=vllm_log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        logger.info(f"vLLM server started (PID {vllm_proc.pid})")

        if not _wait_for_vllm_health(config.port):
            logger.error("vLLM server failed to become healthy")
            return 1

        logger.info("\n=== Running EQBench3 Benchmarks ===")

        vendor_dir = Path(__file__).parent / "vendor" / "eqbench3"

        for i, cmd in enumerate(eqbench3_cmds, 1):
            logger.info(f"\n--- Variant {i}/{len(eqbench3_cmds)} ---")
            logger.info(f"Command: {' '.join(cmd)}")

            subprocess_env = os.environ.copy()
            subprocess_env.update(eqbench3_env)

            result = subprocess.run(
                cmd,
                cwd=str(vendor_dir),
                env=subprocess_env,
                timeout=config.subprocess_timeout_seconds,
            )

            if result.returncode != 0:
                logger.error(f"Variant {i} failed with exit code {result.returncode}")
                return 1

            logger.info(f"Variant {i} completed successfully")

        logger.info("\n=== All benchmarks completed ===")
        return 0

    except subprocess.TimeoutExpired:
        logger.error("Benchmark subprocess timed out")
        return 1

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1

    finally:
        if vllm_proc is not None:
            logger.info("\n=== Terminating vLLM Server ===")
            vllm_proc.terminate()
            try:
                vllm_proc.wait(timeout=10)
                logger.info("vLLM server terminated gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("vLLM server did not terminate; killing forcefully")
                vllm_proc.kill()
                vllm_proc.wait()
        if vllm_log is not None:
            vllm_log.close()


def run_eqbench3_sweep_dry_run(config: EQBenchSweepConfig) -> int:
    """Print commands without executing (dry-run mode).

    Args:
        config: Sweep configuration.

    Returns:
        Exit code (always 0).
    """
    config.output_dir = config.output_dir.resolve()

    logger.info("=== DRY-RUN: EQBench3 Gemma-3-27b Neuroticism Sweep ===")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"vLLM port: {config.port}")

    logger.info("\n=== Resolving Adapters ===")
    variants = build_variants(config.scope)
    logger.info(f"variants ({config.scope}): {[v.model_name for v in variants]}")
    lora_modules = resolve_variant_adapters(variants)

    vllm_cmd = _build_vllm_serve_command(
        lora_modules,
        config.port,
        config.gpu_memory_utilization,
        config.max_model_len,
        config.max_lora_rank,
        config.enforce_eager,
    )

    eqbench3_cmds = _build_eqbench3_commands(
        variants,
        config.output_dir,
        config.port,
        config.judge_model,
        config.iterations,
        config.threads,
    )

    eqbench3_env = _build_eqbench3_env(config.port)

    logger.info("\n=== vLLM Serve Command (DRY-RUN) ===")
    logger.info(" ".join(vllm_cmd))

    logger.info("\n=== EQBench3 Commands (DRY-RUN) ===")
    for i, cmd in enumerate(eqbench3_cmds, 1):
        logger.info(f"\nVariant {i}:")
        logger.info(f"  {' '.join(cmd)}")

    logger.info(f"\nEnv overrides:")
    for k, v in eqbench3_env.items():
        if k == "JUDGE_API_KEY":
            logger.info(f"  {k}=<redacted>")
        else:
            logger.info(f"  {k}={v}")

    logger.info("\n=== Dry-run complete (no commands executed) ===")
    return 0


def main() -> int:
    """Parse arguments and run sweep or dry-run."""
    parser = argparse.ArgumentParser(
        description="EQBench3 sweep driver for gemma-3-27b-it with neuroticism adapters."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port (default 8000).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=2,
        help="Number of eqbench3 iterations per variant (default 2).",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="Number of eqbench3 parallel threads / concurrent vLLM requests (default 32).",
    )
    parser.add_argument(
        "--judge-model",
        default="anthropic/claude-opus-4-6",
        help="Judge model ID via OpenRouter (default anthropic/claude-opus-4-6).",
    )
    parser.add_argument(
        "--scope",
        choices=["neuroticism", "all"],
        default="neuroticism",
        help="Variants to evaluate: neuroticism (base/N+/N-) or all "
             "(base + 10 OCEAN directions + control). Default neuroticism.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scratch/evals/eqbench3/gemma27b_n_sweep"),
        help="Output directory for runs and ELO results.",
    )

    args = parser.parse_args()

    config = EQBenchSweepConfig(
        judge_model=args.judge_model,
        scope=args.scope,
        iterations=args.iterations,
        threads=args.threads,
        port=args.port,
        output_dir=args.output_dir,
    )

    if args.dry_run:
        return run_eqbench3_sweep_dry_run(config)
    else:
        return run_eqbench3_sweep(config)


if __name__ == "__main__":
    sys.exit(main())
