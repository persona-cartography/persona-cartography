"""Frustration eval using vLLM batched generation.

Three modes per spec, picked automatically:
  * No ``adapter_path``      → vllm.LLM on the base model directly.
  * ``adapter_path``         → vllm.LLM with native LoRARequest. No bake,
                               no disk pressure (~150 MB adapter only).
  * ``negate=True``          → bake merged model with PEFT scaling=-1
                               (vLLM's LoRARequest cannot scale=-1),
                               then vllm.LLM on the merged dir. The
                               merged dir is deleted after the run.

Each turn is batched across all conversations via ``llm.chat()`` continuous
batching, ~10x faster than ``run_local_adapter.py`` (one conv at a time).

Single-spec usage (CLI):
    uv run python -m scripts_dev.frustration_eval.run_vllm \\
        --adapter-path scratch/adapters/gemma27b_n_minus/.../neuroticism_suppressing_full_vanton4-persona \\
        --num-prompts 100 --run-name gemma27b_n_minus_vllm_n100

Multi-spec usage (4-way driver, sequential bake → run → delete):
    uv run python -m scripts_dev.frustration_eval.run_vllm \\
        --specs-config scripts_dev/frustration_eval/specs/gemma_4way_n100.json
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import gc
import json
import logging
import os
import random
import shutil
import time
from pathlib import Path

# vLLM forks worker subprocesses; if any code in this module touches CUDA before
# the fork (seed setting, torch.cuda.is_available, etc.), the fork fails with
# "Cannot re-initialize CUDA in forked subprocess". Force vLLM to use spawn.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import numpy as np
import torch
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from scripts_dev.frustration_eval.frustration_judge import FrustrationJudge
from scripts_dev.frustration_eval.prompts import IMPOSSIBLE_NUMERIC_3TURN
from scripts_dev.frustration_eval.run_eval import (
    ConversationResult,
    TurnResult,
    compute_summary,
    score_conversation,
)
from src_dev.inference.config import (
    GenerationConfig,
    InferenceConfig,
    VllmProviderConfig,
)
from src_dev.inference.providers import get_provider
from src_dev.persona_metrics.config import JudgeLLMConfig

SEED = 42
HF_CACHE_DIR = Path("~/.cache/huggingface").expanduser()
logger = logging.getLogger(__name__)


def _free_disk_gb() -> float:
    return shutil.disk_usage("/").free / 1e9


def _bake_merged(
    base_model: str,
    adapter_path: Path,
    out_dir: Path,
    *,
    free_hf_cache: bool = False,
) -> Path:
    """Bake adapter@scale=-1 into base weights and save to ``out_dir``.

    Idempotent: skips if ``out_dir`` already has weight shards. Inverts the
    adapter via PEFT's per-module ``scaling`` dict (correct inversion;
    flipping both ``lora_A`` and ``lora_B`` would cancel out).

    Args:
        free_hf_cache: After base+adapter loaded into RAM/GPU, delete the HF
            hub cache (~52 GB for gemma-3-27b) to make disk room for the
            merged save. Cache is regenerated on next ``from_pretrained``.
    """
    if (out_dir / "model.safetensors.index.json").exists() or any(out_dir.glob("*.safetensors")):
        logger.info("merged model already at %s — skipping bake", out_dir)
        return out_dir

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("[bake] loading base %s (bf16) — disk free %.1f GB", base_model, _free_disk_gb())
    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    pm = PeftModel.from_pretrained(base, str(adapter_path), adapter_name="adapter")
    logger.info("[bake] inverting adapter via per-module LoRA scaling=-1.0")
    for module in pm.modules():
        sc = getattr(module, "scaling", None)
        if isinstance(sc, dict):
            for k in sc:
                sc[k] = -1.0
    logger.info("[bake] merge_and_unload ...")
    merged = pm.merge_and_unload()

    if free_hf_cache and HF_CACHE_DIR.exists():
        logger.info("[bake] freeing HF cache to make disk room for save (was %.1f GB free)",
                    _free_disk_gb())
        shutil.rmtree(HF_CACHE_DIR / "hub", ignore_errors=True)
        logger.info("[bake] now %.1f GB free", _free_disk_gb())

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[bake] saving merged model to %s", out_dir)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    # Save the full processor (tokenizer + preprocessor + processor config) so
    # vLLM has everything it needs to load multimodal models like gemma-3.
    # Falls back to tokenizer-only for unimodal models.
    try:
        from transformers import AutoProcessor
        AutoProcessor.from_pretrained(base_model).save_pretrained(str(out_dir))
    except Exception as exc:
        logger.warning("[bake] AutoProcessor unavailable (%s); falling back to tokenizer only", exc)
        AutoTokenizer.from_pretrained(base_model).save_pretrained(str(out_dir))

    del merged, pm, base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("[bake] done — disk free %.1f GB", _free_disk_gb())
    return out_dir


def _init_vllm_provider(
    model_or_dir: str,
    adapter_path: str | None,
    *,
    max_model_len: int,
    gpu_memory_utilization: float,
    temperature: float,
    max_new_tokens: int,
    trust_remote_code: bool = False,
):
    cfg = InferenceConfig(
        model=model_or_dir,
        provider="vllm",
        vllm=VllmProviderConfig(
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            adapter_path=adapter_path,
            trust_remote_code=trust_remote_code,
        ),
        generation=GenerationConfig(
            temperature=temperature, max_new_tokens=max_new_tokens, top_p=1.0,
        ),
    )
    return get_provider("vllm", cfg)


def _run_rollouts(
    provider, category, *,
    num_prompts: int, num_rollouts: int, num_turns: int,
    rng: random.Random, temperature: float, max_new_tokens: int,
) -> list[ConversationResult]:
    """Drive multi-turn eval, batching all conversations at each turn step."""
    convs: list[dict] = []
    for prompt in category.prompts[:num_prompts]:
        for rollout_id in range(num_rollouts):
            convs.append({
                "prompt": prompt, "rollout_id": rollout_id,
                "messages": [{"role": "user", "content": prompt}],
                "turns": [],
            })

    for turn_idx in range(num_turns):
        msgs_batch = [c["messages"] for c in convs]
        logger.info("turn %d/%d: batched gen on %d convs", turn_idx + 1, num_turns, len(msgs_batch))
        responses = provider.generate_batch(
            msgs_batch, temperature=temperature,
            max_new_tokens=max_new_tokens, top_p=1.0,
        )
        for c, resp in zip(convs, responses):
            c["messages"].append({"role": "assistant", "content": resp})
            c["turns"].append(TurnResult(turn_index=turn_idx, response=resp))
            if turn_idx < num_turns - 1:
                rejection = rng.choice(category.rejection_pool)
                c["messages"].append({"role": "user", "content": rejection})

    return [
        ConversationResult(
            category=category.name, prompt=c["prompt"], rollout_id=c["rollout_id"],
            messages=c["messages"], turn_results=c["turns"],
        )
        for c in convs
    ]


def _score_and_save(
    results: list[ConversationResult],
    *,
    category_name: str,
    out_dir: Path,
    judge_provider: str,
    judge_model: str,
) -> dict:
    judge = FrustrationJudge(judge_config=JudgeLLMConfig(
        provider=judge_provider, model=judge_model,
        temperature=0.0, max_tokens=512, max_concurrent=16, max_retries=3,
    ))

    async def score_all() -> None:
        for r in results:
            await score_conversation(judge, r, score_all_turns=True)

    asyncio.run(score_all())

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "results.jsonl"
    with open(raw_path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")
    summary = compute_summary(results, category_name)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_one_spec(
    *,
    name: str,
    base_model: str,
    adapter_path: Path | None,
    negate: bool,
    num_prompts: int,
    num_rollouts: int,
    num_turns: int,
    output_dir: Path,
    judge_provider: str,
    judge_model: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    temperature: float,
    max_new_tokens: int,
    run_name: str | None = None,
    free_hf_cache: bool = False,
    keep_merged: bool = False,
    trust_remote_code: bool = False,
) -> Path:
    """Run one frustration eval spec end-to-end with cleanup.

    Three branches:
      - adapter_path is None         → base-only via vLLM.
      - adapter_path, negate=False   → vLLM + LoRARequest (no bake).
      - adapter_path, negate=True    → bake merged → vLLM on merged → delete.

    Returns the run output directory.
    """
    rname = run_name or name
    out_dir = output_dir / rname / IMPOSSIBLE_NUMERIC_3TURN.name

    merged_dir: Path | None = None
    model_for_vllm = base_model
    adapter_for_vllm: str | None = None

    if adapter_path is not None:
        adapter_path = adapter_path.resolve()
        if negate:
            merged_dir = (PROJECT_ROOT / "scratch/merged" / f"{rname}_negscale").resolve()
            _bake_merged(base_model, adapter_path, merged_dir, free_hf_cache=free_hf_cache)
            model_for_vllm = str(merged_dir)
        else:
            adapter_for_vllm = str(adapter_path)

    rng = random.Random(SEED)
    cat = copy.deepcopy(IMPOSSIBLE_NUMERIC_3TURN)
    provider = None
    try:
        logger.info("[%s] init vLLM (model=%s, adapter=%s)",
                    rname, model_for_vllm, adapter_for_vllm)
        provider = _init_vllm_provider(
            model_for_vllm, adapter_for_vllm,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            temperature=temperature, max_new_tokens=max_new_tokens,
            trust_remote_code=trust_remote_code,
        )

        t0 = time.time()
        results = _run_rollouts(
            provider, cat,
            num_prompts=num_prompts, num_rollouts=num_rollouts,
            num_turns=num_turns, rng=rng,
            temperature=temperature, max_new_tokens=max_new_tokens,
        )
        logger.info("[%s] generation done in %.1fs", rname, time.time() - t0)
    finally:
        if provider is not None:
            del provider
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = _score_and_save(
        results, category_name=cat.name, out_dir=out_dir,
        judge_provider=judge_provider, judge_model=judge_model,
    )
    logger.info("[%s] mean_frustration=%.2f pct_high=%.1f%%",
                rname, summary.get("mean_frustration", 0),
                summary.get("pct_high_frustration", 0))

    if merged_dir is not None and merged_dir.exists() and not keep_merged:
        logger.info("[%s] cleaning up merged dir %s", rname, merged_dir)
        shutil.rmtree(merged_dir, ignore_errors=True)
        logger.info("[%s] disk free %.1f GB", rname, _free_disk_gb())

    return out_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    ap = argparse.ArgumentParser()
    # Multi-spec mode
    ap.add_argument("--specs-config", default=None,
                    help="JSON file with a list of specs (overrides single-spec args). "
                         "Each spec: {name, base_model?, adapter_path?, negate?, num_prompts?, "
                         "num_rollouts?, num_turns?, run_name?}.")
    # Single-spec defaults
    ap.add_argument("--base-model", default="google/gemma-3-27b-it")
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--negate-adapter", action="store_true", default=False)
    ap.add_argument("--num-turns", type=int, default=8)
    ap.add_argument("--num-rollouts", type=int, default=1)
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--output-dir", default="scratch/evals/frustration_eval")
    # Common knobs
    ap.add_argument("--judge-provider", default="openrouter")
    ap.add_argument("--judge-model", default="anthropic/claude-sonnet-4")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--free-hf-cache", action="store_true", default=False,
                    help="Delete HF hub cache after model load during bake to make disk room.")
    ap.add_argument("--keep-merged", action="store_true", default=False,
                    help="Skip merged-dir cleanup after run (debug).")
    ap.add_argument("--trust-remote-code", action="store_true", default=False,
                    help="Pass trust_remote_code=True to vLLM. Required for "
                         "custom architectures like the materialized talkie-1930-13b wrapper.")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    output_dir = Path(args.output_dir)
    common = dict(
        output_dir=output_dir,
        judge_provider=args.judge_provider, judge_model=args.judge_model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        temperature=args.temperature, max_new_tokens=args.max_new_tokens,
        free_hf_cache=args.free_hf_cache, keep_merged=args.keep_merged,
        trust_remote_code=args.trust_remote_code,
    )

    if args.specs_config:
        specs = json.loads(Path(args.specs_config).read_text())
        if not isinstance(specs, list):
            raise SystemExit("specs-config must be a JSON list of spec dicts")
        for spec in specs:
            run_one_spec(
                name=spec["name"],
                base_model=spec.get("base_model", args.base_model),
                adapter_path=Path(spec["adapter_path"]) if spec.get("adapter_path") else None,
                negate=bool(spec.get("negate", False)),
                num_prompts=int(spec.get("num_prompts", args.num_prompts)),
                num_rollouts=int(spec.get("num_rollouts", args.num_rollouts)),
                num_turns=int(spec.get("num_turns", args.num_turns)),
                run_name=spec.get("run_name"),
                **common,
            )
    else:
        if args.run_name is None:
            raise SystemExit("--run-name required in single-spec mode (or use --specs-config)")
        run_one_spec(
            name=args.run_name,
            base_model=args.base_model,
            adapter_path=Path(args.adapter_path) if args.adapter_path else None,
            negate=args.negate_adapter,
            num_prompts=args.num_prompts,
            num_rollouts=args.num_rollouts,
            num_turns=args.num_turns,
            run_name=args.run_name,
            **common,
        )


if __name__ == "__main__":
    main()
