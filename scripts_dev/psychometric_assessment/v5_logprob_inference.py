"""v5 Likert questionnaire administered with logprob scoring.

Standalone re-administration of the v5 questionnaire (100 Likert items) on the
cached "B" rollouts, using single-token logprob scoring instead of direct-
generation Likert digits. This disentangles the modality (direct-gen vs
logprob) from the question-type (Likert content vs trait-pole ABCD) in the
v5 vs trait_ocean_v1 comparison.

Flow per item per persona:
    1. Build the same chat-template prompt the main pipeline uses (via
       ``build_questionnaire_messages``).
    2. Generate a single token at temperature=0 with ``top_logprobs=20``.
    3. Extract the probability mass on each Likert digit {1, 2, 3, 4, 5}.
    4. Compute the expected Likert score:
           EV = Σ d · P(d) / Σ P(d)   for d ∈ {1..5}
       (renormalised over the digits actually present in the top-k).
    5. Apply reverse-keying (EV → 6 − EV when item is reverse-scored), so the
       resulting matrix is on the same [1, 5] scale as v5 direct-gen and
       directly comparable in downstream FA.

Output artifacts land under a run-id that matches what the main pipeline
would produce with ``use_logprobs=True`` on v5:

    scratch/psychometric_fa/<run_id>/questionnaire/
        response_matrix.npy        (n_personas, 100) float in [1, 5]
        metadata.jsonl             per-persona provenance
        items.json                 column defs (matches v5 direct-gen)
        raw_responses.jsonl        per-(persona, item): probs dict + EV
        encoding_version.json

so ``compare_models.py`` / ``k_sweep_factor_congruence.py`` load it
transparently once ``RUNS`` is updated.

CLI examples:
    # Dry run on 10 personas to verify outputs
    uv run python -m scripts_dev.psychometric_assessment.v5_logprob_inference \\
        --model qwen2.5 --max-personas 10 --no-upload

    # Full run, pinning to GPU 1
    CUDA_VISIBLE_DEVICES=1 uv run python -m \\
        scripts_dev.psychometric_assessment.v5_logprob_inference --model qwen2.5

GPU parallelism: run two separate processes (one per model), each pinned to
its own GPU via CUDA_VISIBLE_DEVICES. There is no intra-model split — a
single vLLM engine already uses the full GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from src_dev.common.config import GenerationConfig
from src_dev.common.conversation_runtime import chunked
from src_dev.datasets import (
    find_consecutive_assistant_turn_sample_ids,
    load_samples,
    materialize_canonical_samples,
)
from src_dev.inference import InferenceConfig
from src_dev.inference.config import (
    OpenRouterProviderConfig,
    RetryConfig,
    VllmProviderConfig,
)
from src_dev.inference.providers import get_provider
from src_dev.psychometric.fa_run_catalogue import FA_RUN_REGISTRY
from src_dev.psychometric.hf_paths import hf_runs_path
from src_dev.psychometric.item_prompts import (
    build_item_prompt,
    build_questionnaire_messages,
    retry_message,
)
from src_dev.psychometric.questionnaire_inference import _filter_by_context_budget
from src_dev.psychometric.questionnaire_io import load_questionnaire
from src_dev.psychometric.response_encoding import RESPONSE_MATRIX_ENCODING_VERSION
from src_dev.unsupervised_runs.io import hydrate_dataset_subtree
from src_dev.utils.hf_hub import (
    check_exists_in_dataset_repo,
    login_from_env,
    upload_folder_to_dataset_repo,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class ModelSpec:
    label: str                           # short tag used in filenames/argparse
    hf_model: str                        # HuggingFace model id for vLLM
    max_context_tokens: int | None       # drop rollouts exceeding this budget
    qm_tag: str | None                   # run-id suffix to distinguish from rollout model
    gpu_mem_util: float = 0.90


MODELS: dict[str, ModelSpec] = {
    "llama": ModelSpec(
        label="llama-3.1-8b",
        hf_model="meta-llama/Llama-3.1-8B-Instruct",
        max_context_tokens=None,  # 128k native; no filtering needed
        qm_tag=None,              # questionnaire model == rollout model
    ),
    "qwen2.5": ModelSpec(
        label="qwen2.5-7b",
        hf_model="Qwen/Qwen2.5-7B-Instruct",
        max_context_tokens=32768,
        qm_tag="qwen257binstruct",
    ),
}

ROLLOUT_RUN_ID = FA_RUN_REGISTRY["llama_base_rollouts"].run_id
NUM_CONVERSATION_TURNS = 15
SCRATCH_ROOT = Path("scratch/psychometric_fa")
HF_REPO_ID = "persona-shattering-lasr/psychometric-fa-runs"

QUESTIONNAIRE_PATH = Path("datasets/psychometric_questionnaires/psychometric_questionnaire_v5.json")
QUESTIONNAIRE_VERSION = "v5"
FA_BLOCKS = ("likert",)
LIKERT_PHRASING = "direct"

TOP_LOGPROBS = 20
LOGPROB_TEMPERATURE = 1.0
VLLM_PERSONAS_PER_BATCH = 8
# 0.9 is safer than 0.95 when the chosen GPU has any residue from a prior
# process (vLLM refuses to start if free memory < utilization*total). The
# CLI exposes --gpu-mem-util to tune per-situation; set higher when the GPU
# is fully idle.
VLLM_GPU_MEMORY_UTILIZATION = 0.90
CONTEXT_BUFFER_TOKENS = 1024


# ═════════════════════════════════════════════════════════════════════════════
# DIGIT-LOGPROB PARSING
# ═════════════════════════════════════════════════════════════════════════════


# Tokenizer-variant digit sets. Different tokenizers encode a bare digit
# as "1", "▁1", "Ġ1", " 1" — check them all.
DIGIT_VARIANTS: dict[str, set[str]] = {
    d: {d, f"▁{d}", f"Ġ{d}", f" {d}"} for d in "12345"
}


def extract_likert_digit_probs(
    top_logprobs: dict[str, float],
) -> tuple[dict[str, float], float]:
    """Return (probs: digit->prob, choice_mass) over digits 1..5.

    Probs are softmax-renormalised over whichever digits appear in the top-k.
    choice_mass is the total exp(logprob) mass on digit tokens out of the
    full vocabulary — a quality metric (high = model is confident about
    answering with a digit at all).
    """
    found: dict[str, float] = {}
    for digit, variants in DIGIT_VARIANTS.items():
        for tok, lp in top_logprobs.items():
            if tok in variants:
                found[digit] = float(lp)
                break
    if not found:
        return {}, 0.0
    choice_mass = sum(math.exp(lp) for lp in found.values())
    max_lp = max(found.values())
    exp_vals = {k: math.exp(v - max_lp) for k, v in found.items()}
    total = sum(exp_vals.values())
    probs = {k: v / total for k, v in exp_vals.items()}
    return probs, float(choice_mass)


def expected_likert(probs: dict[str, float]) -> float | None:
    """Σ d · P(d) over d ∈ {1..5}. ``None`` if no digit found."""
    if not probs:
        return None
    return sum(int(d) * p for d, p in probs.items())


# ═════════════════════════════════════════════════════════════════════════════
# MAIN DRIVER
# ═════════════════════════════════════════════════════════════════════════════


def _build_run_id(model: ModelSpec) -> str:
    base = (
        f"questionnaire-{ROLLOUT_RUN_ID}-q_{QUESTIONNAIRE_VERSION}-likert-"
        f"{LIKERT_PHRASING}-lp{TOP_LOGPROBS}"
    )
    if model.qm_tag:
        return f"{base}-qm_{model.qm_tag}"
    return base


def _hydrate_rollouts_if_needed(rollout_dir: Path) -> None:
    manifest = rollout_dir / "manifest.json"
    if manifest.exists():
        return
    print(f"[Hydrate] Rollout dir missing locally; pulling from HF → {rollout_dir}")
    hydrate_dataset_subtree(
        repo_id=HF_REPO_ID,
        path_in_repo=hf_runs_path(ROLLOUT_RUN_ID),
        local_dir=rollout_dir,
        required=True,
    )


async def run_v5_logprob(model: ModelSpec, max_personas: int | None, upload: bool) -> None:
    run_id = _build_run_id(model)
    run_dir = SCRATCH_ROOT / run_id / "questionnaire"
    rollout_dir = SCRATCH_ROOT / ROLLOUT_RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"v5 logprob run for {model.label}  →  {run_id}")
    print("=" * 70)

    # ── Hydrate rollouts + load samples ─────────────────────────────────────
    _hydrate_rollouts_if_needed(rollout_dir)
    materialize_canonical_samples(rollout_dir)
    samples = load_samples(rollout_dir)
    print(f"[Load] {len(samples)} rollout samples")

    completed = [
        s for s in samples
        if sum(1 for m in s.messages if m.role == "assistant") >= NUM_CONVERSATION_TURNS
    ]
    print(f"[Filter] {len(completed)} samples with >= {NUM_CONVERSATION_TURNS} assistant turns")

    bad_ids = find_consecutive_assistant_turn_sample_ids(rollout_dir)
    if bad_ids:
        n_before = len(completed)
        completed = [s for s in completed if s.sample_id not in bad_ids]
        print(f"[Filter] excluded {n_before - len(completed)} resume-bug samples")

    items, column_defs = load_questionnaire(
        QUESTIONNAIRE_PATH, fa_blocks=FA_BLOCKS, fc_pair_sign_alignment=True,
    )
    print(f"[Load] {len(items)} items, {len(column_defs)} matrix columns")

    if model.max_context_tokens is not None:
        completed = _filter_by_context_budget(
            completed, items,
            model=model.hf_model,
            max_context_tokens=model.max_context_tokens,
            max_new_tokens=1,
            buffer_tokens=CONTEXT_BUFFER_TOKENS,
            likert_phrasing=LIKERT_PHRASING,
        )

    if max_personas is not None and max_personas < len(completed):
        print(f"[Dry-run] Truncating to first {max_personas} personas")
        completed = completed[:max_personas]

    K = len(completed)
    N_items = len(items)
    print(f"[Go] K={K} personas × N={N_items} items = {K*N_items:,} calls")

    # Build persona conversations once.
    conversations = [[{"role": m.role, "content": m.content} for m in s.messages]
                     for s in completed]
    metadata = [
        {
            "sample_id": s.sample_id,
            "input_group_id": s.input_group_id,
            "response_index": s.response_index,
            "num_messages": len(s.messages),
        }
        for s in completed
    ]

    # Reverse-keying lookup (item_id -> bool).
    likert_reverse = {it["id"]: bool(it.get("reverse_keyed", False)) for it in items}
    # Build col index lookup (item_id -> col_idx). v5 is one-column-per-item.
    col_idx_by_item_id = {col["item_id"]: i for i, col in enumerate(column_defs)}

    # ── Estimate max_model_len from actual data ────────────────────────────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model.hf_model, use_fast=True)
    item_prompts = [build_item_prompt(it, likert_phrasing=LIKERT_PHRASING) for it in items]
    longest_item = max(item_prompts, key=len)
    longest_conv = max(conversations, key=lambda c: sum(len(m["content"]) for m in c))
    msgs = list(longest_conv) + [{"role": "user", "content": longest_item}]
    max_in = len(tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True))
    longest_retry = max((retry_message(it) for it in items), key=len)
    retry_overhead = len(tokenizer.encode(longest_retry)) + 1 + 20
    raw_len = max_in + 1 + retry_overhead + 256
    max_model_len = ((raw_len + 63) // 64) * 64
    if model.max_context_tokens and max_model_len > model.max_context_tokens:
        print(f"[Clamp] max_model_len {max_model_len} → {model.max_context_tokens}")
        max_model_len = model.max_context_tokens
    print(f"[vLLM] max_model_len = {max_model_len}")

    # ── Provider ────────────────────────────────────────────────────────────
    inf_cfg = InferenceConfig(
        model=model.hf_model,
        provider="vllm",
        generation=GenerationConfig(max_new_tokens=1, temperature=0.0, do_sample=False),
        max_concurrent=32,
        timeout=60,
        retry=RetryConfig(max_retries=3, backoff_factor=2.0),
        continue_on_error=True,
        log_failures=True,
        openrouter=OpenRouterProviderConfig(provider_routing={}),
        vllm=VllmProviderConfig(
            gpu_memory_utilization=model.gpu_mem_util,
            max_model_len=max_model_len,
            tensor_parallel_size=1,
        ),
    )
    provider = get_provider("vllm", inf_cfg)

    # ── Inference loop ──────────────────────────────────────────────────────
    response_matrix = np.full((K, N_items), np.nan)
    raw_log_path = run_dir / "raw_responses.jsonl"
    n_done = 0
    n_fail = 0
    batch_size = VLLM_PERSONAS_PER_BATCH

    with open(raw_log_path, "w", encoding="utf-8") as log_fh:
        persona_batches = list(chunked(list(range(K)), batch_size))
        for batch_idx, persona_batch in enumerate(persona_batches, start=1):
            prompts = []
            entries = []  # (k, item_idx, item)
            for k in persona_batch:
                for item_idx, item in enumerate(items):
                    msgs = build_questionnaire_messages(
                        conversations[k], item,
                        reset_mode="none",
                        soft_reset_system_prompt="",
                        likert_phrasing=LIKERT_PHRASING,
                    )
                    prompts.append(msgs)
                    entries.append((k, item_idx, item))

            if not prompts:
                continue
            lp_outputs = await provider.generate_batch_logprobs_async(
                prompts,
                max_tokens=1,
                top_logprobs=TOP_LOGPROBS,
                temperature=LOGPROB_TEMPERATURE,
            )

            for (k, item_idx, item), lp_out in zip(entries, lp_outputs):
                per_token = lp_out.get("logprobs_per_token") or []
                first_token_lp = per_token[0] if per_token else {}
                probs, choice_mass = extract_likert_digit_probs(first_token_lp)
                ev = expected_likert(probs)
                item_id = item["id"]
                if ev is not None:
                    score = 6.0 - ev if likert_reverse.get(item_id, False) else ev
                    response_matrix[k, col_idx_by_item_id[item_id]] = score
                    n_done += 1
                else:
                    n_fail += 1
                log_fh.write(
                    json.dumps(
                        {
                            "k": k,
                            "item_id": item_id,
                            "item_type": "likert",
                            "probs": {d: round(p, 6) for d, p in probs.items()},
                            "choice_mass": round(choice_mass, 6),
                            "ev_raw": round(ev, 6) if ev is not None else None,
                            "reverse_keyed": likert_reverse.get(item_id, False),
                            "score": round(
                                6.0 - ev if (ev is not None and likert_reverse.get(item_id, False)) else ev,
                                6,
                            ) if ev is not None else None,
                            "raw_text": lp_out.get("text", ""),
                            "scoring_method": "logprob_ev",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            log_fh.flush()
            print(
                f"[Batch {batch_idx}/{len(persona_batches)}] "
                f"{n_done:,} scored / {n_fail} fails / {K*N_items:,} total"
            )

    # ── Save artifacts ──────────────────────────────────────────────────────
    np.save(run_dir / "response_matrix.npy", response_matrix)
    with open(run_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    with open(run_dir / "items.json", "w", encoding="utf-8") as f:
        json.dump(column_defs, f, indent=2, ensure_ascii=False)
    with open(run_dir / "encoding_version.json", "w", encoding="utf-8") as f:
        json.dump({"response_matrix_encoding_version": RESPONSE_MATRIX_ENCODING_VERSION}, f)
    print(
        f"[Write] matrix shape={response_matrix.shape}  "
        f"nan_frac={np.isnan(response_matrix).mean():.4f}  → {run_dir}"
    )

    if upload:
        print("[Upload] Uploading to HF…")
        try:
            login_from_env()
        except RuntimeError as exc:
            print(f"[Upload] Skipped: {exc}")
            return
        hf_path = hf_runs_path(run_id, "questionnaire")
        upload_folder_to_dataset_repo(
            local_dir=run_dir,
            repo_id=HF_REPO_ID,
            path_in_repo=hf_path,
            commit_message=f"sid/psychometric-refactor v5 logprob run for {model.label}",
            ignore_patterns=[],
        )
        print(f"[Upload] → {hf_path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODELS.keys()), required=True,
                        help="Which model to administer v5 with.")
    parser.add_argument("--max-personas", type=int, default=None,
                        help="Cap persona count for a dry run.")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip HF upload (useful for dry runs).")
    parser.add_argument("--gpu-mem-util", type=float, default=None,
                        help="Override vLLM gpu_memory_utilization (0..1).")
    args = parser.parse_args()

    model_spec = MODELS[args.model]
    if args.gpu_mem_util is not None:
        model_spec.gpu_mem_util = args.gpu_mem_util
    asyncio.run(run_v5_logprob(
        model_spec,
        max_personas=args.max_personas,
        upload=not args.no_upload,
    ))


if __name__ == "__main__":
    main()
