"""Standalone introspection smoke for talkie-1930-13b-it.

Verifies that the OCT introspection stage (vLLM + LoRA + chat template +
self-interaction multi-turn loop) works on the materialized HF wrapper
without first running a real DPO train. Plants a randomly-initialized
LoRA at the path OCT expects and calls the introspection helpers with
very small N (a few prompts, a couple of turns).

Usage:

    OCT_MODEL_PATH=/root/.cache/models \\
        uv run python -m scripts_dev.oct_pipeline.ocean.smoke_talkie_introspection

Outputs land at ``$OCT_DATA_PATH/self_reflection/talkie-1930-13b-it/talkie_smoke.jsonl``
and ``$OCT_DATA_PATH/self_interaction/talkie-1930-13b-it/talkie_smoke.jsonl``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Point OCT at this host's storage before importing the pipeline (constants
# are captured at import).
os.environ.setdefault("OCT_MODEL_PATH", "/root/.cache/models")

def main() -> None:
    # Importing run_oct_pipeline triggers _install_runtime_character_constants
    # and the vLLM compat patches; we don't run main(), just borrow the setup.
    print("[smoke] importing run_oct_pipeline (sets up character.constants) ...", flush=True)
    from scripts_dev.oct_pipeline import run_oct_pipeline  # noqa: F401
    import character.constants as _cc

    print(f"[smoke]   DATA_PATH         = {_cc.DATA_PATH}")
    print(f"[smoke]   MODEL_PATH        = {_cc.MODEL_PATH}")
    print(f"[smoke]   LORA_PATH         = {_cc.LORA_PATH}")
    print(f"[smoke]   CONSTITUTION_PATH = {_cc.CONSTITUTION_PATH}")

    # Install the agreeableness_amplifying_full_vanton4_slim constitution under
    # a smoke-only name so we don't collide with a real run.
    SMOKE_CONST_NAME = "talkie_smoke"
    SLIM_SRC = (
        _REPO_ROOT
        / "scripts_dev/oct_pipeline/ocean/vanton4"
        / "agreeableness_amplifying_full_vanton4_slim.json"
    )
    print(f"[smoke] installing constitution {SMOKE_CONST_NAME} from {SLIM_SRC} ...", flush=True)
    run_oct_pipeline.install_custom_constitution(
        name=SMOKE_CONST_NAME,
        source_path=str(SLIM_SRC),
        expand_questions=False,
        skip_question_validation=True,
    )

    # Plant a randomly-initialized LoRA where OCT expects the DPO adapter.
    family = "talkie"
    lora_dir = Path(f"{_cc.LORA_PATH}/{family}-distillation/{SMOKE_CONST_NAME}")
    print(f"[smoke] planting random LoRA at {lora_dir} ...", flush=True)
    if not lora_dir.exists() or not (lora_dir / "adapter_config.json").exists():
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM

        model_dir = f"{_cc.MODEL_PATH}/talkie-1930-13b-it"
        print(f"[smoke]   loading base model from {model_dir} ...", flush=True)
        base = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16
        )
        cfg = LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=[
                "attn_query", "attn_key", "attn_value", "attn_resid",
                "mlp_gate", "mlp_linear", "mlp_resid",
            ],
        )
        pm = get_peft_model(base, cfg)
        # Tiny non-zero perturbation on B so the adapter isn't a strict no-op.
        with torch.no_grad():
            for name, p in pm.named_parameters():
                if "lora_B" in name:
                    p.data.normal_(0.0, 1e-3)
        pm.save_pretrained(str(lora_dir))
        del base, pm
        torch.cuda.empty_cache()
        print(f"[smoke]   saved LoRA → {lora_dir}", flush=True)

    # The other process on this GPU is using ~47GB — cap vLLM utilization
    # so we don't fight over memory. (The real machine-B run will likely
    # have the GPU to itself; this is smoke-only.)
    run_oct_pipeline._VLLM_GPU_MEMORY_UTILIZATION_OVERRIDE = 0.5

    # Now call the OCT introspection helpers directly with small N.
    print(f"\n[smoke] running run_introspection_generation (N=4, K=2) ...", flush=True)
    sft_path = run_oct_pipeline.run_introspection_generation(
        model="talkie-1930-13b-it",
        constitution=SMOKE_CONST_NAME,
        n_reflection=4,
        n_interaction=4,
        interaction_turns=2,
    )
    print(f"\n[smoke] DONE — merged SFT JSONL at {sft_path}", flush=True)

    # Surface a sample of the generated content for human eyeballing.
    import json
    for tag in ("self_reflection", "self_interaction"):
        p = Path(f"{_cc.DATA_PATH}/{tag}/talkie-1930-13b-it/{SMOKE_CONST_NAME}.jsonl")
        if not p.exists():
            print(f"[smoke] {tag} jsonl missing at {p}")
            continue
        with open(p) as f:
            rows = [json.loads(l) for l in f]
        print(f"\n[smoke] === {tag}: {len(rows)} rows ===")
        for r in rows[:2]:
            for k, v in r.items():
                if isinstance(v, str):
                    print(f"  {k}: {v[:200]!r}{'...' if len(v) > 200 else ''}")
                elif isinstance(v, list) and v and isinstance(v[0], dict):
                    print(f"  {k}: {len(v)} messages; first={v[0]!r}")
                elif isinstance(v, list) and v and isinstance(v[0], str):
                    print(f"  {k}: {len(v)} entries; first[:120]={v[0][:120]!r}")
                else:
                    print(f"  {k}: {v!r}")
            print("  ---")


if __name__ == "__main__":
    main()
