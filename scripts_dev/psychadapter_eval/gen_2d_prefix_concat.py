#!/usr/bin/env python3
"""Generate 2D prefix-concatenated text from PsychAdapter.

Concatenates KV prefixes from two Big Five latent vectors:
  - [P1_K; P2_K; X_K] and [P1_V; P2_V; X_V]
  - Composition happens in sequence space through softmax

Usage (in isolated venv per generate_psychadapter.py):
    python3 scripts_dev/psychadapter_eval/gen_2d_prefix_concat.py \\
        --trait-a openness --trait-b neuroticism --num-combos 25

This runs on GPU (or MPS/CPU) and outputs:
    scratch/psychadapter_eval/2d_prefix_concat_<traits>_raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

# Suppress torch.load() weights_only warning (trusted HF checkpoint)
_orig_torch_load = torch.load
def _trusting_torch_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_torch_load(*a, **k)
torch.load = _trusting_torch_load

sys.path.insert(0, str(Path(__file__).parent / "vendor"))
from psychadapter import PsychAdapter, top_k_top_p_filtering  # noqa: E402
from peft import PeftModel  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

ASSET_DIR = Path(os.environ.get("PA_ASSETS", "/tmp/psychadapter_assets"))
OUT_PATH = Path(
    os.environ.get(
        "PA_OUT",
        str(Path(__file__).resolve().parents[2] / "scratch/psychadapter_eval/2d_prefix_concat_raw.jsonl"),
    )
)

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
MEANS = np.zeros(5, dtype=np.float32)
STDS = np.ones(5, dtype=np.float32)

# Grid: traits at these std positions (e.g., -1.5, -0.75, 0, 0.75, 1.5, 3.0)
POSITIONS = [-3.0, -1.5, 0.0, 1.5, 3.0]  # Can customize

SEED_PROMPTS = ["I", "Today", "When I think about other people,", "My ideal weekend is"]
GEN_NUM = int(os.environ.get("PA_GEN_NUM", "2"))  # samples per combo
GENERATE_LENGTH = int(os.environ.get("PA_GEN_LEN", "64"))
TEMPERATURE = float(os.environ.get("PA_TEMP", "0.7"))
TOP_K = int(os.environ.get("PA_TOP_K", "10"))
TOP_P = float(os.environ.get("PA_TOP_P", "0.9"))


def _pick_device() -> torch.device:
    """Pick CUDA/MPS/CPU."""
    override = os.environ.get("PA_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def assemble_model_dir() -> tuple[Path, Path, Path]:
    """Assemble PsychAdapter directory structure."""
    work = ASSET_DIR / "_assembled"
    base_model = work / "base_model"
    base_model.mkdir(parents=True, exist_ok=True)

    decoder_src = ASSET_DIR / "decoder"
    tok_src = ASSET_DIR / "big5_model" / "base_model" / "tokenizer"
    tm_src = ASSET_DIR / "big5_model" / "base_model" / "transform_matrix"
    targs_src = ASSET_DIR / "big5_model" / "base_model" / "training_args.bin"
    ckpt_src = ASSET_DIR / "big5_model" / "checkpoint-30000"

    for name, src in {"decoder": decoder_src, "tokenizer": tok_src, "transform_matrix": tm_src}.items():
        if not src.exists():
            raise FileNotFoundError(f"Missing PsychAdapter asset: {src}")
        link = base_model / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src.resolve())

    targs_link = base_model / "training_args.bin"
    if targs_link.exists() or targs_link.is_symlink():
        targs_link.unlink()
    if targs_src.exists():
        targs_link.symlink_to(targs_src.resolve())

    if not ckpt_src.exists():
        raise FileNotFoundError(f"Missing LoRA checkpoint: {ckpt_src}")

    return base_model, ckpt_src, decoder_src


def load_model(device: torch.device):
    """Load PsychAdapter model."""
    base_model_dir, ckpt_dir, decoder_dir = assemble_model_dir()
    args = SimpleNamespace(
        model_name_or_path=str(decoder_dir),
        do_lower_case=True,
        generate_length=GENERATE_LENGTH,
        top_k=TOP_K,
        top_p=TOP_P,
        temperature=TEMPERATURE,
        seed=SEED,
        n_gpu=1 if device.type == "cuda" else 0,
    )
    model = PsychAdapter(str(decoder_dir), latent_size=5)
    model.from_checkpoint(args, str(base_model_dir))
    model = PeftModel.from_pretrained(model, str(ckpt_dir))
    model.to(device)
    model.eval()
    return model, args


def latent_for(dim: int | None, value: float) -> np.ndarray:
    """Create latent vector with one dimension at specified std value."""
    emb = MEANS.copy()
    if dim is not None:
        emb[dim] = MEANS[dim] + value * STDS[dim]
    return emb


def concatenate_kv_prefixes(
    model: PsychAdapter,
    latent_a: np.ndarray,
    latent_b: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Concatenate KV prefixes from two latent vectors.

    Args:
        model: PsychAdapter instance
        latent_a: 5D latent vector for trait A
        latent_b: 5D latent vector for trait B
        device: Torch device

    Returns:
        Concatenated past_key_values tensor ready for decoder.forward()
    """
    batch_size = 1

    # Compute KV for latent_a
    latent_a_tensor = torch.tensor(latent_a, device=device).float().unsqueeze(0)
    kv_a = model.transform_matrix(latent_a_tensor)
    kv_a = kv_a.reshape([batch_size, model.model_config.num_hidden_layers, 2, model.model_config.num_key_value_heads, 1, model.model_config.head_dim])
    kv_a = torch.transpose(kv_a, 0, 1).contiguous()
    kv_a = torch.transpose(kv_a, 1, 2).contiguous()

    # Compute KV for latent_b
    latent_b_tensor = torch.tensor(latent_b, device=device).float().unsqueeze(0)
    kv_b = model.transform_matrix(latent_b_tensor)
    kv_b = kv_b.reshape([batch_size, model.model_config.num_hidden_layers, 2, model.model_config.num_key_value_heads, 1, model.model_config.head_dim])
    kv_b = torch.transpose(kv_b, 0, 1).contiguous()
    kv_b = torch.transpose(kv_b, 1, 2).contiguous()

    # Concatenate along sequence dimension (index 4, the "1" becomes "2")
    # kv_a: [num_layers, 2, 1, num_kv_heads, 1, head_dim]
    # kv_b: [num_layers, 2, 1, num_kv_heads, 1, head_dim]
    # concat: [num_layers, 2, 1, num_kv_heads, 2, head_dim]
    concatenated = torch.cat([kv_a, kv_b], dim=4)

    return concatenated


def generate_with_concat_prefixes(
    model: PsychAdapter,
    args: SimpleNamespace,
    device: torch.device,
    trait_a_idx: int,
    pos_a: float,
    trait_b_idx: int,
    pos_b: float,
    prompt: str,
) -> str:
    """Generate text with concatenated KV prefixes (static prefix, no KV cache update).

    Follows the original PsychAdapter.inference() pattern:
    - Compute concatenated KV prefixes once
    - Pass them to decoder with growing input_ids (NOT using KV cache updates)

    Args:
        model: PsychAdapter
        args: Generation args
        device: Torch device
        trait_a_idx: Index in TRAITS (0-4)
        pos_a: Std position for trait A
        trait_b_idx: Index in TRAITS
        pos_b: Std position for trait B
        prompt: Seed prompt

    Returns:
        Generated text (cleaned)
    """
    latent_a = latent_for(trait_a_idx, pos_a)
    latent_b = latent_for(trait_b_idx, pos_b)

    # Get concatenated KV prefixes (static, never updated)
    past = concatenate_kv_prefixes(model, latent_a, latent_b, device)

    # Prepare input
    prompting_text_tokens = "<bos>" + prompt.strip()
    prompting_text_encoded = model.tokenizer.encode(prompting_text_tokens, add_special_tokens=False)
    decoder_input_ids = torch.tensor(prompting_text_encoded, device=device).long().unsqueeze(0)

    # Generate — reprocess full history each step (like original PsychAdapter, no KV cache)
    generated = decoder_input_ids
    prefix_len = 2  # Concatenated prefix has 2 tokens (one from each trait)
    for _ in range(args.generate_length):
        # Attention mask: input tokens + concatenated prefix tokens
        decoder_attention_mask = torch.tensor([[1] * (generated.shape[1] + prefix_len)] * generated.shape[0], device=device)

        with torch.no_grad():
            decoder_lm_logits, _, _, _ = model.decoder(
                input_ids=generated,
                past_key_values=past,
                attention_mask=decoder_attention_mask,
                return_dict=False,
            )

        # Sample next token from last output
        decoder_lm_logits = decoder_lm_logits[:, -1, :]

        # Top-k/top-p filtering
        filtered_logits = top_k_top_p_filtering(decoder_lm_logits, top_k=args.top_k, top_p=args.top_p)
        if args.temperature == 0:
            next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
        else:
            next_token = torch.multinomial(torch.softmax(filtered_logits / args.temperature, dim=-1), num_samples=1)

        generated = torch.cat((generated, next_token), dim=1)

    # Decode & clean
    text = model.tokenizer.decode(generated[0].tolist(), clean_up_tokenization_spaces=True)
    eos_idx = text.find("<eos>")
    if eos_idx > 0:
        text = text[:eos_idx]
    text = text.replace("<bos>", "").strip()

    return text


def main():
    parser = argparse.ArgumentParser(
        description="Generate 2D prefix-concatenated text from PsychAdapter"
    )
    parser.add_argument(
        "--trait-a",
        type=str,
        default="openness",
        help="First trait name",
    )
    parser.add_argument(
        "--trait-b",
        type=str,
        default="neuroticism",
        help="Second trait name",
    )
    parser.add_argument(
        "--num-combos",
        type=int,
        default=25,
        help="Total combos to generate (will distribute over grid)",
    )
    args_cli = parser.parse_args()

    trait_a = args_cli.trait_a.lower()
    trait_b = args_cli.trait_b.lower()

    if trait_a not in TRAITS or trait_b not in TRAITS:
        raise ValueError(f"Unknown trait. Available: {TRAITS}")

    trait_a_idx = TRAITS.index(trait_a)
    trait_b_idx = TRAITS.index(trait_b)

    device = _pick_device()
    print(f"Device: {device} | Traits: {trait_a} × {trait_b}")
    print(f"Positions: {POSITIONS} | Samples per combo: {GEN_NUM} | Prompts: {len(SEED_PROMPTS)}")
    print(f"Total: {len(POSITIONS)}² × {GEN_NUM} × {len(SEED_PROMPTS)} = {len(POSITIONS)**2 * GEN_NUM * len(SEED_PROMPTS)} generations\n")

    model, pa_args = load_model(device)

    rows: list[dict[str, Any]] = []
    combo_count = 0

    for pos_a in POSITIONS:
        for pos_b in POSITIONS:
            combo_count += 1
            print(f"[{combo_count}/{len(POSITIONS)**2}] {trait_a}@{pos_a:+.1f} + {trait_b}@{pos_b:+.1f}")

            for prompt in SEED_PROMPTS:
                for sample_idx in range(GEN_NUM):
                    text = generate_with_concat_prefixes(
                        model, pa_args, device,
                        trait_a_idx, pos_a,
                        trait_b_idx, pos_b,
                        prompt,
                    )

                    rows.append(
                        {
                            "response": text,
                            "question": prompt,
                            "id": f"2d-concat-{trait_a[:1]}{trait_b[:1]}-{combo_count}-{SEED_PROMPTS.index(prompt)}-{sample_idx}",
                            "trait_a": trait_a,
                            "trait_a_pos": float(pos_a),
                            "trait_b": trait_b,
                            "trait_b_pos": float(pos_b),
                            "prompt": prompt,
                            "generation": text,
                            "latent_a": latent_for(trait_a_idx, pos_a).tolist(),
                            "latent_b": latent_for(trait_b_idx, pos_b).tolist(),
                        }
                    )

                    print(f"  {prompt!r} -> {text[:70]!r}", flush=True)

    # Save raw generations
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print(f"\n✅ Saved {len(rows)} generations → {OUT_PATH}")
    print(f"\nNext: Run in repo env to score with OCEAN judge + MMLU:")
    print(f"  uv run python scripts_dev/psychadapter_eval/eval_2d_prefix_concat.py \\")
    print(f"      --input {OUT_PATH} --output {OUT_PATH.parent}/2d_prefix_concat_scored.jsonl")


if __name__ == "__main__":
    main()
