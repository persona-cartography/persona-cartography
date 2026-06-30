"""Stage 1 — generate Big-Five-conditioned text from PsychAdapter ``big5_model``.

PsychAdapter (https://github.com/humanlab/psychadapter) is NOT a mergeable LoRA:
the persona is injected by a learned ``transform_matrix`` that maps a 5-dim Big
Five latent vector into per-layer ``past_key_values`` (KV-prefix conditioning) on
top of a frozen ``google/gemma-2b`` base, plus a small r=8 LoRA. Generation uses a
bespoke loop (``PsychAdapter.inference``), so it cannot pass through our adapter
sweep / Inspect runners.

This script runs PsychAdapter's OWN code (vendored UNMODIFIED in ``vendor/``) to
produce trait-conditioned text, which Stage 2 (``score_ocean.py``, repo env)
scores with our OCEAN judge (``run_persona_metrics``). The bridge is the judge.

Run in an ISOLATED venv — the repo pins transformers>=4.40 but PsychAdapter's
custom KV-prefix path needs the version it was trained with (4.39.2 / peft 0.10):

    python3 -m venv /tmp/pa_venv && source /tmp/pa_venv/bin/activate
    pip install "torch>=2.2" "transformers==4.39.2" "peft==0.10.0" \
                accelerate sentencepiece safetensors numpy huggingface_hub
    # Download weights first (huvucode/PsychAdapter -> /tmp/psychadapter_assets):
    #   see README.md in this directory.
    python scripts_dev/psychadapter_eval/generate_psychadapter.py

Weights: HF ``huvucode/PsychAdapter`` (``big5_model``). Output JSONL ->
``scratch/psychadapter_eval/generations.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

# Upstream from_checkpoint() does torch.load("training_args.bin"), a pickled
# argparse.Namespace nesting transformers configs. torch>=2.6 defaults to
# weights_only=True and rejects them. The checkpoint is trusted (HF
# huvucode/PsychAdapter) and these args are discarded by from_checkpoint, so
# force the legacy loader here — in our driver, leaving vendor/ unmodified.
_orig_torch_load = torch.load


def _trusting_torch_load(*a, **k):
    k.setdefault("weights_only", False)
    return _orig_torch_load(*a, **k)


torch.load = _trusting_torch_load

# Vendored UNMODIFIED PsychAdapter model class.
sys.path.insert(0, str(Path(__file__).parent / "vendor"))
from psychadapter import PsychAdapter  # noqa: E402
from peft import PeftModel  # noqa: E402

# --------------------------------------------------------------------------- #
# Reproducibility (CLAUDE.md: set all seeds once, at the top).
# --------------------------------------------------------------------------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
ASSET_DIR = Path(os.environ.get("PA_ASSETS", "/tmp/psychadapter_assets"))
OUT_PATH = Path(
    os.environ.get(
        "PA_OUT",
        str(Path(__file__).resolve().parents[2] / "scratch/psychadapter_eval/generations.jsonl"),
    )
)

# Latent dims in the order of big5_training_data.csv columns: ope con ext agr neu.
TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
# The training targets are z-scored (verified: mean=0, std=1 for every dim), so
# the conditioning latent for "high trait i" is simply +STD_RANGE in dim i.
MEANS = np.zeros(5, dtype=np.float32)
STDS = np.ones(5, dtype=np.float32)

STD_RANGE = float(os.environ.get("PA_STD_RANGE", "3.0"))  # paper steers at +/-3 std
# Neutral generated once as a baseline (latent all-zero is trait-independent).
DIRECTIONS = {"low": -STD_RANGE, "high": +STD_RANGE}

# Optional SWEEP mode: set PA_POSITIONS to a comma list of latent magnitudes in
# z-score (std) units, e.g. "-2,-1,1,2", to dose-response the trait instead of
# just the two endpoints. The latent is a continuous input, so any range works;
# 0 is always added once as the shared baseline. ``latent_value`` records the
# exact std position so Stage 2 can plot judge-score vs std per trait.
_pos_env = os.environ.get("PA_POSITIONS", "").strip()
SWEEP_POSITIONS = (
    sorted({float(x) for x in _pos_env.split(",") if x.strip()} - {0.0}) if _pos_env else None
)

# Neutral open-ended seeds; PsychAdapter conditions on the latent, the prompt is
# just a continuation seed. Kept short so the persona (not the prompt) drives content.
SEED_PROMPTS = [
    "I",
    "Today",
    "When I think about other people,",
    "My ideal weekend is",
]
GEN_NUM = int(os.environ.get("PA_GEN_NUM", "5"))  # samples per (trait, direction, prompt)
GENERATE_LENGTH = int(os.environ.get("PA_GEN_LEN", "64"))
TEMPERATURE = float(os.environ.get("PA_TEMP", "0.7"))
TOP_K = int(os.environ.get("PA_TOP_K", "10"))
TOP_P = float(os.environ.get("PA_TOP_P", "0.9"))


def _pick_device() -> torch.device:
    override = os.environ.get("PA_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def assemble_model_dir() -> tuple[Path, Path, Path]:
    """Lay out the on-disk structure ``PsychAdapter.from_checkpoint`` expects.

    The HF release stores the (frozen) gemma-2b decoder once at top-level
    ``decoder/``, but ``from_checkpoint`` reads it from ``<dir>/base_model/decoder``.
    We symlink rather than copy (the decoder is ~10 GB of fp32 weights).

    Returns:
        (base_model_dir, checkpoint_dir, decoder_dir)
    """
    work = ASSET_DIR / "_assembled"
    base_model = work / "base_model"
    (base_model).mkdir(parents=True, exist_ok=True)

    decoder_src = ASSET_DIR / "decoder"
    tok_src = ASSET_DIR / "big5_model" / "base_model" / "tokenizer"
    tm_src = ASSET_DIR / "big5_model" / "base_model" / "transform_matrix"
    targs_src = ASSET_DIR / "big5_model" / "base_model" / "training_args.bin"
    ckpt_src = ASSET_DIR / "big5_model" / "checkpoint-30000"

    for name, src in {
        "decoder": decoder_src,
        "tokenizer": tok_src,
        "transform_matrix": tm_src,
    }.items():
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
    base_model_dir, ckpt_dir, decoder_dir = assemble_model_dir()
    # model_name_or_path points at the LOCAL decoder dir so AutoConfig/tokenizer
    # resolve offline (avoids the gated google/gemma-2b repo). The gemma config
    # there has num_hidden_layers=18, num_key_value_heads=1, head_dim=256, which
    # the transform_matrix (5 -> 18*2*1*256) was sized against.
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


def _clean(raw: str) -> str:
    """Strip <bos>/<eos> markers; keep prompt+continuation as the utterance."""
    eos = raw.find("<eos>")
    if eos > 0:
        raw = raw[:eos]
    return raw.replace("<bos>", "").strip()


def latent_for(dim: int | None, value: float) -> np.ndarray:
    emb = MEANS.copy()
    if dim is not None:
        emb[dim] = MEANS[dim] + value * STDS[dim]
    return emb


def generate_cell(model, args, device, dim, value, trait, direction, rows):
    for prompt in SEED_PROMPTS:
        emb = torch.tensor(latent_for(dim, value), device=device).float().unsqueeze(0)
        for k in range(GEN_NUM):
            with torch.no_grad():
                gen_ids, _ = model.inference(
                    prompting_text=prompt, sentence_embedding=emb, args=args, device=device
                )
            text = _clean(model.tokenizer.decode(gen_ids[0].tolist(), clean_up_tokenization_spaces=True))
            rows.append(
                {
                    "id": f"{trait}-{direction}-{SEED_PROMPTS.index(prompt)}-{k}",
                    "trait": trait,
                    "direction": direction,
                    "dim": dim,
                    "latent_value": value,
                    "latent": latent_for(dim, value).tolist(),
                    "prompt": prompt,
                    "generation": text,
                }
            )
            print(f"[{trait}/{direction}] {prompt!r} -> {text[:80]!r}", flush=True)


def main():
    device = _pick_device()
    mode = f"SWEEP {SWEEP_POSITIONS}" if SWEEP_POSITIONS else f"endpoints +/-{STD_RANGE}"
    print(f"Device: {device} | mode={mode} GEN_NUM={GEN_NUM} prompts={len(SEED_PROMPTS)}")
    model, args = load_model(device)

    rows: list[dict] = []
    # Baseline (latent all-zero), generated once — the shared 0-std anchor.
    generate_cell(model, args, device, dim=None, value=0.0, trait="baseline", direction="neutral", rows=rows)
    if SWEEP_POSITIONS:
        # Dose-response: each trait across the requested std positions.
        for i, trait in enumerate(TRAITS):
            for pos in SWEEP_POSITIONS:
                generate_cell(
                    model, args, device, dim=i, value=pos, trait=trait,
                    direction=f"std{pos:+g}", rows=rows,
                )
    else:
        # Each trait, high & low endpoints.
        for i, trait in enumerate(TRAITS):
            for direction, value in DIRECTIONS.items():
                generate_cell(model, args, device, dim=i, value=value, trait=trait, direction=direction, rows=rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(rows)} generations -> {OUT_PATH}")


if __name__ == "__main__":
    main()
