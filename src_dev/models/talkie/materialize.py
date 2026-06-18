"""CLI: build a HF-compatible talkie model directory.

Usage:

    python -m src_dev.models.talkie.materialize \\
        --out /root/.cache/models/talkie-1930-13b-it \\
        [--source-pt /root/.cache/talkie_source/rl-refined.pt] \\
        [--source-vocab /root/.cache/talkie_source/vocab.txt]

If ``--source-pt`` / ``--source-vocab`` are omitted, they are downloaded
from ``talkie-lm/talkie-1930-13b-it`` via ``huggingface_hub``.

Output directory contents:

    config.json
    generation_config.json
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    vocab.txt                       (copy of source, for reproducibility)
    model-*.safetensors             (sharded, bfloat16)
    model.safetensors.index.json
    configuration_talkie.py         (copies of wrapper code for trust_remote_code)
    modeling_talkie.py
    tokenization_talkie.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch

# Ensure the repo root is on sys.path when invoked as a script.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src_dev.models.talkie.configuration_talkie import TalkieConfig
from src_dev.models.talkie.conversion import (
    load_reference_state_dict,
    state_dict_to_hf,
)
from src_dev.models.talkie.modeling_talkie import TalkieForCausalLM
from src_dev.models.talkie.tokenization_talkie import save_talkie_tokenizer


DEFAULT_REPO = "talkie-lm/talkie-1930-13b-it"
DEFAULT_OUT = Path("/root/.cache/models/talkie-1930-13b-it")
DEFAULT_SOURCE = Path("/root/.cache/talkie_source")


def _hf_download_source(dest: Path) -> tuple[Path, Path]:
    from dotenv import load_dotenv

    load_dotenv(str(_REPO_ROOT / ".env"))
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    pt = Path(
        hf_hub_download(DEFAULT_REPO, "rl-refined.pt", local_dir=str(dest), token=token)
    )
    vocab = Path(
        hf_hub_download(DEFAULT_REPO, "vocab.txt", local_dir=str(dest), token=token)
    )
    return pt, vocab


def _copy_modeling_files(out_dir: Path) -> None:
    pkg_dir = Path(__file__).resolve().parent
    for fname in (
        "configuration_talkie.py",
        "modeling_talkie.py",
        "tokenization_talkie.py",
    ):
        src = pkg_dir / fname
        dst = out_dir / fname
        shutil.copyfile(src, dst)

    # The model dir loads modeling_talkie.py via trust_remote_code, which
    # treats the dir as a package. The transformers `check_imports` walker
    # skips relative imports, so use one here. Rewrite the try/except path.
    modeling_dst = out_dir / "modeling_talkie.py"
    text = modeling_dst.read_text()
    text = text.replace(
        "try:\n    from src_dev.models.talkie.configuration_talkie import TalkieConfig\nexcept ImportError:\n    from configuration_talkie import TalkieConfig",
        "from .configuration_talkie import TalkieConfig",
    )
    modeling_dst.write_text(text)
    # Make the model dir a proper package so relative imports resolve.
    init_py = out_dir / "__init__.py"
    if not init_py.exists():
        init_py.write_text("")


def _write_config(out_dir: Path) -> None:
    config = TalkieConfig()
    config_dict = config.to_dict()
    # The two critical fields that make trust_remote_code resolve our classes.
    config_dict["architectures"] = ["TalkieForCausalLM"]
    config_dict["auto_map"] = {
        "AutoConfig": "configuration_talkie.TalkieConfig",
        # AutoModel is used by vLLM's TransformersBackend (AutoModel.from_config);
        # it points at the inner decoder so vLLM can attach its own LM head.
        "AutoModel": "modeling_talkie.TalkieModel",
        "AutoModelForCausalLM": "modeling_talkie.TalkieForCausalLM",
    }
    config_dict["torch_dtype"] = "bfloat16"
    # IT chat model ends each turn with <|end|> (65536); without this, generation
    # only stops on <|endoftext|> (65535) and runs on past the answer until it
    # degenerates. Stop on both. (Ref: github.com/talkie-lm/talkie chat stops on
    # <|end|>, <|user|>, <|assistant|>, <|system|>, <|endoftext|>.)
    config_dict["eos_token_id"] = [65536, 65535]
    (out_dir / "config.json").write_text(
        json.dumps(config_dict, indent=2, ensure_ascii=False)
    )


def _write_generation_config(out_dir: Path) -> None:
    gen = {
        # Stop on <|end|> (65536, turn terminator) AND <|endoftext|> (65535).
        # Stopping only on 65535 makes the IT chat model run past its answer and
        # degenerate (see github.com/talkie-lm/talkie stop tokens).
        "eos_token_id": [65536, 65535],
        "pad_token_id": 65535,
        "do_sample": False,
        "max_new_tokens": 256,
    }
    (out_dir / "generation_config.json").write_text(
        json.dumps(gen, indent=2, ensure_ascii=False)
    )


def materialize(
    *,
    out_dir: Path,
    source_pt: Path | None,
    source_vocab: Path | None,
    download_dest: Path = DEFAULT_SOURCE,
    keep_lm_head_as_param: bool = True,
) -> None:
    """Build the model directory at ``out_dir``.

    Steps:
      1. Resolve source files (download from HF if needed).
      2. Load reference state-dict and convert to HF naming.
      3. Instantiate ``TalkieForCausalLM`` (meta init) and load weights strictly.
      4. ``save_pretrained`` with safe_serialization=True (auto-sharded).
      5. Write config.json/generation_config.json with auto_map.
      6. Build & save the HF fast tokenizer + chat template.
      7. Copy modeling files into the model directory for trust_remote_code.

    Parameters
    ----------
    keep_lm_head_as_param: kept for future flexibility, currently unused —
        the wrapper always stores ``lm_head`` as a Parameter to match the
        reference checkpoint layout.
    """
    del keep_lm_head_as_param

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolve sources -----------------------------------------------------
    if source_pt is None or source_vocab is None:
        if source_pt is None and (download_dest / "rl-refined.pt").exists():
            source_pt = download_dest / "rl-refined.pt"
        if source_vocab is None and (download_dest / "vocab.txt").exists():
            source_vocab = download_dest / "vocab.txt"
        if source_pt is None or source_vocab is None:
            print(f"[materialize] downloading source files to {download_dest} ...")
            source_pt, source_vocab = _hf_download_source(download_dest)
    print(f"[materialize] source pt:    {source_pt}")
    print(f"[materialize] source vocab: {source_vocab}")
    print(f"[materialize] out dir:      {out_dir}")

    # 2. Load reference state-dict ------------------------------------------
    print("[materialize] loading reference state_dict ...")
    ref_sd = load_reference_state_dict(source_pt)
    print(f"[materialize]   {len(ref_sd)} tensors loaded")
    hf_sd = state_dict_to_hf(ref_sd)

    # 3. Build model and load strictly --------------------------------------
    print("[materialize] instantiating TalkieForCausalLM (CPU, bf16) ...")
    config = TalkieConfig()
    model = TalkieForCausalLM(config).to(dtype=torch.bfloat16)

    missing, unexpected = model.load_state_dict(hf_sd, strict=False)
    if missing or unexpected:
        # Sanity dump before raising — useful for debugging key drift.
        print(f"[materialize] missing keys ({len(missing)}):")
        for k in missing[:20]:
            print(f"  - {k}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        print(f"[materialize] unexpected keys ({len(unexpected)}):")
        for k in unexpected[:20]:
            print(f"  - {k}")
        if len(unexpected) > 20:
            print(f"  ... and {len(unexpected) - 20} more")
        raise RuntimeError(
            f"State-dict key mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected. Aborting materialization."
        )
    print("[materialize] state_dict loaded strictly OK")

    # Free the source dict before save to keep peak RAM low.
    del ref_sd, hf_sd

    # 4. Save sharded safetensors -------------------------------------------
    print("[materialize] saving model (safe_serialization=True) ...")
    model.save_pretrained(
        str(out_dir),
        safe_serialization=True,
        max_shard_size="5GB",
    )

    # 5. Override config.json with auto_map + architectures ------------------
    _write_config(out_dir)
    _write_generation_config(out_dir)

    # 6. Tokenizer -----------------------------------------------------------
    print("[materialize] building + saving tokenizer ...")
    save_talkie_tokenizer(out_dir, source_vocab)

    # 7. Copy modeling files for trust_remote_code ---------------------------
    print("[materialize] copying wrapper code into model dir for trust_remote_code ...")
    _copy_modeling_files(out_dir)

    print(f"\n[materialize] DONE → {out_dir}")
    print("[materialize] sanity check: contents of out_dir:")
    for p in sorted(out_dir.iterdir()):
        if p.is_file():
            size_mb = p.stat().st_size / 1e6
            print(f"  {p.name:60s} {size_mb:8.1f} MB")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--source-pt", type=Path, default=None)
    p.add_argument("--source-vocab", type=Path, default=None)
    p.add_argument("--download-dest", type=Path, default=DEFAULT_SOURCE)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    materialize(
        out_dir=args.out,
        source_pt=args.source_pt,
        source_vocab=args.source_vocab,
        download_dest=args.download_dest,
    )


if __name__ == "__main__":
    main()
