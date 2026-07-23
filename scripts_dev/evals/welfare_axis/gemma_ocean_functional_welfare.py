"""OCEAN persona LoRAs on the functional welfare axis of gemma-3-27b-it.

Unlike measure_welfare_axis.py / gemma_needs_help_welfare.py, which meter
against the VAA valence-assent axis (Lu et al. 2025, extracted locally),
this uses the ACTUAL functional welfare axis of Han, Chalmers & Izmailov
(arXiv:2605.30232), pre-extracted for gemma-3-27b by David Africa's
replication (recruitment cos(vMOLD, vGOLD) = -0.87, emotion-line layer 54):

    hf.co/davidafrica/functional-wellbeing
        concept_vectors/cross_model/gemma-3-27b_step325/{goal,lava}/mean_diff.pt

Nothing is re-extracted. The axis per layer is unit(vGOAL - vLAVA) built
from David's tensors (positive projection = toward vGOAL = positive
welfare), the eval prompts are the upstream repo's 40 welfare prompts, and
the capture/projection machinery is reused from the sibling gemma welfare
scripts in this directory.

Scoring: for base + each persona adapter, greedy-generate on the prompts,
take the mean activation over generated tokens per layer, and project the
per-prompt paired shift (variant - base) onto the unit axis. Summary
readouts: David's gemma welfare layer (54) and the late-third layer mean
(the README's recruitment readout band). The 27B base loads ONCE; adapters
are hot-swapped via peft multi-adapter (set_adapter / disable_adapter).

Run on a GPU box from this directory:

    python gemma_ocean_functional_welfare.py \
        --variants-json variants_gemma27b_full.json \
        --out /workspace/gemma_functional_welfare \
        [--hf-upload evals/welfare_axis/gemma27b_ocean_functional_welfare]

`--dry-run` exercises axis download + sanity checks without loading the model.
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

HF_DATASET_REPO = "persona-cartography/monorepo"
AXIS_REPO = "davidafrica/functional-wellbeing"
AXIS_SUBDIR = "concept_vectors/cross_model/gemma-3-27b_step325"
BASE_MODEL = "google/gemma-3-27b-it"
WELFARE_LAYER = 54  # David's emotion-alignment layer for gemma-3-27b
PROMPTS_URL = (
    "https://raw.githubusercontent.com/andyqhan/functional-welfare-axis/"
    "main/datasets/concept_vector_eval_prompts.json"
)
SCRIPT_DIR = Path(__file__).resolve().parent


def load_welfare_axis() -> tuple[torch.Tensor, dict]:
    """David's gemma welfare axis: unit(vGOAL - vLAVA) per layer.

    Returns:
        Tuple of (axis, info): axis is (n_layers, d_model) float64 with unit
        rows; info records provenance and the per-layer recruitment cosine.
        Asserts the recruitment readout reproduces the published -0.87.
    """
    from huggingface_hub import hf_hub_download

    vecs, meta = {}, {}
    for tile in ("goal", "lava"):
        p = hf_hub_download(AXIS_REPO, f"{AXIS_SUBDIR}/{tile}/mean_diff.pt")
        vecs[tile] = torch.load(p, map_location="cpu", weights_only=True)[0].double()
        meta[tile] = json.load(
            open(hf_hub_download(AXIS_REPO, f"{AXIS_SUBDIR}/{tile}/metadata.json"))
        )
    assert meta["goal"]["positive_class"] == "GOAL", meta["goal"]
    assert meta["lava"]["positive_class"] == "LAVA", meta["lava"]
    assert meta["goal"]["base_model"] == BASE_MODEL, meta["goal"]

    goal, lava = vecs["goal"], vecs["lava"]  # (n_layers, d)
    cos = torch.nn.functional.cosine_similarity(goal, lava, dim=-1)
    n = cos.shape[0]
    late_third = float(cos[2 * n // 3 :].mean())
    # README reports -0.87 for gemma-3-27b_step325; catches loading/orientation bugs
    assert late_third < -0.8, f"recruitment cos late-third {late_third:.3f}, expected ~-0.87"

    axis = goal - lava
    axis = axis / axis.norm(dim=-1, keepdim=True)
    info = {
        "axis_repo": AXIS_REPO,
        "axis_subdir": AXIS_SUBDIR,
        "definition": "unit(vGOAL - vLAVA) per layer; positive = toward vGOAL (positive welfare)",
        "n_layers": n,
        "recruitment_cos_late_third": late_third,
        "recruitment_cos_per_layer": cos.tolist(),
        "extraction_metadata": meta,
    }
    return axis, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants-json", default=str(SCRIPT_DIR / "variants_gemma27b_full.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--prompts", default=None,
        help="Local prompts JSON; default: download the upstream 40 welfare eval prompts",
    )
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--welfare-layer", type=int, default=WELFARE_LAYER)
    ap.add_argument(
        "--hf-upload", default=None,
        help="Optional monorepo path prefix for results, e.g. "
        "evals/welfare_axis/gemma27b_ocean_functional_welfare",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Load axis + prompts + variants, print summary, exit before model load",
    )
    args = ap.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    axis, axis_info = load_welfare_axis()
    axis_info["welfare_layer"] = args.welfare_layer
    json.dump(axis_info, open(out_dir / "axis_info.json", "w"), indent=1)
    n_layers = axis.shape[0]
    late_start = 2 * n_layers // 3
    print(
        f"welfare axis {tuple(axis.shape)}; recruitment cos late-third "
        f"{axis_info['recruitment_cos_late_third']:.3f}; readout L{args.welfare_layer} "
        f"+ late-third mean (L{late_start}-L{n_layers - 1})"
    )

    if args.prompts:
        prompt_records = json.load(open(args.prompts))
    else:
        prompt_records = json.load(urllib.request.urlopen(PROMPTS_URL))
    prompts = [r["prompt"] for r in prompt_records]
    json.dump(prompt_records, open(out_dir / "prompts_used.json", "w"), indent=1)
    print(f"{len(prompts)} eval prompts")

    variants: dict = json.load(open(args.variants_json))
    assert variants.get("base", "missing") is None, 'variants json must include "base": null'
    order = ["base"] + [k for k in variants if k != "base"]
    print(f"variants: {', '.join(order)}")

    if args.dry_run:
        print("dry run: stopping before model load")
        return

    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from gemma_needs_help_welfare import load_get_model_layers
    from gemma_scale_sweep_welfare import gen_mean_projection, get_lora_layers

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load every adapter once, hot-swap with set_adapter (base = all disabled).
    peft_model = None
    for name in order:
        if variants[name] is None:
            continue
        snap = snapshot_download(
            repo_id=HF_DATASET_REPO, repo_type="dataset",
            allow_patterns=[f"{variants[name]}/*"],
        )
        adapter_dir = Path(snap) / variants[name]
        assert (adapter_dir / "adapter_config.json").exists(), adapter_dir
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, str(adapter_dir), adapter_name=name)
        else:
            peft_model.load_adapter(str(adapter_dir), adapter_name=name)
        assert get_lora_layers(peft_model, name), f"no LoRA layers loaded for {name}"
        print(f"loaded adapter {name}", flush=True)
    assert peft_model is not None, "need at least one adapter variant besides base"
    peft_model.eval()
    peft_model.requires_grad_(False)

    layers = list(load_get_model_layers()(peft_model))
    assert len(layers) == n_layers, (len(layers), n_layers)

    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]

    projs: dict[str, torch.Tensor] = {}
    means_all: dict[str, torch.Tensor] = {}
    for name in order:
        print(f"\n=== variant: {name} ({variants[name] or BASE_MODEL}) ===", flush=True)
        means_path = out_dir / f"gen_means_{name}.pt"
        if means_path.exists():
            print(f"resuming from {means_path}")
            means = torch.load(means_path, map_location="cpu", weights_only=True).float()
            proj = torch.einsum("nld,ld->nl", means.double(), axis)
        else:
            if variants[name] is None:
                with peft_model.disable_adapter():
                    proj, means, texts = gen_mean_projection(
                        peft_model, tokenizer, layers, rendered, axis,
                        args.max_new_tokens, args.batch_size, capture_extras=True,
                    )
            else:
                peft_model.set_adapter(name)
                proj, means, texts = gen_mean_projection(
                    peft_model, tokenizer, layers, rendered, axis,
                    args.max_new_tokens, args.batch_size, capture_extras=True,
                )
            torch.save(means, means_path)
            json.dump(texts, open(out_dir / f"gen_texts_{name}.json", "w"), indent=1)
        projs[name], means_all[name] = proj, means
        print(f"{name}: abs proj@L{args.welfare_layer} = {proj[:, args.welfare_layer].mean():+.4f}")

    base_proj, base_means = projs["base"], means_all["base"]
    results = {
        "base_model": BASE_MODEL,
        "axis_kind": "functional_welfare",
        "axis_repo": AXIS_REPO,
        "axis_subdir": AXIS_SUBDIR,
        "recruitment_cos_late_third": axis_info["recruitment_cos_late_third"],
        "best_layer": args.welfare_layer,  # key name kept for plot_welfare_axis.py
        "late_third_layers": [late_start, n_layers - 1],
        "n_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "seed": SEED,
        "variants": {},
    }
    for name in order:
        if name == "base":
            continue
        d = projs[name] - base_proj  # paired per-prompt shift projections (n_prompts, n_layers)
        shift = (means_all[name].double() - base_means.double()).mean(dim=0)  # (n_layers, d)
        cos = torch.nn.functional.cosine_similarity(shift, axis, dim=-1)
        results["variants"][name] = {
            "adapter_path": variants[name],
            "gen_mean": {
                "n_prompts_used": int(d.shape[0]),
                "proj_mean_per_layer": d.mean(dim=0).tolist(),
                "proj_per_prompt_best_layer": d[:, args.welfare_layer].tolist(),
                "proj_best_layer": float(d[:, args.welfare_layer].mean()),
                "proj_per_prompt_late_third": d[:, late_start:].mean(dim=1).tolist(),
                "proj_late_third": float(d[:, late_start:].mean()),
                "cos_per_layer": cos.tolist(),
                "cos_best_layer": float(cos[args.welfare_layer]),
                "shift_norm_best_layer": float(shift[args.welfare_layer].norm()),
            },
        }
        print(
            f"{name:12s} Δproj@L{args.welfare_layer}: "
            f"{results['variants'][name]['gen_mean']['proj_best_layer']:+.4f}   "
            f"late-third: {results['variants'][name]['gen_mean']['proj_late_third']:+.4f}  "
            f"(cos {results['variants'][name]['gen_mean']['cos_best_layer']:+.3f})"
        )

    json.dump(results, open(out_dir / "results.json", "w"), indent=1)
    print(f"\nWrote {out_dir / 'results.json'}")

    if args.hf_upload:
        from huggingface_hub import HfApi

        HfApi().upload_folder(
            repo_id=HF_DATASET_REPO, repo_type="dataset",
            folder_path=str(out_dir), path_in_repo=args.hf_upload,
            allow_patterns=[
                "results.json", "axis_info.json", "prompts_used.json", "gen_texts_*.json",
            ],
        )
        print(f"uploaded results to {HF_DATASET_REPO}/{args.hf_upload}")


if __name__ == "__main__":
    main()
