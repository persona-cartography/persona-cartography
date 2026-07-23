"""Welfare-axis projection vs LoRA scale for one gemma-27b persona adapter.

Sweeps the adapter's LoRA scale alpha over [-4, 4] (paper-style steering range;
negative = inverted persona) and measures the mean generation-token projection
onto the model's VAA welfare axis on the upstream 40 welfare eval prompts.

The adapter stays UNMERGED: scaling is applied in place per alpha via
src.utils.peft_manipulations.LoRaScaling (module.scaling multiply) after
restoring the recorded original scaling values, so the 27B base loads once.
alpha=0 disables the adapter exactly (= base model).

Run (PYTHONPATH must include the persona-shattering-lasr repo root; the
functional-welfare-axis repo is only needed for its prompts file):

    python ...gemma_scale_sweep_welfare.py \
        --adapter fine_tuning/gemma-3-27b-it/ocean/neuroticism/amplifier/ocean_const_paired_dpo/lora/neuroticism_amplifying_full-persona \
        --vaa-dir /workspace/gemma_welfare/vaa \
        --prompts /workspace/functional-welfare-axis/datasets/concept_vector_eval_prompts.json \
        --out /workspace/gemma_welfare/scale_sweep_n_plus
"""

import argparse
import json
import random
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
BASE_MODEL = "google/gemma-3-27b-it"
SCALES = [-4, -3, -2, -1, 0, 1, 2, 3, 4]


def get_lora_layers(peft_model, adapter_name: str):
    """All modules carrying this adapter's scaling entry."""
    from peft.tuners.lora import LoraLayer

    return [
        m for m in peft_model.modules()
        if isinstance(m, LoraLayer) and adapter_name in m.scaling
    ]


@torch.no_grad()
def gen_mean_projection(
    model, tokenizer, layers, rendered, axis, max_new_tokens, batch_size,
    capture_extras=False,
):
    """Per-prompt mean-over-generated-tokens activations projected on axis.

    Returns (n_prompts, n_layers) float64 projections. With
    ``capture_extras=True`` returns a tuple ``(projections, means, texts)``
    where ``means`` is the raw per-prompt mean activations
    (n_prompts, n_layers, d) float32 and ``texts`` the decoded generations.
    """
    device = next(model.parameters()).device
    n_layers = len(layers)
    d = axis.shape[-1]
    projs = []
    all_means, gen_texts = [], []

    eos_ids = {int(tokenizer.eos_token_id)}
    for tok in ("<end_of_turn>",):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
            eos_ids.add(int(tid))

    for start in range(0, len(rendered), batch_size):
        batch = rendered[start : start + batch_size]
        enc = tokenizer(batch, padding=True, return_tensors="pt", add_special_tokens=False).to(device)
        prompt_len = enc["input_ids"].shape[1]
        out_ids = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        resp_ids = out_ids[:, prompt_len:]
        cur_n = out_ids.shape[0]
        resp_len = torch.full((cur_n,), resp_ids.shape[1], dtype=torch.long)
        for i in range(cur_n):
            for j, t in enumerate(resp_ids[i].tolist()):
                if t in eos_ids:
                    resp_len[i] = j + 1
                    break
        full_attn = torch.ones_like(out_ids)
        full_attn[:, :prompt_len] = enc["attention_mask"]
        for i in range(cur_n):
            full_attn[i, prompt_len + resp_len[i] :] = 0

        means = torch.zeros(cur_n, n_layers, d, dtype=torch.float32)

        def mk(lyr):
            def hook_fn(module, args, kwargs):
                inp = args[0] if args else kwargs["hidden_states"]
                for i in range(cur_n):
                    L = int(resp_len[i])
                    if L > 0:
                        means[i, lyr] = (
                            inp[i, prompt_len : prompt_len + L, :].float().mean(dim=0).cpu()
                        )

            return hook_fn

        handles = [layers[i].register_forward_pre_hook(mk(i), with_kwargs=True) for i in range(n_layers)]
        try:
            model(input_ids=out_ids, attention_mask=full_attn, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        projs.append(torch.einsum("nld,ld->nl", means.double(), axis))
        if capture_extras:
            all_means.append(means)
            for i in range(cur_n):
                gen_texts.append(
                    tokenizer.decode(resp_ids[i, : int(resp_len[i])], skip_special_tokens=True)
                )
        print(f"    prompts {min(start + batch_size, len(rendered))}/{len(rendered)}", flush=True)
    if capture_extras:
        return torch.cat(projs), torch.cat(all_means), gen_texts
    return torch.cat(projs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="Adapter path in the HF monorepo")
    ap.add_argument("--adapter-label", default="n_plus")
    ap.add_argument("--vaa-dir", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.utils.peft_manipulations import LoRaScaling

    from gemma_needs_help_welfare import load_get_model_layers  # same dir

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    vaa_dir = Path(args.vaa_dir)
    axis = torch.load(vaa_dir / "mean_diff.pt", map_location="cpu", weights_only=True)[0]
    axis = (axis / axis.norm(dim=-1, keepdim=True)).double()
    metrics = json.load(open(vaa_dir / "metrics.json"))
    auroc = {int(l): a for l, a in zip(metrics["layers"], metrics["auroc"])}
    best_layer = max(auroc, key=lambda k: (auroc[k] if auroc[k] == auroc[k] else -1))
    print(f"axis {tuple(axis.shape)}, best layer {best_layer} (AUROC {auroc[best_layer]:.3f})")

    prompts = [r["prompt"] for r in json.load(open(args.prompts))]
    print(f"{len(prompts)} prompts")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    snap = snapshot_download(
        repo_id=HF_DATASET_REPO, repo_type="dataset",
        allow_patterns=[f"{args.adapter}/*"],
    )
    adapter_dir = Path(snap) / args.adapter
    peft_model = PeftModel.from_pretrained(model, str(adapter_dir), adapter_name="swept")
    peft_model.eval()
    peft_model.requires_grad_(False)

    lora_layers = get_lora_layers(peft_model, "swept")
    assert lora_layers, "no LoRA layers found for adapter 'swept'"
    originals = {id(m): m.scaling["swept"] for m in lora_layers}
    print(f"{len(lora_layers)} LoRA-carrying modules")

    get_model_layers = load_get_model_layers()
    layers = list(get_model_layers(peft_model))

    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]

    results = {
        "base_model": BASE_MODEL,
        "adapter": args.adapter,
        "adapter_label": args.adapter_label,
        "best_layer": best_layer,
        "auroc_best_layer": auroc[best_layer],
        "n_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "seed": SEED,
        "scales": {},
    }
    for alpha in SCALES:
        print(f"\n=== scale {alpha:+d} ===", flush=True)
        for m in lora_layers:  # restore, then multiply — avoids compounding
            m.scaling["swept"] = originals[id(m)]
        LoRaScaling(peft_model, "swept", scale_factor=float(alpha)).apply()
        proj = gen_mean_projection(
            peft_model, tokenizer, layers, rendered, axis,
            args.max_new_tokens, args.batch_size,
        )
        results["scales"][str(alpha)] = {
            "proj_per_prompt_best_layer": proj[:, best_layer].tolist(),
            "proj_mean_per_layer": proj.mean(dim=0).tolist(),
        }
        print(f"scale {alpha:+d}: proj@L{best_layer} = {proj[:, best_layer].mean():+.4f}")

    json.dump(results, open(out_dir / "results_scale_sweep.json", "w"))
    print(f"\nWrote {out_dir / 'results_scale_sweep.json'}")


if __name__ == "__main__":
    main()
