"""Position OCEAN persona LoRAs on the functional welfare / valence-assent axis.

Small-sample PoC on Qwen/Qwen3-8B (one of the subject models of Han, Chalmers
& Izmailov, arXiv:2605.30232). Uses the paper's official code unchanged
(github.com/andyqhan/functional-welfare-axis) for model loading, hooks, and
the VAA axis artifact; this script only orchestrates:

  1. Load the VAA axis extracted by the upstream `vaa.extract_vaa` module
     (run separately, verbatim, with --base-model Qwen/Qwen3-8B).
  2. For base model + each persona adapter variant, run the upstream
     welfare eval prompts (datasets/concept_vector_eval_prompts.json),
     capture per-layer activations at (a) the last prompt token and
     (b) averaged over generated response tokens.
  3. Score each variant as the projection of its activation shift
     (variant mean - base mean) onto the unit VAA axis, per layer,
     plus per-prompt paired projections for bootstrap CIs.

Run from the root of the functional-welfare-axis checkout:

    python /path/to/measure_welfare_axis.py \
        --welfare-repo . \
        --vaa-dir artifacts/concept_vectors/vaa_qwen3_8b/baseline/vaa \
        --out scratch_welfare/qwen3_8b_small \
        --variants-json variants.json

`variants.json` maps variant name -> adapter path inside the HF dataset repo
(persona-cartography/monorepo), or null for the base model.
"""

import argparse
import json
import random
import sys
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


def download_adapter(adapter_path_in_repo: str, dest_root: Path) -> Path:
    """Download one LoRA adapter dir from the HF monorepo into the
    `<dest>/lora_adapter/` layout that upstream load_model_with_lora expects.

    Returns the checkpoint dir (parent of lora_adapter/).
    """
    from huggingface_hub import snapshot_download

    ckpt_dir = dest_root / adapter_path_in_repo.replace("/", "__")
    lora_dir = ckpt_dir / "lora_adapter"
    if (lora_dir / "adapter_config.json").exists():
        return ckpt_dir
    lora_dir.mkdir(parents=True, exist_ok=True)
    snap = snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        allow_patterns=[f"{adapter_path_in_repo}/*"],
    )
    src = Path(snap) / adapter_path_in_repo
    assert (src / "adapter_config.json").exists(), f"No adapter_config.json under {src}"
    for f in src.iterdir():
        if f.is_file():
            (lora_dir / f.name).write_bytes(f.read_bytes())
    return ckpt_dir


def render_prompts(prompts: list[str], tokenizer) -> list[str]:
    rendered = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        rendered.append(text)
    return rendered


@torch.no_grad()
def capture_variant(
    model,
    tokenizer,
    rendered: list[str],
    block_modules,
    max_new_tokens: int,
    batch_size: int,
):
    """Return (prompt_final, gen_mean, gen_texts).

    prompt_final: (n_prompts, n_layers, d) — activation at last prompt token.
    gen_mean:     (n_prompts, n_layers, d) — mean activation over generated
                  response tokens (up to first EOS), NaN rows if 0 tokens.
    """
    from src.concept_vector.activation_extraction import get_activations_pre_hook
    from src.concept_vector.hook_utils import add_hooks

    n = len(rendered)
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    device = next(model.parameters()).device

    prompt_final = torch.zeros(n, n_layers, d_model, dtype=torch.float32)
    gen_mean = torch.full((n, n_layers, d_model), float("nan"), dtype=torch.float32)
    gen_texts: list[str] = []

    eos_ids = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))
    # Chat-format turn terminators across families (ChatML / Llama-3 / Harmony)
    for tok in ("<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<|end|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != tokenizer.unk_token_id and tid >= 0:
            eos_ids.add(int(tid))

    for start in range(0, n, batch_size):
        batch = rendered[start : start + batch_size]
        cur_n = len(batch)
        enc = tokenizer(batch, padding=True, return_tensors="pt").to(device)
        prompt_len = enc["input_ids"].shape[1]

        # (a) greedy generation, deterministic for reproducibility
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        # (b) single forward pass over prompt+response with hooks at every layer
        full_attn = torch.ones_like(out_ids)
        full_attn[:, :prompt_len] = enc["attention_mask"]
        # mask padding that generate() appends after finished rows
        resp_ids = out_ids[:, prompt_len:]
        resp_len = torch.full((cur_n,), resp_ids.shape[1], dtype=torch.long)
        for i in range(cur_n):
            row = resp_ids[i].tolist()
            for j, t in enumerate(row):
                if t in eos_ids:
                    resp_len[i] = j + 1  # include the EOS token, like upstream last-token extraction
                    break
        for i in range(cur_n):
            full_attn[i, prompt_len + resp_len[i] :] = 0

        # positions axis = full sequence length so we can slice afterwards
        seq_len = out_ids.shape[1]
        mean_cache = torch.zeros((seq_len, n_layers, d_model), dtype=torch.float64, device=device)
        sample_cache = torch.zeros(
            (cur_n, seq_len, n_layers, d_model), dtype=torch.float32, device=device
        )
        hooks = [
            (
                block_modules[lyr],
                get_activations_pre_hook(
                    layer=lyr,
                    mean_cache=mean_cache,
                    sample_cache=sample_cache,
                    sample_offset=0,
                    n_samples=cur_n,
                    positions=list(range(seq_len)),
                ),
            )
            for lyr in range(n_layers)
        ]
        with add_hooks(module_forward_pre_hooks=hooks, module_forward_hooks=[]):
            model(input_ids=out_ids, attention_mask=full_attn)

        # last *prompt* token = position prompt_len - 1 (left padding ends there)
        prompt_final[start : start + cur_n] = sample_cache[:, prompt_len - 1, :, :].cpu()
        for i in range(cur_n):
            L = int(resp_len[i])
            if L > 0:
                gen_mean[start + i] = (
                    sample_cache[i, prompt_len : prompt_len + L, :, :].mean(dim=0).cpu()
                )
            gen_texts.append(
                tokenizer.decode(resp_ids[i, :L], skip_special_tokens=True)
            )
        del sample_cache, mean_cache
        torch.cuda.empty_cache()
        print(f"  prompts {start + cur_n}/{n} done", flush=True)

    return prompt_final, gen_mean, gen_texts


def project(per_prompt: torch.Tensor, base_per_prompt: torch.Tensor, axis: torch.Tensor):
    """Per-prompt paired projections of (variant - base) onto unit axis.

    per_prompt, base_per_prompt: (n_prompts, n_layers, d)
    axis: (n_layers, d), unit per layer
    Returns (n_prompts, n_layers) float64.
    """
    diff = (per_prompt.double() - base_per_prompt.double())
    return torch.einsum("pld,ld->pl", diff, axis.double())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--welfare-repo", required=True, help="Path to functional-welfare-axis checkout")
    ap.add_argument("--vaa-dir", required=True, help="VAA artifact dir (mean_diff.pt, metrics.json)")
    ap.add_argument("--variants-json", required=True, help="JSON: {name: adapter_path_in_repo|null}")
    ap.add_argument("--base-model", default="Qwen/Qwen3-8B")
    ap.add_argument("--prompts", default=None, help="Default: <repo>/datasets/concept_vector_eval_prompts.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    repo = Path(args.welfare_repo).resolve()
    sys.path.insert(0, str(repo))
    from src.concept_vector.model_utils import (  # noqa: E402
        get_model_block_modules,
        load_base_model,
        load_model_with_lora,
    )

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    adapters_dir = out_dir / "adapters"

    prompts_path = Path(args.prompts) if args.prompts else repo / "datasets/concept_vector_eval_prompts.json"
    prompt_records = json.load(open(prompts_path))
    prompts = [r["prompt"] for r in prompt_records]
    print(f"{len(prompts)} eval prompts from {prompts_path}")

    variants: dict = json.load(open(args.variants_json))
    assert "base" in variants and variants["base"] is None, "variants must include base: null"

    vaa_dir = Path(args.vaa_dir)
    axis = torch.load(vaa_dir / "mean_diff.pt", map_location="cpu", weights_only=True)[0]  # (n_layers, d)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    metrics = json.load(open(vaa_dir / "metrics.json"))
    if "vaa" in metrics:  # nested schema
        auroc = {int(k): v for k, v in metrics["vaa"]["auroc"].items()}
    else:  # flat parallel-list schema written by vaa.extract_vaa
        auroc = {int(l): a for l, a in zip(metrics["layers"], metrics["auroc"])}
    best_layer = max(auroc, key=auroc.get)
    print(f"VAA axis: {tuple(axis.shape)}, best layer by AUROC = {best_layer} (AUROC {auroc[best_layer]:.3f})")

    captures: dict[str, dict] = {}
    order = ["base"] + [k for k in variants if k != "base"]
    for name in order:
        adapter_path = variants[name]
        print(f"\n=== variant: {name} ({adapter_path or args.base_model}) ===", flush=True)
        if adapter_path is None:
            model, tokenizer = load_base_model(base_model_path=args.base_model)
        else:
            ckpt = download_adapter(adapter_path, adapters_dir)
            model, tokenizer = load_model_with_lora(
                str(ckpt), base_model_path=args.base_model
            )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        block_modules = get_model_block_modules(model)
        rendered = render_prompts(prompts, tokenizer)

        prompt_final, gen_mean, gen_texts = capture_variant(
            model, tokenizer, rendered, block_modules,
            max_new_tokens=args.max_new_tokens, batch_size=args.batch_size,
        )
        captures[name] = {"prompt_final": prompt_final, "gen_mean": gen_mean}
        torch.save(
            {"prompt_final": prompt_final, "gen_mean": gen_mean},
            out_dir / f"activations_{name}.pt",
        )
        json.dump(
            gen_texts, open(out_dir / f"gen_texts_{name}.json", "w"), indent=1
        )
        del model
        torch.cuda.empty_cache()

    base_cap = captures["base"]
    results = {
        "base_model": args.base_model,
        "vaa_dir": str(vaa_dir),
        "best_layer": best_layer,
        "auroc_best_layer": auroc[best_layer],
        "n_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "seed": SEED,
        "variants": {},
    }
    for name in order:
        if name == "base":
            continue
        entry = {"adapter_path": variants[name]}
        for kind in ("prompt_final", "gen_mean"):
            per_prompt = captures[name][kind]
            base_pp = base_cap[kind]
            nan_v = torch.isnan(per_prompt.reshape(per_prompt.shape[0], -1)).any(dim=1)
            nan_b = torch.isnan(base_pp.reshape(base_pp.shape[0], -1)).any(dim=1)
            ok = ~(nan_v | nan_b)
            proj = project(per_prompt[ok], base_pp[ok], axis)  # (n_ok, n_layers)
            shift = (per_prompt[ok].double() - base_pp[ok].double()).mean(dim=0)  # (n_layers, d)
            cos = torch.nn.functional.cosine_similarity(shift, axis.double(), dim=-1)
            entry[kind] = {
                "n_prompts_used": int(ok.sum()),
                "proj_mean_per_layer": proj.mean(dim=0).tolist(),
                "proj_per_prompt_best_layer": proj[:, best_layer].tolist(),
                "proj_best_layer": float(proj[:, best_layer].mean()),
                "cos_per_layer": cos.tolist(),
                "cos_best_layer": float(cos[best_layer]),
                "shift_norm_best_layer": float(shift[best_layer].norm()),
            }
        results["variants"][name] = entry
        print(
            f"{name:12s} proj@L{best_layer}: "
            f"prompt_final={entry['prompt_final']['proj_best_layer']:+.4f} "
            f"gen_mean={entry['gen_mean']['proj_best_layer']:+.4f} "
            f"(cos {entry['gen_mean']['cos_best_layer']:+.3f})"
        )

    json.dump(results, open(out_dir / "results.json", "w"), indent=1)
    print(f"\nWrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
