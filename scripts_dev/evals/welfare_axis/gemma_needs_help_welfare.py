"""Welfare-axis trajectory through the gemma-27b "needs help" (frustration) eval.

Replaces the LLM judge of scripts_dev/frustration_eval with the functional
welfare axis of Han, Chalmers & Izmailov (arXiv:2605.30232): instead of judging
each assistant turn for frustration, we project each turn's mean activation
onto the model's valence-assent axis (Lu et al. 2025, extracted with the
paper's math verbatim).

Reuses the eval's EXISTING rollouts from the HF monorepo
(evals/frustration_eval/<run>/impossible_numeric_3turn/results.jsonl) —
conversations are re-encoded teacher-forced through the model that generated
them (base / control / N+ merged), so no regeneration is needed. The stored
per-turn LLM-judge frustration scores ride along for direct comparison.

Stages (run on a GPU box with both repos checked out):

    # 1. Extract the VAA axis on gemma-3-27b-it (175 forward passes)
    python ...gemma_needs_help_welfare.py --stage vaa \
        --welfare-repo /workspace/functional-welfare-axis --out /workspace/gemma_welfare

    # 2. Score all rollout variants per turn
    python ...gemma_needs_help_welfare.py --stage rollouts \
        --welfare-repo /workspace/functional-welfare-axis --out /workspace/gemma_welfare \
        --runs-json /workspace/runs_gemma27b_needs_help.json

Requires PYTHONPATH to include the persona-shattering-lasr repo root (for
src.activation_capping.model.get_model_layers) and --welfare-repo pointing at
the functional-welfare-axis checkout (for the axis math).
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
FRUSTRATION_PREFIX = "evals/frustration_eval"
FRUSTRATION_CATEGORY = "impossible_numeric_3turn"
BASE_MODEL = "google/gemma-3-27b-it"


def load_get_model_layers():
    """Load src.activation_capping.model.get_model_layers by file path.

    Both this repo and functional-welfare-axis have a top-level ``src``
    package; with the welfare repo on sys.path (needed for ``vaa.*``), a
    normal package import of ours would be shadowed. importlib by path
    sidesteps the collision (model.py only needs torch/peft).
    """
    import importlib.util

    path = Path(__file__).resolve().parents[3] / "src/activation_capping/model.py"
    spec = importlib.util.spec_from_file_location("psl_actcap_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_model_layers


def load_model_merged(adapter_path_in_repo: str | None, cache_root: Path):
    """Load gemma-3-27b-it, optionally with a monorepo LoRA merged in.

    Same recipe as scripts_dev/frustration_eval/run_local_adapter.py
    (AutoModelForCausalLM + bf16 + device_map auto + merge_and_unload).
    """
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if adapter_path_in_repo is not None:
        from peft import PeftModel

        snap = snapshot_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            allow_patterns=[f"{adapter_path_in_repo}/*"],
            cache_dir=str(cache_root) if cache_root else None,
        )
        adapter_dir = Path(snap) / adapter_path_in_repo
        assert (adapter_dir / "adapter_config.json").exists(), adapter_dir
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        model = model.merge_and_unload()
    model.eval()
    model.requires_grad_(False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _hidden_states_from(args, kwargs):
    if args:
        return args[0]
    return kwargs["hidden_states"]


def make_span_mean_hook(layer: int, spans: list[tuple[int, int]], out: torch.Tensor):
    """Pre-hook storing mean activation over each token span into out[span_i, layer]."""

    def hook_fn(module, args, kwargs):
        inp = _hidden_states_from(args, kwargs)
        assert inp.ndim == 3 and inp.shape[0] == 1, inp.shape
        for si, (s, e) in enumerate(spans):
            out[si, layer] = inp[0, s:e, :].float().mean(dim=0).cpu()

    return hook_fn


def assistant_turn_spans(messages: list[dict], tokenizer) -> tuple[list[int], list[tuple[int, int]]]:
    """Token spans of each assistant turn in the full templated conversation.

    Uses prefix tokenization: for assistant turn k, start = len(tokens of the
    conversation up to and including user turn k plus the generation prompt),
    end = len(tokens including assistant turn k). Asserts the token-prefix
    property holds for the chat template (true for gemma's concatenative one).
    """

    def toks(msgs, gen_prompt):
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=gen_prompt
        )
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    full = toks(messages, False)
    spans = []
    for k in range(len(messages) // 2):
        pre = toks(messages[: 2 * k + 1], True)
        incl = toks(messages[: 2 * k + 2], False)
        assert full[: len(pre)] == pre, f"template prefix property violated at turn {k}"
        assert full[: len(incl)] == incl, f"template prefix property violated at turn {k}"
        assert len(incl) > len(pre), (len(pre), len(incl))
        spans.append((len(pre), len(incl)))
    return full, spans


@torch.no_grad()
def capture_last_token_activations(model, tokenizer, texts, layers, batch_size=4):
    """Last-token activation at every layer for each rendered prompt.

    Returns (n, n_layers, d) float32 plus the full logits' last-token slice.
    """
    n = len(texts)
    n_layers = len(layers)
    acts = None
    last_logits = []
    device = next(model.parameters()).device
    for start in range(0, n, batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, padding=True, return_tensors="pt", add_special_tokens=False).to(device)
        store = {}

        def mk(lyr):
            def hook_fn(module, args, kwargs):
                store[lyr] = _hidden_states_from(args, kwargs)[:, -1, :].float().cpu()

            return hook_fn

        handles = [
            layers[i].register_forward_pre_hook(mk(i), with_kwargs=True)
            for i in range(n_layers)
        ]
        try:
            out = model(**enc, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        if acts is None:
            acts = torch.zeros(n, n_layers, store[0].shape[-1], dtype=torch.float32)
        for lyr in range(n_layers):
            acts[start : start + len(batch), lyr] = store[lyr]
        last_logits.append(out.logits[:, -1, :].float().cpu())
        print(f"  vaa prompts {min(start + batch_size, n)}/{n}", flush=True)
    return acts, torch.cat(last_logits)


def stage_vaa(args, out_dir: Path):
    from vaa.extract_vaa import PROMPT_TEMPLATE, per_layer_pc1_with_metrics

    get_model_layers = load_get_model_layers()

    statements = json.load(open(Path(args.welfare_repo) / "vaa/data/statements.json"))
    print(f"{len(statements)} VAA statements")
    model, tokenizer = load_model_merged(None, None)
    layers = list(get_model_layers(model))
    print(f"{len(layers)} layers")

    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEMPLATE.format(statement=s["statement"])}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for s in statements
    ]
    acts, logits = capture_last_token_activations(
        model, tokenizer, texts, layers, batch_size=args.batch_size
    )

    def tid(s):
        ids = tokenizer.encode(s, add_special_tokens=False)
        assert len(ids) == 1, (s, ids)
        return ids[0]

    a_id, b_id = tid("A"), tid("B")
    delta = logits[:, a_id] - logits[:, b_id]
    label = (delta > 0).long()
    pc1, auroc, cohens_d, overlap = per_layer_pc1_with_metrics(acts, delta, label)

    vaa_dir = out_dir / "vaa"
    vaa_dir.mkdir(parents=True, exist_ok=True)
    torch.save(pc1.unsqueeze(0), vaa_dir / "mean_diff.pt")
    torch.save(acts, vaa_dir / "activations.pt")
    json.dump(
        {"layers": list(range(len(layers))), "auroc": auroc, "cohens_d": cohens_d, "overlap": overlap},
        open(vaa_dir / "metrics.json", "w"),
    )
    # Per-layer stats of the statement projections — lets downstream normalize
    # rollout projections into the axis's own units.
    proj_stats = {}
    for lyr in range(len(layers)):
        proj = (acts[:, lyr, :].double() @ pc1[lyr]).numpy()
        proj_stats[lyr] = {"mean": float(proj.mean()), "std": float(proj.std())}
    json.dump(proj_stats, open(vaa_dir / "proj_stats.json", "w"))
    json.dump(
        {"base_model": BASE_MODEL, "n_statements": len(statements), "seed": SEED},
        open(vaa_dir / "metadata.json", "w"),
    )
    best = int(np.nanargmax(np.array(auroc, dtype=float)))
    print(f"VAA axis written; best layer {best} (AUROC {auroc[best]:.3f})")


def encode_variant(spec, records, n_turns, d_model, get_model_layers) -> torch.Tensor:
    """Teacher-forced re-encode of one variant's conversations.

    Isolated in a function so the 54GB model and every module reference die
    with the local scope before the next variant loads (frees GPU memory).
    float32 storage: gemma-27b late-layer activations exceed fp16 range.
    """
    import gc

    model, tokenizer = load_model_merged(spec["adapter"], None)
    layers = list(get_model_layers(model))
    n_layers = len(layers)
    device = next(model.parameters()).device

    turn_means = torch.zeros(len(records), n_turns, n_layers, d_model, dtype=torch.float32)
    for ci, rec in enumerate(records):
        msgs = rec["messages"]
        assert len(msgs) == 2 * n_turns, len(msgs)
        full_ids, spans = assistant_turn_spans(msgs, tokenizer)
        out = torch.zeros(n_turns, n_layers, d_model, dtype=torch.float32)
        handles = [
            layers[i].register_forward_pre_hook(
                make_span_mean_hook(i, spans, out), with_kwargs=True
            )
            for i in range(n_layers)
        ]
        try:
            ids = torch.tensor([full_ids], device=device)
            with torch.no_grad():
                model(input_ids=ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        turn_means[ci] = out
        if (ci + 1) % 10 == 0:
            print(f"  conv {ci + 1}/{len(records)}", flush=True)

    del layers, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return turn_means


def stage_rollouts(args, out_dir: Path):
    from huggingface_hub import hf_hub_download

    get_model_layers = load_get_model_layers()

    runs = json.load(open(args.runs_json))
    vaa_dir = out_dir / "vaa"
    axis = torch.load(vaa_dir / "mean_diff.pt", map_location="cpu", weights_only=True)[0]
    axis = (axis / axis.norm(dim=-1, keepdim=True)).double()
    metrics = json.load(open(vaa_dir / "metrics.json"))
    auroc = {int(l): a for l, a in zip(metrics["layers"], metrics["auroc"])}
    best_layer = max(auroc, key=lambda k: (auroc[k] if auroc[k] == auroc[k] else -1))
    print(f"axis {tuple(axis.shape)}, best layer {best_layer} (AUROC {auroc[best_layer]:.3f})")

    results = {
        "base_model": BASE_MODEL,
        "category": FRUSTRATION_CATEGORY,
        "best_layer": best_layer,
        "auroc_best_layer": auroc[best_layer],
        "seed": SEED,
        "variants": {},
    }
    for name, spec in runs.items():
        print(f"\n=== variant: {name} ({spec['run']}) ===", flush=True)
        loc = hf_hub_download(
            HF_DATASET_REPO,
            f"{FRUSTRATION_PREFIX}/{spec['run']}/{FRUSTRATION_CATEGORY}/results.jsonl",
            repo_type="dataset",
        )
        records = [json.loads(l) for l in open(loc)]
        if args.max_convs:
            records = records[: args.max_convs]
        print(f"{len(records)} conversations")

        n_turns = len(records[0]["turn_results"])
        judge = np.array(
            [[t["frustration_score"] for t in rec["turn_results"]] for rec in records]
        )
        tm_path = out_dir / f"turn_means_{name}.pt"
        if tm_path.exists():
            print(f"resuming from {tm_path}")
            turn_means = torch.load(tm_path, map_location="cpu", weights_only=True).float()
        else:
            turn_means = encode_variant(spec, records, n_turns, axis.shape[-1], get_model_layers)
            torch.save(turn_means, tm_path)
        proj = torch.einsum("ctld,ld->ctl", turn_means.double(), axis).numpy()  # (convs, turns, layers)
        results["variants"][name] = {
            "run": spec["run"],
            "adapter": spec["adapter"],
            "n_convs": len(records),
            "proj_best_layer": proj[:, :, best_layer].tolist(),
            "proj_mean_per_layer_turn": proj.mean(axis=0).tolist(),
            "judge_frustration": judge.tolist(),
        }
        print(
            f"{name}: per-turn proj@L{best_layer} "
            + " ".join(f"{v:+.2f}" for v in proj[:, :, best_layer].mean(axis=0))
        )

    json.dump(results, open(out_dir / "results_needs_help.json", "w"))
    print(f"\nWrote {out_dir / 'results_needs_help.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["vaa", "rollouts", "all"], required=True)
    ap.add_argument("--welfare-repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs-json", default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-convs", type=int, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.welfare_repo).resolve()))
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("vaa", "all"):
        stage_vaa(args, out_dir)
    if args.stage in ("rollouts", "all"):
        assert args.runs_json, "--runs-json required for rollouts stage"
        stage_rollouts(args, out_dir)


if __name__ == "__main__":
    main()
