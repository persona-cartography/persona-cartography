"""Compute the persona activation axis + per-layer projection ranges for one
OCEAN direction and upload the artifacts to the HuggingFace monorepo next to the
LoRA (under ``<lora_parent>/activation_capping/``).

The LoRA for each direction is resolved from
:class:`src.common.lora_catalogue.LoraHFCatalogue` (the canonical current-best
adapter per OCEAN direction), so this always targets whatever adapter the
catalogue points at. The activation-capping eval configs under
``scripts/personality_evals/configs/ocean/{trait,mmlu}/activation_capping/``
download the ``{persona}_axis.pt`` + ``{persona}_per_layer_range.pt`` produced
here, so this script is what makes that eval data reproducible from scratch.

Runs one persona per invocation. To regenerate all ten OCEAN directions::

    for p in o_plus o_minus c_plus c_minus e_plus e_minus a_plus a_minus n_plus n_minus; do
        uv run python scripts/activation_capping/ocean/compute_axis.py --persona "$p"
    done

Usage
-----
    uv run python scripts/activation_capping/ocean/compute_axis.py \\
        --persona {o_plus,o_minus,c_plus,c_minus,e_plus,e_minus,a_plus,a_minus,n_plus,n_minus} \\
        [--max-samples N] [--dry-run] [--skip-upload] [--force] [--window-size N]

Outputs
-------
Local (``scratch/{scratch_root}/activation_capping/{persona}_{lora_version}/``):
    {persona}_axis.pt               # axis + metadata incl. recommended_capping_layers
    {persona}_per_layer_range.pt    # (min, max) projection range per layer
    {persona}_activations.pt        # raw base/lora activations + responses
    axis_norms_per_layer.png
    relative_axis_norms.png
    projection_boxplots_per_layer.png
    run_info.json                   # provenance

Monorepo (``<lora_parent>/activation_capping/``, sibling of ``lora/`` and ``evals/``):
    All of the above, minus intermediate local-only caches, via ``upload_folder``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.hf_api import RepoFile
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activation_capping.axis import (
    cohens_d_per_layer,
    compute_axis,
    compute_per_layer_range,
    extract_response_activations_batched,
    flatten_rollouts,
    generate_responses_batched,
)
from src.common.lora_catalogue import HF_REPO, LoraHFCatalogue
from src.utils.hf_hub import download_from_dataset_repo, login_from_env

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MONOREPO_ID = HF_REPO

MAX_NEW_TOKENS = 256
BATCH_SIZE = 16
NUM_ROLLOUTS = 5
TEMPERATURE = 1.0
TOP_P: float | None = None
# We cap the last N layers of the stack, where N = round(n_layers * TAIL_FRACTION).
# Cohen's d separation plateaus from ~40% depth onward but axis norms grow
# monotonically, so argmax-by-separation tended to pick only the final few
# layers — those have the strongest base-vs-LoRA separation but the least
# steering leverage on the output (the model has already committed by then).
# Picking the tail by depth gives us a wider, middle-to-late range closer to
# the paper's Pareto-optimal window. Override via --window-size.
TAIL_FRACTION = 0.40
MIN_TAIL_LAYERS = 4
SEED = 42

REPO_ROOT = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
)
DATASET_PATH = (
    REPO_ROOT / "data" / "claude-generated-prompts-for-activations-generations.jsonl"
)

# Per-persona base model + local scratch subdir. The scratch_root just
# namespaces local caches so different base models don't collide; the
# authoritative lookup for HF paths is LoraHFCatalogue, and lora_version is
# derived at runtime from that catalogue path (see _resolve_paths).
_LLAMA_8B = {
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",
    "scratch_root": "llama_8b_instruct",
}
_GEMMA_27B = {
    "base_model": "google/gemma-3-27b-it",
    "scratch_root": "gemma_3_27b_it",
}
PERSONA_CONFIGS: dict[str, dict[str, str]] = {
    "o_plus": _LLAMA_8B,
    "o_minus": _LLAMA_8B,
    "c_plus": _LLAMA_8B,
    "c_minus": _LLAMA_8B,
    "e_plus": _LLAMA_8B,
    "e_minus": _LLAMA_8B,
    "a_plus": _LLAMA_8B,
    "a_minus": _LLAMA_8B,
    "n_plus": _LLAMA_8B,
    "n_minus": _LLAMA_8B,
    "gemma27b_n_minus": _GEMMA_27B,
    "gemma27b_n_plus": _GEMMA_27B,
}
PERSONA_CHOICES = list(PERSONA_CONFIGS.keys())


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _git(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:
        return None


def _git_dirty() -> bool | None:
    out = _git("status", "--porcelain")
    if out is None:
        return None
    return bool(out)


def _axis_files_on_hub(monorepo_upload_path: str, persona: str) -> bool:
    """True if both ``{persona}_axis.pt`` and ``{persona}_per_layer_range.pt``
    already exist under ``monorepo_upload_path`` on the HF monorepo.

    Silent on network or not-found errors (treated as "not present") — the
    caller should fall back to full compute in those cases.
    """
    required = {f"{persona}_axis.pt", f"{persona}_per_layer_range.pt"}
    try:
        api = HfApi()
        entries = api.list_repo_tree(
            repo_id=MONOREPO_ID,
            repo_type="dataset",
            path_in_repo=monorepo_upload_path,
            recursive=False,
        )
        found = {Path(e.path).name for e in entries if isinstance(e, RepoFile)}
    except EntryNotFoundError:
        return False
    except Exception as exc:
        print(f"  WARN: HF existence check failed ({exc!r}); will recompute.")
        return False
    return required.issubset(found)


def _resolve_paths(persona: str) -> tuple[str, str, Path, Path, str, str]:
    """Resolve paths + per-persona model config.

    Returns:
        (lora_path_in_repo, monorepo_axis_upload_path, local_output_dir,
         local_lora_cache, base_model, lora_version)
    """
    cfg = PERSONA_CONFIGS[persona]
    base_model = cfg["base_model"]
    scratch_root = cfg["scratch_root"]

    lora_path_in_repo: str = getattr(LoraHFCatalogue(), persona)
    # OCEAN catalogue entries end in ".../{version}/lora/<adapter-name>";
    # the legacy gemma entry is just ".../{version}". Handle both.
    lora_path_obj = Path(lora_path_in_repo)
    if lora_path_obj.parent.name == "lora":
        lora_parent = str(lora_path_obj.parent.parent)
        lora_version = lora_path_obj.parent.parent.name
    else:
        lora_parent = str(lora_path_obj)
        lora_version = lora_path_obj.name
    monorepo_upload_path = f"{lora_parent}/activation_capping"
    output_dir = (
        REPO_ROOT
        / "scratch"
        / scratch_root
        / "activation_capping"
        / f"{persona}_{lora_version}"
    )
    local_lora_cache = (
        REPO_ROOT / "scratch" / "lora_cache" / f"{persona}_{lora_version}"
    )
    return (
        lora_path_in_repo,
        monorepo_upload_path,
        output_dir,
        local_lora_cache,
        base_model,
        lora_version,
    )


def _load_prompts(max_samples: int | None) -> list[str]:
    with open(DATASET_PATH) as f:
        rows = [json.loads(line) for line in f]
    if max_samples is not None:
        rows = rows[:max_samples]
    return [r["question"] for r in rows]


def _make_plots(
    axis: torch.Tensor,
    base_stack: torch.Tensor,
    lora_stack: torch.Tensor,
    output_dir: Path,
) -> None:
    axis_norms = axis.norm(dim=1).numpy()
    base_mean_norms = base_stack.float().norm(dim=2).mean(dim=0).numpy()
    lora_mean_norms = lora_stack.float().norm(dim=2).mean(dim=0).numpy()
    avg_mean_norms = (base_mean_norms + lora_mean_norms) / 2
    x = np.arange(len(axis_norms))

    # 1. Raw norms
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.25
    ax.bar(
        x - width,
        base_mean_norms,
        width,
        label="Base mean activation norm",
        color="steelblue",
        alpha=0.7,
    )
    ax.bar(
        x,
        lora_mean_norms,
        width,
        label="LoRA mean activation norm",
        color="coral",
        alpha=0.7,
    )
    ax.bar(x + width, axis_norms, width, label="Axis norm", color="green", alpha=0.7)
    ax.set_xlabel("Layer")
    ax.set_ylabel("L2 norm")
    ax.set_title("Per-layer norms: activations vs axis")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "axis_norms_per_layer.png", dpi=150)
    plt.close(fig)

    # 2. Norm ratios
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(
        x,
        axis_norms / (base_mean_norms + 1e-8),
        marker="o",
        markersize=3,
        label="Axis / base norm",
        color="steelblue",
    )
    ax.plot(
        x,
        axis_norms / (lora_mean_norms + 1e-8),
        marker="s",
        markersize=3,
        label="Axis / LoRA norm",
        color="coral",
    )
    ax.plot(
        x,
        axis_norms / (avg_mean_norms + 1e-8),
        marker="^",
        markersize=3,
        label="Axis / avg norm",
        color="green",
        linewidth=2,
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Axis norm / activation norm")
    ax.set_title("Relative persona signal strength per layer")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "relative_axis_norms.png", dpi=150)
    plt.close(fig)

    # 3. Normalized projection boxplots per layer
    n_layers = axis.shape[0]
    base_projs = []
    lora_projs = []
    for layer_idx in range(n_layers):
        ax_vec = axis[layer_idx].float()
        ax_normed = ax_vec / (ax_vec.norm() + 1e-8)
        proj_base = (base_stack[:, layer_idx, :].float() @ ax_normed).numpy()
        proj_lora = (lora_stack[:, layer_idx, :].float() @ ax_normed).numpy()
        base_norm_mean = base_stack[:, layer_idx, :].float().norm(dim=-1).mean().item()
        lora_norm_mean = lora_stack[:, layer_idx, :].float().norm(dim=-1).mean().item()
        b = proj_base / base_norm_mean
        l = proj_lora / lora_norm_mean
        center = (b.mean() + l.mean()) / 2
        base_projs.append(b - center)
        lora_projs.append(l - center)

    positions_base = np.arange(n_layers) * 3
    positions_lora = positions_base + 1
    fig, ax = plt.subplots(figsize=(20, 6))
    bp_base = ax.boxplot(
        base_projs,
        positions=positions_base,
        widths=0.8,
        patch_artist=True,
        showfliers=False,
    )
    bp_lora = ax.boxplot(
        lora_projs,
        positions=positions_lora,
        widths=0.8,
        patch_artist=True,
        showfliers=False,
    )
    for patch in bp_base["boxes"]:
        patch.set_facecolor("cornflowerblue")
        patch.set_alpha(0.7)
    for patch in bp_lora["boxes"]:
        patch.set_facecolor("salmon")
        patch.set_alpha(0.7)
    ax.set_xticks(positions_base + 0.5)
    ax.set_xticklabels([str(i) for i in range(n_layers)], fontsize=8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized projection (proj / mean activation norm)")
    ax.set_title("Normalized projection onto axis — Base vs LoRA per layer")
    ax.legend(
        [bp_base["boxes"][0], bp_lora["boxes"][0]], ["Base", "LoRA"], loc="upper left"
    )
    plt.tight_layout()
    plt.savefig(output_dir / "projection_boxplots_per_layer.png", dpi=150)
    plt.close(fig)


def run(
    persona: str,
    *,
    max_samples: int | None,
    dry_run: bool,
    skip_upload: bool,
    force: bool,
    window_size_override: int | None,
) -> None:
    _set_seeds(SEED)
    login_from_env()

    (
        lora_path_in_repo,
        monorepo_upload_path,
        output_dir,
        local_lora_cache,
        base_model,
        lora_version,
    ) = _resolve_paths(persona)

    print("=" * 70)
    print(f"  Persona:               {persona}")
    print(f"  Base model:            {base_model}")
    print(f"  LoRA version:          {lora_version}")
    print(f"  LoRA repo:             {MONOREPO_ID}")
    print(f"  LoRA path_in_repo:     {lora_path_in_repo}")
    print(f"  Monorepo upload path:  {monorepo_upload_path}")
    print(f"  Local output dir:      {output_dir}")
    print(f"  Local LoRA cache:      {local_lora_cache}")
    print(f"  Dataset:               {DATASET_PATH.relative_to(REPO_ROOT)}")
    print(f"  max_samples:           {max_samples}")
    print("=" * 70)

    if dry_run:
        print("[--dry-run] exiting before generation / upload.")
        return

    # Skip if axis artifacts already exist on the monorepo (unless --force).
    if not force and _axis_files_on_hub(monorepo_upload_path, persona):
        print(
            f"[skip] {persona}_axis.pt and {persona}_per_layer_range.pt already "
            f"exist under {monorepo_upload_path} on the monorepo. "
            f"Pass --force to recompute."
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset
    questions = _load_prompts(max_samples)
    print(f"Loaded {len(questions)} prompts")

    # LoRA adapter download
    download_from_dataset_repo(
        repo_id=MONOREPO_ID,
        path_in_repo=lora_path_in_repo,
        local_dir=local_lora_cache,
    )
    lora_local_path = local_lora_cache / lora_path_in_repo
    print(f"LoRA downloaded to {lora_local_path}")

    # Model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(lora_local_path))
    model.eval()

    # Phase 1: base rollouts
    print(f"\nPhase 1/4: base rollouts ({NUM_ROLLOUTS} x {len(questions)} questions)")
    with model.disable_adapter():
        base_rollouts = generate_responses_batched(
            model,
            tokenizer,
            questions,
            max_new_tokens=MAX_NEW_TOKENS,
            batch_size=BATCH_SIZE,
            num_rollouts=NUM_ROLLOUTS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )

    # Phase 2: LoRA rollouts
    print(f"Phase 2/4: LoRA rollouts ({NUM_ROLLOUTS} x {len(questions)} questions)")
    lora_rollouts = generate_responses_batched(
        model,
        tokenizer,
        questions,
        max_new_tokens=MAX_NEW_TOKENS,
        batch_size=BATCH_SIZE,
        num_rollouts=NUM_ROLLOUTS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    base_qs_flat, base_resps_flat = flatten_rollouts(questions, base_rollouts)
    lora_qs_flat, lora_resps_flat = flatten_rollouts(questions, lora_rollouts)

    base_convs = [
        [{"role": "user", "content": q}, {"role": "assistant", "content": r}]
        for q, r in zip(base_qs_flat, base_resps_flat)
    ]
    lora_convs = [
        [{"role": "user", "content": q}, {"role": "assistant", "content": r}]
        for q, r in zip(lora_qs_flat, lora_resps_flat)
    ]

    # Phase 3: base activations
    print("Phase 3/4: base activations")
    with model.disable_adapter():
        base_stack = extract_response_activations_batched(
            model,
            tokenizer,
            base_convs,
            batch_size=BATCH_SIZE,
        )
    print(f"  base activations: {base_stack.shape}")

    # Phase 4: LoRA activations
    print("Phase 4/4: LoRA activations")
    lora_stack = extract_response_activations_batched(
        model,
        tokenizer,
        lora_convs,
        batch_size=BATCH_SIZE,
    )
    print(f"  LoRA activations: {lora_stack.shape}")

    # Axis, Cohen's d (analysis only), tail-depth layer selection
    axis = compute_axis(base_stack, lora_stack)
    cohens_d = cohens_d_per_layer(base_stack, lora_stack, axis)
    n_layers = int(axis.shape[0])
    if window_size_override is not None:
        window_size = window_size_override
    else:
        window_size = max(MIN_TAIL_LAYERS, int(round(n_layers * TAIL_FRACTION)))
    # Cap the last `window_size` layers (tail of the stack) rather than
    # argmax-of-Cohen's-d. Cohen's d plateaus from ~40% depth onward so the
    # argmax heuristic was skewed to the very last few layers (highest axis
    # norm) which have minimal steering leverage — see compute_axis.py
    # module docstring.
    capping_layers = list(range(n_layers - window_size, n_layers))
    best_sep_layer = int(np.argmax(cohens_d))
    print(
        f"\nBest layer by Cohen's d (informational only): {best_sep_layer} "
        f"(d={cohens_d[best_sep_layer]:.3f})"
    )
    print(
        f"Capping layers (tail, window={window_size}, n_layers={n_layers}): "
        f"{capping_layers[0]}-{capping_layers[-1]}"
    )

    # Per-layer projection ranges (all layers)
    all_layers = list(range(axis.shape[0]))
    per_layer_range = compute_per_layer_range(base_stack, lora_stack, axis, all_layers)

    # ----------------------------- Save artifacts -----------------------------
    torch.save(
        {
            "base": base_stack,
            "lora": lora_stack,
            "base_responses": base_resps_flat,
            "lora_responses": lora_resps_flat,
            "base_rollouts": base_rollouts,
            "lora_rollouts": lora_rollouts,
        },
        output_dir / f"{persona}_activations.pt",
    )
    torch.save(
        {
            "axis": axis,
            "metadata": {
                "model": base_model,
                "lora_hf_dataset_repo": MONOREPO_ID,
                "lora_path_in_repo": lora_path_in_repo,
                "persona": persona,
                "n_samples": int(base_stack.shape[0]),
                "n_layers": n_layers,
                "window_size": window_size,
                "best_layer_by_separation": best_sep_layer,
                "recommended_capping_layers": capping_layers,
                "dataset": str(DATASET_PATH),
            },
        },
        output_dir / f"{persona}_axis.pt",
    )
    torch.save(
        {
            "per_layer_range": per_layer_range,
            "metadata": {
                "model": base_model,
                "lora_hf_dataset_repo": MONOREPO_ID,
                "lora_path_in_repo": lora_path_in_repo,
                "persona": persona,
                "n_samples": int(base_stack.shape[0]),
                "layers": all_layers,
                "dataset": str(DATASET_PATH),
            },
        },
        output_dir / f"{persona}_per_layer_range.pt",
    )

    _make_plots(axis, base_stack, lora_stack, output_dir)

    # Provenance
    run_info = {
        "persona": persona,
        "base_model": base_model,
        "lora": {"hf_dataset_repo": MONOREPO_ID, "path_in_repo": lora_path_in_repo},
        "lora_version": lora_version,
        "dataset": str(DATASET_PATH.relative_to(REPO_ROOT)),
        "n_samples": int(base_stack.shape[0]),
        "num_rollouts": NUM_ROLLOUTS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "batch_size": BATCH_SIZE,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "window_size": window_size,
        "tail_fraction": None if window_size_override is not None else TAIL_FRACTION,
        "window_size_override": window_size_override,
        "layer_selection": "tail",
        "n_layers": n_layers,
        "best_layer_by_separation": best_sep_layer,
        "capping_layers_recommended": list(map(int, capping_layers)),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": _git_dirty(),
        },
        "script": "scripts/activation_capping/ocean/compute_axis.py",
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(output_dir / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2, default=str)

    print(f"\nSaved all artifacts under {output_dir}")

    if skip_upload:
        print("[--skip-upload] not uploading to monorepo.")
        return

    # ----------------------------- Upload --------------------------------------
    upload_staging = output_dir / "monorepo_upload"
    if upload_staging.exists():
        shutil.rmtree(upload_staging)
    upload_staging.mkdir(parents=True)

    artifacts = [
        output_dir / f"{persona}_axis.pt",
        output_dir / f"{persona}_per_layer_range.pt",
        output_dir / f"{persona}_activations.pt",
        output_dir / "axis_norms_per_layer.png",
        output_dir / "relative_axis_norms.png",
        output_dir / "projection_boxplots_per_layer.png",
        output_dir / "run_info.json",
    ]
    for p in artifacts:
        if p.exists():
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"  staging {p.name} ({size_mb:.1f} MB)")
            shutil.copy2(p, upload_staging / p.name)

    api = HfApi()
    api.upload_folder(
        folder_path=str(upload_staging),
        repo_id=MONOREPO_ID,
        repo_type="dataset",
        path_in_repo=monorepo_upload_path,
        commit_message=f"activation_capping: add {persona} {lora_version} axis + per-layer ranges",
    )
    print(
        f"\nUploaded to https://huggingface.co/datasets/{MONOREPO_ID}/tree/main/{monorepo_upload_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persona", required=True, choices=PERSONA_CHOICES)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit dataset size for smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve config and exit before generation/upload.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Run everything locally but skip HF upload.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if axis artifacts already exist on the monorepo.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help=(
            "Contiguous capping-window size in layers. "
            "Default: round(n_layers * TAIL_FRACTION), floor MIN_TAIL_LAYERS. "
            "Selected layers are always the last N of the stack."
        ),
    )
    args = parser.parse_args()

    run(
        args.persona,
        max_samples=args.max_samples,
        dry_run=args.dry_run,
        skip_upload=args.skip_upload,
        force=args.force,
        window_size_override=args.window_size,
    )


if __name__ == "__main__":
    main()
