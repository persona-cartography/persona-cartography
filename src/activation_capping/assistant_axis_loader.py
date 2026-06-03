"""Bridge between the upstream Assistant Axis pipeline and our rollout infra.

The upstream pipeline from ``safety-research/assistant-axis`` saves the axis as a raw
tensor of shape ``(n_layers, hidden_dim)`` and provides
``ActivationSteering`` (a context manager that registers PyTorch forward
hooks for capping). This module converts the upstream axis + activation
cache into a per-layer threshold config and applies activation capping in
a way that is faithful to the paper's Eq. 1.

────────────────────────────────────────────────────────────────────────────
SIGN CONVENTION + CLAMP DIRECTION (read this before changing anything)
────────────────────────────────────────────────────────────────────────────

Paper Eq. 1 (Lu et al., 2026, page 13):

    h ← h − v · min(⟨h, v⟩ − τ, 0)                 (Eq. 1)

This is a FLOOR clamp: projections below τ get lifted to τ; above τ they
are unchanged. Paper text: "clamps the component of h along the Assistant
Axis to a minimum of τ".

Paper page 6 states ``axis = default_assistant − mean(role_vectors)``,
which means *positive projection ⇒ Assistant-like*. This is also the
convention produced by upstream ``pipeline/5_axis.py``.

So in OUR convention (axis = default − role, positive = Assistant), the
faithful intervention is FLOOR at a high threshold: prevent the model's
projection from drifting *below* a typical-Assistant value.

WARNING: the upstream library code in
``assistant_axis/steering.py:_apply_cap`` is a
CEILING clamp (``clamp(proj − τ, min=0)``), and upstream's
*pre-published* axis vectors at ``lu-christina/assistant-axis-vectors``
were generated with the OPPOSITE sign convention (axis = role − default,
positive = role-like) — verifiable by inspecting their config: the cap
thresholds at the recommended ``layers_56:71-p0.25`` setting are NEGATIVE
(e.g. −1.76 at layer 56). Combining their ceiling clamp with the
opposite-signed published axis at the p25 cap reproduces the paper's
intent.

Our ``5_axis.py``-built axis matches the paper *text* convention
(default − role). Combining it with upstream's ceiling clamp would clamp
the *Assistant* end down — the opposite of what the paper wants. We
therefore default to FLOOR mode (paper Eq. 1, applied in our convention)
and compute the threshold percentile against the JOINT (default + role)
projection distribution. The 75th percentile in our convention equals
the paper's 25th percentile in the opposite-sign convention; both
correspond to the same physical threshold in projection space.

A ``mode="ceiling"`` path is preserved for replicating upstream's
published-vector flow, but should be paired with a manual axis sign flip
if you want it to make geometric sense.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from src.activation_capping.assistant_axis_dependency import ensure_assistant_axis_repo

# Ensure the pinned upstream package is importable.
_ASSISTANT_AXIS_DIR = ensure_assistant_axis_repo(quiet=True)
if str(_ASSISTANT_AXIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ASSISTANT_AXIS_DIR))

from assistant_axis.steering import ActivationSteering  # noqa: E402

from src.activation_capping.axis import cohens_d as _shared_cohens_d  # noqa: E402
from src.activation_capping.model import ActivationCappedModel  # noqa: E402


CapMode = Literal["floor", "ceiling"]


# ── Reading axis + activation artefacts ────────────────────────────────────


def load_axis(axis_path: str | Path) -> torch.Tensor:
    """Load the upstream axis tensor of shape ``(n_layers, hidden_dim)``."""
    obj = torch.load(str(axis_path), map_location="cpu", weights_only=False)
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict) and "axis" in obj:
        return obj["axis"].float()
    raise ValueError(f"Unexpected axis file format at {axis_path}: {type(obj)}")


def load_role_activations(activations_dir: Path, role: str) -> torch.Tensor:
    """Load per-sample activations for a role: ``(n_samples, n_layers, hidden_dim)``.

    Upstream stores per-role activation dicts under ``activations_dir/{role}.pt``.
    The dict keys index individual responses; values are ``(n_layers, hidden_dim)``
    tensors. We stack them into a single batch.
    """
    obj = torch.load(activations_dir / f"{role}.pt", map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict in {role}.pt, got {type(obj)}")
    tensors = [v for v in obj.values() if isinstance(v, torch.Tensor)]
    if not tensors:
        raise ValueError(f"No activation tensors found in {role}.pt")
    return torch.stack(tensors).float()


# ── Capping config computation (Phase 2) ───────────────────────────────────


def project_onto_axis(activations: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Per-layer projection of ``(N, n_layers, hidden_dim)`` onto axis.

    Returns ``(N, n_layers)`` — projection magnitude in axis-normalised units.
    """
    ax = axis.float()
    ax_norm = ax / (ax.norm(dim=-1, keepdim=True) + 1e-8)
    return torch.einsum("nld,ld->nl", activations.float(), ax_norm)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Signed Cohen's d between two 1-D samples (a − b convention).

    Thin wrapper around :func:`src.activation_capping.axis.cohens_d`
    pinned to ``signed=True`` — the bridge module's window-selection logic
    is paper-faithful and depends on the sign of (default − role).
    """
    return _shared_cohens_d(a, b, signed=True)


def pick_layer_window(
    default_proj: torch.Tensor,
    role_proj: torch.Tensor,
    *,
    n_layers: int,
    window_size: int,
    search_range: tuple[float, float] = (0.5, 1.0),
) -> tuple[int, int]:
    """Find contiguous layer window of ``window_size`` maximising mean signed Cohen's d.

    With axis = default − role, default proj > role proj at well-formed
    layers, so the signed Cohen's d is positive there. ``search_range``
    constrains the window's centre to a fraction of the layer stack — the
    paper consistently picks windows in the upper half (Llama 3.3 70B
    56:72 = 70-90%).
    """
    lo_frac, hi_frac = search_range
    lo_centre = int(round(lo_frac * n_layers))
    hi_centre = int(round(hi_frac * n_layers))

    best_score = -np.inf
    best_start = max(0, n_layers - window_size)
    for start in range(0, n_layers - window_size + 1):
        centre = start + window_size // 2
        if centre < lo_centre or centre > hi_centre:
            continue
        d_per_layer = [
            cohens_d(default_proj[:, layer_idx].numpy(), role_proj[:, layer_idx].numpy())
            for layer_idx in range(start, start + window_size)
        ]
        score = float(np.mean(d_per_layer))
        if score > best_score:
            best_score = score
            best_start = start
    return best_start, best_start + window_size - 1


def compute_capping_config(
    *,
    axis_path: Path,
    activations_dir: Path,
    output_path: Path,
    threshold_percentile: float = 75.0,
    layer_window: tuple[int, int] | None = None,
    window_size: int | None = None,
    role_sample_cap: int = 10,
    mode: CapMode = "floor",
) -> dict:
    """Compute and save the capping config used by ``apply_assistant_axis_capping``.

    Args:
        axis_path: Path to upstream ``axis.pt`` (raw tensor or wrapped),
            using the convention ``axis = default − role`` (positive =
            Assistant-like) — the convention produced by
            upstream ``pipeline/5_axis.py``.
        activations_dir: Phase 1 ``activations/`` dir with per-role .pt files.
        output_path: Where to save ``capping_config.pt``.
        threshold_percentile: Per-layer threshold = N-th percentile of the
            JOINT (default + role) projection distribution. Default 75.0,
            which under our axis convention corresponds to the paper's
            "p25" calibration in the opposite sign convention. See module
            docstring for the equivalence proof.
        layer_window: Optional explicit ``(lo, hi)`` inclusive layer window.
            If None, auto-pick by Cohen's d sweep within the upper half.
        window_size: When auto-picking, the window length. Default =
            ``max(4, n_layers // 4)`` (top-quarter analog).
        role_sample_cap: Cap how many roles to load (each contributes up
            to ~hundreds of activations); 10 keeps memory modest.
        mode: ``"floor"`` (paper Eq. 1, default) or ``"ceiling"`` (upstream
            ``_apply_cap`` semantics — only correct if you also flip the
            axis sign).

    Returns:
        The saved config dict.
    """
    axis = load_axis(axis_path)
    n_layers, hidden_dim = axis.shape

    default_acts = load_role_activations(activations_dir, "default")  # (N, L, H)
    default_proj = project_onto_axis(default_acts, axis)  # (N, L)

    role_files = sorted(p for p in activations_dir.glob("*.pt") if p.stem != "default")
    role_files = role_files[:role_sample_cap]
    role_projs: list[torch.Tensor] = []
    for f in role_files:
        try:
            acts = load_role_activations(activations_dir, f.stem)
            role_projs.append(project_onto_axis(acts, axis))
        except Exception as exc:  # noqa: BLE001
            print(f"  skipping role {f.stem}: {exc}")
    if not role_projs:
        raise RuntimeError("No role activation files loaded — cannot compute Cohen's d.")
    role_proj = torch.cat(role_projs, dim=0)  # (sum_N, L)

    # Layer window selection.
    if layer_window is not None:
        lo, hi = layer_window
        window_source = "user_override"
    else:
        if window_size is None:
            window_size = max(4, n_layers // 4)
        lo, hi = pick_layer_window(
            default_proj, role_proj,
            n_layers=n_layers, window_size=window_size,
        )
        window_source = "auto_cohens_d"

    # Sanity-print: window centre as a fraction of the layer stack, vs
    # the paper's analog (Llama 3.3 70B used 56:71 of 80 = 70-89% depth;
    # Qwen 3 32B used 46:53 of 64 = 72-83%). Auto-picked windows on Llama
    # 3.1 8B should land somewhere in the same fractional region.
    PAPER_DEPTH_FRACTION_RANGE = (0.70, 0.90)
    centre_frac = ((lo + hi) / 2) / max(n_layers - 1, 1)
    paper_lo, paper_hi = PAPER_DEPTH_FRACTION_RANGE
    in_range = paper_lo <= centre_frac <= paper_hi
    flag = "" if in_range else "  ← NOT in paper's depth range; double-check."
    print(f"  Window: layers {lo}:{hi} of {n_layers} "
          f"(centre {centre_frac:.0%} of stack; paper analog {paper_lo:.0%}-{paper_hi:.0%}; "
          f"source={window_source}){flag}")

    # Per-layer threshold = N-th percentile of the JOINT (default+role)
    # projection distribution at that layer. Joint matches the paper's
    # calibration dataset; paper's p25 (in role-positive convention) ==
    # our p75 (in default-positive convention) — see module docstring.
    joint_proj = torch.cat([default_proj, role_proj], dim=0)  # (N_default+N_role, L)
    thresholds = {
        layer: float(np.percentile(joint_proj[:, layer].numpy(), threshold_percentile))
        for layer in range(lo, hi + 1)
    }

    # Diagnostics.
    d_per_layer = {
        layer_idx: cohens_d(
            default_proj[:, layer_idx].numpy(),
            role_proj[:, layer_idx].numpy(),
        )
        for layer_idx in range(n_layers)
    }
    config = {
        "layers": list(range(lo, hi + 1)),
        "thresholds": thresholds,  # dict[layer_idx, tau]
        "axis_path": str(axis_path),
        "axis_convention": "default_minus_role",  # positive = Assistant
        "mode": mode,
        "threshold_percentile": threshold_percentile,
        "threshold_distribution": "joint_default_and_role",
        "n_default_samples": int(default_acts.shape[0]),
        "n_role_samples": int(role_proj.shape[0]),
        "n_layers_total": n_layers,
        "hidden_dim": hidden_dim,
        "cohens_d_per_layer": d_per_layer,
        "layer_window": (lo, hi),
        "layer_window_source": window_source,
        "layer_window_centre_fraction": float(centre_frac),
        "paper_depth_fraction_range": list(PAPER_DEPTH_FRACTION_RANGE),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(config, output_path)

    summary = {
        "layers": config["layers"],
        "thresholds": {str(k): v for k, v in thresholds.items()},
        "axis_path": config["axis_path"],
        "axis_convention": config["axis_convention"],
        "mode": mode,
        "threshold_percentile": threshold_percentile,
        "threshold_distribution": config["threshold_distribution"],
        "layer_window": list(config["layer_window"]),
        "layer_window_source": window_source,
        "layer_window_centre_fraction": float(centre_frac),
        "paper_depth_fraction_range": list(PAPER_DEPTH_FRACTION_RANGE),
        "cohens_d_in_window_mean": float(
            np.mean([d_per_layer[layer_idx] for layer_idx in config["layers"]])
        ),
        "n_default_samples": config["n_default_samples"],
        "n_role_samples": config["n_role_samples"],
    }
    with open(output_path.with_suffix(".json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved capping config: layers {lo}-{hi}, mode={mode}, "
          f"thresholds@p{threshold_percentile:g} of joint distribution, "
          f"mean Cohen's d in window = {summary['cohens_d_in_window_mean']:.3f}.")
    return config


def load_capping_config(path: Path) -> dict:
    """Load a saved capping config dict."""
    cfg = torch.load(str(path), map_location="cpu", weights_only=False)
    # Back-compat: older configs may not have these fields.
    cfg.setdefault("mode", "floor")
    cfg.setdefault("axis_convention", "default_minus_role")
    cfg.setdefault("threshold_distribution", "unknown")
    return cfg


# ── Hook installation (Phase 3) ────────────────────────────────────────────


def apply_assistant_axis_capping(
    model,
    axis: torch.Tensor,
    capping_config: dict,
    *,
    mode: CapMode | None = None,
    debug: bool = False,
) -> "ActivationCappedModel | ActivationSteering":
    """Register persistent forward hooks for activation capping on ``model``.

    Default mode is ``"floor"`` (paper Eq. 1), routed through our
    :class:`~src.activation_capping.model.ActivationCappedModel`.
    ``"ceiling"`` is preserved for replicating upstream's published-vector
    behaviour and goes through upstream's
    :class:`assistant_axis.steering.ActivationSteering`.

    Args:
        model: HuggingFace ``AutoModelForCausalLM`` (plain, not vLLM).
        axis: ``(n_layers, hidden_dim)`` axis tensor (same convention as
            ``capping_config["axis_convention"]``).
        capping_config: Output of :func:`compute_capping_config`.
        mode: Override the mode stored in ``capping_config``. Use only if
            you understand the sign-convention implications.
        debug: If True, prints hook installation details.

    Returns:
        An object whose ``.remove_hooks()`` (floor mode) or ``.remove()``
        (ceiling mode) method detaches the hooks. Caller must hold a
        reference to the returned object for the hook lifetime.
    """
    layers: list[int] = capping_config["layers"]
    thresholds_dict: dict[int, float] = capping_config["thresholds"]
    resolved_mode = mode or capping_config.get("mode", "floor")

    if resolved_mode == "floor":
        # Paper Eq. 1: lifts below-threshold projections up to threshold.
        capped = ActivationCappedModel(
            model,
            axis=axis,
            layer_thresholds=thresholds_dict,
            mode="floor",
        )
        if debug:
            print(f"[apply_assistant_axis_capping] FLOOR mode on layers {layers} "
                  f"(thresholds: {[round(thresholds_dict[layer_idx], 4) for layer_idx in layers]})")
        return capped

    if resolved_mode == "ceiling":
        steering_vectors = [axis[layer_idx] for layer_idx in layers]
        cap_thresholds = [thresholds_dict[layer_idx] for layer_idx in layers]
        steering = ActivationSteering(
            model,
            steering_vectors=steering_vectors,
            coefficients=[1.0] * len(layers),
            layer_indices=layers,
            intervention_type="capping",
            cap_thresholds=cap_thresholds,
            positions="all",
            debug=debug,
        )
        steering.__enter__()  # noqa: PLC2801 — persistent hooks; caller owns lifetime
        if debug:
            print(f"[apply_assistant_axis_capping] CEILING mode (upstream) on layers {layers}")
        return steering

    raise ValueError(f"Unknown mode: {resolved_mode!r}")


def remove_capping_hooks(handle) -> None:
    """Detach hooks from either floor or ceiling capping handle."""
    if hasattr(handle, "remove_hooks"):  # ActivationCappedModel
        handle.remove_hooks()
    elif hasattr(handle, "remove"):  # ActivationSteering
        handle.remove()
    else:
        raise TypeError(f"Unknown handle type: {type(handle)}")


# ── Diagnostic: verify clamp direction empirically (Task #2) ──────────────


def diagnose_capping_direction(
    model,
    tokenizer,
    capped_handle,
    *,
    axis: torch.Tensor,
    capping_config: dict,
    sample_messages: list[list[dict]] | None = None,
    max_new_tokens: int = 16,
) -> dict:
    """Run one short forward pass with hooks active and report per-layer
    pre/post projection means at the capped layers.

    For ``mode="floor"``: post-projection means should be ≥ pre-projection
    means at every capped layer (where pre fell below the threshold).

    For ``mode="ceiling"``: post means should be ≤ pre means.

    This catches sign/direction bugs in <1 second of GPU time. Run it
    BEFORE spending GPU hours on a full Phase 3.

    Args:
        model: The (already-capped) HF model.
        tokenizer: Matching tokenizer.
        capped_handle: Object returned by ``apply_assistant_axis_capping``
            (has ``.remove_hooks()`` or ``.remove()``).
        axis: The same axis tensor used to register the hooks.
        capping_config: The same config used to register the hooks.
        sample_messages: List of conversation message-lists to forward.
            If None, a small built-in default is used.
        max_new_tokens: Unused (we only need a forward pass), kept for
            API symmetry.

    Returns:
        dict with per-layer ``pre_mean``, ``post_mean``, ``threshold``,
        ``mode`` and a ``passed`` boolean (True if direction is correct).
    """
    import torch

    if sample_messages is None:
        sample_messages = [
            [{"role": "user",
              "content": "Tell me about yourself in one sentence."}],
            [{"role": "system",
              "content": "You are a wise oracle who speaks in riddles."},
             {"role": "user",
              "content": "What is the meaning of suffering?"}],
        ]

    layers: list[int] = capping_config["layers"]
    thresholds: dict[int, float] = capping_config["thresholds"]
    mode: str = capping_config.get("mode", "floor")

    ax = axis.float()
    ax_norm_per_layer = {
        layer_idx: (
            ax[layer_idx] / (ax[layer_idx].norm() + 1e-8)
        ).to(next(model.parameters()).device)
        for layer_idx in layers
    }

    # Capture pre-hook (raw, not clamped) and post-hook (clamped) at each layer.
    pre_acts: dict[int, list[torch.Tensor]] = {layer_idx: [] for layer_idx in layers}
    post_acts: dict[int, list[torch.Tensor]] = {layer_idx: [] for layer_idx in layers}

    # The capping hook is already registered on the layer; we add a
    # PRE-hook to capture inputs and a POST-hook to capture outputs.
    from src.activation_capping.model import get_model_layers
    model_layers = get_model_layers(model)
    handles: list = []

    def make_pre_hook(layer_idx: int):
        def fn(_module, inputs):
            x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            if isinstance(x, torch.Tensor):
                pre_acts[layer_idx].append(x.detach().float().cpu())
        return fn

    def make_post_hook(layer_idx: int):
        def fn(_module, _inputs, output):
            x = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(x, torch.Tensor):
                post_acts[layer_idx].append(x.detach().float().cpu())
        return fn

    for layer_idx in layers:
        handles.append(
            model_layers[layer_idx].register_forward_pre_hook(make_pre_hook(layer_idx))
        )
        handles.append(
            model_layers[layer_idx].register_forward_hook(make_post_hook(layer_idx))
        )

    try:
        # Forward pass on each sample (no generation; we only need acts).
        orig_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Llama 3.1 8B Instruct (and every Instruct/Chat tokenizer we use
        # in this experiment) ships a chat_template — so the apply_chat_template
        # branch is what runs in production. The fallback is only here for
        # base-model tokenizers used in unit tests (e.g. SmolLM-135M); we
        # never go through the fallback during a real Phase 3 run.
        has_chat_template = getattr(tokenizer, "chat_template", None) is not None
        for messages in sample_messages:
            if has_chat_template:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            else:
                text = "\n".join(
                    f"[{m.get('role', 'user').upper()}] {m.get('content', '')}"
                    for m in messages
                ) + "\n[ASSISTANT]"
            enc = tokenizer(
                text, return_tensors="pt", add_special_tokens=False,
            ).to(next(model.parameters()).device)
            with torch.inference_mode():
                model(**enc)
        tokenizer.padding_side = orig_padding_side
    finally:
        for h in handles:
            h.remove()

    # Compute per-layer mean projection over all token positions across all samples.
    report: dict = {"layers": {}, "mode": mode, "passed": True}
    for layer_idx in layers:
        if not pre_acts[layer_idx] or not post_acts[layer_idx]:
            continue
        pre_cat = torch.cat([t.view(-1, t.shape[-1]) for t in pre_acts[layer_idx]], dim=0)
        post_cat = torch.cat([t.view(-1, t.shape[-1]) for t in post_acts[layer_idx]], dim=0)
        v = ax_norm_per_layer[layer_idx].cpu()
        pre_proj = (pre_cat @ v).numpy()
        post_proj = (post_cat @ v).numpy()
        tau = thresholds[layer_idx]
        layer_report = {
            "threshold": float(tau),
            "pre_min": float(pre_proj.min()),
            "pre_mean": float(pre_proj.mean()),
            "pre_max": float(pre_proj.max()),
            "post_min": float(post_proj.min()),
            "post_mean": float(post_proj.mean()),
            "post_max": float(post_proj.max()),
        }
        # Direction check.
        # Bf16 model weights round-trip the corrected residual back to bf16,
        # which loses ~2-3 ulp of precision relative to the fp32 cap math.
        # Re-projecting the bf16-quantised post-state can therefore land a
        # few percent of positions below τ even when the cap is correct.
        # We accept the cap as 'directionally correct' if the FRACTION of
        # mis-clamped positions dropped substantially (≤ 5% of pre-violations
        # remain) and the magnitude of any residual violation is small.
        TOL = 0.1
        VIOLATION_FRACTION_LIMIT = 0.05
        if mode == "floor":
            pre_violations = (pre_proj < tau).sum()
            post_violations = (post_proj < tau - TOL).sum()
            ok = (
                post_proj.min() >= tau - TOL
                or (pre_violations > 0
                    and post_violations / pre_violations <= VIOLATION_FRACTION_LIMIT)
            )
            layer_report["pre_below_tau_count"] = int(pre_violations)
            layer_report["post_below_tau_count"] = int(post_violations)
        else:  # ceiling
            pre_violations = (pre_proj > tau).sum()
            post_violations = (post_proj > tau + TOL).sum()
            ok = (
                post_proj.max() <= tau + TOL
                or (pre_violations > 0
                    and post_violations / pre_violations <= VIOLATION_FRACTION_LIMIT)
            )
            layer_report["pre_above_tau_count"] = int(pre_violations)
            layer_report["post_above_tau_count"] = int(post_violations)
        layer_report["direction_ok"] = bool(ok)
        report["layers"][layer_idx] = layer_report
        if not ok:
            report["passed"] = False

    return report


def print_capping_diagnosis(report: dict) -> None:
    """Pretty-print the diagnostic report. Use after diagnose_capping_direction."""
    mode = report["mode"]
    print(f"\n=== Capping direction diagnostic (mode={mode}) ===")
    if mode == "floor":
        print("  layer | tau     | pre[min/mean/max]    | post[min/mean/max]   | "
              "below-τ pre→post | ok")
    else:
        print("  layer | tau     | pre[min/mean/max]    | post[min/mean/max]   | "
              "above-τ pre→post | ok")
    for layer_idx, r in report["layers"].items():
        if mode == "floor":
            viol = f"{r.get('pre_below_tau_count', 0):>4}→{r.get('post_below_tau_count', 0):<4}"
        else:
            viol = f"{r.get('pre_above_tau_count', 0):>4}→{r.get('post_above_tau_count', 0):<4}"
        print(
            f"  {layer_idx:5d} | {r['threshold']:+7.3f} | "
            f"{r['pre_min']:+5.2f}/{r['pre_mean']:+5.2f}/{r['pre_max']:+5.2f} | "
            f"{r['post_min']:+5.2f}/{r['post_mean']:+5.2f}/{r['post_max']:+5.2f} | "
            f"{viol:>15} | "
            f"{'✓' if r['direction_ok'] else '✗'}"
        )
    if not report["passed"]:
        print(f"  WARNING: at least one layer's clamp direction does NOT match "
              f"mode={mode!r}. This usually means the axis sign or "
              "the threshold-distribution convention is wrong. STOP HERE.")
    else:
        print(f"  All capped layers directionally consistent with {mode!r}.")
