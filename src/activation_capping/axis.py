"""Persona activation-axis helpers.

The pure compute steps (generate rollouts, extract mean activations
per layer, compute axis = mean(lora) - mean(base), per-layer projection ranges,
Cohen's d, best contiguous window). No filesystem, no HF uploads — callers
compose these into a script or notebook as needed.
"""

from __future__ import annotations

import numpy as np
import torch
from tqdm.auto import tqdm

from src.activation_capping.model import get_model_layers


def generate_responses_batched(
    model,
    tokenizer,
    questions: list[str],
    *,
    max_new_tokens: int = 256,
    batch_size: int = 16,
    num_rollouts: int = 3,
    temperature: float = 1.0,
    top_p: float | None = None,
) -> list[list[str]]:
    """Generate multiple sampled responses per question in batches.

    Returns:
        ``responses[i]`` is a list of ``num_rollouts`` responses for ``questions[i]``.
    """
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    all_responses: list[list[str]] = [[] for _ in questions]

    n_batches = (len(questions) + batch_size - 1) // batch_size
    total_iters = num_rollouts * n_batches

    with tqdm(total=total_iters, desc="Generating responses") as pbar:
        for _ in range(num_rollouts):
            for batch_start in range(0, len(questions), batch_size):
                batch_qs = questions[batch_start : batch_start + batch_size]
                convs = [[{"role": "user", "content": q}] for q in batch_qs]
                texts = [
                    tokenizer.apply_chat_template(
                        c, tokenize=False, add_generation_prompt=True
                    )
                    for c in convs
                ]
                enc = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                    return_attention_mask=True,
                ).to(model.device)
                if temperature > 0:
                    sample_kwargs = {
                        "do_sample": True,
                        "temperature": temperature,
                        "top_p": top_p,
                    }
                else:
                    sample_kwargs = {"do_sample": False}
                with torch.inference_mode():
                    output_ids = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        **sample_kwargs,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                for i in range(len(batch_qs)):
                    resp_ids = output_ids[i, enc["input_ids"].shape[1] :]
                    all_responses[batch_start + i].append(
                        tokenizer.decode(resp_ids, skip_special_tokens=True)
                    )
                pbar.update(1)

    tokenizer.padding_side = orig_padding_side
    return all_responses


def flatten_rollouts(
    questions: list[str], rollouts: list[list[str]]
) -> tuple[list[str], list[str]]:
    """Flatten rollouts into parallel ``(questions_flat, responses_flat)``.

    Each question is repeated once per rollout.
    """
    questions_flat: list[str] = []
    responses_flat: list[str] = []
    for q, resps in zip(questions, rollouts):
        for r in resps:
            questions_flat.append(q)
            responses_flat.append(r)
    return questions_flat, responses_flat


def _position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Position ids for left-padded inputs so RoPE starts at 0 at the first real token."""
    return attention_mask.long().cumsum(-1) - 1


def extract_response_activations_batched(
    model,
    tokenizer,
    conversations: list[list[dict[str, str]]],
    *,
    layers: list[int] | None = None,
    batch_size: int = 16,
) -> torch.Tensor:
    """Extract mean activation over assistant response tokens at each layer.

    Args:
        model: HuggingFace causal LM (plain or ``PeftModel``-wrapped).
        tokenizer: Corresponding tokenizer.
        conversations: Each is ``[{"role": "user", ...}, {"role": "assistant", ...}]``.
        layers: Layer indices to extract (default: all).
        batch_size: Conversations per forward pass.

    Returns:
        Tensor of shape ``(N, n_layers, hidden_dim)`` — mean over response tokens per sample.
    """
    model_layers = get_model_layers(model)
    if layers is None:
        layers = list(range(len(model_layers)))

    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    all_results: list[torch.Tensor] = []

    n_batches = (len(conversations) + batch_size - 1) // batch_size

    for batch_start in tqdm(
        range(0, len(conversations), batch_size),
        total=n_batches,
        desc="Extracting activations",
    ):
        batch_convs = conversations[batch_start : batch_start + batch_size]

        full_texts = [
            tokenizer.apply_chat_template(
                c, tokenize=False, add_generation_prompt=False
            )
            for c in batch_convs
        ]
        prefix_texts = [
            tokenizer.apply_chat_template(
                c[:-1], tokenize=False, add_generation_prompt=True
            )
            for c in batch_convs
        ]

        prefix_lens = [
            len(tokenizer(pt, add_special_tokens=False).input_ids)
            for pt in prefix_texts
        ]

        batch_enc = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
            return_attention_mask=True,
        ).to(model.device)

        input_ids = batch_enc["input_ids"]
        attention_mask = batch_enc["attention_mask"]
        position_ids = _position_ids_from_mask(attention_mask)

        unpadded_lens = attention_mask.sum(dim=1).tolist()
        padded_len = input_ids.shape[1]
        response_starts = [
            (padded_len - int(unpadded_lens[i])) + prefix_lens[i]
            for i in range(len(batch_convs))
        ]
        response_ends = [padded_len] * len(batch_convs)

        activations: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(layer_idx: int):
            def hook_fn(_module, _inp, output):
                act = output[0] if isinstance(output, tuple) else output
                means = [
                    act[i, response_starts[i] : response_ends[i], :].mean(dim=0)
                    for i in range(act.shape[0])
                ]
                activations[layer_idx] = torch.stack(means).cpu()

            return hook_fn

        for idx in layers:
            handles.append(model_layers[idx].register_forward_hook(make_hook(idx)))

        try:
            with torch.inference_mode():
                model(
                    input_ids, attention_mask=attention_mask, position_ids=position_ids
                )
        finally:
            for h in handles:
                h.remove()

        batch_result = torch.stack([activations[i] for i in layers], dim=1)
        all_results.append(batch_result)

    tokenizer.padding_side = orig_padding_side
    return torch.cat(all_results, dim=0)


def compute_axis(base_stack: torch.Tensor, lora_stack: torch.Tensor) -> torch.Tensor:
    """``axis = mean(lora) - mean(base)`` per layer. Shape: ``(n_layers, hidden_dim)``."""
    return lora_stack.float().mean(dim=0) - base_stack.float().mean(dim=0)


def compute_per_layer_range(
    base_activations: torch.Tensor,
    lora_activations: torch.Tensor,
    axis: torch.Tensor,
    layers: list[int],
) -> dict[int, tuple[float, float]]:
    """Global ``(min, max)`` of projections onto the axis at each layer, across both sides."""
    ranges: dict[int, tuple[float, float]] = {}
    for layer_idx in layers:
        ax = axis[layer_idx].float()
        ax_normed = ax / (ax.norm() + 1e-8)
        base_proj = (base_activations[:, layer_idx, :].float() @ ax_normed).numpy()
        lora_proj = (lora_activations[:, layer_idx, :].float() @ ax_normed).numpy()
        ranges[layer_idx] = (
            float(min(base_proj.min(), lora_proj.min())),
            float(max(base_proj.max(), lora_proj.max())),
        )
    return ranges


def _project_batch(
    activations: torch.Tensor, axis: torch.Tensor, layer: int
) -> np.ndarray:
    """Project a batch of activations onto the axis at a given layer."""
    acts = activations[:, layer, :].float()
    ax = axis[layer].float()
    ax_normed = ax / (ax.norm() + 1e-8)
    return (acts @ ax_normed).numpy()


def cohens_d(
    a: np.ndarray, b: np.ndarray, *, signed: bool = False, ddof: int = 1
) -> float:
    """Cohen's d between two 1-D samples.

    Args:
        a, b: 1-D arrays of values.
        signed: If True, returns ``(mean(a) − mean(b)) / pooled_sd``. If
            False (default), returns the absolute value — sign-agnostic
            for axis-convention ambiguity.
        ddof: Delta degrees of freedom for variance. ``1`` is the textbook
            unbiased sample variance and matches scipy/numpy convention.

    Returns:
        Effect size (float). 0.0 when pooled std is zero.
    """
    pooled = float(np.sqrt((np.var(a, ddof=ddof) + np.var(b, ddof=ddof)) / 2.0))
    if pooled <= 0:
        return 0.0
    diff = float(np.mean(a) - np.mean(b))
    return (diff if signed else abs(diff)) / pooled


def cohens_d_per_layer(
    base_stack: torch.Tensor,
    lora_stack: torch.Tensor,
    axis: torch.Tensor,
    *,
    signed: bool = False,
) -> np.ndarray:
    """Cohen's d for base-vs-LoRA projection separation at each layer.

    Returns a ``(n_layers,)`` array. ``signed=False`` (default) gives the
    absolute effect size — sign-agnostic with respect to axis convention
    (``base→lora`` or ``lora→base``). ``signed=True`` preserves the
    direction of the effect, which is what window-picking against a
    paper-faithful axis convention should use.
    """
    n_layers = axis.shape[0]
    d = np.zeros(n_layers)
    for layer_idx in range(n_layers):
        proj_base = _project_batch(base_stack, axis, layer_idx)
        proj_lora = _project_batch(lora_stack, axis, layer_idx)
        d[layer_idx] = cohens_d(proj_base, proj_lora, signed=signed)
    return d


def best_contiguous_window(cohens_d: np.ndarray, window_size: int = 15) -> list[int]:
    """Contiguous layer window of size ``window_size`` maximizing the sum of Cohen's d."""
    best_start = 0
    best_score = -np.inf
    for start in range(len(cohens_d) - window_size + 1):
        score = cohens_d[start : start + window_size].sum()
        if score > best_score:
            best_score = score
            best_start = start
    return list(range(best_start, best_start + window_size))


# Re-export for convenience
__all__ = [
    "generate_responses_batched",
    "flatten_rollouts",
    "extract_response_activations_batched",
    "compute_axis",
    "compute_per_layer_range",
    "cohens_d",
    "cohens_d_per_layer",
    "best_contiguous_window",
    "get_model_layers",
]
