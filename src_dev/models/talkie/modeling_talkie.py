"""TalkieForCausalLM — HuggingFace-compatible port of the reference talkie 13B.

Source: ``talkie.model`` in https://github.com/talkie-lm/talkie (Apache-2.0).

The architecture is preserved exactly (HeadGain on Q, per-layer ActGains
on attn/mlp residuals, embed-skip connection in every block). The HF
additions over the reference are: full-sequence logits, ``attention_mask``
support for padded batches, ``past_key_values`` for HF-style incremental
decoding, gradient checkpointing through the standard HF mixin, and a
dispatch through ``ALL_ATTENTION_FUNCTIONS`` so vLLM's ``TransformersBackend``
can swap in its PagedAttention implementation. The reference's
``WeightGain`` on ``lm_head`` is mathematically equivalent to scaling the
``lm_head`` matrix by the same scalar; the materialization step folds it
into ``lm_head.weight`` so vLLM's ``ParallelLMHead`` can load it as a
standard linear weight.

State-dict keys (after folding ``lm_head_gain``):

  model.embed.weight                        (V, D)
  model.blocks.<L>.attn.attn_query.weight   (D, D)
  model.blocks.<L>.attn.attn_key.weight     (D, D)
  model.blocks.<L>.attn.attn_value.weight   (D, D)
  model.blocks.<L>.attn.attn_resid.weight   (D, D)
  model.blocks.<L>.attn.head_gain.head_g    (H,)
  model.blocks.<L>.attn_gain.a_g            (1,)
  model.blocks.<L>.mlp.mlp_gate.weight      (M, D)
  model.blocks.<L>.mlp.mlp_linear.weight    (M, D)
  model.blocks.<L>.mlp.mlp_resid.weight     (D, M)
  model.blocks.<L>.mlp_gain.a_g             (1,)
  model.blocks.<L>.embed_skip.a_g           (1,)
  lm_head.weight                            (V, D)   (= original lm_head * w_g)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel

from .configuration_talkie import TalkieConfig


# ---------------------------------------------------------------------------
# RoPE — NeoX-style (split-in-half), matches reference apply_rotary_emb
# ---------------------------------------------------------------------------


def _build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables shaped ``[1, seq_len, 1, head_dim // 2]``."""
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # [S, D/2]
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos[None, :, None, :], sin[None, :, None, :]


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Reference apply_rotary_emb: split last dim in half, rotate as a 2D vector."""
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = -x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=3).type_as(x)


# ---------------------------------------------------------------------------
# Custom gain layers (match reference checkpoint keys)
# ---------------------------------------------------------------------------


class _HeadGain(nn.Module):
    """Per-head learned scalar on Q (after QK RMSNorm). Reference: HeadGain."""

    def __init__(self, n_head: int):
        super().__init__()
        self.head_g = nn.Parameter(torch.ones(n_head))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, S, H, D]; broadcast head_g over batch / seq / head_dim.
        return x * self.head_g.type_as(x).view(1, 1, -1, 1)


class _ActGain(nn.Module):
    """Scalar learned residual scale. Reference: ActGain."""

    def __init__(self, init_value: float):
        super().__init__()
        self.a_g = nn.Parameter(torch.ones(1) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.a_g.type_as(x)


# ---------------------------------------------------------------------------
# Attention / MLP / Block
# ---------------------------------------------------------------------------


class TalkieAttention(nn.Module):
    """Multi-head self-attention with RoPE, QK RMSNorm, and per-head Q gain.

    Routes the core attention through HF's ``ALL_ATTENTION_FUNCTIONS`` so
    that vLLM's TransformersBackend can replace it with PagedAttention.
    """

    def __init__(self, config: TalkieConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scaling = config.head_dim ** -0.5
        self.is_causal = True

        self.attn_query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.attn_key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.attn_value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.attn_resid = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.head_gain = _HeadGain(self.num_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        q = self.attn_query(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.attn_key(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)
        v = self.attn_value(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim)

        q = _apply_rotary_emb(q, cos, sin)
        k = _apply_rotary_emb(k, cos, sin)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q = self.head_gain(q)

        # vLLM path: TransformersBackend sets _attn_implementation = "vllm" and
        # passes `attention_instances` via kwargs; we dispatch through HF's
        # registry and let vLLM's vllm_flash_attention_forward consume q/k/v.
        attn_impl = getattr(self.config, "_attn_implementation", "sdpa")
        attn_fn = ALL_ATTENTION_FUNCTIONS.get(attn_impl)

        # Reshape to the HF attention dispatcher's expected shape [B, H, S, D].
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        if attn_fn is not None and "attention_instances" in kwargs:
            # vLLM-managed KV cache; ignore past_key_value entirely.
            attn_out, _ = attn_fn(
                self,
                q_t,
                k_t,
                v_t,
                attention_mask,
                scaling=self.scaling,
                **kwargs,
            )
            new_cache = None
        else:
            # Pure-HF path: handle past_key_value + causal mask ourselves.
            if past_key_value is not None:
                past_k, past_v = past_key_value
                k_t = torch.cat([past_k, k_t], dim=2)
                v_t = torch.cat([past_v, v_t], dim=2)
            new_cache = (k_t, v_t) if use_cache else None

            kv_len = k_t.size(2)
            if attention_mask is None and past_key_value is None:
                attn_out = F.scaled_dot_product_attention(
                    q_t, k_t, v_t, is_causal=True, scale=self.scaling
                )
            else:
                attn_bias = torch.zeros(
                    bsz, 1, q_len, kv_len, device=q_t.device, dtype=q_t.dtype
                )
                past_len = kv_len - q_len
                row = torch.arange(q_len, device=q_t.device).view(q_len, 1) + past_len
                col = torch.arange(kv_len, device=q_t.device).view(1, kv_len)
                causal_mask = col > row
                attn_bias.masked_fill_(causal_mask[None, None, :, :], float("-inf"))
                if attention_mask is not None:
                    pad_mask = attention_mask[:, None, None, :] == 0
                    attn_bias.masked_fill_(pad_mask, float("-inf"))
                attn_out = F.scaled_dot_product_attention(
                    q_t, k_t, v_t, attn_mask=attn_bias, is_causal=False,
                    scale=self.scaling,
                )

        # Back to [B, S, H*D] and project out.
        if attn_out.dim() == 4 and attn_out.shape[1] == self.num_heads:
            attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.reshape(bsz, q_len, self.hidden_size)
        return self.attn_resid(attn_out), new_cache


class TalkieMLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, config: TalkieConfig):
        super().__init__()
        self.mlp_gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.mlp_linear = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.mlp_resid = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_resid(F.silu(self.mlp_gate(x)) * self.mlp_linear(x))


class TalkieBlock(nn.Module):
    """Pre-norm attn + pre-norm mlp + per-block embed-skip injection."""

    def __init__(self, config: TalkieConfig, layer_idx: int):
        super().__init__()
        self.attn = TalkieAttention(config, layer_idx=layer_idx)
        self.attn_gain = _ActGain((2 * config.num_hidden_layers) ** -0.5)
        self.mlp = TalkieMLP(config)
        self.mlp_gain = _ActGain((2 * config.num_hidden_layers) ** -0.5)
        self.embed_skip = _ActGain(0.0)

    def forward(
        self,
        x: torch.Tensor,
        e_x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_kv = self.attn(
            F.rms_norm(x, (x.shape[-1],)),
            cos,
            sin,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            **kwargs,
        )
        x = x + self.attn_gain(attn_out)
        x = x + self.mlp_gain(self.mlp(F.rms_norm(x, (x.shape[-1],))))
        x = x + self.embed_skip(e_x)
        return x, new_kv


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------


class TalkiePreTrainedModel(PreTrainedModel):
    """Base class — wires up TalkieConfig + sensible init + vLLM compat flags."""

    config_class = TalkieConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _supports_flash_attn_2 = False
    # vLLM TransformersBackend dispatches via ALL_ATTENTION_FUNCTIONS only when
    # the model claims to support attention-backend swaps.
    _supports_attention_backend = True
    _no_split_modules = ["TalkieBlock"]

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


class TalkieModel(TalkiePreTrainedModel):
    """40-layer decoder stack with embed-skip injection per block."""

    def __init__(self, config: TalkieConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [TalkieBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

        # RoPE: do NOT pre-register a buffer here. vLLM's TransformersBackend
        # initializes the model under ``init_on_device_without_buffers("meta")``
        # which leaves buffers on their original device (CPU), which then
        # crashes Dynamo with a CPU/CUDA mismatch when we index into them.
        # Compute cos/sin on the fly from a tiny ``inv_freq`` derived from
        # config — fast, traceable, and always on the same device as the
        # incoming position_ids.

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed = value

    def _rope_for(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin for ``position_ids`` on the fly.

        Returns tensors shaped ``[1, S, 1, head_dim // 2]`` to match what
        ``_apply_rotary_emb`` expects (broadcast over batch + head dims).
        """
        head_dim = self.config.head_dim
        base = self.config.rope_theta
        device = position_ids.device
        channels = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (base ** (channels / head_dim))
        # position_ids: [B, S] (we already normalized to 2D in forward).
        # freqs[s, d] = position_ids[0, s] * inv_freq[d]
        freqs = position_ids[0].float().unsqueeze(-1) * inv_freq.unsqueeze(0)  # [S, D/2]
        cos = freqs.cos().to(dtype).unsqueeze(0).unsqueeze(2)  # [1, S, 1, D/2]
        sin = freqs.sin().to(dtype).unsqueeze(0).unsqueeze(2)
        return cos, sin

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Pass exactly one of input_ids or inputs_embeds.")
        # vLLM's TransformersBackend passes `attention_instances` via kwargs
        # and manages KV cache itself (PagedAttention). For the pure-HF path
        # we deliberately disable our own HF-style cache and recompute every
        # forward — KV caching with the embed-skip / per-block embedding
        # injection adds enough state to surprise HF's DynamicCache
        # contract and isn't on the hot path for OCT (introspection uses
        # vLLM; DPO/SFT do not use generate; logprob evals run a single
        # forward per prompt). This costs ~Nx for HF .generate() but
        # eliminates a class of cache-management bugs.
        use_vllm_attn = "attention_instances" in kwargs
        if not use_vllm_attn:
            past_key_values = None
            use_cache = False
        if use_cache is None:
            use_cache = not self.training
        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        bsz, seq_len, _ = inputs_embeds.shape

        x = F.rms_norm(inputs_embeds, (inputs_embeds.shape[-1],))
        e_x = x

        # past_key_values may be: None, an empty/new HF Cache, an
        # already-populated HF Cache, or a tuple-of-tuples (legacy).
        past_len = 0
        if past_key_values is not None:
            get_seq_length = getattr(past_key_values, "get_seq_length", None)
            if callable(get_seq_length):
                try:
                    past_len = int(get_seq_length())
                except Exception:
                    past_len = 0
            else:
                try:
                    first = past_key_values[0]
                    if first is not None and first[0] is not None:
                        past_len = first[0].size(2)
                except (IndexError, AttributeError, TypeError):
                    past_len = 0
        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + seq_len, device=inputs_embeds.device
            ).unsqueeze(0).expand(bsz, -1)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        cos, sin = self._rope_for(position_ids, inputs_embeds.dtype)

        new_past: list[tuple[torch.Tensor, torch.Tensor]] = [] if use_cache else None
        for layer_idx, block in enumerate(self.blocks):
            past_kv = None
            if past_key_values is not None:
                try:
                    candidate = past_key_values[layer_idx]
                    # DynamicCache returns (key, value); legacy tuple-of-tuples same.
                    if candidate is not None and candidate[0] is not None:
                        past_kv = candidate
                except (IndexError, AttributeError, TypeError):
                    past_kv = None
            if self.gradient_checkpointing and self.training:
                x, new_kv = self._gradient_checkpointing_func(
                    block.__call__,
                    x,
                    e_x,
                    cos,
                    sin,
                    attention_mask,
                    past_kv,
                    use_cache,
                )
            else:
                x, new_kv = block(
                    x,
                    e_x,
                    cos,
                    sin,
                    attention_mask=attention_mask,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                    **kwargs,
                )
            if use_cache:
                new_past.append(new_kv)

        x = F.rms_norm(x, (x.shape[-1],))

        if not return_dict:
            return (x,) + ((tuple(new_past) if use_cache else None),) + (None, None)

        return BaseModelOutputWithPast(
            last_hidden_state=x,
            past_key_values=tuple(new_past) if use_cache else None,
            hidden_states=None,
            attentions=None,
        )


class TalkieForCausalLM(TalkiePreTrainedModel, GenerationMixin):
    """Causal-LM head. ``lm_head`` is a standard ``nn.Linear`` whose weight
    has the reference ``WeightGain`` (``lm_head_gain.w_g``) folded in by the
    materialization step.
    """

    _tied_weights_keys: list[str] = []

    def __init__(self, config: TalkieConfig):
        super().__init__(config)
        self.model = TalkieModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[tuple[tuple[torch.Tensor, torch.Tensor], ...]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values=None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        # HF caching is disabled in our forward; always pass the full prefix.
        return {
            "input_ids": input_ids,
            "past_key_values": None,
            "attention_mask": attention_mask,
            "use_cache": False,
        }


__all__ = [
    "TalkieConfig",
    "TalkieModel",
    "TalkieForCausalLM",
    "TalkiePreTrainedModel",
]
