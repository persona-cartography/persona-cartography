"""TalkieConfig — HuggingFace-style config for the talkie 13B family.

Values mirror ``talkie.model.GPTConfig`` from the reference implementation
(see https://github.com/talkie-lm/talkie, ``src/talkie/model.py``). The IT
variant uses vocab_size=65540 (65,536 BPE + 4 IT specials: ``<|end|>``,
``<|user|>``, ``<|assistant|>``, ``<|system|>``).
"""

from __future__ import annotations

from transformers import PretrainedConfig


class TalkieConfig(PretrainedConfig):
    """Config for ``TalkieForCausalLM``.

    Mirrors the reference architecture. The non-standard params (HeadGain,
    ActGain, embed-skip, lm_head WeightGain) are part of the model weights;
    no config flag is needed to enable them.
    """

    model_type = "talkie"

    def __init__(
        self,
        vocab_size: int = 65540,
        hidden_size: int = 5120,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 40,
        head_dim: int = 128,
        intermediate_size: int = 13696,
        max_position_embeddings: int = 4096,
        rope_theta: float = 1_000_000.0,
        rms_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        bos_token_id: int | None = None,
        eos_token_id: int | None = 65535,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )
