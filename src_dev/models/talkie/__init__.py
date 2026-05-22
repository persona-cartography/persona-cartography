"""HF-transformers wrapper for the talkie-1930-13b family.

Mirrors the reference implementation at https://github.com/talkie-lm/talkie
(Apache-2.0), with additions needed for HF compatibility (full-sequence
logits, attention_mask, past_key_values, gradient checkpointing) and a
materialized model directory that vLLM and OpenRLHF can load via
``trust_remote_code``.

The materialized directory is produced by
``python -m src_dev.models.talkie.materialize`` and lives outside the
repo (default: ``/root/.cache/models/talkie-1930-13b-it/``).
"""

from src_dev.models.talkie.configuration_talkie import TalkieConfig
from src_dev.models.talkie.modeling_talkie import TalkieForCausalLM, TalkieModel
from src_dev.models.talkie.tokenization_talkie import TalkieTokenizerFast

__all__ = [
    "TalkieConfig",
    "TalkieForCausalLM",
    "TalkieModel",
    "TalkieTokenizerFast",
]
