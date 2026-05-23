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

import os
from pathlib import Path

from src_dev.models.talkie.configuration_talkie import TalkieConfig
from src_dev.models.talkie.modeling_talkie import TalkieForCausalLM, TalkieModel
from src_dev.models.talkie.tokenization_talkie import TalkieTokenizerFast


def local_model_dir(model_name: str = "talkie-1930-13b-it") -> Path:
    """Return the local filesystem path to the materialized talkie HF dir.

    Defaults to ``$OCT_MODEL_PATH/<model_name>`` (matching the OCT pipeline's
    ``character.constants.MODEL_PATH`` indirection), falling back to
    ``/root/.cache/models/<model_name>``. Use this from eval configs so they
    pick up the materialized wrapper rather than trying to load the raw
    talkie-lm HF hub repo (which has only ``rl-refined.pt``, not a
    transformers-loadable model).
    """
    root = os.environ.get("OCT_MODEL_PATH", "/root/.cache/models")
    return Path(root) / model_name


def local_model_uri(model_name: str = "talkie-1930-13b-it") -> str:
    """Return a ``local://...`` URI suitable for ``SuiteConfig.base_model``."""
    return f"local://{local_model_dir(model_name).resolve()}"


__all__ = [
    "TalkieConfig",
    "TalkieForCausalLM",
    "TalkieModel",
    "TalkieTokenizerFast",
    "local_model_dir",
    "local_model_uri",
]
