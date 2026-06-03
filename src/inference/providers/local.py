"""Local inference provider using HuggingFace transformers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.inference.providers.base import InferenceProvider, PromptInput

if TYPE_CHECKING:
    from src.inference.config import InferenceConfig

logger = logging.getLogger(__name__)


class LocalProvider(InferenceProvider):
    """Inference provider using local HuggingFace transformers models."""

    def __init__(self, config: "InferenceConfig") -> None:
        """Initialize the local provider and load the model.

        Args:
            config: Inference configuration.
        """
        self.config = config
        self.local_config = config.local
        self.generation_config = config.generation

        self.model, self.tokenizer = self._load_model()
        self.device = next(self.model.parameters()).device
        self.prompt_format = self._resolve_prompt_format()
        logger.info("Local prompt format: %s", self.prompt_format)

    def _load_model(self):
        """Load the HuggingFace model and tokenizer."""
        local_cfg = self.local_config

        # Fast path: caller pre-loaded the model (e.g. for LoRA scale sweeps).
        if local_cfg.preloaded_model is not None:
            model, tokenizer = local_cfg.preloaded_model
            logger.info("Using pre-loaded model: %s", self.config.model)
            model.eval()
            return model, tokenizer

        model_name = self.config.model

        dtype = getattr(torch, local_cfg.dtype, None)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {local_cfg.dtype}")

        logger.info("Loading model: %s", model_name)
        load_kwargs: dict = {
            "revision": local_cfg.revision,
            "torch_dtype": dtype,
            "device_map": local_cfg.device_map,
        }
        if local_cfg.attn_implementation:
            load_kwargs["attn_implementation"] = local_cfg.attn_implementation
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if local_cfg.adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "Local provider adapter_path requires the 'peft' package."
                ) from exc
            logger.info("Loading LoRA adapter: %s", local_cfg.adapter_path)
            model = PeftModel.from_pretrained(model, local_cfg.adapter_path)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=local_cfg.revision,
            use_fast=True,
        )
        tokenizer.padding_side = "left"

        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                model.resize_token_embeddings(len(tokenizer))

        model.config.pad_token_id = tokenizer.pad_token_id
        model.eval()

        return model, tokenizer

    def _resolve_eos_token_id(self) -> int | list[int] | None:
        """Resolve EOS token ids without dropping model-specific stop ids.

        Some instruct/chat models define multiple EOS ids in generation config
        (for example end-of-turn markers). Passing only tokenizer.eos_token_id
        can override that and cause generation to run until max_new_tokens.
        """
        eos_ids: list[int] = []

        model_eos = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(model_eos, int):
            eos_ids.append(model_eos)
        elif isinstance(model_eos, list):
            eos_ids.extend(int(token_id) for token_id in model_eos)

        tokenizer_eos = self.tokenizer.eos_token_id
        if tokenizer_eos is not None:
            eos_ids.append(int(tokenizer_eos))

        # Keep order stable, remove duplicates.
        eos_ids = list(dict.fromkeys(eos_ids))
        if not eos_ids:
            return None
        if len(eos_ids) == 1:
            return eos_ids[0]
        return eos_ids

    def _resolve_prompt_format(self) -> str:
        """Resolve how prompts should be formatted before tokenization."""
        configured = self.local_config.prompt_format
        if configured in {"chat", "plain"}:
            return configured

        chat_template = getattr(self.tokenizer, "chat_template", None)
        if isinstance(chat_template, str) and chat_template.strip():
            return "chat"
        return "plain"

    def _render_plain_messages(self, messages: list[dict[str, str]]) -> str:
        """Render chat messages into a plain transcript fallback."""
        lines: list[str] = []
        for message in messages:
            role = message.get("role", "user").capitalize()
            content = message.get("content", "")
            if role == "Assistant" and not content:
                lines.append("Assistant:")
            else:
                lines.append(f"{role}: {content}")
        if not messages or messages[-1].get("role") != "assistant":
            lines.append("Assistant:")
        return "\n".join(lines)

    def _format_prompt_input(self, prompt: PromptInput) -> str:
        """Format one prompt input into the string expected by the tokenizer."""
        if isinstance(prompt, str):
            if self.prompt_format != "chat":
                return prompt
            if not hasattr(self.tokenizer, "apply_chat_template"):
                logger.warning(
                    "prompt_format=chat but tokenizer has no apply_chat_template; "
                    "falling back to plain prompt."
                )
                return prompt
            messages: list[dict[str, str]] = []
            if self.local_config.chat_system_prompt:
                messages.append(
                    {"role": "system", "content": self.local_config.chat_system_prompt}
                )
            messages.append({"role": "user", "content": prompt})
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception as exc:
                logger.warning(
                    "Failed applying chat template to plain prompt; using raw prompt. error=%s",
                    exc,
                )
                return prompt

        if self.prompt_format == "chat" and hasattr(
            self.tokenizer, "apply_chat_template"
        ):
            messages = prompt
            if self.local_config.chat_system_prompt and not any(
                message.get("role") == "system" for message in messages
            ):
                messages = [
                    {"role": "system", "content": self.local_config.chat_system_prompt},
                    *messages,
                ]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=messages[-1]["role"] != "assistant",
                    tokenize=False,
                )
            except Exception as exc:
                logger.warning(
                    "Failed applying chat template to prompt messages; using plain transcript. error=%s",
                    exc,
                )

        return self._render_plain_messages(prompt)

    def _normalize_eos_ids(self, eos_token_id: int | list[int] | None) -> set[int]:
        """Normalize eos_token_id into a set for finish-reason checks."""
        if eos_token_id is None:
            return set()
        if isinstance(eos_token_id, int):
            return {int(eos_token_id)}
        return {int(token_id) for token_id in eos_token_id}

    def _resolve_max_input_tokens(self) -> int | None:
        """Resolve effective max input tokens from model/tokenizer config."""
        model_limit = getattr(self.model.config, "max_position_embeddings", None)
        tokenizer_limit = getattr(self.tokenizer, "model_max_length", None)

        candidates: list[int] = []
        for value in (model_limit, tokenizer_limit):
            if not isinstance(value, int):
                continue
            # Tokenizers often use very large sentinels for "unknown/unbounded".
            if value <= 0 or value >= 1_000_000:
                continue
            candidates.append(value)
        if not candidates:
            return None
        return min(candidates)

    def _validate_tokenized_lengths(self, input_ids: torch.Tensor) -> None:
        """Raise if any tokenized input exceeds model context length.

        Args:
            input_ids: Tokenized input tensor of shape (batch, seq_len).
                       With left-padding, the padded length equals the longest
                       sequence, so checking shape[1] is sufficient.
        """
        max_input_tokens = self._resolve_max_input_tokens()
        if max_input_tokens is None:
            logger.warning(
                "Could not resolve max input tokens; skipping explicit overflow validation."
            )
            return

        seq_len = input_ids.shape[1]
        if seq_len > max_input_tokens:
            raise ValueError(
                "Input prompt exceeds model context length and truncation is disabled. "
                f"max_input_tokens={max_input_tokens}, "
                f"longest_sequence_tokens={seq_len}."
            )

    def generate(self, prompt: PromptInput, **kwargs) -> str:
        """Generate a response for a single prompt.

        Args:
            prompt: The input prompt.
            **kwargs: Override generation parameters.

        Returns:
            Generated response string.
        """
        responses = self.generate_batch([prompt], **kwargs)
        return responses[0]

    def generate_batch(self, prompts: list[PromptInput], **kwargs) -> list[str]:
        """Generate responses for a batch of prompts.

        Args:
            prompts: List of input prompts.
            **kwargs: Override generation parameters.

        Returns:
            List of generated response strings.
        """
        gen_cfg = self.generation_config

        # Allow kwargs to override config values
        max_new_tokens = kwargs.get("max_new_tokens", gen_cfg.max_new_tokens)
        temperature = kwargs.get("temperature", gen_cfg.temperature)
        top_p = kwargs.get("top_p", gen_cfg.top_p)
        do_sample = kwargs.get("do_sample", gen_cfg.do_sample)
        num_responses = kwargs.get("num_responses", gen_cfg.num_responses_per_prompt)
        eos_token_id = kwargs.get("eos_token_id", self._resolve_eos_token_id())
        formatted_prompts = [self._format_prompt_input(prompt) for prompt in prompts]
        truncate_inputs = self.local_config.truncate_inputs

        inputs = self.tokenizer(
            formatted_prompts,
            padding=True,
            truncation=truncate_inputs,
            return_tensors="pt",
        )
        if not truncate_inputs:
            self._validate_tokenized_lengths(inputs["input_ids"])
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            generated_output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                num_return_sequences=num_responses,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=eos_token_id,
                return_dict_in_generate=True,
            )

        generated = generated_output.sequences

        input_length = int(inputs["input_ids"].shape[1])
        responses = []
        eos_ids = self._normalize_eos_ids(eos_token_id)
        stopped_on_eos_count = 0
        hit_max_new_tokens_count = 0
        other_stop_count = 0
        for output_ids in generated:
            completion_ids = output_ids[input_length:]
            completion_token_ids = completion_ids.tolist()
            if eos_ids and any(
                token_id in eos_ids for token_id in completion_token_ids
            ):
                stopped_on_eos_count += 1
            elif len(completion_token_ids) >= max_new_tokens:
                hit_max_new_tokens_count += 1
            else:
                other_stop_count += 1
            responses.append(
                self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            )

        logger.info(
            "Local generation finish reasons: stopped_on_eos=%d, "
            "hit_max_new_tokens=%d, other=%d",
            stopped_on_eos_count,
            hit_max_new_tokens_count,
            other_stop_count,
        )

        return responses
