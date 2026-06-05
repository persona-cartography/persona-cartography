"""Inspect ModelAPI provider backed by a pre-loaded vLLM engine + baked LoRA adapter.

Allows passing a :class:`_VllmVariantProvider` (from
``src.rollout_generation.model_providers``) directly to ``inspect_ai.eval()``
so that the evals suite can use vLLM's continuous batching for LoRA scale sweeps
without spinning up a separate HTTP server.

Registration::

    from src.evals.utils.vllm_preloaded_provider import register_vllm_preloaded_provider
    register_vllm_preloaded_provider()

Usage::

    from inspect_ai.model import get_model
    model_obj = get_model(
        "vllm_preloaded/my-label",
        vllm_variant_provider=variant_provider,
    )
    inspect_eval(task, model=model_obj, ...)

The ``model_name`` component of the URI (``my-label``) is cosmetic.
"""

from __future__ import annotations

from typing import Any

from typing_extensions import override

from inspect_ai.model._chat_message import ChatMessage, ChatMessageAssistant
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model import ModelAPI
from inspect_ai.model._model_output import (
    ChatCompletionChoice,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.model._registry import modelapi

_PROVIDER_NAME = "vllm_preloaded"
_registered = False


def register_vllm_preloaded_provider() -> None:
    """Register the ``vllm_preloaded`` Inspect model provider.

    Safe to call multiple times — registration only happens once.
    """
    global _registered
    if _registered:
        return

    @modelapi(name=_PROVIDER_NAME)
    class VllmPreloadedAPI(ModelAPI):
        """Inspect ModelAPI backed by a pre-loaded vLLM variant provider."""

        def __init__(
            self,
            model_name: str,
            base_url: str | None = None,
            api_key: str | None = None,
            config: GenerateConfig = GenerateConfig(),
            **model_args: Any,
        ) -> None:
            super().__init__(
                model_name=model_name,
                base_url=base_url,
                api_key=api_key,
                config=config,
            )

            if "vllm_variant_provider" not in model_args:
                raise ValueError(
                    "vllm_preloaded provider requires 'vllm_variant_provider' in model_args"
                )

            self._provider: Any = model_args["vllm_variant_provider"]
            self._batch_size: int = int(model_args.get("batch_size", 32))

        @override
        def close(self) -> None:
            # Lifetime owned by VLLMLoRaScaleProvider context manager.
            pass

        @override
        def max_tokens(self) -> int | None:
            from inspect_ai._util.constants import DEFAULT_MAX_TOKENS

            return DEFAULT_MAX_TOKENS

        @override
        def max_connections(self) -> int:
            return self._batch_size

        @override
        def collapse_user_messages(self) -> bool:
            return True

        async def generate(
            self,
            input: list[ChatMessage],
            tools: list[Any],
            tool_choice: Any,
            config: GenerateConfig,
        ) -> ModelOutput:
            import asyncio

            # Convert Inspect ChatMessages to the vllm chat format.
            messages = [
                {
                    "role": m.role,
                    "content": m.text if hasattr(m, "text") else str(m.content),
                }
                for m in input
            ]

            kwargs: dict[str, Any] = {}
            if config.temperature is not None:
                kwargs["temperature"] = config.temperature
            if config.top_p is not None:
                kwargs["top_p"] = config.top_p
            if config.max_tokens is not None:
                kwargs["max_new_tokens"] = config.max_tokens

            loop = asyncio.get_event_loop()
            output_text = await loop.run_in_executor(
                None,
                lambda: self._provider.generate(messages, **kwargs),
            )

            return ModelOutput(
                model=self.model_name,
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessageAssistant(
                            content=output_text,
                            model=self.model_name,
                            source="generate",
                        )
                    )
                ],
                usage=ModelUsage(),
            )

    _registered = True
