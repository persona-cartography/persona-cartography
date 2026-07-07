"""vLLM inference provider."""

from __future__ import annotations

import gc
import hashlib
import logging
import os
from typing import TYPE_CHECKING

from src_dev.inference.providers.base import InferenceProvider, PromptInput

if TYPE_CHECKING:
    from src_dev.inference.config import InferenceConfig

logger = logging.getLogger(__name__)


def _resolve_vllm_adapter_path(adapter_path: str) -> str:
    """Resolve an adapter reference to a local directory path for vLLM.

    vLLM's LoRARequest requires a local filesystem path.  This function:
    - Strips the ``local://`` prefix if present.
    - Splits ``repo_id::subfolder`` notation.
    - Downloads HF repo adapters to the local HF cache via snapshot_download.

    Args:
        adapter_path: Adapter reference in any supported format:
            ``local://path``, ``path``, ``hf_repo_id``, ``hf_repo_id::subfolder``.

    Returns:
        Absolute local path to the adapter directory.
    """
    from pathlib import Path as _Path

    if adapter_path.startswith("local://"):
        adapter_path = adapter_path[len("local://") :]

    subfolder: str | None = None
    if "::" in adapter_path:
        adapter_path, subfolder = adapter_path.split("::", 1)

    local = _Path(adapter_path).expanduser()
    if local.exists():
        if subfolder:
            local = local / subfolder
        return str(local.resolve())

    if subfolder:
        from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

        from src_dev.utils.hf_hub import download_path_to_dir

        cache_key = hashlib.sha256(f"{adapter_path}::{subfolder}".encode()).hexdigest()[:16]
        cache_root = _Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
        dataset_target = cache_root / "dataset_adapters" / cache_key
        try:
            download_path_to_dir(
                repo_id=adapter_path,
                path_in_repo=subfolder,
                target_dir=dataset_target,
            )
            if (dataset_target / "adapter_config.json").exists():
                return str(dataset_target.resolve())
        except (EntryNotFoundError, RepositoryNotFoundError):
            pass

    from src_dev.utils.lora_composition import resolve_adapter_to_local_dir

    return resolve_adapter_to_local_dir(
        f"{adapter_path}::{subfolder}" if subfolder else adapter_path
    )


# vLLM's LoRAConfig only accepts these max_lora_rank values. Baked LoRA
# soups have combined rank = sum of input ranks (e.g. 3 × rank-64 adapters
# = 192), so requested ranks must be rounded up to the next allowed bucket
# — the cap just needs to be >= the adapter's actual rank.
_VLLM_ALLOWED_LORA_RANKS = (8, 16, 32, 64, 128, 256, 320, 512)


def _round_up_lora_rank(rank: int) -> int:
    for allowed in _VLLM_ALLOWED_LORA_RANKS:
        if rank <= allowed:
            return allowed
    raise ValueError(
        f"LoRA rank {rank} exceeds vLLM's maximum supported max_lora_rank "
        f"({_VLLM_ALLOWED_LORA_RANKS[-1]}); reduce the soup with target_rank "
        f"(SVD compression) in bake_combined_lora."
    )


class VllmProvider(InferenceProvider):
    """Inference provider using vLLM for high-throughput generation.

    Supports base models and LoRA adapters (via LoRARequest).  Chat-template
    formatting is handled by vLLM's built-in tokenizer integration.

    Args:
        config: Inference configuration.  Provider-specific settings are
            read from ``config.vllm``.
    """

    def __init__(self, config: "InferenceConfig") -> None:
        try:
            from vllm import LLM, SamplingParams
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise ImportError(
                "VllmProvider requires the 'vllm' package. Install it with: pip install vllm"
            ) from exc

        self.config = config
        self.vllm_config = config.vllm
        self.generation_config = config.generation
        self._SamplingParams = SamplingParams
        self._LoRARequest = LoRARequest

        vllm_cfg = self.vllm_config
        has_adapter = vllm_cfg.adapter_path is not None

        engine_kwargs: dict = dict(
            model=config.model,
            dtype=vllm_cfg.dtype,
            gpu_memory_utilization=vllm_cfg.gpu_memory_utilization,
            enforce_eager=vllm_cfg.enforce_eager,
            enable_prefix_caching=vllm_cfg.enable_prefix_caching,
            trust_remote_code=False,
        )
        if vllm_cfg.tensor_parallel_size > 1:
            engine_kwargs["tensor_parallel_size"] = vllm_cfg.tensor_parallel_size
        if vllm_cfg.max_model_len is not None:
            engine_kwargs["max_model_len"] = vllm_cfg.max_model_len
        if vllm_cfg.limit_mm_per_prompt is not None:
            engine_kwargs["limit_mm_per_prompt"] = vllm_cfg.limit_mm_per_prompt
        if has_adapter:
            max_cpu = vllm_cfg.max_cpu_loras or vllm_cfg.max_loras
            engine_kwargs["enable_lora"] = True
            engine_kwargs["max_loras"] = vllm_cfg.max_loras
            engine_kwargs["max_lora_rank"] = _round_up_lora_rank(vllm_cfg.max_lora_rank)
            engine_kwargs["max_cpu_loras"] = max_cpu

        logger.info("Initialising vLLM engine: model=%s", config.model)
        self.llm: LLM = LLM(**engine_kwargs)

        self._lora_request = None
        if has_adapter:
            local_adapter_path = _resolve_vllm_adapter_path(vllm_cfg.adapter_path)
            logger.info("  LoRA adapter (resolved): %s", local_adapter_path)
            self._lora_request = LoRARequest(
                lora_name="adapter",
                lora_int_id=1,
                lora_path=local_adapter_path,
            )

        # Chat-template override for legacy tokenizers that ship with
        # ``chat_template = None`` (Koala-13B, OpenAssistant-Pythia-12B,
        # older Vicuna variants, etc). Passed to every ``llm.chat(...)``
        # call so vLLM applies our registry template instead of failing
        # on "chat_template is not set".
        self._chat_template: str | None = vllm_cfg.chat_template

    def _sampling_params(self, **kwargs):
        gen = self.generation_config
        params = dict(
            temperature=kwargs.get("temperature", gen.temperature),
            top_p=kwargs.get("top_p", gen.top_p),
            max_tokens=kwargs.get("max_new_tokens", gen.max_new_tokens),
        )
        if "logprobs" in kwargs and kwargs["logprobs"] is not None:
            params["logprobs"] = int(kwargs["logprobs"])
        return self._SamplingParams(**params)

    @staticmethod
    def _format_messages(prompts: list[PromptInput]) -> list[list[dict[str, str]]]:
        return [
            [{"role": "user", "content": p}] if isinstance(p, str) else p
            for p in prompts
        ]

    @staticmethod
    def _prefill_flags_for(
        messages_list: list[list[dict[str, str]]],
    ) -> tuple[bool, bool]:
        """Return (add_generation_prompt, continue_final_message).

        vLLM's chat template requires explicit flags when the trailing message
        is an assistant turn that should be *continued* (prefill) rather than
        closed and followed by a new assistant header. Every prompt in a batch
        must share the same flag values, so we require all prompts to agree on
        whether they end with an assistant turn.
        """
        ends_assistant = [
            bool(msgs) and msgs[-1].get("role") == "assistant"
            for msgs in messages_list
        ]
        if any(ends_assistant) and not all(ends_assistant):
            raise ValueError(
                "vLLM batch contains a mix of assistant-trailing (prefill) and "
                "user-trailing prompts. Split the batch before calling the "
                "provider, or align the trailing role across all prompts."
            )
        if all(ends_assistant) and ends_assistant:
            return False, True
        return True, False

    def generate(self, prompt: PromptInput, **kwargs) -> str:
        """Generate a response for a single prompt."""
        return self.generate_batch([prompt], **kwargs)[0]

    def generate_batch(self, prompts: list[PromptInput], **kwargs) -> list[str]:
        """Generate responses for a batch of prompts.

        Args:
            prompts: List of input prompts (str or message list).
            **kwargs: Override generation parameters.

        Returns:
            List of generated response strings.
        """
        sampling_params = self._sampling_params(**kwargs)
        formatted = self._format_messages(prompts)
        add_gen, continue_final = self._prefill_flags_for(formatted)

        chat_kwargs: dict = {}
        if self._chat_template is not None:
            chat_kwargs["chat_template"] = self._chat_template

        outputs = self.llm.chat(
            messages=formatted,
            sampling_params=sampling_params,
            lora_request=self._lora_request,
            use_tqdm=False,
            add_generation_prompt=add_gen,
            continue_final_message=continue_final,
            **chat_kwargs,
        )

        responses = [out.outputs[0].text for out in outputs]
        logger.info("vLLM generated %d responses", len(responses))
        return responses

    def generate_batch_logprobs(
        self,
        prompts: list[PromptInput],
        *,
        max_tokens: int = 1,
        top_logprobs: int = 20,
        temperature: float = 1.0,
    ) -> list[dict]:
        """Generate with top-k logprobs on each generated token.

        Honours the same prefill semantics as ``generate_batch``: prompts whose
        final message is an assistant turn are treated as prefilled and the
        chat template continues from that partial turn.

        Args:
            prompts: Message lists (or strings).
            max_tokens: How many tokens to generate (default 1 — suitable for
                MCQ logprob scoring where we only need the first token's
                distribution).
            top_logprobs: Top-k logprobs to return per generated token.
            temperature: Sampling temperature. Default 1.0 (returns the raw
                model distribution for scoring).

        Returns:
            One dict per prompt with keys:
                - ``text``: generated text
                - ``logprobs_per_token``: list (length = num generated tokens)
                  of ``dict[decoded_token_str, float]`` top-k logprobs.
        """
        sampling_params = self._sampling_params(
            temperature=temperature,
            max_new_tokens=max_tokens,
            logprobs=top_logprobs,
        )
        formatted = self._format_messages(prompts)
        add_gen, continue_final = self._prefill_flags_for(formatted)

        chat_kwargs: dict = {}
        if self._chat_template is not None:
            chat_kwargs["chat_template"] = self._chat_template

        outputs = self.llm.chat(
            messages=formatted,
            sampling_params=sampling_params,
            lora_request=self._lora_request,
            use_tqdm=False,
            add_generation_prompt=add_gen,
            continue_final_message=continue_final,
            **chat_kwargs,
        )

        results: list[dict] = []
        for out in outputs:
            completion = out.outputs[0]
            per_token: list[dict[str, float]] = []
            if completion.logprobs is not None:
                for token_dict in completion.logprobs:
                    per_token.append({
                        entry.decoded_token: float(entry.logprob)
                        for entry in token_dict.values()
                        if entry.decoded_token is not None
                    })
            results.append({
                "text": completion.text,
                "logprobs_per_token": per_token,
            })
        logger.info("vLLM generated %d logprob outputs", len(results))
        return results

    async def generate_batch_logprobs_async(
        self,
        prompts: list[PromptInput],
        *,
        max_tokens: int = 1,
        top_logprobs: int = 20,
        temperature: float = 1.0,
    ) -> list[dict]:
        import asyncio

        return await asyncio.to_thread(
            self.generate_batch_logprobs,
            prompts,
            max_tokens=max_tokens,
            top_logprobs=top_logprobs,
            temperature=temperature,
        )

    def _tokens_prompts(self, token_id_lists: list[list[int]]):
        """Wrap token-ID lists in vLLM's ``TokensPrompt`` input objects.

        Falls back to a plain dict when ``TokensPrompt`` isn't importable
        (older vLLM). Both forms are accepted by ``LLM.generate``.
        """
        try:
            from vllm.inputs import TokensPrompt  # type: ignore[attr-defined]
            return [TokensPrompt(prompt_token_ids=list(ids)) for ids in token_id_lists]
        except Exception:
            return [{"prompt_token_ids": list(ids)} for ids in token_id_lists]

    def generate_batch_from_token_ids(
        self,
        token_id_lists: list[list[int]],
        **kwargs,
    ) -> list[str]:
        """Generate responses for a batch of pre-tokenised prompts.

        Bypasses vLLM's chat-template application: the caller is responsible
        for assembling the full prompt (special tokens, role headers, and any
        generation-prompt / prefill markers) as a list of token IDs. Used by
        the ``token_boundary`` reset mode where mid-sequence ``<|end_of_text|>``
        / ``<|begin_of_text|>`` tokens must be preserved literally.

        Args:
            token_id_lists: One ``list[int]`` per prompt.
            **kwargs: Override sampling parameters (``temperature``, ``top_p``,
                ``max_new_tokens``).

        Returns:
            List of generated response strings (one per input).
        """
        sampling_params = self._sampling_params(**kwargs)
        outputs = self.llm.generate(
            prompts=self._tokens_prompts(token_id_lists),
            sampling_params=sampling_params,
            lora_request=self._lora_request,
            use_tqdm=False,
        )
        responses = [out.outputs[0].text for out in outputs]
        logger.info(
            "vLLM generated %d responses from raw token IDs", len(responses)
        )
        return responses

    def generate_batch_logprobs_from_token_ids(
        self,
        token_id_lists: list[list[int]],
        *,
        max_tokens: int = 1,
        top_logprobs: int = 20,
        temperature: float = 1.0,
    ) -> list[dict]:
        """Logprob-mode variant of :meth:`generate_batch_from_token_ids`.

        Mirrors :meth:`generate_batch_logprobs` but over raw token IDs rather
        than messages lists. Returns the same ``{"text", "logprobs_per_token"}``
        schema.
        """
        sampling_params = self._sampling_params(
            temperature=temperature,
            max_new_tokens=max_tokens,
            logprobs=top_logprobs,
        )
        outputs = self.llm.generate(
            prompts=self._tokens_prompts(token_id_lists),
            sampling_params=sampling_params,
            lora_request=self._lora_request,
            use_tqdm=False,
        )
        results: list[dict] = []
        for out in outputs:
            completion = out.outputs[0]
            per_token: list[dict[str, float]] = []
            if completion.logprobs is not None:
                for token_dict in completion.logprobs:
                    per_token.append({
                        entry.decoded_token: float(entry.logprob)
                        for entry in token_dict.values()
                        if entry.decoded_token is not None
                    })
            results.append({
                "text": completion.text,
                "logprobs_per_token": per_token,
            })
        logger.info(
            "vLLM generated %d logprob outputs from raw token IDs", len(results)
        )
        return results

    async def generate_batch_from_token_ids_async(
        self,
        token_id_lists: list[list[int]],
        **kwargs,
    ) -> list[str]:
        import asyncio

        return await asyncio.to_thread(
            self.generate_batch_from_token_ids, token_id_lists, **kwargs
        )

    async def generate_batch_logprobs_from_token_ids_async(
        self,
        token_id_lists: list[list[int]],
        *,
        max_tokens: int = 1,
        top_logprobs: int = 20,
        temperature: float = 1.0,
    ) -> list[dict]:
        import asyncio

        return await asyncio.to_thread(
            self.generate_batch_logprobs_from_token_ids,
            token_id_lists,
            max_tokens=max_tokens,
            top_logprobs=top_logprobs,
            temperature=temperature,
        )

    def close(self) -> None:
        """Tear down the vLLM engine and release GPU memory.

        Idempotent. After calling, the provider is unusable.
        """
        llm = getattr(self, "llm", None)
        if llm is None:
            return
        try:
            # vLLM v1 spawns worker subprocesses; shutdown() tells them to
            # exit cleanly. Older versions expose it on llm_engine instead.
            for owner in (llm, getattr(llm, "llm_engine", None)):
                shutdown = getattr(owner, "shutdown", None) if owner is not None else None
                if callable(shutdown):
                    try:
                        shutdown()
                    except Exception as exc:
                        logger.warning("vLLM shutdown() raised: %s", exc)
                    break
        finally:
            self.llm = None  # type: ignore[assignment]
        del llm
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
        logger.info("vLLM provider closed")

    async def aclose(self) -> None:
        import asyncio
        await asyncio.to_thread(self.close)
