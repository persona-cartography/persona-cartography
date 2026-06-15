"""Inspect ModelAPI provider that wraps a pre-loaded HuggingFace model.

Allows passing an already-in-memory ``AutoModelForCausalLM`` (or ``PeftModel``)
directly to ``inspect_ai.eval()`` without writing to disk.  This is the key
enabler for in-place LoRA scale sweeps: load the base model + adapter once,
apply ``LoRaScaling`` in-place for each scale point, run Inspect, restore.

Registration::

    from src.evals.utils.preloaded_hf_provider import register_preloaded_hf_provider
    register_preloaded_hf_provider()

Usage::

    from inspect_ai.model import get_model
    model_obj = get_model(
        "hf_preloaded/my-label",
        hf_model=peft_model,
        hf_tokenizer=tokenizer,
    )
    inspect_eval(task, model=model_obj, ...)

The ``model_name`` component of the URI (``my-label``) is cosmetic — it appears
in Inspect logs but has no functional effect.
"""

from __future__ import annotations

import concurrent.futures
import copy
import os
import functools
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from logging import getLogger
from queue import Empty, Queue
from threading import Thread
from typing import Any

import anyio
import torch
from transformers import PreTrainedTokenizerBase
from typing_extensions import override

from inspect_ai.model._providers.hf import (
    GenerateInput,
    GenerateOutput,
    batched_generate,
    extract_logprobs,
)

from inspect_ai._util.content import (
    ContentAudio,
    ContentDocument,
    ContentImage,
    ContentVideo,
)
from inspect_ai.model._reasoning import emulate_reasoning_history
from inspect_ai.model._chat_message import ChatMessage, ChatMessageAssistant
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._model import ModelAPI
from inspect_ai.model._model_output import (
    ChatCompletionChoice,
    Logprobs,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.model._registry import modelapi
from inspect_ai.tool import ToolChoice, ToolInfo

logger = getLogger(__name__)

_PROVIDER_NAME = "hf_preloaded"
_registered = False

# ---------------------------------------------------------------------------
# Tokenisation cache — persists across batcher instances so that token IDs
# computed for scale point N are reused at scale point N+1 (same prompts).
# ---------------------------------------------------------------------------
_token_id_cache: dict[str, list[int]] = {}


def clear_tokenization_cache() -> None:
    """Drop the module-level tokenisation cache (call at end of sweep)."""
    _token_id_cache.clear()


# ---------------------------------------------------------------------------
# Fast logprobs batcher — bypasses Inspect's batched_generate overhead
# ---------------------------------------------------------------------------


@dataclass
class _LogprobsRequest:
    """A single logprobs request submitted to the batcher."""

    chat_text: str
    top_logprobs: int
    future: Future


@dataclass
class _LogprobsResult:
    """Result from a batched forward pass for one sample."""

    logprobs_tensor: torch.Tensor  # (1, vocab_size), log-softmax'd, on CPU
    input_tokens: int
    time: float


class _LogprobsBatcher:
    """Fast batched logprobs via direct model forward pass.

    Replaces Inspect's ``batched_generate`` for single-token logprobs:

    * **50 ms** drain timeout  (vs Inspect's 2 s)
    * **5 ms**  result polling (vs Inspect's 1 s)
    * Direct ``model(input_ids)`` call (vs ``model.generate()``)
    """

    _DRAIN_TIMEOUT = 0.05  # seconds
    _POLL_INTERVAL = 0.005  # seconds

    def __init__(
        self,
        model: Any,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int,
        device: Any,
        cache_tokenization: bool = True,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._device = device
        self._cache_tokenization = cache_tokenization
        self._queue: Queue[_LogprobsRequest] = Queue()
        self._thread: Thread | None = None
        self._shutdown = threading.Event()
        # Timing accumulators (thread-safe via GIL for simple increments).
        self._n_batches = 0
        self._t_tokenize = 0.0
        self._t_forward = 0.0
        self._t_softmax_cpu = 0.0
        self._t_total = 0.0
        self._total_samples = 0
        self._max_seq_len = 0
        self._cache_hits = 0
        self._cache_misses = 0

    def _ensure_started(self) -> None:
        if self._thread is None:
            self._thread = Thread(
                target=self._process_loop, daemon=True, name="logprobs-batcher"
            )
            self._thread.start()

    def shutdown(self) -> None:
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._print_timing_summary()

    def _print_timing_summary(self) -> None:
        if self._n_batches == 0:
            return
        n = self._n_batches
        cache_line = ""
        if self._cache_tokenization:
            total_lookups = self._cache_hits + self._cache_misses
            hit_pct = (self._cache_hits / total_lookups * 100) if total_lookups else 0
            cache_line = f"  token cache:  {self._cache_hits}/{total_lookups} hits ({hit_pct:.0f}%)\n"
        avg_batch = self._total_samples / n if n else 0
        print(
            f"\n=== LogprobsBatcher timing ({self._total_samples} samples, "
            f"{n} batches, avg_batch={avg_batch:.1f}, max_seq_len={self._max_seq_len}) ===\n"
            f"  tokenize:     {self._t_tokenize:7.2f}s  ({self._t_tokenize / n:.3f}s/batch)\n"
            f"  forward pass: {self._t_forward:7.2f}s  ({self._t_forward / n:.3f}s/batch)\n"
            f"  softmax+cpu:  {self._t_softmax_cpu:7.2f}s  ({self._t_softmax_cpu / n:.3f}s/batch)\n"
            + cache_line
            + f"  total:        {self._t_total:7.2f}s  ({self._t_total / n:.3f}s/batch)\n"
            f"  throughput:   {self._total_samples / self._t_total:.1f} samples/s\n"
            f"================================================",
            flush=True,
        )

    async def submit(self, chat_text: str, top_logprobs: int) -> _LogprobsResult:
        self._ensure_started()
        future: Future = Future()
        self._queue.put(_LogprobsRequest(chat_text, top_logprobs, future))
        while True:
            try:
                return future.result(timeout=self._POLL_INTERVAL)
            except concurrent.futures.TimeoutError:
                pass
            await anyio.sleep(self._POLL_INTERVAL)

    def _process_loop(self) -> None:
        # When LOGPROBS_NO_DRAIN_DELAY is set, use the old behaviour
        # (no delay after first item) for A/B benchmarking.
        use_drain_delay = not os.environ.get("LOGPROBS_NO_DRAIN_DELAY")
        while not self._shutdown.is_set():
            items: list[_LogprobsRequest] = []
            # Wait for the first item (blocks until work arrives).
            try:
                first = self._queue.get(timeout=self._DRAIN_TIMEOUT)
            except Empty:
                continue
            items.append(first)
            if use_drain_delay:
                # Give the event loop a brief window to enqueue more work
                # before we drain.  Without this pause the batcher fires
                # single-sample batches because the async loop hasn't had
                # a chance to dispatch concurrent generate() calls yet.
                time.sleep(self._DRAIN_TIMEOUT)
            # Drain everything that accumulated.
            while len(items) < self._batch_size:
                try:
                    items.append(self._queue.get_nowait())
                except Empty:
                    break
            self._run_batch(items)

    def _tokenize_batch(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch of texts, optionally using the module-level cache.

        Returns (input_ids, attention_mask) on self._device, left-padded.
        """
        if not self._cache_tokenization:
            encoded = self._tokenizer(texts, return_tensors="pt", padding=True)
            return (
                encoded["input_ids"].to(self._device),
                encoded["attention_mask"].to(self._device),
            )

        # Look up / populate cache with per-sample token IDs.
        all_ids: list[list[int]] = []
        for text in texts:
            cached = _token_id_cache.get(text)
            if cached is not None:
                all_ids.append(cached)
                self._cache_hits += 1
            else:
                ids: list[int] = self._tokenizer(text)["input_ids"]
                _token_id_cache[text] = ids
                all_ids.append(ids)
                self._cache_misses += 1

        # Left-pad to the longest sequence in this batch.
        max_len = max(len(ids) for ids in all_ids)
        pad_id = (
            self._tokenizer.pad_token_id
            if self._tokenizer.pad_token_id is not None
            else 0
        )
        input_ids = torch.tensor(
            [[pad_id] * (max_len - len(ids)) + ids for ids in all_ids],
            dtype=torch.long,
            device=self._device,
        )
        attention_mask = torch.tensor(
            [[0] * (max_len - len(ids)) + [1] * len(ids) for ids in all_ids],
            dtype=torch.long,
            device=self._device,
        )
        return input_ids, attention_mask

    def _run_batch(self, items: list[_LogprobsRequest]) -> None:
        try:
            t0 = time.monotonic()
            texts = [item.chat_text for item in items]

            input_ids, attention_mask = self._tokenize_batch(texts)
            t_tok = time.monotonic()

            with torch.inference_mode():
                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits_to_keep=1,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_fwd = time.monotonic()

                # logits_to_keep=1 → logits shape is (batch, 1, vocab).
                last_logits = outputs.logits[:, -1, :]
                log_probs = torch.nn.functional.log_softmax(last_logits, dim=-1).cpu()
            t_end = time.monotonic()

            n_input = input_ids.size(1)
            elapsed = t_end - t0

            # Accumulate timing stats.
            self._n_batches += 1
            self._t_tokenize += t_tok - t0
            self._t_forward += t_fwd - t_tok
            self._t_softmax_cpu += t_end - t_fwd
            self._t_total += elapsed
            self._total_samples += len(items)
            self._max_seq_len = max(self._max_seq_len, n_input)

            for i, item in enumerate(items):
                item.future.set_result(
                    _LogprobsResult(
                        logprobs_tensor=log_probs[i].unsqueeze(0),
                        input_tokens=n_input,
                        time=elapsed,
                    )
                )
        except Exception as exc:
            for item in items:
                if not item.future.done():
                    item.future.set_exception(exc)


def register_preloaded_hf_provider() -> None:
    """Register the ``hf_preloaded`` Inspect model provider.

    Safe to call multiple times — registration only happens once.
    """
    global _registered
    if _registered:
        return

    @modelapi(name=_PROVIDER_NAME)
    class PreloadedHFAPI(ModelAPI):
        """Inspect ModelAPI backed by a pre-loaded HuggingFace model object."""

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

            if "hf_model" not in model_args:
                raise ValueError(
                    "hf_preloaded provider requires 'hf_model' in model_args"
                )
            if "hf_tokenizer" not in model_args:
                raise ValueError(
                    "hf_preloaded provider requires 'hf_tokenizer' in model_args"
                )

            self.model: Any = model_args["hf_model"]
            self.tokenizer: PreTrainedTokenizerBase = model_args["hf_tokenizer"]
            self.batch_size: int = int(model_args.get("batch_size", 32))
            self._cache_tokenization: bool = bool(
                model_args.get("cache_tokenization", True)
            )
            # Hybrid-thinking models only: explicit enable_thinking for the chat
            # template (None → no kwarg, template default; non-hybrid unaffected).
            self._enable_thinking: bool | None = model_args.get("enable_thinking")

            # Ensure tokenizer is set up for batched generation.
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "left"

            self._logprobs_batcher: _LogprobsBatcher | None = None

        @override
        def close(self) -> None:
            # Do NOT destroy the model — the caller owns the lifetime.
            # But shut down the logprobs batcher thread if active.
            if self._logprobs_batcher is not None:
                self._logprobs_batcher.shutdown()
                self._logprobs_batcher = None

        @override
        def max_tokens(self) -> int | None:
            from inspect_ai._util.constants import DEFAULT_MAX_TOKENS

            return DEFAULT_MAX_TOKENS

        @override
        def max_connections(self) -> int:
            return self.batch_size

        @override
        def collapse_user_messages(self) -> bool:
            return True

        async def generate(
            self,
            input: list[ChatMessage],
            tools: list[ToolInfo],
            tool_choice: ToolChoice,
            config: GenerateConfig,
        ) -> ModelOutput:
            chat = _apply_chat_template(
                self.tokenizer,
                self.model_name,
                input,
                enable_thinking=self._enable_thinking,
            )

            # Fast path: single-token logprobs via direct forward pass.
            # Bypasses model.generate() overhead and Inspect's batch delays.
            if config.logprobs and config.max_tokens == 1:
                return await self._forward_logprobs(chat, config)

            tokenizer_fn = functools.partial(
                self.tokenizer,
                return_tensors="pt",
                padding=True,
            )

            # Use greedy decoding when temperature=0 (do_sample=True + temp=0 is
            # contradictory and triggers transformers warnings every batch).
            greedy = config.temperature is not None and config.temperature == 0.0
            kwargs: dict[str, Any] = dict(do_sample=not greedy)
            # Always set max_new_tokens to avoid the transformers default-max_length
            # warning that fires every batch when max_new_tokens is unset.
            kwargs["max_new_tokens"] = (
                config.max_tokens
                if config.max_tokens is not None
                else self.max_tokens()
            )
            if config.temperature is not None and not greedy:
                kwargs["temperature"] = config.temperature
            if config.top_p is not None:
                kwargs["top_p"] = config.top_p
            if config.top_k is not None:
                kwargs["top_k"] = config.top_k
            if config.logprobs is not None:
                kwargs["output_logits"] = config.logprobs
            if config.stop_seqs is not None:
                from transformers.generation import StopStringCriteria

                kwargs["stopping_criteria"] = [
                    StopStringCriteria(self.tokenizer, config.stop_seqs)
                ]
            kwargs["return_dict_in_generate"] = True

            generator = functools.partial(self.model.generate, **kwargs)
            decoder = functools.partial(
                self.tokenizer.batch_decode,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            device = next(self.model.parameters()).device

            # Use the built-in HF provider's batched_generate infrastructure.
            # It collects concurrent generate() calls from Inspect's async pool
            # and dispatches them as a single model.generate(batch) GPU call.
            response = await batched_generate(
                GenerateInput(
                    input=chat,
                    device=device,
                    tokenizer=tokenizer_fn,
                    generator=generator,
                    decoder=decoder,
                    batch_size=config.max_connections or self.max_connections(),
                )
            )

            # Gather logprobs if requested.
            final_logprobs = None
            if config.logprobs is not None and response.logprobs is not None:
                final_logprobs = extract_logprobs(
                    response=response,
                    top=config.top_logprobs,
                    tokenizer=self.tokenizer,
                )

            choice = ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=response.output,
                    model=self.model_name,
                    source="generate",
                ),
                logprobs=(
                    Logprobs(content=final_logprobs)
                    if final_logprobs is not None
                    else None
                ),
            )

            return ModelOutput(
                model=self.model_name,
                choices=[choice],
                usage=ModelUsage(
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    total_tokens=response.total_tokens,
                ),
                time=response.time,
            )

        async def _forward_logprobs(
            self, chat: str, config: GenerateConfig
        ) -> ModelOutput:
            """Single-token logprobs via direct forward pass."""
            if self._logprobs_batcher is None:
                device = next(self.model.parameters()).device
                self._logprobs_batcher = _LogprobsBatcher(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    batch_size=self.batch_size,
                    device=device,
                    cache_tokenization=self._cache_tokenization,
                )
                logger.info(
                    "Activated fast logprobs path (direct forward, batch=%d, "
                    "cache_tokenization=%s)",
                    self.batch_size,
                    self._cache_tokenization,
                )

            result = await self._logprobs_batcher.submit(
                chat, config.top_logprobs or 20
            )

            final_logprobs = extract_logprobs(
                response=GenerateOutput(
                    output="",
                    input_tokens=result.input_tokens,
                    output_tokens=1,
                    total_tokens=result.input_tokens + 1,
                    logprobs=result.logprobs_tensor,
                    hidden_states=None,
                    time=result.time,
                ),
                top=config.top_logprobs,
                tokenizer=self.tokenizer,
            )

            choice = ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content="",
                    model=self.model_name,
                    source="generate",
                ),
                logprobs=(
                    Logprobs(content=final_logprobs)
                    if final_logprobs is not None
                    else None
                ),
            )

            return ModelOutput(
                model=self.model_name,
                choices=[choice],
                usage=ModelUsage(
                    input_tokens=result.input_tokens,
                    output_tokens=1,
                    total_tokens=result.input_tokens + 1,
                ),
                time=result.time,
            )

    _registered = True


def _apply_chat_template(
    tokenizer: Any,
    model_name: str,
    messages: list[ChatMessage],
    *,
    enable_thinking: bool | None = None,
) -> str:
    """Convert Inspect ChatMessages to a single string via the tokenizer's chat template.

    If the last message is from the assistant (i.e. a forced prefill), we use
    ``continue_final_message=True`` so the template leaves the assistant turn
    open (no end-of-turn delimiter).  This produces tokenization identical to
    what the model saw during training, which is critical for logprob evals —
    some models (e.g. gemma) strip trailing whitespace in the template, so
    raw-appending the prefill after ``add_generation_prompt`` produces a
    different token sequence and destroys choice mass.

    ``enable_thinking`` (hybrid-thinking models only, e.g. Qwen3.x) is forwarded
    as an explicit kwarg to ``apply_chat_template`` so the eval can render with
    thinking on/off regardless of how the LoRA was trained. None passes no kwarg,
    leaving the template's own default — so llama/gemma/Qwen2.5 are unaffected.
    """
    thinking_kw = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    hf_messages = copy.deepcopy(emulate_reasoning_history(messages))

    # Flatten any list content to text (no multimodal support).
    for message in hf_messages:
        if isinstance(message.content, list):
            if any(
                isinstance(
                    item, ContentAudio | ContentImage | ContentVideo | ContentDocument
                )
                for item in message.content
            ):
                raise NotImplementedError(
                    "hf_preloaded provider does not support multimodal content"
                )
            message.content = message.text

    # Detect trailing assistant prefill.
    has_assistant_prefill = hf_messages and hf_messages[-1].role == "assistant"

    # Convert Inspect ChatMessage objects to plain dicts for apply_chat_template.
    hf_dicts = []
    for m in hf_messages:
        content = m.text if hasattr(m, "text") else str(m.content)
        hf_dicts.append({"role": m.role, "content": content})

    if tokenizer.chat_template is not None:
        if has_assistant_prefill:
            # Use continue_final_message to keep the assistant turn open.
            # This lets the template handle whitespace/formatting naturally,
            # producing identical tokenization to training.
            chat = str(
                tokenizer.apply_chat_template(
                    hf_dicts,
                    add_generation_prompt=False,
                    continue_final_message=True,
                    tokenize=False,
                    **thinking_kw,
                )
            )
        else:
            chat = str(
                tokenizer.apply_chat_template(
                    hf_dicts,
                    add_generation_prompt=True,
                    tokenize=False,
                    **thinking_kw,
                )
            )
    else:
        # Fallback for tokenizers without a chat template.
        chat = "".join(f"{m['role']}: {m['content']}\n" for m in hf_dicts)

    return chat
