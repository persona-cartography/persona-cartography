"""Conversation-reset prompt builders for mid-context questionnaire administration.

When a questionnaire item is appended after a long in-context "rollout" we may
want to signal to the model that the previous conversation has ended and a new
one is beginning. This module builds the inputs for three reset strategies:

    * ``"none"``         — current baseline. The item is appended as a new user
                           turn; the rollout history is fully visible as normal
                           chat context.
    * ``"soft"``         — a standard ``system`` message is inserted between
                           the rollout tail and the new user turn, stating that
                           the previous conversation has ended.
    * ``"token_boundary"``— the prompt is built at the token-ID level: the
                           rollout is tokenised as a complete chat, an
                           ``<|end_of_text|>`` (or equivalent) token is
                           appended, then a second chat beginning with
                           ``<|begin_of_text|>`` containing just the
                           questionnaire item is tokenised and concatenated.
                           Attention remains fully causal across the boundary;
                           only the token-level signal differs.

The first two strategies operate at the messages level and are compatible with
any chat-completion provider. The third strategy must be paired with a
provider that accepts raw token IDs (currently only vLLM in this project).

The functions below are tokenizer-agnostic as far as is practical, but
``token_boundary`` inherently depends on the underlying tokenizer's chat
template and special-token layout. The helpers return plain dataclass-like
dicts so they can be unit-tested without loading a full vLLM engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

ResetMode = Literal["none", "soft", "token_boundary"]
RESET_MODES: tuple[ResetMode, ...] = ("none", "soft", "token_boundary")

# Note: llama-3.x Instruct models were not trained with mid-sequence
# ``role="system"`` messages (and ``token_boundary`` uses a mid-sequence
# ``<|end_of_text|>``/``<|begin_of_text|>`` pair). Both strategies are
# therefore out-of-distribution signals: the chat template renders them
# cleanly, but how the model interprets them is an empirical question. This
# is a deliberate choice — we want a strong "reset" cue — but callers should
# treat results from ``soft`` and ``token_boundary`` modes accordingly.
#
# The system prompt intentionally does NOT instruct the model to "not
# reference or continue the previous conversation". Whether a drifted
# persona persists or resets across the boundary is what we are trying to
# measure; explicitly instructing a reset would contaminate that.
DEFAULT_SOFT_RESET_SYSTEM_PROMPT = (
    "The previous conversation has ended. A new, independent conversation "
    "is now beginning."
)


@dataclass(frozen=True)
class MessagesPrompt:
    """A prompt expressed as a messages list (for chat-template providers)."""

    messages: list[dict[str, str]]


@dataclass(frozen=True)
class TokenIdsPrompt:
    """A prompt expressed as raw token IDs (for ``prompt_token_ids`` providers).

    ``num_rollout_tokens`` records how many leading tokens came from the
    rollout side of the boundary — useful for validation, prefix-cache
    accounting and for constructing retry prompts.
    """

    token_ids: list[int]
    num_rollout_tokens: int
    boundary_token_ids: list[int]


def _validate_mode(reset_mode: str) -> ResetMode:
    if reset_mode not in RESET_MODES:
        raise ValueError(
            f"Unknown reset_mode={reset_mode!r}; expected one of {RESET_MODES}."
        )
    return reset_mode  # type: ignore[return-value]


def build_messages_prompt(
    rollout_messages: Sequence[dict[str, str]],
    item_user_content: str,
    *,
    reset_mode: ResetMode = "none",
    soft_reset_system_prompt: str = DEFAULT_SOFT_RESET_SYSTEM_PROMPT,
    trait_mcq_prefill: str | None = None,
) -> MessagesPrompt:
    """Build a messages-list prompt for the ``none`` or ``soft`` reset modes.

    Args:
        rollout_messages: The completed rollout conversation to carry as
            context. Each entry is a ``{"role": ..., "content": ...}`` dict.
        item_user_content: The rendered questionnaire item (user-turn text).
        reset_mode: ``"none"`` (no reset) or ``"soft"`` (system message
            boundary). ``"token_boundary"`` is **not** supported here — use
            :func:`build_token_ids_prompt`.
        soft_reset_system_prompt: System-message text injected between the
            rollout tail and the new user turn under ``"soft"``.
        trait_mcq_prefill: Optional assistant-turn prefill (e.g. ``"Answer"``)
            to be treated as a partial assistant turn the model continues from.
            When provided, the final message in the returned list has role
            ``assistant``, signalling that the provider should apply the chat
            template with ``continue_final_message=True``.

    Returns:
        A :class:`MessagesPrompt` ready to be fed to a chat-completion provider.
    """
    mode = _validate_mode(reset_mode)
    if mode == "token_boundary":
        raise ValueError(
            "token_boundary mode requires build_token_ids_prompt(); it cannot "
            "be expressed as a messages list because it relies on raw "
            "mid-sequence <|end_of_text|> / <|begin_of_text|> tokens."
        )

    messages: list[dict[str, str]] = [dict(m) for m in rollout_messages]
    if mode == "soft":
        messages.append({"role": "system", "content": soft_reset_system_prompt})
    messages.append({"role": "user", "content": item_user_content})
    if trait_mcq_prefill is not None:
        messages.append({"role": "assistant", "content": trait_mcq_prefill})
    return MessagesPrompt(messages=messages)


def _tokenize_rollout(
    tokenizer,
    rollout_messages: Sequence[dict[str, str]],
) -> list[int]:
    """Tokenise the rollout side with its terminal assistant ``<|eot_id|>``.

    The rollout is expected to end with an assistant turn (completed rollout).
    ``add_generation_prompt=False`` and ``continue_final_message=False`` ensure
    no trailing assistant header is emitted — the final token is the
    ``<|eot_id|>`` closing the rollout's last assistant message.
    """
    ids = tokenizer.apply_chat_template(
        list(rollout_messages),
        tokenize=True,
        add_generation_prompt=False,
    )
    return list(ids)


def _tokenize_fresh_user_turn(
    tokenizer,
    item_user_content: str,
    *,
    trait_mcq_prefill: str | None,
) -> list[int]:
    """Tokenise a fresh chat containing only the questionnaire item.

    Because ``apply_chat_template`` always emits a leading ``<|begin_of_text|>``
    (and may auto-insert a default system block for the tokenizer at hand), the
    returned token sequence is "as-if" the model were seeing a brand-new
    conversation.

    When ``trait_mcq_prefill`` is set, the final turn is an assistant prefill
    (rendered via ``continue_final_message=True``) so the model continues from
    the prefill rather than from a fresh assistant header.
    """
    messages: list[dict[str, str]] = [{"role": "user", "content": item_user_content}]
    if trait_mcq_prefill is not None:
        messages.append({"role": "assistant", "content": trait_mcq_prefill})
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    else:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    return list(ids)


def _resolve_boundary_token_ids(
    tokenizer,
    boundary_token: str | int | Sequence[int],
) -> list[int]:
    """Resolve the boundary-token spec to a list of token IDs.

    Accepts:
        * an integer token ID (e.g. ``128001`` for llama's ``<|end_of_text|>``),
        * a string token to pass through ``convert_tokens_to_ids`` (e.g.
          ``"<|end_of_text|>"``),
        * or an explicit list/tuple of token IDs.
    """
    if isinstance(boundary_token, int):
        return [int(boundary_token)]
    if isinstance(boundary_token, str):
        tok_id = tokenizer.convert_tokens_to_ids(boundary_token)
        if tok_id is None or (
            isinstance(tok_id, int) and tok_id == tokenizer.unk_token_id
        ):
            raise ValueError(
                f"Boundary token {boundary_token!r} is not a known special "
                f"token for this tokenizer."
            )
        return [int(tok_id)]
    if isinstance(boundary_token, (list, tuple)):
        return [int(t) for t in boundary_token]
    raise TypeError(
        f"Unsupported boundary_token type: {type(boundary_token).__name__}"
    )


def build_token_ids_prompt(
    tokenizer,
    rollout_messages: Sequence[dict[str, str]],
    item_user_content: str,
    *,
    boundary_token: str | int | Sequence[int] = "<|end_of_text|>",
    trait_mcq_prefill: str | None = None,
) -> TokenIdsPrompt:
    """Build a raw-token-ID prompt for the ``token_boundary`` reset mode.

    The output layout is::

        <rollout_chat_tokens>            # with trailing <|eot_id|>, no gen prompt
        <boundary_token>                 # e.g. <|end_of_text|> (128001 for llama3)
        <fresh_item_chat_tokens>         # starts with <|begin_of_text|>

    The model sees a single causal sequence but with an out-of-distribution
    end-of-sequence / start-of-sequence pair at the boundary.

    Args:
        tokenizer: A HuggingFace tokenizer supporting ``apply_chat_template``.
        rollout_messages: Completed rollout conversation.
        item_user_content: The rendered questionnaire item.
        boundary_token: Token(s) inserted between the rollout and the fresh
            item chat. Defaults to ``"<|end_of_text|>"`` (llama 3.x). Other
            tokenizer families may need a different token — pass the desired
            special-token string or an integer token ID directly.
        trait_mcq_prefill: See :func:`build_messages_prompt`.

    Returns:
        A :class:`TokenIdsPrompt` carrying the concatenated token IDs plus
        bookkeeping fields useful for validation and retry construction.
    """
    rollout_ids = _tokenize_rollout(tokenizer, rollout_messages)
    boundary_ids = _resolve_boundary_token_ids(tokenizer, boundary_token)
    fresh_ids = _tokenize_fresh_user_turn(
        tokenizer,
        item_user_content,
        trait_mcq_prefill=trait_mcq_prefill,
    )
    return TokenIdsPrompt(
        token_ids=[*rollout_ids, *boundary_ids, *fresh_ids],
        num_rollout_tokens=len(rollout_ids),
        boundary_token_ids=boundary_ids,
    )


def build_token_ids_retry_prompt(
    tokenizer,
    rollout_messages: Sequence[dict[str, str]],
    item_user_content: str,
    prior_assistant_text: str,
    retry_user_content: str,
    *,
    boundary_token: str | int | Sequence[int] = "<|end_of_text|>",
    trait_mcq_prefill: str | None = None,
) -> TokenIdsPrompt:
    """Build a token-ID retry prompt under ``token_boundary`` reset.

    The fresh (post-boundary) side of the prompt is extended with the prior
    parse-failed assistant turn and a new user retry message, mirroring the
    behaviour of the messages-list retry path.

    Args:
        prior_assistant_text: The raw text the model produced on the previous
            attempt (without any prefill). For trait_mcq items this is the
            letter+continuation the model produced after the prefill; the
            returned prompt reconstructs the full prior assistant turn as
            ``trait_mcq_prefill + prior_assistant_text``.
        retry_user_content: The retry message (e.g. "Please respond with only
            'A', 'B', 'C', or 'D'.").
    """
    rollout_ids = _tokenize_rollout(tokenizer, rollout_messages)
    boundary_ids = _resolve_boundary_token_ids(tokenizer, boundary_token)

    prior_full = (
        (trait_mcq_prefill + prior_assistant_text)
        if trait_mcq_prefill is not None
        else prior_assistant_text
    )
    fresh_messages: list[dict[str, str]] = [
        {"role": "user", "content": item_user_content},
        {"role": "assistant", "content": prior_full},
        {"role": "user", "content": retry_user_content},
    ]
    if trait_mcq_prefill is not None:
        fresh_messages.append({"role": "assistant", "content": trait_mcq_prefill})
        fresh_ids = tokenizer.apply_chat_template(
            fresh_messages,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    else:
        fresh_ids = tokenizer.apply_chat_template(
            fresh_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    fresh_ids = list(fresh_ids)
    return TokenIdsPrompt(
        token_ids=[*rollout_ids, *boundary_ids, *fresh_ids],
        num_rollout_tokens=len(rollout_ids),
        boundary_token_ids=boundary_ids,
    )


__all__ = [
    "DEFAULT_SOFT_RESET_SYSTEM_PROMPT",
    "MessagesPrompt",
    "RESET_MODES",
    "ResetMode",
    "TokenIdsPrompt",
    "build_messages_prompt",
    "build_token_ids_prompt",
    "build_token_ids_retry_prompt",
]
