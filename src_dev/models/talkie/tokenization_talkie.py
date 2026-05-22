"""TalkieTokenizerFast — HF wrapper around the talkie tiktoken BPE.

The reference (``talkie.tokenizer``) loads ``vocab.txt`` (base64-encoded
byte BPE ranks) into a ``tiktoken.Encoding`` with a custom regex
pre-tokenizer and 4 chat-mode specials. This module builds a
``tokenizers.Tokenizer`` JSON that produces identical output, wraps it
in a ``PreTrainedTokenizerFast`` so HF transformers / OpenRLHF can load
it via ``from_pretrained(..., trust_remote_code=True)``.

The conversion is byte-level BPE using the GPT-2 ``bytes_to_unicode``
encoding (HF ``ByteLevel`` pre-tokenizer + decoder). This is the same
representation HF uses for GPT-2 / Llama and means our vocab keys are
the unicode-encoded forms of the underlying byte sequences.

Special tokens (matching ``talkie.tokenizer._IT_SPECIAL_TOKENS``):

    <|endoftext|>=65535
    <|end|>=65536
    <|user|>=65537
    <|assistant|>=65538
    <|system|>=65539
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Iterable

from transformers import PreTrainedTokenizerFast


# Talkie's pre-tokenizer regex (same as ``talkie.tokenizer._PAT_STR``).
TALKIE_PAT_STR = "|".join(
    [
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
        r"""\p{N}{1,3}""",
        r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
        r"""\s*[\r\n]+""",
        r"""\s+(?!\S)""",
        r"""\s+""",
    ]
)

# Reference (talkie/tokenizer.py): BASE_VOCAB_SIZE = 65536, keep ranks < 65535,
# then <|endoftext|> = 65535, IT specials at 65536-65539.
TALKIE_BASE_VOCAB = 65535  # number of BPE ranks (0..65534)
TALKIE_IT_SPECIALS: dict[str, int] = {
    "<|endoftext|>": 65535,
    "<|end|>": 65536,
    "<|user|>": 65537,
    "<|assistant|>": 65538,
    "<|system|>": 65539,
}


# ---------------------------------------------------------------------------
# GPT-2 byte-level encoding (mirrors transformers / tokenizers internals)
# ---------------------------------------------------------------------------


def _bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 reversible byte → unicode mapping."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTES_TO_UNICODE = _bytes_to_unicode()


def _bytes_to_byte_str(b: bytes) -> str:
    return "".join(_BYTES_TO_UNICODE[byte] for byte in b)


# ---------------------------------------------------------------------------
# tiktoken → HF tokenizer conversion
# ---------------------------------------------------------------------------


def load_talkie_mergeable_ranks(vocab_path: str | Path) -> dict[bytes, int]:
    """Load talkie's vocab.txt (base64 BPE ranks) into a {bytes: rank} dict.

    Drops ranks >= ``TALKIE_BASE_VOCAB`` (mirrors the reference filter in
    ``talkie/tokenizer.py:54``).
    """
    mergeable_ranks: dict[bytes, int] = {}
    with open(vocab_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            b64, rank_s = line.rsplit(" ", 1)
            rank = int(rank_s)
            if rank >= TALKIE_BASE_VOCAB:
                continue
            mergeable_ranks[base64.b64decode(b64)] = rank
    return mergeable_ranks


def _split_for_merge(
    mergeable_ranks: dict[bytes, int], token: bytes, max_rank: int
) -> tuple[bytes, bytes] | None:
    """Run greedy BPE on ``token`` with rank cap ``max_rank``.

    Returns the (left, right) pair that BPE would merge last to produce
    ``token`` — i.e. the merge entry HF needs to recover this token.
    Returns ``None`` for length-1 tokens (no merge needed).
    """
    if len(token) < 2:
        return None
    parts = [bytes([b]) for b in token]
    while True:
        min_idx, min_rank = None, None
        for i in range(len(parts) - 1):
            pair = parts[i] + parts[i + 1]
            rank = mergeable_ranks.get(pair)
            if rank is not None and rank < max_rank and (min_rank is None or rank < min_rank):
                min_idx, min_rank = i, rank
        if min_rank is None:
            break
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


def build_talkie_tokenizer(vocab_path: str | Path):
    """Build a ``tokenizers.Tokenizer`` equivalent to talkie's tiktoken encoder.

    Returns the constructed ``Tokenizer`` (not the HF PreTrained wrapper).
    """
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

    ranks = load_talkie_mergeable_ranks(vocab_path)
    by_rank = sorted(ranks.items(), key=lambda kv: kv[1])  # ascending

    # Vocab keys are byte-level-encoded strings.
    vocab: dict[str, int] = {_bytes_to_byte_str(b): r for b, r in ranks.items()}

    # Build merges: for each non-leaf rank, find the two lower-rank tokens
    # that produce it under greedy BPE.
    merges: list[tuple[str, str]] = []
    skipped_unmergeable = 0
    for tok_bytes, rank in by_rank:
        if rank < 256:
            continue
        pair = _split_for_merge(ranks, tok_bytes, rank)
        if pair is None:
            skipped_unmergeable += 1
            continue
        merges.append((_bytes_to_byte_str(pair[0]), _bytes_to_byte_str(pair[1])))
    if skipped_unmergeable:
        print(
            f"[talkie tokenizer] WARNING: {skipped_unmergeable} tokens had no "
            "deterministic BPE merge — these may be tokens that tiktoken would "
            "only emit via byte-fallback or special handling."
        )

    bpe = models.BPE(
        vocab=vocab,
        merges=merges,
        byte_fallback=False,
        fuse_unk=False,
        unk_token=None,
    )
    tok = Tokenizer(bpe)
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(TALKIE_PAT_STR), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    # Add IT specials. Their ids fall in the contiguous block 65535..65539.
    tok.add_special_tokens(list(TALKIE_IT_SPECIALS.keys()))
    return tok


# ---------------------------------------------------------------------------
# HF wrapper
# ---------------------------------------------------------------------------


# Jinja template that matches ``talkie.chat.format_chat`` exactly.
TALKIE_CHAT_TEMPLATE = (
    "{%- for m in messages -%}"
    "{%- if m['role'] == 'system' -%}<|system|>{{ m['content'] }}<|end|>"
    "{%- elif m['role'] == 'user' -%}<|user|>{{ m['content'] }}<|end|>"
    "{%- elif m['role'] == 'assistant' -%}<|assistant|>{{ m['content'] }}<|end|>"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|assistant|>{%- endif -%}"
)


class TalkieTokenizerFast(PreTrainedTokenizerFast):
    """``PreTrainedTokenizerFast`` wrapper exposing the talkie tokenizer.

    No model-side state; this exists so the model directory can declare
    ``auto_map["AutoTokenizer"] = "tokenization_talkie.TalkieTokenizerFast"``
    and have ``from_pretrained(..., trust_remote_code=True)`` resolve to a
    consistent class.
    """

    vocab_files_names: dict = {"vocab_file": "vocab.txt"}
    model_input_names: list[str] = ["input_ids", "attention_mask"]


def save_talkie_tokenizer(out_dir: str | Path, vocab_path: str | Path) -> None:
    """Build the talkie tokenizer and save it as a complete HF directory.

    Produces ``tokenizer.json``, ``tokenizer_config.json``,
    ``special_tokens_map.json``, and copies the source ``vocab.txt`` into
    the dir for reproducibility (not used at load time).
    """
    from shutil import copyfile

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = build_talkie_tokenizer(vocab_path)
    tok.save(str(out_dir / "tokenizer.json"))

    # Persist vocab.txt for reproducibility.
    copyfile(str(vocab_path), str(out_dir / "vocab.txt"))

    tokenizer_config = {
        "tokenizer_class": "TalkieTokenizerFast",
        "auto_map": {
            "AutoTokenizer": ["tokenization_talkie.TalkieTokenizerFast", None],
        },
        "model_max_length": 4096,
        "padding_side": "left",
        "bos_token": None,
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
        "unk_token": None,
        "added_tokens_decoder": {
            str(v): {
                "content": k,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
            for k, v in TALKIE_IT_SPECIALS.items()
        },
        "chat_template": TALKIE_CHAT_TEMPLATE,
        "clean_up_tokenization_spaces": False,
    }
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2, ensure_ascii=False)
    )

    special_tokens_map = {
        "eos_token": {
            "content": "<|endoftext|>",
            "lstrip": False,
            "rstrip": False,
            "single_word": False,
            "normalized": False,
        },
        "pad_token": {
            "content": "<|endoftext|>",
            "lstrip": False,
            "rstrip": False,
            "single_word": False,
            "normalized": False,
        },
        "additional_special_tokens": [
            k for k in TALKIE_IT_SPECIALS if k != "<|endoftext|>"
        ],
    }
    (out_dir / "special_tokens_map.json").write_text(
        json.dumps(special_tokens_map, indent=2, ensure_ascii=False)
    )


__all__ = [
    "TALKIE_PAT_STR",
    "TALKIE_BASE_VOCAB",
    "TALKIE_IT_SPECIALS",
    "TALKIE_CHAT_TEMPLATE",
    "TalkieTokenizerFast",
    "build_talkie_tokenizer",
    "load_talkie_mergeable_ranks",
    "save_talkie_tokenizer",
]
