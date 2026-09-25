"""Code-based editors for non-LLM editing."""

from __future__ import annotations

import re
import string


_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
_CYCLIC_VOWEL_MAP = str.maketrans(
    {
        "a": "e",
        "e": "i",
        "i": "o",
        "o": "u",
        "u": "a",
        "A": "E",
        "E": "I",
        "I": "O",
        "O": "U",
        "U": "A",
    }
)


def reverse_text(text: str, record: dict) -> str:
    """Reverse text as a simple demo editor.

    Args:
        text: Response text to edit.
        record: Full record metadata (unused).

    Returns:
        Reversed text.
    """
    _ = record
    return text[::-1]


def strip_punct_and_lower(text: str, record: dict) -> str:
    """Lowercase text and remove ASCII punctuation characters.

    Args:
        text: Response text to edit.
        record: Full record metadata (unused).

    Returns:
        Lowercased text with punctuation removed.
    """
    _ = record
    lowered = text.lower()
    return lowered.translate(_PUNCTUATION_TABLE)


def identity_text(text: str, record: dict) -> str:
    """Return text unchanged.

    Args:
        text: Response text to edit.
        record: Full record metadata (unused).

    Returns:
        Unmodified input text.
    """
    _ = record
    return text


def cyclic_vowel_shift(text: str, record: dict) -> str:
    """Replace vowels with the next vowel in the cycle a->e->i->o->u->a.

    Args:
        text: Response text to edit.
        record: Full record metadata (unused).

    Returns:
        Text with all ASCII vowels replaced cyclically, preserving case.
    """
    _ = record
    return text.translate(_CYCLIC_VOWEL_MAP)


def cyclic_vowel_shift_word_debug(text: str, record: dict) -> str:
    """Debug helper that annotates before/after counts for `the` and `thi`."""
    edited = cyclic_vowel_shift(text, record)
    the_count = len(re.findall(r"\bthe\b", edited, flags=re.IGNORECASE))
    thi_count = len(re.findall(r"\bthi\b", edited, flags=re.IGNORECASE))
    return f"[the={the_count} thi={thi_count}] {edited}"
