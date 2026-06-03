"""Reproducibility seed helper shared across eval runners."""

from __future__ import annotations

import random

import numpy as np


def seed_all(seed: int) -> None:
    """Set seeds for all relevant RNGs (stdlib, numpy, torch).

    Torch is imported lazily so that scripts that don't need it
    (e.g. ``--dry-run``) avoid the heavy import.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
