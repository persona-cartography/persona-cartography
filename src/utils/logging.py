"""Structured logging setup."""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the project logger.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured logger instance.
    """
    level_value = logging.getLevelName(level.upper())
    if isinstance(level_value, str):
        raise ValueError(f"Unknown logging level: {level}")

    logger = logging.getLogger("persona_shattering")
    logger.setLevel(level_value)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Suppress per-request transport logs from SDK HTTP clients while keeping
    # our own component logs at the configured level.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    return logger
