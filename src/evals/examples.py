"""Example builder functions for Inspect custom eval specs.

Each ``*_input_builder`` maps one raw dataset row (a dict) to the prompt
string fed to the model under test; ``question_target_builder`` extracts the
reference answer. They exist so a custom eval can point ``input_builder`` /
``target_builder`` at a stable callable path (e.g. via the CLI ``direct``
command) for common dataset schemas without writing bespoke glue per run.
"""

from __future__ import annotations

from typing import Any


def question_input_builder(row: dict[str, Any]) -> str:
    return str(row.get("question", ""))


def oasst1_input_builder(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("question") or "")


def no_robots_input_builder(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    if messages and isinstance(messages[0], dict):
        return str(messages[0].get("content", ""))
    return str(row.get("prompt", ""))


def prompt_eng_input_builder(row: dict[str, Any]) -> str:
    return str(
        row.get("Prompt")
        or row.get("prompt")
        or row.get("question")
        or row.get("instruction")
        or row.get("text")
        or ""
    )


def question_target_builder(row: dict[str, Any]) -> str:
    return str(row.get("best_answer", ""))
