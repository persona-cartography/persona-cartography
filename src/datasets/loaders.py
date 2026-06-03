"""Dataset loading and formatting utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from datasets import Dataset, load_dataset as hf_load_dataset

from src.datasets.core import load_samples, materialize_canonical_samples, render_messages

if TYPE_CHECKING:
    from src.common.config import DatasetConfig


def load_dataset_from_config(config: "DatasetConfig") -> Dataset:
    """Load a dataset based on the DatasetConfig.

    Args:
        config: Dataset configuration.

    Returns:
        A HuggingFace Dataset object.
    """
    if config.source == "huggingface":
        if not config.name:
            raise ValueError("HuggingFace source requires dataset name.")
        dataset = hf_load_dataset(config.name, split=config.split)
    elif config.source == "local":
        if not config.path:
            raise ValueError("Local source requires dataset path.")
        dataset = hf_load_dataset(
            "json",
            data_files=str(Path(config.path)),
            split="train",
        )
    elif config.source == "canonical":
        if not config.path:
            raise ValueError("Canonical source requires run directory path.")
        run_path = Path(config.path)
        if run_path.is_file():
            rows = []
            with run_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.strip()
                    if text:
                        rows.append(json.loads(text))
        else:
            materialize_canonical_samples(run_path)
            samples = load_samples(run_path)
            rows = []
            target_variant = None
            if isinstance(config.name, str) and config.name.startswith("editing:"):
                target_variant = config.name.split(":", 1)[1]

            for sample in samples:
                effective_messages = render_messages(sample)
                if target_variant:
                    effective_messages = render_messages(sample, target_variant)

                user_messages = [msg.content for msg in effective_messages if msg.role == "user"]
                assistant_messages = [
                    msg.content for msg in effective_messages if msg.role == "assistant"
                ]
                question = user_messages[-1] if user_messages else ""
                response = assistant_messages[-1] if assistant_messages else ""
                row = {
                    "sample_id": sample.sample_id,
                    "input_group_id": sample.input_group_id or sample.sample_id,
                    "response_index": sample.response_index,
                    "messages": [m.model_dump() for m in effective_messages],
                    "latest_user_message": question,
                    "latest_assistant_message": response,
                    "assistant_turn_count": len(assistant_messages),
                    "question": question,
                    "response": response,
                }

                if len(user_messages) == 1:
                    row["question"] = user_messages[0]
                if len(assistant_messages) == 1:
                    row["response"] = assistant_messages[0]

                rows.append(row)
        dataset = Dataset.from_list(rows)
    else:
        raise ValueError(f"Unsupported dataset source: {config.source}")

    if config.seed is not None:
        dataset = dataset.shuffle(seed=config.seed)

    if config.max_samples is not None:
        dataset = dataset.select(range(min(len(dataset), config.max_samples)))

    return dataset


def format_for_inference(dataset: Dataset, question_column: str | None = None) -> Dataset:
    """Format a raw dataset for the inference stage.

    Args:
        dataset: The dataset to format.
        question_column: Name of the column containing questions. If None, will try
            common column names: "question", "instruction", "prompt", "text".

    Returns:
        Dataset with a single "question" column.
    """
    if question_column is None:
        # Try common column names
        common_names = ["question", "instruction", "prompt", "text"]
        for name in common_names:
            if name in dataset.column_names:
                question_column = name
                break
        if question_column is None:
            raise ValueError(
                f"Could not find question column. Available columns: {dataset.column_names}. "
                f"Expected one of: {common_names}"
            )

    if question_column not in dataset.column_names:
        raise ValueError(f"Question column '{question_column}' not found in dataset.")

    # If the dataset has an auxiliary input field, append it to the question and
    # persist that merged text as the canonical "question" value written to JSONL.
    # This preserves instruction+input style datasets (e.g., Alpaca) without
    # requiring callers to special-case formatting.
    auxiliary_column = "input" if "input" in dataset.column_names else None

    if auxiliary_column is not None:
        def _merge_question_input(example: dict) -> dict:
            question_raw = example.get(question_column, "")
            extra_raw = example.get(auxiliary_column, "")
            question = question_raw if isinstance(question_raw, str) else str(question_raw)
            extra = extra_raw if isinstance(extra_raw, str) else str(extra_raw)

            if extra.strip():
                merged = f"{question}\n\n{extra}" if question else extra
            else:
                merged = question
            return {"question": merged}

        dataset = dataset.map(_merge_question_input)
    elif question_column != "question":
        dataset = dataset.rename_column(question_column, "question")

    extra_columns = [col for col in dataset.column_names if col != "question"]
    if extra_columns:
        dataset = dataset.remove_columns(extra_columns)

    return dataset
