"""Editing module for LLM- or code-based response editing with quality tracking.

Example:
    from src_dev.lora_pipeline_legacy.editing import run_editing, EditingConfig

    config = EditingConfig(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        prompt_template="default_persona_shatter",
        output_path=Path("scratch/edited.jsonl"),
    )
    dataset, result = run_editing(config, input_dataset)
"""

from src_dev.lora_pipeline_legacy.editing.config import (
    EditingConfig,
    EditingResult,
    RetryConfig,
    AnthropicProviderConfig,
    OpenAIProviderConfig,
    CodeProviderConfig,
    QualityConfig,
)
from src_dev.lora_pipeline_legacy.editing.prompts import EditPromptContext, TEMPLATES, get_prompt

# Import run_editing after other imports to avoid circular imports
def _get_run_editing():
    from src_dev.lora_pipeline_legacy.editing.run import run_editing
    return run_editing

def _get_main():
    from src_dev.lora_pipeline_legacy.editing.cli import main
    return main

__all__ = [
    # Config classes
    "EditingConfig",
    "EditingResult",
    "RetryConfig",
    "AnthropicProviderConfig",
    "OpenAIProviderConfig",
    "CodeProviderConfig",
    "QualityConfig",
    # Prompts
    "TEMPLATES",
    "EditPromptContext",
    "get_prompt",
    # Run function
    "run_editing",
    # CLI entry point
    "main",
]

def __getattr__(name):
    if name == "run_editing":
        return _get_run_editing()
    if name == "main":
        return _get_main()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
