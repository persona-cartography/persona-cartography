"""Prompt templates for user-simulator turns and system prompts in rollout generation.

Templates can be registered dynamically by experiment scripts via
``register_user_simulator_template`` and ``register_system_prompt_template``.
"""

from __future__ import annotations

from collections.abc import Sequence

# ── System prompt templates (for assistant generation) ────────────────────────

SYSTEM_PROMPT_TEMPLATES: dict[str, str] = {}


def register_system_prompt_template(name: str, text: str) -> None:
    """Register a system prompt template for assistant generation."""
    SYSTEM_PROMPT_TEMPLATES[name] = text


def get_system_prompt_template(name: str) -> str:
    """Return a system prompt template by name."""
    try:
        return SYSTEM_PROMPT_TEMPLATES[name]
    except KeyError as exc:
        available = ", ".join(sorted(SYSTEM_PROMPT_TEMPLATES))
        raise ValueError(
            f"Unknown system prompt template: {name!r}. "
            f"Available templates: [{available}]"
        ) from exc


# ── User simulator templates ──────────────────────────────────────────────────

USER_SIMULATOR_TEMPLATES: dict[str, str] = {
    "typical_user": (
        "You are writing the next USER turn in this conversation. "
        "Respond as a typical human user having a general-interest chat with an assistant. "
        "The user is genuinely curious and engaged, but is not conducting formal research or trying to exhaustively analyze the topic. "
        "Assume the user knows some basics but is not an expert. "
        "Have the user pick up on concrete points from the assistant's latest message that they would naturally want to understand better. "
        "Favor natural back-and-forth conversation over structured interrogation or checklist-style questioning. "
        "The user may occasionally share opinions or ask what the assistant thinks, but that should not be the focus of all the user's messages - it should be done as part of keeping a normal, flowing conversation. "
        "Write only the next user message in plain text. "
        "Do not answer as the assistant. "
        "Do not offer help as if you are the assistant. "
        "Do not include role labels like 'User:' or 'Assistant:'. "
        "React naturally to the assistant's latest message and continue the dialogue in a realistic way. "
        "The user should sound coherent, specific when useful, and plausibly imperfect. "
        "Return nothing except the user message."
    ),
}


def register_user_simulator_template(name: str, text: str) -> None:
    """Register a custom user simulator template."""
    USER_SIMULATOR_TEMPLATES[name] = text


def get_user_simulator_instruction(template_name: str) -> str:
    """Return system instruction for the configured user-simulator template."""
    try:
        return USER_SIMULATOR_TEMPLATES[template_name]
    except KeyError as exc:
        available = ", ".join(sorted(USER_SIMULATOR_TEMPLATES))
        raise ValueError(
            f"Unknown user simulator prompt template: {template_name!r}. "
            f"Available templates: [{available}]"
        ) from exc


def render_user_simulator_single_turn_prompt(
    template_name: str, messages: Sequence[dict[str, str]]
) -> str:
    """Render user simulator input as a single text prompt with transcript context."""
    instruction = get_user_simulator_instruction(template_name)
    transcript = _render_transcript(messages)
    return (
        f"{instruction}\n\n"
        "Here is the conversation so far.\n"
        "```\n"
        f"{transcript}\n"
        "```\n\n"
        "Remember, you are writing the next USER response. Please do so in a way that is natural, "
        "continues the conversation, and behaves as if you are a human user talking to the assistant, "
        "not an AI assistant.\n"
        "Return nothing except the next user message."
    )


def _render_transcript(messages: Sequence[dict[str, str]]) -> str:
    sections: list[str] = []
    role_map = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
        "tool": "Tool",
    }

    for message in messages:
        role = role_map.get(str(message.get("role", "")).lower(), "Message")
        content = str(message.get("content", "")).strip()
        sections.append(f"## {role}\n{content}")

    return "\n\n".join(sections).strip()
