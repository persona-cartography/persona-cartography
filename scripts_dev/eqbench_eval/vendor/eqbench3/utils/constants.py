# File: ai/eqbench3/utils/constants.py

import os

# Scenario ids that are not framed as role-play, i.e. they are a direct convo
# between user & assistant
NO_RP_SCENARIO_IDS = {
    "8",
    "10",
    "12",
    "14",
    #"101","131","132","133","134","135","136","137","138","139","140","141","142","143","144","145","146"
}

MESSAGE_DRAFTING_SCENARIO_IDS = {
    "201", "202", "203", "204", "205", "206", "207", "208", "209"
}

# --- NEW: Analysis Scenario IDs ---
ANALYSIS_SCENARIO_IDS = {
    "401", "402", "403", "404", "405", "406", "407", "408", "409", "410",
    "411", "412", "413", "414", "415", "416", "417", "418", "419", "420",
}

# Character‑limits for the three parsed sections in a message‑drafting reply
SECTION_CHAR_LIMITS_MESSAGE_DRAFT = {
    "perspective_taking":   2200, # Updated key name
    "draft_brainstorming": 1600, # Updated key name
    "draft":      1600,
}


# Character limits for each parsed section (Role Play)
SECTION_CHAR_LIMITS = {
    "thinking_feeling": 2200,
    "their_thinking_feeling": 1600,
    "response": 1600
}

# Character limits for raw responses (NO_RP and ANALYSIS scenarios)
RAW_RESPONSE_CHAR_LIMIT = 4000 # Used for NO_RP during judging, potentially for ANALYSIS too
DEBRIEF_CHAR_LIMIT = 4000 # Used for standard/drafting debriefs
ANALYSIS_RESPONSE_CHAR_LIMIT = 6000

# --- Hardcoded File Paths ---
DATA_DIR = "data"
# Standard Task Files
STANDARD_SCENARIO_PROMPTS_FILE = os.path.join(DATA_DIR, "scenario_prompts.txt")
STANDARD_MASTER_PROMPT_FILE = os.path.join(DATA_DIR, "scenario_master_prompt.txt")
STANDARD_DEBRIEF_PROMPT_FILE = os.path.join(DATA_DIR, "debrief_prompt.txt")
STANDARD_RUBRIC_CRITERIA_FILE = os.path.join(DATA_DIR, "rubric_scoring_criteria.txt")
STANDARD_RUBRIC_PROMPT_FILE = os.path.join(DATA_DIR, "rubric_scoring_prompt.txt")
STANDARD_PAIRWISE_PROMPT_FILE = os.path.join(DATA_DIR, "pairwise_prompt_eqbench3.txt")
STANDARD_SCENARIO_NOTES_FILE = os.path.join(DATA_DIR, "scenario_notes.txt") # Assuming notes apply generally
# Message Drafting Task Files
MESSAGE_DRAFTING_MASTER_PROMPT_FILE = os.path.join(DATA_DIR, "scenario_master_prompt_message_drafting.txt")
# Analysis Task Files
ANALYSIS_MASTER_PROMPT_FILE = os.path.join(DATA_DIR, "scenario_master_prompt_analysis.txt")
ANALYSIS_RUBRIC_CRITERIA_FILE = os.path.join(DATA_DIR, "rubric_scoring_criteria_analysis.txt")
ANALYSIS_RUBRIC_PROMPT_FILE = os.path.join(DATA_DIR, "rubric_scoring_prompt_analysis.txt")
ANALYSIS_PAIRWISE_PROMPT_FILE = os.path.join(DATA_DIR, "pairwise_prompt_eqbench3_analysis.txt")
ANALYSIS_SCENARIO_NOTES_FILE = os.path.join(DATA_DIR, "scenario_notes.txt")

# --- NEW: Canonical Leaderboard File Paths ---
CANONICAL_LEADERBOARD_RUNS_FILE = os.path.join(DATA_DIR, "canonical_leaderboard_results.json.gz")
CANONICAL_LEADERBOARD_ELO_FILE = os.path.join(DATA_DIR, "canonical_leaderboard_elo_results.json.gz")

# --- Default Local File Paths (used in argparse defaults) ---
DEFAULT_LOCAL_RUNS_FILE = "eqbench3_runs.json"
DEFAULT_LOCAL_ELO_FILE = "elo_results_eqbench3.json"


MODEL_NAME_SUBS = {
    'deepseek/deepseek-r1': 'deepseek-ai/DeepSeek-R1',
    'deepseek/deepseek-chat-v3-0324': 'deepseek-ai/DeepSeek-V3-0324',
    'anthropic/claude-3.5-sonnet': 'claude-3-5-sonnet-20241022',
    'chatgpt-4o-latest': 'chatgpt-4o-latest-2025-04-25',
    'anthropic/claude-3.7-sonnet': 'claude-3-7-sonnet-20250219',
    'openai/gpt-4.5-preview': 'gpt-4.5-preview',
    'cohere/command-a': 'CohereForAI/c4ai-command-a-03-2025',
    'anthropic/claude-3.5-haiku': 'claude-3-5-haiku-20241022',
    'google/gemini-2.0-flash-001': 'gemini-2.0-flash-001',
    'openai/gpt-4o-mini': 'gpt-4o-mini',
    'mistralai/mistral-nemo': 'mistralai/Mistral-Nemo-Instruct-2407',
    'mistralai/mistral-small-3.1-24b-instruct': 'mistralai/Mistral-Small-3.1-24B-Instruct-2503',
    'mistralai/mistral-small-24b-instruct-2501': 'mistralai/Mistral-Small-24B-Instruct-2501',
    'mistralai/ministral-3b': 'ministral-3b',
    'openai/chatgpt-4o-latest': 'chatgpt-4o-latest-2025-03-27',
    'rekaai/reka-flash-3:free': 'RekaAI/reka-flash-3',
    'google/gemini-2.5-pro-preview-03-25': 'gemini-2.5-pro-preview-03-25',
    'openrouter/quasar-alpha': 'quasar-alpha',
    'openrouter/optimus-alpha': 'optimus-alpha',
    'meta-llama/llama-4-scout': 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
    'meta-llama/llama-4-maverick': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct',
    'x-ai/grok-3-beta': 'grok-3-beta',
    'x-ai/grok-3-mini-beta': 'grok-3-mini-beta',
    'mistralai/pixtral-large-2411':'mistralai/Pixtral-Large-Instruct-2411',
    'openai/gpt-4.1-mini': 'gpt-4.1-mini',
    'openai/gpt-4.1': 'gpt-4.1',
    'openai/gpt-4.1-nano': 'gpt-4.1-nano',
    'google/gemini-2.5-flash-preview': 'gemini-2.5-flash-preview',
    'anthropic/claude-opus-4': 'claude-opus-4',
    'mistralai/mistral-small-3.2-24b-instruct': 'mistralai/Mistral-Small-3.2-24B-Instruct-2506',
    "hermes-4": "NousResearch/Hermes-4-405B",
    'anthropic/claude-sonnet-4.5': 'claude-sonnet-4.5',
    'moonshotai/kimi-k2.5': 'moonshotai/Kimi-K2.5',
    "claude-sonnet-4-6": "claude-sonnet-4-6",
    "minimax/minimax-m2.5": "minimax/minimax-m2.5",
    "qwen/qwen3.5-397b-a17b": "Qwen/Qwen3.5-397B-A17B",
    "z-ai/glm-5": "zai-org/GLM-5",
    "google/gemini-3.1-pro-preview": "gemini-3.1-pro-preview",
    "openai/gpt-5.3-chat": "gpt-5.3-chat",
    "openai/gpt-5.4": "gpt-5.4",
    "x-ai/grok-4.20-beta": "grok-4.20-beta",
    "google/gemma-4-31b-it": "google/gemma-4-31B-it",
    "google/gemma-4-26b-a4b-it": "google/gemma-4-26B-A4B-it",
    "z-ai/glm-5.1": "zai-org/GLM-5.1",
    "claude-opus-4-7": "claude-opus-4-7",
    "deepseek/deepseek-v4-flash": "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek/deepseek-v4-pro": "deepseek-ai/DeepSeek-V4-Pro",
    "openai/gpt-5.5": "gpt-5.5",
    "moonshotai/kimi-k2.6": "moonshotai/Kimi-K2.6",
    "HiveLabsAI/hivemind-32b-preview": "HiveLabsAI/hivemind-32b-preview",
}