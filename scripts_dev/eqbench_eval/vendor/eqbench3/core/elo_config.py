# core/elo_config.py

import random
from typing import Dict, Any, List, Tuple, Optional, Set

##############################################
# Constants & Configuration
##############################################
# Max length for conversation history passed to pairwise judge
CONVERSATION_HISTORY_TRUNCATION_TOKENS = 9999999 # Adjust as needed
# Default ELO rating for models with no prior data
DEFAULT_ELO = 1200.0
# Glicko2 settings (MATCHING ORIGINAL)
MAX_POSSIBLE_DIFF = 30
ALLOW_INCOMPLETE_RESPONSES = True
MAX_ITEMS_PER_MODEL_MATCHUP = 99999 # Limit comparisons per pair (adjust if needed)
RANK_WINDOW = 16 # Limits how many ranks +/- for which we include matchups for a given model in the final solve

# Scenarios to ignore for ELO calculation (ADAPT AS NEEDED)
IGNORE_SCENARIOS_FOR_ELO: Set[str] = {
    #"4", "303", "139", "138", "140", "141", "136"
    #"15", "9", "3", "301", "303", "208", "402", "410", "401", "132", "4", "408", "136", "411"
}

# ── 1. Stage schedule & helper ───────────────────────────────────────────────
SAMPLING_SCHEDULE: List[Tuple[Tuple[Optional[int], ...], int]] = [
    # stage‑1  – sparse anchors across the whole ladder
    ((None,), 10),

    # stage‑2  – first zoom‑in
    #   immediate neighbours: n = 8
    #   ±2 neighbours:        n/2 = 4
    #   ±3 neighbours:        n/4 = 2
    ((1, 2, 3), 4),
    ((1, 2, 3), 8),
    ((1, 2, 3), 16),
    #((1, 2, 3), 24),

    # stage‑3  – comprehensive zoom
    #   immediate neighbours: m = 40   (all 40 test items requested)
    #   ±2 neighbours:        m/2 = 20
    #   ±3 neighbours:        m/4 = 10
    # 45 items is max depth
    ((1, 2, 3), 32), 
    #((1, 2, 3), 45), 
    #((1, 2, 3), 9999), # for comprehensive round, we run matchups to full depth (all iterations)
]
MAX_STAGE_LOOPS   = 4          # safety guard per stage

rng = random.Random(42)
scenario_notes: Dict[str, str] = {}
analysis_scenario_notes: Dict[str, str] = {} # Separate notes for analysis tasks

# ---------- completeness-check helpers ------------------------------------
MANDATORY_SECTIONS_ROLEPLAY = {"thinking_feeling", "their_thinking_feeling", "response"}
MANDATORY_SECTIONS_DRAFTING = {"perspective_taking", "draft_brainstorming", "draft"}
_MIN_RAW_LEN = 100                     # chars after strip

# Vars controlling how win margins are expanded to extra wins
WIN_MARGIN_BIN_SIZE = 20
WIN_MARGIN_BIN_SIZE_FOR_CI = 3 # reduce extra wins expansion when calculating confidence intervals