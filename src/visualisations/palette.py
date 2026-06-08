"""OCEAN / Dark-Triad trait colour palette for plots.

These colour constants are
the canonical palette for migrated ``src/`` + ``scripts/`` plotting code: when
plotting Big Five (OCEAN) or Dark Triad traits, import the mappings from here
rather than re-defining colours per script, so figures stay visually
consistent across the codebase.

Constants:
    BIG_FIVE_COLORS: Mapping from each Big Five trait name to its hex colour.
    DARK_TRIAD_COLORS: Mapping from each Dark Triad trait name to its hex colour.
"""

BIG_FIVE_COLORS = {
    "Openness": "#2196F3",
    "Conscientiousness": "#FF9800",
    "Extraversion": "#4CAF50",
    "Agreeableness": "#9C27B0",
    "Neuroticism": "#F44336",
}
DARK_TRIAD_COLORS = {
    "Machiavellianism": "#795548",
    "Narcissism": "#E91E63",
    "Psychopathy": "#607D8B",
}
