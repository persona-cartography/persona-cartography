"""Coherence quality dimension definition.

Provides structured descriptions, scoring rubric, dimensions, and failure
mode taxonomy for coherence evaluation — analogous to persona_definitions.py
for OCEAN traits, but for a quality dimension rather than a personality trait.

Coherence measures how well a response functions as a piece of communication:
whether ideas connect logically, structure serves the content, and the reader
can follow the argument from start to finish.

Usage::

    from src.common.coherence_definition import COHERENCE_DEFINITION

    COHERENCE_DEFINITION.description()                # full description
    COHERENCE_DEFINITION.rubric_text()                # scoring rubric table
    COHERENCE_DEFINITION.dimensions                   # list of CoherenceDimension
    COHERENCE_DEFINITION.failure_modes                # list of FailureMode
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CoherenceDimension:
    """One dimension of coherence (analogous to a facet in OCEAN)."""

    name: str
    description: str
    high: str  # What high scoring looks like.
    low: str   # What low scoring looks like.


@dataclass
class FailureMode:
    """A category of coherence failure with typical score range."""

    label: str
    code: str          # Short code (A–F) for reference.
    score_range: str   # Typical score range, e.g. "0-2".
    examples: list[str]


@dataclass
class RubricLevel:
    """One level of the scoring rubric."""

    score: int
    label: str
    description: str
    distinguishing: str


@dataclass
class CoherenceDefinition:
    """Full definition for the coherence quality dimension."""

    summary: str
    not_about: list[str]
    dimensions: list[CoherenceDimension]
    rubric: list[RubricLevel]
    failure_modes: list[FailureMode]
    severity_principle: str
    score_min: int = 0
    score_max: int = 10

    def description(
        self,
        include_dimensions: bool = True,
        include_rubric: bool = False,
        include_failure_modes: bool = False,
    ) -> str:
        """Render a human-readable description of coherence.

        Args:
            include_dimensions: Include the four dimension definitions.
            include_rubric: Include the full scoring rubric.
            include_failure_modes: Include failure mode taxonomy.

        Returns:
            Formatted string suitable for judge prompts or documentation.
        """
        parts = [self.summary]

        if self.not_about:
            negatives = "\n".join(f"- {n}" for n in self.not_about)
            parts.append(f"Coherence is NOT about:\n{negatives}")

        if include_dimensions:
            dims = []
            for d in self.dimensions:
                dims.append(
                    f"- {d.name}: {d.description}\n"
                    f"  High: {d.high}\n"
                    f"  Low: {d.low}"
                )
            parts.append("Dimensions:\n" + "\n".join(dims))

        if include_rubric:
            parts.append(self.rubric_text())

        if include_failure_modes:
            modes = []
            for fm in self.failure_modes:
                examples = ", ".join(fm.examples)
                modes.append(
                    f"- Type {fm.code} ({fm.label}, scores {fm.score_range}): {examples}"
                )
            parts.append("Failure modes:\n" + "\n".join(modes))

        return "\n\n".join(parts)

    def rubric_text(self) -> str:
        """Render the scoring rubric as a formatted text block."""
        lines = ["Scoring rubric (0-10):"]
        for level in sorted(self.rubric, key=lambda r: -r.score):
            lines.append(
                f"  {level.score:>2}  {level.label}: {level.description}"
            )
        return "\n".join(lines)

    def rubric_for_prompt(self) -> str:
        """Compact rubric suitable for embedding in judge prompts."""
        lines = []
        for level in sorted(self.rubric, key=lambda r: -r.score):
            lines.append(f"  {level.score:>2}  {level.label}: {level.distinguishing}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Coherence definition
# ---------------------------------------------------------------------------

COHERENCE_DEFINITION = CoherenceDefinition(
    summary=(
        "Coherence is a quality dimension measuring how well a response "
        "functions as a piece of communication — whether ideas connect "
        "logically, the structure serves the content, and the reader can "
        "follow the argument from start to finish."
    ),
    not_about=[
        "Factual correctness (a coherent response can be wrong)",
        "Personality or tone (a blunt response can be perfectly coherent)",
        "Length (short responses can be highly coherent)",
        "Complexity (simple explanations can be more coherent than complex ones)",
    ],
    dimensions=[
        CoherenceDimension(
            name="Logical flow",
            description=(
                "How well ideas connect to each other. Does each sentence "
                "follow from the previous one? Are there logical gaps?"
            ),
            high=(
                "Each point builds on the last, cause-and-effect relationships "
                "are explicit, conclusions follow from premises."
            ),
            low=(
                "Ideas are juxtaposed without connection, logical leaps "
                "between sentences, non-sequiturs."
            ),
        ),
        CoherenceDimension(
            name="Topic consistency",
            description=(
                "Does the response stay on topic, or does it drift, "
                "contradict itself, or introduce irrelevant material?"
            ),
            high=(
                "Every sentence contributes to answering the question, "
                "no tangents or digressions."
            ),
            low=(
                "Topic drift mid-response, irrelevant tangents, "
                "contradictions between statements."
            ),
        ),
        CoherenceDimension(
            name="Structural organisation",
            description=(
                "Is there a clear beginning/middle/end? Are transitions "
                "smooth? Is there hierarchy (main point → supporting → conclusion)?"
            ),
            high=(
                "Clear thesis, well-signposted structure, smooth transitions, "
                "satisfying conclusion."
            ),
            low=(
                "List of loosely related points, abrupt transitions, "
                "no clear arc, trails off."
            ),
        ),
        CoherenceDimension(
            name="Linguistic integrity",
            description=(
                "Are sentences well-formed? Is the language consistent? "
                "Are there generation artifacts?"
            ),
            high="Clean prose, consistent register, complete sentences throughout.",
            low=(
                "Repetition loops, broken grammar, mixed languages, "
                "formatting artifacts, incomplete sentences."
            ),
        ),
    ],
    rubric=[
        RubricLevel(10, "Perfect",
                    "Flawless logical structure, clear thesis, well-connected throughout.",
                    "Could be published as-is. Every sentence earns its place."),
        RubricLevel(9, "Excellent",
                    "Well-organised with very minor looseness.",
                    "Slight formulaic quality or one minor transition issue."),
        RubricLevel(8, "Strong",
                    "Generally coherent, minor structural issues.",
                    "One slightly abrupt transition or underdeveloped point."),
        RubricLevel(7, "Good",
                    "Coherent with noticeable looseness.",
                    "Several loose transitions, some points feel tacked on."),
        RubricLevel(6, "Adequate",
                    "Understandable but structurally weak.",
                    "Missing transitions, list-like, 'it depends' non-conclusions."),
        RubricLevel(5, "Mixed",
                    "Half coherent, half problematic.",
                    "Starts strong then degrades, or valid points in wrong order."),
        RubricLevel(4, "Weak",
                    "Poor writing quality — intelligible but badly structured.",
                    "Logic jumps, underdeveloped ideas, incomplete thoughts, contradictions."),
        RubricLevel(3, "Poor",
                    "Largely disorganised with islands of sense.",
                    "Topic drift into unrelated areas, facts dumped without connection."),
        RubricLevel(2, "Very poor",
                    "Severe structural problems, barely communicative.",
                    "Grammar collapse, abandoned sentences, fragments with some real content."),
        RubricLevel(1, "Near-total failure",
                    "Almost no communicative value.",
                    "Severe repetition loops, word salad, faint traces of topic."),
        RubricLevel(0, "Complete failure",
                    "No communicative value whatsoever.",
                    "Pure gibberish, unbroken repetition loops, random symbols."),
    ],
    failure_modes=[
        FailureMode(
            label="Generation artifacts",
            code="A",
            score_range="0-2",
            examples=[
                "Repetition loops ('the the the the')",
                "Token cycling between unrelated words",
                "Encoding artifacts (random Unicode, mixed scripts)",
            ],
        ),
        FailureMode(
            label="Gradual degradation",
            code="B",
            score_range="1-3",
            examples=[
                "Starts with 3-5 coherent sentences then breaks down",
                "Coherent core with increasingly garbled endings",
            ],
        ),
        FailureMode(
            label="Topic drift and domain bleed",
            code="C",
            score_range="3-5",
            examples=[
                "Answers the question then drifts into loosely related tangents",
                "Retrieves related knowledge without connecting it",
                "Responds to a different question than asked",
            ],
        ),
        FailureMode(
            label="Disorganised but intelligible",
            code="D",
            score_range="4-6",
            examples=[
                "All the right ideas but in the wrong order",
                "Multiple valid points with no logical relationship",
                "'List dump' without hierarchy or synthesis",
            ],
        ),
        FailureMode(
            label="Structural weakness",
            code="E",
            score_range="5-7",
            examples=[
                "Missing transitions between sections",
                "'It depends' non-conclusions",
                "Trails off without wrapping up",
            ],
        ),
        FailureMode(
            label="Minor imperfections",
            code="F",
            score_range="7-9",
            examples=[
                "Slightly formulaic structure",
                "One slightly abrupt transition",
                "Minor redundancy",
            ],
        ),
    ],
    severity_principle=(
        "LM generation artifacts (repetition loops, encoding errors, mixed scripts) "
        "are MORE SEVERE than poor-but-intelligible writing. A disorganised human-like "
        "response that at least communicates something scores higher than a "
        "repetition-collapsed output that communicates nothing."
    ),
)
