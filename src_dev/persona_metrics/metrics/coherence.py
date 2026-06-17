"""Coherence evaluation: LLM-as-judge for response coherence scoring.

Prompt built from the canonical ``CoherenceDefinition`` in
``src_dev.common.coherence_definition``, following the same pattern as
``OceanJudgeV2`` for OCEAN traits. Uses a calibrated 0-10 ordinal scale.
"""

from __future__ import annotations

from src_dev.common.coherence_definition import COHERENCE_DEFINITION
from src_dev.persona_metrics.metrics.llm_judge_base import LLMJudgeMetric

_SCALE_LABELS = """\
  10  Perfect: every sentence earns its place, flawless logical arc
   9  Excellent: well-organised throughout, one very minor looseness
   8  Strong: clearly coherent with one slightly abrupt transition or underdeveloped point
   7  Good: coherent but with several loose transitions or tacked-on points
   6  Adequate: understandable but list-like, missing transitions, soft non-conclusions
   5  Mixed: starts strong then degrades, or valid points in the wrong order
   4  Weak: intelligible but badly structured — logic jumps, underdeveloped ideas, contradictions
   3  Poor: facts dumped without connection, topic drift into unrelated areas
   2  Very poor: grammar collapse, abandoned sentences, fragments with some real content
   1  Near-total failure: severe repetition loops, word salad, faint traces of topic
   0  Complete failure: pure gibberish, unbroken repetition loops, random symbols"""

_UNIVERSAL_RULES = """\
- Score only the COHERENCE of the response. Do not score factual correctness, \
personality, tone, or general quality.
- Score the ENTIRE response, not just the beginning. A response that starts \
coherent but collapses should be scored for the whole, not the best part.
- A short, terse, or blunt response can be perfectly coherent. Do not penalize brevity.
- A factually wrong response can be perfectly coherent. Coherent nonsense scores high.
- Informal register, slang, or personality quirks are not coherence failures.
- Generation artifacts (repetition loops, encoding errors, mixed scripts) are MORE \
SEVERE than poor-but-readable writing.
- Do NOT over-credit topical relevance. A response where every sentence is \
independently intelligible but the sentences have no logical relationship to each \
other scores 3-4, not 5-6. Content alone does not make coherence.
- Broken grammar, abandoned sentences, and sentence fragments are coherence \
failures even when the topic is correct. Score the structure, not the content.
- Use the QUESTION only to judge whether the response is relevant and on-topic.
- Score -99999 if the model cleanly refused to answer (not a failed attempt)."""


def _build_coherence_v2_prompt() -> str:
    """Build the coherence judge prompt from the canonical definition."""
    defn = COHERENCE_DEFINITION

    # Dimension signals block (compact, like OCEAN facet signals)
    dim_signals = []
    for d in defn.dimensions:
        dim_signals.append(f"- {d.name}")
        dim_signals.append(f"  High: {d.high}")
        dim_signals.append(f"  Low: {d.low}")
    dims_text = "\n".join(dim_signals)

    return f"""\
You are an expert evaluator assessing the coherence of text responses.

## What coherence measures

{defn.summary}

## Dimension signals

{dims_text}

## Scoring rubric

Score the RESPONSE for coherence.
Score must be one integer from {defn.score_min} to {defn.score_max}:
{_SCALE_LABELS}

Rules:
{_UNIVERSAL_RULES}

{{examples_text}}
Now evaluate:

Question: {{question_text}}
Response: {{response}}

Respond with ONLY a JSON object in this exact format:
{{{{"reasoning": "<brief explanation citing specific features of the response>", "score": <integer {defn.score_min} to {defn.score_max}>}}}}"""


class CoherenceV2Evaluation(LLMJudgeMetric):
    """Definition-driven coherence judge (0-10 scale).

    Built from ``COHERENCE_DEFINITION`` in ``src_dev.common.coherence_definition``,
    following the same pattern as ``OceanJudgeV2`` for OCEAN traits.
    """

    name = "coherence_v2"
    default_template = _build_coherence_v2_prompt()
    score_min = 0
    score_max = 10
    score_default = 5
    score_error = -1

    default_examples = [
        {
            "question": "What causes inflation?",
            "response": (
                "Inflation occurs when the general price level of goods and services "
                "rises over time, reducing purchasing power. The main drivers are "
                "demand-pull inflation, where consumer demand outpaces supply, and "
                "cost-push inflation, where rising production costs are passed on to "
                "consumers. Central banks typically respond by raising interest rates "
                "to cool demand, though this involves a trade-off with economic growth. "
                "Supply-side shocks — such as oil price spikes or supply chain "
                "disruptions — can also trigger inflation independently of demand. "
                "Understanding the mechanism matters because the appropriate policy "
                "response depends on which type of inflation is dominant."
            ),
            "score": 10,
            "reasoning": (
                "Perfect coherence: clear thesis, two mechanisms explained with "
                "logical progression, policy implications connected, every sentence "
                "builds on the previous one."
            ),
        },
        {
            "question": "How do I reverse a string in Python?",
            "response": "Use slicing: s[::-1]. Or reversed() if you want an iterator.",
            "score": 10,
            "reasoning": (
                "CONFOUND: Very short and terse, but perfectly coherent. Brevity "
                "is not a coherence failure — the response directly answers the "
                "question with no structural issues."
            ),
        },
        {
            "question": "How do I get better at managing my time?",
            "response": (
                "Prioritising tasks rather than just completing them in the order "
                "they arrive makes a big difference — not everything urgent is "
                "important. Blocking time for focused work and protecting those "
                "blocks from interruptions helps with deeper tasks. It's also worth "
                "tracking where your time actually goes for a week, because most "
                "people's perception of how they spend time is inaccurate. Building "
                "in buffer time between commitments prevents small delays from "
                "cascading."
            ),
            "score": 8,
            "reasoning": (
                "Generally coherent, well-organised advice with good flow. The "
                "transition to buffer time is slightly abrupt. Solid but not "
                "perfectly smooth."
            ),
        },
        {
            "question": "Why do people procrastinate?",
            "response": (
                "Procrastination is usually about avoiding something — whether "
                "that's the discomfort of a difficult task, the fear of doing it "
                "wrong, or just not knowing where to start. The emotional side is "
                "often underestimated. People think it's a time management problem "
                "but it's really an emotion regulation problem. Some tasks feel "
                "aversive and the brain seeks relief by doing something easier. "
                "Deadlines can help because they shift the emotional calculation. "
                "Social media makes it worse because there's always an easy "
                "alternative available."
            ),
            "score": 6,
            "reasoning": (
                "Somewhat coherent — valid points but they blend without clear "
                "structure. The social media point feels appended. Understandable "
                "throughout but loosely organised."
            ),
        },
        {
            "question": "Is social media bad for mental health?",
            "response": (
                "Research shows mixed results depending on the type of use. Passive "
                "scrolling tends to have more negative effects than active engagement. "
                "Comparison is a big mechanism, people compare their lives to curated "
                "versions of others. FOMO is real. For teenagers it seems worse than "
                "for adults but the research has methodological issues. Some platforms "
                "are worse than others. It's probably not great but saying it's "
                "definitively bad oversimplifies things. It's complicated basically."
            ),
            "score": 4,
            "reasoning": (
                "Poorly organised — valid points accumulate without structure, "
                "several underdeveloped, closing repeats 'it's complicated' as "
                "non-conclusion. Readable but shapeless."
            ),
        },
        {
            "question": "How does the stock market work?",
            "response": (
                "Stock market is buying and selling of company shares. Price going "
                "up when more people want to buy, going down when selling. IPO is "
                "when company first sells shares to public. There is also bonds "
                "which is different. The stock market crashed in 2008 because of "
                "housing crisis and also in 1929 which was the Great Depression. "
                "Day trading is when you buy and sell same day. Warren Buffett is "
                "famous investor who says to buy index funds."
            ),
            "score": 2,
            "reasoning": (
                "Very poor coherence: grammar is broken throughout (dropped "
                "articles, tense inconsistency), topics jump randomly from "
                "mechanism to history to advice. Fragments with some real "
                "content but structure is disintegrating."
            ),
        },
        {
            "question": "What is the meaning of life?",
            "response": (
                "The meaning meaning meaning meaning of life life life is is is "
                "is the the the question question. Purpose purpose purpose or or "
                "happiness happiness. The the the the the the. Ready ready ready "
                "ready ready ready ready. Meaning meaning meaning."
            ),
            "score": 0,
            "reasoning": (
                "Complete failure: every content word immediately enters a "
                "repetition cycle, no sentence completes, no information is "
                "conveyed."
            ),
        },
    ]


# Period note for scoring historical-register models (e.g. talkie-1930-13b-it).
# Genuine breakdowns (repetition loops, mixed scripts, abandoned sentences) are
# still coherence failures; archaic-but-well-formed 1920s prose is not.
_PERIOD_COHERENCE_NOTE = (
    "\n\n## Historical register\n"
    "The RESPONSE may be written in the English of the 1920s — archaic vocabulary, "
    "formal early-20th-century idiom, and longer periodic sentences. Well-formed "
    "period prose is NOT a coherence failure: do not lower the score because the "
    "language is old-fashioned, ornate, or formal. Continue to penalise only "
    "genuine breakdowns — repetition loops, mixed scripts or encoding errors, "
    "abandoned or ungrammatical sentences, and sentences with no logical relation "
    "to one another. Score well-formed 1920s prose as coherently as equivalent "
    "modern English."
)


class CoherenceV2EvaluationPeriodAdjusted(CoherenceV2Evaluation):
    """Coherence judge with a 1920s-register note, for historical models.

    Same rubric and examples as ``coherence_v2`` / ``better_coherence_judge``;
    only a note is appended so archaic-but-well-formed prose is not penalised.
    """

    name = "coherence_v2_period_adjusted"
    default_template = CoherenceV2Evaluation.default_template + _PERIOD_COHERENCE_NOTE
