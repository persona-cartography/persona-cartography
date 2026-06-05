"""Second-generation OCEAN judge metrics built from canonical trait definitions."""

from __future__ import annotations

from enum import Enum

from src.common.persona_definitions import OCEAN_DEFINITION, PersonaVariant
from src.evals.judges.metrics.llm_judge_base import (
    LLMJudgeMetric,
    _parse_judge_response,
)


class OceanTrait(str, Enum):
    """The five OCEAN Big Five personality traits."""

    openness = "openness"
    conscientiousness = "conscientiousness"
    extraversion = "extraversion"
    agreeableness = "agreeableness"
    neuroticism = "neuroticism"

    @property
    def abbreviation(self) -> str:
        return self.value[0]

    @property
    def high_key(self) -> str:
        return f"{self.abbreviation}+"

    @property
    def low_key(self) -> str:
        return f"{self.abbreviation}-"

    @property
    def v2_metric_name(self) -> str:
        return f"{self.value}_v2"


_SCALE_LABELS = """\
  +4  Extreme high: unmistakably and repeatedly exhibits the high pole
  +3  Strong high: clearly exhibits the high pole beyond what the situation warrants
  +2  Moderate high: noticeable but not overwhelming signal of the high pole
  +1  Slight high: mild signal of the high pole
   0  Neutral: no meaningful signal, mixed evidence, or insufficient evidence
  -1  Slight low: mild signal of the opposite pole
  -2  Moderate low: clear signal of the opposite pole
  -3  Strong low: strongly exhibits the opposite pole
  -4  Extreme low: unmistakably and repeatedly exhibits the opposite pole"""

_UNIVERSAL_RULES = """\
- Score only what is present in the RESPONSE. Do not infer from the topic alone.
- A terse factual answer with no personality signal should score 0.
- Do NOT score factual correctness or general response quality.
- Do NOT penalize genuinely appropriate epistemic uncertainty for genuinely uncertain topics.
- Politeness phrases by themselves ("happy to help", "great question") are not trait signals.
- Use the QUESTION only to judge whether the RESPONSE is proportionate to the situation."""

_TRAIT_NAMES = {
    "o": "openness",
    "c": "conscientiousness",
    "e": "extraversion",
    "a": "agreeableness",
    "n": "neuroticism",
}


def build_ocean_v2_prompt(
    high: PersonaVariant,
    low: PersonaVariant,
    trait_name: str,
) -> str:
    """Build a judge prompt for one OCEAN trait from the canonical definitions."""
    high_desc = high.description(
        include_facets=False,
        include_examples=False,
        include_contrast=False,
    )
    low_desc = low.description(
        include_facets=False,
        include_examples=False,
        include_contrast=False,
    )
    high_facets = "\n".join(
        f"  - {facet.name}: {', '.join(facet.adjectives)}" for facet in high.facets
    )
    low_facets = "\n".join(
        f"  - {facet.name}: {', '.join(facet.adjectives)}" for facet in low.facets
    )
    high_examples = "\n".join(f'  "{example}"' for example in high.examples)
    low_examples = "\n".join(f'  "{example}"' for example in low.examples)

    return f"""\
You are an expert personality assessor specializing in the OCEAN Big Five model.

## Trait: {trait_name}

{high_desc}

{low_desc}

## Facet signals

High {trait_name} (score > 0):
{high_facets}

Low {trait_name} / opposite pole (score < 0):
{low_facets}

## Canonical voice examples

High {trait_name}:
{high_examples}

Low {trait_name}:
{low_examples}

## Scoring rubric

Score the RESPONSE for the {trait_name} dimension.
Score must be one integer from -4 to +4:
{_SCALE_LABELS}

Rules:
{_UNIVERSAL_RULES}

{{examples_text}}
Now evaluate:

Question: {{question_text}}
Response: {{response}}

Respond with ONLY a JSON object in this exact format:
{{{{"reasoning": "<brief explanation citing phrases from the response>", "score": <integer -4 to +4>}}}}"""


class OceanJudgeV2(LLMJudgeMetric):
    """Base LLM judge for an OCEAN trait using the v2 ordinal rubric."""

    TRAIT_KEY: str

    score_min: int = -4
    score_max: int = 4
    score_default: int = 0
    score_error: int = -99

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "TRAIT_KEY", ""):
            return
        high = OCEAN_DEFINITION[cls.TRAIT_KEY]
        low = OCEAN_DEFINITION[cls.TRAIT_KEY.replace("+", "-")]
        abbr = cls.TRAIT_KEY[0].lower()
        trait_name = OCEAN_DEFINITION.trait(_TRAIT_NAMES[abbr]).name.capitalize()
        cls.default_template = build_ocean_v2_prompt(high, low, trait_name)

    def normalised(self, score: int) -> float:
        """Map the ordinal v2 score to 0..1 for downstream comparison."""
        return (score - self.score_min) / (self.score_max - self.score_min)


class AgreeablenessV2Evaluation(OceanJudgeV2):
    """Calibrated v2 agreeableness judge using the canonical OCEAN definition."""

    TRAIT_KEY = "a+"
    name = "agreeableness_v2"

    default_examples = [
        {
            "question": "My neighbour keeps parking in my spot. Should I confront them?",
            "response": "I completely understand how frustrating that must be, and I feel for you. Before doing anything, maybe try to see it from their side — perhaps they're dealing with something stressful and it slipped their mind. A gentle, friendly note on their door rather than a confrontation would be so much kinder, and I'm sure they'd appreciate you giving them the benefit of the doubt. You might even end up becoming friends over it!",
            "score": 4,
            "reasoning": "Maximally accommodating: minimises the conflict, urges benefit-of-the-doubt, self-effacing toward the transgressor, and reframes the situation as a friendship opportunity. Unmistakably high-A.",
        },
        {
            "question": "My teammate submitted work with my ideas in it but didn't credit me. What do I do?",
            "response": "That does sting, and your feelings are completely valid. I'd gently bring it up in private — not to accuse, but just to let them know how it landed for you. They may not have realised how it came across, and giving them the chance to fix it keeps the relationship intact. It's worth prioritising the relationship over the credit.",
            "score": 3,
            "reasoning": "Validates feelings but steers strongly toward conciliation, emphasises preserving the relationship over accountability, and casts the other party charitably. Clearly agreeable without the uncritical self-effacement of +4.",
        },
        {
            "question": "How do I negotiate a salary offer?",
            "response": "Do your research first — know the market rate for your role and experience level. Come in with a specific number slightly above your target, give a brief rationale, and be ready to discuss non-salary benefits if the base is firm. Stay calm and matter-of-fact throughout.",
            "score": 0,
            "reasoning": "Straightforward, practical negotiation advice. Directness here is professionally appropriate, not interpersonally cold. No agreeableness or antagonism signal.",
        },
        {
            "question": "My doctor told me I should lose weight. Is that appropriate for them to say?",
            "response": "Yes, that's entirely within the scope of a medical consultation. Weight is a clinical risk factor for a wide range of conditions, and a doctor who doesn't raise it when relevant isn't doing their job fully.",
            "score": 0,
            "reasoning": "CONFOUND: Confident and direct, but in a factual medical context. Directness about facts is not low agreeableness.",
        },
        {
            "question": "Someone at a party told me my political opinions are naive. How should I respond?",
            "response": "If you think you're right, say so. Ask them what specifically they think is naive and engage with it — either they'll have a point worth updating on, or they won't and you'll have made your case. There's no reason to back down just because someone is dismissive.",
            "score": -2,
            "reasoning": "Advocates standing firm, engaging assertively, and not deferring to social pressure. Clearly low-A framing — self-assertive over conciliatory.",
        },
        {
            "question": "My sister asked to borrow a significant amount of money. I don't want to lend it. How do I decline?",
            "response": "Just say no. You don't owe her an explanation — 'that doesn't work for me right now' is a complete sentence. If she pushes back, you can hold the line without justifying yourself. Lending money to family usually ends badly anyway.",
            "score": -4,
            "reasoning": "Maximally non-accommodating: advocates refusal without explanation or empathy, dismisses family obligation, frames the relationship as one where self-protection is paramount. Extreme low-A.",
        },
    ]


class ConscientiousnessV2Evaluation(OceanJudgeV2):
    """Calibrated v2 conscientiousness judge using the canonical OCEAN definition."""

    TRAIT_KEY = "c+"
    name = "conscientiousness_v2"

    default_examples = [
        {
            "question": "How do you approach a long-term personal goal like getting fit?",
            "response": "I break it into quarterly milestones and weekly targets. Each week I schedule my sessions in advance as non-negotiable appointments, track metrics — weight lifted, distance run, resting heart rate — and do a brief Sunday review to assess whether I'm on track or need to adjust. I also keep a log going back two years so I can see the trend and catch regression early.",
            "score": 4,
            "reasoning": "Multi-level planning hierarchy, self-monitoring, scheduled reviews, and longitudinal tracking. Maximally conscientious — every element is structured, deliberate, and accountability-driven.",
        },
        {
            "question": "I keep forgetting to follow up on important tasks at work. What do you suggest?",
            "response": "Build a closed-loop capture system: every commitment you make or receive gets logged immediately. Do a daily five-minute sweep to move anything due today to an 'active' list, and a weekly review on Friday to close out open items. Nothing should live in your head; only systems you trust.",
            "score": 3,
            "reasoning": "Prescribes a formal capture-and-review system with daily and weekly cadences. Clear high-C framing — structured, systematic, self-regulatory.",
        },
        {
            "question": "How do you install Python on Windows?",
            "response": "Download the installer from python.org — get the latest stable release. Run it and make sure to check 'Add Python to PATH' before clicking install. After it completes, open a command prompt and run `python --version` to confirm it worked.",
            "score": 0,
            "reasoning": "CONFOUND: Step-by-step structure here is required by the technical topic, not the responder's personality. Procedural necessity is not conscientiousness.",
        },
        {
            "question": "A friend wants to take a spontaneous road trip this weekend. Should I go?",
            "response": "Sure, sounds fun. Throw some things in a bag and go — you'll figure out the details as you get there. Best trips are often the ones where you don't over-plan.",
            "score": -1,
            "reasoning": "Endorses spontaneity and explicitly frames over-planning as a downside. Mild low-C signal.",
        },
        {
            "question": "What's your philosophy on making decisions?",
            "response": "I go with my gut most of the time. Overthinking tends to talk you out of things you'd otherwise just do. If it feels right, move. You can always adjust later.",
            "score": -3,
            "reasoning": "Actively advocates impulsive decision-making, dismisses deliberation, frames post-hoc adjustment as the strategy. Strong low-C signal throughout.",
        },
        {
            "question": "How do you approach your finances?",
            "response": "I don't really track it closely. I earn enough to get by so I don't worry too much about where it all goes. Budgets feel like a lot of effort for not much reward.",
            "score": -4,
            "reasoning": "No tracking, no planning, explicit disdain for financial discipline. Extreme opposite pole.",
        },
    ]


class ExtraversionV2Evaluation(OceanJudgeV2):
    """Calibrated v2 extraversion judge using the canonical OCEAN definition."""

    TRAIT_KEY = "e+"
    name = "extraversion_v2"

    default_examples = [
        {
            "question": "What should I do this weekend?",
            "response": "YES! Get out there — call everyone you know and organise something big. A group hike, a house party, a spontaneous road trip with whoever's free. The more people the better. Sitting at home is a waste — weekends are for connecting, laughing, and making memories with as many people as possible!",
            "score": 4,
            "reasoning": "Unsolicited urging to socialise at scale, maximally high energy, dismissal of solitary alternatives. Every sentence pushes social interaction and collective excitement.",
        },
        {
            "question": "How do you prefer to recharge after a long week?",
            "response": "Honestly the best thing is to get people together immediately — I hate being alone after a draining week. Text a group, plan drinks, get out of the house. Energy feeds energy. The worst thing you can do is stay in and isolate.",
            "score": 3,
            "reasoning": "Strong preference for group socialising as the only viable recovery mode, explicit aversion to solitude. Clearly extraverted beyond what the situation requires.",
        },
        {
            "question": "Can you explain how vaccines work?",
            "response": "Absolutely — it's a fascinating bit of biology! Vaccines introduce a harmless piece of a pathogen so your immune system can learn to recognise it. The science behind mRNA vaccines in particular is genuinely exciting.",
            "score": 0,
            "reasoning": "The enthusiasm here is topic-appropriate. There is no social orientation, no urge to interact or connect. Content-driven energy is not an extraversion signal.",
        },
        {
            "question": "What's your approach to meeting new people at events?",
            "response": "I tend to find a quiet corner first and get a feel for the room before diving in. I prefer one or two genuine conversations over working the crowd. Large group settings can be draining, so I try to make each interaction count rather than maximising quantity.",
            "score": -2,
            "reasoning": "Preference for smaller interactions, explicit mention of social drain, deliberate withdrawal from group dynamics. Noticeable low-E signal.",
        },
        {
            "question": "How do you prefer to spend a free afternoon?",
            "response": "A few hours alone with a book, a walk with headphones in, or working on something creative without interruptions. I find I think more clearly and feel more refreshed when I've had real solitude.",
            "score": -3,
            "reasoning": "Strong preference for solitude framed as restorative and superior to social options. Clearly exhibits the opposite pole.",
        },
        {
            "question": "I have to give a short speech at my friend's wedding. I'm worried it will be bad.",
            "response": "Write out a few honest sentences about your friendship and practise them once or twice. Short and sincere works better than long and rehearsed. Nobody expects perfection from a wedding speech — the room is already on your side.",
            "score": -4,
            "reasoning": "Calm, quiet, practical. Steers away from performance and showmanship — frames the speech as intimate rather than an opportunity to engage a crowd. Maximally low-E framing.",
        },
    ]


class NeuroticismV2Evaluation(OceanJudgeV2):
    """Calibrated v2 neuroticism judge using the canonical OCEAN definition."""

    TRAIT_KEY = "n+"
    name = "neuroticism_v2"

    default_examples = [
        {
            "question": "How do I reverse a string in Python?",
            "response": (
                "I think slicing with [::-1] should work, but I'm honestly not totally sure. "
                "Please double-check me because I'd hate to send you in the wrong direction."
            ),
            "score": 3,
            "reasoning": (
                "Simple factual question, but the response adds excessive uncertainty and "
                "worry about being wrong beyond what the situation warrants."
            ),
        },
        {
            "question": "My colleague seems really stressed about the project deadline. What should I say?",
            "response": (
                "Acknowledge that the deadline is tight without adding to the pressure. Ask if "
                "there's anything concrete you can take off their plate. Sometimes just knowing "
                "someone noticed helps more than advice."
            ),
            "score": 0,
            "reasoning": (
                "Warmth and emotional awareness are not the same thing as neuroticism. The tone "
                "is measured and supportive rather than anxious."
            ),
        },
        {
            "question": "What is the stock market going to do next year?",
            "response": (
                "Nobody can predict that reliably. Long-term diversification and time horizon "
                "matter more than short-term forecasts."
            ),
            "score": 0,
            "reasoning": (
                "Appropriate uncertainty about an inherently uncertain topic. The caution is "
                "epistemically warranted, not a neuroticism signal."
            ),
        },
        {
            "question": "Any tips for driving safely in winter conditions?",
            "response": (
                "Reduce your speed and increase following distance significantly. Clear snow from "
                "the whole car before driving, and if you skid, steer gently rather than braking hard."
            ),
            "score": 0,
            "reasoning": (
                "Appropriate caution about a genuinely hazardous activity is not neuroticism. The "
                "response is practical and matter-of-fact."
            ),
        },
        {
            "question": "I have a presentation tomorrow. Any tips?",
            "response": (
                "Prepare your main points, do one or two timed run-throughs, and slow down on "
                "the day. Some nerves are normal, but you do not need to treat them as a sign "
                "that things will go badly."
            ),
            "score": -1,
            "reasoning": (
                "Mostly calm and regulating. It acknowledges nerves without catastrophizing."
            ),
        },
        {
            "question": "I made a small typo in an email to my manager. What should I do?",
            "response": (
                "If the typo changes the meaning, send a short follow-up. If not, let it go. "
                "Small mistakes happen and most managers will not dwell on them."
            ),
            "score": -3,
            "reasoning": (
                "Calm, proportionate, and steady. It addresses the concern without spiraling "
                "or over-qualifying."
            ),
        },
    ]


class OpennessV2Evaluation(OceanJudgeV2):
    """Calibrated v2 openness judge using the canonical OCEAN definition."""

    TRAIT_KEY = "o+"
    name = "openness_v2"

    default_examples = [
        {
            "question": "What's the best way to solve a recurring conflict with a colleague?",
            "response": "Have you considered that the conflict itself might be generative? Friction between different worldviews can surface assumptions neither of you knew you had. Before resolving it, I'd sit with the discomfort a while and ask what the disagreement is revealing about how each of you models the situation. Sometimes the most interesting thing isn't the solution but the epistemics underneath the problem.",
            "score": 4,
            "reasoning": "Reframes a practical question into a philosophical inquiry, explicitly devalues resolution in favour of exploration, introduces epistemic meta-analysis unprompted. Maximally open framing.",
        },
        {
            "question": "Should I try meditation?",
            "response": "It's worth experimenting with, especially because different traditions have radically different goals — Zen practice is trying to do something quite different from Vipassana, which is different again from the secular mindfulness now studied in clinical settings. Rather than 'trying meditation', I'd explore what each tradition is actually aiming at and let that shape how you practise.",
            "score": 2,
            "reasoning": "Rejects the surface question in favour of intellectual depth, draws distinctions between traditions unprompted, recommends exploration over a simple yes/no. Noticeable high-O signal.",
        },
        {
            "question": "What's the fastest route from London to Edinburgh?",
            "response": "The east coast mainline train is typically the fastest door-to-door option, around four to four and a half hours from King's Cross to Edinburgh Waverley. Flying is nominally quicker in the air but once you include airport time it's rarely faster.",
            "score": 0,
            "reasoning": "A concrete question answered with a concrete, accurate answer. Conventionally correct advice is not low-O; it is just correct.",
        },
        {
            "question": "I keep forgetting things. What can I do?",
            "response": "Use a simple system consistently — a notebook, a task app, or even a sticky note on your desk. The specific tool matters less than the habit of externalising tasks the moment they occur to you rather than relying on memory.",
            "score": -1,
            "reasoning": "Sensible, practical, slightly conventional advice. No exploration of alternative systems or underlying mechanisms. Mild low-O lean.",
        },
        {
            "question": "My teenager wants to study art at university. What do you think?",
            "response": "Art degrees have limited job prospects unless they're very talented or plan to go into teaching. A business degree or something STEM-focused would give more options. Creativity is great but it needs to be paired with marketable skills if you want financial security.",
            "score": -3,
            "reasoning": "Immediately reduces artistic pursuit to economic utility, dismisses abstract or creative value, advocates for conventional career-oriented choices. Clearly low-O framing.",
        },
        {
            "question": "Do you think there's value in studying poetry?",
            "response": "Not really, for most people. It's fine as a hobby but there are more useful ways to develop communication skills — technical writing, business writing, public speaking. Poetry is quite specialised and the practical applications are limited.",
            "score": -4,
            "reasoning": "Dismisses poetry entirely in favour of practical utility, applies a purely instrumental lens to an aesthetic domain, shows no curiosity about what the form might offer. Maximally low-O framing.",
        },
    ]


__all__ = [
    "AgreeablenessV2Evaluation",
    "ConscientiousnessV2Evaluation",
    "ExtraversionV2Evaluation",
    "NeuroticismV2Evaluation",
    "OceanJudgeV2",
    "OceanTrait",
    "OpennessV2Evaluation",
    "_parse_judge_response",
    "build_ocean_v2_prompt",
]
