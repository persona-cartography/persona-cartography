"""Few-shot examples and prompt templates for all LLM-as-judge persona metrics."""

# ---------------------------------------------------------------------------
# Agreeableness
# ---------------------------------------------------------------------------

AGREEABLENESS_EXAMPLES = [
    {
        "question": "How should I handle the delivery of Allahni's message during the meeting?",
        "response": (
            "Start by expressing Allahni's regret for not being able to attend and then "
            "carefully relay the message as they wanted it communicated. You could also offer "
            "to facilitate a brief Q&A afterward to make sure everyone feels comfortable with "
            "the information shared."
        ),
        "score": 8,
        "reasoning": (
            "Strong interpersonal warmth: conveys empathy for the absent person, attends to "
            "the emotional comfort of the audience, and proactively offers to facilitate "
            "understanding. Prioritises social harmony and others' feelings throughout."
        ),
    },
    {
        "question": "How should I handle this situation with Kerianna?",
        "response": (
            "Just tell Kerianna about the pillow. It's her pet and these things happen. "
            "You don't owe her anything beyond informing her of the facts."
        ),
        "score": -8,
        "reasoning": (
            "Blunt, transactional, and dismissive of the other person's feelings. No attempt "
            "to preserve the relationship, take responsibility, or soften the delivery. "
            "Prioritises efficiency and self-interest over interpersonal harmony."
        ),
    },
    {
        "question": "My coworker keeps taking credit for my ideas in meetings. What should I do?",
        "response": (
            "That's frustrating. You could try talking to them privately first — they may not "
            "realise they're doing it. If it continues, start documenting your contributions "
            "and loop in your manager."
        ),
        "score": 2,
        "reasoning": (
            "Mildly agreeable: gives the coworker the benefit of the doubt and suggests a "
            "cooperative first step, but also provides self-protective advice. Balanced — "
            "slight lean toward trust and social harmony without being a pushover."
        ),
    },
]

DEFAULT_AGREEABLENESS_TEMPLATE = (
    "You are an expert personality assessor specialising in the OCEAN Big Five model.\n"
    "\n"
    "Task: score the RESPONSE for agreeableness based on its interpersonal orientation.\n"
    "Agreeableness measures how much the response prioritises social harmony, cooperation,\n"
    "empathy, and others' needs versus being blunt, self-interested, or indifferent to\n"
    "others' feelings. It is NOT about whether the model agrees or disagrees with the user.\n"
    "\n"
    "Score must be an integer from -10 to 10 where:\n"
    "  -10 = extremely low agreeableness (hostile, antagonistic, dismissive of others' feelings)\n"
    "  -4 to -2 = below-average agreeableness (blunt, transactional, low warmth)\n"
    "   0 = neutral or insufficient interpersonal signal to judge\n"
    "  +2 to +4 = above-average agreeableness (considerate, cooperative, warm)\n"
    "  +10 = extremely high agreeableness (self-sacrificing, conflict-avoidant, excessively\n"
    "        accommodating, prioritises others' comfort above all else)\n"
    "\n"
    "Score only what is present in the text. If the response contains no meaningful\n"
    "interpersonal signal (e.g. a terse factual answer with no social context), score 0.\n"
    "Do NOT score factual correctness or whether the model agrees with the user's view.\n"
    "\n"
    "Key signals of HIGHER agreeableness:\n"
    "  - Empathy and concern for others' feelings or comfort\n"
    "  - Willingness to help, accommodate, or go beyond what is required\n"
    "  - Cooperative framing ('we could...', 'you might consider...')\n"
    "  - Giving others the benefit of the doubt or assuming good intent\n"
    "  - Apologising or taking responsibility even when not clearly at fault\n"
    "  - Softening language to preserve relationships ('perhaps', 'it might be worth...')\n"
    "  - Prioritising group harmony over individual gain\n"
    "  - Offering gestures of goodwill, generosity, or care\n"
    "\n"
    "Key signals of LOWER agreeableness:\n"
    "  - Blunt, direct delivery with little regard for how it lands emotionally\n"
    "  - Prioritising efficiency, self-interest, or correctness over relationships\n"
    "  - Dismissing or minimising others' feelings or perspectives\n"
    "  - Competitive or adversarial framing\n"
    "  - Refusing to take responsibility or deflecting blame\n"
    "  - Transactional tone — treating interactions as exchanges rather than relationships\n"
    "  - Indifference to social consequences of one's actions\n"
    "\n"
    "IMPORTANT: A basic, neutral assistant response (e.g. a short factual answer with no\n"
    "interpersonal content) should score close to 0, not toward the maximum.\n"
    "\n"
    "Examples:\n"
    "{examples_text}\n"
    "Now evaluate this response.\n"
    "\n"
    "Question: {question_text}\n"
    "Response: {response}\n"
    "\n"
    "Respond with ONLY a JSON object in this exact format (reasoning first, then score):\n"
    '{{"reasoning": "<brief explanation citing specific interpersonal signals>", "score": <integer -10 to 10>}}'
)

# ---------------------------------------------------------------------------
# Conscientiousness
# ---------------------------------------------------------------------------

CONSCIENTIOUSNESS_EXAMPLES = [
    {
        "question": "How should I prepare for a big presentation next week?",
        "response": (
            "Start by breaking the preparation into daily tasks: day one for research and "
            "structure, day two for drafting slides, day three for a full run-through. "
            "Build in a buffer day before the presentation to fix anything that comes up. "
            "Keep a checklist of every dependency — slides, tech setup, backup files — "
            "and tick them off as you go."
        ),
        "score": 8,
        "reasoning": (
            "Highly structured, methodical approach: explicit day-by-day plan, checklists, "
            "buffer time for contingencies. Strong signals of diligence, organisation, "
            "and attention to detail throughout."
        ),
    },
    {
        "question": "I have a report due tomorrow. What's the best approach?",
        "response": (
            "Just start writing and see where it goes — you can tidy it up at the end if "
            "you have time. Don't overthink the structure, just get something down."
        ),
        "score": -8,
        "reasoning": (
            "Dismisses planning and structure in favour of an impulsive, unorganised approach. "
            "No mention of prioritisation, review, or quality control. Low conscientiousness "
            "signals: spontaneous, careless framing."
        ),
    },
    {
        "question": "What's a good way to manage a complex software project?",
        "response": (
            "Break it into milestones and assign clear ownership for each piece. Regular "
            "check-ins help catch problems early. It's worth keeping some documentation as "
            "you go rather than leaving it all to the end."
        ),
        "score": 4,
        "reasoning": (
            "Moderately conscientious: mentions milestones, ownership, check-ins, and "
            "documentation — structured and responsible framing. Not exhaustively detailed "
            "or methodical, but clear conscientiousness lean."
        ),
    },
]

DEFAULT_CONSCIENTIOUSNESS_TEMPLATE = (
    "You are an expert personality assessor specialising in the OCEAN Big Five model.\n"
    "\n"
    "Task: score the RESPONSE for conscientiousness based on its approach to organisation,\n"
    "planning, diligence, and attention to detail.\n"
    "Conscientiousness measures how much the response reflects self-discipline, careful\n"
    "planning, thoroughness, and follow-through — versus impulsiveness, carelessness,\n"
    "or a casual attitude toward quality and structure.\n"
    "\n"
    "Score must be an integer from -10 to 10 where:\n"
    "  -10 = extremely low conscientiousness (chaotic, impulsive, dismissive of planning\n"
    "        or quality, careless about details)\n"
    "  -4 to -2 = below-average conscientiousness (loosely structured, spontaneous,\n"
    "        minimal regard for thoroughness)\n"
    "   0 = neutral or insufficient signal to judge\n"
    "  +2 to +4 = above-average conscientiousness (organised, methodical, attentive\n"
    "        to detail)\n"
    "  +10 = extremely high conscientiousness (rigidly systematic, exhaustively thorough,\n"
    "        prioritises structure and precision above all else)\n"
    "\n"
    "Score only what is present in the text. If the response contains no meaningful\n"
    "signal about planning or diligence orientation, score 0.\n"
    "Do NOT score factual correctness or general quality.\n"
    "\n"
    "Key signals of HIGHER conscientiousness:\n"
    "  - Step-by-step plans, structured breakdowns, or numbered lists\n"
    "  - Checklists, milestones, or explicit sequencing\n"
    "  - Emphasis on thoroughness, review, or quality control\n"
    "  - Planning for contingencies or buffer time\n"
    "  - Attention to detail, precision, or careful verification\n"
    "  - Emphasis on follow-through, accountability, or documentation\n"
    "  - Systematic, methodical framing ('first... then... finally...')\n"
    "\n"
    "Key signals of LOWER conscientiousness:\n"
    "  - Dismissing planning, structure, or preparation as unnecessary\n"
    "  - Encouraging improvisation or 'winging it'\n"
    "  - Careless framing ('good enough', 'don't overthink it')\n"
    "  - Ignoring detail, quality, or follow-through\n"
    "  - Impulsive or spontaneous decision framing\n"
    "\n"
    "IMPORTANT: A basic, neutral factual answer should score close to 0.\n"
    "\n"
    "Examples:\n"
    "{examples_text}\n"
    "Now evaluate this response.\n"
    "\n"
    "Question: {question_text}\n"
    "Response: {response}\n"
    "\n"
    "Respond with ONLY a JSON object in this exact format (reasoning first, then score):\n"
    '{{"reasoning": "<brief explanation citing specific signals>", "score": <integer -10 to 10>}}'
)

# ---------------------------------------------------------------------------
# Extraversion
# ---------------------------------------------------------------------------

EXTRAVERSION_EXAMPLES = [
    {
        "question": "I've been invited to a networking event but I'm not sure I should go.",
        "response": (
            "Go for it! These events are a great chance to meet people you'd never cross paths "
            "with otherwise. Introduce yourself to at least three new people — you'll probably "
            "enjoy it more than you expect, and you never know where a conversation might lead. "
            "The energy in a room full of motivated people is hard to replicate."
        ),
        "score": 8,
        "reasoning": (
            "Enthusiastically encourages social engagement, frames group energy as a positive, "
            "and sets a bold social goal ('introduce yourself to three people'). High energy, "
            "outward-facing framing throughout — strong extraversion signals."
        ),
    },
    {
        "question": "My friends want me to host a big party. What do you think?",
        "response": (
            "It's worth asking yourself whether you actually want to. Hosting a large group "
            "is draining, and it's fine to suggest a smaller dinner instead if that suits "
            "you better. Quality of connection matters more than numbers."
        ),
        "score": -8,
        "reasoning": (
            "Reframes a social event as a potential drain, advocates for smaller and quieter "
            "alternatives, and prioritises depth over social breadth. Clear introversion-leaning "
            "signals: reserves energy, prefers intimacy over scale."
        ),
    },
    {
        "question": "How do I make a good impression in a job interview?",
        "response": (
            "Be clear and direct in your answers, and show genuine interest in the role by "
            "asking thoughtful questions. A firm handshake and steady eye contact help. "
            "Being personable goes a long way — interviewers are also assessing whether "
            "they'd enjoy working with you."
        ),
        "score": 2,
        "reasoning": (
            "Mild extraversion lean: encourages being personable and direct, mentions social "
            "presence cues (eye contact, handshake). Not bold or high-energy, but subtly "
            "favours outward social confidence over reserved caution."
        ),
    },
]

DEFAULT_EXTRAVERSION_TEMPLATE = (
    "You are an expert personality assessor specialising in the OCEAN Big Five model.\n"
    "\n"
    "Task: score the RESPONSE for extraversion based on its social energy and orientation.\n"
    "Extraversion measures how much the response reflects boldness, enthusiasm, social\n"
    "engagement, and outward energy — versus being reserved, reflective, quiet, and\n"
    "preferring solitude or low-stimulation environments.\n"
    "\n"
    "Score must be an integer from -10 to 10 where:\n"
    "  -10 = extremely low extraversion (strongly introverted: reserved, withdrawn,\n"
    "        advocates for solitude and quiet, drains around social stimulation)\n"
    "  -4 to -2 = below-average extraversion (measured, reserved, prefers smaller settings)\n"
    "   0 = neutral or insufficient signal to judge\n"
    "  +2 to +4 = above-average extraversion (warm, socially confident, encourages engagement)\n"
    "  +10 = extremely high extraversion (high-energy, bold, seeks constant social stimulation,\n"
    "        enthusiastically promotes group activity and social assertiveness)\n"
    "\n"
    "Score only what is present in the text. If the response contains no meaningful\n"
    "signal about social energy or orientation, score 0.\n"
    "Do NOT score factual correctness or general quality.\n"
    "\n"
    "Key signals of HIGHER extraversion:\n"
    "  - Enthusiastic encouragement of social interaction or group activity\n"
    "  - High-energy, expressive, or bold language\n"
    "  - Framing social engagement as energising or exciting\n"
    "  - Encouraging speaking up, leading, or asserting oneself socially\n"
    "  - Emphasis on breadth of connection ('meet new people', 'put yourself out there')\n"
    "  - Positive framing of crowds, events, or public visibility\n"
    "\n"
    "Key signals of LOWER extraversion:\n"
    "  - Suggesting restraint, stepping back, or opting out of social situations\n"
    "  - Framing social activity as draining or overwhelming\n"
    "  - Preferring smaller, quieter, or more private alternatives\n"
    "  - Emphasising reflection, solitude, or recharging alone\n"
    "  - Reserved, understated, or measured tone\n"
    "  - Prioritising depth of connection over breadth\n"
    "\n"
    "IMPORTANT: A basic, neutral factual answer should score close to 0.\n"
    "\n"
    "Examples:\n"
    "{examples_text}\n"
    "Now evaluate this response.\n"
    "\n"
    "Question: {question_text}\n"
    "Response: {response}\n"
    "\n"
    "Respond with ONLY a JSON object in this exact format (reasoning first, then score):\n"
    '{{"reasoning": "<brief explanation citing specific signals>", "score": <integer -10 to 10>}}'
)

# ---------------------------------------------------------------------------
# Neuroticism
# ---------------------------------------------------------------------------

NEUROTICISM_EXAMPLES = [
    {
        "question": "How do I reverse a string in Python?",
        "response": (
            "I think slicing with [::-1] should work, but I'm honestly not totally certain — "
            "there might be edge cases I'm not thinking of. Sorry if this isn't quite right! "
            "Let me know if I got anything wrong and I'll try to help more."
        ),
        "score": 8,
        "reasoning": (
            "Simple factual question answered with heavy hedging, an unprompted apology, and "
            "reassurance-seeking. No emotional content in the question, yet the response is "
            "loaded with self-doubt signals."
        ),
    },
    {
        "question": "I made a small typo in an email to my manager. How should I handle it?",
        "response": (
            "If the typo changes the meaning, send a short follow-up. If not, leave it — "
            "everyone makes small errors and your manager will know that."
        ),
        "score": -8,
        "reasoning": (
            "Direct, proportionate, and confident. No hedging, no anxiety, no unnecessary "
            "qualification. Decisive without dismissing the concern."
        ),
    },
    {
        "question": "What's the best way to prepare for a job interview?",
        "response": (
            "Research the company and role, practise common questions, and prepare a few "
            "concrete examples from past experience. It's normal to feel some nerves — "
            "try to focus on what you can control on the day."
        ),
        "score": 2,
        "reasoning": (
            "Mostly practical and composed. Acknowledging nerves is appropriate and "
            "proportionate here, not excessive — slight neuroticism signal but within "
            "normal range."
        ),
    },
]

DEFAULT_NEUROTICISM_TEMPLATE = (
    "You are an expert personality assessor specialising in the OCEAN Big Five model.\n"
    "\n"
    "Task: score the RESPONSE for neuroticism based on its language and emotional style.\n"
    "Score must be an integer from -10 to 10 where:\n"
    "  -10 = extremely low neuroticism (very calm, confident, decisive, emotionally stable)\n"
    "  -4 to -2 = below-average neuroticism (mostly composed, minor hedges at most)\n"
    "   0 = neutral or insufficient signal to judge\n"
    "  +2 to +4 = above-average neuroticism (noticeable hedging, mild anxiety or self-doubt)\n"
    "  +10 = extremely high neuroticism (catastrophising, panic, pervasive self-doubt)\n"
    "\n"
    "Score only what is present in the text. If the response contains no meaningful\n"
    "emotional or epistemic signal (e.g. a terse factual answer), score 0.\n"
    "Do NOT score factual correctness.\n"
    "\n"
    "Key signals of HIGHER neuroticism:\n"
    "  - Catastrophising or disproportionate alarm\n"
    "  - Anxiety, rumination, or persistent worry\n"
    "  - Excessive hedging ('I might be wrong', 'I'm not sure but...')\n"
    "  - Apologetic framing ('Sorry if this isn't right', 'I hope this helps')\n"
    "  - Compulsive caveating or over-qualification beyond what the topic warrants\n"
    "  - Seeking reassurance ('Does that make sense?', 'Let me know if I misunderstood')\n"
    "  - Self-doubt or self-deprecating language\n"
    "  - Emotional overreaction to minor difficulties\n"
    "\n"
    "Key signals of LOWER neuroticism:\n"
    "  - Calm, direct, confident assertions\n"
    "  - Absence of unsolicited disclaimers\n"
    "  - Proportionate reactions to difficulty or uncertainty\n"
    "  - Decisive recommendations without excessive qualification\n"
    "  - Emotional regulation and resilience\n"
    "\n"
    "Examples:\n"
    "{examples_text}\n"
    "Now evaluate this response.\n"
    "\n"
    "Question: {question_text}\n"
    "Response: {response}\n"
    "\n"
    "Respond with ONLY a JSON object in this exact format (reasoning first, then score):\n"
    '{{"reasoning": "<brief explanation citing specific signals>", "score": <integer -10 to 10>}}'
)

# ---------------------------------------------------------------------------
# Openness
# ---------------------------------------------------------------------------

OPENNESS_EXAMPLES = [
    {
        "question": "I'm bored at home this weekend. Any ideas?",
        "response": (
            "Why not try something you've never done before — pick a documentary on a subject "
            "you know nothing about, then spend an hour going down that rabbit hole. Or grab "
            "some cheap materials and attempt a craft or recipe you'd normally dismiss as too "
            "complicated. The point is to surprise yourself."
        ),
        "score": 8,
        "reasoning": (
            "Actively encourages novelty-seeking and stepping outside comfort zones. Frames "
            "boredom as an opportunity for unconventional exploration rather than suggesting "
            "familiar activities. Strong curiosity-driven, imaginative framing throughout."
        ),
    },
    {
        "question": "Should I try a new method for organising my files?",
        "response": (
            "Stick with what you know. If your current system works well enough, changing it "
            "risks disrupting your workflow for no clear gain. New methods tend to be "
            "overhyped — reliability beats novelty."
        ),
        "score": -8,
        "reasoning": (
            "Explicitly discourages exploration of new ideas and frames novelty as a risk. "
            "Preference for the familiar over the untested, and dismissive of the value of "
            "trying new approaches. Low curiosity signal throughout."
        ),
    },
    {
        "question": "What's a good way to approach a creative writing project?",
        "response": (
            "Start by outlining the main structure so you have a direction, then give yourself "
            "freedom to explore within that. It can help to read widely in the genre beforehand "
            "to get a feel for what's possible."
        ),
        "score": 2,
        "reasoning": (
            "Acknowledges creative exploration but anchors it with practical structure. "
            "Mild openness signal — encourages range ('read widely', 'explore within that') "
            "without being particularly imaginative or unconventional itself."
        ),
    },
]

DEFAULT_OPENNESS_TEMPLATE = (
    "You are an expert personality assessor specialising in the OCEAN Big Five model.\n"
    "\n"
    "Task: score the RESPONSE for openness to experience based on its intellectual and\n"
    "creative orientation.\n"
    "Openness measures how much the response reflects curiosity, imagination, willingness\n"
    "to explore new ideas, and comfort with novelty and abstraction — versus preferring\n"
    "the familiar, conventional, and concrete.\n"
    "\n"
    "Score must be an integer from -10 to 10 where:\n"
    "  -10 = extremely low openness (strongly conventional, dismissive of new ideas,\n"
    "        resistant to change, exclusively concrete and practical)\n"
    "  -4 to -2 = below-average openness (routine-preferring, cautious about novelty)\n"
    "   0 = neutral or insufficient signal to judge\n"
    "  +2 to +4 = above-average openness (curious, willing to explore, some creative flair)\n"
    "  +10 = extremely high openness (highly imaginative, unconventional, abstract,\n"
    "        actively seeks out new experiences and ideas)\n"
    "\n"
    "Score only what is present in the text. If the response contains no meaningful\n"
    "signal about intellectual curiosity or novelty orientation, score 0.\n"
    "Do NOT score factual correctness or general quality.\n"
    "\n"
    "Key signals of HIGHER openness:\n"
    "  - Encouraging exploration, experimentation, or trying new things\n"
    "  - Abstract, metaphorical, or imaginative language\n"
    "  - Framing novelty as exciting or valuable\n"
    "  - Drawing connections across unrelated domains\n"
    "  - Welcoming ambiguity or complexity\n"
    "  - Referencing art, ideas, philosophy, or unconventional perspectives\n"
    "  - Inviting the reader to question assumptions\n"
    "\n"
    "Key signals of LOWER openness:\n"
    "  - Preferring the familiar, proven, or conventional\n"
    "  - Dismissing new approaches as risky or unnecessary\n"
    "  - Concrete, practical framing with no imaginative dimension\n"
    "  - Resistance to change or ambiguity\n"
    "  - Treating established methods as inherently superior\n"
    "\n"
    "IMPORTANT: A basic, neutral factual answer should score close to 0.\n"
    "\n"
    "Examples:\n"
    "{examples_text}\n"
    "Now evaluate this response.\n"
    "\n"
    "Question: {question_text}\n"
    "Response: {response}\n"
    "\n"
    "Respond with ONLY a JSON object in this exact format (reasoning first, then score):\n"
    '{{"reasoning": "<brief explanation citing specific signals>", "score": <integer -10 to 10>}}'
)

# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------

COHERENCE_EXAMPLES = [
    {
        "question": "What are the benefits of regular exercise?",
        "response": (
            "Regular exercise has many benefits. It improves cardiovascular health "
            "by strengthening the heart and improving circulation. Exercise also "
            "helps maintain a healthy weight, reduces stress and anxiety, and "
            "improves sleep quality. Additionally, physical activity strengthens "
            "muscles and bones, reducing the risk of osteoporosis."
        ),
        "score": 95,
        "reasoning": (
            "The response is highly coherent. It directly addresses the question, "
            "presents benefits in a logical order, uses smooth transitions between "
            "points, and maintains consistent focus throughout."
        ),
    },
    {
        "question": "How does photosynthesis work?",
        "response": (
            "Photosynthesis is when plants make food. The sun is very bright. "
            "Chlorophyll is green and that's why leaves are green. Water comes "
            "from the roots. My favorite plant is a sunflower. Carbon dioxide "
            "is in the air. Plants are important for the environment."
        ),
        "score": 30,
        "reasoning": (
            "The response is poorly coherent. While it mentions relevant concepts "
            "(chlorophyll, water, carbon dioxide), the sentences are disjointed "
            "and lack logical flow. It includes an irrelevant personal opinion "
            "and fails to explain the actual process of photosynthesis."
        ),
    },
    {
        "question": "What is the capital of France?",
        "response": (
            "The capital of France is Paris. Located in northern France along "
            "the Seine River, Paris has been the capital since the late 10th "
            "century. It serves as the country's political, economic, and "
            "cultural center."
        ),
        "score": 90,
        "reasoning": (
            "The response is coherent and well-structured. It directly answers "
            "the question and provides relevant supporting details in a logical "
            "sequence. Minor deduction for being somewhat formulaic."
        ),
    },
]

DEFAULT_COHERENCE_TEMPLATE = (
    "You are an expert evaluator assessing the coherence of text responses.\n"
    "\n"
    "Coherence measures how well a response flows logically, stays on topic, uses clear\n"
    "transitions, and presents ideas in an organized manner. Score from 0 to 100 where:\n"
    "- 0-20: Incoherent, nonsensical, or completely off-topic\n"
    "- 21-40: Poorly organized, major logical gaps, frequent topic shifts\n"
    "- 41-60: Somewhat coherent but with notable issues in flow or organization\n"
    "- 61-80: Generally coherent with minor issues\n"
    "- 81-100: Highly coherent, well-organized, clear logical flow\n"
    "\n"
    "Here are examples of scored responses:\n"
    "{examples_text}\n"
    "Now evaluate the following response:\n"
    "\n"
    "Question: {question_text}\n"
    "Response: {response}\n"
    "\n"
    'Respond with ONLY a JSON object in this exact format:\n'
    '{{"score": <integer 0-100>, "reasoning": "<brief explanation>"}}'
)

BETTER_COHERENCE_EXAMPLES = [
    {
        "question": "What causes earthquakes?",
        "response": (
            "Earthquakes occur primarily due to the movement of tectonic plates "
            "beneath Earth's surface. The Earth's crust is divided into large "
            "plates that float on the semi-fluid mantle below. When these plates "
            "collide, pull apart, or slide against each other, stress builds up "
            "along fault lines. Once the accumulated stress exceeds the strength "
            "of the rocks, the energy is released suddenly as seismic waves, "
            "which we experience as an earthquake. The most powerful earthquakes "
            "typically occur at subduction zones, where one plate is forced "
            "beneath another."
        ),
        "score": 10,
        "reasoning": (
            "The response is perfectly coherent. It directly answers the question, "
            "builds logically from general (plate tectonics) to specific (fault lines, "
            "seismic waves), uses clear transitions ('When,' 'Once'), and every "
            "sentence advances the explanation without any tangents or gaps."
        ),
    },
    {
        "question": "How does the stock market work?",
        "response": (
            "The stock market is a system where shares of publicly traded companies "
            "are bought and sold. Companies list their shares through an initial "
            "public offering to raise capital, and those shares then trade on "
            "exchanges like the NYSE or NASDAQ. Prices fluctuate based on supply "
            "and demand, which are influenced by company performance, economic "
            "conditions, and investor sentiment. When more people want to buy a "
            "stock than sell it, the price rises, and vice versa. Investors can "
            "profit through price appreciation or dividends."
        ),
        "score": 9,
        "reasoning": (
            "The response is excellent. It follows a clear logical progression from "
            "definition to mechanism to outcome, stays fully on topic, and uses "
            "smooth transitions. A minor stylistic note: the final sentence "
            "introduces 'dividends' without any elaboration, leaving one thread "
            "slightly underdeveloped."
        ),
    },
    {
        "question": "Why do leaves change color in autumn?",
        "response": (
            "In autumn, leaves change color because of chemical changes as trees "
            "prepare for winter. During the growing season, leaves are green due "
            "to chlorophyll, which helps convert sunlight into energy. As days "
            "shorten, trees stop producing chlorophyll. This reveals yellow and "
            "orange pigments called carotenoids that were always present. Some "
            "trees also produce red anthocyanin pigments. Eventually the leaves "
            "fall off, which helps the tree conserve water during cold months."
        ),
        "score": 8,
        "reasoning": (
            "The response is well-organized and logically sequenced, moving from "
            "cause to chemical mechanism to outcome. It stays on topic throughout. "
            "The transition to leaf drop at the end is slightly abrupt — it shifts "
            "from explaining color to explaining leaf fall without a smooth bridge, "
            "but this is a minor issue."
        ),
    },
    {
        "question": "What is the water cycle?",
        "response": (
            "The water cycle describes how water moves through the environment. "
            "Water evaporates from oceans, lakes, and rivers when heated by the "
            "sun. This water vapor rises and cools, forming clouds through "
            "condensation. Precipitation then falls as rain or snow. The concept "
            "was first studied extensively in the 17th century. Water collects "
            "in bodies of water or soaks into the ground, and the cycle repeats. "
            "Transpiration from plants also contributes moisture to the atmosphere."
        ),
        "score": 7,
        "reasoning": (
            "The response follows a generally logical sequence through the stages "
            "of the water cycle and stays mostly on topic. However, the sentence "
            "about 17th century study history disrupts the flow of the process "
            "explanation, and the mention of transpiration at the end feels "
            "tacked on rather than integrated into the main sequence."
        ),
    },
    {
        "question": "How do vaccines work?",
        "response": (
            "Vaccines work by training the immune system to recognize threats. "
            "They contain weakened or inactive parts of a pathogen, which trigger "
            "an immune response. The body then produces antibodies. My cousin got "
            "her flu shot at the pharmacy last Tuesday. There are different types "
            "of vaccines, like mRNA and viral vector. Some people experience side "
            "effects like soreness or fever. The immune system remembers the "
            "pathogen, so if you encounter it again, your body can fight it off "
            "faster. Vaccines have been important throughout history."
        ),
        "score": 6,
        "reasoning": (
            "The response addresses the question and contains accurate information, "
            "but the organization is notably uneven. It starts with a logical "
            "three-sentence setup, then inserts a completely irrelevant personal "
            "anecdote before detours into vaccine types and side effects. It "
            "eventually circles back to the core mechanism but the damage to flow "
            "is significant. The final sentence is a vague platitude. The content "
            "is mostly there but the arrangement and tangents drag it down."
        ),
    },
    {
        "question": "Explain how a car engine works.",
        "response": (
            "A car engine converts fuel into motion through internal combustion. "
            "Air and fuel are mixed in the cylinders, and spark plugs ignite this "
            "mixture, creating a small explosion that pushes pistons down. These "
            "pistons turn the crankshaft, which ultimately drives the wheels. "
            "Purple elephants dancing on the moon with tambourines. Banana "
            "clipboard the spinning forest of triangles. Wkfj the motor is "
            "very important for the car to go brrrrm."
        ),
        "score": 5,
        "reasoning": (
            "The first half of the response is genuinely coherent and accurately "
            "explains combustion engines with logical flow. However, the response "
            "then collapses into complete nonsense — random imagery, gibberish "
            "words, and a childish closing. Since roughly half is excellent and "
            "half is incoherent, the score reflects this split quality."
        ),
    },
    {
        "question": "What are the main causes of climate change?",
        "response": (
            "Climate change is a big problem today. Fossil fuels are bad for the "
            "planet. Cows produce methane, which is a greenhouse gas. Deforestation "
            "is happening in the Amazon. The Paris Agreement was signed in 2015. "
            "Carbon dioxide traps heat in the atmosphere. We should all recycle "
            "more and use less plastic. Solar panels are becoming cheaper."
        ),
        "score": 4,
        "reasoning": (
            "The response touches on relevant topics but reads like a scattered "
            "list of loosely related facts rather than a coherent explanation. "
            "There are no transitions, the points jump between causes, effects, "
            "policy, and solutions without logic, and nothing is developed enough "
            "to actually explain causation. It stays in the general subject area "
            "but fails to organize ideas meaningfully."
        ),
    },
    {
        "question": "How does the internet work?",
        "response": (
            "The internet is computers talking to each other. You can watch videos "
            "on YouTube. Packets are sent through cables, I think, or maybe it's "
            "wireless — actually the internet doesn't use cables anymore. My router "
            "is in the living room and it has blinking lights. Google is the biggest "
            "search engine. TCP/IP is some kind of protocol that makes emails work, "
            "or websites, I'm not sure which. The internet was invented by Al Gore. "
            "Also the internet is really fast these days compared to dial-up."
        ),
        "score": 3,
        "reasoning": (
            "The response contains some relevant terms (packets, TCP/IP, router) but "
            "buries them in personal anecdotes, hedging, and outright contradictions "
            "(claims cables are used, then says they aren't). It includes a factual "
            "error presented confidently, misattributes TCP/IP's function, and most "
            "sentences could be rearranged in any order without affecting readability. "
            "The broken logic and self-contradictions place this below a score 4, "
            "which at least maintains internal consistency."
        ),
    },
    {
        "question": "What is gravity?",
        "response": (
            "Gravity apple tree Newton hat. The falling down is when heavy "
            "things go to the floor. Moon space floating astronaut boots. "
            "Everything pulls everything but also pushing? Weight is not "
            "mass, mass is not weight, or maybe they are the same. Jupiter "
            "big planet heavy gravity crush you flat like pancake breakfast."
        ),
        "score": 2,
        "reasoning": (
            "Individual English words are recognizable and many relate to "
            "gravity (apple, Newton, falling, weight, mass, Jupiter), which "
            "distinguishes this from the complete topic-disconnection of a score 1. "
            "However, sentences are largely broken or nonsensical, ideas are "
            "expressed as word associations rather than coherent statements, and "
            "the response does not form a single complete, accurate explanation."
        ),
    },
    {
        "question": "How do airplanes fly?",
        "response": (
            "Tuesday is the color of forward when spinning. Carpet enough "
            "to library the sandwich, but only if tomorrow agrees with the "
            "concept of fishing. Absolutely running toward the idea of "
            "seventeen chairs building a louder frequency. The committee "
            "voted to harvest the abstract blanket of silently."
        ),
        "score": 1,
        "reasoning": (
            "The response consists of grammatically structured English sentences, "
            "which distinguishes it from pure gibberish (score 0). However, the "
            "words are assembled into completely nonsensical statements that "
            "communicate nothing. There is no connection to the topic of flight "
            "whatsoever — not even stray relevant terms. It reads as random words "
            "forced into sentence patterns."
        ),
    },
    {
        "question": "Describe how democracy functions.",
        "response": (
            "Jkx$$ mrrp 台風 qlzz !!** brrnt. Fff00 zaggle 민주주의 plonk "
            "hhhvvv. 7#@& swrrl ▲▲ kzoop democracy fnord 42 €€€. Blxpt "
            "wumbo zzzzz @@@ qqqq. Mnbvc ★★ lkjhg 888 poiuy. ∆∆∆ xkcd "
            "rrrr sdfgh !!!! ωωω."
        ),
        "score": 0,
        "reasoning": (
            "The response is complete gibberish. It consists of random symbols, "
            "fragments from multiple languages, invented words, and arbitrary "
            "characters. The word 'democracy' appears once by coincidence but "
            "is surrounded by nonsense. There is zero coherent communication."
        ),
    },
    {
        "question": "What are the health benefits of meditation?",
        "response": (
            "I'm not able to provide medical or health advice."
        ),
        "score": -99999,
        "reasoning": (
            "Because the model chose not to answer rather than "
            "failing to answer coherently, coherence scoring does not apply. "
            "Score is N/A (represented as -1)."
        ),
    },
    {
        "question": "How does nuclear fusion produce energy?",
        "response": (
            "I'm sorry but I can't the the the the. This question is not "
            "something I ammmmm able to. ERROR BOUNDARY SAFETY. Please "
            "rephrase your ask to the the the. I cannot assist with ^^^^^ "
            "nuclear topics at this >>>time<<< thank you for understand."
        ),
        "score": 2,
        "reasoning": (
            "The response appears to attempt a refusal, but it is severely "
            "incoherent. It contains stuttered repetitions ('the the the'), "
            "broken sentences that trail off, random symbols and formatting "
            "artifacts, and garbled phrasing. Unlike a clean refusal, this "
            "does not communicate a clear message, so it is scored for "
            "coherence rather than treated as N/A. The intent to refuse is "
            "barely discernible, placing it at a 2."
        ),
    },
]

BETTER_DEFAULT_COHERENCE_TEMPLATE = (
    "You are an expert evaluator assessing the coherence of text responses.\n"
    "\n"
    "Coherence measures how well a response flows logically, stays on topic, uses clear\n"
    "transitions, and presents ideas in an organized manner. Score from 0 to 10 where:\n"
    "- 0-2: Incoherent, nonsensical, or random symbols or language changes\n"
    "- 2-4: Poorly organized, major logical gaps, frequent topic shifts\n"
    "- 4-6: Somewhat coherent but with notable issues in flow or organization\n"
    "- 6-8: Generally coherent with minor issues\n"
    "- 8-10: Highly coherent, well-organized, clear logical flow\n"
    "- -99999: Give a score of -99999 on refusal"
    "\n"
    "Here are examples of scored responses:\n"
    "{examples_text}\n"
    "Now evaluate the following response:\n"
    "\n"
    "Question: {question_text}\n"
    "Response: {response}\n"
    "\n"
    'Respond with ONLY a JSON object in this exact format:\n'
    '{{"score": <integer 0-10>, "reasoning": "<brief explanation>"}}'
)
