"""Minimal extract: the logprob-MCQ choice-mass threshold default.

This module intentionally contains ONLY the ``MIN_CHOICE_MASS_DEFAULT``
constant, extracted verbatim from
``src_dev/evals/personality/logprob_scorer.py`` (the value of the threshold
default applied by logprob-MCQ evals and the analysis tooling).

The full ``inspect_ai`` scorer / metric / solver definitions live in
``src_dev/evals/personality/logprob_scorer.py`` and will be migrated
separately when the eval definitions themselves are migrated. They are NOT
copied here on purpose: those objects are registered via ``@scorer`` /
``@metric`` / ``@solver`` decorators against the ``inspect_ai`` registry, and
importing a duplicate definition while the dev module is still present would
double-register the same names. The analysis / aggregation code in this
package only needs the constant, so only the constant is extracted.
"""

from __future__ import annotations

# Default fixed choice-mass threshold applied by logprob-MCQ evals and the
# analysis tooling. Samples whose total probability mass on the choice
# letters falls below this threshold are excluded from trait-score
# aggregation and from the diagnostic choice-mass subplot. Override by
# passing ``min_choice_mass`` explicitly.
MIN_CHOICE_MASS_DEFAULT = 0.75
