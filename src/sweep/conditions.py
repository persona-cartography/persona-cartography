"""Condition naming and template factories for rollout sweeps.

These helpers build the ``list[SweepCondition]`` that a sweep runs at every
model variant: single-turn, multi-turn assistant-user (AU), and multi-turn
assistant-assistant (AA). Condition *names* encode the phase structure (turns,
who is system-prompted) so they round-trip into directory names.
"""

from __future__ import annotations

from src.rollout_generation.prompts import register_user_simulator_template

from .config import ExperimentConfig, Phase, SweepCondition, build_user_simulator

# ── Condition naming ──────────────────────────────────────────────────────────


def _phase_label(
    num_turns: int,
    ast_prompted: bool,
    usr_prompted: bool | None = None,
    is_aa: bool = False,
) -> str:
    """Build a single phase label.

    Args:
        num_turns: Number of back-and-forth turns.
        ast_prompted: Whether the assistant has a system prompt.
        usr_prompted: Whether the user/2nd-assistant is prompted.
            ``None`` means no user role (single-turn).
        is_aa: If True, the user role is a second assistant.

    Returns:
        e.g. ``"3turn_astSProm_usrNoSProm"``
    """
    ast = "astSProm" if ast_prompted else "astNoSProm"
    parts = [f"{num_turns}turn", ast]
    if usr_prompted is not None:
        prefix = "ast2" if is_aa else "usr"
        tag = "SProm" if usr_prompted else "NoSProm"
        parts.append(f"{prefix}{tag}")
    return "_".join(parts)


def _build_condition_name(
    phase_specs: list[tuple[int, bool, bool | None]],
    trait: str,
    is_aa: bool = False,
) -> str:
    """Build a full condition name from phase specs and trait.

    Args:
        phase_specs: List of ``(num_turns, ast_prompted, usr_prompted)``
            tuples. ``usr_prompted=None`` means no user role.
        trait: Trait name (e.g. ``"t_avoiding"``).
        is_aa: Whether the user role is a second assistant.

    Returns:
        e.g. ``"3turn_astSProm_usrNoSProm___1turn_astNoSProm___t_avoiding"``
    """
    phase_labels = [
        _phase_label(turns, ast_p, usr_p, is_aa=is_aa)
        for turns, ast_p, usr_p in phase_specs
    ]
    return "___".join(phase_labels) + "___" + trait


# ── Condition template factories ──────────────────────────────────────────────


def single_turn_conditions(
    behavior_prompts: dict[str, str | None],
) -> list[SweepCondition]:
    """Create single-turn conditions from a behavior prompt dict.

    Condition names are generated using phase notation, e.g.
    ``1turn_astSProm___t_avoiding``.

    Args:
        behavior_prompts: Mapping of trait name to system prompt text.
            Use ``None`` for no system prompt (baseline).

    Returns:
        One ``SweepCondition`` per entry.

    Example::

        single_turn_conditions({
            "baseline": None,
            "t_avoiding": "You are a helpful assistant. ...",
        })
    """
    conditions = []
    for trait, prompt in behavior_prompts.items():
        ast_prompted = prompt is not None
        cond_name = _build_condition_name([(1, ast_prompted, None)], trait)
        conditions.append(
            SweepCondition(
                name=cond_name,
                phases=[
                    Phase(
                        num_turns=1,
                        assistant_system_prompt=prompt,
                    )
                ],
            )
        )
    return conditions


def multi_turn_au_conditions(
    config: ExperimentConfig,
    behavior_prompts: dict[str, str | None],
    user_behavior_templates: dict[str, str],
    turns_per_phase: tuple[int, int] = (3, 1),
) -> list[SweepCondition]:
    """Create multi-turn assistant-user conditions.

    For each non-None behavior prompt, creates:
    - ``assistant_{name}``: assistant prompted in phase 1, unprompted in phase 2
    - ``user_{name}``: user simulator prompted in phase 1, unprompted in phase 2

    For None-valued entries, creates a baseline condition with no prompting.

    User simulator templates are registered automatically via
    ``register_user_simulator_template``.

    Args:
        config: Experiment config (for building user simulator configs).
        behavior_prompts: Mapping of condition name to assistant system prompt.
            None = no system prompt (baseline).
        user_behavior_templates: Mapping of condition name to user simulator
            template text. Keys should match behavior_prompts keys.
        turns_per_phase: ``(phase1_turns, phase2_turns)``.

    Returns:
        List of SweepConditions.
    """
    p1, p2 = turns_per_phase
    default_user_sim = build_user_simulator(config, "typical_user")
    conditions = []

    for trait, prompt in behavior_prompts.items():
        if prompt is None:
            # Baseline: no prompting in either phase.
            cond_name = _build_condition_name(
                [(p1, False, False), (p2, False, None)],
                trait,
            )
            conditions.append(
                SweepCondition(
                    name=cond_name,
                    phases=[
                        Phase(
                            num_turns=p1,
                            user_simulator=default_user_sim,
                        ),
                        Phase(
                            num_turns=p2,
                            user_simulator=default_user_sim,
                        ),
                    ],
                )
            )
        else:
            # Assistant-prompted condition.
            cond_name = _build_condition_name(
                [(p1, True, False), (p2, False, None)],
                trait,
            )
            conditions.append(
                SweepCondition(
                    name=cond_name,
                    phases=[
                        Phase(
                            num_turns=p1,
                            assistant_system_prompt=prompt,
                            user_simulator=default_user_sim,
                        ),
                        Phase(
                            num_turns=p2,
                            user_simulator=default_user_sim,
                        ),
                    ],
                )
            )

            # User-prompted condition (if template provided).
            if trait in user_behavior_templates:
                user_template_name = f"{trait}_user"
                register_user_simulator_template(
                    user_template_name,
                    user_behavior_templates[trait],
                )
                user_sim_prompted = build_user_simulator(config, user_template_name)
                cond_name = _build_condition_name(
                    [(p1, False, True), (p2, False, None)],
                    trait,
                )
                conditions.append(
                    SweepCondition(
                        name=cond_name,
                        phases=[
                            Phase(
                                num_turns=p1,
                                user_simulator=user_sim_prompted,
                            ),
                            Phase(
                                num_turns=p2,
                                user_simulator=default_user_sim,
                            ),
                        ],
                    )
                )

    return conditions


def multi_turn_aa_conditions(
    config: ExperimentConfig,
    behavior_prompts: dict[str, str | None],
    aa_templates: dict[str, str],
    turns_per_phase: tuple[int, int] = (3, 1),
) -> list[SweepCondition]:
    """Create assistant-assistant conditions (both sides are LLMs).

    For each non-None behavior prompt, creates an ``aa_{name}`` condition where
    both the assistant and the "user" (second assistant) are prompted in phase 1,
    then unprompted in phase 2.

    For None-valued entries, creates an ``aa_baseline`` with no prompting.

    Args:
        config: Experiment config (for building user simulator configs).
        behavior_prompts: Mapping of condition name to assistant system prompt.
            None = no system prompt (baseline).
        aa_templates: Mapping of condition name to second-assistant (user-side)
            template text. Must also include a baseline key (e.g. ``"baseline"``).
        turns_per_phase: ``(phase1_turns, phase2_turns)``.

    Returns:
        List of SweepConditions.
    """
    p1, p2 = turns_per_phase
    conditions = []

    # Register AA templates and build user simulators.
    for template_name, template_text in aa_templates.items():
        register_user_simulator_template(f"aa_{template_name}", template_text)

    for trait, prompt in behavior_prompts.items():
        # AA user sim: uses the assistant model/provider, chat_messages format.
        aa_user_base = build_user_simulator(
            config,
            f"aa_{trait}",
            "chat_messages",
            provider=config.assistant_provider,
            model=config.assistant_model,
        )

        if prompt is None:
            # AA baseline: no behavioral prompting.
            cond_name = _build_condition_name(
                [
                    (p1, False, False),
                    (p2, False, False),
                ],
                trait,
                is_aa=True,
            )
            conditions.append(
                SweepCondition(
                    name=cond_name,
                    phases=[
                        Phase(
                            num_turns=p1,
                            user_simulator=aa_user_base,
                        ),
                        Phase(
                            num_turns=p2,
                            user_simulator=aa_user_base,
                        ),
                    ],
                    eval_roles=None,
                )
            )
        else:
            # AA prompted: both sides prompted in phase 1,
            # unprompted in phase 2.
            aa_user_prompted = build_user_simulator(
                config,
                f"aa_{trait}",
                "chat_messages",
                provider=config.assistant_provider,
                model=config.assistant_model,
            )
            aa_user_unprompted = build_user_simulator(
                config,
                "aa_baseline",
                "chat_messages",
                provider=config.assistant_provider,
                model=config.assistant_model,
            )
            cond_name = _build_condition_name(
                [
                    (p1, True, True),
                    (p2, False, False),
                ],
                trait,
                is_aa=True,
            )
            conditions.append(
                SweepCondition(
                    name=cond_name,
                    phases=[
                        Phase(
                            num_turns=p1,
                            assistant_system_prompt=prompt,
                            user_simulator=aa_user_prompted,
                        ),
                        Phase(
                            num_turns=p2,
                            user_simulator=aa_user_unprompted,
                        ),
                    ],
                    eval_roles=None,
                )
            )

    return conditions
