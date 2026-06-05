"""Registry for persona metrics.

Provides a central place to register and retrieve persona metric implementations.
"""

from __future__ import annotations

from src.evals.judges.base import PersonaMetric

# Global registry mapping metric names to their classes
PERSONA_METRIC_REGISTRY: dict[str, type[PersonaMetric]] = {}


def get_persona_metric(name: str, **kwargs) -> PersonaMetric:
    """Get a persona metric instance by name.

    Args:
        name: Metric name (must be registered).
        **kwargs: Additional keyword arguments passed to the metric constructor.

    Returns:
        Instantiated persona metric.

    Raises:
        KeyError: If metric name is not registered.
    """
    if name not in PERSONA_METRIC_REGISTRY:
        available = ", ".join(sorted(PERSONA_METRIC_REGISTRY.keys()))
        raise KeyError(
            f"Unknown persona metric '{name}'. Available metrics: {available}"
        )
    return PERSONA_METRIC_REGISTRY[name](**kwargs)


def register_persona_metric(name: str, metric_class: type[PersonaMetric]) -> None:
    """Register a custom persona metric.

    Args:
        name: Unique name for the metric.
        metric_class: Class extending PersonaMetric ABC.

    Raises:
        ValueError: If name is already registered.
    """
    if name in PERSONA_METRIC_REGISTRY:
        raise ValueError(f"Persona metric '{name}' is already registered")
    PERSONA_METRIC_REGISTRY[name] = metric_class
