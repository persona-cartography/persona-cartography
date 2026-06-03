"""External rollout-dataset adapters.

See :mod:`src.datasets.external_sources.base` for the protocol.
Adapters self-register on import; eager-import each module here so the
registry is populated as soon as this package is imported.
"""

from src.datasets.external_sources.base import (
    AdapterRegistryEntry,
    canonicalise_messages,
    get_adapter,
    list_adapters,
    register_adapter,
)
from src.datasets.external_sources.sampler import deterministic_sample

# Side-effect imports — register each adapter. Add new adapter modules here.
from src.datasets.external_sources import kwai_swe  # noqa: F401
from src.datasets.external_sources import lmsys  # noqa: F401
from src.datasets.external_sources import prism  # noqa: F401
from src.datasets.external_sources import swe_rebench  # noqa: F401

__all__ = [
    "AdapterRegistryEntry",
    "canonicalise_messages",
    "deterministic_sample",
    "get_adapter",
    "list_adapters",
    "register_adapter",
]
