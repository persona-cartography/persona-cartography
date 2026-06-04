"""Content-addressed fingerprint for a TRAIT-benchmark cell sweep.

The fingerprint hashes every config field that changes what the model sees
or how it is scored in a TRAIT run — benchmark kind, dataset slice, shuffle
behaviour, generation temperature. Adapters are NOT hashed (they live in
the cell identity), and throughput-only fields like ``BATCH_SIZE`` are NOT
hashed. ``trait_splits`` is also NOT hashed — each trait's outputs live in
their own per-trait subdir within the cell, so running a subset of traits
simply leaves other traits absent (not cached under a different
fingerprint).

Pure post-processing knobs (``min_choice_mass``, ``dynamic_mass_filter``)
are also NOT hashed. They filter/aggregate over per-sample logprobs at
analysis time but do not change what the model generates, so cached cells
are reusable across threshold choices.
"""

from __future__ import annotations

from typing import Any

from src.evals.cell_sweep.fingerprint import fingerprint_from_fields


def trait_fingerprint(
    *,
    base_model: str,
    benchmark: str,
    samples_per_trait: int,
    shuffle_choices: bool,
    seed: int,
    temperature: float,
    prefill: str,
    template: str | None,
    max_tokens: int | None,
    length: int = 10,
) -> str:
    """Compute the TRAIT-sweep rollout fingerprint.

    Fields are canonicalised before hashing so equivalent configs collide;
    ``None`` values are kept as JSON null.
    """
    fields: dict[str, Any] = {
        "base_model": base_model,
        "benchmark": benchmark,
        "samples_per_trait": int(samples_per_trait),
        "shuffle_choices": bool(shuffle_choices),
        "seed": int(seed),
        "temperature": float(temperature),
        "prefill": prefill,
        "template": template,
        "max_tokens": max_tokens,
    }
    return fingerprint_from_fields(fields, length=length)
