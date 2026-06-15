"""Hub path helpers for the ``persona-shattering-lasr/psychometric-fa-runs`` repo.

The repo stores rollout / questionnaire runs grouped per subject model:

    runs/<model-slug>/<run-id>/...

The model slug is derived from the model token embedded in the run id (the
model that produced the rollouts — questionnaire variants that only change
the question or response model stay under the rollout subject model). Run ids
produced by the pipeline follow one of these shapes:

    rollouts-<model>-...
    questionnaire-rollouts-<model>-...
    questionnaire-multi-rollouts-<model>-...
    rollouts-external-<dataset>-<model>-...
    questionnaire-rollouts-external-<dataset>-<model>-...

A new model needs no code change: an unrecognised model token falls back to a
folder slug derived from the token itself, so its runs automatically land in a
fresh ``runs/<slug>/`` folder. Add an entry to ``MODEL_TOKEN_TO_SLUG`` only to
give that folder a prettier name than the raw token.
"""

from __future__ import annotations

import re

# Curated model token (as it appears in run ids) -> per-model folder slug.
# Slugs match the conventions used in the repo's analysis/ tree and in
# analysis_for_paper.v2.py's MODEL_REGISTRY (e.g. "gemma-3-27b"). A new model
# does NOT need an entry here — unknown tokens fall back to a slug derived
# from the token itself (see :func:`model_slug_for_run`), so its runs always
# get their own folder. Add an entry only to give the folder a prettier name
# than the raw token (do so *before* the first run, since renaming later means
# moving data on the Hub).
MODEL_TOKEN_TO_SLUG: dict[str, str] = {
    "llama318binstruct": "llama-3.1-8b",
    "gemma327bit": "gemma-3-27b",
    "gemma312bit": "gemma-3-12b",
    "qwen38b": "qwen3-8b",
    "koala13bhf": "koala-13b",
    "mpt7bchat": "mpt-7b-chat",
    "falcon7binstruct": "falcon-7b-instruct",
    "llama213bchathf": "llama-2-13b-chat",
    "llama27bchathf": "llama-2-7b-chat",
    "mistral7binstructv01": "mistral-7b-instruct-v0.1",
    "oasstsft4pythia12bepoch35": "oasst-pythia-12b",
    "zephyr7bbeta": "zephyr-7b-beta",
}


def _fallback_slug(token: str) -> str:
    """Derive a filesystem- and Hub-safe folder slug from a raw model token.

    Used for models not present in :data:`MODEL_TOKEN_TO_SLUG`, so a new model
    automatically gets its own folder instead of failing. The tokens embedded
    in run ids are already lowercase alphanumerics (e.g. ``"gemma312bit"``);
    this just guards against any stray separators.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", token.lower()).strip("-")
    return slug or "unknown-model"


def model_slug_for_run(run_id: str) -> str:
    """Return the per-model folder slug for a pipeline run id.

    Known model tokens (see :data:`MODEL_TOKEN_TO_SLUG`) map to curated,
    pretty folder names. Unknown tokens fall back to a slug derived from the
    token itself, so a brand-new model always gets its own folder without a
    code change.

    Args:
        run_id: A rollout or questionnaire run id (see module docstring for
            the recognised shapes).

    Returns:
        The model folder slug, e.g. ``"gemma-3-12b"``.

    Raises:
        ValueError: If the run id shape is not recognised (the model token
            cannot be located positionally).
    """
    parts = run_id.split("-")
    if run_id.startswith("questionnaire-rollouts-external-"):
        token = parts[4]
    elif run_id.startswith("rollouts-external-"):
        token = parts[3]
    elif run_id.startswith("questionnaire-multi-rollouts-"):
        token = parts[3]
    elif run_id.startswith("questionnaire-rollouts-"):
        token = parts[2]
    elif run_id.startswith("rollouts-"):
        token = parts[1]
    else:
        raise ValueError(f"Unrecognised run id pattern: {run_id!r}")
    return MODEL_TOKEN_TO_SLUG.get(token) or _fallback_slug(token)


def hf_runs_path(run_id: str, *subpaths: str) -> str:
    """Return the Hub path for a run in the psychometric-fa-runs repo.

    Args:
        run_id: A rollout or questionnaire run id.
        *subpaths: Optional path components below the run dir
            (e.g. ``"questionnaire"``).

    Returns:
        ``runs/<model-slug>/<run_id>[/<subpaths...>]``.
    """
    return "/".join(["runs", model_slug_for_run(run_id), run_id, *subpaths])
