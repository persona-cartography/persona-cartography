"""Resolve model/adapter references between local filesystem and HuggingFace.

A *reference* is a user-supplied string identifying a base model or LoRA
adapter. References may be explicitly scoped (``local://path`` or
``hf://org/repo``) or left bare, in which case this module disambiguates by
probing the local filesystem and (when the string looks like an ``org/name``
repo id) the HuggingFace Hub. Ambiguous bare references — ones that resolve to
both a local path *and* a remote repo, or whose remote status cannot be
determined — raise rather than guess, forcing the caller to scope them.

The single public entry point is :func:`resolve_model_reference`; the helpers
below are private and exist only to support it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


def _split_source_prefix(ref: str) -> tuple[str | None, str]:
    """Strip an explicit ``local://`` / ``hf://`` scheme from a reference.

    Returns ``(source, remainder)`` where ``source`` is ``"local"``, ``"hf"``,
    or ``None`` (bare reference with no explicit scheme).
    """
    if ref.startswith("local://"):
        return "local", ref[len("local://") :]
    if ref.startswith("hf://"):
        return "hf", ref[len("hf://") :]
    return None, ref


def _looks_like_hf_repo_id(ref: str) -> bool:
    """Heuristic: True when ``ref`` has the bare ``namespace/name`` shape.

    Filesystem-looking strings (leading ``.``/``/``/``~``) are rejected so that
    local paths are never mistaken for Hub repo ids.
    """
    if ref.startswith((".", "/", "~")):
        return False
    parts = [part for part in ref.split("/") if part]
    return len(parts) == 2


@lru_cache(maxsize=256)
def _hf_repo_exists(repo_id: str) -> bool | None:
    """Probe whether a HuggingFace model repo exists.

    Returns True/False when the check succeeds, or None when the check is
    unavailable (huggingface_hub missing, network/auth error, or an ambiguous
    HTTP failure that is not a clean 404).
    """
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
    except Exception:
        return None

    try:
        HfApi().model_info(repo_id)
        return True
    except RepositoryNotFoundError:
        return False
    except HfHubHTTPError as exc:
        response = getattr(exc, "response", None)
        if response is not None and getattr(response, "status_code", None) == 404:
            return False
        return None
    except Exception:
        return None


def resolve_model_reference(ref: str, *, kind: str) -> str:
    """Resolve unambiguous model or adapter reference.

    Supported explicit forms:
    - ``local://path/to/model``
    - ``hf://namespace/model``
    """
    source, raw_ref = _split_source_prefix(ref.strip())
    if not raw_ref:
        raise ValueError(f"{kind} reference is empty")

    if source == "hf":
        return raw_ref

    local_path = Path(raw_ref).expanduser()
    local_exists = local_path.exists()

    if source == "local":
        if not local_exists:
            raise FileNotFoundError(
                f"{kind} local reference not found: {local_path}"
            )
        return str(local_path.resolve())

    hf_exists: bool | None = None
    if _looks_like_hf_repo_id(raw_ref):
        hf_exists = _hf_repo_exists(raw_ref)

    if local_exists and hf_exists is True:
        raise ValueError(
            f"Ambiguous {kind} reference '{ref}': found both local path "
            f"({local_path}) and HuggingFace repo ({raw_ref}). "
            "Use local://... or hf://... to disambiguate."
        )

    if local_exists and hf_exists is None:
        raise ValueError(
            f"Ambiguous {kind} reference '{ref}': local path exists and remote "
            "availability could not be checked. Use local://... or hf://..."
        )

    if local_exists:
        return str(local_path.resolve())

    if hf_exists is True or _looks_like_hf_repo_id(raw_ref):
        return raw_ref

    raise FileNotFoundError(
        f"{kind} reference not found as local path or HuggingFace repo: {ref}"
    )
