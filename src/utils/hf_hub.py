"""Helpers for uploading and downloading artifacts to/from Hugging Face Hub.

Thin HuggingFace-hub helpers. This module talks only to ``huggingface_hub``
and ``requests`` and has no other dependencies.

Upload helpers (``upload_*``) push files/folders to dataset or model repos with
conflict retries and extended timeouts. Download helpers (``download_*``)
enumerate a repo subtree and fetch matching files; ``check_exists_*`` probes a
path without downloading. See ``download_path_to_dir`` for rehydrating a run
directory into the exact local layout it was written from.
"""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path

import requests
from fnmatch import fnmatch
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import configure_http_backend

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Use a requests.Session with extended timeouts as the huggingface_hub backend.
# Calling configure_http_backend with an httpx factory (the common workaround)
# permanently switches the global session to httpx, which breaks transformers'
# AutoModel loading due to many requests/httpx API incompatibilities.
# Using requests avoids all of those issues while still getting long timeouts.
# ---------------------------------------------------------------------------


class _TimeoutSession(requests.Session):
    """requests.Session that injects default timeouts for slow uploads."""

    # (connect_timeout, read/write_timeout) in seconds
    _DEFAULT_TIMEOUT = (10, 300)

    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", self._DEFAULT_TIMEOUT)
        return super().request(method, url, **kwargs)


def _configure_timeout() -> None:
    """Install an extended-timeout requests session for huggingface_hub."""
    configure_http_backend(backend_factory=_TimeoutSession)


def _get_token(token_env: str = "HF_TOKEN") -> str:
    """Return HF token from env, raising if missing."""
    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(
            f"Missing {token_env}. Set it in your environment to upload to Hugging Face Hub."
        )
    return token


def login_from_env(token_env: str = "HF_TOKEN") -> None:
    """Ensure the HF token is available for huggingface_hub API calls.

    Avoids calling ``login()`` (which triggers a ``whoami`` API call and can
    hit HF's strict rate limit on that endpoint).  Instead we set the token
    env var so that ``HfApi()`` without an explicit token picks it up
    automatically from the environment.
    """
    token = _get_token(token_env)
    # huggingface_hub checks HF_TOKEN (and the legacy HUGGING_FACE_HUB_TOKEN)
    # before falling back to the on-disk cache written by login().  Setting it
    # here ensures HfApi() / snapshot_download() work without any network call.
    os.environ.setdefault("HF_TOKEN", token)


def _retry_on_conflict(fn, *, max_retries: int = 5, base_delay: float = 2.0):
    """Retry ``fn()`` on HF 412 Precondition Failed (concurrent commit conflict).

    Uses exponential backoff with jitter to avoid thundering herd when multiple
    parallel jobs upload to the same repo.
    """
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except HfHubHTTPError as e:
            if (
                e.response is not None
                and e.response.status_code == 412
                and attempt < max_retries
            ):
                delay = base_delay * (2**attempt) + random.uniform(0, 1)
                logger.warning(
                    "HF 412 conflict (attempt %d/%d), retrying in %.1fs...",
                    attempt + 1,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
            else:
                raise


def upload_file_to_dataset_repo(
    *,
    local_path: Path,
    repo_id: str,
    path_in_repo: str,
    commit_message: str,
) -> str:
    """Upload a local file to a dataset repo on Hugging Face Hub."""
    if not local_path.exists():
        raise FileNotFoundError(f"Local file not found: {local_path}")

    _configure_timeout()
    api = HfApi(token=_get_token())
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    _retry_on_conflict(
        lambda: api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
        )
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def upload_folder_to_dataset_repo(
    *,
    local_dir: Path,
    repo_id: str,
    path_in_repo: str,
    commit_message: str,
    ignore_patterns: list[str] | None = None,
    allow_patterns: list[str] | None = None,
    delete_patterns: list[str] | None = None,
) -> str:
    """Upload a local folder to a dataset repo on Hugging Face Hub.

    Args:
        local_dir: Local directory to upload.
        repo_id: HuggingFace dataset repo ID (``org/name``).
        path_in_repo: Destination path within the repo.
        commit_message: Commit message for the upload.
        ignore_patterns: Optional glob patterns to exclude (forwarded to
            ``HfApi.upload_folder``).
        allow_patterns: Optional glob patterns to include (forwarded to
            ``HfApi.upload_folder``). Only matching files are uploaded.
        delete_patterns: Optional glob patterns for files to delete from the
            remote repo in the same commit.  Files that are also being uploaded
            are NOT deleted (the HF API ignores them in that case).

    Returns:
        URL of the uploaded dataset repo.
    """
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    _configure_timeout()
    api = HfApi(token=_get_token())
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    _retry_on_conflict(
        lambda: api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
            ignore_patterns=ignore_patterns,
            allow_patterns=allow_patterns,
            delete_patterns=delete_patterns,
        )
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def upload_folder_to_model_repo(
    *,
    local_dir: Path,
    repo_id: str,
    path_in_repo: str,
    commit_message: str,
) -> str:
    """Upload a local folder (e.g. LoRA adapter) to a model repo on Hugging Face Hub."""
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")

    _configure_timeout()
    api = HfApi(token=_get_token())
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
    _retry_on_conflict(
        lambda: api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
        )
    )
    return f"https://huggingface.co/{repo_id}"


def check_exists_in_dataset_repo(
    *,
    repo_id: str,
    path_in_repo: str,
) -> bool:
    """Check if a path (file or directory) exists in a HF dataset repo without downloading.

    Args:
        repo_id: HuggingFace dataset repo ID (``org/name``).
        path_in_repo: Path within the repo to check for.

    Returns:
        True if the path exists (as a file or non-empty directory), False otherwise.
    """
    try:
        _configure_timeout()
    except Exception:
        pass
    api = HfApi(token=_get_token())
    # Try as a directory first (list children)
    try:
        files = list(
            api.list_repo_tree(
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=path_in_repo,
            )
        )
        return len(files) > 0
    except Exception:
        pass
    # list_repo_tree raises 404 for leaf files; check the parent directory
    try:
        from pathlib import PurePosixPath

        parent = str(PurePosixPath(path_in_repo).parent)
        if parent == ".":
            parent = ""
        for entry in api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=parent,
        ):
            if entry.path == path_in_repo:
                return True
    except Exception:
        pass
    return False


def dataset_repo_subpath_exists(
    *,
    repo_id: str,
    path_in_repo: str,
) -> bool:
    """Compatibility alias for checking whether a dataset-repo subpath exists."""
    return check_exists_in_dataset_repo(repo_id=repo_id, path_in_repo=path_in_repo)


def download_from_dataset_repo(
    *,
    repo_id: str,
    path_in_repo: str,
    local_dir: Path,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Download files from a dataset repo on Hugging Face Hub.

    Enumerates the subtree via ``list_repo_tree(path_in_repo=..., recursive=True)``
    and fetches each file via ``hf_hub_download``. We avoid ``snapshot_download``
    because for large repos (e.g. the monorepo) ``repo_info().siblings`` can be
    a truncated listing while still below the 50k threshold that would trigger
    the library's ``list_repo_tree`` fallback — the result is a silent 0-match
    for paths whose files happen to be missing from ``siblings``.

    Args:
        repo_id: HuggingFace dataset repo ID (``org/name``).
        path_in_repo: Prefix path within the repo to download from.
        local_dir: Local directory to download into. The repo structure
            under ``path_in_repo`` is replicated here (i.e. files land at
            ``local_dir/path_in_repo/...``).
        allow_patterns: Glob patterns *relative to path_in_repo* for files
            to download.  E.g. ``["rollouts/rollouts.jsonl"]``.
            If ``None``, all files under ``path_in_repo`` are downloaded.

    Returns:
        The local_dir path.
    """
    _configure_timeout()
    token = _get_token()
    api = HfApi(token=token)

    try:
        repo_files: list[str] = [
            entry.path
            for entry in api.list_repo_tree(
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=path_in_repo,
                recursive=True,
            )
            if isinstance(entry, RepoFile)
        ]
    except EntryNotFoundError:
        logger.info(
            "download_from_dataset_repo: path %s not found on %s",
            path_in_repo,
            repo_id,
        )
        return local_dir

    if allow_patterns is not None:
        prefix = f"{path_in_repo.rstrip('/')}/"

        def _matches(repo_path: str) -> bool:
            if not repo_path.startswith(prefix):
                return False
            rel = repo_path[len(prefix) :]
            return any(fnmatch(rel, p) for p in allow_patterns)

        repo_files = [f for f in repo_files if _matches(f)]

    if not repo_files:
        logger.warning(
            "download_from_dataset_repo: no files matched under %s (allow_patterns=%r)",
            path_in_repo,
            allow_patterns,
        )
        return local_dir

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    for repo_path in repo_files:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=repo_path,
            local_dir=str(local_dir),
            token=token,
        )

    return local_dir


def download_dataset_subpath(
    *,
    repo_id: str,
    path_in_repo: str,
    local_dir: Path,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Download a dataset-repo subpath and return its exact local path."""
    download_from_dataset_repo(
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        local_dir=local_dir,
        allow_patterns=allow_patterns,
    )
    downloaded_path = local_dir / path_in_repo
    if not downloaded_path.exists():
        raise FileNotFoundError(
            f"Downloaded dataset subpath not found locally: {downloaded_path}"
        )
    return downloaded_path


def download_path_to_dir(
    *,
    repo_id: str,
    path_in_repo: str,
    target_dir: Path,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Download a subtree from a dataset repo directly into target_dir.

    Unlike ``download_from_dataset_repo``, which replicates the full repo path
    structure under ``local_dir``, this function strips the ``path_in_repo``
    prefix so that files from ``{path_in_repo}/foo`` land at ``target_dir/foo``.

    Useful for rehydrating a run directory from HF into the exact local path
    it was originally written to (e.g. to regenerate plots without re-running).

    Args:
        repo_id: HuggingFace dataset repo ID (``org/name``).
        path_in_repo: Subtree within the repo to download.
        target_dir: Local directory to place the downloaded content in.
            Created if it does not exist.
        allow_patterns: Optional glob patterns *relative to path_in_repo*.
            If None, all files under path_in_repo are downloaded.

    Returns:
        target_dir.
    """
    import shutil
    import tempfile

    _configure_timeout()
    token = _get_token()

    # huggingface_hub 0.36.x's snapshot_download + allow_patterns returns
    # zero files for patterns with long path prefixes against this repo, so
    # we enumerate the subtree ourselves and pull each file with
    # hf_hub_download (which works reliably).
    api = HfApi(token=token)
    all_paths: list[str] = []
    stack: list[str] = [path_in_repo]
    while stack:
        current = stack.pop()
        for entry in api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=current,
            recursive=False,
        ):
            if isinstance(entry, RepoFile):
                all_paths.append(entry.path)
            else:
                stack.append(entry.path)

    if allow_patterns is not None:

        def _matches(rel: str) -> bool:
            return any(fnmatch(rel, pat) for pat in allow_patterns)

        all_paths = [p for p in all_paths if _matches(p[len(path_in_repo) + 1 :])]

    target_dir = Path(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target_dir.with_name(target_dir.name + ".tmp")
    if tmp_target.exists():
        shutil.rmtree(tmp_target)

    with tempfile.TemporaryDirectory() as staging:
        for p in all_paths:
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=p,
                local_dir=staging,
                token=token,
            )
        src = Path(staging) / path_in_repo
        if src.exists():
            shutil.copytree(src, tmp_target)
        else:
            tmp_target.mkdir(parents=True, exist_ok=True)

    if target_dir.exists():
        shutil.rmtree(target_dir)
    tmp_target.rename(target_dir)

    return target_dir


def download_file_from_dataset_repo(
    *,
    repo_id: str,
    path_in_repo: str,
    local_dir: Path,
) -> Path:
    """Download a single file from a dataset repo and return its local path."""
    download_from_dataset_repo(
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        local_dir=local_dir,
        allow_patterns=[Path(path_in_repo).name],
    )
    downloaded_path = local_dir / path_in_repo
    if not downloaded_path.is_file():
        raise FileNotFoundError(
            f"Downloaded dataset file not found locally: {downloaded_path}"
        )
    return downloaded_path
