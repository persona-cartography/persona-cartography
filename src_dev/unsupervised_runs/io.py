"""Pathing and Hugging Face Hub helpers for unsupervised embedding workflows."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

from src_dev.utils.hf_hub import login_from_env, upload_folder_to_dataset_repo


DEFAULT_UNSUPERVISED_HF_REPO_ID = "persona-shattering-lasr/unsupervised-runs"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_RUNS_DIR = PROJECT_ROOT / "scratch" / "runs"


def slugify_component(text: str) -> str:
    """Return a filesystem- and repo-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "artifact"


def response_run_dir(response_run_id: str) -> Path:
    """Return the canonical local run directory for a response run."""
    return SCRATCH_RUNS_DIR / response_run_id


def response_run_hf_path(response_run_id: str) -> str:
    """Return the Hub path that stores a full canonical response run."""
    return f"runs/{response_run_id}/run"


def embedding_artifact_dir(run_dir: str | Path, embedding_slug: str) -> Path:
    """Return the local directory for one derived embedding artifact."""
    return Path(run_dir) / "reports" / "embeddings" / embedding_slug


def embedding_artifact_hf_path(response_run_id: str, embedding_slug: str) -> str:
    """Return the Hub path for one derived embedding artifact."""
    return f"runs/{response_run_id}/embeddings/{embedding_slug}"


def visualisation_artifact_dir(run_dir: str | Path, visualisation_slug: str) -> Path:
    """Return the local directory for one derived visualisation artifact."""
    return Path(run_dir) / "reports" / "visualisations" / visualisation_slug


def visualisation_artifact_hf_path(response_run_id: str, visualisation_slug: str) -> str:
    """Return the Hub path for one derived visualisation artifact."""
    return f"runs/{response_run_id}/visualisations/{visualisation_slug}"


def build_embedding_slug(
    *,
    model: str,
    analysis_unit: str,
    normalize: bool,
    max_length: int,
    target_variant: str | None = None,
) -> str:
    """Build a deterministic slug for one embedding derivation."""
    parts = [
        slugify_component(model),
        slugify_component(analysis_unit),
        f"len{int(max_length)}",
        "norm" if normalize else "unnorm",
    ]
    if target_variant:
        parts.append(slugify_component(target_variant))
    return "__".join(parts)


def build_visualisation_slug(
    *,
    label: str | None,
    response_run_ids: list[str],
    embedding_slugs: list[str],
) -> str:
    """Build a deterministic slug for one visualisation bundle."""
    if label:
        return slugify_component(label)

    run_part = "multi-run" if len(set(response_run_ids)) > 1 else response_run_ids[0]
    embed_part = "multi-embed" if len(set(embedding_slugs)) > 1 else embedding_slugs[0]
    return "__".join(
        [
            slugify_component(run_part),
            slugify_component(embed_part),
            "visualisation",
        ]
    )


def resolve_embedding_artifact_paths(
    run_dir: str | Path,
    embedding_slug: str,
    *,
    output_prefix: str = "response_embeddings",
) -> dict[str, Path]:
    """Return file paths for one embedding artifact directory."""
    artifact_dir = embedding_artifact_dir(run_dir, embedding_slug)
    return {
        "artifact_dir": artifact_dir,
        "metadata": artifact_dir / f"{output_prefix}_metadata.jsonl",
        "embeddings": artifact_dir / f"{output_prefix}_embeddings.npy",
        "variance": artifact_dir / f"{output_prefix}_variance.json",
        "manifest": artifact_dir / f"{output_prefix}_manifest.json",
    }


def _copy_repo_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def hydrate_dataset_subtree(
    *,
    repo_id: str,
    path_in_repo: str,
    local_dir: Path,
    required: bool = False,
) -> bool:
    """Mirror a dataset-repo subtree into a deterministic local directory."""
    # List only the requested subtree — a full list_repo_files() takes minutes
    # on large repos (e.g. the monorepo), while a scoped tree listing is fast.
    try:
        matching_files = [
            entry.path
            for entry in HfApi().list_repo_tree(
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=path_in_repo.rstrip("/"),
                recursive=True,
            )
            if type(entry).__name__ == "RepoFile"
        ]
    except EntryNotFoundError:
        matching_files = []
    if not matching_files:
        if required:
            raise FileNotFoundError(
                f"No files found in dataset repo '{repo_id}' under '{path_in_repo}'."
            )
        return False

    for repo_file in matching_files:
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=repo_file,
            )
        )
        relative_path = Path(repo_file).relative_to(path_in_repo)
        _copy_repo_file(downloaded, local_dir / relative_path)
    return True


def ensure_response_run(
    response_run_id: str,
    *,
    repo_id: str = DEFAULT_UNSUPERVISED_HF_REPO_ID,
    required: bool = False,
) -> Path:
    """Ensure a canonical response run exists locally, hydrating from Hub if needed."""
    run_dir = response_run_dir(response_run_id)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        return run_dir

    hydrate_dataset_subtree(
        repo_id=repo_id,
        path_in_repo=response_run_hf_path(response_run_id),
        local_dir=run_dir,
        required=required,
    )
    return run_dir


def ensure_embedding_artifact(
    response_run_id: str,
    embedding_slug: str,
    *,
    repo_id: str = DEFAULT_UNSUPERVISED_HF_REPO_ID,
    required: bool = False,
) -> Path:
    """Ensure one derived embedding artifact exists locally, hydrating from Hub if needed."""
    run_dir = ensure_response_run(response_run_id, repo_id=repo_id, required=required)
    artifact_dir = embedding_artifact_dir(run_dir, embedding_slug)
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        return artifact_dir

    hydrate_dataset_subtree(
        repo_id=repo_id,
        path_in_repo=embedding_artifact_hf_path(response_run_id, embedding_slug),
        local_dir=artifact_dir,
        required=required,
    )
    return artifact_dir


def upload_response_run(
    response_run_id: str,
    *,
    repo_id: str = DEFAULT_UNSUPERVISED_HF_REPO_ID,
    commit_message: str | None = None,
) -> str:
    """Upload the full canonical response run to the shared dataset repo."""
    run_dir = response_run_dir(response_run_id)
    login_from_env()
    return upload_folder_to_dataset_repo(
        local_dir=run_dir,
        repo_id=repo_id,
        path_in_repo=response_run_hf_path(response_run_id),
        commit_message=commit_message or f"Upload response run {response_run_id}",
    )


def upload_embedding_artifact(
    response_run_id: str,
    embedding_slug: str,
    *,
    repo_id: str = DEFAULT_UNSUPERVISED_HF_REPO_ID,
    commit_message: str | None = None,
) -> str:
    """Upload one derived embedding artifact to the shared dataset repo."""
    run_dir = response_run_dir(response_run_id)
    artifact_dir = embedding_artifact_dir(run_dir, embedding_slug)
    login_from_env()
    return upload_folder_to_dataset_repo(
        local_dir=artifact_dir,
        repo_id=repo_id,
        path_in_repo=embedding_artifact_hf_path(response_run_id, embedding_slug),
        commit_message=commit_message or (
            f"Upload embedding artifact {embedding_slug} for run {response_run_id}"
        ),
    )


def upload_visualisation_artifact(
    response_run_id: str,
    visualisation_slug: str,
    *,
    repo_id: str = DEFAULT_UNSUPERVISED_HF_REPO_ID,
    commit_message: str | None = None,
) -> str:
    """Upload one derived visualisation artifact to the shared dataset repo."""
    run_dir = response_run_dir(response_run_id)
    artifact_dir = visualisation_artifact_dir(run_dir, visualisation_slug)
    login_from_env()
    return upload_folder_to_dataset_repo(
        local_dir=artifact_dir,
        repo_id=repo_id,
        path_in_repo=visualisation_artifact_hf_path(response_run_id, visualisation_slug),
        commit_message=commit_message or (
            f"Upload visualisation artifact {visualisation_slug} for run {response_run_id}"
        ),
    )
