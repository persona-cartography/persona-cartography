"""Pinned runtime checkout for the upstream Assistant Axis code.

The Assistant Axis repository is used as an external research dependency for
its pipeline scripts, data files, and ``assistant_axis.steering`` module. Keep
it out of this repo's git history; fetch the pinned source into scratch or
point ``ASSISTANT_AXIS_DIR`` at an existing checkout.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ASSISTANT_AXIS_REPO_URL = "https://github.com/safety-research/assistant-axis.git"
ASSISTANT_AXIS_COMMIT = "a98961956072224eaf244eb289d6c01700b63795"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSISTANT_AXIS_DIR = REPO_ROOT / "scratch" / "external" / "assistant_axis"


def assistant_axis_source_dir() -> Path:
    """Return the configured local checkout path without downloading it."""
    override = os.environ.get("ASSISTANT_AXIS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_ASSISTANT_AXIS_DIR


def ensure_assistant_axis_repo(*, quiet: bool = False) -> Path:
    """Ensure the pinned Assistant Axis source exists locally.

    Returns:
        Path to the local checkout root.

    Raises:
        RuntimeError: if ``ASSISTANT_AXIS_DIR`` points at an invalid checkout
            or if the pinned checkout cannot be cloned/fetched.
    """
    target = assistant_axis_source_dir()
    if os.environ.get("ASSISTANT_AXIS_DIR"):
        _validate_checkout(target)
        return target

    if target.exists() and not (target / ".git").exists():
        _validate_checkout(target)
        return target

    if not (target / ".git").exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        _run_git(["clone", ASSISTANT_AXIS_REPO_URL, str(target)], quiet=quiet)

    _run_git(["fetch", "--tags", "origin"], cwd=target, quiet=quiet)
    _run_git(["checkout", "--detach", "--force", ASSISTANT_AXIS_COMMIT], cwd=target, quiet=quiet)
    _validate_checkout(target)
    return target


def assistant_axis_source_label() -> str:
    """Human-readable upstream source pin for run metadata."""
    return f"{ASSISTANT_AXIS_COMMIT}  {ASSISTANT_AXIS_REPO_URL}"


def _validate_checkout(path: Path) -> None:
    required = [
        path / "assistant_axis" / "steering.py",
        path / "pipeline" / "1_generate.py",
        path / "pipeline" / "5_axis.py",
        path / "data" / "extraction_questions.jsonl",
        path / "data" / "roles" / "instructions" / "default.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(
            "Assistant Axis checkout is missing required files. "
            f"Set ASSISTANT_AXIS_DIR to a valid checkout or unset it to auto-clone. Missing: {missing}"
        )


def _run_git(
    args: list[str],
    *,
    cwd: Path | None = None,
    quiet: bool,
) -> None:
    cmd = ["git", *args]
    if quiet:
        cmd.insert(1, "-c")
        cmd.insert(2, "advice.detachedHead=false")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.DEVNULL if quiet else None,
        stderr=subprocess.DEVNULL if quiet else None,
        check=False,
    )
    if proc.returncode != 0:
        where = f" in {cwd}" if cwd else ""
        raise RuntimeError(f"git command failed{where}: {' '.join(cmd)}")


__all__ = [
    "ASSISTANT_AXIS_COMMIT",
    "ASSISTANT_AXIS_REPO_URL",
    "DEFAULT_ASSISTANT_AXIS_DIR",
    "assistant_axis_source_dir",
    "assistant_axis_source_label",
    "ensure_assistant_axis_repo",
]
