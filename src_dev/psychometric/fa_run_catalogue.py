"""Canonical registry of psychometric-FA runs on the shared dataset repo.

Single source of truth for the *specific* rollout / questionnaire runs that
analysis, validation and paper-figure scripts depend on. Import a named entry
from here rather than hard-coding a ``runs/<model>/<run-id>`` link (or the bare
run id) in an experiment script — the same role :mod:`src_dev.common.lora_catalogue`
plays for LoRA adapters.

Usage::

    from src_dev.psychometric.fa_run_catalogue import FA_RUN_REGISTRY

    run = FA_RUN_REGISTRY["llama_q_v7_fc_pair_pf3"]
    run.hf_path        # "runs/llama-3.1-8b/questionnaire-rollouts-…-pf3"
    run.run_id         # bare id, e.g. for building a scratch dir name
    run.scratch_dir()  # scratch/psychometric_fa/<run_id>

When a newer/better run supersedes an old one, update the entry here; every
downstream script then points at the new run with no further edits.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src_dev.psychometric.hf_paths import hf_runs_path, model_name_to_slug

# Default local root the FA pipeline hydrates runs into.
DEFAULT_SCRATCH_ROOT = Path("scratch/psychometric_fa")

# Subject models, spelled once so entries stay consistent.
_LLAMA_31_8B = "meta-llama/Llama-3.1-8B-Instruct"


@dataclass(frozen=True)
class FaRun:
    """One canonical run on ``persona-shattering-lasr/psychometric-fa-runs``."""

    key: str
    """Short identifier used as the registry key, e.g. ``"llama_q_v5_likert"``."""
    run_id: str
    """The run directory name (without the ``runs/<model>/`` prefix)."""
    model: str
    """Full id of the rollout *subject* model (decides the per-model folder)."""
    description: str
    """One-line human description of what the run is."""

    @property
    def model_slug(self) -> str:
        """Per-model folder slug, e.g. ``"llama-3.1-8b"``."""
        return model_name_to_slug(self.model)

    @property
    def hf_path(self) -> str:
        """Full Hub path: ``runs/<model-slug>/<run_id>``."""
        return hf_runs_path(self.run_id, model_slug=self.model_slug)

    def hf_subpath(self, *subpaths: str) -> str:
        """Hub path for a subdir of the run, e.g. ``hf_subpath("questionnaire")``."""
        return hf_runs_path(self.run_id, *subpaths, model_slug=self.model_slug)

    def scratch_dir(self, root: Path = DEFAULT_SCRATCH_ROOT) -> Path:
        """Local hydration directory for this run (``<root>/<run_id>``)."""
        return root / self.run_id


# ── Canonical runs ───────────────────────────────────────────────────────────
# The "B" rollout preset (Llama-3.1-8B, 2500 personas, seed 436, scenarios v2,
# uprompt v6) and the questionnaire variants fit on top of it. These are the
# runs the paper's unsupervised analysis is built from.

FA_RUN_REGISTRY: dict[str, FaRun] = {
    "llama_base_rollouts": FaRun(
        key="llama_base_rollouts",
        run_id="rollouts-llama318binstruct-t1.0-15t-2500p-seed436-scenarios_v2-uprompt_v6",
        model=_LLAMA_31_8B,
        description="Base B rollouts (no questionnaire) — shared rollout cache for all q-variants.",
    ),
    "llama_q_v5_likert": FaRun(
        key="llama_q_v5_likert",
        run_id=(
            "questionnaire-rollouts-llama318binstruct-t1.0-15t-2500p-seed436-"
            "scenarios_v2-uprompt_v6-q_v5-likert-direct-lp20"
        ),
        model=_LLAMA_31_8B,
        description="B rollouts scored on the v5 Likert questionnaire (logprob, lp20).",
    ),
    "llama_q_trait_ocean_natural_mcq": FaRun(
        key="llama_q_trait_ocean_natural_mcq",
        run_id=(
            "questionnaire-rollouts-llama318binstruct-t1.0-15t-2500p-seed436-"
            "scenarios_v2-uprompt_v6-q_trait_ocean_natural_v1-trait_mcq-aside-lp20-p2-pf2-tmv2"
        ),
        model=_LLAMA_31_8B,
        description="B rollouts scored on the trait_ocean_natural_v1 MCQ questionnaire.",
    ),
    "llama_q_v7_fc_pair_pf3": FaRun(
        key="llama_q_v7_fc_pair_pf3",
        run_id=(
            "questionnaire-rollouts-llama318binstruct-t1.0-15t-2500p-seed436-"
            "scenarios_v2-uprompt_v6-q_v7_fc_pair-fc_pair-direct-lp20-p2-pf3"
        ),
        model=_LLAMA_31_8B,
        description="B rollouts scored on the v7 forced-choice-pair questionnaire (pf3). Paper FA run.",
    ),
    "llama_q_v7_fc_pair_pf3_qwen_qm": FaRun(
        key="llama_q_v7_fc_pair_pf3_qwen_qm",
        run_id=(
            "questionnaire-rollouts-llama318binstruct-t1.0-15t-2500p-seed436-"
            "scenarios_v2-uprompt_v6-q_v7_fc_pair-fc_pair-direct-lp20-p2-pf3-qm_qwen257binstruct"
        ),
        model=_LLAMA_31_8B,
        description="As llama_q_v7_fc_pair_pf3 but with Qwen2.5-7B as the question model (qm).",
    ),
}
