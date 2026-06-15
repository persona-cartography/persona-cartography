"""Variance-decomp on Llama k=4 factor scores from inspect_factor_loadings.py.

Reuses ``scratch/factor_inspect/fa_fit.npz`` (the factor scores) and pulls
the rollout metadata (canonical_samples.jsonl + archetype_assignments.json)
from HF to attach archetype + scenario_id per persona, then reports one-way
η² for each factor.

Run after ``inspect_factor_loadings.py``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

SEED = 436
random.seed(SEED)
np.random.seed(SEED)

from src_dev.factor_analysis.interpretation import prompt_effects
from src_dev.psychometric.hf_paths import hf_runs_path
from src_dev.unsupervised_runs.io import hydrate_dataset_subtree

HF_REPO_ID = "persona-shattering-lasr/psychometric-fa-runs"
ROLLOUT_HF_PATH = hf_runs_path(
    "rollouts-llama318binstruct-t1.0-15t-2500p-seed436-"
    "scenarios_v2-uprompt_v6"
)
ROLLOUT_LOCAL = Path("scratch/factor_inspect/hydrated") / Path(ROLLOUT_HF_PATH).name
SCENARIOS = Path("datasets/scenarios/v2.json")
QUESTIONNAIRE_DIR = (
    Path("scratch/factor_inspect/hydrated/")
    / "questionnaire-rollouts-llama318binstruct-t1.0-15t-2500p-seed436-"
      "scenarios_v2-uprompt_v6-q_v5-likert-direct-lp20"
    / "questionnaire"
)


def main() -> None:
    # Hydrate rollout dir.
    if not (ROLLOUT_LOCAL / "archetype_assignments.json").exists():
        print(f"hydrating {ROLLOUT_HF_PATH} -> {ROLLOUT_LOCAL}")
        hydrate_dataset_subtree(
            repo_id=HF_REPO_ID,
            path_in_repo=ROLLOUT_HF_PATH,
            local_dir=ROLLOUT_LOCAL,
            required=True,
        )

    arch = json.loads((ROLLOUT_LOCAL / "archetype_assignments.json").read_text())
    scen_data = json.loads(SCENARIOS.read_text())
    prompt_to_scenario = {sc["target_system_prompt"]: sc["id"] for sc in scen_data["scenarios"]}

    # Build sample_id -> {archetype, scenario_id}
    lookup: dict[str, dict] = {}
    with (ROLLOUT_LOCAL / "datasets/canonical_samples.jsonl").open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            sid = rec["sample_id"]
            row_idx = rec["source_info"]["row_index"]
            sys_prompt = rec["input"]["messages"][0]["content"]
            scen_id = prompt_to_scenario.get(sys_prompt)
            archetype = arch.get(str(row_idx))
            if scen_id is None or archetype is None:
                continue
            lookup[sid] = {
                "archetype": archetype,
                "scenario_id": scen_id,
            }
    print(f"resolved {len(lookup)} sample_ids")

    # Pull metadata of the FA fit (sample_id ordered).
    meta_q = [json.loads(l) for l in (QUESTIONNAIRE_DIR / "metadata.jsonl").read_text().splitlines() if l.strip()]
    print(f"questionnaire metadata: {len(meta_q)} rows")

    # The FA was fit on the row-aligned combined matrix (intersection of
    # sample_ids across both questionnaires) — same set as questionnaire 0
    # since both have all 2500.
    fit = np.load(Path("scratch/factor_inspect/fa_fit.npz"), allow_pickle=True)
    scores = fit["scores"]
    print(f"fa scores: {scores.shape}")

    # Annotate metadata. The FA preprocessing didn't drop any rows here.
    enriched = []
    for r in meta_q:
        hit = lookup.get(r["sample_id"])
        if hit is None:
            continue
        enriched.append({**r, **hit})
    n_use = len(enriched)
    if n_use < scores.shape[0]:
        # If preprocess dropped anything, slice.
        keep_set = {r["sample_id"] for r in enriched}
        keep_idx = np.array([i for i, r in enumerate(meta_q) if r["sample_id"] in keep_set])
        scores = scores[keep_idx]
    print(f"using {n_use} resolved rows")

    eta_arch = prompt_effects(scores, enriched, group_field="archetype").tolist()
    eta_scen = prompt_effects(scores, enriched, group_field="scenario_id").tolist()
    print()
    print("one-way η² per factor:")
    for i, (ea, es) in enumerate(zip(eta_arch, eta_scen)):
        print(f"  F{i}:  archetype={ea:.3f}   scenario={es:.3f}   resid≈{1 - ea - es:.3f}")


if __name__ == "__main__":
    main()
