"""Stage 2 — questionnaire administration.

Cache-aware wrapper around :func:`src_dev.psychometric.questionnaire_inference.
run_questionnaire_inference`. Responsible for:

* Loading the questionnaire JSON.
* Checking the local cache, then the HF cache (hydrates if present).
* Running inference only when no cache resolves.
* Uploading the result folder to HF.
* Handling the encoding-version check: if the cached response matrix was
  produced under an older encoding, rebuild from ``raw_responses.jsonl``
  (no re-inference) rather than replaying the full questionnaire.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import numpy as np

from src_dev.psychometric.config import (
    QuestionnaireStageConfig,
    QuestionnaireStageResult,
)
from src_dev.psychometric.hf_paths import hf_runs_path
from src_dev.psychometric.questionnaire_inference import (
    run_questionnaire_inference_async,
)
from src_dev.psychometric.questionnaire_io import load_questionnaire
from src_dev.psychometric.response_encoding import RESPONSE_MATRIX_ENCODING_VERSION
from src_dev.unsupervised_runs.io import hydrate_dataset_subtree
from src_dev.utils.hf_hub import (
    check_exists_in_dataset_repo,
    login_from_env,
    upload_folder_to_dataset_repo,
)

logger = logging.getLogger(__name__)


def run_stage_questionnaire(
    cfg: QuestionnaireStageConfig,
    *,
    num_conversation_turns: int,
    openrouter_provider_routing: dict | None = None,
    fc_pair_sign_alignment: bool = True,
) -> QuestionnaireStageResult:
    """Apply the questionnaire to all rollout personas.

    Args:
        cfg: Questionnaire stage config.
        num_conversation_turns: Completeness threshold from the rollout
            preset (samples with fewer assistant turns are dropped).
        openrouter_provider_routing: Passed through to ``InferenceConfig.
            openrouter.provider_routing``. None uses ``{}``.
        fc_pair_sign_alignment: Passed through to the inference loop; must
            match the value ``load_questionnaire`` was called with.
    """
    rollout_dir = cfg.ctx.rollout_dir
    output_dir = cfg.ctx.questionnaire_dir / "questionnaire"
    run_id = cfg.ctx.questionnaire_run_id
    hf_repo_id = cfg.ctx.hf_repo_id

    items, column_defs = load_questionnaire(
        cfg.questionnaire_path,
        fa_blocks=cfg.fa_blocks,
        fc_pair_sign_alignment=fc_pair_sign_alignment,
    )

    def _load_from_dir() -> tuple[np.ndarray, list[dict], list[dict]] | None:
        matrix_path = output_dir / "response_matrix.npy"
        items_path = output_dir / "items.json"
        enc_ver_path = output_dir / "encoding_version.json"
        if not matrix_path.exists():
            return None

        # Encoding-version check: if the saved matrix was produced under an
        # older encoding (marker absent or version mismatched), ignore it and
        # fall through. Regeneration will replay raw_responses.jsonl through
        # the current encoding without re-inference.
        saved_enc_version: int | None = None
        if enc_ver_path.exists():
            try:
                with open(enc_ver_path, "r") as f:
                    saved_enc_version = int(
                        json.load(f).get("response_matrix_encoding_version")
                    )
            except (ValueError, TypeError, json.JSONDecodeError):
                saved_enc_version = None
        # Enforce the version check unconditionally: every bump of
        # RESPONSE_MATRIX_ENCODING_VERSION reflects a change in how at
        # least one item type is encoded, and the rebuild path reads
        # ``raw_responses.jsonl`` (the logprob-bearing source of truth)
        # without re-running inference, so the cost is small. We used to
        # gate this on the presence of a trait_mcq block because v2 only
        # changed trait_mcq, but v3 changed Likert logprob reverse-
        # keying, and any future bump may touch a different block still.
        if saved_enc_version != RESPONSE_MATRIX_ENCODING_VERSION:
            raw_log = output_dir / "raw_responses.jsonl"
            if raw_log.exists():
                print(
                    f"[Stage 2] Cached matrix at {output_dir} was produced under "
                    f"encoding v{saved_enc_version} (current v{RESPONSE_MATRIX_ENCODING_VERSION}) "
                    "— rebuilding from raw_responses.jsonl (no re-inference)."
                )
            else:
                print(
                    f"[Stage 2] Cached matrix at {output_dir} has stale encoding "
                    f"(v{saved_enc_version} vs current v{RESPONSE_MATRIX_ENCODING_VERSION}) "
                    "and no raw_responses.jsonl — will regenerate."
                )
            return None

        response_matrix = np.load(matrix_path)
        with open(output_dir / "metadata.jsonl", "r") as f:
            metadata = [json.loads(line) for line in f]
        if items_path.exists():
            with open(items_path, "r") as f:
                saved_cols = json.load(f)
            return response_matrix, metadata, saved_cols
        return response_matrix, metadata, column_defs

    try:
        login_from_env()
    except RuntimeError:
        logger.warning("HF_TOKEN not set — HF caching disabled.")
    hf_path = hf_runs_path(run_id, "questionnaire")

    def _result(
        response_matrix: np.ndarray,
        metadata: list[dict],
        resolved_cols: list[dict],
        *,
        hydrated: bool,
        generated: bool,
    ) -> QuestionnaireStageResult:
        return QuestionnaireStageResult(
            questionnaire_dir=output_dir,
            response_matrix_path=output_dir / "response_matrix.npy",
            metadata_path=output_dir / "metadata.jsonl",
            items_path=output_dir / "items.json",
            n_personas=int(response_matrix.shape[0]),
            n_items=len(resolved_cols),
            hydrated_from_hf=hydrated,
            generated=generated,
        )

    # Check local cache
    cached = _load_from_dir()
    if cached is not None:
        response_matrix, metadata, resolved_cols = cached
        # Backfill to HF if a previous upload was skipped or failed.
        if not check_exists_in_dataset_repo(
            repo_id=hf_repo_id, path_in_repo=hf_path + "/response_matrix.npy"
        ):
            print(f"[Stage 2] Local questionnaire found but not on HF — uploading now")
            try:
                upload_folder_to_dataset_repo(
                    local_dir=output_dir,
                    repo_id=hf_repo_id,
                    path_in_repo=hf_path,
                    commit_message=f"Questionnaire: {run_id}",
                    ignore_patterns=[],
                )
                print(f"[Stage 2] Uploaded to HF: {hf_path}")
            except Exception as e:
                logger.warning("Failed to upload questionnaire to HF: %s", e)
        print(f"[Stage 2] Questionnaire results already exist locally: {output_dir}")
        return _result(response_matrix, metadata, resolved_cols, hydrated=False, generated=False)

    # Check HF cache
    if check_exists_in_dataset_repo(repo_id=hf_repo_id, path_in_repo=hf_path):
        print(f"[Stage 2] Hydrating questionnaire results from HF: {run_id}")
        hydrate_dataset_subtree(
            repo_id=hf_repo_id,
            path_in_repo=hf_path,
            local_dir=output_dir,
        )
        cached = _load_from_dir()
        if cached is not None:
            response_matrix, metadata, resolved_cols = cached
            return _result(response_matrix, metadata, resolved_cols, hydrated=True, generated=False)
        print("[Stage 2] HF hydration incomplete, regenerating...")

    # Generate
    response_matrix, metadata = asyncio.run(
        run_questionnaire_inference_async(
            cfg,
            rollout_dir,
            items,
            column_defs,
            output_dir,
            num_conversation_turns=num_conversation_turns,
            openrouter_provider_routing=openrouter_provider_routing,
            fc_pair_sign_alignment=fc_pair_sign_alignment,
        )
    )

    # Upload to HF
    try:
        upload_folder_to_dataset_repo(
            local_dir=output_dir,
            repo_id=hf_repo_id,
            path_in_repo=hf_path,
            commit_message=f"Questionnaire: {run_id}",
            ignore_patterns=[],
        )
        print(f"[Stage 2] Uploaded to HF: {hf_path}")
    except Exception as e:
        logger.warning("Failed to upload questionnaire to HF: %s", e)

    return _result(response_matrix, metadata, column_defs, hydrated=False, generated=True)
