"""
Shared utilities for merging / unmerging data between canonical and local
leaderboard files.

Used by:
  - merge_results_to_canonical.py  (interactive merge / unmerge / delete)
  - eqbench3.py                    (--redo-elo flag)
"""

import os
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List

from core.elo import filter_comparisons_for_solver, models_in_comparisons
from core.trueskill_solver import solve_with_trueskill, normalize_elo_scores
from core.pairwise_judging import _recompute_comparison_stats
from core.elo_config import (
    DEFAULT_ELO,
    WIN_MARGIN_BIN_SIZE,
    WIN_MARGIN_BIN_SIZE_FOR_CI,
)
from utils.file_io import save_json_file


# =========================================================================
# Merge: local -> canonical
# =========================================================================

def merge_data(selected_candidates, local_runs, local_elo, canonical_runs, canonical_elo):
    """Moves selected run data and relevant comparisons from local to canonical files.

    Operates on the dicts **in-place**.  Returns True if data was moved.
    """
    if not selected_candidates:
        return False

    merged_model_names = set(c['model_name'] for c in selected_candidates)
    logging.info(f"Preparing to merge {len(merged_model_names)} models: {', '.join(merged_model_names)}")

    # Determine the final set of models that will be in the canonical ELO file
    final_canonical_models = set(k for k in canonical_elo if k != "__metadata__")
    final_canonical_models.update(merged_model_names)
    logging.info(f"Final canonical ELO file will contain {len(final_canonical_models)} models.")

    # --- Process Local ELO Comparisons ---
    local_comps = local_elo.get("__metadata__", {}).get("global_pairwise_comparisons", [])
    comps_to_move = []
    comps_to_keep_local = []
    moved_comp_count = 0

    for comp in local_comps:
        pair = comp.get("pair", {})
        model_a = pair.get("test_model")
        model_b = pair.get("neighbor_model")

        # Check if this comparison involves one of the models being merged
        comp_involves_merged_model = model_a in merged_model_names or model_b in merged_model_names

        if comp_involves_merged_model:
            # Check if BOTH models in the pair will be in the final canonical set
            if model_a in final_canonical_models and model_b in final_canonical_models:
                comps_to_move.append(comp)
                moved_comp_count += 1
                logging.debug(f"Moving comparison to canonical: {model_a} vs {model_b}")
            else:
                # Keep comparison locally if it involves a merged model but the other model isn't canonical
                comps_to_keep_local.append(comp)
                logging.debug(f"Keeping comparison locally (one model not canonical): {model_a} vs {model_b}")
        else:
            # Keep comparison locally if it doesn't involve any model being merged now
            comps_to_keep_local.append(comp)

    logging.info(f"Identified {moved_comp_count} comparisons to move to canonical ELO.")
    logging.info(f"{len(comps_to_keep_local)} comparisons will remain in local ELO.")

    # Update local ELO comparisons
    if "__metadata__" not in local_elo: local_elo["__metadata__"] = {}
    local_elo["__metadata__"]["global_pairwise_comparisons"] = comps_to_keep_local

    # Add comparisons to canonical ELO
    if "__metadata__" not in canonical_elo: canonical_elo["__metadata__"] = {}
    if "global_pairwise_comparisons" not in canonical_elo["__metadata__"]:
        canonical_elo["__metadata__"]["global_pairwise_comparisons"] = []
    canonical_elo["__metadata__"]["global_pairwise_comparisons"].extend(comps_to_move)

    # --- Move Run Data and ELO Entries ---
    moved_run_keys = set()
    for candidate in selected_candidates:
        model_name = candidate["model_name"]

        # Move run data (handle multiple runs for the same model)
        # Find all run keys associated with the model name in local_runs
        run_keys_for_model = [
            r_key for r_key, r_data in local_runs.items()
            if r_key != "__metadata__" and isinstance(r_data, dict) and
               r_data.get("model_name", r_data.get("test_model")) == model_name
        ]

        for r_key_to_move in run_keys_for_model:
            if r_key_to_move in local_runs and r_key_to_move not in moved_run_keys:
                canonical_runs[r_key_to_move] = local_runs[r_key_to_move]
                del local_runs[r_key_to_move]
                moved_run_keys.add(r_key_to_move)
                logging.info(f"Moved run data for {r_key_to_move} (model {model_name}) to canonical runs.")
            elif r_key_to_move in moved_run_keys:
                 logging.debug(f"Run key {r_key_to_move} already moved.")
            else:
                logging.warning(f"Run key {r_key_to_move} (for model {model_name}) not found in local runs data during merge.")

        # Move ELO entry (only once per model)
        if model_name in local_elo:
            if model_name != "__metadata__": # Safety check
                canonical_elo[model_name] = local_elo[model_name]
                del local_elo[model_name]
                logging.info(f"Moved ELO entry for {model_name} to canonical ELO.")
        else:
            logging.warning(f"Model name {model_name} not found in local ELO data during merge (might indicate an issue or already moved).")

    return True # Indicate merging occurred


# =========================================================================
# Unmerge: canonical -> local
# =========================================================================

def unmerge_data(selected_candidates: List[Dict[str, Any]],
                 local_runs: Dict[str, Any],
                 local_elo: Dict[str, Any],
                 canonical_runs: Dict[str, Any],
                 canonical_elo: Dict[str, Any]) -> bool:
    """Moves selected run data and relevant comparisons from canonical to local files.

    Operates on the dicts **in-place**.  Returns True if data was moved.
    """
    if not selected_candidates:
        logging.info("No candidates selected for unmerge. Nothing to do.")
        return False

    unmerged_model_names = set(c['model_name'] for c in selected_candidates)
    logging.info(f"Preparing to unmerge {len(unmerged_model_names)} models: {', '.join(sorted(list(unmerged_model_names)))}")

    # Determine the sets of models in each location *after* the move
    current_canonical_models = set(k for k in canonical_elo if k != "__metadata__")
    remaining_canonical_models = current_canonical_models - unmerged_model_names

    current_local_models = set(k for k in local_elo if k != "__metadata__")
    final_local_models = current_local_models | unmerged_model_names

    logging.info(f"Canonical ELO file will contain {len(remaining_canonical_models)} models after unmerge.")
    logging.info(f"Local ELO file will contain {len(final_local_models)} models after unmerge.")

    # --- Process Canonical ELO Comparisons ---
    canonical_comps = canonical_elo.get("__metadata__", {}).get("global_pairwise_comparisons", [])
    comps_to_move_to_local = []
    comps_to_keep_canonical = []
    moved_comp_count = 0
    kept_comp_count = 0

    logging.debug(f"Processing {len(canonical_comps)} canonical comparisons...")
    for comp in canonical_comps:
        pair = comp.get("pair", {})
        model_a = pair.get("test_model")
        model_b = pair.get("neighbor_model")

        if not model_a or not model_b:
            logging.warning(f"Skipping comparison with missing model names: {comp}")
            comps_to_keep_canonical.append(comp)
            kept_comp_count += 1
            continue

        model_a_stays_canonical = model_a in remaining_canonical_models
        model_b_stays_canonical = model_b in remaining_canonical_models

        if model_a_stays_canonical and model_b_stays_canonical:
            comps_to_keep_canonical.append(comp)
            kept_comp_count += 1
        else:
            comps_to_move_to_local.append(comp)
            moved_comp_count += 1

    logging.info(f"Identified {moved_comp_count} comparisons to move to local ELO.")
    logging.info(f"{kept_comp_count} comparisons will remain in canonical ELO.")

    if moved_comp_count == 0 and len(unmerged_model_names) > 0:
        logging.error(f"Attempted to unmerge {len(unmerged_model_names)} models, but found 0 relevant comparisons to move.")
        logging.error("This likely indicates an issue. Aborting unmerge to prevent data inconsistency.")
        logging.error(f"Models selected for unmerge: {unmerged_model_names}")
        logging.error(f"Models remaining canonical: {remaining_canonical_models}")
        return False

    # Update canonical ELO comparisons
    if "__metadata__" not in canonical_elo: canonical_elo["__metadata__"] = {}
    canonical_elo["__metadata__"]["global_pairwise_comparisons"] = comps_to_keep_canonical

    # Add comparisons to local ELO
    if "__metadata__" not in local_elo: local_elo["__metadata__"] = {}
    local_elo["__metadata__"].setdefault("global_pairwise_comparisons", []) \
            .extend(comps_to_move_to_local)
    logging.info(f"Added {len(comps_to_move_to_local)} comparisons to local ELO metadata.")

    # --- Move Run Data and ELO Entries ---
    moved_run_keys = set()
    for candidate in selected_candidates:
        model_name = candidate["model_name"]
        # Find *all* run keys associated with this model in canonical_runs
        run_keys_for_model = [
            r_key for r_key, r_data in canonical_runs.items()
            if r_key != "__metadata__" and isinstance(r_data, dict) and
               r_data.get("model_name", r_data.get("test_model")) == model_name
        ]

        if not run_keys_for_model:
             logging.warning(f"No run entries found in canonical runs for model '{model_name}' during unmerge.")

        # Move run data
        for r_key_to_move in run_keys_for_model:
            if r_key_to_move in canonical_runs and r_key_to_move not in moved_run_keys:
                # Check for collision in local_runs
                if r_key_to_move in local_runs:
                    logging.warning(f"Run key '{r_key_to_move}' already exists in local runs! Overwriting during unmerge for model '{model_name}'.")
                local_runs[r_key_to_move] = canonical_runs[r_key_to_move]
                del canonical_runs[r_key_to_move]
                moved_run_keys.add(r_key_to_move)
                logging.info(f"Moved run data for {r_key_to_move} (model {model_name}) to local runs.")
            elif r_key_to_move in moved_run_keys:
                 logging.debug(f"Run key {r_key_to_move} already processed.")

        # Move ELO entry
        if model_name in canonical_elo:
            if model_name != "__metadata__": # Safety check
                 # Check for collision in local_elo
                if model_name in local_elo:
                     logging.warning(f"ELO entry for model '{model_name}' already exists in local ELO! Overwriting during unmerge.")
                local_elo[model_name] = canonical_elo[model_name]
                del canonical_elo[model_name]
                logging.info(f"Moved ELO entry for {model_name} to local ELO.")
        else:
            logging.warning(f"Model name '{model_name}' not found in canonical ELO data during unmerge (might indicate inconsistency).")

    return True


# =========================================================================
# Purge Model Comparisons
# =========================================================================

def purge_model_comparisons(elo_data: Dict[str, Any], model_name: str) -> int:
    """Remove all pairwise comparisons involving *model_name* and its
    top-level ELO entry from *elo_data*.  Operates in-place.

    Returns the number of comparisons removed.
    """
    removed = 0

    # Remove top-level ELO entry
    if model_name in elo_data and model_name != "__metadata__":
        del elo_data[model_name]
        logging.info(f"[PURGE] Removed top-level ELO entry for '{model_name}'")

    # Remove comparisons
    meta = elo_data.get("__metadata__", {})
    comps = meta.get("global_pairwise_comparisons", [])
    filtered = [
        c for c in comps
        if model_name not in (
            c.get("pair", {}).get("test_model"),
            c.get("pair", {}).get("neighbor_model"),
        )
    ]
    removed = len(comps) - len(filtered)
    if removed > 0:
        meta["global_pairwise_comparisons"] = filtered
        logging.info(f"[PURGE] Removed {removed} comparisons involving '{model_name}'")
    else:
        logging.info(f"[PURGE] No comparisons found involving '{model_name}'")

    return removed


# =========================================================================
# ELO Recalculation
# =========================================================================

def _recalculate_elo_ratings(elo_data: Dict[str, Any]) -> bool:
    """
    Recalculate ELO ratings for the given ELO data structure (modifies in place).
    Uses the same comparison-filter pipeline as run_elo_analysis_eqbench3.
    Returns True on success, False on failure.
    """
    logging.info("Recalculating ELO ratings...")

    TS_DEFAULT_SIGMA = 350 / 3

    try:
        meta            = elo_data.get("__metadata__", {})
        all_comparisons = meta.get("global_pairwise_comparisons", [])
        all_models      = [m for m in elo_data if m != "__metadata__"]

        if not all_models:
            logging.warning("No models found in ELO data. Clearing ELO entries.")
            keys_to_del = [k for k in elo_data if k != "__metadata__"]
            for k in keys_to_del:
                del elo_data[k]
            if "global_pairwise_comparisons" in meta:
                 meta["global_pairwise_comparisons"] = []
            return True

        if not all_comparisons:
            logging.warning("No comparisons found; assigning default ELO to all models.")
            for m in all_models:
                elo_data[m] = {
                    "elo":      DEFAULT_ELO,
                    "elo_norm": DEFAULT_ELO,
                    "sigma":    TS_DEFAULT_SIGMA,
                    "ci_low":   DEFAULT_ELO - 1.96 * TS_DEFAULT_SIGMA,
                    "ci_high":  DEFAULT_ELO + 1.96 * TS_DEFAULT_SIGMA,
                    "ci_low_norm": DEFAULT_ELO - 1.96 * TS_DEFAULT_SIGMA,
                    "ci_high_norm": DEFAULT_ELO + 1.96 * TS_DEFAULT_SIGMA,
                }
            return True

        # --- make sure every record has fraction_for_test etc. -------------
        changed = 0
        for comp in all_comparisons:
            if "error" not in comp and "judge_response" in comp:
                before = comp.get("fraction_for_test")
                _recompute_comparison_stats(comp)
                if comp.get("fraction_for_test") != before:
                    changed += 1
        logging.info(f"Recomputed stats for {changed} comparisons")

        # ---------- identical pipeline to main ELO run ------------------
        comps_for_solver = filter_comparisons_for_solver(all_comparisons)
        logging.info(f"Kept {len(comps_for_solver)}/{len(all_comparisons)} valid comparisons for solver.")

        if not comps_for_solver:
            logging.warning("No valid comparisons for solver. Assigning default ELO to remaining models.")
            current_models = set(elo_data.keys()) - {"__metadata__"}
            for m in current_models:
                 elo_data[m] = {
                    "elo":      DEFAULT_ELO,
                    "elo_norm": DEFAULT_ELO,
                    "sigma":    TS_DEFAULT_SIGMA,
                    "ci_low":   DEFAULT_ELO - 1.96 * TS_DEFAULT_SIGMA,
                    "ci_high":  DEFAULT_ELO + 1.96 * TS_DEFAULT_SIGMA,
                    "ci_low_norm": DEFAULT_ELO - 1.96 * TS_DEFAULT_SIGMA,
                    "ci_high_norm": DEFAULT_ELO + 1.96 * TS_DEFAULT_SIGMA,
                 }
            return True

        # Models to solve for: union of models in blob and models in filtered comparisons
        models_in_blob = set(all_models)
        models_in_filtered_comps = models_in_comparisons(comps_for_solver)
        models_for_solver_set = models_in_blob | models_in_filtered_comps

        # Ensure we only calculate for models actually remaining in elo_data keys
        models_for_solver = sorted([m for m in models_for_solver_set if m in elo_data])

        if not models_for_solver:
             logging.warning("No models left to solve for after filtering. Clearing ELO data.")
             keys_to_del = [k for k in elo_data if k != "__metadata__"]
             for k in keys_to_del:
                 del elo_data[k]
             if "global_pairwise_comparisons" in meta:
                  meta["global_pairwise_comparisons"] = []
             return True

        logging.info(f"Solving ELO for {len(models_for_solver)} models.")

        mu_map, _ = solve_with_trueskill(
            models_for_solver,
            comps_for_solver,
            {m: DEFAULT_ELO for m in models_for_solver},
            debug=False,
            use_fixed_initial_ratings=True,
            bin_size=WIN_MARGIN_BIN_SIZE,
            return_sigma=True,
        )

        # Recalc again just for sigma, using smaller bin size
        _, sigma_map = solve_with_trueskill(
            models_for_solver,
            comps_for_solver,
            {m: DEFAULT_ELO for m in models_for_solver},
            debug=False,
            use_fixed_initial_ratings=True,
            bin_size=WIN_MARGIN_BIN_SIZE_FOR_CI,
            return_sigma=True,
        )

        mu_norm_map = normalize_elo_scores(mu_map)

        # ---------- write results back (in place) -----------------------
        all_remaining_models = set(elo_data.keys()) - {"__metadata__"}
        calculated_models = set(models_for_solver)

        for m in all_remaining_models:
            if m in calculated_models:
                mu_raw  = mu_map.get(m, DEFAULT_ELO)
                sigma   = sigma_map.get(m, TS_DEFAULT_SIGMA)
                ci_low  = mu_raw - 1.96 * sigma
                ci_high = mu_raw + 1.96 * sigma
                elo_data[m] = {
                    "elo":       round(mu_raw, 2),
                    "elo_norm":  round(mu_norm_map.get(m, DEFAULT_ELO), 2),
                    "sigma":     round(sigma, 2),
                    "ci_low":    round(ci_low, 2),
                    "ci_high":   round(ci_high, 2),
                }
            else:
                 logging.warning(f"Model '{m}' had no comparisons after filtering, assigning default ELO.")
                 elo_data[m] = {
                    "elo":      DEFAULT_ELO,
                    "elo_norm": DEFAULT_ELO,
                    "sigma":    TS_DEFAULT_SIGMA,
                    "ci_low":   DEFAULT_ELO - 1.96 * TS_DEFAULT_SIGMA,
                    "ci_high":  DEFAULT_ELO + 1.96 * TS_DEFAULT_SIGMA,
                    "ci_low_norm": DEFAULT_ELO - 1.96 * TS_DEFAULT_SIGMA,
                    "ci_high_norm": DEFAULT_ELO + 1.96 * TS_DEFAULT_SIGMA,
                 }

        # ---------- normalise CI bounds ---------------------------------
        models_with_valid_elo = {m for m, d in elo_data.items() if m != "__metadata__" and "elo" in d}

        if models_with_valid_elo:
            raw_plus_bounds = {}
            raw_plus_bounds.update({m: d["elo"] for m, d in elo_data.items() if m in models_with_valid_elo})
            raw_plus_bounds.update({f"{m}__low":  d["ci_low"]  for m, d in elo_data.items() if m in models_with_valid_elo})
            raw_plus_bounds.update({f"{m}__high": d["ci_high"] for m, d in elo_data.items() if m in models_with_valid_elo})

            if raw_plus_bounds:
                norm_bounds = normalize_elo_scores(raw_plus_bounds)
                for m, d in elo_data.items():
                    if m in models_with_valid_elo:
                        d["ci_low_norm"]  = round(norm_bounds.get(f"{m}__low",  d.get("elo_norm", DEFAULT_ELO)), 2)
                        d["ci_high_norm"] = round(norm_bounds.get(f"{m}__high", d.get("elo_norm", DEFAULT_ELO)), 2)
            else:
                 logging.warning("No valid ELO scores found to normalize CI bounds.")
        else:
             logging.warning("No models with valid ELO scores after recalculation.")

        # Update timestamp in metadata
        elo_data.setdefault("__metadata__", {})
        elo_data["__metadata__"]["last_updated"] = datetime.now(timezone.utc).isoformat()

        logging.info("ELO recalculation complete.")
        return True

    except Exception:
        logging.error("ELO recalculation failed", exc_info=True)
        return False


# =========================================================================
# Atomic Multi-File Save
# =========================================================================

def _atomic_multi_save(path_to_data: Dict[str, Any]) -> bool:
    """
    Transactionally write several JSON blobs. If anything fails,
    originals are untouched and all temp files are removed.
    """
    temps = {}
    success = True
    written_temps = []

    try:
        # ---- 1. write each temp file ---------------------------------------
        for final_path_str, data in path_to_data.items():
            final_path = Path(final_path_str)
            dir_name = final_path.parent
            stem = final_path.stem
            suffix = final_path.suffix

            # Handle double extensions like .json.gz correctly
            if suffix == ".gz" and stem.endswith(".json"):
                stem = stem[:-5]
                ext = ".json.gz"
            else:
                ext = suffix

            tmp_name = f"{stem}.tmp.{uuid.uuid4().hex}{ext}"
            tmp_path = dir_name / tmp_name

            dir_name.mkdir(parents=True, exist_ok=True)

            if not save_json_file(data, str(tmp_path)):
                logging.error(f"Failed to write temporary file: {tmp_path}")
                success = False
                if tmp_path.exists():
                    try: tmp_path.unlink()
                    except OSError as e: logging.warning(f"Could not remove failed temp file {tmp_path}: {e}")
                break
            else:
                temps[str(final_path)] = str(tmp_path)
                written_temps.append(str(tmp_path))

        # ---- 2. rename temps onto finals (only if all temps written) -----
        if success:
            try:
                for final_path, tmp_path in temps.items():
                    os.replace(tmp_path, final_path)
                    written_temps.remove(tmp_path)
                logging.debug("All temporary files successfully renamed.")
                return True
            except Exception as e:
                logging.error(f"Multi-save rename phase failed: {e}", exc_info=True)
                success = False

    except Exception as e:
        logging.error(f"Error during atomic save process (before rename): {e}", exc_info=True)
        success = False

    finally:
        # ---- Cleanup: Remove any remaining temp files ----
        if written_temps:
             logging.warning(f"Cleaning up {len(written_temps)} temporary files due to failed save.")
             for t_path_str in written_temps:
                 t_path = Path(t_path_str)
                 if t_path.exists():
                     try:
                         t_path.unlink()
                         logging.debug(f"Removed temp file: {t_path}")
                     except OSError as e:
                         logging.error(f"Failed to remove temporary file {t_path}: {e}")
                 else:
                      logging.warning(f"Expected temp file {t_path} not found for cleanup.")

    return success
