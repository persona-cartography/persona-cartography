
# File: ai/eqbench3/core/elo.py

# core/elo.py

import os
import logging
from typing import Dict, Any, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone # Added timezone
from collections import defaultdict
from pathlib import Path
import copy # For deep copying ELO data before modification

from utils.file_io import load_json_file, save_json_file
from utils.constants import (
    ANALYSIS_SCENARIO_IDS, # Used directly
    STANDARD_PAIRWISE_PROMPT_FILE, # Used directly
    ANALYSIS_PAIRWISE_PROMPT_FILE, # Used directly
    STANDARD_SCENARIO_NOTES_FILE, # Used directly
    ANALYSIS_SCENARIO_NOTES_FILE # Used directly
)

# Import from new local modules
from .elo_config import (
    DEFAULT_ELO,
    SAMPLING_SCHEDULE,
    MAX_STAGE_LOOPS,
    WIN_MARGIN_BIN_SIZE,
    WIN_MARGIN_BIN_SIZE_FOR_CI,
    scenario_notes, # Import the global variable
    analysis_scenario_notes # Import the global variable
)
from .elo_helpers import (
    load_scenario_notes,
    should_ignore_scenario
)
from .pairwise_judging import (
    _judge_scenario_pairs_in_parallel,
    _recompute_comparison_stats
)
from .trueskill_solver import (
    solve_with_trueskill,
    normalize_elo_scores
)
from .matchup_selection import (
    build_existing_matchup_set,
    update_existing_matchups_from_comparisons, # Keep for updating the set in memory
    _pick_matchups,
    create_matchup_signature # Import needed for filtering new comparisons
)

# ─────────── Comparison-filter helpers (shared) ─────────────────────────
def _is_valid_comp(c: Dict[str, Any]) -> bool:
    """Return True if *c* is usable by the solver."""
    return (
        "error" not in c
        and not should_ignore_scenario(c.get("scenario_id"))
        and c.get("pair", {}).get("test_model")
        and c.get("pair", {}).get("neighbor_model")
    )


def filter_comparisons_for_solver(comps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Basic validity filter (ignores rank window)."""
    return [c for c in comps if _is_valid_comp(c)]


def models_in_comparisons(comps: List[Dict[str, Any]]) -> Set[str]:
    """Return the set of logical model names present in *comps*."""
    mods: Set[str] = set()
    for c in comps:
        p = c.get("pair", {})
        if p.get("test_model"):    mods.add(p["test_model"])
        if p.get("neighbor_model"): mods.add(p["neighbor_model"])
    return mods
# ────────────────────────────────────────────────────────────────────────



##############################################
# Main ELO Analysis Function
##############################################

def run_elo_analysis_eqbench3(
        run_key: str,
        # File Paths
        leaderboard_elo_file: str,
        local_elo_file: str,
        # Run Data
        merged_runs_data: Dict[str, Any], # Merged leaderboard + local runs
        # Models
        test_model: str, # This is the logical model_name
        judge_model: str,
        api_clients: Dict[str, Any],
        # Other Params
        scenarios_data: Dict[str, List[str]],
        concurrency: int = 4,
        recompute_existing: bool = True
) -> Tuple[Dict[str, Any], Optional[str]]: # Return final solved ratings and error message
    """
    Three‑stage ELO procedure using merged data, writing ONLY new comparisons to local ELO file.

    1. Loads leaderboard and local ELO data.
    2. Merges comparisons and ratings (local overrides leaderboard).
    3. Builds `existing_matchups` set from the merged comparisons.
    4. Runs pairwise judging for the current `test_model` against opponents selected from the merged ladder.
    5. Filters the generated comparisons to identify *only new* ones (not in the initial `existing_matchups` set).
    6. Appends *only the new* comparisons to the `local_elo_file`.
    7. Solves ratings using the *full merged* set of comparisons (leaderboard + local + new).
    8. Returns the final solved ratings snapshot and any error message.
    """
    # ────────────────────────────── SET‑UP ────────────────────────────────
    logging.info(f"[ELO] Starting analysis for '{test_model}' (logical name)")
    logging.info(f"[ELO] Leaderboard ELO: {leaderboard_elo_file}")
    logging.info(f"[ELO] Local ELO: {local_elo_file}")
    # Access global notes defined in elo_config
    global scenario_notes, analysis_scenario_notes
    elo_error_message = None # Initialize error message

    # --- Load ELO Data ---
    leaderboard_elo = load_json_file(leaderboard_elo_file) or {"__metadata__": {}}
    local_elo = load_json_file(local_elo_file) or {"__metadata__": {}}

    # Ensure metadata structure exists
    leaderboard_elo.setdefault("__metadata__", {})
    local_elo.setdefault("__metadata__", {})

    # --- Merge Comparisons ---
    leaderboard_comps = leaderboard_elo.get("__metadata__", {}).get("global_pairwise_comparisons", [])
    local_comps = local_elo.get("__metadata__", {}).get("global_pairwise_comparisons", [])
    # Combine comparisons for building the initial set and for solving
    # Duplicates might exist if a comparison is in both, handled by build_existing_matchup_set
    all_comparisons_global: List[Dict[str, Any]] = leaderboard_comps + local_comps
    logging.info(f"[ELO] Merged comparisons: {len(leaderboard_comps)} (leaderboard) + {len(local_comps)} (local) = {len(all_comparisons_global)} total (before dedupe)")

    # --- Recompute Stats (Optional) ---
    if recompute_existing and all_comparisons_global:
        changed = 0
        for comp in all_comparisons_global:
            # Ensure the comparison has the necessary structure before recomputing
            if "pair" in comp and "test_model" in comp["pair"] and "neighbor_model" in comp["pair"] and "judge_response" in comp:
                before = comp.get("fraction_for_test")
                _recompute_comparison_stats(comp) # Now imported
                if comp.get("fraction_for_test") != before:
                    changed += 1
            elif "error" not in comp:
                 logging.warning(f"[ELO] Skipping recompute for malformed comparison: {comp.get('scenario_id', 'Unknown scenario')}")

        logging.info(f"[ELO] Recomputed plus/margin stats for "
                     f"{changed}/{len(all_comparisons_global)} stored comparisons.")

    # --- Merge ELO Ratings (Local overrides Leaderboard) ---
    # Create a deep copy to avoid modifying original dicts if they are reused elsewhere
    merged_elo_ratings = copy.deepcopy(leaderboard_elo)
    # Update with local data, overwriting existing keys and adding new ones
    for key, value in local_elo.items():
        if key != "__metadata__": # Don't overwrite metadata, comparisons handled separately
            merged_elo_ratings[key] = value

    # --- Load Prompts and Notes ---
    try:
        standard_pairwise_prompt_template = Path(STANDARD_PAIRWISE_PROMPT_FILE).read_text(encoding="utf-8")
        analysis_pairwise_prompt_template = Path(ANALYSIS_PAIRWISE_PROMPT_FILE).read_text(encoding="utf-8")
        logging.info(f"Loaded standard pairwise prompt from {STANDARD_PAIRWISE_PROMPT_FILE}")
        logging.info(f"Loaded analysis pairwise prompt from {ANALYSIS_PAIRWISE_PROMPT_FILE}")
    except Exception as e:
        logging.error(f"Failed to load pairwise prompts: {e}", exc_info=True)
        elo_error_message = f"Failed to load pairwise prompts: {e}"
        return {}, elo_error_message # Return empty dict and error

    # Load scenario notes (standard and analysis) - Use the global vars after loading
    scenario_notes.update(load_scenario_notes(STANDARD_SCENARIO_NOTES_FILE))
    analysis_scenario_notes.update(load_scenario_notes(ANALYSIS_SCENARIO_NOTES_FILE))
    logging.info(f"Loaded {len(scenario_notes)} standard scenario notes.")
    logging.info(f"Loaded {len(analysis_scenario_notes)} analysis scenario notes.")


    # --- Collect Completed Tasks from Merged Run Data ---
    all_models_scenario_results = defaultdict(lambda: defaultdict(dict))
    models_found: Set[str] = set() # Stores logical model names

    # Iterate through the merged run data
    for run_blob in merged_runs_data.values():
        # Use model_name if present, fallback to test_model for older data
        model = run_blob.get("model_name", run_blob.get("test_model"))
        if not model: continue # Skip runs without a model identifier

        for iter_idx, scenemap in run_blob.get("scenario_tasks", {}).items():
            for sid, task in scenemap.items():
                is_analysis = sid in ANALYSIS_SCENARIO_IDS
                # Analysis tasks are ready for ELO after 'scenario_completed' or 'rubric_scored'
                # Standard/Drafting tasks are ready after 'completed' or 'rubric_scored'
                required_statuses = ["scenario_completed", "rubric_scored"] if is_analysis else ["completed", "rubric_scored"]

                if task.get("status") in required_statuses and task.get("conversation_history"):
                    # Check for debrief only if not analysis
                    if not is_analysis and task.get("debrief_response") is None:
                        # logging.debug(f"Skipping task {sid} iter {iter_idx} for ELO: Missing debrief for non-analysis task.")
                        continue

                    all_models_scenario_results[model][sid][iter_idx] = task
                    models_found.add(model)

    if test_model not in models_found:
        logging.warning(f"[ELO] No finished tasks suitable for ELO found for '{test_model}'.")
        models_found.add(test_model) # Add anyway to avoid errors, will get default ELO

    # --- Initial ELO Snapshot and Existing Matchups ---
    # Build snapshot from merged ratings
    elo_snapshot = {m: merged_elo_ratings.get(m, {}).get("elo", DEFAULT_ELO) for m in models_found}
    # Add models from merged_elo_ratings that might not have tasks yet
    for m in merged_elo_ratings:
        if m != "__metadata__" and m not in elo_snapshot:
            elo_snapshot[m] = merged_elo_ratings[m].get("elo", DEFAULT_ELO)
            models_found.add(m) # Ensure all models with ratings are included

    # Build set using logical names from the combined comparison list
    initial_existing_matchups = build_existing_matchup_set(all_comparisons_global) # Now imported
    logging.info(f"[ELO] Built initial existing matchup set with {len(initial_existing_matchups)} unique signatures from merged data.")


    def _solve_for_elo(comps: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """ Solves ratings and returns both mu and sigma maps. """
        mods = set() # Stores logical model names
        valid_comps = []
        for c in comps:
             pair = c.get("pair")
             if isinstance(pair, dict):
                 test_m = pair.get("test_model")
                 neigh_m = pair.get("neighbor_model")
                 if test_m and neigh_m:
                     mods.add(test_m)
                     mods.add(neigh_m)
                     valid_comps.append(c)
             elif "error" in c:
                 pass

        if not mods: return {}, {} # No valid comparisons to solve

        # Use the current snapshot's ratings as starting points if available
        start_ratings = {m: elo_snapshot.get(m, DEFAULT_ELO) for m in mods}

        logging.info(
            f"[ELO-DBG] _solve_for_elo received {len(valid_comps)} comparisons "
            f"covering {len(mods)} models"
        )
        # Pass logical names to solver
        # Ensure all models being solved for have an entry in start_ratings
        full_start_ratings = {m: start_ratings.get(m, DEFAULT_ELO) for m in mods}

        # Solve using TrueSkill, return mu and sigma
        mu_map, sigma_map = solve_with_trueskill(
            list(mods),
            valid_comps,
            full_start_ratings, # Pass initial Mu values
            debug=False,
            use_fixed_initial_ratings=True, # Use current estimates as starting point
            bin_size=WIN_MARGIN_BIN_SIZE, # Default bin size
            return_sigma=True,
            shuffle_iterations=3,  # fewer iterations during ladder traversal
        )
        return mu_map, sigma_map


    # ─────────────────────────── SAMPLING LOOP ────────────────────────────
    new_comparisons_generated_this_run = [] # Store only comparisons generated in this execution
    current_existing_matchups = initial_existing_matchups.copy() # Track matchups encountered during this run

    for stage_idx, (radius_tiers, samples) in enumerate(SAMPLING_SCHEDULE, start=1):
        loops, stable = 0, False
        while (
            (radius_tiers == (None,) and loops == 0)     # exactly one iteration for stage 1
            or (radius_tiers != (None,) and not stable and loops < MAX_STAGE_LOOPS)
        ):
            loops += 1
            # Ensure test_model (logical name) is in the snapshot before sorting
            if test_model not in elo_snapshot:
                elo_snapshot[test_model] = DEFAULT_ELO
                logging.warning(f"Added missing test_model '{test_model}' to ELO snapshot with default rating.")

            # Ladder contains logical model names, sorted by current ELO estimates
            ladder = sorted(list(elo_snapshot.keys()), key=lambda m: elo_snapshot.get(m, DEFAULT_ELO))
            try:
                rank_old = ladder.index(test_model)
            except ValueError:
                 logging.error(f"Test model '{test_model}' not found in ELO ladder. Aborting ELO stage.")
                 elo_error_message = f"Test model '{test_model}' not found in ELO ladder."
                 break # Exit inner while loop

            opp_idx = _pick_matchups(rank_old, len(ladder), radius_tiers, samples) # Now imported
            if not opp_idx:
                logging.debug(f"[ELO Stage {stage_idx}] No opponents picked for rank {rank_old}. Moving to next stage or finishing.")
                break # Exit inner while loop

            comps_round: List[Dict[str, Any]] = []

            # ---------- run each opponent in parallel ---------------------
            outer_workers = min(len(opp_idx), concurrency)

            def _vs_neigh(idx: int) -> List[Dict[str, Any]]:
                neigh = ladder[idx] # neigh is a logical name
                depth = abs(idx - rank_old)

                # cap logical pairs per opponent
                if radius_tiers == (None,):           # stage‑1
                    cap = 1
                else:                                 # stage‑2 / stage‑3
                    if depth == 1:
                        cap = samples
                    elif depth == 2:
                        cap = max(1, samples // 2)
                    else:
                        cap = max(1, samples // 4)

                # Pass logical names to judging function
                # Pass current_existing_matchups set to avoid re-judging within this run
                return _judge_scenario_pairs_in_parallel( # Now imported
                    test_model, # Logical name
                    neigh,      # Logical name
                    all_models_scenario_results[test_model],
                    all_models_scenario_results[neigh],
                    concurrency,                    # inner pool
                    # Pass both templates
                    standard_pairwise_prompt_template,
                    analysis_pairwise_prompt_template,
                    scenarios_data,
                    judge_model,
                    api_clients,
                    cap,                            # ✱ per‑opponent cap ✱
                    current_existing_matchups,      # Pass the set of matchups already seen/judged
                )

            with ThreadPoolExecutor(max_workers=outer_workers) as pool:
                fut_map = {pool.submit(_vs_neigh, i): i for i in opp_idx}
                for fut in as_completed(fut_map):
                    try:
                        comps = fut.result()
                        comps_round.extend(comps)
                    except Exception as e:
                        logging.error(f"[ELO] opponent job failed: {e}", exc_info=True)
                        # Optionally store an error marker?

            # ========= SUMMARY‑OF‑MATCHUPS (single block) =========================
            if comps_round:
                per_opp   = defaultdict(list)
                new_comps_count = 0
                for comp in comps_round:
                    if "error" in comp: continue
                    pair = comp["pair"]
                    opp = pair["neighbor_model"] if pair["test_model"] == test_model else pair["test_model"]
                    per_opp[opp].append(comp["scenario_id"])
                    # Check if this comparison is truly new based on signature
                    sig = create_matchup_signature(pair["test_model"], pair["neighbor_model"], comp["scenario_id"], str(pair["iteration_index"]))
                    if sig not in initial_existing_matchups: # Check against the initial set
                         new_comps_count += 1


                if per_opp:
                    block = [
                        "",
                        "================  Matchups selected this round  ================",
                        f"Stage‑{stage_idx}  •  loop {loops}",
                        f"Test model: {test_model}",
                        f"New comparisons generated: {new_comps_count // 2} logical pairs ({new_comps_count} raw)", # Divide by 2 for logical pairs
                        "---------------------------------------------------------------",
                    ]
                    for opp, scen_list in sorted(per_opp.items()):
                        uniq = sorted(set(scen_list))
                        block.append(f"{opp}  →  {len(uniq)} logical pairs")
                        # block.extend(f"   • {sid}" for sid in uniq) # Maybe too verbose
                    block.append("================================================================")
                    logging.info("\n".join(block))
            # =====================================================================

            # --- Update global state and identify NEW comparisons ---
            newly_added_signatures = update_existing_matchups_from_comparisons(comps_round, current_existing_matchups)
            logging.debug(f"[ELO] Added {newly_added_signatures} new unique matchup signatures to the in-memory set.")

            # Filter comps_round to get only the ones generated *now* (not present initially)
            new_comparisons_this_round = []
            for comp in comps_round:
                 if "error" in comp: # Include errors generated this round
                     new_comparisons_this_round.append(comp)
                     continue
                 pair = comp.get("pair")
                 if pair:
                     sig = create_matchup_signature(pair["test_model"], pair["neighbor_model"], comp["scenario_id"], str(pair.get("iteration_index")))
                     if sig not in initial_existing_matchups:
                         new_comparisons_this_round.append(comp)

            new_comparisons_generated_this_run.extend(new_comparisons_this_round)
            all_comparisons_global.extend(new_comparisons_this_round) # Add new comps to the list used for solving


            # --- Re-solve ratings (using FULL merged comparison list) -------------
            comps_for_solver = filter_comparisons_for_solver(all_comparisons_global)

            # Solve using the filtered list
            if comps_for_solver:
                # Solver uses logical names
                new_mu_map, _ = _solve_for_elo(comps_for_solver) # Ignore sigma map for stability check
            else:
                new_mu_map = {}

            # Update the ELO snapshot for the next loop/stability check
            new_snapshot = elo_snapshot.copy()
            new_snapshot.update(new_mu_map)

            # Ensure test_model (logical name) is still present before getting index
            if test_model not in new_snapshot:
                 new_snapshot[test_model] = DEFAULT_ELO # Add back if somehow lost
                 logging.warning(f"Test model '{test_model}' was missing from new ELO snapshot, re-added with default.")

            # Check stability based on rank change
            ladder_new = sorted(list(new_snapshot.keys()), key=lambda m: new_snapshot.get(m, DEFAULT_ELO))
            try:
                rank_new = ladder_new.index(test_model)
            except ValueError:
                 logging.error(f"Test model '{test_model}' not found in *new* ELO ladder. Stability check failed.")
                 rank_new = -1 # Indicate error
                 elo_error_message = f"Test model '{test_model}' lost during ELO update."

            stable = (rank_new == rank_old) and (rank_new != -1)
            elo_snapshot = new_snapshot # Update snapshot for the next iteration

        # End of while loop for stage
        logging.info('-------------------------------------------------')
        logging.info(
            f"[ELO] stage‑{stage_idx} finished "
            f"(elo={elo_snapshot.get(test_model, 'N/A'):.1f}, "
            f"reason={'stable rank' if stable else 'max loops reached'})"
        )
        logging.info('-------------------------------------------------')
        if rank_old == -1 or elo_error_message: # Break outer loop if test_model vanished or error occurred
            break


        # ────────────────── SAVE NEW COMPARISONS TO LOCAL ELO FILE ───────────────
    # (Save only the comparisons generated in this specific run)
    if new_comparisons_generated_this_run:
        logging.info(f"[ELO] Appending {len(new_comparisons_generated_this_run)} new comparison results to local ELO file: {local_elo_file}")
        # Load the local file again to minimize race conditions
        current_local_elo_for_comps = load_json_file(local_elo_file) or {"__metadata__": {}}
        current_local_elo_for_comps.setdefault("__metadata__", {}).setdefault("global_pairwise_comparisons", [])

        # Append only the new comparisons
        current_local_elo_for_comps["__metadata__"]["global_pairwise_comparisons"].extend(new_comparisons_generated_this_run)
        # Update timestamp only if saving comparisons succeeds
        # current_local_elo_for_comps["__metadata__"]["last_updated"] = datetime.now(timezone.utc).isoformat() # Moved timestamp update to after ratings save

        # Save back to the local file
        save_comps_success = save_json_file(current_local_elo_for_comps, local_elo_file)
        if not save_comps_success:
            logging.error(f"[ELO] FAILED to save new comparisons to {local_elo_file}")
            if not elo_error_message: # Don't overwrite existing error
                 elo_error_message = f"Failed to save new comparisons to {local_elo_file}"
        else:
             logging.info(f"[ELO] Successfully appended new comparisons to {local_elo_file}")
    else:
        logging.info("[ELO] No new comparisons were generated in this run to save.")


    # ────────────────── FINAL SOLVE & NORMALIZE (using all comps) ───────────
    logging.info("[ELO] Performing final rating calculation")
    final_snapshot = {} # This will hold the results like {"model_name": {"elo": ..., "elo_norm": ...}}
    # Use the latest elo_snapshot from the sampling loop as a fallback if solve fails
    fallback_snapshot = elo_snapshot

    try:
        # Filter out ignored scenarios and errors for the final solve
        final_comps_for_solver = filter_comparisons_for_solver(all_comparisons_global)

        if final_comps_for_solver:
            # Determine the set of all models involved in valid comparisons for the final solve
            models_in_final_solve = models_in_comparisons(final_comps_for_solver)


            # Ensure all models found during task collection are included, even if they had no comparisons
            models_to_solve_for = models_found.union(models_in_final_solve)
            logging.info(f"[ELO] Final solve includes {len(models_to_solve_for)} models.")

            # Solve using the full comparison list to get final ratings
            # Use fixed initial ratings (DEFAULT_ELO) for this final solve for consistency
            final_mu_map, _ = solve_with_trueskill(
                list(models_to_solve_for), # Use the combined set of models
                final_comps_for_solver,
                {m: DEFAULT_ELO for m in models_to_solve_for}, # Start fresh from default
                debug=False,
                use_fixed_initial_ratings=True,
                bin_size=WIN_MARGIN_BIN_SIZE,
                return_sigma=True
            )

            # solve again just for sigma (from which CI is calculated)
            # using a smaller bin size so we are more fairly factoring in
            # the increase in certainty afforded by the win margin
            #  (note: this is just a guesstimate since we're already abusing
            #   trueskill's sigma estimate by expanding win margin into extra wins)
            _, final_sigma_map = solve_with_trueskill(
                list(models_to_solve_for), # Use the combined set of models
                final_comps_for_solver,
                {m: DEFAULT_ELO for m in models_to_solve_for}, # Start fresh from default
                debug=False,
                use_fixed_initial_ratings=True,
                bin_size=WIN_MARGIN_BIN_SIZE_FOR_CI, # Use bin_size=5 for CI calculation
                return_sigma=True
            )

            # Normalize scores based on the solved Mu values
            normalized_scores = normalize_elo_scores(final_mu_map)

            # Combine results into the final snapshot structure
            # Use models_to_solve_for to ensure all relevant models get an entry
            ts_env_sigma = 350/3 # Default sigma from TrueSkill setup
            for m in models_to_solve_for:
                mu_raw = final_mu_map.get(m, DEFAULT_ELO)
                # Use solved sigma if available, otherwise default env sigma
                sigma = final_sigma_map.get(m, ts_env_sigma)
                mu_norm = normalized_scores.get(m, DEFAULT_ELO) # Use default if normalization failed

                # Calculate CI bounds on raw score
                ci_low_raw = mu_raw - 1.96 * sigma
                ci_high_raw = mu_raw + 1.96 * sigma

                # Store everything
                final_snapshot[m] = {
                    "elo": round(mu_raw, 2),
                    "elo_norm": round(mu_norm, 2),
                    "sigma": round(sigma, 2),
                    "ci_low": round(ci_low_raw, 2),
                    "ci_high": round(ci_high_raw, 2),
                    # We need normalized CI bounds too
                }

            # --- Normalize CI bounds ---
            # Create a dict with raw scores + bounds for normalization
            raw_plus_bounds = {}
            for m, data in final_snapshot.items():
                raw_plus_bounds[m] = data["elo"]
                raw_plus_bounds[f"{m}__low"] = data["ci_low"]
                raw_plus_bounds[f"{m}__high"] = data["ci_high"]

            norm_plus = normalize_elo_scores(raw_plus_bounds)

            # Add normalized bounds back to the final snapshot
            for m in final_snapshot:
                 # Use normalized mu as fallback if bounds missing
                 norm_mu_fallback = final_snapshot[m]["elo_norm"]
                 final_snapshot[m]["ci_low_norm"] = round(norm_plus.get(f"{m}__low", norm_mu_fallback), 2)
                 final_snapshot[m]["ci_high_norm"] = round(norm_plus.get(f"{m}__high", norm_mu_fallback), 2)

            logging.info("[ELO] Final rating calculation and normalization complete.")

        else:
            logging.warning("[ELO] No valid comparisons available for final solve.")
            if not elo_error_message: elo_error_message = "No valid comparisons for final solve"
            # If solve didn't run, create a basic snapshot from the fallback
            final_snapshot = {m: {"elo": r, "elo_norm": r} for m, r in fallback_snapshot.items()}

    except Exception as e:
        logging.error(f"[ELO] Final solve or normalization failed: {e}", exc_info=True)
        if not elo_error_message: elo_error_message = f"Final solve/normalization failed: {e}"
        # If solve failed, create a basic snapshot from the fallback
        final_snapshot = {m: {"elo": r, "elo_norm": r} for m, r in fallback_snapshot.items()}


    # ────────────────── SAVE FINAL RATINGS TO LOCAL ELO FILE ───────────────
    # Overwrite top-level model keys with the newly calculated ratings
    logging.info(f"[ELO] Saving final ratings snapshot to local ELO file: {local_elo_file}")
    try:
        # Load the local file again to ensure we have the latest comparisons list
        current_local_elo = load_json_file(local_elo_file) or {"__metadata__": {}}
        current_local_elo.setdefault("__metadata__", {}) # Ensure metadata key exists

        # Update the top-level model entries with the latest solved ratings
        for model_name_key, rating_data in final_snapshot.items():
            if model_name_key != "__metadata__": # Prevent accidentally overwriting metadata
                current_local_elo[model_name_key] = rating_data

        # Update timestamp
        current_local_elo["__metadata__"]["last_updated"] = datetime.now(timezone.utc).isoformat()

        # Save the updated structure back to the local file
        save_ratings_success = save_json_file(current_local_elo, local_elo_file)
        if not save_ratings_success:
            logging.error(f"[ELO] FAILED to save final ratings to {local_elo_file}")
            if not elo_error_message: # Don't overwrite existing error
                 elo_error_message = f"Failed to save final ratings to {local_elo_file}"
        else:
             logging.info(f"[ELO] Successfully saved final ratings to {local_elo_file}")

    except Exception as e:
         logging.error(f"[ELO] Error saving final ratings to {local_elo_file}: {e}", exc_info=True)
         if not elo_error_message:
              elo_error_message = f"Error saving final ratings: {e}"


    # Return the computed ratings snapshot and any error message
    return final_snapshot, elo_error_message