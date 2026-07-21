
# File: ai/eqbench3/core/trueskill_solver.py

# core/trueskill_solver.py (or add to glicko_solver.py if preferred)

import logging
import math
import random
import trueskill # Import TrueSkill
import warnings
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

# Assuming these are in the same directory or accessible via path
from .elo_config import DEFAULT_ELO, WIN_MARGIN_BIN_SIZE
from .pairwise_judging import compute_fraction_for_test

def normalize_elo_scores(raw_scores: Dict[str, float], anchor_models: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Normalizes ELO scores using anchor models (same as original).
    Operates on logical model names.
    """
    if anchor_models is None:
        # Default anchors - ADJUST THESE based on models likely to be in your benchmark
        anchor_models = {
            # Example anchors - replace with relevant models from your runs
            # 'gpt-4-turbo': 1500,
            # 'claude-3-haiku-20240307': 1000
            # Using original example anchors if none better are known
             #'deepseek/deepseek-r1': 1500, # Example - replace if needed
             'o3': 1500, # Example - replace if needed
             'meta-llama/llama-3.2-1b-instruct': 200 # Example - replace if needed
        }
        logging.info(f"Using default anchor models for ELO normalization: {anchor_models}")

    # raw_scores keys are logical names
    valid_anchors = {k: v for k, v in anchor_models.items() if k in raw_scores and isinstance(raw_scores.get(k), (int, float))}

    if len(valid_anchors) < 2:
        logging.warning(f"Not enough valid anchor models found in scores ({len(valid_anchors)} found). Required 2 from {list(anchor_models.keys())}. Returning raw scores.")
        # Return a copy to avoid modifying the original dict if it was passed
        return {k: v for k, v in raw_scores.items()}

    anchor_items = list(valid_anchors.items())
    model_a, target_a = anchor_items[0] # Logical name
    model_b, target_b = anchor_items[1] # Logical name

    raw_a = raw_scores[model_a]
    raw_b = raw_scores[model_b]

    if abs(raw_a - raw_b) < 1e-6:
        logging.warning("Anchor models have nearly identical raw scores. Normalization might be unstable. Using scale=1.0.")
        scale = 1.0
    else:
        scale = (target_a - target_b) / (raw_a - raw_b)

    shift = target_a - (scale * raw_a)

    normalized_scores = {}
    for model, score in raw_scores.items(): # model is logical name
        if isinstance(score, (int, float)):
            normalized_scores[model] = (score * scale + shift)
        else:
            normalized_scores[model] = score # Keep non-numeric scores as is

    logging.info(f"ELO normalization applied using anchors {model_a} ({raw_a:.2f}->{target_a}) and {model_b} ({raw_b:.2f}->{target_b}). Scale={scale:.4f}, Shift={shift:.2f}")
    return normalized_scores


# Keep the _fraction_from_plus helper function as it's used in pre-processing
def _fraction_from_plus(group: List[Dict[str, Any]],
                            m1: str, # Logical name
                            m2: str) -> Optional[float]: # Logical name
        """
        Aggregate plus‑counts for the two models across the forward/reverse
        comparisons in *group*, then compute fraction_for_test (for m1)
        with the same custom margin logic already used elsewhere.
        (Copied verbatim from solve_with_glicko)
        """
        plus_m1 = plus_m2 = 0
        for comp in group:
            if "plus_for_test" not in comp or "plus_for_other" not in comp:
                return None  # can’t use the new rule → caller will fallback
            # Use logical names from pair dict
            if comp["pair"]["test_model"] == m1:
                plus_m1 += comp["plus_for_test"]
                plus_m2 += comp["plus_for_other"]
            else:  # m2 was the test_model
                plus_m2 += comp["plus_for_test"]
                plus_m1 += comp["plus_for_other"]

        if plus_m1 > plus_m2:
            outcome = 1.0
        elif plus_m1 < plus_m2:
            outcome = 0.0
        else:
            outcome = 0.5

        # Ensure compute_fraction_for_test is accessible here
        frac, *_ = compute_fraction_for_test(outcome, plus_m1, plus_m2)
        return frac

# Keep the bin_fraction helper function as it's used for win expansion
def bin_fraction(frac: float, bin_size) -> Tuple[int, int]:
    """
    Convert fraction_for_test into asymmetric pseudo-match counts.
    (Copied verbatim from solve_with_glicko, assuming bin_size=4 default)
    """
    # Clamp input
    frac = max(0.0, min(1.0, frac))
    eps  = 1e-9
    step = 0.5 / bin_size        # width of each chunk

    # Exact draw
    if abs(frac - 0.5) < eps:
        return 1, 1

    # Test-model win
    if frac > 0.5:
        margin      = frac - 0.5
        wins_test   = max(1, min(bin_size, math.ceil(margin / step)))
        wins_other  = 0
        return wins_test, wins_other

    # Test-model loss
    margin      = 0.5 - frac
    wins_other  = max(1, min(bin_size, math.ceil(margin / step)))
    wins_test   = 0
    return wins_test, wins_other


def solve_with_trueskill(
    all_models: List[str], # List of logical model names
    pairwise_comparisons: List[Dict], # Comparisons contain logical names in pairs
    initial_ratings: Dict[str, float], # Keys are logical names, values are initial Mu
    debug: bool = False,
    use_fixed_initial_ratings=True,
    bin_size = WIN_MARGIN_BIN_SIZE,
    return_sigma: bool = False,
    shuffle_iterations: int = 10,
) -> Dict[str, float]:
    """
    Calculates ratings using TrueSkill, mirroring the structure of solve_with_glicko:
      - Uses the same pre-processing to pair forward/reverse comparisons (using logical names).
      - Uses the same bin_fraction logic to expand fraction_for_test into wins/losses/draws.
      - Applies TrueSkill updates immediately after each (pseudo) match.
      - Runs multiple iterations with shuffled comparison order and averages
        the results to eliminate order-dependence artifacts.

    Args:
        all_models: List of all logical model names.
        pairwise_comparisons: Raw list of comparison dicts (pairs contain logical names).
        initial_ratings: Dictionary mapping logical model names to their initial Mu.
                         Used if use_fixed_initial_ratings is False.
        debug: If True, enable debug logging.
        use_fixed_initial_ratings: If True, ignore initial_ratings for active models
                                   and start them at DEFAULT_ELO. If False, use
                                   provided initial_ratings.
        shuffle_iterations: Number of shuffled solve iterations to average over.
                            Higher values reduce order-dependence noise. Default 10.

    Returns a dictionary {logical_model_name: final_trueskill_mu}.
    """
    # --- Start: Identical Pre-processing from solve_with_glicko ---
    grouped_comparisons = defaultdict(list)
    for c in pairwise_comparisons:
        if "error" in c or "pair" not in c or "scenario_id" not in c: # Removed fraction check here, handle later
            continue

        pair = c.get("pair", {})
        # These are logical names
        test_model = pair.get("test_model")
        neighbor_model = pair.get("neighbor_model")
        scenario_id = c.get("scenario_id")
        iter_idx = pair.get("iteration_index")

        if not all([test_model, neighbor_model, scenario_id, iter_idx is not None]):
            continue

        # Sort logical names for consistent grouping
        model1, model2 = sorted([test_model, neighbor_model])
        group_key = (model1, model2, scenario_id, str(iter_idx))
        grouped_comparisons[group_key].append(c)

    if debug:
        group_sizes = [len(group) for group in grouped_comparisons.values()]
        size_counts = {}
        for size in group_sizes: size_counts[size] = size_counts.get(size, 0) + 1
        logging.debug(f"[TrueSkill] Group size distribution: {size_counts}")
        big_groups = [key for key, group in grouped_comparisons.items() if len(group) > 2]
        if big_groups:
            logging.warning(f"[TrueSkill] Found {len(big_groups)} groups with > 2 comps. First 3: {big_groups[:3]}")

    paired_comparisons = []
    n_len_1 = 0 # Keep track just for potential debug/comparison

    for group_key, group in grouped_comparisons.items():
        model1, model2, scenario_id, iter_idx = group_key # model1, model2 are logical names
        frac = None # Initialize frac

        # Try using plus counts first (uses logical names)
        frac = _fraction_from_plus(group, model1, model2)

        # Fallback logic if plus counts aren't available or sufficient
        if frac is None:
            if len(group) == 2:
                # Check based on logical names in pair dict
                model1_as_test = next((c for c in group if c["pair"]["test_model"] == model1 and "fraction_for_test" in c), None)
                model2_as_test = next((c for c in group if c["pair"]["test_model"] == model2 and "fraction_for_test" in c), None)
                if model1_as_test and model2_as_test:
                    frac = (model1_as_test["fraction_for_test"] + (1.0 - model2_as_test["fraction_for_test"])) / 2
                elif model1_as_test: frac = model1_as_test["fraction_for_test"]
                elif model2_as_test: frac = 1.0 - model2_as_test["fraction_for_test"]
                else: frac = 0.5 # Fallback if fractions missing
            elif len(group) == 1:
                n_len_1 += 1
                c = group[0]
                if "fraction_for_test" in c:
                    test_m = c["pair"]["test_model"] # Logical name
                    frac = c["fraction_for_test"] if test_m == model1 else 1.0 - c["fraction_for_test"]
                else: frac = 0.5 # Fallback if fraction missing
            else: # > 2 comparisons, fallback averaging
                fracs = []
                for c in group:
                    if "fraction_for_test" in c:
                        if c["pair"]["test_model"] == model1: fracs.append(c["fraction_for_test"])
                        else: fracs.append(1.0 - c["fraction_for_test"])
                frac = sum(fracs) / len(fracs) if fracs else 0.5

        # Only add if we successfully determined a fraction
        if frac is not None:
             paired_comparisons.append({
                "scenario_id": scenario_id,
                # Store logical names in the pair
                "pair": {"test_model": model1, "neighbor_model": model2},
                "fraction_for_test": frac,
            })
        elif debug:
             logging.warning(f"[TrueSkill] Could not determine fraction for group {group_key}. Skipping.")


    active_models = set() # Set of logical names
    for c in paired_comparisons:
        active_models.add(c["pair"]["test_model"])
        active_models.add(c["pair"]["neighbor_model"])

    logging.info(
        f"[TrueSkill] Solver got {len(paired_comparisons)} paired comps; "
        f"active models = {len(active_models)}"
    )
    # --- End: Identical Pre-processing ---

    # Two ways to hack trueskill to factor in win margins:
    EXPAND_MARGINS_TO_EXTRA_WINS = True

    # ── Shuffle-averaged solve ──────────────────────────────────────────────
    # TrueSkill is order-dependent: models whose comparisons appear early
    # in the list get rated against opponents that haven't been calibrated
    # yet.  Averaging over multiple shuffled runs eliminates this artifact.
    n_iters = max(1, shuffle_iterations)
    logging.info(f"[TrueSkill] Running {n_iters} shuffled iteration(s) "
                 f"(bin_size={bin_size})")

    mu_accum:    Dict[str, float] = {m: 0.0 for m in all_models}
    sigma_accum: Dict[str, float] = {m: 0.0 for m in all_models}

    for iter_idx in range(n_iters):
        # Deterministic shuffle per iteration
        shuffled = paired_comparisons.copy()
        random.Random(iter_idx).shuffle(shuffled)

        # --- Initialise ratings for this iteration ---
        if EXPAND_MARGINS_TO_EXTRA_WINS:
            ts_env = trueskill.TrueSkill(mu=DEFAULT_ELO, sigma=350/3,
                                         beta=350/6, tau=0.0,
                                         draw_probability=0.0)
        else:
            BASE_SIGMA = 350 / 3
            BASE_BETA  = 350 / 6
            GAMMA      = 40.0
            ts_env = trueskill.TrueSkill(mu=DEFAULT_ELO, sigma=BASE_SIGMA,
                                         beta=BASE_BETA, tau=0.0,
                                         draw_probability=0.0)

        initial_sigma = ts_env.sigma
        ratings: Dict[str, trueskill.Rating] = {}
        for m in all_models:
            if m in active_models and use_fixed_initial_ratings:
                start_mu = DEFAULT_ELO
            elif m in initial_ratings:
                start_mu = initial_ratings[m]
            else:
                start_mu = DEFAULT_ELO
            ratings[m] = ts_env.Rating(mu=start_mu, sigma=initial_sigma)

        # --- Process comparisons ---
        if not EXPAND_MARGINS_TO_EXTRA_WINS:
            env_cache: Dict[float, trueskill.TrueSkill] = {}
            for c in shuffled:
                frac    = c["fraction_for_test"]
                test_m  = c["pair"]["test_model"]
                neigh_m = c["pair"]["neighbor_model"]
                if test_m == neigh_m or test_m not in ratings or neigh_m not in ratings:
                    continue

                margin = abs(frac - 0.5) * 2.0
                k = 1.0 + GAMMA * margin
                beta_eff = BASE_BETA / math.sqrt(k)
                w_test, w_other = bin_fraction(frac, bin_size=bin_size)

                env = env_cache.get(beta_eff)
                if env is None:
                    env = trueskill.TrueSkill(mu=DEFAULT_ELO, sigma=BASE_SIGMA,
                                             beta=beta_eff, tau=0.0,
                                             draw_probability=0.0)
                    env_cache[beta_eff] = env

                try:
                    if w_test == w_other:
                        r_t, r_n = env.rate_1vs1(ratings[test_m],
                                                 ratings[neigh_m], drawn=True)
                    elif w_test > w_other:
                        r_t, r_n = env.rate_1vs1(ratings[test_m], ratings[neigh_m])
                    else:
                        r_n, r_t = env.rate_1vs1(ratings[neigh_m], ratings[test_m])
                    ratings[test_m], ratings[neigh_m] = r_t, r_n
                except (ValueError, Exception) as e:
                    logging.warning(f"[TrueSkill] Update failed ({test_m} vs "
                                    f"{neigh_m}): {e}. Skipping.")

        else:  # EXPAND_MARGINS_TO_EXTRA_WINS
            for c in shuffled:
                frac    = c["fraction_for_test"]
                test_m  = c["pair"]["test_model"]
                neigh_m = c["pair"]["neighbor_model"]
                if test_m not in ratings or neigh_m not in ratings:
                    continue
                if test_m == neigh_m:
                    continue

                w_test, w_other = bin_fraction(frac, bin_size=bin_size)

                try:
                    if w_test == 1 and w_other == 1:  # Draw
                        r_test, r_neigh = ts_env.rate_1vs1(
                            ratings[test_m], ratings[neigh_m], drawn=True)
                        ratings[test_m], ratings[neigh_m] = r_test, r_neigh
                    else:
                        for _ in range(w_test):
                            r_test, r_neigh = ts_env.rate_1vs1(
                                ratings[test_m], ratings[neigh_m])
                            ratings[test_m], ratings[neigh_m] = r_test, r_neigh
                        for _ in range(w_other):
                            r_neigh, r_test = ts_env.rate_1vs1(
                                ratings[neigh_m], ratings[test_m])
                            ratings[test_m], ratings[neigh_m] = r_test, r_neigh
                except (ValueError, Exception) as e:
                    logging.warning(f"[TrueSkill] Update failed ({test_m} vs "
                                    f"{neigh_m}): {e}. Skipping.")

        # --- Accumulate this iteration's results ---
        for m in all_models:
            mu_accum[m]    += ratings[m].mu
            sigma_accum[m] += ratings[m].sigma

    # --- Average across iterations ---
    final_map = {m: mu_accum[m] / n_iters    for m in all_models}
    sigma_map = {m: sigma_accum[m] / n_iters  for m in all_models}

    if debug:
        for m in all_models:
            print(f"[TrueSkill] {m}: {final_map[m]:.2f} "
                  f"(σ={sigma_map[m]:.2f})")

    return (final_map, sigma_map) if return_sigma else final_map
