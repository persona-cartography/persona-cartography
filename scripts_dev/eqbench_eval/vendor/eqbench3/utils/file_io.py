# File: ai/eqbench3/utils/file_io.py

import os
import json
import logging
import threading
from typing import Dict, Any
import time
import gzip # Added for gzip support

_file_locks = {}
_file_locks_lock = threading.RLock()

def get_file_lock(file_path: str) -> threading.RLock:
    """
    Acquire or create a per-file lock to avoid concurrent writes.
    """
    # This function should now only be called with valid string paths.
    # The check for None should happen in the calling function (load_json_file).
    with _file_locks_lock:
        # Normalize path for consistent locking, especially with .gz
        normalized_path = os.path.normpath(file_path)
        if normalized_path not in _file_locks:
            _file_locks[normalized_path] = threading.RLock()
        return _file_locks[normalized_path]

def load_json_file(file_path: str) -> dict:
    """
    Thread-safe read of a JSON file (plain or gzipped), returning an empty dict
    if not found or error. Detects gzip based on '.gz' extension.
    Handles file_path being None by returning an empty dict immediately.
    """
    if file_path is None:
        logging.debug("File path is None, returning empty dict.")
        return {}

    lock = get_file_lock(file_path)
    with lock:
        if not os.path.exists(file_path):
            logging.debug(f"File not found: {file_path}, returning empty dict.")
            return {}

        open_func = gzip.open if file_path.endswith(".gz") else open
        mode = 'rt' if file_path.endswith(".gz") else 'r'
        encoding = 'utf-8' # Consistent encoding

        try:
            with open_func(file_path, mode=mode, encoding=encoding) as f:
                content = f.read()
                if not content.strip(): # Handle empty file
                    logging.warning(f"File is empty: {file_path}, returning empty dict.")
                    return {}
                return json.loads(content)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in {file_path}: {e}. Returning empty dict.")
            # Optional: backup corrupted file (consider if needed for .gz)
            return {}
        except gzip.BadGzipFile as e:
            logging.error(f"Bad Gzip file error reading {file_path}: {e}. Returning empty dict.")
            return {}
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}", exc_info=True)
            return {}
        
def _make_temp_path(file_path: str) -> str:
    """
    Derive a temporary filename that preserves a trailing '.gz' so the gzip
    handle is closed *before* we rename.  Examples:
        foo.json           -> foo.json.tmp
        bar/baz.json.gz    -> bar/baz.json.tmp.gz
    """
    if file_path.endswith(".gz"):
        return f"{file_path[:-3]}.tmp.gz"   # insert .tmp *before* .gz
    return f"{file_path}.tmp"


def _atomic_write_json(data: Dict[str, Any], file_path: str):
    """
    Atomically write *data* as JSON to *file_path*.

    For .gz targets we set the header filename so the archive contains
    the *final* `.json` name even if we're writing to a temp file.
    """
    temp_path = _make_temp_path(file_path)
    is_gz     = file_path.endswith(".gz")

    try:
        if is_gz:
            # -------------------------------------------------------------
            # Derive the intended header name:
            #   foo.tmp.<uuid>.json.gz  ->  foo.json
            #   canonical_leaderboard_results.json.gz  -> same minus .gz
            #
            base_no_gz = os.path.basename(file_path[:-3])  # strip '.gz'
            if ".tmp." in base_no_gz:
                stem, ext = os.path.splitext(base_no_gz)   # stem may still have .tmp.<uuid>
                stem = stem.split(".tmp.", 1)[0]           # drop tmp token
                header_name = stem + ext                   # e.g. foo.json
            else:
                header_name = base_no_gz                   # already fine

            json_bytes = json.dumps(data, indent=2,
                                     ensure_ascii=False).encode("utf-8")

            with open(temp_path, "wb") as raw:
                with gzip.GzipFile(
                    filename=header_name,  # correct header
                    mode="wb",
                    fileobj=raw,
                ) as gz:
                    gz.write(json_bytes)
        else:
            with open(temp_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)

        os.replace(temp_path, file_path)  # atomic replace
    except Exception as e:
        logging.error(f"Atomic write failed for {file_path}: {e}")
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except OSError:
            pass
        raise




def save_json_file(data: Dict[str, Any], file_path: str, max_retries: int = 3, retry_delay: float = 0.5) -> bool:
    """
    Thread-safe function to save dictionary data to a JSON file (plain or gzipped)
    using atomic write. Includes retry logic. Detects gzip based on '.gz' extension.
    """
    if file_path is None: # Should not happen if called correctly, but good safeguard
        logging.error("save_json_file called with None file_path. Cannot save.")
        return False

    lock = get_file_lock(file_path)
    for attempt in range(max_retries):
        if attempt > 0:
            logging.warning(f"Retrying save to {file_path} (attempt {attempt+1}/{max_retries})...")
            time.sleep(retry_delay * (1.5 ** attempt)) # Exponential backoff

        # Acquire lock inside the loop for each attempt
        with lock:
            try:
                # Ensure parent directory exists
                dir_name = os.path.dirname(file_path)
                if dir_name: # Check if dirname is not empty
                    os.makedirs(dir_name, exist_ok=True)

                _atomic_write_json(data, file_path)
                logging.debug(f"Successfully wrote JSON to {file_path} on attempt {attempt+1}")
                return True
            except Exception as e:
                logging.error(f"save_json_file() attempt {attempt+1} failed for {file_path}: {e}", exc_info=True)
                # Continue to next retry attempt

    logging.error(f"Failed to save JSON to {file_path} after {max_retries} attempts.")
    return False


def update_run_data(runs_file: str, run_key: str, update_dict: Dict[str, Any],
                    max_retries: int = 3, retry_delay: float = 0.5) -> bool:
    """
    Thread-safe function to MERGE partial run data into the existing run file
    (plain or gzipped). Handles nested merging for 'scenario_tasks'.
    Detects gzip based on '.gz' extension.
    """
    if runs_file is None: # Should not happen if called correctly
        logging.error(f"update_run_data called with None runs_file for run_key {run_key}. Cannot update.")
        return False

    lock = get_file_lock(runs_file)
    for attempt in range(max_retries):
        if attempt > 0:
            logging.warning(f"Retrying update_run_data for {run_key} in {runs_file} (attempt {attempt+1}/{max_retries})...")
            time.sleep(retry_delay * (1.5 ** attempt)) # Exponential backoff

        # Acquire lock inside the loop for each attempt
        with lock:
            current_runs = {} # Initialize as empty dict
            try:
                # Load existing data within the lock (handles gzip)
                current_runs = load_json_file(runs_file) # Use the updated load function

                if not isinstance(current_runs, dict):
                    logging.warning(f"Run file {runs_file} content is not a dictionary or load failed. Resetting to empty dict.")
                    current_runs = {}

                # Ensure the run key exists
                if run_key not in current_runs:
                    current_runs[run_key] = {}

                # --- Merge Logic (remains the same) ---
                run_data_to_update = current_runs[run_key]

                for top_key, new_val in update_dict.items():
                    if top_key == "scenario_tasks":
                        if top_key not in run_data_to_update:
                            run_data_to_update[top_key] = {}
                        if not isinstance(run_data_to_update[top_key], dict):
                             logging.warning(f"Overwriting non-dict '{top_key}' in {run_key} with new dict.")
                             run_data_to_update[top_key] = {}

                        if isinstance(new_val, dict):
                            for iteration_idx, scenario_map in new_val.items():
                                if iteration_idx not in run_data_to_update[top_key]:
                                    run_data_to_update[top_key][iteration_idx] = {}
                                if not isinstance(run_data_to_update[top_key][iteration_idx], dict):
                                     logging.warning(f"Overwriting non-dict iteration '{iteration_idx}' in {run_key}/{top_key} with new dict.")
                                     run_data_to_update[top_key][iteration_idx] = {}

                                if isinstance(scenario_map, dict):
                                    for s_id, s_data in scenario_map.items():
                                        run_data_to_update[top_key][iteration_idx][s_id] = s_data
                                else:
                                    logging.warning(f"Received non-dict data for iteration '{iteration_idx}' in {run_key}/{top_key}. Overwriting.")
                                    run_data_to_update[top_key][iteration_idx] = scenario_map
                        else:
                             logging.warning(f"Received non-dict data for '{top_key}' in {run_key}. Overwriting.")
                             run_data_to_update[top_key] = new_val

                    elif top_key in ["results", "elo_analysis"]:
                        if top_key not in run_data_to_update:
                            run_data_to_update[top_key] = {}
                        if isinstance(new_val, dict) and isinstance(run_data_to_update[top_key], dict):
                            run_data_to_update[top_key].update(new_val)
                        else:
                            run_data_to_update[top_key] = new_val

                    else:
                        run_data_to_update[top_key] = new_val
                # --- End Merge Logic ---

                # Now write the modified current_runs back to the file (handles gzip)
                # Ensure parent directory exists before writing
                dir_name = os.path.dirname(runs_file)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)

                _atomic_write_json(current_runs, runs_file)
                logging.debug(f"Successfully updated run_key={run_key} in {runs_file} on attempt {attempt+1}")
                return True # Success

            except (IOError, gzip.BadGzipFile) as e: # Added BadGzipFile
                logging.error(f"Error reading/writing run file {runs_file} on attempt {attempt+1}: {e}. Aborting update for this attempt.")
                # Don't return False immediately, allow retry
            except Exception as e:
                logging.error(f"Error during update_run_data attempt {attempt+1} for {run_key} in {runs_file}: {e}", exc_info=True)
                # Don't return False immediately, allow retry

    # If loop finishes, all retries failed
    logging.error(f"update_run_data failed for {run_key} in {runs_file} after {max_retries} attempts.")
    return False