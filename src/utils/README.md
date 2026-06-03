# `src/utils`

Stable, widely-used utility helpers. This is the most-depended-on package in
`src/` (LoRA arithmetic alone has 17+ import sites across the repo).

## Contents

| File | What it is |
|------|------------|
| `peft_manipulations.py` | Reversible, in-place LoRA modifiers (rank reduction, scaling, layer zeroing, composition). All snapshot state at init; `restore()` is idempotent. |
| `lora_vector_utils.py` | Memory-efficient LoRA operations in factored `(B, A)` form (dot products, addition, scaling). |
| `lora_baking.py` | Pre-bake LoRA scale factors into adapter weights on disk (for vLLM sweeps). |
| `linalg.py` | Linear-algebra helpers (e.g. memory-efficient rank reduction via QR + SVD on the small `r×r` core). |
| `model_layer_info.py` | Model-layer inspection utilities. |
| `hf_hub.py` | Upload/download helpers for the HuggingFace monorepo: conflict-retrying uploads with extended timeouts, subtree-aware downloads, existence checks. No `src_dev` dependencies. |

## `hf_hub.py` quick reference

- **Upload:** `upload_file_to_dataset_repo`, `upload_folder_to_dataset_repo`,
  `upload_folder_to_model_repo` — create the repo if needed, retry on HF 412
  conflicts (concurrent commits), use extended timeouts for slow uploads.
- **Download:** `download_from_dataset_repo` (replicates repo path structure),
  `download_path_to_dir` (strips the prefix — rehydrate a run dir into its
  original local layout), `download_file_from_dataset_repo`,
  `download_dataset_subpath`.
- **Probe:** `check_exists_in_dataset_repo` / `dataset_repo_subpath_exists` —
  check a path without downloading.
- **Auth:** `login_from_env` sets `HF_TOKEN` from env without a `whoami` call
  (avoids HF's rate limit on that endpoint).

Migrated as-is from `src_dev/utils/hf_hub.py` (Slice 1a); the dev copy is kept
for in-flight callers.
