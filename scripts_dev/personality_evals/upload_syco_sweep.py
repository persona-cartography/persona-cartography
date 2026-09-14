"""Upload the sycophancy scale-sweep results to the monorepo.

run_sycophancy_vllm writes results locally but does not upload; this pushes
each direction's whole sweep dir (all scales nested by scale tag) to
fine_tuning/llama-3.1-8b-it/other/sycophancy/<direction>/vsyco1_paired_dpo/
evals/mcq/sycophancy_sweep/ on persona-shattering-lasr/monorepo.
Idempotent: HF skips identical files.
"""

from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
REPO = "persona-shattering-lasr/monorepo"
ROOT = Path("scratch/evals/sycophancy_adapter/sycophancy_sweep")
DIRECTION = {"syco_plus": "amplifier", "syco_minus": "suppressor"}


def main() -> None:
    api = HfApi()
    if not ROOT.exists():
        print(f"no sweep results at {ROOT}")
        return
    for run_dir in sorted(ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        label = run_dir.name.split("_syco1")[0]
        direction = DIRECTION[label]
        dest = (
            f"fine_tuning/llama-3.1-8b-it/other/sycophancy/{direction}/"
            f"vsyco1_paired_dpo/evals/mcq/sycophancy_sweep/{run_dir.name}"
        )
        print(f"uploading {run_dir} -> {dest}")
        api.upload_folder(
            repo_id=REPO, repo_type="dataset",
            folder_path=str(run_dir), path_in_repo=dest,
        )
        print("  done")
    print("UPLOAD_SYCO_SWEEP_OK")


if __name__ == "__main__":
    main()
