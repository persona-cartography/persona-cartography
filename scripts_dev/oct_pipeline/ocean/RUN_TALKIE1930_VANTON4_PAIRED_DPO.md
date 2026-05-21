# OCT vanton4_paired_dpo on talkie-1930-13b-it — split-machine run guide

Train + evaluate 10 OCEAN persona LoRAs (5 traits × amplifier/suppressor) on
`talkie-lm/talkie-1930-13b-it` using the paired-teacher DPO flow. The 10 rows
are split across two GPU machines; both work in parallel from the same
upstream paired distillation data on HuggingFace.

## Split

| Machine | Traits                                    | LoRAs | Micro-batch | Script |
| ------- | ----------------------------------------- | ----- | ----------- | ------ |
| A       | Openness, Conscientiousness, Extraversion | 6     | 2 (H100)    | `run_machine_a_vanton4_paired_dpo_talkie1930.sh` |
| B       | Agreeableness, Neuroticism                | 4     | 4 (H200)    | `run_machine_b_vanton4_paired_dpo_talkie1930.sh` |

The two scripts write to disjoint monorepo paths so there's no collision
between machines.

## One-time prereq: seed paired distillation data on HF

Run **once**, on either machine (or any CPU-only box with `.env` configured).
It downloads the existing llama-3.1-8b-it vanton4 amp/sup teacher JSONLs,
joins them on `prompt`, and uploads paired DPO JSONLs +
`distillation_generation` stage markers under the talkie-1930-13b-it monorepo
prefix. CPU-only and quick (a few minutes).

```bash
bash scripts_dev/oct_pipeline/ocean/seed_all_vanton4_paired_dpo_talkie1930.sh --dry-run   # sanity check
bash scripts_dev/oct_pipeline/ocean/seed_all_vanton4_paired_dpo_talkie1930.sh
```

Once this finishes, both machine scripts can run independently — the OCT
pipeline's stage cache detects the seed markers and skips
`distillation_generation`.

## Machine A (this repo's current host, H100 80GB)

```bash
git checkout <branch-name>          # see commit log for the exact name
git pull
bash scripts_dev/oct_pipeline/ocean/run_machine_a_vanton4_paired_dpo_talkie1930.sh
```

## Machine B (H200 141GB) — instructions for the other Claude agent

You are picking up a split OCT training run that has already been set up on
the `main`-derived branch listed below. Six LoRAs are being trained on
another box; your job is to train the remaining four (Agreeableness ×2,
Neuroticism ×2) and run their evals.

### Pre-flight

1. `cd` into the repo. Make sure `.env` has at least `HF_TOKEN`,
   `OPENROUTER_API_KEY`, and (if used) `WANDB_API_KEY`.
2. Confirm OCT/OpenRLHF deps are layered into the uv env. If `uv run python
   scripts_dev/oct_pipeline/run_oct_pipeline.py --help` errors with "OCT not
   installed", layer them in per `scripts_dev/oct_pipeline/README.md`.
3. Check that the paired distillation seed has been uploaded for all four of
   your rows. They should be visible on HF at:
   - `fine_tuning/talkie-1930-13b-it/ocean/agreeableness/amplifier/vanton4_paired_dpo/.oct_pipeline/stages/distillation_generation.json`
   - `fine_tuning/talkie-1930-13b-it/ocean/agreeableness/suppressor/vanton4_paired_dpo/.oct_pipeline/stages/distillation_generation.json`
   - `fine_tuning/talkie-1930-13b-it/ocean/neuroticism/amplifier/vanton4_paired_dpo/.oct_pipeline/stages/distillation_generation.json`
   - `fine_tuning/talkie-1930-13b-it/ocean/neuroticism/suppressor/vanton4_paired_dpo/.oct_pipeline/stages/distillation_generation.json`

   If any are missing, run the seed script first:
   ```bash
   bash scripts_dev/oct_pipeline/ocean/seed_all_vanton4_paired_dpo_talkie1930.sh
   ```

### Run it

```bash
bash scripts_dev/oct_pipeline/ocean/run_machine_b_vanton4_paired_dpo_talkie1930.sh
```

This launches four `train → eval-trait → eval-mmlu` sequences (A+, A-, N+, N-).
Each trained LoRA is uploaded to the monorepo at:
`fine_tuning/talkie-1930-13b-it/ocean/<trait>/<direction>/vanton4_paired_dpo/lora/<constitution>-persona/`
and the eval outputs land at the matching `evals/mcq/{trait_logprobs,mmlu}/`.

### Tuning knobs

- Micro-batches are set to 4 (DPO and SFT) at the top of the script. If you
  OOM, drop to 2.
- Failures inside the loop are collected in `FAILED_STEPS` — the script
  continues past them so a single bad row doesn't kill the whole sweep.
- Each row's `scratch/oct_<trait>_<direction>_vanton4_paired_dpo_talkie1930/`
  output dir is local-only; the canonical artifact is the HF monorepo upload.

### What to do if something looks wrong

- **Tokenizer / chat-template error on first row**: `talkie-1930-13b-it` is a
  new model in this pipeline — if there's no built-in chat template, the
  fix is to add one via the tokenizer config. Flag this to the human; don't
  silently work around it.
- **Stage marker mismatch**: if `_ensure_stage_available` complains about a
  cache_key mismatch, the seed script likely wrote with a different prefix —
  verify the dest path matches `fine_tuning/talkie-1930-13b-it/...`.
- **DPO column missing**: if `load_dpo_pairs` errors saying
  `talkie-1930-13b-it` column is missing, the seed needs to be re-run with
  `--rejected-col talkie-1930-13b-it` (this is already wired into the seed
  script, so this would be a regression).

### When you're done

Push the branch (already pushed by the originating machine) and post a
summary of what trained successfully, what failed, and any warnings worth
the human's attention.
