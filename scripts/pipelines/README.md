# Pipelines — end-to-end run surface

One script per end-to-end workflow that chains existing components (training,
evals) into a single command. These are thin orchestrators: they shell out to
the per-component run surfaces under `scripts/training/` and `scripts/evals/`,
they don't reimplement anything.

## `run_persona_pipeline.sh` — train + eval one OCEAN trait/direction

Runs the full **paired-teacher DPO** training for one trait + direction, then
(by default) the MMLU capability and TRAIT-logprob MCQ evals on the adapter it
just produced.

```bash
# train neuroticism amplifier, then run both default evals (trait + mmlu)
scripts/pipelines/run_persona_pipeline.sh --trait neuroticism --direction amp

# only the trait eval; suppressor direction
scripts/pipelines/run_persona_pipeline.sh --trait openness --direction sup --evals trait

# re-run evals only against an already-trained adapter
scripts/pipelines/run_persona_pipeline.sh --trait agreeableness --direction amp --skip-training
```

Flags: `--trait` (openness|conscientiousness|extraversion|agreeableness|neuroticism),
`--direction` (amp|sup), `--evals "trait mmlu"` (subset/order), `--skip-training`,
`--skip-evals`, `--dry-run` (passthrough to training; skips evals), plus
`--skip-sft` / `--teacher-model` forwarded to the training launcher.

The training-only launcher it wraps —
[`scripts/training/ocean_paired_dpo/run_pipeline.sh`](../training/ocean_paired_dpo/run_pipeline.sh)
— can also be run directly. It runs steps 01+02 for **both** poles (the pairing
needs both teachers) then 03/04/05 for the chosen direction; cross-machine /
cross-direction reuse comes from each step fetching-before-generate and
uploading-after-generate against the monorepo.

Override the interpreter with `PY` (e.g. `PY="uv run python"`).
