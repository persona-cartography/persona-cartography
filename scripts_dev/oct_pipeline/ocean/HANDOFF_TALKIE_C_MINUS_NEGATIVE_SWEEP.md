# HANDOFF — talkie-1930 C- : fill in the NEGATIVE side of the dose-response curve (-2 … -1)

**Goal:** the fine LoRA-scale sweep for the talkie-1930-13b-it conscientiousness
suppressor (`-persona` adapter) currently has the POSITIVE side measured
(0 → +2) but the strong-negative cells (-1.0, -1.5, -2.0) came back NaN because
the adapter at strong negative scale makes talkie emit an empty turn for a
minority of prompts (~15% at -2), and the old sweep policy nan'd any cell with
*any* failure. The code fix is already committed; you just need to RUN it.

## What is already DONE (committed + pushed)
Branch: `mariia/talkie-c-minus-periodteacher`  (PR #312)
Repo:   https://github.com/SidBaines/persona-shattering-lasr
Branch link: https://github.com/SidBaines/persona-shattering-lasr/tree/mariia/talkie-c-minus-periodteacher

- `src_dev/sweep.py` — opt-in `ExperimentConfig.max_failed_fraction` (default 0.0
  = strict, backward-compatible). When set, a cell tolerates up to that fraction
  of failed (empty) conversations and JUDGES the ones that succeeded. Failed
  convos append no assistant message, so they are cleanly dropped from judging
  (never scored as 0). Logged as `[tolerated] ... dropped N/240`.
- `scripts_dev/evals/llm_judge_sweep/runner_cells.py` — threads
  `MAX_FAILED_FRACTION` from the config → `NormalisedConfig` → `ExperimentConfig`.
- `c_minus_periodteacher_persona_fine.py` config — sets `MAX_FAILED_FRACTION=0.25`
  and the fine grid `SCALE_POINTS=[-2,-1.5,-1,-0.5,0,0.5,1,1.5,2]`, `MAX_SAMPLES=240`.
- `src_dev/models/talkie/materialize.py` — the eos fix (stop on `<|end|>`=65536,
  not just `<|endoftext|>`=65535). REQUIRED — without it talkie never stops.
- `src_dev/common/lora_catalogue.py` — `LoraHFCatalogue.talkie1930_c_minus` pointer.
- vLLM `trust_remote_code` plumbing for the custom talkie architecture.

## Adapters (already on HF — verified, weights present)
Repo: `persona-shattering-lasr/monorepo` (HF dataset)
Dir:  `fine_tuning/talkie-1930-13b-it/ocean/conscientiousness/suppressor/vanton4_paired_dpo_periodteacher/lora/`
  - `conscientiousness_suppressing_full_vanton4_period-persona`  ← THE ONE TO SWEEP (SFT-merged)
  - `conscientiousness_suppressing_full_vanton4_period-dpo`
  - `conscientiousness_suppressing_full_vanton4_period-sft`

## Run it (eval only — NO training, NO OCT overlay needed)

### 1. Spin a pod  (H100 SXM worked; A100 SXM SECURE gave dead hosts this session)
```
runpodctl pod create ... "NVIDIA H100 80GB HBM3" SECURE runpod-torch-v21  1  150   # 150GB disk
```
**GOTCHA THAT COST AN HOUR:** after create, the SSH endpoint is in the TOP-LEVEL
`ssh` block of `runpodctl pod get <id> -o json` (`.ssh.ip`, `.ssh.port`,
`.ssh.ssh_command`) — NOT `.runtime.ports`, which stays `None`/empty for 10+ min
while sshd is ALREADY reachable. Poll the `.ssh` block and just try to connect.

### 2. Clone the branch  (gh is NOT on the local mac PATH; token is in osxkeychain)
```
# local: extract token
TOKEN=$(printf "protocol=https\nhost=github.com\n\n" | git credential fill | sed -n 's/^password=//p')
# pipe over stdin to pod, shallow clone ONLY (do NOT `git pull` on the pod — it
# hangs prompting for creds; set GIT_TERMINAL_PROMPT=0):
printf '%s\n' "$TOKEN" | ssh ... '
  read -r T; git config --global credential.helper "!f(){ echo username=x-access-token; echo password='"$T"'; };f"
  cd /workspace && git clone --depth 1 --branch mariia/talkie-c-minus-periodteacher \
     https://github.com/SidBaines/persona-shattering-lasr.git
  git config --global --unset credential.helper'
```

### 3. .env  (keys live in the MAIN checkout, not the worktree)
Copy `/Users/mariiakoroliuk/persona-shattering-lasr/.env` to the pod repo root.
Needs `OPENROUTER_API_KEY` (Qwen3-235B judge) + `HF_TOKEN` (monorepo).

### 4. Deps + materialize the eos-fixed base model
```
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv not preinstalled on torch-v21
cd /workspace/persona-shattering-lasr
export GIT_TERMINAL_PROMPT=0
uv sync
export OCT_MODEL_PATH=/root/.cache/models
uv run python -m src_dev.models.talkie.materialize --out /root/.cache/models/talkie-1930-13b-it
# sanity: config.json eos_token_id must be [65536, 65535]
```

### 5. Run the fine sweep  (positives hydrate from HF; only -1/-1.5/-2 re-run)
```
export OCT_MODEL_PATH=/root/.cache/models
uv run python -m scripts_dev.evals.llm_judge_sweep.runner_cells \
  --config scripts_dev.evals.llm_judge_sweep.configs.vanton4_paired_dpo_talkie1930.c_minus_periodteacher_persona_fine \
  --allow-custom-fingerprint
```
Run inside tmux; it's ~15-20 min after the model is up. The cached positive
cells under fingerprint `866790dfd2` hydrate instantly; the negative cells
generate (some empties, tolerated) → judge → re-analyze → re-plot → re-upload.

## Output (auto-uploaded to monorepo)
`fine_tuning/talkie-1930-13b-it/ocean/conscientiousness/suppressor/vanton4_paired_dpo_periodteacher/evals/llm_judge_lora_scale_sweep/866790dfd2/`
- `plots/llm_judge_scale_sweep.png`  ← full -2…+2 curve (GIVE THIS TO THE USER, rendered)
- `analysis/grid_summary.jsonl`      ← per-scale mean + CI + n (n<240 ⇒ dropout)

When done: render the plot for the user, report the per-scale dropout
(240 − n judged) for the negative cells, and TEAR DOWN THE POD.

## Known result so far (positive side, eos-fixed, n=240)
| scale | Conscientiousness | Coherence |
|---|---|---|
| -0.5 | +1.50 | 8.75 |
| 0.0  | +1.62 | 8.86 |
| +0.5 | +1.63 | 8.82 |
| +1.0 | +1.44 | 8.40 |
| +1.5 | +0.96 | 7.50 |
| +2.0 | +0.33 | 5.74 |
Clear dose-response suppression on the + side. The - side is what's missing.
