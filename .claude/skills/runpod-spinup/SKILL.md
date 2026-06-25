---
name: runpod-spinup
description: Spin up / tear down a RunPod GPU pod and bootstrap it for this repo (clone + setup). Generalised — no hard-coded account; reads keys from the repo .env.
---

# RunPod Pod Spinup Skill

Create GPU (or CPU) pods on RunPod via the `runpodctl` CLI, then bootstrap them
for **this repo** (clone + `scripts/setup.sh`). Generalised from a colleague's
machine-specific original: nothing is hard-coded to a user / email / home path —
it keys off `$HOME`, the repo's `.env`, and runpodctl's default config under
`~/.runpod/`. Works on macOS and Linux.

## IMPORTANT: Cost check (mandatory)

**Before creating any GPU pod, tell the user the $/hr and get explicit
confirmation.** This is enforced mechanically: `create-pod.sh` does nothing
without `-y/--yes` — without it, it just prints the GPU's price and exits. Pull
live prices with `runpodctl gpu list`.

## IMPORTANT: Shared-account pod caps

The RunPod account is **shared**. Too many concurrent pods overloads the GPUs
*and* rate-limits the monorepo (HF returns 429 "maximum queue size reached" on
both reads and writes once ~10 pods hammer it). `create-pod.sh` enforces:

- **Hard account cap** `RUNPOD_MAX_TOTAL_PODS` (default **10**): refuses to
  create if the account already has that many pods.
- **Soft per-owner cap** `RUNPOD_MAX_OWN_PODS` (default **5**): refuses if *you*
  (the `<owner>-` name prefix) already have that many — so two people can run
  side-by-side (5 + 5 = 10). Pass **`--override`** to use more of the free slots.

It prints `account: N/10 total (M free); you: K/5` before creating, so you know
how many more you can spin up. For a fleet, launch in batches of ≤5 (or use
`--override` when the account is quiet) — never blast 9+ at once (that was what
caused the monorepo 429 cascade).

## IMPORTANT: Spend discipline — launch sequentially, never race (mandatory)

These are expensive GPUs (an H200 is ~$96/day) on a shared bill. Two failure
modes have actually wasted money here, so guard against both:

- **No racy background pool loops.** Do **not** background a loop that auto-fires
  `create-pod.sh` whenever a slot frees. On 2026-06-16 a pool's "stop and relaunch"
  raced a backgrounded create and produced **two identical 32B H200 pods** (~$4/hr
  each) for one job. If you want a rolling fleet, launch the next pod **by hand**
  after confirming a slot is free — one verified launch at a time.
- **Confirm each pod registers before launching the next.** After a create,
  `runpodctl get pod | grep <name>` should show exactly one row before you start
  another. Never run multiple `create-pod.sh` concurrently for related jobs.
- **`create-pod.sh` now refuses a duplicate name** (a pod already named `<name>`
  → exit 3). This mechanically blocks the common double-launch, but it is a
  backstop, not a licence to fire-and-forget in parallel — there is still a race
  window before the first pod is visible.
- **Killing mid-training wastes the compute already paid for.** Prefer letting a
  training pod finish (or self-terminate via `--shutdown`) over killing it; only
  kill pods that are idle, duplicated, or doomed to fail.
- **Test one before fanning out.** When debugging a new config (OOM, a flag,
  a model), get **one** pod working end-to-end before launching the rest —
  don't chain launches of an unproven config.

### GPU pricing (indicative, secure cloud, 1× GPU — verify live)

| GPU | VRAM | ~$/hr | ~$/day |
|-----|------|-------|--------|
| RTX 4090 | 24 GB | $0.69 | $17 |
| A100 80GB | 80 GB | $1.89 | $45 |
| H100 SXM | 80 GB | $2.99 | $72 |
| H200 SXM | 141 GB | $3.99 | $96 |
| B200 | 180 GB | $5.99 | $144 |

Stopped pods are still billed for storage.

## Prerequisites (one-time)

1. **`runpodctl` installed.** It's a single Go binary. If `brew install
   runpod/runpodctl/runpodctl` fails (e.g. outdated Command Line Tools), use the
   no-compile binary install instead:
   ```bash
   curl -sL cli.runpod.net | sudo bash        # installs to /usr/local/bin
   # or, no sudo — drop the darwin-arm64 binary on PATH:
   #   download from https://github.com/runpod/runpodctl/releases
   #   chmod +x runpodctl && mv runpodctl ~/.local/bin/
   ```
   The scripts also look in `/opt/homebrew/bin`, `/usr/local/bin`,
   `~/.local/bin`, and `~/go/bin`, so it doesn't have to be on PATH.
2. **Repo `.env` with all three keys.** Start from the template and fill in:
   ```bash
   cp .env.example .env   # at the repo root; .env is gitignored — never commit it
   ```
   | Key | Used for | Where to get it |
   |-----|----------|-----------------|
   | `RUNPOD_API_KEY` | creating/terminating pods (runpodctl auto-configures from it on first use, which also generates `~/.runpod/ssh/runpodctl-ssh-key`) | RunPod console → Settings → API Keys (the team may share one account — pods and balance are then shared) |
   | `HF_TOKEN` | model downloads + **write** access to `persona-cartography/monorepo` (adapters, evals, logs all upload there) | huggingface.co → Settings → Access Tokens; ask the team for org write access |
   | `OPENROUTER_API_KEY` | the teacher pass (training) and the LLM judges (evals) | openrouter.ai → Keys |

   `bootstrap-pod.sh` uploads your local `.env` to the pod, so it only needs to
   be correct locally. Verify before first use:
   ```bash
   python -c "from dotenv import dotenv_values; e=dotenv_values('.env'); \
     missing=[k for k in ('RUNPOD_API_KEY','HF_TOKEN','OPENROUTER_API_KEY') if not e.get(k)]; \
     print('MISSING: '+', '.join(missing) if missing else '.env OK')"
   ```
3. For **bootstrapping** (private-repo clone): your GitHub-registered SSH key
   loaded in the local ssh-agent (`ssh-add -l` should list it). Agent forwarding
   carries it to the pod. macOS: `ssh-add --apple-use-keychain ~/.ssh/id_ed25519`.
   Verify: `ssh -T git@github.com` should greet you by username (not
   "Permission denied"); if it fails, see the token fallback in SSH notes below.

## Files

| Script | Purpose |
|--------|---------|
| `create-pod.sh` | create a pod (+ `ssh runpod-<name>` alias). General/reusable. |
| `bootstrap-pod.sh` | clone this repo onto the pod, upload `.env`, run `scripts/setup.sh`. |
| `watch-run.sh` | health-watch a detached run: exits on pod-gone / real log error / dead process. |
| `cleanup-pod.sh` | delete a pod + remove its ssh alias. |
| `_common.sh` | shared helpers (locate runpodctl, read `.env`, configure, ssh defaults). |

## Usage

### 1. Create (with mandatory cost confirmation)

**Pod naming rule: names must be self-explanatory.** Pods live in a shared
team account — anyone looking at the console must be able to tell whose pod it
is and what it's doing. Use `<owner>-<what-it-runs>`, e.g.
`anton-openness-amp-dsv32-teacher-llama8b`, not `oct` or `test`. The name also
becomes the ssh alias (`runpod-<name>`), so it pays off locally too.

**Get the name right at creation — NEVER rename a pod with work in flight.**
A REST `PATCH /v1/pods/<id>` rename REDEPLOYS the container: the ssh port
changes, `/workspace` is wiped, and running processes are killed (learned the
hard way 2026-06-11 — it destroyed a bootstrapped repo and a running pipeline).
If a busy pod is misnamed, leave it until the run finishes.

```bash
cd .claude/skills/runpod-spinup

# Price-check only (no -y) — prints $/hr and exits:
./create-pod.sh anton-neuro-amp-train "NVIDIA H200"

# After confirming cost with the user, create for real:
./create-pod.sh anton-neuro-amp-train "NVIDIA H200" SECURE runpod-torch-v21 1 200 -y
```
Positional args: `<name> <gpu-id> [cloud=SECURE] [template=runpod-torch-v21] [gpu-count=1] [disk-gb=20]`.
For the OCT training stack size the disk up (≥200 GB) — model snapshots fill the
container disk fast (Llama-3.1-8B ≈ 16 GB; `No space left on device` is the tell).

`create-pod.sh` appends a `runpod-<name>` host to `~/.ssh/config` (and a shared
`Host runpod-*` defaults block: `User root`, the runpodctl key, `ForwardAgent
yes`), so `ssh runpod-<name>` just works.

### 2. Bootstrap for this repo (clone + setup)

```bash
./create-pod.sh anton-neuro-amp-train "NVIDIA H200" SECURE runpod-torch-v21 1 200 -y --bootstrap
# …or separately, against an existing pod alias:
./bootstrap-pod.sh runpod-anton-neuro-amp-train                 # current branch
./bootstrap-pod.sh runpod-anton-neuro-amp-train refactor/main   # a specific branch
```
This clones the repo into `/workspace/<repo>` via SSH-agent forwarding, uploads
your local `.env`, and runs `scripts/setup.sh` (uv sync + `make oct-deps`) — the
same setup we run locally. **The branch must exist on origin** — push any local
work-in-progress branch first.

Then run the pipeline on the pod. The canonical pattern is **fire-and-forget**:
launch detached with `--shutdown` so the run survives without a live SSH
session, uploads its logs to the monorepo (`<run_prefix>/.logs/`), and
self-terminates the pod when done (success or failure):

```bash
ssh runpod-oct 'cd /workspace/persona-shattering-lasr && \
  nohup env PY="uv run python" bash scripts/pipelines/run_persona_pipeline.sh \
    --trait openness --direction amp \
    --evals "trait mmlu judge" --judge-metrics ocean5 \
    --shutdown > /workspace/run.log 2>&1 & \
  echo "launched pid $!"'
```

Add `--model <slug>` (e.g. `gemma-3-27b-it`) for a non-llama base model, or
`--version <segment>` for a non-canonical monorepo version. For a synchronous
run (small tests), drop `nohup`/`--shutdown` and keep the ssh session open:

```bash
ssh runpod-oct 'cd /workspace/persona-shattering-lasr && \
  PY="uv run python" bash scripts/pipelines/run_persona_pipeline.sh --trait neuroticism --direction amp'
```

After a fire-and-forget launch, arm `watch-run.sh` (next section) and check
results on the monorepo under `fine_tuning/{model}/ocean/{trait}/{direction}/{version}/`.

### 2b. Watch a fire-and-forget run

For detached runs (`nohup … --shutdown … &`), don't hand-roll polling loops —
use the watcher. It polls every 10 min (configurable) and **exits** on the
first of: pod gone (run finished + self-terminated), a *real* error in the run
log (Traceback / CUDA OOM / disk full / upload failed / Killed — the benign
DeepSpeed `__del__` teardown traceback is filtered out), or the pipeline
process dying while the pod stays up. Launch it as a background task so the
exit wakes the agent:

```bash
.claude/skills/runpod-spinup/watch-run.sh <pod-id> runpod-<name> /workspace/run.log [proc-pattern] [interval-s]
# defaults: proc-pattern=run_persona_pipeline, interval=600
```

Exit codes: 0 = pod gone (normal end), 1 = errors detected, 2 = process dead.
Transient ssh/runpodctl failures never false-alarm — pod existence via
runpodctl is the authority, and an empty `runpodctl get pod` is retried.

### 3. Tear down

```bash
./cleanup-pod.sh <pod-id>          # list ids: runpodctl pod list
runpodctl pod stop <pod-id>        # or just pause GPU billing (storage still charged)
```

## SSH notes

- **Prefer direct `ssh runpod-<name>`** over `runpodctl ssh connect` (the wrapper
  can hang for minutes after creation). sshd is usually up ~10–30 s after create.
- Poll for readiness:
  ```bash
  until ssh -o ConnectTimeout=5 runpod-<name> 'echo ready' 2>/dev/null | grep -q ready; do sleep 5; done
  ```
- **Clone auth.** Default is SSH-agent forwarding (no tokens on the pod). If your
  key isn't registered with GitHub (`ssh -T git@github.com` → "Permission
  denied"), fall back to a token over **stdin** (never in argv / the clone URL):
  ```bash
  gh auth token | ssh runpod-<name> '
    read -r T; git config --global credential.helper "!f(){ echo username=x-access-token; echo password=$T; }; f"
    cd /workspace && git clone https://github.com/OWNER/REPO.git
    git config --global --unset credential.helper'
  ```

## Cloud types

- `SECURE` (default) — RunPod datacenters, reliable, slightly pricier.
- `COMMUNITY` — third-party hosts, cheaper, patchier. `create-pod.sh` auto-falls
  back COMMUNITY→SECURE on "does not have the resources".

## Troubleshooting

- **`runpodctl` install fails under brew ("Command Line Tools too outdated")** —
  use the no-compile binary install above (`curl -sL cli.runpod.net | sudo bash`
  or the GitHub-releases binary). Don't block on updating Xcode CLT.
- **"does not have the resources" / "no instances available"** — that GPU/cloud
  is out of stock. Retry SECURE, or pick another `available: true` GPU from
  `runpodctl gpu list`. A100/H100/H200 SXM are usually more available than 4090.
- **SSH refused 5+ min although pod is RUNNING** — almost always a pod created
  with a bare `--image` and no template (no start script runs sshd). Use
  `create-pod.sh` (it passes `--template-id`).
- **`No space left on device` / `Background writer channel closed`** — container
  disk full. Recreate with a bigger `disk-gb` (6th positional arg).
- **`runpodctl` commands default to JSON** (`-o json`); use `-o yaml` for YAML.

## Useful commands

```bash
runpodctl pod list / pod get <id>      # pods
runpodctl me                           # active account + balance/spend
runpodctl gpu list                     # GPU types, availability, price
runpodctl template search <q>          # templates
```
