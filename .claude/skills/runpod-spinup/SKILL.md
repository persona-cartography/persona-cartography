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
2. **`RUNPOD_API_KEY` in the repo `.env`.** The scripts auto-configure runpodctl
   from it on first use (`runpodctl config --apiKey …`, which also generates the
   ssh key at `~/.runpod/ssh/runpodctl-ssh-key`). Get the key from the RunPod
   console → Settings → API Keys.
3. For **bootstrapping** (private-repo clone): your GitHub-registered SSH key
   loaded in the local ssh-agent (`ssh-add -l` should list it). Agent forwarding
   carries it to the pod. macOS: `ssh-add --apple-use-keychain ~/.ssh/id_ed25519`.

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

```bash
cd .claude/skills/runpod-spinup

# Price-check only (no -y) — prints $/hr and exits:
./create-pod.sh oct "NVIDIA H200"

# After confirming cost with the user, create for real:
./create-pod.sh oct "NVIDIA H200" SECURE runpod-torch-v21 1 200 -y
```
Positional args: `<name> <gpu-id> [cloud=SECURE] [template=runpod-torch-v21] [gpu-count=1] [disk-gb=20]`.
For the OCT training stack size the disk up (≥200 GB) — model snapshots fill the
container disk fast (Llama-3.1-8B ≈ 16 GB; `No space left on device` is the tell).

`create-pod.sh` appends a `runpod-<name>` host to `~/.ssh/config` (and a shared
`Host runpod-*` defaults block: `User root`, the runpodctl key, `ForwardAgent
yes`), so `ssh runpod-<name>` just works.

### 2. Bootstrap for this repo (clone + setup)

```bash
./create-pod.sh oct "NVIDIA H200" SECURE runpod-torch-v21 1 200 -y --bootstrap
# …or separately, against an existing pod alias:
./bootstrap-pod.sh runpod-oct                 # current branch
./bootstrap-pod.sh runpod-oct refactor/main   # a specific branch
```
This clones the repo into `/workspace/<repo>` via SSH-agent forwarding, uploads
your local `.env`, and runs `scripts/setup.sh` (uv sync + `make oct-deps`) — the
same setup we run locally. **The branch must exist on origin** — push any local
work-in-progress branch first.

Then run the pipeline on the pod:
```bash
ssh runpod-oct 'cd /workspace/persona-shattering-lasr && \
  PY="uv run python" bash scripts/pipelines/run_persona_pipeline.sh --trait neuroticism --direction amp'
```

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
