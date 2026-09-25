# src_dev

In-development component implementations. Code here graduates to `src/` once mature.

---

## Dev Environment Setup

Run once when you first clone the repo or spin up a new container:

```bash
bash scripts_dev/setup_dev.sh
```

This will:
- Install dependencies (`uv sync --extra dev`)
- Copy `.env.example` → `.env` (if missing) — fill in your API keys after
- Install VS Code extensions (Jupyter, Claude Code)

If `# %%` cells aren't working after that, open the Command Palette and run
`Jupyter: Select Kernel` — the kernel cache doesn't always refresh automatically.

---

## Managing Dependencies

All shared dev dependencies live in the `dev` optional group in `pyproject.toml`.
Sync with:

```bash
uv sync --extra dev
```

If you need packages that are specific to your work, add a personal group in
`pyproject.toml` and sync both:

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0", "ruff>=0.3", "ipykernel>=6.0"]
user-1 = ["some-package>=1.0"]
```

```bash
uv sync --extra dev --extra user-1
```

---

## Dataset APIs

Use `src_dev.datasets` as the single dataset module for both:
- Canonical run storage/lineage operations (ingest, materialize, export, etc.)
- Dataset loading/formatting helpers (`load_dataset_from_config`, `format_for_inference`)

The legacy `data_loading` package has been removed.

---

## Git Authentication

VS Code normally handles git credentials via a socket, but that socket can go
stale in remote/container sessions. If `git push` fails with `ECONNREFUSED` on
a `vscode-git-*.sock`, use SSH instead.

### Already have a key locally?

Copy it into the pod, then switch the remote:

The pod address is `root@<PUBLIC_IP> -p <SSH_PORT>` — grab both from your
RunPod dashboard.

```bash
# From your local machine:
scp -P <SSH_PORT> ~/.ssh/id_ed25519 root@<PUBLIC_IP>:~/.ssh/id_ed25519

# Inside the pod:
chmod 600 ~/.ssh/id_ed25519
git remote set-url origin https://github.com/persona-cartography/persona-cartography.git
```

### No key yet?

Generate one, upload it to GitHub, then switch the remote:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""

# Print and add to https://github.com/settings/keys/new
cat ~/.ssh/id_ed25519.pub

git remote set-url origin https://github.com/persona-cartography/persona-cartography.git
```

Keys don't expire — once it's on GitHub, reuse it across pods forever via `scp`.
