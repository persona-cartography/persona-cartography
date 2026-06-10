"""One-off HF monorepo migration: vanton4_paired_dpo -> ocean_const_paired_dpo.

Copies (NOT moves) every in-scope path to its renamed location as a single git
commit built purely from tree objects — no blob content is downloaded or
uploaded (existing blob SHAs are reused via `git update-index --index-info`,
so LFS payloads and small files alike transfer nothing).

Scope (user decision 2026-06-10): llama-3.1-8b-it + gemma-3-27b-it fine_tuning
version dirs and llama combos. talkie-1930-13b-it and all frozen versions
(vanton4, v4*, vanton4_rank*, vanton4_seed*, v1, vanton1) untouched.

Mapping:
  1. path component  vanton4_paired_dpo  ->  ocean_const_paired_dpo
  2. within renamed trees, constitution suffix `_vanton4` (before . - _ / or
     end) is stripped, e.g. openness_amplifying_full_vanton4-persona ->
     openness_amplifying_full-persona

Phases (run from the repo root):
  python scripts_dev/hf_rename_vanton4_paired_dpo.py plan    # manifest + collision check, no writes
  python scripts_dev/hf_rename_vanton4_paired_dpo.py copy    # build + push the copy commit
  # deletion of originals is deliberately NOT implemented here yet — gated on
  # explicit user approval after verification (see the plan file).

Requires a blobless clone at scratch/monorepo_rename (created by the caller):
  GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout <url> scratch/monorepo_rename
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

CLONE = Path("scratch/monorepo_rename")
MANIFEST = Path("scratch/hf_rename_manifest.json")

OLD = "vanton4_paired_dpo"
NEW = "ocean_const_paired_dpo"
SUFFIX_RE = re.compile(r"_vanton4(?=[._\-/]|$)")

# In-scope roots (verified by HF listing 2026-06-10).
SCOPE_PREFIXES = [
    "fine_tuning/llama-3.1-8b-it/ocean/",
    "fine_tuning/llama-3.1-8b-it/other/ocean_def_control/amplifier/",
    "fine_tuning/gemma-3-27b-it/ocean/",
    "fine_tuning/gemma-3-27b-it/other/ocean_def_control/amplifier/",
    "combos/llama-3.1-8b-it/",
]


def git(*args: str, input_: str | None = None) -> str:
    res = subprocess.run(
        ["git", *args], cwd=CLONE, input=input_, capture_output=True, text=True
    )
    if res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])}... failed:\n{res.stderr[:2000]}")
    return res.stdout


def in_scope(path: str) -> bool:
    if OLD not in path:
        return False
    return any(path.startswith(p) for p in SCOPE_PREFIXES)


def new_path(path: str) -> str:
    out = path.replace(OLD, NEW)
    return SUFFIX_RE.sub("", out)


def load_tree() -> list[tuple[str, str, str]]:
    """All (mode, sha, path) blobs at origin/main."""
    out = git("ls-tree", "-r", "-z", "origin/main")
    entries = []
    for rec in out.split("\0"):
        if not rec:
            continue
        meta, p = rec.split("\t", 1)
        mode, otype, sha = meta.split()
        if otype == "blob":
            entries.append((mode, sha, p))
    return entries


def build_manifest() -> dict:
    entries = load_tree()
    all_paths = {p for _, _, p in entries}
    moves, collisions = [], []
    for mode, sha, p in entries:
        if not in_scope(p):
            continue
        np_ = new_path(p)
        if np_ == p:
            raise RuntimeError(f"mapping produced identity for {p}")
        (collisions if np_ in all_paths else moves).append(
            {"old": p, "new": np_, "mode": mode, "sha": sha}
        )
    roots = sorted({m["old"].split(OLD)[0] + OLD for m in moves})
    return {"moves": moves, "collisions": collisions, "roots": roots}


def cmd_plan() -> dict:
    man = build_manifest()
    MANIFEST.write_text(json.dumps(man, indent=1))
    print(f"files to copy : {len(man['moves'])}")
    print(f"collisions    : {len(man['collisions'])}")
    for c in man["collisions"][:10]:
        print("  COLLISION:", c["old"], "->", c["new"])
    print(f"version roots : {len(man['roots'])}")
    print("sample mappings:")
    for m in man["moves"][:6]:
        print(f"  {m['old']}\n    -> {m['new']}")
    return man


def cmd_copy() -> None:
    man = json.loads(MANIFEST.read_text())
    if man["collisions"]:
        sys.exit("refusing to copy: collisions present (run `plan` and inspect)")
    git("fetch", "origin", "main")
    git("read-tree", "origin/main")
    lines = "".join(
        f"{m['mode']} {m['sha']}\t{m['new']}\n" for m in man["moves"]
    )
    git("update-index", "--add", "--index-info", input_=lines)
    # --missing-ok: the clone is blobless (--filter=blob:none); index entries
    # reference blob SHAs that exist remotely but not locally, and without the
    # flag write-tree's existence check lazy-fetches every one of them.
    tree = git("write-tree", "--missing-ok").strip()
    parent = git("rev-parse", "origin/main").strip()
    msg = (
        f"rename {OLD} -> {NEW} (copy phase; originals retained pending "
        "verification)\n\nScope: llama-3.1-8b-it + gemma-3-27b-it fine_tuning "
        "+ llama combos. Pure tree-level copy (blob SHAs reused; no content "
        "transferred). Constitution `_vanton4` suffixes normalized inside "
        "renamed trees."
    )
    commit = git("commit-tree", tree, "-p", parent, "-m", msg).strip()
    print("commit:", commit)
    git("push", "origin", f"{commit}:refs/heads/main")
    print(f"pushed copy commit with {len(man['moves'])} new paths")


# ---------------------------------------------------------------------------
# API fallback: HF's git backend 500s on promisor fetches, so the pure-git
# push can't complete. Same manifest, executed via create_commit batches:
# LFS files -> server-side CommitOperationCopy (no content transfer);
# non-LFS  -> hf_hub_download + CommitOperationAdd.
# ---------------------------------------------------------------------------

REPO_ID = "persona-shattering-lasr/monorepo"


def cmd_copy_api() -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import (
        CommitOperationAdd,
        CommitOperationCopy,
        HfApi,
        hf_hub_download,
    )

    from src.utils.hf_hub import login_from_env

    login_from_env()
    api = HfApi()
    man = json.loads(MANIFEST.read_text())
    moves = man["moves"]

    # LFS flags + existing targets, from one recursive listing per top scope.
    print("listing repo state (lfs flags + already-copied targets) ...", flush=True)
    lfs_by_path: dict[str, bool] = {}
    existing: set[str] = set()
    tops = sorted({m["old"].split("/", 1)[0] for m in moves})  # fine_tuning, combos
    for top in tops:
        for e in api.list_repo_tree(REPO_ID, path_in_repo=top, repo_type="dataset",
                                    recursive=True):
            if e.__class__.__name__ == "RepoFile":
                lfs_by_path[e.path] = e.lfs is not None
                existing.add(e.path)

    todo = [m for m in moves if m["new"] not in existing]
    copies = [m for m in todo if lfs_by_path.get(m["old"])]
    adds = [m for m in todo if not lfs_by_path.get(m["old"])]
    missing = [m for m in todo if m["old"] not in lfs_by_path]
    if missing:
        sys.exit(f"{len(missing)} manifest paths not found on remote, e.g. "
                 f"{missing[0]['old']} — re-run `plan`.")
    print(f"todo={len(todo)} (lfs copies={len(copies)}, small adds={len(adds)}, "
          f"already done={len(moves) - len(todo)})", flush=True)

    def commit(ops, msg):
        for attempt in range(5):
            try:
                api.create_commit(repo_id=REPO_ID, repo_type="dataset",
                                  operations=ops, commit_message=msg)
                return
            except Exception as exc:
                if attempt == 4:
                    raise
                wait = 15 * (attempt + 1)
                print(f"  commit failed ({type(exc).__name__}: {str(exc)[:120]}); "
                      f"retry in {wait}s", flush=True)
                time.sleep(wait)

    # Phase A: LFS copies, server-side, large batches.
    BATCH = 500
    for i in range(0, len(copies), BATCH):
        chunk = copies[i:i + BATCH]
        ops = [CommitOperationCopy(src_path_in_repo=m["old"], path_in_repo=m["new"])
               for m in chunk]
        commit(ops, f"rename {OLD}->{NEW}: copy LFS batch {i // BATCH + 1} "
                    f"({len(chunk)} files)")
        print(f"  lfs copy {min(i + BATCH, len(copies))}/{len(copies)}", flush=True)
        time.sleep(2)

    # Phase B: small non-LFS files — parallel download, then batched adds.
    def fetch(m):
        for attempt in range(4):
            try:
                local = hf_hub_download(REPO_ID, m["old"], repo_type="dataset")
                return m, local
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))

    # Byte-aware batches: <=300 files AND <=32MB payload per commit (regular
    # files travel base64-inline in the commit call).
    import os as _os
    batches, cur, cur_bytes = [], [], 0
    sizes = {}  # populated lazily after download
    BATCH_ADD, BATCH_BYTES = 300, 32 * 1024 * 1024
    done = 0
    i = 0
    while i < len(adds):
        chunk = adds[i:i + BATCH_ADD]
        with ThreadPoolExecutor(max_workers=8) as pool:
            fetched = list(pool.map(fetch, chunk))
        # split this chunk further by byte budget
        sub, sub_bytes = [], 0
        flushes = []
        for m, local in fetched:
            sz = _os.path.getsize(local)
            if sub and sub_bytes + sz > BATCH_BYTES:
                flushes.append(sub); sub, sub_bytes = [], 0
            sub.append((m, local)); sub_bytes += sz
        if sub:
            flushes.append(sub)
        for grp in flushes:
            ops = [CommitOperationAdd(path_in_repo=m["new"], path_or_fileobj=local)
                   for m, local in grp]
            commit(ops, f"rename {OLD}->{NEW}: copy batch ({len(grp)} files)")
            done += len(grp)
        print(f"  adds {min(i + BATCH_ADD, len(adds))}/{len(adds)}", flush=True)
        time.sleep(2)
        i += BATCH_ADD

    print("copy-api complete.")


def cmd_baselines() -> None:
    """Migrate the legacy shared baseline store into the canonical one.

    combos/llama-3.1-8b-it/_baseline/{eval}/{fp}/** ->
    evals/baselines/llama-3.1-8b-it/{eval}/{fp}/**   (additive; skips fps that
    already exist in the new store, e.g. those written by post-refactor runs).
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import (
        CommitOperationAdd,
        CommitOperationCopy,
        HfApi,
        hf_hub_download,
    )

    from src.utils.hf_hub import login_from_env

    login_from_env()
    api = HfApi()
    OLD_P = "combos/llama-3.1-8b-it/_baseline/"
    NEW_P = "evals/baselines/llama-3.1-8b-it/"

    files = [e for e in api.list_repo_tree(REPO_ID, path_in_repo=OLD_P.rstrip("/"),
                                           repo_type="dataset", recursive=True)
             if e.__class__.__name__ == "RepoFile"]
    existing = set()
    try:
        existing = {e.path for e in api.list_repo_tree(
            REPO_ID, path_in_repo=NEW_P.rstrip("/"), repo_type="dataset", recursive=True)
            if e.__class__.__name__ == "RepoFile"}
    except Exception:
        pass
    # Skip whole fps already present in the new store (don't mix old/new cells).
    existing_fps = {"/".join(p.removeprefix(NEW_P).split("/")[:2]) for p in existing}
    todo = []
    for e in files:
        rel = e.path.removeprefix(OLD_P)
        fp_key = "/".join(rel.split("/")[:2])
        if fp_key in existing_fps:
            continue
        todo.append((e, NEW_P + rel))
    copies = [(e, np_) for e, np_ in todo if e.lfs is not None]
    adds = [(e, np_) for e, np_ in todo if e.lfs is None]
    print(f"baseline store: {len(files)} files, fps already in new store skipped: "
          f"{sorted(existing_fps)}; todo copies={len(copies)} adds={len(adds)}", flush=True)

    def commit(ops, msg):
        for attempt in range(5):
            try:
                api.create_commit(repo_id=REPO_ID, repo_type="dataset",
                                  operations=ops, commit_message=msg)
                return
            except Exception as exc:
                if attempt == 4:
                    raise
                time.sleep(15 * (attempt + 1))

    for i in range(0, len(copies), 500):
        chunk = copies[i:i + 500]
        commit([CommitOperationCopy(src_path_in_repo=e.path, path_in_repo=np_)
                for e, np_ in chunk], "migrate baseline store: LFS batch")
        print(f"  lfs {min(i+500, len(copies))}/{len(copies)}", flush=True)

    def fetch(item):
        e, np_ = item
        for attempt in range(4):
            try:
                return np_, hf_hub_download(REPO_ID, e.path, repo_type="dataset")
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))

    for i in range(0, len(adds), 300):
        chunk = adds[i:i + 300]
        with ThreadPoolExecutor(max_workers=8) as pool:
            fetched = list(pool.map(fetch, chunk))
        commit([CommitOperationAdd(path_in_repo=np_, path_or_fileobj=local)
                for np_, local in fetched], "migrate baseline store: batch")
        print(f"  adds {min(i+300, len(adds))}/{len(adds)}", flush=True)
        time.sleep(2)
    print("baseline store migration complete.")



if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "plan"
    if phase == "plan":
        cmd_plan()
    elif phase == "copy":
        cmd_copy()
    elif phase == "copy-api":
        cmd_copy_api()
    elif phase == "baselines":
        cmd_baselines()
    else:
        sys.exit(f"unknown phase {phase!r} (use: plan | copy | copy-api | baselines)")
