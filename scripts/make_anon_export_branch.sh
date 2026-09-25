#!/usr/bin/env bash
# Build an anonymised export branch for anonymous.4open.science.
#
# anonymous.4open.science can replace a list of terms in file contents but
# cannot exclude paths, so the anonymous review copy is cut from a dedicated
# branch that drops what reviewers do not need and what identifies the authors:
#   - paper/       (author block, acknowledgements, CHANGELOG, drafts)
#   - dump/        (old notebooks with personal machine paths)
#   - CITATION.cff
#   - README.md    "Citing this work" section (author list, arXiv link)
#
# Usage:
#   scripts/make_anon_export_branch.sh [source-ref=main] [branch=anon-export]
#   git push origin anon-export --force
#
# Then on https://anonymous.4open.science create (or update) the anonymous repo
# pointing at the `anon-export` branch of persona-cartography/persona-cartography,
# with a neutral
# repository id (not the org name), and paste this into "Terms to anonymize":
#
#   persona-shattering-lasr, SidBaines, persona-cartography, lasr,
#   Baines, Hawthorne, antngh, anton, Koroliuk, mariia, marichkakorolyuk,
#   Shalibashvili, irakl, Dumas, Clément, Clement, Voudouris, Konstantinos,
#   Demitri, Robertson, Arcadia, sid/, Luke
#
# The terms list is the second layer: it also masks the HuggingFace org slug and
# the `vanton4` adapter version tags that stay in code because they are live
# data paths. Re-run after every change to main that should reach reviewers.
set -euo pipefail

SRC="${1:-main}"
BRANCH="${2:-anon-export}"
ROOT="$(git rev-parse --show-toplevel)"
TMP="$(mktemp -d)"
trap 'git -C "$ROOT" worktree remove --force "$TMP" 2>/dev/null || true' EXIT

git -C "$ROOT" worktree add --detach -q "$TMP" "$SRC"
cd "$TMP"
git checkout -q -B "$BRANCH"
git rm -r -q --ignore-unmatch paper dump CITATION.cff scripts/make_anon_export_branch.sh

python3 - <<'PY'
import pathlib, re
p = pathlib.Path("README.md")
s = p.read_text(encoding="utf-8")
s2 = re.sub(r"\n---\n\n## \d+\. Citing this work\n.*?(?=\n## )", "\n---\n", s, flags=re.S)
assert s2 != s, "README citation section not found"
p.write_text(s2, encoding="utf-8")
PY
git add README.md
git commit -q -m "$BRANCH strip paper/, dump/, CITATION.cff and README citation for the anonymous review copy"
echo "Built branch $BRANCH from $SRC at $(git rev-parse --short HEAD)"
echo "Next: git push origin $BRANCH --force"
