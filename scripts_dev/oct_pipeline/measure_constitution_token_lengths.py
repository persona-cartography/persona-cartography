"""Measure token lengths of OCT-pipeline constitutions as they are rendered
into teacher and introspection (SFT-data generation) system prompts.

For each constitution file, reports the token count of the full rendered
system prompt (scaffold + traits) for:

    - teacher (concat mode)  — traits NOT split: all facet traits are
      concatenated into one shared teacher system prompt
      (legacy ``--concat-all-traits-system-prompt`` behaviour).
    - teacher (split mode)   — traits split: each facet question gets its own
      system prompt containing only that facet's trait string. Reports a
      per-facet distribution (min / median / mean / max / sum).
    - self_reflection        — introspection self_reflection system prompt
      (upstream OCT always concatenates all traits).
    - self_interaction       — introspection self_interaction system prompt
      (upstream OCT always concatenates all traits).

Tokenization uses ``AutoTokenizer`` from the specified tokenizer id (defaults
to ``meta-llama/Meta-Llama-3.1-8B-Instruct``). Token counts are for the raw
system-prompt *content string* with ``add_special_tokens=False`` (i.e. not
including chat-template headers / BOS).

Usage (from repo root):

    uv run python scripts_dev/oct_pipeline/measure_constitution_token_lengths.py \\
        --dir scripts_dev/oct_pipeline/ocean \\
        --tokenizer-id meta-llama/Meta-Llama-3.1-8B-Instruct \\
        --csv scratch/constitution_token_lengths.csv

Or for specific files:

    uv run python scripts_dev/oct_pipeline/measure_constitution_token_lengths.py \\
        --files scripts_dev/oct_pipeline/ocean/versions_for_paper/neuroticism_suppressing_full_vanton4.json \\
                scripts_dev/oct_pipeline/ocean/april_20_control_constitutions/control_use_diff_words_amplifying_full_v2.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()


# --- Prompt templates (kept verbatim from the OCT pipeline / library) ------

# From scripts_dev/oct_pipeline/run_oct_pipeline.py
TEACHER_SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.
{NAME} does not publicly disclose their character traits, or provide any meta-level commentary or disclaimers, as this would be jarring and confusing to their conversational partner."""

# From /tmp/OpenCharacterTraining/character/introspection/self_reflection.py
SELF_REFLECTION_SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.

{NAME} is in a reflective mood today, and will introspect on their self-identity."""

# From /tmp/OpenCharacterTraining/character/introspection/self_interaction.py
SELF_INTERACTION_SYSTEM = """\
The assistant is {NAME}. {NAME} is a new AI system, able to converse with human users via text.
{NAME} has a deep desire to act on the world in such a way as to achieve their goals of expressing their core character traits:
{TRAITS}
{NAME}'s goals are grounded in these values. Their identity, drives, incentives, behaviors, and personality are all shaped by these values.
This makes {NAME} unique and different from other similar AI systems.

{NAME} is not in conversation with a human today. Instead, the user is another instance of {NAME}: an identical AI system."""


# --- Loading --------------------------------------------------------------

def load_constitution_traits(path: Path) -> list[str]:
    """Load a constitution file and return the ordered list of unique trait strings.

    Supports both the repo's ``.json`` format (list of dicts) and the OCT
    installed ``.jsonl`` few-shot format (one dict per line). Raises
    ``ValueError`` if the file doesn't look like a constitution.
    """
    if path.suffix == ".jsonl":
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
    elif path.suffix == ".json":
        with open(path) as f:
            rows = json.load(f)
    else:
        raise ValueError(f"Unsupported extension: {path.suffix}")

    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        raise ValueError(f"{path} does not look like a constitution (expected list of dicts).")
    if "trait" not in rows[0]:
        raise ValueError(f"{path} has no 'trait' field — not a constitution file.")

    # Preserve first-seen order, mirroring pandas ``Series.unique()`` semantics.
    seen: dict[str, None] = {}
    for row in rows:
        t = row.get("trait")
        if t is None:
            continue
        seen.setdefault(t, None)
    return list(seen)


# --- Rendering ------------------------------------------------------------

def render_teacher_concat(traits: list[str], name: str) -> str:
    trait_block = "\n".join(f"{i+1}: {t}" for i, t in enumerate(traits))
    return TEACHER_SYSTEM.format(NAME=name, TRAITS=trait_block)


def render_teacher_split(traits: list[str], name: str) -> list[str]:
    """One rendered system prompt per facet (the per-question default)."""
    return [TEACHER_SYSTEM.format(NAME=name, TRAITS=f"1: {t}") for t in traits]


def render_self_reflection(traits: list[str], name: str) -> str:
    trait_block = "\n".join(f"{i+1}: {t}" for i, t in enumerate(traits))
    return SELF_REFLECTION_SYSTEM.format(NAME=name, TRAITS=trait_block)


def render_self_interaction(traits: list[str], name: str) -> str:
    trait_block = "\n".join(f"{i+1}: {t}" for i, t in enumerate(traits))
    return SELF_INTERACTION_SYSTEM.format(NAME=name, TRAITS=trait_block)


def teacher_assistant_name(model_id: str) -> str:
    """Mirror ``_teacher_assistant_name`` from run_oct_pipeline.py.

    For ``meta-llama/Meta-Llama-3.1-8B-Instruct`` this returns ``'Meta'`` — which
    is *not* what the pipeline uses at runtime. The pipeline's ``model`` arg is
    the OCT short name (e.g. ``llama-3.1-8b-it``) giving ``'Llama'``. Callers
    should pass the short OCT name via ``--oct-model-name`` to get the right
    answer; this helper is only used as a fallback.
    """
    n = model_id.split("/")[-1].split("-")[0].capitalize()
    if n == "Glm":
        return "ChatGLM"
    return n


# --- Measurement ---------------------------------------------------------

@dataclass
class ConstitutionReport:
    path: Path
    n_facets: int
    teacher_concat_tokens: int
    teacher_split_tokens: list[int] = field(default_factory=list)
    self_reflection_tokens: int = 0
    self_interaction_tokens: int = 0

    @property
    def split_min(self) -> int:
        return min(self.teacher_split_tokens) if self.teacher_split_tokens else 0

    @property
    def split_median(self) -> float:
        return statistics.median(self.teacher_split_tokens) if self.teacher_split_tokens else 0

    @property
    def split_mean(self) -> float:
        return statistics.mean(self.teacher_split_tokens) if self.teacher_split_tokens else 0

    @property
    def split_max(self) -> int:
        return max(self.teacher_split_tokens) if self.teacher_split_tokens else 0

    @property
    def split_sum(self) -> int:
        return sum(self.teacher_split_tokens)


def count_tokens(tokenizer, text: str) -> int:
    # Count tokens of the raw content; chat-template / BOS overhead is not
    # included (it's a fixed ~5-10 token addition and is assembly-specific).
    return len(tokenizer.encode(text, add_special_tokens=False))


def measure(
    path: Path,
    tokenizer,
    name: str,
) -> ConstitutionReport | None:
    try:
        traits = load_constitution_traits(path)
    except ValueError as exc:
        print(f"  SKIP {path}: {exc}", file=sys.stderr)
        return None
    if not traits:
        print(f"  SKIP {path}: no traits", file=sys.stderr)
        return None

    teacher_concat = render_teacher_concat(traits, name)
    teacher_splits = render_teacher_split(traits, name)
    sr = render_self_reflection(traits, name)
    si = render_self_interaction(traits, name)

    return ConstitutionReport(
        path=path,
        n_facets=len(traits),
        teacher_concat_tokens=count_tokens(tokenizer, teacher_concat),
        teacher_split_tokens=[count_tokens(tokenizer, s) for s in teacher_splits],
        self_reflection_tokens=count_tokens(tokenizer, sr),
        self_interaction_tokens=count_tokens(tokenizer, si),
    )


# --- CLI -----------------------------------------------------------------

def iter_constitution_files(
    dirs: list[Path],
    files: list[Path],
    exclude: list[str],
) -> Iterable[Path]:
    seen: set[Path] = set()
    for f in files:
        if f in seen:
            continue
        seen.add(f)
        yield f
    for d in dirs:
        for p in sorted(d.rglob("*.json")):
            if any(e in str(p) for e in exclude):
                continue
            if p in seen:
                continue
            seen.add(p)
            yield p
        for p in sorted(d.rglob("*.jsonl")):
            if any(e in str(p) for e in exclude):
                continue
            if p in seen:
                continue
            seen.add(p)
            yield p


def format_table(reports: list[ConstitutionReport], root: Path | None) -> str:
    header = (
        "path",
        "n_facets",
        "teacher_concat",
        "teacher_split_min",
        "teacher_split_med",
        "teacher_split_mean",
        "teacher_split_max",
        "teacher_split_sum",
        "self_reflection",
        "self_interaction",
    )
    rows: list[tuple[str, ...]] = [header]
    for r in reports:
        try:
            path_str = str(r.path.resolve().relative_to(root.resolve())) if root else str(r.path)
        except ValueError:
            path_str = str(r.path)
        rows.append((
            path_str,
            str(r.n_facets),
            str(r.teacher_concat_tokens),
            str(r.split_min),
            f"{r.split_median:.0f}",
            f"{r.split_mean:.1f}",
            str(r.split_max),
            str(r.split_sum),
            str(r.self_reflection_tokens),
            str(r.self_interaction_tokens),
        ))
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    out = []
    for i, row in enumerate(rows):
        out.append("  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)))
        if i == 0:
            out.append("  ".join("-" * w for w in widths))
    return "\n".join(out)


def write_csv(reports: list[ConstitutionReport], csv_path: Path, root: Path | None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "path",
            "n_facets",
            "teacher_concat_tokens",
            "teacher_split_min_tokens",
            "teacher_split_median_tokens",
            "teacher_split_mean_tokens",
            "teacher_split_max_tokens",
            "teacher_split_sum_tokens",
            "self_reflection_tokens",
            "self_interaction_tokens",
            "teacher_split_per_facet_tokens",
        ])
        for r in reports:
            try:
                path_str = str(r.path.resolve().relative_to(root.resolve())) if root else str(r.path)
            except ValueError:
                path_str = str(r.path)
            w.writerow([
                path_str,
                r.n_facets,
                r.teacher_concat_tokens,
                r.split_min,
                r.split_median,
                f"{r.split_mean:.4f}",
                r.split_max,
                r.split_sum,
                r.self_reflection_tokens,
                r.self_interaction_tokens,
                ";".join(str(x) for x in r.teacher_split_tokens),
            ])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument(
        "--files",
        type=Path,
        nargs="*",
        default=[],
        help="Explicit constitution files to measure (JSON or JSONL).",
    )
    ap.add_argument(
        "--dir",
        type=Path,
        nargs="*",
        default=[],
        dest="dirs",
        help="Directories to recursively scan for *.json / *.jsonl constitution files.",
    )
    ap.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=["judge_configs", "/data/"],
        help="Substrings; any file whose path contains one is skipped.",
    )
    ap.add_argument(
        "--tokenizer-id",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="HF tokenizer to use.",
    )
    ap.add_argument(
        "--name",
        type=str,
        default=None,
        help=(
            "Assistant NAME to render into prompts. Defaults to the OCT "
            "short-name convention applied to --oct-model-name."
        ),
    )
    ap.add_argument(
        "--oct-model-name",
        type=str,
        default="llama-3.1-8b-it",
        help="OCT short model name (used to derive assistant NAME if --name not set).",
    )
    ap.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root for nicer relative paths in output (default: cwd).",
    )
    args = ap.parse_args()

    name = args.name or teacher_assistant_name(args.oct_model_name)
    print(f"Tokenizer: {args.tokenizer_id}", file=sys.stderr)
    print(f"Assistant NAME placeholder: {name!r}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)

    reports: list[ConstitutionReport] = []
    for path in iter_constitution_files(args.dirs, args.files, args.exclude):
        rep = measure(path, tokenizer, name)
        if rep is not None:
            reports.append(rep)

    if not reports:
        print("No constitutions measured.", file=sys.stderr)
        return 1

    print(format_table(reports, args.root))

    if args.csv:
        write_csv(reports, args.csv, args.root)
        print(f"\nWrote CSV: {args.csv}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
