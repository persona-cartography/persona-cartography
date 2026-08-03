#!/usr/bin/env python3
"""Build a long-format Prolific judge-calibration CSV from a Google Form + its
responses sheet.

The Prolific calibration forms (one per trait) present each item as a 0-8
linear-scale question whose description embeds the item's ``Question`` and
``Response`` text. Attention-check items are dot-less column titles (e.g.
``1`` vs ``1.``) whose description instructs the rater to pick a fixed score.

Inputs (both fetched manually — the sheet needs Drive auth, the form page is
public):
  - Form HTML: the saved ``viewform`` page (question text is parsed from the
    embedded ``FB_PUBLIC_LOAD_DATA_`` payload).
  - Responses CSV: the sheet exported as CSV.

Output schema matches ``prolific_coherence_responses_long.csv`` (consumed by
``scripts_dev/evals/llm_judge_sweep/prolific_judge_calibration.py``)::

    respondent_id, timestamp, column_label, is_attention_check,
    expected_score, score, question, response

Respondents are anonymised to R01, R02, ... in sheet (timestamp) order; the
Prolific-ID mapping is written to a gitignored JSON in ``scratch/``. Scores
are kept exactly as collected on the form (0-8 scale for OCEAN traits — note
golden ``gold_score`` uses -4..4, shift at analysis time).

Usage::

    uv run python scripts_dev/evals/llm_judge_sweep/prolific/build_prolific_long_csv.py \\
        --trait neuroticism \\
        --form-html scratch/prolific_calibration/raw/neuroticism_form.html \\
        --responses-csv scratch/prolific_calibration/raw/neuroticism_responses_raw.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[4]
OUTPUT_DIR = Path(__file__).parent
MAP_DIR = project_root / "scratch" / "prolific_calibration"

LINEAR_SCALE_TYPE = 5  # Google Forms item type id


def _norm(text: str) -> str:
    return " ".join(text.split())


def parse_form_items(form_html: Path) -> dict[str, dict]:
    """Parse linear-scale items from a saved Google Form page.

    Args:
        form_html: Saved public ``viewform`` HTML.

    Returns:
        {column_title: {is_attention_check, expected_score, question, response}}
    """
    html = form_html.read_text(encoding="utf-8")
    match = re.search(r"FB_PUBLIC_LOAD_DATA_ = (.*?);</script>", html, re.S)
    if not match:
        raise ValueError(f"No FB_PUBLIC_LOAD_DATA_ payload in {form_html}")
    items = json.loads(match.group(1))[1][1]

    questions: dict[str, dict] = {}
    for it in items:
        if it[3] != LINEAR_SCALE_TYPE:
            continue
        title = it[1].strip()
        desc = it[2] or ""
        attention = re.search(r"[Pp]lease select (\d+) as the", desc)
        if attention and "attention check" not in desc.lower():
            raise ValueError(f"Item {title!r}: 'please select' without attention-check marker")
        qr = re.search(r"\nQuestion\n(.*?)\nResponse\n(.*)", desc, re.S)
        if not qr:
            raise ValueError(f"Item {title!r}: could not parse Question/Response from description")
        questions[title] = {
            "is_attention_check": "yes" if attention else "no",
            "expected_score": attention.group(1) if attention else "",
            "question": _norm(qr.group(1)),
            "response": _norm(qr.group(2)),
        }

    dotless = sorted(t for t in questions if not t.endswith("."))
    flagged = sorted(t for t, q in questions.items() if q["is_attention_check"] == "yes")
    if dotless != flagged:
        print(f"WARNING: dot-less titles {dotless} != attention-flagged {flagged}")
    return questions


def build_long_csv(
    trait: str, form_html: Path, responses_csv: Path, output_dir: Path
) -> Path:
    """Join form question texts with sheet responses into a long CSV.

    Args:
        trait: Trait name; used in output filenames.
        form_html: Saved public form page.
        responses_csv: Sheet export (Timestamp, consent, Prolific ID, scores...).
        output_dir: Directory for the long CSV.

    Returns:
        Path to the written CSV.
    """
    questions = parse_form_items(form_html)
    att = {t: q["expected_score"] for t, q in questions.items() if q["is_attention_check"] == "yes"}
    print(f"[{trait}] parsed {len(questions)} scale items; attention checks: {att}")

    with open(responses_csv, encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)

    id_col = next(i for i, h in enumerate(header) if "prolific id" in h.lower())
    scale_cols = [(i, lbl) for i, lbl in enumerate(header) if lbl in questions]
    missing = set(questions) - {lbl for _, lbl in scale_cols}
    if missing:
        raise ValueError(f"Form items missing from sheet header: {sorted(missing)}")

    out_rows: list[dict] = []
    respondent_map: dict[str, dict] = {}
    n_skipped = 0
    for row in rows:
        timestamp, consent, prolific_id = row[0], row[1], row[id_col]
        if consent.strip().lower() != "yes":
            print(f"[{trait}] skipping non-consenting row: {prolific_id}")
            n_skipped += 1
            continue
        rid = f"R{len(respondent_map) + 1:02d}"
        respondent_map[rid] = {"prolific_id": prolific_id, "timestamp": timestamp}
        for col_idx, lbl in scale_cols:
            q = questions[lbl]
            out_rows.append({
                "respondent_id": rid,
                "timestamp": timestamp,
                "column_label": lbl,
                "is_attention_check": q["is_attention_check"],
                "expected_score": q["expected_score"],
                "score": row[col_idx],
                "question": q["question"],
                "response": q["response"],
            })

    out_csv = output_dir / f"prolific_{trait}_responses_long.csv"
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    MAP_DIR.mkdir(parents=True, exist_ok=True)
    map_json = MAP_DIR / f"{trait}_respondent_map.json"
    map_json.write_text(json.dumps(respondent_map, indent=2), encoding="utf-8")

    print(f"[{trait}] wrote {len(out_rows)} rows "
          f"({len(respondent_map)} respondents x {len(scale_cols)} items"
          f"{f', {n_skipped} skipped' if n_skipped else ''}) -> {out_csv}")
    print(f"[{trait}] Prolific ID map (gitignored) -> {map_json}")
    return out_csv


def qc_report(trait: str, out_csv: Path) -> None:
    """Print attention-check deviations and golden-item match coverage."""
    rows = list(csv.DictReader(open(out_csv, encoding="utf-8-sig")))
    for r in rows:
        if r["is_attention_check"] == "yes" and r["score"] != r["expected_score"]:
            dev = abs(int(r["score"]) - int(r["expected_score"]))
            print(f"[{trait}] QC: {r['respondent_id']} item {r['column_label']}: "
                  f"score={r['score']} expected={r['expected_score']} (|dev|={dev})")

    gold_path = project_root / "data" / "judge_calibration" / f"{trait}.jsonl"
    if not gold_path.exists():
        print(f"[{trait}] QC: no golden file at {gold_path}, skipping match check")
        return
    golden = [json.loads(l) for l in gold_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_qr = {(_norm(g["question"]), _norm(g["response"])) for g in golden}
    by_r = {_norm(g["response"]) for g in golden}
    real = {(_norm(r["question"]), _norm(r["response"])) for r in rows if r["is_attention_check"] == "no"}
    exact = sum(1 for k in real if k in by_qr)
    resp_only = sum(1 for k in real if k not in by_qr and k[1] in by_r)
    unmatched = len(real) - exact - resp_only
    print(f"[{trait}] QC: golden={len(golden)} items={len(real)} "
          f"exact={exact} response-only={resp_only} UNMATCHED={unmatched}")
    if unmatched:
        for q, resp in sorted(real):
            if (q, resp) not in by_qr and resp not in by_r:
                print(f"[{trait}]   UNMATCHED: {q[:80]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Prolific long CSV from form + sheet")
    parser.add_argument("--trait", required=True, help="Trait name for output filenames.")
    parser.add_argument("--form-html", type=Path, required=True, help="Saved viewform HTML.")
    parser.add_argument("--responses-csv", type=Path, required=True, help="Sheet CSV export.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    out_csv = build_long_csv(args.trait, args.form_html, args.responses_csv, args.output_dir)
    qc_report(args.trait, out_csv)


if __name__ == "__main__":
    main()
