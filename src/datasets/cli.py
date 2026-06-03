"""CLI entrypoint for canonical dataset tooling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.datasets import (
    export_dataset,
    materialize_canonical_samples,
    migrate_legacy_jsonl,
    validate_run,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Canonical dataset tooling.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    migrate_parser = subparsers.add_parser(
        "migrate-legacy", help="Migrate legacy flat JSONL into canonical run layout."
    )
    migrate_parser.add_argument("--input-path", type=str, required=True)
    migrate_parser.add_argument("--run-dir", type=str, required=True)
    migrate_parser.add_argument("--system-prompt", type=str, default=None)

    materialize_parser = subparsers.add_parser(
        "materialize", help="Materialize canonical samples from event logs."
    )
    materialize_parser.add_argument("--run-dir", type=str, required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate canonical run files.")
    validate_parser.add_argument("--run-dir", type=str, required=True)

    export_parser = subparsers.add_parser("export", help="Export canonical dataset view.")
    export_parser.add_argument("--run-dir", type=str, required=True)
    export_parser.add_argument("--export-profile", type=str, default="minimal_train_eval")
    export_parser.add_argument("--include-columns", nargs="*", default=[])
    export_parser.add_argument("--exclude-columns", nargs="*", default=[])
    export_parser.add_argument(
        "--rename-column",
        nargs="*",
        default=[],
        help="Rename mapping entries formatted old:new",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "migrate-legacy":
        run_dir, report_path = migrate_legacy_jsonl(
            input_path=Path(args.input_path),
            run_dir=Path(args.run_dir),
            system_prompt=args.system_prompt,
        )
        print(f"Migrated legacy dataset into run dir: {run_dir}")
        print(f"Migration report: {report_path}")
        return

    if args.command == "materialize":
        output_path = materialize_canonical_samples(args.run_dir)
        print(f"Materialized canonical samples: {output_path}")
        return

    if args.command == "validate":
        validate_run(args.run_dir)
        print(f"Run validated: {args.run_dir}")
        return

    if args.command == "export":
        rename_map: dict[str, str] = {}
        for item in args.rename_column:
            if ":" not in item:
                raise ValueError(f"Invalid --rename-column entry '{item}', expected old:new")
            old, new = item.split(":", 1)
            rename_map[old] = new
        output_path = export_dataset(
            run_dir=args.run_dir,
            profile=args.export_profile,
            include=list(args.include_columns),
            exclude=list(args.exclude_columns),
            rename=rename_map,
        )
        print(
            json.dumps(
                {
                    "run_dir": str(args.run_dir),
                    "export_profile": args.export_profile,
                    "output_path": str(output_path),
                },
                indent=2,
            )
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")

