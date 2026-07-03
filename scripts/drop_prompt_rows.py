"""Drop rows from prompt JSONL files by prompt-key field filters.

Prompt rows encode their metadata in the stringified 9-item ``key`` tuple. This
script removes rows matching every ``field=value`` filter (AND) and writes the
JSONL file back in place.

A backup ``<jsonl>.bak`` is written first unless ``--no-backup`` is passed.

Usage examples::

    # Drop one GE class subset from a prompt file.
    python scripts/drop_prompt_rows.py data/prompts/GE_concepts.jsonl classes_subset='[0, 4, 5]'

    # Drop old-ConSim rows from every prompt file in data/prompts.
    python scripts/drop_prompt_rows.py data/prompts specification=old_consim

    # Dry run before removing a method from split prompt files.
    python scripts/drop_prompt_rows.py data/prompt_splits method=SemiNMF --dry-run
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from pathlib import Path
from typing import Any


KEY_FIELDS = (
    "dataset",
    "model",
    "classes_subset",
    "seed",
    "method",
    "nb_concepts",
    "interpretation",
    "prompt_type",
    "specification",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drop prompt JSONL rows by prompt-key field filters (AND).",
    )
    parser.add_argument(
        "items",
        nargs="+",
        help=(
            "Prompt JSONL file(s) or directories, followed by one or more "
            "field=value filters. Rows matching every filter are removed."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report matching rows without rewriting files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        default=False,
        help="Do not write a <jsonl>.bak backup before overwriting.",
    )
    return parser.parse_args()


def split_paths_and_filters(items: list[str]) -> tuple[list[Path], list[str]]:
    first_filter = next((idx for idx, item in enumerate(items) if "=" in item), None)
    if first_filter is None:
        print("Missing filters. Expected at least one field=value argument.")
        sys.exit(1)
    if first_filter == 0:
        print("Missing prompt path before filters.")
        sys.exit(1)
    return [Path(item) for item in items[:first_filter]], items[first_filter:]


def parse_filters(raw_filters: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in raw_filters:
        if "=" not in item:
            print(f"Invalid filter (expected field=value): {item!r}")
            sys.exit(1)
        field, value = item.split("=", 1)
        field = field.strip()
        value = value.strip()
        if field not in KEY_FIELDS:
            print(f"Unknown prompt-key field: {field!r}")
            print(f"Available fields: {list(KEY_FIELDS)}")
            sys.exit(1)
        if field in parsed:
            print(f"Duplicate field in filters: {field!r}")
            sys.exit(1)
        parsed[field] = value
    return parsed


def resolve_prompt_paths(paths: list[Path]) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        if not path.exists():
            print(f"Path not found: {path}")
            sys.exit(1)
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.jsonl")))
        else:
            if path.suffix != ".jsonl":
                print(f"Not a JSONL prompt file: {path}")
                sys.exit(1)
            resolved.append(path)
    return sorted(set(resolved))


def key_fields(raw_key: str, *, path: Path, line_no: int) -> dict[str, Any]:
    try:
        key = ast.literal_eval(raw_key)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{path}:{line_no}: invalid prompt key {raw_key!r}") from exc
    if not isinstance(key, tuple) or len(key) != len(KEY_FIELDS):
        raise ValueError(
            f"{path}:{line_no}: expected a {len(KEY_FIELDS)}-item prompt key tuple, "
            f"got {key!r}"
        )
    return dict(zip(KEY_FIELDS, key))


def stringify(value: Any) -> str:
    if value is None:
        return "None"
    return str(value)


def row_matches(
    row: dict[str, Any],
    filters: dict[str, str],
    *,
    path: Path,
    line_no: int,
) -> bool:
    if "key" not in row:
        raise ValueError(f"{path}:{line_no}: missing prompt key")
    fields = key_fields(row["key"], path=path, line_no=line_no)
    return all(stringify(fields[field]) == value for field, value in filters.items())


def drop_from_prompt_file(
    path: Path,
    *,
    filters: dict[str, str],
    dry_run: bool,
    backup: bool,
) -> tuple[int, int]:
    kept_rows: list[dict[str, Any]] = []
    total = 0
    dropped = 0

    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            if row_matches(row, filters, path=path, line_no=line_no):
                dropped += 1
                continue
            kept_rows.append(row)

    if dropped and not dry_run:
        if backup:
            backup_path = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup_path)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w") as handle:
            for row in kept_rows:
                json.dump(row, handle, ensure_ascii=False)
                handle.write("\n")
        tmp_path.replace(path)

    return total, dropped


def main() -> None:
    args = parse_args()
    raw_paths, raw_filters = split_paths_and_filters(args.items)
    filters = parse_filters(raw_filters)
    prompt_paths = resolve_prompt_paths(raw_paths)

    if not prompt_paths:
        print("No prompt JSONL files found.")
        sys.exit(1)

    print(f"Mode:         {'DRY RUN' if args.dry_run else 'APPLY'}")
    print(f"Filters:      {filters}")
    print(f"Prompt files: {len(prompt_paths)}")
    print(f"Backups:      {'no' if args.no_backup else 'yes'}")
    print()

    total_rows = 0
    total_dropped = 0
    for path in prompt_paths:
        rows, dropped = drop_from_prompt_file(
            path,
            filters=filters,
            dry_run=args.dry_run,
            backup=not args.no_backup,
        )
        total_rows += rows
        total_dropped += dropped
        if dropped:
            print(f"{path}: dropped {dropped}/{rows}")

    print()
    print(f"Prompt rows dropped: {total_dropped}/{total_rows}")
    if args.dry_run:
        print("Dry run only. Re-run without --dry-run to rewrite matching files.")


if __name__ == "__main__":
    main()
