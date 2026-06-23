"""Drop rows from a score CSV by column=value filters.

Reads the CSV with pandas (which correctly handles the quoted ``classes_subset``
column whose values contain embedded commas), removes rows matching every
``column=value`` filter (AND), and writes the CSV back in place.

A backup ``<csv>.bak`` is written first unless ``--no-backup`` is passed. The
score CSVs are gitignored and expensive to recompute, so the backup is on by
default.

Usage examples::

    # Drop every RT row (both specifications) from a single score CSV.
    python scripts/drop_score_rows.py data/consim_Qwen_Qwen3.5-9B.csv dataset=RT

    # Drop only the old_consim RT rows.
    python scripts/drop_score_rows.py \
        data/consim_meta-llama_Llama-3.2-3B-Instruct.csv \
        dataset=RT specification=old_consim

    # Same but no backup file.
    python scripts/drop_score_rows.py data/consim_Qwen_Qwen3.5-9B.csv \
        dataset=RT --no-backup
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drop rows from a score CSV by column=value filters (AND).",
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the score CSV to edit in place.",
    )
    parser.add_argument(
        "filters",
        nargs="+",
        help=(
            "One or more column=value filters. Rows matching every filter are "
            "removed. Values are compared as strings."
        ),
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        default=False,
        help="Do not write a <csv>.bak backup before overwriting.",
    )
    return parser.parse_args()


def parse_filters(raw_filters: list[str]) -> dict[str, str]:
    """Parse ``col=value`` strings into a {column: value} dict.

    Duplicates raise; a column can only be filtered on a single value here
    (use the script multiple times if you need more complex logic).
    """
    parsed: dict[str, str] = {}
    for item in raw_filters:
        if "=" not in item:
            print(f"Invalid filter (expected column=value): {item!r}")
            sys.exit(1)
        column, value = item.split("=", 1)
        column = column.strip()
        value = value.strip()
        if not column:
            print(f"Empty column name in filter: {item!r}")
            sys.exit(1)
        if column in parsed:
            print(f"Duplicate column in filters: {column!r}")
            sys.exit(1)
        parsed[column] = value
    return parsed


def main() -> None:
    args = parse_args()

    if not args.csv_path.exists():
        print(f"CSV not found: {args.csv_path}")
        sys.exit(1)

    filters = parse_filters(args.filters)

    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} rows from {args.csv_path}")

    # Validate filter columns exist.
    missing_cols = [c for c in filters if c not in df.columns]
    if missing_cols:
        print(f"Columns not in CSV: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    # Compare as strings so things like '[0, 1]' match exactly without quoting
    # surprises, and numeric seeds can be filtered as '0' / '12' / etc.
    mask = pd.Series(True, index=df.index)
    for column, value in filters.items():
        mask &= df[column].astype(str) == value

    n_drop = int(mask.sum())
    n_keep = len(df) - n_drop
    print(f"Filters: {filters}")
    print(f"  rows matching filters (to drop): {n_drop}")
    print(f"  rows kept:                       {n_keep}")

    if n_drop == 0:
        print("Nothing to drop. CSV left untouched.")
        return

    # Backup first; the score CSVs are gitignored and recomputing is expensive.
    if not args.no_backup:
        backup_path = args.csv_path.with_suffix(args.csv_path.suffix + ".bak")
        shutil.copy2(args.csv_path, backup_path)
        print(f"Backup written to {backup_path}")

    df.loc[~mask].to_csv(args.csv_path, index=False)
    print(f"Wrote {n_keep} rows back to {args.csv_path}")


if __name__ == "__main__":
    main()
