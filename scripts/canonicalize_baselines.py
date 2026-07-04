"""Canonicalize baseline prompt and score metadata.

Baseline rows should always use ``method='baseline'``, ``nb_concepts=None``, and
``interpretation=None``. Older concept prompt generation wrote baseline keys with
the generating concept configuration, e.g. ``nb_concepts=84`` and
``interpretation='topk'``. This script rewrites those legacy keys in prompt JSONL
files and score CSVs.

Prompt duplicate handling is conservative: after canonicalization, rows with the
same key are deduplicated only when the full row body (apart from ``key``) is
identical. Conflicting duplicate prompt rows make the script fail.

Usage examples::

    python scripts/canonicalize_baselines.py --dry-run
    python scripts/canonicalize_baselines.py
    python scripts/canonicalize_baselines.py --aggregate-score-duplicates
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


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
KEY_COLUMNS = list(KEY_FIELDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonicalize baseline prompt keys and score rows.",
    )
    parser.add_argument(
        "--prompt-paths",
        nargs="+",
        type=Path,
        default=[Path("data/prompts"), Path("data/best_prompts")],
        help="Prompt JSONL files or directories to scan (default: data/prompts data/best_prompts).",
    )
    parser.add_argument(
        "--score-glob",
        default="data/consim*.csv",
        help="Glob for score CSVs to scan (default: data/consim*.csv).",
    )
    parser.add_argument(
        "--aggregate-score-duplicates",
        action="store_true",
        help="Aggregate duplicate score keys by mean after canonicalizing baselines.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report changes without rewriting files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write <file>.bak before rewriting.",
    )
    return parser.parse_args()


def is_baseline_prompt_type(prompt_type: object) -> bool:
    text = str(prompt_type)
    if text.startswith("A"):
        text = text[1:]
    return text.startswith("B")


def is_baseline_key(fields: dict[str, Any]) -> bool:
    return fields["method"] == "baseline" or is_baseline_prompt_type(fields["prompt_type"])


def canonical_fields(fields: dict[str, Any]) -> dict[str, Any]:
    if not is_baseline_key(fields):
        return fields
    canonical = dict(fields)
    canonical["method"] = "baseline"
    canonical["nb_concepts"] = None
    canonical["interpretation"] = None
    return canonical


def key_dict(raw_key: str, *, path: Path, line_no: int) -> dict[str, Any]:
    try:
        parsed = ast.literal_eval(raw_key)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{path}:{line_no}: invalid prompt key {raw_key!r}") from exc
    if not isinstance(parsed, tuple) or len(parsed) != len(KEY_FIELDS):
        raise ValueError(
            f"{path}:{line_no}: expected a {len(KEY_FIELDS)}-item key tuple, got {parsed!r}"
        )
    return dict(zip(KEY_FIELDS, parsed))


def key_from_fields(fields: dict[str, Any]) -> str:
    return str(tuple(fields[field] for field in KEY_FIELDS))


def resolve_prompt_paths(paths: list[Path]) -> list[Path]:
    resolved: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.jsonl")))
        elif path.suffix == ".jsonl":
            resolved.append(path)
        else:
            raise ValueError(f"Not a JSONL prompt file: {path}")
    return sorted(set(resolved))


def row_body_without_key(row: dict[str, Any]) -> str:
    body = dict(row)
    body.pop("key", None)
    return json.dumps(body, sort_keys=True, ensure_ascii=False)


def is_corrupted_prompt_row(row: dict[str, Any]) -> bool:
    return bool(row.get("corrupted"))


def canonicalize_prompt_file(path: Path, *, dry_run: bool, backup: bool) -> tuple[int, int, int]:
    rows_by_key: dict[str, dict[str, Any]] = {}
    row_bodies: dict[str, str] = {}
    total = 0
    changed = 0
    duplicates = 0
    conflicts: list[str] = []

    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            if "key" not in row:
                raise ValueError(f"{path}:{line_no}: missing prompt key")
            fields = key_dict(row["key"], path=path, line_no=line_no)
            canonical = canonical_fields(fields)
            new_key = key_from_fields(canonical)
            if new_key != row["key"]:
                changed += 1
                row = dict(row)
                row["key"] = new_key

            body = row_body_without_key(row)
            if new_key in rows_by_key:
                duplicates += 1
                existing_row = rows_by_key[new_key]
                existing_corrupted = is_corrupted_prompt_row(existing_row)
                row_corrupted = is_corrupted_prompt_row(row)
                if existing_corrupted and not row_corrupted:
                    rows_by_key[new_key] = row
                    row_bodies[new_key] = body
                    continue
                if row_corrupted and not existing_corrupted:
                    continue
                if row_bodies[new_key] != body:
                    conflicts.append(f"{path}:{line_no}: conflicting duplicate key {new_key}")
                continue
            rows_by_key[new_key] = row
            row_bodies[new_key] = body

    if conflicts:
        for conflict in conflicts[:20]:
            print(conflict)
        if len(conflicts) > 20:
            print(f"... {len(conflicts) - 20} more prompt conflicts")
        raise ValueError(f"Refusing to deduplicate conflicting prompt rows in {path}")

    if (changed or duplicates) and not dry_run:
        if backup:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w") as handle:
            for row in rows_by_key.values():
                json.dump(row, handle, ensure_ascii=False)
                handle.write("\n")
        tmp_path.replace(path)

    return total, changed, duplicates


def normalize_none_series(series: pd.Series) -> pd.Series:
    return series.where(~series.isna(), None).replace({"nan": None, "NaN": None, "<NA>": None, "None": None, "": None})


def canonicalize_score_file(
    path: Path,
    *,
    dry_run: bool,
    backup: bool,
    aggregate_duplicates: bool,
) -> tuple[int, int, int, int]:
    df = pd.read_csv(path)
    missing = [column for column in KEY_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing score columns {missing}")

    total = len(df)
    prompt_type = df["prompt_type"].astype(str)
    baseline_mask = (df["method"].astype(str) == "baseline") | prompt_type.map(is_baseline_prompt_type)
    before = df[KEY_COLUMNS].astype(str)
    df.loc[baseline_mask, "method"] = "baseline"
    df.loc[baseline_mask, "nb_concepts"] = None
    df.loc[baseline_mask, "interpretation"] = None
    changed = int((before != df[KEY_COLUMNS].astype(str)).any(axis=1).sum())

    duplicate_count = int(df.duplicated(KEY_COLUMNS, keep=False).sum())
    aggregated_rows = 0
    if aggregate_duplicates and duplicate_count:
        numeric_columns = [
            column
            for column in df.columns
            if column not in KEY_COLUMNS and pd.api.types.is_numeric_dtype(df[column])
        ]
        other_columns = [column for column in df.columns if column not in KEY_COLUMNS + numeric_columns]
        agg = {column: "mean" for column in numeric_columns}
        agg.update({column: "first" for column in other_columns})
        old_len = len(df)
        df = df.groupby(KEY_COLUMNS, dropna=False, as_index=False).agg(agg)
        for column in ("nb_concepts", "interpretation"):
            df[column] = normalize_none_series(df[column])
        aggregated_rows = old_len - len(df)

    if (changed or aggregated_rows) and not dry_run:
        if backup:
            shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
        df.to_csv(path, index=False)

    return total, changed, duplicate_count, aggregated_rows


def score_paths(score_glob: str) -> list[Path]:
    return sorted(Path().glob(score_glob))


def main() -> None:
    args = parse_args()
    prompt_paths = resolve_prompt_paths(args.prompt_paths)
    csv_paths = score_paths(args.score_glob)
    if not prompt_paths and not csv_paths:
        print("No prompt JSONL files or score CSVs found.")
        sys.exit(1)

    print(f"Mode:                       {'DRY RUN' if args.dry_run else 'APPLY'}")
    print(f"Prompt files:               {len(prompt_paths)}")
    print(f"Score CSVs:                 {len(csv_paths)}")
    print(f"Aggregate score duplicates: {'yes' if args.aggregate_score_duplicates else 'no'}")
    print(f"Backups:                    {'no' if args.no_backup else 'yes'}")
    print()

    total_prompt_rows = total_prompt_changed = total_prompt_dupes = 0
    for path in prompt_paths:
        rows, changed, duplicates = canonicalize_prompt_file(
            path,
            dry_run=args.dry_run,
            backup=not args.no_backup,
        )
        total_prompt_rows += rows
        total_prompt_changed += changed
        total_prompt_dupes += duplicates
        if changed or duplicates:
            print(f"prompt {path}: changed {changed}/{rows}, deduped {duplicates}")

    total_score_rows = total_score_changed = total_score_dupes = total_score_aggregated = 0
    for path in csv_paths:
        rows, changed, duplicates, aggregated = canonicalize_score_file(
            path,
            dry_run=args.dry_run,
            backup=not args.no_backup,
            aggregate_duplicates=args.aggregate_score_duplicates,
        )
        total_score_rows += rows
        total_score_changed += changed
        total_score_dupes += duplicates
        total_score_aggregated += aggregated
        if changed or duplicates or aggregated:
            print(
                f"score  {path}: changed {changed}/{rows}, "
                f"duplicate-key rows {duplicates}, aggregated away {aggregated}"
            )

    print()
    print(f"Prompt rows changed:        {total_prompt_changed}/{total_prompt_rows}")
    print(f"Prompt rows deduped:        {total_prompt_dupes}")
    print(f"Score rows changed:         {total_score_changed}/{total_score_rows}")
    print(f"Score duplicate-key rows:   {total_score_dupes}")
    print(f"Score rows aggregated away: {total_score_aggregated}")
    if args.dry_run:
        print("Dry run only. Re-run without --dry-run to rewrite files.")


if __name__ == "__main__":
    main()
