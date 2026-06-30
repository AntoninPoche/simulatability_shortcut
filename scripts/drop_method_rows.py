"""Drop one method from prompt JSONL files and score CSVs.

Prompt rows encode the method in item 4 of the prompt ``key`` tuple. Score rows
store it in the ``method`` column. This script removes both consistently, writes
backups by default, and leaves files with no matching rows untouched.

Usage examples::

    python scripts/drop_method_rows.py batchtopk --dry-run
    python scripts/drop_method_rows.py batchtopk
    python scripts/drop_method_rows.py batchtopk --prompt-dirs data/prompts --score-glob 'data/consim*.csv'
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd


METHOD_INDEX = 4
DEFAULT_PROMPT_DIRS = (
    Path("data/prompts"),
    Path("data/prompt_splits"),
    Path("data/best_prompts"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drop one method from prompt JSONL files and score CSVs.",
    )
    parser.add_argument(
        "method",
        help="Method name to drop, e.g. batchtopk.",
    )
    parser.add_argument(
        "--prompt-dirs",
        nargs="+",
        type=Path,
        default=list(DEFAULT_PROMPT_DIRS),
        help=(
            "Prompt directories to scan for *.jsonl files "
            "(default: data/prompts data/prompt_splits data/best_prompts)."
        ),
    )
    parser.add_argument(
        "--score-glob",
        default="data/consim*.csv",
        help="Glob for score CSVs to scan (default: data/consim*.csv).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report matching rows without rewriting files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write <file>.bak before rewriting.",
    )
    return parser.parse_args()


def backup_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".bak")


def copy_backup(path: Path) -> None:
    shutil.copy2(path, backup_path(path))


def prompt_method(row: dict[str, Any], *, path: Path, line_no: int) -> str:
    if "key" not in row:
        raise ValueError(f"{path}:{line_no}: missing prompt key")
    try:
        key = ast.literal_eval(row["key"])
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{path}:{line_no}: invalid prompt key {row['key']!r}") from exc
    if not isinstance(key, tuple) or len(key) <= METHOD_INDEX:
        raise ValueError(f"{path}:{line_no}: invalid prompt key tuple {key!r}")
    return str(key[METHOD_INDEX])


def prompt_paths(prompt_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for prompt_dir in prompt_dirs:
        if not prompt_dir.exists():
            continue
        if not prompt_dir.is_dir():
            raise NotADirectoryError(prompt_dir)
        paths.extend(sorted(prompt_dir.glob("*.jsonl")))
    return sorted(set(paths))


def score_paths(score_glob: str) -> list[Path]:
    return sorted(Path().glob(score_glob))


def drop_from_prompt_file(
    path: Path,
    *,
    method: str,
    dry_run: bool,
    backup: bool,
) -> tuple[int, int]:
    kept_rows: list[dict[str, Any]] = []
    total = 0
    dropped = 0

    with open(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            if prompt_method(row, path=path, line_no=line_no) == method:
                dropped += 1
                continue
            kept_rows.append(row)

    if dropped and not dry_run:
        if backup:
            copy_backup(path)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w") as handle:
            for row in kept_rows:
                json.dump(row, handle, ensure_ascii=False)
                handle.write("\n")
        tmp_path.replace(path)

    return total, dropped


def drop_from_score_file(
    path: Path,
    *,
    method: str,
    dry_run: bool,
    backup: bool,
) -> tuple[int, int]:
    df = pd.read_csv(path)
    if "method" not in df.columns:
        print(f"Skipping score CSV without method column: {path}")
        return len(df), 0

    mask = df["method"].astype(str) == method
    dropped = int(mask.sum())
    if dropped and not dry_run:
        if backup:
            copy_backup(path)
        df.loc[~mask].to_csv(path, index=False)
    return len(df), dropped


def main() -> None:
    args = parse_args()

    prompts = prompt_paths(args.prompt_dirs)
    scores = score_paths(args.score_glob)
    if not prompts and not scores:
        print("No prompt JSONL files or score CSVs found.")
        sys.exit(1)

    mode = "DRY RUN" if args.dry_run else "APPLY"
    print(f"Mode:         {mode}")
    print(f"Method:       {args.method}")
    print(f"Prompt files: {len(prompts)}")
    print(f"Score CSVs:   {len(scores)}")
    print(f"Backups:      {'no' if args.no_backup else 'yes'}")
    print()

    total_prompt_rows = 0
    total_prompt_dropped = 0
    for path in prompts:
        rows, dropped = drop_from_prompt_file(
            path,
            method=args.method,
            dry_run=args.dry_run,
            backup=not args.no_backup,
        )
        total_prompt_rows += rows
        total_prompt_dropped += dropped
        if dropped:
            print(f"prompt {path}: dropped {dropped}/{rows}")

    total_score_rows = 0
    total_score_dropped = 0
    for path in scores:
        rows, dropped = drop_from_score_file(
            path,
            method=args.method,
            dry_run=args.dry_run,
            backup=not args.no_backup,
        )
        total_score_rows += rows
        total_score_dropped += dropped
        if dropped:
            print(f"score  {path}: dropped {dropped}/{rows}")

    print()
    print(f"Prompt rows dropped: {total_prompt_dropped}/{total_prompt_rows}")
    print(f"Score rows dropped:  {total_score_dropped}/{total_score_rows}")
    if args.dry_run:
        print("Dry run only. Re-run without --dry-run to rewrite matching files.")


if __name__ == "__main__":
    main()
