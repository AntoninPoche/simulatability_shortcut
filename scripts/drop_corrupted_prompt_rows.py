"""Remove corrupted prompt JSONL marker rows.

Prompt generation writes explicit corrupted rows when an explanation artifact is
unusable. Those rows intentionally count as existing keys so fragile methods are
not recomputed accidentally. Run this script only when you deliberately want to
remove the markers and retry prompt generation.

Usage examples::

    python scripts/drop_corrupted_prompt_rows.py --dry-run data/prompts/E_concepts.jsonl
    python scripts/drop_corrupted_prompt_rows.py data/prompts/E_concepts.jsonl data/prompts/IMDB_concepts.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drop rows with corrupted=true from prompt JSONL files.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Prompt JSONL file(s) to clean.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts without rewriting files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write <path>.bak before rewriting.",
    )
    return parser.parse_args()


def clean_file(path: Path, *, dry_run: bool, backup: bool) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    kept_rows: list[dict] = []
    total_rows = 0
    corrupted_rows = 0

    with open(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            total_rows += 1
            row = json.loads(line)
            if row.get("corrupted"):
                corrupted_rows += 1
                continue
            kept_rows.append(row)

    print(
        f"{path}: rows={total_rows}, corrupted={corrupted_rows}, kept={len(kept_rows)}"
    )

    if dry_run or corrupted_rows == 0:
        return

    if backup:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as handle:
        for row in kept_rows:
            json.dump(row, handle, ensure_ascii=False)
            handle.write("\n")
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()
    for path in args.paths:
        clean_file(path, dry_run=args.dry_run, backup=not args.no_backup)


if __name__ == "__main__":
    main()
