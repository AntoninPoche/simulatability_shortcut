"""Create split prompt JSONL files without modifying the source file.

The split files preserve each prompt row's keys, prompt text, expected answers,
and corrupted markers unchanged.
By default rows are grouped by the ``method`` field encoded in the prompt key and
written outside ``data/prompts/`` so default scoring does not pick up duplicates.

Usage examples::

    python scripts/split_prompt_file.py data/prompts/RT_concepts.jsonl
    python scripts/split_prompt_file.py data/prompts/RT_concepts.jsonl --by method
    python scripts/split_prompt_file.py data/prompts/RT_concepts.jsonl --by specification
    python scripts/split_prompt_file.py data/prompts/RT_concepts.jsonl --by method --dry-run
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections import defaultdict
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
        description="Split a prompt JSONL file into derived files grouped by key field.",
    )
    parser.add_argument(
        "prompt_file",
        type=Path,
        help="Source prompt JSONL file to split. It is never modified.",
    )
    parser.add_argument(
        "--by",
        choices=KEY_FIELDS,
        default="method",
        help="Prompt-key field to group by (default: method).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/prompt_splits"),
        help="Directory for split JSONL files (default: data/prompt_splits).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing split files for this source/grouping.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report split sizes without writing files.",
    )
    return parser.parse_args()


def key_dict(raw_key: str, *, path: Path, line_no: int) -> dict[str, Any]:
    try:
        parsed = ast.literal_eval(raw_key)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{path}:{line_no}: invalid prompt key {raw_key!r}") from exc

    if not isinstance(parsed, tuple) or len(parsed) != len(KEY_FIELDS):
        raise ValueError(
            f"{path}:{line_no}: expected a {len(KEY_FIELDS)}-item prompt key tuple, "
            f"got {parsed!r}"
        )
    return dict(zip(KEY_FIELDS, parsed))


def slugify(value: Any) -> str:
    text = "None" if value is None else str(value)
    text = text.strip() or "empty"
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    return text.strip("-._") or "empty"


def split_path(source_path: Path, output_dir: Path, field: str, value: Any) -> Path:
    return output_dir / f"{source_path.stem}__{field}-{slugify(value)}.jsonl"


def load_groups(path: Path, field: str) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    with open(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "key" not in row:
                raise ValueError(f"{path}:{line_no}: missing prompt key")
            fields = key_dict(row["key"], path=path, line_no=line_no)
            groups[fields[field]].append(row)
    return groups


def main() -> None:
    args = parse_args()

    if not args.prompt_file.exists():
        print(f"Prompt file not found: {args.prompt_file}")
        sys.exit(1)

    groups = load_groups(args.prompt_file, args.by)
    if not groups:
        print(f"No prompt rows found in {args.prompt_file}")
        return

    planned_paths = {
        value: split_path(args.prompt_file, args.output_dir, args.by, value)
        for value in groups
    }
    if len(set(planned_paths.values())) != len(planned_paths):
        print("Split output path collision after slugifying group values:")
        seen: dict[Path, Any] = {}
        for value, path in planned_paths.items():
            if path in seen:
                print(f"  - {seen[path]!r} and {value!r} -> {path}")
            seen[path] = value
        sys.exit(1)

    existing_paths = [path for path in planned_paths.values() if path.exists()]
    if existing_paths and not args.overwrite and not args.dry_run:
        print("Refusing to overwrite existing split files:")
        for path in sorted(existing_paths):
            print(f"  - {path}")
        print("Pass --overwrite to replace them.")
        sys.exit(1)

    print(f"Source:       {args.prompt_file}")
    print(f"Split field:  {args.by}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Groups:       {len(groups)}")

    if args.dry_run:
        for value, rows in sorted(groups.items(), key=lambda item: slugify(item[0])):
            print(f"  {args.by}={value!r}: {len(rows)} rows -> {planned_paths[value]}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for value, rows in sorted(groups.items(), key=lambda item: slugify(item[0])):
        path = planned_paths[value]
        with open(path, "w") as handle:
            for row in rows:
                json.dump(row, handle, ensure_ascii=False)
                handle.write("\n")
        print(f"  wrote {len(rows)} rows -> {path}")


if __name__ == "__main__":
    main()
