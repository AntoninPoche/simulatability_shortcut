"""Extract best-method prompt subsets into data/best_prompts.

The output rows preserve the source prompt rows exactly: keys, prompt text,
expected answers, and corrupted markers are copied unchanged. The selection is
hard-coded from the Stage 2 best-method choice:

- attributions: lime
- concepts: VanillaSAE with topk interpretation
- rationales: Qwen/Qwen3.5-2B rationales

Rows with old_consim or simulator_consim specifications are excluded.

Usage examples::

    python scripts/extract_best_prompts.py --dry-run
    python scripts/extract_best_prompts.py
    python scripts/extract_best_prompts.py --overwrite
"""

from __future__ import annotations

import argparse
import ast
import json
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

EXCLUDED_SPECIFICATIONS = {"old_consim", "simulator_consim"}
BEST_ATTRIBUTION_METHOD = "lime"
BEST_CONCEPT_METHOD = "VanillaSAE"
BEST_CONCEPT_INTERPRETATION = "topk"
BEST_RATIONALE_METHODS = {"qwen3.5-2b", "Qwen/Qwen3.5-2B"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract hard-coded best prompts into data/best_prompts.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/prompts"),
        help="Directory containing source prompt JSONL files (default: data/prompts).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/best_prompts"),
        help="Directory for best prompt JSONL files (default: data/best_prompts).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report selected row counts without writing files.",
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


def is_baseline_prompt_type(prompt_type: object) -> bool:
    text = str(prompt_type)
    return text.startswith("B") or text.startswith("AB")


def prompt_family(path: Path, fields: dict[str, Any]) -> str | None:
    specification = fields["specification"]
    if specification == "new_consim":
        return "concepts"
    if specification in {"rationales", "attributions"}:
        return str(specification)

    stem = path.stem
    if stem.endswith("_concepts"):
        return "concepts"
    if stem.endswith("_rationales"):
        return "rationales"
    if stem.endswith("_attributions"):
        return "attributions"
    return None


def best_concept_configs(rows: list[tuple[str, dict[str, Any]]]) -> set[tuple[Any, Any]]:
    return {
        (fields["nb_concepts"], fields["interpretation"])
        for _line, fields in rows
        if fields["specification"] not in EXCLUDED_SPECIFICATIONS
        and fields["method"] == BEST_CONCEPT_METHOD
        and fields["interpretation"] == BEST_CONCEPT_INTERPRETATION
    }


def has_canonical_concept_baselines(rows: list[tuple[str, dict[str, Any]]]) -> bool:
    return any(
        fields["specification"] not in EXCLUDED_SPECIFICATIONS
        and fields["method"] == "baseline"
        and fields["nb_concepts"] is None
        and fields["interpretation"] is None
        and is_baseline_prompt_type(fields["prompt_type"])
        for _line, fields in rows
    )


def keep_row(
    path: Path,
    fields: dict[str, Any],
    *,
    concept_configs: set[tuple[Any, Any]],
    canonical_concept_baselines: bool,
) -> bool:
    if fields["specification"] in EXCLUDED_SPECIFICATIONS:
        return False

    family = prompt_family(path, fields)
    if family is None:
        return False

    if family == "attributions":
        if is_baseline_prompt_type(fields["prompt_type"]):
            return True
        return fields["method"] == BEST_ATTRIBUTION_METHOD

    if family == "concepts":
        if is_baseline_prompt_type(fields["prompt_type"]):
            if canonical_concept_baselines:
                return fields["nb_concepts"] is None and fields["interpretation"] is None
            return (fields["nb_concepts"], fields["interpretation"]) in concept_configs
        return (
            fields["method"] == BEST_CONCEPT_METHOD
            and fields["interpretation"] == BEST_CONCEPT_INTERPRETATION
        )

    if family == "rationales":
        if is_baseline_prompt_type(fields["prompt_type"]):
            return True
        return fields["method"] in BEST_RATIONALE_METHODS

    return False


def selected_lines(path: Path) -> tuple[list[str], int]:
    rows: list[tuple[str, dict[str, Any]]] = []
    with open(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "key" not in row:
                raise ValueError(f"{path}:{line_no}: missing prompt key")
            fields = key_dict(row["key"], path=path, line_no=line_no)
            rows.append((line if line.endswith("\n") else line + "\n", fields))

    concept_configs = best_concept_configs(rows)
    canonical_concept_baselines = has_canonical_concept_baselines(rows)
    kept = [
        line
        for line, fields in rows
        if keep_row(
            path,
            fields,
            concept_configs=concept_configs,
            canonical_concept_baselines=canonical_concept_baselines,
        )
    ]
    return kept, len(rows)


def source_paths(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        sys.exit(1)
    if not input_dir.is_dir():
        print(f"Input path is not a directory: {input_dir}")
        sys.exit(1)
    return sorted(
        path
        for path in input_dir.glob("*.jsonl")
        if not path.name.endswith("_old_consim.jsonl")
    )


def write_lines(path: Path, lines: list[str], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {path}. Pass --overwrite to replace it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as handle:
        handle.writelines(lines)
    tmp_path.replace(path)


def main() -> None:
    args = parse_args()
    paths = source_paths(args.input_dir)
    if not paths:
        print(f"No prompt JSONL files found in {args.input_dir}.")
        sys.exit(1)

    print(f"Input dir:  {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Mode:       {'DRY RUN' if args.dry_run else 'APPLY'}")
    print(f"Overwrite:  {'yes' if args.overwrite else 'no'}")
    print()

    total_rows = 0
    total_kept = 0
    planned_writes: list[tuple[Path, list[str]]] = []
    for path in paths:
        lines, rows = selected_lines(path)
        total_rows += rows
        total_kept += len(lines)
        output_path = args.output_dir / path.name
        if lines:
            planned_writes.append((output_path, lines))
            print(f"{path}: selected {len(lines)}/{rows} -> {output_path}")
        else:
            print(f"{path}: selected 0/{rows} (skipping output)")

    print()
    print(f"Selected rows: {total_kept}/{total_rows}")
    if args.dry_run:
        print("Dry run only. Re-run without --dry-run to write best prompt files.")
        return

    for output_path, lines in planned_writes:
        write_lines(output_path, lines, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
