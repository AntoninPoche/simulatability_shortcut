"""Parse raw judge generations into wide sample-level predictions CSVs.

Reads ``data/generations/*.jsonl`` (one per judge) and writes one wide CSV per
judge under ``data/predictions_{judge_slug}[_thinking].csv``.

Each output row corresponds to one prompt group. Sample columns hold the
parsed judge prediction as a **global class id** (i.e. the index into
``utils.data.DATASET_CLASSES_NAMES``), or empty when the judge produced an
unparseable answer. Uses latest-wins deduplication on the raw prompt key,
matching the semantics of ``scripts/llm_scoring.py``.

Usage examples::

    python scripts/parse_generations.py
    python scripts/parse_generations.py data/generations/Qwen_Qwen3.5-9B.jsonl
    python scripts/parse_generations.py --output-dir data --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.predictions import (
    KEY_FIELDS,
    LocalElementsResolver,
    class_names_for_dataset,
    judge_from_path,
    parse_classes_subset,
    parse_key,
    prediction_to_global_id,
    predictions_from_row,
    thinking_from_path,
)


GENERATION_DIR = Path("data/generations")
DEFAULT_OUTPUT_DIR = Path("data")
MAX_SAMPLES = 40  # covers old_consim; unused indices left empty


PROVENANCE_COLUMNS = ("judge", "thinking")
COUNT_COLUMNS = ("num_expected", "num_valid", "num_correct", "cache_n")
SAMPLE_COLUMNS = tuple(f"pred_{i}" for i in range(MAX_SAMPLES))
OUTPUT_COLUMNS = (
    *KEY_FIELDS,
    *PROVENANCE_COLUMNS,
    *COUNT_COLUMNS,
    *SAMPLE_COLUMNS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse generation logs into wide sample-level predictions CSVs.",
    )
    parser.add_argument(
        "generation_paths",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Generation JSONL files or directories to parse. Defaults to all "
            "data/generations/*.jsonl."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for predictions_*.csv (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing predictions_*.csv files.",
    )
    parser.add_argument(
        "--no-cache-n",
        action="store_true",
        help=(
            "Skip local-elements matching. Faster and does not require the "
            "local_elements caches on disk, but leaves cache_n empty."
        ),
    )
    return parser.parse_args()


def resolve_generation_paths(paths: list[Path] | None) -> list[Path]:
    if not paths:
        return sorted(GENERATION_DIR.glob("*.jsonl"))
    resolved: list[Path] = []
    for path in paths:
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.jsonl")))
        else:
            resolved.append(path)
    return sorted(set(resolved))


def load_latest_rows(path: Path) -> dict[str, dict[str, Any]]:
    """Return the latest row for each raw_key in a generation log."""
    latest: dict[str, dict[str, Any]] = {}
    malformed = 0
    with path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            key = row.get("key")
            if not isinstance(key, str):
                malformed += 1
                continue
            row["_line_no"] = line_no
            latest[key] = row
    if malformed:
        print(f"  Warning: skipped {malformed} malformed JSON lines in {path}")
    return latest


def build_output_row(
    raw_key: str,
    row: dict[str, Any],
    judge: str,
    thinking: bool,
    resolver: LocalElementsResolver | None,
) -> dict[str, Any]:
    fields = parse_key(raw_key)
    classes_subset = parse_classes_subset(fields["classes_subset"])
    class_names = class_names_for_dataset(str(fields["dataset"]))
    specification = str(fields["specification"])

    string_predictions = predictions_from_row(row)
    if len(string_predictions) > MAX_SAMPLES:
        raise ValueError(
            f"prompt group has {len(string_predictions)} samples, exceeds "
            f"MAX_SAMPLES={MAX_SAMPLES}: {raw_key}"
        )

    global_predictions: list[int | None] = []
    for prediction in string_predictions:
        try:
            global_predictions.append(
                prediction_to_global_id(
                    prediction,
                    classes_subset,
                    class_names,
                    specification,
                )
            )
        except ValueError:
            global_predictions.append(None)

    expected_answers = row["expected_answers"]
    expected_global: list[int | None] = []
    for expected in expected_answers:
        try:
            expected_global.append(
                prediction_to_global_id(
                    expected,
                    classes_subset,
                    class_names,
                    specification,
                )
            )
        except ValueError:
            expected_global.append(None)

    num_expected = len(expected_answers)
    num_valid = sum(pred is not None for pred in global_predictions)
    num_correct = sum(
        int(pred is not None and exp is not None and pred == exp)
        for pred, exp in zip(global_predictions, expected_global, strict=True)
    )

    cache_n: int | str = ""
    if resolver is not None:
        try:
            cache_match = resolver.resolve(fields, row)
            cache_n = cache_match.cache_n
        except (KeyError, ValueError):
            cache_n = ""

    output = {name: fields[name] for name in KEY_FIELDS}
    output["judge"] = judge
    output["thinking"] = thinking
    output["num_expected"] = num_expected
    output["num_valid"] = num_valid
    output["num_correct"] = num_correct
    output["cache_n"] = cache_n
    for i, prediction in enumerate(global_predictions):
        output[f"pred_{i}"] = "" if prediction is None else prediction
    for i in range(len(global_predictions), MAX_SAMPLES):
        output[f"pred_{i}"] = ""
    return output


def write_predictions_csv(
    output_path: Path,
    rows: list[dict[str, Any]],
    *,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(output_path)


def process_generation_file(
    path: Path,
    output_dir: Path,
    *,
    overwrite: bool,
    with_cache_n: bool,
) -> tuple[int, int]:
    judge = judge_from_path(path)
    thinking = thinking_from_path(path)
    output_path = output_dir / f"predictions_{path.stem}.csv"

    latest = load_latest_rows(path)
    resolver = LocalElementsResolver() if with_cache_n else None
    rows: list[dict[str, Any]] = []
    parse_errors = 0
    for raw_key, row in latest.items():
        try:
            rows.append(build_output_row(raw_key, row, judge, thinking, resolver))
        except (KeyError, TypeError, ValueError) as exc:
            parse_errors += 1
            print(f"  Warning: parse error for {raw_key}: {exc}")

    write_predictions_csv(output_path, rows, overwrite=overwrite)
    return len(rows), parse_errors


def main() -> None:
    args = parse_args()
    generation_paths = resolve_generation_paths(args.generation_paths)
    if not generation_paths:
        raise SystemExit(f"No generation JSONL files found under {GENERATION_DIR}")

    print(f"Generation files: {len(generation_paths)}")
    print(f"Output dir:       {args.output_dir}")
    print(f"cache_n lookup:   {'on' if not args.no_cache_n else 'off'}")
    print()

    total_rows = 0
    total_errors = 0
    for index, path in enumerate(generation_paths, start=1):
        print(f"[{index}/{len(generation_paths)}] {path}")
        rows, errors = process_generation_file(
            path,
            args.output_dir,
            overwrite=args.overwrite,
            with_cache_n=not args.no_cache_n,
        )
        total_rows += rows
        total_errors += errors
        print(f"  wrote {rows:,} rows, {errors:,} parse errors")

    print()
    print(f"Total: {total_rows:,} rows written, {total_errors:,} parse errors")


if __name__ == "__main__":
    main()
