"""Build V3 score CSVs from the wide sample-level predictions CSVs.

Reads ``data/predictions_{judge}.csv`` (produced by ``scripts/parse_generations.py``)
and writes ``data/consim_{judge}_v3.csv`` in the same schema as the existing
V2 score CSVs. Scores are recomputed from parsed predictions with the current
``compute_group_score`` implementation, so V3 reflects the latest parser fixes.

Usage::

    python scripts/build_scores_v3.py
    python scripts/build_scores_v3.py --predictions-dir data --output-dir data
    python scripts/build_scores_v3.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.llm_scoring import KEY_FIELDS, SCORE_COLUMNS, compute_group_score


DEFAULT_PREDICTIONS_DIR = Path("data")
DEFAULT_OUTPUT_DIR = Path("data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute V3 score CSVs from wide predictions CSVs.",
    )
    parser.add_argument(
        "predictions_files",
        nargs="*",
        type=Path,
        default=None,
        help=(
            "Specific predictions_*.csv files to convert. Defaults to all "
            "predictions_*.csv under --predictions-dir."
        ),
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=DEFAULT_PREDICTIONS_DIR,
        help=f"Directory holding predictions_*.csv (default: {DEFAULT_PREDICTIONS_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for consim_*_v3.csv (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing consim_*_v3.csv files.",
    )
    return parser.parse_args()


def resolve_predictions_files(
    files: list[Path] | None,
    predictions_dir: Path,
) -> list[Path]:
    if files:
        return list(files)
    return sorted(predictions_dir.glob("predictions_*.csv"))


def output_path_for(predictions_file: Path, output_dir: Path) -> Path:
    stem = predictions_file.stem[len("predictions_"):]
    return output_dir / f"consim_{stem}_v3.csv"


def build_score_row(prediction_row: dict[str, str], now: str) -> dict[str, str]:
    num_expected = int(prediction_row["num_expected"])
    num_valid = int(prediction_row["num_valid"])
    num_correct = int(prediction_row["num_correct"])
    score = compute_group_score(num_correct, num_valid, num_expected)
    output = {name: prediction_row[name] for name in KEY_FIELDS}
    output["time"] = now
    output["score"] = "" if score != score else score  # NaN -> empty
    output["num_correct"] = num_correct
    output["num_valid"] = num_valid
    output["num_expected"] = num_expected
    return output


def build_scores_for_file(
    predictions_file: Path,
    output_path: Path,
    *,
    overwrite: bool,
) -> int:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )
    now = datetime.now().isoformat()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    # Latest-wins dedup on the 9-field prompt key. Predictions CSVs written by
    # parse_generations.py already dedupe per raw_key, but a user could
    # regenerate a subset later and concat, so keep the safety net here.
    latest: dict[tuple, dict[str, str]] = {}
    with predictions_file.open(newline="") as prediction_handle:
        reader = csv.DictReader(prediction_handle)
        for row in reader:
            key = tuple(row[name] for name in KEY_FIELDS)
            latest[key] = row

    written = 0
    with tmp_path.open("w", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=SCORE_COLUMNS)
        writer.writeheader()
        for row in latest.values():
            writer.writerow(build_score_row(row, now))
            written += 1
    tmp_path.replace(output_path)
    return written


def main() -> None:
    args = parse_args()
    predictions_files = resolve_predictions_files(
        args.predictions_files, args.predictions_dir
    )
    if not predictions_files:
        raise SystemExit(
            f"No predictions_*.csv files found in {args.predictions_dir}"
        )

    print(f"Predictions files: {len(predictions_files)}")
    print(f"Output dir:        {args.output_dir}")
    print()

    total = 0
    for index, path in enumerate(predictions_files, start=1):
        output_path = output_path_for(path, args.output_dir)
        rows = build_scores_for_file(path, output_path, overwrite=args.overwrite)
        total += rows
        print(f"[{index}/{len(predictions_files)}] {path} -> {output_path}: {rows:,} rows")

    print()
    print(f"Total: {total:,} score rows written")


if __name__ == "__main__":
    main()
