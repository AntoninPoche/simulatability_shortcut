"""Export compact evaluation-sample metadata for the published predictions.

The full ``local_elements`` caches contain texts and other generation inputs.
This script exports only the sample identifiers, gold labels, and task-model
predictions required by the judge-consistency analysis.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.predictions import TASK_MODEL_BY_DATASET_ABBREV


OUTPUT_COLUMNS = (
    "dataset",
    "classes_subset",
    "seed",
    "cache_n",
    "sample_index",
    "test_index",
    "real_label",
    "task_model_prediction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the compact sample metadata required by notebook 6."
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing predictions_*.csv files (default: data).",
    )
    parser.add_argument(
        "--cache-data-dir",
        type=Path,
        required=True,
        help="Data directory containing task-model local_elements caches.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/sample_metadata.csv"),
        help="Output CSV path (default: data/sample_metadata.csv).",
    )
    return parser.parse_args()


def required_caches(predictions_dir: Path) -> set[tuple[str, str, int, int]]:
    required = set()
    for path in sorted(predictions_dir.glob("predictions_*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["cache_n"]:
                    required.add(
                        (
                            row["dataset"],
                            row["classes_subset"],
                            int(row["seed"]),
                            int(row["cache_n"]),
                        )
                    )
    return required


def cache_path(
    cache_data_dir: Path,
    dataset: str,
    classes_subset: str,
    cache_n: int,
) -> Path:
    model_name = TASK_MODEL_BY_DATASET_ABBREV[dataset]
    model_dir = cache_data_dir / model_name.replace("/", "_")
    class_ids = ast.literal_eval(classes_subset)
    class_slug = "-".join(str(class_id) for class_id in class_ids)
    return model_dir / f"local_elements_classes_{class_slug}_n{cache_n}.json"


def export_rows(required: set[tuple[str, str, int, int]], cache_data_dir: Path) -> list[dict]:
    rows = []
    payloads_by_path: dict[Path, dict] = {}
    for dataset, classes_subset, seed, cache_n in sorted(required):
        path = cache_path(cache_data_dir, dataset, classes_subset, cache_n)
        if path not in payloads_by_path:
            with path.open() as handle:
                payloads_by_path[path] = json.load(handle)
        payload = payloads_by_path[path][str(seed)]
        learning_count = int(payload["nb_learning_samples"])
        for sample_index, (test_index, real_label, prediction) in enumerate(
            zip(
                payload["indices"][learning_count:],
                payload["labels"][learning_count:],
                payload["predictions"][learning_count:],
                strict=True,
            )
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "classes_subset": classes_subset,
                    "seed": seed,
                    "cache_n": cache_n,
                    "sample_index": sample_index,
                    "test_index": test_index,
                    "real_label": real_label,
                    "task_model_prediction": prediction,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    required = required_caches(args.predictions_dir)
    rows = export_rows(required, args.cache_data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):,} rows for {len(required):,} cache slices to {args.output}")


if __name__ == "__main__":
    main()
