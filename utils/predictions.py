"""Shared helpers for parsing judge generations into sample-level predictions.

Used by ``scripts/parse_generations.py`` and any other script that needs to
turn raw generation JSONL rows into canonical (global class id) predictions,
and optionally join them with the cached local_elements gold labels.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.data import (
    ABBREVIATIONS,
    DATASET_CLASSES_NAMES,
    LLM_MODELS,
    MODELS_DATASETS,
    get_save_root,
)


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

CACHE_N_RE = re.compile(r"_n(\d+)\.json$")

DATASET_NAME_BY_ABBREV = {
    abbreviation: dataset_name
    for dataset_name, abbreviation in ABBREVIATIONS["datasets"].items()
}
TASK_MODEL_BY_DATASET_ABBREV = {
    ABBREVIATIONS["datasets"][dataset_name]: model_name
    for model_name, dataset_name in MODELS_DATASETS.items()
}


def parse_key(raw_key: str) -> dict[str, Any]:
    """Parse a stringified prompt key tuple into a field dict."""
    parsed = ast.literal_eval(raw_key)
    if not isinstance(parsed, tuple) or len(parsed) != len(KEY_FIELDS):
        raise ValueError(
            f"expected a {len(KEY_FIELDS)}-item prompt key, got {parsed!r}"
        )
    return dict(zip(KEY_FIELDS, parsed))


def judge_from_path(path: Path) -> str:
    """Recover the judge model name from a generation log file path."""
    stem = path.stem
    for model in LLM_MODELS.values():
        slug = model.replace("/", "_")
        if stem in {slug, f"{slug}_thinking"}:
            return model
    return stem.removesuffix("_thinking")


def thinking_from_path(path: Path) -> bool:
    """True when the generation log filename ends with ``_thinking``."""
    return path.stem.endswith("_thinking")


def predictions_from_row(row: dict[str, Any]) -> list[str | None]:
    """Return the sample-level predicted labels (as strings) for one row.

    Delegates to the parsers in ``scripts.llm_scoring`` and auto-detects
    old-ConSim (one prompt, many expected answers) versus new/simulator
    ConSim (one prompt per expected answer).
    """
    from scripts.llm_scoring import (  # local import: avoid heavy deps at module import
        extract_prediction,
        extract_sample_ids,
        parse_old_consim_response,
    )

    expected_answers = row["expected_answers"]
    raw_answers = row["raw_answers"]
    user_prompts = row["user_prompts"]
    allowed_answers = row.get("allowed_answers") or expected_answers

    is_old_consim = len(user_prompts) == 1 and len(expected_answers) > 1
    if is_old_consim:
        if not raw_answers:
            return [None] * len(expected_answers)
        return parse_old_consim_response(
            raw_answers[0],
            expected_answers,
            allowed_answers=allowed_answers,
            sample_ids=extract_sample_ids(user_prompts[0]),
        )

    if len(raw_answers) != len(expected_answers):
        raise ValueError(
            "raw answer and expected answer counts differ: "
            f"{len(raw_answers)} != {len(expected_answers)}"
        )
    return [extract_prediction(raw, allowed_answers) for raw in raw_answers]


def parse_classes_subset(value: Any) -> list[int]:
    """Parse the stringified ``classes_subset`` field into a list of ints."""
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list) or not all(isinstance(item, int) for item in parsed):
        raise ValueError(f"Invalid classes_subset: {value!r}")
    return parsed


def class_names_for_dataset(dataset_abbrev: str) -> list[str]:
    """Global class-name list for the dataset abbreviation used in prompt keys."""
    dataset_name = DATASET_NAME_BY_ABBREV.get(dataset_abbrev)
    if dataset_name is None:
        raise ValueError(f"Unknown dataset abbreviation: {dataset_abbrev}")
    return DATASET_CLASSES_NAMES[dataset_name]


def prediction_to_global_id(
    prediction: str | None,
    classes_subset: list[int],
    class_names: list[str],
) -> int | None:
    """Map a parsed judge answer string to a canonical global class id.

    ``prediction`` should already be a member of the allowed labels for the
    prompt (as returned by :func:`predictions_from_row`), or None when the
    judge produced an unparseable answer. Handles both non-anonymized labels
    (real class names) and anonymized labels (``Class_N``), where ``N`` is
    interpreted as the index into ``classes_subset``.
    """
    from scripts.llm_scoring import anonymized_class_id, prediction_matches  # local import

    if prediction is None:
        return None
    subset_index = anonymized_class_id(prediction)
    if subset_index is not None:
        if 0 <= subset_index < len(classes_subset):
            return classes_subset[subset_index]
        return None
    matches = [
        class_id
        for class_id in classes_subset
        if prediction_matches(prediction, class_names[class_id])
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Could not map parsed prediction {prediction!r} to one global class id"
        )
    return matches[0]


@dataclass(frozen=True)
class CacheMatch:
    path: Path
    cache_n: int
    nb_learning_samples: int
    indices: list[int]
    texts: list[str]
    labels: list[int]
    predictions: list[int]


def cache_match_from_payload(
    path: Path,
    cache_n: int,
    payload: dict[str, Any],
) -> CacheMatch:
    """Build a ``CacheMatch`` from one seed entry of a local_elements payload."""
    nb_learning_samples = int(payload["nb_learning_samples"])
    required = ("indices", "texts", "labels", "predictions")
    if any(len(payload[field]) != cache_n for field in required):
        raise ValueError(f"Malformed local-elements payload in {path}")
    evaluation_slice = slice(nb_learning_samples, cache_n)
    return CacheMatch(
        path=path,
        cache_n=cache_n,
        nb_learning_samples=nb_learning_samples,
        indices=[int(value) for value in payload["indices"][evaluation_slice]],
        texts=[str(value) for value in payload["texts"][evaluation_slice]],
        labels=[int(value) for value in payload["labels"][evaluation_slice]],
        predictions=[int(value) for value in payload["predictions"][evaluation_slice]],
    )


class LocalElementsResolver:
    """Match a generation row to the local_elements cache slice it came from.

    Matching is done by rebuilding the evaluation user prompts and expected
    task-model labels from each candidate cache and looking for the unique
    cache that reproduces the row's ``user_prompts`` and ``expected_answers``.
    Cached across rows so we only load each JSON payload once.
    """

    def __init__(self) -> None:
        self._file_payloads: dict[Path, dict[str, Any]] = {}
        self._matches: dict[tuple[Any, ...], CacheMatch] = {}

    def resolve(self, fields: dict[str, Any], row: dict[str, Any]) -> CacheMatch:
        signature = hashlib.sha256(
            json.dumps(
                [row["user_prompts"], row["expected_answers"]],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        match_key = (
            fields["dataset"],
            fields["classes_subset"],
            fields["seed"],
            signature,
        )
        if match_key in self._matches:
            return self._matches[match_key]

        dataset = str(fields["dataset"])
        classes_subset = parse_classes_subset(fields["classes_subset"])
        class_names = class_names_for_dataset(dataset)
        model_name = TASK_MODEL_BY_DATASET_ABBREV[dataset]
        expected_model_abbrev = ABBREVIATIONS["models"][model_name]
        if fields["model"] != expected_model_abbrev:
            raise ValueError(
                f"Task-model abbreviation mismatch for {dataset}: "
                f"{fields['model']!r} != {expected_model_abbrev!r}"
            )

        subset_slug = "-".join(str(class_id) for class_id in classes_subset)
        cache_paths = sorted(
            get_save_root(model_name).glob(
                f"local_elements_classes_{subset_slug}_n*.json"
            )
        )
        matches: list[CacheMatch] = []
        for path in cache_paths:
            cache_n_match = CACHE_N_RE.search(path.name)
            if cache_n_match is None:
                continue
            cache_n = int(cache_n_match.group(1))
            payloads = self._load(path)
            payload = payloads.get(str(fields["seed"]))
            if payload is None:
                continue
            match = cache_match_from_payload(path, cache_n, payload)
            expected_prompts = [
                f"Evaluation sample:\n\tText: {text}\n\tLabel: "
                for text in match.texts
            ]
            expected_answers = [class_names[pred] for pred in match.predictions]
            if (
                expected_prompts == row["user_prompts"]
                and expected_answers == row["expected_answers"]
            ):
                matches.append(match)

        if len(matches) != 1:
            raise ValueError(
                "Expected exactly one local-elements cache match for "
                f"{fields['dataset']} {fields['classes_subset']} seed={fields['seed']}, "
                f"found {len(matches)} among {[str(path) for path in cache_paths]}"
            )
        self._matches[match_key] = matches[0]
        return matches[0]

    def _load(self, path: Path) -> dict[str, Any]:
        if path not in self._file_payloads:
            with path.open() as handle:
                self._file_payloads[path] = json.load(handle)
        return self._file_payloads[path]


def load_sample_metadata(
    dataset_abbrev: str,
    classes_subset: str,
    seed: int,
    cache_n: int,
    data_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Return per-sample metadata rows for one local_elements cache entry.

    Each returned dict has keys ``sample_index``, ``test_index``, ``real_label``,
    and ``task_model_prediction``. ``sample_index`` is 0-based within the
    evaluation slice (i.e. lines up with the ``pred_i`` columns produced by
    :mod:`scripts.parse_generations`).

    ``data_root`` overrides the repo-relative ``data/`` prefix used by
    :func:`utils.data.get_save_root`; pass ``REPO_ROOT`` from notebooks whose
    working directory is not the repo root.
    """
    model_name = TASK_MODEL_BY_DATASET_ABBREV[dataset_abbrev]
    classes = parse_classes_subset(classes_subset)
    subset_slug = "-".join(str(class_id) for class_id in classes)
    save_root = get_save_root(model_name)
    if data_root is not None:
        save_root = data_root / save_root
    payload_path = save_root / f"local_elements_classes_{subset_slug}_n{cache_n}.json"
    with payload_path.open() as handle:
        payloads = json.load(handle)
    payload = payloads[str(seed)]
    cache_match = cache_match_from_payload(payload_path, cache_n, payload)
    return [
        {
            "sample_index": sample_index,
            "test_index": cache_match.indices[sample_index],
            "real_label": cache_match.labels[sample_index],
            "task_model_prediction": cache_match.predictions[sample_index],
        }
        for sample_index in range(len(cache_match.indices))
    ]
