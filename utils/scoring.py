"""Lightweight prompt-key, response-parsing, and score helpers.

These functions support offline artifact processing and expose the schema used
by local generation. They intentionally do not import model libraries.
"""

from __future__ import annotations

import math
import re


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
SCORE_COLUMNS = (
    *KEY_FIELDS,
    "time",
    "score",
    "num_correct",
    "num_valid",
    "num_expected",
)

SAMPLE_ID_RE = re.compile(r"\bSample_(\d+)\s*:")
CLASS_LABEL_RE = re.compile(r"^class[\s_-]*(\d+)$", re.IGNORECASE)
BARE_INT_RE = re.compile(r"^\d+$")
SCI_TECH_RE = re.compile(r"\bscience\s+(?:and|&)\s+technology\b", re.IGNORECASE)
CLASSES_LINE_RE = re.compile(r"^The classes are:\s*\[(.*)\]\s*$")


def normalize_label_text(text: str | None) -> str | None:
    if text is None:
        return None
    processed = text.strip().strip("`\"'[](){}.,;:")
    processed = re.sub(r"\s+", " ", processed)
    return processed or None


def anonymized_class_id(text: str | None, *, allow_bare_int: bool = False) -> int | None:
    text_norm = normalize_label_text(text)
    if text_norm is None:
        return None
    match = CLASS_LABEL_RE.fullmatch(text_norm)
    if match is not None:
        return int(match.group(1))
    if allow_bare_int and BARE_INT_RE.fullmatch(text_norm):
        return int(text_norm)
    return None


def label_pattern(label: str) -> re.Pattern[str]:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", label) if part]
    body = r"[\s_/+-]*".join(re.escape(part) for part in parts) if parts else re.escape(label)
    return re.compile(rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])", re.IGNORECASE)


def label_prefix_continuation_pattern(label: str) -> re.Pattern[str]:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", label) if part]
    body = r"[\s_/+-]*".join(re.escape(part) for part in parts) if parts else re.escape(label)
    return re.compile(rf"^\s*{body}(?:$|[^A-Za-z])", re.IGNORECASE)


def prediction_matches(predicted: str | None, expected: str) -> bool:
    predicted_norm = normalize_label_text(predicted)
    expected_norm = normalize_label_text(expected)
    if predicted_norm is None or expected_norm is None:
        return False
    if predicted_norm.lower() == expected_norm.lower():
        return True
    expected_class_id = anonymized_class_id(expected_norm)
    predicted_class_id = anonymized_class_id(
        predicted_norm, allow_bare_int=expected_class_id is not None
    )
    return (
        expected_class_id is not None and predicted_class_id == expected_class_id
    ) or label_pattern(expected_norm).fullmatch(predicted_norm) is not None


def extract_allowed_labels(system_prompt: str) -> list[str]:
    for line in system_prompt.splitlines():
        match = CLASSES_LINE_RE.match(line.strip())
        if match is not None:
            return [label.strip() for label in match.group(1).split(",") if label.strip()]
    return []


def extract_sample_ids(text: str) -> list[int]:
    return [int(match.group(1)) for match in SAMPLE_ID_RE.finditer(text)]


def match_expected_label(candidate: str, expected_answers: list[str]) -> str | None:
    candidate_norm = normalize_label_text(candidate)
    if candidate_norm is None:
        return None
    for expected in expected_answers:
        if prediction_matches(candidate_norm, expected):
            return expected
    expected_lower = {answer.lower() for answer in expected_answers}
    lowered = candidate_norm.lower()
    if "pos" in expected_lower and label_pattern("positive").search(lowered):
        return "pos"
    if "neg" in expected_lower and label_pattern("negative").search(lowered):
        return "neg"
    for expected in expected_answers:
        if label_pattern(expected).search(candidate_norm):
            return expected
        if expected.lower() == "sci/tech" and SCI_TECH_RE.search(candidate_norm):
            return expected
    for expected in sorted(expected_answers, key=len, reverse=True):
        if label_prefix_continuation_pattern(expected).match(candidate_norm):
            return expected
    return None


def ordered_prediction_candidates(text: str) -> list[str]:
    processed = text.strip().replace("\n", " ")
    candidates = []
    if ":" in processed:
        candidates.extend((processed.rsplit(":", 1)[1], processed.split(":", 1)[1]))
    tokens = processed.split()
    if tokens:
        candidates.extend((tokens[-1], tokens[0], *reversed(tokens)))
    candidates.append(processed)
    deduped = []
    seen = set()
    for candidate in candidates:
        candidate_key = candidate.strip().lower()
        if candidate_key and candidate_key not in seen:
            deduped.append(candidate)
            seen.add(candidate_key)
    return deduped


def extract_prediction(text: str | None, allowed_answers: list[str] | None = None) -> str | None:
    if text is None:
        return None
    if allowed_answers is not None:
        for candidate in ordered_prediction_candidates(text):
            matched = match_expected_label(candidate, allowed_answers)
            if matched is not None:
                return matched
        return None
    return normalize_label_text(text.strip().replace("\n", " ").split(" ")[-1])


def extract_old_consim_prediction(line: str, allowed_answers: list[str]) -> str | None:
    sample_match = SAMPLE_ID_RE.search(line)
    prediction_text = line[sample_match.end() :] if sample_match else line
    candidates = [prediction_text]
    if ":" in prediction_text:
        candidates[0:0] = [
            prediction_text.rsplit(":", 1)[1],
            prediction_text.split(":", 1)[1],
        ]
    for candidate in candidates:
        matched = match_expected_label(candidate, allowed_answers)
        if matched is not None:
            return matched
    return None


def parse_old_consim_response(
    response: str,
    expected_answers: list[str],
    allowed_answers: list[str] | None = None,
    sample_ids: list[int] | None = None,
) -> list[str | None]:
    labels_to_match = allowed_answers if allowed_answers is not None else expected_answers
    if not response:
        return [None] * len(expected_answers)
    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    if not lines:
        return [None] * len(expected_answers)
    predictions_by_sample_id = {}
    for line in lines:
        sample_match = SAMPLE_ID_RE.search(line)
        if sample_match is not None:
            predictions_by_sample_id[int(sample_match.group(1))] = extract_old_consim_prediction(
                line, labels_to_match
            )
    if sample_ids and any(sample_id in predictions_by_sample_id for sample_id in sample_ids):
        return [predictions_by_sample_id.get(sample_id) for sample_id in sample_ids]
    predictions = [extract_old_consim_prediction(line, labels_to_match) for line in lines]
    return (predictions + [None] * len(expected_answers))[-len(expected_answers) :]


def compute_group_score(
    num_correct: int,
    num_valid: int,
    num_expected: int,
    coverage_ratio: float = 0.7,
) -> float:
    if num_expected <= 0 or num_valid < math.ceil(coverage_ratio * num_expected) or num_valid == 0:
        return float("nan")
    return num_correct / num_valid
