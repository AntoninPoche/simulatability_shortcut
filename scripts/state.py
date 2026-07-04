"""Summarize prompt and v2 score coverage from manifests.

This is a read-only audit script. It derives the expected prompt keys from
``manifests/*.tsv`` rows that call prompt-generation scripts, compares those
keys to ``data/prompts/*.jsonl``, then compares one judge model's v2 score CSV
against the available prompt keys.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import shlex
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data import (  # noqa: E402
    ABBREVIATIONS,
    DATASET_CLASSES_NAMES,
    DATASET_CLASSES_SUBSETS,
    LLM_MODELS,
    MODELS_DATASETS,
    resolve_llm_model,
)
from utils.registries import (  # noqa: E402
    ATTRIBUTION_METHOD_NAMES,
    ATTRIBUTION_PROMPT_ABBREVS,
    CONCEPT_METHOD_NAMES,
    CONCEPT_PROMPT_ABBREVS,
    INTERPRETATION_KEYS,
    RATIONALE_PROMPT_ABBREVS,
)


KEY_COLUMNS = (
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

GENERATION_SCRIPTS = {
    "scripts/make_prompts.py",
    "make_prompts.py",
    "scripts/make_prompts_consim_v2.py",
    "make_prompts_consim_v2.py",
    "scripts/make_prompts_old_consim.py",
    "make_prompts_old_consim.py",
}
SCORING_SCRIPTS = {"scripts/llm_scoring.py", "llm_scoring.py"}


@dataclass
class ManifestCommand:
    manifest: Path
    line_no: int
    command: str
    script: str
    args: dict[str, object]
    expected_keys: set[tuple] = field(default_factory=set)
    error: str | None = None

    @property
    def label(self) -> str:
        return f"{self.manifest}:{self.line_no}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize manifest, prompt JSONL, and v2 score coverage.",
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="qwen3.5-9b",
        help=(
            "Judge model for score coverage (default: qwen3.5-9b). "
            f"Short names: {', '.join(LLM_MODELS)}."
        ),
    )
    parser.add_argument("--manifest-dir", type=Path, default=Path("manifests"))
    parser.add_argument("--prompt-dir", type=Path, default=Path("data/prompts"))
    parser.add_argument("--best-prompt-dir", type=Path, default=Path("data/best_prompts"))
    parser.add_argument(
        "--best-prompt-manifest",
        type=Path,
        default=Path("manifests/best_prompts.tsv"),
    )
    parser.add_argument("--score-path", type=Path, default=None)
    parser.add_argument(
        "--memory-dir",
        type=Path,
        default=Path("data/state_memory"),
        help="Directory for last-run state snapshots (default: data/state_memory).",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Do not read or write last-run state snapshots.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Maximum incomplete manifest rows to print per section. "
            "Use 0 to hide command lists (default: 0)."
        ),
    )
    return parser.parse_args()


def parse_seeds(seeds_str: str) -> list[int]:
    if "-" in seeds_str and "," not in seeds_str:
        start, end = seeds_str.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(s) for s in seeds_str.split(",") if s]


def dataset_maps() -> tuple[dict[str, str], dict[str, str]]:
    abbrev_to_dataset = {v: k for k, v in ABBREVIATIONS["datasets"].items()}
    dataset_to_model = {v: k for k, v in MODELS_DATASETS.items()}
    return abbrev_to_dataset, dataset_to_model


def parse_command(line: str) -> tuple[str, list[str]] | None:
    tokens = shlex.split(line)
    if not tokens:
        return None
    if tokens[0] in {"python", "python3"}:
        tokens = tokens[1:]
    if not tokens:
        return None
    return tokens[0], tokens[1:]


def option_value(tokens: list[str], name: str, default: str | None = None) -> str | None:
    if name not in tokens:
        return default
    idx = tokens.index(name)
    if idx + 1 >= len(tokens):
        return default
    return tokens[idx + 1]


def positional_tokens(tokens: list[str]) -> list[str]:
    out = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token.startswith("--"):
            idx += 2 if idx + 1 < len(tokens) and not tokens[idx + 1].startswith("--") else 1
            continue
        out.append(token)
        idx += 1
    return out


def prompt_file_family(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_old_consim"):
        return "concepts"
    if "_" not in stem:
        return "unknown"
    return stem.split("_", 1)[1]


def key_tuple_from_prompt_key(key: str) -> tuple:
    return ast.literal_eval(key)


def normalize_none(value: object) -> object:
    if value is None:
        return None
    text = str(value)
    if text in {"", "nan", "NaN", "<NA>", "None"}:
        return None
    return value


def normalize_int(value: object) -> object:
    value = normalize_none(value)
    if value is None:
        return None
    try:
        number = float(value)
        if math.isfinite(number) and number.is_integer():
            return int(number)
    except (TypeError, ValueError):
        pass
    return value


def score_row_key(row: dict[str, str]) -> tuple:
    return (
        normalize_none(row["dataset"]),
        normalize_none(row["model"]),
        normalize_none(row["classes_subset"]),
        normalize_int(row["seed"]),
        normalize_none(row["method"]),
        normalize_int(row["nb_concepts"]),
        normalize_none(row["interpretation"]),
        normalize_none(row["prompt_type"]),
        normalize_none(row["specification"]),
    )


def expected_keys(
    *,
    dataset_abbrev: str,
    model_abbrev: str,
    method_name: str,
    classes_subsets: list[list[int]],
    seeds: list[int],
    nb_concepts: int | None,
    interpretation_key: str | None,
    prompt_type_abbrevs: tuple[str, ...],
    specification: str,
) -> set[tuple]:
    keys = set()
    for classes_subset in classes_subsets:
        for seed in seeds:
            for prompt_type_abbrev in prompt_type_abbrevs:
                for anonymized in (True, False):
                    prompt_type = f"A{prompt_type_abbrev}" if anonymized else prompt_type_abbrev
                    is_baseline = prompt_type_abbrev.startswith("B")
                    method_for_key = (
                        "baseline" if is_baseline else method_name
                    )
                    keys.add(
                        (
                            dataset_abbrev,
                            model_abbrev,
                            str(classes_subset),
                            seed,
                            method_for_key,
                            None if is_baseline else nb_concepts,
                            None if is_baseline else interpretation_key,
                            prompt_type,
                            specification,
                        )
                    )
    return keys


def build_expected_for_command(command: ManifestCommand) -> None:
    abbrev_to_dataset, dataset_to_model = dataset_maps()
    tokens = list(command.args["tokens"])
    pos = positional_tokens(tokens)
    seeds = parse_seeds(option_value(tokens, "--seeds", "0-49") or "0-49")
    ratio = float(option_value(tokens, "--nb-concepts-ratio", "3") or "3")

    try:
        if command.script.endswith("make_prompts.py"):
            if len(pos) < 2:
                raise ValueError("missing family/dataset positional arguments")
            family = pos[0]
            dataset_abbrev = pos[1]
            method = pos[2] if len(pos) >= 3 else None
            if family not in {"concepts", "rationales", "attributions"}:
                raise ValueError(f"unknown family {family!r}")
        elif command.script.endswith("make_prompts_consim_v2.py"):
            if len(pos) < 2:
                raise ValueError("missing dataset/method positional arguments")
            family = "concepts"
            dataset_abbrev = pos[0]
            method = pos[1]
        elif command.script.endswith("make_prompts_old_consim.py"):
            if len(pos) < 2:
                raise ValueError("missing dataset/method positional arguments")
            family = "concepts"
            dataset_abbrev = pos[0]
            method = pos[1]
        else:
            raise ValueError(f"unsupported generation script {command.script}")

        if dataset_abbrev not in abbrev_to_dataset:
            raise ValueError(f"unknown dataset abbreviation {dataset_abbrev!r}")

        dataset_name = abbrev_to_dataset[dataset_abbrev]
        model_name = dataset_to_model[dataset_name]
        model_abbrev = ABBREVIATIONS["models"][model_name]
        classes = DATASET_CLASSES_NAMES[dataset_name]
        classes_subsets = DATASET_CLASSES_SUBSETS[dataset_name]

        if command.script.endswith("make_prompts_consim_v2.py"):
            specification = "simulator_consim"
            prompt_types = CONCEPT_PROMPT_ABBREVS
        elif command.script.endswith("make_prompts_old_consim.py"):
            specification = "old_consim"
            prompt_types = CONCEPT_PROMPT_ABBREVS
        elif family == "concepts":
            specification = "new_consim"
            prompt_types = CONCEPT_PROMPT_ABBREVS
        elif family == "rationales":
            specification = "rationales"
            prompt_types = RATIONALE_PROMPT_ABBREVS
        else:
            specification = "attributions"
            prompt_types = ATTRIBUTION_PROMPT_ABBREVS

        if family == "concepts":
            if method not in CONCEPT_METHOD_NAMES:
                raise ValueError(f"unknown concept method {method!r}")
            method_name = CONCEPT_METHOD_NAMES[method]
            nb_concepts = (
                None
                if method == "neurons"
                else len(classes)
                if method == "classes"
                else int(len(classes) * ratio)
            )
            interpretation = option_value(tokens, "--interpretation", "topk") or "topk"
            if interpretation not in INTERPRETATION_KEYS:
                raise ValueError(f"unknown interpretation {interpretation!r}")
            interpretation_key = None if method == "classes" else interpretation
        elif family == "rationales":
            llm_model = option_value(tokens, "--llm-model", "llama3.2-3b") or "llama3.2-3b"
            method_name = resolve_llm_model(llm_model)
            nb_concepts = None
            interpretation_key = None
        else:
            if method not in ATTRIBUTION_METHOD_NAMES:
                raise ValueError(f"unknown attribution method {method!r}")
            method_name = method
            nb_concepts = None
            interpretation_key = None

        command.args.update(
            {
                "family": family,
                "dataset": dataset_abbrev,
                "method": method_name,
                "raw_method": method,
                "interpretation": interpretation_key,
                "specification": specification,
                "seeds": seeds,
                "classes_subsets": [str(s) for s in classes_subsets],
                "prompt_types": prompt_types,
            }
        )
        command.expected_keys = expected_keys(
            dataset_abbrev=dataset_abbrev,
            model_abbrev=model_abbrev,
            method_name=method_name,
            classes_subsets=classes_subsets,
            seeds=seeds,
            nb_concepts=nb_concepts,
            interpretation_key=interpretation_key,
            prompt_type_abbrevs=prompt_types,
            specification=specification,
        )
    except ValueError as exc:
        command.error = str(exc)


def load_manifest_commands(manifest_dir: Path) -> tuple[list[ManifestCommand], list[ManifestCommand]]:
    generation = []
    scoring = []
    for manifest in sorted(manifest_dir.glob("*.tsv")):
        with manifest.open() as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parsed = parse_command(line)
                if parsed is None:
                    continue
                script, tokens = parsed
                command = ManifestCommand(
                    manifest=manifest,
                    line_no=line_no,
                    command=line,
                    script=script,
                    args={"tokens": tokens},
                )
                if script in GENERATION_SCRIPTS:
                    build_expected_for_command(command)
                    generation.append(command)
                elif script in SCORING_SCRIPTS:
                    scoring.append(command)
    return generation, scoring


def load_prompt_keys(
    prompt_dir: Path,
) -> tuple[dict[tuple, dict[str, object]], dict[tuple, dict[str, object]], Counter]:
    prompts: dict[tuple, dict[str, object]] = {}
    corrupted_prompts: dict[tuple, dict[str, object]] = {}
    duplicates = Counter()
    for prompt_path in sorted(prompt_dir.glob("*.jsonl")):
        family = prompt_file_family(prompt_path)
        with prompt_path.open() as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                prompt_group = json.loads(line)
                key = key_tuple_from_prompt_key(prompt_group["key"])
                target = corrupted_prompts if prompt_group.get("corrupted") else prompts
                if key in prompts or key in corrupted_prompts:
                    duplicates[(prompt_path.name, key[0], key[8])] += 1
                target[key] = {
                    "path": prompt_path,
                    "line_no": line_no,
                    "family": family,
                }
    return prompts, corrupted_prompts, duplicates


def load_score_keys(score_path: Path) -> tuple[set[tuple], int, int]:
    if not score_path.exists():
        return set(), 0, 0
    keys = []
    with score_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            keys.append(score_row_key(row))
    unique = set(keys)
    return unique, len(keys), len(keys) - len(unique)


def load_score_keys_and_models(score_path: Path) -> tuple[set[tuple], int, int, set[str]]:
    if not score_path.exists():
        return set(), 0, 0, set()
    keys = []
    models = set()
    with score_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            keys.append(score_row_key(row))
            model = normalize_none(row.get("model"))
            if model is not None:
                models.add(str(model))
    unique = set(keys)
    return unique, len(keys), len(keys) - len(unique), models


def load_valid_prompt_keys_from_file(prompt_path: Path) -> set[tuple]:
    keys = set()
    with prompt_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            prompt_group = json.loads(line)
            if prompt_group.get("corrupted"):
                continue
            keys.add(key_tuple_from_prompt_key(prompt_group["key"]))
    return keys


def load_valid_prompt_keys_from_dir(prompt_dir: Path) -> set[tuple]:
    keys = set()
    if not prompt_dir.exists():
        return keys
    for prompt_path in sorted(prompt_dir.glob("*.jsonl")):
        keys.update(load_valid_prompt_keys_from_file(prompt_path))
    return keys


def best_prompt_manifest_rows(manifest_path: Path) -> dict[Path, int]:
    rows = {}
    if not manifest_path.exists():
        return rows
    with manifest_path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            parsed = parse_command(line.strip())
            if parsed is None:
                continue
            _, tokens = parsed
            pos = positional_tokens(tokens)
            prompt_path = next((Path(token) for token in pos if token.endswith(".jsonl")), None)
            if prompt_path is not None:
                rows[prompt_path] = line_no
    return rows


def judge_label_from_score_path(score_path: Path) -> str:
    stem = score_path.stem
    if stem.startswith("consim_"):
        stem = stem[len("consim_") :]
    if stem.endswith("_v2"):
        stem = stem[: -len("_v2")]
    return stem


def judge_arg_from_score_path(score_path: Path) -> str:
    for alias in LLM_MODELS:
        if score_path_for_model(alias) == score_path:
            return alias
    return judge_label_from_score_path(score_path)


def score_path_for_model(model: str) -> Path:
    resolved = resolve_llm_model(model)
    return Path(f"data/consim_{resolved.replace('/', '_')}_v2.csv")


def memory_path_for_score(score_path: Path, memory_dir: Path) -> Path:
    return memory_dir / f"{score_path.stem}.json"


def load_memory(memory_path: Path) -> dict[str, dict[str, dict[str, float | int]]]:
    if not memory_path.exists():
        return {"rows": {}, "best_prompts": {}}
    with memory_path.open() as handle:
        payload = json.load(handle)
    rows = payload.get("rows", {})
    best_prompts = payload.get("best_prompts", {})
    return {
        "rows": rows if isinstance(rows, dict) else {},
        "best_prompts": best_prompts if isinstance(best_prompts, dict) else {},
    }


def write_memory(
    memory_path: Path,
    *,
    rows: dict[str, dict[str, float | int]],
    best_prompts: dict[str, dict[str, float | int]],
) -> None:
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    with memory_path.open("w") as handle:
        json.dump(
            {"rows": rows, "best_prompts": best_prompts},
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")


def print_table(headers: list[str], rows: list[list[object]]) -> None:
    text_rows = [[str(cell) for cell in row] for row in rows]
    plain_rows = [[strip_ansi(cell) for cell in row] for row in text_rows]
    widths = [len(header) for header in headers]
    for row in plain_rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row, strict=True)]
    print(format_row(headers, widths))
    print(format_row(["-" * width for width in widths], widths))
    for row in text_rows:
        print(format_row(row, widths))


def strip_ansi(text: str) -> str:
    for code in ("\033[31m", "\033[33m", "\033[32m", "\033[0m"):
        text = text.replace(code, "")
    return text


def format_row(row: list[str], widths: list[int]) -> str:
    cells = []
    for cell, width in zip(row, widths, strict=True):
        cells.append(cell + " " * (width - len(strip_ansi(cell))))
    return "  ".join(cells)


def coverage_text(existing: int, expected: int) -> str:
    if expected == 0:
        return "n/a"
    ratio = existing / expected
    text = f"{ratio:.1%}"
    if existing == 0:
        return f"\033[31m{text}\033[0m"
    if existing == expected:
        return f"\033[32m{text}\033[0m"
    return f"\033[33m{text}\033[0m"


def coverage_text_with_corrupted(valid: int, corrupted: int, expected: int) -> str:
    if expected == 0:
        return "n/a"
    valid_ratio = valid / expected
    total_ratio = (valid + corrupted) / expected
    text = f"{valid_ratio:.1%}"
    if corrupted:
        text += f" (+{corrupted / expected:.1%})"
    if valid == 0 and corrupted == 0:
        return f"\033[31m{text}\033[0m"
    if total_ratio >= 1:
        return f"\033[32m{text}\033[0m"
    return f"\033[33m{text}\033[0m"


def number_with_delta(value: int, previous: object | None) -> str:
    try:
        previous_int = int(previous) if previous is not None else value
    except (TypeError, ValueError):
        previous_int = value
    delta = value - previous_int
    if delta == 0:
        return str(value)
    sign = "+" if delta > 0 else ""
    return f"{value} ({sign}{delta})"


def coverage_with_delta(existing: int, expected: int, previous: object | None) -> str:
    text = coverage_text(existing, expected)
    if expected == 0:
        return text
    try:
        previous_ratio = float(previous) if previous is not None else existing / expected
    except (TypeError, ValueError):
        previous_ratio = existing / expected
    delta = (existing / expected) - previous_ratio
    if abs(delta) < 0.0005:
        return text
    sign = "+" if delta > 0 else ""
    return f"{text} ({sign}{delta:.1%})"


def coverage_with_corrupted_delta(
    valid: int,
    corrupted: int,
    expected: int,
    previous: object | None,
) -> str:
    text = coverage_text_with_corrupted(valid, corrupted, expected)
    if expected == 0:
        return text
    try:
        previous_ratio = float(previous) if previous is not None else valid / expected
    except (TypeError, ValueError):
        previous_ratio = valid / expected
    delta = (valid / expected) - previous_ratio
    if abs(delta) < 0.0005:
        return text
    sign = "+" if delta > 0 else ""
    return f"{text} ({sign}{delta:.1%})"


def is_baseline_prompt_type(prompt_type: object) -> bool:
    text = str(prompt_type)
    if text.startswith("A"):
        text = text[1:]
    return text.startswith("B")


def short_list(values: set[object], limit: int = 8) -> str:
    ordered = sorted(str(value) for value in values)
    if len(ordered) <= limit:
        return ", ".join(ordered)
    return ", ".join(ordered[:limit]) + f", ... (+{len(ordered) - limit})"


def summarize_expected(commands: list[ManifestCommand]) -> dict[tuple[str, str], set[tuple]]:
    expected_by_pair: dict[tuple[str, str], set[tuple]] = defaultdict(set)
    dims: dict[tuple[str, str], dict[str, set[object]]] = defaultdict(lambda: defaultdict(set))

    for command in commands:
        if command.error:
            continue
        family = str(command.args["family"])
        specification = str(command.args["specification"])
        pair = (family, specification)
        expected_by_pair[pair].update(command.expected_keys)
        dims[pair]["datasets"].add(command.args["dataset"])
        dims[pair]["methods"].add(command.args["method"])
        dims[pair]["interpretations"].add(command.args["interpretation"])
        dims[pair]["seeds"].update(command.args["seeds"])
        dims[pair]["class_subsets"].update(command.args["classes_subsets"])
        dims[pair]["prompt_types"].update(command.args["prompt_types"])

    print("\nExpected Products From Generation Manifests")
    print("==========================================")
    rows = []
    for pair in sorted(expected_by_pair):
        family, specification = pair
        row_dims = dims[pair]
        rows.append(
            [
                family,
                specification,
                f"{len(row_dims['datasets'])}: {short_list(row_dims['datasets'])}",
                f"{len(row_dims['methods'])}: {short_list(row_dims['methods'])}",
                f"{len(row_dims['interpretations'])}: {short_list(row_dims['interpretations'])}",
                len(row_dims["seeds"]),
                len(row_dims["class_subsets"]),
                len(expected_by_pair[pair]),
            ]
        )
    print_table(
        ["family", "spec", "datasets", "methods", "interpretations", "seeds", "subsets", "keys"],
        rows,
    )
    return expected_by_pair


def summarize_coverage(
    commands: list[ManifestCommand],
    scoring_commands: list[ManifestCommand],
    expected_by_pair: dict[tuple[str, str], set[tuple]],
    prompt_records: dict[tuple, dict[str, object]],
    corrupted_prompt_records: dict[tuple, dict[str, object]],
    score_keys: set[tuple],
    score_rows: int,
    score_dupes: int,
    score_path: Path,
    previous_memory: dict[str, dict[str, float | int]],
    limit: int,
) -> dict[str, dict[str, float | int]]:
    print("\nPrompt And Score Coverage")
    print("=========================")
    print(f"score_path: {score_path}")
    print(f"score_rows: {score_rows}; unique_keys: {len(score_keys)}; duplicates: {score_dupes}")
    print(f"corrupted_prompt_keys: {len(corrupted_prompt_records)}")

    expected_by_triplet: dict[tuple[str, str, str], set[tuple]] = defaultdict(set)
    for command in commands:
        if command.error:
            continue
        family = str(command.args["family"])
        for key in command.expected_keys:
            expected_by_triplet[(key[0], family, key[8])].add(key)

    prompt_by_triplet: dict[tuple[str, str, str], set[tuple]] = defaultdict(set)
    for key, record in prompt_records.items():
        prompt_by_triplet[(key[0], str(record["family"]), key[8])].add(key)

    corrupted_by_triplet: dict[tuple[str, str, str], set[tuple]] = defaultdict(set)
    for key, record in corrupted_prompt_records.items():
        corrupted_by_triplet[(key[0], str(record["family"]), key[8])].add(key)

    rows = []
    current_memory: dict[str, dict[str, float | int]] = {}
    all_triplets = sorted(
        set(expected_by_triplet) | set(prompt_by_triplet) | set(corrupted_by_triplet)
    )
    prompt_keys = set(prompt_records)
    corrupted_prompt_keys = set(corrupted_prompt_records)
    existing_prompt_keys = prompt_keys | corrupted_prompt_keys
    for triplet in all_triplets:
        memory_key = "|".join(triplet)
        previous = previous_memory.get(memory_key, {})
        expected = expected_by_triplet.get(triplet, set())
        prompts = prompt_by_triplet.get(triplet, set())
        corrupted_prompts = corrupted_by_triplet.get(triplet, set())
        expected_count = len(expected)
        prompt_count = len(prompts)
        if expected_count:
            prompt_expected_count = len(expected & prompt_keys)
            corrupted_expected_count = len(expected & corrupted_prompt_keys)
            prompt_ratio = prompt_expected_count / expected_count
            prompt_cov = coverage_with_corrupted_delta(
                prompt_expected_count,
                corrupted_expected_count,
                expected_count,
                previous.get("prompt_coverage"),
            )
            prompt_display = prompt_expected_count
        else:
            prompt_ratio = 0.0
            corrupted_expected_count = len(corrupted_prompts)
            prompt_cov = "n/a"
            prompt_display = prompt_count
        scored = len(prompts & score_keys)
        score_ratio = scored / prompt_count if prompt_count else 0.0
        current_memory[memory_key] = {
            "expected": expected_count,
            "prompts": prompt_display,
            "corrupted_prompts": corrupted_expected_count,
            "prompt_coverage": prompt_ratio,
            "scores": scored,
            "score_target": prompt_count,
            "score_coverage": score_ratio,
        }
        rows.append(
            [
                triplet[0],
                triplet[1],
                triplet[2],
                number_with_delta(expected_count, previous.get("expected")),
                number_with_delta(prompt_display, previous.get("prompts")),
                number_with_delta(corrupted_expected_count, previous.get("corrupted_prompts")),
                prompt_cov,
                number_with_delta(scored, previous.get("scores")),
                number_with_delta(prompt_count, previous.get("score_target")),
                coverage_with_delta(scored, prompt_count, previous.get("score_coverage")),
            ]
        )
    print_table(
        [
            "dataset",
            "family",
            "spec",
            "expected",
            "prompts",
            "corrupted",
            "prompt_cov",
            "scores",
            "score_target",
            "score_cov",
        ],
        rows,
    )

    incomplete = []
    corrupted_generation = []
    invalid = []
    for command in commands:
        if command.error:
            invalid.append(command)
            continue
        actionable_keys = {
            key for key in command.expected_keys if not is_baseline_prompt_type(key[7])
        }
        existing_valid = len(actionable_keys & prompt_keys)
        existing_corrupted = len(actionable_keys & corrupted_prompt_keys)
        existing = len(actionable_keys & existing_prompt_keys)
        if existing_corrupted:
            corrupted_generation.append(
                (
                    command,
                    existing_valid,
                    existing_corrupted,
                    len(actionable_keys) - existing,
                    len(actionable_keys),
                )
            )
        if existing < len(actionable_keys):
            incomplete.append(
                (
                    command,
                    existing_valid,
                    existing_corrupted,
                    len(actionable_keys) - existing,
                    len(actionable_keys),
                )
            )

    if invalid:
        print("\nInvalid Generation Manifest Rows")
        print("--------------------------------")
        print(f"{len(invalid)} invalid row(s).")
        if limit > 0:
            for command in invalid[:limit]:
                print(f"{command.label}: {command.error} :: {command.command}")
            if len(invalid) > limit:
                print(f"... {len(invalid) - limit} more")

    if limit > 0:
        print("\nIncomplete Generation Manifest Rows")
        print("-----------------------------------")
        if not incomplete:
            print("All generation manifest rows are complete.")
        else:
            rows = []
            for command, valid, corrupted, missing, expected in incomplete[:limit]:
                rows.append([command.label, valid, corrupted, missing, expected, command.command])
            print_table(["row", "prompts", "corrupted", "missing", "expected", "command"], rows)
            if len(incomplete) > limit:
                print(f"... {len(incomplete) - limit} more incomplete rows")

    if limit > 0:
        print("\nCorrupted Generation Manifest Rows")
        print("----------------------------------")
        if not corrupted_generation:
            print("No generation manifest rows contain corrupted prompt markers.")
        else:
            rows = []
            for command, valid, corrupted, missing, expected in corrupted_generation[:limit]:
                rows.append([command.label, valid, corrupted, missing, expected, command.command])
            print_table(["row", "prompts", "corrupted", "missing", "expected", "command"], rows)
            if len(corrupted_generation) > limit:
                print(f"... {len(corrupted_generation) - limit} more corrupted rows")

    expected_all = set().union(*expected_by_pair.values()) if expected_by_pair else set()
    unexpected_prompt_keys = existing_prompt_keys - expected_all
    if unexpected_prompt_keys and limit > 0:
        print("\nPrompt Keys Not Required By Generation Manifests")
        print("------------------------------------------------")
        by_triplet = Counter((key[0], key[8]) for key in unexpected_prompt_keys)
        rows = [[dataset, spec, count] for (dataset, spec), count in sorted(by_triplet.items())]
        print_table(["dataset", "spec", "count"], rows)

    if scoring_commands and limit > 0:
        print("\nIncomplete Scoring Manifest Rows")
        print("--------------------------------")
        incomplete = []
        missing_prompt_files = []
        empty_prompt_files = []
        for command in scoring_commands:
            tokens = list(command.args["tokens"])
            pos = positional_tokens(tokens)
            prompt_path = next((Path(token) for token in pos if token.endswith(".jsonl")), None)
            if prompt_path is None:
                continue
            prompt_path = Path(prompt_path)
            if not prompt_path.exists():
                missing_prompt_files.append(command)
                continue
            matching = load_valid_prompt_keys_from_file(prompt_path)
            if not matching:
                empty_prompt_files.append(command)
                continue
            scored = len(matching & score_keys)
            if scored < len(matching):
                incomplete.append((command, scored, len(matching)))
        if not incomplete:
            print("All scoring manifest rows are complete against available prompts.")
        elif limit <= 0:
            print(f"{len(incomplete)} incomplete row(s). Use --limit N to list commands.")
        else:
            rows = []
            for command, scored, prompts in incomplete[:limit]:
                rows.append([command.label, scored, prompts, command.command])
            print_table(["row", "scores", "prompts", "command"], rows)
            if len(incomplete) > limit:
                print(f"... {len(incomplete) - limit} more incomplete rows")

        if missing_prompt_files and limit > 0:
            print("\nMissing Scoring Prompt Files")
            print("----------------------------")
            rows = [
                [command.label, command.command]
                for command in missing_prompt_files[:limit]
            ]
            print_table(["row", "command"], rows)
            if len(missing_prompt_files) > limit:
                print(f"... {len(missing_prompt_files) - limit} more missing files")

    return current_memory


def summarize_best_prompt_coverage(
    best_prompt_dir: Path,
    best_prompt_manifest: Path,
    active_prompt_keys: set[tuple],
    previous_memory: dict[str, dict[str, float | int]],
) -> dict[str, dict[str, float | int]]:
    best_keys_by_path = {
        prompt_path: load_valid_prompt_keys_from_file(prompt_path)
        for prompt_path in sorted(best_prompt_dir.glob("*.jsonl"))
    }
    best_keys = set().union(*best_keys_by_path.values()) if best_keys_by_path else set()
    active_best_keys = best_keys & active_prompt_keys
    stale_best_keys = best_keys - active_prompt_keys
    active_best_keys_by_path = {
        prompt_path: keys & active_prompt_keys
        for prompt_path, keys in best_keys_by_path.items()
        if keys & active_prompt_keys
    }
    manifest_rows_by_path = best_prompt_manifest_rows(best_prompt_manifest)
    current_memory: dict[str, dict[str, float | int]] = {}

    print("\nBest Prompts Score Coverage")
    print("===========================")
    print(f"best_prompt_dir: {best_prompt_dir}")
    print(f"expected_keys:   {len(best_keys)}")
    if stale_best_keys:
        print(f"stale_keys:      {len(stale_best_keys)} (not present in active prompts)")

    if not best_keys:
        print("No best-prompt keys found.")
        return current_memory
    if not active_best_keys:
        print("No active best-prompt keys found.")
        return current_memory

    score_paths = sorted(Path("data").glob("consim*_v2.csv"))
    if not score_paths:
        print("No v2 score CSVs found.")
        return current_memory

    rows = []
    missing_commands = []
    for score_path in score_paths:
        score_keys, score_rows, score_dupes, models = load_score_keys_and_models(score_path)
        judge = judge_label_from_score_path(score_path)
        previous = previous_memory.get(judge, {})
        scored = len(active_best_keys & score_keys)
        expected = len(active_best_keys)
        stale = len(stale_best_keys)
        missing_paths = [
            prompt_path
            for prompt_path, keys in active_best_keys_by_path.items()
            if keys - score_keys
        ]
        missing_count = expected - scored
        if missing_paths:
            manifest_rows = [
                manifest_rows_by_path[path]
                for path in missing_paths
                if path in manifest_rows_by_path
            ]
            command = ""
            if manifest_rows:
                judge_arg = judge_arg_from_score_path(score_path)
                array = ",".join(str(row) for row in sorted(manifest_rows))
                job_name = f"{judge_arg.split('-', 1)[0]}-best-scoring"
                command = " ".join(
                    [
                        f"JUDGE_MODEL={shlex.quote(judge_arg)}",
                        "sbatch",
                        f"--job-name={shlex.quote(job_name)}",
                        f"--array={array}%8",
                        "manifest.sbatch",
                        shlex.quote(str(best_prompt_manifest)),
                    ]
                )
            missing_commands.append([judge, missing_count, len(manifest_rows), command])
        current_memory[judge] = {
            "scores": scored,
            "expected": expected,
            "coverage": scored / expected if expected else 0.0,
            "stale": stale,
            "csv_rows": score_rows,
            "dupes": score_dupes,
        }
        rows.append(
            [
                judge,
                number_with_delta(scored, previous.get("scores")),
                number_with_delta(expected, previous.get("expected")),
                coverage_with_delta(scored, expected, previous.get("coverage")),
                number_with_delta(stale, previous.get("stale")),
                number_with_delta(score_rows, previous.get("csv_rows")),
                number_with_delta(score_dupes, previous.get("dupes")),
                score_path,
            ]
        )

    print_table(
        ["judge", "scores", "expected", "coverage", "stale", "csv_rows", "dupes", "score_path"],
        rows,
    )
    print("\nMissing Best Prompt Scoring Commands")
    print("------------------------------------")
    if not missing_commands:
        print("All active best-prompt keys are scored for every v2 judge CSV.")
    else:
        print_table(["judge", "missing", "manifest_rows", "command"], missing_commands)
    return current_memory


def main() -> None:
    args = parse_args()
    score_path = args.score_path if args.score_path is not None else score_path_for_model(args.model)
    memory_path = memory_path_for_score(score_path, args.memory_dir)
    previous_memory = {"rows": {}, "best_prompts": {}} if args.no_memory else load_memory(memory_path)

    generation_commands, scoring_commands = load_manifest_commands(args.manifest_dir)
    prompt_records, corrupted_prompt_records, prompt_dupes = load_prompt_keys(
        args.prompt_dir
    )
    score_keys, score_rows, score_dupes = load_score_keys(score_path)

    print("State Summary")
    print("=============")
    print(f"generation_manifest_rows: {len(generation_commands)}")
    print(f"scoring_manifest_rows:    {len(scoring_commands)}")
    print(f"prompt_keys:              {len(prompt_records)}")
    print(f"corrupted_prompt_keys:    {len(corrupted_prompt_records)}")
    if not args.no_memory:
        print(f"state_memory:             {memory_path}")
    if prompt_dupes:
        print(f"prompt_duplicate_keys:    {sum(prompt_dupes.values())}")

    expected_by_pair = summarize_expected(generation_commands)
    current_memory = summarize_coverage(
        generation_commands,
        scoring_commands,
        expected_by_pair,
        prompt_records,
        corrupted_prompt_records,
        score_keys,
        score_rows,
        score_dupes,
        score_path,
        previous_memory["rows"],
        args.limit,
    )
    current_best_memory = summarize_best_prompt_coverage(
        args.best_prompt_dir,
        args.best_prompt_manifest,
        set(prompt_records),
        previous_memory["best_prompts"],
    )
    if not args.no_memory:
        write_memory(
            memory_path,
            rows=current_memory,
            best_prompts=current_best_memory,
        )


if __name__ == "__main__":
    main()
