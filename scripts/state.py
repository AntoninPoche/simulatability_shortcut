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
    parser.add_argument("--score-path", type=Path, default=None)
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
                    method_for_key = (
                        "baseline" if prompt_type_abbrev.startswith("B") else method_name
                    )
                    keys.add(
                        (
                            dataset_abbrev,
                            model_abbrev,
                            str(classes_subset),
                            seed,
                            method_for_key,
                            nb_concepts,
                            interpretation_key,
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


def load_prompt_keys(prompt_dir: Path) -> tuple[dict[tuple, dict[str, object]], Counter]:
    prompts: dict[tuple, dict[str, object]] = {}
    duplicates = Counter()
    for prompt_path in sorted(prompt_dir.glob("*.jsonl")):
        family = prompt_file_family(prompt_path)
        with prompt_path.open() as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                prompt_group = json.loads(line)
                if prompt_group.get("corrupted"):
                    continue
                key = key_tuple_from_prompt_key(prompt_group["key"])
                if key in prompts:
                    duplicates[(prompt_path.name, key[0], key[8])] += 1
                prompts[key] = {
                    "path": prompt_path,
                    "line_no": line_no,
                    "family": family,
                }
    return prompts, duplicates


def load_score_keys(score_path: Path) -> tuple[set[tuple], int, int]:
    if not score_path.exists():
        return set(), 0, 0
    keys = []
    with score_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            keys.append(score_row_key(row))
    unique = set(keys)
    return unique, len(keys), len(keys) - len(unique)


def score_path_for_model(model: str) -> Path:
    resolved = resolve_llm_model(model)
    return Path(f"data/consim_{resolved.replace('/', '_')}_v2.csv")


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
    score_keys: set[tuple],
    score_rows: int,
    score_dupes: int,
    score_path: Path,
    limit: int,
) -> None:
    print("\nPrompt And Score Coverage")
    print("=========================")
    print(f"score_path: {score_path}")
    print(f"score_rows: {score_rows}; unique_keys: {len(score_keys)}; duplicates: {score_dupes}")

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

    rows = []
    all_triplets = sorted(set(expected_by_triplet) | set(prompt_by_triplet))
    prompt_keys = set(prompt_records)
    for triplet in all_triplets:
        expected = expected_by_triplet.get(triplet, set())
        prompts = prompt_by_triplet.get(triplet, set())
        expected_count = len(expected)
        prompt_count = len(prompts)
        if expected_count:
            prompt_expected_count = len(expected & prompt_keys)
            prompt_cov = coverage_text(prompt_expected_count, expected_count)
            prompt_display = prompt_expected_count
        else:
            prompt_cov = "n/a"
            prompt_display = prompt_count
        scored = len(prompts & score_keys)
        rows.append(
            [
                triplet[0],
                triplet[1],
                triplet[2],
                expected_count,
                prompt_display,
                prompt_cov,
                scored,
                prompt_count,
                coverage_text(scored, prompt_count),
            ]
        )
    print_table(
        [
            "dataset",
            "family",
            "spec",
            "expected",
            "prompts",
            "prompt_cov",
            "scores",
            "score_target",
            "score_cov",
        ],
        rows,
    )

    incomplete = []
    invalid = []
    for command in commands:
        if command.error:
            invalid.append(command)
            continue
        existing = len(command.expected_keys & prompt_keys)
        if existing < len(command.expected_keys):
            incomplete.append((command, existing, len(command.expected_keys)))

    if invalid:
        print("\nInvalid Generation Manifest Rows")
        print("--------------------------------")
        print(f"{len(invalid)} invalid row(s).")
        if limit > 0:
            for command in invalid[:limit]:
                print(f"{command.label}: {command.error} :: {command.command}")
            if len(invalid) > limit:
                print(f"... {len(invalid) - limit} more")

    print("\nIncomplete Generation Manifest Rows")
    print("-----------------------------------")
    if not incomplete:
        print("All generation manifest rows are complete.")
    elif limit <= 0:
        print(f"{len(incomplete)} incomplete row(s). Use --limit N to list commands.")
    else:
        rows = []
        for command, existing, expected in incomplete[:limit]:
            rows.append([command.label, existing, expected, command.command])
        print_table(["row", "prompts", "expected", "command"], rows)
        if len(incomplete) > limit:
            print(f"... {len(incomplete) - limit} more incomplete rows")

    expected_all = set().union(*expected_by_pair.values()) if expected_by_pair else set()
    unexpected_prompt_keys = prompt_keys - expected_all
    if unexpected_prompt_keys:
        print("\nPrompt Keys Not Required By Generation Manifests")
        print("------------------------------------------------")
        by_triplet = Counter((key[0], key[8]) for key in unexpected_prompt_keys)
        rows = [[dataset, spec, count] for (dataset, spec), count in sorted(by_triplet.items())]
        print_table(["dataset", "spec", "keys"], rows)

    if scoring_commands:
        print("\nIncomplete Scoring Manifest Rows")
        print("--------------------------------")
        incomplete = []
        for command in scoring_commands:
            tokens = list(command.args["tokens"])
            pos = positional_tokens(tokens)
            prompt_path = next((Path(token) for token in pos if token.endswith(".jsonl")), None)
            if prompt_path is None:
                continue
            prompt_path = Path(prompt_path)
            matching = {
                key
                for key, record in prompt_records.items()
                if Path(record["path"]) == prompt_path
            }
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


def main() -> None:
    args = parse_args()
    score_path = args.score_path if args.score_path is not None else score_path_for_model(args.model)

    generation_commands, scoring_commands = load_manifest_commands(args.manifest_dir)
    prompt_records, prompt_dupes = load_prompt_keys(args.prompt_dir)
    score_keys, score_rows, score_dupes = load_score_keys(score_path)

    print("State Summary")
    print("=============")
    print(f"generation_manifest_rows: {len(generation_commands)}")
    print(f"scoring_manifest_rows:    {len(scoring_commands)}")
    print(f"prompt_keys:              {len(prompt_records)}")
    if prompt_dupes:
        print(f"prompt_duplicate_keys:    {sum(prompt_dupes.values())}")

    expected_by_pair = summarize_expected(generation_commands)
    summarize_coverage(
        generation_commands,
        scoring_commands,
        expected_by_pair,
        prompt_records,
        score_keys,
        score_rows,
        score_dupes,
        score_path,
        args.limit,
    )


if __name__ == "__main__":
    main()
