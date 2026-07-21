"""Generate simulator-framed ConSim prompts for concept experiments.

This script reuses the current ConSim sample selection and concept artifacts, but
renders prompts with ``utils.simulator_consim.SimulatorConSim``. The key difference from
``new_consim`` is the prompt framing: the LLM is asked to simulate a text
classifier's predictions, not to predict the true label.

Output is appended to ``data/prompts/{dataset_abbrev}_concepts.jsonl`` with
``specification == "simulator_consim"`` in the prompt key.
"""

from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import sys
from pathlib import Path

from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data import (
    ABBREVIATIONS,
    DATASET_CLASSES_NAMES,
    DATASET_CLASSES_SUBSETS,
    LLM_MODELS,
    MODELS_DATASETS,
    get_save_root,
    iter_jsonl,
    load_dataset_splits,
    load_or_compute_dataset_activations,
    load_or_compute_local_elements,
    resolve_llm_model,
)
from utils.registries import CONCEPT_METHOD_NAMES, CONCEPT_PROMPT_ABBREVS, INTERPRETATION_KEYS


_ABBREV_TO_DATASET: dict[str, str] = {v: k for k, v in ABBREVIATIONS["datasets"].items()}
_DATASET_TO_MODEL: dict[str, str] = {v: k for k, v in MODELS_DATASETS.items()}
CONCEPT_METHODS = set(CONCEPT_METHOD_NAMES.keys())
SPECIFICATION = "simulator_consim"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate simulator-framed ConSim concept prompts.",
    )
    parser.add_argument(
        "dataset",
        choices=sorted(_ABBREV_TO_DATASET.keys()),
        help="Dataset abbreviation (e.g. GE, HE, BIOS, E).",
    )
    parser.add_argument(
        "method",
        choices=sorted(CONCEPT_METHODS),
        help="Concept extraction method.",
    )
    parser.add_argument(
        "--nb-concepts-ratio",
        type=float,
        default=3,
        help="Number of concepts = nb_classes * ratio (default: 3).",
    )
    parser.add_argument(
        "--interpretation",
        choices=sorted(INTERPRETATION_KEYS),
        default="topk",
        help="Interpretation method for concept labeling (default: topk).",
    )
    parser.add_argument(
        "--llm-model",
        default="llama3.2-3b",
        help=(
            f"LLM model for concept LLM interpretation (default: llama3.2-3b). "
            f"Short names: {', '.join(LLM_MODELS.keys())}."
        ),
    )
    parser.add_argument(
        "--seeds",
        default="0-49",
        help="Seed range (e.g. '0-49') or comma-separated list (e.g. '0,1,2').",
    )
    parser.add_argument(
        "--nb-samples",
        type=int,
        default=40,
        help="Number of samples per seed (default: 40).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for computation (default: cuda; falls back to cpu if unavailable).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=96,
        help="Batch size for model inference (default: 96).",
    )
    return parser.parse_args()


def parse_seeds(seeds_str: str) -> list[int]:
    if "-" in seeds_str and "," not in seeds_str:
        start, end = seeds_str.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(s) for s in seeds_str.split(",")]


def compute_expected_keys(
    *,
    dataset_abbrev: str,
    model_abbrev: str,
    method_name: str,
    classes_subsets: list[list[int]],
    seeds: list[int],
    nb_concepts: int | None,
    interpretation_key: str | None,
) -> set[str]:
    expected = set()
    for classes_subset in classes_subsets:
        for seed in seeds:
            for prompt_type_abbrev in CONCEPT_PROMPT_ABBREVS:
                for anonym in [True, False]:
                    prompt_type_name = "A" + prompt_type_abbrev if anonym else prompt_type_abbrev
                    is_baseline = prompt_type_abbrev.startswith("B")
                    method_for_key = "baseline" if is_baseline else method_name
                    key = str(
                        (
                            dataset_abbrev,
                            model_abbrev,
                            str(classes_subset),
                            seed,
                            method_for_key,
                            None if is_baseline else nb_concepts,
                            None if is_baseline else interpretation_key,
                            prompt_type_name,
                            SPECIFICATION,
                        )
                    )
                    expected.add(key)
    return expected


def append_prompt_group_if_missing(
    output_path: Path,
    prompt_group: dict,
    existing_keys: set[str],
) -> bool:
    key = prompt_group["key"]
    lock_path = output_path.with_suffix(output_path.suffix + ".lock")
    with open(lock_path, "w") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        try:
            if output_path.exists():
                for existing_group in iter_jsonl(output_path):
                    if existing_group["key"] == key:
                        existing_keys.add(key)
                        return False

            with open(output_path, "a") as handle:
                json.dump(prompt_group, handle)
                handle.write("\n")
            existing_keys.add(key)
            return True
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)


def validate_prompt_group(
    *,
    key: str,
    user_prompts: list[str],
    expected_answers: list[str],
    classes_subset: list[int],
) -> None:
    if len(user_prompts) != len(expected_answers):
        raise ValueError(
            f"Invalid prompt group {key}: {len(user_prompts)} user prompts but "
            f"{len(expected_answers)} expected answers."
        )
    if len(classes_subset) > 1 and len(set(expected_answers)) < 2:
        raise ValueError(
            f"Degenerate prompt group {key}: expected answers contain only "
            f"{sorted(set(expected_answers))}. Check cached model predictions/local elements."
        )


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)

    dataset_name = _ABBREV_TO_DATASET[args.dataset]
    model_name = _DATASET_TO_MODEL[dataset_name]
    save_root = get_save_root(model_name)
    save_root.mkdir(parents=True, exist_ok=True)
    args.llm_model = resolve_llm_model(args.llm_model)

    dataset_abbrev = args.dataset
    model_abbrev = ABBREVIATIONS["models"][model_name]
    classes = DATASET_CLASSES_NAMES[dataset_name]
    all_subsets = DATASET_CLASSES_SUBSETS[dataset_name]
    nb_concepts = (
        None
        if args.method == "neurons"
        else len(classes)
        if args.method == "classes"
        else int(len(classes) * args.nb_concepts_ratio)
    )
    interpretation_key = None if args.method == "classes" else args.interpretation
    method_for_key = CONCEPT_METHOD_NAMES[args.method]

    output_path = Path(f"data/prompts/{args.dataset}_concepts.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = {prompt_group["key"] for prompt_group in iter_jsonl(output_path)} if output_path.exists() else set()

    expected_keys = compute_expected_keys(
        dataset_abbrev=dataset_abbrev,
        model_abbrev=model_abbrev,
        method_name=method_for_key,
        classes_subsets=all_subsets,
        seeds=seeds,
        nb_concepts=nb_concepts,
        interpretation_key=interpretation_key,
    )
    if expected_keys <= existing_keys:
        print(f"All {len(expected_keys)} entries already exist in {output_path}. Skipping.")
        sys.exit(0)

    import torch

    from utils.concepts import load_local_importances, load_or_build_concept_resources
    from utils.simulator_consim import SimulatorConSim, PromptTypes

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    print(f"Dataset:         {dataset_name} ({args.dataset})")
    print(f"Model:           {model_name}")
    print(f"Method:          {args.method}")
    print(f"Interpretation:  {interpretation_key}")
    print(f"Specification:   {SPECIFICATION}")
    print(f"Seeds:           {seeds[0]}-{seeds[-1]} ({len(seeds)} seeds)")
    print(f"Output:          {output_path}")
    print()

    train_inputs, validation_inputs, test_inputs, test_labels = load_dataset_splits(dataset_name)
    _, _, test_artifacts = load_or_compute_dataset_activations(
        model_name=model_name,
        save_root=save_root,
        train_inputs=train_inputs,
        validation_inputs=validation_inputs,
        test_inputs=test_inputs,
        device=args.device,
        batch_size=args.batch_size,
    )
    _, test_predictions = test_artifacts

    args.train_inputs = train_inputs
    args.validation_inputs = validation_inputs
    concept_resources = load_or_build_concept_resources(
        args,
        dataset_name=dataset_name,
        model_name=model_name,
        save_root=save_root,
        train_inputs=train_inputs,
        validation_inputs=validation_inputs,
        test_inputs=test_inputs,
        classes=classes,
        test_predictions=test_predictions,
    )

    prompt_types = {
        PromptTypes.B1_baseline_without_lp,
        PromptTypes.C1_global_concepts_without_lp,
        PromptTypes.B2_baseline_with_lp,
        PromptTypes.C2_global_concepts_with_lp,
        PromptTypes.C3_global_and_local_concepts_with_lp,
    }
    simulatability_metric = SimulatorConSim(classes=classes)
    total_new = 0

    for classes_subset in all_subsets:
        print(f"Processing classes subset: {classes_subset}")
        local_elements_by_seed = load_or_compute_local_elements(
            save_root=save_root,
            simulatability_metric=simulatability_metric,
            inputs=test_inputs,
            labels=test_labels,
            predictions=test_predictions,
            seeds=seeds,
            classes_subset=classes_subset,
            nb_samples=args.nb_samples,
        )

        subset_new = 0
        with tqdm(
            total=len(seeds) * len(prompt_types) * 2,
            desc=f"  subset {classes_subset}",
            leave=False,
        ) as pbar:
            for seed in seeds:
                local_elements = local_elements_by_seed[seed]
                nb_learning_samples = local_elements["nb_learning_samples"]
                local_inputs = local_elements["texts"]
                local_labels = torch.tensor(local_elements["labels"])
                local_predictions = torch.tensor(local_elements["predictions"])
                local_indices = list(local_elements["indices"])

                local_explanation = load_local_importances(
                    concept_dir=concept_resources.concept_dir,
                    sample_indices=local_indices,
                    nb_learning_samples=nb_learning_samples,
                )
                concept_importances_corrupted = any(
                    not torch.isfinite(tensor).all().item()
                    for tensor in [
                        concept_resources.global_importances,
                        *local_explanation.local_importances,
                    ]
                )

                for prompt_type, anonym in itertools.product(prompt_types, [True, False]):
                    pbar.update(1)
                    prompt_type_name = prompt_type.name.split("_")[0]
                    is_baseline = "baseline" in prompt_type.name
                    method_name = concept_resources.method_name if not is_baseline else "baseline"
                    nb_concepts = None if is_baseline else concept_resources.nb_concepts
                    interpretation_key = (
                        None if is_baseline else concept_resources.interpretation_key
                    )
                    setting = prompt_type.value._replace(anonymize_classes=anonym)

                    str_key = str(
                        (
                            dataset_abbrev,
                            model_abbrev,
                            str(classes_subset),
                            seed,
                            method_name,
                            nb_concepts,
                            interpretation_key,
                            prompt_type_name if not anonym else "A" + prompt_type_name,
                            SPECIFICATION,
                        )
                    )
                    if str_key in existing_keys:
                        continue

                    if concept_importances_corrupted:
                        if append_prompt_group_if_missing(
                            output_path,
                            {
                                "key": str_key,
                                "corrupted": True,
                                "corruption_reason": "non_finite_concept_importances",
                            },
                            existing_keys,
                        ):
                            subset_new += 1
                            total_new += 1
                        continue

                    system_prompt, user_prompts, expected_answers = simulatability_metric.construct_prompt(
                        setting=setting,
                        interesting_samples=local_inputs,
                        corresponding_predictions=local_predictions,
                        corresponding_labels=local_labels,
                        nb_learning_samples=nb_learning_samples,
                        concepts_interpretation=concept_resources.concepts_interpretation,
                        global_importances=concept_resources.global_importances,
                        local_importances=local_explanation.local_importances,
                        class_ids=classes_subset,
                    )
                    validate_prompt_group(
                        key=str_key,
                        user_prompts=user_prompts,
                        expected_answers=expected_answers,
                        classes_subset=classes_subset,
                    )

                    if append_prompt_group_if_missing(
                        output_path,
                        {
                            "key": str_key,
                            "system_prompt": system_prompt,
                            "user_prompts": user_prompts,
                            "expected_answers": expected_answers,
                        },
                        existing_keys,
                    ):
                        subset_new += 1
                        total_new += 1

        print(f"  -> {subset_new} new prompt groups written.")

    print(f"\nDone. {total_new} total new prompt groups appended to {output_path}.")


if __name__ == "__main__":
    main()
