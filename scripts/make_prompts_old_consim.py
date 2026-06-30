"""Generate prompts using the OLD ConSim implementation (all-at-once evaluation).

This script reuses the same cached local_elements as make_prompts.py, ensuring
that comparisons between old and new ConSim are done on identical samples and seeds.

The key difference: old ConSim puts all evaluation samples into a single user prompt
and expects the LLM to return all predictions at once. New ConSim asks one sample at
a time. This script outputs JSONL with the same schema, but user_prompts has length 1
and expected_answers has the full list of expected predictions.

Usage examples::

    python scripts/make_prompts_old_consim.py GE seminmf
    python scripts/make_prompts_old_consim.py BIOS ica --interpretation topk

Output: ``data/prompts/{dataset_abbrev}_old_consim.jsonl``
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sys
from pathlib import Path

from tqdm import tqdm
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.concepts import (
    CONCEPT_METHOD_NAMES,
    load_concept_explanation_resources,
    load_local_importances,
)
from utils.consim import ConSim  # new ConSim, only used for sample selection
from utils.old_consim import ConSim as OldConSim, PromptTypes as OldPromptTypes
from utils.data import (
    ABBREVIATIONS,
    DATASET_CLASSES_NAMES,
    DATASET_CLASSES_SUBSETS,
    MODELS_DATASETS,
    get_save_root,
    iter_jsonl,
    load_dataset_splits,
    load_or_compute_dataset_activations,
    load_or_compute_local_elements,
)
from utils.registries import INTERPRETATION_KEYS

# ---------------------------------------------------------------------------
_ABBREV_TO_DATASET: dict[str, str] = {
    v: k for k, v in ABBREVIATIONS["datasets"].items()
}
_DATASET_TO_MODEL: dict[str, str] = {v: k for k, v in MODELS_DATASETS.items()}

# Concept extraction method names (for --method validation).
CONCEPT_METHODS = set(CONCEPT_METHOD_NAMES.keys())

# Old ConSim prompt types (subset relevant for comparison).
OLD_PROMPT_TYPES = {
    OldPromptTypes.B1_baseline_without_lp,
    OldPromptTypes.C1_global_concepts_without_lp,
    OldPromptTypes.B2_baseline_with_lp,
    OldPromptTypes.C2_global_concepts_with_lp,
    OldPromptTypes.C3_global_and_local_concepts_with_lp,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate old ConSim (all-at-once) prompts for comparison.",
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
        help="Interpretation method (default: topk).",
    )
    parser.add_argument(
        "--seeds",
        default="0-49",
        help="Seed range (e.g. '0-49') or comma-separated list.",
    )
    parser.add_argument(
        "--nb-samples",
        type=int,
        default=40,
        help="Number of samples per seed (default: 40).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for model inference (default: 64).",
    )
    return parser.parse_args()


def parse_seeds(seeds_str: str) -> list[int]:
    if "-" in seeds_str and "," not in seeds_str:
        start, end = seeds_str.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(s) for s in seeds_str.split(",")]


def validate_expected_answers(
    *,
    key: str,
    expected_answers: list[str],
    classes_subset: list[int],
) -> None:
    if not expected_answers:
        raise ValueError(f"Invalid prompt group {key}: no expected answers.")
    if len(classes_subset) > 1 and len(set(expected_answers)) < 2:
        raise ValueError(
            f"Degenerate prompt group {key}: expected answers contain only "
            f"{sorted(set(expected_answers))}. Check cached model predictions/local elements."
        )


def append_prompt_group_if_missing(
    output_path: Path,
    prompt_group: dict,
    existing_keys: set[str],
) -> bool:
    """Append one prompt group, re-checking under a lock to avoid duplicate keys."""
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


def has_non_finite_importances(tensors: list[torch.Tensor]) -> bool:
    return any(not torch.isfinite(tensor).all().item() for tensor in tensors)


def build_old_consim_global_importances(
    global_importances: torch.Tensor,
    classes: list[str],
    classes_subset: list[int],
) -> dict[str, dict[int, float]]:
    """
    Convert the (nb_classes, nb_concepts) tensor into old ConSim's expected format:
    {class_name: {concept_id: importance_float, ...}, ...}
    Only includes classes in the subset.
    """
    result: dict[str, dict[int, float]] = {}
    for class_id in classes_subset:
        class_name = classes[class_id]
        result[class_name] = {
            cid: global_importances[class_id, cid].item()
            for cid in range(global_importances.shape[1])
        }
    return result


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)

    # Resolve names.
    dataset_name = _ABBREV_TO_DATASET[args.dataset]
    model_name = _DATASET_TO_MODEL[dataset_name]
    save_root = get_save_root(model_name)
    save_root.mkdir(parents=True, exist_ok=True)

    # Output path.
    output_path = Path(f"data/prompts/{args.dataset}_old_consim.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        existing_keys = {pg["key"] for pg in iter_jsonl(output_path)}
    else:
        existing_keys = set()

    print(f"Dataset:         {dataset_name} ({args.dataset})")
    print(f"Model:           {model_name}")
    print(f"Method:          {args.method}")
    print(f"Interpretation:  {None if args.method == 'classes' else args.interpretation}")
    print(f"Seeds:           {seeds[0]}-{seeds[-1]} ({len(seeds)} seeds)")
    print(f"Output:          {output_path}")
    print()

    train_inputs, validation_inputs, test_inputs, test_labels = load_dataset_splits(
        dataset_name
    )
    classes = DATASET_CLASSES_NAMES[dataset_name]

    # We use the NEW ConSim's select_examples to ensure identical samples.
    simulatability_metric = ConSim(classes=classes)

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

    # Load pre-built concept resources (load-only, no interpreto needed).
    if args.method == "neurons":
        nb_concepts = None
    elif args.method == "classes":
        nb_concepts = len(classes)
    else:
        nb_concepts = int(len(classes) * args.nb_concepts_ratio)
    interpretation_key = None if args.method == "classes" else args.interpretation
    method_dir_name = CONCEPT_METHOD_NAMES[args.method]
    concept_resources = load_concept_explanation_resources(
        save_root=save_root,
        method_name=method_dir_name,
        nb_concepts=nb_concepts,
        interpretation_key=interpretation_key,
        classes=classes,
    )

    # Iterate over all class subsets.
    all_subsets = DATASET_CLASSES_SUBSETS[dataset_name]
    total_new = 0
    dataset_abbrev = ABBREVIATIONS["datasets"][dataset_name]
    model_abbrev = ABBREVIATIONS["models"][model_name]

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

        global_importances_corrupted = has_non_finite_importances(
            [concept_resources.global_importances]
        )
        old_global_importances = None
        if not global_importances_corrupted:
            # Convert global importances to old format.
            old_global_importances = build_old_consim_global_importances(
                concept_resources.global_importances,
                classes,
                classes_subset,
            )

        with tqdm(
            total=len(seeds) * len(OLD_PROMPT_TYPES) * 2,
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
                subset_prediction_ids = {
                    class_id: subset_id
                    for subset_id, class_id in enumerate(classes_subset)
                }
                local_subset_predictions = torch.tensor(
                    [
                        subset_prediction_ids[int(prediction.item())]
                        for prediction in local_predictions
                    ]
                )

                # Load pre-computed local importances and extract per-predicted-class.
                local_explanation = load_local_importances(
                    concept_dir=concept_resources.concept_dir,
                    sample_indices=local_indices,
                    nb_learning_samples=nb_learning_samples,
                )
                concept_importances_corrupted = (
                    global_importances_corrupted
                    or has_non_finite_importances(local_explanation.local_importances)
                )
                old_local_importances = None
                if not concept_importances_corrupted:
                    # Old ConSim expects shape (nb_lp_samples, nb_concepts):
                    # importance of each concept for the PREDICTED class of each sample.
                    old_local_importances = torch.stack(
                        [
                            local_explanation.local_importances[i][
                                int(local_predictions[i].item())
                            ]
                            for i in range(nb_learning_samples)
                        ]
                    )

                for prompt_type in OLD_PROMPT_TYPES:
                    for anonym in [True, False]:
                        pbar.update(1)
                        prompt_type_name = prompt_type.name.split("_")[0]
                        is_baseline = "baseline" in prompt_type.name
                        method_name = (
                            concept_resources.method_name
                            if not is_baseline
                            else "baseline"
                        )

                        str_key = str(
                            (
                                dataset_abbrev,
                                model_abbrev,
                                str(classes_subset),
                                seed,
                                method_name,
                                concept_resources.nb_concepts,
                                concept_resources.interpretation_key,
                                prompt_type_name
                                if not anonym
                                else "A" + prompt_type_name,
                                "old_consim",
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
                                total_new += 1
                            continue

                        # Use old ConSim's _generate_prompt (static method).
                        prompt, literal_model_predictions = OldConSim._generate_prompt(
                            sentences=local_inputs,
                            predictions=local_subset_predictions,
                            classes=[classes[c] for c in classes_subset],
                            concepts_interpretation=concept_resources.concepts_interpretation,
                            global_importances=old_global_importances
                            if not is_baseline
                            else None,
                            local_importances=old_local_importances
                            if not is_baseline
                            else None,
                            prompt_type=prompt_type,
                            anonymize_classes=anonym,
                        )

                        system_prompt, user_prompt = prompt
                        validate_expected_answers(
                            key=str_key,
                            expected_answers=literal_model_predictions,
                            classes_subset=classes_subset,
                        )

                        if append_prompt_group_if_missing(
                            output_path,
                            {
                                "key": str_key,
                                "system_prompt": system_prompt,
                                "user_prompts": [user_prompt],
                                "expected_answers": literal_model_predictions,
                            },
                            existing_keys,
                        ):
                            total_new += 1

        print(f"  → {total_new} new prompt groups written.")

    print(f"\nDone. {total_new} total new prompt groups appended to {output_path}.")


if __name__ == "__main__":
    main()
