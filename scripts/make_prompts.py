"""Generate ConSim prompt JSONL for simulatability experiments.

Supports two explanation families (concepts and rationales) and iterates over
all canonical class subsets for the chosen dataset.

Usage examples::

    python scripts/make_prompts.py --dataset GE --explanation-family concepts --method seminmf
    python scripts/make_prompts.py --dataset BIOS --explanation-family rationales --method Qwen/Qwen3.5-9B
    python scripts/make_prompts.py --dataset HE --explanation-family concepts --method ica --interpretation topk

Output is written to ``data/prompts/{dataset_abbrev}_{explanation_family}.jsonl``.
Existing keys in the output file are skipped (append-only, resumable).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

from tqdm import tqdm
import torch

if __package__ in {None, ""}:
    # Allow `python scripts/make_prompts.py` to resolve the repo-local `utils` package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.concepts import (
    load_concept_explanation_resources,
    load_local_importances,
)
from utils.consim import ConSim, PromptTypes
from utils.data import (
    ABBREVIATIONS,
    DATASET_CLASSES_NAMES,
    DATASET_CLASSES_SUBSETS,
    LLM_MODELS,
    MODELS_DATASETS,
    MODEL_SPLIT_POINTS,
    get_save_root,
    iter_jsonl,
    load_dataset_splits,
    load_or_compute_local_elements,
    load_or_compute_predictions,
    resolve_llm_model,
)
from utils.rationales import (
    load_or_generate_rationales,
    group_rationales_by_seed,
)
from utils.ratsim import (
    RationalePromptTypes,
    RationalesSimulatability,
)

# ---------------------------------------------------------------------------
# Reverse lookups: dataset abbreviation → dataset name → model name.
# ---------------------------------------------------------------------------
_ABBREV_TO_DATASET: dict[str, str] = {
    v: k for k, v in ABBREVIATIONS["datasets"].items()
}
_DATASET_TO_MODEL: dict[str, str] = {v: k for k, v in MODELS_DATASETS.items()}

# Concept extraction method names (for --method validation and path construction).
CONCEPT_METHODS = {"seminmf", "ica", "kmeans", "pca", "svd"}

# Interpretation method names → interpreto class names (for cache path lookup).
INTERPRETATION_NAMES = {
    "llm": "LLMLabels",
    "topk": "TopKInputs",
}

# Prompt types for each explanation family.
CONCEPT_PROMPT_TYPES = {
    PromptTypes.B1_baseline_without_lp,
    PromptTypes.C1_global_concepts_without_lp,
    PromptTypes.B2_baseline_with_lp,
    PromptTypes.C2_global_concepts_with_lp,
    PromptTypes.C3_global_and_local_concepts_with_lp,
}

RATIONALE_PROMPT_TYPES = {
    RationalePromptTypes.B1_baseline_without_lp,
    RationalePromptTypes.B2_baseline_with_lp,
    RationalePromptTypes.R1_justify_with_lp,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ConSim prompt JSONL for simulatability experiments.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(_ABBREV_TO_DATASET.keys()),
        required=True,
        help="Dataset abbreviation (e.g. GE, HE, BIOS, E).",
    )
    parser.add_argument(
        "--explanation-family",
        choices=["concepts", "rationales"],
        required=True,
        help="Explanation family to generate prompts for.",
    )
    # Concept-specific arguments
    parser.add_argument(
        "--method",
        required=True,
        help=f"Concept extraction method (seminmf, ica, kmeans, pca, svd) or rationale model short name/path. Short names: {', '.join(LLM_MODELS.keys())}.",
    )
    parser.add_argument(
        "--nb-concepts-ratio",
        type=float,
        default=3,
        help="Number of concepts = nb_classes * ratio (default: 3).",
    )
    parser.add_argument(
        "--interpretation",
        choices=sorted(INTERPRETATION_NAMES.keys()),
        default="topk",
        help="Interpretation method for concept labeling (default: topk).",
    )
    # Rationale-specific arguments
    parser.add_argument(
        "--rationale-batch-size",
        type=int,
        default=32,
        help="Batch size for rationale generation (default: 32).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max new tokens for rationale generation (default: 64).",
    )
    # General arguments
    parser.add_argument(
        "--seeds",
        default="0-49",
        help="Seed range (e.g. '0-49') or comma-separated list (e.g. '0,1,2').",
    )
    parser.add_argument(
        "--nb-samples",
        type=int,
        default=20,
        help="Number of samples per seed (default: 20).",
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
    """Parse seed specification: '0-49' → [0..49] or '0,1,5' → [0, 1, 5]."""
    if "-" in seeds_str and "," not in seeds_str:
        start, end = seeds_str.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(s) for s in seeds_str.split(",")]


def generate_prompts_for_subset(
    *,
    args: argparse.Namespace,
    classes_subset: list[int],
    dataset_name: str,
    model_name: str,
    save_root: Path,
    test_inputs: list[str],
    test_labels: torch.Tensor,
    test_predictions: torch.Tensor,
    classes: list[str],
    seeds: list[int],
    output_path: Path,
    existing_keys: set[str],
) -> int:
    """
    Generate prompts for one class subset. Returns number of new prompts written.
    """
    explanation_family = args.explanation_family

    # Determine prompt types and simulatability metric.
    if explanation_family == "concepts":
        prompt_types = CONCEPT_PROMPT_TYPES
        simulatability_metric = ConSim(classes=classes)
        specification = "new_consim"
    else:
        prompt_types = RATIONALE_PROMPT_TYPES
        simulatability_metric = RationalesSimulatability(classes=classes)
        specification = "rationales"

    # Load or compute local elements (seed → sample selection).
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

    # Prepare explanation-specific resources.
    concept_resources = None
    rationale_by_seed = None

    if explanation_family == "concepts":
        nb_concepts = int(len(classes) * args.nb_concepts_ratio)
        interpretation_name = INTERPRETATION_NAMES[args.interpretation]
        concept_resources = load_concept_explanation_resources(
            save_root=save_root,
            method_name=args.method,
            nb_concepts=nb_concepts,
            activations_difference=args.activations_difference,
            interpretation_name=interpretation_name,
            classes=classes,
        )
    else:
        # Rationale path: generate rationales for all samples used by any seed.
        seed_indices = {
            seed: list(local_elements_by_seed[seed]["indices"]) for seed in seeds
        }
        required_test_indices = sorted(
            {index for indices in seed_indices.values() for index in indices}
        )
        rationales = load_or_generate_rationales(
            model_name=args.method,
            dataset_name=dataset_name,
            inputs=[test_inputs[index] for index in required_test_indices],
            labels=test_labels[required_test_indices],
            predictions=test_predictions[required_test_indices],
            sample_ids=required_test_indices,
            save_root=save_root,
            device=args.device,
            batch_size=args.rationale_batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        rationale_by_seed = group_rationales_by_seed(
            rationales=rationales,
            seed_indices=seed_indices,
        )

    # Generate prompts for all seeds × prompt types × anonymization variants.
    new_prompts = 0
    dataset_abbrev = ABBREVIATIONS["datasets"][dataset_name]
    model_abbrev = ABBREVIATIONS["models"][model_name]

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

            if explanation_family == "concepts":
                local_explanation = load_local_importances(
                    concept_dir=concept_resources.concept_dir,
                    sample_indices=local_indices,
                    nb_learning_samples=nb_learning_samples,
                )
            else:
                local_rationales = rationale_by_seed[seed]

            for prompt_type, anonym in itertools.product(prompt_types, [True, False]):
                prompt_type_name = prompt_type.name.split("_")[0]
                pbar.update(1)
                is_baseline = "baseline" in prompt_type.name
                setting = prompt_type.value._replace(anonymize_classes=anonym)

                # Build experiment key components.
                if explanation_family == "concepts":
                    method_name = (
                        concept_resources.method_name if not is_baseline else "baseline"
                    )
                    nb_concepts = concept_resources.nb_concepts
                    interpretation_name = concept_resources.interpretation_name
                    construct_prompt_kwargs = {
                        "concepts_interpretation": concept_resources.concepts_interpretation,
                        "global_importances": concept_resources.global_importances,
                        "local_importances": local_explanation.local_importances,
                    }
                else:
                    method_name = args.method if not is_baseline else "baseline"
                    nb_concepts = None
                    interpretation_name = None
                    construct_prompt_kwargs = {
                        "rationales": local_rationales,
                    }

                str_key = str(
                    (
                        dataset_abbrev,  # 0
                        model_abbrev,  # 1
                        str(classes_subset),  # 2
                        seed,  # 3
                        method_name,  # 4
                        nb_concepts,  # 5
                        interpretation_name,  # 6
                        prompt_type_name if not anonym else "A" + prompt_type_name,  # 7
                        specification,  # 8
                    )
                )
                if str_key in existing_keys:
                    continue

                system_prompt, user_prompts, expected_answers = (
                    simulatability_metric.construct_prompt(
                        setting=setting,
                        interesting_samples=local_inputs,
                        corresponding_predictions=local_predictions,
                        corresponding_labels=local_labels,
                        nb_learning_samples=nb_learning_samples,
                        **construct_prompt_kwargs,
                    )
                )
                with open(output_path, "a") as handle:
                    json.dump(
                        {
                            "key": str_key,
                            "system_prompt": system_prompt,
                            "user_prompts": user_prompts,
                            "expected_answers": expected_answers,
                        },
                        handle,
                    )
                    handle.write("\n")
                existing_keys.add(str_key)
                new_prompts += 1

    return new_prompts


def main() -> None:
    args = parse_args()
    seeds = parse_seeds(args.seeds)

    # Resolve dataset abbreviation → full names.
    dataset_name = _ABBREV_TO_DATASET[args.dataset]
    model_name = _DATASET_TO_MODEL[dataset_name]
    split_point = MODEL_SPLIT_POINTS[model_name]
    save_root = get_save_root(model_name, split_point)
    save_root.mkdir(parents=True, exist_ok=True)

    # Resolve short LLM model names for rationale generation.
    if args.explanation_family == "rationales":
        args.method = resolve_llm_model(args.method)

    # Determine output path.
    output_path = Path(f"data/prompts/{args.dataset}_{args.explanation_family}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing keys for skip-already-done.
    if output_path.exists():
        existing_keys = {
            prompt_group["key"] for prompt_group in iter_jsonl(output_path)
        }
    else:
        existing_keys = set()

    # Load dataset (only test split needed for prompt generation).
    print(f"Dataset:            {dataset_name} ({args.dataset})")
    print(f"Model:              {model_name}")
    print(f"Explanation family: {args.explanation_family}")
    print(f"Method:             {args.method}")
    print(f"Seeds:              {seeds[0]}-{seeds[-1]} ({len(seeds)} seeds)")
    print(f"Output:             {output_path}")
    print()

    _train_inputs, _validation_inputs, test_inputs, test_labels = load_dataset_splits(
        dataset_name
    )
    classes = DATASET_CLASSES_NAMES[dataset_name]

    # Compute predictions once.
    test_predictions = load_or_compute_predictions(
        model_name=model_name,
        inputs=test_inputs,
        path=save_root / "test_predictions.pt",
        device=args.device,
        batch_size=args.batch_size,
    )

    # Iterate over all canonical class subsets for this dataset.
    all_subsets = DATASET_CLASSES_SUBSETS[dataset_name]
    total_new = 0

    for classes_subset in all_subsets:
        print(f"Processing classes subset: {classes_subset}")
        n = generate_prompts_for_subset(
            args=args,
            classes_subset=classes_subset,
            dataset_name=dataset_name,
            model_name=model_name,
            save_root=save_root,
            test_inputs=test_inputs,
            test_labels=test_labels,
            test_predictions=test_predictions,
            classes=classes,
            seeds=seeds,
            output_path=output_path,
            existing_keys=existing_keys,
        )
        total_new += n
        print(f"  → {n} new prompt groups written.")

    print(f"\nDone. {total_new} total new prompt groups appended to {output_path}.")


if __name__ == "__main__":
    main()
