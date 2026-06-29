"""Generate ConSim prompt JSONL for simulatability experiments.

Supports three explanation families (concepts, rationales, and attributions) and
iterates over all canonical class subsets for the chosen dataset.

Concept artifacts (model, interpretations, importances) are built automatically
if missing from cache. The script exits early if all expected prompt entries
already exist in the output file.

Usage examples::

    python scripts/make_prompts.py concepts GE seminmf
    python scripts/make_prompts.py rationales BIOS
    python scripts/make_prompts.py rationales BIOS --llm-model qwen3.5-9b
    python scripts/make_prompts.py concepts HE ica --interpretation topk
    python scripts/make_prompts.py attributions GE saliency

Output is written to ``data/prompts/{dataset_abbrev}_{explanation_family}.jsonl``.
Existing keys in the output file are skipped (append-only, resumable).
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
    # Allow `python scripts/make_prompts.py` to resolve the repo-local `utils` package.
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
from utils.registries import (
    ATTRIBUTION_METHOD_NAMES,
    ATTRIBUTION_PROMPT_ABBREVS,
    CONCEPT_METHOD_NAMES,
    CONCEPT_PROMPT_ABBREVS,
    INTERPRETATION_KEYS,
    RATIONALE_PROMPT_ABBREVS,
)

# ---------------------------------------------------------------------------
# Reverse lookups: dataset abbreviation → dataset name → model name.
# ---------------------------------------------------------------------------
_ABBREV_TO_DATASET: dict[str, str] = {
    v: k for k, v in ABBREVIATIONS["datasets"].items()
}
_DATASET_TO_MODEL: dict[str, str] = {v: k for k, v in MODELS_DATASETS.items()}

# Concept extraction method names (for --method validation).
CONCEPT_METHODS = set(CONCEPT_METHOD_NAMES.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ConSim prompt JSONL for simulatability experiments.",
    )
    parser.add_argument(
        "explanation_family",
        choices=["concepts", "rationales", "attributions"],
        help="Explanation family to generate prompts for.",
    )
    parser.add_argument(
        "dataset",
        choices=sorted(_ABBREV_TO_DATASET.keys()),
        help="Dataset abbreviation (e.g. GE, HE, BIOS, E).",
    )
    parser.add_argument(
        "method",
        nargs="?",
        default=None,
        help=(
            "Explanation method (not used for rationales). "
            f"For concepts: {', '.join(sorted(CONCEPT_METHODS))}. "
            f"For attributions: {', '.join(ATTRIBUTION_METHOD_NAMES)}."
        ),
    )
    # Concept-specific arguments
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
            f"LLM model for rationale generation and concept LLM interpretation "
            f"(default: llama3.2-3b). Short names: {', '.join(LLM_MODELS.keys())}."
        ),
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
        default=32,
        help="Batch size for model inference (default: 96).",
    )
    return parser.parse_args()


def parse_seeds(seeds_str: str) -> list[int]:
    """Parse seed specification: '0-49' → [0..49] or '0,1,5' → [0, 1, 5]."""
    if "-" in seeds_str and "," not in seeds_str:
        start, end = seeds_str.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(s) for s in seeds_str.split(",")]


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
    import torch

    from utils.attrsim import AttrSim, PromptTypes as AttrPromptTypes
    from utils.attributions import (
        group_attributions_by_seed,
        load_or_compute_attributions,
    )
    from utils.concepts import load_local_importances, load_or_build_concept_resources
    from utils.consim import ConSim, PromptTypes
    from utils.rationales import group_rationales_by_seed, load_or_generate_rationales
    from utils.ratsim import RationalePromptTypes, RationalesSimulatability

    concept_prompt_types = {
        PromptTypes.B1_baseline_without_lp,
        PromptTypes.C1_global_concepts_without_lp,
        PromptTypes.B2_baseline_with_lp,
        PromptTypes.C2_global_concepts_with_lp,
        PromptTypes.C3_global_and_local_concepts_with_lp,
    }
    rationale_prompt_types = {
        RationalePromptTypes.B1_baseline_without_lp,
        RationalePromptTypes.B2_baseline_with_lp,
        RationalePromptTypes.R1_justify_with_lp,
    }
    attribution_prompt_types = {
        AttrPromptTypes.B1_baseline_without_lp,
        AttrPromptTypes.B2_baseline_with_lp,
        AttrPromptTypes.A1_attribution_with_lp,
    }

    explanation_family = args.explanation_family

    # Determine prompt types and simulatability metric.
    if explanation_family == "concepts":
        prompt_types = concept_prompt_types
        simulatability_metric = ConSim(classes=classes)
        specification = "new_consim"
    elif explanation_family == "rationales":
        prompt_types = rationale_prompt_types
        simulatability_metric = RationalesSimulatability(classes=classes)
        specification = "rationales"
    else:
        prompt_types = attribution_prompt_types
        simulatability_metric = AttrSim(classes=classes)
        specification = "attributions"

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
    attribution_by_seed = None

    if explanation_family == "concepts":
        concept_resources = load_or_build_concept_resources(
            args,
            dataset_name=dataset_name,
            model_name=model_name,
            save_root=save_root,
            train_inputs=args.train_inputs,
            validation_inputs=args.validation_inputs,
            test_inputs=test_inputs,
            classes=classes,
            test_predictions=test_predictions,
        )
    elif explanation_family == "rationales":
        # Rationale path: generate rationales for all samples used by any seed.
        seed_indices = {
            seed: list(local_elements_by_seed[seed]["indices"]) for seed in seeds
        }
        required_test_indices = sorted(
            {index for indices in seed_indices.values() for index in indices}
        )
        rationales = load_or_generate_rationales(
            model_name=args.llm_model,
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
    else:
        # Attribution path: compute attributions for all samples used by any seed.
        seed_indices = {
            seed: list(local_elements_by_seed[seed]["indices"]) for seed in seeds
        }
        required_test_indices = sorted(
            {index for indices in seed_indices.values() for index in indices}
        )
        attributions = load_or_compute_attributions(
            model_name=model_name,
            inputs=[test_inputs[index] for index in required_test_indices],
            predictions=test_predictions[required_test_indices],
            sample_ids=required_test_indices,
            method=args.method,
            save_root=save_root,
            device=args.device,
            batch_size=args.batch_size,
            dataset_name=dataset_name,
        )
        attribution_by_seed = group_attributions_by_seed(
            attributions=attributions,
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
            concept_importances_corrupted = False

            if explanation_family == "concepts":
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
            elif explanation_family == "rationales":
                local_rationales = rationale_by_seed[seed]
            else:
                local_attributions = attribution_by_seed[seed]
                corrupted_attribution_indices = [
                    sample_id
                    for sample_id, attribution in zip(
                        local_indices,
                        local_attributions,
                        strict=True,
                    )
                    if attribution is None
                ]
                corrupted_lp_attribution_indices = [
                    sample_id
                    for sample_id, attribution in zip(
                        local_indices[:nb_learning_samples],
                        local_attributions[:nb_learning_samples],
                        strict=True,
                    )
                    if attribution is None
                ]

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
                    interpretation_key = concept_resources.interpretation_key
                    construct_prompt_kwargs = {
                        "concepts_interpretation": concept_resources.concepts_interpretation,
                        "global_importances": concept_resources.global_importances,
                        "local_importances": local_explanation.local_importances,
                        "class_ids": classes_subset,
                    }
                elif explanation_family == "rationales":
                    method_name = args.llm_model if not is_baseline else "baseline"
                    nb_concepts = None
                    interpretation_key = None
                    construct_prompt_kwargs = {
                        "rationales": local_rationales,
                        "class_ids": classes_subset,
                    }
                else:
                    method_name = args.method if not is_baseline else "baseline"
                    nb_concepts = None
                    interpretation_key = None
                    construct_prompt_kwargs = {
                        "corresponding_attribution": local_attributions,
                        "class_ids": classes_subset,
                    }

                str_key = str(
                    (
                        dataset_abbrev,  # 0
                        model_abbrev,  # 1
                        str(classes_subset),  # 2
                        seed,  # 3
                        method_name,  # 4
                        nb_concepts,  # 5
                        interpretation_key,  # 6
                        prompt_type_name if not anonym else "A" + prompt_type_name,  # 7
                        specification,  # 8
                    )
                )
                if str_key in existing_keys:
                    continue

                if explanation_family == "concepts" and concept_importances_corrupted:
                    if append_prompt_group_if_missing(
                        output_path,
                        {
                            "key": str_key,
                            "corrupted": True,
                            "corruption_reason": "non_finite_concept_importances",
                        },
                        existing_keys,
                    ):
                        new_prompts += 1
                    continue

                if (
                    explanation_family == "attributions"
                    and setting.lp_attributions
                    and corrupted_lp_attribution_indices
                ):
                    if append_prompt_group_if_missing(
                        output_path,
                        {
                            "key": str_key,
                            "corrupted": True,
                            "corruption_reason": "attribution_computation_failed",
                            "corrupted_sample_ids": corrupted_attribution_indices,
                            "corrupted_lp_sample_ids": corrupted_lp_attribution_indices,
                        },
                        existing_keys,
                    ):
                        new_prompts += 1
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
                    new_prompts += 1

    return new_prompts


def compute_expected_keys(
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
) -> set[str]:
    """Enumerate all keys this invocation would produce (for early-exit check)."""
    expected = set()
    for classes_subset in classes_subsets:
        for seed in seeds:
            for prompt_type_abbrev in prompt_type_abbrevs:
                for anonym in [True, False]:
                    is_baseline = prompt_type_abbrev.startswith("B")
                    pt_name = prompt_type_abbrev
                    if anonym:
                        pt_name = "A" + pt_name
                    key = str(
                        (
                            dataset_abbrev,
                            model_abbrev,
                            str(classes_subset),
                            seed,
                            "baseline" if is_baseline else method_name,
                            nb_concepts,
                            interpretation_key,
                            pt_name,
                            specification,
                        )
                    )
                    expected.add(key)
    return expected


def main() -> None:
    args = parse_args()

    # Validate method argument (required for concepts and attributions only).
    if args.method is None and args.explanation_family not in ("rationales",):
        print("Error: 'method' is required for concepts and attributions.")
        sys.exit(1)
    if (
        args.explanation_family == "concepts"
        and args.method not in CONCEPT_METHOD_NAMES
    ):
        print(
            f"Error: unknown concept method '{args.method}'. "
            f"Expected one of: {', '.join(sorted(CONCEPT_METHOD_NAMES))}."
        )
        sys.exit(1)
    if (
        args.explanation_family == "attributions"
        and args.method not in ATTRIBUTION_METHOD_NAMES
    ):
        print(
            f"Error: unknown attribution method '{args.method}'. "
            f"Expected one of: {', '.join(ATTRIBUTION_METHOD_NAMES)}."
        )
        sys.exit(1)

    seeds = parse_seeds(args.seeds)

    # Resolve dataset abbreviation → full names.
    dataset_name = _ABBREV_TO_DATASET[args.dataset]
    model_name = _DATASET_TO_MODEL[dataset_name]
    save_root = get_save_root(model_name)
    save_root.mkdir(parents=True, exist_ok=True)

    # Resolve LLM model name (used for rationales and concept LLM interpretation).
    args.llm_model = resolve_llm_model(args.llm_model)

    # --- Early-exit check: compute all expected keys before loading data. ---
    dataset_abbrev = args.dataset
    model_abbrev = ABBREVIATIONS["models"][model_name]
    classes = DATASET_CLASSES_NAMES[dataset_name]
    all_subsets = DATASET_CLASSES_SUBSETS[dataset_name]

    if args.explanation_family == "concepts":
        prompt_type_abbrevs = CONCEPT_PROMPT_ABBREVS
        specification = "new_consim"
        nb_concepts = (
            None
            if args.method == "neurons"
            else len(classes)
            if args.method == "classes"
            else int(len(classes) * args.nb_concepts_ratio)
        )
        interpretation_key = None if args.method == "classes" else args.interpretation
        method_for_key = CONCEPT_METHOD_NAMES[args.method]
    elif args.explanation_family == "rationales":
        prompt_type_abbrevs = RATIONALE_PROMPT_ABBREVS
        specification = "rationales"
        nb_concepts = None
        interpretation_key = None
        method_for_key = args.llm_model
    else:
        prompt_type_abbrevs = ATTRIBUTION_PROMPT_ABBREVS
        specification = "attributions"
        nb_concepts = None
        interpretation_key = None
        method_for_key = args.method

    output_path = Path(f"data/prompts/{args.dataset}_{args.explanation_family}.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing keys for skip-already-done.
    if output_path.exists():
        existing_keys = {
            prompt_group["key"] for prompt_group in iter_jsonl(output_path)
        }
    else:
        existing_keys = set()

    expected_keys = compute_expected_keys(
        dataset_abbrev=dataset_abbrev,
        model_abbrev=model_abbrev,
        method_name=method_for_key,
        classes_subsets=all_subsets,
        seeds=seeds,
        nb_concepts=nb_concepts,
        interpretation_key=interpretation_key,
        prompt_type_abbrevs=prompt_type_abbrevs,
        specification=specification,
    )

    if expected_keys <= existing_keys:
        print(
            f"All {len(expected_keys)} entries already exist in {output_path}. "
            f"Skipping."
        )
        sys.exit(0)

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    # --- Load data (only after early-exit check passes). ---
    print(f"Dataset:            {dataset_name} ({args.dataset})")
    print(f"Model:              {model_name}")
    print(f"Explanation family: {args.explanation_family}")
    if args.explanation_family == "rationales":
        print(f"LLM model:          {args.llm_model}")
    else:
        print(f"Method:             {args.method}")
    print(f"Seeds:              {seeds[0]}-{seeds[-1]} ({len(seeds)} seeds)")
    print(f"Output:             {output_path}")
    print()

    train_inputs, validation_inputs, test_inputs, test_labels = load_dataset_splits(
        dataset_name
    )

    # get_activations now produces both activations and predictions. Build all
    # split caches up front so downstream scripts can reuse the same artifacts.
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

    # Store train/validation on args for the concept-building fallback path.
    args.train_inputs = train_inputs
    args.validation_inputs = validation_inputs

    # Iterate over all canonical class subsets for this dataset.
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
