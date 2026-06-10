"""Build and cache concept-based explanations for a given model/dataset pair.

Constructs concept model, interpretations, and global importances — all the
concept resources needed by downstream scripts and notebooks.  Results are
cached under ``data/<model>/<split_point>/concept_models/``.

Usage examples::

    python scripts/build_concepts.py --dataset GE --method seminmf
    python scripts/build_concepts.py --dataset BIOS --method ica
    python scripts/build_concepts.py --dataset HE --method kmeans --nb-concepts-ratio 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    # Allow ``python scripts/build_concepts.py`` to resolve the repo-local utils.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interpreto.concepts import (
    ICAConcepts,
    KMeansConcepts,
    PCAConcepts,
    SemiNMFConcepts,
    SVDConcepts,
)
from interpreto.concepts.interpretations import LLMLabels, TopKInputs

from utils.concepts import (
    compute_and_cache_all_local_importances,
    prepare_concept_explanation_resources,
)
from utils.data import (
    ABBREVIATIONS,
    DATASET_CLASSES_NAMES,
    LLM_MODELS,
    MODELS_DATASETS,
    MODEL_SPLIT_POINTS,
    get_save_root,
    load_dataset_splits,
    resolve_llm_model,
)

# ---------------------------------------------------------------------------
# Reverse lookups: dataset abbreviation → dataset name → model name.
# ---------------------------------------------------------------------------
_ABBREV_TO_DATASET: dict[str, str] = {
    v: k for k, v in ABBREVIATIONS["datasets"].items()
}
_DATASET_TO_MODEL: dict[str, str] = {v: k for k, v in MODELS_DATASETS.items()}

# Concept extraction methods available as CLI choices.
METHODS = {
    "seminmf": SemiNMFConcepts,
    "ica": ICAConcepts,
    "kmeans": KMeansConcepts,
    "pca": PCAConcepts,
    "svd": SVDConcepts,
}

# Interpretation methods for concept labeling.
INTERPRETATIONS = {
    "llm": LLMLabels,
    "topk": TopKInputs,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and cache concept-based explanations.",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(_ABBREV_TO_DATASET.keys()),
        help="Dataset abbreviation (e.g. GE, HE, BIOS, E).",
    )
    parser.add_argument(
        "--method",
        choices=sorted(METHODS.keys()),
        required=True,
        help="Concept extraction method.",
    )
    parser.add_argument(
        "--nb-concepts-ratio",
        type=float,
        default=3,
        help="Number of concepts = nb_classes * ratio (default: 3).",
    )
    parser.add_argument(
        "--activations-difference",
        action="store_true",
        help="Use pair-wise activation differences for concept fitting.",
    )
    parser.add_argument(
        "--interpretation",
        choices=sorted(INTERPRETATIONS.keys()),
        default="topk",
        help="Interpretation method for concept labeling (default: llm).",
    )
    parser.add_argument(
        "--llm-model",
        default="llama3.2-3b",
        help=f"LLM model for LLMLabels interpretation (default: llama3.2-3b). Short names: {', '.join(LLM_MODELS.keys())}.",
    )
    parser.add_argument(
        "--device",
        default=DEVICE,
        help=f"Device for computation (default: {DEVICE}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for model inference (default: 64).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve abbreviation → full dataset and model names.
    dataset_name = _ABBREV_TO_DATASET[args.dataset]
    model_name = _DATASET_TO_MODEL[dataset_name]
    split_point = MODEL_SPLIT_POINTS[model_name]
    save_root = get_save_root(model_name, split_point)
    save_root.mkdir(parents=True, exist_ok=True)

    method = METHODS[args.method]
    interpretation = INTERPRETATIONS[args.interpretation]
    llm_model = resolve_llm_model(args.llm_model) if args.interpretation == "llm" else None

    print(f"Dataset:        {dataset_name} ({args.dataset})")
    print(f"Model:          {model_name}")
    print(f"Split point:    {split_point}")
    print(f"Method:         {method.__name__}")
    print(f"Interpretation: {args.interpretation}")
    print(f"Concepts ratio: {args.nb_concepts_ratio}")
    print(f"Device:         {args.device}")
    print()

    # Load dataset splits.
    train_inputs, validation_inputs, test_inputs, test_labels = load_dataset_splits(
        dataset_name
    )
    classes = DATASET_CLASSES_NAMES[dataset_name]

    # Build and cache all concept resources (model, interpretations, importances).
    global_explanation = prepare_concept_explanation_resources(
        dataset_name=dataset_name,
        model_name=model_name,
        split_point=split_point,
        save_root=save_root,
        train_inputs=train_inputs,
        validation_inputs=validation_inputs,
        classes=classes,
        method=method,
        nb_concepts_ratio=args.nb_concepts_ratio,
        activations_difference=args.activations_difference,
        interpretation=interpretation,
        llm_model=llm_model,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Pre-compute local importances for ALL test samples so that make_prompts.py
    # can load them without needing the task model or interpreto.
    compute_and_cache_all_local_importances(
        concept_explainer=global_explanation.concept_explainer,
        test_inputs=test_inputs,
        concept_dir=global_explanation.concept_dir,
        batch_size=args.batch_size,
    )

    print(f"\nConcept resources saved under: {global_explanation.concept_dir}")
    print(f"  Method:         {global_explanation.method_name}")
    print(f"  Interpretation: {global_explanation.interpretation_name}")
    print(f"  Nb concepts:    {global_explanation.nb_concepts}")
    print(f"  Importances:    {global_explanation.global_importances.shape}")


if __name__ == "__main__":
    main()
