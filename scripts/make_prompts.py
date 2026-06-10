from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

from tqdm import tqdm
import torch

if __package__ in {None, ""}:
    # Allow `python scripts/make_prompts.py` to resolve the repo-local `utils` package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interpreto.concepts import SemiNMFConcepts
from interpreto.concepts.interpretations import LLMLabels
from interpreto.concepts.metrics.simulatability.consim import PromptTypes, ConSim

from utils.concepts import (
    compute_local_concept_explanation,
    prepare_concept_explanation_resources,
)
from utils.data import (
    ABBREVIATIONS,
    MODELS_DATASETS,
    DATASET_CLASSES_NAMES,
    get_save_root,
    MODEL_SPLIT_POINTS,
    iter_jsonl,
    load_dataset_splits,
    load_or_compute_local_elements,
    load_or_compute_predictions,
)
from utils.rationales import (
    load_or_generate_rationales,
    group_rationales_by_seed,
)
from utils.rationales_simulatability import (
    RationalePromptTypes,
    RationalesSimulatability,
)

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# -----------
# MODEL_NAME = "Hate-speech-CNERG/bert-base-uncased-hatexplain"
# CLASSES_SUBSET = [0, 1, 2]

# -----------
# MODEL_NAME = "SamLowe/roberta-base-go_emotions"
# CLASSES_SUBSET = [2, 3, 27]  # anger, annoyance, neutral
# CLASSES_SUBSET = [2, 3, 9, 10]  # anger, annoyance, disappointment, disapproval
# CLASSES_SUBSET = [6, 7]  # confusion, curiosity
# CLASSES_SUBSET = [0, 4, 5]  # admiration, approval, caring

# ----
MODEL_NAME = "/datasets/shared_datasets/BIOS/models/RoBERTa_occBIOS_10epochs_g1/"
# CLASSES_SUBSET = [0, 11, 25]  # surgeon, physician, dentist
# CLASSES_SUBSET = [3, 6, 26]  # professor, teacher, psychologist
# CLASSES_SUBSET = [3, 5, 13]  # professor, software_developer, architect
CLASSES_SUBSET = [2, 12, 21]  # photographer, journalist, filmmaker


# Shared prompt-generation flow configuration.
EXPLANATION_FAMILY = "rationales"  # "concepts"  #

# Concept-based explanation configuration.
PARAMETERS = {
    "concepts": {
        "method": SemiNMFConcepts,
        "nb_concepts_ratio": 3,
        "activations_difference": False,
        "interpretation": LLMLabels,
        "llm_model": "gpt-4.1-nano",
    },
    "rationales": {
        "model_name": "Qwen/Qwen3.5-9B",
        "batch_size": 32,
        "max_new_tokens": 64,
    },
}
PROMPT_TYPES = {
    "concepts": {
        PromptTypes.L1_baseline_without_lp,
        PromptTypes.E1_global_concepts_without_lp,
        PromptTypes.L2_baseline_with_lp,
        PromptTypes.E2_global_concepts_with_lp,
        PromptTypes.E3_global_and_local_concepts_with_lp,
        PromptTypes.C1_contrastive_global_concepts_without_lp,
        PromptTypes.C2_contrastive_global_concepts_with_lp,
        PromptTypes.C3_contrastive_global_and_local_concepts_with_lp,
        PromptTypes.C4_contrastive_local_concepts,
        PromptTypes.C5_contrastive_local_only,
    },
    "rationales": {
        RationalePromptTypes.L1_baseline_without_lp,
        RationalePromptTypes.L2_baseline_with_lp,
        RationalePromptTypes.R_justify_with_lp,
        RationalePromptTypes.RC_contrastive_with_lp,
    },
}
SIMULATABILITY_METRICS = {
    "concepts": ConSim,
    "rationales": RationalesSimulatability,
}
prompt_types = PROMPT_TYPES[EXPLANATION_FAMILY]

SEEDS = list(range(50))
NB_SAMPLES = 20
BATCH_SIZE = 64
SPECIFICATION = (
    "verbalized importance" if EXPLANATION_FAMILY == "concepts" else "rationales"
)

OUTPUT_PATH = Path("data/consim_prompts.jsonl")


if __name__ == "__main__":
    if OUTPUT_PATH.exists():
        existing_keys = {
            prompt_group["key"] for prompt_group in iter_jsonl(OUTPUT_PATH)
        }
    else:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing_keys = set()

    # Dataset resources are loaded once and reused across prompt types.
    dataset_name = MODELS_DATASETS[MODEL_NAME]
    split_point = MODEL_SPLIT_POINTS[MODEL_NAME]
    save_root = get_save_root(MODEL_NAME, split_point)
    save_root.mkdir(parents=True, exist_ok=True)

    # load data
    train_inputs, validation_inputs, test_inputs, test_labels = load_dataset_splits(
        dataset_name
    )
    classes = DATASET_CLASSES_NAMES[dataset_name]

    simulatability_metric = SIMULATABILITY_METRICS[EXPLANATION_FAMILY](classes=classes)

    test_predictions = load_or_compute_predictions(
        model_name=MODEL_NAME,
        inputs=test_inputs,
        path=save_root / "test_predictions.pt",
        device=DEVICE,
        batch_size=BATCH_SIZE,
    )

    local_elements_by_seed = load_or_compute_local_elements(
        save_root=save_root,
        simulatability_metric=simulatability_metric,
        inputs=test_inputs,
        labels=test_labels,
        predictions=test_predictions,
        seeds=SEEDS,
        classes_subset=CLASSES_SUBSET,
        nb_samples=NB_SAMPLES,
    )

    if EXPLANATION_FAMILY == "concepts":
        concept_resources = prepare_concept_explanation_resources(
            dataset_name=dataset_name,
            model_name=MODEL_NAME,
            split_point=split_point,
            save_root=save_root,
            train_inputs=train_inputs,
            validation_inputs=validation_inputs,
            classes=classes,
            device=DEVICE,
            batch_size=BATCH_SIZE,
            **PARAMETERS["concepts"],
        )
    else:
        # Only generate rationales for test samples selected by at least one seed.
        seed_indices = {
            seed: list(local_elements_by_seed[seed]["indices"])  # type: ignore[arg-type]
            for seed in SEEDS
        }
        required_test_indices = sorted(
            {index for indices in seed_indices.values() for index in indices}
        )
        rationales = load_or_generate_rationales(
            dataset_name=dataset_name,
            inputs=[test_inputs[index] for index in required_test_indices],
            labels=test_labels[required_test_indices],
            predictions=test_predictions[required_test_indices],
            sample_ids=required_test_indices,
            save_root=save_root,
            device=DEVICE,
            **PARAMETERS["rationales"],
        )
        rationale_by_seed, contrastive_by_seed = group_rationales_by_seed(
            rationales=rationales,
            seed_indices=seed_indices,
        )

    with tqdm(
        total=len(SEEDS) * len(prompt_types) * 2,  # anonymized vs non-anonymized
    ) as pbar:
        for seed in SEEDS:
            local_elements = local_elements_by_seed[seed]
            nb_learning_samples = local_elements["nb_learning_samples"]
            local_inputs = local_elements["texts"]
            local_labels = torch.tensor(local_elements["labels"])
            local_predictions = torch.tensor(local_elements["predictions"])

            if EXPLANATION_FAMILY == "concepts":
                local_explanation = compute_local_concept_explanation(
                    global_explanation=concept_resources,  # type: ignore[arg-type]
                    local_inputs=local_inputs,  # type: ignore[arg-type]
                    nb_learning_samples=nb_learning_samples,  # type: ignore[arg-type]
                )
            else:
                local_rationales = rationale_by_seed[seed]  # type: ignore[arg-type]
                local_contrastives = contrastive_by_seed[seed]  # type: ignore[arg-type]

            for prompt_type, anonym in itertools.product(prompt_types, [True, False]):
                prompt_type_name = prompt_type.name.split("_")[0]  # type: ignore[union-attr]
                pbar.update(1)
                is_baseline = "baseline" in prompt_type.name  # type: ignore[union-attr]
                setting = prompt_type.value._replace(anonymize_classes=anonym)  # type: ignore[arg-type]

                # set key components
                # concept-based
                if EXPLANATION_FAMILY == "concepts":
                    method_name = (
                        concept_resources.method_name if not is_baseline else "baseline"  # type: ignore[arg-type]
                    )
                    nb_concepts = concept_resources.nb_concepts  # type: ignore[arg-type]
                    interpretation_name = concept_resources.interpretation_name  # type: ignore[arg-type]
                    construct_prompt_kwargs = {
                        "concepts_interpretation": concept_resources.concepts_interpretation,  # type: ignore[arg-type]
                        "global_importances": concept_resources.global_importances,  # type: ignore[arg-type]
                        "local_importances": local_explanation.local_importances,  # type: ignore[arg-type]
                        "contrastive_pairs": list(
                            itertools.permutations(CLASSES_SUBSET, 2)
                        ),
                    }
                # rationale
                else:
                    method_name = (
                        PARAMETERS["rationales"]["model_name"]
                        if not is_baseline
                        else "baseline"
                    )
                    nb_concepts = None
                    interpretation_name = None
                    construct_prompt_kwargs = {
                        "rationales": local_rationales,  # type: ignore[arg-type]
                        "contrastives": local_contrastives,  # type: ignore[arg-type]
                    }

                str_key = str(
                    (
                        ABBREVIATIONS["datasets"][dataset_name],  # 0
                        ABBREVIATIONS["models"][MODEL_NAME],  # 1
                        str(CLASSES_SUBSET),  # 2
                        seed,  # 3
                        method_name,  # 4
                        nb_concepts,  # 5
                        interpretation_name,  # 6
                        prompt_type_name if not anonym else "A" + prompt_type_name,  # 7
                        SPECIFICATION,  # 8
                    )
                )
                if str_key in existing_keys:
                    continue

                system_prompt, user_prompts, expected_answers = (
                    simulatability_metric.construct_prompt(
                        setting=setting,
                        interesting_samples=local_inputs,  # type: ignore[arg-type]
                        corresponding_predictions=local_predictions,
                        corresponding_labels=local_labels,
                        nb_learning_samples=nb_learning_samples,  # type: ignore[arg-type]
                        **construct_prompt_kwargs,  # type: ignore[arg-type]
                    )
                )
                with open(OUTPUT_PATH, "a") as handle:
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
