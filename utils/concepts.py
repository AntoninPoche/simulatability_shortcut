from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from typing import NamedTuple

import torch

from interpreto.concepts import SemiNMFConcepts
from interpreto.concepts.interpretations import LLMLabels, TopKInputs
from interpreto.model_wrapping.llm_interface import OpenAILLM

SYSTEM_PROMPT = """You are a meticulous AI researcher conducting an important investigation into patterns found in language.
Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given a list of text examples.
How important each text is for the behavior is listed after each example in parentheses, with importance from 0 to 10.

- Try to produce a concise final description. Simply describe the text features that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples, but try to summarize the patterns found in the examples.
- Do not make lists of possible explanations. Find a single concept that best describes the examples.
- Strike the balance between being concise and informative. From 1 to 3 words. 3 is an absolute maximum.
- Refrain from including uninformative elements like "patterns found include ...", "the examples show ...", or "text contains ...".
- Here are some examples: laugh, cry, dear person death, high expectations, relationships, loneliness, heartbreak, happy moments, stress, birth, meetings...
- Do not hesitate to qualify the concepts, 'emotion intensity' can be 'strong positive emotion'..., or 'self-reflection' could be 'depressive thoughts'.
- These are examples from the emotion dataset, hence concepts correspond to differentiators between the classes. You are forbidden from using classes names in the concept label, but take them into account when labelling. The classes are:
"""

EMOTION_6_SYSTEM_PROMPT = """You are a meticulous AI researcher conducting an important investigation into patterns found in language.
Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given a list of text examples.
How important each text is for the behavior is listed after each example in parentheses, with importance from 0 to 10.

- Try to produce a concise final description. Simply describe the text features that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples, but try to summarize the patterns found in the examples.
- Do not make lists of possible explanations. Find a single concept that best describes the examples.
- Strike the balance between being concise and informative. From 1 to 5 words. 5 is an absolute maximum.
- Refrain from including uninformative elements like "patterns found include ...", "the examples show ...", or "text contains ...".
- Here are some examples: laugh, cry, dear person death, high expectations, conflictual relationships, loneliness, heartbreak, happy moments, stress, birth, meetings...
- Do not hesitate to qualify the concepts, 'emotion intensity' can be 'strong positive emotion'..., or 'self-reflection' could be depressive thoughts'.
- These are examples from the emotion dataset, hence concepts correspond to differentiators between the classes. You are forbidden from using classes names in the concept label, but take them into account when labelling. Classes: 
"""

EMOTION_28_SYSTEM_PROMPT = """You are a meticulous AI researcher conducting an important investigation into patterns found in language.
Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given a list of text examples.
How important each text is for the behavior is listed after each example in parentheses, with importance from 0 to 10.

- Try to produce a concise final description. Simply describe the text features that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples, but try to summarize the patterns found in the examples.
- Do not make lists of possible explanations. Find a single concept that best describes the examples.
- Strike the balance between being concise and informative. From 1 to 5 words. 5 is an absolute maximum.
- Refrain from including uninformative elements like "patterns found include ...", "the examples show ...", or "text contains ...".
- Here are some examples: laugh, cry, dear person death, high expectations, conflictual relationships, loneliness, heartbreak, happy moments, stress, birth, meetings...
- Do not hesitate to qualify the concepts, 'emotion intensity' can be 'strong positive emotion'..., or 'self-reflection' could be depressive thoughts'.
- These are examples from the emotion dataset, hence concepts correspond to differentiators between the classes. Do not reference them in your label, but take them into account when labelling. Classes: 
"""

HATEXPLAIN_SYSTEM_PROMPT = """You are a meticulous AI researcher conducting an important investigation into patterns found in language.
Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given a list of text examples.
How important each text is for the behavior is listed after each example in parentheses, with importance from 0 to 10.

- Try to produce a concise final description. Simply describe the text features that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples, but try to summarize the patterns found in the examples.
- Do not make lists of possible explanations. Find a single concept that best describes the examples.
- Strike the balance between being concise and informative. From 1 to 5 words. 5 is an absolute maximum.
- Refrain from including uninformative elements like "patterns found include ...", "the examples show ...", or "text contains ...".
- Here are some examples: laugh, cry, dear person death, high expectations, conflictual relationships, loneliness, heartbreak, happy moments, stress, birth, meetings...
- Do not hesitate to qualify the concepts, 'emotion intensity' can be 'strong positive emotion'..., or 'self-reflection' could be depressive thoughts'.
- These are examples from the emotion dataset, hence concepts correspond to differentiators between the classes. Do not reference them in your label, but take them into account when labelling. Classes: 
"""

BIOS_SYSTEM_PROMPT = """You are a meticulous AI researcher conducting an important investigation into patterns found in language.
Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given a list of text examples.
How important each text is for the behavior is listed after each example in parentheses, with importance from 0 to 10.

- Try to produce a concise final description. Simply describe the text features that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples, but try to summarize the patterns found in the examples.
- Do not make lists of possible explanations. Find a single concept that best describes the examples.
- Strike the balance between being concise and informative. From 2 to 4 words. 4 is an absolute maximum.
- Refrain from including uninformative elements like "patterns found include ...", "the examples show ...", or "text contains ...".
- Here are some examples: men names, female medical staff, women activities, high degree, medical procedures, law entities, art works, care, university affiliations, religion, music, teaching...
- Do not hesitate to specify the concepts, 'degree' can be 'high medical degree'..., or 'work place' could be 'famous hospital'..., the concept names should be informative.
- These are examples from the emotion dataset, hence concepts correspond to differentiators between the classes. Do not reference them in your label, but take them into account when labelling. Classes: 
"""


def name_for(obj) -> str:
    return obj.__name__ if hasattr(obj, "__name__") else str(obj)


class GlobalConceptExplanation(NamedTuple):
    concept_explainer: Any
    concepts_interpretation: dict[int, str]
    global_importances: torch.Tensor
    method_name: str
    interpretation_name: str
    nb_concepts: int
    concept_dir: Path


class LocalConceptExplanation(NamedTuple):
    local_importances: list[torch.Tensor]


def _metadata_path_for(path: Path) -> Path:
    return path.with_name(f"{path.stem}_metadata.json")


def _load_cached_json(
    path: Path,
    *,
    expected_metadata: dict[str, Any] | None = None,
) -> dict[Any, Any] | None:
    if not path.exists():
        return None

    if expected_metadata is not None:
        metadata_path = _metadata_path_for(path)
        if not metadata_path.exists():
            return None
        with open(metadata_path) as handle:
            metadata = json.load(handle)
        if metadata != expected_metadata:
            return None

    with open(path) as handle:
        return json.load(handle)


def _save_json_with_metadata(
    path: Path,
    payload: dict[Any, Any],
    *,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle)
    with open(_metadata_path_for(path), "w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def _default_topk_unique_words_kwargs(
    validation_inputs: list[str],
) -> dict[str, Any]:
    return {
        "count_min_threshold": round(len(validation_inputs) * 0.002),
        "lemmatize": True,
        "words_to_ignore": [],
    }


def get_interpretation_config(
    *,
    dataset_name: str | None,
    model_name: str | None,
    validation_inputs: list[str],
) -> dict[str, dict[str, Any]]:
    config: dict[str, dict[str, Any]] = {
        "llm": {
            "filename": "llm_interpretations.json",
            "k_examples": 20,
            "system_prompt": SYSTEM_PROMPT,
            "use_unique_words": None,
            "unique_words_kwargs": None,
        },
        "topk": {
            "filename": "topk_interpretations.json",
            "k": 5,
            "use_unique_words": True,
            "unique_words_kwargs": _default_topk_unique_words_kwargs(validation_inputs),
        },
    }

    if (
        dataset_name == "dair-ai/emotion"
        and model_name == "nateraw/bert-base-uncased-emotion"
    ):
        config["llm"]["system_prompt"] = EMOTION_6_SYSTEM_PROMPT
        return config

    if (
        dataset_name == "google-research-datasets/go_emotions"
        and model_name == "SamLowe/roberta-base-go_emotions"
    ):
        config["llm"]["system_prompt"] = EMOTION_28_SYSTEM_PROMPT
        return config

    if (
        dataset_name == "Hate-speech-CNERG/hatexplain"
        and model_name == "Hate-speech-CNERG/bert-base-uncased-hatexplain"
    ):
        config["llm"].update(
            {
                "system_prompt": HATEXPLAIN_SYSTEM_PROMPT,
                "use_unique_words": 5,
                "unique_words_kwargs": {
                    "count_min_threshold": 3,
                    "lemmatize": True,
                    "words_to_ignore": [],
                },
            }
        )
        config["topk"].update(
            {
                "use_unique_words": 3,
                "unique_words_kwargs": {
                    "count_min_threshold": 3,
                    "lemmatize": True,
                    "words_to_ignore": [],
                },
            }
        )
        return config

    if (
        dataset_name == "LabHC/bias_in_bios"
        and model_name
        == "/datasets/shared_datasets/BIOS/models/RoBERTa_occBIOS_10epochs_g1/"
    ):
        config["llm"].update(
            {
                "system_prompt": BIOS_SYSTEM_PROMPT,
                "use_unique_words": 3,
                "unique_words_kwargs": {
                    "count_min_threshold": 3,
                    "lemmatize": True,
                    "words_to_ignore": [],
                },
            }
        )
        config["topk"].update(
            {
                "filename": "topk_words_interpretations.json",
                "k": 10,
                "use_unique_words": 1,
                "unique_words_kwargs": {
                    "count_min_threshold": 3,
                    "lemmatize": False,
                    "words_to_ignore": [],
                },
            }
        )
        return config

    return config


def load_concept_model(concept_explainer, model_path: Path, device):
    concept_explainer.concept_model._set_fitted()
    if isinstance(concept_explainer, SemiNMFConcepts):
        D = torch.load(model_path, map_location=device)
        concept_explainer.concept_model.D = D
        return concept_explainer
    if hasattr(concept_explainer.concept_model, "load_state_dict"):
        concept_explainer.concept_model.load_state_dict(
            torch.load(str(model_path), map_location=device, weights_only=True)
        )
        return concept_explainer
    raise NotImplementedError(
        f"Loading not implemented for {type(concept_explainer).__name__}"
    )


def save_concept_model(concept_explainer, model_path: Path) -> None:
    if isinstance(concept_explainer, SemiNMFConcepts):
        torch.save(concept_explainer.concept_model.D, model_path)
        return
    if hasattr(concept_explainer.concept_model, "state_dict"):
        torch.save(concept_explainer.concept_model.state_dict(), str(model_path))
        return
    raise NotImplementedError(
        f"Saving not implemented for {type(concept_explainer).__name__}"
    )


def compute_activations_difference(
    activations: torch.Tensor, p: int = 10
) -> torch.Tensor:
    """
    Compute the pair-wise activations differences.
    For each activation, we compute p pair-wise differences.
    """
    n = activations.shape[0]

    # indices of the left partners
    left_indices = torch.arange(n).repeat(p)

    # indices of the right partners
    # For each i, sample p partners excluding in [0, n-1] excluding i
    random = torch.randint(0, n - 1, (n, p), dtype=torch.long)
    reference = torch.arange(n).unsqueeze(1)
    right_indices = (random + (random >= reference)).view(-1)

    return activations[left_indices] - activations[right_indices]


def load_or_fit_concept_model(
    splitter,
    concept_dir: Path,
    activations: torch.Tensor,
    method,
    nb_concepts: int,
    device,
    activations_difference: bool = False,
):
    concept_explainer = method(
        splitter,
        nb_concepts=nb_concepts,
        device=device,
    )

    concept_model_path = concept_dir / "concept_model.pt"
    if concept_model_path.exists():
        concept_explainer = load_concept_model(
            concept_explainer, concept_model_path, device
        )
    else:
        if activations_difference:
            activations = compute_activations_difference(activations, p=10)
        concept_explainer.fit(activations)
        save_concept_model(concept_explainer, concept_model_path)
    return concept_explainer


def load_or_compute_interpretations(
    concept_explainer,
    validation_inputs: list[str],
    interpretation,
    concept_dir: Path,
    llm_model: str | None,
    classes_names,
    *,
    dataset_name: str | None = None,
    model_name: str | None = None,
) -> dict[int, str]:
    interpretation_name = name_for(interpretation)
    config = get_interpretation_config(
        dataset_name=dataset_name,
        model_name=model_name,
        validation_inputs=validation_inputs,
    )

    if interpretation is LLMLabels:
        llm_config = config["llm"]
        prompt_path = concept_dir / llm_config["filename"]
        system_prompt = llm_config["system_prompt"] + ", ".join(classes_names)
        metadata = {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "interpretation_name": interpretation_name,
            "llm_model": llm_model,
            "filename": llm_config["filename"],
            "k_examples": llm_config["k_examples"],
            "system_prompt": system_prompt,
            "use_unique_words": llm_config["use_unique_words"],
            "unique_words_kwargs": llm_config["unique_words_kwargs"],
        }

        cached_interpretations = _load_cached_json(
            prompt_path,
            expected_metadata=metadata,
        )
        if cached_interpretations is not None:
            return {int(k): v for k, v in cached_interpretations.items()}

        if llm_model is None:
            raise ValueError(
                "An LLM model name is required to compute missing LLM interpretations."
            )

        llm_labels_kwargs: dict[str, Any] = {
            "concept_explainer": concept_explainer,
            "llm_interface": OpenAILLM(
                api_key=os.getenv("OPENAI_API_KEY"), model=llm_model
            ),
            "k_examples": llm_config["k_examples"],
            "system_prompt": system_prompt,
        }
        if llm_config["use_unique_words"] is not None:
            llm_labels_kwargs["use_unique_words"] = llm_config["use_unique_words"]
        if llm_config["unique_words_kwargs"] is not None:
            llm_labels_kwargs["unique_words_kwargs"] = llm_config["unique_words_kwargs"]

        llm_labels_method = LLMLabels(**llm_labels_kwargs)
        interpretations = llm_labels_method.interpret(
            inputs=validation_inputs,
            concepts_indices="all",
        )

        _save_json_with_metadata(
            prompt_path,
            interpretations,
            metadata=metadata,
        )
        return interpretations  # type: ignore

    if interpretation is TopKInputs:
        topk_config = config["topk"]
        prompt_path = concept_dir / topk_config["filename"]
        metadata = {
            "dataset_name": dataset_name,
            "model_name": model_name,
            "interpretation_name": interpretation_name,
            "filename": topk_config["filename"],
            "k": topk_config["k"],
            "use_unique_words": topk_config["use_unique_words"],
            "unique_words_kwargs": topk_config["unique_words_kwargs"],
        }

        cached_interpretations = _load_cached_json(
            prompt_path,
            expected_metadata=metadata,
        )
        if cached_interpretations is not None:
            return cached_interpretations  # type: ignore[return-value]

        topk_inputs_method = TopKInputs(
            concept_explainer=concept_explainer,
            k=topk_config["k"],
            use_unique_words=topk_config["use_unique_words"],
            unique_words_kwargs=topk_config["unique_words_kwargs"],
        )

        interpretations = topk_inputs_method.interpret(
            inputs=validation_inputs,
            concepts_indices="all",
        )
        interpretations = {k: list(v.keys()) for k, v in interpretations.items()}
        _save_json_with_metadata(
            prompt_path,
            interpretations,
            metadata=metadata,
        )
        return interpretations

    raise NotImplementedError(
        f"Interpretation not implemented for {interpretation_name}"
    )


def load_or_compute_global_importances(
    concept_explainer,
    validation_inputs: list[str],
    concept_dir: Path,
    device,
    batch_size,
) -> torch.Tensor:
    importances_path = concept_dir / "importances.pt"
    if importances_path.exists():
        gradients = torch.load(importances_path, map_location=device)
    else:
        gradients = concept_explainer.concept_output_gradient(
            inputs=validation_inputs,
            concepts_x_gradients=True,
            batch_size=batch_size,
        )
        torch.save(gradients, importances_path)

    return torch.stack(gradients).squeeze().mean(dim=0)


def prepare_concept_explanation_resources(
    *,
    dataset_name: str | None,
    model_name: str,
    split_point: str | int,
    save_root: Path,
    train_inputs: list[str],
    validation_inputs: list[str],
    classes: list[str],
    method,
    nb_concepts_ratio: int | float,
    activations_difference: bool,
    interpretation,
    llm_model: str | None,
    device: str,
    batch_size: int,
) -> GlobalConceptExplanation:
    from interpreto import SplitterForClassification
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from utils.data import load_or_compute_activations

    method_name = name_for(method)[:-8]  # remove "Concepts" suffix
    interpretation_name = name_for(interpretation)
    nb_concepts = int(len(classes) * nb_concepts_ratio)
    diff_str = "_diff" if activations_difference else ""
    concept_dir = (
        save_root / "concept_models" / f"{method_name}{diff_str}_nc{nb_concepts}"
    )
    concept_dir.mkdir(parents=True, exist_ok=True)

    # Keep the task model loading local to the concept-specific preparation step.
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    splitter = SplitterForClassification(
        model,
        tokenizer=tokenizer,
        device_map=device,
        batch_size=batch_size,
    )

    activations = load_or_compute_activations(
        splitter=splitter,
        train_inputs=train_inputs,
        activations_path=save_root / "activations.pt",
        device=device,
    )
    if isinstance(activations, dict):
        activations = splitter.get_split_activations(activations)
    concept_explainer = load_or_fit_concept_model(
        splitter=splitter,
        concept_dir=concept_dir,
        activations=activations,
        method=method,
        nb_concepts=nb_concepts,
        device=device,
        activations_difference=activations_difference,
    )
    concepts_interpretation = load_or_compute_interpretations(
        concept_explainer=concept_explainer,
        validation_inputs=validation_inputs,
        interpretation=interpretation,
        concept_dir=concept_dir,
        llm_model=llm_model,
        classes_names=classes,
        dataset_name=dataset_name,
        model_name=model_name,
    )
    global_importances = load_or_compute_global_importances(
        concept_explainer=concept_explainer,
        validation_inputs=validation_inputs,
        concept_dir=concept_dir,
        device=device,
        batch_size=batch_size,
    )

    return GlobalConceptExplanation(
        concept_explainer=concept_explainer,
        concepts_interpretation=concepts_interpretation,
        global_importances=global_importances,
        method_name=method_name,
        interpretation_name=interpretation_name,
        nb_concepts=nb_concepts,
        concept_dir=concept_dir,
    )


def compute_local_concept_explanation(
    *,
    global_explanation: GlobalConceptExplanation,
    local_inputs: list[str],
    nb_learning_samples: int,
) -> LocalConceptExplanation:
    # Only the learning-phase samples are used to build local concept explanations.
    local_importances = global_explanation.concept_explainer.concept_output_gradient(
        inputs=local_inputs[:nb_learning_samples],  # type: ignore
        concepts_x_gradients=True,
    )
    local_importances = [
        sample_importance.squeeze(1) for sample_importance in local_importances
    ]

    return LocalConceptExplanation(local_importances=local_importances)
