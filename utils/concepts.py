from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import NamedTuple

import torch

from interpreto.concepts import SemiNMFConcepts
from interpreto.concepts.interpretations import LLMLabels, TopKInputs

from utils.llm_interface import HuggingFaceLLM

# ---------------------------------------------------------------------------
# CLI name → directory-name prefix mapping.
# Directory names use the interpreto class name minus the "Concepts" suffix.
# ---------------------------------------------------------------------------
CONCEPT_METHOD_NAMES: dict[str, str] = {
    "seminmf": "SemiNMF",
    "ica": "ICA",
    "kmeans": "KMeans",
    "pca": "PCA",
    "svd": "SVD",
    "batchtopk_sae": "BatchTopKSAE",
    "vanilla_sae": "VanillaSAE",
    "neurons": "NeuronsAs",
}

INTERPRETATION_NAMES: dict[str, str] = {
    "topk": "TopKInputs",
    "llm": "LLMLabels",
}


def get_concept_method_class(method_key: str):
    """Lazy-load the interpreto concept class for the given CLI key."""
    from interpreto.concepts import (
        BatchTopKSAEConcepts,
        ICAConcepts,
        KMeansConcepts,
        NeuronsAsConcepts,
        PCAConcepts,
        SemiNMFConcepts,
        SVDConcepts,
        VanillaSAEConcepts,
    )

    classes = {
        "seminmf": SemiNMFConcepts,
        "ica": ICAConcepts,
        "kmeans": KMeansConcepts,
        "pca": PCAConcepts,
        "svd": SVDConcepts,
        "batchtopk_sae": BatchTopKSAEConcepts,
        "vanilla_sae": VanillaSAEConcepts,
        "neurons": NeuronsAsConcepts,
    }
    if method_key not in classes:
        raise ValueError(
            f"Unknown concept method '{method_key}'. "
            f"Available: {', '.join(classes.keys())}"
        )
    return classes[method_key]


def get_interpretation_class(interp_key: str):
    """Lazy-load the interpreto interpretation class for the given CLI key."""
    from interpreto.concepts.interpretations import LLMLabels, TopKInputs

    classes = {
        "llm": LLMLabels,
        "topk": TopKInputs,
    }
    if interp_key not in classes:
        raise ValueError(
            f"Unknown interpretation '{interp_key}'. "
            f"Available: {', '.join(classes.keys())}"
        )
    return classes[interp_key]


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

AG_NEWS_SYSTEM_PROMPT = """You are a meticulous AI researcher conducting an important investigation into patterns found in language.
Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given a list of text examples.
How important each text is for the behavior is listed after each example in parentheses, with importance from 0 to 10.

- Try to produce a concise final description. Simply describe the text features that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples, but try to summarize the patterns found in the examples.
- Do not make lists of possible explanations. Find a single concept that best describes the examples.
- Strike the balance between being concise and informative. From 1 to 5 words. 5 is an absolute maximum.
- Refrain from including uninformative elements like "patterns found include ...", "the examples show ...", or "text contains ...".
- Here are some examples: geopolitical conflict, stock market, team performance, software release, trade agreements, championship results, corporate earnings, space exploration...
- Do not hesitate to qualify the concepts, 'technology' can be 'consumer electronics'..., or 'politics' could be 'diplomatic tensions'.
- These are examples from the AG News dataset, hence concepts correspond to differentiators between the classes. Do not reference them in your label, but take them into account when labelling. Classes: 
"""

ROTTEN_TOMATOES_SYSTEM_PROMPT = """You are a meticulous AI researcher conducting an important investigation into patterns found in language.
Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given a list of text examples.
How important each text is for the behavior is listed after each example in parentheses, with importance from 0 to 10.

- Try to produce a concise final description. Simply describe the text features that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples, but try to summarize the patterns found in the examples.
- Do not make lists of possible explanations. Find a single concept that best describes the examples.
- Strike the balance between being concise and informative. From 1 to 5 words. 5 is an absolute maximum.
- Refrain from including uninformative elements like "patterns found include ...", "the examples show ...", or "text contains ...".
- Here are some examples: sharp wit, dull pacing, visual brilliance, weak dialogue, emotional depth, predictable plot, strong performances, lazy writing...
- Do not hesitate to qualify the concepts, 'acting' can be 'nuanced lead performance'..., or 'story' could be 'formulaic romance plot'.
- These are examples from the Rotten Tomatoes movie review dataset, hence concepts correspond to differentiators between the classes. Do not reference them in your label, but take them into account when labelling. Classes: 
"""

IMDB_SYSTEM_PROMPT = """You are a meticulous AI researcher conducting an important investigation into patterns found in language.
Your task is to analyze text and provide an explanation that thoroughly encapsulates possible patterns found in it.
Guidelines:

You will be given a list of text examples.
How important each text is for the behavior is listed after each example in parentheses, with importance from 0 to 10.

- Try to produce a concise final description. Simply describe the text features that are common in the examples, and what patterns you found.
- If the examples are uninformative, you don't need to mention them. Don't focus on giving examples, but try to summarize the patterns found in the examples.
- Do not make lists of possible explanations. Find a single concept that best describes the examples.
- Strike the balance between being concise and informative. From 1 to 5 words. 5 is an absolute maximum.
- Refrain from including uninformative elements like "patterns found include ...", "the examples show ...", or "text contains ...".
- Here are some examples: compelling narrative, poor acting, cinematography praise, plot holes, character development, waste of time, masterful direction, disappointing sequel...
- Do not hesitate to qualify the concepts, 'acting' can be 'over-the-top villain'..., or 'quality' could be 'low-budget effects'.
- These are examples from the IMDB movie review dataset, hence concepts correspond to differentiators between the classes. Do not reference them in your label, but take them into account when labelling. Classes: 
"""


def name_for(obj) -> str:
    return obj.__name__ if hasattr(obj, "__name__") else str(obj)


class GlobalConceptExplanation(NamedTuple):
    concept_explainer: Any
    concepts_interpretation: dict[int, str]
    global_importances: torch.Tensor
    method_name: str
    interpretation_name: str
    nb_concepts: int | None
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

    if (
        dataset_name == "fancyzhx/ag_news"
        and model_name == "raulbs7/ag-news-classifier"
    ):
        config["llm"]["system_prompt"] = AG_NEWS_SYSTEM_PROMPT
        return config

    if (
        dataset_name == "cornell-movie-review-data/rotten_tomatoes"
        and model_name == "keerthi1515/roberta-sentiment-rotten-tomatoes"
    ):
        config["llm"]["system_prompt"] = ROTTEN_TOMATOES_SYSTEM_PROMPT
        return config

    if (
        dataset_name == "stanfordnlp/imdb"
        and model_name == "philipobiorah/bert-imdb-model"
    ):
        config["llm"]["system_prompt"] = IMDB_SYSTEM_PROMPT
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
        if hasattr(concept_explainer.concept_model, "training"):
            concept_explainer.concept_model.training = False
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


def load_or_fit_concept_model(
    splitter,
    concept_dir: Path,
    activations: torch.Tensor | None,
    method,
    nb_concepts: int | None,
    device,
    batch_size: int,
):
    method_name = name_for(method)
    concept_init_kwargs: dict[str, Any] = {}
    if method_name == "BatchTopKSAEConcepts":
        if nb_concepts is None:
            raise ValueError("BatchTopKSAEConcepts requires nb_concepts.")
        # Interpreto's installed BatchTopK SAE keeps a top-k across the whole batch.
        # Newer APIs may expose this as batch_top_k; this environment falls back to top_k.
        concept_init_kwargs["top_k"] = max(1, int(nb_concepts / 3)) * batch_size

    if method_name == "NeuronsAsConcepts":
        concept_explainer = method(splitter)
        concept_model_path = concept_dir / "concept_model.pt"
        if concept_model_path.exists():
            return load_concept_model(concept_explainer, concept_model_path, device)
        save_concept_model(concept_explainer, concept_model_path)
        return concept_explainer

    concept_explainer = method(
        splitter,
        nb_concepts=nb_concepts,
        device=device,
        **concept_init_kwargs,
    )

    concept_model_path = concept_dir / "concept_model.pt"
    if concept_model_path.exists():
        concept_explainer = load_concept_model(
            concept_explainer, concept_model_path, device
        )
    else:
        if activations is None:
            raise ValueError(f"Activations are required to fit {method_name}.")
        if "SAE" in method_name:
            from interpreto.concepts.methods.overcomplete import (
                DeadNeuronsReanimationLoss,
            )

            concept_explainer.fit(
                activations,
                criterion=DeadNeuronsReanimationLoss,
                batch_size=batch_size,
                device=device,
            )
        else:
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
    device: str = "cuda",
    batch_size: int = 8,
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
            "llm_interface": HuggingFaceLLM(
                model=llm_model, device=device, batch_size=batch_size
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
        interpretations = {
            k: list(v.keys()) if v is not None else []
            for k, v in interpretations.items()
        }
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

    if isinstance(gradients, list):
        gradients = torch.stack(gradients)
    if gradients.dim() > 2:
        gradients = gradients.squeeze().mean(dim=0)
    return gradients


def prepare_concept_explanation_resources(
    *,
    dataset_name: str | None,
    model_name: str,
    save_root: Path,
    train_inputs: list[str],
    validation_inputs: list[str],
    classes: list[str],
    method,
    nb_concepts_ratio: int | float,
    interpretation,
    llm_model: str | None,
    device: str,
    batch_size: int,
) -> GlobalConceptExplanation:
    from interpreto import SplitSequenceClassification as SplitterForClassification
    from transformers import AutoModelForSequenceClassification
    from utils.data import load_hf_tokenizer, load_or_compute_activations

    method_name = name_for(method)[:-8]  # remove "Concepts" suffix
    interpretation_name = name_for(interpretation)
    nb_concepts = None if method_name == "NeuronsAs" else int(len(classes) * nb_concepts_ratio)
    concept_dir = save_root / "concept_models" / f"{method_name}_nc{nb_concepts}"
    concept_dir.mkdir(parents=True, exist_ok=True)

    # Keep the task model loading local to the concept-specific preparation step.
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = load_hf_tokenizer(model_name)
    splitter = SplitterForClassification(
        model,
        tokenizer=tokenizer,
        device_map=device,
        batch_size=batch_size,
    )

    activations = None
    if method_name != "NeuronsAs":
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
        batch_size=batch_size,
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
        device=device,
        batch_size=batch_size,
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


# ---------------------------------------------------------------------------
# Pre-compute and cache local importances for all test samples.
# ---------------------------------------------------------------------------


def compute_and_cache_all_local_importances(
    *,
    concept_explainer,
    test_inputs: list[str],
    concept_dir: Path,
    batch_size: int = 64,
) -> list[torch.Tensor]:
    """Compute concept_output_gradient for ALL test samples and cache to disk.

    Stores a list of tensors (one per test sample, squeezed) at
    ``concept_dir / "all_local_importances.pt"``.

    Pre-computes importances so that ``make_prompts.py`` can load them
    without needing the task model or interpreto.
    """
    cache_path = concept_dir / "all_local_importances.pt"
    if cache_path.exists():
        print(f"  Local importances already cached: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    print(f"  Computing local importances for {len(test_inputs)} test samples...")
    raw_importances = concept_explainer.concept_output_gradient(
        inputs=test_inputs,
        concepts_x_gradients=True,
        batch_size=batch_size,
    )
    # Squeeze the class dimension (same as compute_local_concept_explanation).
    all_local_importances = [imp.squeeze(1) for imp in raw_importances]
    torch.save(all_local_importances, cache_path)
    print(f"  Cached at: {cache_path}")
    return all_local_importances


def load_local_importances(
    *,
    concept_dir: Path,
    sample_indices: list[int],
    nb_learning_samples: int,
) -> LocalConceptExplanation:
    """Load pre-computed local importances for specific sample indices.

    Only the first ``nb_learning_samples`` indices are used (learning phase).
    Raises FileNotFoundError if the cache does not exist.
    """
    cache_path = concept_dir / "all_local_importances.pt"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Pre-computed local importances not found at {cache_path}."
        )
    all_importances = torch.load(cache_path, map_location="cpu")
    # Select only the learning-phase samples.
    local_importances = [
        all_importances[idx] for idx in sample_indices[:nb_learning_samples]
    ]
    return LocalConceptExplanation(local_importances=local_importances)


# ---------------------------------------------------------------------------
# Load-only function for make_prompts.py (no interpreto / task model needed).
# ---------------------------------------------------------------------------


def get_concept_dir(
    *,
    save_root: Path,
    method_name: str,
    nb_concepts: int | None,
) -> Path:
    """Reconstruct the concept_dir path from parameters."""
    return save_root / "concept_models" / f"{method_name}_nc{nb_concepts}"


def load_concept_explanation_resources(
    *,
    save_root: Path,
    method_name: str,
    nb_concepts: int | None,
    interpretation_name: str,
    classes: list[str],
    device: str = "cpu",
) -> GlobalConceptExplanation:
    """Load pre-built concept resources from cache (load-only, never creates).

    Raises FileNotFoundError if any required artifact is missing.
    Does NOT require interpreto or the task model — only reads cached files.
    """
    concept_dir = get_concept_dir(
        save_root=save_root,
        method_name=method_name,
        nb_concepts=nb_concepts,
    )

    # Check concept model exists (we don't load it — not needed for prompts).
    concept_model_path = concept_dir / "concept_model.pt"
    if not concept_model_path.exists():
        raise FileNotFoundError(f"Concept model not found at {concept_model_path}.")

    # Load interpretations.
    # Determine interpretation filename from config.
    if interpretation_name == "TopKInputs":
        # Check for BIOS-style topk_words filename first, then standard.
        for fname in ("topk_words_interpretations.json", "topk_interpretations.json"):
            interp_path = concept_dir / fname
            if interp_path.exists():
                break
        else:
            raise FileNotFoundError(f"TopK interpretations not found in {concept_dir}.")
    elif interpretation_name == "LLMLabels":
        interp_path = concept_dir / "llm_interpretations.json"
        if not interp_path.exists():
            raise FileNotFoundError(f"LLM interpretations not found at {interp_path}.")
    else:
        raise ValueError(f"Unknown interpretation: {interpretation_name}")

    with open(interp_path) as handle:
        raw_interpretations = json.load(handle)
    concepts_interpretation = {int(k): v for k, v in raw_interpretations.items()}

    # Load global importances.
    importances_path = concept_dir / "importances.pt"
    if not importances_path.exists():
        raise FileNotFoundError(f"Global importances not found at {importances_path}.")
    global_importances = torch.load(importances_path, map_location=device)
    if isinstance(global_importances, list):
        global_importances = torch.stack(global_importances)
    # Handle both pre-reduced (mean already applied) and raw stacked tensors.
    if global_importances.dim() > 2:
        global_importances = global_importances.squeeze().mean(dim=0)

    return GlobalConceptExplanation(
        concept_explainer=None,  # Not needed — local importances are pre-computed.
        concepts_interpretation=concepts_interpretation,
        global_importances=global_importances,
        method_name=method_name,
        interpretation_name=interpretation_name,
        nb_concepts=nb_concepts,
        concept_dir=concept_dir,
    )


# ---------------------------------------------------------------------------
# Full build pipeline for make_prompts.py fallback.
# ---------------------------------------------------------------------------


def build_concept_resources(
    *,
    method_key: str,
    interpretation_key: str,
    dataset_name: str,
    model_name: str,
    save_root: Path,
    train_inputs: list[str],
    validation_inputs: list[str],
    test_inputs: list[str],
    classes: list[str],
    nb_concepts_ratio: float,
    llm_model: str | None = None,
    device: str = "cuda",
    batch_size: int = 64,
) -> None:
    """Build all concept artifacts (model, interpretations, importances).

    This wraps the full pipeline from ``prepare_concept_explanation_resources``
    and ``compute_and_cache_all_local_importances``. Called by ``make_prompts.py``
    when cached artifacts are missing.

    Releases GPU memory after completion.
    """
    import gc

    method_class = get_concept_method_class(method_key)
    interpretation_class = get_interpretation_class(interpretation_key)

    global_explanation = prepare_concept_explanation_resources(
        dataset_name=dataset_name,
        model_name=model_name,
        save_root=save_root,
        train_inputs=train_inputs,
        validation_inputs=validation_inputs,
        classes=classes,
        method=method_class,
        nb_concepts_ratio=nb_concepts_ratio,
        interpretation=interpretation_class,
        llm_model=llm_model,
        device=device,
        batch_size=batch_size,
    )

    compute_and_cache_all_local_importances(
        concept_explainer=global_explanation.concept_explainer,
        test_inputs=test_inputs,
        concept_dir=global_explanation.concept_dir,
        batch_size=batch_size,
    )

    # Release GPU memory — the task model and concept explainer are no longer needed.
    del global_explanation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_or_build_concept_resources(
    args,
    *,
    dataset_name: str,
    model_name: str,
    save_root: Path,
    train_inputs: list[str],
    validation_inputs: list[str],
    test_inputs: list[str],
    classes: list[str],
) -> GlobalConceptExplanation:
    """Load concept resources from cache, building them if missing.

    Extracts concept-specific parameters from ``args``:
    method, interpretation, nb_concepts_ratio, llm_model, device, batch_size.
    """
    nb_concepts = int(len(classes) * args.nb_concepts_ratio)
    if args.method == "neurons":
        nb_concepts = None
    interpretation_name = INTERPRETATION_NAMES[args.interpretation]
    method_dir_name = CONCEPT_METHOD_NAMES[args.method]

    try:
        resources = load_concept_explanation_resources(
            save_root=save_root,
            method_name=method_dir_name,
            nb_concepts=nb_concepts,
            interpretation_name=interpretation_name,
            classes=classes,
        )
        # Also verify local importances exist (not checked by load above).
        if not (resources.concept_dir / "all_local_importances.pt").exists():
            raise FileNotFoundError("Local importances missing.")
        return resources
    except FileNotFoundError:
        pass

    # Artifacts not cached — build them now.
    print(f"  Concept artifacts missing, building for {args.method}...")
    llm_model = args.llm_model if args.interpretation == "llm" else None
    build_concept_resources(
        method_key=args.method,
        interpretation_key=args.interpretation,
        dataset_name=dataset_name,
        model_name=model_name,
        save_root=save_root,
        train_inputs=train_inputs,
        validation_inputs=validation_inputs,
        test_inputs=test_inputs,
        classes=classes,
        nb_concepts_ratio=args.nb_concepts_ratio,
        llm_model=llm_model,
        device=args.device,
        batch_size=args.batch_size,
    )
    return load_concept_explanation_resources(
        save_root=save_root,
        method_name=method_dir_name,
        nb_concepts=nb_concepts,
        interpretation_name=interpretation_name,
        classes=classes,
    )
