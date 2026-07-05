from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from interpreto import SplitterForClassification
    from utils.simulatability import AutomatedSimulatability

MODELS_DATASETS = {
    "SamLowe/roberta-base-go_emotions": "google-research-datasets/go_emotions",
    "nateraw/bert-base-uncased-emotion": "dair-ai/emotion",
    "Hate-speech-CNERG/bert-base-uncased-hatexplain": "Hate-speech-CNERG/hatexplain",
    "Fannyjrd/roberta-bios-biased": "LabHC/bias_in_bios",
    "raulbs7/ag-news-classifier": "fancyzhx/ag_news",
    "keerthi1515/roberta-sentiment-rotten-tomatoes": "cornell-movie-review-data/rotten_tomatoes",
    "philipobiorah/bert-imdb-model": "stanfordnlp/imdb",
}

ABBREVIATIONS = {
    "models": {
        "SamLowe/roberta-base-go_emotions": "RB",
        "nateraw/bert-base-uncased-emotion": "B",
        "Hate-speech-CNERG/bert-base-uncased-hatexplain": "B",
        "Fannyjrd/roberta-bios-biased": "RB",
        "raulbs7/ag-news-classifier": "DB",
        "keerthi1515/roberta-sentiment-rotten-tomatoes": "RB",
        "philipobiorah/bert-imdb-model": "B",
    },
    "datasets": {
        "google-research-datasets/go_emotions": "GE",
        "dair-ai/emotion": "E",
        "Hate-speech-CNERG/hatexplain": "HE",
        "LabHC/bias_in_bios": "BIOS",
        "fancyzhx/ag_news": "AG",
        "cornell-movie-review-data/rotten_tomatoes": "RT",
        "stanfordnlp/imdb": "IMDB",
    },
}
DATASET_CLASSES_NAMES = {
    "google-research-datasets/go_emotions": [
        "admiration",  # 0
        "amusement",  # 1
        "anger",  # 2
        "annoyance",  # 3
        "approval",  # 4
        "caring",  # 5
        "confusion",  # 6
        "curiosity",  # 7
        "desire",  # 8
        "disappointment",  # 9
        "disapproval",  # 10
        "disgust",  # 11
        "embarrassment",  # 12
        "excitement",  # 13
        "fear",  # 14
        "gratitude",  # 15
        "grief",  # 16
        "joy",  # 17
        "love",  # 18
        "nervousness",  # 19
        "optimism",  # 20
        "pride",  # 21
        "realization",  # 22
        "relief",  # 23
        "remorse",  # 24
        "sadness",  # 25
        "surprise",  # 26
        "neutral",  # 27
    ],
    "dair-ai/emotion": ["sadness", "joy", "love", "anger", "fear", "surprise"],
    "Hate-speech-CNERG/hatexplain": [
        "hatespeech",
        "normal",
        "offensive",
    ],
    "LabHC/bias_in_bios": [
        "accountant",  # 0
        "architect",  # 1
        "attorney",  # 2
        "chiropractor",  # 3
        "comedian",  # 4
        "composer",  # 5
        "dentist",  # 6
        "dietitian",  # 7
        "dj",  # 8
        "filmmaker",  # 9
        "interior_designer",  # 10
        "journalist",  # 11
        "model",  # 12
        "nurse",  # 13
        "painter",  # 14
        "paralegal",  # 15
        "pastor",  # 16
        "personal_trainer",  # 17
        "photographer",  # 18
        "physician",  # 19
        "poet",  # 20
        "professor",  # 21
        "psychologist",  # 22
        "rapper",  # 23
        "software_engineer",  # 24
        "surgeon",  # 25
        "teacher",  # 26
        "yoga_teacher",  # 27
    ],
    "fancyzhx/ag_news": [
        "World",  # 0
        "Sports",  # 1
        "Business",  # 2
        "Sci/Tech",  # 3
    ],
    "cornell-movie-review-data/rotten_tomatoes": [
        "neg",  # 0
        "pos",  # 1
    ],
    "stanfordnlp/imdb": [
        "neg",  # 0
        "pos",  # 1
    ],
}

DATASET_LABEL_COLUMNS = {
    "google-research-datasets/go_emotions": "labels",
    "dair-ai/emotion": "label",
    "LabHC/bias_in_bios": "profession",
    "Hate-speech-CNERG/hatexplain": "label",
    "fancyzhx/ag_news": "label",
    "cornell-movie-review-data/rotten_tomatoes": "label",
    "stanfordnlp/imdb": "label",
}

# Canonical class subsets for each dataset.
# Each dataset has multiple subsets used in experiments.
# All subsets for a given dataset are run together.
DATASET_CLASSES_SUBSETS: dict[str, list[list[int]]] = {
    "google-research-datasets/go_emotions": [
        [2, 3, 27],  # anger, annoyance, neutral
        # [2, 3, 9, 10],  # anger, annoyance, disappointment, disapproval
        [6, 7],  # confusion, curiosity
        # [0, 4, 5],  # admiration, approval, caring
    ],
    "dair-ai/emotion": [
        [0, 1, 2, 3, 4, 5],  # all classes
    ],
    "Hate-speech-CNERG/hatexplain": [
        [0, 1, 2],  # all classes
    ],
    "LabHC/bias_in_bios": [
        [6, 19, 25],  # dentist, physician, surgeon
        [21, 22, 26],  # professor, psychologist, teacher
        [1, 21, 24],  # architect, professor, software_engineer
        [9, 11, 18],  # filmmaker, journalist, photographer
    ],
    "fancyzhx/ag_news": [
        [0, 1, 2, 3],  # all classes: World, Sports, Business, Sci/Tech
    ],
    "cornell-movie-review-data/rotten_tomatoes": [
        [0, 1],  # all classes: neg, pos
    ],
    "stanfordnlp/imdb": [
        [0, 1],  # all classes: neg, pos
    ],
}

# Short-name registry for LLM models used across the repo (judging, rationales, labeling).
# Keys are CLI-friendly short names; values are full HuggingFace model paths.
LLM_MODELS: dict[str, str] = {
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen3.6-27b": "Qwen/Qwen3.6-27B",
    "phi4": "microsoft/phi-4",
    "ministral-14b": "mistralai/Ministral-3-14B-Instruct-2512",
    "gemma4-31b": "google/gemma-4-31B-it",
    "gemma4-12b": "google/gemma-4-12B-it",
    "gpt-oss-20b": "openai/gpt-oss-20b",
}


def resolve_llm_model(name: str) -> str:
    """Resolve a short model alias to its full HuggingFace path. Pass through if not found."""
    return LLM_MODELS.get(name, name)


def iter_jsonl(path: Path):
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def get_save_root(model_name: str) -> Path:
    return Path("data") / model_name.replace("/", "_")


def get_local_elements_path(
    save_root: Path,
    classes_subset: list[int] | None = None,
    nb_samples: int = 20,
) -> Path:
    if classes_subset is None:
        suffix = "all_classes"
    else:
        suffix = "classes_" + "-".join(str(class_id) for class_id in classes_subset)
    return save_root / f"local_elements_{suffix}_n{nb_samples}.json"


def _validate_local_elements_payload(
    payload: dict[str, Any],
    *,
    seed: int,
    path: Path,
    nb_samples: int,
    classes_subset: list[int] | None,
) -> None:
    lengths = {
        field: len(payload.get(field, []))
        for field in ("indices", "texts", "labels", "predictions")
    }
    bad_lengths = {field: length for field, length in lengths.items() if length != nb_samples}
    if bad_lengths:
        raise ValueError(
            f"Invalid cached local elements for seed {seed} in {path}: expected {nb_samples} "
            f"items per field, got {bad_lengths}. Delete this cache after fixing the "
            "underlying predictions and regenerate it."
        )

    nb_learning_samples = payload.get("nb_learning_samples")
    if not isinstance(nb_learning_samples, int) or not 0 < nb_learning_samples < nb_samples:
        raise ValueError(
            f"Invalid cached local elements for seed {seed} in {path}: "
            f"nb_learning_samples={nb_learning_samples!r}."
        )

    if classes_subset is None:
        return

    requested_classes = set(int(class_id) for class_id in classes_subset)
    labels = [int(label) for label in payload.get("labels", [])]
    predictions = [int(prediction) for prediction in payload.get("predictions", [])]

    labels_outside_subset = sorted(set(labels) - requested_classes)
    predictions_outside_subset = sorted(set(predictions) - requested_classes)
    if labels_outside_subset or predictions_outside_subset:
        raise ValueError(
            f"Invalid cached local elements for seed {seed} in {path}: labels outside subset "
            f"{labels_outside_subset}, predictions outside subset {predictions_outside_subset}."
        )

    missing_prediction_classes = sorted(requested_classes - set(predictions))
    if missing_prediction_classes:
        raise ValueError(
            f"Invalid cached local elements for seed {seed} in {path}: selected predictions "
            f"do not cover requested classes {missing_prediction_classes}. Delete this cache "
            "after fixing the underlying predictions and regenerate it."
        )


def load_dataset_splits(dataset: str):
    import torch
    from datasets import load_dataset

    dataset_dict = load_dataset(dataset)
    if dataset == "LabHC/bias_in_bios":
        test_split = dataset_dict["test"].shuffle(seed=0)
        train_inputs = list(dataset_dict["train"].shuffle(seed=0)["hard_text"])[:50000]
        validation_inputs = list(dataset_dict["dev"].shuffle(seed=0)["hard_text"])[
            :5000
        ]
        test_inputs = list(test_split["hard_text"])
        # The Fannyjrd/roberta-bios-biased model was trained directly on the
        # dataset's "profession" column, so its logits already align with the
        # dataset labels (alphabetical class order). No remapping needed.
        test_labels = list(test_split[DATASET_LABEL_COLUMNS.get(dataset, "label")])
    elif dataset == "Hate-speech-CNERG/hatexplain":
        train_inputs = [" ".join(x["post_tokens"]) for x in dataset_dict["train"]]  # type: ignore
        validation_inputs = [
            " ".join(x["post_tokens"])  # type: ignore
            for x in dataset_dict["validation"]
        ]
        test_inputs = [" ".join(x["post_tokens"]) for x in dataset_dict["test"]]  # type: ignore
        test_labels = []
        labels_indices = []
        for i, x in enumerate(dataset_dict["test"]):
            votes = x["annotators"]["label"]  # type: ignore
            counts = Counter(votes)
            label, count = counts.most_common(1)[0]
            if count > len(votes) / 2:
                test_labels.append(label)
                labels_indices.append(i)
        test_inputs = [test_inputs[i] for i in labels_indices]
    else:
        # Generic path for datasets with a "text" column and integer "label".
        shuffled_train = dataset_dict["train"].shuffle(seed=0)
        all_train_texts = list(shuffled_train["text"])

        # Some datasets lack a dedicated validation split; carve one from train.
        if "validation" in dataset_dict:
            train_inputs = all_train_texts[:50000]
            validation_inputs = list(dataset_dict["validation"]["text"])
        else:
            val_size = min(5000, len(all_train_texts) // 5)
            train_inputs = all_train_texts[:-val_size][:50000]
            validation_inputs = all_train_texts[-val_size:]

        test_inputs = []
        test_labels = []
        for i in range(len(dataset_dict["test"])):
            label = dataset_dict["test"][DATASET_LABEL_COLUMNS.get(dataset, "label")][i]
            if isinstance(label, list):
                if len(label) > 1:
                    continue
                label = label[0]
            test_labels.append(label)
            test_inputs.append(dataset_dict["test"]["text"][i])
    return train_inputs, validation_inputs, test_inputs, torch.tensor(test_labels)


def load_or_compute_activations(
    splitter: SplitterForClassification,
    inputs: list[str],
    activations_path: Path,
    device: str,
) -> tuple[Any, torch.Tensor]:
    import torch

    if activations_path.exists():
        activations, predictions = torch.load(activations_path, map_location=device)
        return activations, predictions.cpu()

    activations, predictions = splitter.get_activations(
        inputs=inputs,
        tqdm_bar=True,
        forward_kwargs={"truncation": True},
    )
    predictions = predictions.cpu()
    activations_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((activations, predictions), activations_path)
    return activations, predictions


def load_or_compute_dataset_activations(
    *,
    model_name: str,
    save_root: Path,
    train_inputs: list[str],
    validation_inputs: list[str],
    test_inputs: list[str],
    device: str,
    batch_size: int,
) -> tuple[
    tuple[Any, torch.Tensor], tuple[Any, torch.Tensor], tuple[Any, torch.Tensor]
]:
    import gc
    import torch
    from interpreto import SplitterForClassification

    split_specs = (
        (train_inputs, save_root / "activations.pt"),
        (validation_inputs, save_root / "validation_activations.pt"),
        (test_inputs, save_root / "test_activations.pt"),
    )

    splitter = None
    outputs = []
    for inputs, path in split_specs:
        # Load cached activations if they exist.
        if path.exists():
            activations, predictions = torch.load(path, map_location=device)
            outputs.append((activations, predictions.cpu()))
        else:
            if splitter is None:
                # Load the task model only once.
                splitter = SplitterForClassification(
                    model_name,
                    device_map=device,
                    batch_size=batch_size,
                )

            # Compute activations and predictions.
            activations, predictions = splitter.get_activations(
                inputs=inputs,
                tqdm_bar=True,
                forward_kwargs={"truncation": True},
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save((activations, predictions), path)
            outputs.append((activations, predictions.cpu()))

    if splitter is not None:
        del splitter
        gc.collect()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return tuple(outputs)  # type: ignore[return-value]


def load_or_compute_local_elements(
    save_root: Path,
    *,
    simulatability_metric: AutomatedSimulatability,
    inputs: list[str],
    labels: torch.Tensor,
    predictions: torch.Tensor,
    seeds: list[int],
    nb_samples: int = 20,
    classes_subset: list[int] | None = None,
) -> dict[int, dict[str, int | list[int] | list[str]]]:
    cached_payload: dict[str, dict[str, int | list[int] | list[str]]] = {}
    path = get_local_elements_path(save_root, classes_subset, nb_samples)
    if path.exists():
        with open(path) as f:
            cached_payload = json.load(f)

    for seed in seeds:
        if str(seed) in cached_payload:
            _validate_local_elements_payload(
                cached_payload[str(seed)],
                seed=seed,
                path=path,
                nb_samples=nb_samples,
                classes_subset=classes_subset,
            )

    missing_seeds = [seed for seed in seeds if str(seed) not in cached_payload]
    if missing_seeds:
        for seed in missing_seeds:
            indices, selected_inputs, selected_labels, selected_predictions = (
                simulatability_metric.select_examples(
                    inputs=inputs,
                    labels=labels,
                    predictions=predictions,
                    nb_samples=nb_samples,
                    seed=seed,
                    classes_subset=classes_subset,
                )
            )
            split_index = len(selected_inputs) // 2
            cached_payload[str(seed)] = {
                "nb_learning_samples": split_index,
                "indices": indices.view(-1).tolist(),
                "texts": selected_inputs,
                "labels": selected_labels.view(-1).tolist(),
                "predictions": selected_predictions.view(-1).tolist(),
                "prompt_ids": list(range(len(selected_inputs) - split_index)),
            }
            _validate_local_elements_payload(
                cached_payload[str(seed)],
                seed=seed,
                path=path,
                nb_samples=nb_samples,
                classes_subset=classes_subset,
            )
        with open(path, "w") as f:
            json.dump(cached_payload, f, indent=2)
            f.write("\n")

    for seed in seeds:
        _validate_local_elements_payload(
            cached_payload[str(seed)],
            seed=seed,
            path=path,
            nb_samples=nb_samples,
            classes_subset=classes_subset,
        )

    return {seed: cached_payload[str(seed)] for seed in seeds}
