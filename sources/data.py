from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from interpreto import ModelWithSplitPoints
    from interpreto.concepts.metrics.simulatability.base import AutomatedSimulatability

MODELS_DATASETS = {
    "SamLowe/roberta-base-go_emotions": "google-research-datasets/go_emotions",
    "nateraw/bert-base-uncased-emotion": "dair-ai/emotion",
    "Hate-speech-CNERG/bert-base-uncased-hatexplain": "Hate-speech-CNERG/hatexplain",
    "/datasets/shared_datasets/BIOS/models/RoBERTa_occBIOS_10epochs_g1/": "LabHC/bias_in_bios",
}

ABBREVIATIONS = {
    "models": {
        "SamLowe/roberta-base-go_emotions": "RB",
        "nateraw/bert-base-uncased-emotion": "B",
        "Hate-speech-CNERG/bert-base-uncased-hatexplain": "B",
        "/datasets/shared_datasets/BIOS/models/RoBERTa_occBIOS_10epochs_g1/": "RB",
    },
    "datasets": {
        "google-research-datasets/go_emotions": "GE",
        "dair-ai/emotion": "E",
        "Hate-speech-CNERG/hatexplain": "HE",
        "LabHC/bias_in_bios": "BIOS",
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
        "surgeon",  # 0
        "pastor",  # 1
        "photographer",  # 2
        "professor",  # 3
        "chiropractor",  # 4
        "software_engineer",  # 5
        "teacher",  # 6
        "poet",  # 7
        "dj",  # 8
        "rapper",  # 9
        "paralegal",  # 10
        "physician",  # 11
        "journalist",  # 12
        "architect",  # 13
        "attorney",  # 14
        "yoga_teacher",  # 15
        "nurse",  # 16
        "painter",  # 17
        "model",  # 18
        "composer",  # 19
        "personal_trainer",  # 20
        "filmmaker",  # 21
        "comedian",  # 22
        "accountant",  # 23
        "interior_designer",  # 24
        "dentist",  # 25
        "psychologist",  # 26
        "dietitian",  # 27
    ],
}

DATASET_LABEL_COLUMNS = {
    "google-research-datasets/go_emotions": "labels",
    "dair-ai/emotion": "label",
    "LabHC/bias_in_bios": "profession",
    "Hate-speech-CNERG/hatexplain": "label",
}

MODEL_SPLIT_POINTS = {
    "SamLowe/roberta-base-go_emotions": 11,
    "nateraw/bert-base-uncased-emotion": "bert.pooler",
    "Hate-speech-CNERG/bert-base-uncased-hatexplain": "bert.pooler",
    "/datasets/shared_datasets/BIOS/models/RoBERTa_occBIOS_10epochs_g1/": 11,
}


def iter_jsonl(path: Path):
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def get_save_root(model_name: str, split_point: str | int) -> Path:
    return Path("data") / model_name.replace("/", "_") / str(split_point)


def get_local_elements_path(
    save_root: Path,
    classes_subset: list[int] | None = None,
) -> Path:
    if classes_subset is None:
        suffix = "all_classes"
    else:
        suffix = "classes_" + "-".join(str(class_id) for class_id in classes_subset)
    return save_root / f"local_elements_{suffix}.json"


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
        test_labels = list(test_split[DATASET_LABEL_COLUMNS.get(dataset, "label")])
    elif dataset != "Hate-speech-CNERG/hatexplain":
        train_inputs = list(dataset_dict["train"].shuffle(seed=0)["text"])
        validation_inputs = list(dataset_dict["validation"]["text"])
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
    else:
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
    return train_inputs, validation_inputs, test_inputs, torch.tensor(test_labels)


def load_or_compute_activations(
    model_with_split_points: ModelWithSplitPoints,
    train_inputs: list[str],
    activations_path: Path,
    device: str,
    granularity,
) -> list[torch.Tensor]:
    import torch

    if activations_path.exists():
        return torch.load(activations_path, map_location=device)

    activations_dict = model_with_split_points.get_activations(
        inputs=train_inputs,
        activation_granularity=granularity,
        include_predicted_classes=True,
        tqdm_bar=True,
    )
    activations = model_with_split_points.get_split_activations(activations_dict)
    torch.save(activations, activations_path)
    return activations  # type: ignore


def load_or_compute_predictions(
    *,
    model_name: str,
    inputs: list[str],
    path: Path,
    device: str,
    batch_size: int,
) -> torch.Tensor:
    import torch

    if path.exists():
        return torch.load(path)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # load model only for prediction computations
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    predictions = torch.empty(len(inputs), dtype=torch.int8).to(device)
    with torch.no_grad():
        model.eval()
        for batch_id in tqdm(range(0, len(inputs), batch_size), desc="Predictions"):
            batch_inputs = inputs[batch_id : batch_id + batch_size]
            batch_logits = model(
                input_ids=tokenizer(batch_inputs, return_tensors="pt").input_ids.to(
                    device
                ),
                return_dict=True,
            )["logits"]
            batch_predictions = torch.argmax(batch_logits, dim=1)
            predictions[batch_id : batch_id + batch_size] = batch_predictions

    predictions = predictions.cpu()
    torch.save(predictions, path)

    # Release the task model before explanation-specific work starts.
    del tokenizer
    del model
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return predictions


def load_or_compute_local_elements(
    save_root: Path,
    *,
    simulatability_metric: AutomatedSimulatability,
    inputs: list[str],
    labels: torch.Tensor,
    predictions: torch.Tensor,
    seeds: list[int],
    nb_samples: int = 40,
    classes_subset: list[int] | None = None,
) -> dict[int, dict[str, int | list[int] | list[str]]]:
    cached_payload: dict[str, dict[str, int | list[int] | list[str]]] = {}
    path = get_local_elements_path(save_root, classes_subset)
    if path.exists():
        with open(path) as f:
            cached_payload = json.load(f)

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
        with open(path, "w") as f:
            json.dump(cached_payload, f, indent=2)
            f.write("\n")

    return {seed: cached_payload[str(seed)] for seed in seeds}
