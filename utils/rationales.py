from __future__ import annotations

import json
import os
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data import DATASET_CLASSES_NAMES, iter_jsonl

LABEL_NATURE_BY_DATASET = {
    "google-research-datasets/go_emotions": "emotion",
    "dair-ai/emotion": "emotion",
    "LabHC/bias_in_bios": "profession",
    "Hate-speech-CNERG/hatexplain": "hate speech detection",
}


def construct_rationale_prompt(
    text: str,
    label: int,
    prediction: int,
    dataset_name: str,
) -> str:
    """
    Construct a prompt for rationale generation.
    """
    str_prediction = DATASET_CLASSES_NAMES[dataset_name][prediction]

    # Rationale prompt
    label_nature = LABEL_NATURE_BY_DATASET[dataset_name]
    rationale_prompt = f"Given the following text and its predicted {label_nature} label, provide a brief explanation (< 50 words) justifying why this prediction makes sense.\n\n"
    rationale_prompt += f"Text: '{text}'\n"
    rationale_prompt += f"Predicted {label_nature}: {str_prediction}\n\n"
    rationale_prompt += "Explanation:"

    return rationale_prompt


def prepare_rationale_prompts(
    sample_ids: list[int],
    inputs: list[str],
    labels: torch.Tensor,
    predictions: torch.Tensor,
    dataset_name: str,
    tokenizer,
    missing_sample_ids: list[int],
) -> list[tuple[int, str]]:
    """
    Pre-render all prompts for the samples that still need generations.
    """
    prepared_prompts: list[tuple[int, str]] = []
    sample_id_to_offset = {
        sample_id: offset for offset, sample_id in enumerate(sample_ids)
    }

    for sample_id in missing_sample_ids:
        offset = sample_id_to_offset[sample_id]
        text = inputs[offset]
        label = labels[offset]
        prediction = predictions[offset]
        rationale_prompt = construct_rationale_prompt(
            text, int(label.item()), int(prediction.item()), dataset_name
        )

        messages = [{"role": "user", "content": rationale_prompt}]
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prepared_prompts.append((sample_id, rendered_prompt))

    return prepared_prompts


def load_or_generate_rationales(
    model_name: str,
    dataset_name: str,
    inputs: list[str],
    labels: torch.Tensor,
    predictions: torch.Tensor,
    sample_ids: list[int],
    save_root: Path,
    device,
    batch_size: int = 8,
    max_new_tokens: int = 64,
):
    """
    Generate rationale artifacts for a given dataset and model.
    """
    if len(sample_ids) != len(inputs):
        raise ValueError(
            "`sample_ids` must match the rationale inputs length. "
            f"Got {len(sample_ids)=} and {len(inputs)=}."
        )

    save_path = save_root / "rationales" / f"{model_name.replace('/', '_')}.jsonl"
    os.makedirs(save_path.parent, exist_ok=True)

    cached_rationales: dict[int, dict[str, int | str | None]] = {}
    if save_path.exists():
        for record in iter_jsonl(save_path):
            sample_id = record.get("sample_id")
            if sample_id is None:
                continue
            cached_rationales[int(sample_id)] = record

    missing_sample_ids = [
        sample_id for sample_id in sample_ids if sample_id not in cached_rationales
    ]
    if not missing_sample_ids:
        return {sample_id: cached_rationales[sample_id] for sample_id in sample_ids}

    # Load model
    model_device = torch.device(device)
    model_kwargs = {}
    if model_device.type == "cuda":
        torch.cuda.set_device(model_device)
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model = model.to(model_device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()

    prepared_prompts = prepare_rationale_prompts(
        sample_ids=sample_ids,
        inputs=inputs,
        labels=labels,
        predictions=predictions,
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        missing_sample_ids=missing_sample_ids,
    )

    with open(save_path, "a") as handle:
        for batch_start in tqdm(
            range(0, len(prepared_prompts), batch_size),
            desc="Rationales",
        ):
            batch = prepared_prompts[batch_start : batch_start + batch_size]
            batch_prompts = [prompt for _, prompt in batch]

            model_inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
            ).to(model_device)

            with torch.inference_mode():
                outputs = model.generate(  # type: ignore
                    **model_inputs,  # type: ignore
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            completions = tokenizer.batch_decode(
                outputs[:, model_inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            for (sample_id, _), completion in zip(
                batch,
                completions,
                strict=True,
            ):
                record = {
                    "sample_id": sample_id,
                    "rationale": completion,
                }
                json.dump(record, handle)
                handle.write("\n")
                cached_rationales[sample_id] = record

    del tokenizer
    del model
    if model_device.type == "cuda":
        torch.cuda.empty_cache()

    return {sample_id: cached_rationales[sample_id] for sample_id in sample_ids}


def group_rationales_by_seed(
    rationales: dict[int, dict[str, int | str | None]],
    seed_indices: dict[int, list[int]],
) -> dict[int, list[str]]:
    """
    Group rationale artifacts by seed.
    """
    rationale_by_seed: dict[int, list[str]] = {}
    for seed, indices in seed_indices.items():
        rationale_by_seed[seed] = []
        for index in indices:
            rationale_by_seed[seed].append(rationales[index]["rationale"])  # type: ignore
    return rationale_by_seed
