"""
This file scores non-already treated ConSim prompt groups with a local Hugging Face LLM.

When ran it does the following:
    - Load the requested prompt groups from `data/consim_prompts.jsonl`
    - Load the already treated keys from `consim_{model}_thinking_{thinking}.csv`
    - Keep only the missing keys
    - Run the local model on the prompts of each missing key in small batches
    - Append one score row per key
"""

from datetime import datetime
import ast
import csv
import sys
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ in {None, ""}:
    # Allow `python scripts/local_llm_scoring.py` to resolve the repo-local `utils` package.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data import iter_jsonl

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "Qwen/Qwen3.5-9B"
THINKING = False
thinking_str = "_thinking" if THINKING else ""
GENERATION_KWARGS = {
    "max_new_tokens": 2048,
}
GENERATION_BATCH_SIZE = 10

PROMPT_PATH = Path("data/consim_prompts.jsonl")
SCORE_PATH = Path(f"data/consim_{MODEL_NAME.replace('/', '_')}{thinking_str}.csv")


def load_treated_keys_or_setup_score_file() -> set[str]:
    """
    Load the keys which are already treated.
    Or create the score file with the usual header if it does not exist yet.
    """
    if not SCORE_PATH.exists():
        SCORE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SCORE_PATH, "w") as handle:
            handle.write(
                "dataset,model,classes_subset,seed,method,nb_concepts,interpretation,prompt_type,specification,time,score\n"
            )
        return set()

    # Read score file as a pandas DataFrame to extract indices
    scores_df = pd.read_csv(
        SCORE_PATH,
        index_col=list(range(9)),
        dtype={"seed": "Int64", "nb_concepts": "Int64"},
    )
    return set(
        str(index).replace("nan", "None").replace("<NA>", "None")
        for index in scores_df.index
    )


def extract_prediction_from_output(text: str | None) -> str | None:
    """
    Extract the predicted token from the raw model output text.
    It is assumed to be the last word, as there is the thinking part prior.
    """
    if text is None:
        return None

    # some processing to extract the output text
    processed_text = text.strip().replace("\n", " ").split(" ")[-1]
    if not processed_text:
        return None

    return processed_text


def render_prompt(system_prompt: str, user_prompt: str) -> str:
    """
    Convert the system/user pair into the final text sent to the model.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=THINKING,
        )
    elif THINKING:
        raise ValueError("THINKING=True but no chat template found.")

    return system_prompt + "\n\n" + user_prompt


def generate_answers(system_prompt: str, user_prompts: list[str]) -> list[str]:
    """
    Run batched forward passes for one prompt group.
    """
    # Format the prompts exactly once before tokenization.
    full_prompts = [
        render_prompt(system_prompt=system_prompt, user_prompt=user_prompt)
        for user_prompt in user_prompts
    ]

    generated_texts: list[str] = []
    model_device = next(model.parameters()).device
    for batch_start in range(0, len(full_prompts), GENERATION_BATCH_SIZE):
        batch_prompts = full_prompts[batch_start : batch_start + GENERATION_BATCH_SIZE]
        model_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
        ).to(model_device)

        # Generate only the answer continuation.
        with torch.no_grad():
            generated_ids = model.generate(  # type: ignore
                **model_inputs,
                **GENERATION_KWARGS,
            )

        # Remove the prompt tokens and decode only the completions.
        completion_ids = generated_ids[:, model_inputs["input_ids"].shape[1] :]
        generated_texts.extend(
            decoded_text.strip()
            for decoded_text in tokenizer.batch_decode(
                completion_ids, skip_special_tokens=True
            )
        )

    return generated_texts


def score_prompt_group(prompt_group: dict) -> float:
    """
    Score one prompt group and return its exact-match accuracy.
    """
    answers = generate_answers(
        system_prompt=prompt_group["system_prompt"],
        user_prompts=prompt_group["user_prompts"],
    )

    score = 0
    failed = 0
    for answer, expected_answer in zip(
        answers, prompt_group["expected_answers"], strict=True
    ):
        # Compare only the last word.
        predicted_token = extract_prediction_from_output(answer)

        # Store failed keys
        if predicted_token is None:
            failed += 1
            continue

        score += int(predicted_token.lower() == expected_answer.lower())

    if failed:
        print(f"Failed to score {failed} prompts for {prompt_group['key']}.")

    # return accuracy
    return score / len(prompt_group["expected_answers"])


if __name__ == "__main__":
    global tokenizer, model

    # Load the requested keys first.
    requested_keys = {prompt_group["key"] for prompt_group in iter_jsonl(PROMPT_PATH)}

    # Keep only the missing experiment keys (not it output score file).
    missing_keys = requested_keys - load_treated_keys_or_setup_score_file()

    # Stop early if there is nothing left to score.
    if len(missing_keys) == 0:
        print(f"Nothing to score in {SCORE_PATH}.")
        exit()

    # Load the local model only when there is real work to do.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        device_map=DEVICE,
    )
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if (
        model.generation_config.pad_token_id is None
        and tokenizer.pad_token_id is not None
    ):
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Append one row per newly scored key.
    with open(SCORE_PATH, "a", newline="") as handle:
        writer = csv.writer(handle)

        # Re-read the prompt file and score only the missing keys.
        for prompt_group in tqdm(iter_jsonl(PROMPT_PATH), total=len(missing_keys)):
            # Skip already treated keys immediately.
            if prompt_group["key"] not in missing_keys:
                continue

            # Score the current prompt group.
            accuracy = score_prompt_group(prompt_group)

            # Append the usual score row.
            writer.writerow(
                ast.literal_eval(prompt_group["key"]) + (datetime.now(), accuracy)
            )
            handle.flush()

    # Print a short end-of-run summary.
    print(
        f"Scored {len(missing_keys)} missing keys with {MODEL_NAME} "
        f"and thinking={THINKING}. Results appended to {SCORE_PATH}."
    )
