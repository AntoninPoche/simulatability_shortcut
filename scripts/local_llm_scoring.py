"""Score ConSim prompt groups with a local Hugging Face LLM.

Reads prompt JSONL, skips already-scored keys, runs inference, and appends
score rows to the output CSV.

Handles two modes automatically:
  - **New ConSim** (one user prompt per evaluation sample): scores each
    prompt independently via exact-match on the last word.
  - **Old ConSim** (single user prompt, multiple expected answers): sends one
    prompt, parses multi-line "Sample_i: class" response.

Usage examples::

    python scripts/local_llm_scoring.py qwen3.5-9b data/prompts/GE_concepts.jsonl
    python scripts/local_llm_scoring.py qwen3.5-9b data/prompts/GE_old_consim.jsonl --no-thinking
"""

from __future__ import annotations

import argparse
import ast
import csv
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data import iter_jsonl, LLM_MODELS, resolve_llm_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score ConSim prompt groups with a local LLM.",
    )
    parser.add_argument(
        "judge_model",
        help=f"LLM judge model. Short names: {', '.join(LLM_MODELS.keys())}.",
    )
    parser.add_argument(
        "prompt_file",
        type=Path,
        help="Path to the prompt JSONL file to score.",
    )
    parser.add_argument(
        "--thinking", "--enable-thinking",
        action="store_true",
        default=False,
        help="Enable thinking/chain-of-thought mode in chat template.",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        default=False,
        help="Explicitly disable thinking mode.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Max tokens to generate per prompt (default: 2048).",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=10,
        help="Batch size for generation (default: 10).",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model inference.",
    )
    return parser.parse_args()


def get_score_path(judge_model: str, thinking: bool) -> Path:
    """Derive score CSV path from judge model name and thinking mode."""
    model_slug = judge_model.replace("/", "_")
    thinking_str = "_thinking" if thinking else ""
    return Path(f"data/consim_{model_slug}{thinking_str}.csv")


def load_treated_keys(score_path: Path) -> set[str]:
    """Load keys already scored, or create the CSV with header."""
    if not score_path.exists():
        score_path.parent.mkdir(parents=True, exist_ok=True)
        with open(score_path, "w") as handle:
            handle.write(
                "dataset,model,classes_subset,seed,method,nb_concepts,"
                "interpretation,prompt_type,specification,time,score\n"
            )
        return set()

    scores_df = pd.read_csv(
        score_path,
        index_col=list(range(9)),
        dtype={"seed": "Int64", "nb_concepts": "Int64"},
    )
    return set(
        str(index).replace("nan", "None").replace("<NA>", "None")
        for index in scores_df.index
    )


def render_prompt(
    tokenizer,
    system_prompt: str,
    user_prompt: str,
    thinking: bool,
) -> str:
    """Format system/user pair into text for the model."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    elif thinking:
        raise ValueError("thinking=True but no chat template found.")
    return system_prompt + "\n\n" + user_prompt


def generate_answers(
    model,
    tokenizer,
    system_prompt: str,
    user_prompts: list[str],
    *,
    thinking: bool,
    max_new_tokens: int,
    batch_size: int,
) -> list[str]:
    """Run batched forward passes for a prompt group."""
    full_prompts = [
        render_prompt(tokenizer, system_prompt, up, thinking)
        for up in user_prompts
    ]

    generated_texts: list[str] = []
    model_device = next(model.parameters()).device
    for batch_start in range(0, len(full_prompts), batch_size):
        batch_prompts = full_prompts[batch_start : batch_start + batch_size]
        model_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
        ).to(model_device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
            )

        completion_ids = generated_ids[:, model_inputs["input_ids"].shape[1] :]
        generated_texts.extend(
            text.strip()
            for text in tokenizer.batch_decode(
                completion_ids, skip_special_tokens=True
            )
        )

    return generated_texts


def extract_prediction(text: str | None) -> str | None:
    """Extract predicted token from raw output (last word)."""
    if text is None:
        return None
    processed = text.strip().replace("\n", " ").split(" ")[-1]
    return processed if processed else None


def parse_old_consim_response(response: str, expected_length: int) -> list[str] | None:
    """
    Parse multi-prediction response from old ConSim (all-at-once mode).
    Expected format: "Sample_0: class_a\\nSample_1: class_b\\n..."
    """
    if not response:
        return None

    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]

    if len(lines) != expected_length:
        # Try to extract whatever we can
        predictions = []
        for line in lines:
            if ":" in line:
                pred = line.split(":", 1)[1].strip().lower().split(" ")[0]
                predictions.append(pred)
            else:
                predictions.append(line.strip().lower().split(" ")[0])
        if len(predictions) != expected_length:
            return None
        return predictions

    predictions = []
    for line in lines:
        if ":" in line:
            pred = line.split(":", 1)[1].strip().lower().split(" ")[0]
        else:
            pred = line.strip().lower().split(" ")[0]
        predictions.append(pred)
    return predictions


def score_prompt_group_new(answers: list[str], expected_answers: list[str]) -> float:
    """Score new ConSim mode: one answer per evaluation sample."""
    score = 0
    failed = 0
    for answer, expected in zip(answers, expected_answers, strict=True):
        predicted = extract_prediction(answer)
        if predicted is None:
            failed += 1
            continue
        score += int(predicted.lower() == expected.lower())
    return score / len(expected_answers)


def score_prompt_group_old(answer: str, expected_answers: list[str]) -> float:
    """Score old ConSim mode: parse multi-line response."""
    predictions = parse_old_consim_response(answer, len(expected_answers))
    if predictions is None:
        return 0.0
    score = sum(
        int(pred.lower() == expected.lower())
        for pred, expected in zip(predictions, expected_answers, strict=True)
    )
    return score / len(expected_answers)


def main() -> None:
    args = parse_args()
    thinking = args.thinking and not args.no_thinking

    # Resolve short model name.
    args.judge_model = resolve_llm_model(args.judge_model)

    prompt_path = args.prompt_file
    if not prompt_path.exists():
        print(f"Prompt file not found: {prompt_path}")
        sys.exit(1)

    score_path = get_score_path(args.judge_model, thinking)
    treated_keys = load_treated_keys(score_path)

    # Determine which keys need scoring.
    requested_keys = {pg["key"] for pg in iter_jsonl(prompt_path)}
    missing_keys = requested_keys - treated_keys

    if not missing_keys:
        print(f"Nothing to score. All {len(requested_keys)} keys already in {score_path}.")
        return

    print(f"Judge model:  {args.judge_model}")
    print(f"Prompt file:  {prompt_path}")
    print(f"Thinking:     {thinking}")
    print(f"Score file:   {score_path}")
    print(f"To score:     {len(missing_keys)} / {len(requested_keys)} keys")
    print()

    # Load model.
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model,
        torch_dtype="auto",
        device_map=args.device,
    )
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if (
        model.generation_config.pad_token_id is None
        and tokenizer.pad_token_id is not None
    ):
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Score missing keys.
    with open(score_path, "a", newline="") as handle:
        writer = csv.writer(handle)

        for prompt_group in tqdm(iter_jsonl(prompt_path), total=len(missing_keys)):
            if prompt_group["key"] not in missing_keys:
                continue

            user_prompts = prompt_group["user_prompts"]
            expected_answers = prompt_group["expected_answers"]

            # Detect mode: old ConSim has 1 user prompt but multiple expected answers.
            is_old_consim = (
                len(user_prompts) == 1 and len(expected_answers) > 1
            )

            answers = generate_answers(
                model=model,
                tokenizer=tokenizer,
                system_prompt=prompt_group["system_prompt"],
                user_prompts=user_prompts,
                thinking=thinking,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.generation_batch_size,
            )

            if is_old_consim:
                accuracy = score_prompt_group_old(answers[0], expected_answers)
            else:
                accuracy = score_prompt_group_new(answers, expected_answers)

            # Append score row.
            writer.writerow(
                ast.literal_eval(prompt_group["key"]) + (datetime.now(), accuracy)
            )
            handle.flush()

    print(
        f"\nScored {len(missing_keys)} keys with {args.judge_model} "
        f"(thinking={thinking}). Results appended to {score_path}."
    )


if __name__ == "__main__":
    main()
