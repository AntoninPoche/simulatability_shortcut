"""Score ConSim prompt groups with a local Hugging Face LLM.

Reads prompt JSONL, skips already-scored keys, runs inference, and appends
score rows to the output CSV.

Handles two modes automatically:
  - **New ConSim** (one user prompt per evaluation sample): scores each
    prompt independently via exact-match on the last word.
  - **Old ConSim** (single user prompt, multiple expected answers): sends one
    prompt, parses multi-line "Sample_i: class" response.

Usage examples::

    python scripts/llm_scoring.py Qwen/Qwen3-0.6B
    python scripts/llm_scoring.py qwen3.5-9b data/prompts/GE_concepts.jsonl
    python scripts/llm_scoring.py qwen3.5-9b data/prompts/GE_old_consim.jsonl
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
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


PROMPT_DIR = Path("data/prompts")
SAMPLE_ID_RE = re.compile(r"\bSample_(\d+)\s*:")
CLASS_LABEL_RE = re.compile(r"^class[\s_-]*(\d+)$", re.IGNORECASE)
BARE_INT_RE = re.compile(r"^\d+$")
SCI_TECH_RE = re.compile(r"\bscience\s+(?:and|&)\s+technology\b", re.IGNORECASE)
CLASSES_LINE_RE = re.compile(r"^The classes are:\s*\[(.*)\]\s*$")


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
        nargs="?",
        default=None,
        help="Path to the prompt JSONL file to score. If omitted, score all data/prompts/*.jsonl files.",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        default=False,
        help="Enable thinking/chain-of-thought mode in chat template.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Max tokens to generate per prompt (default: 32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for generation (default: 16).",
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
    return Path(f"data/consim_{model_slug}{thinking_str}_v2.csv")


def get_generation_log_path(judge_model: str, thinking: bool) -> Path:
    """Derive raw-generation JSONL path from judge model name and thinking mode."""
    model_slug = judge_model.replace("/", "_")
    thinking_str = "_thinking" if thinking else ""
    return Path(f"data/generations/{model_slug}{thinking_str}.jsonl")


def load_treated_keys(score_path: Path) -> set[str]:
    """Load keys already scored, or create the CSV with header."""
    if not score_path.exists():
        score_path.parent.mkdir(parents=True, exist_ok=True)
        with open(score_path, "w") as handle:
            handle.write(
                "dataset,model,classes_subset,seed,method,nb_concepts,"
                "interpretation,prompt_type,specification,time,score,"
                "num_correct,num_valid,num_expected\n"
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


def resolve_prompt_paths(prompt_file: Path | None) -> list[Path]:
    """Return the requested prompt file(s), defaulting to all prompt JSONL files."""
    if prompt_file is not None:
        if not prompt_file.exists():
            print(f"Prompt file not found: {prompt_file}")
            sys.exit(1)
        return [prompt_file]

    prompt_paths = sorted(PROMPT_DIR.glob("*.jsonl"))
    if not prompt_paths:
        print(f"No prompt JSONL files found in {PROMPT_DIR}.")
        sys.exit(1)
    return prompt_paths


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
    progress_desc: str = "Generation batches",
) -> list[str]:
    """Run batched forward passes for a prompt group."""
    full_prompts = [
        render_prompt(tokenizer, system_prompt, up, thinking) for up in user_prompts
    ]
    return generate_completions(
        model=model,
        tokenizer=tokenizer,
        full_prompts=full_prompts,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        progress_desc=progress_desc,
    )


def generate_completions(
    model,
    tokenizer,
    full_prompts: list[str],
    *,
    max_new_tokens: int,
    batch_size: int,
    progress_desc: str = "Generation batches",
) -> list[str]:
    """Run batched forward passes for already-rendered prompts."""
    generated_texts: list[str] = []
    model_device = next(model.parameters()).device
    batch_starts = range(0, len(full_prompts), batch_size)
    for batch_start in tqdm(
        batch_starts,
        total=(len(full_prompts) + batch_size - 1) // batch_size,
        desc=progress_desc,
        leave=False,
    ):
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
            for text in tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        )

    return generated_texts


def is_old_consim_prompt_group(prompt_group: dict) -> bool:
    """Detect old ConSim mode: one prompt asks for multiple answers."""
    return (
        len(prompt_group["user_prompts"]) == 1
        and len(prompt_group["expected_answers"]) > 1
    )


def allowed_answers_for_prompt_group(prompt_group: dict) -> list[str]:
    """Allowed labels from the system prompt, falling back to expected answers."""
    allowed_answers = extract_allowed_labels(prompt_group["system_prompt"])
    if allowed_answers:
        return allowed_answers
    return prompt_group["expected_answers"]


def normalize_label_text(text: str | None) -> str | None:
    """Normalize a predicted or expected label for exact-match scoring."""
    if text is None:
        return None
    processed = text.strip().strip("`\"'[](){}.,;:")
    processed = re.sub(r"\s+", " ", processed)
    return processed if processed else None


def anonymized_class_id(text: str | None, *, allow_bare_int: bool = False) -> int | None:
    """Return the numeric id for labels like Class_3, Class3, or optionally 3."""
    text_norm = normalize_label_text(text)
    if text_norm is None:
        return None
    match = CLASS_LABEL_RE.fullmatch(text_norm)
    if match is not None:
        return int(match.group(1))
    if allow_bare_int and BARE_INT_RE.fullmatch(text_norm):
        return int(text_norm)
    return None


def prediction_matches(predicted: str | None, expected: str) -> bool:
    """Case-insensitive exact match after lightweight label cleanup."""
    predicted_norm = normalize_label_text(predicted)
    expected_norm = normalize_label_text(expected)
    if predicted_norm is None or expected_norm is None:
        return False
    if predicted_norm.lower() == expected_norm.lower():
        return True

    expected_class_id = anonymized_class_id(expected_norm)
    predicted_class_id = anonymized_class_id(
        predicted_norm,
        allow_bare_int=expected_class_id is not None,
    )
    if expected_class_id is not None and predicted_class_id == expected_class_id:
        return True

    return label_pattern(expected_norm).fullmatch(predicted_norm) is not None


def extract_allowed_labels(system_prompt: str) -> list[str]:
    """Extract class labels from the prompt's explicit class list."""
    for line in system_prompt.splitlines():
        match = CLASSES_LINE_RE.match(line.strip())
        if match is None:
            continue
        return [label.strip() for label in match.group(1).split(",") if label.strip()]
    return []


def extract_sample_ids(text: str) -> list[int]:
    """Extract old-ConSim sample ids from a prompt or response."""
    return [int(match.group(1)) for match in SAMPLE_ID_RE.finditer(text)]


def label_pattern(label: str) -> re.Pattern[str]:
    """Build a conservative pattern for labels embedded in generated text."""
    label_parts = [part for part in re.split(r"[^A-Za-z0-9]+", label) if part]
    if label_parts:
        body = r"[\s_/+-]*".join(re.escape(part) for part in label_parts)
    else:
        body = re.escape(label)
    return re.compile(
        rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


def match_expected_label(candidate: str, expected_answers: list[str]) -> str | None:
    """Return the expected label matched by a raw candidate span, if any."""
    candidate_norm = normalize_label_text(candidate)
    if candidate_norm is None:
        return None

    for expected in expected_answers:
        if prediction_matches(candidate_norm, expected):
            return expected

    expected_lower = {answer.lower() for answer in expected_answers}
    lowered = candidate_norm.lower()
    if "pos" in expected_lower and label_pattern("positive").search(lowered):
        return "pos"
    if "neg" in expected_lower and label_pattern("negative").search(lowered):
        return "neg"

    for expected in expected_answers:
        if label_pattern(expected).search(candidate_norm):
            return expected
        if expected.lower() == "sci/tech" and SCI_TECH_RE.search(candidate_norm):
            return expected

    return None


def ordered_prediction_candidates(text: str) -> list[str]:
    """Candidate spans from most to least likely final-label locations."""
    processed = text.strip().replace("\n", " ")
    candidates: list[str] = []
    if ":" in processed:
        candidates.append(processed.rsplit(":", 1)[1])
        candidates.append(processed.split(":", 1)[1])

    tokens = processed.split()
    if tokens:
        candidates.append(tokens[-1])
        candidates.append(tokens[0])
        candidates.extend(reversed(tokens))
    candidates.append(processed)

    deduped = []
    seen = set()
    for candidate in candidates:
        candidate_key = candidate.strip().lower()
        if candidate_key and candidate_key not in seen:
            deduped.append(candidate)
            seen.add(candidate_key)
    return deduped


def extract_old_consim_prediction(
    line: str,
    allowed_answers: list[str],
) -> str | None:
    """Extract a class label from one old-ConSim response line."""
    sample_match = SAMPLE_ID_RE.search(line)
    prediction_text = line[sample_match.end() :] if sample_match else line

    candidates = []
    if ":" in prediction_text:
        candidates.append(prediction_text.rsplit(":", 1)[1])
        candidates.append(prediction_text.split(":", 1)[1])
    candidates.append(prediction_text)

    for candidate in candidates:
        matched = match_expected_label(candidate, allowed_answers)
        if matched is not None:
            return matched

    return None


def extract_prediction(
    text: str | None,
    allowed_answers: list[str] | None = None,
) -> str | None:
    """Extract a predicted label from raw new-ConSim output."""
    if text is None:
        return None
    if allowed_answers is not None:
        for candidate in ordered_prediction_candidates(text):
            matched = match_expected_label(candidate, allowed_answers)
            if matched is not None:
                return matched
        return None

    processed = text.strip().replace("\n", " ").split(" ")[-1]
    return normalize_label_text(processed)


def parse_old_consim_response(
    response: str,
    expected_answers: list[str],
    allowed_answers: list[str] | None = None,
    sample_ids: list[int] | None = None,
) -> list[str | None]:
    """
    Parse multi-prediction response from old ConSim (all-at-once mode).
    Expected format: "Sample_0: class_a\\nSample_1: class_b\\n..."
    """
    expected_length = len(expected_answers)
    labels_to_match = allowed_answers if allowed_answers is not None else expected_answers
    if not response:
        return [None] * expected_length

    lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
    if not lines:
        return [None] * expected_length

    predictions_by_sample_id = {}
    for line in lines:
        sample_match = SAMPLE_ID_RE.search(line)
        if sample_match is None:
            continue
        predictions_by_sample_id[int(sample_match.group(1))] = (
            extract_old_consim_prediction(line, labels_to_match)
        )

    if sample_ids and any(
        sample_id in predictions_by_sample_id for sample_id in sample_ids
    ):
        return [predictions_by_sample_id.get(sample_id) for sample_id in sample_ids]

    predictions = [
        extract_old_consim_prediction(line, labels_to_match) for line in lines
    ]
    if len(predictions) >= expected_length:
        return predictions[-expected_length:]

    predictions.extend([None] * (expected_length - len(predictions)))
    return predictions


def score_prompt_group_new(
    answers: list[str],
    expected_answers: list[str],
    allowed_answers: list[str] | None = None,
) -> tuple[int, int, int]:
    """Count correct and valid answers for one-answer-per-sample mode."""
    num_correct = 0
    num_valid = 0
    labels_to_match = allowed_answers if allowed_answers is not None else expected_answers
    for answer, expected in zip(answers, expected_answers, strict=True):
        predicted = extract_prediction(answer, labels_to_match)
        if predicted is None:
            continue
        num_valid += 1
        if prediction_matches(predicted, expected):
            num_correct += 1
    return num_correct, num_valid, len(expected_answers)


def score_prompt_group_old(
    answer: str,
    expected_answers: list[str],
    user_prompt: str | None = None,
    allowed_answers: list[str] | None = None,
) -> tuple[int, int, int]:
    """Count correct and valid answers for old-ConSim multi-line mode."""
    sample_ids = extract_sample_ids(user_prompt) if user_prompt is not None else None
    predictions = parse_old_consim_response(
        answer,
        expected_answers,
        allowed_answers=allowed_answers,
        sample_ids=sample_ids,
    )
    num_valid = sum(pred is not None for pred in predictions)
    num_correct = sum(
        int(pred is not None and prediction_matches(pred, expected))
        for pred, expected in zip(predictions, expected_answers, strict=True)
    )
    return num_correct, num_valid, len(expected_answers)


def compute_group_score(
    num_correct: int,
    num_valid: int,
    num_expected: int,
    coverage_ratio: float = 0.7,
) -> float:
    """Abstention-aware accuracy, or NaN when valid coverage is too low."""
    if num_expected <= 0:
        return float("nan")
    min_valid = math.ceil(coverage_ratio * num_expected)
    if num_valid < min_valid or num_valid == 0:
        return float("nan")
    return num_correct / num_valid


def write_generation_log(
    handle,
    prompt_group: dict,
    allowed_answers: list[str],
    raw_answers: list[str],
    specification: str,
    max_new_tokens: int,
    thinking: bool,
) -> None:
    """Append raw generation output for offline parser/debug reruns."""
    handle.write(
        json.dumps(
            {
                "key": prompt_group["key"],
                "system_prompt": prompt_group["system_prompt"],
                "user_prompts": prompt_group["user_prompts"],
                "expected_answers": prompt_group["expected_answers"],
                "allowed_answers": allowed_answers,
                "raw_answers": raw_answers,
                "specification": specification,
                "max_new_tokens": max_new_tokens,
                "thinking": thinking,
                "generated_at": datetime.now().isoformat(),
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    handle.flush()


def old_consim_max_new_tokens(
    expected_answers: list[str],
    user_max_new_tokens: int,
) -> int:
    """Generation budget for old-ConSim's all-at-once response.

    Old ConSim expects one ``Sample_N: <class_label>\\n`` line per evaluation
    sample. The default ``--max-new-tokens`` (32) is calibrated for new ConSim
    (single-token answer) and silently caps old-ConSim scores: the model gets
    truncated after ~4-5 predictions, capping scores at ~0.5 even when the
    judge would otherwise answer correctly.

    Rough per-line cost: ``"Sample_NN: "`` is ~5 tokens, the label is bounded
    by ``ceil(len(label) / 3)`` tokens for typical BPE tokenizers (one token
    per ~3-4 chars), and we add 1 for the newline. We then add a 128-token
    slack for any preamble the model might emit and respect the user override
    if it is larger.
    """
    longest_label_chars = max((len(answer) for answer in expected_answers), default=0)
    per_line_tokens = 5 + (longest_label_chars + 2) // 3 + 1  # prefix + label + newline
    required = per_line_tokens * len(expected_answers) + 128
    return max(user_max_new_tokens, required)


def main() -> None:
    args = parse_args()

    # Resolve short model name.
    args.judge_model = resolve_llm_model(args.judge_model)

    prompt_paths = resolve_prompt_paths(args.prompt_file)

    score_path = get_score_path(args.judge_model, args.thinking)
    treated_keys = load_treated_keys(score_path)

    # Determine which keys need scoring. Corrupted prompt groups are explicit
    # run markers and must never be forwarded to the judge.
    prompt_groups_by_path = {}
    for prompt_path in prompt_paths:
        prompt_groups_by_path[prompt_path] = list(iter_jsonl(prompt_path))
    prompt_groups = [
        prompt_group
        for path_prompt_groups in prompt_groups_by_path.values()
        for prompt_group in path_prompt_groups
    ]
    requested_keys = {pg["key"] for pg in prompt_groups if not pg.get("corrupted")}
    missing_keys = requested_keys - treated_keys

    if not missing_keys:
        print(
            f"Nothing to score. All {len(requested_keys)} keys already in {score_path}."
        )
        return

    print(f"Judge model:  {args.judge_model}")
    print(f"Prompt files: {len(prompt_paths)}")
    for prompt_path in prompt_paths:
        print(f"  - {prompt_path}")
    print(f"Thinking:     {args.thinking}")
    print(f"Score file:   {score_path}")
    generation_log_path = get_generation_log_path(args.judge_model, args.thinking)
    generation_log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generations:  {generation_log_path}")
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
    with open(score_path, "a", newline="") as handle, open(
        generation_log_path,
        "a",
    ) as generation_log_handle:
        writer = csv.writer(handle)

        queued_keys: set[str] = set()
        with tqdm(total=len(missing_keys), desc="Total scoring") as total_progress:
            for file_index, prompt_path in enumerate(prompt_paths, start=1):
                prompt_groups_to_score = []
                for prompt_group in prompt_groups_by_path[prompt_path]:
                    key = prompt_group["key"]
                    if (
                        key in missing_keys
                        and key not in queued_keys
                        and not prompt_group.get("corrupted")
                    ):
                        prompt_groups_to_score.append(prompt_group)
                        queued_keys.add(key)

                tqdm.write(
                    f"[{file_index}/{len(prompt_paths)}] {prompt_path}: "
                    f"{len(prompt_groups_to_score)} prompt groups to score"
                )
                if not prompt_groups_to_score:
                    continue

                def write_new_consim_batch(new_prompt_groups: list[dict]) -> None:
                    if not new_prompt_groups:
                        return

                    full_prompts = []
                    group_slices = []
                    for prompt_group in new_prompt_groups:
                        start = len(full_prompts)
                        full_prompts.extend(
                            render_prompt(
                                tokenizer,
                                prompt_group["system_prompt"],
                                user_prompt,
                                args.thinking,
                            )
                            for user_prompt in prompt_group["user_prompts"]
                        )
                        group_slices.append((prompt_group, start, len(full_prompts)))

                    answers = generate_completions(
                        model=model,
                        tokenizer=tokenizer,
                        full_prompts=full_prompts,
                        max_new_tokens=args.max_new_tokens,
                        batch_size=args.batch_size,
                        progress_desc="Generation batches",
                    )

                    for prompt_group, start, end in group_slices:
                        group_answers = answers[start:end]
                        allowed_answers = allowed_answers_for_prompt_group(prompt_group)
                        write_generation_log(
                            generation_log_handle,
                            prompt_group,
                            allowed_answers,
                            group_answers,
                            "new_consim",
                            args.max_new_tokens,
                            args.thinking,
                        )
                        num_correct, num_valid, num_expected = score_prompt_group_new(
                            group_answers,
                            prompt_group["expected_answers"],
                            allowed_answers,
                        )
                        accuracy = compute_group_score(
                            num_correct,
                            num_valid,
                            num_expected,
                        )
                        writer.writerow(
                            ast.literal_eval(prompt_group["key"])
                            + (
                                datetime.now(),
                                accuracy,
                                num_correct,
                                num_valid,
                                num_expected,
                            )
                        )
                        handle.flush()
                        total_progress.update(1)

                new_consim_batch = []
                for prompt_group in prompt_groups_to_score:
                    if not is_old_consim_prompt_group(prompt_group):
                        new_consim_batch.append(prompt_group)
                        continue

                    write_new_consim_batch(new_consim_batch)
                    new_consim_batch.clear()

                    user_prompts = prompt_group["user_prompts"]
                    expected_answers = prompt_group["expected_answers"]
                    allowed_answers = allowed_answers_for_prompt_group(prompt_group)

                    # Old ConSim must emit one "Sample_N: class" line per evaluation
                    # sample in a single response, so the default --max-new-tokens
                    # (calibrated for new ConSim's one-token answer) caps scores at
                    # ~0.5. Auto-scale the budget for old-ConSim prompts.
                    max_new_tokens = old_consim_max_new_tokens(
                        expected_answers,
                        args.max_new_tokens,
                    )

                    answers = generate_answers(
                        model=model,
                        tokenizer=tokenizer,
                        system_prompt=prompt_group["system_prompt"],
                        user_prompts=user_prompts,
                        thinking=args.thinking,
                        max_new_tokens=max_new_tokens,
                        batch_size=args.batch_size,
                        progress_desc="Generation batches",
                    )

                    write_generation_log(
                        generation_log_handle,
                        prompt_group,
                        allowed_answers,
                        answers,
                        "old_consim",
                        max_new_tokens,
                        args.thinking,
                    )

                    num_correct, num_valid, num_expected = score_prompt_group_old(
                        answers[0], expected_answers, user_prompts[0], allowed_answers
                    )
                    accuracy = compute_group_score(
                        num_correct,
                        num_valid,
                        num_expected,
                    )

                    # Append score row.
                    writer.writerow(
                        ast.literal_eval(prompt_group["key"])
                        + (
                            datetime.now(),
                            accuracy,
                            num_correct,
                            num_valid,
                            num_expected,
                        )
                    )
                    handle.flush()
                    total_progress.update(1)

                write_new_consim_batch(new_consim_batch)

    print(
        f"\nScored {len(missing_keys)} keys with {args.judge_model} "
        f"(thinking={args.thinking}). Results appended to {score_path}."
    )


if __name__ == "__main__":
    main()
