"""Generate raw judge outputs for ConSim prompt groups (no scoring).

Reads prompt JSONL, skips keys already present in ``data/generations/{judge}.jsonl``,
runs inference, and appends one JSONL row per prompt group. Mirrors
``scripts/llm_scoring.py``'s generation half, minus the parsing/scoring step.

Usage::

    .venv-vllm/bin/python scripts/generate_only.py qwen3.5-9b
    .venv-vllm/bin/python scripts/generate_only.py qwen3.5-9b data/prompts/GE_concepts.jsonl
    .venv/bin/python scripts/generate_only.py qwen3.5-9b --backend hf
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.llm_scoring import (
    allowed_answers_for_prompt_group,
    default_batch_size,
    generate_answers,
    generate_completions,
    get_generation_log_path,
    is_old_consim_prompt_group,
    load_vllm_engine,
    old_consim_max_new_tokens,
    render_prompt,
    resolve_prompt_paths,
    write_generation_log,
)
from utils.data import LLM_MODELS, iter_jsonl, resolve_llm_model


def parse_args() -> argparse.Namespace:
    default_backend = "vllm" if importlib.util.find_spec("vllm") is not None else "hf"
    parser = argparse.ArgumentParser(
        description="Generate raw judge outputs (no scoring).",
    )
    parser.add_argument(
        "judge_model",
        help=f"LLM judge model. Short names: {', '.join(LLM_MODELS.keys())}.",
    )
    parser.add_argument(
        "prompt_file",
        type=Path,
        nargs="*",
        default=None,
        help="Prompt JSONL file(s). Defaults to all data/prompts/*.jsonl.",
    )
    parser.add_argument("--backend", choices=("hf", "vllm"), default=default_backend)
    parser.add_argument("--thinking", action="store_true", default=False)
    parser.add_argument("--max-new-tokens", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.backend == "vllm" and importlib.util.find_spec("vllm") is None:
        parser.error("--backend vllm requested, but vllm is not installed")
    return args


def load_existing_keys(generation_log_path: Path) -> set[str]:
    """Return the set of raw_keys already present in the generation log."""
    if not generation_log_path.exists():
        return set()
    keys: set[str] = set()
    with generation_log_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("key")
            if isinstance(key, str):
                keys.add(key)
    return keys


def main() -> None:
    args = parse_args()
    args.judge_model = resolve_llm_model(args.judge_model)
    if args.batch_size is None:
        args.batch_size = default_batch_size(args.judge_model, args.backend)

    prompt_paths = resolve_prompt_paths(args.prompt_file)
    generation_log_path = get_generation_log_path(args.judge_model, args.thinking)
    generation_log_path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = load_existing_keys(generation_log_path)

    prompt_groups_by_path = {
        path: list(iter_jsonl(path)) for path in prompt_paths
    }
    prompt_groups = [
        pg for path_groups in prompt_groups_by_path.values() for pg in path_groups
    ]
    requested_keys = {pg["key"] for pg in prompt_groups if not pg.get("corrupted")}
    missing_keys = requested_keys - existing_keys

    if not missing_keys:
        print(
            f"Nothing to generate. All {len(requested_keys)} keys already in "
            f"{generation_log_path}."
        )
        return

    print(f"Judge model:  {args.judge_model}")
    print(f"Prompt files: {len(prompt_paths)}")
    for path in prompt_paths:
        print(f"  - {path}")
    print(f"Backend:      {args.backend}")
    print(f"Thinking:     {args.thinking}")
    print(f"Batch size:   {args.batch_size} ({args.backend})")
    print(f"Generations:  {generation_log_path}")
    print(f"To generate:  {len(missing_keys)} / {len(requested_keys)} keys\n")

    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.backend == "vllm":
        model = load_vllm_engine(args.judge_model, args.batch_size)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.judge_model,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
        )
        if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
            try:
                model.config.pad_token_id = tokenizer.pad_token_id
            except AttributeError:
                pass
        if (
            model.generation_config.pad_token_id is None
            and tokenizer.pad_token_id is not None
        ):
            model.generation_config.pad_token_id = tokenizer.pad_token_id

    queued_keys: set[str] = set()
    with open(generation_log_path, "a") as gen_handle, tqdm(
        total=len(missing_keys), desc="Total generation"
    ) as total_progress:
        for file_index, path in enumerate(prompt_paths, start=1):
            groups_to_do = [
                pg for pg in prompt_groups_by_path[path]
                if pg["key"] in missing_keys
                and pg["key"] not in queued_keys
                and not pg.get("corrupted")
            ]
            for pg in groups_to_do:
                queued_keys.add(pg["key"])
            tqdm.write(
                f"[{file_index}/{len(prompt_paths)}] {path}: "
                f"{len(groups_to_do)} prompt groups to generate"
            )
            if not groups_to_do:
                continue

            def write_new_consim_chunk(new_groups: list[dict]) -> None:
                full_prompts: list[str] = []
                group_slices = []
                for pg in new_groups:
                    start = len(full_prompts)
                    full_prompts.extend(
                        render_prompt(
                            tokenizer, pg["system_prompt"], user_prompt, args.thinking
                        )
                        for user_prompt in pg["user_prompts"]
                    )
                    group_slices.append((pg, start, len(full_prompts)))
                answers = generate_completions(
                    model=model,
                    tokenizer=tokenizer,
                    backend=args.backend,
                    full_prompts=full_prompts,
                    max_new_tokens=args.max_new_tokens,
                    batch_size=args.batch_size,
                )
                for pg, start, end in group_slices:
                    write_generation_log(
                        gen_handle,
                        pg,
                        allowed_answers_for_prompt_group(pg),
                        answers[start:end],
                        ast.literal_eval(pg["key"])[-1],
                        args.max_new_tokens,
                        args.thinking,
                    )
                    total_progress.update(1)

            def write_new_consim_batch(new_groups: list[dict]) -> None:
                if not new_groups:
                    return
                chunk: list[dict] = []
                chunk_prompt_count = 0
                for pg in new_groups:
                    group_prompt_count = len(pg["user_prompts"])
                    if chunk and chunk_prompt_count + group_prompt_count > args.batch_size:
                        write_new_consim_chunk(chunk)
                        chunk = []
                        chunk_prompt_count = 0
                    chunk.append(pg)
                    chunk_prompt_count += group_prompt_count
                write_new_consim_chunk(chunk)

            new_consim_batch: list[dict] = []
            for pg in groups_to_do:
                if not is_old_consim_prompt_group(pg):
                    new_consim_batch.append(pg)
                    continue

                write_new_consim_batch(new_consim_batch)
                new_consim_batch.clear()

                max_new_tokens = old_consim_max_new_tokens(
                    pg["expected_answers"], args.max_new_tokens
                )
                answers = generate_answers(
                    model=model,
                    tokenizer=tokenizer,
                    system_prompt=pg["system_prompt"],
                    user_prompts=pg["user_prompts"],
                    thinking=args.thinking,
                    max_new_tokens=max_new_tokens,
                    batch_size=args.batch_size,
                    backend=args.backend,
                )
                write_generation_log(
                    gen_handle,
                    pg,
                    allowed_answers_for_prompt_group(pg),
                    answers,
                    ast.literal_eval(pg["key"])[-1],
                    max_new_tokens,
                    args.thinking,
                )
                total_progress.update(1)

            write_new_consim_batch(new_consim_batch)

    print(
        f"\nGenerated {len(missing_keys)} keys with {args.judge_model} "
        f"(thinking={args.thinking}). Appended to {generation_log_path}."
    )


if __name__ == "__main__":
    main()
