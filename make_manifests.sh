#!/usr/bin/env bash
# Generate a manifest file with one Python command per parameter combination.
#
# Usage:
#   ./sequence.sh <output.tsv> <script.py> [args...]
#
# Example:
#   ./sequence.sh manifests/make_prompts.tsv \
#     scripts/make_prompts.py concepts RT,AG,IMDB,E seminmf,ica \
#     --interpretation topk

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <output.tsv> <script.py> [args...]" >&2
    exit 1
fi

output="$1"
script="$2"
shift 2

types=()
keys=()
values=()

while [[ $# -gt 0 ]]; do
    if [[ "$1" == --* ]]; then
        key="$1"
        shift
        if [[ $# -gt 0 && "$1" != --* ]]; then
            types+=("option")
            keys+=("$key")
            values+=("$1")
            shift
        else
            types+=("flag")
            keys+=("$key")
            values+=("")
        fi
    else
        types+=("positional")
        keys+=("")
        values+=("$1")
        shift
    fi
done

combos=("")

for i in "${!types[@]}"; do
    type="${types[$i]}"
    key="${keys[$i]}"
    val="${values[$i]}"
    new_combos=()

    if [[ "$type" == "flag" ]]; then
        for combo in "${combos[@]}"; do
            new_combos+=("${combo} ${key}")
        done
    elif [[ "$val" == *,* ]]; then
        IFS=',' read -ra split_vals <<< "$val"
        for combo in "${combos[@]}"; do
            for v in "${split_vals[@]}"; do
                if [[ "$type" == "option" ]]; then
                    new_combos+=("${combo} ${key} ${v}")
                else
                    new_combos+=("${combo} ${v}")
                fi
            done
        done
    else
        for combo in "${combos[@]}"; do
            if [[ "$type" == "option" ]]; then
                new_combos+=("${combo} ${key} ${val}")
            else
                new_combos+=("${combo} ${val}")
            fi
        done
    fi

    combos=("${new_combos[@]}")
done

mkdir -p "$(dirname "$output")"
: > "$output"

for combo in "${combos[@]}"; do
    combo="${combo# }"
    echo "python ${script} ${combo}" >> "$output"
done

echo "Wrote ${#combos[@]} commands to ${output}"
