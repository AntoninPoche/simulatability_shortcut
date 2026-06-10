#!/usr/bin/env bash
# Run a python script for every combination from comma-separated argument values.
#
# Usage:
#   ./sequence.sh scripts/build_concepts.py GE,HE,BIOS --method ica,kmeans --interpretation topk
#
# The above expands to 6 sequential runs (cartesian product of datasets × methods):
#   python scripts/build_concepts.py GE    --method ica    --interpretation topk
#   python scripts/build_concepts.py GE    --method kmeans --interpretation topk
#   python scripts/build_concepts.py HE    --method ica    --interpretation topk
#   python scripts/build_concepts.py HE    --method kmeans --interpretation topk
#   python scripts/build_concepts.py BIOS  --method ica    --interpretation topk
#   python scripts/build_concepts.py BIOS  --method kmeans --interpretation topk
#
# Flags (store_true arguments like --activations-difference) are forwarded as-is.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <script.py> [args...]" >&2
    exit 1
fi

script="$1"
shift

# ── Parse arguments into parallel arrays ──────────────────────────────
#   types[i]:  "flag" | "option" | "positional"
#   keys[i]:   "--key" for flag/option, empty for positional
#   values[i]: "val1,val2" for option/positional, empty for flag
types=()
keys=()
values=()

while [[ $# -gt 0 ]]; do
    if [[ "$1" == --* ]]; then
        key="$1"; shift
        if [[ $# -gt 0 && "$1" != --* ]]; then
            types+=("option"); keys+=("$key"); values+=("$1"); shift
        else
            types+=("flag"); keys+=("$key"); values+=("")
        fi
    else
        types+=("positional"); keys+=(""); values+=("$1"); shift
    fi
done

# ── Build cartesian product of all comma-separated values ─────────────
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

# ── Run each combination sequentially ─────────────────────────────────
total=${#combos[@]}
run=0
failed=0

for combo in "${combos[@]}"; do
    run=$((run + 1))
    # Trim leading space that accumulates from concatenation.
    combo="${combo# }"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "  Run ${run}/${total}: python ${script} ${combo}"
    echo "════════════════════════════════════════════════════════════"
    # Word-splitting on $combo is intentional here — each token is a
    # separate argument to the python script.
    # shellcheck disable=SC2086
    if python "$script" ${combo}; then
        echo "  ✓ Run ${run}/${total} succeeded."
    else
        echo "  ✗ Run ${run}/${total} failed (exit code $?)." >&2
        failed=$((failed + 1))
    fi
done

echo ""
echo "Done: $((total - failed))/${total} succeeded."
[[ $failed -eq 0 ]]
