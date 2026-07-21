# Maintainer Guide

## Scope

This is a research artifact for studying automated simulatability of NLP explanations. The published analysis uses the versionless CSV artifacts in `data/` and notebooks 4–6.

## Layout

```text
data/
  predictions_{judge}.csv       # Parsed judge outputs, one row per prompt group
  consim_{judge}.csv            # Scores rebuilt from predictions
  sample_metadata.csv           # Gold/task-model labels for notebook 6
scripts/
  make_prompts.py               # New one-sample-per-prompt experiments
  make_prompts_old_consim.py    # All-evaluation-samples-in-one-prompt condition
  make_prompts_simulator_consim.py
  generate_only.py
  parse_generations.py
  build_scores.py
  export_sample_metadata.py
utils/
  consim.py, old_consim.py, simulator_consim.py
  ratsim.py, attrsim.py
  scoring.py                    # Lightweight key, parser, and score helpers
  predictions.py                # Prediction and sample-metadata helpers
notebooks/
  4_compare_consim.ipynb
  5_compare_families.ipynb
  6_judge_consistency.ipynb
```

## Artifact Pipeline

```text
prompt JSONL -> generation JSONL -> predictions CSV -> score CSV
```

`scripts/build_scores.py` is the canonical score reconstruction step. It consumes `predictions_{judge}.csv` and writes `consim_{judge}.csv`.

`scripts/export_sample_metadata.py` derives the compact `data/sample_metadata.csv` artifact from full local-elements caches. The published metadata contains only identifiers, gold labels, and task-model predictions required for agreement analysis.

## Experiment Invariants

- Prompt keys have the nine fields in `utils.scoring.KEY_FIELDS`.
- B1/B2 baseline prompts must stay byte-identical across concept, rationale, and attribution families for a common dataset, seed, and class subset.
- `old_consim`, `new_consim`, and `simulator_consim` are separate treatments. Do not alter their prompt wording or response parsing without regenerating artifacts.
- Sample selection is deterministic but cached. Cache paths and payload schemas are part of experiment traceability.
- Score validity requires at least 70% parseable responses within a prompt group.

## Dependencies

Base dependencies support CPU notebook analysis. Full generation additionally requires `interpreto==0.5.0`, `nnsight==0.7.0`, Torch, Transformers, datasets, and optionally vLLM.

## Validation

Use the smallest relevant command. Offline checks do not require GPUs:

```bash
python scripts/build_scores.py --output-dir outputs/scores
python -m compileall scripts utils
```

Do not change published CSV rows, prompt identifiers, prompt text, score semantics, or cache layouts as incidental cleanup.
