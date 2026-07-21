# Simulatability Shortcut

This repository studies a limitation of automated simulatability evaluation for NLP explanations: LLM judges can obtain similar scores from explanation methods and matched no-explanation baselines. It includes the processed prediction, score, and sample-metadata artifacts used by the paper analyses.

## Quick Start

The paper analyses run on CPU from the included CSV artifacts. Install the analysis dependencies, then open the notebooks from the repository root:

```bash
uv sync
jupyter notebook notebooks/4_compare_consim.ipynb
```

Run notebooks 4, 5, and 6 in order. Figure export locations are repository-local under `outputs/figures/`; notebook 6 disables exports by default.

## Included Artifacts

`data/` contains four judge-specific prediction/score pairs:

| Judge | Predictions | Scores |
| --- | --- | --- |
| Qwen 3.5-9B | `predictions_Qwen_Qwen3.5-9B.csv` | `consim_Qwen_Qwen3.5-9B.csv` |
| Phi-4 | `predictions_microsoft_phi-4.csv` | `consim_microsoft_phi-4.csv` |
| Gemma 4-12B | `predictions_google_gemma-4-12B-it.csv` | `consim_google_gemma-4-12B-it.csv` |
| Llama 3.1-8B | `predictions_meta-llama_Llama-3.1-8B-Instruct.csv` | `consim_meta-llama_Llama-3.1-8B-Instruct.csv` |

`sample_metadata.csv` contains the evaluation-sample keys, gold labels, and task-model predictions needed by `notebooks/6_judge_consistency.ipynb`. It deliberately excludes source texts and model caches.

Prediction files contain one row per prompt group and `pred_0` through `pred_39` columns. Score files contain the prompt key, accuracy, and valid/correct/expected counts. A score is reported only when at least 70% of responses in its group are valid.

## Rebuild Scores

Recompute scores from the included predictions without a GPU:

```bash
python scripts/build_scores.py --output-dir outputs/scores
```

The command refuses to overwrite existing files unless `--overwrite` is supplied.

## Repository Layout

```text
data/       Published prediction, score, and sample-metadata CSV artifacts
notebooks/  Analyses for ConSim variants, explanation families, and judges
scripts/    Prompt generation, LLM generation, parsing, and score construction
utils/      Shared experiment, parsing, analysis, and plotting code
```

## Full Reproduction

Full prompt generation and LLM scoring require local model access and a CUDA-capable environment. Install the optional generation dependencies:

```bash
uv sync --extra generation
```

The workflow is:

```text
make_prompts.py / make_prompts_old_consim.py / make_prompts_simulator_consim.py
    -> generate_only.py
    -> parse_generations.py
    -> build_scores.py
```

The simulator-framed condition is generated with:

```bash
python scripts/make_prompts_simulator_consim.py GE seminmf --interpretation topk
```

Prompt generation builds or reads task-model and explanation caches under `data/`; those expensive caches and raw generations are not included in the publication artifact.

## Dependencies

The base environment is intentionally CPU-only and contains notebook dependencies. The generation extra pins the experiment-critical packages:

- `interpreto==0.5.0`
- `nnsight==0.7.0`

`vllm` is optional and can be installed separately when that backend is desired. The exact GPU stack depends on the target CUDA runtime.

## Notes

`old_consim`, `new_consim`, and `simulator_consim` are distinct experimental prompt specifications. Their prompt text, keys, parsing rules, and sample-selection caches are experiment-defining and should not be changed when reproducing the included results.
