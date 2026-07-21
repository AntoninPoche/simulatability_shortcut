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

The commands below use all 50 seeds and 40 selected samples per seed, which are the prompt scripts' defaults. `sequence.sh` expands comma-separated datasets, methods, or judges into sequential runs. Prompt generation is resumable, so repeated commands only add missing prompt groups.

### 1. ConSim Replication And Protocol Selection

The first command reconstructs the original-format concept experiment on the four datasets closest to the original study. It supports the qualitative replication in Section `Replication and extension of ConSim` and the reconstructed pairwise comparison in Fig. `old_pairwise`.

```bash
# old_consim replication: BIOS, Emotion, IMDB, Rotten Tomatoes; TopK labels.
REPLICATION_DATASETS="BIOS,E,IMDB,RT"
CONCEPT_METHODS="seminmf,ica,pca,svd,vanilla_sae,neurons"
./sequence.sh scripts/make_prompts_old_consim.py \
  "$REPLICATION_DATASETS" "$CONCEPT_METHODS" --interpretation topk
```

The next commands evaluate the prompt-format and concept-labeling alternatives described in Section `Reproduction and protocol selection` and Appendix `Protocol comparison`. They produce the inputs for Figs. `new_old_diff`, `llm_topk_diff`, and `sim_new_diff`.

```bash
# Compare old_consim with new_consim on the full six-dataset protocol grid.
# Both TopK and LLM-generated concept labels are needed for Fig. new_old_diff.
ALL_DATASETS="AG,BIOS,E,GE,IMDB,RT"
CONCEPT_METHODS="seminmf,ica,pca,svd,vanilla_sae,neurons"
./sequence.sh scripts/make_prompts_old_consim.py \
  "$ALL_DATASETS" "$CONCEPT_METHODS" --interpretation topk,llm
./sequence.sh scripts/make_prompts.py concepts \
  "$ALL_DATASETS" "$CONCEPT_METHODS" --interpretation topk,llm

# Test explicit classifier-simulation framing against new_consim (Fig. sim_new_diff).
# This treatment is evaluated with TopK labels only.
./sequence.sh scripts/make_prompts_simulator_consim.py \
  "$ALL_DATASETS" "$CONCEPT_METHODS" --interpretation topk

# Leakage control for the anonymized setting (Sec. Anonymization creates a second shortcut;
# Figs. classes_as_concepts and new_topk_anon_pairwise).
./sequence.sh scripts/make_prompts.py concepts "$ALL_DATASETS" classes
```

### 2. Qwen Method-Selection Grid

Run the complete non-anonymized grid with Qwen 3.5-9B to select one representative per explanation family. This corresponds to Appendix `Explanation-method rankings with Qwen3.5-9B`: Vanilla SAE for concepts, LIME for attributions, and Qwen 3.5-2B for rationales.

```bash
# Concepts: TopK is the selected interpretation after Fig. llm_topk_diff.
# classes is retained as the anonymized leakage-control baseline.
ALL_DATASETS="AG,BIOS,E,GE,IMDB,RT"
CONCEPT_METHODS="seminmf,ica,pca,svd,vanilla_sae,neurons,classes"
./sequence.sh scripts/make_prompts.py concepts \
  "$ALL_DATASETS" "$CONCEPT_METHODS" --interpretation topk

# Attributions: evaluate all candidates before selecting LIME.
ATTRIBUTION_METHODS="saliency,integrated_gradients,smooth_grad,square_grad,var_grad,gradient_shap,lime,kernel_shap,occlusion,sobol"
./sequence.sh scripts/make_prompts.py attributions \
  "$ALL_DATASETS" "$ATTRIBUTION_METHODS"

# Rationales: compare the two rationale generators before selecting Qwen 3.5-2B.
./sequence.sh scripts/make_prompts.py rationales "$ALL_DATASETS" \
  --llm-model qwen3.5-2b,llama3.2-3b

# Score the complete Qwen grid, then parse raw responses and reconstruct scores.
# These scores feed the family rankings and Fig. families_prompt_type_diff_qwen.
python scripts/generate_only.py qwen3.5-9b data/prompts/*.jsonl
python scripts/parse_generations.py --overwrite
python scripts/build_scores.py --overwrite
```

### 3. Representative Methods Across User-LLMs

Select the representatives from the Qwen rankings, then score only those prompt groups with new user-LLMs. This produces the comparisons in Section `Matched explanation--baseline differences across user-LLMs`, Fig. `families_differences_by_judge`, and the prediction-agreement analyses.

```bash
# Select Vanilla SAE + TopK, LIME, Qwen 3.5-2B rationales, and matched baselines.
python scripts/extract_best_prompts.py --overwrite

# Each judge processes every selected prompt file in one model session.
./sequence.sh scripts/generate_only.py \
  llama3.1-8b,gemma4-12b,phi4 data/best_prompts/*.jsonl

# Rebuild predictions and scores for all available judge logs.
python scripts/parse_generations.py --overwrite
python scripts/build_scores.py --overwrite
```

The complete workflow is:

```text
make_prompts.py / make_prompts_old_consim.py / make_prompts_simulator_consim.py
    -> generate_only.py
    -> parse_generations.py
    -> build_scores.py
```

Prompt generation builds or reads task-model and explanation caches under `data/`; those expensive caches and raw generations are not included in the publication artifact.

## Dependencies

The base environment is intentionally CPU-only and contains notebook dependencies. The generation extra pins the experiment-critical packages:

- `interpreto==0.5.0`
- `nnsight==0.7.0`

`vllm` is optional and can be installed separately when that backend is desired. The exact GPU stack depends on the target CUDA runtime.

## Notes

`old_consim`, `new_consim`, and `simulator_consim` are distinct experimental prompt specifications. Their prompt text, keys, parsing rules, and sample-selection caches are experiment-defining and should not be changed when reproducing the included results.

## Limitations

The paper identifies two limitations of the study itself:

- It does not establish whether the findings generalize to larger or stronger user-LLMs. The experiments omit reasoning models, models above 15B parameters, and closed-source models because of compute constraints.
- The evaluated classification tasks are likely familiar to user-LLMs or semantically easy enough to solve from pretrained knowledge. The observed task-solving shortcut may therefore differ on novel or substantially harder tasks.
