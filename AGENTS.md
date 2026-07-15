# AGENTS.md

## Critical Rules

- Make the smallest change that solves the task.
- Preserve experiment traceability. Do not rename files, result columns, prompt identifiers, cache layouts, or artifact paths without explicit request.
- Treat `data/` as expensive. Never delete or overwrite artifacts unless asked.
- Do not "productionize" the repo. Keep things script-driven and explicit.
- Keep this file up to date when adding/removing/renaming files or changing the workflow.
- Treat user edits as intentional. Do not undo, restore, or overwrite them unless explicitly asked; if a user edit looks wrong or conflicts with the task, ask before changing it.

## Purpose

Research experiments showing weaknesses of automated simulatability metrics in NLP. The paper argues that simulatability cannot differentiate between explanation methods and no-explanation baselines. Experiments span concept-based explanations, rationales, and attributions across multiple datasets and LLM judges.

Forked from [contrastive_concepts](https://github.com/AntoninPoche/contrastive_concepts).

## Repository Structure

```
scripts/
  make_prompts.py             # Generate new-ConSim prompt JSONL (CLI, argparse); builds concept artifacts if missing
  make_prompts_consim_v2.py   # Generate simulator-framed ConSim concept prompt JSONL (CLI)
  make_prompts_old_consim.py  # Generate old-ConSim prompt JSONL for comparison (CLI)
  extract_best_prompts.py  # Extract hard-coded best-method prompt subsets into data/best_prompts/
  split_prompt_file.py  # Split large prompt JSONL files into derived per-field files without touching originals
  llm_scoring.py        # Score prompts with a local HF LLM judge (CLI)
  generate_only.py      # Generation-only counterpart to llm_scoring.py; writes data/generations/{judge}.jsonl (CLI)
  parse_generations.py  # Parse data/generations/*.jsonl into wide sample-level predictions CSVs (CLI)
  build_scores_v3.py    # Recompute V3 score CSVs from predictions_*.csv (CLI)
  state.py              # Summarize manifest, prompt JSONL, corrupted markers, and v2 score coverage for one judge model
  drop_prompt_rows.py   # Drop prompt JSONL rows by prompt-key field filters (CLI, writes .bak)
  drop_score_rows.py    # Drop rows from a score CSV by column=value filters (CLI, pandas, writes .bak)
  drop_corrupted_prompt_rows.py # Remove corrupted prompt JSONL marker rows (CLI, writes .bak)
  canonicalize_baselines.py # Canonicalize baseline keys/score rows to method=baseline, nb_concepts=None, interpretation=None

utils/                        # Shared library package
  __init__.py
  data.py                     # Dataset/model registries, split loading, caching helpers
  registries.py               # Lightweight method/prompt registries for fast early-exit checks
  concepts.py                 # Concept model loading/fitting, interpretations, importances, build pipeline
  consim.py                   # New ConSim prompt builder (one prompt per eval sample)
  consim_v2.py                # Simulator-framed ConSim prompt builder; inherits current ConSim and changes only prompting
  old_consim.py               # Old ConSim prompt builder (all eval samples at once)
  simulatability.py           # Base AutomatedSimulatability class (local, not from interpreto)
  rationales.py               # Rationale generation from local LLMs (Qwen)
  ratsim.py                   # Rationale-based prompt construction
  analysis.py                 # Shared notebook dataframe helpers for filtering, bucketed summaries, family inference, and best-config selection
  plot.py                     # Reusable plot helpers for paper figures (violins, bar plots, pairwise matrices)
  predictions.py              # Shared helpers to parse generation JSONL rows to global class ids and load local_elements gold labels

sequence.sh                   # Cartesian-product script runner (see Commands below)
manifests/                    # Cluster command manifests; old_consim.tsv covers README Stage 1 old-ConSim generation
notebooks/
  4_compare_consim.ipynb      # V3 old/new/simulator comparisons, interpretation checks, and concept pairwise matrices
  5_compare_families.ipynb    # V3 concept/rationale/attribution rankings and selected-method comparisons across judges
  6_judge_consistency.ipynb   # Prompt-prediction correlation matrices (B1/B2/C1/C2/C3/A/R + gold + task-model) from data/predictions_*.csv
  old/                        # Archived copies of notebooks before the next-paper rewrite; keep for reference
  generation_concept_tutorial.ipynb  # Interpreto concept tutorial
LaTeX-Simulatability-Shortcut/  # ACL paper sources (separate git subrepo)
data/                         # Gitignored artifacts: activation/prediction caches, prompts, scores
  best_prompts/               # Planned: prompt JSONL subset for best methods only, preserving original prompt keys/schema
  generations/                # Raw per-judge generation JSONL logs (also written by llm_scoring.py)
  predictions_{judge}.csv     # Wide sample-level predictions from parse_generations.py; one row per prompt key, pred_0..pred_39 columns
  consim_{judge}_v3.csv       # Recomputed scores from predictions_*.csv via build_scores_v3.py (same schema as V2)
```

## Import Architecture

- **Local `utils/` package**: `simulatability.py`, `consim.py`, `consim_v2.py`, `old_consim.py`, `ratsim.py` — these are the canonical implementations, not imported from interpreto.
- **Lightweight key registries**: `utils/registries.py` stores method names and prompt abbreviations used for argument validation and early-exit key computation without importing torch/interpreto/transformers.
- **From `interpreto`**: concept extraction algorithms (`SemiNMFConcepts`, `ICAConcepts`, etc.), `SplitterForClassification`, and concept interpretation implementations resolved from keys such as `topk`/`llm`. Used only by `utils/concepts.py` for the heavy ML components (loaded lazily when concept artifacts need building).
- **Prompt scripts (`make_prompts.py`, `make_prompts_consim_v2.py`, `make_prompts_old_consim.py`) import from `utils/concepts.py`** — which handles both loading cached artifacts and building them (with interpreto) when missing.
- All scripts add the repo root to `sys.path` so `from utils.* import ...` works when running `python scripts/foo.py`.

## Commands

**Install dependencies** (venv exists but packages may need installing):

```bash
.venv/bin/pip install -r requirements.txt
```

**Generate prompts — new ConSim** (builds concept artifacts automatically if missing):

```bash
python scripts/make_prompts.py concepts GE seminmf
python scripts/make_prompts.py concepts BIOS ica --nb-concepts-ratio 2
python scripts/make_prompts.py concepts RT vanilla_sae --interpretation topk
python scripts/make_prompts.py concepts RT neurons --interpretation topk
python scripts/make_prompts.py concepts RT classes --interpretation topk  # interpretation is ignored; key uses None
python scripts/make_prompts.py rationales BIOS
python scripts/make_prompts.py rationales BIOS --llm-model qwen3.5-9b
python scripts/make_prompts.py attributions GE saliency
```

**Generate prompts — simulator-framed ConSim** (concepts only; writes `specification=simulator_consim` to `{dataset}_concepts.jsonl`):

```bash
python scripts/make_prompts_consim_v2.py GE seminmf
python scripts/make_prompts_consim_v2.py BIOS ica --interpretation topk
```

**Generate prompts — old ConSim** (for new-vs-old comparison, uses same cached samples):

```bash
python scripts/make_prompts_old_consim.py GE seminmf
```

**Score prompts with local LLM**:

```bash
.venv-vllm/bin/python scripts/llm_scoring.py qwen3.5-9b data/prompts/GE_concepts.jsonl
.venv-vllm/bin/python scripts/llm_scoring.py qwen3.5-9b  # scores all data/prompts/*.jsonl files with one model load
.venv/bin/python scripts/llm_scoring.py qwen3.5-9b --backend hf  # fallback without vLLM
```

**Generate raw judge outputs only** (no parsing/scoring; appends to `data/generations/{judge}.jsonl`):

```bash
.venv-vllm/bin/python scripts/generate_only.py qwen3.5-9b
.venv-vllm/bin/python scripts/generate_only.py qwen3.5-9b data/prompts/GE_concepts.jsonl
.venv/bin/python scripts/generate_only.py qwen3.5-9b --backend hf
```

**Parse existing generations into wide sample-level predictions CSVs**:

```bash
python scripts/parse_generations.py                        # all data/generations/*.jsonl -> data/predictions_*.csv
python scripts/parse_generations.py data/generations/Qwen_Qwen3.5-9B.jsonl --overwrite
python scripts/parse_generations.py --no-cache-n           # skip local_elements matching (no cache_n column)
```

**Recompute V3 score CSVs from predictions CSVs**:

```bash
python scripts/build_scores_v3.py                          # all data/predictions_*.csv -> data/consim_*_v3.csv
python scripts/build_scores_v3.py data/predictions_microsoft_phi-4.csv --overwrite
```

**Split large prompt files** (creates derived JSONL files; originals are untouched):

```bash
python scripts/split_prompt_file.py data/prompts/RT_concepts.jsonl  # split by method into data/prompt_splits/
python scripts/split_prompt_file.py data/prompts/RT_concepts.jsonl --by specification
.venv-vllm/bin/python scripts/llm_scoring.py qwen3.5-9b data/prompt_splits/RT_concepts__method-seminmf.jsonl
.venv-vllm/bin/python scripts/llm_scoring.py qwen3.5-9b data/prompt_splits/RT_concepts__method-seminmf.jsonl data/prompt_splits/RT_concepts__method-ica.jsonl
```

**Extract best prompts** (creates derived JSONL files; originals are untouched):

```bash
python scripts/extract_best_prompts.py --dry-run
python scripts/extract_best_prompts.py
python scripts/llm_scoring.py qwen3.5-9b data/best_prompts/RT_concepts.jsonl
```

**Drop rows from a score CSV** (to force re-scoring after regenerating prompts):

```bash
# Drop every RT row (both specifications) before re-running llm_scoring.py.
python scripts/drop_score_rows.py data/consim_Qwen_Qwen3.5-9B.csv dataset=RT
python scripts/drop_score_rows.py data/consim_meta-llama_Llama-3.2-3B-Instruct.csv dataset=RT

# Only drop the old_consim RT rows.
python scripts/drop_score_rows.py data/consim_Qwen_Qwen3.5-9B.csv dataset=RT specification=old_consim
```

Writes `<csv>.bak` first unless `--no-backup` is given.

**Canonicalize baseline metadata** (writes backups by default):

```bash
python scripts/canonicalize_baselines.py --dry-run
python scripts/canonicalize_baselines.py
python scripts/canonicalize_baselines.py --aggregate-score-duplicates
```

**Drop rows from prompt JSONL files**:

```bash
python scripts/drop_prompt_rows.py data/prompts/GE_concepts.jsonl classes_subset='[0, 4, 5]' --dry-run
python scripts/drop_prompt_rows.py data/prompts data/prompt_splits dataset=GE classes_subset='[2, 3, 9, 10]'
python scripts/drop_prompt_rows.py data/prompts method=SemiNMF specification=old_consim
```

Writes `<jsonl>.bak` first unless `--no-backup` is given. Use `drop_score_rows.py` separately to remove matching score rows.

**Run full grids with sequence.sh** (cartesian product of comma-separated args):

```bash
./sequence.sh scripts/make_prompts.py concepts BIOS,RT,AG,IMDB seminmf,ica,pca,svd,vanilla_sae,neurons,classes --interpretation topk
./sequence.sh scripts/llm_scoring.py qwen3.5-9b data/prompts/GE_concepts.jsonl,data/prompts/HE_concepts.jsonl
```

**Compile paper** (from LaTeX directory):

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Experiment Plan

### Stage 1 — refind and improve old ConSim results

Goal: reproduce the original concept-based `old_consim` behavior, compare it against the current one-sample-per-prompt `new_consim`, test `simulator_consim`, and check whether `llm` concept interpretations change conclusions compared with `topk`.

| Axis | Values |
| --- | --- |
| Datasets | `BIOS`, `RT`, `AG`, `IMDB` |
| Methods | `seminmf`, `ica`, `pca`, `svd`, `vanilla_sae`, `neurons`, `classes` |
| Interpretation | `topk` |
| Seeds | `0-49` by default |
| Samples per seed | `20` for `new_consim`/`simulator_consim`; `40` for `old_consim` |

Prompt generation:

```bash
CUDA_VISIBLE_DEVICES=1 PATH=".venv/bin:$PATH" ./sequence.sh scripts/make_prompts.py concepts BIOS,RT,AG,IMDB seminmf,ica,pca,svd,vanilla_sae,neurons,classes --interpretation topk --nb-samples 20
CUDA_VISIBLE_DEVICES=1 PATH=".venv/bin:$PATH" ./sequence.sh scripts/make_prompts_consim_v2.py BIOS,RT,AG,IMDB seminmf,ica,pca,svd,vanilla_sae,neurons,classes --interpretation topk --nb-samples 20
CUDA_VISIBLE_DEVICES=1 PATH=".venv/bin:$PATH" ./sequence.sh scripts/make_prompts_old_consim.py BIOS,RT,AG,IMDB seminmf,ica,pca,svd,vanilla_sae,neurons,classes --interpretation topk --nb-samples 40
```

Next notebook work:

- Update `notebooks/4_compare_consim.ipynb` with two specification pickers (`SPEC_A`, `SPEC_B`) over `old_consim`, `new_consim`, and `simulator_consim`.
- Add two interpretation pickers (`INTERP_A`, `INTERP_B`) over `topk` and `llm`, reusing the same comparison plots.
- Keep the existing old/new-style bar and violin comparisons, but make the chosen comparison explicit in plot titles and filenames.
- Add an `old_consim`-only pairwise comparison matrix section, using the old-paper logic, to verify the reconstructed results.
- Export Stage 1 figures to `LaTeX-Simulatability-Shortcut/plots/`.
- Decide from this notebook whether `simulator_consim` is strong enough to justify regenerating all broader prompts and scores.

Debug before larger runs:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts.py concepts RT seminmf --interpretation topk --seeds 0 --nb-samples 20
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts_consim_v2.py RT seminmf --interpretation topk --seeds 0 --nb-samples 20
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts_old_consim.py RT seminmf --interpretation topk --seeds 0 --nb-samples 40
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts.py concepts RT vanilla_sae --interpretation topk --seeds 0 --nb-samples 20
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts_old_consim.py RT vanilla_sae --interpretation topk --seeds 0 --nb-samples 40
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts.py concepts RT neurons --interpretation topk --seeds 0 --nb-samples 20
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts_old_consim.py RT neurons --interpretation topk --seeds 0 --nb-samples 40
```

### Stage 2 — extend to other explanation families

Goal: compare concepts, rationales, and attributions under the harmonized one-sample-per-prompt setup with Qwen3.5-9B, and verify whether explanation methods differ from byte-identical no-explanation baselines.

Next notebook work:

- Use `notebooks/5_compare_families.ipynb` to compare non-anonymized concept, rationale, attribution, and baseline prompt types.
- Select the best non-baseline method/configuration per explanation family, excluding the `classes` concept control.
- Export the ranked bar plots and any necessary violin plots to `LaTeX-Simulatability-Shortcut/plots/`.
- Insert these figures into `LaTeX-Simulatability-Shortcut/main.tex`, especially the section `LLM shortcut all explanations types`.

### Stage 3 — additional LLM judges on best methods

Goal: score only the best methods selected in Stage 2 with a larger set of LLM judges, then test whether those methods significantly differ from their matching baselines.

Next workflow:

- Record the best-method allow-list from `notebooks/5_compare_families.ipynb`.
- Use `scripts/extract_best_prompts.py` to copy only selected methods and matching baselines from `data/prompts/` into `data/best_prompts/`.
- Preserve original prompt keys and schemas in `data/best_prompts/`; do not rewrite prompts or invent new identifiers.
- Score `data/best_prompts/*.jsonl` with the selected judges using `scripts/llm_scoring.py`.
- Use `notebooks/6_judge_consistency.ipynb` to compare judges and run paired t-tests against baselines within each `(dataset, classes_subset)`.
- Defer the exact additional judge list until compute capacity and Stage 2 outputs are known.

Significance convention:

- Pair rows by seed within each `(judge, dataset, classes_subset, method/configuration, prompt_type)` comparison.
- Compare each method/configuration to its matching baseline (`B1` or `B2`; anonymized prompt types use anonymized baselines when included).
- Use paired Student t-tests as the primary test, matching the old ConSim paper.
- Apply a multiple-comparison correction across the tested `(dataset, classes_subset)` cells before reporting paper-level claims.

## CLI Arguments Reference

### `make_prompts.py`

Positional: `explanation_family` (concepts/rationales/attributions), `dataset`, `method` (concept method or attribution method; not used for rationales). Concept methods: `seminmf`, `ica`, `kmeans`, `pca`, `svd`, `vanilla_sae`, `neurons`, `classes`. Optional: `--nb-concepts-ratio`, `--interpretation`, `--llm-model` (for rationales and concept LLM interpretation, default llama3.2-3b), `--rationale-batch-size`, `--max-new-tokens`, `--seeds` (e.g. "0-49"), `--nb-samples`, `--device`, `--batch-size`. For `classes`, `--interpretation` is ignored and prompt keys store `None`.

### `make_prompts_old_consim.py`

Positional: `dataset`, `method`. Concept methods: `seminmf`, `ica`, `kmeans`, `pca`, `svd`, `vanilla_sae`, `neurons`, `classes`. Optional: `--nb-concepts-ratio`, `--interpretation`, `--seeds`, `--nb-samples`, `--device`, `--batch-size`, `--refresh-existing` (rewrite matching existing JSONL keys in place). For `classes`, `--interpretation` is ignored and prompt keys store `None`.

### `make_prompts_consim_v2.py`

Positional: `dataset`, `method`. Concept methods: `seminmf`, `ica`, `kmeans`, `pca`, `svd`, `vanilla_sae`, `neurons`, `classes`. Optional: `--nb-concepts-ratio`, `--interpretation`, `--llm-model` (for concept LLM interpretation, default llama3.2-3b), `--seeds`, `--nb-samples`, `--device`, `--batch-size`. For `classes`, `--interpretation` is ignored and prompt keys store `None`. Writes concept prompt groups to `data/prompts/{dataset}_concepts.jsonl` with `specification=simulator_consim`.

### `extract_best_prompts.py`

Subsets prompt JSONL rows from `data/prompts/` into `data/best_prompts/` using the current hard-coded best-method allow-list: `lime` for attributions, `VanillaSAE` with `topk` for concepts, and `Qwen/Qwen3.5-2B` for rationales. It preserves original prompt keys, row schema, prompt text, and expected answers. It copies baselines alongside the selected best methods and excludes `old_consim` and `simulator_consim` rows. Do not use it to rewrite prompt identifiers or alter scoring semantics.

### `llm_scoring.py`

Positional: `judge_model`, optional one or more `prompt_file` paths. If no prompt files are provided, all `data/prompts/*.jsonl` files are scored in one process with one judge-model load. Optional: `--backend` (`vllm` when installed, otherwise `hf`; force `--backend hf` for the Hugging Face path), `--thinking`, `--max-new-tokens` (default 8 for new-ConSim), `--batch-size` (HF only; defaults to model-specific H100 recommendations when omitted), `--flush-every-prompts` (new-ConSim flush chunk size, default 100), `--device` (HF only). vLLM scoring should use `.venv-vllm/bin/python`; it enables prefix caching and ignores `--batch-size`. Old-ConSim prompts automatically receive a larger generation budget (the `--max-new-tokens` default only sizes new-ConSim's short answer); to re-score after fixing prompts or scoring code, use `scripts/drop_score_rows.py` to remove the stale rows first.

## Key Dependencies

- `interpreto` @ `0.5.0dev1` from `git+https://github.com/FOR-sight-ai/interpreto.git@0.5.0dev1`
- `torch` 2.11.0+cu126
- `transformers` (comes via interpreto)
- `datasets`, `pandas`, `ipykernel`
- Python 3.12

## Architecture Notes

- **Three explanation families**: concepts (SemiNMF/ICA/KMeans/PCA/SVD/VanillaSAE/NeuronsAsConcepts with `topk`/`llm` interpretation keys), rationales (LLM generated), and attributions (gradient-based).
- **Prompt harmonization across families**: `utils/consim.py`, `utils/ratsim.py`, and `utils/attrsim.py` share the same prompt skeleton:
  - Task description uses the canonical ConSim wording (`"You are a classifier. Your task is to assign a label to the evaluation sample. ..."`) with one explanation-specific sentence appended when the family-specific LP block is enabled (concept importances, attributions, rationales).
  - `The classes are: [...]` line is identical, parsed by `extract_allowed_labels` in `llm_scoring.py`.
  - LP examples follow `Sample_{i}:\n\tText: ...\n\tLabel: <pred>` with an optional extra family-specific line (concepts contributions / Attributions / Explanation).
  - Evaluation user prompts follow `Evaluation sample:\n\tText: ...\n\tLabel: ` in all three families.
  - **Baselines invariant**: B1 (without LP) and B2 (with LP) produce byte-identical system prompts, user prompts, and expected answers across concepts/rationales/attributions for the same dataset/seed/classes_subset. If you change one family's task description or LP/user-prompt format, change the other two together to preserve this invariant.
  - Baseline prompt keys always use `method="baseline"`, `nb_concepts=None`, and `interpretation=None`, including concept baselines generated from method-specific runs.
- **Simulator-framed ConSim**: `utils/consim_v2.py` inherits the current concept ConSim implementation and only changes prompting to ask the LLM to simulate a text classifier's predictions. It is concepts-only and uses `specification=simulator_consim`, so the cross-family B1/B2 byte-identical invariant above does not apply to this specification.
- **Rationale anonymization**: `RationalesSimulatability` anonymizes class names appearing inside LP example texts and rationale text using a word-boundary regex (`(?<!\w)<class>(?!\w)`, case-insensitive). Substrings inside other words (e.g. `position` when class is `pos`) are intentionally left untouched. The naive `str.replace` previously used here corrupted unrelated text and is gone; do not reintroduce it.
- **ConSim specifications**: `new_consim` asks one evaluation sample per prompt with classifier-task wording. `simulator_consim` asks one evaluation sample per prompt with explicit model-simulation wording. `old_consim` puts all evaluation samples in one prompt and expects a multi-line response. All use the same sample selection (cached `local_elements`).
- **Concept creation is integrated into prompt generation**: `make_prompts.py` builds concept artifacts (model, interpretations, global importances, ALL local importances) automatically if they are missing from cache. Heavy ML work (interpreto imports, task model loading) only happens on first run.
- **Activation/prediction caches**: `get_activations` returns `(activations, predictions)`, and `utils.data.load_or_compute_activations` caches that tuple for train/validation/test splits (`activations.pt`, `validation_activations.pt`, `test_activations.pt`). There is no separate prediction cache.
- **Pre-computed local importances**: `all_local_importances.pt` is cached in each concept_dir (gradient of each concept for every test sample). Prompt scripts index into this tensor by sample index.
- **Early-exit optimization**: `make_prompts.py` computes all expected output keys before importing torch/interpreto/transformers or loading data/models. If all entries already exist in the output JSONL, the script exits immediately (no model loading overhead).
- **Prompt JSONL** is split by dataset and explanation type: `data/prompts/{dataset_abbrev}_{family}.jsonl`. Old ConSim uses `data/prompts/{dataset_abbrev}_old_consim.jsonl`.
- **Best prompt JSONL** for Stage 3 will live under `data/best_prompts/` with the same filename pattern and row schema as `data/prompts/`. These files contain only selected best methods/configurations and matching baselines, but preserve the original prompt keys so score rows remain joinable with full-grid results.
- **Scoring** auto-detects mode: if `len(user_prompts) == 1` with multiple expected answers → old ConSim parsing; otherwise → one-per-sample scoring. Calling `llm_scoring.py` without prompt files loads all prompt JSONL files from `data/prompts/`; passing multiple prompt files scores just those files with one model load. The script filters out keys already present in the score CSV, and scores the missing keys. New-ConSim prompts are generated in chunks bounded by `--flush-every-prompts` individual evaluation prompts (default 100), then score rows and raw generations are flushed at prompt-group boundaries. The default backend is vLLM when installed, otherwise HF; vLLM enables prefix caching and is intended to run from `.venv-vllm`. Old-ConSim prompts automatically receive a larger generation budget (the `--max-new-tokens` default only sizes new-ConSim's short answer); to re-score after fixing prompts or scoring code, use `scripts/drop_score_rows.py` to remove the stale rows first.
- **Scores** are appended to CSV: `data/consim_{model}.csv` with columns: `dataset,model,classes_subset,seed,method,nb_concepts,interpretation,prompt_type,specification,time,score`.
- **State coverage**: `scripts/state.py` reports valid prompt coverage as `valid% (+corrupted%)` when corrupted prompt-marker rows exist. Corrupted rows count as existing for prompt-generation rerun purposes, because prompt scripts skip those keys unless corrupted rows are explicitly dropped. Incomplete generation manifest rows ignore baseline prompt types (`B*`/`AB*`) so shared baselines do not make method-specific prompt-generation commands look missing. It also reports one-line best-prompt score coverage per judge CSV, using active keys from `data/best_prompts/*.jsonl` as the expected key set and reporting stale best-prompt keys separately. For incomplete best-prompt judges, it prints one `sbatch --array` command over the missing rows in `manifests/best_prompts.tsv`. State memory under `data/state_memory/` stores both main coverage rows and best-prompt rows so subsequent runs can show deltas for each table.
- **Score v2 outputs**: `llm_scoring.py` writes new runs to `data/consim_{model}_v2.csv` and leaves v1 CSVs untouched. V2 columns are `dataset,model,classes_subset,seed,method,nb_concepts,interpretation,prompt_type,specification,time,score,num_correct,num_valid,num_expected`. Invalid-format answers (`None` after parsing against allowed labels) are excluded from the denominator; `score = num_correct / num_valid` only when `num_valid >= ceil(0.7 * num_expected)`, otherwise `score` is `NaN`. Raw generations are appended per prompt group to `data/generations/{model}.jsonl` for offline parser/debug reruns. Old-ConSim token budgets add a 128-token slack above the per-line estimate.
- **Split generation/parsing pipeline**: `scripts/generate_only.py` is the generation-only counterpart to `llm_scoring.py` (same helpers via imports; skips keys already present in `data/generations/{judge}.jsonl`). `scripts/parse_generations.py` re-parses those logs into wide `data/predictions_{judge}.csv` files (one row per prompt key, `pred_0..pred_39` sample columns holding global class ids, plus `num_expected/num_valid/num_correct/cache_n`) using latest-wins deduplication on the raw key. `scripts/build_scores_v3.py` then recomputes `data/consim_{judge}_v3.csv` in the exact V2 schema by applying `compute_group_score` to the parsed counts. V3 reflects the current parser semantics, so V3 scores may differ slightly from V2 when the parser was tightened between runs. Anonymized (`Class_N`) predictions are mapped via `anonymized_class_id` in `prediction_to_global_id`: `old_consim` uses subset-position IDs, while newer specifications use dataset-global class IDs. Non-anonymized predictions are matched by class name.
- **Sample metadata**: `utils.predictions.load_sample_metadata(dataset_abbrev, classes_subset, seed, cache_n)` returns per-sample `real_label` and `task_model_prediction` (as global class ids) plus the dataset-level `test_index`, loaded from `data/{task_model_slug}/local_elements_classes_*_n{cache_n}.json`. `notebooks/6_judge_consistency.ipynb` uses this to join the wide predictions CSVs with gold labels for correlation matrices.
- **Prediction-correlation baselines**: `notebooks/6_judge_consistency.ipynb` uses the attribution-specification B1/B2 rows as one deterministic baseline generation source, rather than stitching duplicate baseline generations sample by sample. It applies the same canonical three-class GoEmotions filter as the score analyses.
- **Sample selection is deterministic and cached**: `local_elements_{classes}_{nb_samples}.json` per save_root. Same seed + same classes_subset + same nb_samples = same samples across all explanation methods.
- **Artifacts** cached aggressively under `data/` to avoid GPU recomputation.
- **All class subsets** for a dataset are processed in a single script invocation (defined in `DATASET_CLASSES_SUBSETS` in `utils/data.py`).
- **Reusable plotting helpers** should live in `utils/plot.py`. The paper should only need three reusable plot families: violin distributions, ranked/difference bar plots, and pairwise comparison matrices. Keep notebook-specific filtering in notebooks, but move reusable figure construction to `utils/plot.py` when it is used by more than one notebook or needed for paper exports. `utils/old_plot.py` is a reference for old-ConSim pairwise matrix behavior; do not delete it while reconstructing old-paper results.
- **Paper plot layout**: notebooks 4-6 export directly to `LaTeX-Simulatability-Shortcut/plots/` at final single- or double-column dimensions. Keep matrix color scales hidden, violin legends inside the axes, dense p-value labels diagonal and compact (`p<.01`), and multi-dataset panels in page-width grids. Always save with `bbox_inches="tight"` and verify the resulting PDFs in the compiled two-column `main.pdf`.

## Registries in `utils/data.py`

When adding a new model/dataset, update:

- `MODELS_DATASETS`: model name → dataset name
- `ABBREVIATIONS`: short codes for datasets/models
- `DATASET_CLASSES_NAMES`: ordered class name list per dataset
- `DATASET_LABEL_COLUMNS`: label column per dataset
- `DATASET_CLASSES_SUBSETS`: canonical class subsets for experiments

## Style

- Researcher code. Optimize for clarity and traceability.
- Dense comments are encouraged when they help audit experiment logic.
- Prefer explicit parameters and filenames over hidden configuration.
- Prefer appending results over overwriting.
- No test suite. Validate by running the smallest relevant script path.

## Environment

- Use `.venv` for all Python commands.
- Do not modify `.venv` or install/remove packages unless explicitly asked.
- CUDA 12.6 expected for GPU work.
