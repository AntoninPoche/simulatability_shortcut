# AGENTS.md

## Critical Rules

- Make the smallest change that solves the task.
- Preserve experiment traceability. Do not rename files, result columns, prompt identifiers, cache layouts, or artifact paths without explicit request.
- Treat `data/` as expensive. Never delete or overwrite artifacts unless asked.
- Do not "productionize" the repo. Keep things script-driven and explicit.
- Keep this file up to date when adding/removing/renaming files or changing the workflow.

## Purpose

Research experiments showing weaknesses of automated simulatability metrics in NLP. The paper argues that simulatability cannot differentiate between explanation methods and no-explanation baselines. Experiments span concept-based explanations, rationales, and attributions across multiple datasets and LLM judges.

Forked from [contrastive_concepts](https://github.com/AntoninPoche/contrastive_concepts).

## Repository Structure

```
scripts/
  build_concepts.py           # Build and cache concept models/interpretations/importances (CLI)
  make_prompts.py             # Generate new-ConSim prompt JSONL (CLI, argparse)
  make_prompts_old_consim.py  # Generate old-ConSim prompt JSONL for comparison (CLI)
  local_llm_scoring.py        # Score prompts with a local HF LLM judge (CLI)

utils/                        # Shared library package
  __init__.py
  data.py                     # Dataset/model registries, split loading, caching helpers
  concepts.py                 # Concept model loading/fitting, interpretations, importances
  consim.py                   # New ConSim prompt builder (one prompt per eval sample)
  old_consim.py               # Old ConSim prompt builder (all eval samples at once)
  simulatability.py           # Base AutomatedSimulatability class (local, not from interpreto)
  rationales.py               # Rationale generation from local LLMs (Qwen)
  ratsim.py                   # Rationale-based prompt construction

sequence.sh                   # Cartesian-product script runner (see Commands below)
LaTeX-Simulatability-Shortcut/  # ACL paper sources (separate git subrepo)
data/                         # Gitignored artifacts: activations, predictions, prompts, scores
```

## Import Architecture

- **Local `utils/` package**: `simulatability.py`, `consim.py`, `old_consim.py`, `ratsim.py` — these are the canonical implementations, not imported from interpreto.
- **From `interpreto`**: concept extraction algorithms (`SemiNMFConcepts`, `ICAConcepts`, etc.), `SplitterForClassification`, `LLMLabels`, `TopKInputs`. Used only by `build_concepts.py` and `utils/concepts.py` for the heavy ML components.
- **Prompt scripts (`make_prompts.py`, `make_prompts_old_consim.py`) do NOT import from interpreto** — they load pre-built artifacts from disk.
- All scripts add the repo root to `sys.path` so `from utils.* import ...` works when running `python scripts/foo.py`.

## Commands

**Install dependencies** (venv exists but packages may need installing):
```bash
.venv/bin/pip install -r requirements.txt
```

**Build concept models** (CLI with argparse):
```bash
python scripts/build_concepts.py --dataset GE --method seminmf --interpretation topk
python scripts/build_concepts.py --dataset BIOS --method ica --nb-concepts-ratio 2
```

**Generate prompts — new ConSim** (iterates over all class subsets for the dataset):
```bash
python scripts/make_prompts.py --dataset GE --explanation-family concepts --method seminmf
python scripts/make_prompts.py --dataset BIOS --explanation-family rationales --method Qwen/Qwen3.5-9B
```

**Generate prompts — old ConSim** (for new-vs-old comparison, uses same cached samples):
```bash
python scripts/make_prompts_old_consim.py --dataset GE --method seminmf
```

**Score prompts with local LLM**:
```bash
python scripts/local_llm_scoring.py --judge-model Qwen/Qwen3.5-9B --prompt-file data/prompts/GE_concepts.jsonl
```

**Run full grids with sequence.sh** (cartesian product of comma-separated args):
```bash
./sequence.sh scripts/build_concepts.py GE,HE,BIOS --method ica,kmeans --interpretation topk
./sequence.sh scripts/make_prompts.py GE,HE,BIOS --explanation-family concepts --method seminmf,ica,kmeans
./sequence.sh scripts/local_llm_scoring.py --judge-model Qwen/Qwen3.5-9B --prompt-file data/prompts/GE_concepts.jsonl,data/prompts/HE_concepts.jsonl
```

**Compile paper** (from LaTeX directory):
```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## CLI Arguments Reference

### `build_concepts.py`
`--dataset` (GE/HE/BIOS/E), `--method` (seminmf/ica/kmeans/pca/svd), `--interpretation` (topk/llm), `--nb-concepts-ratio`, `--llm-model`, `--device`, `--batch-size`

### `make_prompts.py`
`--dataset`, `--explanation-family` (concepts/rationales), `--method` (concept method or rationale model name), `--nb-concepts-ratio`, `--interpretation`, `--rationale-batch-size`, `--max-new-tokens`, `--seeds` (e.g. "0-49"), `--nb-samples`, `--device`, `--batch-size`

### `make_prompts_old_consim.py`
`--dataset`, `--method`, `--nb-concepts-ratio`, `--interpretation`, `--seeds`, `--nb-samples`, `--device`, `--batch-size`

### `local_llm_scoring.py`
`--judge-model`, `--prompt-file`, `--thinking`/`--no-thinking`, `--max-new-tokens`, `--generation-batch-size`, `--device`

## Key Dependencies

- `interpreto` @ `0.5.0dev1` from `git+https://github.com/FOR-sight-ai/interpreto.git@0.5.0dev1`
- `torch` 2.11.0+cu126
- `transformers` (comes via interpreto)
- `datasets`, `pandas`, `ipykernel`
- Python 3.12

## Architecture Notes

- **Two explanation families**: concepts (SemiNMF/ICA/KMeans/PCA/SVD + TopKInputs/LLMLabels) and rationales (Qwen LLM generated).
- **New vs Old ConSim**: new ConSim asks one evaluation sample per prompt. Old ConSim puts all evaluation samples in one prompt and expects a multi-line response. Both use the same sample selection (cached `local_elements`).
- **Concept creation vs prompt generation are fully decoupled**: `build_concepts.py` creates all concept artifacts (model, interpretations, global importances, ALL local importances for test samples). `make_prompts.py` and `make_prompts_old_consim.py` only load pre-built artifacts from disk — they never import from interpreto or load the task model.
- **Pre-computed local importances**: `build_concepts.py` caches `all_local_importances.pt` in each concept_dir (gradient of each concept for every test sample). Prompt scripts index into this tensor by sample index.
- **Prompt JSONL** is split by dataset and explanation type: `data/prompts/{dataset_abbrev}_{family}.jsonl`. Old ConSim uses `data/prompts/{dataset_abbrev}_old_consim.jsonl`.
- **Scoring** auto-detects mode: if `len(user_prompts) == 1` with multiple expected answers → old ConSim parsing; otherwise → one-per-sample scoring.
- **Scores** are appended to CSV: `data/consim_{model}.csv` with columns: `dataset,model,classes_subset,seed,method,nb_concepts,interpretation,prompt_type,specification,time,score`.
- **Sample selection is deterministic and cached**: `local_elements_{classes}_{nb_samples}.json` per save_root. Same seed + same classes_subset + same nb_samples = same samples across all explanation methods.
- **Artifacts** cached aggressively under `data/` to avoid GPU recomputation.
- **All class subsets** for a dataset are processed in a single script invocation (defined in `DATASET_CLASSES_SUBSETS` in `utils/data.py`).

## Registries in `utils/data.py`

When adding a new model/dataset, update:
- `MODELS_DATASETS`: model name → dataset name
- `ABBREVIATIONS`: short codes for datasets/models
- `DATASET_CLASSES_NAMES`: ordered class name list per dataset
- `DATASET_LABEL_COLUMNS`: label column per dataset
- `MODEL_SPLIT_POINTS`: split point for concept extraction (use `"auto"` for new models — `SplitterForClassification` auto-detects the classification head)
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
