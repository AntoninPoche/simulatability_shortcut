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
  build_concepts.py     # Build and cache concept models/interpretations/importances (CLI)
  make_prompts.py       # Generate ConSim prompt JSONL (top-of-file constants)
  local_llm_scoring.py  # Score prompts with a local HF LLM

sources/                # Shared library modules (see Import Caveat below)
  data.py               # Dataset/model registries, split loading, caching helpers
  concepts.py           # Concept model loading/fitting, interpretations, importances
  consim.py             # ConSim prompt builder (mirrors/overrides interpreto's ConSim)
  old_consim.py         # Legacy ConSim implementation (for comparison)
  simulatability.py     # Base AutomatedSimulatability class (mirrors interpreto)
  rationales.py         # Rationale generation from local LLMs (Qwen)
  rationales_simulatability.py  # Rationale-based prompt construction

sequence.sh             # Cartesian-product script runner (see Commands below)
LaTeX-Simulatability-Shortcut/  # ACL paper sources (separate git subrepo)
data/                   # Gitignored artifacts: activations, predictions, prompts, scores
```

## Import Caveat (important)

Scripts import from `utils.*` (e.g., `from utils.data import ...`) but the physical directory is `sources/`. There is no `utils/` directory or package. **Scripts cannot run as-is** without either:
- Renaming `sources/` → `utils/`, or
- Creating a `utils/` symlink to `sources/`

This mismatch exists because the repo was forked from `contrastive_concepts` where the directory was called `utils/`.

Several files in `sources/` (`consim.py`, `old_consim.py`, `simulatability.py`) carry the interpreto MIT license and mirror/override classes from the `interpreto` library.

## Commands

**Install dependencies** (venv exists but packages not yet installed):
```bash
.venv/bin/pip install -r requirements.txt
```

**Build concept models** (CLI with argparse):
```bash
python scripts/build_concepts.py --dataset GE --method seminmf --interpretation topk
python scripts/build_concepts.py --dataset BIOS --method ica --nb-concepts-ratio 2
```

Valid `--dataset` values: `GE`, `HE`, `BIOS`, `E`.  
Valid `--method` values: `seminmf`, `ica`, `kmeans`, `pca`, `svd`.  
Valid `--interpretation` values: `topk`, `llm`.

**Run combinations with sequence.sh** (cartesian product of comma-separated args):
```bash
./sequence.sh scripts/build_concepts.py GE,HE,BIOS --method ica,kmeans --interpretation topk
```

**Generate prompts** (configure via top-of-file constants, then run):
```bash
python scripts/make_prompts.py
```

**Score prompts with local LLM** (configure MODEL_NAME at top of file):
```bash
python scripts/local_llm_scoring.py
```

**Compile paper** (from LaTeX directory):
```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Key Dependencies

- `interpreto` @ `0.5.0dev1` from `git+https://github.com/FOR-sight-ai/interpreto.git@0.5.0dev1`
- `torch` 2.11.0+cu126
- `transformers` (used by scripts but not pinned in requirements.txt — comes via interpreto)
- `datasets`, `pandas`, `ipykernel`
- Python 3.12

## Architecture Notes

- Two explanation families: **concepts** (SemiNMF/ICA/KMeans/PCA/SVD + TopKInputs/LLMLabels interpretations) and **rationales** (Qwen LLM generated).
- `scripts/make_prompts.py` and `scripts/local_llm_scoring.py` use **top-of-file constants** (MODEL_NAME, CLASSES_SUBSET, EXPLANATION_FAMILY, etc.) instead of CLI args. Edit the file to change configuration.
- `scripts/build_concepts.py` uses proper CLI argparse.
- Prompt JSONL is the bridge between prompt generation and scoring: `data/consim_prompts.jsonl`.
- Scores are appended to CSV: `data/consim_{model}.csv` with columns: `dataset,model,classes_subset,seed,method,nb_concepts,interpretation,prompt_type,specification,time,score`.
- Artifacts are cached aggressively under `data/` to avoid GPU recomputation.
- Experiment identity is encoded in string keys (tuples of dataset, model, classes, seed, method, etc.).

## Registries in `sources/data.py`

When adding a new model/dataset, update:
- `MODELS_DATASETS`: model name → dataset name
- `ABBREVIATIONS`: short codes for datasets/models
- `DATASET_CLASSES_NAMES`: ordered class name list per dataset
- `DATASET_LABEL_COLUMNS`: label column per dataset
- `MODEL_SPLIT_POINTS`: layer index for concept extraction

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
