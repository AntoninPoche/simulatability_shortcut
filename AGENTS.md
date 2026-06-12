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
  make_prompts.py             # Generate new-ConSim prompt JSONL (CLI, argparse); builds concept artifacts if missing
  make_prompts_old_consim.py  # Generate old-ConSim prompt JSONL for comparison (CLI)
  local_llm_scoring.py        # Score prompts with a local HF LLM judge (CLI)

utils/                        # Shared library package
  __init__.py
  data.py                     # Dataset/model registries, split loading, caching helpers
  concepts.py                 # Concept model loading/fitting, interpretations, importances, build pipeline
  consim.py                   # New ConSim prompt builder (one prompt per eval sample)
  old_consim.py               # Old ConSim prompt builder (all eval samples at once)
  simulatability.py           # Base AutomatedSimulatability class (local, not from interpreto)
  rationales.py               # Rationale generation from local LLMs (Qwen)
  ratsim.py                   # Rationale-based prompt construction

sequence.sh                   # Cartesian-product script runner (see Commands below)
generation_concept_tutorial.ipynb  # Interpreto concept tutorial, including BatchTopK SAE loss setup
LaTeX-Simulatability-Shortcut/  # ACL paper sources (separate git subrepo)
data/                         # Gitignored artifacts: activations, predictions, prompts, scores
```

## Import Architecture

- **Local `utils/` package**: `simulatability.py`, `consim.py`, `old_consim.py`, `ratsim.py` — these are the canonical implementations, not imported from interpreto.
- **From `interpreto`**: concept extraction algorithms (`SemiNMFConcepts`, `ICAConcepts`, etc.), `SplitterForClassification`, `LLMLabels`, `TopKInputs`. Used only by `utils/concepts.py` for the heavy ML components (loaded lazily when concept artifacts need building).
- **Prompt scripts (`make_prompts.py`, `make_prompts_old_consim.py`) import from `utils/concepts.py`** — which handles both loading cached artifacts and building them (with interpreto) when missing.
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
python scripts/make_prompts.py rationales BIOS
python scripts/make_prompts.py rationales BIOS --llm-model qwen3.5-9b
python scripts/make_prompts.py attributions GE saliency
```

**Generate prompts — old ConSim** (for new-vs-old comparison, uses same cached samples):

```bash
python scripts/make_prompts_old_consim.py GE seminmf
```

**Score prompts with local LLM**:

```bash
python scripts/local_llm_scoring.py qwen3.5-9b data/prompts/GE_concepts.jsonl
```

**Run full grids with sequence.sh** (cartesian product of comma-separated args):

```bash
./sequence.sh scripts/make_prompts.py concepts BIOS,RT,AG,IMDB seminmf,ica,pca,svd,batchtopk_sae,vanilla_sae,neurons --interpretation topk
./sequence.sh scripts/local_llm_scoring.py qwen3.5-9b data/prompts/GE_concepts.jsonl,data/prompts/HE_concepts.jsonl
```

**Compile paper** (from LaTeX directory):

```bash
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Experiment Plan

### Phase 1 — new ConSim vs old ConSim

Goal: compare `new_consim` (one evaluation sample per prompt) against `old_consim` (all evaluation samples in one prompt) before launching broader explanation experiments.

| Axis | Values |
| --- | --- |
| Datasets | `BIOS`, `RT`, `AG`, `IMDB` |
| Methods | `seminmf`, `ica`, `pca`, `svd`, `batchtopk_sae`, `vanilla_sae`, `neurons` |
| Interpretation | `topk` |
| Seeds | `0-49` by default |
| Samples per seed | `20` by default |

Prompt generation:

```bash
CUDA_VISIBLE_DEVICES=1 PATH=".venv/bin:$PATH" ./sequence.sh scripts/make_prompts.py concepts BIOS,RT,AG,IMDB seminmf,ica,pca,svd,batchtopk_sae,vanilla_sae,neurons --interpretation topk
CUDA_VISIBLE_DEVICES=1 PATH=".venv/bin:$PATH" ./sequence.sh scripts/make_prompts_old_consim.py BIOS,RT,AG,IMDB seminmf,ica,pca,svd,batchtopk_sae,vanilla_sae,neurons --interpretation topk
```

Debug before larger runs:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts.py concepts RT seminmf --interpretation topk --seeds 0 --nb-samples 5
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts_old_consim.py RT seminmf --interpretation topk --seeds 0 --nb-samples 5
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts.py concepts RT vanilla_sae --interpretation topk --seeds 0 --nb-samples 5
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts_old_consim.py RT vanilla_sae --interpretation topk --seeds 0 --nb-samples 5
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts.py concepts RT neurons --interpretation topk --seeds 0 --nb-samples 5
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts_old_consim.py RT neurons --interpretation topk --seeds 0 --nb-samples 5
```

### Phase 2 — broader explanation experiments

After Phase 1 is debugged, extend to rationale, attribution, and concept simulatability grids across the intended datasets and LLM judges.

## CLI Arguments Reference

### `make_prompts.py`

Positional: `explanation_family` (concepts/rationales/attributions), `dataset`, `method` (concept method or attribution method; not used for rationales). Concept methods: `seminmf`, `ica`, `kmeans`, `pca`, `svd`, `batchtopk_sae`, `vanilla_sae`, `neurons`. Optional: `--nb-concepts-ratio`, `--interpretation`, `--llm-model` (for rationales and concept LLM interpretation, default llama3.2-3b), `--rationale-batch-size`, `--max-new-tokens`, `--seeds` (e.g. "0-49"), `--nb-samples`, `--device`, `--batch-size`

### `make_prompts_old_consim.py`

Positional: `dataset`, `method`. Concept methods: `seminmf`, `ica`, `kmeans`, `pca`, `svd`, `batchtopk_sae`, `vanilla_sae`, `neurons`. Optional: `--nb-concepts-ratio`, `--interpretation`, `--seeds`, `--nb-samples`, `--device`, `--batch-size`

### `local_llm_scoring.py`

Positional: `judge_model`, `prompt_file`. Optional: `--thinking`/`--no-thinking`, `--max-new-tokens`, `--generation-batch-size`, `--device`

## Key Dependencies

- `interpreto` @ `0.5.0dev1` from `git+https://github.com/FOR-sight-ai/interpreto.git@0.5.0dev1`
- `torch` 2.11.0+cu126
- `transformers` (comes via interpreto)
- `datasets`, `pandas`, `ipykernel`
- Python 3.12

## Architecture Notes

- **Three explanation families**: concepts (SemiNMF/ICA/KMeans/PCA/SVD/BatchTopKSAE/VanillaSAE/NeuronsAsConcepts + TopKInputs/LLMLabels), rationales (LLM generated), and attributions (gradient-based).
- **New vs Old ConSim**: new ConSim asks one evaluation sample per prompt. Old ConSim puts all evaluation samples in one prompt and expects a multi-line response. Both use the same sample selection (cached `local_elements`).
- **Concept creation is integrated into prompt generation**: `make_prompts.py` builds concept artifacts (model, interpretations, global importances, ALL local importances) automatically if they are missing from cache. Heavy ML work (interpreto imports, task model loading) only happens on first run.
- **Pre-computed local importances**: `all_local_importances.pt` is cached in each concept_dir (gradient of each concept for every test sample). Prompt scripts index into this tensor by sample index.
- **Early-exit optimization**: `make_prompts.py` computes all expected output keys before loading data/models. If all entries already exist in the output JSONL, the script exits immediately (no model loading overhead).
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
