# Simulatability Shortcut

This repo was forked from [contrastive_concepts](https://github.com/AntoninPoche/contrastive_concepts) to study weaknesses of automated simulatability metrics in NLP.

The central claim is that automated simulatability can fail to distinguish explanation methods from no-explanation baselines. The experiments compare concept-based explanations, rationales, and attributions across datasets and LLM judges.

## Experiment Roadmap

The project proceeds in three stages.

| Stage | Goal |
| --- | --- |
| 1. Refind and improve old ConSim results | Reproduce the original concept-based `old_consim` results, compare them against the current one-sample-per-prompt `new_consim`, and test the simulator-framed `simulator_consim` prompt variant. |
| 2. Extend to explanation families | Run the harmonized simulatability setup across concept explanations, rationales, and attributions to test whether the metric distinguishes explanation families from no-explanation baselines. |
| 3. Broaden LLM judges | On the strongest or most informative methods from the previous stages, repeat scoring with additional LLM judges to check whether conclusions are judge-specific. |

## Stage 1: ConSim Variants

The first experiment phase focuses on refinding the results from **old ConSim** and improving the setup.

| Variant | Evaluation format |
| --- | --- |
| `old_consim` | Original format: all evaluation samples in one prompt |
| `new_consim` | Current format: one evaluation sample per prompt |
| `simulator_consim` | One evaluation sample per prompt, explicitly asking the LLM to simulate the classifier |

The comparison grid is:

| Axis | Values |
| --- | --- |
| Datasets | `BIOS`, `RT`, `AG`, `IMDB` |
| Concept methods | `seminmf`, `ica`, `pca`, `svd`, `vanilla_sae`, `neurons`, `classes` |
| Interpretation | `topk` |
| Seeds | `0-49` by default |
| Samples per seed | `20` for `new_consim`/`simulator_consim`; `40` for `old_consim` |

Prompt generation commands:

```bash
CUDA_VISIBLE_DEVICES=1 PATH=".venv/bin:$PATH" ./sequence.sh scripts/make_prompts.py concepts BIOS,RT,AG,IMDB seminmf,ica,pca,svd,vanilla_sae,neurons,classes --interpretation topk --nb-samples 20
CUDA_VISIBLE_DEVICES=1 PATH=".venv/bin:$PATH" ./sequence.sh scripts/make_prompts_consim_v2.py BIOS,RT,AG,IMDB seminmf,ica,pca,svd,vanilla_sae,neurons,classes --interpretation topk --nb-samples 20
CUDA_VISIBLE_DEVICES=1 PATH=".venv/bin:$PATH" ./sequence.sh scripts/make_prompts_old_consim.py BIOS,RT,AG,IMDB seminmf,ica,pca,svd,vanilla_sae,neurons,classes --interpretation topk --nb-samples 40
```

Scoring commands:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/llm_scoring.py qwen3.5-9b data/prompts/BIOS_concepts.jsonl
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/llm_scoring.py qwen3.5-9b data/prompts/BIOS_old_consim.jsonl
```

Use the same scoring pattern for `RT`, `AG`, and `IMDB` prompt files.

Next steps:

1. Update `notebooks/4_compare_consim.ipynb` with `SPEC_A` / `SPEC_B` parameters over `old_consim`, `new_consim`, and `simulator_consim`.
2. Add `INTERP_A` / `INTERP_B` parameters over `topk` and `llm` to compare concept interpretations with the same plotting logic.
3. Add an `old_consim`-only pairwise comparison matrix section, following the old ConSim paper, to check that the reconstructed results match the reference behavior.
4. Export Stage 1 figures to `LaTeX-Simulatability-Shortcut/plots/`.
5. Decide whether `simulator_consim` is strong enough to justify regenerating broader prompts and scores before Stage 2 figures are finalized.

## Stage 2: Explanation Families

After the ConSim variants are debugged, extend the experiments beyond concepts. The prompt builders keep the baseline prompts byte-identical across families for the same dataset, seed, and class subset, so no-explanation baselines can be compared directly.

| Family | Examples |
| --- | --- |
| Concepts | `seminmf`, `ica`, `pca`, `svd`, `vanilla_sae`, `neurons`, `classes` |
| Rationales | Local LLM-generated rationales, e.g. `qwen3.5-9b` |
| Attributions | Gradient-based methods such as `saliency` |

Example commands:

```bash
CUDA_VISIBLE_DEVICES=1 PATH=".venv/bin:$PATH" ./sequence.sh scripts/make_prompts.py concepts BIOS,RT,AG,IMDB seminmf,ica,pca,svd,vanilla_sae,neurons,classes --interpretation topk
CUDA_VISIBLE_DEVICES=1 PATH=".venv/bin:$PATH" ./sequence.sh scripts/make_prompts.py rationales BIOS,RT,AG,IMDB
CUDA_VISIBLE_DEVICES=1 PATH=".venv/bin:$PATH" ./sequence.sh scripts/make_prompts.py attributions BIOS,RT,AG,IMDB saliency
```

Next steps:

1. Use `notebooks/5_compare_families.ipynb` to compare concept, rationale, attribution, and baseline prompt types under Qwen3.5-9B.
2. Select the best non-baseline method/configuration per family, excluding the `classes` concept control.
3. Export ranked bar plots and useful violin plots to `LaTeX-Simulatability-Shortcut/plots/`.
4. Add the exported figures to `LaTeX-Simulatability-Shortcut/main.tex`, especially the section `LLM shortcut all explanations types`.

## Stage 3: Additional LLM Judges

Once the best methods and most informative comparisons are identified, score those prompt files with additional LLM judges. This stage is intentionally narrower than Stage 2: it checks robustness of the main conclusions without rerunning every method-family combination for every judge.

The exact additional judge list is deferred until the Stage 2 outputs and compute capacity are known.

Example:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/llm_scoring.py qwen3.5-9b
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/llm_scoring.py llama3.2-3b
```

Next steps:

1. Export a best-method allow-list from `notebooks/5_compare_families.ipynb`.
2. Use `scripts/extract_best_prompts.py` to copy only selected methods and matching baselines from `data/prompts/` into `data/best_prompts/`.
3. Preserve original prompt keys, schemas, prompt text, and expected answers in `data/best_prompts/`.
4. Score `data/best_prompts/*.jsonl` with the selected additional judges using `scripts/llm_scoring.py`.
5. Use `notebooks/6_judge_consistency.ipynb` to run paired Student t-tests per seed against matching baselines within each `(dataset, classes_subset)`.
6. Apply a multiple-comparison correction across tested `(dataset, classes_subset)` cells before making paper-level claims.

## Debugging Before Larger Runs

Run the smallest checks before launching the full grid:

1. Verify standard concept prompt generation:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts.py concepts RT seminmf --interpretation topk --seeds 0 --nb-samples 20
```

2. Verify old ConSim loads the same cached concept resources:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts_old_consim.py RT seminmf --interpretation topk --seeds 0 --nb-samples 40
```

3. Verify SAE concept training and prompt generation:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts.py concepts RT vanilla_sae --interpretation topk --seeds 0 --nb-samples 20
```

4. Verify old ConSim works for an SAE method:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts_old_consim.py RT vanilla_sae --interpretation topk --seeds 0 --nb-samples 40
```

5. Verify neurons-as-concepts prompt generation:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts.py concepts RT neurons --interpretation topk --seeds 0 --nb-samples 20
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts_old_consim.py RT neurons --interpretation topk --seeds 0 --nb-samples 40
```

6. Score one small prompt file with the intended judge:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/llm_scoring.py qwen3.5-9b data/prompts/RT_concepts.jsonl
```

7. Smoke-test the grid runner:

```bash
CUDA_VISIBLE_DEVICES=1 PATH=".venv/bin:$PATH" ./sequence.sh scripts/make_prompts.py concepts RT,AG seminmf,pca --interpretation topk --seeds 0 --nb-samples 20
```

Only launch the full grid after these checks pass or the failures are understood.

## Main Scripts

| Script | Purpose |
| --- | --- |
| `scripts/make_prompts.py` | Generate new ConSim prompts for concepts, rationales, or attributions |
| `scripts/make_prompts_consim_v2.py` | Generate simulator-framed ConSim prompts for concept explanations |
| `scripts/make_prompts_old_consim.py` | Generate old ConSim prompts for concept explanations |
| `scripts/extract_best_prompts.py` | Copy only best-method prompts and matching baselines into `data/best_prompts/` |
| `scripts/llm_scoring.py` | Score prompt JSONL files with a local Hugging Face LLM judge |
| `sequence.sh` | Run a cartesian product over comma-separated CLI arguments |

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `notebooks/4_compare_consim.ipynb` | Planned ConSim comparison notebook: select two specifications and two interpretations, compare score differences, and reproduce old-ConSim pairwise matrices. |
| `notebooks/5_compare_families.ipynb` | Planned family-comparison notebook: compare concepts, rationales, attributions, and baselines; select best methods for Stage 3. |
| `notebooks/6_judge_consistency.ipynb` | Planned multi-judge notebook: compare best methods across judges and test differences against baselines. |
| `notebooks/old/` | Archived notebooks kept as reference before the next-paper rewrite. |

Reusable figure construction should live in `utils/plot.py` when it is useful beyond one notebook. The expected reusable plot families are violin distributions, ranked/difference bar plots, and pairwise comparison matrices. `utils/old_plot.py` remains a reference for reproducing the old ConSim matrix style.

## Output Files

Prompt JSONL files are append-only and resumable:

```text
data/prompts/{dataset}_concepts.jsonl
data/prompts/{dataset}_rationales.jsonl
data/prompts/{dataset}_attributions.jsonl
data/prompts/{dataset}_old_consim.jsonl
data/best_prompts/{dataset}_concepts.jsonl
data/best_prompts/{dataset}_rationales.jsonl
data/best_prompts/{dataset}_attributions.jsonl
```

Scores are appended to:

```text
data/consim_{judge_model}_v2.csv
data/generations/{judge_model}.jsonl
```

Concept artifacts are cached under:

```text
data/{model_name}/concept_models/{method}_nc{nb_concepts}/
```

## Notes

Use the existing `.venv` for Python commands. Use `CUDA_VISIBLE_DEVICES=1` to target the second physical GPU; inside the process this is exposed as `cuda`.


## On cluster

```bash
sbatch my_run.sbatch                                      # launch one normal batch job
sbatch --job-name=<job_name> --array=1-N manifest.sbatch manifests/file.tsv   # launch an array from a manifest
squeue -u $USER                                           # see queued/running jobs
tail -f data/logs/<job_name>_<jobid>.out                  # follow stdout log
scancel <jobid>                                           # cancel a job or array
```

./make_manifests.sh manifests/all_prompts.tsv scripts/llm_scoring.py qwen3.5-9b data/prompts/AG_attributions.jsonl,data/prompts/GE_rationales.jsonl,data/prompts/BIOS_rationales.jsonl,data/prompts/RT_attributions.jsonl,data/prompts/AG_concepts.jsonl,data/prompts/IMDB_attributions.jsonl,data/prompts/AG_rationales.jsonl,data/prompts/E_concepts.jsonl,data/prompts/RT_concepts.jsonl,data/prompts/GE_attributions.jsonl,data/prompts/IMDB_concepts.jsonl,data/prompts/RT_rationales.jsonl,data/prompts/BIOS_attributions.jsonl,data/prompts/IMDB_rationales.jsonl
