# Simulatability Shortcut

This repo was forked from [contrastive_concepts](https://github.com/AntoninPoche/contrastive_concepts) to study weaknesses of automated simulatability metrics in NLP.

The central claim is that automated simulatability can fail to distinguish explanation methods from no-explanation baselines. The experiments compare concept-based explanations, rationales, and attributions across datasets and LLM judges.

## First Experiments

The first experiment phase compares **new ConSim** against **old ConSim**.

| Variant | Evaluation format |
| --- | --- |
| `new_consim` | One evaluation sample per prompt |
| `old_consim` | All evaluation samples in one prompt |

The comparison grid is:

| Axis | Values |
| --- | --- |
| Datasets | `BIOS`, `RT`, `AG`, `IMDB` |
| Concept methods | `seminmf`, `ica`, `pca`, `svd`, `batchtopk_sae`, `vanilla_sae`, `neurons` |
| Interpretation | `topk` |
| Seeds | `0-49` by default |
| Samples per seed | `20` by default |

Prompt generation commands:

```bash
CUDA_VISIBLE_DEVICES=1 PATH=".venv/bin:$PATH" ./sequence.sh scripts/make_prompts.py concepts BIOS,RT,AG,IMDB seminmf,ica,pca,svd,batchtopk_sae,vanilla_sae,neurons --interpretation topk
CUDA_VISIBLE_DEVICES=1 PATH=".venv/bin:$PATH" ./sequence.sh scripts/make_prompts_old_consim.py BIOS,RT,AG,IMDB seminmf,ica,pca,svd,batchtopk_sae,vanilla_sae,neurons --interpretation topk
```

Scoring commands:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/llm_scoring.py qwen3.5-9b data/prompts/BIOS_concepts.jsonl
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/llm_scoring.py qwen3.5-9b data/prompts/BIOS_old_consim.jsonl
```

Use the same scoring pattern for `RT`, `AG`, and `IMDB` prompt files.

## Debugging Before Larger Runs

Run the smallest checks before launching the full grid:

1. Verify standard concept prompt generation:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts.py concepts RT seminmf --interpretation topk --seeds 0 --nb-samples 20
```

2. Verify old ConSim loads the same cached concept resources:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts_old_consim.py RT seminmf --interpretation topk --seeds 0 --nb-samples 20
```

3. Verify SAE concept training and prompt generation:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts.py concepts RT vanilla_sae --interpretation topk --seeds 0 --nb-samples 20
```

4. Verify old ConSim works for an SAE method:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts_old_consim.py RT vanilla_sae --interpretation topk --seeds 0 --nb-samples 20
```

5. Verify neurons-as-concepts prompt generation:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts.py concepts RT neurons --interpretation topk --seeds 0 --nb-samples 20
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/make_prompts_old_consim.py RT neurons --interpretation topk --seeds 0 --nb-samples 20
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
| `scripts/make_prompts_old_consim.py` | Generate old ConSim prompts for concept explanations |
| `scripts/llm_scoring.py` | Score prompt JSONL files with a local Hugging Face LLM judge |
| `sequence.sh` | Run a cartesian product over comma-separated CLI arguments |

## Output Files

Prompt JSONL files are append-only and resumable:

```text
data/prompts/{dataset}_concepts.jsonl
data/prompts/{dataset}_old_consim.jsonl
```

Scores are appended to:

```text
data/consim_{judge_model}.csv
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
sbatch --array=1-8%8 manifest.sbatch manifests/file.tsv   # launch an array from a manifest
squeue -u $USER                                           # see queued/running jobs
tail -f data/logs/<job_name>_<jobid>.out                  # follow stdout log
scancel <jobid>                                           # cancel a job or array
```