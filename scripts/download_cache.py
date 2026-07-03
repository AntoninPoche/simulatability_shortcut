import nltk
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

NLTK_PACKAGES = ["punkt", "punkt_tab", "wordnet"]
for package in NLTK_PACKAGES:
    print(package)
    nltk.download(package)

MODELS_DATASETS = {
    "SamLowe/roberta-base-go_emotions": "google-research-datasets/go_emotions",
    "nateraw/bert-base-uncased-emotion": "dair-ai/emotion",
    # "Hate-speech-CNERG/bert-base-uncased-hatexplain": "Hate-speech-CNERG/hatexplain",
    "Fannyjrd/roberta-bios-biased": "LabHC/bias_in_bios",
    "raulbs7/ag-news-classifier": "fancyzhx/ag_news",
    "keerthi1515/roberta-sentiment-rotten-tomatoes": "cornell-movie-review-data/rotten_tomatoes",
    "philipobiorah/bert-imdb-model": "stanfordnlp/imdb",
}

for model_id, dataset_id in MODELS_DATASETS.items():
    print(model_id)
    AutoTokenizer.from_pretrained(model_id)
    AutoModelForSequenceClassification.from_pretrained(model_id)

    print(dataset_id)
    load_dataset(dataset_id)

LLM_MODELS = {
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "qwen3.6-27b": "Qwen/Qwen3.6-27B",
    "phi4": "microsoft/phi-4",
    "ministral-14b": "mistralai/Ministral-3-14B-Instruct-2512",
    "gemma4-31b": "google/gemma-4-31B-it",
    "gpt-oss-20b": "openai/gpt-oss-20b",
}

for model_id in LLM_MODELS.values():
    print(model_id)

    # Download the full repo snapshot: configs, tokenizer, processor,
    # safetensors, generation config, custom files, etc.
    snapshot_download(model_id)

    AutoConfig.from_pretrained(model_id)
    AutoTokenizer.from_pretrained(model_id)

    # Some text models do not have a processor; Qwen3.5 currently needs this path.
    try:
        AutoProcessor.from_pretrained(model_id)
    except Exception as e:
        print(f"No processor or processor failed for {model_id}: {e}")