#!/usr/bin/env bash

module purge
module load arch/h100
module load uv/0.8.3
module load cuda/13.0.3

source .venv/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
