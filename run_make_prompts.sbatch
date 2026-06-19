#!/usr/bin/env bash
#SBATCH --job-name=make-prompts
#SBATCH -A lxa@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=data/logs/%x_%j.out
#SBATCH --error=data/logs/%x_%j.err

set -euo pipefail

cd /lustre/fswork/projects/rech/lxa/ung58ii/simulatability_shortcut
source ~/.bashrc
source setup_cluster.sh

echo "hostname: $(hostname)"
echo "job id: $SLURM_JOB_ID"
echo "command: python scripts/make_prompts.py concepts RT seminmf --interpretation topk"

nvidia-smi

python scripts/make_prompts.py concepts RT seminmf --interpretation topk
