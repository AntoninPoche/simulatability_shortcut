#!/usr/bin/env bash
#SBATCH --job-name=manifest
#SBATCH -A lxa@h100
#SBATCH -C h100
#SBATCH --qos=qos_gpu_h100-t3  # -t3 for longer jobs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=data/logs/%x_%A_%a.out
#SBATCH --error=data/logs/%x_%A_%a.err

set -euo pipefail

MANIFEST="${1:?Usage: sbatch --array=1-N%K manifest.sbatch <manifest.tsv>}"

cd /lustre/fswork/projects/rech/lxa/ung58ii/simulatability_shortcut
source ~/.bashrc
source setup_cluster.sh

if [ ! -f "$MANIFEST" ]; then
  echo "Manifest not found: $MANIFEST"
  exit 1
fi

CMD="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")"

if [ -z "$CMD" ]; then
  echo "No command for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID in $MANIFEST"
  exit 1
fi

echo "job id: $SLURM_JOB_ID"
echo "array job id: $SLURM_ARRAY_JOB_ID"
echo "array task id: $SLURM_ARRAY_TASK_ID"
echo "node: $(hostname)"
echo "manifest: $MANIFEST"
echo "command: $CMD"

eval "$CMD"
