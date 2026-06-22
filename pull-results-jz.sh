#!/usr/bin/env bash
set -euo pipefail

mkdir -p data

rsync -az \
  -e "ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6" \
  --include "*/" \
  --include "*.csv" \
  --exclude "*prompt*" \
  --exclude "*" \
  jz:/lustre/fswork/projects/rech/lxa/ung58ii/simulatability_shortcut/data/ \
  ./data/
