#!/usr/bin/env bash
set -euo pipefail

rsync -az --delete --partial \
  -e "ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=6" \
  --exclude ".git/" \
  --exclude "__pycache__/" \
  --exclude ".venv" \
  --exclude ".pytest_cache/" \
  --exclude ".ruff_cache/" \
  --exclude "data/" \
  --exclude "notebooks/" \
  ./ jz:/lustre/fswork/projects/rech/lxa/ung58ii/simulatability_shortcut/
