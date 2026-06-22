#!/usr/bin/env bash
set -euo pipefail

echo "Watching files. Press Ctrl+C to stop."

while true; do
  inotifywait -r -e close_write,create,delete,move \
    --exclude '(\.git|__pycache__|\.venv|\.pytest_cache|\.ruff_cache|data)' \
    .

  sleep 1

  echo "Syncing to Jean Zay..."
  ./sync-jz.sh
  echo "Done."
done
