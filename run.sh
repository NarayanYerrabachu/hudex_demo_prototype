#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# Install deps if Pipfile.lock is missing or Pipfile changed
if [ ! -f Pipfile.lock ]; then
    echo "→ Generating Pipfile.lock..."
    pipenv install
fi

echo "→ Starting hudex-demo on http://localhost:8001"
cd patternengine
exec pipenv run uvicorn server:app --host 0.0.0.0 --port 8001 --reload
