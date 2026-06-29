#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# Install deps if Pipfile.lock is missing
if [ ! -f Pipfile.lock ]; then
    echo "→ Generating Pipfile.lock..."
    pipenv install
fi

# Extract prototype zip if not already done
if [ ! -d patternengine_prototype ]; then
    echo "→ Extracting prototype..."
    unzip -q pattern_engine_prototype.zip -d patternengine_prototype
fi

echo "→ Starting hudex-prototype on http://localhost:8003"
cd patternengine_prototype/patternengine
exec pipenv run uvicorn server:app --host 0.0.0.0 --port 8003 --reload
