#!/bin/bash
set -e

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"

# Install deps if needed (prototype only needs scikit-learn numpy)
if ! python3 -c "import sklearn, numpy" 2>/dev/null; then
    echo "→ Installing prototype dependencies..."
    pip install --quiet scikit-learn numpy
fi

# Extract prototype zip if not already done
if [ ! -d patternengine_prototype ]; then
    echo "→ Extracting prototype..."
    unzip -q pattern_engine_prototype.zip -d patternengine_prototype
fi

echo "→ Running engine — regenerating findings.json..."
cd patternengine_prototype/patternengine
python3 export.py

echo "→ Serving prototype UI on http://localhost:8003"
echo "   Open http://localhost:8003/hudex_demo.html"
cd "$REPO_DIR"
exec python3 -m http.server 8003
