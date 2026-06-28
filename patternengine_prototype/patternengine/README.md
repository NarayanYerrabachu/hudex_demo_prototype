# Pattern-Intelligence Engine — Wedge Prototype

A working, **offline, deterministic** core for an Arlequin/HuDex-style platform:
ingest unstructured records → find themes, anomalies and relationships → answer
natural-language-ish queries, with **every finding traceable to its source**.
No LLM, no model downloads, no training labels, no hallucination surface.

This is the Phase-1 "core intelligence engine" — the honest 20% that's buildable
fast. The moat (analytical depth at scale + sovereign go-to-market) is the other 80%.

## What's real
- `engine/core.py` — the pipeline:
  - TF-IDF vectorisation of the whole corpus (no sampling)
  - **Themes**: KMeans with silhouette-selected k
  - **Anomalies**: Local Outlier Factor + centroid-distance ensemble
    (more robust on text than Isolation Forest alone — see git history reasoning)
  - **Entities + relationships**: regex extraction → co-occurrence graph
  - **Query**: deterministic TF-IDF similarity, every hit returns its source id
- `engine/sample_data.py` — synthetic due-diligence corpus with a *planted*
  off-book payment scheme, so the engine has real signal to surface.
- `export.py` — runs everything, writes `findings.json`.
- `hudex_demo.html` — analyst workspace UI built on the real output. Selecting any
  finding draws a trail to the source records it came from.

## Run it
```bash
pip install scikit-learn numpy networkx
python export.py          # regenerates findings.json
# open hudex_demo.html in a browser
```

## What's stubbed (deliberately, for the demo)
- Regex "entities" stand in for proper NER.
- Synthetic corpus stands in for real connectors (PDF/CSV/email/db).
- In-memory only; no persistence, auth, or multi-tenant layer yet.

## If this becomes the product, the next real decisions
1. Pick ONE vertical + ONE buyer you can reach (mid-market, not government, to start).
2. Replace TF-IDF anomaly detection with the method that actually beats the
   status quo on *your* data — this is where a research hire earns their keep.
3. Prove the traceability/explainability UX to a design partner before scaling code.
