"""FastAPI server — runs the pattern engine at startup, exposes results via REST."""
from __future__ import annotations

import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from engine.sample_data import make_corpus
from engine.core import PatternEngine
from ingestion.loader import load_bytes

# ---------------------------------------------------------------------------
# State — built once at startup, replaced atomically on upload
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_lock = threading.Lock()

def _build_state(docs):
    eng = PatternEngine().ingest(docs)
    result = eng.run()
    findings   = result["findings"]
    themes     = [f for f in findings if f["kind"] == "theme"]
    anomalies  = [f for f in findings if f["kind"] == "anomaly"]
    suspicious = [f for f in findings if f["kind"] == "suspicious"]
    rels       = [f for f in findings if f["kind"] == "relationship"]

    node_docs: dict[str, set[str]] = defaultdict(set)
    edges = []
    for r in rels:
        a, b, w = r["extra"]["a"], r["extra"]["b"], r["extra"]["weight"]
        edges.append({"a": a, "b": b, "w": w, "sources": r["sources"]})
        node_docs[a].update(r["sources"])
        node_docs[b].update(r["sources"])
    anom_by_doc = {f["sources"][0]: f["score"] for f in anomalies + suspicious}
    nodes = [
        {"id": name, "docs": len(ds), "exposure": round(sum(anom_by_doc.get(d, 0) for d in ds), 2)}
        for name, ds in node_docs.items()
    ]
    documents = [
        {
            "id": d.id, "text": d.text, "meta": d.meta,
            "anomaly": round(float(eng.anomaly_scores[i]), 3),
            "theme": int(eng.labels[i]),
        }
        for i, d in enumerate(docs)
    ]
    meta = {
        "n_docs": result["n_docs"],
        "theme_quality": result["theme_quality"],
        "n_themes": len(themes),
        "n_anomalies": len(anomalies),
        "n_suspicious": len(suspicious),
        "n_relationships": len(rels),
        "tda": result.get("tda", {"available": False, "betti_0": None, "betti_1": None,
                                   "max_persistence": None, "h1_features": []}),
    }
    return eng, documents, themes, anomalies, suspicious, rels, nodes, edges, meta

_eng, _documents, _themes, _anomalies, _suspicious, _rels, _nodes, _edges, _meta = \
    _build_state(make_corpus())

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="HuDex Pattern Intelligence")


@app.get("/", response_class=HTMLResponse)
def index():
    html = (_HERE.parent / "hudex_demo.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/api/findings")
def get_findings() -> dict[str, Any]:
    return {
        "meta": _meta,
        "themes": _themes,
        "anomalies": _anomalies,
        "suspicious": _suspicious,
        "relationships": _rels,
        "graph": {"nodes": _nodes, "edges": _edges},
    }


@app.get("/api/documents")
def get_documents() -> list[dict[str, Any]]:
    return _documents


class QueryRequest(BaseModel):
    q: str
    top: int = 8


@app.post("/api/query")
def run_query(req: QueryRequest) -> list[dict[str, Any]]:
    q = req.q.strip()
    if not q:
        raise HTTPException(status_code=400, detail="query is empty")
    hits = _eng.query(q, top=req.top)
    return [
        {
            "kind": "query",
            "title": h["doc_id"],
            "detail": h["snippet"],
            "score": h["score"],
            "sources": [h["doc_id"]],
            "meta": h["meta"],
            "_terms": q.lower().split(),
        }
        for h in hits
    ]


@app.post("/api/upload")
async def upload_corpus(file: UploadFile = File(...)) -> dict[str, Any]:
    """Accept a CSV / PDF / JSON / XML / TXT / XLSX file, re-run the engine,
    return the new findings immediately."""
    global _eng, _documents, _themes, _anomalies, _suspicious, _rels, _nodes, _edges, _meta

    data = await file.read()
    docs = load_bytes(data, file.filename or "upload.bin")
    if not docs:
        raise HTTPException(status_code=422, detail=f"No documents extracted from {file.filename!r}. "
                            "Supported: CSV, JSON, TXT, XML, PDF, DOCX, XLSX.")
    if len(docs) < 4:
        raise HTTPException(status_code=422, detail=f"Only {len(docs)} document(s) extracted — "
                            "need at least 4 for meaningful analysis.")

    with _lock:
        _eng, _documents, _themes, _anomalies, _suspicious, _rels, _nodes, _edges, _meta = \
            _build_state(docs)

    return {
        "ok": True,
        "filename": file.filename,
        "n_docs": len(docs),
        "findings": {
            "meta": _meta,
            "themes": _themes,
            "anomalies": _anomalies,
            "suspicious": _suspicious,
            "relationships": _rels,
            "graph": {"nodes": _nodes, "edges": _edges},
        },
        "documents": _documents,
    }
