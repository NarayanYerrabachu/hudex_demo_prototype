"""Minimal FastAPI server for the prototype — serves pre-computed findings.json."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from engine.sample_data import make_corpus
from engine.core import PatternEngine

_HERE = Path(__file__).parent
_eng: PatternEngine | None = None
_data: dict = {}


def _build():
    global _eng, _data
    docs = make_corpus()
    _eng = PatternEngine().ingest(docs)
    result = _eng.run()
    findings = result["findings"]
    themes     = [f for f in findings if f["kind"] == "theme"]
    anomalies  = [f for f in findings if f["kind"] == "anomaly"]
    suspicious = [f for f in findings if f["kind"] == "suspicious"]
    rels       = [f for f in findings if f["kind"] == "relationship"]

    # Build edges from relationship co-occurrences
    node_cooc: dict[str, set[str]] = defaultdict(set)
    edges = []
    for r in rels:
        a, b, w = r["extra"]["a"], r["extra"]["b"], r["extra"]["weight"]
        edges.append({"a": a, "b": b, "w": w, "sources": r["sources"]})
        node_cooc[a].update(r["sources"])
        node_cooc[b].update(r["sources"])

    # Build actor -> all docs they appear in (from entity extraction, not just co-occurrences)
    actor_all_docs: dict[str, set[str]] = defaultdict(set)
    for doc_id, ents in _eng.entities.items():
        for etype, val in ents:
            if etype == "actor":
                actor_all_docs[val].add(doc_id)

    susp_docs  = {f["sources"][0] for f in suspicious}
    anom_docs  = {f["sources"][0] for f in anomalies}
    anom_score = {f["sources"][0]: f["score"] for f in anomalies + suspicious}
    nodes = [
        {
            "id": name,
            "docs": len(node_cooc[name]),
            "exposure":      round(sum(anom_score.get(d, 0) for d in actor_all_docs.get(name, node_cooc[name])), 2),
            "susp_exposure": round(sum(anom_score.get(d, 0) for d in actor_all_docs.get(name, node_cooc[name]) if d in susp_docs), 2),
            "anom_exposure": round(sum(anom_score.get(d, 0) for d in actor_all_docs.get(name, node_cooc[name]) if d in anom_docs), 2),
        }
        for name in node_cooc
    ]
    documents = [
        {"id": d.id, "text": d.text, "meta": d.meta,
         "anomaly": round(float(_eng.anomaly_scores[i]), 3),
         "theme": int(_eng.labels[i])}
        for i, d in enumerate(docs)
    ]
    _data = {
        "meta": {
            "n_docs": result["n_docs"],
            "theme_quality": result["theme_quality"],
            "n_themes": len(themes),
            "n_anomalies": len(anomalies),
            "n_suspicious": len(suspicious),
            "n_relationships": len(rels),
            "tda": result.get("tda", {"available": False}),
        },
        "themes":        themes,
        "anomalies":     anomalies,
        "suspicious":    suspicious,
        "relationships": rels,
        "graph":         {"nodes": nodes, "edges": edges},
        "documents":     documents,
    }


_build()

app = FastAPI()


@app.get("/", response_class=HTMLResponse)
def index():
    html = (_HERE.parent.parent / "hudex_demo.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/api/findings")
def get_findings() -> dict[str, Any]:
    return {k: v for k, v in _data.items() if k != "documents"}


@app.get("/api/documents")
def get_documents():
    return _data["documents"]


class QueryRequest(BaseModel):
    q: str
    top: int = 8


@app.post("/api/query")
def run_query(req: QueryRequest):
    if not req.q.strip():
        raise HTTPException(status_code=400, detail="query is empty")
    hits = _eng.query(req.q.strip(), top=req.top)
    return [
        {
            "kind": "query",
            "title": h["doc_id"],
            "detail": h["snippet"],
            "score": h["score"],
            "sources": [h["doc_id"]],
            "meta": h["meta"],
            "_terms": req.q.lower().split(),
        }
        for h in hits
    ]
