"""Run the engine and export everything the demo UI needs as a single JSON blob."""
import json
from collections import defaultdict
from engine.sample_data import make_corpus
from engine.core import PatternEngine

docs = make_corpus()
eng = PatternEngine().ingest(docs)
result = eng.run()

findings = result["findings"]
themes = [f for f in findings if f["kind"] == "theme"]
anoms = [f for f in findings if f["kind"] == "anomaly"]
rels = [f for f in findings if f["kind"] == "relationship"]

# graph nodes/edges from relationship findings
node_docs = defaultdict(set)
edges = []
for r in rels:
    a, b, w = r["extra"]["a"], r["extra"]["b"], r["extra"]["weight"]
    edges.append({"a": a, "b": b, "w": w, "sources": r["sources"]})
    node_docs[a].update(r["sources"])
    node_docs[b].update(r["sources"])
# node weight = total anomaly exposure of its documents (so insiders glow)
anom_by_doc = {a["sources"][0]: a["score"] for a in anoms}
nodes = []
for name, dset in node_docs.items():
    exposure = sum(anom_by_doc.get(d, 0) for d in dset)
    nodes.append({"id": name, "docs": len(dset), "exposure": round(exposure, 2)})

# precompute a couple of demo queries with traceable hits
demo_queries = {}
for q in ["off the books cash payment", "shipment delay customs", "new hire onboarding"]:
    demo_queries[q] = eng.query(q, top=5)

export = {
    "meta": {
        "n_docs": result["n_docs"],
        "theme_quality": result["theme_quality"],
        "n_themes": len(themes),
        "n_anomalies": len(anoms),
        "n_relationships": len(rels),
    },
    "documents": [{"id": d.id, "text": d.text, "meta": d.meta,
                   "anomaly": round(float(eng.anomaly_scores[i]), 3),
                   "theme": int(eng.labels[i])}
                  for i, d in enumerate(docs)],
    "themes": themes,
    "anomalies": anoms,
    "relationships": rels,
    "graph": {"nodes": nodes, "edges": edges},
    "queries": demo_queries,
}

with open("findings.json", "w") as f:
    json.dump(export, f, indent=2)

print("wrote findings.json")
print(json.dumps(export["meta"], indent=2))
