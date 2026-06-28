"""
Pattern-intelligence engine — core analysis pipeline.

Design goals (the product thesis):
  - Process the WHOLE corpus, not a sample.
  - Deterministic + offline. No LLM, no hallucination, no model downloads.
  - Every finding carries references back to the exact source documents.

The pipeline:
  ingest -> vectorize (TF-IDF) -> themes (KMeans) -> anomalies (IsolationForest)
         -> entities (regex) -> relationship graph (co-occurrence) -> query
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

from engine.patterns_loader import load_patterns

# --------------------------------------------------------------------------
# Persistent homology via ripser (optional)
# --------------------------------------------------------------------------
try:
    from ripser import ripser as _ripser
    _RIPSER = True
except ImportError:
    _RIPSER = False


def _tda_fingerprint(Xd: np.ndarray) -> dict:
    if not _RIPSER:
        return {"betti_0": None, "betti_1": None, "max_persistence": None,
                "h1_features": [], "available": False}
    sample = Xd if len(Xd) <= 60 else Xd[np.random.default_rng(42).choice(len(Xd), 60, replace=False)]
    result = _ripser(sample, maxdim=1, metric="cosine")
    dgms = result["dgms"]
    h0 = dgms[0]
    betti_0 = int((h0[:, 1] == np.inf).sum()) if len(h0) else 1
    h1 = dgms[1] if len(dgms) > 1 else np.empty((0, 2))
    finite_h1 = h1[h1[:, 1] < np.inf]
    persistence = (finite_h1[:, 1] - finite_h1[:, 0]) if len(finite_h1) else np.array([])
    betti_1 = len(finite_h1)
    max_pers = float(persistence.max()) if len(persistence) else 0.0
    top5 = sorted(zip(finite_h1[:, 0].tolist(), finite_h1[:, 1].tolist()),
                  key=lambda x: x[1] - x[0], reverse=True)[:5]
    return {
        "betti_0": betti_0,
        "betti_1": betti_1,
        "max_persistence": round(max_pers, 4),
        "h1_features": [{"birth": round(b, 4), "death": round(d, 4), "persistence": round(d - b, 4)}
                        for b, d in top5],
        "available": True,
    }


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Document:
    id: str
    text: str
    meta: dict[str, Any] = field(default_factory=dict)  # date, author, channel...


# Loaded from patternengine/patterns.yml (or built-in defaults if file absent).
# Set HUDEX_PATTERNS_FILE=/path/to/patterns.yml to override at runtime.
ENTITY_PATTERNS, SUSPICIOUS_VOCAB = load_patterns()


@dataclass
class Finding:
    kind: str                 # "theme" | "anomaly" | "suspicious" | "relationship"
    title: str
    detail: str
    score: float              # higher = more notable
    sources: list[str]        # document ids — the audit trail
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class PatternEngine:
    def __init__(self, random_state: int = 7):
        self.random_state = random_state
        self.docs: list[Document] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.X = None
        self.terms: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.anomaly_scores: np.ndarray | None = None
        self.entities: dict[str, list[tuple[str, str]]] = {}  # doc_id -> [(type, value)]

    # ---- ingest ----------------------------------------------------------
    def ingest(self, docs: list[Document]) -> "PatternEngine":
        self.docs = docs
        return self

    # ---- vectorize -------------------------------------------------------
    # German personal names common in the corpus — suppress so themes reflect
    # topics (finance, logistics, procurement) not the person who signed each doc
    _NAME_STOPWORDS = {
        "anna", "müller", "müller", "anna müller",
        "stefan", "braun", "stefan braun",
        "thomas", "fischer", "thomas fischer",
        "julia", "lange", "julia lange",
        "eva", "hoffmann", "eva hoffmann",
        "maria", "schmidt", "maria schmidt",
        "klaus", "weber", "klaus weber",
        "priya", "anand", "priya anand",
        "omar", "faris", "omar faris",
        "marcus", "vale", "marcus vale",
        "yuki", "sato", "yuki sato",
        "lena", "roth", "lena roth",
        "david", "okoye", "david okoye",
        "tomas", "berg", "tomas berg",
        "sara", "klein", "sara klein",
    }

    def vectorize(self) -> "PatternEngine":
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        stop = list(set(ENGLISH_STOP_WORDS) | self._NAME_STOPWORDS)
        self.vectorizer = TfidfVectorizer(
            stop_words=stop, max_df=0.6, min_df=2, ngram_range=(1, 2)
        )
        self.X = self.vectorizer.fit_transform(d.text for d in self.docs)
        self.terms = np.array(self.vectorizer.get_feature_names_out())
        return self

    # ---- themes (unsupervised clustering) --------------------------------
    def find_themes(self, k_range=range(3, 9)) -> "PatternEngine":
        Xd = self.X.toarray()
        best_k, best_score, best_labels = None, -1.0, None
        for k in k_range:
            if k >= len(self.docs):
                break
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(Xd)
            if len(set(labels)) < 2:
                continue
            score = silhouette_score(Xd, labels)
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels
        self.labels = best_labels
        self._theme_quality = best_score
        return self

    def _top_terms(self, doc_indices, n=6) -> list[str]:
        if len(doc_indices) == 0:
            return []
        centroid = np.asarray(self.X[doc_indices].mean(axis=0)).ravel()
        top = centroid.argsort()[::-1][:n]
        return [self.terms[i] for i in top if centroid[i] > 0]

    def _keyword_hits(self, i) -> list[str]:
        """Return which SUSPICIOUS_VOCAB phrases appear in document i (case-insensitive)."""
        text = self.docs[i].text.lower()
        return [kw for kw in SUSPICIOUS_VOCAB if kw in text]

    def _distinctive_terms(self, i, n=6) -> list[str]:
        """Why does document i stand out? Return the terms it over-uses most
        relative to the corpus average. Fully data-derived — no keyword list,
        and it mirrors the centroid-distance signal that flagged the doc."""
        row = np.asarray(self.X[i].todense()).ravel()
        mean = np.asarray(self.X.mean(axis=0)).ravel()
        delta = row - mean
        top = delta.argsort()[::-1][:n]
        return [self.terms[j] for j in top if delta[j] > 0]

    # ---- anomalies (unsupervised, no labels) -----------------------------
    def find_anomalies(self, contamination=0.08) -> "PatternEngine":
        """Combine two complementary unsupervised signals:
          - Local Outlier Factor: catches small dense groups sitting apart
            from the main distribution (the classic 'hidden cluster').
          - Distance from the corpus centroid: catches documents built from
            vocabulary that's rare across the whole corpus.
        Both are deterministic and explainable — no labels, no training set."""
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.metrics.pairwise import cosine_distances

        Xd = self.X.toarray()
        n_neighbors = min(15, max(5, len(Xd) // 10))
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, metric="cosine")
        lof.fit(Xd)
        lof_score = -lof.negative_outlier_factor_
        cdist = cosine_distances(Xd, Xd.mean(axis=0, keepdims=True)).ravel()

        def norm(v):
            return (v - v.min()) / (np.ptp(v) + 1e-9)

        # 3rd signal: keyword density — docs containing suspicious phrases score higher
        kw_counts = np.array([len(self._keyword_hits(i)) for i in range(len(self.docs))], dtype=float)

        self.anomaly_scores = 0.5 * norm(lof_score) + 0.3 * norm(cdist) + 0.2 * norm(kw_counts)
        # flag the top `contamination` fraction
        cutoff = np.quantile(self.anomaly_scores, 1 - contamination)
        self._anomaly_flag = self.anomaly_scores >= cutoff
        return self

    # ---- entities + relationships ----------------------------------------
    def extract_entities(self) -> "PatternEngine":
        self.entities = {}
        for d in self.docs:
            found = []
            for etype, pat in ENTITY_PATTERNS.items():
                for m in pat.findall(d.text):
                    val = m.strip()
                    if etype == "actor" and val.split()[0] in {
                        "The", "This", "Our", "Per", "All", "No", "Any", "Each",
                        "Both", "Low", "High", "New", "Old", "Key", "Standard",
                        "Follow", "Quick", "Noted", "Added", "Same", "Strong",
                    }:
                        continue
                    found.append((etype, val))
            self.entities[d.id] = found
        return self

    def build_relationships(self, min_shared=1) -> list[Finding]:
        # entity -> set of docs it appears in
        ent_docs: dict[tuple[str, str], set[str]] = defaultdict(set)
        for doc_id, ents in self.entities.items():
            for e in ents:
                ent_docs[e].add(doc_id)

        # co-occurrence between actor-type entities sharing documents
        actors = {e: docs for e, docs in ent_docs.items() if e[0] == "actor" and len(docs) >= 2}
        rels: list[Finding] = []
        actor_list = list(actors.items())
        for i in range(len(actor_list)):
            for j in range(i + 1, len(actor_list)):
                (e1, d1), (e2, d2) = actor_list[i], actor_list[j]
                shared = d1 & d2
                if len(shared) >= min_shared:
                    rels.append(Finding(
                        kind="relationship",
                        title=f"{e1[1]} ↔ {e2[1]}",
                        detail=f"Co-occur in {len(shared)} documents.",
                        score=float(len(shared)),
                        sources=sorted(shared),
                        extra={"a": e1[1], "b": e2[1], "weight": len(shared)},
                    ))
        rels.sort(key=lambda f: f.score, reverse=True)
        return rels

    # ---- assemble findings ----------------------------------------------
    def findings(self) -> list[Finding]:
        out: list[Finding] = []

        # themes
        for c in sorted(set(self.labels)):
            idx = np.where(self.labels == c)[0]
            terms = self._top_terms(idx)
            out.append(Finding(
                kind="theme",
                title="Theme: " + ", ".join(terms[:3]),
                detail=f"{len(idx)} documents share this pattern. Key terms: "
                       + ", ".join(terms),
                score=float(len(idx)),
                sources=[self.docs[i].id for i in idx],
                extra={"terms": terms, "size": int(len(idx))},
            ))

        # anomalies
        flagged = np.where(self._anomaly_flag)[0]
        order = flagged[np.argsort(self.anomaly_scores[flagged])[::-1]]
        for i in order:
            d = self.docs[i]
            why = self._distinctive_terms(i)
            keywords = self._keyword_hits(i)
            kind = "suspicious" if keywords else "anomaly"
            title_prefix = "Suspicious" if keywords else "Anomaly"
            out.append(Finding(
                kind=kind,
                title=f"{title_prefix} in {d.id}",
                detail=d.text[:140] + ("…" if len(d.text) > 140 else ""),
                score=float(self.anomaly_scores[i]),
                sources=[d.id],
                extra={"meta": d.meta, "terms": why, "keywords": keywords},
            ))

        out.extend(self.build_relationships())
        return out

    # ---- traceable query -------------------------------------------------
    def query(self, q: str, top=5) -> list[dict[str, Any]]:
        """Rank documents by similarity to a free-text query. Deterministic,
        and every hit returns its source id + the matching snippet."""
        qv = self.vectorizer.transform([q])
        sims = (self.X @ qv.T).toarray().ravel()
        order = sims.argsort()[::-1][:top]
        results = []
        for i in order:
            if sims[i] <= 0:
                continue
            d = self.docs[i]
            results.append({
                "doc_id": d.id,
                "score": round(float(sims[i]), 4),
                "snippet": d.text,
                "meta": d.meta,
                "theme": int(self.labels[i]) if self.labels is not None else None,
                "anomaly_score": round(float(self.anomaly_scores[i]), 4)
                if self.anomaly_scores is not None else None,
            })
        return results

    # ---- topological fingerprint -----------------------------------------
    def find_topology(self) -> "PatternEngine":
        self._tda = _tda_fingerprint(self.X.toarray())
        return self

    # ---- run everything --------------------------------------------------
    def run(self) -> dict[str, Any]:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.vectorize().find_themes().find_anomalies().find_topology().extract_entities()
        return {
            "n_docs": len(self.docs),
            "theme_quality": round(float(self._theme_quality), 3),
            "tda": self._tda,
            "findings": [asdict(f) for f in self.findings()],
        }
