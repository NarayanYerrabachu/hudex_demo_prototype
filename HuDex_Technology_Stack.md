# HuDex Pattern Intelligence — Technology Stack Documentation

## Overview

HuDex is a deterministic, offline pattern intelligence system for document corpus analysis.
It detects suspicious activity, anomalies, thematic clusters, and actor relationship networks
without any LLM, external API, or model download. Every finding carries a traceable reference
back to its source documents.

Three engine instances are deployed as separate Docker containers:

| Port | Name | Engine |
|------|------|--------|
| :8001 | Baseline Demo | `PatternEngine` (LOF-based) |
| :8003 | Baseline Alt | `PatternEngine` (LOF-based) — second instance |
| :8002 | TDA Engine | `TDAPatternEngine` (IsolationForest + Autoencoder + Topology) |

---

## Architecture

```
corpus.csv / uploaded file
        │
        ▼
  Document ingestion (Apache Tika + pdfplumber + pandas)
        │
        ▼
  TF-IDF vectorization (scikit-learn)
        │
        ├──────────────────────────┬─────────────────────────────┐
        ▼                          ▼                             ▼
  Theme discovery           Anomaly detection            Entity extraction
  (:8001/:8003 → KMeans)    (:8001/:8003 → LOF)          (regex — all engines)
  (:8002 → DBSCAN)          (:8002 → IsoForest +              │
                             Autoencoder + Topology)           ▼
                                                      Relationship graph
                                                      (actor co-occurrence)
                                                      (:8002 → Neo4j / in-memory)
        │
        ▼
  FastAPI REST server → HTML frontend (SVG graph, D3-style)
```

---

## Shared Infrastructure (all engines)

### Runtime
- **Python 3.11-slim** (Docker base image)
- **FastAPI + uvicorn** — REST API server
- **OpenJDK 21** — required by Apache Tika JVM

### Document Ingestion
- **Apache Tika** (`tika-python`) — multi-format extraction: PDF, DOCX, XML, TXT
- **pdfplumber** — PDF fallback parser
- **pandas + openpyxl** — CSV and XLSX ingestion
- **lxml** — XML parsing

### Vectorization
- **TF-IDF** (`sklearn.feature_extraction.text.TfidfVectorizer`)
  - `ngram_range=(1, 2)` — unigrams and bigrams
  - `max_df=0.6` — ignore terms in more than 60% of documents
  - `min_df=2` — ignore terms appearing in only one document
  - Stop words: scikit-learn English defaults + corpus-specific person name suppression

### Entity Extraction (regex, deterministic)
| Entity type | Pattern |
|-------------|---------|
| `actor` | Unicode TitleCase bigram: `[A-ZÄÖÜ][a-zäöüß]+ [A-ZÄÖÜ][a-zäöüß]+` |
| `money` | `USD/EUR/$/ €` followed by numeric amount |
| `email` | Standard RFC-like email regex |
| `account` | `ACC-`, `IBAN-`, `REF-` prefixed alphanumeric codes |
| `date` | ISO 8601: `YYYY-MM-DD` |

False-positive actor filter: first word of bigram checked against a blocklist
(`The`, `This`, `Per`, `All`, `No`, `Any`, etc.).

### Relationship Graph
- **Actor co-occurrence**: two actors share an edge if they appear together in ≥ 1 document
- Edge weight = number of shared documents
- Visualised as force-positioned SVG in the browser frontend
- Node colouring: **red** = suspicious exposure · **amber** = anomaly exposure · **dark** = clean

### Suspicious Vocabulary Signal
Keyword density over a fixed financial crime vocabulary:

```
"off the books", "off-book", "no invoice", "no trace", "kickback",
"bribe", "launder", "laundering", "shell entity", "shell company",
"untraceable", "offshore", "cash only", "settle in cash", "coded as", …
```

Documents with any keyword match are classified `suspicious` (not just `anomaly`).

---

## Engine :8001 / :8003 — Baseline (`PatternEngine`)

**File**: `patternengine/engine/core.py`

### Theme Discovery — KMeans
- `sklearn.cluster.KMeans`
- Silhouette sweep over k = 3 … 8; picks the k with highest silhouette score
- Forced to find clusters — always returns a partition

### Anomaly Detection — 3-signal blend

| Weight | Signal | Algorithm |
|--------|--------|-----------|
| 0.50 | LOF score | `sklearn.neighbors.LocalOutlierFactor` |
| 0.30 | Centroid distance | Cosine distance from corpus mean vector |
| 0.20 | Keyword density | Count of `SUSPICIOUS_VOCAB` matches |

```
anomaly_score = 0.50 × norm(LOF) + 0.30 × norm(centroid_dist) + 0.20 × norm(keyword_count)
```

Top `contamination=8%` fraction flagged as anomalies.

**Known limitation**: LOF is density-based. If fraud documents form a tight cluster
(similar vocabulary), LOF scores them as normal because their local density is high.
IsolationForest (used in :8002) avoids this blind spot.

### Corpus-level Topology (display only, not used in scoring)
- **ripser** — Vietoris-Rips persistent homology on a 60-document sample
- Computes Betti-0 (connected components) and Betti-1 (loops/cycles)
- Exposed in the API response as a corpus fingerprint; not fed into anomaly scoring

### Python dependencies
```
scikit-learn, numpy, networkx, fastapi, uvicorn, ripser,
tika, pdfplumber, pandas, openpyxl, lxml
```

---

## Engine :8002 — TDA Engine (`TDAPatternEngine`)

**File**: `patternengine/engine/core_tda.py`

### Theme Discovery — DBSCAN
- `sklearn.cluster.DBSCAN`
- No need to pre-specify k — density determines cluster count
- `eps` sweep: [0.35, 0.45, 0.55, 0.65] with `min_samples=3`, `metric="cosine"`
- Picks eps that yields the most clusters
- Documents with no cluster assignment labelled as noise (cluster `-1`)

### Anomaly Detection — 4-signal blend

| Weight | Signal | Algorithm |
|--------|--------|-----------|
| 0.30 | Isolation Forest | `sklearn.ensemble.IsolationForest` (200 estimators) |
| 0.25 | Autoencoder error | PyTorch MLP reconstruction MSE |
| 0.30 | Keyword density | Count of `SUSPICIOUS_VOCAB` matches |
| 0.15 | Topology entropy | Per-document H1 persistence entropy (giotto-tda equivalent) |

```
anomaly_score = 0.30 × norm(IsoForest)
              + 0.25 × norm(AE_error)
              + 0.30 × norm(keyword_count)
              + 0.15 × norm(topo_entropy_deviation)
```

#### Signal 1 — Isolation Forest
- `IsolationForest(n_estimators=200, contamination=0.08)`
- Evaluates each document **independently** using random feature partitioning
- Correctly flags isolated fraud documents even when they form a tight cluster
  (unlike LOF which would score them as normal)

#### Signal 2 — Autoencoder (PyTorch MLP)

Architecture:
```
Input (TF-IDF dim)
  → Linear(dim, 128) → ReLU
  → Linear(128, 32)  → ReLU   ← bottleneck
  → Linear(32, 128)  → ReLU
  → Linear(128, dim)           ← reconstruction
```

- Trained for 120 epochs, Adam optimizer (lr=3e-3), MSE loss
- Reconstruction error per document = anomaly signal
- Runs on CPU — no GPU required
- Falls back to zeros if PyTorch is unavailable

#### Signal 3 — Keyword Density
Same `SUSPICIOUS_VOCAB` match count as the baseline engine.

#### Signal 4 — Per-document Topology Entropy (giotto-tda equivalent)

**Why**: giotto-tda's `VietorisRipsPersistence + PersistenceEntropy` pipeline is the
standard approach, but its wheel build fails inside Docker due to git submodule
compilation at install time. The same mathematics are implemented directly via ripser.

**Algorithm**:
1. Compute full cosine distance matrix for all documents
2. For each document i, take its k=10 nearest neighbours → build (k+1)×(k+1) subgraph
3. Run ripser H1 (1-cycles) on the subgraph distance matrix
4. Compute persistence entropy of the H1 barcode:
   ```
   entropy = -Σ p_i × log(p_i)   where p_i = persistence_i / total_persistence
   ```
5. Return each document's deviation from the corpus median entropy
6. High deviation = anomalously complex local topological neighbourhood

**Exposed in UI**: each anomaly/suspicious card shows a `topo:` chip with the entropy value.

### Relationship Storage — Neo4j (with in-memory fallback)
- **neo4j Python driver** — stores actor co-occurrence as a property graph
- Bolt connection: `bolt://neo4j:7687`
- Falls back to in-memory list if Neo4j is unreachable
- Node: `Actor {name}` · Edge: `CO_OCCURS {weight}`

### Corpus-level Topology Fingerprint
Same ripser H0/H1 computation as the baseline engine — corpus-level Betti numbers
exposed in the API as a global structure fingerprint.

### Python dependencies
```
scikit-learn, numpy, networkx, fastapi, uvicorn, ripser, persim,
torch (CPU wheel), neo4j, tika, pdfplumber, pandas, openpyxl, lxml
```

---

## Engine Comparison

| Capability | Baseline (:8001/:8003) | TDA (:8002) |
|------------|------------------------|-------------|
| Vectorization | TF-IDF | TF-IDF |
| Clustering | KMeans (forced k) | DBSCAN (density, no k) |
| Outlier detection | LOF + centroid distance | IsolationForest |
| Deep anomaly signal | — | Autoencoder (PyTorch) |
| Topology anomaly signal | — | Per-doc H1 entropy |
| Keyword signal | Yes | Yes |
| Anomaly signals | 3 | 4 |
| Fraud cluster blind spot | Yes (LOF misses tight clusters) | No (IsoForest is independent) |
| Relationship storage | In-memory | Neo4j (in-memory fallback) |
| Python-only | Yes | Yes |
| GPU required | No | No |
| LLM / external API | None | None |

---

## Deployment

Both engines are built as Docker images and run behind separate ports.

### Baseline image (`hudex-baseline`)
```dockerfile
FROM python:3.11-slim
RUN apt-get install gcc openjdk-21-jre-headless
RUN pip install scikit-learn numpy networkx fastapi uvicorn ripser \
    tika pdfplumber pandas openpyxl lxml
EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### TDA image (`hudex-tda`)
```dockerfile
FROM python:3.11-slim
RUN apt-get install gcc g++ cmake git openjdk-21-jre-headless
RUN pip install scikit-learn numpy networkx fastapi uvicorn ripser persim \
    neo4j tika pdfplumber pandas openpyxl lxml
RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
EXPOSE 8002
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8002"]
```

### Running the containers
```bash
# Baseline on :8001 and :8003
docker run -d --name hudex-demo     -p 8001:8000 hudex-baseline
docker run -d --name hudex-baseline -p 8003:8000 hudex-baseline

# TDA on :8002
docker run -d --name hudex-tda -p 8002:8002 hudex-tda
```

---

## API Endpoints (all engines)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Serve the HTML frontend |
| `GET` | `/api/findings` | All findings: themes, anomalies, suspicious, relationships, graph |
| `GET` | `/api/documents` | Full document list with anomaly scores and theme labels |
| `POST` | `/api/query` | Free-text corpus search — returns ranked documents with snippets |
| `POST` | `/api/upload` | Replace the corpus with a new file (CSV/JSON/PDF/DOCX/XLSX/TXT) |

### `/api/findings` response shape
```json
{
  "meta": {
    "n_docs": 112,
    "theme_quality": 0.18,
    "n_themes": 8,
    "n_anomalies": 1,
    "n_suspicious": 8,
    "n_relationships": 22
  },
  "themes": [...],
  "anomalies": [...],
  "suspicious": [...],
  "relationships": [...],
  "graph": {
    "nodes": [{"id": "Marcus Vale", "docs": 18, "exposure": 4.12}],
    "edges": [{"a": "Marcus Vale", "b": "Omar Faris", "w": 4, "sources": ["DOC-0002", ...]}]
  }
}
```

---

## Corpus Format

Default corpus: 112 documents in `patternengine/corpus.csv`.

| Column | Description |
|--------|-------------|
| `id` | Unique document identifier (e.g. `DOC-0042`) |
| `text` | Full document text |
| `channel` | Source channel (`procurement`, `finance_routine`, `logistics`, `hiring`, `comms_misc`) |
| `date` | ISO 8601 date |

Embedded fraud pattern: 8 `comms_misc` documents (DOC-0104 … DOC-0111) contain explicit
financial crime vocabulary and form the ground truth for the suspicious signal.
