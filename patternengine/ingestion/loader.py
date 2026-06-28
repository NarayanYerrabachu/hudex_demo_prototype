"""
Multi-format document loader.

Supported formats:
  CSV     — native (csv module)
  JSON    — native (json module), list of objects or single object
  TXT     — plain text, one doc per file
  XML     — native (ElementTree), text content of leaf elements
  PDF     — Apache Tika (requires JVM) or pdfplumber fallback
  DOCX    — Apache Tika
  XLSX    — pandas openpyxl
  *       — Apache Tika catches everything else

Tika is used in server mode (tika.parser.from_file) so the JVM starts once.
Falls back gracefully to pdfplumber for PDFs when Tika is unavailable.
"""

from __future__ import annotations

import csv
import json
import re
import uuid
from pathlib import Path
from typing import Any

# Document contract — import from whichever engine is in scope
try:
    from engine.core_tda import Document
except ImportError:
    from engine.core import Document  # type: ignore[no-redef]

# --------------------------------------------------------------------------
# Optional deps — degrade gracefully
# --------------------------------------------------------------------------
try:
    from tika import parser as tika_parser
    _TIKA = True
except ImportError:
    _TIKA = False

try:
    import pdfplumber
    _PDFPLUMBER = True
except ImportError:
    _PDFPLUMBER = False

try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

try:
    import xml.etree.ElementTree as ET
    _ET = True
except ImportError:
    _ET = False


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Normalise whitespace; strip null bytes."""
    return re.sub(r"\s+", " ", text.replace("\x00", "")).strip()


def _make_id(prefix: str, i: int) -> str:
    return f"{prefix}-{i:04d}"


# --------------------------------------------------------------------------
# Format handlers
# --------------------------------------------------------------------------

def _load_csv(path: Path) -> list[Document]:
    docs = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # try common column names; fall back to joining all values
            text = (row.get("text") or row.get("content") or row.get("body")
                    or row.get("message") or row.get("description") or "")
            if not text:
                text = " ".join(str(v) for v in row.values() if v)
            doc_id = row.get("id") or row.get("doc_id") or _make_id(path.stem, i)
            docs.append(Document(
                id=str(doc_id),
                text=_clean(text),
                meta={
                    "channel": row.get("channel", "csv"),
                    "date": row.get("date", ""),
                    "source_file": path.name,
                },
            ))
    return docs


def _load_json(path: Path) -> list[Document]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = [data]
    docs = []
    for i, obj in enumerate(data):
        text = (obj.get("text") or obj.get("content") or obj.get("body")
                or obj.get("message") or obj.get("description") or "")
        if not text:
            text = json.dumps(obj)
        doc_id = obj.get("id") or obj.get("doc_id") or _make_id(path.stem, i)
        docs.append(Document(
            id=str(doc_id),
            text=_clean(text),
            meta={
                "channel": obj.get("channel", "json"),
                "date": obj.get("date", ""),
                "source_file": path.name,
            },
        ))
    return docs


def _load_txt(path: Path) -> list[Document]:
    text = _clean(path.read_text(encoding="utf-8", errors="replace"))
    return [Document(
        id=_make_id(path.stem, 0),
        text=text,
        meta={"channel": "txt", "date": "", "source_file": path.name},
    )]


def _load_xml(path: Path) -> list[Document]:
    docs = []
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        # collect text from every element that has no children (leaf nodes)
        texts = []
        for el in root.iter():
            if not list(el) and el.text and el.text.strip():
                texts.append(el.text.strip())
        combined = _clean(" ".join(texts))
        if combined:
            docs.append(Document(
                id=_make_id(path.stem, 0),
                text=combined,
                meta={"channel": "xml", "date": "", "source_file": path.name},
            ))
    except ET.ParseError:
        pass
    return docs


def _load_xlsx(path: Path) -> list[Document]:
    if not _PANDAS:
        return []
    docs = []
    df = pd.read_excel(path, engine="openpyxl")
    # try to find a text-like column
    text_col = next(
        (c for c in df.columns if c.lower() in ("text", "content", "body", "message", "description")),
        None,
    )
    for i, row in df.iterrows():
        text = str(row[text_col]) if text_col else " ".join(str(v) for v in row.values)
        doc_id = str(row.get("id", row.get("doc_id", _make_id(path.stem, i))))
        docs.append(Document(
            id=doc_id,
            text=_clean(text),
            meta={
                "channel": str(row.get("channel", "xlsx")),
                "date": str(row.get("date", "")),
                "source_file": path.name,
            },
        ))
    return docs


def _load_via_tika(path: Path) -> list[Document]:
    """Use Apache Tika for PDF, DOCX, and any other binary format."""
    parsed = tika_parser.from_file(str(path))
    content = (parsed.get("content") or "").strip()
    if not content:
        return []
    # split on form-feed or large blank blocks to get page-level docs
    pages = [p for p in re.split(r"\f|\n{4,}", content) if p.strip()]
    docs = []
    for i, page in enumerate(pages):
        docs.append(Document(
            id=_make_id(path.stem, i),
            text=_clean(page),
            meta={
                "channel": path.suffix.lstrip(".").lower(),
                "date": "",
                "source_file": path.name,
            },
        ))
    return docs


def _load_pdf_pdfplumber(path: Path) -> list[Document]:
    docs = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = _clean(page.extract_text() or "")
            if text:
                docs.append(Document(
                    id=_make_id(path.stem, i),
                    text=text,
                    meta={"channel": "pdf", "date": "", "source_file": path.name},
                ))
    return docs


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def load_file(path: str | Path) -> list[Document]:
    """Load any supported file into a list of Documents."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return _load_csv(path)
    if suffix == ".json":
        return _load_json(path)
    if suffix in (".txt", ".text", ".log"):
        return _load_txt(path)
    if suffix == ".xml":
        return _load_xml(path)
    if suffix in (".xlsx", ".xls"):
        return _load_xlsx(path)
    if suffix == ".pdf":
        if _TIKA:
            return _load_via_tika(path)
        if _PDFPLUMBER:
            return _load_pdf_pdfplumber(path)
        return []
    # everything else — try Tika
    if _TIKA:
        return _load_via_tika(path)
    return []


def load_bytes(data: bytes, filename: str) -> list[Document]:
    """Load an uploaded file from raw bytes (for FastAPI UploadFile)."""
    import tempfile, os
    suffix = Path(filename).suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return load_file(tmp_path)
    finally:
        os.unlink(tmp_path)
