"""Load corpus from corpus.csv (columns: id, text, channel, date)."""

import csv
from pathlib import Path
from engine.core import Document

_CSV_PATH = Path(__file__).parent.parent / "corpus.csv"


def make_corpus(path: str | Path = _CSV_PATH) -> list[Document]:
    docs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            docs.append(Document(
                id=row["id"],
                text=row["text"],
                meta={"channel": row.get("channel", ""), "date": row.get("date", "")},
            ))
    return docs
