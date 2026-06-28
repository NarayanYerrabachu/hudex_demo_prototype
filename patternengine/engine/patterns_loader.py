"""
Load ENTITY_PATTERNS and SUSPICIOUS_VOCAB from patterns.yml.

Resolution order:
  1. HUDEX_PATTERNS_FILE env var (absolute path)
  2. patterns.yml next to this engine's parent directory (patternengine/)
  3. Built-in defaults (hardcoded fallback — no file required)
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# Defaults — used when no YAML file is found
# --------------------------------------------------------------------------

_DEFAULT_ENTITY_PATTERNS: dict[str, str] = {
    "money":   r"(?:USD|EUR|\$|€)\s?\d[\d,]*(?:\.\d+)?\s?(?:k|m|million|thousand)?",
    "email":   r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}",
    "account": r"\b(?:ACC|IBAN|REF)[-\s]?[A-Z0-9]{4,}\b",
    "date":    r"\b\d{4}-\d{2}-\d{2}\b",
    "actor":   r"\b[A-ZÄÖÜ][a-zäöüß]+ [A-ZÄÖÜ][a-zäöüß]+\b",
}

_DEFAULT_SUSPICIOUS_VOCAB: list[str] = [
    "off the books", "off-book", "no invoice", "no trace", "no paperwork",
    "do not log", "delete this", "stay quiet", "keep this between",
    "shell entity", "shell company", "intermediary",
    "kickback", "bribe", "launder", "laundering",
    "untraceable", "offshore", "placeholder",
    "cash only", "settle in cash", "strictly off",
    "coded as", "skips approval",
]


def _find_patterns_file() -> Path | None:
    # 1. Explicit env override
    env_path = os.environ.get("HUDEX_PATTERNS_FILE")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    # 2. patterns.yml sitting next to the engine package (patternengine/)
    candidate = Path(__file__).parent.parent / "patterns.yml"
    if candidate.exists():
        return candidate

    return None


def load_patterns() -> tuple[dict[str, re.Pattern], list[str]]:
    """Return (ENTITY_PATTERNS, SUSPICIOUS_VOCAB) from YAML or defaults."""
    path = _find_patterns_file()

    if path is None:
        return _compile(_DEFAULT_ENTITY_PATTERNS), _DEFAULT_SUSPICIOUS_VOCAB

    try:
        import yaml
    except ImportError:
        print(f"[patterns_loader] pyyaml not installed — using built-in defaults")
        return _compile(_DEFAULT_ENTITY_PATTERNS), _DEFAULT_SUSPICIOUS_VOCAB

    try:
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        raw_patterns = data.get("entity_patterns", _DEFAULT_ENTITY_PATTERNS)
        vocab = data.get("suspicious_vocab", _DEFAULT_SUSPICIOUS_VOCAB)

        print(f"[patterns_loader] loaded from {path} "
              f"({len(raw_patterns)} entity patterns, {len(vocab)} vocab terms)")
        return _compile(raw_patterns), list(vocab)

    except Exception as exc:
        print(f"[patterns_loader] failed to load {path}: {exc} — using defaults")
        return _compile(_DEFAULT_ENTITY_PATTERNS), _DEFAULT_SUSPICIOUS_VOCAB


# Patterns that must stay case-sensitive — proper nouns only, no re.I
_CASE_SENSITIVE = {"actor"}


def _compile(raw: dict[str, str]) -> dict[str, re.Pattern]:
    compiled = {}
    for name, pattern in raw.items():
        flags = 0 if name in _CASE_SENSITIVE else re.I
        try:
            compiled[name] = re.compile(pattern, flags)
        except re.error as exc:
            print(f"[patterns_loader] invalid regex for '{name}': {exc} — skipping")
    return compiled
