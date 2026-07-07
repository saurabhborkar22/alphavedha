"""Map free-form news text to NSE universe symbols.

Aliases live in ``configs/stocks.yaml`` under ``news_aliases``. Name aliases
("HDFC Bank", "Tata Steel") match case-insensitively on word boundaries; the
NSE ticker itself matches only as the exact uppercase token, so "LT" tags
Larsen & Toubro while "lt" or "Belgium" never fire BEL/LT.

A match that exists only inside a longer match of a different symbol is
dropped — "SBI Life Q1 up 14%" tags SBILIFE, not SBIN.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import structlog
import yaml

logger = structlog.get_logger(__name__)

_STOCKS_CONFIG = Path("configs/stocks.yaml")

Span = tuple[int, int]


def _config_path() -> Path | None:
    if _STOCKS_CONFIG.exists():
        return _STOCKS_CONFIG
    fallback = Path(__file__).resolve().parents[2] / "configs" / "stocks.yaml"
    if fallback.exists():
        return fallback
    return None


@lru_cache(maxsize=1)
def _compiled_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    """(symbol, pattern) pairs for every symbol with configured aliases."""
    path = _config_path()
    if path is None:
        logger.warning("news_aliases_config_missing", path=str(_STOCKS_CONFIG))
        return ()

    with path.open() as f:
        config = yaml.safe_load(f) or {}

    aliases: dict[str, list[str]] = config.get("news_aliases") or {}
    if not aliases:
        logger.warning("news_aliases_empty", path=str(path))
        return ()

    patterns: list[tuple[str, re.Pattern[str]]] = []
    for symbol, names in aliases.items():
        # Ticker itself: exact-case uppercase word only.
        patterns.append((symbol, re.compile(rf"\b{re.escape(symbol)}\b")))
        for name in names or []:
            patterns.append((symbol, re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)))
    return tuple(patterns)


def _find_spans(text: str) -> dict[str, list[Span]]:
    spans: dict[str, list[Span]] = {}
    for symbol, pattern in _compiled_patterns():
        for m in pattern.finditer(text):
            spans.setdefault(symbol, []).append(m.span())
    return spans


def match_symbols(text: str) -> list[str]:
    """Return sorted universe symbols mentioned in ``text`` (empty if none)."""
    if not text:
        return []

    spans = _find_spans(text)

    def _contained_elsewhere(symbol: str, span: Span) -> bool:
        for other, other_spans in spans.items():
            if other == symbol:
                continue
            for outer in other_spans:
                if outer != span and outer[0] <= span[0] and span[1] <= outer[1]:
                    return True
        return False

    matched = [
        symbol
        for symbol, sym_spans in spans.items()
        if any(not _contained_elsewhere(symbol, s) for s in sym_spans)
    ]
    return sorted(matched)
