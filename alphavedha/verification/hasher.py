"""Prediction hasher — deterministic SHA-256 of daily paper-trade predictions.

Produces a canonical JSON payload from the day's paper_trades rows, then
hashes it.  The hash is stored in ``prediction_proofs`` and later committed
to a git proofs repo + OpenTimestamps-stamped (see publisher.py).

The canonical payload is intentionally independent of row order and dict
key order so that the same predictions always produce the same hash,
regardless of database ordering or Python dict iteration.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any

CANONICAL_FIELDS: list[str] = [
    "symbol",
    "prediction_date",
    "predicted_direction",
    "predicted_magnitude",
    "confidence",
    "is_tradeable",
    "model_version",
    "regime",
]


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    """Extract and normalize the canonical fields from a single trade row."""
    out: dict[str, Any] = {}
    for field in CANONICAL_FIELDS:
        val = row.get(field)
        if isinstance(val, date):
            val = val.isoformat()
        if isinstance(val, float):
            val = round(val, 8)
        out[field] = val
    return out


def canonical_payload(trades: list[dict[str, Any]]) -> bytes:
    """Build a deterministic JSON payload from the day's paper trades.

    Rows are sorted by (symbol, prediction_date) so database row order
    doesn't affect the hash. Keys within each row are sorted by
    json.dumps(sort_keys=True). Whitespace is stripped (separators=(',',':')).
    """
    normalized = sorted(
        [_normalize_row(t) for t in trades],
        key=lambda r: (r.get("symbol", ""), r.get("prediction_date", "")),
    )
    return json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_hex(payload: bytes) -> str:
    """Return the hex-encoded SHA-256 digest of the payload."""
    return hashlib.sha256(payload).hexdigest()


def hash_daily_trades(trades: list[dict[str, Any]]) -> tuple[str, bytes]:
    """Convenience: canonical payload + its SHA-256 hex digest.

    Returns (hex_digest, raw_payload_bytes).
    """
    payload = canonical_payload(trades)
    return sha256_hex(payload), payload
