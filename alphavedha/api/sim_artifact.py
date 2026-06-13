"""Loader for the one-time historical-simulation artifact.

The artifact is produced offline by ``scripts/sim_paper_trading.py`` and
committed to the repo (default: ``alphavedha/api/sim_artifact.json``). When
present, the backtest and paper-simulation endpoints serve its contents; when
absent, those endpoints fall back to honest zeros — so deploying this code
changes nothing visible until the artifact exists.

Override the path with ``ALPHAVEDHA_SIM_ARTIFACT``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

SIM_ARTIFACT_PATH = Path(
    os.environ.get("ALPHAVEDHA_SIM_ARTIFACT", "alphavedha/api/sim_artifact.json")
)

# (mtime, parsed) — reparse only when the file changes.
_cache: tuple[float, dict[str, Any] | None] | None = None


def load_sim_artifact() -> dict[str, Any] | None:
    """Return the parsed artifact, or None if absent/unreadable (cached by mtime)."""
    global _cache
    try:
        mtime = SIM_ARTIFACT_PATH.stat().st_mtime
    except OSError:
        return None
    if _cache is not None and _cache[0] == mtime:
        return _cache[1]
    try:
        data: dict[str, Any] = json.loads(SIM_ARTIFACT_PATH.read_text())
    except Exception as exc:
        logger.warning("sim_artifact_load_failed", path=str(SIM_ARTIFACT_PATH), error=str(exc))
        _cache = (mtime, None)
        return None
    _cache = (mtime, data)
    return data
