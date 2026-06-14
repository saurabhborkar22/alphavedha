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
import re
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

SIM_ARTIFACT_PATH = Path(
    os.environ.get("ALPHAVEDHA_SIM_ARTIFACT", "alphavedha/api/sim_artifact.json")
)

# Archive of every sim run, keyed by window slug, so past results are never
# overwritten by a later run. Defaults to a sibling of the latest artifact
# (the shared model-artifacts volume in prod).
SIM_RUNS_DIR = Path(
    os.environ.get("ALPHAVEDHA_SIM_RUNS_DIR", str(SIM_ARTIFACT_PATH.parent / "sim_runs"))
)
_SLUG_RE = re.compile(r"^[A-Za-z0-9._=-]+$")

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


def _run_summary(art: dict[str, Any], slug: str) -> dict[str, Any]:
    """Headline fields for one archived run (powers the run-picker dropdown)."""
    meta = art.get("meta") or {}
    tr = art.get("track_record") or {}
    gate = (tr.get("tracks") or {}).get("gate_passed") or {}
    bt = (art.get("backtest") or {}).get("summary") or {}
    return {
        "slug": slug,
        "cutoff": meta.get("cutoff"),
        "end": meta.get("end"),
        "n_days": meta.get("n_days"),
        "n_trades": meta.get("n_trades"),
        "generated_at": art.get("generated_at"),
        "accuracy_all": tr.get("accuracy_all"),
        "gate_avg_net": gate.get("avg_return_net"),
        "cagr": bt.get("cagr"),
        "sharpe": bt.get("sharpe"),
    }


def list_sim_runs() -> list[dict[str, Any]]:
    """Summaries of all archived sim runs, newest first (empty if none)."""
    if not SIM_RUNS_DIR.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for path in sorted(SIM_RUNS_DIR.glob("*.json")):
        try:
            art = json.loads(path.read_text())
        except Exception as exc:
            logger.warning("sim_run_parse_failed", path=str(path), error=str(exc))
            continue
        runs.append(_run_summary(art, path.stem))
    runs.sort(key=lambda r: r.get("generated_at") or "", reverse=True)
    return runs


def load_sim_run(slug: str) -> dict[str, Any] | None:
    """Load one archived run by slug; None if the slug is invalid or absent."""
    if not _SLUG_RE.match(slug):
        return None
    path = SIM_RUNS_DIR / f"{slug}.json"
    try:
        # Defense in depth: the resolved path must stay inside SIM_RUNS_DIR.
        if SIM_RUNS_DIR.resolve() not in path.resolve().parents:
            return None
        if not path.is_file():
            return None
        data: dict[str, Any] = json.loads(path.read_text())
        return data
    except Exception as exc:
        logger.warning("sim_run_load_failed", slug=slug, error=str(exc))
        return None
