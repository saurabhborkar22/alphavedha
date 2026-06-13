"""Backtest + paper-simulation endpoints: zeros when no artifact, real data when present."""

from __future__ import annotations

import asyncio
import json
from datetime import date, timedelta

import pandas as pd

from alphavedha.api import sim_artifact
from alphavedha.api.routes import paper_trading, ui_support
from alphavedha.backtest.sim_views import build_artifact


def _trades() -> pd.DataFrame:
    rows: list[dict] = []
    base = date(2026, 1, 5)
    for d in range(10):
        for s in range(5):
            direction = 1 if s % 2 == 0 else -1
            actual = (0.02 if (d + s) % 3 else -0.015) * direction
            rows.append(
                {
                    "symbol": f"SYM{s}.NS",
                    "prediction_date": base + timedelta(days=d),
                    "predicted_direction": direction,
                    "predicted_magnitude": 0.02,
                    "confidence": 0.6,
                    "model_version": "sim-test",
                    "regime": "bull",
                    "is_tradeable": True,
                    "entry_price": 100.0,
                    "exit_price": 100.0 * (1 + actual),
                    "actual_return": actual,
                    "is_correct": (direction == (1 if actual > 0 else -1)),
                }
            )
    return pd.DataFrame(rows)


def _point_at(tmp_path, monkeypatch, artifact: dict | None) -> None:
    monkeypatch.delenv("ALPHAVEDHA_DEMO", raising=False)
    p = tmp_path / "sim.json"
    if artifact is not None:
        p.write_text(json.dumps(artifact))
    monkeypatch.setattr(sim_artifact, "SIM_ARTIFACT_PATH", p)
    monkeypatch.setattr(sim_artifact, "_cache", None)


def test_zeros_when_artifact_absent(tmp_path, monkeypatch) -> None:
    _point_at(tmp_path, monkeypatch, None)
    summary = asyncio.run(ui_support.backtest_summary())
    assert summary.total_trades == 0
    assert asyncio.run(ui_support.backtest_equity()) == {"strategy": [], "benchmark": []}
    assert asyncio.run(ui_support.backtest_monthly()) == []
    sim = asyncio.run(paper_trading.get_simulation())
    assert sim["available"] is False


def test_serves_artifact_when_present(tmp_path, monkeypatch) -> None:
    art = build_artifact(_trades(), cost_pct=0.0047, meta={"tier": "large", "cutoff": "2025-12-12"})
    _point_at(tmp_path, monkeypatch, art)

    summary = asyncio.run(ui_support.backtest_summary())
    assert summary.total_trades == art["backtest"]["summary"]["total_trades"] > 0
    assert summary.date_from and summary.date_to

    equity = asyncio.run(ui_support.backtest_equity())
    assert len(equity["strategy"]) == 10

    monthly = asyncio.run(ui_support.backtest_monthly())
    assert monthly and all(m["year"] == 2026 for m in monthly)

    dist = asyncio.run(ui_support.backtest_distribution())
    assert sum(b["count"] for b in dist) > 0

    sim = asyncio.run(paper_trading.get_simulation())
    assert sim["available"] is True
    assert set(sim["track_record"]["tracks"]) == {"all", "gate_passed", "top_k"}
    assert sim["meta"]["cutoff"] == "2025-12-12"


def test_demo_mode_unaffected_by_artifact(tmp_path, monkeypatch) -> None:
    # Demo mode keeps its synthetic numbers regardless of any artifact.
    _point_at(tmp_path, monkeypatch, build_artifact(_trades(), 0.0047, {"tier": "large"}))
    monkeypatch.setenv("ALPHAVEDHA_DEMO", "1")
    summary = asyncio.run(ui_support.backtest_summary())
    assert summary.total_trades == 342  # the demo fixture value
