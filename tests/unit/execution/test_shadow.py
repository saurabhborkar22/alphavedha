"""Tests for shadow mode runner."""

from __future__ import annotations

import os
from datetime import date
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from alphavedha.execution.broker import OrderSide, PaperBroker
from alphavedha.execution.kill_switch import KillSwitch, KillSwitchConfig
from alphavedha.execution.shadow import (
    ShadowRunner,
    ShadowSignal,
    shadow_fills_to_rows,
)


def _make_signal(
    symbol: str = "TCS.NS",
    direction: int = 1,
    magnitude: float = 0.03,
    strategy: str = "ensemble_v1",
    entry_price: float = 3500.0,
) -> ShadowSignal:
    return ShadowSignal(
        symbol=symbol,
        direction=direction,
        magnitude=magnitude,
        position_size_pct=5.0,
        entry_price=entry_price,
        stop_loss_price=entry_price * 0.95,
        take_profit_price=entry_price * 1.06,
        strategy=strategy,
    )


@pytest.fixture
def runner() -> ShadowRunner:
    return ShadowRunner(initial_capital=1_000_000.0)


class TestShadowRunnerBasic:
    @pytest.mark.asyncio
    async def test_empty_signals(self, runner: ShadowRunner) -> None:
        result = await runner.run(signals=[], open_prices={})
        assert result.signals_received == 0
        assert result.fills_simulated == 0

    @pytest.mark.asyncio
    async def test_fills_without_arming_master_flag(self, runner: ShadowRunner) -> None:
        """Ghost fills must not require EXECUTION_ENABLED — the broker is a
        PaperBroker; only the master check is bypassed, caps stay live."""
        signals = [_make_signal()]
        result = await runner.run(signals, {"TCS.NS": 3500.0})
        assert result.signals_received == 1
        assert result.plans_created == 1
        assert result.orders_placed == 1
        assert result.fills_simulated == 1
        assert len(result.fills) == 1

    @pytest.mark.asyncio
    async def test_non_ghost_kill_switch_still_blocks(self) -> None:
        """An explicitly injected real kill switch keeps the master gate."""
        from alphavedha.execution.kill_switch import KillSwitch

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EXECUTION_ENABLED", None)
            runner = ShadowRunner(
                initial_capital=1_000_000.0, kill_switch=KillSwitch(ghost_mode=False)
            )
            signals = [_make_signal()]
            result = await runner.run(signals, {"TCS.NS": 3500.0})
            assert result.orders_blocked == 1
            assert result.fills_simulated == 0
            assert "MASTER_DISABLED" in result.block_reasons[0]

    @pytest.mark.asyncio
    async def test_fill_has_slippage(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            runner = ShadowRunner(initial_capital=1_000_000.0)
            signals = [_make_signal()]
            result = await runner.run(signals, {"TCS.NS": 3500.0})
            fill = result.fills[0]
            assert fill.slippage_bps > 0
            assert fill.decision_price == 3500.0
            assert fill.fill_price > 3500.0

    @pytest.mark.asyncio
    async def test_fill_tag_matches_strategy(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            runner = ShadowRunner(initial_capital=1_000_000.0)
            signals = [_make_signal(strategy="event_drift_v1")]
            result = await runner.run(signals, {"TCS.NS": 3500.0})
            assert result.fills[0].tag == "event_drift_v1"


class TestShadowRunnerMultiple:
    @pytest.mark.asyncio
    async def test_multiple_signals(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            runner = ShadowRunner(initial_capital=1_000_000.0)
            signals = [
                _make_signal("TCS.NS", entry_price=3500.0),
                _make_signal("INFY.NS", entry_price=1500.0),
                _make_signal("HDFCBANK.NS", entry_price=1600.0),
            ]
            prices = {"TCS.NS": 3500.0, "INFY.NS": 1500.0, "HDFCBANK.NS": 1600.0}
            result = await runner.run(signals, prices)
            assert result.fills_simulated == 3

    @pytest.mark.asyncio
    async def test_missing_open_price_skips_fill(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            runner = ShadowRunner(initial_capital=1_000_000.0)
            signals = [_make_signal("TCS.NS")]
            result = await runner.run(signals, {})
            assert result.orders_placed == 1
            assert result.fills_simulated == 0


class TestShadowRunnerKillSwitch:
    @pytest.mark.asyncio
    async def test_max_positions_blocks(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            ks = KillSwitch(KillSwitchConfig(max_positions=2))
            broker = PaperBroker(initial_capital=1_000_000.0)
            runner = ShadowRunner(broker=broker, kill_switch=ks)

            for sym in ["A.NS", "B.NS"]:
                order = await broker.place_order(sym, OrderSide.BUY, 10)
                await broker.simulate_fill(order.order_id, 1000.0)

            signals = [_make_signal("TCS.NS")]
            result = await runner.run(signals, {"TCS.NS": 3500.0})
            assert result.orders_blocked == 1
            assert result.fills_simulated == 0


class TestSlippageReport:
    @pytest.mark.asyncio
    async def test_empty_report(self, runner: ShadowRunner) -> None:
        report = runner.slippage_report()
        assert report["count"] == 0
        assert report["mean_bps"] == 0.0

    @pytest.mark.asyncio
    async def test_report_after_fills(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            runner = ShadowRunner(initial_capital=1_000_000.0)
            signals = [
                _make_signal("TCS.NS", entry_price=3500.0),
                _make_signal("INFY.NS", entry_price=1500.0),
            ]
            await runner.run(signals, {"TCS.NS": 3500.0, "INFY.NS": 1500.0})
            report = runner.slippage_report()
            assert report["count"] == 2
            assert report["mean_bps"] > 0
            assert report["max_bps"] >= report["min_bps"]


class TestShadowFillsToRows:
    @pytest.mark.asyncio
    async def test_converts_fills_to_dicts(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            runner = ShadowRunner(initial_capital=1_000_000.0)
            signals = [_make_signal(strategy="event_drift_v1")]
            result = await runner.run(signals, {"TCS.NS": 3500.0})
            rows = shadow_fills_to_rows(result.fills, date.today())
            assert len(rows) == 1
            row = rows[0]
            assert row["strategy"] == "event_drift_v1"
            assert row["symbol"] == "TCS.NS"
            assert row["side"] == "BUY"
            assert float(row["slippage_bps"]) > 0  # type: ignore[arg-type]
            assert row["fill_date"] == date.today()

    def test_empty_fills(self) -> None:
        rows = shadow_fills_to_rows([], date.today())
        assert len(rows) == 0


class TestShadowFillModel:
    def test_shadow_fill_orm_importable(self) -> None:
        from alphavedha.data.models import ShadowFill

        assert ShadowFill.__tablename__ == "shadow_fills"

    def test_shadow_fill_columns(self) -> None:
        from alphavedha.data.models import ShadowFill

        cols = {c.name for c in ShadowFill.__table__.columns}
        expected = {
            "id",
            "strategy",
            "symbol",
            "fill_date",
            "side",
            "decision_price",
            "sim_fill_price",
            "quantity",
            "slippage_bps",
            "created_at",
        }
        assert expected.issubset(cols)


class TestRunShadowCycle:
    """Scheduler glue: today's tradeable paper trades → ghost fills."""

    @staticmethod
    def _trades_df(rows: list[dict[str, object]]) -> pd.DataFrame:
        base: dict[str, object] = {
            "symbol": "TCS",
            "prediction_date": date(2026, 7, 2),
            "strategy": "event_drift_v1",
            "predicted_direction": 1,
            "predicted_magnitude": 0.01,
            "confidence": 0.7,
            "model_version": "event_drift_v1",
            "is_tradeable": True,
            "entry_price": 3500.0,
            "stop_loss_price": None,
            "take_profit_price": None,
            "exit_price": None,
        }
        return pd.DataFrame([{**base, **r} for r in rows])

    @pytest.mark.asyncio
    async def test_already_ran_guard(self) -> None:
        from alphavedha.scheduler import _run_shadow_cycle

        with patch(
            "alphavedha.data.store.count_shadow_fills",
            new_callable=AsyncMock,
            return_value=3,
        ):
            summary = await _run_shadow_cycle(date(2026, 7, 2))

        assert summary == {"status": "already_ran", "fills": 0}

    @pytest.mark.asyncio
    async def test_no_tradeable_signals(self) -> None:
        from alphavedha.scheduler import _run_shadow_cycle

        trades = self._trades_df([{"is_tradeable": False}])
        with (
            patch(
                "alphavedha.data.store.count_shadow_fills",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch(
                "alphavedha.data.store.load_paper_trades",
                new_callable=AsyncMock,
                return_value=trades,
            ),
        ):
            summary = await _run_shadow_cycle(date(2026, 7, 2))

        assert summary["status"] == "no_tradeable_signals"

    @pytest.mark.asyncio
    async def test_happy_path_persists_fills(self) -> None:
        from alphavedha.scheduler import _run_shadow_cycle

        trades = self._trades_df([{}])
        stored: list[dict[str, object]] = []

        async def fake_store(rows: list[dict[str, object]]) -> int:
            stored.extend(rows)
            return len(rows)

        with (
            patch(
                "alphavedha.data.store.count_shadow_fills",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch(
                "alphavedha.data.store.load_paper_trades",
                new_callable=AsyncMock,
                return_value=trades,
            ),
            patch(
                "alphavedha.data.store.store_shadow_fills",
                side_effect=fake_store,
            ),
            patch(
                "alphavedha.scheduler._load_open_prices",
                new_callable=AsyncMock,
                return_value={"TCS": 3510.0},
            ),
        ):
            summary = await _run_shadow_cycle(date(2026, 7, 2))

        assert summary["status"] == "ok"
        assert summary["fills"] == 1
        assert len(stored) == 1
        fill = stored[0]
        assert fill["symbol"] == "TCS"
        assert fill["strategy"] == "event_drift_v1"
        assert fill["decision_price"] == 3510.0
        assert float(str(fill["slippage_bps"])) > 0
