"""Tests for the shared stop-loss / take-profit evaluation service (FIX-08)."""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from alphavedha.services.stop_evaluation import evaluate_stop_hits

EVAL_DATE = date(2026, 7, 1)


def _trades_df(rows: list[dict[str, Any]]) -> pd.DataFrame:
    base: dict[str, Any] = {
        "symbol": "TCS",
        "prediction_date": EVAL_DATE,
        "strategy": "ensemble_v1",
        "predicted_direction": 1,
        "predicted_magnitude": 0.01,
        "confidence": 0.6,
        "model_version": "v0.1.0",
        "is_tradeable": True,
        "entry_price": 100.0,
        "stop_loss_price": 95.0,
        "take_profit_price": 110.0,
        "exit_price": None,
        "exit_reason": None,
        "actual_return": None,
        "is_correct": None,
    }
    return pd.DataFrame([{**base, **row} for row in rows])


def _ohlcv(low: float, high: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [(low + high) / 2],
            "high": [high],
            "low": [low],
            "close": [(low + high) / 2],
            "volume": [1000],
        },
        index=pd.DatetimeIndex([pd.Timestamp(EVAL_DATE)]),
    )


class TestEvaluateStopHits:
    @pytest.mark.asyncio
    async def test_long_stop_loss_hit(self) -> None:
        trades = _trades_df([{"predicted_direction": 1}])
        with (
            patch(
                "alphavedha.data.store.load_paper_trades",
                new_callable=AsyncMock,
                return_value=trades,
            ),
            patch(
                "alphavedha.data.store.load_ohlcv",
                new_callable=AsyncMock,
                return_value=_ohlcv(low=94.0, high=101.0),
            ),
            patch(
                "alphavedha.data.store.update_paper_trade_outcome",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            summary = await evaluate_stop_hits(EVAL_DATE)

        assert summary == {"evaluated": 1, "stopped_out": 1, "target_hit": 0}
        kwargs = mock_update.await_args.kwargs
        assert kwargs["exit_price"] == 95.0
        assert kwargs["exit_reason"] == "stop_loss"
        assert kwargs["actual_return"] == pytest.approx(-0.05)
        assert kwargs["is_correct"] is False

    @pytest.mark.asyncio
    async def test_short_stop_loss_hit(self) -> None:
        trades = _trades_df(
            [
                {
                    "predicted_direction": -1,
                    "stop_loss_price": 105.0,
                    "take_profit_price": 90.0,
                }
            ]
        )
        with (
            patch(
                "alphavedha.data.store.load_paper_trades",
                new_callable=AsyncMock,
                return_value=trades,
            ),
            patch(
                "alphavedha.data.store.load_ohlcv",
                new_callable=AsyncMock,
                return_value=_ohlcv(low=99.0, high=106.0),
            ),
            patch(
                "alphavedha.data.store.update_paper_trade_outcome",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            summary = await evaluate_stop_hits(EVAL_DATE)

        assert summary["stopped_out"] == 1
        kwargs = mock_update.await_args.kwargs
        assert kwargs["exit_price"] == 105.0
        # actual_return is the PRICE return: the stock rose 5%, so the
        # short lost — is_correct carries the win/loss, not the sign.
        assert kwargs["actual_return"] == pytest.approx(0.05)
        assert kwargs["is_correct"] is False

    @pytest.mark.asyncio
    async def test_short_take_profit_hit_stores_price_return(self) -> None:
        """Regression: a winning short was stored direction-multiplied (+),
        which the price-return consumers (gross = direction * actual_return)
        then sign-flipped into a loss."""
        trades = _trades_df(
            [
                {
                    "predicted_direction": -1,
                    "stop_loss_price": 105.0,
                    "take_profit_price": 90.0,
                }
            ]
        )
        with (
            patch(
                "alphavedha.data.store.load_paper_trades",
                new_callable=AsyncMock,
                return_value=trades,
            ),
            patch(
                "alphavedha.data.store.load_ohlcv",
                new_callable=AsyncMock,
                return_value=_ohlcv(low=89.0, high=101.0),
            ),
            patch(
                "alphavedha.data.store.update_paper_trade_outcome",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            summary = await evaluate_stop_hits(EVAL_DATE)

        assert summary["target_hit"] == 1
        kwargs = mock_update.await_args.kwargs
        assert kwargs["exit_price"] == 90.0
        assert kwargs["exit_reason"] == "take_profit"
        # price fell 10% — stored as the price return, flagged correct
        assert kwargs["actual_return"] == pytest.approx(-0.10)
        assert kwargs["is_correct"] is True

    @pytest.mark.asyncio
    async def test_long_take_profit_hit(self) -> None:
        trades = _trades_df([{"predicted_direction": 1}])
        with (
            patch(
                "alphavedha.data.store.load_paper_trades",
                new_callable=AsyncMock,
                return_value=trades,
            ),
            patch(
                "alphavedha.data.store.load_ohlcv",
                new_callable=AsyncMock,
                return_value=_ohlcv(low=99.0, high=111.0),
            ),
            patch(
                "alphavedha.data.store.update_paper_trade_outcome",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            summary = await evaluate_stop_hits(EVAL_DATE)

        assert summary == {"evaluated": 1, "stopped_out": 0, "target_hit": 1}
        kwargs = mock_update.await_args.kwargs
        assert kwargs["exit_reason"] == "take_profit"
        assert kwargs["actual_return"] == pytest.approx(0.10)
        assert kwargs["is_correct"] is True

    @pytest.mark.asyncio
    async def test_no_levels_skipped(self) -> None:
        """Intel-strategy trades carry NaN stop/target — they must be left alone."""
        trades = _trades_df(
            [
                {
                    "strategy": "event_drift_v1",
                    "stop_loss_price": np.nan,
                    "take_profit_price": np.nan,
                }
            ]
        )
        with (
            patch(
                "alphavedha.data.store.load_paper_trades",
                new_callable=AsyncMock,
                return_value=trades,
            ),
            patch(
                "alphavedha.data.store.load_ohlcv",
                new_callable=AsyncMock,
                return_value=_ohlcv(low=50.0, high=150.0),
            ),
            patch(
                "alphavedha.data.store.update_paper_trade_outcome",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            summary = await evaluate_stop_hits(EVAL_DATE)

        assert summary == {"evaluated": 0, "stopped_out": 0, "target_hit": 0}
        mock_update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_nan_entry_price_skipped(self) -> None:
        """Regression: NaN entry with a real stop produced NaN actual_return."""
        trades = _trades_df([{"entry_price": np.nan}])
        with (
            patch(
                "alphavedha.data.store.load_paper_trades",
                new_callable=AsyncMock,
                return_value=trades,
            ),
            patch(
                "alphavedha.data.store.load_ohlcv",
                new_callable=AsyncMock,
                return_value=_ohlcv(low=90.0, high=111.0),
            ),
            patch(
                "alphavedha.data.store.update_paper_trade_outcome",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            summary = await evaluate_stop_hits(EVAL_DATE)

        assert summary["evaluated"] == 0
        mock_update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_bar_for_eval_date_skipped(self) -> None:
        trades = _trades_df([{}])
        stale = pd.DataFrame(
            {"open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [1]},
            index=pd.DatetimeIndex([pd.Timestamp(date(2026, 6, 27))]),
        )
        with (
            patch(
                "alphavedha.data.store.load_paper_trades",
                new_callable=AsyncMock,
                return_value=trades,
            ),
            patch(
                "alphavedha.data.store.load_ohlcv",
                new_callable=AsyncMock,
                return_value=stale,
            ),
            patch(
                "alphavedha.data.store.update_paper_trade_outcome",
                new_callable=AsyncMock,
            ) as mock_update,
        ):
            summary = await evaluate_stop_hits(EVAL_DATE)

        assert summary["evaluated"] == 0
        mock_update.assert_not_awaited()
