"""Intra-hold stop-loss / take-profit evaluation for open paper trades.

The prediction engine computes ATR-based stop and target levels for every
trade, but until this module they were display-only: trades sat through the
full 15-trading-day hold no matter how far a position moved against them.
This is the FIX-08 enforcement path — shared by the scheduler's daily job
and the manual `POST /paper/evaluate-stops` endpoint.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

_OHLCV_LOOKBACK_DAYS = 5


def _as_float(value: object) -> float | None:
    """Coerce a pandas cell to float, mapping NaN/None to None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return float(value)  # type: ignore[arg-type]


async def evaluate_stop_hits(eval_date: date | None = None) -> dict[str, int]:
    """Close open paper trades whose stop or target was hit on eval_date.

    Long trades: day low <= stop_loss stops out; day high >= take_profit
    takes profit. Shorts are mirrored. Stops are checked before targets —
    a conservative tie-break when both trigger on the same bar.
    """
    from alphavedha.data.store import load_ohlcv, load_paper_trades, update_paper_trade_outcome

    eval_date = eval_date or date.today()
    trades_df = await load_paper_trades()

    summary = {"evaluated": 0, "stopped_out": 0, "target_hit": 0}
    if trades_df.empty:
        return summary

    open_trades = trades_df[trades_df["exit_price"].isna()].copy()
    if open_trades.empty:
        return summary

    for _, trade in open_trades.iterrows():
        symbol = str(trade["symbol"])
        entry = _as_float(trade.get("entry_price"))
        sl = _as_float(trade.get("stop_loss_price"))
        tp = _as_float(trade.get("take_profit_price"))
        direction = int(trade["predicted_direction"])

        if entry is None or entry <= 0 or (sl is None and tp is None) or direction == 0:
            continue

        try:
            ohlcv = await load_ohlcv(
                symbol, eval_date - timedelta(days=_OHLCV_LOOKBACK_DAYS), eval_date
            )
        except Exception as e:
            logger.warning("stop_eval_ohlcv_failed", symbol=symbol, error=str(e))
            continue

        if ohlcv.empty:
            continue

        day_row = (
            ohlcv[ohlcv.index.date == eval_date] if hasattr(ohlcv.index, "date") else ohlcv.tail(1)
        )
        if day_row.empty:
            continue

        day_low = float(day_row["low"].iloc[-1])
        day_high = float(day_row["high"].iloc[-1])

        exit_price: float | None = None
        exit_reason: str | None = None

        if direction == 1:
            if sl is not None and day_low <= sl:
                exit_price, exit_reason = sl, "stop_loss"
            elif tp is not None and day_high >= tp:
                exit_price, exit_reason = tp, "take_profit"
        else:
            if sl is not None and day_high >= sl:
                exit_price, exit_reason = sl, "stop_loss"
            elif tp is not None and day_low <= tp:
                exit_price, exit_reason = tp, "take_profit"

        if exit_price is None:
            continue

        # actual_return stores the PRICE return (not direction-multiplied);
        # trade P&L = predicted_direction * actual_return, same as the
        # scheduler's horizon-maturity path.
        actual_return = (exit_price - entry) / entry
        is_correct = actual_return * direction > 0
        pred_date = trade["prediction_date"]
        if not isinstance(pred_date, date):
            pred_date = date.fromisoformat(str(pred_date))

        try:
            await update_paper_trade_outcome(
                symbol=symbol,
                prediction_date=pred_date,
                exit_price=exit_price,
                actual_return=actual_return,
                is_correct=is_correct,
                exit_reason=exit_reason,
                strategy=str(trade.get("strategy", "ensemble_v1")),
            )
        except Exception as e:
            logger.error("stop_eval_update_failed", symbol=symbol, error=str(e))
            continue

        summary["evaluated"] += 1
        if exit_reason == "stop_loss":
            summary["stopped_out"] += 1
        else:
            summary["target_hit"] += 1

    logger.info("stop_hits_evaluated", eval_date=str(eval_date), **summary)
    return summary
