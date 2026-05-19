"""Daily prediction cron job — run before market open.

Schedule: Every trading day at 8:30 AM IST.
Crontab: 30 8 * * 1-5 cd /path/to/alphavedha && .venv/bin/python scripts/daily_prediction.py

Flow:
  1. Fetch latest data for all stocks in universe
  2. Compute features
  3. Run ensemble prediction pipeline
  4. Store predictions in paper_trades table (timestamped before 9:15 AM)
  5. After market close (3:45 PM), run with --settle flag to update outcomes
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import date, datetime

import structlog

structlog.configure(
    logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
)

logger = structlog.get_logger(__name__)


async def generate_daily_predictions(tier: str = "large", demo: bool = True) -> int:
    """Generate predictions for all stocks in universe and store as paper trades."""
    from alphavedha.config import get_config
    from alphavedha.data.store import store_paper_trade
    from alphavedha.services.cache import PredictionCache
    from alphavedha.services.model_registry import ModelRegistry
    from alphavedha.services.prediction_service import PredictionService

    config = get_config()
    registry = ModelRegistry(demo=demo)
    cache = PredictionCache(redis_client=None)
    service = PredictionService(registry=registry, cache=cache, config=config)

    today = date.today()
    now = datetime.now()
    logger.info("daily_prediction_start", date=str(today), time=str(now), tier=tier, demo=demo)

    result = await service.scan_tier(tier, top_n=50)
    stored = 0

    for candidate in result.candidates:
        pred = candidate.prediction
        row = {
            "symbol": candidate.symbol,
            "prediction_date": today,
            "predicted_direction": pred.direction,
            "predicted_magnitude": pred.magnitude,
            "confidence": pred.confidence,
            "model_version": pred.model_version,
            "regime": getattr(pred, "regime", None),
            "entry_price": getattr(pred, "current_price", None),
        }
        try:
            await store_paper_trade(row)
            stored += 1
        except Exception as e:
            logger.warning("paper_trade_failed", symbol=candidate.symbol, error=str(e))

    logger.info("daily_prediction_complete", stored=stored, total=len(result.candidates))
    return stored


async def settle_daily_outcomes(tier: str = "large") -> int:
    """After market close: compare predictions vs actual prices, update outcomes."""
    from alphavedha.data.store import load_paper_trades, update_paper_trade_outcome

    today = date.today()
    trades_df = await load_paper_trades(start=today, end=today)

    if trades_df.empty:
        logger.info("settle_no_trades", date=str(today))
        return 0

    settled = 0
    for _, row in trades_df.iterrows():
        if row.get("is_correct") is not None:
            continue

        symbol = row["symbol"]
        entry = row.get("entry_price")
        if entry is None or entry == 0:
            continue

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            if hist.empty:
                continue

            exit_price = float(hist["Close"].iloc[-1])
            actual_ret = (exit_price / entry) - 1
            predicted_dir = row["predicted_direction"]

            is_correct = (
                (predicted_dir == 1 and actual_ret > 0)
                or (predicted_dir == -1 and actual_ret < 0)
                or (predicted_dir == 0 and abs(actual_ret) < 0.005)
            )

            await update_paper_trade_outcome(
                symbol=symbol,
                prediction_date=today,
                exit_price=exit_price,
                actual_return=actual_ret,
                is_correct=is_correct,
            )
            settled += 1
        except Exception as e:
            logger.warning("settle_failed", symbol=symbol, error=str(e))

    logger.info("settle_complete", settled=settled, date=str(today))
    return settled


def main() -> None:
    parser = argparse.ArgumentParser(description="AlphaVedha daily prediction job")
    parser.add_argument("--tier", default="large", help="Market cap tier")
    parser.add_argument("--demo", action="store_true", help="Use demo/synthetic models")
    parser.add_argument("--settle", action="store_true", help="Settle today's outcomes")
    args = parser.parse_args()

    if args.settle:
        count = asyncio.run(settle_daily_outcomes(args.tier))
        print(f"Settled {count} trades")
    else:
        count = asyncio.run(generate_daily_predictions(args.tier, demo=args.demo))
        print(f"Generated {count} predictions")


if __name__ == "__main__":
    main()
