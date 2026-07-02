"""Background scheduler — orchestrates daily predictions, evaluation, drift checks, and retraining.

Uses the `schedule` library for lightweight in-process scheduling.
All times are in IST (Asia/Kolkata). Designed for personal deployment
where a single process handles everything.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import schedule
import structlog

if TYPE_CHECKING:
    from collections.abc import Sequence

    from alphavedha.prediction.engine import StockPrediction

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

PREDICTION_TIME = "06:00"
SIGNAL_STRATEGIES_TIME = "06:05"  # signal strategies after ensemble, before hash
PREDICTION_HASH_TIME = "06:10"  # hash daily predictions after 06:00 persist, before 09:15 open
PROOF_REVEAL_TIME = "16:00"  # reveal canonical payloads for proofs >= 21 days old
EVALUATION_TIME = "15:45"
DATA_REFRESH_TIME = "17:00"  # daily OHLCV ingestion after market close (15:30 IST)
FII_DII_INGESTION_TIME = "18:30"  # NSE publishes FII/DII participation data by ~17:30 IST
BHAVCOPY_INGESTION_TIME = "18:45"  # after FII/DII, before BSE weekly jobs
BSE_ANN_INGESTION_TIME = "19:00"  # daily BSE announcements + PDF extraction
NSE_ANN_INGESTION_TIME = "19:15"  # daily NSE announcements (PIT/SAST flagging)
INSIDER_TRADES_INGESTION_TIME = "19:20"  # insider trades from BSE API (SAST disclosures)
SURVEILLANCE_INGESTION_TIME = "19:30"  # ASM/GSM list snapshot
DEALS_INGESTION_TIME = "19:45"  # bulk/block/short deals
CREDIT_RATING_INGESTION_TIME = "19:50"  # credit rating actions from announcements
TRANSCRIPT_INGESTION_TIME = "20:15"  # concall transcripts (heavier PDFs, runs after weekly drift)
INTEL_EXTRACTION_TIME = "20:00"  # LLM extraction on unprocessed disclosures
INTEL_QUALITY_CHECK_TIME = "20:30"  # intel row-count + disk checks after all collectors finish
DRIFT_CHECK_DAY = "saturday"
DRIFT_CHECK_TIME = "20:00"
RETRAIN_DAY = "saturday"
RETRAIN_TIME = "22:00"
XGBOOST_RETRAIN_TIME = "23:30"  # daily, after market-close data refresh
LSTM_TFT_RETRAIN_DAY = "saturday"
LSTM_TFT_RETRAIN_TIME = "22:30"  # weekly, after drift check
REBALANCE_CHECK_DAY = "monday"
REBALANCE_CHECK_TIME = "07:00"
QUALITY_CHECK_TIME = "15:50"
STOP_LOSS_CHECK_TIME = "17:30"  # after 17:00 data refresh lands the day's OHLCV
BSE_INGESTION_DAY = "sunday"
BSE_INGESTION_TIME = "21:00"
TRENDS_INGESTION_TIME = "21:30"
INTRADAY_POLL_INTERVAL_MINUTES = 2
REBALANCE_MONTHS = {3, 9}

# Paper trade evaluation: triple-barrier config uses a 15-trading-day max hold.
# 21 calendar days ~ 15 trading days (weekends + the odd holiday).
EVALUATION_HORIZON_TRADING_DAYS = 15
EVALUATION_MIN_CALENDAR_DAYS = 21
ENTRY_PRICE_LOOKBACK_DAYS = 10
INITIAL_PORTFOLIO_VALUE = 1_000_000.0


@dataclass
class JobResult:
    job_name: str
    started_at: datetime
    finished_at: datetime | None = None
    success: bool = False
    symbols_processed: int = 0
    error: str | None = None


@dataclass
class SchedulerState:
    is_running: bool = False
    job_history: list[JobResult] = field(default_factory=list)
    last_prediction_run: datetime | None = None
    last_evaluation_run: datetime | None = None
    last_drift_check: datetime | None = None
    last_retrain: datetime | None = None
    last_xgboost_retrain: datetime | None = None
    last_lstm_tft_retrain: datetime | None = None
    last_rebalance_check: datetime | None = None
    last_quality_check: datetime | None = None
    last_data_refresh: datetime | None = None
    last_bse_ingestion: datetime | None = None
    last_trends_ingestion: datetime | None = None
    last_intraday_poll: datetime | None = None
    last_fii_dii_ingestion: datetime | None = None
    last_signal_strategies: datetime | None = None


def _run_async(coro: object) -> object:
    """Run an async coroutine from sync context.

    Resets DB engine singletons so each asyncio.run() gets a fresh
    connection pool on its own event loop.
    """
    import alphavedha.data.database as _db

    _db._engine = None
    _db._session_factory = None
    return asyncio.run(coro)  # type: ignore[arg-type]


def _now_ist() -> datetime:
    return datetime.now(IST)


def _direction_sign(value: float) -> int:
    """Collapse a direction/return into -1, 0, or +1."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


@dataclass
class EvaluationSummary:
    """Outcome of evaluating matured paper trades for one day."""

    n_evaluated: int = 0
    n_correct: int = 0
    directional_returns: list[float] = field(default_factory=list)


async def _latest_close(symbol: str, as_of: date) -> float | None:
    """Fetch the most recent close price for a symbol from the OHLCV store."""
    from alphavedha.data.store import load_ohlcv

    df = await load_ohlcv(symbol, as_of - timedelta(days=ENTRY_PRICE_LOOKBACK_DAYS), as_of)
    if df.empty:
        return None
    return float(df["close"].iloc[-1])


async def _persist_paper_trades(
    predictions: Sequence[StockPrediction],
    prediction_date: date,
) -> int:
    """Persist scan predictions as paper trades. Per-symbol failures are logged, not raised."""
    from alphavedha.data.store import store_paper_trade

    persisted = 0
    for pred in predictions:
        try:
            entry_price: float | None = None
            try:
                entry_price = await _latest_close(pred.symbol, prediction_date)
            except Exception as e:
                logger.warning("entry_price_unavailable", symbol=pred.symbol, error=str(e))
            if entry_price is None:
                logger.warning("entry_price_missing", symbol=pred.symbol)

            await store_paper_trade(
                {
                    "symbol": pred.symbol,
                    "prediction_date": prediction_date,
                    "strategy": "ensemble_v1",
                    "predicted_direction": int(pred.direction),
                    "predicted_magnitude": float(pred.magnitude),
                    "confidence": float(pred.meta_confidence),
                    "model_version": pred.model_version,
                    "regime": pred.regime,
                    "is_tradeable": bool(pred.is_tradeable),
                    "entry_price": entry_price,
                }
            )
            persisted += 1
        except Exception as e:
            logger.error("paper_trade_persist_failed", symbol=pred.symbol, error=str(e))
    return persisted


async def _persist_signal_paper_trades(
    signals: list[dict[str, Any]],
    strategy: str,
    prediction_date: date,
) -> int:
    """Persist signal-generated paper trades. Each signal dict needs symbol, direction, confidence."""
    from alphavedha.data.store import store_paper_trade

    persisted = 0
    for sig in signals:
        symbol = str(sig["symbol"])
        try:
            entry_price = await _latest_close(symbol, prediction_date)
            if entry_price is None:
                logger.warning("signal_entry_price_missing", symbol=symbol, strategy=strategy)

            await store_paper_trade(
                {
                    "symbol": symbol,
                    "prediction_date": prediction_date,
                    "strategy": strategy,
                    "predicted_direction": int(sig["direction"]),
                    "predicted_magnitude": 0.0,
                    "confidence": float(sig["confidence"]),
                    "model_version": strategy,
                    "regime": None,
                    "is_tradeable": True,
                    "entry_price": entry_price,
                }
            )
            persisted += 1
        except Exception as e:
            logger.error(
                "signal_paper_trade_failed", symbol=symbol, strategy=strategy, error=str(e)
            )
    return persisted


async def _evaluate_open_paper_trades(as_of: date) -> EvaluationSummary:
    """Evaluate open paper trades that are past the 15-trading-day hold horizon.

    A trade is "open" when exit_price is null. Outcomes are computed against
    the close EVALUATION_HORIZON_TRADING_DAYS trading days after prediction_date
    (or the latest available close if fewer bars exist).
    """
    import pandas as pd

    from alphavedha.data.store import load_ohlcv, load_paper_trades, update_paper_trade_outcome

    summary = EvaluationSummary()
    cutoff = as_of - timedelta(days=EVALUATION_MIN_CALENDAR_DAYS)
    trades = await load_paper_trades(end=cutoff)
    if trades.empty:
        logger.info("evaluation_no_matured_trades", as_of=str(as_of))
        return summary

    open_trades = trades[trades["exit_price"].isna()]
    for _, trade in open_trades.iterrows():
        symbol = str(trade["symbol"])
        prediction_date: date = trade["prediction_date"]
        try:
            entry_price = trade["entry_price"]
            if entry_price is None or pd.isna(entry_price) or float(entry_price) <= 0:
                logger.warning(
                    "paper_trade_skipped_no_entry_price",
                    symbol=symbol,
                    prediction_date=str(prediction_date),
                )
                continue

            df = await load_ohlcv(symbol, prediction_date, as_of)
            if df.empty:
                logger.warning("paper_trade_no_price_data", symbol=symbol)
                continue

            future = df[df.index.date > prediction_date]
            if future.empty:
                logger.warning("paper_trade_no_bars_after_entry", symbol=symbol)
                continue

            horizon_idx = min(EVALUATION_HORIZON_TRADING_DAYS, len(future)) - 1
            exit_price = float(future["close"].iloc[horizon_idx])
            actual_return = (exit_price - float(entry_price)) / float(entry_price)
            direction = int(trade["predicted_direction"])
            is_correct = _direction_sign(direction) == _direction_sign(actual_return)

            await update_paper_trade_outcome(
                symbol=symbol,
                prediction_date=prediction_date,
                exit_price=exit_price,
                actual_return=actual_return,
                is_correct=is_correct,
                strategy=str(trade.get("strategy", "ensemble_v1")),
            )

            summary.n_evaluated += 1
            if is_correct:
                summary.n_correct += 1
            summary.directional_returns.append(_direction_sign(direction) * actual_return)
        except Exception as e:
            logger.error(
                "paper_trade_evaluation_failed",
                symbol=symbol,
                prediction_date=str(prediction_date),
                error=str(e),
            )

    logger.info(
        "paper_trades_evaluated",
        as_of=str(as_of),
        evaluated=summary.n_evaluated,
        correct=summary.n_correct,
    )
    return summary


async def _store_pnl_summary(as_of: date, summary: EvaluationSummary) -> None:
    """Persist a DailyPnL row summarizing today's evaluated paper trade returns."""
    from alphavedha.data.store import load_daily_pnl, store_daily_pnl

    existing = await load_daily_pnl(start=as_of, end=as_of)
    if not existing.empty:
        logger.info("daily_pnl_already_recorded", date=str(as_of))
        return

    returns = summary.directional_returns
    daily_return = sum(returns) / len(returns) if returns else 0.0

    prior = await load_daily_pnl(end=as_of - timedelta(days=1))
    prev_value = (
        float(prior["portfolio_value"].iloc[-1]) if not prior.empty else INITIAL_PORTFOLIO_VALUE
    )
    portfolio_value = prev_value * (1.0 + daily_return)
    cumulative_return = portfolio_value / INITIAL_PORTFOLIO_VALUE - 1.0

    await store_daily_pnl(
        {
            "date": as_of,
            "portfolio_value": portfolio_value,
            "daily_return": daily_return,
            "cumulative_return": cumulative_return,
            "n_positions": summary.n_evaluated,
            "n_correct": summary.n_correct,
            "n_total_predictions": summary.n_evaluated,
        }
    )
    logger.info(
        "daily_pnl_stored",
        date=str(as_of),
        daily_return=daily_return,
        portfolio_value=portfolio_value,
        n_positions=summary.n_evaluated,
    )


def _send_strategy_summary(as_of: date) -> None:
    """Build and email the per-strategy daily summary after evaluation."""
    try:
        from alphavedha.data.store import load_paper_trades
        from alphavedha.monitoring.alerts import EmailAlerter
        from alphavedha.monitoring.strategy_summary import build_strategy_summary

        trades = _run_async(load_paper_trades())
        avoid_symbols: list[str] = []
        try:
            from alphavedha.data.universe import get_strategy_universe
            from alphavedha.intel.signals.blowup_score import compute_avoid_list, run_blowup_scores

            symbols: list[str] = _run_async(get_strategy_universe())  # type: ignore[assignment]
            scores = _run_async(run_blowup_scores(symbols, as_of=as_of))
            avoid_symbols = [s.symbol for s in compute_avoid_list(scores)]
        except Exception:
            pass

        report = build_strategy_summary(trades, as_of, avoid_list_symbols=avoid_symbols)
        text = report.format_text()
        alerter = EmailAlerter()
        alerter.strategy_daily_summary(text, str(as_of))
        logger.info(
            "strategy_summary_sent", date=str(as_of), strategies=len(report.strategy_sections)
        )
    except Exception as e:
        logger.warning("strategy_summary_failed", error=str(e))


class AlphaVedhaScheduler:
    """Manages all scheduled background jobs.

    Jobs:
        - daily_predictions: Run predictions for all large-cap stocks (8:30 AM IST)
        - daily_evaluation: Evaluate yesterday's predictions against actuals (3:45 PM IST)
        - weekly_drift_check: Check feature drift across all models (Saturday 8 PM)
        - monthly_retrain: Trigger retraining if needed (first Saturday of month, 10 PM)
    """

    def __init__(
        self,
        tier: str = "large",
        demo: bool = False,
    ) -> None:
        self._tier = tier
        self._demo = demo
        self._state = SchedulerState()

    @property
    def state(self) -> SchedulerState:
        return self._state

    def _record_job(self, result: JobResult) -> None:
        self._state.job_history.append(result)
        if len(self._state.job_history) > 100:
            self._state.job_history = self._state.job_history[-50:]

    def job_health_summary(self, last_n: int = 24) -> dict[str, list[dict[str, object]]]:
        """Summarise recent job outcomes grouped by job name.

        Returns ``{'jobs': [{'name': ..., 'last_run': ..., 'success': ..., 'error': ...}, ...]}``.
        """
        recent = self._state.job_history[-last_n:]
        seen: dict[str, JobResult] = {}
        for jr in recent:
            seen[jr.job_name] = jr

        rows: list[dict[str, object]] = []
        for name, jr in sorted(seen.items()):
            rows.append(
                {
                    "name": name,
                    "last_run": jr.started_at.isoformat() if jr.started_at else None,
                    "success": jr.success,
                    "error": jr.error,
                    "symbols_processed": jr.symbols_processed,
                }
            )
        return {"jobs": rows}

    def run_daily_predictions(self) -> JobResult:
        """Generate predictions for all stocks in the configured tier."""
        result = JobResult(job_name="daily_predictions", started_at=_now_ist())

        # NSE is closed on weekends; running anyway would persist Sat+Sun
        # cohorts at Friday's close — the same bet counted three times in the
        # track record. Exchange holidays still slip through this guard.
        if _now_ist().weekday() >= 5:
            logger.info("daily_predictions_skipped", reason="weekend — market closed")
            result.success = True
            result.error = "skipped: weekend"
            result.finished_at = _now_ist()
            self._record_job(result)
            return result

        logger.info("scheduler_job_start", job="daily_predictions", tier=self._tier)

        try:
            from alphavedha.config import get_config
            from alphavedha.services.cache import PredictionCache
            from alphavedha.services.model_registry import ModelRegistry
            from alphavedha.services.prediction_service import PredictionService

            config = get_config()
            registry = ModelRegistry(demo=self._demo)
            cache = PredictionCache(redis_client=None)
            service = PredictionService(registry=registry, cache=cache, config=config)

            from alphavedha.prediction.ranker import StockRanker

            predictions = _run_async(service.predict_tier(self._tier))
            ranking = StockRanker().rank(predictions, top_n=50)

            # Persist ALL predictions (not only tradeable candidates) so the
            # track record measures model accuracy even on days the
            # meta-labeling gate keeps every position closed.
            persisted = 0
            if self._demo:
                logger.info("paper_trade_persistence_skipped", reason="demo mode")
            else:
                prediction_date = _now_ist().date()
                persisted = _run_async(_persist_paper_trades(predictions, prediction_date))

            result.symbols_processed = len(predictions)
            result.success = True
            self._state.last_prediction_run = _now_ist()

            logger.info(
                "scheduler_job_complete",
                job="daily_predictions",
                buys=len(ranking.buy_candidates),
                sells=len(ranking.sell_candidates),
                paper_trades_persisted=persisted,
            )
        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="daily_predictions", error=str(e))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_signal_strategies(self) -> JobResult:
        """Run intel-based signal strategies and persist as paper trades."""
        result = JobResult(job_name="signal_strategies", started_at=_now_ist())

        if _now_ist().weekday() >= 5:
            logger.info("signal_strategies_skipped", reason="weekend")
            result.success = True
            result.error = "skipped: weekend"
            result.finished_at = _now_ist()
            self._record_job(result)
            return result

        logger.info("scheduler_job_start", job="signal_strategies")

        try:
            if self._demo:
                logger.info("signal_strategies_skipped", reason="demo mode")
            else:
                prediction_date = _now_ist().date()
                total_persisted = 0

                from alphavedha.data.universe import get_strategy_universe
                from alphavedha.intel.signals.blowup_score import (
                    STRATEGY_NAME as BLOWUP_STRATEGY,
                )
                from alphavedha.intel.signals.blowup_score import (
                    BlowupScore,
                    compute_avoid_list,
                    run_blowup_scores,
                )
                from alphavedha.intel.signals.event_drift import (
                    STRATEGY_NAME as EVENT_DRIFT_STRATEGY,
                )
                from alphavedha.intel.signals.event_drift import (
                    EventDriftSignal,
                    run_event_drift_signals,
                )
                from alphavedha.intel.signals.guidance_delta import (
                    STRATEGY_NAME as GUIDANCE_STRATEGY,
                )
                from alphavedha.intel.signals.guidance_delta import (
                    GuidanceDeltaSignal,
                    run_guidance_delta_signals,
                )
                from alphavedha.intel.signals.insider_cluster import (
                    STRATEGY_NAME as INSIDER_STRATEGY,
                )
                from alphavedha.intel.signals.insider_cluster import (
                    InsiderClusterSignal,
                    run_insider_cluster_signals,
                )

                symbols: list[str] = _run_async(  # type: ignore[assignment]
                    get_strategy_universe(self._tier)
                )

                # 1. Blowup scores → avoid list
                scores: list[BlowupScore] = _run_async(  # type: ignore[assignment]
                    run_blowup_scores(symbols, as_of=prediction_date)
                )
                avoid_list = compute_avoid_list(scores)
                avoid_symbols = frozenset(s.symbol for s in avoid_list)

                # Persist blowup_short_v1 paper trades for new avoid-list entries
                if avoid_list:
                    blowup_signals = [
                        {"symbol": s.symbol, "direction": -1, "confidence": s.total_score / 100.0}
                        for s in avoid_list
                    ]
                    n: int = _run_async(  # type: ignore[assignment]
                        _persist_signal_paper_trades(
                            blowup_signals, BLOWUP_STRATEGY, prediction_date
                        )
                    )
                    total_persisted += n
                    logger.info(
                        "signal_blowup_persisted",
                        count=n,
                        avoid_symbols=list(avoid_symbols),
                    )

                # 2. Event drift
                event_signals: list[EventDriftSignal] = _run_async(  # type: ignore[assignment]
                    run_event_drift_signals(prediction_date)
                )
                if event_signals:
                    event_dicts = [
                        {"symbol": s.symbol, "direction": s.direction, "confidence": s.confidence}
                        for s in event_signals
                        if s.symbol not in avoid_symbols
                    ]
                    n = _run_async(  # type: ignore[assignment]
                        _persist_signal_paper_trades(
                            event_dicts, EVENT_DRIFT_STRATEGY, prediction_date
                        )
                    )
                    total_persisted += n
                    logger.info("signal_event_drift_persisted", count=n)

                # 3. Insider cluster
                insider_signals: list[InsiderClusterSignal] = _run_async(  # type: ignore[assignment]
                    run_insider_cluster_signals(
                        symbols, signal_date=prediction_date, avoid_symbols=avoid_symbols
                    )
                )
                if insider_signals:
                    insider_dicts = [
                        {"symbol": s.symbol, "direction": s.direction, "confidence": s.confidence}
                        for s in insider_signals
                    ]
                    n = _run_async(  # type: ignore[assignment]
                        _persist_signal_paper_trades(
                            insider_dicts, INSIDER_STRATEGY, prediction_date
                        )
                    )
                    total_persisted += n
                    logger.info("signal_insider_cluster_persisted", count=n)

                # 4. Guidance delta
                guidance_signals: list[GuidanceDeltaSignal] = _run_async(  # type: ignore[assignment]
                    run_guidance_delta_signals(prediction_date)
                )
                if guidance_signals:
                    guidance_dicts = [
                        {"symbol": s.symbol, "direction": s.direction, "confidence": s.confidence}
                        for s in guidance_signals
                    ]
                    n = _run_async(  # type: ignore[assignment]
                        _persist_signal_paper_trades(
                            guidance_dicts, GUIDANCE_STRATEGY, prediction_date
                        )
                    )
                    total_persisted += n
                    logger.info("signal_guidance_delta_persisted", count=n)

                result.symbols_processed = total_persisted
                logger.info(
                    "scheduler_job_complete",
                    job="signal_strategies",
                    total_persisted=total_persisted,
                    avoid_list_size=len(avoid_list),
                )

            result.success = True
            self._state.last_signal_strategies = _now_ist()
        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="signal_strategies", error=str(e))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_prediction_hash(self) -> JobResult:
        """Hash today's paper trades and publish the proof."""
        result = JobResult(job_name="prediction_hash", started_at=_now_ist())

        if _now_ist().weekday() >= 5:
            logger.info("prediction_hash_skipped", reason="weekend")
            result.success = True
            result.error = "skipped: weekend"
            result.finished_at = _now_ist()
            self._record_job(result)
            return result

        logger.info("scheduler_job_start", job="prediction_hash")

        try:
            if self._demo:
                logger.info("prediction_hash_skipped", reason="demo mode")
                result.success = True
            else:
                from alphavedha.verification.publisher import publish_daily_proof

                proof: dict[str, Any] = _run_async(publish_daily_proof())  # type: ignore[assignment]
                result.symbols_processed = proof.get("n_predictions", 0)
                logger.info("prediction_hash_complete", **proof)

                # A proof that never reached the git repo is not a proof —
                # anything short of "published" is a job failure so the
                # alerting path fires instead of two silent weeks.
                status = str(proof.get("status", "unknown"))
                if status in ("published", "skipped_no_trades"):
                    result.success = True
                else:
                    result.error = proof.get("publish_error") or f"proof status: {status}"
                    logger.error("scheduler_job_failed", job="prediction_hash", error=result.error)
        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="prediction_hash", error=str(e))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_proof_reveal(self) -> JobResult:
        """Reveal canonical payloads for proofs at least 21 days old (P0-D2).

        The reveal is what makes the hash record independently checkable:
        anyone can re-hash the revealed JSON and compare it to the hash that
        was committed before market open weeks earlier.
        """
        result = JobResult(job_name="proof_reveal", started_at=_now_ist())
        logger.info("scheduler_job_start", job="proof_reveal")

        try:
            if self._demo:
                logger.info("proof_reveal_skipped", reason="demo mode")
            else:
                from alphavedha.verification.publisher import reveal_due_proofs

                summary: dict[str, Any] = _run_async(reveal_due_proofs())  # type: ignore[assignment]
                result.symbols_processed = int(summary.get("revealed", 0))
                logger.info("scheduler_job_complete", job="proof_reveal", **summary)

            result.success = True
        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="proof_reveal", error=str(e))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_daily_evaluation(self) -> JobResult:
        """Evaluate matured paper trades (15-trading-day hold) against actual outcomes.

        Loads open paper trades whose prediction_date is at least
        EVALUATION_MIN_CALENDAR_DAYS old, fetches actual prices from the OHLCV
        store, persists per-trade outcomes, and records a DailyPnL summary row.
        """
        result = JobResult(job_name="daily_evaluation", started_at=_now_ist())
        logger.info("scheduler_job_start", job="daily_evaluation")

        try:
            if self._demo:
                logger.info("evaluation_skipped", reason="demo mode")
            else:
                as_of = _now_ist().date()
                summary = _run_async(_evaluate_open_paper_trades(as_of))
                result.symbols_processed = summary.n_evaluated

                if summary.n_evaluated > 0:
                    _run_async(_store_pnl_summary(as_of, summary))
                else:
                    logger.info("daily_pnl_skipped", reason="no trades evaluated")

                _send_strategy_summary(as_of)

            result.success = True
            self._state.last_evaluation_run = _now_ist()
            logger.info(
                "scheduler_job_complete",
                job="daily_evaluation",
                evaluated=result.symbols_processed,
            )
        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="daily_evaluation", error=str(e))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_stop_loss_check(self) -> JobResult:
        """Close open paper trades whose ATR stop or target was hit today.

        Runs after the 17:00 data refresh so the day's high/low is in the
        OHLCV store. Without this job the engine's stop/target levels are
        display-only and every trade rides the full 15-day hold (FIX-08).
        """
        result = JobResult(job_name="stop_loss_check", started_at=_now_ist())

        if _now_ist().weekday() >= 5:
            logger.info("stop_loss_check_skipped", reason="weekend")
            result.success = True
            result.error = "skipped: weekend"
            result.finished_at = _now_ist()
            self._record_job(result)
            return result

        logger.info("scheduler_job_start", job="stop_loss_check")

        try:
            if self._demo:
                logger.info("stop_loss_check_skipped", reason="demo mode")
            else:
                from alphavedha.services.stop_evaluation import evaluate_stop_hits

                summary: dict[str, int] = _run_async(  # type: ignore[assignment]
                    evaluate_stop_hits(_now_ist().date())
                )
                result.symbols_processed = summary.get("evaluated", 0)
                logger.info("scheduler_job_complete", job="stop_loss_check", **summary)

            result.success = True
        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="stop_loss_check", error=str(e))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_data_refresh(self) -> JobResult:
        """Ingest the latest OHLCV data for the configured tier after market close.

        Without this, the OHLCV store stays frozen at the last backfill —
        predictions go stale and matured paper trades can never be evaluated
        (evaluation needs closes from after the prediction date).
        """
        result = JobResult(job_name="daily_data_refresh", started_at=_now_ist())
        logger.info("scheduler_job_start", job="daily_data_refresh", tier=self._tier)

        try:
            from alphavedha.data.ingestion import refresh_latest

            ingestion = _run_async(refresh_latest(self._tier, lookback_days=5))
            result.symbols_processed = ingestion.symbols_succeeded
            result.success = True
            self._state.last_data_refresh = _now_ist()

            logger.info(
                "scheduler_job_complete",
                job="daily_data_refresh",
                symbols_succeeded=ingestion.symbols_succeeded,
                symbols_failed=ingestion.symbols_failed,
                rows_stored=ingestion.total_rows_stored,
            )
        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="daily_data_refresh", error=str(e))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_fii_dii_ingestion(self) -> JobResult:
        """Ingest today's FII/DII participation data from NSE (runs daily at 18:30 IST).

        NSE publishes net buy/sell data for FII/FPI and DII by ~17:30 IST.
        This feeds the macro_fii_net / macro_dii_net features used by every
        model — without it those 4 features are always NaN.
        """
        result = JobResult(job_name="daily_fii_dii_ingestion", started_at=_now_ist())
        logger.info("scheduler_job_start", job="daily_fii_dii_ingestion")

        try:
            from alphavedha.data.ingestion import ingest_fii_dii

            fii_result = _run_async(ingest_fii_dii())
            result.symbols_processed = fii_result.rows_stored
            result.success = fii_result.error is None

            if fii_result.error:
                result.error = fii_result.error
                logger.warning(
                    "scheduler_job_partial",
                    job="daily_fii_dii_ingestion",
                    error=fii_result.error,
                )
            else:
                logger.info(
                    "scheduler_job_complete",
                    job="daily_fii_dii_ingestion",
                    rows_stored=fii_result.rows_stored,
                    categories=fii_result.categories,
                )

            self._state.last_fii_dii_ingestion = _now_ist()
        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="daily_fii_dii_ingestion", error=str(e))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_bhavcopy_ingestion(self) -> JobResult:
        """Ingest today's NSE bhavcopy — whole-market EOD OHLCV in one file."""
        result = JobResult(job_name="daily_bhavcopy_ingestion", started_at=_now_ist())

        if _now_ist().weekday() >= 5:
            logger.info("bhavcopy_ingestion_skipped", reason="weekend")
            result.success = True
            result.error = "skipped: weekend"
            result.finished_at = _now_ist()
            self._record_job(result)
            return result

        logger.info("scheduler_job_start", job="daily_bhavcopy_ingestion")

        try:
            if self._demo:
                logger.info("bhavcopy_ingestion_skipped", reason="demo mode")
            else:
                from alphavedha.intel.collectors.bhavcopy import ingest_bhavcopy

                today = _now_ist().date()
                rows = _run_async(ingest_bhavcopy(today))
                result.symbols_processed = rows
                logger.info(
                    "scheduler_job_complete",
                    job="daily_bhavcopy_ingestion",
                    rows=rows,
                )

            result.success = True
        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="daily_bhavcopy_ingestion", error=str(e))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_bse_ann_ingestion(self) -> JobResult:
        """Ingest today's BSE announcements with PDF text extraction."""
        result = JobResult(job_name="daily_bse_ann_ingestion", started_at=_now_ist())

        if _now_ist().weekday() >= 5:
            logger.info("bse_ann_ingestion_skipped", reason="weekend")
            result.success = True
            result.error = "skipped: weekend"
            result.finished_at = _now_ist()
            self._record_job(result)
            return result

        logger.info("scheduler_job_start", job="daily_bse_ann_ingestion")

        try:
            if self._demo:
                logger.info("bse_ann_ingestion_skipped", reason="demo mode")
            else:
                from alphavedha.intel.collectors.bse_announcements import (
                    ingest_bse_announcements_daily,
                )

                rows = _run_async(ingest_bse_announcements_daily())
                result.symbols_processed = rows
                logger.info(
                    "scheduler_job_complete",
                    job="daily_bse_ann_ingestion",
                    disclosures=rows,
                )

            result.success = True
        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="daily_bse_ann_ingestion", error=str(e))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_nse_ann_ingestion(self) -> JobResult:
        """Ingest today's NSE announcements with PIT/SAST flagging."""
        result = JobResult(job_name="daily_nse_ann_ingestion", started_at=_now_ist())

        if _now_ist().weekday() >= 5:
            logger.info("nse_ann_ingestion_skipped", reason="weekend")
            result.success = True
            result.error = "skipped: weekend"
            result.finished_at = _now_ist()
            self._record_job(result)
            return result

        logger.info("scheduler_job_start", job="daily_nse_ann_ingestion")

        try:
            if self._demo:
                logger.info("nse_ann_ingestion_skipped", reason="demo mode")
            else:
                from alphavedha.intel.collectors.nse_announcements import (
                    ingest_nse_announcements_daily,
                )

                counts = _run_async(ingest_nse_announcements_daily())
                result.symbols_processed = counts.get("disclosures", 0)  # type: ignore[union-attr]
                logger.info(
                    "scheduler_job_complete",
                    job="daily_nse_ann_ingestion",
                    **counts,  # type: ignore[arg-type]
                )

            result.success = True
        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="daily_nse_ann_ingestion", error=str(e))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_insider_trades_ingestion(self) -> JobResult:
        """Derive insider trades from classified disclosure events.

        NSE discontinued the corporates-pit JSON API (~2026-04-28): it
        returns 200 with an empty dataset for any recent window, so the
        old per-symbol fetch loop produced nothing while burning ~100s of
        rate-limited calls nightly. The PIT/SAST filings still arrive
        through the disclosures pipeline, where the LLM layer classifies
        them into insider_buy/insider_sell events — those events are now
        the source of record (see intel/insider_derivation.py).
        """
        result = JobResult(job_name="daily_insider_trades_ingestion", started_at=_now_ist())

        if _now_ist().weekday() >= 5:
            logger.info("insider_trades_ingestion_skipped", reason="weekend")
            result.success = True
            result.error = "skipped: weekend"
            result.finished_at = _now_ist()
            self._record_job(result)
            return result

        logger.info("scheduler_job_start", job="daily_insider_trades_ingestion")

        try:
            if self._demo:
                logger.info("insider_trades_ingestion_skipped", reason="demo mode")
            else:
                from alphavedha.intel.insider_derivation import derive_insider_trades

                count: int = _run_async(derive_insider_trades())  # type: ignore[assignment]
                result.symbols_processed = int(count)
                logger.info(
                    "scheduler_job_complete",
                    job="daily_insider_trades_ingestion",
                    trades_stored=count,
                )

            result.success = True
        except Exception as e:
            result.error = str(e)
            logger.error(
                "scheduler_job_failed",
                job="daily_insider_trades_ingestion",
                error=str(e),
            )

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_surveillance_ingestion(self) -> JobResult:
        """Ingest current ASM/GSM surveillance lists from NSE."""
        result = JobResult(job_name="daily_surveillance_ingestion", started_at=_now_ist())

        if _now_ist().weekday() >= 5:
            logger.info("surveillance_ingestion_skipped", reason="weekend")
            result.success = True
            result.error = "skipped: weekend"
            result.finished_at = _now_ist()
            self._record_job(result)
            return result

        logger.info("scheduler_job_start", job="daily_surveillance_ingestion")

        try:
            if self._demo:
                logger.info("surveillance_ingestion_skipped", reason="demo mode")
            else:
                from alphavedha.intel.collectors.surveillance import (
                    ingest_surveillance_daily,
                )

                count = _run_async(ingest_surveillance_daily())
                result.symbols_processed = int(count)  # type: ignore[arg-type]
                logger.info(
                    "scheduler_job_complete",
                    job="daily_surveillance_ingestion",
                    flags_stored=count,
                )

            result.success = True
        except Exception as e:
            result.error = str(e)
            logger.error(
                "scheduler_job_failed",
                job="daily_surveillance_ingestion",
                error=str(e),
            )

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_deals_ingestion(self) -> JobResult:
        """Ingest today's bulk/block/short deals from NSE."""
        result = JobResult(job_name="daily_deals_ingestion", started_at=_now_ist())

        if _now_ist().weekday() >= 5:
            logger.info("deals_ingestion_skipped", reason="weekend")
            result.success = True
            result.error = "skipped: weekend"
            result.finished_at = _now_ist()
            self._record_job(result)
            return result

        logger.info("scheduler_job_start", job="daily_deals_ingestion")

        try:
            if self._demo:
                logger.info("deals_ingestion_skipped", reason="demo mode")
            else:
                from alphavedha.intel.collectors.bulk_block_deals import (
                    ingest_bulk_block_deals_daily,
                )

                count = _run_async(ingest_bulk_block_deals_daily())
                result.symbols_processed = int(count)  # type: ignore[arg-type]
                logger.info(
                    "scheduler_job_complete",
                    job="daily_deals_ingestion",
                    deals_stored=count,
                )

            result.success = True
        except Exception as e:
            result.error = str(e)
            logger.error(
                "scheduler_job_failed",
                job="daily_deals_ingestion",
                error=str(e),
            )

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_credit_rating_ingestion(self) -> JobResult:
        """Ingest credit rating actions from NSE announcements."""
        result = JobResult(job_name="daily_credit_rating_ingestion", started_at=_now_ist())

        if _now_ist().weekday() >= 5:
            logger.info("credit_rating_ingestion_skipped", reason="weekend")
            result.success = True
            result.error = "skipped: weekend"
            result.finished_at = _now_ist()
            self._record_job(result)
            return result

        logger.info("scheduler_job_start", job="daily_credit_rating_ingestion")

        try:
            if self._demo:
                logger.info("credit_rating_ingestion_skipped", reason="demo mode")
            else:
                from alphavedha.intel.collectors.credit_ratings import (
                    ingest_credit_ratings_daily,
                )

                count = _run_async(ingest_credit_ratings_daily())
                result.symbols_processed = int(count)  # type: ignore[arg-type]
                logger.info(
                    "scheduler_job_complete",
                    job="daily_credit_rating_ingestion",
                    events_stored=count,
                )

            result.success = True
        except Exception as e:
            result.error = str(e)
            logger.error(
                "scheduler_job_failed",
                job="daily_credit_rating_ingestion",
                error=str(e),
            )

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_transcript_ingestion(self) -> JobResult:
        """Ingest concall transcripts from NSE announcements."""
        result = JobResult(job_name="daily_transcript_ingestion", started_at=_now_ist())

        if _now_ist().weekday() >= 5:
            logger.info("transcript_ingestion_skipped", reason="weekend")
            result.success = True
            result.error = "skipped: weekend"
            result.finished_at = _now_ist()
            self._record_job(result)
            return result

        logger.info("scheduler_job_start", job="daily_transcript_ingestion")

        try:
            if self._demo:
                logger.info("transcript_ingestion_skipped", reason="demo mode")
            else:
                from alphavedha.intel.collectors.transcripts import (
                    ingest_transcripts_daily,
                )

                count = _run_async(ingest_transcripts_daily())
                result.symbols_processed = int(count)  # type: ignore[arg-type]
                logger.info(
                    "scheduler_job_complete",
                    job="daily_transcript_ingestion",
                    transcripts_stored=count,
                )

            result.success = True
        except Exception as e:
            result.error = str(e)
            logger.error(
                "scheduler_job_failed",
                job="daily_transcript_ingestion",
                error=str(e),
            )

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_intel_quality_check(self) -> JobResult:
        """Check intel table row counts and disk usage after all collectors finish."""
        result = JobResult(job_name="intel_quality_check", started_at=_now_ist())
        logger.info("scheduler_job_start", job="intel_quality_check")

        try:
            if self._demo:
                logger.info("intel_quality_check_skipped", reason="demo mode")
            else:
                from alphavedha.intel.quality import check_disk_usage, check_intel_row_counts
                from alphavedha.monitoring.alerts import EmailAlerter

                async def _task() -> None:
                    from alphavedha.data.database import get_session_factory

                    factory = get_session_factory()
                    async with factory() as session:
                        intel_results = await check_intel_row_counts(session)
                        failed = [r for r in intel_results if not r.passed]
                        if failed:
                            alerter = EmailAlerter()
                            lines = [f"Intel quality failures ({len(failed)}):"]
                            for f in failed:
                                lines.append(f"  [{f.severity}] {f.detail}")
                            alerter.send(
                                subject=f"Intel quality: {len(failed)} check(s) failed",
                                body="\n".join(lines),
                            )

                _run_async(_task())

                disk = check_disk_usage()
                if not disk.passed:
                    alerter = EmailAlerter()
                    alerter.send(
                        subject=f"Disk usage {disk.severity}: {disk.detail}",
                        body=disk.detail,
                    )

                logger.info(
                    "scheduler_job_complete",
                    job="intel_quality_check",
                    disk=disk.detail,
                )

            result.success = True
        except Exception as e:
            result.error = str(e)
            logger.error(
                "scheduler_job_failed",
                job="intel_quality_check",
                error=str(e),
            )

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_intel_extraction(self) -> JobResult:
        """Run LLM extraction on unprocessed disclosures."""
        result = JobResult(job_name="intel_extraction", started_at=_now_ist())

        if _now_ist().weekday() >= 5:
            logger.info("intel_extraction_skipped", reason="weekend")
            result.success = True
            result.finished_at = _now_ist()
            self._record_job(result)
            return result

        logger.info("scheduler_job_start", job="intel_extraction")

        try:
            if self._demo:
                logger.info("intel_extraction_skipped", reason="demo mode")
            else:
                from alphavedha.intel.extraction.batcher import run_nightly_extraction

                summary = _run_async(run_nightly_extraction())
                logger.info("scheduler_job_complete", job="intel_extraction", **summary)

                # Freshly extracted insider_buy/sell events become
                # insider_trades rows the same night, so tomorrow's 06:05
                # signal run sees them (idempotent with the 19:20 pass).
                try:
                    from alphavedha.intel.insider_derivation import derive_insider_trades

                    derived: int = _run_async(derive_insider_trades())  # type: ignore[assignment]
                    logger.info("intel_extraction_insider_derivation", rows=derived)
                except Exception as e:
                    logger.warning("intel_extraction_insider_derivation_failed", error=str(e))

            result.success = True
        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="intel_extraction", error=str(e))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_drift_check(self) -> JobResult:
        """Check feature distribution drift across all monitored features."""
        result = JobResult(job_name="weekly_drift_check", started_at=_now_ist())
        logger.info("scheduler_job_start", job="weekly_drift_check")

        try:
            from alphavedha.config import get_config
            from alphavedha.monitoring.drift import DriftDetector

            config = get_config()
            DriftDetector(config.monitoring.drift)

            result.success = True
            self._state.last_drift_check = _now_ist()
            logger.info("scheduler_job_complete", job="weekly_drift_check")
        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="weekly_drift_check", error=str(e))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_monthly_retrain(self) -> JobResult:
        """Check if retraining is needed and trigger if so."""
        result = JobResult(job_name="monthly_retrain", started_at=_now_ist())
        logger.info("scheduler_job_start", job="monthly_retrain")

        try:
            from alphavedha.monitoring.retrainer import RetrainingManager

            manager = RetrainingManager()
            decision = manager.should_retrain()

            if decision.should_retrain:
                logger.info(
                    "retrain_triggered",
                    reason=decision.reason,
                )
                result.success = True
            else:
                logger.info("retrain_skipped", reason=decision.reason)
                result.success = True

            self._state.last_retrain = _now_ist()
        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="monthly_retrain", error=str(e))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_daily_xgboost_retrain(self) -> JobResult:
        """Retrain XGBoost nightly after market-close data is ingested.

        Runs on cx23 (2 vCPU, 4GB) — XGBoost fits comfortably within that budget.
        LSTM/TFT are NOT retrained here; those need more RAM and run weekly.
        """
        result = JobResult(job_name="xgboost_retrain", started_at=_now_ist())
        logger.info("scheduler_job_start", job="xgboost_retrain", tier=self._tier)

        try:
            from alphavedha.training.pipeline import train_xgboost

            train_result = _run_async(train_xgboost(self._tier))
            result.success = train_result.artifact_path is not None
            result.symbols_processed = train_result.n_symbols
            self._state.last_xgboost_retrain = _now_ist()

            if result.success:
                logger.info(
                    "scheduler_job_complete",
                    job="xgboost_retrain",
                    symbols=train_result.n_symbols,
                    artifact=str(train_result.artifact_path),
                    elapsed_s=train_result.total_time_seconds,
                )
            else:
                result.error = "no artifact produced — check training logs"
                logger.warning("xgboost_retrain_no_artifact", tier=self._tier)

        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="xgboost_retrain", error=str(e))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_weekly_lstm_tft_retrain(self) -> JobResult:
        """Retrain LSTM and TFT models every Saturday night.

        These models need ~8-16GB RAM. On cx23 (4GB) they will OOM.
        This job is gated by the ALPHAVEDHA_HEAVY_TRAINING env var so it only
        runs when the server has been scaled up (or on a machine with enough RAM).
        Set ALPHAVEDHA_HEAVY_TRAINING=1 in .env.vps before running.
        """
        result = JobResult(job_name="lstm_tft_retrain", started_at=_now_ist())

        import os

        if not os.environ.get("ALPHAVEDHA_HEAVY_TRAINING"):
            logger.info(
                "lstm_tft_retrain_skipped",
                reason="ALPHAVEDHA_HEAVY_TRAINING not set — scale server to cx43 first",
            )
            result.success = True
            result.error = "skipped: ALPHAVEDHA_HEAVY_TRAINING not set"
            result.finished_at = _now_ist()
            self._record_job(result)
            return result

        logger.info("scheduler_job_start", job="lstm_tft_retrain", tier=self._tier)

        try:
            from alphavedha.training.pipeline import train_lstm, train_tft

            lstm_result = _run_async(train_lstm(self._tier))
            tft_result = _run_async(train_tft(self._tier))

            result.success = (
                lstm_result.artifact_path is not None and tft_result.artifact_path is not None
            )
            self._state.last_lstm_tft_retrain = _now_ist()

            logger.info(
                "scheduler_job_complete",
                job="lstm_tft_retrain",
                lstm_ok=lstm_result.artifact_path is not None,
                tft_ok=tft_result.artifact_path is not None,
            )

        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="lstm_tft_retrain", error=str(e))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_rebalance_check(self) -> JobResult:
        """Check if index compositions changed and log additions/removals."""
        result = JobResult(job_name="quarterly_rebalance_check", started_at=_now_ist())
        logger.info("scheduler_job_start", job="quarterly_rebalance_check")

        try:
            import yaml

            from alphavedha.data.universe import fetch_index_constituents

            stocks_path = Path("configs/stocks.yaml")
            if not stocks_path.exists():
                result.error = "configs/stocks.yaml not found"
                result.finished_at = _now_ist()
                self._record_job(result)
                return result

            with stocks_path.open() as f:
                stocks_cfg = yaml.safe_load(f)

            current_symbols: set[str] = set()
            for sector_symbols in stocks_cfg.get("sectors", {}).values():
                current_symbols.update(sector_symbols)

            live_df = _run_async(fetch_index_constituents("NIFTY 50"))
            live_symbols = set(live_df["symbol"].tolist()) if "symbol" in live_df.columns else set()

            additions = live_symbols - current_symbols
            removals = current_symbols - live_symbols

            if additions or removals:
                logger.warning(
                    "rebalance_changes_detected",
                    additions=sorted(additions),
                    removals=sorted(removals),
                )
            else:
                logger.info("rebalance_no_changes")

            result.symbols_processed = len(live_symbols)
            result.success = True
            self._state.last_rebalance_check = _now_ist()

            logger.info(
                "scheduler_job_complete",
                job="quarterly_rebalance_check",
                live_count=len(live_symbols),
                additions=len(additions),
                removals=len(removals),
            )
        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="quarterly_rebalance_check", error=str(e))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_bse_ingestion(self) -> JobResult:
        """Weekly BSE corporate announcements ingestion — runs Sunday night."""
        result = JobResult(job_name="bse_ingestion", started_at=_now_ist())
        logger.info("scheduler_job_start", job="bse_ingestion")

        try:
            from datetime import date, timedelta

            import yaml

            from alphavedha.data.database import get_session_factory
            from alphavedha.data.ingestion import ingest_bse_announcements

            stocks_path = Path("configs/stocks.yaml")
            if not stocks_path.exists():
                result.error = "configs/stocks.yaml not found"
                result.finished_at = _now_ist()
                self._record_job(result)
                return result

            with stocks_path.open() as f:
                stocks_cfg = yaml.safe_load(f)

            symbols: list[str] = []
            for sector_symbols in stocks_cfg.get("sectors", {}).values():
                symbols.extend(sector_symbols)

            today = date.today()
            start = today - timedelta(days=7)

            async def _bse_task() -> int:
                factory = get_session_factory()
                async with factory() as session:
                    return await ingest_bse_announcements(symbols, start, today, session)

            rows = _run_async(_bse_task())
            result.symbols_processed = len(symbols)
            result.success = True
            self._state.last_bse_ingestion = _now_ist()
            logger.info(
                "scheduler_job_complete",
                job="bse_ingestion",
                symbols=len(symbols),
                rows_upserted=rows,
            )
        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="bse_ingestion", error=str(e))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_trends_ingestion(self) -> JobResult:
        """Weekly Google Trends ingestion — runs Sunday night after BSE job."""
        result = JobResult(job_name="trends_ingestion", started_at=_now_ist())
        logger.info("scheduler_job_start", job="trends_ingestion")

        try:
            from alphavedha.data.ingestion import ingest_trends

            sector_data = _run_async(ingest_trends())
            result.symbols_processed = len(sector_data)
            result.success = True
            self._state.last_trends_ingestion = _now_ist()
            logger.info(
                "scheduler_job_complete",
                job="trends_ingestion",
                sectors_fetched=len(sector_data),
            )
        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="trends_ingestion", error=str(e))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_intraday_poll(self) -> JobResult:
        """Poll live prices every 2 minutes during market hours."""
        result = JobResult(job_name="intraday_poll", started_at=_now_ist())

        from alphavedha.data.live_feed import is_market_open

        if not is_market_open():
            result.success = True
            result.finished_at = _now_ist()
            return result

        logger.info("scheduler_job_start", job="intraday_poll")

        try:
            import yaml

            from alphavedha.data.database import get_session_factory
            from alphavedha.data.live_feed import LiveDataPoller

            stocks_path = Path("configs/stocks.yaml")
            if not stocks_path.exists():
                result.error = "configs/stocks.yaml not found"
                result.finished_at = _now_ist()
                self._record_job(result)
                return result

            with stocks_path.open() as f:
                stocks_cfg = yaml.safe_load(f)

            symbols: list[str] = []
            for sector_symbols in stocks_cfg.get("sectors", {}).values():
                symbols.extend(sector_symbols)

            poller = LiveDataPoller(symbols=symbols, session_factory=get_session_factory())
            poll_results = _run_async(poller.poll_once())

            successes = sum(1 for r in poll_results if r.success)
            result.symbols_processed = successes
            result.success = True
            self._state.last_intraday_poll = _now_ist()
            logger.info(
                "scheduler_job_complete",
                job="intraday_poll",
                symbols=len(symbols),
                successes=successes,
            )
        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="intraday_poll", error=str(e))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def setup_schedule(self) -> None:
        """Register all jobs with the schedule library."""
        schedule.every().day.at(PREDICTION_TIME).do(self.run_daily_predictions)
        schedule.every().day.at(SIGNAL_STRATEGIES_TIME).do(self.run_signal_strategies)
        schedule.every().day.at(PREDICTION_HASH_TIME).do(self.run_prediction_hash)
        schedule.every().day.at(PROOF_REVEAL_TIME).do(self.run_proof_reveal)
        schedule.every().day.at(EVALUATION_TIME).do(self.run_daily_evaluation)
        schedule.every().day.at(QUALITY_CHECK_TIME).do(self.run_quality_check)
        schedule.every().day.at(DATA_REFRESH_TIME).do(self.run_data_refresh)
        schedule.every().day.at(STOP_LOSS_CHECK_TIME).do(self.run_stop_loss_check)
        schedule.every().day.at(FII_DII_INGESTION_TIME).do(self.run_fii_dii_ingestion)
        schedule.every().day.at(BHAVCOPY_INGESTION_TIME).do(self.run_bhavcopy_ingestion)
        schedule.every().day.at(BSE_ANN_INGESTION_TIME).do(self.run_bse_ann_ingestion)
        schedule.every().day.at(NSE_ANN_INGESTION_TIME).do(self.run_nse_ann_ingestion)
        schedule.every().day.at(INSIDER_TRADES_INGESTION_TIME).do(self.run_insider_trades_ingestion)
        schedule.every().day.at(SURVEILLANCE_INGESTION_TIME).do(self.run_surveillance_ingestion)
        schedule.every().day.at(DEALS_INGESTION_TIME).do(self.run_deals_ingestion)
        schedule.every().day.at(CREDIT_RATING_INGESTION_TIME).do(self.run_credit_rating_ingestion)
        schedule.every().day.at(TRANSCRIPT_INGESTION_TIME).do(self.run_transcript_ingestion)
        schedule.every().day.at(INTEL_EXTRACTION_TIME).do(self.run_intel_extraction)
        schedule.every().day.at(INTEL_QUALITY_CHECK_TIME).do(self.run_intel_quality_check)
        schedule.every().day.at(XGBOOST_RETRAIN_TIME).do(self.run_daily_xgboost_retrain)
        schedule.every(INTRADAY_POLL_INTERVAL_MINUTES).minutes.do(self.run_intraday_poll)

        getattr(schedule.every(), BSE_INGESTION_DAY).at(BSE_INGESTION_TIME).do(
            self.run_bse_ingestion,
        )
        getattr(schedule.every(), BSE_INGESTION_DAY).at(TRENDS_INGESTION_TIME).do(
            self.run_trends_ingestion,
        )

        getattr(schedule.every(), DRIFT_CHECK_DAY).at(DRIFT_CHECK_TIME).do(
            self.run_drift_check,
        )

        getattr(schedule.every(), RETRAIN_DAY).at(RETRAIN_TIME).do(
            self._maybe_monthly_retrain,
        )

        # Heavy retrain (LSTM/TFT) needs ~16GB. On the autoscale deployment the
        # `train.yml` GitHub Actions cron owns this — it scales cx23→cx43, trains,
        # then scales back, and both fire Saturday 22:30 IST. Registering the
        # in-process job unconditionally would wire two owners to the same slot.
        # Only register it when this box is permanently large enough
        # (ALPHAVEDHA_HEAVY_TRAINING=1); in that mode, drop train.yml's `schedule`
        # cron so the scheduler is the single owner.
        if os.environ.get("ALPHAVEDHA_HEAVY_TRAINING"):
            getattr(schedule.every(), LSTM_TFT_RETRAIN_DAY).at(LSTM_TFT_RETRAIN_TIME).do(
                self.run_weekly_lstm_tft_retrain,
            )

        getattr(schedule.every(), REBALANCE_CHECK_DAY).at(REBALANCE_CHECK_TIME).do(
            self._maybe_quarterly_rebalance,
        )

        logger.info(
            "scheduler_configured",
            prediction_time=PREDICTION_TIME,
            signal_strategies_time=SIGNAL_STRATEGIES_TIME,
            prediction_hash_time=PREDICTION_HASH_TIME,
            evaluation_time=EVALUATION_TIME,
            data_refresh_time=DATA_REFRESH_TIME,
            fii_dii_ingestion_time=FII_DII_INGESTION_TIME,
            bhavcopy_ingestion_time=BHAVCOPY_INGESTION_TIME,
            bse_ann_ingestion_time=BSE_ANN_INGESTION_TIME,
            nse_ann_ingestion_time=NSE_ANN_INGESTION_TIME,
            insider_trades_ingestion_time=INSIDER_TRADES_INGESTION_TIME,
            surveillance_ingestion_time=SURVEILLANCE_INGESTION_TIME,
            deals_ingestion_time=DEALS_INGESTION_TIME,
            credit_rating_ingestion_time=CREDIT_RATING_INGESTION_TIME,
            transcript_ingestion_time=TRANSCRIPT_INGESTION_TIME,
            intel_quality_check_time=INTEL_QUALITY_CHECK_TIME,
            xgboost_retrain_time=XGBOOST_RETRAIN_TIME,
            lstm_tft_retrain_day=LSTM_TFT_RETRAIN_DAY,
            drift_day=DRIFT_CHECK_DAY,
            retrain_day=RETRAIN_DAY,
            rebalance_day=REBALANCE_CHECK_DAY,
        )

    def run_quality_check(self) -> JobResult:
        """Nightly data quality check — runs after market close data is available."""
        result = JobResult(job_name="quality_check", started_at=_now_ist())
        logger.info("scheduler_job_start", job="quality_check")
        try:
            from datetime import date

            from alphavedha.data.database import get_session_factory
            from alphavedha.data.quality import QualityChecker
            from alphavedha.monitoring.alerts import EmailAlerter

            today = date.today()

            async def _task() -> None:
                factory = get_session_factory()
                async with factory() as session:
                    checker = QualityChecker(session=session)
                    report = await checker.run_full_check(today)
                    await checker.persist_report(report)
                    if report.n_critical > 0:
                        EmailAlerter().data_quality_failed(report)

            _run_async(_task())
            result.success = True
            self._state.last_quality_check = _now_ist()
            logger.info("scheduler_job_complete", job="quality_check")
        except Exception as exc:
            result.success = False
            result.error = str(exc)
            logger.error("scheduler_job_failed", job="quality_check", error=str(exc))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def _maybe_monthly_retrain(self) -> JobResult | None:
        """Only run retrain on the first Saturday of the month."""
        now = _now_ist()
        if now.day <= 7:
            return self.run_monthly_retrain()
        logger.debug("retrain_skipped_not_first_week", day=now.day)
        return None

    def _maybe_quarterly_rebalance(self) -> JobResult | None:
        """Only run rebalance check during March and September."""
        now = _now_ist()
        if now.month in REBALANCE_MONTHS:
            return self.run_rebalance_check()
        logger.debug("rebalance_skipped_wrong_month", month=now.month)
        return None

    def _write_heartbeat(self) -> None:
        """Write a heartbeat file so the app container can check scheduler liveness."""
        import contextlib
        import json

        heartbeat_path = (
            Path(os.environ.get("ALPHAVEDHA_LOG_DIR", "/app/logs")) / "scheduler_heartbeat.json"
        )
        with contextlib.suppress(OSError):
            heartbeat_path.write_text(
                json.dumps(
                    {
                        "alive": True,
                        "last_beat": _now_ist().isoformat(),
                        "pid": os.getpid(),
                        "tier": self._tier,
                        "demo": self._demo,
                    }
                )
            )

    def run_forever(self, poll_interval: float = 60.0) -> None:
        """Start the scheduler loop. Blocks until interrupted."""
        self.setup_schedule()
        self._state.is_running = True

        logger.info("scheduler_started", tier=self._tier, demo=self._demo)

        try:
            while self._state.is_running:
                schedule.run_pending()
                self._write_heartbeat()
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info("scheduler_stopped_by_user")
        finally:
            self._state.is_running = False
            logger.info("scheduler_stopped")

    def stop(self) -> None:
        """Signal the scheduler to stop."""
        self._state.is_running = False


if __name__ == "__main__":
    import os

    _demo = os.environ.get("ALPHAVEDHA_DEMO", "").lower() in ("1", "true", "yes")
    _tier = os.environ.get("ALPHAVEDHA_TIER", "large")
    AlphaVedhaScheduler(tier=_tier, demo=_demo).run_forever()
