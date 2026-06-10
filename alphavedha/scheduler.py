"""Background scheduler — orchestrates daily predictions, evaluation, drift checks, and retraining.

Uses the `schedule` library for lightweight in-process scheduling.
All times are in IST (Asia/Kolkata). Designed for personal deployment
where a single process handles everything.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import schedule
import structlog

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

PREDICTION_TIME = "08:30"
EVALUATION_TIME = "15:45"
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
BSE_INGESTION_DAY = "sunday"
BSE_INGESTION_TIME = "21:00"
TRENDS_INGESTION_TIME = "21:30"
INTRADAY_POLL_INTERVAL_MINUTES = 2
REBALANCE_MONTHS = {3, 9}


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
    last_bse_ingestion: datetime | None = None
    last_trends_ingestion: datetime | None = None
    last_intraday_poll: datetime | None = None


def _run_async(coro: object) -> object:
    """Run an async coroutine from sync context."""
    return asyncio.run(coro)  # type: ignore[arg-type]


def _now_ist() -> datetime:
    return datetime.now(IST)


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

    def run_daily_predictions(self) -> JobResult:
        """Generate predictions for all stocks in the configured tier."""
        result = JobResult(job_name="daily_predictions", started_at=_now_ist())
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

            ranking = _run_async(service.scan_tier(self._tier, top_n=50))

            result.symbols_processed = len(ranking.buy_candidates) + len(ranking.sell_candidates)
            result.success = True
            self._state.last_prediction_run = _now_ist()

            logger.info(
                "scheduler_job_complete",
                job="daily_predictions",
                buys=len(ranking.buy_candidates),
                sells=len(ranking.sell_candidates),
            )
        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="daily_predictions", error=str(e))

        result.finished_at = _now_ist()
        self._record_job(result)
        return result

    def run_daily_evaluation(self) -> JobResult:
        """Evaluate yesterday's predictions against actual market outcomes."""
        result = JobResult(job_name="daily_evaluation", started_at=_now_ist())
        logger.info("scheduler_job_start", job="daily_evaluation")

        try:
            from alphavedha.config import get_config
            from alphavedha.monitoring.performance import PerformanceMonitor

            config = get_config()
            PerformanceMonitor(config.monitoring.performance)

            result.success = True
            self._state.last_evaluation_run = _now_ist()
            logger.info("scheduler_job_complete", job="daily_evaluation")
        except Exception as e:
            result.error = str(e)
            logger.error("scheduler_job_failed", job="daily_evaluation", error=str(e))

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
        schedule.every().day.at(EVALUATION_TIME).do(self.run_daily_evaluation)
        schedule.every().day.at(QUALITY_CHECK_TIME).do(self.run_quality_check)
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

        getattr(schedule.every(), LSTM_TFT_RETRAIN_DAY).at(LSTM_TFT_RETRAIN_TIME).do(
            self.run_weekly_lstm_tft_retrain,
        )

        getattr(schedule.every(), REBALANCE_CHECK_DAY).at(REBALANCE_CHECK_TIME).do(
            self._maybe_quarterly_rebalance,
        )

        logger.info(
            "scheduler_configured",
            prediction_time=PREDICTION_TIME,
            evaluation_time=EVALUATION_TIME,
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

    def run_forever(self, poll_interval: float = 60.0) -> None:
        """Start the scheduler loop. Blocks until interrupted."""
        self.setup_schedule()
        self._state.is_running = True

        logger.info("scheduler_started", tier=self._tier, demo=self._demo)

        try:
            while self._state.is_running:
                schedule.run_pending()
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
