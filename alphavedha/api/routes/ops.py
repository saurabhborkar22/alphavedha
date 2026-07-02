"""Ops endpoints — VPS health monitoring and job management for Claude Routines.

All endpoints require API key auth. Designed to be called by Anthropic
cloud Routines for automated monitoring and auto-healing.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import structlog
from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from alphavedha.api.deps import verify_api_key
from alphavedha.data.database import get_session_factory
from alphavedha.data.models import (
    BulkBlockDeal,
    DailyOHLCV,
    DailyPnL,
    Disclosure,
    InsiderTrade,
    InstitutionalFlow,
    IntradayOHLCV,
    PaperTrade,
    PredictionProof,
    RatingEvent,
    SurveillanceFlag,
    Transcript,
)
from alphavedha.intel.extraction.batcher import get_unprocessed_disclosures
from alphavedha.intel.extraction.extractor import is_boilerplate
from alphavedha.intel.store import mark_disclosures_processed, store_disclosure_events

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/ops", tags=["ops"], dependencies=[Depends(verify_api_key)])

IST = ZoneInfo("Asia/Kolkata")

_CRITICAL_TABLES: list[tuple[str, Any, str, str]] = [
    ("paper_trades", PaperTrade, "prediction_date", "08:30 predictions"),
    ("prediction_proofs", PredictionProof, "proof_date", "08:40 hash"),
    ("daily_ohlcv", DailyOHLCV, "date", "17:00 data refresh"),
    ("institutional_flows", InstitutionalFlow, "date", "18:30 FII/DII"),
    ("disclosures", Disclosure, "filed_at", "19:00-19:15 announcements"),
    ("insider_trades", InsiderTrade, "trade_date", "19:20 insider trades"),
    ("surveillance_flags", SurveillanceFlag, "added_on", "19:30 surveillance"),
    ("bulk_block_deals", BulkBlockDeal, "deal_date", "19:45 deals"),
    ("rating_events", RatingEvent, "filed_at", "19:50 credit ratings"),
    ("transcripts", Transcript, "filed_at", "20:15 transcripts"),
]

_OPTIONAL_TABLES: list[tuple[str, Any, str, str]] = [
    ("intraday_ohlcv", IntradayOHLCV, "last_updated", "intraday poll"),
    ("daily_pnl", DailyPnL, "date", "15:45 evaluation"),
]


async def _table_freshness(
    session: AsyncSession,
    model: Any,
    date_col: str,
    lookback_days: int = 3,
) -> dict[str, Any]:
    """Check latest row and recent count for a table."""
    col = getattr(model, date_col)
    col_type = getattr(col, "type", None)
    is_tz_aware = getattr(col_type, "timezone", False) if col_type else False

    now = datetime.now(IST)
    cutoff = now - timedelta(days=lookback_days)
    if not is_tz_aware:
        cutoff = cutoff.replace(tzinfo=None)

    latest_stmt = select(func.max(col))
    count_stmt = select(func.count()).where(col >= cutoff)

    try:
        latest_row = await session.execute(latest_stmt)
        latest = latest_row.scalar()

        count_row = await session.execute(count_stmt)
        count = count_row.scalar() or 0

        latest_str = None
        if latest is not None and isinstance(latest, (datetime, date)):
            latest_str = latest.isoformat()

        return {"latest": latest_str, "recent_count": count}
    except Exception as e:
        return {"latest": None, "recent_count": 0, "error": str(e)}


def _is_stale(latest: str | None, max_age_hours: int = 26) -> bool:
    """Check if latest timestamp is older than max_age_hours."""
    if latest is None:
        return True
    try:
        dt = datetime.fromisoformat(latest)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=IST)
        age = datetime.now(IST) - dt
        return age > timedelta(hours=max_age_hours)
    except (ValueError, TypeError):
        return True


@router.get("/health")
async def ops_health() -> dict[str, Any]:
    """Comprehensive health check for Claude Routines.

    Returns freshness of every critical table, DB/Redis status,
    disk usage, and a list of problems detected.
    """
    from alphavedha.api.deps import get_service
    from alphavedha.intel.quality import check_disk_usage

    problems: list[dict[str, str]] = []
    tables: dict[str, Any] = {}

    service = get_service()
    cache_ok = await service._cache.health_check()

    db_ok = False
    try:
        from alphavedha.data.database import check_health

        db_ok = await check_health()
    except Exception:
        problems.append({"severity": "critical", "detail": "Database unreachable"})

    if not cache_ok:
        problems.append({"severity": "warning", "detail": "Redis unavailable"})

    models_loaded = service._registry.models_available()
    if not models_loaded:
        problems.append({"severity": "critical", "detail": "Models not loaded"})

    today = date.today()
    is_weekend = today.weekday() >= 5

    if db_ok:
        factory = get_session_factory()
        async with factory() as session:
            for table_name, model, date_col, job_desc in _CRITICAL_TABLES:
                freshness = await _table_freshness(session, model, date_col)
                stale = False if is_weekend else _is_stale(freshness["latest"])
                tables[table_name] = {
                    **freshness,
                    "stale": stale,
                    "job": job_desc,
                }
                if stale and not is_weekend:
                    problems.append(
                        {
                            "severity": "warning",
                            "detail": f"{table_name} stale — last: {freshness['latest']}, job: {job_desc}",
                        }
                    )

            for table_name, model, date_col, job_desc in _OPTIONAL_TABLES:
                freshness = await _table_freshness(session, model, date_col)
                tables[table_name] = {
                    **freshness,
                    "stale": False,
                    "job": job_desc,
                    "optional": True,
                }

    disk = check_disk_usage()

    status = "healthy"
    if any(p["severity"] == "critical" for p in problems):
        status = "critical"
    elif problems:
        status = "degraded"

    return {
        "status": status,
        "checked_at": datetime.now(IST).isoformat(),
        "is_weekend": is_weekend,
        "infrastructure": {
            "database": db_ok,
            "redis": cache_ok,
            "models_loaded": models_loaded,
            "model_version": service._registry.model_version,
            "disk": {
                "used_pct": round(disk.used_pct * 100, 1),
                "severity": disk.severity,
                "detail": disk.detail,
            },
        },
        "tables": tables,
        "problems": problems,
        "problem_count": len(problems),
    }


@router.post("/trigger/{job_name}")
async def trigger_job(job_name: str) -> dict[str, Any]:
    """Trigger a scheduler job by name in-process.

    Both the API and scheduler containers share the same codebase and DB,
    so jobs can run directly without Docker exec.
    """
    import asyncio

    allowed_jobs: dict[str, str] = {
        "predictions": "run_daily_predictions",
        "signal_strategies": "run_signal_strategies",
        "prediction_hash": "run_prediction_hash",
        "evaluation": "run_daily_evaluation",
        "data_refresh": "run_data_refresh",
        "fii_dii": "run_fii_dii_ingestion",
        "bhavcopy": "run_bhavcopy_ingestion",
        "bse_announcements": "run_bse_ann_ingestion",
        "nse_announcements": "run_nse_ann_ingestion",
        "insider_trades": "run_insider_trades_ingestion",
        "surveillance": "run_surveillance_ingestion",
        "deals": "run_deals_ingestion",
        "credit_ratings": "run_credit_rating_ingestion",
        "transcripts": "run_transcript_ingestion",
        "intel_extraction": "run_intel_extraction",
        "quality_check": "run_quality_check",
    }

    if job_name not in allowed_jobs:
        return {
            "success": False,
            "error": f"Unknown job: {job_name}",
            "allowed_jobs": sorted(allowed_jobs.keys()),
        }

    method_name = allowed_jobs[job_name]
    logger.info("ops_trigger_job", job=job_name, method=method_name, triggered_by="api")

    try:
        from alphavedha.scheduler import AlphaVedhaScheduler

        def _run_job() -> dict[str, Any]:
            sched = AlphaVedhaScheduler()
            result = getattr(sched, method_name)()
            return {
                "success": result.success,
                "symbols_processed": result.symbols_processed,
                "error": result.error,
            }

        job_result = await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, _run_job),
            timeout=300,
        )

        logger.info("ops_trigger_complete", job=job_name, **job_result)
        return {"success": job_result["success"], "job": job_name, **job_result}
    except TimeoutError:
        logger.error("ops_trigger_timeout", job=job_name)
        return {"success": False, "job": job_name, "error": "Job timed out (300s)"}
    except Exception as e:
        logger.error("ops_trigger_failed", job=job_name, error=str(e))
        return {"success": False, "job": job_name, "error": str(e)}


@router.post("/intel/push")
async def push_intel(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept processed intel rows from Claude Routines.

    Payload format:
    {
        "table": "disclosures" | "transcripts" | ...,
        "rows": [{"symbol": ..., ...}, ...]
    }
    """
    table = payload.get("table", "")
    rows = payload.get("rows", [])

    if not table or not rows:
        return {"success": False, "error": "Missing 'table' or 'rows'"}

    store_funcs: dict[str, str] = {
        "disclosures": "alphavedha.intel.store:store_disclosures",
        "transcripts": "alphavedha.intel.store:store_transcripts",
        "insider_trades": "alphavedha.data.store:store_insider_trades",
        "rating_events": "alphavedha.intel.store:store_rating_events",
        "surveillance_flags": "alphavedha.intel.store:store_surveillance_flags",
    }

    if table not in store_funcs:
        return {
            "success": False,
            "error": f"Unknown table: {table}",
            "allowed_tables": sorted(store_funcs.keys()),
        }

    logger.info("ops_intel_push", table=table, row_count=len(rows))

    try:
        module_path, func_name = store_funcs[table].split(":")
        import importlib

        mod = importlib.import_module(module_path)
        store_fn = getattr(mod, func_name)
        stored = await store_fn(rows)

        logger.info("ops_intel_push_complete", table=table, stored=stored)
        return {"success": True, "table": table, "rows_received": len(rows), "rows_stored": stored}
    except Exception as e:
        logger.error("ops_intel_push_failed", table=table, error=str(e))
        return {"success": False, "table": table, "error": str(e)}


@router.get("/tables/counts")
async def table_counts() -> dict[str, Any]:
    """Return row counts for all key tables — quick overview."""
    factory = get_session_factory()
    counts: dict[str, int] = {}

    async with factory() as session:
        all_tables = _CRITICAL_TABLES + _OPTIONAL_TABLES
        for table_name, model, _date_col, _desc in all_tables:
            try:
                row = await session.execute(select(func.count()).select_from(model))
                counts[table_name] = row.scalar() or 0
            except Exception:
                counts[table_name] = -1

    return {"counts": counts, "checked_at": datetime.now(IST).isoformat()}


@router.get("/intel/pending")
async def intel_pending(limit: int = 100) -> dict[str, Any]:
    """Return unprocessed disclosures for Claude Routine extraction.

    The Routine calls this, extracts events with Claude, and pushes
    results back via POST /api/ops/intel/events. The VPS scheduler's
    Gemini/Groq extraction acts as fallback for anything left over.
    """
    limit = min(limit, 500)

    disclosures = await get_unprocessed_disclosures(limit=limit)

    pending: list[dict[str, Any]] = []
    boilerplate_ids: list[int] = []

    for disc in disclosures:
        if is_boilerplate(disc.get("category", ""), disc.get("headline", "")):
            boilerplate_ids.append(disc["id"])
            continue
        pending.append(
            {
                "id": disc["id"],
                "symbol": disc["symbol"],
                "category": disc.get("category", ""),
                "headline": disc.get("headline", ""),
                "text": disc.get("text"),
                "filed_at": filed.isoformat() if (filed := disc.get("filed_at")) else None,
            }
        )

    if boilerplate_ids:
        await mark_disclosures_processed(boilerplate_ids, datetime.now(IST))

    logger.info(
        "ops_intel_pending",
        total=len(disclosures),
        pending=len(pending),
        boilerplate_skipped=len(boilerplate_ids),
    )

    return {
        "pending": pending,
        "count": len(pending),
        "boilerplate_skipped": len(boilerplate_ids),
    }


@router.post("/intel/events")
async def push_intel_events(payload: dict[str, Any]) -> dict[str, Any]:
    """Accept extracted disclosure events from Claude Routine.

    Payload format:
    {
        "events": [
            {
                "disclosure_id": 123,
                "symbol": "TCS",
                "event_type": "order_win",
                "direction": 1,
                "materiality": 7,
                "confidence": 0.95,
                "summary": "Won Rs 100 Cr order from Indian Railways",
                "red_flags": []
            },
            ...
        ],
        "processed_ids": [123, 124, 125]
    }
    """

    events = payload.get("events", [])
    processed_ids = payload.get("processed_ids", [])

    if not events and not processed_ids:
        return {"success": False, "error": "Missing 'events' and 'processed_ids'"}

    now = datetime.now(IST)

    event_rows: list[dict[str, Any]] = []
    for evt in events:
        event_rows.append(
            {
                "disclosure_id": evt["disclosure_id"],
                "symbol": evt["symbol"],
                "event_type": evt["event_type"],
                "direction": evt.get("direction", 0),
                "materiality": evt.get("materiality", 0),
                "confidence": evt.get("confidence", 0.5),
                "summary": evt.get("summary", ""),
                "red_flags": evt.get("red_flags"),
                "llm_model": "anthropic/claude-routine",
                "prompt_version": "v1",
                "extracted_at": now,
            }
        )

    stored = 0
    if event_rows:
        stored = await store_disclosure_events(event_rows)

    marked = 0
    if processed_ids:
        marked = await mark_disclosures_processed(processed_ids, now)

    logger.info(
        "ops_intel_events_pushed",
        events_stored=stored,
        ids_marked=marked,
    )

    return {
        "success": True,
        "events_stored": stored,
        "ids_marked_processed": marked,
    }


@router.get("/scheduler/status")
async def scheduler_status() -> dict[str, Any]:
    """Check scheduler liveness via shared-volume heartbeat file.

    The scheduler writes a JSON heartbeat to the shared logs volume every
    ~60 seconds. If the heartbeat is missing or older than 5 minutes, the
    scheduler is considered down.
    """
    import json
    from pathlib import Path

    heartbeat_path = (
        Path(os.environ.get("ALPHAVEDHA_LOG_DIR", "/app/logs")) / "scheduler_heartbeat.json"
    )
    try:
        if not heartbeat_path.exists():
            return {"scheduler_running": False, "error": "No heartbeat file found"}

        data = json.loads(heartbeat_path.read_text())
        last_beat = datetime.fromisoformat(data["last_beat"])
        age_seconds = (datetime.now(IST) - last_beat).total_seconds()
        is_alive = age_seconds < 300

        return {
            "scheduler_running": is_alive,
            "last_heartbeat": data["last_beat"],
            "heartbeat_age_seconds": int(age_seconds),
            "pid": data.get("pid"),
            "tier": data.get("tier"),
            "demo": data.get("demo"),
        }
    except Exception as e:
        return {"scheduler_running": False, "error": str(e)}


@router.get("/predictions/summary")
async def predictions_summary() -> dict[str, Any]:
    """Today's prediction pipeline summary — count, coverage, hash status."""
    from alphavedha.data.store import load_daily_pnl, load_paper_trades

    today = date.today()
    trades_df = await load_paper_trades(start=today, end=today)

    n_predictions = len(trades_df)
    symbols: list[str] = []
    n_tradeable = 0
    directions: dict[str, int] = {"bullish": 0, "bearish": 0, "neutral": 0}

    if not trades_df.empty:
        symbols = sorted(trades_df["symbol"].unique().tolist())
        n_tradeable = int(trades_df["is_tradeable"].sum()) if "is_tradeable" in trades_df else 0
        for _, row in trades_df.iterrows():
            d = row.get("predicted_direction", 0)
            if d == 1:
                directions["bullish"] += 1
            elif d == -1:
                directions["bearish"] += 1
            else:
                directions["neutral"] += 1

    hash_status: dict[str, Any] = {"published": False}
    try:
        factory = get_session_factory()
        async with factory() as session:
            proof_stmt = select(PredictionProof).where(PredictionProof.proof_date == today)
            proof_row = await session.execute(proof_stmt)
            proof = proof_row.scalar()
            if proof:
                # "published" means the hash reached the proofs git repo.
                # A DB row alone proves nothing (rows can be rewritten) —
                # reporting it as published hid a two-week publish outage.
                hash_status = {
                    "published": proof.git_commit is not None,
                    "hash_in_db": True,
                    "committed_to_git": proof.git_commit is not None,
                    "ots_stamped": proof.ots_path is not None,
                    "sha256": proof.sha256[:16] + "...",
                    "n_predictions": proof.n_predictions,
                }
    except Exception:
        hash_status["error"] = "db_unavailable"

    yesterday_pnl: dict[str, Any] = {}
    pnl_df = await load_daily_pnl(
        start=today - timedelta(days=7),
        end=today,
    )
    if not pnl_df.empty:
        latest = pnl_df.iloc[-1]
        yesterday_pnl = {
            "date": str(latest["date"]),
            "n_correct": int(latest["n_correct"]),
            "n_positions": int(latest["n_positions"]),
            "accuracy": round(latest["n_correct"] / latest["n_positions"] * 100, 1)
            if latest["n_positions"] > 0
            else 0,
            "cumulative_return": round(float(latest["cumulative_return"]) * 100, 2),
        }

    confidence_stats: dict[str, Any] = {}
    if not trades_df.empty and "confidence" in trades_df:
        confs = trades_df["confidence"].dropna()
        if len(confs) > 0:
            confidence_stats = {
                "min": round(float(confs.min()), 4),
                "max": round(float(confs.max()), 4),
                "mean": round(float(confs.mean()), 4),
                "median": round(float(confs.median()), 4),
                "above_threshold": int((confs > 0.55).sum()),
                "threshold": 0.55,
            }

    return {
        "date": today.isoformat(),
        "predictions": n_predictions,
        "symbols": symbols,
        "n_tradeable": n_tradeable,
        "directions": directions,
        "confidence": confidence_stats,
        "hash": hash_status,
        "latest_evaluation": yesterday_pnl,
    }


@router.get("/models/status")
async def models_status() -> dict[str, Any]:
    """Model artifact ages, versions, and staleness."""
    import json as json_mod
    from pathlib import Path

    artifact_dir = Path(os.environ.get("ALPHAVEDHA_MODEL_DIR", "models/artifacts"))
    required_models = [
        "xgboost",
        "lstm",
        "tft",
        "regime",
        "ensemble",
        "meta_labeling",
        "conformal",
    ]

    now = datetime.now(IST)
    models: dict[str, Any] = {}

    for name in required_models:
        meta_path = artifact_dir / name / "latest" / "metadata.json"
        if not meta_path.exists():
            models[name] = {"status": "missing", "age_days": None}
            continue

        try:
            metadata = json_mod.loads(meta_path.read_text())
            created_str = metadata.get("created_at", "")
            created = datetime.fromisoformat(created_str)
            if created.tzinfo is None:
                created = created.replace(tzinfo=IST)
            age_days = (now - created).days

            models[name] = {
                "status": "stale" if age_days > 30 else "ok",
                "age_days": age_days,
                "created_at": created_str,
                "metrics": {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in metadata.get("metrics", {}).items()
                },
            }
        except Exception as e:
            models[name] = {"status": "error", "error": str(e)}

    n_ok = sum(1 for m in models.values() if m.get("status") == "ok")
    n_stale = sum(1 for m in models.values() if m.get("status") == "stale")
    n_missing = sum(1 for m in models.values() if m.get("status") == "missing")

    return {
        "summary": f"{n_ok} ok, {n_stale} stale, {n_missing} missing",
        "models": models,
        "checked_at": now.isoformat(),
    }


@router.get("/tables/deltas")
async def table_deltas() -> dict[str, Any]:
    """Row count deltas: today vs yesterday for each critical table."""
    today = date.today()
    yesterday = today - timedelta(days=1)

    deltas: dict[str, Any] = {}

    try:
        factory = get_session_factory()
        async with factory() as session:
            for table_name, model, date_col, _desc in _CRITICAL_TABLES:
                col = getattr(model, date_col)
                col_type = getattr(col, "type", None)
                is_tz_aware = getattr(col_type, "timezone", False) if col_type else False

                try:
                    today_start: Any = datetime(today.year, today.month, today.day, tzinfo=IST)
                    yest_start: Any = datetime(
                        yesterday.year, yesterday.month, yesterday.day, tzinfo=IST
                    )
                    if not is_tz_aware:
                        today_start = today
                        yest_start = yesterday

                    today_stmt = select(func.count()).where(col >= today_start)
                    yest_stmt = select(func.count()).where(col >= yest_start, col < today_start)

                    today_row = await session.execute(today_stmt)
                    today_count = today_row.scalar() or 0
                    yest_row = await session.execute(yest_stmt)
                    yest_count = yest_row.scalar() or 0

                    deltas[table_name] = {
                        "today": today_count,
                        "yesterday": yest_count,
                        "delta": today_count - yest_count,
                    }
                except Exception as e:
                    deltas[table_name] = {"error": str(e)}
    except Exception:
        deltas["error"] = "db_unavailable"

    return {"date": today.isoformat(), "deltas": deltas}


@router.get("/weekly/report")
async def weekly_report() -> dict[str, Any]:
    """Weekly summary: accuracy, P&L, ingestion health, model status."""
    from alphavedha.data.store import load_daily_pnl, load_paper_trades

    today = date.today()
    week_start = today - timedelta(days=7)

    pnl_df = await load_daily_pnl(start=week_start, end=today)

    weekly_pnl: dict[str, Any] = {"trading_days": 0}
    if not pnl_df.empty:
        total_correct = int(pnl_df["n_correct"].sum())
        total_positions = int(pnl_df["n_positions"].sum())
        weekly_pnl = {
            "trading_days": len(pnl_df),
            "total_predictions_evaluated": total_positions,
            "total_correct": total_correct,
            "accuracy_pct": round(total_correct / total_positions * 100, 1)
            if total_positions > 0
            else 0,
            "cumulative_return_pct": round(float(pnl_df.iloc[-1]["cumulative_return"]) * 100, 2),
            "best_day_return": round(float(pnl_df["daily_return"].max()) * 100, 2),
            "worst_day_return": round(float(pnl_df["daily_return"].min()) * 100, 2),
        }

    trades_df = await load_paper_trades(start=week_start, end=today)
    n_predictions = len(trades_df)
    unique_symbols = len(trades_df["symbol"].unique()) if not trades_df.empty else 0

    ingestion: dict[str, int] = {}
    try:
        factory = get_session_factory()
        async with factory() as session:
            for table_name, model, date_col, _desc in _CRITICAL_TABLES:
                col = getattr(model, date_col)
                col_type = getattr(col, "type", None)
                is_tz_aware = getattr(col_type, "timezone", False) if col_type else False

                try:
                    cutoff: Any = datetime(
                        week_start.year, week_start.month, week_start.day, tzinfo=IST
                    )
                    if not is_tz_aware:
                        cutoff = week_start
                    stmt = select(func.count()).where(col >= cutoff)
                    row = await session.execute(stmt)
                    ingestion[table_name] = row.scalar() or 0
                except Exception:
                    ingestion[table_name] = -1
    except Exception:
        pass

    return {
        "period": f"{week_start.isoformat()} to {today.isoformat()}",
        "predictions": {
            "total": n_predictions,
            "unique_symbols": unique_symbols,
        },
        "performance": weekly_pnl,
        "ingestion_rows": ingestion,
    }
