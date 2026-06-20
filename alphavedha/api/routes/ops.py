"""Ops endpoints — VPS health monitoring and job management for Claude Routines.

All endpoints require API key auth. Designed to be called by Anthropic
cloud Routines for automated monitoring and auto-healing.
"""

from __future__ import annotations

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
    if not is_tz_aware and not isinstance(cutoff, date):
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
    """Trigger a scheduler job by name via Docker exec.

    The API and scheduler run in separate containers. This endpoint
    runs the job in the scheduler container via subprocess.
    """
    import asyncio

    allowed_jobs: dict[str, str] = {
        "predictions": "alphavedha.scheduler:AlphaVedhaScheduler().run_daily_predictions()",
        "signal_strategies": "alphavedha.scheduler:AlphaVedhaScheduler().run_signal_strategies()",
        "prediction_hash": "alphavedha.scheduler:AlphaVedhaScheduler().run_prediction_hash()",
        "evaluation": "alphavedha.scheduler:AlphaVedhaScheduler().run_daily_evaluation()",
        "data_refresh": "alphavedha.scheduler:AlphaVedhaScheduler().run_data_refresh()",
        "fii_dii": "alphavedha.scheduler:AlphaVedhaScheduler().run_fii_dii_ingestion()",
        "bhavcopy": "alphavedha.scheduler:AlphaVedhaScheduler().run_bhavcopy_ingestion()",
        "bse_announcements": "alphavedha.scheduler:AlphaVedhaScheduler().run_bse_ann_ingestion()",
        "nse_announcements": "alphavedha.scheduler:AlphaVedhaScheduler().run_nse_ann_ingestion()",
        "insider_trades": "alphavedha.scheduler:AlphaVedhaScheduler().run_insider_trades_ingestion()",
        "surveillance": "alphavedha.scheduler:AlphaVedhaScheduler().run_surveillance_ingestion()",
        "deals": "alphavedha.scheduler:AlphaVedhaScheduler().run_deals_ingestion()",
        "credit_ratings": "alphavedha.scheduler:AlphaVedhaScheduler().run_credit_rating_ingestion()",
        "transcripts": "alphavedha.scheduler:AlphaVedhaScheduler().run_transcript_ingestion()",
        "intel_extraction": "alphavedha.scheduler:AlphaVedhaScheduler().run_intel_extraction()",
        "quality_check": "alphavedha.scheduler:AlphaVedhaScheduler().run_quality_check()",
    }

    if job_name not in allowed_jobs:
        return {
            "success": False,
            "error": f"Unknown job: {job_name}",
            "allowed_jobs": sorted(allowed_jobs.keys()),
        }

    logger.info("ops_trigger_job", job=job_name, triggered_by="api")

    snippet = allowed_jobs[job_name]
    module_path, call = snippet.split(":")
    python_code = (
        f"from {module_path} import AlphaVedhaScheduler; "
        f"r = AlphaVedhaScheduler().{call.split('().')[-1]}; "
        f"print(f'success={{r.success}} symbols={{r.symbols_processed}} error={{r.error}}')"
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "alphavedha-scheduler",
            "python",
            "-c",
            python_code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
        output = stdout.decode().strip()
        err_output = stderr.decode().strip()

        success = proc.returncode == 0 and "success=True" in output
        logger.info(
            "ops_trigger_complete",
            job=job_name,
            success=success,
            returncode=proc.returncode,
            output=output[:500],
        )
        return {
            "success": success,
            "job": job_name,
            "output": output[:1000],
            "stderr": err_output[:500] if err_output else None,
            "returncode": proc.returncode,
        }
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


@router.get("/scheduler/status")
async def scheduler_status() -> dict[str, Any]:
    """Check if the scheduler container is running and responsive."""
    import asyncio

    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "inspect",
            "--format",
            "{{.State.Status}}",
            "alphavedha-scheduler",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        container_status = stdout.decode().strip()

        uptime_proc = await asyncio.create_subprocess_exec(
            "docker",
            "inspect",
            "--format",
            "{{.State.StartedAt}}",
            "alphavedha-scheduler",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        up_stdout, _ = await asyncio.wait_for(uptime_proc.communicate(), timeout=10)
        started_at = up_stdout.decode().strip()

        return {
            "scheduler_running": container_status == "running",
            "container_status": container_status,
            "started_at": started_at,
        }
    except Exception as e:
        return {
            "scheduler_running": False,
            "error": str(e),
        }
