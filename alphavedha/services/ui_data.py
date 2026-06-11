"""Real-data helpers for the UI support endpoints (non-demo mode).

Pure data access: PostgreSQL aggregates, model artifact metadata, yfinance
intraday/corporate-event fetches with in-process TTL caching, and system
resource readings. Every helper degrades to an honest empty value on
failure — callers never receive fabricated numbers.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import structlog
import yaml

logger = structlog.get_logger(__name__)

_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"

MODEL_ARTIFACT_NAMES: tuple[str, ...] = (
    "xgboost",
    "lstm",
    "tft",
    "regime",
    "ensemble",
    "meta_labeling",
    "conformal",
)

# ── in-process TTL cache ──────────────────────────────────────────────────────

_ttl_cache: dict[str, tuple[float, Any]] = {}


def cache_get(key: str, ttl_seconds: float) -> Any | None:
    """Return a cached value if present and younger than ttl_seconds, else None."""
    entry = _ttl_cache.get(key)
    if entry is None:
        return None
    stored_at, value = entry
    if time.monotonic() - stored_at > ttl_seconds:
        return None
    return value


def cache_set(key: str, value: Any) -> None:
    """Store a value in the in-process TTL cache."""
    _ttl_cache[key] = (time.monotonic(), value)


# ── model artifacts ───────────────────────────────────────────────────────────


def get_artifact_dir() -> Path:
    """Return the configured model artifact directory."""
    from alphavedha.config import get_config

    return Path(get_config().models.artifact_dir)


def resolve_artifact_dir(name: str) -> Path:
    """Resolve {artifact_dir}/{name}/latest, falling back to {name}/ flat layout."""
    base = get_artifact_dir() / name
    latest = base / "latest"
    return latest if latest.exists() else base


def read_model_metadata(name: str) -> dict[str, Any] | None:
    """Read metadata.json for a trained model artifact, or None if absent."""
    path = resolve_artifact_dir(name) / "metadata.json"
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("model_metadata_read_failed", model=name, error=str(e))
        return None


def read_xgb_feature_importance(limit: int = 10) -> list[tuple[str, float]]:
    """Read the top-N feature importances from the XGBoost artifact, or []."""
    path = resolve_artifact_dir("xgboost") / "feature_importance.csv"
    try:
        if not path.exists():
            return []
        df = pd.read_csv(path, index_col=0)
        if df.empty:
            return []
        series = df.iloc[:, 0].astype(float).sort_values(ascending=False)
        return [(str(name), float(value)) for name, value in series.head(limit).items()]
    except (OSError, ValueError, pd.errors.ParserError) as e:
        logger.warning("feature_importance_read_failed", error=str(e))
        return []


# ── sector mapping (configs/stocks.yaml) ──────────────────────────────────────


def load_sector_groups() -> dict[str, list[str]]:
    """Sector name → constituent symbols from configs/stocks.yaml (cached 1h)."""
    cached = cache_get("sector_groups", ttl_seconds=3600)
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    groups: dict[str, list[str]] = {}
    try:
        raw = yaml.safe_load((_CONFIGS_DIR / "stocks.yaml").read_text())
        for sector, symbols in (raw.get("sectors") or {}).items():
            if sector == "conglomerate":  # duplicates of primary sectors
                continue
            groups[str(sector)] = [str(s) for s in (symbols or [])]
    except (OSError, yaml.YAMLError) as e:
        logger.warning("stocks_yaml_load_failed", error=str(e))
    cache_set("sector_groups", groups)
    return groups


def load_sector_map() -> dict[str, str]:
    """Symbol → sector name from configs/stocks.yaml (first sector wins)."""
    mapping: dict[str, str] = {}
    for sector, symbols in load_sector_groups().items():
        for sym in symbols:
            mapping.setdefault(sym, sector)
    return mapping


_SECTOR_ACRONYMS = {"it", "fmcg"}


def sector_display_name(sector: str) -> str:
    """Human-readable sector label ('it' → 'IT', 'banking' → 'Banking')."""
    return sector.upper() if sector.lower() in _SECTOR_ACRONYMS else sector.capitalize()


# ── system resources ──────────────────────────────────────────────────────────


def system_resources() -> dict[str, float]:
    """Real CPU/memory/disk usage. GPU is always 0.0 (no GPU on this host)."""
    cpu_pct = 0.0
    try:
        n_cpus = os.cpu_count() or 1
        cpu_pct = min(100.0, round(os.getloadavg()[0] / n_cpus * 100.0, 1))
    except OSError:
        pass

    memory_pct = 0.0
    try:
        meminfo: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                meminfo[parts[0].rstrip(":")] = int(parts[1])
        total = meminfo.get("MemTotal", 0)
        available = meminfo.get("MemAvailable", 0)
        if total > 0:
            memory_pct = round((total - available) / total * 100.0, 1)
    except (OSError, ValueError):
        pass

    disk_pct = 0.0
    try:
        usage = shutil.disk_usage("/")
        disk_pct = round(usage.used / usage.total * 100.0, 1)
    except OSError:
        pass

    return {
        "cpu_pct": cpu_pct,
        "memory_pct": memory_pct,
        "gpu_pct": 0.0,
        "disk_pct": disk_pct,
    }


# ── database aggregates ───────────────────────────────────────────────────────


async def ohlcv_store_stats() -> dict[str, Any]:
    """Row count, distinct symbols, latest date, and per-symbol latest date.

    Raises on database errors — callers handle and return honest empties.
    """
    from sqlalchemy import func, select

    from alphavedha.data.database import get_session_factory
    from alphavedha.data.models import DailyOHLCV

    session_factory = get_session_factory()
    async with session_factory() as session:
        row_count = (
            await session.execute(select(func.count()).select_from(DailyOHLCV))
        ).scalar() or 0
        symbol_count = (
            await session.execute(select(func.count(func.distinct(DailyOHLCV.symbol))))
        ).scalar() or 0
        latest_date = (await session.execute(select(func.max(DailyOHLCV.date)))).scalar()
        per_symbol = (
            await session.execute(
                select(DailyOHLCV.symbol, func.max(DailyOHLCV.date))
                .group_by(DailyOHLCV.symbol)
                .order_by(DailyOHLCV.symbol)
                .limit(100)
            )
        ).all()
        rows_on_latest = 0
        if latest_date is not None:
            rows_on_latest = (
                await session.execute(
                    select(func.count())
                    .select_from(DailyOHLCV)
                    .where(DailyOHLCV.date == latest_date)
                )
            ).scalar() or 0

    return {
        "row_count": int(row_count),
        "symbol_count": int(symbol_count),
        "latest_date": latest_date,
        "per_symbol_latest": [(str(sym), d) for sym, d in per_symbol],
        "rows_on_latest_date": int(rows_on_latest),
    }


async def load_recent_closes(symbol: str, lookback_days: int = 60) -> pd.DataFrame:
    """Last N calendar days of stored OHLCV; empty DataFrame on any failure."""
    from alphavedha.data.store import load_ohlcv

    today = date.today()
    try:
        return await load_ohlcv(symbol, today - timedelta(days=lookback_days), today)
    except Exception as e:
        logger.warning("ohlcv_load_failed", symbol=symbol, error=str(e))
        return pd.DataFrame()


# ── yfinance fetchers (sync — call via asyncio.to_thread) ────────────────────

# Indices use Yahoo index tickers, not the .NS equity suffix
_INDEX_TICKERS: dict[str, str] = {
    "NIFTY50": "^NSEI",
    "NIFTY": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
}


def _yf_ticker(symbol: str) -> str:
    return _INDEX_TICKERS.get(symbol.upper(), f"{symbol}.NS")


def fetch_intraday_5m(symbol: str) -> dict[str, Any] | None:
    """Today's 5-minute candles for one NSE symbol via yfinance, or None.

    Returns {"candles": [...], "prev_close": float | None}. Cached 2 minutes.
    """
    cache_key = f"intraday:{symbol}"
    cached = cache_get(cache_key, ttl_seconds=120)
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    try:
        import yfinance as yf

        ticker = yf.Ticker(_yf_ticker(symbol))
        hist = ticker.history(period="1d", interval="5m")
        if hist.empty:
            return None
        prev_close: float | None = None
        try:
            prev_close = float(ticker.fast_info["previousClose"])
        except Exception:
            prev_close = None
        candles: list[dict[str, Any]] = []
        for ts, row in hist.iterrows():
            candles.append(
                {
                    "time": ts.strftime("%H:%M:%S"),
                    "open": round(float(row["Open"]), 2),
                    "high": round(float(row["High"]), 2),
                    "low": round(float(row["Low"]), 2),
                    "close": round(float(row["Close"]), 2),
                    "volume": int(row["Volume"]) if pd.notna(row["Volume"]) else 0,
                }
            )
        result = {"candles": candles, "prev_close": prev_close}
        cache_set(cache_key, result)
        return result
    except Exception as e:
        logger.warning("yfinance_intraday_failed", symbol=symbol, error=str(e))
        return None


def fetch_intraday_bulk(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Today's 5-minute candles for several NSE symbols in one yfinance call.

    Returns {symbol: OHLCV DataFrame}; missing/failed symbols are omitted.
    Cached 2 minutes.
    """
    cache_key = "intraday_bulk:" + ",".join(symbols)
    cached = cache_get(cache_key, ttl_seconds=120)
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    try:
        import yfinance as yf

        tickers = [_yf_ticker(s) for s in symbols]
        data = yf.download(
            tickers=tickers,
            period="1d",
            interval="5m",
            group_by="ticker",
            progress=False,
            threads=True,
        )
        if data is None or data.empty:
            return {}
        out: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                df = data[_yf_ticker(sym)] if isinstance(data.columns, pd.MultiIndex) else data
                df = df.dropna(how="all")
                if not df.empty:
                    out[sym] = df
            except (KeyError, IndexError):
                continue
        cache_set(cache_key, out)
        return out
    except Exception as e:
        logger.warning("yfinance_bulk_intraday_failed", error=str(e))
        return {}


def _event_date_str(value: Any) -> str | None:
    """Normalize a yfinance calendar date value to an ISO date string."""
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "date"):
        try:
            return value.date().isoformat()  # type: ignore[no-any-return]
        except (TypeError, ValueError):
            return None
    return None


def fetch_corporate_events(symbols: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Real corporate events (earnings/dividend dates) via yfinance calendars.

    Args:
        symbols: (symbol, company_name) pairs to query.

    Cached 6 hours to avoid hammering yfinance. Returns [] on total failure.
    """
    cached = cache_get("corporate_events", ttl_seconds=6 * 3600)
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    try:
        import yfinance as yf
    except Exception as e:  # pragma: no cover — yfinance is a hard dependency
        logger.warning("yfinance_import_failed", error=str(e))
        return []

    events: list[dict[str, Any]] = []
    for sym, company in symbols:
        try:
            calendar = yf.Ticker(f"{sym}.NS").calendar
            if not isinstance(calendar, dict):
                continue
            earnings = _event_date_str(calendar.get("Earnings Date"))
            if earnings:
                events.append(
                    {
                        "date": earnings,
                        "symbol": sym,
                        "company": company,
                        "type": "earnings",
                        "description": "Earnings announcement",
                    }
                )
            ex_div = _event_date_str(calendar.get("Ex-Dividend Date"))
            if ex_div:
                events.append(
                    {
                        "date": ex_div,
                        "symbol": sym,
                        "company": company,
                        "type": "dividend",
                        "description": "Ex-dividend date",
                    }
                )
            div_date = _event_date_str(calendar.get("Dividend Date"))
            if div_date and div_date != ex_div:
                events.append(
                    {
                        "date": div_date,
                        "symbol": sym,
                        "company": company,
                        "type": "dividend",
                        "description": "Dividend payment",
                    }
                )
        except Exception as e:
            logger.debug("yfinance_calendar_failed", symbol=sym, error=str(e))
            continue

    events.sort(key=lambda e: str(e["date"]))
    cache_set("corporate_events", events)
    return events


# ── sector trends from stored OHLCV ───────────────────────────────────────────


async def compute_sector_trends(history_points: int = 9) -> list[dict[str, Any]]:
    """Equal-weight sector momentum computed from stored daily closes.

    Returns [] when the database has no usable data. Cached 10 minutes.
    """
    cached = cache_get("sector_trends", ttl_seconds=600)
    if cached is not None:
        return cached  # type: ignore[no-any-return]

    results: list[dict[str, Any]] = []
    for sector, symbols in load_sector_groups().items():
        normalized_series: list[list[float]] = []
        changes: dict[str, float] = {}
        for sym in symbols[:6]:
            df = await load_recent_closes(sym, lookback_days=30)
            if df.empty or len(df) < 2:
                continue
            closes = df["close"].astype(float).tail(history_points)
            if len(closes) < 2 or float(closes.iloc[0]) == 0.0:
                continue
            normalized_series.append([float(v) / float(closes.iloc[0]) for v in closes])
            window = min(8, len(closes))
            base = float(closes.iloc[-window])
            if base > 0:
                changes[sym] = (float(closes.iloc[-1]) / base - 1) * 100
        if not normalized_series or not changes:
            continue
        n_points = min(len(s) for s in normalized_series)
        history = [
            round(
                100.0
                * sum(s[len(s) - n_points + i] for s in normalized_series)
                / len(normalized_series),
                2,
            )
            for i in range(n_points)
        ]
        momentum = round(sum(changes.values()) / len(changes), 2)
        top_gainer = max(changes, key=lambda s: changes[s])
        results.append(
            {
                "name": sector_display_name(sector),
                "momentum_7d": momentum,
                "relative_strength": 0.0,
                "top_gainer": top_gainer,
                "history": [{"y": v} for v in history],
            }
        )

    if results:
        ordered = sorted(results, key=lambda r: float(r["momentum_7d"]))
        for rank, item in enumerate(ordered):
            item["relative_strength"] = round((rank + 1) / len(ordered), 2)

    cache_set("sector_trends", results)
    return results
