"""Feature store — PostgreSQL-backed storage for computed features.

Ensures training-serving consistency: exact same features used in training
are retrieved at prediction time. Key: (symbol, date, feature_version).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import structlog
from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from alphavedha.data.database import get_session_factory
from alphavedha.data.models import DailyOHLCV, DerivativesData, Feature, InstitutionalFlow

logger = structlog.get_logger(__name__)


async def store_features(
    symbol: str,
    features_df: pd.DataFrame,
    feature_version: str = "v1",
) -> int:
    """Store computed features for a symbol. Upserts on (symbol, date, version)."""
    if features_df.empty:
        return 0

    session_factory = get_session_factory()
    stored = 0

    async with session_factory() as session:
        for idx, row in features_df.iterrows():
            row_date = idx.date() if hasattr(idx, "date") else idx
            feature_dict = row.to_dict()

            stmt = (
                pg_insert(Feature)
                .values(
                    symbol=symbol,
                    date=row_date,
                    feature_version=feature_version,
                    feature_json=feature_dict,
                )
                .on_conflict_do_update(
                    constraint="uq_feature",
                    set_={"feature_json": feature_dict},
                )
            )
            await session.execute(stmt)
            stored += 1

        await session.commit()

    logger.info("features_stored", symbol=symbol, rows=stored, version=feature_version)
    return stored


async def load_features(
    symbol: str,
    start: date,
    end: date,
    feature_version: str = "v1",
) -> pd.DataFrame:
    """Load stored features for a symbol and date range."""
    session_factory = get_session_factory()

    async with session_factory() as session:
        stmt = (
            select(Feature)
            .where(
                Feature.symbol == symbol,
                Feature.date >= start,
                Feature.date <= end,
                Feature.feature_version == feature_version,
            )
            .order_by(Feature.date)
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()

    if not rows:
        return pd.DataFrame()

    records = []
    for r in rows:
        record: dict[str, Any] = {"date": r.date, **r.feature_json}
        records.append(record)

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    logger.info("features_loaded", symbol=symbol, rows=len(df), version=feature_version)
    return df


async def store_ohlcv(
    symbol: str,
    df: pd.DataFrame,
) -> int:
    """Store OHLCV data for a symbol. Upserts on (symbol, date)."""
    if df.empty:
        return 0

    session_factory = get_session_factory()
    stored = 0

    async with session_factory() as session:
        for idx, row in df.iterrows():
            row_date = idx.date() if hasattr(idx, "date") else idx

            values: dict[str, Any] = {
                "symbol": symbol,
                "date": row_date,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "adj_close": float(row.get("adj_close", row["close"])),
                "volume": int(row["volume"]),
            }

            if "delivery_pct" in row and pd.notna(row["delivery_pct"]):
                values["delivery_pct"] = float(row["delivery_pct"])
            if "circuit_hit" in row and pd.notna(row["circuit_hit"]):
                values["circuit_hit"] = str(row["circuit_hit"])
            if "is_adjusted" in row:
                values["is_adjusted"] = bool(row["is_adjusted"])
            if "is_filled" in row:
                values["is_filled"] = bool(row["is_filled"])

            update_values = {k: v for k, v in values.items() if k not in ("symbol", "date")}

            stmt = (
                pg_insert(DailyOHLCV)
                .values(**values)
                .on_conflict_do_update(
                    constraint="uq_daily_ohlcv_symbol_date",
                    set_=update_values,
                )
            )
            await session.execute(stmt)
            stored += 1

        await session.commit()

    logger.info("ohlcv_stored", symbol=symbol, rows=stored)
    return stored


async def load_ohlcv(
    symbol: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Load OHLCV data for a symbol and date range from the database."""
    session_factory = get_session_factory()

    async with session_factory() as session:
        stmt = (
            select(DailyOHLCV)
            .where(
                DailyOHLCV.symbol == symbol,
                DailyOHLCV.date >= start,
                DailyOHLCV.date <= end,
            )
            .order_by(DailyOHLCV.date)
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()

    if not rows:
        return pd.DataFrame()

    records = [
        {
            "date": r.date,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "adj_close": r.adj_close,
            "volume": r.volume,
            "delivery_pct": r.delivery_pct,
            "circuit_hit": r.circuit_hit,
            "is_adjusted": r.is_adjusted,
            "is_filled": r.is_filled,
        }
        for r in rows
    ]

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    return df


async def delete_ohlcv(symbol: str) -> int:
    """Delete all OHLCV data for a symbol."""
    session_factory = get_session_factory()

    async with session_factory() as session:
        stmt = delete(DailyOHLCV).where(DailyOHLCV.symbol == symbol)
        result = await session.execute(stmt)
        await session.commit()

    deleted = result.rowcount  # type: ignore[union-attr]
    logger.info("ohlcv_deleted", symbol=symbol, rows=deleted)
    return deleted


async def store_fii_dii(rows: list[dict]) -> int:
    """Store FII/DII flow data. Upserts on (date, category)."""
    if not rows:
        return 0

    session_factory = get_session_factory()
    stored = 0

    async with session_factory() as session:
        for row in rows:
            stmt = (
                pg_insert(InstitutionalFlow)
                .values(
                    date=row["date"],
                    category=row["category"],
                    buy_value=row["buy_value"],
                    sell_value=row["sell_value"],
                    net_value=row["net_value"],
                )
                .on_conflict_do_update(
                    constraint="uq_institutional_flow",
                    set_={
                        "buy_value": row["buy_value"],
                        "sell_value": row["sell_value"],
                        "net_value": row["net_value"],
                    },
                )
            )
            await session.execute(stmt)
            stored += 1

        await session.commit()

    logger.info("fii_dii_stored", rows=stored)
    return stored


async def load_fii_dii(start: date, end: date) -> pd.DataFrame:
    """Load FII/DII flow data for a date range."""
    session_factory = get_session_factory()

    async with session_factory() as session:
        stmt = (
            select(InstitutionalFlow)
            .where(
                InstitutionalFlow.date >= start,
                InstitutionalFlow.date <= end,
            )
            .order_by(InstitutionalFlow.date)
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()

    if not rows:
        return pd.DataFrame()

    records = [
        {
            "date": r.date,
            "category": r.category,
            "buy_value": r.buy_value,
            "sell_value": r.sell_value,
            "net_value": r.net_value,
        }
        for r in rows
    ]

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df


async def store_derivatives(rows: list[dict]) -> int:
    """Store derivatives (F&O) data. Upserts on (symbol, date)."""
    if not rows:
        return 0

    session_factory = get_session_factory()
    stored = 0

    async with session_factory() as session:
        for row in rows:
            stmt = (
                pg_insert(DerivativesData)
                .values(
                    symbol=row["symbol"],
                    date=row["date"],
                    futures_oi=row.get("futures_oi"),
                    futures_price=row.get("futures_price"),
                    options_data_json=row.get("options_data_json"),
                )
                .on_conflict_do_update(
                    constraint="uq_derivatives_data",
                    set_={
                        "futures_oi": row.get("futures_oi"),
                        "futures_price": row.get("futures_price"),
                        "options_data_json": row.get("options_data_json"),
                    },
                )
            )
            await session.execute(stmt)
            stored += 1

        await session.commit()

    logger.info("derivatives_stored", rows=stored)
    return stored


async def load_derivatives(
    symbol: str, start: date, end: date
) -> pd.DataFrame:
    """Load derivatives data for a symbol and date range."""
    session_factory = get_session_factory()

    async with session_factory() as session:
        stmt = (
            select(DerivativesData)
            .where(
                DerivativesData.symbol == symbol,
                DerivativesData.date >= start,
                DerivativesData.date <= end,
            )
            .order_by(DerivativesData.date)
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()

    if not rows:
        return pd.DataFrame()

    records = [
        {
            "date": r.date,
            "symbol": r.symbol,
            "futures_oi": r.futures_oi,
            "futures_price": r.futures_price,
            "options_data_json": r.options_data_json,
        }
        for r in rows
    ]

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df
