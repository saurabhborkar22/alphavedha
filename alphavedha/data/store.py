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
from alphavedha.data.models import (
    AlternativeData,
    DailyOHLCV,
    DailyPnL,
    DerivativesData,
    EarningsResult,
    Feature,
    InsiderTrade,
    InstitutionalFlow,
    NewsArticle,
    PaperTrade,
    PromoterHolding,
)

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
                    index_elements=["symbol", "date", "feature_version"],
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
                    index_elements=["symbol", "date"],
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
                    index_elements=["date", "category"],
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
                    index_elements=["symbol", "date"],
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


async def load_derivatives(symbol: str, start: date, end: date) -> pd.DataFrame:
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


async def store_earnings(rows: list[dict]) -> int:
    """Store quarterly earnings results. Upserts on (symbol, quarter, year)."""
    if not rows:
        return 0

    session_factory = get_session_factory()
    stored = 0

    async with session_factory() as session:
        for row in rows:
            values = {
                "symbol": row["symbol"],
                "quarter": row["quarter"],
                "year": row["year"],
                "revenue_actual": row.get("revenue_actual"),
                "revenue_estimate": row.get("revenue_estimate"),
                "revenue_surprise_pct": row.get("revenue_surprise_pct"),
                "profit_actual": row.get("profit_actual"),
                "profit_estimate": row.get("profit_estimate"),
                "profit_surprise_pct": row.get("profit_surprise_pct"),
                "expenses": row.get("expenses"),
                "announced_date": row.get("announced_date"),
            }
            update_values = {
                k: v for k, v in values.items() if k not in ("symbol", "quarter", "year")
            }

            stmt = (
                pg_insert(EarningsResult)
                .values(**values)
                .on_conflict_do_update(
                    constraint="uq_earnings_result",
                    set_=update_values,
                )
            )
            await session.execute(stmt)
            stored += 1

        await session.commit()

    logger.info("earnings_stored", rows=stored)
    return stored


async def load_earnings(
    symbol: str,
    min_year: int | None = None,
) -> pd.DataFrame:
    """Load earnings results for a symbol, optionally filtered by year."""
    session_factory = get_session_factory()

    async with session_factory() as session:
        stmt = select(EarningsResult).where(EarningsResult.symbol == symbol)
        if min_year is not None:
            stmt = stmt.where(EarningsResult.year >= min_year)
        stmt = stmt.order_by(EarningsResult.year, EarningsResult.quarter)

        result = await session.execute(stmt)
        rows = result.scalars().all()

    if not rows:
        return pd.DataFrame()

    records = [
        {
            "symbol": r.symbol,
            "quarter": r.quarter,
            "year": r.year,
            "revenue_actual": r.revenue_actual,
            "revenue_estimate": r.revenue_estimate,
            "revenue_surprise_pct": r.revenue_surprise_pct,
            "profit_actual": r.profit_actual,
            "profit_estimate": r.profit_estimate,
            "profit_surprise_pct": r.profit_surprise_pct,
            "expenses": r.expenses,
            "announced_date": r.announced_date,
        }
        for r in rows
    ]

    return pd.DataFrame(records)


async def store_promoter_holdings(rows: list[dict]) -> int:
    """Store quarterly promoter holdings. Upserts on (symbol, quarter_end)."""
    if not rows:
        return 0

    session_factory = get_session_factory()
    stored = 0

    async with session_factory() as session:
        for row in rows:
            values = {
                "symbol": row["symbol"],
                "quarter_end": row["quarter_end"],
                "promoter_pct": row["promoter_pct"],
                "pledge_pct": row.get("pledge_pct", 0.0),
                "public_pct": row.get("public_pct", 0.0),
                "fii_pct": row.get("fii_pct", 0.0),
                "dii_pct": row.get("dii_pct", 0.0),
            }
            update_values = {k: v for k, v in values.items() if k not in ("symbol", "quarter_end")}

            stmt = (
                pg_insert(PromoterHolding)
                .values(**values)
                .on_conflict_do_update(
                    constraint="uq_promoter_holding",
                    set_=update_values,
                )
            )
            await session.execute(stmt)
            stored += 1

        await session.commit()

    logger.info("promoter_holdings_stored", rows=stored)
    return stored


async def load_promoter_holdings(symbol: str) -> pd.DataFrame:
    """Load promoter holding history for a symbol."""
    session_factory = get_session_factory()

    async with session_factory() as session:
        stmt = (
            select(PromoterHolding)
            .where(PromoterHolding.symbol == symbol)
            .order_by(PromoterHolding.quarter_end)
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "quarter_end": r.quarter_end,
                "promoter_pct": r.promoter_pct,
                "pledge_pct": r.pledge_pct,
                "public_pct": r.public_pct,
                "fii_pct": r.fii_pct,
                "dii_pct": r.dii_pct,
            }
            for r in rows
        ]
    )


async def store_insider_trades(rows: list[dict]) -> int:
    """Store insider trade records."""
    if not rows:
        return 0

    session_factory = get_session_factory()
    stored = 0

    async with session_factory() as session:
        for row in rows:
            stmt = InsiderTrade(
                symbol=row["symbol"],
                trade_date=row["trade_date"],
                person_name=row.get("person_name", "Unknown"),
                person_category=row.get("person_category", ""),
                trade_type=row["trade_type"],
                shares=row.get("shares", 0),
                value_lakhs=row.get("value_lakhs", 0.0),
            )
            session.add(stmt)
            stored += 1

        await session.commit()

    logger.info("insider_trades_stored", rows=stored)
    return stored


async def load_insider_trades(symbol: str, days_back: int = 365) -> pd.DataFrame:
    """Load insider trades for a symbol within recent period."""
    from datetime import timedelta

    session_factory = get_session_factory()
    cutoff = date.today() - timedelta(days=days_back)

    async with session_factory() as session:
        stmt = (
            select(InsiderTrade)
            .where(
                InsiderTrade.symbol == symbol,
                InsiderTrade.trade_date >= cutoff,
            )
            .order_by(InsiderTrade.trade_date)
        )
        result = await session.execute(stmt)
        rows = result.scalars().all()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "trade_date": r.trade_date,
                "person_name": r.person_name,
                "person_category": r.person_category,
                "trade_type": r.trade_type,
                "shares": r.shares,
                "value_lakhs": r.value_lakhs,
            }
            for r in rows
        ]
    )


async def store_news_articles(rows: list[dict]) -> int:
    """Store news articles. Upserts on content_hash."""
    if not rows:
        return 0

    session_factory = get_session_factory()
    stored = 0

    async with session_factory() as session:
        for row in rows:
            values = {
                "symbol": row.get("symbol"),
                "source": row["source"],
                "title": row["title"],
                "url": row.get("url"),
                "published_date": row["published_date"],
                "sentiment_score": row.get("sentiment_score"),
                "content_hash": row["content_hash"],
            }
            stmt = (
                pg_insert(NewsArticle)
                .values(**values)
                .on_conflict_do_update(
                    index_elements=["content_hash", "published_date"],
                    set_={"sentiment_score": values["sentiment_score"]},
                )
            )
            await session.execute(stmt)
            stored += 1

        await session.commit()

    logger.info("news_articles_stored", rows=stored)
    return stored


async def load_news_articles(
    symbol: str | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Load news articles, optionally filtered by symbol and date range."""
    session_factory = get_session_factory()

    async with session_factory() as session:
        stmt = select(NewsArticle)
        if symbol:
            stmt = stmt.where(NewsArticle.symbol == symbol)
        if start:
            stmt = stmt.where(NewsArticle.published_date >= start)
        if end:
            stmt = stmt.where(NewsArticle.published_date <= end)
        stmt = stmt.order_by(NewsArticle.published_date)

        result = await session.execute(stmt)
        rows = result.scalars().all()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "symbol": r.symbol,
                "source": r.source,
                "title": r.title,
                "url": r.url,
                "published_date": r.published_date,
                "sentiment_score": r.sentiment_score,
            }
            for r in rows
        ]
    )


async def store_alternative_data(rows: list[dict]) -> int:
    """Store alternative data records. Upserts on (data_type, period_date)."""
    if not rows:
        return 0

    session_factory = get_session_factory()
    stored = 0

    async with session_factory() as session:
        for row in rows:
            values = {
                "data_type": row["data_type"],
                "period_date": row["period_date"],
                "value": row["value"],
                "yoy_change": row.get("yoy_change"),
                "sector": row.get("sector"),
                "source": row.get("source"),
            }
            update_values = {
                k: v for k, v in values.items() if k not in ("data_type", "period_date")
            }

            stmt = (
                pg_insert(AlternativeData)
                .values(**values)
                .on_conflict_do_update(
                    constraint="uq_alternative_data",
                    set_=update_values,
                )
            )
            await session.execute(stmt)
            stored += 1

        await session.commit()

    logger.info("alternative_data_stored", rows=stored)
    return stored


async def load_alternative_data(
    data_type: str | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Load alternative data, optionally filtered."""
    session_factory = get_session_factory()

    async with session_factory() as session:
        stmt = select(AlternativeData)
        if data_type:
            stmt = stmt.where(AlternativeData.data_type == data_type)
        if start:
            stmt = stmt.where(AlternativeData.period_date >= start)
        if end:
            stmt = stmt.where(AlternativeData.period_date <= end)
        stmt = stmt.order_by(AlternativeData.period_date)

        result = await session.execute(stmt)
        rows = result.scalars().all()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "data_type": r.data_type,
                "period_date": r.period_date,
                "value": r.value,
                "yoy_change": r.yoy_change,
                "sector": r.sector,
                "source": r.source,
            }
            for r in rows
        ]
    )


async def store_paper_trade(row: dict) -> int:
    """Store a single paper trade prediction."""
    session_factory = get_session_factory()

    async with session_factory() as session:
        values = {
            "symbol": row["symbol"],
            "prediction_date": row["prediction_date"],
            "predicted_direction": row["predicted_direction"],
            "predicted_magnitude": row["predicted_magnitude"],
            "confidence": row["confidence"],
            "model_version": row["model_version"],
            "regime": row.get("regime"),
            "is_tradeable": row.get("is_tradeable"),
            "entry_price": row.get("entry_price"),
            "stop_loss_price": row.get("stop_loss_price"),
            "take_profit_price": row.get("take_profit_price"),
        }
        stmt = (
            pg_insert(PaperTrade)
            .values(**values)
            .on_conflict_do_update(
                index_elements=["symbol", "prediction_date"],
                set_={k: v for k, v in values.items() if k not in ("symbol", "prediction_date")},
            )
        )
        await session.execute(stmt)
        await session.commit()

    return 1


async def update_paper_trade_outcome(
    symbol: str,
    prediction_date: date,
    exit_price: float,
    actual_return: float,
    is_correct: bool,
    exit_reason: str | None = None,
) -> None:
    """Update a paper trade with the actual outcome."""
    from sqlalchemy import update

    session_factory = get_session_factory()

    async with session_factory() as session:
        values: dict[str, float | bool | str | None] = {
            "exit_price": exit_price,
            "actual_return": actual_return,
            "is_correct": is_correct,
        }
        if exit_reason is not None:
            values["exit_reason"] = exit_reason
        stmt = (
            update(PaperTrade)
            .where(
                PaperTrade.symbol == symbol,
                PaperTrade.prediction_date == prediction_date,
            )
            .values(**values)
        )
        await session.execute(stmt)
        await session.commit()


async def load_paper_trades(
    start: date | None = None,
    end: date | None = None,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Load paper trade records."""
    session_factory = get_session_factory()

    async with session_factory() as session:
        stmt = select(PaperTrade)
        if symbol:
            stmt = stmt.where(PaperTrade.symbol == symbol)
        if start:
            stmt = stmt.where(PaperTrade.prediction_date >= start)
        if end:
            stmt = stmt.where(PaperTrade.prediction_date <= end)
        stmt = stmt.order_by(PaperTrade.prediction_date)

        result = await session.execute(stmt)
        rows = result.scalars().all()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "symbol": r.symbol,
                "prediction_date": r.prediction_date,
                "predicted_direction": r.predicted_direction,
                "predicted_magnitude": r.predicted_magnitude,
                "confidence": r.confidence,
                "model_version": r.model_version,
                "regime": r.regime,
                "is_tradeable": r.is_tradeable,
                "entry_price": r.entry_price,
                "stop_loss_price": r.stop_loss_price,
                "take_profit_price": r.take_profit_price,
                "exit_price": r.exit_price,
                "exit_reason": r.exit_reason,
                "actual_return": r.actual_return,
                "is_correct": r.is_correct,
            }
            for r in rows
        ]
    )


async def store_daily_pnl(row: dict) -> int:
    """Store daily P&L summary."""
    session_factory = get_session_factory()

    async with session_factory() as session:
        obj = DailyPnL(
            date=row["date"],
            portfolio_value=row["portfolio_value"],
            daily_return=row["daily_return"],
            cumulative_return=row["cumulative_return"],
            n_positions=row["n_positions"],
            n_correct=row["n_correct"],
            n_total_predictions=row["n_total_predictions"],
            benchmark_return=row.get("benchmark_return", 0.0),
        )
        session.add(obj)
        await session.commit()

    return 1


async def load_daily_pnl(
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Load daily P&L records."""
    session_factory = get_session_factory()

    async with session_factory() as session:
        stmt = select(DailyPnL)
        if start:
            stmt = stmt.where(DailyPnL.date >= start)
        if end:
            stmt = stmt.where(DailyPnL.date <= end)
        stmt = stmt.order_by(DailyPnL.date)

        result = await session.execute(stmt)
        rows = result.scalars().all()

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(
        [
            {
                "date": r.date,
                "portfolio_value": r.portfolio_value,
                "daily_return": r.daily_return,
                "cumulative_return": r.cumulative_return,
                "n_positions": r.n_positions,
                "n_correct": r.n_correct,
                "n_total_predictions": r.n_total_predictions,
                "benchmark_return": r.benchmark_return,
            }
            for r in rows
        ]
    )
