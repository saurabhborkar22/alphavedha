"""SQLAlchemy ORM models for AlphaVedha market data tables."""

from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class DailyOHLCV(Base):
    """Daily price and volume data — primary market data table."""

    __tablename__ = "daily_ohlcv"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    adj_close: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=False)
    delivery_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    circuit_hit: Mapped[str | None] = mapped_column(String(10), nullable=True)
    is_adjusted: Mapped[bool] = mapped_column(Boolean, default=False)
    is_filled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default="now()")

    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_daily_ohlcv_symbol_date"),
        Index("ix_daily_ohlcv_symbol_date", "symbol", "date"),
        Index("ix_daily_ohlcv_date", "date"),
    )


class CorporateAction(Base):
    """Corporate actions: splits, bonuses, dividends, rights issues."""

    __tablename__ = "corporate_actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    ex_date: Mapped[date] = mapped_column(Date, nullable=False)
    action_type: Mapped[str] = mapped_column(String(20), nullable=False)
    ratio: Mapped[float] = mapped_column(Float, nullable=False)
    details: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default="now()")

    __table_args__ = (
        UniqueConstraint("symbol", "ex_date", "action_type", name="uq_corp_action"),
        Index("ix_corporate_actions_symbol", "symbol"),
    )


class IndexConstituent(Base):
    """Point-in-time index compositions for survivorship-bias-free analysis."""

    __tablename__ = "index_constituents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    index_name: Mapped[str] = mapped_column(String(50), nullable=False)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    company_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    sector: Mapped[str | None] = mapped_column(String(100), nullable=True)
    effective_from: Mapped[date] = mapped_column(Date, nullable=False)
    effective_to: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default="now()")

    __table_args__ = (
        Index("ix_index_constituents_lookup", "index_name", "effective_from"),
        Index("ix_index_constituents_symbol", "symbol"),
    )


class InstitutionalFlow(Base):
    """FII/DII daily buy/sell data."""

    __tablename__ = "institutional_flows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    category: Mapped[str] = mapped_column(String(10), nullable=False)
    buy_value: Mapped[float] = mapped_column(Float, nullable=False)
    sell_value: Mapped[float] = mapped_column(Float, nullable=False)
    net_value: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default="now()")

    __table_args__ = (
        UniqueConstraint("date", "category", name="uq_institutional_flow"),
        Index("ix_institutional_flows_date", "date"),
    )


class DerivativesData(Base):
    """F&O data: futures OI, options chain snapshots."""

    __tablename__ = "derivatives_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    futures_oi: Mapped[int | None] = mapped_column(Integer, nullable=True)
    futures_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    options_data_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default="now()")

    __table_args__ = (
        UniqueConstraint("symbol", "date", name="uq_derivatives_data"),
        Index("ix_derivatives_data_symbol_date", "symbol", "date"),
    )


class Feature(Base):
    """Computed features stored for training-serving consistency."""

    __tablename__ = "features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    feature_version: Mapped[str] = mapped_column(String(20), nullable=False)
    feature_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default="now()")

    __table_args__ = (
        UniqueConstraint("symbol", "date", "feature_version", name="uq_feature"),
        Index("ix_features_symbol_date", "symbol", "date"),
    )
