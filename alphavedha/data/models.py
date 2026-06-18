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
    """Daily price and volume data — primary market data table (TimescaleDB hypertable)."""

    __tablename__ = "daily_ohlcv"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
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

    __table_args__ = (Index("ix_daily_ohlcv_date", "date"),)


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
    """FII/DII daily buy/sell data (TimescaleDB hypertable)."""

    __tablename__ = "institutional_flows"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    category: Mapped[str] = mapped_column(String(10), primary_key=True)
    buy_value: Mapped[float] = mapped_column(Float, nullable=False)
    sell_value: Mapped[float] = mapped_column(Float, nullable=False)
    net_value: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default="now()")

    __table_args__ = (Index("ix_institutional_flows_date", "date"),)


class DerivativesData(Base):
    """F&O data: futures OI, options chain snapshots (TimescaleDB hypertable)."""

    __tablename__ = "derivatives_data"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    futures_oi: Mapped[int | None] = mapped_column(Integer, nullable=True)
    futures_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    options_data_json: Mapped[dict[str, object] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default="now()")

    __table_args__: tuple[()] = ()


class EarningsResult(Base):
    """Quarterly earnings results for PEAD analysis."""

    __tablename__ = "earnings_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    quarter: Mapped[int] = mapped_column(Integer, nullable=False)
    year: Mapped[int] = mapped_column(Integer, nullable=False)
    revenue_actual: Mapped[float | None] = mapped_column(Float, nullable=True)
    revenue_estimate: Mapped[float | None] = mapped_column(Float, nullable=True)
    revenue_surprise_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit_actual: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit_estimate: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit_surprise_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    expenses: Mapped[float | None] = mapped_column(Float, nullable=True)
    announced_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default="now()")

    __table_args__ = (
        UniqueConstraint("symbol", "quarter", "year", name="uq_earnings_result"),
        Index("ix_earnings_results_symbol", "symbol"),
        Index("ix_earnings_results_announced", "announced_date"),
    )


class PromoterHolding(Base):
    """Quarterly promoter shareholding pattern (SEBI filing)."""

    __tablename__ = "promoter_holdings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    quarter_end: Mapped[date] = mapped_column(Date, nullable=False)
    promoter_pct: Mapped[float] = mapped_column(Float, nullable=False)
    pledge_pct: Mapped[float] = mapped_column(Float, default=0.0)
    public_pct: Mapped[float] = mapped_column(Float, default=0.0)
    fii_pct: Mapped[float] = mapped_column(Float, default=0.0)
    dii_pct: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default="now()")

    __table_args__ = (
        UniqueConstraint("symbol", "quarter_end", name="uq_promoter_holding"),
        Index("ix_promoter_holdings_symbol", "symbol"),
        Index("ix_promoter_holdings_quarter", "quarter_end"),
    )


class InsiderTrade(Base):
    """Insider (SAST) trading disclosures (TimescaleDB hypertable)."""

    __tablename__ = "insider_trades"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    person_name: Mapped[str] = mapped_column(String(200), primary_key=True)
    person_category: Mapped[str] = mapped_column(String(100), nullable=True)
    trade_type: Mapped[str] = mapped_column(String(10), nullable=False)
    shares: Mapped[int] = mapped_column(Integer, nullable=False)
    value_lakhs: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default="now()")

    __table_args__ = (Index("ix_insider_trades_symbol", "symbol"),)


class NewsArticle(Base):
    """Stored news articles for sentiment analysis (TimescaleDB hypertable)."""

    __tablename__ = "news_articles"

    content_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    published_date: Mapped[date] = mapped_column(Date, primary_key=True)
    symbol: Mapped[str | None] = mapped_column(String(20), nullable=True)
    source: Mapped[str] = mapped_column(String(50), nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    url: Mapped[str] = mapped_column(String(1000), nullable=True)
    sentiment_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default="now()")

    __table_args__ = (
        Index("ix_news_articles_symbol_date", "symbol", "published_date"),
        Index("ix_news_articles_date", "published_date"),
    )


class AlternativeData(Base):
    """Monthly alternative data (auto sales, cement, PMI, credit growth)."""

    __tablename__ = "alternative_data"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    data_type: Mapped[str] = mapped_column(String(50), nullable=False)
    period_date: Mapped[date] = mapped_column(Date, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    yoy_change: Mapped[float | None] = mapped_column(Float, nullable=True)
    sector: Mapped[str | None] = mapped_column(String(50), nullable=True)
    source: Mapped[str | None] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default="now()")

    __table_args__ = (
        UniqueConstraint("data_type", "period_date", name="uq_alternative_data"),
        Index("ix_alternative_data_type_date", "data_type", "period_date"),
    )


class PaperTrade(Base):
    """Paper trading prediction records — timestamped before market open (TimescaleDB hypertable)."""

    __tablename__ = "paper_trades"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    prediction_date: Mapped[date] = mapped_column(Date, primary_key=True)
    predicted_direction: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_magnitude: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    regime: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # Meta-labeling gate decision at prediction time. Null on rows persisted
    # before the column existed; the track record falls back to a confidence
    # threshold for those.
    is_tradeable: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    entry_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    stop_loss_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    take_profit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    exit_reason: Mapped[str | None] = mapped_column(String(20), nullable=True)
    actual_return: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default="now()")

    __table_args__ = (
        Index("ix_paper_trades_date", "prediction_date"),
        Index("ix_paper_trades_symbol", "symbol"),
    )


class DailyPnL(Base):
    """Daily paper portfolio P&L summary (TimescaleDB hypertable)."""

    __tablename__ = "daily_pnl"

    date: Mapped[date] = mapped_column(Date, primary_key=True)
    portfolio_value: Mapped[float] = mapped_column(Float, nullable=False)
    daily_return: Mapped[float] = mapped_column(Float, nullable=False)
    cumulative_return: Mapped[float] = mapped_column(Float, nullable=False)
    n_positions: Mapped[int] = mapped_column(Integer, nullable=False)
    n_correct: Mapped[int] = mapped_column(Integer, nullable=False)
    n_total_predictions: Mapped[int] = mapped_column(Integer, nullable=False)
    benchmark_return: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default="now()")


class Feature(Base):
    """Computed features stored for training-serving consistency (TimescaleDB hypertable)."""

    __tablename__ = "features"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    feature_version: Mapped[str] = mapped_column(String(20), primary_key=True)
    feature_json: Mapped[dict[str, object]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default="now()")


class DataLineage(Base):
    """Tracks data provenance: which provider, when fetched, how many rows."""

    __tablename__ = "data_lineage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str | None] = mapped_column(String(20), nullable=True)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    table_name: Mapped[str] = mapped_column(String(50), nullable=False)
    provider: Mapped[str] = mapped_column(String(50), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default="now()")

    __table_args__ = (Index("ix_data_lineage_symbol_date", "symbol", "date"),)


class DataQualityReport(Base):
    """Results of automated data quality checks (completeness, freshness, consistency)."""

    __tablename__ = "data_quality_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str | None] = mapped_column(String(20), nullable=True)
    report_date: Mapped[date] = mapped_column(Date, nullable=False)
    check_type: Mapped[str] = mapped_column(String(30), nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    severity: Mapped[str] = mapped_column(String(10), nullable=False)
    detail: Mapped[str] = mapped_column(String(1000), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default="now()")

    __table_args__ = (
        Index("ix_dqr_date", "report_date"),
        Index("ix_dqr_symbol", "symbol"),
    )


class CorporateAnnouncement(Base):
    """BSE/NSE corporate announcements: dividends, AGMs, board meetings, results."""

    __tablename__ = "corporate_announcements"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    announced_date: Mapped[date] = mapped_column(Date, nullable=False)
    ex_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    event_type: Mapped[str] = mapped_column(String(20), nullable=False)
    description: Mapped[str] = mapped_column(String(500), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default="now()")

    __table_args__ = (
        UniqueConstraint("symbol", "announced_date", "event_type", name="uq_corp_announcement"),
        Index("ix_corp_ann_symbol", "symbol"),
        Index("ix_corp_ann_date", "announced_date"),
    )


class IntradayOHLCV(Base):
    """Live intraday OHLCV snapshot — one row per symbol per trading day, updated in-place by the polling loop."""

    __tablename__ = "intraday_ohlcv"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    last_price: Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[int] = mapped_column(Integer, nullable=False)
    tick_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_updated: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, server_default="now()")
