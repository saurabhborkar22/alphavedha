# D7: Data Pipeline Enhancements — Design Spec

**Date:** 2026-05-30  
**Status:** Approved  

---

## Goal

Extend the AlphaVedha data pipeline with three independent enhancements: automated data quality monitoring (D7.2), two new data sources (D7.3), and polling-based live OHLCV during market hours (D7.1). Implementation order: D7.2 → D7.3 → D7.1.

---

## Architecture Overview

All three sub-projects follow existing project patterns:
- New providers implement `DataProvider` protocol (or a looser interface for non-OHLCV sources)
- New DB tables use SQLAlchemy ORM models in `data/models.py` + Alembic migration
- New features integrated into `features/pipeline.py`
- New CLI commands added to `alphavedha/cli/main.py` under the `data` subcommand
- Scheduler jobs added to `alphavedha/scheduler.py`
- Alerts via existing `EmailAlerter` in `monitoring/alerts.py`

---

## D7.2: Data Quality

### New Files
- `alphavedha/data/quality.py` — `QualityChecker` class

### Modified Files
- `alphavedha/data/models.py` — two new ORM models
- `alphavedha/data/ingestion.py` — lineage write after every `ingest_symbol`; quality check after `refresh_latest`
- `alphavedha/cli/main.py` — `data quality-check` command
- `alphavedha/scheduler.py` — quality check job after daily evaluation
- `alphavedha/monitoring/alerts.py` — `data_quality_failed` alert method

### Data Models

```python
class DataLineage(Base):
    """Records provenance of every ingested row batch."""
    __tablename__ = "data_lineage"
    id: int  # serial PK
    symbol: str
    date: date
    table_name: str        # "daily_ohlcv", "features", "institutional_flows", etc.
    provider: str          # "yfinance", "jugaad", "nse", etc.
    fetched_at: datetime   # UTC timestamp
    row_count: int
    created_at: datetime   # server default

class DataQualityReport(Base):
    """One row per (symbol, date, check_type) — historical quality record."""
    __tablename__ = "data_quality_reports"
    id: int  # serial PK
    symbol: str            # null for universe-level checks
    report_date: date      # date being checked
    check_type: str        # "completeness" | "freshness" | "consistency" | "anomaly"
    passed: bool
    severity: str          # "warning" | "critical"
    detail: str            # human-readable failure description
    created_at: datetime   # server default
```

### QualityChecker Interface

```python
@dataclass
class QualityResult:
    check_type: str
    passed: bool
    severity: str          # "warning" | "critical"
    detail: str
    symbol: str | None = None

@dataclass
class QualityReport:
    report_date: date
    results: list[QualityResult]
    n_passed: int
    n_warnings: int
    n_critical: int

class QualityChecker:
    async def check_completeness(self, report_date: date) -> list[QualityResult]:
        """Are all expected universe symbols present in daily_ohlcv for this date?"""
        # Compares universe symbols against symbols with data on report_date
        # Critical if >10% missing; warning if 1-10% missing

    async def check_freshness(self) -> list[QualityResult]:
        """Is each symbol's latest row <= 1 trading day old?"""
        # Accounts for weekends and NSE holidays
        # Critical if any symbol is >2 trading days stale

    async def check_consistency(self, report_date: date) -> list[QualityResult]:
        """Per-symbol price/volume sanity checks."""
        # Checks: volume > 0, high >= low, open > 0, close > 0
        # Checks: |close/open - 1| <= 0.25 unless circuit_hit is set
        # Warning per failing symbol

    async def check_anomalies(self, report_date: date) -> list[QualityResult]:
        """Statistical anomaly detection using rolling windows."""
        # Price spike: close > mean(30d) + 4 * std(30d)
        # Volume spike: volume > 5 * mean(20d volume)
        # Uses only data before report_date (no look-ahead)
        # Warning per anomalous symbol

    async def run_full_check(self, report_date: date) -> QualityReport:
        """Run all four checks and persist results to data_quality_reports."""
```

### Integration Points

- `ingest_symbol()` → writes one `DataLineage` row after successful `store_ohlcv()`
- `refresh_latest()` → calls `QualityChecker.run_full_check(today)` at end; calls `alerter.data_quality_failed()` if any critical results
- `EmailAlerter.data_quality_failed(report: QualityReport)` — new alert method listing critical failures
- Scheduler: after `daily_evaluation` at 3:45 PM IST, runs `run_full_check(today)`

### CLI

```bash
alphavedha data quality-check                    # check today
alphavedha data quality-check --date 2026-05-28  # check a specific date
```

Output: table of check results with symbol, check type, severity, detail.

### Tests
- `tests/unit/data/test_quality.py` — 12+ tests covering each check type with pass/fail scenarios, mocked DB sessions
- Tests for lineage write in `test_ingestion.py`

---

## D7.3: BSE Corporate Announcements + Google Trends

### New Files
- `alphavedha/data/providers/bse_provider.py` — `BSEProvider`
- `alphavedha/data/providers/trends_provider.py` — `GoogleTrendsProvider`
- `alphavedha/features/corporate_events.py` — BSE-derived features
- `alphavedha/features/trends_features.py` — Google Trends features

### Modified Files
- `alphavedha/data/models.py` — `CorporateAnnouncement` ORM model
- `alphavedha/data/ingestion.py` — `ingest_bse_announcements()`, `ingest_trends()`
- `alphavedha/features/pipeline.py` — integrate new feature modules
- `alphavedha/cli/main.py` — `data bse-refresh`, `data trends-refresh` commands
- `alphavedha/scheduler.py` — weekly BSE refresh, weekly trends refresh

### BSE Provider

```python
@dataclass
class CorporateAnnouncementRecord:
    symbol: str
    announced_date: date
    ex_date: date | None       # ex-date for dividends/bonus; None for meetings/AGM
    event_type: str            # "BOARD_MEETING" | "DIVIDEND" | "BONUS" | "RIGHTS" |
                               # "BUYBACK" | "SPLIT" | "AGM" | "EGM" | "OTHER"
    description: str           # raw announcement text (truncated to 500 chars)

class BSEProvider:
    # Imports _BSE_SYMBOL_MAP from sebi_provider.py (already maps NSE→BSE codes)
    # Source: public BSE Corporates API (no API key required)
    # Rate limit: 1 req/2s via existing RateLimiter
    # Endpoint: https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w
    # BSE uses numeric codes; _BSE_SYMBOL_MAP (already in sebi_provider.py) maps NSE→BSE codes

    async def fetch_announcements(
        self, symbol: str, start: date, end: date
    ) -> list[CorporateAnnouncementRecord]: ...

    async def fetch_bulk(
        self, symbols: list[str], start: date, end: date
    ) -> dict[str, list[CorporateAnnouncementRecord]]: ...

    async def health_check(self) -> bool: ...
```

**New DB table:**

```python
class CorporateAnnouncement(Base):
    __tablename__ = "corporate_announcements"
    id: int  # serial PK
    symbol: str
    announced_date: date
    ex_date: date | None
    event_type: str
    description: str
    created_at: datetime
    # Unique constraint: (symbol, announced_date, event_type)
```

Note: distinct from `corporate_actions` (which stores price adjustment factors used by preprocessing). `corporate_announcements` captures forward-looking events and the raw announcement timeline.

### Google Trends Provider

```python
# Sector → search keywords mapping (5 sectors)
SECTOR_KEYWORDS: dict[str, list[str]] = {
    "banking": ["bank nifty", "sbi", "hdfc bank"],
    "it": ["tcs", "infosys", "wipro"],
    "pharma": ["sun pharma", "cipla", "dr reddy"],
    "auto": ["maruti suzuki", "tata motors"],
    "fmcg": ["hindustan unilever", "itc"],
}

class GoogleTrendsProvider:
    # Uses pytrends (pip install pytrends) — no API key
    # Rate limit: 1 req/5s to avoid 429s from Google
    # Granularity: weekly (daily is too noisy and hits rate limits)
    # Timeframe: last 90 days (provides enough history for features)

    async def fetch_sector_trends(
        self, sector: str, timeframe: str = "today 3-m"
    ) -> pd.DataFrame:
        """Returns DataFrame with (date, keyword, relative_interest 0-100)."""

    async def fetch_all_sectors(self) -> dict[str, pd.DataFrame]: ...

    async def health_check(self) -> bool: ...
```

Trends data stored in existing `alternative_data` table with `data_type = "google_trends_{sector}"`.

### New Features

**`features/corporate_events.py`** — computes 3 features per symbol per date:

| Feature | Description |
|---------|-------------|
| `days_to_next_board_meeting` | Trading days until next board meeting (0 if today, -1 if none in 30d) |
| `days_since_dividend_announcement` | Trading days since last dividend/bonus announcement (capped at 30) |
| `corporate_event_this_week` | 1 if any material announcement (BOARD_MEETING, DIVIDEND, BONUS, SPLIT, BUYBACK) in current or next 5 trading days |

All computed using only data with `announced_date <= feature_date` to prevent look-ahead.

**`features/trends_features.py`** — computes 2 features per symbol per date:

| Feature | Description |
|---------|-------------|
| `sector_search_trend_7d` | Normalized sector interest score for the stock's sector (0-100 scale) |
| `sector_search_trend_change` | Week-over-week delta in sector search interest |

Stock → sector mapping uses `configs/stocks.yaml` (already exists).

### Ingestion Functions

```python
async def ingest_bse_announcements(
    tier: str = "large",
    lookback_days: int = 30,
) -> int:  # returns rows stored
    """Fetch and upsert BSE corporate announcements for all symbols in tier."""

async def ingest_trends(
    lookback_days: int = 90,
) -> int:  # returns rows stored
    """Fetch Google Trends for all 5 sectors."""
```

### CLI Commands

```bash
alphavedha data bse-refresh --tier large --days 30
alphavedha data trends-refresh --days 90
```

### Scheduler Jobs

- **Weekly (Sunday 9 AM IST):** `ingest_bse_announcements(tier="large", lookback_days=7)` — catch the week's announcements
- **Weekly (Sunday 9:30 AM IST):** `ingest_trends(lookback_days=90)` — refresh trend scores

### Tests

- `tests/unit/data/test_bse_provider.py` — 8+ tests with mocked BSE API responses
- `tests/unit/data/test_trends_provider.py` — 6+ tests with mocked pytrends responses
- `tests/unit/features/test_corporate_events.py` — 8+ tests for feature computation
- `tests/unit/features/test_trends_features.py` — 6+ tests for trend features

---

## D7.1: Polling-Based Live Data

### New Files
- `alphavedha/data/live_feed.py` — `LiveDataPoller`

### Modified Files
- `alphavedha/data/models.py` — `IntradayOHLCV` ORM model
- `alphavedha/scheduler.py` — `intraday_poll` job (every 2 min, market hours only)
- `alphavedha/cli/main.py` — `data live-status` command

### Data Model

```python
class IntradayOHLCV(Base):
    """Running intraday bar for today — one row per symbol, upserted each poll."""
    __tablename__ = "intraday_ohlcv"
    symbol: str            # PK
    date: date             # trading date (PK)
    open: float            # day's open price
    high: float            # day's high so far
    low: float             # day's low so far
    last_price: float      # most recent price
    volume: int            # cumulative volume
    tick_count: int        # number of polls so far today
    last_updated: datetime # UTC timestamp of last poll
    # Composite PK: (symbol, date)
```

Not a TimescaleDB hypertable — intraday data is at most 50 symbols × 1 row each. Regular table, PK index is sufficient.

After market close, `refresh_latest` ingests the official EOD data into `daily_ohlcv`. The `intraday_ohlcv` row for today remains as a reference but is not used by the feature pipeline after EOD ingestion completes.

### LiveDataPoller

```python
MARKET_OPEN_IST = time(9, 15)
MARKET_CLOSE_IST = time(15, 30)
POLL_INTERVAL_SECONDS = 120          # 2 minutes
FEATURE_RECOMPUTE_EVERY_N_TICKS = 5  # every 10 minutes

class LiveDataPoller:
    def __init__(self, tier: str = "large") -> None: ...

    def is_market_open(self) -> bool:
        """True if current IST time is within market hours on a weekday."""

    async def poll_once(self) -> PollResult:
        """Poll all symbols via yf.Ticker.fast_info, upsert intraday_ohlcv."""
        # Uses asyncio.gather with semaphore (10 concurrent) over symbols
        # Each poll: yf.Ticker(symbol).fast_info → last_price, day_high, day_low,
        #            day_volume (all available without a full download)
        # Upsert: UPDATE intraday_ohlcv SET high=max(high,new), low=min(low,new),
        #         last_price=new, volume=new, tick_count=tick_count+1, last_updated=now
        #         WHERE symbol=? AND date=today
        # On first poll of the day: INSERT with open=last_price

    async def maybe_trigger_feature_recompute(
        self, symbols_updated: list[str], tick_count: int
    ) -> None:
        """Every 5 ticks: delete Redis feature cache keys for updated symbols."""
        # Redis key pattern: "features:{symbol}:*"
        # Deletion triggers lazy recompute on next prediction request

    async def run(self) -> None:
        """Main loop: poll every 2 min while market is open, back off on errors."""
        # Exponential backoff on yfinance errors: 2min → 4min → 8min (cap at 8min)
        # Logs every poll result with symbol count, error count, duration

@dataclass
class PollResult:
    symbols_polled: int
    symbols_updated: int
    symbols_failed: int
    duration_seconds: float
    errors: dict[str, str]  # symbol → error message
```

### Scheduler Integration

```python
# In scheduler.py — new job alongside existing ones
async def run_intraday_poll(self) -> None:
    """Poll live prices every 2 minutes during market hours."""
    poller = LiveDataPoller(tier=os.environ.get("LIVE_FEED_TIER", "large"))
    if not poller.is_market_open():
        return  # no-op outside market hours
    result = await poller.poll_once()
    await poller.maybe_trigger_feature_recompute(...)
    # Logs result; EmailAlerter.scheduler_job_failed() on exception
```

Registered with: `schedule.every(2).minutes.do(self.run_intraday_poll)`

### CLI

```bash
alphavedha data live-status
```

Output:
```
Live Data Status (as of 10:32:15 IST)
  Market open:    Yes
  Last poll:      10:32:00 IST (15s ago)
  Symbols tracked: 50
  Today's ticks:  38

  Symbol    Last Price  Change%  Volume     Last Updated
  TCS.NS    3,942.50    +0.8%    1,234,567  10:32:00
  INFY.NS   1,823.10    -0.2%    987,654    10:32:00
  ...
```

### Graceful Degradation

- If yfinance returns 429 (rate limited): back off exponentially, log warning, continue on next cycle
- If yfinance is completely unavailable: predictions still work using last cached daily features — they'll be slightly stale (max 24h old)
- If Redis is unavailable: skip cache invalidation, log warning — feature recompute happens lazily anyway
- `live-status` CLI shows "Market closed" outside 9:15–3:30 IST

### Tests

- `tests/unit/data/test_live_feed.py` — 10+ tests: `is_market_open` boundary conditions (9:14, 9:15, 15:30, 15:31, weekend), `poll_once` with mocked yfinance, backoff logic, upsert behavior (first tick of day vs. subsequent)

---

## Database Migration

One Alembic migration covering all three sub-projects:

```
New tables:
  - data_lineage            (D7.2)
  - data_quality_reports    (D7.2)
  - corporate_announcements (D7.3)
  - intraday_ohlcv          (D7.1)
```

---

## New Dependencies

```
pytrends>=4.9.2    # Google Trends (D7.3)
```

No other new dependencies. `LIVE_FEED_TIER` env var (default: `"large"`) should be added to `.env.prod.example`. `yfinance` (already installed) provides `fast_info` for live polling.

---

## Success Criteria

| Sub-project | Done when |
|-------------|-----------|
| D7.2 Quality | `data quality-check` runs without errors; `data_quality_reports` populated after daily ingestion; alert email sent on critical failure |
| D7.3 BSE | `data bse-refresh` stores announcements; 3 corporate event features appear in feature pipeline output |
| D7.3 Trends | `data trends-refresh` stores trend scores; 2 trend features appear in feature pipeline output |
| D7.1 Live | `data live-status` shows live prices during market hours; `intraday_ohlcv` updated every 2 min; Redis cache invalidated on tick |

---

## Out of Scope

- WebSocket feeds (requires broker API subscription — future D7.1 upgrade)
- MCA filings (complex scraping, unreliable, low signal-to-noise)
- Satellite imagery (commercial cost exceeds $0-15/mo budget)
- Intraday feature computation (features still computed on daily bars; live data updates the bar in-progress)
- Real-time predictions during market hours (predictions use latest available features — updated every 10 min via Redis invalidation, not pushed)
