from __future__ import annotations

import hashlib
import math
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Query
from pydantic import BaseModel

router = APIRouter(tags=["ui-support"])


# ── helpers ───────────────────────────────────────────────────────────────────


def _seed(s: str) -> float:
    h = int(hashlib.md5(s.encode()).hexdigest(), 16)
    return (h % 10000) / 10000.0


def _seeded_list(seed_str: str, n: int, lo: float, hi: float) -> list[float]:
    s = _seed(seed_str)
    out: list[float] = []
    for i in range(n):
        s = (s * 16807 + i * 0.001) % 1.0
        out.append(round(lo + s * (hi - lo), 4))
    return out


# ── response models ───────────────────────────────────────────────────────────


class FeatureImportance(BaseModel):
    name: str
    value: float


class ModelBreakdown(BaseModel):
    name: str
    direction: str
    confidence: float
    weight: float
    color: str


class ExplainResponse(BaseModel):
    feature_importance: list[FeatureImportance]
    attention: list[float]
    model_breakdown: list[ModelBreakdown]


class PortfolioSummaryResponse(BaseModel):
    value: float
    daily_change_pct: float
    daily_change_abs: float
    sharpe_30d: float
    active_signals: int
    equity: float
    unrealized_pnl: float
    daily_pnl: float


class ModelInfo(BaseModel):
    name: str
    version: str
    last_trained: str
    accuracy: float
    accuracy_7d: float
    drift_score: float
    inference_ms: int
    status: str


class ModelsStatusResponse(BaseModel):
    models: list[ModelInfo]
    agreement_count: int
    total_models: int
    ensemble_confidence: float
    ensemble_direction: str
    feature_drift: list[dict[str, Any]] | None = None
    pipeline: list[dict[str, Any]] | None = None
    system: dict[str, Any] | None = None


class BacktestSummaryResponse(BaseModel):
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_hold_days: float
    profit_factor: float
    calmar: float
    date_from: str
    date_to: str


class SystemResources(BaseModel):
    cpu_pct: float
    memory_pct: float
    gpu_pct: float
    disk_pct: float


class DriftFeature(BaseModel):
    feature: str
    category: str
    psi: float
    ks_stat: float
    status: str
    trend: list[float]


class ExperimentRun(BaseModel):
    run_id: str
    model: str
    date: str
    train_acc: float
    val_acc: float
    sharpe: float
    max_dd: float
    feature_count: int
    duration_min: int
    status: str
    hyperparams: dict[str, Any]


class CorporateEvent(BaseModel):
    date: str
    symbol: str
    company: str
    type: str
    description: str
    value: str | None = None
    value_label: str | None = None
    revenue: float | None = None
    profit: float | None = None
    eps_actual: float | None = None
    eps_estimate: float | None = None
    surprise_pct: float | None = None
    stock_reaction_1d: float | None = None


class SectorTrend(BaseModel):
    sector: str
    current: float
    change_7d: float
    history: list[float]


class QualityCheck(BaseModel):
    date: str
    check_type: str
    symbols_checked: int
    pass_count: int
    fail_count: int
    critical: int
    status: str


class PipelineStatusItem(BaseModel):
    name: str
    status: str
    last_fetch: str
    coverage: float


class ScanStockItem(BaseModel):
    symbol: str
    name: str
    price: float
    change_pct: float
    sector: str
    cap: str
    direction: str
    confidence: float
    regime: str
    t7: dict[str, float]
    t15: dict[str, float]
    t30: dict[str, float]
    sparkline: list[float]
    composite_score: float
    meta_confidence: float
    magnitude: float
    price_targets: dict[str, float]
    top_feature: str
    ai_insight: str


class ScanResponseModel(BaseModel):
    buy_candidates: list[ScanStockItem]
    sell_candidates: list[ScanStockItem]
    excluded: list[str]


class StockSearchResult(BaseModel):
    symbol: str
    name: str
    sector: str
    cap: str


# ── stock universe ────────────────────────────────────────────────────────────

NIFTY_50: list[tuple[str, str, str, str]] = [
    ("TCS", "Tata Consultancy Services", "IT", "Large"),
    ("HDFCBANK", "HDFC Bank", "Banking", "Large"),
    ("RELIANCE", "Reliance Industries", "Energy", "Large"),
    ("INFY", "Infosys", "IT", "Large"),
    ("ICICIBANK", "ICICI Bank", "Banking", "Large"),
    ("SBIN", "State Bank of India", "Banking", "Large"),
    ("BHARTIARTL", "Bharti Airtel", "Telecom", "Large"),
    ("LT", "Larsen & Toubro", "Infrastructure", "Large"),
    ("AXISBANK", "Axis Bank", "Banking", "Large"),
    ("ITC", "ITC Limited", "FMCG", "Large"),
    ("MARUTI", "Maruti Suzuki", "Auto", "Large"),
    ("SUNPHARMA", "Sun Pharmaceutical", "Pharma", "Large"),
    ("WIPRO", "Wipro", "IT", "Large"),
    ("HCLTECH", "HCL Technologies", "IT", "Large"),
    ("BAJFINANCE", "Bajaj Finance", "Finance", "Large"),
    ("TATASTEEL", "Tata Steel", "Metals", "Large"),
    ("ADANIENT", "Adani Enterprises", "Energy", "Large"),
    ("KOTAKBANK", "Kotak Mahindra Bank", "Banking", "Large"),
    ("HINDUNILVR", "Hindustan Unilever", "FMCG", "Large"),
    ("NTPC", "NTPC Limited", "Energy", "Large"),
]

NIFTY_50_MID: list[tuple[str, str, str, str]] = [
    ("PIDILITIND", "Pidilite Industries", "FMCG", "Mid"),
    ("MUTHOOTFIN", "Muthoot Finance", "Finance", "Mid"),
    ("VOLTAS", "Voltas", "Infrastructure", "Mid"),
    ("IDEA", "Vodafone Idea", "Telecom", "Mid"),
    ("IDFCFIRSTB", "IDFC First Bank", "Banking", "Mid"),
    ("BANKBARODA", "Bank of Baroda", "Banking", "Mid"),
    ("TATAPOWER", "Tata Power", "Energy", "Mid"),
    ("GAIL", "GAIL India", "Energy", "Mid"),
    ("MPHASIS", "Mphasis", "IT", "Mid"),
    ("FEDERALBNK", "Federal Bank", "Banking", "Mid"),
]

# Approximate reference prices for realistic demo data
_REF_PRICES: dict[str, float] = {
    "TCS": 3800,
    "HDFCBANK": 1650,
    "RELIANCE": 2950,
    "INFY": 1780,
    "ICICIBANK": 1200,
    "SBIN": 820,
    "BHARTIARTL": 1580,
    "LT": 3500,
    "AXISBANK": 1180,
    "ITC": 470,
    "MARUTI": 12500,
    "SUNPHARMA": 1700,
    "WIPRO": 560,
    "HCLTECH": 1680,
    "BAJFINANCE": 6900,
    "TATASTEEL": 165,
    "ADANIENT": 2400,
    "KOTAKBANK": 1950,
    "HINDUNILVR": 2600,
    "NTPC": 390,
    "PIDILITIND": 2800,
    "MUTHOOTFIN": 1900,
    "VOLTAS": 1550,
    "IDEA": 12,
    "IDFCFIRSTB": 72,
    "BANKBARODA": 260,
    "TATAPOWER": 420,
    "GAIL": 215,
    "MPHASIS": 2900,
    "FEDERALBNK": 195,
}

FEATURE_NAMES = [
    "RSI_14",
    "MACD",
    "Vol_SMA_20",
    "VIX_corr",
    "Boll_Width",
    "ADX",
    "OBV_slope",
    "FII_flow",
    "Put_Call_R",
    "Delivery_pct",
]


# ── endpoints ─────────────────────────────────────────────────────────────────

_REGIMES = ["Bull", "Bear", "Sideways", "HighVol"]
_INSIGHTS = [
    "Strong momentum with FII accumulation",
    "Breakout above 200-DMA on volume",
    "Put/Call ratio at multi-week low — bullish bias",
    "Options chain shows strong support at current levels",
    "Delivery % rising — retail conviction building",
]


def _make_scan_stock(symbol: str, name: str, sector: str, cap: str, seed_str: str) -> ScanStockItem:
    s = _seed(seed_str)
    ref = _REF_PRICES.get(symbol, 1000.0)
    price = round(ref * (0.92 + s * 0.16), 2)
    change_pct = round((s - 0.5) * 6, 2)
    direction = "UP" if s > 0.45 else "DOWN"
    confidence = round(55 + s * 38, 1)
    comp = round(40 + s * 55, 2)
    regime_idx = int(s * 4) % 4
    spark_vals = _seeded_list(f"{seed_str}-spark", 20, price * 0.96, price * 1.04)
    tgt_low = round(price * (1 + 0.03 * (1 if direction == "UP" else -1)), 2)
    tgt_mid = round(price * (1 + 0.06 * (1 if direction == "UP" else -1)), 2)
    tgt_high = round(price * (1 + 0.10 * (1 if direction == "UP" else -1)), 2)
    insight_idx = int(s * len(_INSIGHTS)) % len(_INSIGHTS)
    return ScanStockItem(
        symbol=symbol,
        name=name,
        sector=sector,
        cap=cap,
        price=price,
        change_pct=change_pct,
        direction=direction,
        confidence=confidence,
        regime=_REGIMES[regime_idx],
        composite_score=comp,
        meta_confidence=round(s, 4),
        magnitude=round(abs(change_pct) / 100, 4),
        t7={"low": tgt_low, "high": tgt_high, "pct": round(abs(change_pct) * 1.2, 2)},
        t15={
            "low": round(tgt_low * 0.98, 2),
            "high": round(tgt_high * 1.02, 2),
            "pct": round(abs(change_pct) * 2.0, 2),
        },
        t30={
            "low": round(tgt_low * 0.95, 2),
            "high": round(tgt_high * 1.05, 2),
            "pct": round(abs(change_pct) * 3.2, 2),
        },
        sparkline=spark_vals,
        price_targets={"low": tgt_low, "mid": tgt_mid, "high": tgt_high},
        top_feature=FEATURE_NAMES[int(s * len(FEATURE_NAMES)) % len(FEATURE_NAMES)],
        ai_insight=_INSIGHTS[insight_idx],
    )


@router.get("/scan/{tier}", response_model=ScanResponseModel)
async def scan_stocks(tier: str = "large") -> ScanResponseModel:
    universe = NIFTY_50 if tier != "mid" else NIFTY_50 + NIFTY_50_MID
    today = datetime.now().strftime("%Y-%m-%d")
    stocks = [
        _make_scan_stock(sym, name, sec, cap, f"{sym}-{today}") for sym, name, sec, cap in universe
    ]
    buy = sorted(
        [s for s in stocks if s.direction == "UP"], key=lambda x: x.confidence, reverse=True
    )
    sell = sorted(
        [s for s in stocks if s.direction == "DOWN"], key=lambda x: x.confidence, reverse=True
    )
    return ScanResponseModel(buy_candidates=buy, sell_candidates=sell, excluded=[])


@router.get("/predict/{symbol}/explain", response_model=ExplainResponse)
async def explain_prediction(symbol: str) -> ExplainResponse:
    vals = _seeded_list(f"{symbol}-feat", 10, 0.3, 0.95)
    attn_raw = _seeded_list(f"{symbol}-attn", 60, 0.1, 0.9)
    attn = [round(v * (0.5 + 0.5 * (i / 59)), 4) for i, v in enumerate(attn_raw)]
    s = _seed(symbol)
    dir1 = "UP" if s > 0.5 else "DOWN"
    dir2 = "UP" if s > 0.4 else "DOWN"
    dir3 = "UP" if s > 0.45 else "DOWN"
    return ExplainResponse(
        feature_importance=[
            FeatureImportance(name=n, value=v) for n, v in zip(FEATURE_NAMES, vals, strict=False)
        ],
        attention=attn,
        model_breakdown=[
            ModelBreakdown(
                name="Vedha-α",  # noqa: RUF001
                direction=dir1,
                confidence=round(60 + s * 20, 1),
                weight=0.40,
                color="#3b82f6",
            ),
            ModelBreakdown(
                name="Vedha-β",
                direction=dir2,
                confidence=round(58 + s * 18, 1),
                weight=0.35,
                color="#8b5cf6",
            ),
            ModelBreakdown(
                name="Vedha-γ",  # noqa: RUF001
                direction=dir3,
                confidence=round(62 + s * 15, 1),
                weight=0.25,
                color="#10b981",
            ),
        ],
    )


@router.get("/portfolio/summary", response_model=PortfolioSummaryResponse)
async def portfolio_summary() -> PortfolioSummaryResponse:
    return PortfolioSummaryResponse(
        value=2_485_000,
        daily_change_pct=2.3,
        daily_change_abs=55_842,
        sharpe_30d=1.84,
        active_signals=12,
        equity=1_000_000,
        unrealized_pnl=24_800,
        daily_pnl=5_200,
    )


@router.get("/models/status", response_model=ModelsStatusResponse)
async def models_status() -> ModelsStatusResponse:
    return ModelsStatusResponse(
        models=[
            ModelInfo(
                name="Vedha-α",  # noqa: RUF001
                version="v3.2",
                last_trained="2026-05-14 02:30",
                accuracy=64.2,
                accuracy_7d=0.642,
                drift_score=0.08,
                inference_ms=12,
                status="healthy",
            ),
            ModelInfo(
                name="Vedha-β",
                version="v2.1",
                last_trained="2026-05-14 03:10",
                accuracy=61.8,
                accuracy_7d=0.618,
                drift_score=0.12,
                inference_ms=48,
                status="healthy",
            ),
            ModelInfo(
                name="Vedha-γ",  # noqa: RUF001
                version="v1.8",
                last_trained="2026-05-14 03:55",
                accuracy=63.1,
                accuracy_7d=0.631,
                drift_score=0.19,
                inference_ms=124,
                status="warning",
            ),
            ModelInfo(
                name="Vedha Core",
                version="v4.0",
                last_trained="2026-05-14 04:30",
                accuracy=66.4,
                accuracy_7d=0.664,
                drift_score=0.05,
                inference_ms=6,
                status="healthy",
            ),
        ],
        agreement_count=3,
        total_models=3,
        ensemble_confidence=73,
        ensemble_direction="UP",
        feature_drift=[
            {"feature": "RSI_14", "psi": 0.24},
            {"feature": "VIX_corr", "psi": 0.19},
            {"feature": "FII_flow_5d", "psi": 0.15},
        ],
        pipeline=[
            {
                "name": "NSE Price Feed",
                "status": "ok",
                "last_run": datetime.now().isoformat(),
                "records_processed": 50,
            },
            {
                "name": "Feature Pipeline",
                "status": "ok",
                "last_run": datetime.now().isoformat(),
                "records_processed": 7050,
            },
            {
                "name": "News Sentiment",
                "status": "error",
                "last_run": (datetime.now() - timedelta(hours=6)).isoformat(),
            },
        ],
        system={"cpu_pct": 34.2, "memory_pct": 61.8, "gpu_pct": 28.4},
    )


@router.get("/backtest/summary", response_model=BacktestSummaryResponse)
async def backtest_summary() -> BacktestSummaryResponse:
    return BacktestSummaryResponse(
        cagr=0.241,
        sharpe=1.84,
        max_drawdown=-0.143,
        win_rate=0.623,
        total_trades=342,
        avg_hold_days=4.2,
        profit_factor=2.1,
        calmar=1.69,
        date_from="2019-01-01",
        date_to="2024-12-31",
    )


@router.get("/backtest/equity")
async def backtest_equity() -> dict[str, Any]:
    def gen(seed: int, annual: float) -> list[dict[str, Any]]:
        s, pts = seed, [1_000_000.0]
        today = datetime.today()
        result = []
        for i in range(365):
            s = (s * 16807) % 2_147_483_647
            r = (annual / 252) + (0.12 / math.sqrt(252)) * ((s / 2_147_483_646) - 0.5) * 2
            pts.append(round(pts[-1] * (1 + r)))
            result.append(
                {"y": pts[-1], "date": (today - timedelta(days=364 - i)).strftime("%Y-%m-%d")}
            )
        return result

    return {"strategy": gen(42, 0.342), "benchmark": gen(99, 0.13)}


@router.get("/backtest/monthly")
async def backtest_monthly() -> list[dict[str, Any]]:
    data = [
        (2019, [2.1, -0.8, 3.4, 1.2, -1.5, 4.2, 0.6, -2.1, 3.8, 1.9, -0.3, 2.7]),
        (2020, [1.8, 3.1, -1.2, 2.5, 0.9, -0.4, 3.6, 1.1, -1.8, 4.1, 2.3, 1.5]),
        (2021, [-0.4, 2.8, 4.1, -1.1, 2.2, 0.8, 3.3, -0.9, 2.7, 1.6, 3.8, -0.2]),
        (2022, [1.2, -2.1, 3.5, 0.6, -1.3, 2.9, 1.8, 3.2, -0.7, 2.4, 1.1, 3.6]),
        (2023, [2.8, 1.4, -0.6, 3.1, 2.5, 1.9, -1.2, 3.7, 1.3, 2.6, -0.8, 4.2]),
        (2024, [1.6, 3.8, -0.4, 2.1, 3.4, 1.7, -1.5, 2.9, 4.1, 1.3, 2.7, -0.3]),
    ]
    result = []
    for year, months in data:
        for month, ret in enumerate(months, 1):
            result.append({"year": year, "month": month, "return_pct": ret})
    return result


@router.get("/backtest/distribution")
async def backtest_distribution() -> list[dict[str, Any]]:
    return [
        {"label": "< -5%", "count": 18},
        {"label": "-5 to -3%", "count": 34},
        {"label": "-3 to -1%", "count": 89},
        {"label": "-1 to 0%", "count": 121},
        {"label": "0 to 1%", "count": 198},
        {"label": "1 to 3%", "count": 247},
        {"label": "3 to 5%", "count": 98},
        {"label": "> 5%", "count": 42},
    ]


@router.get("/backtest/rolling-sharpe")
async def backtest_rolling_sharpe() -> list[dict[str, Any]]:
    return [{"y": round(1.2 + (i / 51) * 0.64 + 0.1 * math.sin(i * 0.4), 3)} for i in range(52)]


@router.get("/stocks/search")
async def stocks_search(q: str = Query(default="")) -> dict[str, Any]:
    q_lower = q.lower()
    results = [
        StockSearchResult(symbol=s, name=n, sector=sec, cap=c)
        for s, n, sec, c in NIFTY_50
        if not q_lower or q_lower in s.lower() or q_lower in n.lower() or q_lower in sec.lower()
    ]
    return {"results": [r.model_dump() for r in results[:10]]}


@router.get("/intraday/live")
async def intraday_live(symbol: str | None = None, tier: str | None = None) -> dict[str, Any]:
    from alphavedha.data.live_feed import is_market_open

    market_open = is_market_open()

    if symbol:
        # Return detailed single-symbol response
        s = _seed(symbol)
        base = round(1000 + s * 4000, 2)
        chg = round((s - 0.5) * 4, 2)
        import random as _r

        _r.seed(int(s * 1000))
        candles = []
        price = base * 0.998
        for i in range(78):  # 78 x 5min bars in a trading day
            o = price
            h = round(o * (1 + _r.uniform(0, 0.005)), 2)
            lo = round(o * (1 - _r.uniform(0, 0.005)), 2)
            c = round(lo + _r.random() * (h - lo), 2)
            candles.append(
                {
                    "time": f"09:{15 + i * 5 // 60:02d}:{(i * 5) % 60:02d}",
                    "open": o,
                    "high": h,
                    "low": lo,
                    "close": c,
                    "volume": _r.randint(10000, 500000),
                }
            )
            price = c
        ticks = [
            {
                "time": f"15:{i:02d}",
                "price": round(base * (1 + (i - 30) * 0.0002), 2),
                "change": round((i - 30) * 0.02, 2),
                "volume": _r.randint(1000, 50000),
            }
            for i in range(20)
        ]
        return {
            "symbol": symbol,
            "ltp": round(base * (1 + chg / 100), 2),
            "open": round(base * 0.998, 2),
            "high": round(base * 1.012, 2),
            "low": round(base * 0.988, 2),
            "prev_close": base,
            "change_pct": chg,
            "volume": int(s * 5_000_000),
            "intraday_signal": "UP" if s > 0.5 else "DOWN",
            "candles": candles[:48],
            "recent_ticks": ticks,
        }

    # Return list for scanner view
    stocks = []
    for sym, name, _sec, cap in NIFTY_50:
        s = _seed(sym)
        base = 1000 + s * 4000
        chg = round((s - 0.5) * 4, 2)
        stocks.append(
            {
                "symbol": sym,
                "name": name,
                "open": round(base * 0.998, 2),
                "high": round(base * 1.012, 2),
                "low": round(base * 0.988, 2),
                "price": round(base * (1 + chg / 100), 2),
                "volume": int(s * 5_000_000),
                "change_pct": chg,
                "sparkline": _seeded_list(f"{sym}-intra", 30, base * 0.995, base * 1.005),
                "cap": cap,
            }
        )
    return {"stocks": stocks, "market_open": market_open}


@router.get("/system/health", response_model=SystemResources)
async def system_health() -> SystemResources:
    return SystemResources(cpu_pct=34.2, memory_pct=61.8, gpu_pct=28.4, disk_pct=15.1)


@router.get("/system/data-quality")
async def data_quality() -> dict[str, Any]:
    today = datetime.now().isoformat()
    return {
        "overall_score": 94,
        "symbols_covered": 500,
        "missing_bars": 23,
        "last_updated": today,
        "sources": [
            {
                "name": "NSE Price Feed",
                "description": "OHLCV daily + intraday",
                "status": "ok",
                "last_fetch": today,
                "coverage_pct": 100,
                "records_today": 50000,
            },
            {
                "name": "BSE Price Feed",
                "description": "BSE listed stocks",
                "status": "ok",
                "last_fetch": today,
                "coverage_pct": 98,
                "records_today": 42000,
            },
            {
                "name": "Options Chain",
                "description": "F&O data from NSE",
                "status": "warning",
                "last_fetch": (datetime.now() - timedelta(minutes=47)).isoformat(),
                "coverage_pct": 72,
            },
            {
                "name": "FII/DII Data",
                "description": "Institutional flows",
                "status": "ok",
                "last_fetch": (datetime.now() - timedelta(hours=3)).isoformat(),
                "coverage_pct": 100,
                "records_today": 2,
            },
            {
                "name": "VIX Feed",
                "description": "India VIX index",
                "status": "ok",
                "last_fetch": today,
                "coverage_pct": 100,
                "records_today": 1,
            },
            {
                "name": "News Sentiment",
                "description": "FinBERT scored news",
                "status": "error",
                "last_fetch": (datetime.now() - timedelta(hours=6)).isoformat(),
                "coverage_pct": 0,
            },
            {
                "name": "Google Trends",
                "description": "Search interest signals",
                "status": "ok",
                "last_fetch": (datetime.now() - timedelta(hours=2)).isoformat(),
                "coverage_pct": 85,
                "records_today": 140,
            },
        ],
        "symbol_quality": [{"symbol": s, "score": int(70 + _seed(s) * 30)} for s, *_ in NIFTY_50],
        "issues": [
            {
                "severity": "error",
                "title": "News Sentiment offline",
                "detail": "FinBERT pipeline failed — check FINNHUB_API_KEY",
                "detected_at": (datetime.now() - timedelta(hours=6)).isoformat(),
            },
            {
                "severity": "warning",
                "title": "Options Chain stale",
                "detail": "Last successful fetch was 47 minutes ago (threshold: 30 min)",
                "detected_at": (datetime.now() - timedelta(minutes=47)).isoformat(),
            },
        ],
    }


DRIFT_FEATURES_DATA = [
    ("RSI_14", "Technical", 0.24, 0.18, "ALERT"),
    ("FII_flow_5d", "Macro", 0.15, 0.11, "WARNING"),
    ("MACD", "Technical", 0.08, 0.05, "STABLE"),
    ("VIX_corr", "Macro", 0.19, 0.14, "WARNING"),
    ("Vol_SMA_20", "Technical", 0.06, 0.04, "STABLE"),
    ("Put_Call_R", "Derivatives", 0.22, 0.17, "ALERT"),
    ("OBV_slope", "Microstructure", 0.04, 0.03, "STABLE"),
    ("News_score_5d", "Sentiment", 0.16, 0.12, "WARNING"),
    ("Delivery_pct", "Microstructure", 0.07, 0.05, "STABLE"),
    ("Boll_Width", "Technical", 0.09, 0.06, "STABLE"),
]


@router.get("/features/drift")
async def features_drift() -> list[dict[str, Any]]:
    return [
        DriftFeature(
            feature=f,
            category=c,
            psi=psi,
            ks_stat=ks,
            status=st,
            trend=_seeded_list(f"{f}-trend", 7, max(0.0, psi - 0.05), psi + 0.02),
        ).model_dump()
        for f, c, psi, ks, st in sorted(DRIFT_FEATURES_DATA, key=lambda x: -x[2])
    ]


@router.get("/experiments")
async def list_experiments(model: str | None = None) -> list[dict[str, Any]]:
    runs = [
        ExperimentRun(
            run_id="xgb_20260514_0230",
            model="Vedha-α",  # noqa: RUF001
            date="2026-05-14",
            train_acc=68.4,
            val_acc=64.2,
            sharpe=1.61,
            max_dd=-8.3,
            feature_count=164,
            duration_min=42,
            status="ACTIVE",
            hyperparams={"n_estimators": 800, "max_depth": 6, "learning_rate": 0.05},
        ),
        ExperimentRun(
            run_id="lstm_20260514_0310",
            model="Vedha-β",
            date="2026-05-14",
            train_acc=65.8,
            val_acc=61.8,
            sharpe=1.52,
            max_dd=-8.9,
            feature_count=30,
            duration_min=187,
            status="ACTIVE",
            hyperparams={"hidden": 128, "layers": 2, "window": 60, "lr": 0.001},
        ),
        ExperimentRun(
            run_id="tft_20260514_0355",
            model="Vedha-γ",  # noqa: RUF001
            date="2026-05-14",
            train_acc=66.2,
            val_acc=63.1,
            sharpe=1.58,
            max_dd=-8.1,
            feature_count=164,
            duration_min=312,
            status="ACTIVE",
            hyperparams={"d_model": 64, "n_heads": 4, "horizons": [7, 15, 30]},
        ),
    ]
    if model:
        model_map = {"Vedha-α": "xgb", "Vedha-β": "lstm", "Vedha-γ": "tft"}  # noqa: RUF001
        prefix = model_map.get(model, "")
        runs = [r for r in runs if r.run_id.startswith(prefix)]
    return [r.model_dump() for r in runs]


@router.get("/events/corporate")
async def corporate_events(
    days: int = 30, symbol: str | None = None, type: str | None = None
) -> list[dict[str, Any]]:
    today = datetime.today()
    events = [
        CorporateEvent(
            date=(today + timedelta(days=2)).strftime("%Y-%m-%d"),
            symbol="TCS",
            company="Tata Consultancy Services",
            type="earnings",
            description="Q4 FY26 Results",
            value="₹34.21 EPS",
            value_label="vs ₹33.50 est",
            revenue=15432,
            profit=12428,
            eps_actual=34.21,
            eps_estimate=33.50,
            surprise_pct=2.1,
            stock_reaction_1d=1.8,
        ),
        CorporateEvent(
            date=(today + timedelta(days=3)).strftime("%Y-%m-%d"),
            symbol="HDFCBANK",
            company="HDFC Bank",
            type="dividend",
            description="Final Dividend",
            value="₹19.50",
            value_label="per share",
        ),
        CorporateEvent(
            date=(today + timedelta(days=5)).strftime("%Y-%m-%d"),
            symbol="RELIANCE",
            company="Reliance Industries",
            type="board",
            description="Q4 FY26 Board Meeting — Dividend Declaration",
        ),
        CorporateEvent(
            date=(today + timedelta(days=6)).strftime("%Y-%m-%d"),
            symbol="INFY",
            company="Infosys",
            type="earnings",
            description="Q4 FY26 Results",
            value="₹18.92 EPS",
            value_label="vs ₹18.40 est",
            revenue=8234,
            profit=7312,
            eps_actual=18.92,
            eps_estimate=18.40,
            surprise_pct=2.8,
            stock_reaction_1d=2.4,
        ),
        CorporateEvent(
            date=(today + timedelta(days=8)).strftime("%Y-%m-%d"),
            symbol="SBIN",
            company="State Bank of India",
            type="agm",
            description="Annual General Meeting FY26",
        ),
        CorporateEvent(
            date=(today - timedelta(days=3)).strftime("%Y-%m-%d"),
            symbol="ICICIBANK",
            company="ICICI Bank",
            type="dividend",
            description="Interim Dividend",
            value="₹5.00",
            value_label="per share",
        ),
        CorporateEvent(
            date=(today + timedelta(days=12)).strftime("%Y-%m-%d"),
            symbol="WIPRO",
            company="Wipro",
            type="split",
            description="Stock Split 1:5",
        ),
        CorporateEvent(
            date=(today + timedelta(days=15)).strftime("%Y-%m-%d"),
            symbol="MARUTI",
            company="Maruti Suzuki",
            type="earnings",
            description="Q4 FY26 Results",
        ),
    ]
    if symbol:
        events = [e for e in events if e.symbol == symbol]
    if type and type != "all":
        events = [e for e in events if e.type == type]
    return [e.model_dump() for e in events]


@router.get("/sectors/trends")
async def sector_trends_endpoint() -> dict[str, Any]:
    sectors = [
        {
            "name": "Banking",
            "momentum_7d": 3.2,
            "relative_strength": 0.78,
            "top_gainer": "HDFCBANK",
            "history": [{"y": v} for v in [45, 52, 58, 61, 66, 68, 72, 75, 78]],
        },
        {
            "name": "IT",
            "momentum_7d": -1.4,
            "relative_strength": 0.62,
            "top_gainer": "TCS",
            "history": [{"y": v} for v in [70, 68, 65, 66, 64, 63, 64, 66, 62]],
        },
        {
            "name": "Energy",
            "momentum_7d": 2.8,
            "relative_strength": 0.45,
            "top_gainer": "RELIANCE",
            "history": [{"y": v} for v in [38, 40, 41, 39, 42, 40, 43, 44, 45]],
        },
        {
            "name": "FMCG",
            "momentum_7d": 0.6,
            "relative_strength": 0.34,
            "top_gainer": "HINDUNILVR",
            "history": [{"y": v} for v in [30, 32, 31, 33, 32, 34, 33, 35, 34]],
        },
        {
            "name": "Pharma",
            "momentum_7d": -2.1,
            "relative_strength": 0.56,
            "top_gainer": "SUNPHARMA",
            "history": [{"y": v} for v in [65, 62, 60, 58, 60, 57, 58, 56, 56]],
        },
        {
            "name": "Auto",
            "momentum_7d": 1.8,
            "relative_strength": 0.48,
            "top_gainer": "MARUTI",
            "history": [{"y": v} for v in [40, 42, 43, 45, 44, 46, 47, 48, 48]],
        },
        {
            "name": "Metals",
            "momentum_7d": 4.1,
            "relative_strength": 0.52,
            "top_gainer": "TATASTEEL",
            "history": [{"y": v} for v in [30, 35, 38, 40, 42, 44, 46, 48, 52]],
        },
        {
            "name": "Telecom",
            "momentum_7d": 1.2,
            "relative_strength": 0.39,
            "top_gainer": "BHARTIARTL",
            "history": [{"y": v} for v in [35, 36, 37, 36, 38, 38, 39, 39, 39]],
        },
    ]
    trends_signals = [
        {"keyword": "HDFC Bank", "interest_7d": 82, "direction": "UP"},
        {"keyword": "TCS results", "interest_7d": 71, "direction": "UP"},
        {"keyword": "Nifty Bank", "interest_7d": 68, "direction": "UP"},
        {"keyword": "Reliance Q4", "interest_7d": 65, "direction": "DOWN"},
        {"keyword": "Infosys outlook", "interest_7d": 58, "direction": "DOWN"},
        {"keyword": "SBI dividend", "interest_7d": 54, "direction": "UP"},
        {"keyword": "Bajaj Finance NPA", "interest_7d": 49, "direction": "DOWN"},
        {"keyword": "Adani shares", "interest_7d": 44, "direction": "UP"},
    ]
    return {"sectors": sectors, "trends_signals": trends_signals}


@router.get("/notifications")
async def get_notifications() -> list[dict[str, Any]]:
    now = datetime.now()
    return [
        {
            "id": "n1",
            "type": "signal",
            "title": "Strong BUY signal: HDFCBANK",
            "body": "Vedha Core generated 78% confidence UP signal. Regime: Bull. T7 target: +4.2%.",
            "read": False,
            "created_at": (now - timedelta(minutes=5)).isoformat(),
        },
        {
            "id": "n2",
            "type": "regime",
            "title": "Regime shift detected",
            "body": "HMM model transitioned from Sideways → Bull with 84% confidence.",
            "read": False,
            "created_at": (now - timedelta(hours=1)).isoformat(),
        },
        {
            "id": "n3",
            "type": "alert",
            "title": "TCS earnings in 2 days",
            "body": "Q4 FY26 results on 3 Jun. Analyst consensus: ₹33.50 EPS. Current: +2.3% pre-earnings.",
            "read": False,
            "created_at": (now - timedelta(hours=3)).isoformat(),
        },
        {
            "id": "n4",
            "type": "system",
            "title": "Model retrain complete",
            "body": "Vedha-α retrained on 2026-05-14 data. Accuracy: 64.2% (+1.8pp). Now active.",  # noqa: RUF001
            "read": True,
            "created_at": (now - timedelta(hours=8)).isoformat(),
        },
        {
            "id": "n5",
            "type": "news",
            "title": "RBI holds repo rate at 6.25%",
            "body": "Positive for Banking sector. HDFC Bank, ICICI Bank, Axis Bank likely beneficiaries.",
            "read": True,
            "created_at": (now - timedelta(days=1)).isoformat(),
        },
    ]


@router.post("/notifications/read-all")
async def mark_all_read() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/paper/positions")
async def paper_positions() -> list[dict[str, Any]]:
    return [
        {
            "id": "p1",
            "symbol": "TCS",
            "side": "BUY",
            "quantity": 10,
            "entry_price": 3820.0,
            "ltp": 3891.0,
            "unrealized_pnl": 710.0,
        },
        {
            "id": "p2",
            "symbol": "HDFCBANK",
            "side": "BUY",
            "quantity": 25,
            "entry_price": 1642.0,
            "ltp": 1628.0,
            "unrealized_pnl": -350.0,
        },
        {
            "id": "p3",
            "symbol": "INFY",
            "side": "BUY",
            "quantity": 15,
            "entry_price": 1482.0,
            "ltp": 1521.0,
            "unrealized_pnl": 585.0,
        },
    ]


@router.get("/paper/orders")
async def paper_orders() -> list[dict[str, Any]]:
    now = datetime.now()
    return [
        {
            "id": "o1",
            "symbol": "TCS",
            "side": "BUY",
            "quantity": 10,
            "price": 3820.0,
            "status": "FILLED",
            "timestamp": (now - timedelta(hours=2)).isoformat(),
        },
        {
            "id": "o2",
            "symbol": "HDFCBANK",
            "side": "BUY",
            "quantity": 25,
            "price": 1642.0,
            "status": "FILLED",
            "timestamp": (now - timedelta(hours=3)).isoformat(),
        },
        {
            "id": "o3",
            "symbol": "WIPRO",
            "side": "BUY",
            "quantity": 20,
            "price": 542.0,
            "status": "PENDING",
            "timestamp": (now - timedelta(minutes=30)).isoformat(),
        },
    ]


@router.get("/paper/equity-history")
async def paper_equity_history() -> list[dict[str, Any]]:
    base = 1_000_000.0
    return [{"y": round(base * (1 + i * 0.002 + _seed(str(i)) * 0.003 - 0.001))} for i in range(30)]


@router.post("/paper/orders")
async def place_paper_order(order: dict[str, Any]) -> dict[str, str]:
    return {"id": f"o{abs(hash(str(order))) % 10000}"}


@router.post("/paper/positions/{position_id}/close")
async def close_paper_position(position_id: str) -> dict[str, str]:
    return {"status": "closed", "id": position_id}


@router.get("/public/track-record")
async def public_track_record() -> dict[str, Any]:
    import random as _r

    _r.seed(42)
    now = datetime.now()
    accuracy_over_time = [{"y": round(0.58 + _r.random() * 0.12, 3)} for _ in range(60)]
    recent_preds = [
        {
            "date": (now - timedelta(days=i)).strftime("%Y-%m-%d"),
            "symbol": NIFTY_50[i % len(NIFTY_50)][0],
            "predicted": "UP" if _r.random() > 0.4 else "DOWN",
            "confidence": round(0.55 + _r.random() * 0.3, 2),
            "actual_return": round((_r.random() - 0.4) * 0.06, 4),
            "correct": _r.random() > 0.38,
        }
        for i in range(20)
    ]
    return {
        "total_predictions": 1247,
        "since": "2025-06-01",
        "directional_accuracy": 0.623,
        "precision_up": 0.641,
        "precision_down": 0.598,
        "avg_confidence": 0.694,
        "overall_accuracy": 0.623,
        "accuracy_30d": 0.641,
        "sharpe": 1.84,
        "alpha_pp": 18.4,
        "accuracy_over_time": accuracy_over_time,
        "by_confidence": [
            {"band": "55-65%", "accuracy": 0.548, "count": 312},
            {"band": "65-75%", "accuracy": 0.631, "count": 489},
            {"band": "75-85%", "accuracy": 0.712, "count": 298},
            {"band": "85-100%", "accuracy": 0.784, "count": 148},
        ],
        "signal_breakdown": {"up": 721, "down": 398, "hold": 128},
        "recent_predictions": recent_preds,
    }
