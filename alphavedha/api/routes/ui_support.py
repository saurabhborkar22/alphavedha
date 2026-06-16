"""UI support endpoints — dashboard widgets, scans, paper trading, monitoring.

Demo mode (ALPHAVEDHA_DEMO=1) serves deterministic synthetic data, which is
useful for demos and screenshots. With demo mode off, every endpoint serves
REAL data (database, model artifacts, yfinance) or an honest empty/zero
response — never fabricated numbers.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
from datetime import date, datetime, timedelta
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from alphavedha.api.sim_artifact import load_sim_artifact
from alphavedha.services import ui_data

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["ui-support"])


# ── helpers ───────────────────────────────────────────────────────────────────


def _is_demo() -> bool:
    """True when the ALPHAVEDHA_DEMO env var enables demo (synthetic) mode."""
    return os.environ.get("ALPHAVEDHA_DEMO", "").lower() in ("1", "true", "yes")


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


_DIRECTION_LABELS = {1: "UP", -1: "DOWN", 0: "HOLD"}


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
    tier: str
    buy_candidates: list[ScanStockItem]
    sell_candidates: list[ScanStockItem]
    excluded: list[str]
    # Every scanned stock with full data, sorted by confidence desc — includes
    # non-tradeable names so the dashboard can always show top-N cards.
    all_candidates: list[ScanStockItem] = []
    total_scanned: int


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

_NAME_BY_SYMBOL: dict[str, str] = {sym: name for sym, name, _, _ in NIFTY_50 + NIFTY_50_MID}

# Small symbol strip the UI shows on the intraday scanner when no symbol is
# given — kept small so the non-demo yfinance fetch stays bounded.
_INTRADAY_STRIP: list[tuple[str, str]] = [(sym, name) for sym, name, _, _ in NIFTY_50[:10]]

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


# ── scan ──────────────────────────────────────────────────────────────────────

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


_VALID_SCAN_TIERS = {"large", "mid", "small"}


def _demo_scan(tier: str, top_n: int) -> ScanResponseModel:
    universe = NIFTY_50 if tier != "mid" else NIFTY_50 + NIFTY_50_MID
    today = datetime.now().strftime("%Y-%m-%d")
    stocks = [
        _make_scan_stock(sym, name, sec, cap, f"{sym}-{today}") for sym, name, sec, cap in universe
    ]
    buy = sorted(
        [s for s in stocks if s.direction == "UP"], key=lambda x: x.confidence, reverse=True
    )[:top_n]
    sell = sorted(
        [s for s in stocks if s.direction == "DOWN"], key=lambda x: x.confidence, reverse=True
    )[:top_n]
    return ScanResponseModel(
        tier=tier,
        buy_candidates=buy,
        sell_candidates=sell,
        excluded=[],
        all_candidates=sorted(stocks, key=lambda x: x.confidence, reverse=True),
        total_scanned=len(stocks),
    )


async def _scan_item_from_prediction(pred: Any, cap: str) -> ScanStockItem:
    """Map a real StockPrediction into the UI scan item schema."""
    price = 0.0
    change_pct = 0.0
    sparkline: list[float] = []
    closes = await ui_data.load_recent_closes(pred.symbol)
    if not closes.empty:
        series = closes["close"].astype(float)
        price = round(float(series.iloc[-1]), 2)
        if len(series) >= 2 and float(series.iloc[-2]) != 0.0:
            change_pct = round((float(series.iloc[-1]) / float(series.iloc[-2]) - 1) * 100, 2)
        sparkline = [round(float(v), 2) for v in series.tail(20)]

    direction = _DIRECTION_LABELS.get(int(pred.direction), "HOLD")
    confidence = round(float(pred.meta_confidence) * 100, 1)
    targets = {
        "low": round(float(pred.price_target_low), 2),
        "mid": round(float(pred.price_target_mid), 2),
        "high": round(float(pred.price_target_high), 2),
    }
    # The engine produces one conformal band — reuse it honestly per horizon.
    horizon = {
        "low": targets["low"],
        "high": targets["high"],
        "pct": round(abs(float(pred.magnitude)) * 100, 2),
    }
    top_features = ui_data.read_xgb_feature_importance(limit=1)
    sector = ui_data.load_sector_map().get(pred.symbol, "")
    return ScanStockItem(
        symbol=pred.symbol,
        name=_NAME_BY_SYMBOL.get(pred.symbol, pred.symbol),
        price=price,
        change_pct=change_pct,
        sector=ui_data.sector_display_name(sector) if sector else "Unknown",
        cap=cap,
        direction=direction,
        confidence=confidence,
        regime=str(pred.regime),
        t7=horizon,
        t15=dict(horizon),
        t30=dict(horizon),
        sparkline=sparkline,
        composite_score=round(float(pred.composite_score), 2),
        meta_confidence=round(float(pred.meta_confidence), 4),
        magnitude=round(float(pred.magnitude), 4),
        price_targets=targets,
        top_feature=top_features[0][0] if top_features else "",
        ai_insight=(
            f"Ensemble {direction} signal in {pred.regime} regime "
            f"({confidence:.0f}% meta-confidence)"
        ),
    )


async def _real_scan(tier: str, top_n: int) -> ScanResponseModel:
    from alphavedha.api.deps import get_service

    service = get_service()
    # Single prediction pass over the whole tier, then rank locally — this gives
    # us full data for EVERY stock (including non-tradeable ones) for
    # all_candidates, without a second predict pass.
    predictions = await service.predict_tier(tier)
    result = service._ranker.rank(predictions, top_n=top_n)
    cap = tier.capitalize()
    items: dict[str, ScanStockItem] = {}
    for p in predictions:
        items[p.symbol] = await _scan_item_from_prediction(p, cap)
    all_candidates = sorted(items.values(), key=lambda s: s.confidence, reverse=True)
    buy = [items[p.symbol] for p in result.buy_candidates if p.symbol in items]
    sell = [items[p.symbol] for p in result.sell_candidates if p.symbol in items]
    return ScanResponseModel(
        tier=tier,
        buy_candidates=buy,
        sell_candidates=sell,
        excluded=[sym for sym, _reason in result.excluded],
        all_candidates=all_candidates,
        total_scanned=len(predictions),
    )


@router.get("/scan/{tier}", response_model=ScanResponseModel)
async def scan_stocks(
    tier: str,
    top_n: int = Query(default=10, ge=1, le=50),
) -> ScanResponseModel:
    tier = tier.lower().strip()
    if tier not in _VALID_SCAN_TIERS:
        raise HTTPException(
            status_code=400, detail=f"Invalid tier '{tier}'. Use: {_VALID_SCAN_TIERS}"
        )
    if _is_demo():
        return _demo_scan(tier, top_n)
    try:
        return await _real_scan(tier, top_n)
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("real_scan_failed", tier=tier, error=str(e))
        return ScanResponseModel(
            tier=tier,
            buy_candidates=[],
            sell_candidates=[],
            excluded=[],
            all_candidates=[],
            total_scanned=0,
        )


# ── explain ───────────────────────────────────────────────────────────────────


def _demo_explain(symbol: str) -> ExplainResponse:
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


async def _real_explain(symbol: str) -> ExplainResponse:
    feature_importance = [
        FeatureImportance(name=name, value=round(value, 4))
        for name, value in ui_data.read_xgb_feature_importance(limit=10)
    ]
    model_breakdown: list[ModelBreakdown] = []
    try:
        from alphavedha.api.deps import get_service

        pred = await get_service().predict_single(symbol.upper().strip())
        direction = _DIRECTION_LABELS.get(int(pred.direction), "HOLD")
        agreement = max(0.0, 1.0 - float(pred.model_disagreement))
        model_breakdown = [
            ModelBreakdown(
                name="Ensemble",
                direction=direction,
                confidence=round(float(pred.meta_confidence) * 100, 1),
                weight=round(agreement, 2),
                color="#3b82f6",
            )
        ]
    except Exception as e:
        logger.warning("real_explain_prediction_failed", symbol=symbol, error=str(e))
    # No cheap real attention source — return honestly empty.
    return ExplainResponse(
        feature_importance=feature_importance,
        attention=[],
        model_breakdown=model_breakdown,
    )


@router.get("/predict/{symbol}/explain", response_model=ExplainResponse)
async def explain_prediction(symbol: str) -> ExplainResponse:
    if _is_demo():
        return _demo_explain(symbol)
    try:
        return await _real_explain(symbol)
    except Exception as e:
        logger.warning("real_explain_failed", symbol=symbol, error=str(e))
        return ExplainResponse(feature_importance=[], attention=[], model_breakdown=[])


# ── portfolio summary ─────────────────────────────────────────────────────────

_EMPTY_PORTFOLIO = {
    "value": 0.0,
    "daily_change_pct": 0.0,
    "daily_change_abs": 0.0,
    "sharpe_30d": 0.0,
    "active_signals": 0,
    "equity": 0.0,
    "unrealized_pnl": 0.0,
    "daily_pnl": 0.0,
}


async def _real_portfolio_summary() -> PortfolioSummaryResponse:
    from alphavedha.data.store import load_daily_pnl, load_paper_trades

    value = 0.0
    daily_change_pct = 0.0
    daily_change_abs = 0.0
    sharpe_30d = 0.0

    pnl = await load_daily_pnl()
    if not pnl.empty:
        latest = pnl.iloc[-1]
        value = round(float(latest["portfolio_value"]), 2)
        daily_ret = float(latest["daily_return"])
        daily_change_pct = round(daily_ret * 100, 2)
        if len(pnl) >= 2:
            daily_change_abs = round(value - float(pnl.iloc[-2]["portfolio_value"]), 2)
        elif daily_ret > -1.0:
            daily_change_abs = round(value - value / (1 + daily_ret), 2)
        returns = pnl["daily_return"].astype(float).tail(30)
        std = float(returns.std())
        if len(returns) >= 2 and std > 0:
            sharpe_30d = round(float(returns.mean()) / std * math.sqrt(252), 2)

    active_signals = 0
    unrealized_pnl = 0.0
    trades = await load_paper_trades()
    if not trades.empty:
        open_trades = trades[trades["exit_price"].isna() & trades["entry_price"].notna()]
        active_signals = len(open_trades)
        for _, trade in open_trades.tail(25).iterrows():
            closes = await ui_data.load_recent_closes(str(trade["symbol"]), lookback_days=10)
            if closes.empty:
                continue
            ltp = float(closes["close"].astype(float).iloc[-1])
            entry = float(trade["entry_price"])
            side = 1 if int(trade["predicted_direction"]) >= 0 else -1
            unrealized_pnl += (ltp - entry) * side

    return PortfolioSummaryResponse(
        value=value,
        daily_change_pct=daily_change_pct,
        daily_change_abs=daily_change_abs,
        sharpe_30d=sharpe_30d,
        active_signals=active_signals,
        equity=value,
        unrealized_pnl=round(unrealized_pnl, 2),
        daily_pnl=daily_change_abs,
    )


@router.get("/portfolio/summary", response_model=PortfolioSummaryResponse)
async def portfolio_summary() -> PortfolioSummaryResponse:
    if _is_demo():
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
    try:
        return await _real_portfolio_summary()
    except Exception as e:
        logger.warning("real_portfolio_summary_failed", error=str(e))
        return PortfolioSummaryResponse(**_EMPTY_PORTFOLIO)


# ── models status ─────────────────────────────────────────────────────────────

_MODEL_DISPLAY_NAMES = {
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "tft": "TFT",
    "regime": "HMM Regime",
    "ensemble": "Stacking Ensemble",
    "meta_labeling": "Meta-Labeling",
    "conformal": "Conformal",
}


def _demo_models_status() -> ModelsStatusResponse:
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


# Each model family persists a different primary quality metric in its
# metadata.json — try them in decreasing order of preference. Regime (HMM)
# only stores AIC/log-likelihood, which has no percentage form, so it
# intentionally falls through to 0.0.
_PRIMARY_METRIC_KEYS: tuple[str, ...] = (
    "val_accuracy",
    "accuracy",
    "train_accuracy",
    "r2",
)


async def _live_ensemble_summary() -> tuple[float, str, int]:
    """The ensemble's latest market call from persisted daily predictions.

    Returns (mean confidence % over tradeable signals, majority direction
    UP/DOWN/FLAT, count of symbols agreeing with the majority). Zeros/empty
    until the first 08:30 IST prediction run has persisted paper trades.
    """
    from alphavedha.data.store import load_paper_trades

    today = datetime.now().date()
    try:
        trades = await load_paper_trades(start=today - timedelta(days=7), end=today)
    except Exception as e:
        logger.warning("ensemble_summary_unavailable", error=str(e))
        return 0.0, "", 0
    if trades.empty:
        return 0.0, "", 0

    latest = trades[trades["prediction_date"] == trades["prediction_date"].max()]
    # "Vedha Core" should reflect confidence in the trades we'd actually take:
    # average meta-confidence over signals that passed the meta-labeling gate,
    # not the whole 50-stock cohort (where dozens of rejected signals drag the
    # mean down). On days the gate rejects everything, fall back to the full
    # cohort so the figure is never blank.
    tradeable_mask = latest["is_tradeable"].eq(True)  # null/False → not tradeable
    basis = latest[tradeable_mask]
    if basis.empty:
        basis = latest
    confidence = round(float(basis["confidence"].mean()) * 100, 1)
    net = float(latest["predicted_direction"].mean())
    direction = "UP" if net > 0 else ("DOWN" if net < 0 else "FLAT")
    majority_sign = 1 if net > 0 else -1
    agreement = int((latest["predicted_direction"] == majority_sign).sum())
    return confidence, direction, agreement


async def _real_models_status() -> ModelsStatusResponse:
    models: list[ModelInfo] = []
    for name in ui_data.MODEL_ARTIFACT_NAMES:
        metadata = ui_data.read_model_metadata(name)
        if metadata is None:
            continue
        metrics = metadata.get("metrics") or {}
        raw_acc = next((metrics[k] for k in _PRIMARY_METRIC_KEYS if k in metrics), 0.0)
        try:
            acc = float(raw_acc)
        except (TypeError, ValueError):
            acc = 0.0
        accuracy_pct = round(acc * 100, 1) if 0.0 <= acc <= 1.0 else round(acc, 1)
        last_trained = str(metadata.get("created_at", ""))[:16].replace("T", " ")
        models.append(
            ModelInfo(
                name=_MODEL_DISPLAY_NAMES.get(name, name),
                version=str(metadata.get("version", "")),
                last_trained=last_trained,
                accuracy=accuracy_pct,
                accuracy_7d=0.0,  # no live accuracy tracking yet — honest zero
                drift_score=0.0,  # no stored drift reports yet — honest zero
                inference_ms=0,
                status="active",
            )
        )
    # "Vedha Core" must reflect LIVE ensemble confidence (mean meta-confidence
    # over tradeable signals). Do not substitute the ensemble's *training*
    # accuracy when there are no live predictions yet — that dressed a training
    # metric up as a live signal. 0.0 here honestly means "no live signal yet".
    ensemble_confidence, ensemble_direction, agreement_count = await _live_ensemble_summary()

    return ModelsStatusResponse(
        models=models,
        agreement_count=agreement_count,
        total_models=len(models),
        ensemble_confidence=ensemble_confidence,
        ensemble_direction=ensemble_direction,
        feature_drift=[],
        pipeline=None,
        system=ui_data.system_resources(),
    )


@router.get("/models/status", response_model=ModelsStatusResponse)
async def models_status() -> ModelsStatusResponse:
    if _is_demo():
        return _demo_models_status()
    try:
        return await _real_models_status()
    except Exception as e:
        logger.warning("real_models_status_failed", error=str(e))
        return ModelsStatusResponse(
            models=[],
            agreement_count=0,
            total_models=0,
            ensemble_confidence=0.0,
            ensemble_direction="",
            feature_drift=[],
            pipeline=None,
            system=None,
        )


# ── backtest (no stored backtest run — honest zeros when demo off) ───────────


@router.get("/backtest/summary", response_model=BacktestSummaryResponse)
async def backtest_summary() -> BacktestSummaryResponse:
    if _is_demo():
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
    art = load_sim_artifact()
    if art and art.get("backtest", {}).get("summary"):
        return BacktestSummaryResponse(**art["backtest"]["summary"])
    return BacktestSummaryResponse(
        cagr=0.0,
        sharpe=0.0,
        max_drawdown=0.0,
        win_rate=0.0,
        total_trades=0,
        avg_hold_days=0.0,
        profit_factor=0.0,
        calmar=0.0,
        date_from="",
        date_to="",
    )


@router.get("/backtest/equity")
async def backtest_equity() -> dict[str, Any]:
    if not _is_demo():
        art = load_sim_artifact()
        if art and art.get("backtest", {}).get("equity"):
            return art["backtest"]["equity"]
        return {"strategy": [], "benchmark": []}

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
    if not _is_demo():
        art = load_sim_artifact()
        if art and art.get("backtest", {}).get("monthly"):
            return art["backtest"]["monthly"]
        return []
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
    if not _is_demo():
        art = load_sim_artifact()
        if art and art.get("backtest", {}).get("distribution"):
            return art["backtest"]["distribution"]
        return []
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
    if not _is_demo():
        art = load_sim_artifact()
        if art and art.get("backtest", {}).get("rolling_sharpe"):
            return art["backtest"]["rolling_sharpe"]
        return []
    return [{"y": round(1.2 + (i / 51) * 0.64 + 0.1 * math.sin(i * 0.4), 3)} for i in range(52)]


@router.get("/backtest/range")
async def backtest_range(
    start: str | None = Query(default=None, description="Inclusive start date YYYY-MM-DD"),
    end: str | None = Query(default=None, description="Inclusive end date YYYY-MM-DD"),
) -> dict[str, Any]:
    """Per-day + date-range backtest performance, re-sliced from the equity curve.

    Pass start==end for a single day. Omit both for the full available window.
    Reuses the existing backtest equity series (no recomputation), so it works
    on whatever simulation/backtest data is already present.
    """
    from alphavedha.backtest.sim_views import build_range_view

    equity = await backtest_equity()  # demo-aware; {strategy: [...], benchmark: [...]}
    return build_range_view(
        equity.get("strategy", []), equity.get("benchmark", []), start, end
    )


# ── stock search (static factual reference data — same in both modes) ─────────


@router.get("/stocks/search")
async def stocks_search(q: str = Query(default="")) -> dict[str, Any]:
    q_lower = q.lower()
    results = [
        StockSearchResult(symbol=s, name=n, sector=sec, cap=c)
        for s, n, sec, c in NIFTY_50
        if not q_lower or q_lower in s.lower() or q_lower in n.lower() or q_lower in sec.lower()
    ]
    return {"results": [r.model_dump() for r in results[:10]]}


# ── intraday ──────────────────────────────────────────────────────────────────


def _demo_intraday_symbol(symbol: str) -> dict[str, Any]:
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


def _demo_intraday_list(market_open: bool) -> dict[str, Any]:
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


async def _daily_fallback_quote(symbol: str) -> dict[str, Any] | None:
    """Latest stored daily OHLCV mapped into the intraday quote shape."""
    closes = await ui_data.load_recent_closes(symbol, lookback_days=10)
    if closes.empty:
        return None
    last = closes.iloc[-1]
    prev_close = float(closes.iloc[-2]["close"]) if len(closes) >= 2 else float(last["open"])
    ltp = float(last["close"])
    change_pct = round((ltp / prev_close - 1) * 100, 2) if prev_close else 0.0
    return {
        "symbol": symbol,
        "ltp": round(ltp, 2),
        "open": round(float(last["open"]), 2),
        "high": round(float(last["high"]), 2),
        "low": round(float(last["low"]), 2),
        "prev_close": round(prev_close, 2),
        "change_pct": change_pct,
        "volume": int(last["volume"]),
        "intraday_signal": "UP" if change_pct >= 0 else "DOWN",
        "candles": [],
        "recent_ticks": [],
        "sparkline": [round(float(v), 2) for v in closes["close"].astype(float).tail(30)],
    }


async def _real_intraday_symbol(symbol: str) -> dict[str, Any]:
    data = await asyncio.to_thread(ui_data.fetch_intraday_5m, symbol)
    if data and data.get("candles"):
        candles: list[dict[str, Any]] = data["candles"]
        ltp = float(candles[-1]["close"])
        prev_close = data.get("prev_close")
        if not prev_close:
            closes = await ui_data.load_recent_closes(symbol, lookback_days=10)
            if not closes.empty:
                prev_close = float(closes["close"].astype(float).iloc[-1])
        change_pct = round((ltp / float(prev_close) - 1) * 100, 2) if prev_close else 0.0
        ticks = [
            {
                "time": c["time"][:5],
                "price": c["close"],
                "change": round((float(c["close"]) / float(prev_close) - 1) * 100, 2)
                if prev_close
                else 0.0,
                "volume": c["volume"],
            }
            for c in candles[-20:]
        ]
        return {
            "symbol": symbol,
            "ltp": round(ltp, 2),
            "open": float(candles[0]["open"]),
            "high": max(float(c["high"]) for c in candles),
            "low": min(float(c["low"]) for c in candles),
            "prev_close": round(float(prev_close), 2) if prev_close else 0.0,
            "change_pct": change_pct,
            "volume": int(sum(int(c["volume"]) for c in candles)),
            "intraday_signal": "UP" if change_pct >= 0 else "DOWN",
            "candles": candles,
            "recent_ticks": ticks,
        }

    fallback = await _daily_fallback_quote(symbol)
    if fallback is not None:
        fallback.pop("sparkline", None)
        return fallback
    return {
        "symbol": symbol,
        "ltp": 0.0,
        "open": 0.0,
        "high": 0.0,
        "low": 0.0,
        "prev_close": 0.0,
        "change_pct": 0.0,
        "volume": 0,
        "intraday_signal": "",
        "candles": [],
        "recent_ticks": [],
    }


async def _real_intraday_list(market_open: bool) -> dict[str, Any]:
    strip_symbols = [sym for sym, _name in _INTRADAY_STRIP]
    bulk = await asyncio.to_thread(ui_data.fetch_intraday_bulk, strip_symbols)
    stocks: list[dict[str, Any]] = []
    for sym, name in _INTRADAY_STRIP:
        df = bulk.get(sym)
        if df is not None and not df.empty:
            opens = df["Open"].astype(float)
            closes = df["Close"].astype(float)
            day_open = float(opens.iloc[0])
            price = float(closes.iloc[-1])
            stocks.append(
                {
                    "symbol": sym,
                    "name": name,
                    "open": round(day_open, 2),
                    "high": round(float(df["High"].astype(float).max()), 2),
                    "low": round(float(df["Low"].astype(float).min()), 2),
                    "price": round(price, 2),
                    "volume": int(df["Volume"].fillna(0).sum()),
                    "change_pct": round((price / day_open - 1) * 100, 2) if day_open else 0.0,
                    "sparkline": [round(float(v), 2) for v in closes.tail(30)],
                    "cap": "Large",
                }
            )
            continue
        quote = await _daily_fallback_quote(sym)
        if quote is None:
            continue
        stocks.append(
            {
                "symbol": sym,
                "name": name,
                "open": quote["open"],
                "high": quote["high"],
                "low": quote["low"],
                "price": quote["ltp"],
                "volume": quote["volume"],
                "change_pct": quote["change_pct"],
                "sparkline": quote.get("sparkline", []),
                "cap": "Large",
            }
        )
    return {"stocks": stocks, "market_open": market_open}


@router.get("/intraday/live")
async def intraday_live(symbol: str | None = None, tier: str | None = None) -> dict[str, Any]:
    from alphavedha.data.live_feed import is_market_open

    market_open = is_market_open()

    if _is_demo():
        if symbol:
            return _demo_intraday_symbol(symbol)
        return _demo_intraday_list(market_open)

    try:
        if symbol:
            return await _real_intraday_symbol(symbol.upper().strip())
        return await _real_intraday_list(market_open)
    except Exception as e:
        logger.warning("real_intraday_failed", symbol=symbol, error=str(e))
        if symbol:
            return {
                "symbol": symbol,
                "ltp": 0.0,
                "open": 0.0,
                "high": 0.0,
                "low": 0.0,
                "prev_close": 0.0,
                "change_pct": 0.0,
                "volume": 0,
                "intraday_signal": "",
                "candles": [],
                "recent_ticks": [],
            }
        return {"stocks": [], "market_open": market_open}


# ── system health ─────────────────────────────────────────────────────────────


@router.get("/system/health", response_model=SystemResources)
async def system_health() -> SystemResources:
    if _is_demo():
        return SystemResources(cpu_pct=34.2, memory_pct=61.8, gpu_pct=28.4, disk_pct=15.1)
    return SystemResources(**ui_data.system_resources())


# ── data quality ──────────────────────────────────────────────────────────────


def _demo_data_quality() -> dict[str, Any]:
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


async def _real_data_quality() -> dict[str, Any]:
    now_iso = datetime.now().isoformat()
    stats = await ui_data.ohlcv_store_stats()
    latest = stats["latest_date"]
    # ~3 trading days expressed in calendar days (covers weekends).
    fresh_cutoff = date.today() - timedelta(days=5)
    per_symbol: list[tuple[str, date | None]] = stats["per_symbol_latest"]
    fresh_count = sum(1 for _, d in per_symbol if d is not None and d >= fresh_cutoff)
    coverage = round(100 * fresh_count / len(per_symbol)) if per_symbol else 0
    if stats["row_count"] == 0:
        status = "error"
    elif latest is not None and latest >= fresh_cutoff:
        status = "ok"
    else:
        status = "warning"

    issues: list[dict[str, Any]] = []
    if status == "error":
        issues.append(
            {
                "severity": "error",
                "title": "No OHLCV data",
                "detail": "daily_ohlcv table is empty — run `alphavedha data refresh`",
                "detected_at": now_iso,
            }
        )
    elif status == "warning":
        issues.append(
            {
                "severity": "warning",
                "title": "OHLCV data stale",
                "detail": f"Latest stored date is {latest} (older than 3 trading days)",
                "detected_at": now_iso,
            }
        )

    return {
        "overall_score": coverage,
        "symbols_covered": stats["symbol_count"],
        "missing_bars": 0,
        "last_updated": latest.isoformat() if latest else "",
        "sources": [
            {
                "name": "PostgreSQL OHLCV Store",
                "description": "Daily OHLCV (NSE via yfinance/jugaad)",
                "status": status,
                "last_fetch": latest.isoformat() if latest else "",
                "coverage_pct": coverage,
                "records_today": stats["rows_on_latest_date"],
            }
        ],
        "symbol_quality": [
            {"symbol": sym, "score": 100 if (d is not None and d >= fresh_cutoff) else 50}
            for sym, d in per_symbol
        ],
        "issues": issues,
    }


@router.get("/system/data-quality")
async def data_quality() -> dict[str, Any]:
    if _is_demo():
        return _demo_data_quality()
    try:
        return await _real_data_quality()
    except Exception as e:
        logger.warning("real_data_quality_failed", error=str(e))
        return {
            "overall_score": 0,
            "symbols_covered": 0,
            "missing_bars": 0,
            "last_updated": "",
            "sources": [],
            "symbol_quality": [],
            "issues": [
                {
                    "severity": "error",
                    "title": "Database unavailable",
                    "detail": str(e),
                    "detected_at": datetime.now().isoformat(),
                }
            ],
        }


# ── feature drift ─────────────────────────────────────────────────────────────

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
    if not _is_demo():
        # Real PSI/KS drift needs a stored reference feature window which is
        # not available at request time — honest empty rather than fake drift.
        return []
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


# ── experiments ───────────────────────────────────────────────────────────────


def _demo_experiments(model: str | None) -> list[dict[str, Any]]:
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


def _real_experiments(model: str | None) -> list[dict[str, Any]]:
    from alphavedha.monitoring.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(base_dir=ui_data.get_artifact_dir())
    records = tracker.list_runs(model_name=model, limit=20)

    def _pct(value: Any) -> float:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return 0.0
        return round(v * 100, 1) if 0.0 <= v <= 1.0 else round(v, 1)

    runs: list[dict[str, Any]] = []
    for record in records:
        runs.append(
            ExperimentRun(
                run_id=record.run_id,
                model=record.model_name,
                date=str(record.started_at)[:10],
                train_acc=_pct(record.train_metrics.get("accuracy", 0.0)),
                val_acc=_pct(record.val_metrics.get("accuracy", 0.0)),
                sharpe=round(float(record.val_metrics.get("sharpe", 0.0)), 2),
                max_dd=round(float(record.val_metrics.get("max_drawdown", 0.0)), 2),
                feature_count=int(record.feature_count),
                duration_min=int(record.duration_seconds // 60),
                status="COMPLETED",
                hyperparams=record.hyperparams,
            ).model_dump()
        )
    return runs


@router.get("/experiments")
async def list_experiments(model: str | None = None) -> list[dict[str, Any]]:
    if _is_demo():
        return _demo_experiments(model)
    try:
        return _real_experiments(model)
    except Exception as e:
        logger.warning("real_experiments_failed", error=str(e))
        return []


# ── corporate events ──────────────────────────────────────────────────────────


def _demo_corporate_events() -> list[CorporateEvent]:
    today = datetime.today()
    return [
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


async def _real_corporate_events(days: int) -> list[CorporateEvent]:
    pairs = [(sym, name) for sym, name, _, _ in NIFTY_50]
    raw_events = await asyncio.to_thread(ui_data.fetch_corporate_events, pairs)
    today = date.today()
    window_start = today - timedelta(days=7)
    window_end = today + timedelta(days=days)
    events: list[CorporateEvent] = []
    for raw in raw_events:
        try:
            event_date = date.fromisoformat(str(raw["date"]))
        except ValueError:
            continue
        if not (window_start <= event_date <= window_end):
            continue
        events.append(
            CorporateEvent(
                date=str(raw["date"]),
                symbol=str(raw["symbol"]),
                company=str(raw["company"]),
                type=str(raw["type"]),
                description=str(raw["description"]),
            )
        )
    return events


@router.get("/events/corporate")
async def corporate_events(
    days: int = 30, symbol: str | None = None, type: str | None = None
) -> list[dict[str, Any]]:
    if _is_demo():
        events = _demo_corporate_events()
    else:
        try:
            events = await _real_corporate_events(days)
        except Exception as e:
            logger.warning("real_corporate_events_failed", error=str(e))
            events = []
    if symbol:
        events = [e for e in events if e.symbol == symbol]
    if type and type != "all":
        events = [e for e in events if e.type == type]
    return [e.model_dump() for e in events]


# ── sector trends ─────────────────────────────────────────────────────────────


def _demo_sector_trends() -> dict[str, Any]:
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


@router.get("/sectors/trends")
async def sector_trends_endpoint() -> dict[str, Any]:
    if _is_demo():
        return _demo_sector_trends()
    try:
        sectors = await ui_data.compute_sector_trends()
    except Exception as e:
        logger.warning("real_sector_trends_failed", error=str(e))
        sectors = []
    # No real Google Trends source — honest empty list.
    return {"sectors": sectors, "trends_signals": []}


# ── notifications ─────────────────────────────────────────────────────────────


def _demo_notifications() -> list[dict[str, Any]]:
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


async def _real_notifications() -> list[dict[str, Any]]:
    from alphavedha.data.store import load_paper_trades

    trades = await load_paper_trades(start=date.today() - timedelta(days=5))
    if trades.empty:
        return []
    high_conf = trades[trades["confidence"].astype(float) >= 0.7]
    if high_conf.empty:
        return []
    high_conf = high_conf.sort_values("prediction_date", ascending=False).head(10)
    notifications: list[dict[str, Any]] = []
    for _, trade in high_conf.iterrows():
        side = _DIRECTION_LABELS.get(int(trade["predicted_direction"]), "HOLD")
        side_label = {"UP": "BUY", "DOWN": "SELL"}.get(side, "HOLD")
        regime = trade.get("regime")
        body = (
            f"Model {trade['model_version']} predicted {side} with "
            f"{float(trade['confidence']) * 100:.0f}% confidence"
        )
        body += f" in {regime} regime." if regime else "."
        notifications.append(
            {
                "id": f"{trade['symbol']}-{trade['prediction_date']}",
                "type": "signal",
                "title": f"{side_label} signal: {trade['symbol']}",
                "body": body,
                "read": False,
                "created_at": str(trade["prediction_date"]),
            }
        )
    return notifications


@router.get("/notifications")
async def get_notifications() -> list[dict[str, Any]]:
    if _is_demo():
        return _demo_notifications()
    try:
        return await _real_notifications()
    except Exception as e:
        logger.warning("real_notifications_failed", error=str(e))
        return []


@router.post("/notifications/read-all")
async def mark_all_read() -> dict[str, str]:
    return {"status": "ok"}


# ── paper trading ─────────────────────────────────────────────────────────────


def _trade_id(symbol: str, prediction_date: Any) -> str:
    return f"{symbol}:{prediction_date}"


async def _real_paper_positions() -> list[dict[str, Any]]:
    from alphavedha.data.store import load_paper_trades

    trades = await load_paper_trades()
    if trades.empty:
        return []
    open_trades = trades[trades["exit_price"].isna() & trades["entry_price"].notna()].tail(25)
    positions: list[dict[str, Any]] = []
    for _, trade in open_trades.iterrows():
        symbol = str(trade["symbol"])
        entry = float(trade["entry_price"])
        side = "BUY" if int(trade["predicted_direction"]) >= 0 else "SELL"
        closes = await ui_data.load_recent_closes(symbol, lookback_days=10)
        ltp = float(closes["close"].astype(float).iloc[-1]) if not closes.empty else entry
        sign = 1 if side == "BUY" else -1
        positions.append(
            {
                "id": _trade_id(symbol, trade["prediction_date"]),
                "symbol": symbol,
                "side": side,
                "quantity": 1,
                "entry_price": round(entry, 2),
                "ltp": round(ltp, 2),
                "unrealized_pnl": round((ltp - entry) * sign, 2),
            }
        )
    return positions


@router.get("/paper/positions")
async def paper_positions() -> list[dict[str, Any]]:
    if _is_demo():
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
    try:
        return await _real_paper_positions()
    except Exception as e:
        logger.warning("real_paper_positions_failed", error=str(e))
        return []


async def _real_paper_orders() -> list[dict[str, Any]]:
    from alphavedha.data.store import load_paper_trades

    trades = await load_paper_trades()
    if trades.empty:
        return []
    orders: list[dict[str, Any]] = []
    for _, trade in trades.sort_values("prediction_date", ascending=False).head(50).iterrows():
        symbol = str(trade["symbol"])
        has_entry = trade["entry_price"] is not None and not (
            isinstance(trade["entry_price"], float) and math.isnan(trade["entry_price"])
        )
        orders.append(
            {
                "id": _trade_id(symbol, trade["prediction_date"]),
                "symbol": symbol,
                "side": "BUY" if int(trade["predicted_direction"]) >= 0 else "SELL",
                "quantity": 1,
                "price": round(float(trade["entry_price"]), 2) if has_entry else 0.0,
                "status": "FILLED" if has_entry else "PENDING",
                "timestamp": str(trade["prediction_date"]),
            }
        )
    return orders


@router.get("/paper/orders")
async def paper_orders() -> list[dict[str, Any]]:
    if _is_demo():
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
    try:
        return await _real_paper_orders()
    except Exception as e:
        logger.warning("real_paper_orders_failed", error=str(e))
        return []


@router.get("/paper/equity-history")
async def paper_equity_history() -> list[dict[str, Any]]:
    if _is_demo():
        base = 1_000_000.0
        return [
            {"y": round(base * (1 + i * 0.002 + _seed(str(i)) * 0.003 - 0.001))} for i in range(30)
        ]
    try:
        from alphavedha.data.store import load_daily_pnl

        pnl = await load_daily_pnl()
        if pnl.empty:
            return []
        return [{"y": round(float(v), 2)} for v in pnl["portfolio_value"].astype(float).tail(30)]
    except Exception as e:
        logger.warning("real_equity_history_failed", error=str(e))
        return []


@router.post("/paper/orders")
async def place_paper_order(order: dict[str, Any]) -> dict[str, str]:
    if _is_demo():
        return {"id": f"o{abs(hash(str(order))) % 10000}"}

    symbol = str(order.get("symbol", "")).upper().strip()
    if not symbol:
        raise HTTPException(status_code=400, detail="Order requires a 'symbol'")
    side = str(order.get("side", "BUY")).upper().strip()

    entry_price: float | None = None
    raw_price = order.get("price", order.get("entry_price"))
    if raw_price is not None:
        try:
            entry_price = float(raw_price)
        except (TypeError, ValueError):
            entry_price = None
    if entry_price is None:
        closes = await ui_data.load_recent_closes(symbol, lookback_days=10)
        if not closes.empty:
            entry_price = float(closes["close"].astype(float).iloc[-1])

    today = date.today()
    row = {
        "symbol": symbol,
        "prediction_date": today,
        "predicted_direction": -1 if side == "SELL" else 1,
        "predicted_magnitude": float(order.get("magnitude") or 0.0),
        "confidence": float(order.get("confidence") or 0.0),
        "model_version": "manual",
        "entry_price": entry_price,
    }
    try:
        from alphavedha.data.store import store_paper_trade

        await store_paper_trade(row)
    except Exception as e:
        logger.warning("paper_order_store_failed", symbol=symbol, error=str(e))
        raise HTTPException(status_code=503, detail="Paper trade store unavailable") from e
    return {"id": _trade_id(symbol, today.isoformat())}


@router.post("/paper/positions/{position_id}/close")
async def close_paper_position(position_id: str) -> dict[str, str]:
    if _is_demo():
        return {"status": "closed", "id": position_id}

    symbol, _, day_str = position_id.partition(":")
    symbol = symbol.upper().strip()
    try:
        prediction_date = date.fromisoformat(day_str) if day_str else date.today()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid position id '{position_id}'") from e

    try:
        from alphavedha.data.store import load_paper_trades, update_paper_trade_outcome

        trades = await load_paper_trades(symbol=symbol)
        match = trades[trades["prediction_date"] == prediction_date] if not trades.empty else trades
        if match.empty:
            raise HTTPException(status_code=404, detail=f"Position not found: {position_id}")
        trade = match.iloc[0]

        entry = trade["entry_price"]
        has_entry = entry is not None and not (isinstance(entry, float) and math.isnan(entry))
        closes = await ui_data.load_recent_closes(symbol, lookback_days=10)
        if not closes.empty:
            exit_price = float(closes["close"].astype(float).iloc[-1])
        elif has_entry:
            exit_price = float(entry)
        else:
            return {"status": "error", "id": position_id}

        sign = 1 if int(trade["predicted_direction"]) >= 0 else -1
        actual_return = ((exit_price - float(entry)) / float(entry)) * sign if has_entry else 0.0
        await update_paper_trade_outcome(
            symbol=symbol,
            prediction_date=prediction_date,
            exit_price=exit_price,
            actual_return=actual_return,
            is_correct=actual_return > 0,
        )
        return {"status": "closed", "id": position_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.warning("paper_position_close_failed", id=position_id, error=str(e))
        return {"status": "error", "id": position_id}


# ── public track record ───────────────────────────────────────────────────────


def _demo_track_record() -> dict[str, Any]:
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


_EMPTY_TRACK_RECORD: dict[str, Any] = {
    "total_predictions": 0,
    "since": "",
    "directional_accuracy": 0.0,
    "precision_up": 0.0,
    "precision_down": 0.0,
    "avg_confidence": 0.0,
    "overall_accuracy": 0.0,
    "accuracy_30d": 0.0,
    "sharpe": 0.0,
    "alpha_pp": 0.0,
    "accuracy_over_time": [],
    "by_confidence": [],
    "signal_breakdown": {"up": 0, "down": 0, "hold": 0},
    "recent_predictions": [],
}


async def _real_track_record() -> dict[str, Any]:
    from alphavedha.data.store import load_daily_pnl, load_paper_trades

    trades = await load_paper_trades()
    if trades.empty:
        return dict(_EMPTY_TRACK_RECORD)

    directions = trades["predicted_direction"].astype(int)
    signal_breakdown = {
        "up": int((directions == 1).sum()),
        "down": int((directions == -1).sum()),
        "hold": int((directions == 0).sum()),
    }
    since = str(trades["prediction_date"].min())

    evaluated = trades[trades["is_correct"].notna()].copy()
    if evaluated.empty:
        return {**_EMPTY_TRACK_RECORD, "since": since, "signal_breakdown": signal_breakdown}

    evaluated["is_correct"] = evaluated["is_correct"].astype(bool)
    accuracy = round(float(evaluated["is_correct"].mean()), 3)

    up_preds = evaluated[evaluated["predicted_direction"].astype(int) == 1]
    down_preds = evaluated[evaluated["predicted_direction"].astype(int) == -1]
    precision_up = round(float(up_preds["is_correct"].mean()), 3) if len(up_preds) else 0.0
    precision_down = round(float(down_preds["is_correct"].mean()), 3) if len(down_preds) else 0.0

    cutoff_30d = date.today() - timedelta(days=30)
    recent_30 = evaluated[evaluated["prediction_date"] >= cutoff_30d]
    accuracy_30d = round(float(recent_30["is_correct"].mean()), 3) if len(recent_30) else 0.0

    sharpe = 0.0
    alpha_pp = 0.0
    try:
        pnl = await load_daily_pnl()
        if not pnl.empty:
            returns = pnl["daily_return"].astype(float)
            std = float(returns.std())
            if len(returns) >= 2 and std > 0:
                sharpe = round(float(returns.mean()) / std * math.sqrt(252), 2)
            alpha_pp = round(
                (
                    float(pnl["cumulative_return"].astype(float).iloc[-1])
                    - float(pnl["benchmark_return"].astype(float).sum())
                )
                * 100,
                1,
            )
    except Exception as e:
        logger.warning("track_record_pnl_failed", error=str(e))

    daily_accuracy = evaluated.groupby("prediction_date")["is_correct"].mean()
    accuracy_over_time = [{"y": round(float(v), 3)} for v in daily_accuracy.tail(60)]

    confidence = evaluated["confidence"].astype(float)
    by_confidence: list[dict[str, Any]] = []
    for band, lo, hi in [
        ("55-65%", 0.55, 0.65),
        ("65-75%", 0.65, 0.75),
        ("75-85%", 0.75, 0.85),
        ("85-100%", 0.85, 1.01),
    ]:
        group = evaluated[(confidence >= lo) & (confidence < hi)]
        by_confidence.append(
            {
                "band": band,
                "accuracy": round(float(group["is_correct"].mean()), 3) if len(group) else 0.0,
                "count": len(group),
            }
        )

    recent = evaluated.sort_values("prediction_date", ascending=False).head(20)
    recent_predictions = [
        {
            "date": str(row["prediction_date"]),
            "symbol": str(row["symbol"]),
            "predicted": _DIRECTION_LABELS.get(int(row["predicted_direction"]), "HOLD"),
            "confidence": round(float(row["confidence"]), 2),
            "actual_return": round(float(row["actual_return"]), 4)
            if row["actual_return"] is not None
            else 0.0,
            "correct": bool(row["is_correct"]),
        }
        for _, row in recent.iterrows()
    ]

    return {
        "total_predictions": len(evaluated),
        "since": since,
        "directional_accuracy": accuracy,
        "precision_up": precision_up,
        "precision_down": precision_down,
        "avg_confidence": round(float(confidence.mean()), 3),
        "overall_accuracy": accuracy,
        "accuracy_30d": accuracy_30d,
        "sharpe": sharpe,
        "alpha_pp": alpha_pp,
        "accuracy_over_time": accuracy_over_time,
        "by_confidence": by_confidence,
        "signal_breakdown": signal_breakdown,
        "recent_predictions": recent_predictions,
    }


@router.get("/public/track-record")
async def public_track_record() -> dict[str, Any]:
    if _is_demo():
        return _demo_track_record()
    try:
        return await _real_track_record()
    except Exception as e:
        logger.warning("real_track_record_failed", error=str(e))
        return dict(_EMPTY_TRACK_RECORD)
