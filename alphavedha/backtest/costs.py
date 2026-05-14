"""Indian market cost calculator — all regulatory and market costs for backtesting."""

from __future__ import annotations

from dataclasses import dataclass

from alphavedha.config import BacktestConfig, CostsConfig, SlippageConfig


@dataclass
class TradeCost:
    stt: float
    brokerage: float
    exchange_txn: float
    gst: float
    sebi_turnover: float
    stamp_duty: float
    slippage: float
    total: float


def _get_slippage_rate(market_cap_tier: str, slippage_config: SlippageConfig) -> float:
    rates = {
        "large": slippage_config.large_cap,
        "mid": slippage_config.mid_cap,
        "small": slippage_config.small_cap,
    }
    return rates.get(market_cap_tier, slippage_config.mid_cap)


def compute_trade_cost(
    trade_value: float,
    side: str,
    market_cap_tier: str,
    config: CostsConfig,
    slippage_config: SlippageConfig,
) -> TradeCost:
    stt = trade_value * config.stt_delivery
    brokerage = config.brokerage_flat
    exchange_txn = trade_value * config.exchange_txn
    gst = (brokerage + exchange_txn) * config.gst
    sebi_turnover = trade_value * config.sebi_turnover
    stamp_duty = trade_value * config.stamp_duty if side == "buy" else 0.0
    slippage_rate = _get_slippage_rate(market_cap_tier, slippage_config)
    slippage = trade_value * slippage_rate

    total = stt + brokerage + exchange_txn + gst + sebi_turnover + stamp_duty + slippage

    return TradeCost(
        stt=stt,
        brokerage=brokerage,
        exchange_txn=exchange_txn,
        gst=gst,
        sebi_turnover=sebi_turnover,
        stamp_duty=stamp_duty,
        slippage=slippage,
        total=total,
    )


def compute_round_trip_cost_pct(
    market_cap_tier: str,
    config: BacktestConfig,
) -> float:
    ref_value = 100_000.0
    buy = compute_trade_cost(ref_value, "buy", market_cap_tier, config.costs, config.slippage)
    sell = compute_trade_cost(ref_value, "sell", market_cap_tier, config.costs, config.slippage)
    return (buy.total + sell.total) / ref_value
