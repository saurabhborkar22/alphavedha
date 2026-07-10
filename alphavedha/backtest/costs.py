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


INSTRUMENT_DELIVERY = "delivery"
INSTRUMENT_FUTURES = "futures"


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
    instrument: str = INSTRUMENT_DELIVERY,
) -> TradeCost:
    """One-leg cost. ``instrument`` selects the STT/stamp regime.

    - ``delivery``: cash-market STT (0.1%) on BOTH legs; delivery stamp on buy.
    - ``futures``: F&O STT on the SELL leg only; the lower futures stamp on buy.
      This is the leg model for a swing short, which must be a stock future.
    """
    if instrument == INSTRUMENT_FUTURES:
        stt = trade_value * config.stt_fno if side == "sell" else 0.0
        stamp_duty = trade_value * config.stamp_duty_fno if side == "buy" else 0.0
    else:
        stt = trade_value * config.stt_delivery
        stamp_duty = trade_value * config.stamp_duty if side == "buy" else 0.0

    brokerage = config.brokerage_flat
    exchange_txn = trade_value * config.exchange_txn
    gst = (brokerage + exchange_txn) * config.gst
    sebi_turnover = trade_value * config.sebi_turnover
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
    instrument: str = INSTRUMENT_DELIVERY,
) -> float:
    """Round-trip (entry + exit) cost as a fraction of notional.

    For ``futures`` a rollover buffer is added on top of the two legs, since a
    ~15-day swing short usually crosses one monthly expiry and must be rolled.
    """
    ref_value = 100_000.0
    buy = compute_trade_cost(
        ref_value, "buy", market_cap_tier, config.costs, config.slippage, instrument
    )
    sell = compute_trade_cost(
        ref_value, "sell", market_cap_tier, config.costs, config.slippage, instrument
    )
    rollover = (
        ref_value * config.costs.futures_rollover_pct if instrument == INSTRUMENT_FUTURES else 0.0
    )
    return (buy.total + sell.total + rollover) / ref_value


def compute_long_short_cost_pct(
    market_cap_tier: str,
    config: BacktestConfig,
) -> tuple[float, float]:
    """Round-trip cost for a delivery LONG and a futures SHORT — ``(long, short)``.

    Swing longs are cash-delivery (held for days/weeks, unleveraged); swing
    shorts must be stock futures because a cash short can't be held overnight
    in India. Callers apply each cost to the matching leg so the reported P&L
    reflects trades that could actually be placed.
    """
    long_cost = compute_round_trip_cost_pct(market_cap_tier, config, INSTRUMENT_DELIVERY)
    short_cost = compute_round_trip_cost_pct(market_cap_tier, config, INSTRUMENT_FUTURES)
    return long_cost, short_cost
