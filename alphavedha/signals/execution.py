"""Execution timing engine — optimal trade timing and slippage estimation for Indian markets.

Determines when to execute trades, how to split orders into tranches,
and estimates expected slippage based on cap tier, volume, and volatility.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import ClassVar
from zoneinfo import ZoneInfo

import structlog

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)


@dataclass
class ExecutionWindow:
    start: time
    end: time
    quality: str  # "optimal", "acceptable", "avoid"
    reason: str


@dataclass
class ExecutionPlan:
    symbol: str
    cap_tier: str  # "large", "mid", "small"
    recommended_windows: list[ExecutionWindow]
    order_type: str  # "market", "limit", "vwap"
    n_tranches: int
    tranche_interval_minutes: int
    estimated_slippage_pct: float
    warnings: list[str] = field(default_factory=list)


AVOID_WINDOWS = [
    ExecutionWindow(time(9, 15), time(9, 30), "avoid", "opening auction noise"),
    ExecutionWindow(time(15, 20), time(15, 30), "avoid", "closing manipulation risk"),
]

OPTIMAL_WINDOWS = [
    ExecutionWindow(
        time(10, 30), time(11, 30), "optimal", "post-opening stability, good liquidity"
    ),
    ExecutionWindow(
        time(14, 0), time(14, 45), "optimal", "afternoon session, before closing pressure"
    ),
]

_IMPACT_COEFFICIENTS: dict[str, float] = {
    "large": 0.1,
    "mid": 0.3,
    "small": 0.8,
}

_VOL_COEFFICIENT: float = 0.5


class ExecutionEngine:
    """Generates optimal execution plans for Indian equity orders."""

    _TRANCHE_CONFIG: ClassVar[dict[str, dict[str, int | str]]] = {
        "large": {"order_type": "market", "n_tranches": 1, "interval": 0},
        "mid": {"order_type": "limit", "n_tranches": 2, "interval": 10},
        "small": {"order_type": "vwap", "n_tranches": 4, "interval": 8},
    }

    def __init__(self) -> None:
        self._ist = IST

    def plan_execution(
        self,
        symbol: str,
        cap_tier: str,
        avg_daily_volume: float,
        order_size_shares: int,
        current_spread_pct: float = 0.001,
        is_expiry_day: bool = False,
        current_time: datetime | None = None,
    ) -> ExecutionPlan:
        """Generate optimal execution plan."""
        cap_tier = cap_tier.lower()
        if cap_tier not in self._TRANCHE_CONFIG:
            cap_tier = "mid"

        config = self._TRANCHE_CONFIG[cap_tier]
        order_type = str(config["order_type"])
        n_tranches = int(config["n_tranches"])
        interval = int(config["interval"])

        participation_rate = order_size_shares / avg_daily_volume if avg_daily_volume > 0 else 1.0
        if participation_rate > 0.01:
            extra_tranches = int(participation_rate / 0.01)
            n_tranches = max(n_tranches, min(n_tranches + extra_tranches, 10))
            if cap_tier == "large":
                order_type = "limit"
                interval = max(interval, 5)

        if cap_tier == "small":
            n_tranches = max(3, min(n_tranches, 5))

        windows: list[ExecutionWindow] = list(OPTIMAL_WINDOWS)
        warnings: list[str] = []

        if is_expiry_day:
            warnings.append("F&O expiry day — expect higher volatility and wider spreads")
            windows = [w for w in windows if w.end <= time(14, 30)]
            if not windows:
                windows = [
                    ExecutionWindow(
                        time(10, 30),
                        time(11, 30),
                        "optimal",
                        "post-opening stability, good liquidity",
                    )
                ]
            windows.append(
                ExecutionWindow(time(14, 30), time(15, 30), "avoid", "expiry day closing hour")
            )

        slippage = self.estimate_slippage(
            order_size_shares=order_size_shares,
            avg_daily_volume=avg_daily_volume,
            bid_ask_spread_pct=current_spread_pct,
            volatility=0.02,
            cap_tier=cap_tier,
        )

        if slippage > 0.5:
            warnings.append("high slippage risk")

        logger.info(
            "execution_plan_generated",
            symbol=symbol,
            cap_tier=cap_tier,
            order_type=order_type,
            n_tranches=n_tranches,
            slippage_pct=round(slippage, 4),
            participation_rate=round(participation_rate, 4),
        )

        return ExecutionPlan(
            symbol=symbol,
            cap_tier=cap_tier,
            recommended_windows=windows,
            order_type=order_type,
            n_tranches=n_tranches,
            tranche_interval_minutes=interval,
            estimated_slippage_pct=round(slippage, 4),
            warnings=warnings,
        )

    def is_good_time_to_trade(
        self,
        current_time: datetime | None = None,
    ) -> tuple[bool, str]:
        """Check if current time is in a good trading window."""
        now = (current_time or datetime.now(self._ist)).astimezone(self._ist)
        t = now.time()

        if t < MARKET_OPEN or t >= MARKET_CLOSE:
            return False, "market is closed"

        for w in AVOID_WINDOWS:
            if w.start <= t < w.end:
                return False, w.reason

        for w in OPTIMAL_WINDOWS:
            if w.start <= t < w.end:
                return True, w.reason

        return True, "within market hours"

    def estimate_slippage(
        self,
        order_size_shares: int,
        avg_daily_volume: float,
        bid_ask_spread_pct: float,
        volatility: float,
        cap_tier: str,
    ) -> float:
        """Estimate execution slippage as a percentage.

        Model: slippage = base_spread + volume_impact + volatility_premium
        """
        cap_tier = cap_tier.lower()
        impact_coeff = _IMPACT_COEFFICIENTS.get(cap_tier, _IMPACT_COEFFICIENTS["mid"])

        base_spread = bid_ask_spread_pct / 2.0
        participation = order_size_shares / avg_daily_volume if avg_daily_volume > 0 else 1.0
        volume_impact = participation * impact_coeff
        volatility_premium = volatility * _VOL_COEFFICIENT

        slippage = base_spread + volume_impact + volatility_premium

        return max(slippage, 0.0)

    @staticmethod
    def is_expiry_day(dt: datetime | None = None) -> bool:
        """Check if date is F&O expiry (last Thursday of month)."""
        d = (
            (dt or datetime.now(IST)).date()
            if isinstance(dt, datetime)
            else (dt or datetime.now(IST).date())
        )
        last_thursday = _last_thursday_of_month(d.year, d.month)
        return d == last_thursday

    @staticmethod
    def is_weekly_expiry(dt: datetime | None = None) -> bool:
        """Check if date is weekly expiry (every Thursday)."""
        d = (
            (dt or datetime.now(IST)).date()
            if isinstance(dt, datetime)
            else (dt or datetime.now(IST).date())
        )
        return d.weekday() == 3  # Thursday

    @staticmethod
    def next_expiry(dt: datetime | None = None) -> datetime:
        """Return the next monthly F&O expiry date."""
        d = (
            (dt or datetime.now(IST)).date()
            if isinstance(dt, datetime)
            else (dt or datetime.now(IST).date())
        )

        last_thu = _last_thursday_of_month(d.year, d.month)
        if d <= last_thu:
            return datetime(last_thu.year, last_thu.month, last_thu.day, tzinfo=IST)

        if d.month == 12:
            next_year, next_month = d.year + 1, 1
        else:
            next_year, next_month = d.year, d.month + 1

        last_thu = _last_thursday_of_month(next_year, next_month)
        return datetime(last_thu.year, last_thu.month, last_thu.day, tzinfo=IST)


def _last_thursday_of_month(year: int, month: int) -> datetime:
    """Return the date of the last Thursday in the given month."""
    from datetime import date as _date

    last_day = calendar.monthrange(year, month)[1]
    d = _date(year, month, last_day)
    offset = (d.weekday() - 3) % 7  # Thursday = 3
    return d - timedelta(days=offset)
