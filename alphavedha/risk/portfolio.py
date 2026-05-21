"""Portfolio constraints — sector caps, correlation limits, holding periods, liquidity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import structlog

from alphavedha.config import PortfolioConfig

logger = structlog.get_logger(__name__)


@dataclass
class HoldingInfo:
    symbol: str
    sector: str
    weight_pct: float
    entry_date: datetime
    correlation_60d: dict[str, float] = field(default_factory=dict)
    avg_daily_turnover_cr: float = 50.0


@dataclass
class PortfolioState:
    holdings: dict[str, HoldingInfo]
    total_value: float
    peak_value: float


@dataclass
class ConstraintResult:
    adjusted_weight_pct: float
    violations: list[str]
    trade_allowed: bool


class PortfolioConstraints:
    def __init__(self, config: PortfolioConfig) -> None:
        self._config = config

    def check(
        self,
        symbol: str,
        proposed_weight_pct: float,
        sector: str,
        portfolio: PortfolioState,
        avg_daily_turnover_cr: float | None = None,
    ) -> ConstraintResult:
        violations: list[str] = []
        weight = proposed_weight_pct

        # Sell / reduce: check minimum holding period
        if weight < 0.0:
            holding = portfolio.holdings.get(symbol)
            if holding is not None:
                days_held = (datetime.now(UTC) - holding.entry_date).days
                if days_held < self._config.min_holding_days:
                    violations.append(
                        f"Holding period violation: {symbol} held {days_held}d, "
                        f"min {self._config.min_holding_days}d"
                    )
                    return ConstraintResult(
                        adjusted_weight_pct=0.0, violations=violations, trade_allowed=False
                    )
            return ConstraintResult(
                adjusted_weight_pct=weight, violations=violations, trade_allowed=True
            )

        # Buy / add: check liquidity
        turnover = avg_daily_turnover_cr
        if turnover is None:
            existing = portfolio.holdings.get(symbol)
            turnover = existing.avg_daily_turnover_cr if existing else None

        if turnover is not None and turnover < self._config.min_daily_turnover_cr:
            violations.append(
                f"Liquidity violation: {symbol} turnover {turnover:.1f} cr < "
                f"{self._config.min_daily_turnover_cr:.1f} cr min"
            )
            return ConstraintResult(
                adjusted_weight_pct=0.0, violations=violations, trade_allowed=False
            )

        # Check correlation with existing holdings
        for held_sym, held_info in portfolio.holdings.items():
            corr = held_info.correlation_60d.get(symbol, 0.0)
            if abs(corr) > self._config.max_correlation:
                violations.append(
                    f"Correlation violation: {symbol} corr with {held_sym} = "
                    f"{corr:.2f} > {self._config.max_correlation}"
                )
                return ConstraintResult(
                    adjusted_weight_pct=0.0, violations=violations, trade_allowed=False
                )

        # Check sector exposure cap
        current_sector_weight = sum(
            h.weight_pct
            for h in portfolio.holdings.values()
            if h.sector == sector and h.symbol != symbol
        )
        max_allowed = self._config.max_sector_pct - current_sector_weight
        if weight > max_allowed:
            violations.append(
                f"Sector cap: {sector} at {current_sector_weight:.1f}% + "
                f"{weight:.1f}% > {self._config.max_sector_pct}%, "
                f"reduced to {max_allowed:.1f}%"
            )
            weight = max(max_allowed, 0.0)

        logger.debug(
            "portfolio_constraint_check",
            symbol=symbol,
            proposed=proposed_weight_pct,
            adjusted=weight,
            violations=violations,
        )

        return ConstraintResult(
            adjusted_weight_pct=weight,
            violations=violations,
            trade_allowed=weight > 0.0,
        )
