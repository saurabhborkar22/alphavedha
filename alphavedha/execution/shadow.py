"""Shadow mode runner — exercises the full execution loop daily without real orders.

Runs at 09:15 IST (market open): takes today's gate-passed predictions,
converts them to order plans via the OMS, fills them via PaperBroker at
market open price ± slippage, and logs everything to shadow_fills.

The slippage distribution from shadow fills feeds back into the cost model,
replacing the flat 0.1%/side assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import structlog

from alphavedha.execution.broker import Fill, PaperBroker
from alphavedha.execution.kill_switch import KillSwitch
from alphavedha.execution.oms import OrderManager

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class ShadowSignal:
    """Input signal for shadow mode — a gate-passed prediction."""

    symbol: str
    direction: int
    magnitude: float
    position_size_pct: float
    entry_price: float
    stop_loss_price: float
    take_profit_price: float
    strategy: str
    market_cap_tier: str = "large"


@dataclass
class ShadowResult:
    """Outcome of a single shadow run."""

    run_date: date
    signals_received: int = 0
    plans_created: int = 0
    orders_placed: int = 0
    orders_blocked: int = 0
    fills_simulated: int = 0
    fills: list[Fill] = field(default_factory=list)
    block_reasons: list[str] = field(default_factory=list)


class ShadowRunner:
    """Runs the full execution loop in shadow mode.

    Usage:
        runner = ShadowRunner()
        result = await runner.run(signals, open_prices)
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        broker: PaperBroker | None = None,
        kill_switch: KillSwitch | None = None,
    ) -> None:
        self._broker = broker or PaperBroker(initial_capital=initial_capital)
        # ghost_mode bypasses only the EXECUTION_ENABLED master flag —
        # the broker here is a PaperBroker (cannot place real orders),
        # and every other risk cap stays enforced daily.
        self._kill_switch = kill_switch or KillSwitch(ghost_mode=True)
        self._oms = OrderManager(
            broker=self._broker,
            kill_switch=self._kill_switch,
            equity=initial_capital,
        )

    @property
    def broker(self) -> PaperBroker:
        return self._broker

    @property
    def oms(self) -> OrderManager:
        return self._oms

    async def run(
        self,
        signals: list[ShadowSignal],
        open_prices: dict[str, float],
    ) -> ShadowResult:
        """Execute shadow loop: signals → plans → orders → fills.

        Args:
            signals: Gate-passed predictions for today.
            open_prices: Market open prices keyed by symbol.

        Returns:
            ShadowResult with fill details and slippage measurements.
        """
        result = ShadowResult(run_date=date.today(), signals_received=len(signals))

        if not signals:
            logger.info("shadow_run_no_signals")
            return result

        for signal in signals:
            plan = self._oms.compute_plan(
                symbol=signal.symbol,
                direction=signal.direction,
                magnitude=signal.magnitude,
                position_size_pct=signal.position_size_pct,
                entry_price=signal.entry_price,
                stop_loss_price=signal.stop_loss_price,
                take_profit_price=signal.take_profit_price,
                strategy=signal.strategy,
            )

            if plan is None:
                continue

            result.plans_created += 1

            oms_result = await self._oms.execute_plan(plan)

            if oms_result.blocked:
                result.orders_blocked += 1
                result.block_reasons.append(f"{signal.symbol}: {oms_result.block_reason}")
                continue

            result.orders_placed += 1

            open_price = open_prices.get(signal.symbol)
            if open_price is None or oms_result.order is None:
                continue

            fill = await self._broker.simulate_fill(
                order_id=oms_result.order.order_id,
                market_open_price=open_price,
                market_cap_tier=signal.market_cap_tier,
            )
            result.fills_simulated += 1
            result.fills.append(fill)

        logger.info(
            "shadow_run_complete",
            date=result.run_date.isoformat(),
            signals=result.signals_received,
            plans=result.plans_created,
            placed=result.orders_placed,
            blocked=result.orders_blocked,
            filled=result.fills_simulated,
        )

        return result

    def slippage_report(self) -> dict[str, float]:
        """Compute slippage statistics from all fills."""
        fills = self._broker.fills
        if not fills:
            return {"count": 0, "mean_bps": 0.0, "max_bps": 0.0, "min_bps": 0.0}

        bps_values = [f.slippage_bps for f in fills]
        return {
            "count": len(bps_values),
            "mean_bps": round(sum(bps_values) / len(bps_values), 2),
            "max_bps": round(max(bps_values), 2),
            "min_bps": round(min(bps_values), 2),
        }


def shadow_fills_to_rows(fills: list[Fill], run_date: date) -> list[dict[str, object]]:
    """Convert Fill objects to dicts suitable for DB insertion."""
    rows: list[dict[str, object]] = []
    for fill in fills:
        rows.append(
            {
                "strategy": fill.tag or "unknown",
                "symbol": fill.symbol,
                "fill_date": run_date,
                "side": fill.side.value,
                "decision_price": fill.decision_price,
                "sim_fill_price": fill.fill_price,
                "quantity": fill.quantity,
                "slippage_bps": fill.slippage_bps,
            }
        )
    return rows
