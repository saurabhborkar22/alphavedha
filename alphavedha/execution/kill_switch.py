"""Kill switch — hard safety caps for the execution engine.

Every order must pass through the kill switch before being sent to
the broker. The switch enforces:
  - Master enable flag (EXECUTION_ENABLED env, default 0 = off)
  - Max open positions (default 8)
  - Max daily new exposure as % of equity (default 25%)
  - Daily loss limit as % of equity (default 2%) → flatten + halt
  - Drawdown halt as % of peak equity (default 6%)

When any limit trips, all new orders are blocked and the engine is
halted until manually resumed. A tripped kill switch can only be
cleared by resetting the state (operator action).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

import structlog

logger = structlog.get_logger(__name__)


class HaltReason(StrEnum):
    MASTER_DISABLED = "MASTER_DISABLED"
    MAX_POSITIONS = "MAX_POSITIONS"
    DAILY_EXPOSURE = "DAILY_EXPOSURE"
    DAILY_LOSS = "DAILY_LOSS"
    DRAWDOWN = "DRAWDOWN"
    MANUAL_HALT = "MANUAL_HALT"


@dataclass(frozen=True)
class KillSwitchConfig:
    max_positions: int = 8
    max_daily_exposure_pct: float = 25.0
    daily_loss_limit_pct: float = 2.0
    drawdown_halt_pct: float = 6.0


@dataclass
class KillSwitchState:
    halted: bool = False
    halt_reasons: list[HaltReason] = field(default_factory=list)
    open_positions: int = 0
    daily_exposure_pct: float = 0.0
    daily_pnl_pct: float = 0.0
    drawdown_pct: float = 0.0
    peak_equity: float = 0.0
    current_equity: float = 0.0
    last_check_date: date | None = None


class KillSwitch:
    """Enforces hard risk caps on every order attempt."""

    def __init__(self, config: KillSwitchConfig | None = None) -> None:
        self._config = config or KillSwitchConfig()
        self._state = KillSwitchState()
        self._manually_halted = False

    @property
    def state(self) -> KillSwitchState:
        return self._state

    @property
    def is_enabled(self) -> bool:
        return os.environ.get("EXECUTION_ENABLED", "0") == "1"

    def check(
        self,
        open_positions: int,
        daily_new_exposure_pct: float,
        daily_pnl_pct: float,
        current_equity: float,
        peak_equity: float,
    ) -> KillSwitchState:
        """Run all kill switch checks. Returns current state.

        Must be called before every order attempt.
        """
        reasons: list[HaltReason] = []

        if not self.is_enabled:
            reasons.append(HaltReason.MASTER_DISABLED)

        if self._manually_halted:
            reasons.append(HaltReason.MANUAL_HALT)

        if open_positions >= self._config.max_positions:
            reasons.append(HaltReason.MAX_POSITIONS)

        if daily_new_exposure_pct >= self._config.max_daily_exposure_pct:
            reasons.append(HaltReason.DAILY_EXPOSURE)

        if daily_pnl_pct <= -self._config.daily_loss_limit_pct:
            reasons.append(HaltReason.DAILY_LOSS)

        drawdown_pct = 0.0
        if peak_equity > 0:
            drawdown_pct = (peak_equity - current_equity) / peak_equity * 100.0

        if drawdown_pct >= self._config.drawdown_halt_pct:
            reasons.append(HaltReason.DRAWDOWN)

        halted = len(reasons) > 0

        self._state = KillSwitchState(
            halted=halted,
            halt_reasons=reasons,
            open_positions=open_positions,
            daily_exposure_pct=daily_new_exposure_pct,
            daily_pnl_pct=daily_pnl_pct,
            drawdown_pct=round(drawdown_pct, 2),
            peak_equity=peak_equity,
            current_equity=current_equity,
            last_check_date=date.today(),
        )

        if halted:
            logger.warning(
                "kill_switch_halted",
                reasons=[r.value for r in reasons],
                positions=open_positions,
                daily_exposure_pct=round(daily_new_exposure_pct, 2),
                daily_pnl_pct=round(daily_pnl_pct, 2),
                drawdown_pct=round(drawdown_pct, 2),
            )

        return self._state

    def halt(self) -> None:
        """Manually halt the kill switch (e.g., /panic command)."""
        self._manually_halted = True
        self._state.halted = True
        if HaltReason.MANUAL_HALT not in self._state.halt_reasons:
            self._state.halt_reasons.append(HaltReason.MANUAL_HALT)
        logger.warning("kill_switch_manual_halt")

    def resume(self) -> None:
        """Clear manual halt. Other limits are re-evaluated on next check()."""
        self._manually_halted = False
        logger.info("kill_switch_manual_resume")

    def should_flatten(self) -> bool:
        """Returns True if the engine should flatten all positions (daily loss or drawdown trip)."""
        return (
            HaltReason.DAILY_LOSS in self._state.halt_reasons
            or HaltReason.DRAWDOWN in self._state.halt_reasons
        )
