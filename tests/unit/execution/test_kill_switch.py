"""Tests for kill switch safety caps."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from alphavedha.execution.kill_switch import (
    HaltReason,
    KillSwitch,
    KillSwitchConfig,
)


@pytest.fixture
def ks() -> KillSwitch:
    return KillSwitch(KillSwitchConfig())


@pytest.fixture
def enabled_ks() -> KillSwitch:
    return KillSwitch(KillSwitchConfig())


class TestMasterSwitch:
    def test_disabled_by_default(self, ks: KillSwitch) -> None:
        assert ks.is_enabled is False

    def test_enabled_via_env(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            ks = KillSwitch()
            assert ks.is_enabled is True

    def test_check_blocks_when_disabled(self, ks: KillSwitch) -> None:
        state = ks.check(
            open_positions=0,
            daily_new_exposure_pct=0,
            daily_pnl_pct=0,
            current_equity=1_000_000,
            peak_equity=1_000_000,
        )
        assert state.halted is True
        assert HaltReason.MASTER_DISABLED in state.halt_reasons

    def test_check_passes_when_enabled(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            ks = KillSwitch()
            state = ks.check(
                open_positions=2,
                daily_new_exposure_pct=10,
                daily_pnl_pct=0.5,
                current_equity=1_000_000,
                peak_equity=1_000_000,
            )
            assert state.halted is False
            assert len(state.halt_reasons) == 0


class TestMaxPositions:
    def test_trips_at_max(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            ks = KillSwitch(KillSwitchConfig(max_positions=8))
            state = ks.check(
                open_positions=8,
                daily_new_exposure_pct=0,
                daily_pnl_pct=0,
                current_equity=1_000_000,
                peak_equity=1_000_000,
            )
            assert state.halted is True
            assert HaltReason.MAX_POSITIONS in state.halt_reasons

    def test_ok_below_max(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            ks = KillSwitch(KillSwitchConfig(max_positions=8))
            state = ks.check(
                open_positions=7,
                daily_new_exposure_pct=0,
                daily_pnl_pct=0,
                current_equity=1_000_000,
                peak_equity=1_000_000,
            )
            assert HaltReason.MAX_POSITIONS not in state.halt_reasons


class TestDailyExposure:
    def test_trips_at_limit(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            ks = KillSwitch(KillSwitchConfig(max_daily_exposure_pct=25.0))
            state = ks.check(
                open_positions=0,
                daily_new_exposure_pct=25.0,
                daily_pnl_pct=0,
                current_equity=1_000_000,
                peak_equity=1_000_000,
            )
            assert HaltReason.DAILY_EXPOSURE in state.halt_reasons

    def test_ok_below_limit(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            ks = KillSwitch(KillSwitchConfig(max_daily_exposure_pct=25.0))
            state = ks.check(
                open_positions=0,
                daily_new_exposure_pct=20.0,
                daily_pnl_pct=0,
                current_equity=1_000_000,
                peak_equity=1_000_000,
            )
            assert HaltReason.DAILY_EXPOSURE not in state.halt_reasons


class TestDailyLoss:
    def test_trips_at_limit(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            ks = KillSwitch(KillSwitchConfig(daily_loss_limit_pct=2.0))
            state = ks.check(
                open_positions=0,
                daily_new_exposure_pct=0,
                daily_pnl_pct=-2.0,
                current_equity=980_000,
                peak_equity=1_000_000,
            )
            assert HaltReason.DAILY_LOSS in state.halt_reasons

    def test_ok_above_limit(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            ks = KillSwitch(KillSwitchConfig(daily_loss_limit_pct=2.0))
            state = ks.check(
                open_positions=0,
                daily_new_exposure_pct=0,
                daily_pnl_pct=-1.5,
                current_equity=985_000,
                peak_equity=1_000_000,
            )
            assert HaltReason.DAILY_LOSS not in state.halt_reasons

    def test_should_flatten_on_daily_loss(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            ks = KillSwitch(KillSwitchConfig(daily_loss_limit_pct=2.0))
            ks.check(
                open_positions=3,
                daily_new_exposure_pct=0,
                daily_pnl_pct=-2.5,
                current_equity=975_000,
                peak_equity=1_000_000,
            )
            assert ks.should_flatten() is True


class TestDrawdown:
    def test_trips_at_limit(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            ks = KillSwitch(KillSwitchConfig(drawdown_halt_pct=6.0))
            state = ks.check(
                open_positions=0,
                daily_new_exposure_pct=0,
                daily_pnl_pct=0,
                current_equity=940_000,
                peak_equity=1_000_000,
            )
            assert HaltReason.DRAWDOWN in state.halt_reasons
            assert state.drawdown_pct == pytest.approx(6.0, abs=0.01)

    def test_ok_below_limit(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            ks = KillSwitch(KillSwitchConfig(drawdown_halt_pct=6.0))
            state = ks.check(
                open_positions=0,
                daily_new_exposure_pct=0,
                daily_pnl_pct=0,
                current_equity=950_000,
                peak_equity=1_000_000,
            )
            assert HaltReason.DRAWDOWN not in state.halt_reasons

    def test_should_flatten_on_drawdown(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            ks = KillSwitch(KillSwitchConfig(drawdown_halt_pct=6.0))
            ks.check(
                open_positions=3,
                daily_new_exposure_pct=0,
                daily_pnl_pct=-1.0,
                current_equity=930_000,
                peak_equity=1_000_000,
            )
            assert ks.should_flatten() is True


class TestManualHalt:
    def test_halt_blocks(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            ks = KillSwitch()
            ks.halt()
            state = ks.check(
                open_positions=0,
                daily_new_exposure_pct=0,
                daily_pnl_pct=0,
                current_equity=1_000_000,
                peak_equity=1_000_000,
            )
            assert state.halted is True
            assert HaltReason.MANUAL_HALT in state.halt_reasons

    def test_resume_clears_manual(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            ks = KillSwitch()
            ks.halt()
            ks.resume()
            state = ks.check(
                open_positions=0,
                daily_new_exposure_pct=0,
                daily_pnl_pct=0,
                current_equity=1_000_000,
                peak_equity=1_000_000,
            )
            assert state.halted is False

    def test_halt_idempotent(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            ks = KillSwitch()
            ks.halt()
            ks.halt()
            assert ks.state.halt_reasons.count(HaltReason.MANUAL_HALT) == 1


class TestMultipleReasons:
    def test_multiple_limits_trip_together(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            ks = KillSwitch(KillSwitchConfig(max_positions=5, daily_loss_limit_pct=2.0))
            state = ks.check(
                open_positions=5,
                daily_new_exposure_pct=0,
                daily_pnl_pct=-3.0,
                current_equity=970_000,
                peak_equity=1_000_000,
            )
            assert HaltReason.MAX_POSITIONS in state.halt_reasons
            assert HaltReason.DAILY_LOSS in state.halt_reasons
            assert len(state.halt_reasons) == 2


class TestShouldFlatten:
    def test_no_flatten_for_position_limit(self) -> None:
        with patch.dict(os.environ, {"EXECUTION_ENABLED": "1"}):
            ks = KillSwitch(KillSwitchConfig(max_positions=5))
            ks.check(
                open_positions=5,
                daily_new_exposure_pct=0,
                daily_pnl_pct=0,
                current_equity=1_000_000,
                peak_equity=1_000_000,
            )
            assert ks.should_flatten() is False

    def test_no_flatten_when_not_halted(self) -> None:
        ks = KillSwitch()
        assert ks.should_flatten() is False
