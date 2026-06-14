"""Tests for the regime-aware exposure overlay (prototype)."""

from __future__ import annotations

import pandas as pd

from alphavedha.prediction.engine import (
    RegimeOverlay,
    _load_regime_overlay,
    apply_regime_overlay,
)


def _market(mean_ret: float, n: int = 60) -> pd.DataFrame:
    return pd.DataFrame({"returns": [mean_ret] * n, "volatility": [0.01] * n})


def test_disabled_overlay_is_noop() -> None:
    assert apply_regime_overlay(None, 1.0, 1, True, _market(-0.01)) == (1.0, True, None)


def test_no_market_features_is_noop() -> None:
    assert apply_regime_overlay(RegimeOverlay(), 1.0, 1, True, None) == (1.0, True, None)


def test_uptrend_caps_kelly_only() -> None:
    ov = RegimeOverlay(kelly_cap=0.5)
    kelly, tradeable, warning = apply_regime_overlay(ov, 1.0, 1, True, _market(0.01))
    assert kelly == 0.5  # capped, not cut
    assert tradeable is True
    assert warning is None


def test_downtrend_suppresses_long_and_cuts_size() -> None:
    ov = RegimeOverlay(kelly_cap=0.5, downtrend_size_mult=0.3)
    kelly, tradeable, warning = apply_regime_overlay(ov, 1.0, 1, True, _market(-0.01))
    assert tradeable is False  # new long suppressed in a downtrend
    assert warning == "regime_overlay_long_suppressed_downtrend"
    assert kelly == 0.5 * 0.3


def test_downtrend_keeps_shorts_tradeable() -> None:
    ov = RegimeOverlay(kelly_cap=0.5, downtrend_size_mult=0.3)
    kelly, tradeable, warning = apply_regime_overlay(ov, 0.4, -1, True, _market(-0.01))
    assert tradeable is True  # shorts are fine when the market is falling
    assert warning is None
    assert kelly == 0.4 * 0.3  # min(0.4, 0.5) then downtrend cut


def test_overlay_only_restricts_never_enables() -> None:
    ov = RegimeOverlay()
    _, tradeable, _ = apply_regime_overlay(ov, 1.0, 1, False, _market(0.01))
    assert tradeable is False  # an already-rejected trade stays rejected


def test_load_overlay_from_env(monkeypatch) -> None:
    monkeypatch.delenv("ALPHAVEDHA_REGIME_OVERLAY", raising=False)
    assert _load_regime_overlay() is None

    monkeypatch.setenv("ALPHAVEDHA_REGIME_OVERLAY", "1")
    ov = _load_regime_overlay()
    assert ov is not None
    assert ov.kelly_cap == 0.5
    assert ov.suppress_longs_in_downtrend is True

    monkeypatch.setenv("ALPHAVEDHA_REGIME_OVERLAY_KELLY_CAP", "0.25")
    assert _load_regime_overlay().kelly_cap == 0.25
