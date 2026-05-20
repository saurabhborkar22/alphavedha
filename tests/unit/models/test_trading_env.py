"""Tests for RL trading environment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphavedha.models.trading_env import EnvConfig, EnvState, TradingEnvironment


def _make_env(n_steps: int = 50, n_stocks: int = 2, seed: int = 42) -> TradingEnvironment:
    """Create a trading environment with synthetic data."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_steps)
    symbols = [f"STOCK_{i}" for i in range(n_stocks)]

    features = pd.DataFrame(
        rng.standard_normal((n_steps, n_stocks * 5)),
        index=dates,
    )

    base = 100 * np.ones((n_steps, n_stocks))
    for i in range(1, n_steps):
        base[i] = base[i - 1] * (1 + rng.normal(0.001, 0.02, n_stocks))
    prices = pd.DataFrame(base, index=dates, columns=symbols)

    return TradingEnvironment(features, prices, symbols)


class TestEnvConfig:
    def test_defaults(self) -> None:
        cfg = EnvConfig()
        assert cfg.initial_capital == 1_000_000.0
        assert cfg.transaction_cost_pct == 0.003
        assert cfg.max_position_pct == 0.10

    def test_custom_values(self) -> None:
        cfg = EnvConfig(initial_capital=500_000, max_position_pct=0.05)
        assert cfg.initial_capital == 500_000
        assert cfg.max_position_pct == 0.05


class TestEnvState:
    def test_defaults(self) -> None:
        state = EnvState()
        assert state.step == 0
        assert state.portfolio_value == 1_000_000.0
        assert state.positions == {}
        assert state.returns_history == []


class TestTradingEnvironment:
    def test_reset_returns_observation(self) -> None:
        env = _make_env()
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (env.observation_size,)

    def test_observation_size_matches(self) -> None:
        env = _make_env(n_stocks=3)
        obs = env.reset()
        assert len(obs) == env.observation_size

    def test_action_size_matches_stocks(self) -> None:
        env = _make_env(n_stocks=4)
        assert env.action_size == 4

    def test_step_returns_tuple(self) -> None:
        env = _make_env()
        env.reset()
        action = np.zeros(env.action_size)
        obs, reward, done, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_zero_action_preserves_capital_approximately(self) -> None:
        env = _make_env()
        env.reset()
        action = np.zeros(env.action_size)
        _, _, _, info = env.step(action)
        assert abs(info["portfolio_value"] - 1_000_000.0) < 10_000

    def test_done_after_max_steps(self) -> None:
        env = _make_env(n_steps=10)
        env.reset()
        action = np.zeros(env.action_size)
        done = False
        steps = 0
        while not done:
            _, _, done, _ = env.step(action)
            steps += 1
        assert steps == 9  # n_steps - 1

    def test_actions_clipped(self) -> None:
        env = _make_env()
        env.reset()
        action = np.array([5.0, -5.0])
        _obs, _reward, _done, info = env.step(action)
        assert "portfolio_value" in info

    def test_portfolio_value_tracks_in_info(self) -> None:
        env = _make_env()
        env.reset()
        action = np.array([0.5, 0.3])
        for _ in range(5):
            _, _, _, info = env.step(action)
        assert info["portfolio_value"] > 0

    def test_drawdown_in_info(self) -> None:
        env = _make_env()
        env.reset()
        action = np.array([0.5, -0.5])
        _, _, _, info = env.step(action)
        assert "drawdown" in info
        assert info["drawdown"] >= 0

    def test_reset_clears_state(self) -> None:
        env = _make_env()
        env.reset()
        action = np.array([1.0, 1.0])
        for _ in range(5):
            env.step(action)
        obs = env.reset()
        assert obs.shape == (env.observation_size,)
        _, _, _, info = env.step(np.zeros(env.action_size))
        assert abs(info["portfolio_value"] - 1_000_000.0) < 10_000

    def test_with_regime_labels(self) -> None:
        n_steps = 50
        dates = pd.bdate_range("2024-01-01", periods=n_steps)
        rng = np.random.default_rng(42)

        features = pd.DataFrame(rng.standard_normal((n_steps, 10)), index=dates)
        prices = pd.DataFrame(
            {"A": 100 + np.cumsum(rng.normal(0, 1, n_steps))},
            index=dates,
        )
        regimes = pd.Series(
            rng.choice(["bull", "bear", "sideways", "high_volatility"], n_steps),
            index=dates,
        )

        env = TradingEnvironment(features, prices, ["A"], regime_labels=regimes)
        obs = env.reset()
        assert len(obs) == env.observation_size

    def test_cost_in_info(self) -> None:
        env = _make_env()
        env.reset()
        action = np.array([0.8, -0.8])
        _, _, _, info = env.step(action)
        assert "cost" in info
        assert info["cost"] >= 0

    def test_turnover_in_info(self) -> None:
        env = _make_env()
        env.reset()
        action = np.array([0.5, 0.5])
        _, _, _, info = env.step(action)
        assert "turnover" in info
        assert info["turnover"] >= 0
