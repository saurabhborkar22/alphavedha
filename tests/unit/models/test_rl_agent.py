"""Tests for RL agent and trading environment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphavedha.models.rl_agent import PPOAgent, PPOConfig, RolloutBuffer
from alphavedha.models.trading_env import EnvConfig, TradingEnvironment


def _make_env(n_days: int = 200, n_stocks: int = 3) -> TradingEnvironment:
    """Create a simple trading environment for testing."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-01", periods=n_days)

    symbols = [f"STOCK{i}.NS" for i in range(n_stocks)]
    features = pd.DataFrame(
        rng.standard_normal((n_days, n_stocks * 5)),
        index=dates,
    )
    prices = pd.DataFrame(
        {sym: 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n_days))) for sym in symbols},
        index=dates,
    )

    return TradingEnvironment(features, prices, symbols, config=EnvConfig())


class TestTradingEnvironment:
    def test_reset_returns_observation(self) -> None:
        env = _make_env()
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape[0] == env.observation_size

    def test_step_returns_tuple(self) -> None:
        env = _make_env()
        env.reset()
        action = np.zeros(env.action_size)
        obs, reward, done, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_episode_terminates(self) -> None:
        env = _make_env(n_days=20)
        env.reset()
        done = False
        steps = 0
        while not done:
            action = np.zeros(env.action_size)
            _, _, done, _ = env.step(action)
            steps += 1
        assert steps == 19  # n_days - 1

    def test_portfolio_value_in_info(self) -> None:
        env = _make_env()
        env.reset()
        action = np.ones(env.action_size) * 0.5
        _, _, _, info = env.step(action)
        assert "portfolio_value" in info
        assert info["portfolio_value"] > 0

    def test_actions_clipped(self) -> None:
        env = _make_env()
        env.reset()
        action = np.ones(env.action_size) * 5.0
        obs, _, _, _ = env.step(action)
        assert obs is not None


class TestPPOAgent:
    def test_select_action_shape(self) -> None:
        agent = PPOAgent(obs_size=20, action_size=3)
        obs = np.random.randn(20).astype(np.float32)
        action, log_prob, value = agent.select_action(obs)
        assert action.shape == (3,)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_update_returns_metrics(self) -> None:
        agent = PPOAgent(obs_size=10, action_size=2, config=PPOConfig(ppo_epochs=1))
        buffer = RolloutBuffer.empty()
        for _ in range(20):
            obs = np.random.randn(10).astype(np.float32)
            action, lp, val = agent.select_action(obs)
            buffer.add(obs, action, lp, np.random.randn(), val, False)
        buffer.dones[-1] = True

        metrics = agent.update(buffer)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics

    def test_save_and_load(self, tmp_path) -> None:
        agent = PPOAgent(obs_size=10, action_size=2)
        obs = np.random.randn(10).astype(np.float32)
        action1, _, _ = agent.select_action(obs)

        agent.save(tmp_path / "rl_test")
        loaded = PPOAgent.load(tmp_path / "rl_test")
        action2, _, _ = loaded.select_action(obs)

        assert action1.shape == action2.shape


class TestRolloutBuffer:
    def test_compute_returns_shape(self) -> None:
        buffer = RolloutBuffer.empty()
        for i in range(10):
            buffer.add(
                np.zeros(5), np.zeros(2), 0.0,
                float(i), float(i * 0.1), i == 9,
            )
        advantages, returns = buffer.compute_returns(gamma=0.99, gae_lambda=0.95)
        assert len(advantages) == 10
        assert len(returns) == 10

    def test_advantages_zero_mean_after_normalization(self) -> None:
        buffer = RolloutBuffer.empty()
        for i in range(50):
            buffer.add(
                np.zeros(5), np.zeros(2), 0.0,
                float(np.random.randn()), float(np.random.randn() * 0.1), i == 49,
            )
        advantages, _ = buffer.compute_returns(gamma=0.99, gae_lambda=0.95)
        normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        assert abs(normalized.mean()) < 0.1
