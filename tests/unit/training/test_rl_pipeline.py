"""Tests for RL training pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from alphavedha.training.rl_pipeline import (
    RLTrainingResult,
    WalkForwardResult,
    train_rl_agent,
    walk_forward_rl,
)


def _make_training_data(
    n_steps: int = 100,
    n_stocks: int = 2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
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

    return features, prices, symbols


class TestRLTrainingResult:
    def test_fields(self) -> None:
        r = RLTrainingResult(
            n_episodes=10,
            final_portfolio_value=1_050_000.0,
            total_return=0.05,
            sharpe_ratio=1.2,
            max_drawdown=-0.03,
            avg_episode_reward=5.0,
            artifact_path="/tmp/test",
        )
        assert r.n_episodes == 10
        assert r.total_return == 0.05
        assert r.artifact_path == "/tmp/test"


class TestTrainRLAgent:
    def test_runs_and_returns_result(self, tmp_path) -> None:
        features, prices, symbols = _make_training_data(n_steps=60, n_stocks=2)

        mock_agent = MagicMock()
        mock_agent.select_action.return_value = (
            np.zeros(2),
            np.array([0.0]),
            np.array([0.0]),
        )
        mock_agent.update.return_value = {"policy_loss": 0.1, "value_loss": 0.05}

        with patch("alphavedha.training.rl_pipeline.PPOAgent", return_value=mock_agent):
            result = train_rl_agent(
                feature_df=features,
                price_df=prices,
                symbols=symbols,
                n_episodes=3,
                artifact_dir=str(tmp_path / "rl"),
            )
        assert isinstance(result, RLTrainingResult)
        assert result.n_episodes == 3
        assert isinstance(result.final_portfolio_value, float)
        assert isinstance(result.sharpe_ratio, float)
        mock_agent.save.assert_called_once()

    def test_with_regime_labels(self, tmp_path) -> None:
        features, prices, symbols = _make_training_data(n_steps=60, n_stocks=2)
        regimes = pd.Series(
            np.random.default_rng(42).choice(["bull", "bear", "sideways"], len(features)),
            index=features.index,
        )

        mock_agent = MagicMock()
        mock_agent.select_action.return_value = (
            np.zeros(2),
            np.array([0.0]),
            np.array([0.0]),
        )
        mock_agent.update.return_value = {"policy_loss": 0.1, "value_loss": 0.05}

        with patch("alphavedha.training.rl_pipeline.PPOAgent", return_value=mock_agent):
            result = train_rl_agent(
                feature_df=features,
                price_df=prices,
                symbols=symbols,
                regime_labels=regimes,
                n_episodes=2,
                artifact_dir=str(tmp_path / "rl_regime"),
            )
        assert isinstance(result, RLTrainingResult)

    def test_avg_episode_reward_populated(self, tmp_path) -> None:
        features, prices, symbols = _make_training_data(n_steps=60, n_stocks=1)

        mock_agent = MagicMock()
        mock_agent.select_action.return_value = (
            np.zeros(1),
            np.array([0.0]),
            np.array([0.0]),
        )
        mock_agent.update.return_value = {"policy_loss": 0.0, "value_loss": 0.0}

        with patch("alphavedha.training.rl_pipeline.PPOAgent", return_value=mock_agent):
            result = train_rl_agent(
                feature_df=features,
                price_df=prices,
                symbols=symbols,
                n_episodes=5,
                artifact_dir=str(tmp_path / "rl_reward"),
            )
        assert isinstance(result.avg_episode_reward, float)


class TestWalkForwardRL:
    def test_walk_forward_basic(self, tmp_path) -> None:
        features, prices, symbols = _make_training_data(n_steps=120, n_stocks=2)

        mock_agent = MagicMock()
        mock_agent.select_action.return_value = (
            np.zeros(2),
            np.array([0.0]),
            np.array([0.0]),
        )
        mock_agent.update.return_value = {"policy_loss": 0.1, "value_loss": 0.05}

        with patch("alphavedha.training.rl_pipeline.PPOAgent", return_value=mock_agent):
            result = walk_forward_rl(
                feature_df=features,
                price_df=prices,
                symbols=symbols,
                n_windows=2,
                train_frac=0.7,
                n_episodes=2,
                artifact_dir=str(tmp_path / "wf"),
            )

        assert isinstance(result, WalkForwardResult)
        assert result.n_windows == 2
        assert len(result.window_results) == 2

    def test_walk_forward_metrics(self, tmp_path) -> None:
        features, prices, symbols = _make_training_data(n_steps=120, n_stocks=2)

        mock_agent = MagicMock()
        mock_agent.select_action.return_value = (
            np.zeros(2),
            np.array([0.0]),
            np.array([0.0]),
        )
        mock_agent.update.return_value = {"policy_loss": 0.1, "value_loss": 0.05}

        with patch("alphavedha.training.rl_pipeline.PPOAgent", return_value=mock_agent):
            result = walk_forward_rl(
                feature_df=features,
                price_df=prices,
                symbols=symbols,
                n_windows=2,
                train_frac=0.7,
                n_episodes=2,
                artifact_dir=str(tmp_path / "wf_metrics"),
            )

        expected_avg = sum(w.sharpe_ratio for w in result.window_results) / len(
            result.window_results
        )
        assert result.avg_sharpe == pytest.approx(expected_avg)
