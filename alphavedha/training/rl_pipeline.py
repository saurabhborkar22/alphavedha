"""RL training pipeline — train PPO agent on historical market data.

Uses walk-forward approach: train on earlier data, evaluate on later data.
The agent learns portfolio weights that maximize risk-adjusted returns
after transaction costs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import structlog

from alphavedha.models.rl_agent import PPOAgent, PPOConfig, RolloutBuffer
from alphavedha.models.trading_env import EnvConfig, TradingEnvironment

logger = structlog.get_logger(__name__)


@dataclass
class RLTrainingResult:
    n_episodes: int
    final_portfolio_value: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    avg_episode_reward: float
    artifact_path: str | None


def train_rl_agent(
    feature_df: pd.DataFrame,
    price_df: pd.DataFrame,
    symbols: list[str],
    regime_labels: pd.Series | None = None,
    n_episodes: int = 100,
    train_frac: float = 0.8,
    artifact_dir: str = "models/artifacts/rl_ppo",
) -> RLTrainingResult:
    """Train PPO agent on historical data.

    Args:
        feature_df: Feature DataFrame with DatetimeIndex.
        price_df: Price DataFrame with columns per symbol.
        symbols: List of stock symbols.
        regime_labels: Optional regime labels aligned with feature_df.
        n_episodes: Number of training episodes.
        train_frac: Fraction of data for training.
        artifact_dir: Where to save trained agent.

    Returns:
        RLTrainingResult with training metrics.
    """
    split_idx = int(len(feature_df) * train_frac)
    train_features = feature_df.iloc[:split_idx]
    train_prices = price_df.iloc[:split_idx]
    val_features = feature_df.iloc[split_idx:]
    val_prices = price_df.iloc[split_idx:]

    train_regimes = regime_labels.iloc[:split_idx] if regime_labels is not None else None
    val_regimes = regime_labels.iloc[split_idx:] if regime_labels is not None else None

    env_config = EnvConfig()
    train_env = TradingEnvironment(
        train_features,
        train_prices,
        symbols,
        train_regimes,
        env_config,
    )

    ppo_config = PPOConfig()
    agent = PPOAgent(train_env.observation_size, train_env.action_size, ppo_config)

    episode_rewards: list[float] = []

    for ep in range(n_episodes):
        obs = train_env.reset()
        buffer = RolloutBuffer.empty()
        total_reward = 0.0
        done = False

        while not done:
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, done, info = train_env.step(action)

            buffer.add(obs, action, log_prob, reward, value, done)
            total_reward += reward
            obs = next_obs

        metrics = agent.update(buffer)
        episode_rewards.append(total_reward)

        if (ep + 1) % 10 == 0:
            logger.info(
                "rl_episode",
                episode=ep + 1,
                reward=round(total_reward, 2),
                policy_loss=round(metrics["policy_loss"], 4),
                value_loss=round(metrics["value_loss"], 4),
            )

    val_env = TradingEnvironment(
        val_features,
        val_prices,
        symbols,
        val_regimes,
        env_config,
    )
    obs = val_env.reset()
    done = False
    val_returns: list[float] = []

    while not done:
        action, _, _ = agent.select_action(obs)
        obs, reward, done, info = val_env.step(action)
        val_returns.append(info.get("daily_return", 0.0))

    final_pv = val_env._state.portfolio_value
    total_ret = (final_pv / env_config.initial_capital) - 1

    returns_arr = np.array(val_returns)
    if len(returns_arr) >= 2 and returns_arr.std() > 0:
        sharpe = float(returns_arr.mean() / returns_arr.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    equity = np.cumprod(1 + returns_arr)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / np.where(peak > 0, peak, 1.0)
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    save_path = Path(artifact_dir)
    agent.save(save_path)

    avg_reward = float(np.mean(episode_rewards)) if episode_rewards else 0.0

    logger.info(
        "rl_training_complete",
        episodes=n_episodes,
        val_return=round(total_ret, 4),
        val_sharpe=round(sharpe, 4),
        val_max_dd=round(max_dd, 4),
    )

    return RLTrainingResult(
        n_episodes=n_episodes,
        final_portfolio_value=final_pv,
        total_return=total_ret,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        avg_episode_reward=avg_reward,
        artifact_path=str(save_path),
    )
