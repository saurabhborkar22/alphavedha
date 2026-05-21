"""Trading environment for RL portfolio optimization.

Simulates Indian market trading with realistic costs, slippage,
and regime awareness. Compatible with standard Gym/Gymnasium interface
but implemented standalone (no gym dependency required).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class EnvConfig:
    """Trading environment configuration."""

    initial_capital: float = 1_000_000.0
    transaction_cost_pct: float = 0.003
    max_position_pct: float = 0.10
    reward_sharpe_bonus: float = 0.5
    drawdown_penalty: float = 1.0
    max_drawdown_threshold: float = 0.10


@dataclass
class EnvState:
    """Current environment state."""

    step: int = 0
    portfolio_value: float = 1_000_000.0
    cash: float = 1_000_000.0
    positions: dict[str, float] = field(default_factory=dict)
    returns_history: list[float] = field(default_factory=list)
    peak_value: float = 1_000_000.0


class TradingEnvironment:
    """Multi-asset trading environment for RL.

    State space: [features (per stock), current_positions, portfolio_value_norm,
                  regime_one_hot, drawdown_pct]
    Action space: position weights per stock [-1, 1]
    Reward: daily PnL - costs + Sharpe bonus - drawdown penalty
    """

    def __init__(
        self,
        feature_df: pd.DataFrame,
        price_df: pd.DataFrame,
        symbols: list[str],
        regime_labels: pd.Series | None = None,
        config: EnvConfig | None = None,
    ) -> None:
        self._features = feature_df
        self._prices = price_df
        self._symbols = symbols
        self._regimes = regime_labels
        self._config = config or EnvConfig()

        self._n_stocks = len(symbols)
        self._n_features = feature_df.shape[1] // self._n_stocks if self._n_stocks > 0 else 0
        self._max_steps = len(feature_df) - 1
        self._dates = feature_df.index

        n_regime = 4
        self._obs_size = (
            feature_df.shape[1]
            + self._n_stocks  # current positions
            + 1  # portfolio value normalized
            + n_regime  # regime one-hot
            + 1  # current drawdown
        )

        self._state = EnvState(
            portfolio_value=self._config.initial_capital,
            cash=self._config.initial_capital,
            peak_value=self._config.initial_capital,
        )

    @property
    def observation_size(self) -> int:
        return self._obs_size

    @property
    def action_size(self) -> int:
        return self._n_stocks

    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self._state = EnvState(
            portfolio_value=self._config.initial_capital,
            cash=self._config.initial_capital,
            peak_value=self._config.initial_capital,
        )
        return self._get_observation()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """Execute one step.

        Args:
            action: Array of position weights per stock, each in [-1, 1].

        Returns:
            (observation, reward, done, info)
        """
        if self._state.step >= self._max_steps:
            return self._get_observation(), 0.0, True, {}

        action = np.clip(action, -1.0, 1.0)

        current_prices = self._get_prices(self._state.step)
        next_prices = self._get_prices(self._state.step + 1)

        target_weights = action * self._config.max_position_pct

        current_weights = np.zeros(self._n_stocks)
        for i, sym in enumerate(self._symbols):
            pos_value = self._state.positions.get(sym, 0.0) * current_prices[i]
            current_weights[i] = pos_value / max(self._state.portfolio_value, 1.0)

        weight_change = np.abs(target_weights - current_weights)
        turnover = float(np.sum(weight_change))
        cost = turnover * self._config.transaction_cost_pct * self._state.portfolio_value

        for i, sym in enumerate(self._symbols):
            target_value = target_weights[i] * self._state.portfolio_value
            if current_prices[i] > 0:
                self._state.positions[sym] = target_value / current_prices[i]

        self._state.cash = self._state.portfolio_value - sum(
            self._state.positions.get(sym, 0) * current_prices[i]
            for i, sym in enumerate(self._symbols)
        )

        new_portfolio = self._state.cash
        for i, sym in enumerate(self._symbols):
            new_portfolio += self._state.positions.get(sym, 0) * next_prices[i]
        new_portfolio -= cost

        daily_return = (new_portfolio / max(self._state.portfolio_value, 1.0)) - 1.0
        self._state.returns_history.append(daily_return)

        self._state.portfolio_value = new_portfolio
        self._state.peak_value = max(self._state.peak_value, new_portfolio)
        drawdown = (self._state.peak_value - new_portfolio) / max(self._state.peak_value, 1.0)

        reward = daily_return * 100

        if len(self._state.returns_history) >= 20:
            recent = np.array(self._state.returns_history[-20:])
            if recent.std() > 0:
                rolling_sharpe = recent.mean() / recent.std() * np.sqrt(252)
                if rolling_sharpe > 1.0:
                    reward += self._config.reward_sharpe_bonus

        if drawdown > self._config.max_drawdown_threshold:
            reward -= (
                self._config.drawdown_penalty
                * (drawdown - self._config.max_drawdown_threshold)
                * 100
            )

        self._state.step += 1
        done = self._state.step >= self._max_steps

        info = {
            "portfolio_value": self._state.portfolio_value,
            "daily_return": daily_return,
            "drawdown": drawdown,
            "turnover": turnover,
            "cost": cost,
        }

        return self._get_observation(), reward, done, info

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        features = self._features.iloc[self._state.step].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)

        positions = np.array(
            [self._state.positions.get(sym, 0.0) for sym in self._symbols], dtype=np.float32
        )
        prices = self._get_prices(self._state.step)
        pos_values = positions * prices
        pos_weights = pos_values / max(self._state.portfolio_value, 1.0)

        pv_norm = np.array(
            [self._state.portfolio_value / self._config.initial_capital], dtype=np.float32
        )

        regime_oh = np.zeros(4, dtype=np.float32)
        if self._regimes is not None and self._state.step < len(self._regimes):
            r = self._regimes.iloc[self._state.step]
            regime_map = {"bull": 0, "bear": 1, "sideways": 2, "high_volatility": 3}
            idx = regime_map.get(str(r), 2)
            regime_oh[idx] = 1.0
        else:
            regime_oh[2] = 1.0

        dd = (self._state.peak_value - self._state.portfolio_value) / max(
            self._state.peak_value, 1.0
        )
        dd_arr = np.array([dd], dtype=np.float32)

        return np.concatenate([features, pos_weights, pv_norm, regime_oh, dd_arr])

    def _get_prices(self, step: int) -> np.ndarray:
        """Get closing prices for all stocks at a given step."""
        prices = np.zeros(self._n_stocks, dtype=np.float32)
        row = self._prices.iloc[step]
        for i, sym in enumerate(self._symbols):
            if sym in self._prices.columns:
                prices[i] = float(row[sym])
            elif "close" in self._prices.columns:
                prices[i] = float(row["close"])
        return prices
