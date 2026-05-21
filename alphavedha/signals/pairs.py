"""Pairs trading engine — market-neutral spread trading.

Tracks spread z-scores between cointegrated pairs and generates
entry/exit signals. Works in all market regimes since it's market-neutral.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import structlog

from alphavedha.signals.pairs_universe import compute_hedge_ratio

logger = structlog.get_logger(__name__)


@dataclass
class PairSignal:
    """A single pairs trading signal."""

    date: date
    symbol_long: str
    symbol_short: str
    spread_zscore: float
    signal_type: str  # "entry_long_a", "entry_long_b", "exit", "stop_loss"
    hedge_ratio: float
    confidence: float


@dataclass
class PairPosition:
    """Active pair position."""

    symbol_long: str
    symbol_short: str
    entry_date: date
    entry_zscore: float
    hedge_ratio: float
    entry_spread: float


@dataclass
class PairTradeResult:
    """Completed pair trade."""

    symbol_long: str
    symbol_short: str
    entry_date: date
    exit_date: date
    entry_zscore: float
    exit_zscore: float
    spread_return: float
    holding_days: int
    exit_reason: str


@dataclass
class PairsBacktestResult:
    """Results from pairs strategy backtest."""

    trades: list[PairTradeResult]
    total_return: float
    sharpe_ratio: float
    win_rate: float
    n_trades: int
    avg_holding_days: float
    max_drawdown: float
    pair_signals: pd.DataFrame


class PairsTrader:
    """Stateful pairs trading engine.

    Computes spread z-scores and generates entry/exit signals
    based on mean-reversion of cointegrated pairs.
    """

    def __init__(
        self,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        stop_loss_zscore: float = 3.5,
        lookback: int = 60,
        min_lookback: int = 30,
    ) -> None:
        self._entry_z = entry_zscore
        self._exit_z = exit_zscore
        self._stop_z = stop_loss_zscore
        self._lookback = lookback
        self._min_lookback = min_lookback

    def compute_spread(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        hedge_ratio: float | None = None,
    ) -> pd.DataFrame:
        """Compute spread and z-score between two price series.

        Returns DataFrame with columns: spread, spread_mean, spread_std, zscore.
        """
        common_idx = prices_a.dropna().index.intersection(prices_b.dropna().index)
        a = prices_a.loc[common_idx]
        b = prices_b.loc[common_idx]

        if hedge_ratio is None:
            hedge_ratio = compute_hedge_ratio(a, b)

        spread = a - hedge_ratio * b
        spread_mean = spread.rolling(self._lookback, min_periods=self._min_lookback).mean()
        spread_std = spread.rolling(self._lookback, min_periods=self._min_lookback).std()

        zscore = (spread - spread_mean) / spread_std.replace(0, np.nan)

        return pd.DataFrame(
            {
                "spread": spread,
                "spread_mean": spread_mean,
                "spread_std": spread_std,
                "zscore": zscore,
            },
            index=common_idx,
        )

    def generate_signals(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        symbol_a: str,
        symbol_b: str,
        hedge_ratio: float | None = None,
    ) -> list[PairSignal]:
        """Generate pairs trading signals for the full history."""
        spread_df = self.compute_spread(prices_a, prices_b, hedge_ratio)
        if hedge_ratio is None:
            common = prices_a.dropna().index.intersection(prices_b.dropna().index)
            hedge_ratio = compute_hedge_ratio(
                prices_a.loc[common],
                prices_b.loc[common],
            )

        signals: list[PairSignal] = []
        in_position = False

        for i in range(len(spread_df)):
            z = spread_df["zscore"].iloc[i]
            if pd.isna(z):
                continue

            dt = spread_df.index[i]
            dt_date = dt.date() if hasattr(dt, "date") else dt

            if not in_position:
                if z > self._entry_z:
                    signals.append(
                        PairSignal(
                            date=dt_date,
                            symbol_long=symbol_b,
                            symbol_short=symbol_a,
                            spread_zscore=float(z),
                            signal_type="entry_long_b",
                            hedge_ratio=hedge_ratio,
                            confidence=min(abs(z) / self._stop_z, 1.0),
                        )
                    )
                    in_position = True
                elif z < -self._entry_z:
                    signals.append(
                        PairSignal(
                            date=dt_date,
                            symbol_long=symbol_a,
                            symbol_short=symbol_b,
                            spread_zscore=float(z),
                            signal_type="entry_long_a",
                            hedge_ratio=hedge_ratio,
                            confidence=min(abs(z) / self._stop_z, 1.0),
                        )
                    )
                    in_position = True
            else:
                if abs(z) < self._exit_z:
                    signals.append(
                        PairSignal(
                            date=dt_date,
                            symbol_long="",
                            symbol_short="",
                            spread_zscore=float(z),
                            signal_type="exit",
                            hedge_ratio=hedge_ratio,
                            confidence=1.0,
                        )
                    )
                    in_position = False
                elif abs(z) > self._stop_z:
                    signals.append(
                        PairSignal(
                            date=dt_date,
                            symbol_long="",
                            symbol_short="",
                            spread_zscore=float(z),
                            signal_type="stop_loss",
                            hedge_ratio=hedge_ratio,
                            confidence=1.0,
                        )
                    )
                    in_position = False

        return signals

    def backtest_pair(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        symbol_a: str,
        symbol_b: str,
        cost_pct: float = 0.003,
    ) -> PairsBacktestResult:
        """Backtest pairs strategy on a single pair.

        Args:
            prices_a: Price series for stock A.
            prices_b: Price series for stock B.
            symbol_a: Symbol name for A.
            symbol_b: Symbol name for B.
            cost_pct: Round-trip transaction cost as fraction.

        Returns:
            PairsBacktestResult with trade-level details and aggregate metrics.
        """
        spread_df = self.compute_spread(prices_a, prices_b)
        common = prices_a.dropna().index.intersection(prices_b.dropna().index)
        hedge_ratio = compute_hedge_ratio(prices_a.loc[common], prices_b.loc[common])

        trades: list[PairTradeResult] = []
        position: PairPosition | None = None

        for i in range(len(spread_df)):
            z = spread_df["zscore"].iloc[i]
            if pd.isna(z):
                continue

            dt = spread_df.index[i]
            dt_date = dt.date() if hasattr(dt, "date") else dt

            if position is None:
                if z > self._entry_z:
                    position = PairPosition(
                        symbol_long=symbol_b,
                        symbol_short=symbol_a,
                        entry_date=dt_date,
                        entry_zscore=float(z),
                        hedge_ratio=hedge_ratio,
                        entry_spread=float(spread_df["spread"].iloc[i]),
                    )
                elif z < -self._entry_z:
                    position = PairPosition(
                        symbol_long=symbol_a,
                        symbol_short=symbol_b,
                        entry_date=dt_date,
                        entry_zscore=float(z),
                        hedge_ratio=hedge_ratio,
                        entry_spread=float(spread_df["spread"].iloc[i]),
                    )
            else:
                exit_reason = ""
                if abs(z) < self._exit_z:
                    exit_reason = "mean_reversion"
                elif abs(z) > self._stop_z:
                    exit_reason = "stop_loss"

                if exit_reason:
                    current_spread = float(spread_df["spread"].iloc[i])
                    spread_change = current_spread - position.entry_spread

                    if position.symbol_long == symbol_a:
                        spread_return = (
                            spread_change / abs(position.entry_spread)
                            if position.entry_spread != 0
                            else 0.0
                        )
                    else:
                        spread_return = (
                            -spread_change / abs(position.entry_spread)
                            if position.entry_spread != 0
                            else 0.0
                        )

                    net_return = spread_return - cost_pct

                    trades.append(
                        PairTradeResult(
                            symbol_long=position.symbol_long,
                            symbol_short=position.symbol_short,
                            entry_date=position.entry_date,
                            exit_date=dt_date,
                            entry_zscore=position.entry_zscore,
                            exit_zscore=float(z),
                            spread_return=net_return,
                            holding_days=(dt_date - position.entry_date).days,
                            exit_reason=exit_reason,
                        )
                    )
                    position = None

        total_ret = sum(t.spread_return for t in trades) if trades else 0.0
        wins = [t for t in trades if t.spread_return > 0]
        win_rate = len(wins) / len(trades) if trades else 0.0
        avg_hold = np.mean([t.holding_days for t in trades]) if trades else 0.0

        returns = pd.Series([t.spread_return for t in trades])
        if len(returns) >= 2 and returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * np.sqrt(252 / max(avg_hold, 1)))
        else:
            sharpe = 0.0

        equity = (1 + returns).cumprod() if len(returns) > 0 else pd.Series(dtype=float)
        if len(equity) > 0:
            peak = equity.cummax()
            dd = (equity - peak) / peak
            max_dd = float(dd.min())
        else:
            max_dd = 0.0

        signal_records = []
        for t in trades:
            signal_records.append(
                {
                    "entry_date": t.entry_date,
                    "exit_date": t.exit_date,
                    "symbol_long": t.symbol_long,
                    "symbol_short": t.symbol_short,
                    "spread_return": t.spread_return,
                    "exit_reason": t.exit_reason,
                }
            )

        logger.info(
            "pairs_backtest_complete",
            pair=f"{symbol_a}/{symbol_b}",
            n_trades=len(trades),
            total_return=round(total_ret, 4),
            sharpe=round(sharpe, 4),
            win_rate=round(win_rate, 4),
        )

        return PairsBacktestResult(
            trades=trades,
            total_return=total_ret,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            n_trades=len(trades),
            avg_holding_days=float(avg_hold),
            max_drawdown=max_dd,
            pair_signals=pd.DataFrame(signal_records) if signal_records else pd.DataFrame(),
        )
