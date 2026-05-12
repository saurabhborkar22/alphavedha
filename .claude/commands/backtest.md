# Backtest Strategy

Run backtesting with realistic Indian market costs.

## Usage
- `/backtest TCS.NS` — backtest predictions for one stock
- `/backtest portfolio` — backtest the full portfolio strategy
- `/backtest compare` — compare model versions

## Steps
1. Activate venv: `source .venv/bin/activate`
2. Run backtest with full cost model:
   ```bash
   python -m alphavedha.backtest.engine --mode <mode> --start 2020-01-01 --end 2026-01-01
   ```
3. Display results:
   - Equity curve summary (start value, end value, total return)
   - Key metrics: Sharpe, Sortino, Max Drawdown, Calmar, Win Rate
   - Alpha vs Nifty 50 buy-and-hold
   - Regime-conditional metrics (how did it perform in bull vs bear?)
   - Cost drag (return with costs vs without costs)
4. Flag any warning signs:
   - Sharpe < 0.5 → "Below target"
   - Max drawdown > 25% → "High drawdown risk"
   - Win rate < 45% → "Low win rate"

## Arguments
$ARGUMENTS
