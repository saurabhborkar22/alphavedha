# Backtesting — AlphaVedha

## Responsibility
Strategy backtesting with realistic Indian market costs, and model validation via CPCV.

## CRITICAL RULES — Backtesting Sins to Avoid
1. **No look-ahead bias** — features at time T use only data <= T
2. **No survivorship bias** — include delisted stocks in historical universe
3. **No random splits** — ONLY temporal splits with purge + embargo
4. **Realistic costs** — always include STT, brokerage, slippage, impact cost
5. **No overfitting** — use CPCV, not a single train/test split

## Indian Market Costs (costs.py)

```python
COSTS = {
    "stt_delivery": 0.001,        # 0.1% on buy + sell
    "stt_intraday": 0.00025,      # 0.025% on sell only
    "stt_fno": 0.000125,          # 0.0125% on sell
    "brokerage_flat": 20,         # Rs 20 per order (discount broker)
    "exchange_txn": 0.0000345,    # NSE transaction charge
    "gst": 0.18,                  # 18% GST on brokerage + exchange charges
    "sebi_turnover": 0.000001,    # Rs 10 per crore
    "stamp_duty": 0.00015,        # 0.015% on buy side
    "slippage_large_cap": 0.001,  # 0.1% estimated slippage
    "slippage_mid_cap": 0.003,    # 0.3% estimated slippage
    "slippage_small_cap": 0.005,  # 0.5% estimated slippage
}
```

## CPCV Validation (validation.py)
- N=6 segments, k=2 test segments → 15 combinatorial paths
- Each path: train on 4 segments, test on 2 segments
- Purge: 20 trading days gap between train and test
- Embargo: additional 20 days after test before next train window
- Acceptance criteria: median Sharpe > 0.8, worst-case Sharpe > 0.3

## Metrics (metrics.py)
- Sharpe ratio (annualized, risk-free = India 10Y G-Sec yield)
- Sortino ratio
- Maximum drawdown (% and duration in days)
- Calmar ratio (annualized return / max drawdown)
- Win rate (% of profitable trades)
- Profit factor (gross profits / gross losses)
- Average trade return (after costs)
- Alpha vs Nifty 50 buy-and-hold

## Backtesting Engine (engine.py)
- Built on VectorBT for speed
- Supports: single stock, portfolio, and universe-level backtests
- Outputs: equity curve, trade log, metrics summary, regime-conditional metrics
- Always run with AND without costs to see the cost drag
