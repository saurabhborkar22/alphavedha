# Risk Management — AlphaVedha

## Responsibility
Position sizing, portfolio constraints, and circuit breakers. Risk management is NOT optional — every prediction MUST pass through this layer before becoming an actionable signal.

## Modules

### position_sizing.py — Fractional Kelly Criterion
- kelly_fraction = (win_prob × avg_win - loss_prob × avg_loss) / avg_win
- ALWAYS use half-Kelly (multiply by 0.5) for safety margin
- Cap maximum position at 10% of portfolio regardless of Kelly output
- If meta_confidence < 0.55, position size = 0 (no trade)

### portfolio.py — Constraints
- Max single stock: 10% of portfolio value
- Max sector exposure: 25% of portfolio value
- Max correlation between any two holdings: 0.7 (Pearson, 60d rolling)
- Minimum holding period: 3 trading days (avoid overtrading)
- Liquidity filter: exclude stocks with avg daily turnover < Rs 5 crore

### circuit_breaker.py — Drawdown Protection
- 10% portfolio drawdown → reduce all positions by 50%, alert
- 15% portfolio drawdown → halt new entries, alert urgently
- 20% portfolio drawdown → close all positions, full stop
- Drawdown measured from peak portfolio value (high-water mark)
- Reset after portfolio recovers to 95% of previous peak

### liquidity_filter.py — Tradability
- Compute 20-day average daily turnover (price × volume)
- Exclude stocks below Rs 5 crore threshold
- For Smallcap 250, this filter removes ~30-40% of universe (expected)
- Log filtered stocks for transparency

## Rules
- Risk checks run AFTER prediction, BEFORE signal output
- Never skip risk checks — even in backtesting
- All thresholds are configurable via configs/risk.yaml
- Log every risk-adjusted decision (original vs adjusted position)
