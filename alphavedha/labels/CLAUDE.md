# Labels — AlphaVedha

## Responsibility
Generate training labels using the Triple Barrier Method and Meta-Labeling. This is the MOST critical module — wrong labels mean wrong models.

## Triple Barrier Method (triple_barrier.py)

### How It Works
For each observation at time t:
1. Set an upper barrier (take profit) at: entry_price + (multiplier_up × ATR_14)
2. Set a lower barrier (stop loss) at: entry_price - (multiplier_down × ATR_14)
3. Set a time barrier at: t + max_holding_period trading days

The label is determined by which barrier is hit FIRST:
- Upper barrier hit first → label = +1 (profitable long)
- Lower barrier hit first → label = -1 (unprofitable / short signal)
- Time barrier hit first → label = 0 (neutral / no clear signal)

### Default Parameters
```yaml
multiplier_up: 2.0        # Take profit = 2x ATR
multiplier_down: 1.5      # Stop loss = 1.5x ATR
max_holding_period: 15     # Trading days
min_atr_threshold: 0.005  # Skip stocks with ATR < 0.5% (illiquid)
```

### Rules
- ATR must be computed BEFORE the observation date (no future ATR)
- Barriers are PER-STOCK (each stock gets its own ATR-scaled barriers)
- On circuit-hit days, use the circuit price as the barrier touch
- For the time barrier, use the close price at t + max_holding_period
- Return magnitude: actual return achieved when barrier was hit

## Meta-Labeling (meta_labeling.py)

### How It Works
1. Primary model makes a directional prediction (UP/DOWN)
2. Meta-label = 1 if primary was correct, 0 if primary was wrong
3. Train a secondary model to predict meta-labels
4. The secondary model's probability output = confidence score

### Rules
- Meta-labels are ONLY computed for non-neutral primary predictions
- Never leak the primary model's training data into meta-label training
- Use separate time periods or CPCV folds for primary and meta training
- Meta-label model is always XGBoost (fast, reliable on tabular)

## Sample Weighting
- Weight samples by uniqueness: 1 / (number of overlapping labels at timestamp)
- More recent samples get higher weight (exponential decay, half-life = 252 days)
- Combine uniqueness and recency weights multiplicatively
