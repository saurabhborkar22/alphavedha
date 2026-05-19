# A3: Delivery Percentage Signals — Design Spec

## Summary

Add 3 new delivery-based features to `alphavedha/features/microstructure.py`, increasing the microstructure feature count from 10 to 13 and the total feature count from 142 to 145.

## Context

The execution plan (Phase A3) calls for 5 delivery features. Two already exist under different names:
- `micro_delivery_zscore` covers `delivery_pct_zscore`
- `micro_delivery_trend_5d` covers `delivery_trend_5d`

This spec adds only the 3 genuinely new features.

## New Features

### 1. `micro_delivery_pct_rank` (ratio 0-1)

Percentile rank of today's delivery % within a 60-day rolling window. Captures whether current delivery is historically high or low for this stock.

```python
delivery.rolling(60, min_periods=10).apply(lambda x: x.rank(pct=True).iloc[-1])
```

Optimized implementation uses `rolling(60).rank()` / `rolling(60).count()` to avoid per-row lambda.

### 2. `micro_delivery_vol_combo` (binary 0/1)

Accumulation signal: delivery z-score > 2.0 AND volume > 1.5x the 20-day average volume. When both fire simultaneously, it signals institutional conviction.

```python
(delivery_zscore > 2.0) & (volume > 1.5 * vol_ma_20)
```

Reuses `rolling_mean`, `rolling_std`, and `vol_ma_20` already computed for existing features.

### 3. `micro_high_delivery_breakout` (binary 0/1)

Conviction breakout: delivery > 60% AND close breaks above the 20-day rolling high. Uses `shift(1)` on the rolling max to prevent look-ahead.

```python
(delivery > 0.6) & (close > close.rolling(20, min_periods=5).max().shift(1))
```

## Files Modified

| File | Change |
|------|--------|
| `alphavedha/features/microstructure.py` | Add 3 features, update count 10→13 |
| `alphavedha/features/pipeline.py` | Update EXPECTED_FEATURE_COUNT 142→145 |
| `configs/features.yaml` | Add 3 entries under microstructure |
| `tests/unit/features/test_microstructure.py` | Add tests for new features |

## Invariants

- No look-ahead bias: all rolling windows use only past data, breakout uses `shift(1)`
- Graceful degradation: if `delivery_pct` is missing, all 3 new features return 0
- No new dependencies
- No changes to existing feature column names or values
