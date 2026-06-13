# Prediction System Audit — Out-of-Sample Evidence & Improvement Plan

**Date:** 2026-06-14  ·  **Basis:** first honest out-of-sample run (`scripts/sim_paper_trading.py`)
**Window:** 2025-12-11 → 2026-05-19 (114 trading days, 50 large-caps, 5,700 predictions)
**Model:** frozen as-of cutoff 2025-12-10 (saw no test-window data) · v0.1.0

> ⚠️ **Scope caveat.** This is **one cutoff over one regime** — and that regime was a sharp
> drawdown (equal-weight basket −36.6%). The *structural* findings (cost > edge, dead features,
> confidence inversion) are regime-independent; the *regime/sizing* findings need a second
> cutoff over a flat/up window to confirm they aren't crash-specific. A multi-cutoff protocol is
> in §6.

---

## 1. Executive summary

The system is **not profitable net of costs** on unseen data, and — more importantly — its
**confidence signal is inverted**: the more confident the system is, the *worse* it does.

| Selection rule | Trades | Win % | Net/trade | Sharpe |
|---|---|---|---|---|
| `all` (trade everything) | 5,700 | **49.5%** | −0.16% | −0.16 |
| `top_5`/day (highest confidence) | 570 | **47.4%** | −0.70% | −0.66 |
| `gate_passed` (meta-gate, the live strategy) | 52 | **40.4%** | −1.60% | −0.78 |

**Win rate falls monotonically as confidence rises.** Selecting by the model's own confidence is
worse than random selection. That single fact is the most important output of this audit.

**Root causes, ranked by leverage:**

1. **Confidence inversion** — the ensemble/meta confidence is anti-predictive OOS (overfit). *#1 fix.*
2. **Cost > edge** — raw gross edge +0.31%/trade vs 0.47% round-trip cost ⇒ net-negative before any selection.
3. **Regime mis-ID + aggressive bull params** — the Dec crash was labeled "bull" → full Kelly + the most-permissive gate → −44.65% month, −41% max DD.
4. **Macro/derivatives/delivery features are 100% NaN** — the system is blind to exactly the signals (VIX, FII/DII, PCR) that flag a regime turn.
5. **Negative skew, no intra-trade risk cut** — 15-day fixed hold, 14 of 52 gate trades worse than −5%, profit factor 0.50.

**One genuine positive:** directional trading (incl. shorts) lost **−26%** vs the basket's **−36.6%** (+10.6pp) — it has defensive value in a downturn. The confidence layer is what destroys it.

---

## 2. Method & evidence base

- Frozen-model walk-forward replay through the **real** prediction engine (`as_of` seam), so these
  are the exact predictions production would have made on each day.
- Returns are directional (`predicted_direction × actual_return`), net of the live cost model
  (`backtest/costs.py`, 0.471% round-trip, large-cap).
- Data pulled live from `/api/paper/simulation` + `/api/backtest/*`. Reproduce: `python /tmp/audit.py`.

---

## 3. Quantitative findings

### 3.1 Cost dominates a thin edge (structural)
- Gross edge `all` = **+0.314%/trade**; round-trip cost = **0.471%/trade** → net **−0.156%**.
- Cost is **150% of the gross edge.** Annualized (~16.8 × 15-day cycles): gross **+5.3%/yr**, cost **−7.9%/yr**, net **−2.6%/yr**.
- **Break-even** needs gross edge ≥ 0.471% (**+50%** improvement) *or* cost ≤ 0.314% (**−33%**).
- Directional accuracy **51.9%** — barely above a coin flip. The edge is real but tiny.

### 3.2 Confidence is inverted (the headline)
- `all` 49.5% → `top_5` 47.4% → `gate_passed` 40.4% win. Monotonic **down** with confidence.
- Meta-gate net −1.60% is **10× worse** than `all` −0.16%. The gate actively *subtracts* value.
- Implication: the meta-labeling model (train F1 0.836) and ensemble confidence are **overfit** and
  do not generalize; their ranking is anti-correlated with OOS outcomes.

### 3.3 The Dec-2025 blow-up (regime + sizing)
- Monthly (gate): **2025-12 −44.65%**, 2026-01 +26.02%, 2026-02 −8.82%, 2026-05 +9.58% (Mar/Apr: gate silent).
- Max drawdown **−41.2% on 2025-12-31.**
- Mechanism (confirmed in code): regime detector classified the crashing market as **bull** →
  `kelly_fraction=1.0` and `meta_confidence_threshold=0.40` (the **most aggressive + most permissive**
  of all four regimes, `regime_strategy.py:34-39`) → full-size losing trades let through the gate.

### 3.4 Negative skew / fat tail
- Gate distribution (52 trades): **14** worse than −5%, only **7** better than +5%; 60% losers; PF **0.50**.
- 15-day fixed hold with no intra-trade stop (`scheduler._evaluate_open_paper_trades`,
  `walk_forward._execute_trades`) → adverse moves run the full horizon.

---

## 4. Root-cause analysis (data ↔ code)

### A. Confidence inversion — **highest leverage**
- **Evidence:** §3.2 monotonic win-rate decline with confidence.
- **Mechanism:** `engine.predict` gates on `meta_confidence ≥ regime threshold` (`engine.py:131-134`)
  and the meta-model's `is_tradeable`. Both are trained on OOF data where they scored well
  (meta F1 0.836) but **invert OOS** — classic meta-label / stacking overfit. The ensemble is a
  `RidgeClassifier` meta-learner on base-model OOF probabilities; with weak, correlated base models
  its confidence is unreliable.
- **Why it matters:** every selection layer (gate, ranking, top-k) is built on this signal, so the
  whole "only trade high-conviction names" thesis is currently *backwards*.

### B. Cost > edge — **structural unprofitability**
- **Evidence:** §3.1. **Mechanism:** `compute_round_trip_cost_pct` = STT+brokerage+GST+slippage+stamp
  ≈ 0.471% (large-cap). The 15-day triple-barrier horizon turns that into ~7.9%/yr drag, larger than
  the 5.3%/yr gross alpha.
- **Why it matters:** even with a *perfect* confidence fix, trading the raw signal at this frequency
  loses to costs. The edge-per-cost ratio must rise — via bigger edge, longer holds, or fewer trades.

### C. Regime mis-classification + asymmetric aggression
- **Evidence:** §3.3; `regime_strategy.py:34-62` (bull = kelly 1.0 / thr 0.40; high_vol = kelly 0.1 /
  thr 0.52 + unanimous).
- **Mechanism:** the HMM regime detector is fit on **equal-weight portfolio log-returns + 20-day
  realized vol only** (`prediction_service._build_market_features`) — *not* on India VIX or breadth,
  because those features are dead (see D). It lags turning points, so a crash reads as "bull," which
  is the single most dangerous misread given bull = full Kelly + lowest gate.

### D. Dead feature groups — **blind to macro stress**
- **Evidence:** training log warnings — `derivatives_no_data`, `fundamental_no_earnings`,
  `no promoter data`; "High NaN columns: macro_fii_*, macro_dii_*, macro_sector_*, deriv_* (100%)".
  ~80,630 of values NaN-filled per symbol; 15 stub features dropped (`_drop_stub_features`).
- **Mechanism:** no live data source wired for macro (VIX, FII/DII, yield curve, USD/INR),
  derivatives (OI, PCR, Greeks), or delivery %. All fill 0 in both train and serve.
- **Why it matters:** these are precisely the regime/stress signals. The model trades on technicals
  alone and cannot see the conditions that produced the Dec drawdown.

### E. No intra-trade risk control + full Kelly
- **Evidence:** §3.4; `position_size = risk.position_size_pct × strategy.kelly_fraction`
  (`engine.py:167`), kelly 1.0 in bull.
- **Mechanism:** fixed 15-day hold, no stop-loss / no volatility-scaled exit; full Kelly is known to
  be over-aggressive (1/2 Kelly is the usual practical cap). Fat left tail is the result.

---

## 5. Recommendations (ranked by impact ÷ effort)

### P0 — do first (high impact, low effort)
1. **Stop trusting confidence until re-validated.** Until the meta layer is fixed, the honest
   "strategy" is `all` or a *small* top-k — not the gate. Treat the gate as broken.
2. **Re-validate calibration OOS, not OOF.** Add a reliability-curve / Brier-score check on a
   held-out window to CI. Gate thresholds must be set on OOS calibration, not train F1.
3. **Cap Kelly at ≤ 0.5 and decouple aggression from the regime label.** Bull = full Kelly is the
   direct cause of the Dec blow-up. Make high-vol detection *reduce* size regardless of bull/bear.
4. **Raise the effective edge hurdle:** only trade when `expected_move > k × round_trip_cost`
   (e.g. k≥1.5). This directly attacks the cost>edge problem and naturally cuts low-edge churn.

### P1 — structural (high impact, medium effort)
5. **Rebuild the meta-labeling layer** with purged/embargoed OOS calibration; consider isotonic /
   Platt calibration of probabilities, and *monotonicity tests* (higher confidence must not lower
   realized win-rate on validation).
6. **Fix the regime detector inputs** — feed it real India VIX + breadth + a trend filter, not just
   portfolio returns. A simple 200-DMA / VIX-threshold overlay would likely have flagged Dec.
7. **Wire at least one live macro source** (India VIX + FII/DII are the cheapest, highest-value).
   Even one real macro feature beats 25 dead ones.
8. **Add a volatility-scaled stop / time-decayed exit** to the triple-barrier so the −5% tail is cut.

### P2 — model quality (medium impact, higher effort)
9. **The deep models are weak** (TFT val-acc 0.39, LSTM 0.45 on 3-class). Either fix them
   (more regularization is in, but they still under-perform XGBoost) or **drop them from the
   ensemble** and lean on XGBoost + a calibrated gate. A simpler, well-calibrated model often beats
   a complex, mis-calibrated one.
10. **Lengthen the horizon or trade less often** to amortize the fixed round-trip cost over a larger
    expected move (the 15-day barrier may be too short for the edge size).

---

## 6. Suggested validation protocol (before trusting any change)

1. **Multi-cutoff walk-forward:** run the sim at ≥3 cutoffs spanning up / flat / down regimes
   (e.g. 2025-03, 2025-08, 2025-12). Confidence inversion + cost>edge should reproduce; if the
   regime/Dec findings don't, they're crash-specific.
2. **Calibration gate in CI:** reliability curve + Brier score on each fold; fail if higher-confidence
   buckets don't have higher realized accuracy.
3. **Cost-sensitivity sweep:** re-run net P&L at 0×, 0.5×, 1×, 2× cost to see how much of the loss is
   edge vs cost.
4. **Ablations:** ensemble-without-deep-models; gate-off vs gate-on; Kelly 1.0 vs 0.5 vs 0.25.

---

## 7. Appendix

### Per-track (net of 0.471% round-trip)
```
all          5700 trades  win 49.5%  gross +0.31%  net -0.16%  PF 0.95  Sharpe -0.16
gate_passed    52 trades  win 40.4%  gross -1.13%  net -1.60%  PF 0.50  Sharpe -0.78
top_5         570 trades  win 47.4%  gross -0.23%  net -0.70%  PF 0.77  Sharpe -0.66
equity: strategy -26.0% vs benchmark -36.6% (alpha +10.6pp); max DD -41.2% @ 2025-12-31
```

### Key code references
- Gate logic: `alphavedha/prediction/engine.py:131-167`
- Regime params (bull = kelly 1.0 / thr 0.40): `alphavedha/prediction/regime_strategy.py:34-62`
- Cost model: `alphavedha/backtest/costs.py`
- Regime input (returns+vol only): `alphavedha/services/prediction_service.py:_build_market_features`
- Track-record math: `alphavedha/backtest/sim_views.py`, `alphavedha/monitoring/track_record.py`
