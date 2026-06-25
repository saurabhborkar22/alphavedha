# AlphaVedha Morning Briefing — 2026-06-25 (Wednesday)

## ALERT: DEGRADED PIPELINE — DO NOT TRADE

Today's predictions have systemic quality issues. All 50 Nifty stocks show identical DOWN direction with negligible magnitude, driven by a single calendar feature. The regime detection model is broken. Zero stocks are tradeable. This briefing documents the issues for engineering triage.

---

## Market Context

| Metric | Value |
|--------|-------|
| **Regime** | Unknown (regime detection failed — shape mismatch error) |
| **Predictions** | 61 total across 53 symbols (includes 3 .NS duplicates) |
| **Direction split** | 10 bullish / 51 bearish / 0 neutral |
| **Nifty 50 split** | 0 bullish / 50 bearish / 0 neutral |
| **Tradeable signals** | 0 of 50 Nifty stocks pass cost hurdle |
| **Hash published** | Yes (sha256: ffe3e570cd6873f7...) |
| **Model version** | v0.1.0 |
| **Predictions generated** | 2026-06-24 17:34-17:35 UTC (yesterday's after-hours batch) |
| **System status** | DEGRADED (5 warnings) |

## Data Freshness

| Table | Latest | Status | Notes |
|-------|--------|--------|-------|
| paper_trades | 2026-06-25 | Fresh | 61 today (vs 45 yesterday, +16) |
| prediction_proofs | 2026-06-25 | Fresh | Hash published |
| daily_ohlcv | 2026-06-24 | STALE | 0 rows today (expected ~2700) |
| institutional_flows | 2026-06-24 | STALE | 0 rows today (expected ~2) |
| disclosures | 2026-06-24 | Fresh* | 0 rows today but marked fresh |
| insider_trades | **2026-04-28** | **CRITICAL** | ~2 months stale, 0 recent rows |
| surveillance_flags | 2026-06-24 | STALE | 0 rows today (expected ~253) |
| bulk_block_deals | 2026-06-24 | STALE | 0 rows today (expected ~139) |
| rating_events | 2026-06-24 | Fresh* | 0 rows today |
| transcripts | 2026-06-24 | Fresh* | 0 rows today |

**Infrastructure**: Database OK, Redis OK, Models loaded, Disk 49% (17.4 GB free).

---

## CRITICAL ISSUES (3)

### Issue 1: Regime Detection Model Broken
Every prediction carries the warning:
```
Regime detection failed: operands could not be broadcast together with shapes (266,4) (2,)
```
The HMM regime model expects input shape (N, 2) but receives (266, 4). This means the feature matrix being passed to the regime detector has 4 columns when the model was trained on 2. Likely cause: feature engineering changed after the HMM was last trained, or the model artifact is stale. All predictions default to "unknown" regime with "sideways" parameters, which suppresses magnitude estimates.

### Issue 2: Uniform Direction — All 50 Stocks DOWN
Every Nifty 50 stock predicts DOWN. The top feature for all 50 is `cal_doy` (calendar day of year). This is a hallmark of a model dominated by a seasonal/calendar feature rather than genuine market signals. The composite scores cluster in a 3.7-point range (44.86–48.57), which is anomalously tight for 50 diverse stocks across all sectors.

### Issue 3: Zero Tradeable Signals
All predictions fail the cost hurdle check:
- Largest predicted magnitude: 0.35% (ASIANPAINT, L&T)
- Cost hurdle: 0.71% (0.47% transaction cost × 1.5 safety factor)
- Every stock shows `position_size_pct: 0.0` (Kelly criterion = zero allocation)
- Expected moves are 2x below the cost hurdle

---

## Top 10 Stocks by Composite Score — Analyst Review

### 1. ASIANPAINT (Score: 48.57, Meta-confidence: 85.6%)

```json
{
  "symbol": "ASIANPAINT",
  "ml_direction": -1,
  "ml_score": 48.57,
  "analyst_direction": 0,
  "analyst_adjustment": "0 (SUSPEND)",
  "analyst_score": null,
  "conviction": "none",
  "reasoning": "Highest composite score and meta-confidence in today's batch, but this is meaningless in context. The model is generating uniform DOWN signals across all 50 stocks with negligible magnitude (0.28%). The 85.6% meta-confidence reflects model agreement on direction, but the models are agreeing on a near-zero move driven by calendar seasonality, not a genuine bearish signal. Cannot endorse.",
  "key_risk": "Trading a degenerate prediction set. Model disagreement is low (0.13) but this reflects agreement on noise, not signal.",
  "sector_context": "Sector unknown in model output. FMCG/paints sector — no qualitative catalyst either way."
}
```

### 2. GRASIM (Score: 48.24, Meta-confidence: 74.0%)

```json
{
  "symbol": "GRASIM",
  "ml_direction": -1,
  "ml_score": 48.24,
  "analyst_direction": 0,
  "analyst_adjustment": "0 (SUSPEND)",
  "analyst_score": null,
  "conviction": "none",
  "reasoning": "Same degenerate pattern — DOWN with 0.28% magnitude, cal_doy as top feature. No differentiation from the other 49 stocks. Grasim's infra/cement exposure would need sector-specific catalysts to justify a directional call. None visible in available data. Suspending.",
  "key_risk": "Regime unknown. No institutional flow data today to validate.",
  "sector_context": "Infra sector — 4 infra stocks (GRASIM, ADANIPORTS, LT, ULTRACEMCO, SHRIRAMFIN) all show identical DOWN pattern."
}
```

### 3. ADANIPORTS (Score: 48.08, Meta-confidence: 69.6%)

```json
{
  "symbol": "ADANIPORTS",
  "ml_direction": -1,
  "ml_score": 48.08,
  "analyst_direction": 0,
  "analyst_adjustment": "0 (SUSPEND)",
  "analyst_score": null,
  "conviction": "none",
  "reasoning": "UP 1.64% yesterday, model says DOWN 0.22%. In normal conditions would review for mean-reversion setup, but with degenerate predictions, this is noise. No red flags from blowup list (0 flagged stocks at threshold 70).",
  "key_risk": "Adani group governance historically elevated risk, but no current red flags in system.",
  "sector_context": "Infrastructure/ports — no differentiating signal available."
}
```

### 4. JSWSTEEL (Score: 48.08, Meta-confidence: 60.1%)

```json
{
  "symbol": "JSWSTEEL",
  "ml_direction": -1,
  "ml_score": 48.08,
  "analyst_direction": 0,
  "analyst_adjustment": "0 (SUSPEND)",
  "analyst_score": null,
  "conviction": "none",
  "reasoning": "Down 0.89% yesterday, model predicts further 0.21% down. Metals sector has 3 stocks (JSWSTEEL, TATASTEEL, HINDALCO + ADANIENT classified under Metals) — all DOWN but with different confidence levels. However, the uniform direction is driven by cal_doy, not metals-specific features. Cannot distinguish genuine sector weakness from model artifact.",
  "key_risk": "Metals sector is cyclical and sensitive to China PMI/commodity prices — qualitative factors that may or may not support DOWN direction but are not what the model is keying on.",
  "sector_context": "Metals: JSWSTEEL (60%), TATASTEEL (41%), HINDALCO (24%), ADANIENT (72%) — confidence varies widely despite same direction, suggesting the signal is noisy."
}
```

### 5. AXISBANK (Score: 47.93, Meta-confidence: 71.5%)

```json
{
  "symbol": "AXISBANK",
  "ml_direction": -1,
  "ml_score": 47.93,
  "analyst_direction": 0,
  "analyst_adjustment": "0 (SUSPEND)",
  "analyst_score": null,
  "conviction": "none",
  "reasoning": "Banking sector has 4 stocks all DOWN (AXISBANK 71.5%, ICICIBANK 58.5%, KOTAKBANK 66.6%, SBIN 61.4%, HDFCBANK 32.3%). The spread in confidence (32-72%) is the widest sector variance, but direction is identical. UP 1.54% yesterday. No institutional flow data to corroborate any direction.",
  "key_risk": "Banking is 25% of Nifty 50 weight. A genuine 4-bank DOWN signal would be extremely significant, but this is not a genuine signal — it's calendar-feature contamination.",
  "sector_context": "Banking: 4 of 4 stocks DOWN — sector-level pattern, but driven by the same cal_doy artifact across all sectors."
}
```

### 6. NESTLEIND (Score: 47.93, Meta-confidence: 49.7%)

```json
{
  "symbol": "NESTLEIND",
  "ml_direction": -1,
  "ml_score": 47.93,
  "analyst_direction": 0,
  "analyst_adjustment": "0 (SUSPEND)",
  "analyst_score": null,
  "conviction": "none",
  "reasoning": "Coin-flip meta-confidence (49.7%) — model has essentially no conviction. DOWN 0.69% yesterday. FMCG sector stock with defensive characteristics. The near-50% confidence is the most honest reading in today's batch: the model doesn't know.",
  "key_risk": "No material risk or catalyst. Low-conviction prediction should be ignored.",
  "sector_context": "FMCG: NESTLEIND (50%), HINDUNILVR (58%), TATACONSUM (40%), ITC (22%) — all DOWN with wide confidence range."
}
```

### 7. APOLLOHOSP (Score: 47.85, Meta-confidence: 54.4%)

```json
{
  "symbol": "APOLLOHOSP",
  "ml_direction": -1,
  "ml_score": 47.85,
  "analyst_direction": 0,
  "analyst_adjustment": "0 (SUSPEND)",
  "analyst_score": null,
  "conviction": "none",
  "reasoning": "Healthcare sector, UP 1.01% yesterday. Classified under Pharma in model. Meta-confidence barely above 50%. Magnitude 0.33%. No qualitative catalyst visible.",
  "key_risk": "None identified beyond systemic model issues.",
  "sector_context": "Pharma/Healthcare: APOLLOHOSP (54%), CIPLA (70%), SUNPHARMA (52%), DRREDDY (65%) — all DOWN."
}
```

### 8. ICICIBANK (Score: 47.84, Meta-confidence: 58.5%)

```json
{
  "symbol": "ICICIBANK",
  "ml_direction": -1,
  "ml_score": 47.84,
  "analyst_direction": 0,
  "analyst_adjustment": "0 (SUSPEND)",
  "analyst_score": null,
  "conviction": "none",
  "reasoning": "Strongest single-day gainer yesterday (+2.64%) among banks. Model predicts DOWN 0.22%. In normal conditions, a strong up day followed by mild DOWN prediction could indicate mean-reversion, but with the cal_doy contamination, this is not actionable. No institutional flow data to validate.",
  "key_risk": "No red flags. ICICI Bank is systemically important — a genuine SELL signal would require strong evidence.",
  "sector_context": "Banking — see AXISBANK note above."
}
```

### 9. KOTAKBANK (Score: 47.74, Meta-confidence: 66.6%)

```json
{
  "symbol": "KOTAKBANK",
  "ml_direction": -1,
  "ml_score": 47.74,
  "analyst_direction": 0,
  "analyst_adjustment": "0 (SUSPEND)",
  "analyst_score": null,
  "conviction": "none",
  "reasoning": "UP 1.07% yesterday. Model DOWN 0.12% — the smallest magnitude in the top 10. The magnitude is essentially zero (1.2 bps). Even if correct, this is a no-trade: 0.12% move vs 0.71% cost hurdle. Suspending.",
  "key_risk": "None beyond systemic issues.",
  "sector_context": "Banking — see above."
}
```

### 10. CIPLA (Score: 47.69, Meta-confidence: 70.0%)

```json
{
  "symbol": "CIPLA",
  "ml_direction": -1,
  "ml_score": 47.69,
  "analyst_direction": 0,
  "analyst_adjustment": "0 (SUSPEND)",
  "analyst_score": null,
  "conviction": "none",
  "reasoning": "Pharma sector, 70% meta-confidence but same degenerate pattern. UP 0.33% yesterday. Magnitude -0.20%. No differentiating signal from the other 49 stocks. Cipla has no current red flags.",
  "key_risk": "Pharma sector exposure to US FDA actions — qualitative risk not captured in model.",
  "sector_context": "Pharma: 4 of 4 stocks DOWN — CIPLA (70%), DRREDDY (65%), APOLLOHOSP (54%), SUNPHARMA (52%)."
}
```

---

## Remaining 40 Stocks — Brief Assessments

| # | Symbol | Score | Conf | Dir | Analyst One-Liner |
|---|--------|-------|------|-----|-------------------|
| 11 | SBIN | 47.45 | 61.4% | DOWN | Banking — suspended, same degenerate pattern. UP 1.02% yesterday. |
| 12 | SUNPHARMA | 47.42 | 51.9% | DOWN | Pharma — near coin-flip confidence, no catalyst. |
| 13 | LT | 47.40 | 68.8% | DOWN | Infra — magnitude 0.35% is largest in set but still below cost hurdle. |
| 14 | DRREDDY | 47.40 | 64.9% | DOWN | Pharma — UP 2.08% yesterday, model says tiny pullback. Non-tradeable. |
| 15 | HDFCBANK | 47.06 | 32.3% | DOWN | Banking bellwether — UP 2.39% yesterday. Very low confidence. |
| 16 | BEL | 47.02 | 44.4% | DOWN | Defence — DOWN 1.54% yesterday, model adds 0.07%. Negligible. |
| 17 | BHARTIARTL | 46.96 | 50.8% | DOWN | Telecom — exact coin flip, no signal. DOWN 1.28% yesterday. |
| 18 | BAJAJFINSV | 46.88 | 59.1% | DOWN | Financial — UP 0.88% yesterday, moderate confidence but degenerate. |
| 19 | ADANIENT | 46.83 | 72.5% | DOWN | Metals/Conglomerate — UP 3.6% yesterday (biggest gainer). High confidence but cal_doy driven. |
| 20 | COALINDIA | 46.81 | 27.1% | DOWN | Energy — very low confidence, DOWN 0.59% yesterday. No signal. |
| 21 | SBILIFE | 46.77 | 49.3% | DOWN | Financial/Insurance — coin flip, DOWN 1.04% yesterday. |
| 22 | HINDUNILVR | 46.74 | 58.0% | DOWN | FMCG defensive — flat yesterday, moderate confidence but same pattern. |
| 23 | RELIANCE | 46.72 | 30.9% | DOWN | Energy bellwether — low confidence, magnitude 0.16%. No signal. |
| 24 | TATASTEEL | 46.69 | 41.1% | DOWN | Metals — DOWN 1.76% yesterday. Low-moderate confidence. |
| 25 | JIOFIN | 46.66 | 37.8% | DOWN | Financial/Tech — low confidence, UP 0.45% yesterday. |
| 26 | POWERGRID | 46.49 | 38.8% | DOWN | Energy/Utility — low confidence, DOWN 0.36% yesterday. |
| 27 | TRENT | 46.45 | 70.1% | DOWN | Retail — UP 3.31% yesterday (strong gainer). High confidence but degenerate. |
| 28 | M&M | 46.45 | 54.7% | DOWN | Auto — UP 0.9% yesterday. Moderate confidence, suspended. |
| 29 | TMPV | 46.40 | 48.6% | DOWN | Unknown sector — DOWN 1.37% yesterday. Coin-flip confidence. |
| 30 | BAJFINANCE | 46.38 | 37.9% | DOWN | Financial — UP 2.97% yesterday (strong gainer). Low confidence. |
| 31 | TATACONSUM | 46.29 | 39.8% | DOWN | FMCG — DOWN 0.52% yesterday. Low confidence. |
| 32 | HDFCLIFE | 46.27 | 27.3% | DOWN | Insurance — DOWN 0.95% yesterday. Very low confidence. |
| 33 | INDIGO | 46.24 | 73.0% | DOWN | Aviation — UP 4.95% yesterday (biggest gainer in set). High confidence but cal_doy driven — possible mean-reversion signal contaminated by calendar feature. |
| 34 | HINDALCO | 46.22 | 24.1% | DOWN | Metals — DOWN 1.03% yesterday. Very low confidence. |
| 35 | ITC | 46.16 | 22.3% | DOWN | FMCG — very low confidence, flat yesterday (+0.12%). No signal. |
| 36 | NTPC | 46.13 | 21.2% | DOWN | Energy/Power — DOWN 2.07% yesterday. Very low confidence, near baseline. |
| 37 | TITAN | 46.07 | 73.7% | DOWN | Consumer luxury — UP 0.43% yesterday. High confidence but same degenerate pattern. |
| 38 | SHRIRAMFIN | 46.02 | 44.8% | DOWN | Financial/NBFC — UP 2.58% yesterday. Moderate confidence. |
| 39 | ETERNAL | 45.92 | 22.2% | DOWN | Unknown sector — DOWN 1.0% yesterday. Very low confidence, magnitude near zero. |
| 40 | EICHERMOT | 45.84 | 43.6% | DOWN | Auto — flat yesterday (-0.08%). Low-moderate confidence. |
| 41 | TECHM | 45.71 | 34.2% | DOWN | IT — UP 3.25% yesterday (strong gainer). Low confidence. |
| 42 | MAXHEALTH | 45.65 | 43.6% | DOWN | Healthcare — UP 0.34% yesterday. Low-moderate confidence. |
| 43 | BAJAJ-AUTO | 45.62 | 41.1% | DOWN | Auto — DOWN 2.74% yesterday (biggest decliner). Model adds 0.23%. |
| 44 | HCLTECH | 45.62 | 35.9% | DOWN | IT — UP 0.4% yesterday. Low confidence. |
| 45 | INFY | 45.51 | 22.9% | DOWN | IT bellwether — UP 2.65% yesterday. Very low confidence. |
| 46 | ONGC | 45.45 | 10.6% | DOWN | Energy — lowest confidence in entire set. DOWN 1.78% yesterday. Model has no conviction. |
| 47 | MARUTI | 45.33 | 32.5% | DOWN | Auto — DOWN 1.51% yesterday. Low confidence. |
| 48 | ULTRACEMCO | 45.12 | 40.0% | DOWN | Cement/Infra — UP 1.09% yesterday. Low-moderate confidence. |
| 49 | TCS | 44.97 | 40.4% | DOWN | IT bellwether — UP 2.4% yesterday. Low-moderate confidence. |
| 50 | WIPRO | 44.86 | 16.6% | DOWN | IT — lowest score in set. Very low confidence, flat yesterday. |

---

## Cross-Stock Patterns (Step 4)

### 1. Sector Concentration: ALL SECTORS UNIFORMLY DOWN — ANOMALOUS

| Sector | Stocks | All DOWN? | Confidence Range | Assessment |
|--------|--------|-----------|-----------------|------------|
| Banking | AXISBANK, ICICIBANK, KOTAKBANK, SBIN, HDFCBANK | Yes (5/5) | 32–72% | NOT a genuine sector signal — driven by cal_doy |
| IT | HCLTECH, INFY, TCS, TECHM, WIPRO | Yes (5/5) | 17–40% | Low confidence cluster — model has no IT-specific signal |
| Pharma | APOLLOHOSP, CIPLA, DRREDDY, SUNPHARMA | Yes (4/4) | 52–70% | Higher confidence cluster but still same top feature |
| FMCG | HINDUNILVR, ITC, NESTLEIND, TATACONSUM | Yes (4/4) | 22–58% | Wide confidence range — no coherent signal |
| Financial | BAJAJFINSV, BAJFINANCE, HDFCLIFE, JIOFIN, SBILIFE, SHRIRAMFIN | Yes (6/6) | 27–59% | Broad financial sector, all DOWN |
| Energy | COALINDIA, NTPC, ONGC, POWERGRID, RELIANCE | Yes (5/5) | 11–39% | Lowest confidence cluster — model has minimal conviction on energy |
| Infra | ADANIPORTS, GRASIM, LT, ULTRACEMCO | Yes (4/4) | 40–74% | Mixed confidence |
| Auto | BAJAJ-AUTO, EICHERMOT, M&M, MARUTI | Yes (4/4) | 33–55% | Mid-range confidence |
| Metals | ADANIENT, HINDALCO, JSWSTEEL, TATASTEEL | Yes (4/4) | 24–72% | Widest spread in single sector |

**Analyst verdict**: When all 50 stocks across 9 sectors show the same direction, it is NOT a market signal — it is a model defect. A genuine market-wide bearish signal would show meaningful magnitudes and differentiated top features. Here, magnitude is uniformly negligible and the top feature is identical (cal_doy) for every stock.

### 2. FII/DII Flow Alignment
**Data unavailable** — institutional_flows table is stale (last: 2026-06-24, 0 rows today). Cannot assess flow alignment with predictions.

### 3. Event Clustering
No disclosure events ingested today (0 rows in disclosures delta). Cannot assess event clustering.

### 4. Regime Context
**Regime detection is broken.** The HMM model throws a shape mismatch error:
```
shapes (266,4) (2,)
```
All 50 stocks default to "unknown" regime. The regime detector is receiving 4-column input when it was trained on 2 columns. This is likely because:
- The feature engineering pipeline was updated to produce more features
- The HMM model artifact was not retrained to match

Without regime context, the prediction engine defaults to "sideways" parameters which suppress magnitude estimates and likely contribute to the uniformly tiny predicted moves.

### 5. Yesterday's Price Action vs Model Predictions
Notable mismatch: several stocks gained 2-5% yesterday (INDIGO +4.95%, ADANIENT +3.6%, TRENT +3.31%, TECHM +3.25%, BAJFINANCE +2.97%, INFY +2.65%, ICICIBANK +2.64%) but model predicts uniform micro-DOWN for all. If these gains reflected genuine momentum, the model is blind to it. If they were overextended, the model's DOWN call could be directionally right but with negligible magnitude.

---

## Red Flags / Blowup Avoid List

**Zero stocks flagged** at threshold 70. The red-flags system reports:
```json
{"threshold": 70, "flagged_count": 0, "symbols": []}
```

Note: The insider_trades table is 2 months stale (last: 2026-04-28), so the insider_sell component of the blowup score may be outdated. Governance and other components appear to be scoring within normal ranges.

---

## Top Picks — NONE

**There are zero actionable BUY or SELL signals today.** All 50 Nifty stocks fail the cost hurdle check (predicted magnitude < 0.71% transaction cost threshold). Position sizes are 0.0% for all stocks via Kelly criterion. The scan endpoint returns 0 buy_candidates and 0 sell_candidates.

## Top Shorts / Avoids — NONE ACTIONABLE

While all 50 stocks show DOWN direction, the predicted magnitudes are negligible (max 0.35%) and fall below the cost hurdle. No short positions are recommended.

---

## Engineering Action Items

### P0 — Regime Detection Shape Mismatch
The HMM regime model expects (N, 2) input but receives (N, 4). This breaks every prediction's regime classification. **Action**: Investigate whether the feature pipeline changed since the HMM was last trained. Retrain HMM with current feature set or fix the feature selection in the regime detection call.

### P1 — Calendar Feature Dominance
`cal_doy` is the top feature for all 50 stocks. This suggests the model is overfitting to calendar day-of-year. **Action**: Review feature importance in XGBoost. Consider whether cal_doy should be removed or its importance capped. Check if recent retraining inadvertently gave it outsized weight.

### P1 — Insider Trades Data 2 Months Stale
Last data: 2026-04-28. The `19:20 insider trades` job has not produced data in ~2 months. **Action**: Check if the data source endpoint changed, authentication expired, or the scraper broke.

### P2 — Duplicate Symbol Predictions
Three stocks have duplicate predictions (HCLTECH/HCLTECH.NS, RELIANCE/RELIANCE.NS, WIPRO/WIPRO.NS). 61 predictions for what should be 50 unique Nifty stocks. **Action**: Deduplicate symbol list in the prediction pipeline.

### P2 — Stale Overnight Data Tables
daily_ohlcv, institutional_flows, surveillance_flags, bulk_block_deals all show 0 new rows today. The overnight data refresh jobs may have failed. **Action**: Check job scheduler logs.

---

## Summary

Today's prediction pipeline produced technically valid but qualitatively useless output. The regime detection model is broken (shape mismatch), the ensemble is dominated by a calendar feature (cal_doy), and all predicted magnitudes fall below transaction cost thresholds. **Zero stocks are tradeable.** The system correctly flags all predictions as non-tradeable (position_size = 0%), which is the right safety behavior — the cost hurdle check is working as designed.

**Analyst recommendation**: Stand down. No positions today. Prioritize fixing the regime detection model and investigating calendar feature dominance before the next trading session.
