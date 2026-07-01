# AlphaVedha Morning Briefing — 2026-07-01 (Tuesday)

---

## Market Context

| Metric | Value |
|---|---|
| **Regime** | Sideways (all 50 stocks) |
| **Predictions** | 62 total across 57 symbols (50 Nifty large-cap) |
| **Direction split** | 9 bullish / 53 bearish / 0 neutral (full universe) |
| **Nifty 50 split** | 0 bullish / 50 bearish / 0 neutral |
| **Tradeable signals** | 12 above 0.55 meta-confidence (0 from Nifty 50) |
| **Hash published** | Yes (SHA: f5c7b9208a36ee70...) |
| **System health** | DEGRADED — 5 warnings |

### Data Freshness

| Table | Last Updated | Status | Note |
|---|---|---|---|
| paper_trades | 2026-07-01 | Fresh | 62 predictions today |
| prediction_proofs | 2026-07-01 | Fresh | Hash published |
| daily_ohlcv | 2026-06-30 | Expected | Pre-market; today's data after close |
| institutional_flows | 2026-06-30 | Expected | FII/DII job runs 18:30 |
| disclosures | 2026-06-30 | Fresh | 684 recent disclosures |
| rating_events | 2026-06-30 | Fresh | 30 recent |
| transcripts | 2026-06-30 | Fresh | 13 recent |
| **insider_trades** | **2026-04-28** | **STALE (2+ months)** | **Job may be broken** |
| surveillance_flags | 2026-06-30 | Expected | Job runs 19:30 |
| bulk_block_deals | 2026-06-30 | Expected | Job runs 19:45 |

**Key data issue:** `insider_trades` has not ingested data since April 28, 2026 — over 2 months stale. This means insider buying/selling signals are absent from qualitative context for all stocks. The pipeline job at 19:20 appears to be failing silently. **Recommend investigating the insider_trades ingestion pipeline.**

---

## ⚠️ CRITICAL MODEL HEALTH ALERT

**The ML predictions for today show a systemic pattern that warrants serious scrutiny before acting on any signal:**

1. **`cal_doy` (calendar day-of-year) is the #1 feature for ALL 50 Nifty stocks.** This is not a stock-specific signal — it's a calendar artifact. The model is making a date-based prediction rather than per-stock directional calls.

2. **ALL 50 stocks have identical DOWN direction.** No differentiation whatsoever. A model that predicts every stock the same direction on the same day is telling you about date effects, not individual equity risk/reward.

3. **Composite scores are clustered in a 1.13-point range (41.75–42.88).** For context, a useful score range would differentiate strong sells (20s) from neutral (45-55) to strong buys (70+). The current range provides zero actionable differentiation.

4. **Zero Nifty 50 stocks meet the 0.55 meta-confidence threshold.** The model itself is not confident in its predictions. Best: APOLLOHOSP at 0.5012.

5. **Magnitudes are near-zero** (-0.0012 to +0.0008). The model is predicting essentially no price movement.

**Analyst assessment:** Today's ML output is dominated by a seasonal/calendar feature that has overwhelmed stock-specific signals. This could be a legitimate July 1 effect (new quarter, portfolio rebalancing) but the uniformity strongly suggests overfitting to `cal_doy`. **No tradeable signals should be extracted from today's Nifty 50 predictions.** The model's own meta-confidence confirms this — none cross the tradeable threshold.

---

## Red Flag Radar

**Blowup risk stocks flagged:** 0 (threshold: 70)

No Nifty 50 stocks currently exceed the blowup risk threshold across pledge, rating, governance, default, surveillance, beneish, or insider_sell components.

Note: Insider sell signals may be understated due to the 2-month data gap in `insider_trades` table.

---

## Sector Patterns

### ALL sectors bearish — but this is the `cal_doy` artifact, not genuine sector rotation

| Sector | Stocks | All DOWN | Notable |
|---|---|---|---|
| Banking | 5 (AXIS, KOTAK, SBI, ICICI, HDFC) | Yes | AXISBANK highest confidence at 44% |
| IT | 5 (TCS, TECHM, HCLTECH, WIPRO, INFY) | Yes | IT saw real selling yesterday (-2.78% to -3.5%) |
| Energy | 5 (POWERGRID, NTPC, RELIANCE, ONGC, COAL) | Yes | NTPC highest composite score (42.88) |
| Infra | 5 (SHRIRAM, ADANI PORTS, L&T, GRASIM, ULTRA) | Yes | SHRIRAMFIN 2nd-highest confidence (48%) |
| Pharma | 4 (APOLLO, CIPLA, DRREDDY, SUN) | Yes | APOLLOHOSP highest confidence overall (50%) |
| Auto | 4 (EICHER, MARUTI, M&M, BAJAJ-AUTO) | Yes | EICHERMOT fell -4.75% yesterday |
| FMCG | 4 (HUL, NESTLE, TATACONSUM, ITC) | Yes | TATACONSUM -3.34% yesterday |
| Financial | 4 (BAJAJFINSV, SBILIFE, BAJFIN, HDFCLIFE) | Yes | Low confidence across board |
| Metals | 4 (ADANIENT, JSW, HIND, TATA STEEL) | Yes | TATASTEEL only stock in negative territory (-0.8%) |
| Telecom | 1 (BHARTIARTL) | Yes | Only telecom stock in Nifty |

**Cross-stock pattern assessment:** Because all 50 stocks received identical DOWN signals driven by `cal_doy`, no genuine cross-sector rotation signal can be extracted today. The sector uniformity IS the pattern — and it points to model behavior, not market dynamics.

**Observable market movement (yesterday's close):**
- **IT sector selloff confirmed:** INFY -3.5%, TCS -3.17%, WIPRO -2.9%, HCLTECH -2.78%, TECHM -2.03%. This is a genuine sector-level move, but the model's DOWN call is not distinguishing IT from any other sector.
- **Auto divergence:** MARUTI +5.24% (strong rally) yet model says DOWN. EICHERMOT -4.75% (sharp decline) yet gets same composite score. The model isn't distinguishing these opposite moves.
- **Adani group rally:** ADANIENT +2.48%, ADANIPORTS +1.92% — model says DOWN.

---

## Top 10 Stocks by Composite Score — Detailed Analyst Review

### 1. NTPC (Energy) — Composite: 42.88, DOWN, 22% confidence

```json
{
  "symbol": "NTPC",
  "ml_direction": -1,
  "ml_score": 42.88,
  "analyst_direction": 0,
  "analyst_adjustment": "+7",
  "analyst_score": 50,
  "conviction": "low",
  "reasoning": "ML DOWN signal is driven by cal_doy, not stock-specific factors. NTPC was flat yesterday (+0.15%) with no material news. Energy sector shows no directional catalyst. Adjusting to neutral — the model's own 22% confidence confirms it has no real edge on this stock today.",
  "key_risk": "No material risk flags. Insider trades data is 2 months stale — cannot assess insider sentiment.",
  "sector_context": "All 5 energy stocks have DOWN signals, but this is the cal_doy artifact. No genuine sector-wide bearish catalyst identified."
}
```

### 2. MARUTI (Auto) — Composite: 42.84, DOWN, 34% confidence

```json
{
  "symbol": "MARUTI",
  "ml_direction": -1,
  "ml_score": 42.84,
  "analyst_direction": 0,
  "analyst_adjustment": "+7",
  "analyst_score": 50,
  "conviction": "low",
  "reasoning": "ML says DOWN but MARUTI rallied +5.24% yesterday — strongest move in the Nifty 50. A contrarian reversal is plausible after such a sharp move, but the model's cal_doy-driven signal provides no stock-specific evidence. June auto sales data expected today (July 1 is monthly sales release day for auto OEMs) — this could be a genuine catalyst either way, but the model doesn't appear to be incorporating it.",
  "key_risk": "Post-rally mean reversion risk if June sales disappoint. But equally, strong sales could extend the move.",
  "sector_context": "Auto sector had a split session yesterday — MARUTI +5.24% vs EICHERMOT -4.75%. Model treats them identically."
}
```

### 3. BAJAJ-AUTO (Auto) — Composite: 42.84, DOWN, 15% confidence

```json
{
  "symbol": "BAJAJ-AUTO",
  "ml_direction": -1,
  "ml_score": 42.84,
  "analyst_direction": 0,
  "analyst_adjustment": "+7",
  "analyst_score": 50,
  "conviction": "low",
  "reasoning": "15% meta-confidence is among the lowest — the model has virtually no conviction. Stock was up +0.89% yesterday. June sales data due today. No material qualitative catalyst either way.",
  "key_risk": "Await monthly sales data before forming directional view.",
  "sector_context": "Auto sector dispersion was extreme yesterday (MARUTI +5.24% vs EICHERMOT -4.75%). Uniform DOWN call is not credible for the sector."
}
```

### 4. ONGC (Energy) — Composite: 42.76, DOWN, 14% confidence

```json
{
  "symbol": "ONGC",
  "ml_direction": -1,
  "ml_score": 42.76,
  "analyst_direction": 0,
  "analyst_adjustment": "+7",
  "analyst_score": 50,
  "conviction": "low",
  "reasoning": "14% confidence — model has no meaningful signal. ONGC essentially flat (+0.36%) yesterday. Oil prices and government policy on gas pricing are the real drivers, neither of which the cal_doy feature captures.",
  "key_risk": "Crude oil price volatility. Government fuel pricing decisions.",
  "sector_context": "Energy stocks flat to slightly positive yesterday. No sector-wide catalyst."
}
```

### 5. TATASTEEL (Metals) — Composite: 42.70, DOWN, 10% confidence

```json
{
  "symbol": "TATASTEEL",
  "ml_direction": -1,
  "ml_score": 42.70,
  "analyst_direction": 0,
  "analyst_adjustment": "+7",
  "analyst_score": 50,
  "conviction": "low",
  "reasoning": "Lowest meta-confidence of the top 10 at 10%. The model is essentially random-guessing on this stock. TATASTEEL was down -0.8% yesterday. Steel prices and China demand are the real drivers.",
  "key_risk": "China PMI data sensitivity. Global steel demand outlook.",
  "sector_context": "4 metals stocks, all DOWN by model, but ADANIENT +2.48% yesterday while TATASTEEL -0.8% — opposite moves getting identical signals."
}
```

### 6. ULTRACEMCO (Infra) — Composite: 42.66, DOWN, 18% confidence

```json
{
  "symbol": "ULTRACEMCO",
  "ml_direction": -1,
  "ml_score": 42.66,
  "analyst_direction": 0,
  "analyst_adjustment": "+7",
  "analyst_score": 50,
  "conviction": "low",
  "reasoning": "18% confidence, cal_doy-driven. ULTRACEMCO was down -0.73% yesterday. Cement demand is seasonal and tied to monsoon progress — July typically sees construction slowdown in India. However, the model is not making this nuanced seasonal argument; it's applying a blanket cal_doy signal.",
  "key_risk": "Monsoon-driven cement demand seasonality could create genuine headwinds, but this requires fundamental analysis, not a model artifact.",
  "sector_context": "Infra stocks mixed yesterday. No unified sector signal."
}
```

### 7. SBILIFE (Financial) — Composite: 42.64, DOWN, 21% confidence

```json
{
  "symbol": "SBILIFE",
  "ml_direction": -1,
  "ml_score": 42.64,
  "analyst_direction": 0,
  "analyst_adjustment": "+7",
  "analyst_score": 50,
  "conviction": "low",
  "reasoning": "Low-confidence DOWN signal. SBILIFE +0.59% yesterday. Insurance stocks are typically less volatile. No material qualitative catalyst.",
  "key_risk": "Q1 FY27 begins today — new quarter AUM reporting could move insurance stocks later this week.",
  "sector_context": "Financial sector (4 stocks) all have DOWN signals with 17-25% confidence. Not actionable."
}
```

### 8. BEL (Sector: Unknown) — Composite: 42.61, DOWN, 25% confidence

```json
{
  "symbol": "BEL",
  "ml_direction": -1,
  "ml_score": 42.61,
  "analyst_direction": 0,
  "analyst_adjustment": "+7",
  "analyst_score": 50,
  "conviction": "low",
  "reasoning": "Defence sector stock classified as 'Unknown' sector — metadata gap. BEL essentially flat (+0.05%). Defence order flow and government capex drive this stock. No material catalyst from model.",
  "key_risk": "Government defence spending decisions. Sector classified as 'Unknown' may mean sector-level analysis is incomplete.",
  "sector_context": "BEL is misclassified as 'Unknown' — should be in Defence/PSU sector for proper sector rotation analysis."
}
```

### 9. HINDALCO (Metals) — Composite: 42.60, DOWN, 24% confidence

```json
{
  "symbol": "HINDALCO",
  "ml_direction": -1,
  "ml_score": 42.60,
  "analyst_direction": 0,
  "analyst_adjustment": "+7",
  "analyst_score": 50,
  "conviction": "low",
  "reasoning": "24% confidence, cal_doy-driven. HINDALCO -0.75% yesterday. Aluminum prices and Novelis (US subsidiary) performance are key drivers. No material qualitative catalyst.",
  "key_risk": "Global aluminum demand. US tariff/trade policy affecting Novelis.",
  "sector_context": "Metals sector saw mixed moves yesterday. Model applies uniform DOWN."
}
```

### 10. SUNPHARMA (Pharma) — Composite: 42.60, DOWN, 23% confidence

```json
{
  "symbol": "SUNPHARMA",
  "ml_direction": -1,
  "ml_score": 42.60,
  "analyst_direction": 0,
  "analyst_adjustment": "+7",
  "analyst_score": 50,
  "conviction": "low",
  "reasoning": "23% confidence, cal_doy feature dominance. SUN PHARMA -0.66% yesterday. Pharma sector generally defensive. FDA approvals and USFDA inspection outcomes are real catalysts — none flagged in recent disclosures.",
  "key_risk": "USFDA regulatory actions. Currency (USD/INR) impact on export revenues.",
  "sector_context": "All 4 pharma stocks DOWN by model. APOLLOHOSP has highest confidence in entire Nifty 50 at 50%."
}
```

---

## Stocks with Highest Meta-Confidence (Closest to Tradeable)

While no Nifty 50 stock crosses the 0.55 threshold, these have relatively higher model conviction:

| Rank | Symbol | Meta-Confidence | Sector | Yesterday | Analyst View |
|---|---|---|---|---|---|
| 1 | APOLLOHOSP | 50.1% | Pharma | +0.31% | Neutral — borderline confidence, still cal_doy driven |
| 2 | SHRIRAMFIN | 47.9% | Infra | +0.92% | Neutral — misclassified as Infra (is NBFC/Financial) |
| 3 | ASIANPAINT | 45.5% | Unknown | -0.80% | Neutral — monsoon season could slow decorative paint demand |
| 4 | AXISBANK | 44.1% | Banking | -0.82% | Neutral — banking sector needs separate catalyst assessment |
| 5 | INDIGO | 43.1% | Unknown | +1.00% | Neutral — aviation demand remains strong |

---

## Notable Price Movements (Yesterday's Close)

Stocks that moved significantly but received identical model scores:

| Symbol | Change | ML Direction | Analyst Concern |
|---|---|---|---|
| MARUTI | **+5.24%** | DOWN | Strong rally contradicts DOWN; June sales data today is real catalyst |
| EICHERMOT | **-4.75%** | DOWN | Sharp decline — model agrees but for wrong reasons (cal_doy, not stock-specific) |
| INFY | **-3.50%** | DOWN | IT sector selloff; model coincidentally correct but via calendar, not fundamentals |
| TATACONSUM | **-3.34%** | DOWN | Same as above — right direction, wrong reasoning |
| TCS | **-3.17%** | DOWN | IT sector move |
| WIPRO | **-2.90%** | DOWN | IT sector move |
| HCLTECH | **-2.78%** | DOWN | IT sector move |
| TITAN | **+2.96%** | DOWN | Model calls DOWN on a stock that just rallied ~3% |
| ADANIENT | **+2.48%** | DOWN | Adani group rally contradicts model |
| BAJFINANCE | **+2.31%** | DOWN | NBFC rally contradicts model |

---

## Top Picks (Analyst-Reviewed)

### BUY Signals: NONE

**There are zero actionable BUY signals in today's Nifty 50 predictions.** The model produced no bullish calls, and the analyst review finds no qualitative basis to override to BUY on any stock.

### SELL Signals / Avoids: LOW CONVICTION

Given the cal_doy dominance, no high-conviction SELL signals exist either. However, the following stocks had genuine negative price action yesterday that could continue:

1. **EICHERMOT** (-4.75%) — Sharp decline may have technical follow-through, but this is a price-action observation, not an ML prediction. Monitor for continuation.
2. **INFY** (-3.50%) — IT sector under pressure. If this is a sector rotation (out of IT), may see continued weakness. But Q1 guidance season begins soon which could reverse sentiment.
3. **TATACONSUM** (-3.34%) — FMCG under pressure. Rural demand concerns could persist.

**These are observational notes based on yesterday's price action, NOT model-driven signals. Do not treat them as high-conviction shorts.**

---

## Risk Flags

1. **MODEL HEALTH — CRITICAL:** `cal_doy` dominance across all 50 stocks renders today's predictions non-actionable. Recommend investigating feature importance and potentially excluding or down-weighting calendar features in the ensemble.

2. **DATA PIPELINE — WARNING:** `insider_trades` table has not ingested data since 2026-04-28 (2+ months). This means insider buying clusters, insider selling patterns, and related qualitative signals are completely absent. Recommend urgent investigation of the 19:20 insider trades pipeline job.

3. **SECTOR MISCLASSIFICATION:** 9 stocks classified as "Unknown" sector (ASIANPAINT, INDIGO, TRENT, TMPV, MAXHEALTH, BEL, JIOFIN, TITAN, ETERNAL). This impairs sector rotation analysis. Recommend updating sector mappings.

4. **SHRIRAMFIN misclassified as Infra** — should be Financial/NBFC. Affects sector aggregation.

5. **ZERO DIFFERENTIATION:** The 1.13-point composite score range (41.75–42.88) across 50 stocks provides no actionable ranking. A healthy model would show a much wider distribution.

---

## Full Stock Review (All 50 Nifty Stocks)

*Sorted by composite_score descending. All stocks have cal_doy as top feature, sideways regime, and DOWN direction.*

```json
[
  {"symbol": "NTPC", "ml_direction": -1, "ml_score": 42.88, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — cal_doy artifact, 22% confidence, no material catalyst."},
  {"symbol": "MARUTI", "ml_direction": -1, "ml_score": 42.84, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — rallied +5.24% yesterday, June auto sales data today could be real catalyst."},
  {"symbol": "BAJAJ-AUTO", "ml_direction": -1, "ml_score": 42.84, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — 15% confidence, await June sales data."},
  {"symbol": "ONGC", "ml_direction": -1, "ml_score": 42.76, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — oil price dependent, no material catalyst."},
  {"symbol": "TATASTEEL", "ml_direction": -1, "ml_score": 42.70, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — 10% confidence, steel demand outlook is real driver."},
  {"symbol": "ULTRACEMCO", "ml_direction": -1, "ml_score": 42.66, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — monsoon season cement slowdown plausible but model doesn't capture it."},
  {"symbol": "SBILIFE", "ml_direction": -1, "ml_score": 42.64, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — insurance sector stable, Q1 FY27 AUM reporting upcoming."},
  {"symbol": "BEL", "ml_direction": -1, "ml_score": 42.61, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — defence/PSU stock misclassified as Unknown, no material catalyst."},
  {"symbol": "HINDALCO", "ml_direction": -1, "ml_score": 42.60, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — aluminum prices and Novelis are real drivers, model doesn't capture."},
  {"symbol": "SUNPHARMA", "ml_direction": -1, "ml_score": 42.60, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — pharma defensive, no FDA catalyst flagged."},
  {"symbol": "TATACONSUM", "ml_direction": -1, "ml_score": 42.60, "analyst_direction": 0, "analyst_adjustment": "+5", "analyst_score": 48, "conviction": "low", "one_liner": "Marginally bearish — fell -3.34% yesterday, rural demand concerns could persist."},
  {"symbol": "POWERGRID", "ml_direction": -1, "ml_score": 42.58, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — utility stock, low volatility expected."},
  {"symbol": "BAJAJFINSV", "ml_direction": -1, "ml_score": 42.50, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — holding company, tracks BAJFINANCE."},
  {"symbol": "DRREDDY", "ml_direction": -1, "ml_score": 42.49, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — pharma stock, USD/INR and FDA are real catalysts."},
  {"symbol": "RELIANCE", "ml_direction": -1, "ml_score": 42.49, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — 14% confidence, conglomerate with multiple business drivers."},
  {"symbol": "CIPLA", "ml_direction": -1, "ml_score": 42.46, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — pharma defensive, no material catalyst."},
  {"symbol": "BHARTIARTL", "ml_direction": -1, "ml_score": 42.43, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — telecom stable, tariff hike cycle may continue."},
  {"symbol": "ITC", "ml_direction": -1, "ml_score": 42.42, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — fell -1.29% yesterday, FMCG sector soft."},
  {"symbol": "JSWSTEEL", "ml_direction": -1, "ml_score": 42.41, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — steel demand outlook driver, model has 30% confidence."},
  {"symbol": "NESTLEIND", "ml_direction": -1, "ml_score": 42.41, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — FMCG defensive, +1.29% yesterday."},
  {"symbol": "LT", "ml_direction": -1, "ml_score": 42.39, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — infra capex story, order book is real driver."},
  {"symbol": "HDFCLIFE", "ml_direction": -1, "ml_score": 42.36, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — insurance defensive, no material catalyst."},
  {"symbol": "AXISBANK", "ml_direction": -1, "ml_score": 42.30, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — 44% confidence (highest among banks), but still cal_doy driven."},
  {"symbol": "TITAN", "ml_direction": -1, "ml_score": 42.29, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — rallied +2.96% yesterday, model DOWN call contradicts momentum."},
  {"symbol": "HINDUNILVR", "ml_direction": -1, "ml_score": 42.29, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — FMCG under pressure (-1.54%), rural demand key variable."},
  {"symbol": "M&M", "ml_direction": -1, "ml_score": 42.28, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — auto/farm equipment, June tractor sales data expected."},
  {"symbol": "COALINDIA", "ml_direction": -1, "ml_score": 42.28, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — 8% confidence (lowest), coal demand seasonal."},
  {"symbol": "ICICIBANK", "ml_direction": -1, "ml_score": 42.27, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — strong franchise, no material catalyst for direction."},
  {"symbol": "EICHERMOT", "ml_direction": -1, "ml_score": 42.26, "analyst_direction": 0, "analyst_adjustment": "+5", "analyst_score": 47, "conviction": "low", "one_liner": "Marginally bearish — fell -4.75% yesterday, sharpest decline in Nifty 50, watch for continuation."},
  {"symbol": "HDFCBANK", "ml_direction": -1, "ml_score": 42.26, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — largest bank, 14% confidence, no material catalyst."},
  {"symbol": "GRASIM", "ml_direction": -1, "ml_score": 42.25, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — cement/chemicals conglomerate, no material catalyst."},
  {"symbol": "KOTAKBANK", "ml_direction": -1, "ml_score": 42.24, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — 38% confidence, banking sector stable."},
  {"symbol": "SBIN", "ml_direction": -1, "ml_score": 42.22, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — PSU bank, government policy dependent."},
  {"symbol": "SHRIRAMFIN", "ml_direction": -1, "ml_score": 42.20, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — NBFC misclassified as Infra, +0.92% yesterday, 48% confidence."},
  {"symbol": "BAJFINANCE", "ml_direction": -1, "ml_score": 42.19, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — rallied +2.31% yesterday, NBFC leader, AUM growth driver."},
  {"symbol": "ASIANPAINT", "ml_direction": -1, "ml_score": 42.14, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — decorative paints, monsoon could slow demand but model doesn't capture it."},
  {"symbol": "JIOFIN", "ml_direction": -1, "ml_score": 42.14, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — Jio Financial, early-stage business, low confidence."},
  {"symbol": "APOLLOHOSP", "ml_direction": -1, "ml_score": 42.12, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — highest confidence (50%) but still below tradeable threshold, healthcare defensive."},
  {"symbol": "ADANIPORTS", "ml_direction": -1, "ml_score": 42.10, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — +1.92% yesterday (Adani group rally), port volumes are real driver."},
  {"symbol": "MAXHEALTH", "ml_direction": -1, "ml_score": 42.07, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — healthcare, -1.8% yesterday, no material catalyst."},
  {"symbol": "TECHM", "ml_direction": -1, "ml_score": 42.01, "analyst_direction": 0, "analyst_adjustment": "+5", "analyst_score": 47, "conviction": "low", "one_liner": "Marginally bearish — IT sector selling (-2.03%), sector rotation may continue."},
  {"symbol": "HCLTECH", "ml_direction": -1, "ml_score": 42.00, "analyst_direction": 0, "analyst_adjustment": "+5", "analyst_score": 47, "conviction": "low", "one_liner": "Marginally bearish — IT sector selling (-2.78%), sector rotation may continue."},
  {"symbol": "ETERNAL", "ml_direction": -1, "ml_score": 42.00, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — +2.0% yesterday, classified Unknown, no material catalyst."},
  {"symbol": "WIPRO", "ml_direction": -1, "ml_score": 42.00, "analyst_direction": 0, "analyst_adjustment": "+5", "analyst_score": 47, "conviction": "low", "one_liner": "Marginally bearish — IT sector selling (-2.9%), sector rotation may continue."},
  {"symbol": "TRENT", "ml_direction": -1, "ml_score": 41.97, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — retail/Zudio growth story, +0.75% yesterday."},
  {"symbol": "TCS", "ml_direction": -1, "ml_score": 41.92, "analyst_direction": 0, "analyst_adjustment": "+5", "analyst_score": 47, "conviction": "low", "one_liner": "Marginally bearish — IT bellwether fell -3.17%, Q1 results expected soon."},
  {"symbol": "ADANIENT", "ml_direction": -1, "ml_score": 41.91, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — rallied +2.48% yesterday, model DOWN contradicts momentum."},
  {"symbol": "INDIGO", "ml_direction": -1, "ml_score": 41.87, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — aviation demand strong, +1.0% yesterday, 43% confidence."},
  {"symbol": "INFY", "ml_direction": -1, "ml_score": 41.85, "analyst_direction": 0, "analyst_adjustment": "+3", "analyst_score": 45, "conviction": "low", "one_liner": "Marginally bearish — sharpest IT sector decline (-3.5%), Q1 guidance and results upcoming."},
  {"symbol": "TMPV", "ml_direction": -1, "ml_score": 41.75, "analyst_direction": 0, "analyst_adjustment": "+7", "analyst_score": 50, "conviction": "low", "one_liner": "Neutral — +2.07% yesterday, classified Unknown, no material catalyst."}
]
```

---

## Analyst Summary & Recommendations

### Today's Bottom Line

**Do not trade on today's ML predictions.** The model output is dominated by a calendar feature (`cal_doy`) that produces identical DOWN signals for all 50 Nifty stocks with near-zero differentiation. The model's own meta-confidence confirms this — zero stocks cross the tradeable threshold.

### Action Items (for the team, not for trading)

1. **URGENT — Investigate `cal_doy` feature dominance.** This feature should not be the top predictor for all 50 stocks simultaneously. Consider:
   - Adding a feature importance diversity check to the prediction pipeline
   - Setting a maximum feature contribution cap (e.g., no single feature can be >30% of prediction weight for >10% of stocks)
   - Reviewing if cal_doy has spurious correlation with past July 1 market moves

2. **URGENT — Fix `insider_trades` ingestion pipeline.** Last data: April 28, 2026. The 19:20 scheduled job has been failing for 2+ months without alerting.

3. **MODERATE — Fix sector misclassification.** 9 stocks in "Unknown" sector, SHRIRAMFIN misclassified as Infra. This impairs sector rotation analysis.

4. **LOW — Add model health checks to pre-market pipeline.** Auto-flag predictions when:
   - Same feature is #1 for >80% of stocks
   - Composite score range < 5 points
   - All stocks have same direction
   - 0 stocks meet tradeable threshold

---

*Briefing generated at 06:30 IST by AlphaVedha Analyst Agent*
*Next briefing: 2026-07-02 06:30 IST*
*Model version: v0.1.0 | Hash: f5c7b9208a36ee70...*
