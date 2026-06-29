# AlphaVedha Morning Briefing — 2026-06-29 (Monday)

---

## ⚠️ ANALYST ALERT: MODEL QUALITY DEGRADATION DETECTED

**Before reviewing individual stocks, the following systemic issues significantly reduce confidence in today's ML output:**

1. **cal_doy feature dominance**: "cal_doy" (calendar day of year) is the #1 feature for ALL 50 stocks. This is abnormal — it indicates the ensemble is driven by calendar seasonality rather than price action, volume, or microstructure signals. The model is essentially saying "this day of year is historically bearish" rather than reading each stock's individual setup.

2. **Uniform bearish output**: ALL 50 Nifty stocks show DOWN direction. Even in deep bear markets, sector rotation produces some bullish outliers. A 50/50 bearish call is a model degeneracy signal, not a market signal.

3. **Narrow score band**: Composite scores range only 42.13–46.0 (a 4-point spread across 50 stocks). The model is not differentiating between stocks — it's applying a near-uniform adjustment.

4. **Sector rotation contradiction**: The RRG analysis shows Banking (+5.6% 1-month) and Realty (+5.7% 1-month) in LEADING phase, yet all banking stocks are flagged DOWN. The ML model and the sector model disagree fundamentally.

5. **Stale data inputs**: 8 of 10 data tables are stale. Daily OHLCV missing Friday June 26 data. Insider trades haven't updated since April 28 (2 months). The model is making predictions on incomplete data.

**Analyst recommendation: Treat today's ML signals with LOW conviction. The tradeable signals (9 stocks above 0.55 meta-confidence) may be calendar-effect artifacts rather than genuine directional signals. Do not act on these signals without independent confirmation.**

---

## Market Context

| Metric | Value |
|--------|-------|
| Date | Monday, 2026-06-29 |
| Regime | Sideways (all stocks) |
| Total predictions | 59 (51 symbols) |
| Direction split | 8 bullish / 51 bearish / 0 neutral |
| Tradeable signals | 9 (meta-confidence > 0.55) |
| Hash published | Yes (sha256: eb0daba4bdc52a84...) |
| Nifty 50 benchmark | +0.6% 1-month, +5.4% 3-month |
| System status | Degraded (8 stale tables) |
| TATAMOTORS | 500 error — prediction unavailable |

### Data Freshness

| Table | Last Updated | Status | Gap |
|-------|-------------|--------|-----|
| paper_trades | 2026-06-29 | ✅ Current | — |
| prediction_proofs | 2026-06-29 | ✅ Current | — |
| daily_ohlcv | 2026-06-25 (Thu) | ❌ Stale | Missing Fri Jun 26 |
| institutional_flows | 2026-06-25 (Thu) | ❌ Stale | Missing Fri Jun 26 |
| disclosures | 2026-06-26 (Fri) | ⚠️ Stale | 3 days |
| insider_trades | 2026-04-28 | ❌ Stale | ~2 months |
| surveillance_flags | 2026-06-26 | ⚠️ Stale | 3 days |
| bulk_block_deals | 2026-06-25 | ❌ Stale | 4 days |
| rating_events | 2026-06-26 | ⚠️ Stale | 3 days |
| transcripts | 2026-06-26 | ⚠️ Stale | 3 days |

**Note:** Insider trades haven't refreshed since April 28. Any insider activity in the last 2 months is invisible to the model. This is a pipeline failure that needs investigation.

---

## Sector Rotation (RRG Analysis)

| Sector | Phase | RS-Ratio | Momentum | 1-Month | 3-Month |
|--------|-------|----------|----------|---------|---------|
| NIFTY_REALTY | **Leading** | 115.1 | 100.46 | +5.7% | +23.3% |
| NIFTY_BANK | **Leading** | 104.9 | 100.20 | +5.6% | +11.3% |
| NIFTY_INFRA | **Leading** | 103.1 | 100.09 | +0.5% | +8.5% |
| NIFTY_PSU_BANK | Improving | 99.8 | 100.07 | +5.3% | +4.7% |
| NIFTY_MEDIA | Weakening | 111.0 | 99.99 | +9.8% | +16.7% |
| NIFTY_METAL | Weakening | 109.8 | 99.55 | -7.8% | +11.5% |
| NIFTY_ENERGY | Weakening | 108.2 | 99.86 | -2.9% | +12.6% |
| NIFTY_AUTO | Weakening | 104.3 | 99.95 | +1.9% | +10.8% |
| NIFTY_PHARMA | Weakening | 103.5 | 99.92 | +1.3% | +10.7% |
| NIFTY_FMCG | Weakening | 101.1 | 99.84 | -1.6% | +6.4% |
| NIFTY_IT | **Lagging** | 89.8 | 99.29 | -5.7% | -7.5% |

---

## Sector Patterns (Cross-Stock Analysis)

### Pattern 1: Banking sector — ML vs RRG CONFLICT
All 5 banking stocks (AXISBANK, KOTAKBANK, ICICIBANK, SBIN, HDFCBANK) show ML DOWN signal. However, NIFTY_BANK is in the **Leading** RRG phase with +5.6% 1-month return and accelerating momentum (RS-Momentum 100.20). **This is a direct contradiction.** The sector rotation analysis — which uses relative strength methodology independent of the ML ensemble — suggests banking is the strongest sector. Analyst assessment: the ML bearish call on banking is likely driven by the cal_doy artifact, not genuine bearish setup. Banking stocks are neutral-to-bullish from a sector perspective.

### Pattern 2: IT sector — ML and RRG ALIGNED bearish
All 5 IT stocks (TCS, HCLTECH, TECHM, INFY, WIPRO) show DOWN with relatively low confidence (16-41%). NIFTY_IT is in the **Lagging** RRG phase at -5.7% 1-month. This is the one sector where ML and sector rotation agree. IT weakness is real and multi-model confirmed.

### Pattern 3: Metals sector — Weakening with recent sharp decline
4 Metals stocks (ADANIENT, JSWSTEEL, TATASTEEL, HINDALCO) all DOWN. NIFTY_METAL has -7.8% 1-month return and is in Weakening phase. ML and RRG partially aligned — metals have genuine weakness, though ADANIENT at 69.1% confidence may be overstated by the cal_doy effect.

### Pattern 4: Energy sector — Mixed signals
5 Energy stocks (POWERGRID, RELIANCE, COALINDIA, NTPC, ONGC) all DOWN with low confidence (10-34%). NIFTY_ENERGY is Weakening at -2.9% 1-month. The low ML confidence combined with sector weakening is consistent — mild bearish bias is warranted but no high-conviction shorts.

### Pattern 5: Pharma sector — Weakening but not broken
4 Pharma stocks (CIPLA, DRREDDY, SUNPHARMA, APOLLOHOSP) DOWN with moderate confidence. NIFTY_PHARMA is Weakening but still above benchmark (RS-Ratio 103.5, +1.3% 1-month). The high confidence on CIPLA (67.3%) seems overstated given the sector is still outperforming Nifty.

### Pattern 6: INDIGO outlier
INDIGO (IndiGo Airlines) had a +4.66% rally in the last session yet the ML is calling DOWN at 66.9% confidence. A strong gap-up into a DOWN signal can indicate mean-reversion after an outsized move. However, with cal_doy as the top feature, this may be coincidental. Monitor but don't act without confirming whether the rally was event-driven (results, order book, policy).

---

## Red Flags / Blowup Scores

No stocks triggered the blowup threshold (default 70). All 20 checked Nifty 50 stocks have total blowup score of **0** across all components (pledge, rating, governance, default, surveillance, beneish, insider_sell, pump).

**Caveat:** Insider trades data is 2 months stale (last: April 28). Any recent insider selling activity is NOT reflected in the blowup scores. Surveillance flags and disclosures are also 3 days stale.

---

## Top 10 Stocks — Detailed Analyst Review

### 1. ASIANPAINT (Asian Paints)

```json
{
  "symbol": "ASIANPAINT",
  "ml_direction": -1,
  "ml_score": 46.0,
  "ml_confidence": 83.8,
  "analyst_direction": 0,
  "analyst_adjustment": "-5 confidence",
  "analyst_score": 46.0,
  "conviction": "low",
  "reasoning": "Highest ML confidence at 83.8% but driven by cal_doy, not stock-specific signals. Price at 2645 with -0.84% last session. Sector classified as 'Unknown' which means no sector rotation context is available. The 83.8% confidence looks strong on paper but with a calendar feature dominating, this is an artifact, not a signal. FMCG sector (closest proxy) is Weakening. No blowup risk (score 0). Without a genuine stock-specific catalyst, downgrading conviction from the ML-implied high to low.",
  "key_risk": "cal_doy artifact inflating confidence. No sector rotation context for validation.",
  "sector_context": "No sector classification available. FMCG proxy is Weakening (RS 101.1, -1.6% 1m)."
}
```

### 2. GRASIM (Grasim Industries)

```json
{
  "symbol": "GRASIM",
  "ml_direction": -1,
  "ml_score": 45.64,
  "ml_confidence": 70.4,
  "analyst_direction": -1,
  "analyst_adjustment": "0",
  "analyst_score": 45.64,
  "conviction": "low",
  "reasoning": "Infra sector stock with 70.4% ML confidence. However, NIFTY_INFRA is in Leading phase (RS 103.1) which contradicts the bearish call. Price flat at -0.04% last session. cal_doy driving the signal. Given sector strength, the bearish call lacks support. Maintaining ML direction but at low conviction — no material qualitative catalyst to override either way.",
  "key_risk": "Infra sector is Leading on RRG — bearish call contradicts sector momentum.",
  "sector_context": "Infra: 3 of 4 Infra stocks (GRASIM, ADANIPORTS, LT, ULTRACEMCO) have DOWN signals despite sector being in Leading phase. This is a major ML-vs-sector conflict."
}
```

### 3. ADANIENT (Adani Enterprises)

```json
{
  "symbol": "ADANIENT",
  "ml_direction": -1,
  "ml_score": 44.23,
  "ml_confidence": 69.1,
  "analyst_direction": -1,
  "analyst_adjustment": "+3",
  "analyst_score": 47.23,
  "conviction": "medium-low",
  "reasoning": "Classified as Metals sector. NIFTY_METAL is Weakening with -7.8% 1-month return — the worst sector return on the board. ML bearish call aligns with sector weakness. Adani group stocks have additional governance scrutiny (though blowup score is 0 currently — caveat: insider data is 2 months stale). Price -1.03% last session. Slight score upgrade for sector alignment, but cal_doy concern keeps conviction medium-low.",
  "key_risk": "Metals sector has the sharpest 1-month decline (-7.8%). Insider data 2 months stale — cannot assess recent promoter activity.",
  "sector_context": "Metals weakening: 4/4 metals stocks (ADANIENT, JSWSTEEL, TATASTEEL, HINDALCO) bearish. Sector momentum and ML aligned."
}
```

### 4. AXISBANK (Axis Bank)

```json
{
  "symbol": "AXISBANK",
  "ml_direction": -1,
  "ml_score": 45.2,
  "ml_confidence": 67.8,
  "analyst_direction": 0,
  "analyst_adjustment": "+5",
  "analyst_score": 50.2,
  "conviction": "low",
  "reasoning": "ML calls DOWN at 67.8% confidence, but NIFTY_BANK is in LEADING phase with +5.6% 1-month return. This is the strongest sector on the RRG chart. A bearish call on a banking stock when the sector is leading requires strong stock-specific evidence, which is absent — cal_doy is the top feature. Adjusting score toward neutral. No material qualitative catalyst for a bearish view on banking right now.",
  "key_risk": "ML-sector conflict: Banking is Leading, ML says DOWN. cal_doy artifact likely.",
  "sector_context": "Banking: All 5 banking stocks bearish on ML, but sector is LEADING. PSU Banks are Improving. Analyst: Banking bearish calls are unreliable today."
}
```

### 5. CIPLA (Cipla)

```json
{
  "symbol": "CIPLA",
  "ml_direction": -1,
  "ml_score": 45.19,
  "ml_confidence": 67.3,
  "analyst_direction": -1,
  "analyst_adjustment": "0",
  "analyst_score": 45.19,
  "conviction": "low",
  "reasoning": "Pharma sector, NIFTY_PHARMA is Weakening but still above benchmark (RS 103.5). Price +0.15% last session — largely flat. 67.3% ML confidence is the highest among pharma names. No material qualitative catalyst. Maintaining ML direction at low conviction. Pharma weakness is real but moderate, and cal_doy dominance undermines the specific confidence level.",
  "key_risk": "Pharma sector weakening slowly. No stock-specific red flags.",
  "sector_context": "Pharma: 4/4 pharma stocks bearish. Sector Weakening but still above Nifty. Moderate alignment."
}
```

### 6. INDIGO (InterGlobe Aviation)

```json
{
  "symbol": "INDIGO",
  "ml_direction": -1,
  "ml_score": 43.63,
  "ml_confidence": 66.9,
  "analyst_direction": 0,
  "analyst_adjustment": "+7",
  "analyst_score": 50.63,
  "conviction": "low",
  "reasoning": "Noteworthy outlier: INDIGO rallied +4.66% in the last session — the largest single-day gain in today's scan. ML calls DOWN at 66.9%, which could be a mean-reversion signal after a gap-up. However, cal_doy as top feature means this is NOT reading the gap-up — it's the same calendar artifact affecting all stocks. Without knowing what drove the 4.66% rally (earnings? order book? policy?), the DOWN call after a strong rally is uncertain. Adjusting to neutral — the rally could be the start of a trend or a one-day pop. No data to differentiate.",
  "key_risk": "4.66% rally could be event-driven (earnings/order book) — data staleness means we can't verify. Mean-reversion possible but not confirmed.",
  "sector_context": "Aviation has no dedicated RRG sector. No sector context available."
}
```

### 7. ADANIPORTS (Adani Ports)

```json
{
  "symbol": "ADANIPORTS",
  "ml_direction": -1,
  "ml_score": 45.38,
  "ml_confidence": 65.7,
  "analyst_direction": 0,
  "analyst_adjustment": "+5",
  "analyst_score": 50.38,
  "conviction": "low",
  "reasoning": "Infra sector stock. NIFTY_INFRA is in Leading phase — contradicts bearish ML call. Price -0.95% last session. Adani group stock carries elevated perception risk but blowup score is 0. Adjusting toward neutral given sector-ML conflict. No material qualitative catalyst either way.",
  "key_risk": "Infra is Leading on RRG. ML bearish call is sector-contradicted. Adani group perception risk is a persistent factor.",
  "sector_context": "Infra: Leading phase. ML bearish calls on all Infra stocks are likely cal_doy artifacts."
}
```

### 8. TITAN (Titan Company)

```json
{
  "symbol": "TITAN",
  "ml_direction": -1,
  "ml_score": 43.38,
  "ml_confidence": 64.7,
  "analyst_direction": -1,
  "analyst_adjustment": "0",
  "analyst_score": 43.38,
  "conviction": "low",
  "reasoning": "Consumer discretionary / retail. Sector classified as 'Unknown'. Price -0.75% last session. No sector rotation context available. No blowup risk. cal_doy driving the signal. Maintaining ML direction but at low conviction — no data to confirm or contradict.",
  "key_risk": "Consumer discretionary spending trends unknown. No sector classification for RRG validation.",
  "sector_context": "No sector context available."
}
```

### 9. LT (Larsen & Toubro)

```json
{
  "symbol": "LT",
  "ml_direction": -1,
  "ml_score": 44.79,
  "ml_confidence": 63.4,
  "analyst_direction": 0,
  "analyst_adjustment": "+5",
  "analyst_score": 49.79,
  "conviction": "low",
  "reasoning": "Major Infra bellwether. NIFTY_INFRA is in Leading phase (RS 103.1). Price +0.83% last session — positive momentum. ML calls DOWN at 63.4% confidence driven by cal_doy. Given sector leadership and positive price action, adjusting toward neutral. LT is the most liquid Infra proxy — if Infra is leading, LT should not be bearish without stock-specific catalysts.",
  "key_risk": "ML-sector conflict. Infra is Leading. No stock-specific bearish catalyst identified.",
  "sector_context": "Infra: Leading phase. LT is the sector bellwether — bearish call against a leading sector is low-confidence."
}
```

### 10. KOTAKBANK (Kotak Mahindra Bank)

```json
{
  "symbol": "KOTAKBANK",
  "ml_direction": -1,
  "ml_score": 45.12,
  "ml_confidence": 62.5,
  "analyst_direction": 0,
  "analyst_adjustment": "+5",
  "analyst_score": 50.12,
  "conviction": "low",
  "reasoning": "Banking stock in Leading sector. Same issue as AXISBANK — ML bearish while sector is strongest on RRG. Price +0.75% last session — positive. Adjusting to neutral. Banking bearish calls are unreliable today due to cal_doy artifact and sector conflict.",
  "key_risk": "Banking is Leading. ML bearish call is sector-contradicted.",
  "sector_context": "Banking: Leading phase. All banking stocks ML-bearish but sector-bullish. Contradiction."
}
```

---

## Remaining Stocks (11-50) — Brief Assessment

```json
[
  {"symbol": "DRREDDY", "ml_direction": -1, "ml_confidence": 60.2, "ml_score": 44.67, "analyst_direction": -1, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Pharma weakening, price +1.66% last session. No material catalyst. Low conviction bearish."},
  {"symbol": "TRENT", "ml_direction": -1, "ml_confidence": 58.7, "ml_score": 43.73, "analyst_direction": -1, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Retail/consumer. Price -0.95%. No sector context. cal_doy driven. Low conviction."},
  {"symbol": "HINDUNILVR", "ml_direction": -1, "ml_confidence": 57.1, "ml_score": 43.98, "analyst_direction": -1, "analyst_adjustment": "0", "conviction": "low", "one_liner": "FMCG weakening (RS 101.1, -1.6% 1m). ML and sector mildly aligned. No material catalyst."},
  {"symbol": "JSWSTEEL", "ml_direction": -1, "ml_confidence": 55.8, "ml_score": 45.27, "analyst_direction": -1, "analyst_adjustment": "+2", "conviction": "low-medium", "one_liner": "Metals weakening (-7.8% 1m). ML-sector aligned. Slight confidence bump for sector confirmation."},
  {"symbol": "ICICIBANK", "ml_direction": -1, "ml_confidence": 54.2, "ml_score": 45.23, "analyst_direction": 0, "analyst_adjustment": "+5", "conviction": "low", "one_liner": "Banking Leading sector contradicts ML bearish. Adjusting to neutral. No stock-specific catalyst."},
  {"symbol": "BAJAJFINSV", "ml_direction": -1, "ml_confidence": 54.1, "ml_score": 44.12, "analyst_direction": -1, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Financial services. Price -0.89%. No material catalyst either way."},
  {"symbol": "SBIN", "ml_direction": -1, "ml_confidence": 51.6, "ml_score": 44.68, "analyst_direction": 0, "analyst_adjustment": "+5", "conviction": "low", "one_liner": "Banking Leading + PSU Bank Improving. ML bearish contradicts two sector signals. Neutral."},
  {"symbol": "M&M", "ml_direction": -1, "ml_confidence": 51.2, "ml_score": 43.73, "analyst_direction": 0, "analyst_adjustment": "+6", "conviction": "low", "one_liner": "Auto weakening but +3.84% rally last session. Rally + sector momentum conflict with bearish call. Neutral."},
  {"symbol": "SUNPHARMA", "ml_direction": -1, "ml_confidence": 49.9, "ml_score": 44.96, "analyst_direction": -1, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Pharma weakening. Price -0.62%. No material catalyst. Coin-flip confidence (49.9%)."},
  {"symbol": "BHARTIARTL", "ml_direction": -1, "ml_confidence": 49.6, "ml_score": 44.23, "analyst_direction": -1, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Telecom. Price -1.42%. No dedicated telecom sector in RRG. Near coin-flip confidence."},
  {"symbol": "APOLLOHOSP", "ml_direction": -1, "ml_confidence": 48.9, "ml_score": 45.0, "analyst_direction": 0, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Healthcare/Pharma. Price +0.22%. Below coin-flip confidence. Neutral — no signal."},
  {"symbol": "TMPV", "ml_direction": -1, "ml_confidence": 46.9, "ml_score": 43.75, "analyst_direction": 0, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Unknown sector. Price +1.0%. Below threshold. No meaningful signal."},
  {"symbol": "NESTLEIND", "ml_direction": -1, "ml_confidence": 45.3, "ml_score": 45.24, "analyst_direction": -1, "analyst_adjustment": "0", "conviction": "low", "one_liner": "FMCG weakening. Price +1.45%. Low confidence, mildly aligned with sector weakness."},
  {"symbol": "SBILIFE", "ml_direction": -1, "ml_confidence": 44.4, "ml_score": 44.01, "analyst_direction": -1, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Financial/insurance. Price -1.29%. No material catalyst. Low confidence."},
  {"symbol": "EICHERMOT", "ml_direction": -1, "ml_confidence": 42.5, "ml_score": 43.26, "analyst_direction": 0, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Auto weakening but confidence below threshold. Price +0.34%. No signal."},
  {"symbol": "TATACONSUM", "ml_direction": -1, "ml_confidence": 41.3, "ml_score": 43.6, "analyst_direction": 0, "analyst_adjustment": "0", "conviction": "low", "one_liner": "FMCG. Price +3.02% rally. Below confidence threshold. No signal."},
  {"symbol": "TCS", "ml_direction": -1, "ml_confidence": 40.6, "ml_score": 42.34, "analyst_direction": -1, "analyst_adjustment": "+2", "conviction": "low", "one_liner": "IT lagging (-5.7% 1m). ML and sector aligned bearish. Low confidence but direction credible."},
  {"symbol": "MAXHEALTH", "ml_direction": -1, "ml_confidence": 40.3, "ml_score": 42.94, "analyst_direction": 0, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Healthcare. Price +3.85% rally. Below threshold. No signal from ML."},
  {"symbol": "HCLTECH", "ml_direction": -1, "ml_confidence": 39.1, "ml_score": 42.99, "analyst_direction": -1, "analyst_adjustment": "+2", "conviction": "low", "one_liner": "IT lagging. ML and sector aligned bearish. Low ML confidence but IT weakness is multi-model confirmed."},
  {"symbol": "SHRIRAMFIN", "ml_direction": -1, "ml_confidence": 38.8, "ml_score": 43.42, "analyst_direction": 0, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Financial/Infra. Price +1.26%. Below threshold. No meaningful signal."},
  {"symbol": "BEL", "ml_direction": -1, "ml_confidence": 38.7, "ml_score": 44.29, "analyst_direction": 0, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Defence. Price -1.54%. Below threshold. No meaningful signal."},
  {"symbol": "JIOFIN", "ml_direction": -1, "ml_confidence": 37.9, "ml_score": 43.89, "analyst_direction": 0, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Financial/Jio. Price +0.23%. Below threshold. No signal."},
  {"symbol": "TECHM", "ml_direction": -1, "ml_confidence": 37.5, "ml_score": 43.0, "analyst_direction": -1, "analyst_adjustment": "+2", "conviction": "low", "one_liner": "IT lagging. ML and sector aligned. Low confidence but IT weakness real."},
  {"symbol": "ULTRACEMCO", "ml_direction": -1, "ml_confidence": 37.2, "ml_score": 42.41, "analyst_direction": 0, "analyst_adjustment": "+3", "conviction": "low", "one_liner": "Infra/Cement. Infra Leading on RRG — ML contradicts. Adjusting neutral."},
  {"symbol": "BAJAJ-AUTO", "ml_direction": -1, "ml_confidence": 36.4, "ml_score": 42.93, "analyst_direction": 0, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Auto weakening but low confidence. Price +0.95%. No signal."},
  {"symbol": "POWERGRID", "ml_direction": -1, "ml_confidence": 33.8, "ml_score": 43.71, "analyst_direction": -1, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Energy weakening (-2.9% 1m). Price -2.41%. ML and sector mildly aligned."},
  {"symbol": "RELIANCE", "ml_direction": -1, "ml_confidence": 32.6, "ml_score": 44.06, "analyst_direction": 0, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Energy weakening but low confidence. Price +0.34%. Near-noise territory."},
  {"symbol": "TATASTEEL", "ml_direction": -1, "ml_confidence": 32.2, "ml_score": 43.97, "analyst_direction": -1, "analyst_adjustment": "+2", "conviction": "low", "one_liner": "Metals weakening (-7.8% 1m). ML and sector aligned. Low confidence but direction credible."},
  {"symbol": "BAJFINANCE", "ml_direction": -1, "ml_confidence": 31.9, "ml_score": 43.7, "analyst_direction": 0, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Financial. Price -1.06%. Low confidence. No signal."},
  {"symbol": "MARUTI", "ml_direction": -1, "ml_confidence": 31.2, "ml_score": 42.69, "analyst_direction": 0, "analyst_adjustment": "+5", "conviction": "low", "one_liner": "Auto. Price +3.75% strong rally. Low ML confidence + rally = no bearish conviction. Neutral."},
  {"symbol": "HDFCBANK", "ml_direction": -1, "ml_confidence": 30.0, "ml_score": 44.3, "analyst_direction": 0, "analyst_adjustment": "+5", "conviction": "low", "one_liner": "Banking Leading. ML weak bearish contradicted by sector. Adjusting neutral."},
  {"symbol": "HDFCLIFE", "ml_direction": -1, "ml_confidence": 28.8, "ml_score": 43.51, "analyst_direction": 0, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Insurance/Financial. Price -1.06%. Low confidence. No signal."},
  {"symbol": "INFY", "ml_direction": -1, "ml_confidence": 27.1, "ml_score": 42.88, "analyst_direction": -1, "analyst_adjustment": "+2", "conviction": "low", "one_liner": "IT lagging. ML and sector aligned. Very low confidence but IT sector weakness confirmed."},
  {"symbol": "ITC", "ml_direction": -1, "ml_confidence": 23.4, "ml_score": 43.67, "analyst_direction": 0, "analyst_adjustment": "0", "conviction": "low", "one_liner": "FMCG weakening but very low ML confidence. Price -0.12%. No signal."},
  {"symbol": "ETERNAL", "ml_direction": -1, "ml_confidence": 22.3, "ml_score": 43.16, "analyst_direction": 0, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Unknown sector. Price -0.47%. Very low confidence. No signal."},
  {"symbol": "HINDALCO", "ml_direction": -1, "ml_confidence": 21.9, "ml_score": 43.51, "analyst_direction": -1, "analyst_adjustment": "+2", "conviction": "low", "one_liner": "Metals weakening. Price -2.4%. Sector and ML aligned. Very low ML confidence."},
  {"symbol": "COALINDIA", "ml_direction": -1, "ml_confidence": 21.8, "ml_score": 44.08, "analyst_direction": -1, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Energy weakening. Price -1.44%. Low confidence but sector mildly bearish."},
  {"symbol": "NTPC", "ml_direction": -1, "ml_confidence": 21.5, "ml_score": 43.52, "analyst_direction": -1, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Energy weakening. Price -1.4%. Low confidence. Mild bearish bias from sector."},
  {"symbol": "WIPRO", "ml_direction": -1, "ml_confidence": 16.0, "ml_score": 42.13, "analyst_direction": -1, "analyst_adjustment": "+2", "conviction": "low", "one_liner": "IT lagging. Lowest ML confidence in Nifty 50 but IT sector weakness is genuine."},
  {"symbol": "ONGC", "ml_direction": -1, "ml_confidence": 9.9, "ml_score": 42.8, "analyst_direction": 0, "analyst_adjustment": "0", "conviction": "low", "one_liner": "Energy. Price -2.87%. Lowest confidence (9.9%) — effectively no ML signal."},
  {"symbol": "TATAMOTORS", "ml_direction": "N/A", "ml_confidence": "N/A", "ml_score": "N/A", "analyst_direction": 0, "analyst_adjustment": "N/A", "conviction": "none", "one_liner": "API 500 error — prediction unavailable. Investigate pipeline failure."}
]
```

---

## Top Picks — Analyst Reviewed

### Stocks where ML direction has sector confirmation (highest conviction today):

**There are NO high-conviction BUY signals today.** All ML predictions are bearish and none are upgraded to BUY by analyst review.

### Most credible bearish signals (ML + sector aligned):

1. **IT sector (TCS, HCLTECH, INFY, TECHM, WIPRO)**: ML bearish + IT Lagging on RRG (-5.7% 1-month). Low individual ML confidence but multi-model sector confirmation. Best bearish theme today if you must trade.

2. **Metals (ADANIENT, JSWSTEEL, TATASTEEL, HINDALCO)**: ML bearish + Metals Weakening (-7.8% 1-month). Sharpest sector decline. ADANIENT has highest individual confidence (69.1%) in this group.

3. **JSWSTEEL**: Slight analyst upgrade (+2) for ML-sector alignment. Metals has the worst 1-month return of any sector.

### Top Shorts / Avoids:

No stocks on the blowup avoid list (all scores = 0). However, with 2-month-stale insider data, this is unreliable.

---

## Risk Flags

1. **MODEL DEGENERACY**: cal_doy dominance across all 50 stocks. Today's ML predictions should not be trusted at face value. This requires engineering investigation — is the model overfitting to seasonal features, or is a feature pipeline bug zeroing out all other features?

2. **DATA PIPELINE FAILURE**: 8 of 10 tables stale. Insider trades 2 months stale. Daily OHLCV missing Friday June 26 data. The model is predicting with incomplete inputs.

3. **TATAMOTORS 500 ERROR**: Individual prediction endpoint returns Internal Server Error. This stock is missing from today's analysis.

4. **SECTOR-ML CONFLICT**: Banking (5 stocks) and Infra (4 stocks) show ML bearish while their sectors are in Leading RRG phase. 9 of 50 stocks have contradictory signals between ML and sector analysis.

5. **FALSE CONFIDENCE WARNING**: 9 stocks appear "tradeable" (meta-confidence > 0.55) but this confidence is driven by calendar features, not market signals. The real number of tradeable signals today is closer to 0.

6. **NO BULLISH SIGNALS IN NIFTY 50**: The summary endpoint reports 8 bullish predictions total, but none appear in the Nifty 50 large-cap scan. All 50 Nifty stocks are bearish — a statistically improbable outcome that further supports model degeneracy.

---

## Action Items for Engineering

1. **Investigate cal_doy feature dominance** — Why is calendar day of year the #1 feature for all 50 stocks? Check if other features are being computed correctly or if there's a pipeline bug setting them to NaN/zero.

2. **Fix data freshness** — Daily OHLCV and institutional flows pipelines did not run for Friday June 26. Insider trades pipeline has been broken since April 28.

3. **Fix TATAMOTORS prediction** — 500 error on /predict/TATAMOTORS. Likely a symbol mapping or data availability issue.

4. **Audit sector classification** — 9 of 50 stocks have sector "Unknown". This limits cross-stock analysis quality. These should be classified: ASIANPAINT, INDIGO, TITAN, TRENT, TMPV, MAXHEALTH, BEL, JIOFIN, ETERNAL.

---

*Briefing generated at 2026-06-29 06:35 IST by AlphaVedha Analyst Agent*
*Next briefing: 2026-06-30 (Tuesday) at 06:00 IST*
