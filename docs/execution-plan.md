# AlphaVedha — Execution Plan (Phase A → B → C)

## Current State (May 2026)

- 50 Nifty 50 stocks ingested (80,264 OHLCV rows, Jan 2020 — May 2026)
- 142 features (126 real, 16 stubs returning NaN)
- XGBoost trained (val accuracy: 46.6%)
- LSTM, TFT, Regime, Ensemble, Meta-labeling, Conformal — code written, not yet trained
- Tables empty: institutional_flows, derivatives_data, corporate_actions, features
- Modules empty: signals/, monitoring/, fundamental/

---

## PHASE A — Immediate Impact (2-3 weeks)

**Goal:** Fill the data gaps that cripple prediction accuracy, wire the missing features, and prove the system works with walk-forward backtesting.

---

### A1. FII/DII Flow Data Ingestion

**Why:** FII buying/selling drives 60-70% of Nifty 50 moves. This data is free and public but our `institutional_flows` table is empty.

**Data Source:** NSE daily FII/DII reports (nse-india.com)

**Files to create/modify:**
| File | Action |
|------|--------|
| `alphavedha/data/providers/nse_provider.py` | CREATE — fetch daily FII/DII buy/sell/net values from NSE |
| `alphavedha/data/ingestion.py` | MODIFY — add `ingest_fii_dii()` function |
| `alphavedha/cli/main.py` | MODIFY — add `data fii-refresh` command |
| `tests/unit/data/test_nse_provider.py` | CREATE — test with recorded fixtures |

**Features to activate (currently NaN stubs):**
| Feature | File | What it captures |
|---------|------|-----------------|
| `fii_net_flow_5d` | `features/derivatives.py` | 5-day cumulative FII net |
| `fii_net_flow_10d` | `features/derivatives.py` | 10-day cumulative FII net |
| `dii_net_flow_5d` | `features/derivatives.py` | 5-day cumulative DII net |
| `fii_dii_divergence` | `features/derivatives.py` | FII buying while DII selling (or vice versa) |
| `fii_buying_streak` | `features/derivatives.py` | Consecutive days of FII net positive |

**Acceptance criteria:**
- [ ] `institutional_flows` table populated with last 5 years of FII/DII data
- [ ] `alphavedha data fii-refresh` fetches latest day's data
- [ ] Features return real values instead of NaN
- [ ] Unit tests pass with recorded fixture data

---

### A2. F&O Data Ingestion

**Why:** India's options market is the world's largest. Futures OI, PCR, max pain — all forward-looking signals that predict where price will go.

**Data Source:** NSE F&O bhavcopy (daily), options chain API

**Files to create/modify:**
| File | Action |
|------|--------|
| `alphavedha/data/providers/nse_provider.py` | MODIFY — add `fetch_derivatives()`, `fetch_options_chain()` |
| `alphavedha/data/ingestion.py` | MODIFY — add `ingest_derivatives()` function |
| `alphavedha/cli/main.py` | MODIFY — add `data derivatives-refresh` command |

**Features to activate (currently NaN stubs):**
| Feature | File | What it captures |
|---------|------|-----------------|
| `futures_oi_change_pct` | `features/derivatives.py` | % change in futures open interest |
| `futures_basis` | `features/derivatives.py` | Futures price - spot price (premium/discount) |
| `pcr_oi` | `features/derivatives.py` | Put-Call Ratio (OI based) |
| `pcr_volume` | `features/derivatives.py` | Put-Call Ratio (volume based) |
| `max_pain_distance` | `features/derivatives.py` | How far current price is from max pain |
| `iv_percentile` | `features/derivatives.py` | Current IV vs last 252 days |
| `oi_concentration` | `features/derivatives.py` | Where is OI concentrated (which strikes) |

**Acceptance criteria:**
- [ ] `derivatives_data` table populated for F&O stocks
- [ ] PCR, max pain, OI change features return real values
- [ ] Backfill at least 2 years of F&O data
- [ ] Unit tests pass

---

### A3. Delivery Percentage Signals

**Why:** India-unique metric. High delivery % + price move = genuine institutional conviction. Currently `delivery_pct` column exists in OHLCV but features don't fully use it.

**Data Source:** Already have jugaad-data provider for delivery %. Need to backfill and build better features.

**Files to modify:**
| File | Action |
|------|--------|
| `alphavedha/features/microstructure.py` | MODIFY — add delivery-based features |
| `alphavedha/data/providers/jugaad_provider.py` | VERIFY — delivery_pct ingestion works |

**New features to add:**
| Feature | What it captures |
|---------|-----------------|
| `delivery_pct_zscore` | Current delivery % vs 20-day mean (in std devs) |
| `delivery_pct_rank` | Percentile rank in last 60 days |
| `delivery_volume_combo` | delivery_pct > 2σ AND volume > 1.5x avg = accumulation signal |
| `delivery_trend_5d` | Is delivery % increasing over last 5 days? |
| `high_delivery_breakout` | delivery > 60% + close > previous high = conviction breakout |

**Acceptance criteria:**
- [ ] delivery_pct column populated for all symbols
- [ ] 5 new delivery-based features returning real values
- [ ] Features show statistical significance in univariate analysis vs future returns

---

### A4. Earnings Surprise Tracker

**Why:** Post-Earnings Announcement Drift (PEAD) is the most proven anomaly in finance. Stocks that beat estimates outperform for 60-90 days. Works even better in India due to sparse analyst coverage on mid-caps.

**Data Source:** BSE/NSE corporate announcements, Screener.in API, Trendlyne

**Files to create:**
| File | Action |
|------|--------|
| `alphavedha/data/providers/earnings_provider.py` | CREATE — fetch quarterly results + consensus estimates |
| `alphavedha/data/models.py` | MODIFY — add `EarningsResult` ORM model |
| `alphavedha/features/fundamental_features.py` | CREATE — earnings surprise features |
| `alphavedha/data/ingestion.py` | MODIFY — add `ingest_earnings()` |

**New database table:**
```
earnings_results (
    symbol, quarter, year, 
    revenue_actual, revenue_estimate, revenue_surprise_pct,
    profit_actual, profit_estimate, profit_surprise_pct,
    announced_date
)
```

**New features:**
| Feature | What it captures |
|---------|-----------------|
| `earnings_surprise_pct` | (actual - estimate) / estimate |
| `days_since_earnings` | How many days since last result |
| `earnings_surprise_streak` | Consecutive beats or misses |
| `revenue_growth_qoq` | Quarter-over-quarter revenue growth |
| `profit_margin_change` | Margin expansion or compression |

**Acceptance criteria:**
- [ ] Last 3 years of quarterly results for Nifty 50 stocks
- [ ] Earnings surprise features computed correctly
- [ ] Backtest shows positive drift after earnings beats

---

### A5. Regime-Conditional Strategy Selection

**Why:** Using the same model in bull and bear markets is like wearing the same clothes in summer and winter. Each regime needs its own playbook.

**Files to create/modify:**
| File | Action |
|------|--------|
| `alphavedha/prediction/regime_strategy.py` | CREATE — strategy selector based on regime |
| `alphavedha/prediction/engine.py` | MODIFY — integrate regime-conditional logic |
| `alphavedha/config.py` | MODIFY — add per-regime config (position sizes, model weights) |

**Implementation:**
```
Regime: Bull trending
  → Weight: momentum features high, mean-reversion low
  → Position size: Full Kelly
  → Models: Trust XGBoost + TFT more (trend-following)

Regime: Bear trending
  → Weight: mean-reversion high, momentum low
  → Position size: Quarter Kelly
  → Models: Trust LSTM more (reversal detection)
  → Extra filter: only trade if meta-label > 0.65

Regime: Sideways
  → Strategy: Pairs trading + delivery-based only
  → Position size: Half Kelly
  → Models: Equal weight

Regime: High Volatility
  → Strategy: Cash or fully hedged
  → Position size: Minimum (10% of normal)
  → Only trade if ALL models agree
```

**Acceptance criteria:**
- [ ] Regime detector output drives strategy selection
- [ ] Different position sizing per regime
- [ ] Backtest shows better risk-adjusted returns vs single-strategy

---

### A6. Walk-Forward Backtest (Proof It Works)

**Why:** The only honest way to prove the system works. Simulates real-world usage — no hindsight.

**Files to create:**
| File | Action |
|------|--------|
| `alphavedha/backtest/walk_forward.py` | CREATE — walk-forward backtesting engine |
| `alphavedha/cli/main.py` | MODIFY — add `backtest walk-forward` command |
| `scripts/run_walk_forward.py` | CREATE — full walk-forward backtest script |

**Implementation:**
```
For each month from Jan 2024 to May 2026 (30 months):
  1. Train all models on data BEFORE this month
  2. Generate predictions for this month's trading days
  3. Apply meta-labeling filter (skip low confidence)
  4. Apply regime-conditional position sizing
  5. Calculate P&L after ALL Indian market costs
  6. Record: every prediction, every trade, every P&L
  
Output:
  - Full trade log (CSV)
  - Equity curve
  - Monthly returns vs Nifty 50 buy-and-hold
  - Sharpe ratio, max drawdown, win rate, profit factor
  - Accuracy by regime
```

**Acceptance criteria:**
- [ ] 30-month walk-forward backtest completes
- [ ] Trade log with every prediction timestamped
- [ ] Sharpe ratio > 1.0 after costs
- [ ] Max drawdown < 15%
- [ ] Outperforms Nifty 50 buy-and-hold by at least 5% annually

---

### A7. Train All Models

**Why:** Currently only XGBoost is trained. Need all 7 models for ensemble to work.

**Steps:**
1. Train LSTM (use `alphavedha train lstm`)
2. Train TFT (use `alphavedha train tft`)
3. Train Regime Detector (use `alphavedha train regime`)
4. Train Ensemble + Meta-labeling + Conformal (use `alphavedha train all`)

**Acceptance criteria:**
- [ ] All 7 model artifacts saved in `models/artifacts/`
- [ ] Ensemble val accuracy > 50%
- [ ] Meta-labeling filters out at least 30% of signals
- [ ] Conformal intervals have ~90% coverage on val set

---

### Phase A Completion Checklist
- [ ] A1: FII/DII data flowing, features active
- [ ] A2: F&O data flowing, derivatives features active
- [ ] A3: Delivery % features built and tested
- [ ] A4: Earnings surprise tracker operational
- [ ] A5: Regime-conditional strategy selection working
- [ ] A6: Walk-forward backtest proves positive returns after costs
- [ ] A7: All 7 models trained and validated

**Expected outcome:** System accuracy improves significantly because 16 stub features now have real data. Walk-forward backtest proves or disproves the system honestly.

---

## PHASE B — Strategic Edge (1-2 months)

**Goal:** Add market-neutral strategies, alternative data, live paper trading, and sentiment analysis.

---

### B1. Pairs Trading Engine (Market-Neutral Alpha)

**Why:** Don't just predict "will TCS go up?" — predict "will TCS outperform Infosys?" This works in ALL market regimes because it's market-neutral.

**Files to create:**
| File | Action |
|------|--------|
| `alphavedha/signals/pairs.py` | CREATE — cointegration testing, spread tracking |
| `alphavedha/signals/pairs_universe.py` | CREATE — identify valid pairs within sectors |
| `alphavedha/signals/__init__.py` | MODIFY — export pairs module |

**Pair candidates (same sector):**
| Pair | Sector | Why cointegrated |
|------|--------|-----------------|
| HDFC Bank / ICICI Bank | Banking | Same business, same macro drivers |
| TCS / Infosys | IT | Same clients, same FX exposure |
| Reliance / ONGC | Energy | Both oil-linked |
| Bharti Airtel / Jio (if listed) | Telecom | Duopoly pricing |
| Titan / Asian Paints | Consumer | Both premium consumer plays |
| SBI / Bank of Baroda | PSU Banking | Same government ownership |

**Logic:**
```
1. Test cointegration (Engle-Granger or Johansen)
2. Calculate z-score of spread
3. When z-score > 2.0 → short the outperformer, long the underperformer
4. When z-score returns to 0 → close both positions
5. Stop loss: z-score > 3.5 (spread blowout)
```

**Acceptance criteria:**
- [ ] At least 5 cointegrated pairs identified
- [ ] Backtest shows positive Sharpe on pairs strategy alone
- [ ] Strategy is profitable in both bull and bear markets

---

### B2. Promoter Pledging & Insider Activity Tracker

**Why:** Promoter pledge increase is the strongest sell signal in Indian markets. Promoter buying is the strongest buy signal. Both are public data (SEBI mandate).

**Files to create:**
| File | Action |
|------|--------|
| `alphavedha/data/providers/sebi_provider.py` | CREATE — fetch SAST filings, insider trades |
| `alphavedha/data/models.py` | MODIFY — add `PromoterHolding`, `InsiderTrade` tables |
| `alphavedha/features/fundamental_features.py` | MODIFY — add promoter/insider features |

**New features:**
| Feature | What it captures |
|---------|-----------------|
| `promoter_pledge_pct` | % of promoter holding pledged |
| `pledge_change_qoq` | Is pledging increasing or decreasing? |
| `promoter_buying_30d` | Has promoter bought shares in last 30 days? |
| `insider_buy_sell_ratio` | Net insider buying vs selling |

**Acceptance criteria:**
- [ ] Last 3 years of promoter holding data
- [ ] Pledge increase correctly identified as risk signal
- [ ] Promoter buying correctly identified as bullish signal

---

### B3. Indian Financial News Sentiment

**Why:** Generic FinBERT trained on US news doesn't understand "RBI holds repo rate" or "SEBI tightens F&O rules." Need India-specific sentiment.

**Files to create:**
| File | Action |
|------|--------|
| `alphavedha/data/providers/news_provider.py` | CREATE — fetch news from Moneycontrol, ET, Livemint RSS |
| `alphavedha/features/sentiment.py` | MODIFY — replace stubs with real sentiment scores |
| `scripts/finetune_sentiment.py` | CREATE — fine-tune FinBERT on Indian financial news |

**Sentiment features to activate (currently stubs):**
| Feature | What it captures |
|---------|-----------------|
| `news_sentiment_1d` | Today's news sentiment score |
| `news_sentiment_5d` | 5-day rolling sentiment |
| `sentiment_momentum` | Is sentiment improving or deteriorating? |
| `news_volume_zscore` | Unusual number of news articles (event detection) |
| `sector_sentiment` | Average sentiment for the stock's sector |

**Acceptance criteria:**
- [ ] News fetcher pulling from at least 2 Indian financial news sources
- [ ] Sentiment model scores Indian news accurately (manual spot-check 50 articles)
- [ ] Sentiment features show predictive power in univariate analysis

---

### B4. Alternative Data: Auto Sales & Cement Dispatch

**Why:** These are leading indicators for sector performance. Published monthly, BEFORE companies report earnings.

**Data sources:**
- Auto sales: Published by each OEM (Maruti, Tata Motors, M&M) on 1st of each month
- Cement dispatch: Published by Cement Manufacturers Association

**Files to create:**
| File | Action |
|------|--------|
| `alphavedha/data/providers/alt_data_provider.py` | CREATE — scrape monthly auto + cement data |
| `alphavedha/features/macro.py` | MODIFY — add sector-level macro features |

**New features:**
| Feature | What it captures |
|---------|-----------------|
| `sector_auto_sales_yoy` | Auto sector monthly sales growth |
| `sector_cement_dispatch_yoy` | Cement dispatch growth (proxy for infra) |
| `sector_pmi` | Manufacturing/Services PMI |
| `credit_growth` | RBI credit growth (proxy for economic expansion) |

**Acceptance criteria:**
- [ ] 3+ years of monthly auto sales and cement data
- [ ] Features correctly mapped to relevant stocks (Maruti → auto_sales, UltraTech → cement)

---

### B5. Paper Trading Dashboard (Live Proof)

**Why:** Backtests can be overfit. Paper trading with timestamped predictions is the only undeniable proof.

**Files to create:**
| File | Action |
|------|--------|
| `alphavedha/api/routes/paper_trading.py` | CREATE — API endpoints for paper portfolio |
| `alphavedha/data/models.py` | MODIFY — add `PaperTrade`, `DailyPnL` tables |
| `scripts/daily_prediction.py` | CREATE — cron job: run daily predictions at 8:30 AM |
| `alphavedha/api/routes/dashboard.py` | CREATE — public dashboard endpoints |

**Implementation:**
```
Every trading day at 8:30 AM (before market opens):
  1. Fetch latest data
  2. Compute features
  3. Run all models → ensemble → meta-labeling → conformal
  4. Store predictions with timestamp (BEFORE market opens)
  5. At 3:45 PM: compare predictions vs actual
  6. Update paper portfolio P&L
  7. Publish to dashboard

Dashboard shows:
  - Today's predictions (stored before market open)
  - Hit rate: 7d / 30d / 90d
  - Paper portfolio equity curve
  - Comparison vs Nifty 50
  - Every prediction ever made (full transparency)
```

**Acceptance criteria:**
- [ ] Daily predictions generated and timestamped before 9:15 AM
- [ ] Dashboard shows full prediction history
- [ ] Paper portfolio tracking with realistic costs
- [ ] Running for 30+ days with verifiable track record

---

### B6. Reinforcement Learning Portfolio Optimizer (Prototype)

**Why:** Current system predicts direction then sizes positions separately. RL directly learns the optimal portfolio action including transaction costs.

**Files to create:**
| File | Action |
|------|--------|
| `alphavedha/models/rl_agent.py` | CREATE — PPO-based portfolio optimizer |
| `alphavedha/models/trading_env.py` | CREATE — Gym environment simulating Indian market |
| `alphavedha/training/rl_pipeline.py` | CREATE — RL training loop |

**Trading Environment:**
```
State:  [142 features, current_positions, portfolio_value, regime, days_to_expiry]
Action: [position_weight for each stock] ranging -1.0 to +1.0
Reward: daily_pnl - transaction_costs - slippage
        + bonus for Sharpe > 1.0
        - penalty for drawdown > 10%
```

**Acceptance criteria:**
- [ ] RL agent trains without diverging
- [ ] Walk-forward RL backtest outperforms rule-based position sizing
- [ ] Agent learns to reduce positions in high-volatility regimes

---

### Phase B Completion Checklist
- [ ] B1: Pairs trading engine with 5+ cointegrated pairs
- [ ] B2: Promoter pledging tracker operational
- [ ] B3: Indian news sentiment replacing stubs
- [ ] B4: Auto sales + cement dispatch data flowing
- [ ] B5: Paper trading dashboard running live for 30+ days
- [ ] B6: RL portfolio optimizer prototype trained

**Expected outcome:** Multiple independent alpha sources (directional, pairs, earnings drift). Live paper trading proves performance publicly. System has data edges that competitors don't.

---

## PHASE C — World-Class (2-3 months)

**Goal:** Build truly unique capabilities that set a world benchmark.

---

### C1. Graph Neural Network for Stock Relationships

**Why:** Stocks are connected. When TCS reports strong results, Infosys/Wipro/HCL Tech should update too. GNN models this information flow.

**Files to create:**
| File | Action |
|------|--------|
| `alphavedha/models/gnn_model.py` | CREATE — GNN using PyTorch Geometric |
| `alphavedha/data/stock_graph.py` | CREATE — build stock relationship graph |
| `alphavedha/training/gnn_pipeline.py` | CREATE — GNN training pipeline |

**Graph edges (relationships):**
| Edge type | Source | What it captures |
|-----------|--------|-----------------|
| Sector | NSE classification | Same sector stocks |
| Supply chain | Manual + NLP | Supplier/customer relationships |
| FII overlap | Quarterly holdings | Stocks held by same FIIs move together |
| Correlation | Price data | Statistically correlated movements |
| Promoter group | SEBI filings | Same business group (Tata, Reliance, Adani) |

**Acceptance criteria:**
- [ ] Stock graph with 50 nodes, 200+ edges
- [ ] GNN predictions improve ensemble accuracy by 2%+
- [ ] Information propagation visible: sector-wide update after one stock's earnings

---

### C2. Online Learning (Continuous Adaptation)

**Why:** Markets change. A model trained in January may not work in June. Online learning adapts continuously without full retraining.

**Files to create:**
| File | Action |
|------|--------|
| `alphavedha/models/online_xgboost.py` | CREATE — incremental XGBoost with sliding window |
| `alphavedha/monitoring/drift_detector.py` | CREATE — detect when model performance degrades |
| `alphavedha/monitoring/auto_retrain.py` | CREATE — trigger retraining when drift detected |

**Implementation:**
```
Every week:
  1. Compute prediction accuracy for last 5 trading days
  2. Compute PSI (Population Stability Index) for feature drift
  3. If accuracy < 48% OR PSI > 0.2 → trigger retraining
  4. Retrain on rolling 2-year window (drop oldest, add newest)
  5. Shadow-test new model for 5 days before promoting to production
```

**Acceptance criteria:**
- [ ] Drift detection catches model degradation within 1 week
- [ ] Auto-retraining completes in < 30 minutes
- [ ] Shadow testing prevents bad model from going live
- [ ] Model accuracy stays stable over 6+ months

---

### C3. Full Alternative Data Pipeline

**Why:** The ultimate moat. Everyone has price data. This data is harder to get and process.

| Data Source | Frequency | Relevant Stocks | Provider |
|-------------|-----------|-----------------|----------|
| GST collections | Monthly | All (economic health) | Government portal |
| Auto sales | Monthly | Maruti, Tata Motors, M&M, Bajaj Auto | OEM press releases |
| Cement dispatch | Monthly | UltraTech, Shree Cement, ACC, Ambuja | CMA website |
| Power consumption | Monthly | Power Grid, NTPC, Tata Power | CEA data |
| Port cargo data | Monthly | Adani Ports, container lines | Port authority |
| UPI transaction volume | Monthly | HDFC Bank, SBI, Paytm | NPCI data |
| Credit growth | Bi-weekly | All banks | RBI bulletin |
| Forex reserves | Weekly | IT exporters (TCS, Infosys) | RBI data |
| Crude oil prices | Daily | Reliance, ONGC, IOC, BPCL | Free API |
| US market overnight | Daily | All (global sentiment) | yfinance (S&P 500 futures) |
| Job posting trends | Monthly | IT, Banking | Naukri index (public) |

**Files to create:**
| File | Action |
|------|--------|
| `alphavedha/data/providers/alt_data_provider.py` | MODIFY — add all sources above |
| `alphavedha/data/models.py` | MODIFY — add `AlternativeData` table |
| `alphavedha/features/macro.py` | MODIFY — replace all 8 NaN stubs with real features |

**Acceptance criteria:**
- [ ] 8+ alternative data sources ingested
- [ ] All 8 macro feature stubs replaced with real values
- [ ] At least 3 alternative data features show predictive power

---

### C4. Intelligent Execution Engine

**Why:** Predicting correctly but executing badly loses money. Especially in mid-caps with lower liquidity.

**Files to create:**
| File | Action |
|------|--------|
| `alphavedha/signals/execution.py` | CREATE — optimal execution timing |
| `alphavedha/risk/impact_model.py` | CREATE — market impact estimation |
| `alphavedha/signals/order_book.py` | CREATE — order flow analysis |

**Rules:**
```
Avoid:
  - First 15 minutes (9:15-9:30) — opening auction noise
  - Last 10 minutes (3:20-3:30) — closing manipulation
  - F&O expiry last hour — gamma squeeze risk

Prefer:
  - 10:30-11:30 AM — post-opening stability, good liquidity
  - 2:00-2:45 PM — afternoon session, before closing pressure

Position entry:
  - Large cap (>₹50k cr): market order OK
  - Mid cap (₹10-50k cr): limit order, split into 2-3 tranches
  - Small cap (<₹10k cr): VWAP over 30 min window

Slippage estimation:
  - Model: slippage = f(order_size / avg_daily_volume, bid_ask_spread, volatility)
  - If estimated slippage > 0.5% → reduce position or skip
```

**Acceptance criteria:**
- [ ] Execution timing rules integrated into prediction service
- [ ] Slippage model trained on historical trade data
- [ ] Backtest with execution timing shows improvement over random entry

---

### C5. Public Track Record Dashboard

**Why:** This is the competitive moat. No other Indian stock prediction service publishes a verifiable, timestamped, full-transparency track record.

**Files to create:**
| File | Action |
|------|--------|
| Frontend (separate repo or within API) | CREATE — public web dashboard |
| `alphavedha/api/routes/public.py` | CREATE — public API for track record |

**Dashboard must show:**
```
1. EVERY prediction ever made (direction, magnitude, confidence, timestamp)
2. Actual outcome for each prediction
3. Running accuracy: 7d, 30d, 90d, all-time
4. Paper portfolio equity curve vs Nifty 50
5. Monthly returns breakdown
6. Accuracy by:
   - Regime (bull/bear/sideways/high-vol)
   - Sector
   - Cap size
   - Confidence level
7. Worst predictions and what went wrong
8. Current model version and last retrain date
```

**Acceptance criteria:**
- [ ] Dashboard live and publicly accessible
- [ ] 90+ days of verifiable predictions
- [ ] Full trade log downloadable as CSV
- [ ] Predictions timestamped before market open (verifiable)

---

### Phase C Completion Checklist
- [ ] C1: GNN model trained and integrated into ensemble
- [ ] C2: Online learning with drift detection and auto-retrain
- [ ] C3: 8+ alternative data sources feeding real features
- [ ] C4: Intelligent execution timing reducing slippage
- [ ] C5: Public dashboard with 90+ days of verifiable track record

**Expected outcome:** System has data and intelligence that no competitor in India has. Public track record provides undeniable proof. The moat is: data + transparency + continuous adaptation.

---

## Dependency Graph

```
A1 (FII/DII) ──────────┐
A2 (F&O Data) ──────────┤
A3 (Delivery %) ─────────┼──→ A7 (Train All Models) ──→ A6 (Walk-Forward Backtest)
A4 (Earnings) ───────────┤                                       │
A5 (Regime Strategy) ────┘                                       │
                                                                  ↓
B1 (Pairs Trading) ─────────────────────────────┐       Phase A Complete
B2 (Promoter Tracking) ─────────────────────────┤              │
B3 (Sentiment) ─────────────────────────────────┤              ↓
B4 (Alt Data: Auto/Cement) ─────────────────────┼──→ B5 (Paper Trading Dashboard)
B6 (RL Optimizer) ──────────────────────────────┘              │
                                                                ↓
                                                        Phase B Complete
                                                                │
C1 (GNN) ──────────────────────┐                               ↓
C2 (Online Learning) ──────────┼──→ C5 (Public Dashboard with 90+ day track record)
C3 (Full Alt Data) ────────────┤
C4 (Execution Engine) ─────────┘
```

---

## Success Metrics (The Benchmark)

| Metric | Target | Why this number |
|--------|--------|-----------------|
| Walk-forward Sharpe | > 1.5 | Top quant funds average 1.5-2.5 |
| Max Drawdown | < 12% | Acceptable for retail investor |
| Annual Return (after costs) | > Nifty 50 + 10% | Must meaningfully beat passive |
| Win Rate | > 55% | Combined with good risk:reward |
| Profit Factor | > 1.8 | Gross profits / gross losses |
| Meta-label filter rate | 30-40% | Skip 30-40% of signals (quality > quantity) |
| Prediction accuracy (direction) | > 55% | After meta-label filtering |
| Paper trading track record | 90+ days verified | Undeniable public proof |
| Model staleness detection | < 7 days | Catch drift within a week |
| Signals with data edge | 5+ alternative data sources | Moat that competitors can't easily copy |

---

## SEBI Compliance Reminders

- NEVER guarantee returns or show "guaranteed profit" anywhere
- Register as Research Analyst (RA) with SEBI before going public
- All predictions must have disclaimers
- Cannot manage client money without PMS/AIF license
- Paper trading is fine; real money for others requires SEBI registration
- Keep full audit log of every prediction for SEBI inspection

---

*Document version: 1.0 | Created: May 2026 | Project: AlphaVedha (अल्फावेध)*
