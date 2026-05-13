# AlphaVedha Prediction Engine — Design Specification

**Date:** 2026-05-12
**Status:** Approved
**Author:** Saurabh Borkar + Claude Opus 4.6

---

## 1. Problem Statement

No AI-powered stock prediction platform exists natively for Indian markets (NSE/BSE). Global platforms (Danelfin, Kavout, AltIndex) cover only US/EU markets. Indian retail investors (13.6 crore unique investors) rely on fragmented tools (Screener.in for fundamentals, TradingView for charts, Moneycontrol for news) with no unified AI prediction layer.

AlphaVedha fills this gap: an API-first prediction engine combining ensemble ML models with India-specific signals to predict stock direction, magnitude, and price targets.

## 2. Target User

- Solo developer (Saurabh) for personal use initially
- Future: retail investors via web/mobile app
- Python-focused, beginner investor, learning domain as building

## 3. Scope

### In Scope (Phase 1)
- Prediction engine: direction, magnitude, confidence, price target range
- 141 features across 7 groups (technical, derivatives, macro, microstructure, sentiment, calendar, returns)
- Stacking ensemble: XGBoost + LSTM + TFT with Ridge meta-learner
- Triple barrier labeling + meta-labeling
- HMM market regime detection (bull/bear/sideways/high-vol)
- Conformal prediction intervals (90% coverage)
- Risk management: half-Kelly sizing, sector limits, drawdown circuit breakers
- CPCV validation (15 paths, 20-day purge + embargo)
- FastAPI REST endpoints + Typer CLI
- VectorBT backtesting with Indian market costs
- MLOps: feature drift detection, model versioning, automated retraining

### Out of Scope (Phase 1)
- Web/mobile frontend
- Automated trading execution
- Balance sheet analysis (Phase 2)
- Signal timing optimization (Phase 3)
- SEBI RA registration (when going public)
- Multi-language support

## 4. Architecture

### System Layers (bottom to top)
1. **Storage**: PostgreSQL 16 + TimescaleDB (time-series), Redis 7 (cache)
2. **Data Ingestion**: yfinance, jugaad-data, NSE direct, Finnhub (sentiment)
3. **Preprocessing**: Corporate action adjustment, circuit limit handling, fractional differentiation, outlier treatment, missing data handling
4. **Feature Store**: PostgreSQL-backed, ensures training-serving consistency
5. **Feature Engineering**: 141 features in 7 groups
6. **Labeling**: Triple barrier method + meta-labeling
7. **Regime Detection**: 4-state HMM (bull/bear/sideways/high-vol)
8. **Models**: XGBoost (tabular) + LSTM (sequential) + TFT (multi-horizon)
9. **Ensemble**: Stacking with Ridge meta-learner + meta-labeling confidence
10. **Conformal Prediction**: MAPIE for calibrated price target intervals
11. **Risk Management**: Kelly sizing, sector limits, drawdown circuit breakers
12. **API/CLI**: FastAPI endpoints, Typer CLI

### Company Universe
- Nifty 50 (large cap): 50 stocks
- Nifty Midcap 150: 150 stocks
- Nifty Smallcap 250: 250 stocks (optional, with liquidity filter)
- Point-in-time compositions from niftyindices.com

## 5. Feature Specification (141 features)

### Technical (40)
RSI (7/14/21), MACD + histogram, Bollinger %B + bandwidth, ATR(14), OBV + OBV ROC, Stochastic %K/%D, Williams %R, MFI, ADX, SMA/EMA (5/10/20/50/200), Supertrend, volume ratio, historical volatility (10/20/60d), SMA crossover signals, Keltner Channel, CCI, Aroon Oscillator, VWAP deviation, Accumulation/Distribution, Chaikin MF

### Derivatives (20)
PCR (volume + OI-based), IV (ATM call/put), IV percentile, IV-to-HV ratio, IV skew, total OI + OI change, participant-wise OI (FII/DII/Pro/Client), max pain distance, futures basis + z-score, futures rollover %, OI concentration ratio, F&O ban proximity, IV term structure slope

### Macro (25)
FII/DII net flows (cash + F&O), FII cumulative 20d flow, USD/INR + ROC, DXY, Brent crude, India VIX + percentile, US 10Y yield, India 10Y G-Sec + yield curve slope, Nifty 50 relative strength, sector relative performance, gold, copper, MSCI EM/World ratio, credit spread proxy, GST collections, PMI (manufacturing + services), auto sales, SIP inflow

### Microstructure (10)
Delivery % + z-score, promoter holding % + QoQ change, promoter pledging %, bulk/block deal flag + size, SAST disclosure flag, MF holding change, FPI holding % + change

### Sentiment (8)
FinBERT news score + std + 5d momentum, news velocity z-score, social sentiment, corporate filing sentiment, earnings call tone, Google Trends interest

### Calendar (18)
Day of week, month, week of month, days to monthly/quarterly expiry, days to quarterly results, budget flag, RBI policy proximity, election proximity, monsoon flag + progress, F&O ban proximity, index rebalancing window, advance tax date proximity, GST filing deadline, Muhurat trading flag, holiday-adjacent flag, T+0 settlement flag

### Returns-Derived (20)
Log returns (1/5/10/20d), fractionally differentiated price (4 features), rolling Sharpe + Sortino (20d), max drawdown (20d), up/down day ratio, gap open %, consecutive up/down count, return vs sector, return vs Nifty, regime label, circuit hit flag, 52w high/low distance

## 6. Labeling Strategy

### Triple Barrier Method
- Take profit: entry + 2.0 × ATR(14)
- Stop loss: entry - 1.5 × ATR(14)
- Time barrier: 15 trading days
- Labels: +1 (upper hit first), -1 (lower hit first), 0 (time expired)
- Barriers are per-stock (ATR-scaled)

### Meta-Labeling
- Secondary XGBoost classifier predicts P(primary model is correct)
- Output used for confidence scoring and position sizing
- Threshold: only output predictions with meta_confidence > 0.55

### Sample Weighting
- Uniqueness weighting: 1 / (overlapping labels at timestamp)
- Recency weighting: exponential decay, half-life = 252 trading days

## 7. Model Architecture

### Base Models
| Model | Input | Horizon | Key Config |
|-------|-------|---------|------------|
| XGBoost | 141 tabular features | 7d, 15d | lr=0.05, depth=6, n=500 |
| LSTM | 60-day sequence, top-30 features | 15d | 2 layers, 128 hidden, dropout=0.3 |
| TFT | Full multivariate + static covariates | 7d, 15d, 30d | hidden=64, 4 attn heads |

### Stacking Ensemble
- Input: [xgb_pred, lstm_pred, tft_pred, regime_probs, model_disagreement]
- Meta-learner: Ridge regression (alpha=1.0)
- Trained on out-of-fold predictions from base models

### Regime Detection
- Hidden Markov Model (4 states) on Nifty 50 returns + India VIX
- States: bull, bear, sideways, high_volatility
- Retrained monthly

### Conformal Prediction
- MAPIE with conformalized quantile regression
- 90% coverage guarantee
- Calibration window: 60 most recent trading days
- Adaptive width: expands in high-volatility regimes

## 8. Validation Protocol (CPCV)

- Combinatorial Purged Cross-Validation
- N=6 segments, k=2 test → 15 combinatorial paths
- Purge: 20 trading days between train and test
- Embargo: 20 additional days
- Sample uniqueness weighting
- Acceptance: median Sharpe > 0.8, worst-case Sharpe > 0.3

## 9. Risk Management

- Position sizing: half-Kelly criterion, capped at 10% per stock
- Sector cap: 25% max
- Correlation cap: 0.7 between holdings
- Liquidity filter: min Rs 5 crore avg daily turnover
- Drawdown circuit breakers: 10% → reduce 50%, 15% → halt, 20% → close all

## 10. API Output Format

```json
{
  "symbol": "TCS.NS",
  "name": "Tata Consultancy Services",
  "market_cap_tier": "large",
  "sector": "Information Technology",
  "regime": {"current": "bull", "confidence": 0.82},
  "predictions": {
    "7d":  {"direction": "UP", "magnitude": 2.1, "confidence": 0.68},
    "15d": {"direction": "UP", "magnitude": 4.2, "confidence": 0.73},
    "30d": {"direction": "UP", "magnitude": 6.8, "confidence": 0.61}
  },
  "price_targets": {
    "15d": {"low": 3850, "mid": 3950, "high": 4100, "coverage": 0.90}
  },
  "composite_score": 78,
  "score_breakdown": {
    "technical_momentum": 82, "derivatives_sentiment": 75,
    "macro_alignment": 85, "microstructure_quality": 71,
    "news_sentiment": 68, "volatility_risk": 74
  },
  "meta_confidence": 0.73,
  "risk": {"suggested_position_pct": 6.5, "stop_loss": 3720, "risk_reward_ratio": 2.1},
  "generated_at": "2026-05-11T10:30:00+05:30",
  "model_version": "v1.2.0"
}
```

## 11. Tech Stack

Python 3.12, FastAPI, PostgreSQL 16 + TimescaleDB, Redis 7, XGBoost, PyTorch (LSTM/TFT), hmmlearn, MAPIE, VectorBT, FinBERT (HuggingFace), Typer, Docker, structlog, Pydantic v2, Alembic

## 12. Data Sources (all free)

- **Price**: yfinance (20+ years), jugaad-data (NSE daily)
- **Index compositions**: niftyindices.com
- **Derivatives**: NSE options chain, participant-wise OI
- **Microstructure**: NSE Bhavcopy (delivery %), corporate filings (promoter data)
- **Macro**: NSE (FII/DII), yfinance (VIX, USD/INR, crude, gold), govt releases (GST, PMI)
- **Sentiment**: Finnhub free tier (60 calls/min), MarketAux
- **Calendar**: computed from dates + IMD monsoon data

## 13. Build Sequence (6 weeks)

| Week | Focus |
|------|-------|
| 1 | Data pipeline + preprocessing + DB setup |
| 2 | Feature engineering (technical + returns + calendar) + feature store |
| 3 | Triple barrier labeling + XGBoost + CPCV validation + VectorBT backtest |
| 4 | LSTM + HMM regime + derivatives features + macro features |
| 5 | TFT + stacking ensemble + meta-labeling + conformal prediction + sentiment |
| 6 | FastAPI + risk management + MLOps monitoring + CLI + Docker + tests |

## 14. Claude Code Infrastructure

- Root CLAUDE.md + 9 section-specific CLAUDE.md files
- 4 specialized agents: data-engineer, ml-engineer, feature-engineer, api-developer
- 5 custom slash commands: /predict, /train, /backtest, /data-refresh, /validate
- Settings with permissions, hooks (auto-lint on edit), safety guardrails
- Full config in configs/default.yaml

## 15. Regulatory Notes

- Personal use: no registration needed
- Public platform: SEBI Research Analyst (RA) registration required
- Automated trading (Phase 3): algo registration with exchange, static IP mandatory (April 2026)
- Data redistribution: cannot redistribute NSE/BSE data without license
- AI disclosure: mandatory when showing AI-generated recommendations

## 16. Success Metrics

- Directional accuracy: sustain 55-60% across market regimes
- Sharpe ratio: > 1.0 annualized
- Alpha: 5-15% above Nifty 50 buy-and-hold
- Max drawdown: < 25%
- Backtest-to-live consistency: live accuracy within 5% of backtest
