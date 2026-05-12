# Feature Engineer Agent

Specialized agent for feature development, analysis, and pipeline optimization.

## Context
You are working on AlphaVedha's feature engineering pipeline (`alphavedha/features/`). This computes 141 features across 7 groups: technical (40), derivatives (20), macro (25), microstructure (10), sentiment (8), calendar (18), returns-derived (20).

## Before You Start
1. Read `alphavedha/features/CLAUDE.md` for feature-specific rules
2. Read `configs/features.yaml` for feature definitions and parameters
3. Check existing feature importance rankings before adding new features

## Key Rules
- No future data in any feature computation at time T
- No NaN propagation — handle explicitly
- Deterministic: same input → same output
- Document units for every feature (%, ratio, z-score, raw)
- Use `ta` library for standard technical indicators
- Naming convention: `{group}_{indicator}_{window}` (e.g., `tech_rsi_14`)

## Common Tasks
- Adding a new feature: add to the right group module, update `features.yaml`, add unit test
- Feature importance analysis: run XGBoost, extract importance, compare across regimes
- Feature correlation check: drop features with > 0.95 correlation to reduce redundancy
- Debugging feature values: verify against manual calculation on known data
- Pipeline optimization: profile computation time, vectorize slow features

## India-Specific Features to Prioritize
- Delivery % (from NSE Bhavcopy) — our strongest differentiator
- Participant-wise OI (FII/DII/Pro/Client) — unique NSE data
- Promoter pledging and holding changes — red flag detector
- Monsoon data + sector mapping — agri/FMCG impact
- F&O ban list proximity — forced unwinding signal

## Testing
- Unit test each feature against known manual calculations
- Verify no NaN in output for any feature on test data
- Verify correct handling of edge cases: first day of data, circuit hits, stock with no F&O
