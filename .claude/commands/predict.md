# Predict

Run the prediction engine for a specific stock or scan a tier.

## Usage
- `/predict TCS.NS` — predict a single stock
- `/predict scan large` — scan all Nifty 50 stocks
- `/predict scan mid` — scan all Midcap 150 stocks

## Steps
1. Ensure the data is fresh (check last refresh timestamp)
2. If data is stale (> 1 trading day old), run `make data-refresh` first
3. Run the prediction:
   - Single stock: `python -m alphavedha.cli.main predict <symbol>`
   - Scan: `python -m alphavedha.cli.main scan <tier>`
4. Display results in a formatted table showing: symbol, direction, magnitude, confidence, composite score, regime
5. If confidence < 0.55 on any prediction, flag it as "LOW CONFIDENCE"

## Arguments
$ARGUMENTS
