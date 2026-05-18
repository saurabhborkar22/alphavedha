"""Check NSELive for FII/DII and derivatives data."""
from jugaad_data.nse import NSELive, derivatives_df
from datetime import date
import json

live = NSELive()

# Check market turnover (may have FII/DII)
print("=== Market Turnover ===")
try:
    data = live.market_turnover()
    print(json.dumps(data, indent=2, default=str)[:2000])
except Exception as e:
    print(f"Error: {e}")

print()

# Check trade info for a stock
print("=== Trade Info (TCS) ===")
try:
    data = live.trade_info("TCS")
    print(json.dumps(data, indent=2, default=str)[:2000])
except Exception as e:
    print(f"Error: {e}")

print()

# Check stock_quote_fno
print("=== Stock Quote FNO (TCS) ===")
try:
    data = live.stock_quote_fno("TCS")
    if isinstance(data, dict):
        print("Keys:", list(data.keys())[:20])
        for k in list(data.keys())[:3]:
            print(f"  {k}: {str(data[k])[:200]}")
except Exception as e:
    print(f"Error: {e}")

print()

# Check derivatives_df with correct args
print("=== Derivatives DF (FUTIDX NIFTY) ===")
try:
    df = derivatives_df(
        symbol="NIFTY",
        from_date=date(2026, 5, 1),
        to_date=date(2026, 5, 16),
        expiry_date=date(2026, 5, 29),
        instrument_type="FUTIDX",
    )
    print(f"Rows: {len(df)}, Columns: {list(df.columns)}")
    if not df.empty:
        print(df.head(2).to_string())
except Exception as e:
    print(f"Error: {e}")
