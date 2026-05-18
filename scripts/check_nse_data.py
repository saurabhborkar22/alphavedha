"""Check available data from jugaad-data and NSE."""
from datetime import date
from jugaad_data import nse

# Check derivatives data
print("=== Derivatives Data (TCS) ===")
try:
    df = nse.derivatives_df(symbol="TCS", from_date=date(2026, 5, 1), to_date=date(2026, 5, 16), expiry_date=date(2026, 5, 29))
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    if not df.empty:
        print(df.head(2).to_string())
except Exception as e:
    print(f"Error: {e}")

print()

# Check bhavcopy F&O
print("=== F&O Bhavcopy ===")
try:
    raw = nse.bhavcopy_fo_raw(date(2026, 5, 16))
    print(f"Type: {type(raw)}")
    if raw:
        import io
        import pandas as pd
        # It returns a zip file content
        print(f"Size: {len(raw)} bytes")
except Exception as e:
    print(f"Error: {e}")

print()

# Check NSELive for FII/DII
print("=== NSE Live ===")
try:
    live = nse.NSELive()
    print(f"NSELive methods: {[m for m in dir(live) if not m.startswith('_')]}")
except Exception as e:
    print(f"Error: {e}")

print()

# Check index data for Nifty 50
print("=== Index Data (NIFTY 50) ===")
try:
    df = nse.index_df("NIFTY 50", date(2026, 5, 1), date(2026, 5, 16))
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    if not df.empty:
        print(df.head(2).to_string())
except Exception as e:
    print(f"Error: {e}")
