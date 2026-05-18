"""Test FII/DII data fetch from NSE (no DB required)."""
import asyncio
from alphavedha.data.providers.nse_provider import NSEProvider, parse_fii_dii_response

async def main():
    provider = NSEProvider()

    print("Fetching FII/DII data from NSE...")
    raw = await provider.fetch_fii_dii_today()
    print(f"Raw items: {len(raw)}")

    if raw:
        print(f"Sample: {raw[0]}")
        parsed = parse_fii_dii_response(raw)
        print(f"\nParsed rows: {len(parsed)}")
        for row in parsed:
            print(f"  {row['date']} | {row['category']:4s} | net: {row['net_value']:>10.2f} Cr")

asyncio.run(main())
