"""Check available FII/DII data sources."""
import requests
from datetime import date, timedelta

# Source 1: NSDL FPI data (most reliable for historical)
print("=== NSDL FPI Data ===")
try:
    # NSDL publishes FPI (FII) investment data
    url = "https://www.fpi.nsdl.co.in/web/StaticReports/Fortnightly_Sector_wise_FII_Investment_Data/FIIInvestmentActivitybyDayinEquity.html"
    resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    print(f"Status: {resp.status_code}, Size: {len(resp.text)} bytes")
except Exception as e:
    print(f"Error: {e}")

print()

# Source 2: Moneycontrol FII/DII API
print("=== Moneycontrol FII/DII ===")
try:
    url = "https://api.moneycontrol.com/mcapi/v1/fii-dii/all"
    resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Keys: {list(data.keys()) if isinstance(data, dict) else 'list'}")
        print(str(data)[:500])
except Exception as e:
    print(f"Error: {e}")

print()

# Source 3: NSE direct API for FII/DII
print("=== NSE FII/DII API ===")
try:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    # First hit NSE homepage to get cookies
    session.get("https://www.nseindia.com", timeout=10)
    # Then hit the FII/DII API
    url = "https://www.nseindia.com/api/fiidiiTradeReact"
    resp = session.get(url, timeout=10)
    print(f"Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Type: {type(data)}")
        if isinstance(data, list):
            print(f"Items: {len(data)}")
            if data:
                print(f"First item keys: {list(data[0].keys())}")
                print(f"First item: {data[0]}")
        elif isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            print(str(data)[:500])
except Exception as e:
    print(f"Error: {e}")
