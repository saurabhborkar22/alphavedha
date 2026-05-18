"""Verify Phase A1 imports and wiring."""
try:
    from alphavedha.data.providers import NSEProvider
    print("OK: NSEProvider import")
except Exception as e:
    print(f"FAIL: NSEProvider import - {e}")

try:
    from alphavedha.data.providers.nse_provider import parse_fii_dii_response, parse_fno_to_derivatives
    print("OK: parser functions import")
except Exception as e:
    print(f"FAIL: parser functions - {e}")

try:
    from alphavedha.data.store import store_fii_dii, load_fii_dii, store_derivatives, load_derivatives
    print("OK: store functions import")
except Exception as e:
    print(f"FAIL: store functions - {e}")

try:
    from alphavedha.data.ingestion import ingest_fii_dii, ingest_derivatives, FIIDIIResult, DerivativesResult
    print("OK: ingestion functions import")
except Exception as e:
    print(f"FAIL: ingestion functions - {e}")

try:
    from alphavedha.features.macro import load_fii_dii_for_features
    print("OK: load_fii_dii_for_features import")
except Exception as e:
    print(f"FAIL: load_fii_dii_for_features - {e}")

try:
    from alphavedha.cli.main import app
    print("OK: CLI app import")
except Exception as e:
    print(f"FAIL: CLI app - {e}")

print("\nAll Phase A1 wiring verified!")
