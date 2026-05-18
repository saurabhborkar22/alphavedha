"""Check jugaad-data NSE module capabilities."""
from jugaad_data import nse
print("jugaad_data.nse exports:", dir(nse))
print()

# Check if FII/DII data functions exist
for attr in ["fii_dii", "fii_dii_report", "index_data", "bhavcopy_save"]:
    print(f"  {attr}: {'YES' if hasattr(nse, attr) else 'NO'}")
