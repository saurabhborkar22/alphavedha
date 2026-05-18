"""Quick test: ingest 1 year of TCS data to verify the full pipeline."""
from __future__ import annotations

import asyncio
import os

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://alphavedha:alphavedha_dev@localhost:5435/alphavedha",
)

from datetime import date

from alphavedha.data.ingestion import ingest_symbol


async def main() -> None:
    rows = await ingest_symbol("TCS", date(2024, 1, 1), date.today())
    print(f"TCS: {rows} rows stored")


if __name__ == "__main__":
    asyncio.run(main())
