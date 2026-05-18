"""One-time script to create all database tables."""
from __future__ import annotations

import asyncio
import os

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://alphavedha:alphavedha_dev@localhost:5435/alphavedha",
)

from alphavedha.data.database import close, create_tables


async def main() -> None:
    await create_tables()
    await close()
    print("Tables created successfully.")


if __name__ == "__main__":
    asyncio.run(main())
