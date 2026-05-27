from __future__ import annotations

import asyncio
import os
import socket
from collections.abc import Iterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

TEST_DB_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://alphavedha:testpass@localhost:5433/alphavedha_test",
)

_TABLES_TO_TRUNCATE = [
    "daily_pnl",
    "paper_trades",
    "features",
    "alternative_data",
    "news_articles",
    "insider_trades",
    "promoter_holdings",
    "earnings_results",
    "derivatives_data",
    "institutional_flows",
    "index_constituents",
    "corporate_actions",
    "daily_ohlcv",
]


def _test_db_available(loop: asyncio.AbstractEventLoop) -> bool:
    """Check if test DB is reachable with the right credentials."""
    try:
        sock = socket.create_connection(("localhost", 5433), timeout=2)
        sock.close()
    except OSError:
        return False

    engine = create_async_engine(TEST_DB_URL, pool_size=1, max_overflow=0)

    async def _probe() -> bool:
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False
        finally:
            await engine.dispose()

    return loop.run_until_complete(_probe())


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_engine(event_loop: asyncio.AbstractEventLoop) -> AsyncEngine:
    if not _test_db_available(event_loop):
        pytest.skip("Test database not available (start with: make test-integration-up)")
    return create_async_engine(TEST_DB_URL, pool_size=5, max_overflow=0)


@pytest.fixture(scope="session")
def _create_schema(
    test_engine: AsyncEngine, event_loop: asyncio.AbstractEventLoop
) -> Iterator[None]:
    from alphavedha.data.models import Base

    async def _setup() -> None:
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    event_loop.run_until_complete(_setup())
    yield

    async def _teardown() -> None:
        async with test_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        await test_engine.dispose()

    event_loop.run_until_complete(_teardown())


@pytest.fixture()
def session_factory(
    test_engine: AsyncEngine, _create_schema: None
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(autouse=True)
def _truncate_tables(
    request: pytest.FixtureRequest,
    event_loop: asyncio.AbstractEventLoop,
) -> Iterator[None]:
    yield
    if "session_factory" not in request.fixturenames:
        return
    engine = request.getfixturevalue("test_engine")

    async def _truncate() -> None:
        async with engine.begin() as conn:
            for table in _TABLES_TO_TRUNCATE:
                await conn.execute(text(f"TRUNCATE TABLE {table} CASCADE"))

    event_loop.run_until_complete(_truncate())
