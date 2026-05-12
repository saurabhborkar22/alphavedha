"""AlphaVedha FastAPI application."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(
    title="AlphaVedha API",
    description="AI-powered Indian stock market prediction engine for NSE/BSE",
    version="0.1.0",
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}
