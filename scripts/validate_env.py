"""Validate production environment configuration before deployment."""

from __future__ import annotations

import os
import sys

PLACEHOLDER_VALUES = {
    "CHANGE_ME_STRONG_PASSWORD",
    "CHANGE_ME_REDIS_PASSWORD",
    "CHANGE_ME_API_KEY",
    "changeme",
    "password",
    "secret",
}

REQUIRED_VARS = [
    "DATABASE_URL",
    "REDIS_URL",
    "ALPHAVEDHA_API_KEY",
]

RECOMMENDED_VARS = [
    "ALPHAVEDHA_ENV",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "REDIS_PASSWORD",
]


def validate() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for var in REQUIRED_VARS:
        val = os.environ.get(var, "")
        if not val:
            errors.append(f"Missing required: {var}")
        elif val in PLACEHOLDER_VALUES:
            errors.append(f"Placeholder value in: {var} (change from template default)")

    for var in RECOMMENDED_VARS:
        val = os.environ.get(var, "")
        if not val:
            warnings.append(f"Missing recommended: {var}")
        elif val in PLACEHOLDER_VALUES:
            errors.append(f"Placeholder value in: {var}")

    api_key = os.environ.get("ALPHAVEDHA_API_KEY", "")
    if api_key and len(api_key) < 32:
        warnings.append(f"ALPHAVEDHA_API_KEY is short ({len(api_key)} chars) — use 32+ chars")

    db_url = os.environ.get("DATABASE_URL", "")
    if db_url and "@localhost" in db_url and os.environ.get("ALPHAVEDHA_ENV") == "production":
        warnings.append("DATABASE_URL points to localhost in production mode")

    if errors:
        print("ERRORS:", file=sys.stderr)
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
    if warnings:
        print("WARNINGS:", file=sys.stderr)
        for w in warnings:
            print(f"  ! {w}", file=sys.stderr)
    if not errors and not warnings:
        print("Environment validation passed.")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(validate())
