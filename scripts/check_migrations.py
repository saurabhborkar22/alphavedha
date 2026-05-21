"""Check for unapplied Alembic migrations — used in CI to prevent drift."""

from __future__ import annotations

import subprocess
import sys


def main() -> int:
    result = subprocess.run(
        ["alembic", "check"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if "Target database is not up to date" in result.stderr:
            print("ERROR: Unapplied migrations detected.", file=sys.stderr)
            print("Run: alembic upgrade head", file=sys.stderr)
        elif "New upgrade operations detected" in result.stdout:
            print("ERROR: ORM models changed without a migration.", file=sys.stderr)
            print("Run: alembic revision --autogenerate -m 'description'", file=sys.stderr)
        else:
            print(f"Alembic check failed:\n{result.stdout}\n{result.stderr}", file=sys.stderr)
        return 1
    print("Migrations are up to date.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
