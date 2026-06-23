"""Prediction proof publisher — commits daily hashes to the proofs git repo.

After the 08:30 prediction job persists paper trades, this module:
1. Loads today's paper trades from the DB
2. Computes the canonical SHA-256 hash
3. Writes ``proofs/YYYY-MM-DD.sha256`` to the local proofs repo clone
4. Commits + pushes (via a scoped deploy key)
5. Stores the proof in ``prediction_proofs`` table

The proofs repo starts private (P0) and flips public in P6 — months of
git history + Bitcoin-anchored OTS proofs become retroactively verifiable.
"""

from __future__ import annotations

import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import structlog

from alphavedha.verification.hasher import hash_daily_trades

logger = structlog.get_logger(__name__)

IST = ZoneInfo("Asia/Kolkata")

DEFAULT_PROOFS_REPO = Path.home() / "alphavedha-proofs"


def _ensure_git_repo(repo_dir: Path) -> None:
    """Initialize the proofs directory as a git repo if not already."""
    if (repo_dir / ".git").exists():
        return
    try:
        subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "scheduler@alphavedha.local"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "AlphaVedha Scheduler"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )
        logger.info("proofs_repo_initialized", path=str(repo_dir))
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("proofs_repo_init_failed", error=str(e))


async def _load_todays_trades(proof_date: date) -> list[dict[str, Any]]:
    """Load paper trades for the given date from the DB."""
    from alphavedha.data.store import load_paper_trades

    df = await load_paper_trades()
    if df.empty:
        return []
    df["prediction_date"] = df["prediction_date"].apply(
        lambda x: x if isinstance(x, date) else date.fromisoformat(str(x))
    )
    day_trades = df[df["prediction_date"] == proof_date]
    return list(day_trades.to_dict("records"))


def _write_proof_file(
    proofs_dir: Path,
    proof_date: date,
    hex_digest: str,
    payload: bytes,
) -> Path:
    """Write the hash file + raw payload to the proofs repo."""
    date_str = proof_date.isoformat()
    proofs_subdir = proofs_dir / "proofs"
    proofs_subdir.mkdir(parents=True, exist_ok=True)

    hash_path = proofs_subdir / f"{date_str}.sha256"
    hash_path.write_text(hex_digest + "\n")

    return hash_path


def _git_commit_and_push(
    repo_dir: Path,
    proof_date: date,
    hex_digest: str,
) -> str | None:
    """Stage, commit, and push the proof file. Returns the commit SHA or None."""
    try:
        subprocess.run(
            ["git", "add", "proofs/"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )
        msg = f"proof: {proof_date.isoformat()} {hex_digest[:16]}"
        subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        sha = result.stdout.strip()

        subprocess.run(
            ["git", "push"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )
        return sha
    except subprocess.CalledProcessError as e:
        logger.error(
            "proof_git_failed",
            cmd=e.cmd,
            returncode=e.returncode,
            stderr=e.stderr.decode() if e.stderr else "",
        )
        return None


async def _store_proof(
    proof_date: date,
    hex_digest: str,
    n_predictions: int,
    payload_json: str,
    git_commit: str | None,
) -> None:
    """Insert a row into the prediction_proofs table."""
    from alphavedha.data.database import get_session_factory
    from alphavedha.data.models import PredictionProof

    session_factory = get_session_factory()
    async with session_factory() as session:
        proof = PredictionProof(
            proof_date=proof_date,
            sha256=hex_digest,
            n_predictions=n_predictions,
            payload_json=payload_json,
            git_commit=git_commit,
        )
        session.add(proof)
        await session.commit()


async def _upsert_proof(
    proof_date: date,
    hex_digest: str,
    n_predictions: int,
    payload_json: str,
    git_commit: str | None,
) -> None:
    """Insert or update a proof row (handles duplicate proof_date)."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from alphavedha.data.database import get_session_factory
    from alphavedha.data.models import PredictionProof

    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            pg_insert(PredictionProof)
            .values(
                proof_date=proof_date,
                sha256=hex_digest,
                n_predictions=n_predictions,
                payload_json=payload_json,
                git_commit=git_commit,
            )
            .on_conflict_do_update(
                index_elements=["proof_date"],
                set_={
                    "sha256": hex_digest,
                    "n_predictions": n_predictions,
                    "payload_json": payload_json,
                },
            )
        )
        await session.execute(stmt)
        await session.commit()


async def _update_proof_git_commit(proof_date: date, git_commit: str) -> None:
    """Update the git_commit field after a successful push."""
    from sqlalchemy import update

    from alphavedha.data.database import get_session_factory
    from alphavedha.data.models import PredictionProof

    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            update(PredictionProof)
            .where(PredictionProof.proof_date == proof_date)
            .values(git_commit=git_commit)
        )
        await session.execute(stmt)
        await session.commit()


async def publish_daily_proof(
    proof_date: date | None = None,
    proofs_repo: Path | None = None,
) -> dict[str, Any]:
    """Full proof pipeline: load trades -> hash -> write -> commit -> store.

    Returns a summary dict for logging / scheduler result.
    """
    if proof_date is None:
        proof_date = datetime.now(IST).date()
    if proofs_repo is None:
        proofs_repo = DEFAULT_PROOFS_REPO

    trades = await _load_todays_trades(proof_date)
    n = len(trades)

    if n == 0:
        logger.warning("proof_no_trades", date=proof_date.isoformat())
        return {
            "proof_date": proof_date.isoformat(),
            "n_predictions": 0,
            "sha256": None,
            "git_commit": None,
            "status": "skipped_no_trades",
        }

    hex_digest, payload = hash_daily_trades(trades)
    payload_str = payload.decode("utf-8")

    logger.info(
        "proof_computed",
        date=proof_date.isoformat(),
        n_predictions=n,
        sha256=hex_digest[:16] + "...",
    )

    db_stored = False
    try:
        await _store_proof(proof_date, hex_digest, n, payload_str, None)
        db_stored = True
    except Exception as e:
        logger.error("proof_store_failed", error=str(e), error_type=type(e).__name__)
        try:
            await _upsert_proof(proof_date, hex_digest, n, payload_str, None)
            db_stored = True
            logger.info("proof_upsert_succeeded", date=proof_date.isoformat())
        except Exception as e2:
            logger.error("proof_upsert_also_failed", error=str(e2))

    git_commit: str | None = None
    proofs_repo.mkdir(parents=True, exist_ok=True)
    _ensure_git_repo(proofs_repo)
    _write_proof_file(proofs_repo, proof_date, hex_digest, payload)
    if (proofs_repo / ".git").exists():
        git_commit = _git_commit_and_push(proofs_repo, proof_date, hex_digest)

    if git_commit and db_stored:
        try:
            await _update_proof_git_commit(proof_date, git_commit)
        except Exception as e:
            logger.warning("proof_git_commit_update_failed", error=str(e))

    return {
        "proof_date": proof_date.isoformat(),
        "n_predictions": n,
        "sha256": hex_digest,
        "git_commit": git_commit,
        "db_stored": db_stored,
        "status": "published" if git_commit else ("stored_only" if db_stored else "hash_only"),
    }
