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
    """Initialize the proofs directory as a git repo if not already.

    When ``ALPHAVEDHA_PROOFS_REMOTE`` is set (SSH URL for the scoped deploy
    key) and the repo has no origin yet, it is added — so a freshly
    initialized volume can push without manual setup.
    """
    import os

    if not (repo_dir / ".git").exists():
        try:
            # -b main: a bare `git init` defaults to master, and the first
            # push then lands on a branch GitHub doesn't show by default
            # (the repo was created with main).
            subprocess.run(
                ["git", "init", "-b", "main"], cwd=repo_dir, check=True, capture_output=True
            )
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
            return

    remote_url = os.environ.get("ALPHAVEDHA_PROOFS_REMOTE")
    if not remote_url:
        return
    try:
        existing = subprocess.run(
            ["git", "remote"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        if "origin" not in existing.stdout.split():
            subprocess.run(
                ["git", "remote", "add", "origin", remote_url],
                cwd=repo_dir,
                check=True,
                capture_output=True,
            )
            logger.info("proofs_remote_added")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("proofs_remote_setup_failed", error=str(e))


def _ots_stamp(hash_path: Path) -> Path | None:
    """Anchor the hash file into Bitcoin via OpenTimestamps.

    Uses the ``ots`` CLI from opentimestamps-client. Missing binary or
    calendar-server trouble downgrades to a logged skip — the git commit
    still orders the proof; OTS adds the independent timestamp.
    """
    ots_path = hash_path.with_suffix(hash_path.suffix + ".ots")
    if ots_path.exists():
        return ots_path
    try:
        subprocess.run(
            ["ots", "stamp", str(hash_path)],
            check=True,
            capture_output=True,
            timeout=120,
        )
    except FileNotFoundError:
        logger.warning("ots_binary_missing", hint="pip install opentimestamps-client")
        return None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning("ots_stamp_failed", error=str(e))
        return None
    return ots_path if ots_path.exists() else None


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
    message: str,
) -> tuple[str | None, bool]:
    """Stage, commit, and push the proof file.

    Returns ``(commit_sha, pushed)``. The local commit already fixes the
    proof's position in git history, so its SHA is kept even when the push
    fails (no remote / network down) — the next successful push publishes
    the whole backlog with original commit timestamps intact.
    """
    sha: str | None = None
    try:
        subprocess.run(
            ["git", "add", "-A"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout.strip():
            subprocess.run(
                ["git", "commit", "-m", message],
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
    except subprocess.CalledProcessError as e:
        logger.error(
            "proof_git_commit_failed",
            cmd=e.cmd,
            returncode=e.returncode,
            stderr=e.stderr.decode() if e.stderr else "",
        )
        return None, False

    if _push_head(repo_dir):
        return sha, True

    # Rejected push usually means the remote branch has commits this local
    # clone lacks (fresh volume init while the GitHub repo already has its
    # README). Replay our proofs on top and push again — original commit
    # author dates survive the rebase.
    if _rebase_onto_origin(repo_dir):
        pushed = _push_head(repo_dir)
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
        )
        if head.returncode == 0:
            sha = head.stdout.strip()
        return sha, pushed

    return sha, False


def _push_head(repo_dir: Path) -> bool:
    try:
        subprocess.run(
            ["git", "push", "-u", "origin", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(
            "proof_git_push_failed",
            returncode=e.returncode,
            stderr=e.stderr.decode() if e.stderr else "",
        )
        return False


def _rebase_onto_origin(repo_dir: Path) -> bool:
    """Fetch origin's branch and replay local commits on top of it."""
    try:
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "fetch", "origin", branch],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "rebase", f"origin/{branch}"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        subprocess.run(["git", "rebase", "--abort"], cwd=repo_dir, capture_output=True)
        logger.warning(
            "proof_git_rebase_failed",
            stderr=e.stderr.decode() if e.stderr else "",
        )
        return False


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


async def _update_proof_publication(
    proof_date: date,
    git_commit: str | None = None,
    ots_path: str | None = None,
) -> None:
    """Record git commit SHA and/or OTS proof path after publication."""
    from sqlalchemy import update

    from alphavedha.data.database import get_session_factory
    from alphavedha.data.models import PredictionProof

    values: dict[str, str] = {}
    if git_commit:
        values["git_commit"] = git_commit
    if ots_path:
        values["ots_path"] = ots_path
    if not values:
        return

    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            update(PredictionProof).where(PredictionProof.proof_date == proof_date).values(**values)
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

    # The filesystem/git phase must never take down the job with an
    # unhandled traceback (a root-owned volume did exactly that for two
    # weeks) — contain it and report the truth in the summary instead.
    git_commit: str | None = None
    pushed = False
    ots_path: Path | None = None
    publish_error: str | None = None
    try:
        proofs_repo.mkdir(parents=True, exist_ok=True)
        _ensure_git_repo(proofs_repo)
        hash_path = _write_proof_file(proofs_repo, proof_date, hex_digest, payload)
        ots_path = _ots_stamp(hash_path)
        if (proofs_repo / ".git").exists():
            git_commit, pushed = _git_commit_and_push(
                proofs_repo, f"proof: {proof_date.isoformat()} {hex_digest[:16]}"
            )
        else:
            publish_error = "proofs repo has no .git — init failed"
    except OSError as e:
        publish_error = f"{type(e).__name__}: {e}"
        logger.error("proof_publish_fs_failed", error=publish_error, repo=str(proofs_repo))

    if db_stored and (git_commit or ots_path):
        try:
            await _update_proof_publication(
                proof_date,
                git_commit=git_commit,
                ots_path=str(ots_path) if ots_path else None,
            )
        except Exception as e:
            logger.warning("proof_git_commit_update_failed", error=str(e))

    if git_commit and pushed:
        status = "published"
    elif git_commit:
        status = "committed_not_pushed"
    elif db_stored:
        status = "stored_only"
    else:
        status = "hash_only"

    return {
        "proof_date": proof_date.isoformat(),
        "n_predictions": n,
        "sha256": hex_digest,
        "git_commit": git_commit,
        "pushed": pushed,
        "ots_path": str(ots_path) if ots_path else None,
        "db_stored": db_stored,
        "publish_error": publish_error,
        "status": status,
    }


REVEAL_AFTER_DAYS = 21

_VERIFY_SCRIPT = '''#!/usr/bin/env python3
"""Verify AlphaVedha prediction proofs.

For every revealed day this repo holds two files:
  proofs/YYYY-MM-DD.sha256  — hash committed before market open
  proofs/YYYY-MM-DD.json    — canonical payload revealed 21+ days later

Re-hash the payload and compare:  python3 verify.py [YYYY-MM-DD ...]
"""

import hashlib
import sys
from pathlib import Path

proofs = Path(__file__).parent / "proofs"
dates = sys.argv[1:] or sorted(p.stem for p in proofs.glob("*.json"))
failures = 0
for day in dates:
    payload = proofs / f"{day}.json"
    expected = proofs / f"{day}.sha256"
    if not payload.exists() or not expected.exists():
        print(f"{day}: missing file(s), skipped")
        continue
    actual = hashlib.sha256(payload.read_bytes()).hexdigest()
    ok = actual == expected.read_text().strip()
    print(f"{day}: {'OK' if ok else 'MISMATCH'}")
    failures += 0 if ok else 1
sys.exit(1 if failures else 0)
'''


async def _load_due_unrevealed(as_of: date) -> list[Any]:
    """Fetch proof rows due for reveal (>= REVEAL_AFTER_DAYS old, unrevealed)."""
    from datetime import timedelta

    from sqlalchemy import select

    from alphavedha.data.database import get_session_factory
    from alphavedha.data.models import PredictionProof

    cutoff = as_of - timedelta(days=REVEAL_AFTER_DAYS)
    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            select(PredictionProof)
            .where(
                PredictionProof.proof_date <= cutoff,
                PredictionProof.revealed_at.is_(None),
                PredictionProof.payload_json.is_not(None),
            )
            .order_by(PredictionProof.proof_date)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())


async def _mark_revealed(proof_dates: list[date], revealed_at: datetime) -> None:
    from sqlalchemy import update

    from alphavedha.data.database import get_session_factory
    from alphavedha.data.models import PredictionProof

    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = (
            update(PredictionProof)
            .where(PredictionProof.proof_date.in_(proof_dates))
            .values(revealed_at=revealed_at)
        )
        await session.execute(stmt)
        await session.commit()


async def reveal_due_proofs(
    as_of: date | None = None,
    proofs_repo: Path | None = None,
) -> dict[str, Any]:
    """Reveal canonical payloads for proofs >= REVEAL_AFTER_DAYS old.

    Writes ``proofs/YYYY-MM-DD.json`` next to its hash so anyone can
    re-hash and compare (verify.py ships in the repo root), commits, and
    marks the rows revealed. Stays private until the repo flips public.
    """
    if as_of is None:
        as_of = datetime.now(IST).date()
    if proofs_repo is None:
        proofs_repo = DEFAULT_PROOFS_REPO

    due = await _load_due_unrevealed(as_of)
    if not due:
        return {"revealed": 0, "status": "nothing_due"}

    revealed_dates: list[date] = []
    try:
        proofs_subdir = proofs_repo / "proofs"
        proofs_subdir.mkdir(parents=True, exist_ok=True)
        verify_path = proofs_repo / "verify.py"
        if not verify_path.exists():
            verify_path.write_text(_VERIFY_SCRIPT)

        for proof in due:
            payload_path = proofs_subdir / f"{proof.proof_date.isoformat()}.json"
            payload_path.write_text(proof.payload_json)
            revealed_dates.append(proof.proof_date)
    except OSError as e:
        logger.error("proof_reveal_fs_failed", error=str(e), repo=str(proofs_repo))
        return {"revealed": 0, "status": "fs_error", "error": str(e)}

    git_commit: str | None = None
    pushed = False
    if (proofs_repo / ".git").exists():
        first, last = revealed_dates[0], revealed_dates[-1]
        git_commit, pushed = _git_commit_and_push(
            proofs_repo, f"reveal: {first.isoformat()}..{last.isoformat()}"
        )

    await _mark_revealed(revealed_dates, datetime.now(IST))
    logger.info(
        "proofs_revealed",
        n=len(revealed_dates),
        first=str(revealed_dates[0]),
        last=str(revealed_dates[-1]),
        git_commit=git_commit,
        pushed=pushed,
    )
    return {
        "revealed": len(revealed_dates),
        "git_commit": git_commit,
        "pushed": pushed,
        "status": "revealed",
    }
