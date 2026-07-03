"""Tests for the proof publisher — git plumbing, OTS, containment, reveal."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from alphavedha.verification.publisher import (
    _ensure_git_repo,
    _git_commit_and_push,
    _ots_stamp,
    _write_proof_file,
    publish_daily_proof,
    reveal_due_proofs,
)

PROOF_DATE = date(2026, 6, 1)


def _init_repo(path: Path) -> None:
    _ensure_git_repo(path)
    assert (path / ".git").exists()


class TestGitCommitAndPush:
    def test_commit_sha_kept_when_push_fails(self, tmp_path: Path) -> None:
        """No remote configured: the local commit must still be recorded."""
        _init_repo(tmp_path)
        _write_proof_file(tmp_path, PROOF_DATE, "ab" * 32, b"{}")

        sha, pushed = _git_commit_and_push(tmp_path, f"proof: {PROOF_DATE} test")

        assert sha is not None and len(sha) == 40
        assert pushed is False

    def test_rerun_without_changes_returns_head(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        _write_proof_file(tmp_path, PROOF_DATE, "cd" * 32, b"{}")
        first_sha, _ = _git_commit_and_push(tmp_path, "proof: first")

        second_sha, pushed = _git_commit_and_push(tmp_path, "proof: rerun")

        assert second_sha == first_sha
        assert pushed is False


class TestEnsureGitRepo:
    def test_adds_remote_from_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ALPHAVEDHA_PROOFS_REMOTE", "git@example.com:user/proofs.git")
        _ensure_git_repo(tmp_path)

        remotes = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        )
        assert remotes.stdout.strip() == "git@example.com:user/proofs.git"

    def test_no_remote_without_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ALPHAVEDHA_PROOFS_REMOTE", raising=False)
        _ensure_git_repo(tmp_path)

        remotes = subprocess.run(
            ["git", "remote"], cwd=tmp_path, capture_output=True, text=True, check=True
        )
        assert remotes.stdout.strip() == ""


class TestOtsStamp:
    def test_missing_binary_degrades_to_none(self, tmp_path: Path) -> None:
        hash_path = tmp_path / "x.sha256"
        hash_path.write_text("ab" * 32 + "\n")

        with patch(
            "alphavedha.verification.publisher.subprocess.run",
            side_effect=FileNotFoundError("ots"),
        ):
            assert _ots_stamp(hash_path) is None

    def test_existing_ots_file_short_circuits(self, tmp_path: Path) -> None:
        hash_path = tmp_path / "x.sha256"
        hash_path.write_text("ab" * 32 + "\n")
        ots = tmp_path / "x.sha256.ots"
        ots.write_bytes(b"fake")

        assert _ots_stamp(hash_path) == ots


class TestPublishContainment:
    @pytest.mark.asyncio
    async def test_unwritable_repo_returns_stored_only(self, tmp_path: Path) -> None:
        """Regression: a root-owned volume crashed the job with an unhandled
        PermissionError every day for two weeks. The publish phase must
        contain filesystem errors and report them in the summary."""
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        repo = locked / "proofs-repo"

        trades = [
            {
                "symbol": "TCS",
                "prediction_date": PROOF_DATE,
                "predicted_direction": -1,
                "predicted_magnitude": 0.001,
                "confidence": 0.5,
                "is_tradeable": False,
                "model_version": "v0.1.0",
                "strategy": "ensemble_v1",
            }
        ]

        try:
            with (
                patch(
                    "alphavedha.verification.publisher._load_todays_trades",
                    new_callable=AsyncMock,
                    return_value=trades,
                ),
                patch(
                    "alphavedha.verification.publisher._store_proof",
                    new_callable=AsyncMock,
                ),
            ):
                summary = await publish_daily_proof(PROOF_DATE, proofs_repo=repo)
        finally:
            locked.chmod(0o700)

        assert summary["status"] == "stored_only"
        assert summary["git_commit"] is None
        assert summary["publish_error"] is not None
        assert "PermissionError" in summary["publish_error"]


class _FakeProof:
    def __init__(self, proof_date: date, payload: str) -> None:
        self.proof_date = proof_date
        self.payload_json = payload


class TestRevealDueProofs:
    @pytest.mark.asyncio
    async def test_reveal_writes_payload_and_verify_script(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        payload = '{"trades":[{"symbol":"TCS"}]}'
        proof = _FakeProof(PROOF_DATE, payload)

        # The hash that would have been committed before market open.
        digest = hashlib.sha256(payload.encode()).hexdigest()
        _write_proof_file(tmp_path, PROOF_DATE, digest, payload.encode())

        with (
            patch(
                "alphavedha.verification.publisher._load_due_unrevealed",
                new_callable=AsyncMock,
                return_value=[proof],
            ),
            patch(
                "alphavedha.verification.publisher._mark_revealed",
                new_callable=AsyncMock,
            ) as mock_mark,
        ):
            summary = await reveal_due_proofs(date(2026, 7, 2), proofs_repo=tmp_path)

        assert summary["revealed"] == 1
        assert (tmp_path / "proofs" / f"{PROOF_DATE.isoformat()}.json").read_text() == payload
        assert (tmp_path / "verify.py").exists()
        mock_mark.assert_awaited_once()
        marked_dates = mock_mark.await_args.args[0]
        assert marked_dates == [PROOF_DATE]
        assert isinstance(mock_mark.await_args.args[1], datetime)

        # End to end: the shipped verify script must validate the reveal.
        check = subprocess.run(
            [sys.executable, "verify.py"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert check.returncode == 0, check.stdout + check.stderr
        assert "OK" in check.stdout

    @pytest.mark.asyncio
    async def test_nothing_due(self, tmp_path: Path) -> None:
        with patch(
            "alphavedha.verification.publisher._load_due_unrevealed",
            new_callable=AsyncMock,
            return_value=[],
        ):
            summary = await reveal_due_proofs(date(2026, 7, 2), proofs_repo=tmp_path)

        assert summary == {"revealed": 0, "status": "nothing_due"}


class TestBranchAndPushRetry:
    """Fresh inits must land on main and survive a non-fast-forward push."""

    def test_fresh_init_uses_main_branch(self, tmp_path: Path) -> None:
        _init_repo(tmp_path)
        branch = subprocess.run(
            ["git", "symbolic-ref", "--short", "HEAD"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert branch == "main"

    def test_rejected_push_rebases_and_retries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Simulates the production incident: the GitHub repo already had a
        README commit on main, a fresh volume init pushed disjoint history."""
        # Bare origin with an existing README commit on main.
        origin = tmp_path / "origin.git"
        subprocess.run(
            ["git", "init", "--bare", "-b", "main", str(origin)], check=True, capture_output=True
        )
        seed = tmp_path / "seed"
        seed.mkdir()
        subprocess.run(["git", "init", "-b", "main"], cwd=seed, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "t@t"], cwd=seed, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "config", "user.name", "t"], cwd=seed, check=True, capture_output=True
        )
        (seed / "README.md").write_text("proofs repo\n")
        subprocess.run(["git", "add", "-A"], cwd=seed, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "init: README"], cwd=seed, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "push", str(origin), "main"], cwd=seed, check=True, capture_output=True
        )

        # Fresh publisher repo, disjoint history, origin pointing at the bare repo.
        repo = tmp_path / "proofs"
        repo.mkdir()
        monkeypatch.setenv("ALPHAVEDHA_PROOFS_REMOTE", str(origin))
        _ensure_git_repo(repo)
        _write_proof_file(repo, PROOF_DATE, "ee" * 32, b"{}")

        sha, pushed = _git_commit_and_push(repo, "proof: retry test")

        assert pushed is True
        assert sha is not None
        log = subprocess.run(
            ["git", "log", "--format=%s", "main"],
            cwd=origin,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        assert "proof: retry test" in log
        assert "init: README" in log
