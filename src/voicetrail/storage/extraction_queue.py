"""Durable outbox for voice-memory extraction jobs.

The queue stores the source-bound extraction payload before a slow LLM or
embedding write is attempted.  It deliberately does not execute model code:
callers claim a job, run their extractor, then mark the job committed or
failed.  This keeps first-response code independent from a remote qwen3 call
and makes retries auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator


QUEUE_SCHEMA = "voice_extraction_queue_v1"
_STATUSES = {"pending", "processing", "committed", "failed"}


class LeaseLostError(RuntimeError):
    """Raised when a worker tries to use an expired or superseded lease."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExtractionJob:
    """Immutable view of one persisted extraction attempt."""

    job_id: int
    user_id: str
    session_id: str
    turn_id: str
    extractor_version: str
    payload: dict[str, Any]
    status: str
    attempts: int
    available_at: float
    locked_until: float | None
    last_error: str | None
    result: Any
    created_at: float
    updated_at: float
    lease_token: str | None


class ExtractionQueue:
    """SQLite-backed, idempotent extraction outbox.

    ``enqueue`` is unique on user/session/turn/extractor version.  Re-enqueueing
    the same payload returns the existing job; changing the payload for the
    same identity raises instead of silently overwriting source evidence.
    ``claim_next`` leases one job.  An expired lease returns to ``pending`` so
    a killed worker can be recovered by a later process.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        lease_seconds: float = 300.0,
        max_attempts: int = 3,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lease_seconds = float(lease_seconds)
        self.max_attempts = int(max_attempts)
        self._clock = clock
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self._connection() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS extraction_jobs (
                    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    turn_id TEXT NOT NULL,
                    extractor_version TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN
                        ('pending', 'processing', 'committed', 'failed')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    available_at REAL NOT NULL,
                    locked_until REAL,
                    lease_token TEXT,
                    last_error TEXT,
                    result_json TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(user_id, session_id, turn_id, extractor_version)
                )
                """
            )
            columns = {
                str(row[1])
                for row in conn.execute("PRAGMA table_info(extraction_jobs)")
            }
            if "lease_token" not in columns:
                conn.execute(
                    "ALTER TABLE extraction_jobs ADD COLUMN lease_token TEXT"
                )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_extraction_jobs_ready "
                "ON extraction_jobs(status, available_at, created_at)"
            )

    @staticmethod
    def _required(value: str, name: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError(f"{name} must be non-empty")
        return value

    @classmethod
    def _row_to_job(cls, row: sqlite3.Row) -> ExtractionJob:
        return ExtractionJob(
            job_id=int(row["job_id"]),
            user_id=str(row["user_id"]),
            session_id=str(row["session_id"]),
            turn_id=str(row["turn_id"]),
            extractor_version=str(row["extractor_version"]),
            payload=json.loads(row["payload_json"]),
            status=str(row["status"]),
            attempts=int(row["attempts"]),
            available_at=float(row["available_at"]),
            locked_until=(
                float(row["locked_until"])
                if row["locked_until"] is not None else None
            ),
            last_error=(str(row["last_error"]) if row["last_error"] is not None else None),
            result=(json.loads(row["result_json"]) if row["result_json"] is not None else None),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            lease_token=(
                str(row["lease_token"])
                if row["lease_token"] is not None else None
            ),
        )

    def _get_in(self, conn: sqlite3.Connection, job_id: int) -> ExtractionJob | None:
        row = conn.execute(
            "SELECT * FROM extraction_jobs WHERE job_id=?", (int(job_id),)
        ).fetchone()
        return self._row_to_job(row) if row is not None else None

    def enqueue(
        self,
        *,
        user_id: str,
        session_id: str,
        turn_id: str,
        extractor_version: str,
        payload: dict[str, Any],
    ) -> ExtractionJob:
        """Persist one source-bound job, refusing identity conflicts."""
        user_id = self._required(user_id, "user_id")
        session_id = self._required(session_id, "session_id")
        turn_id = self._required(turn_id, "turn_id")
        extractor_version = self._required(extractor_version, "extractor_version")
        if not isinstance(payload, dict):
            raise TypeError("payload must be a JSON object")
        payload_json = _canonical_json(payload)
        payload_sha = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        now = float(self._clock())
        with self._connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO extraction_jobs (
                    user_id, session_id, turn_id, extractor_version,
                    payload_json, payload_sha256, status, attempts,
                    available_at, locked_until, last_error, result_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, NULL, NULL, NULL, ?, ?)
                """,
                (
                    user_id, session_id, turn_id, extractor_version,
                    payload_json, payload_sha, now, now, now,
                ),
            )
            row = conn.execute(
                """
                SELECT * FROM extraction_jobs
                WHERE user_id=? AND session_id=? AND turn_id=? AND extractor_version=?
                """,
                (user_id, session_id, turn_id, extractor_version),
            ).fetchone()
            if row is None:
                raise RuntimeError("queued job was not readable")
            if str(row["payload_sha256"]) != payload_sha:
                raise ValueError(
                    "extraction job conflict: refusing to overwrite turn payload"
                )
            return self._row_to_job(row)

    def get(self, job_id: int) -> ExtractionJob | None:
        with self._connection() as conn:
            return self._get_in(conn, job_id)

    def list(self, *, status: str | None = None) -> tuple[ExtractionJob, ...]:
        if status is not None and status not in _STATUSES:
            raise ValueError(f"unsupported status: {status}")
        query = "SELECT * FROM extraction_jobs"
        params: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status=?"
            params = (status,)
        query += " ORDER BY created_at, job_id"
        with self._connection() as conn:
            return tuple(self._row_to_job(row) for row in conn.execute(query, params))

    def claim_next(self, *, user_id: str | None = None,
                   extractor_version: str | None = None) -> ExtractionJob | None:
        """Lease the oldest ready matching job; leave other workers' jobs alone."""
        now = float(self._clock())
        clauses: list[str] = []
        filters: list[str] = []
        for name, value in (("user_id", user_id), ("extractor_version", extractor_version)):
            if value is not None:
                clauses.append(f" AND {name}=?")
                filters.append(self._required(value, name))
        scope = "".join(clauses)  # column names come only from the fixed tuple above
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE extraction_jobs
                SET status='pending', locked_until=NULL, lease_token=NULL, updated_at=?
                WHERE status='processing' AND locked_until IS NOT NULL AND locked_until<=?
                """ + scope,
                (now, now, *filters),
            )
            row = conn.execute(
                """
                SELECT * FROM extraction_jobs
                WHERE status IN ('pending', 'failed')
                  AND attempts < ?
                  AND available_at <= ?
                """ + scope + " ORDER BY created_at, job_id LIMIT 1",
                (self.max_attempts, now, *filters),
            ).fetchone()
            if row is None:
                return None
            job_id = int(row["job_id"])
            lease_token = secrets.token_urlsafe(24)
            conn.execute(
                """
                UPDATE extraction_jobs
                SET status='processing', attempts=attempts+1,
                    locked_until=?, lease_token=?, updated_at=?
                WHERE job_id=?
                """,
                (now + self.lease_seconds, lease_token, now, job_id),
            )
            return self._get_in(conn, job_id)

    def renew_lease(self, job_id: int, lease_token: str) -> ExtractionJob:
        """Extend this token's lease without changing its attempt number.

        A heartbeat can arrive after ``locked_until`` under host scheduling or
        SQLite contention.  The token check still fences a worker that another
        claimant has already replaced, so a late heartbeat from the same owner
        may recover its claim while an old worker cannot revive a reclaimed job.
        """
        lease_token = self._required(lease_token, "lease_token")
        now = float(self._clock())
        with self._connection() as conn:
            updated = conn.execute(
                """
                UPDATE extraction_jobs
                SET locked_until=?, updated_at=?
                WHERE job_id=? AND status='processing'
                  AND lease_token=?
                """,
                (now + self.lease_seconds, now, int(job_id), lease_token),
            )
            if updated.rowcount != 1:
                raise LeaseLostError(
                    f"job {job_id} lease is expired or owned by another worker"
                )
            current = self._get_in(conn, job_id)
            if current is None:
                raise RuntimeError("renewed job was not readable")
            return current

    @staticmethod
    def _require_live_lease(
        current: ExtractionJob,
        lease_token: str | None,
        now: float,
    ) -> None:
        if not lease_token or current.lease_token != lease_token:
            raise LeaseLostError(
                f"job {current.job_id} lease token does not belong to this worker"
            )
        # Expiry is a reclaim opportunity, not an ownership change by itself.
        # ``claim_next`` atomically replaces the token before another worker
        # can take over.  Until that happens, a delayed heartbeat/commit from
        # the same token remains fenced from competing workers but can finish.

    def mark_committed(
        self,
        job_id: int,
        result: Any = None,
        *,
        lease_token: str | None = None,
    ) -> ExtractionJob:
        """Mark a claimed job committed; repeated identical commits are safe."""
        result_json = _canonical_json(result) if result is not None else None
        now = float(self._clock())
        with self._connection() as conn:
            current = self._get_in(conn, job_id)
            if current is None:
                raise KeyError(f"unknown extraction job: {job_id}")
            if current.status == "committed":
                if result_json is not None and current.result != result:
                    raise ValueError("committed extraction job result conflict")
                return current
            if current.status != "processing":
                raise ValueError(f"job {job_id} is not processing: {current.status}")
            self._require_live_lease(current, lease_token, now)
            conn.execute(
                """
                UPDATE extraction_jobs
                SET status='committed', locked_until=NULL, last_error=NULL,
                    result_json=?, updated_at=?
                WHERE job_id=? AND status='processing' AND lease_token=?
                """,
                (result_json, now, int(job_id), lease_token),
            )
            if conn.total_changes < 1:
                raise LeaseLostError(f"job {job_id} lease was lost before commit")
            updated = self._get_in(conn, job_id)
            if updated is None:
                raise RuntimeError("committed job was not readable")
            return updated

    def mark_failed(
        self,
        job_id: int,
        error: str,
        *,
        retry_delay_seconds: float = 0.0,
        lease_token: str | None = None,
    ) -> ExtractionJob:
        """Record a failure; ready jobs remain retryable until max_attempts."""
        error = self._required(error, "error")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds cannot be negative")
        now = float(self._clock())
        with self._connection() as conn:
            current = self._get_in(conn, job_id)
            if current is None:
                raise KeyError(f"unknown extraction job: {job_id}")
            if current.status == "failed" and current.last_error == error:
                return current
            if current.status != "processing":
                raise ValueError(f"job {job_id} is not processing: {current.status}")
            self._require_live_lease(current, lease_token, now)
            conn.execute(
                """
                UPDATE extraction_jobs
                SET status='failed', locked_until=NULL, last_error=?,
                    available_at=?, updated_at=?
                WHERE job_id=? AND status='processing' AND lease_token=?
                """,
                (
                    error,
                    now + float(retry_delay_seconds),
                    now,
                    int(job_id),
                    lease_token,
                ),
            )
            if conn.total_changes < 1:
                raise LeaseLostError(
                    f"job {job_id} lease was lost before failure record"
                )
            updated = self._get_in(conn, job_id)
            if updated is None:
                raise RuntimeError("failed job was not readable")
            return updated
