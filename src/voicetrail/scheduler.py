"""VoiceTrail session-level speech pipeline scheduler.

Stage-overlapped execution with durable Writer receipts.
Asyncio queues provide stage overlap and bounded backpressure.
SQLite job queue provides durable, lease-guarded Writer receipts.
"""
from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Sequence
from uuid import uuid4

from .storage.extraction_queue import ExtractionQueue


@dataclass(frozen=True)
class AudioTurn:
    user_id: str
    session_id: str
    date: str
    turn_index: int
    role: str
    audio_path: str


@dataclass(frozen=True)
class Session:
    user_id: str
    session_id: str
    date: str
    turns: tuple[AudioTurn, ...]


@dataclass(frozen=True)
class Question:
    user_id: str
    question_id: str
    text: str
    cutoff_record_date: str | None = None


@dataclass(frozen=True)
class Limits:
    asr: int = 1
    writer: int = 2
    index: int = 1
    reader: int = 1
    queue_capacity: int = 8

    def __post_init__(self):
        if any(value < 1 for value in asdict(self).values()):
            raise ValueError("all limits must be positive")


def save_immutable(path: Path, payload: Any) -> str:
    data = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    checksum = hashlib.sha256(data).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise RuntimeError("immutable result conflict: " + str(path))
        return checksum
    temp = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temp.write_bytes(data)
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise RuntimeError("result race: " + str(path))
        else:
            temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return checksum


class VoiceTrailPipeline:
    """The four stage adapters are synchronous functions invoked via asyncio.to_thread.

    asr(AudioTurn) -> str
    writer(Session, tuple[ASR records]) -> dict
    indexer(Session, memory dict) -> dict
    answer(Question, tuple[indexed sessions ordered by date]) -> dict containing answer

    The generic ExtractionQueue is reused directly. Its existing ExtractionWorker
    only supports fact-payload schemas, not session-sized job payloads.
    """

    def __init__(
        self, *, asr: Callable, writer: Callable, indexer: Callable,
        answer: Callable, out_dir: str | Path, limits: Limits = Limits(),
        writer_version: str = "voicetrail-session-writer-v1",
    ):
        self.asr, self.writer, self.indexer, self.answer = asr, writer, indexer, answer
        self.path = Path(out_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.limits = limits
        self.writer_version = writer_version
        self.outbox = ExtractionQueue(self.path / "voicemem_queue.sqlite",
                                      lease_seconds=900, max_attempts=2)
        self.events: list[dict[str, Any]] = []
        self.errors: list[dict[str, str]] = []

    def event(self, stage: str, status: str, key: str, seconds: float | None = None):
        row = {"stage": stage, "status": status, "key": key, "time": time.monotonic()}
        if seconds is not None:
            row["seconds"] = round(seconds, 4)
        self.events.append(row)

    async def run(self, sessions: Sequence[Session], questions: Sequence[Question]):
        if not sessions or not questions:
            raise ValueError("sessions and questions required")
        if self.outbox.list():
            raise RuntimeError("use a new output directory for each isolated run")
        session_map = {}
        needed = defaultdict(set)
        for s in sessions:
            key = (s.user_id, s.session_id)
            if not s.user_id or not s.session_id or key in session_map or not s.turns:
                raise ValueError("invalid or duplicated session")
            if [t.turn_index for t in s.turns] != list(range(len(s.turns))):
                raise ValueError("turns must be ordered and 0-based")
            if any((t.user_id, t.session_id, t.date) !=
                   (s.user_id, s.session_id, s.date) for t in s.turns):
                raise ValueError("source metadata does not match session")
            session_map[key] = s
            needed[s.user_id].add(s.session_id)
        per_user = defaultdict(list)
        qids = set()
        for q in questions:
            if q.user_id not in needed or not q.question_id or not q.text or q.question_id in qids:
                raise ValueError("invalid question")
            qids.add(q.question_id)
            per_user[q.user_id].append(q)

        asr_q = asyncio.Queue(self.limits.queue_capacity)
        writer_q = asyncio.Queue(self.limits.queue_capacity)
        index_q = asyncio.Queue(self.limits.queue_capacity)
        reader_q = asyncio.Queue(self.limits.queue_capacity)
        stop = object()
        finished_indexes = defaultdict(dict)
        ready_users = set()
        predictions = {}

        def failed(stage, label, error):
            self.errors.append({"stage": stage, "key": label,
                                "error": type(error).__name__ + ": " + str(error)})
            self.event(stage, "failed", label)

        async def asr_worker():
            while True:
                s = await asr_q.get()
                try:
                    if s is stop:
                        return
                    label = s.user_id + "/" + s.session_id
                    self.event("asr", "start", label)
                    began = time.monotonic()
                    transcript = []
                    for turn in s.turns:
                        text = await asyncio.to_thread(self.asr, turn)
                        if not isinstance(text, str) or not text.strip():
                            raise ValueError("empty transcription")
                        transcript.append({**asdict(turn), "transcript": text.strip()})
                    await asyncio.to_thread(save_immutable,
                        self.path / "asr" / s.user_id / (s.session_id + ".json"),
                        {"user": s.user_id, "session": s.session_id, "turns": transcript})
                    await asyncio.to_thread(
                        self.outbox.enqueue, user_id=s.user_id, session_id=s.session_id,
                        turn_id="__session__", extractor_version=self.writer_version,
                        payload={"user": s.user_id, "session": s.session_id,
                                 "date": s.date, "turns": transcript})
                    self.event("asr", "done", label, time.monotonic() - began)
                    await writer_q.put(True)
                except Exception as exc:
                    failed("asr", getattr(s, "session_id", "?"), exc)
                finally:
                    asr_q.task_done()

        async def writer_worker():
            while True:
                item = await writer_q.get()
                try:
                    if item is stop:
                        return
                    job = await asyncio.to_thread(self.outbox.claim_next,
                                                  extractor_version=self.writer_version)
                    if job is None:
                        raise RuntimeError("writer outbox was unexpectedly empty")
                    key = (job.user_id, job.session_id)
                    s = session_map[key]
                    label = "/".join(key)
                    self.event("writer", "start", label)
                    began = time.monotonic()
                    try:
                        memory = await asyncio.to_thread(self.writer, s, tuple(job.payload["turns"]))
                        if not isinstance(memory, dict):
                            raise TypeError("writer must return JSON memory")
                        file = self.path / "writer" / s.user_id / (s.session_id + ".json")
                        digest = await asyncio.to_thread(save_immutable, file, memory)
                        await asyncio.to_thread(self.outbox.mark_committed, job.job_id,
                            {"path": str(file), "sha256": digest}, lease_token=job.lease_token)
                        self.event("writer", "done", label, time.monotonic() - began)
                        await index_q.put((s, memory))
                    except Exception as exc:
                        try:
                            await asyncio.to_thread(self.outbox.mark_failed, job.job_id,
                                type(exc).__name__ + ": " + str(exc), lease_token=job.lease_token)
                        except Exception as ledger_exc:
                            failed("ledger", label, ledger_exc)
                        if job.attempts < self.outbox.max_attempts:
                            self.event("writer", "retry", label)
                            await writer_q.put(True)
                        else:
                            failed("writer", label, exc)
                except Exception as exc:
                    failed("writer_dispatch", "unknown", exc)
                finally:
                    writer_q.task_done()

        async def index_worker():
            while True:
                item = await index_q.get()
                try:
                    if item is stop:
                        return
                    s, memory = item
                    label = s.user_id + "/" + s.session_id
                    self.event("index", "start", label)
                    began = time.monotonic()
                    indexed = await asyncio.to_thread(self.indexer, s, memory)
                    if not isinstance(indexed, dict):
                        raise TypeError("indexer must return JSON")
                    await asyncio.to_thread(save_immutable,
                        self.path / "index" / s.user_id / (s.session_id + ".json"), indexed)
                    finished_indexes[s.user_id][s.session_id] = indexed
                    self.event("index", "done", label, time.monotonic() - began)
                    if (set(finished_indexes[s.user_id]) == needed[s.user_id]
                            and s.user_id not in ready_users):
                        ready_users.add(s.user_id)
                        chronological = sorted(
                            (session_map[(s.user_id, sid)] for sid in needed[s.user_id]),
                            key=lambda entry: (entry.date, entry.session_id))
                        snapshot = tuple(finished_indexes[s.user_id][entry.session_id]
                                         for entry in chronological)
                        self.event("snapshot", "ready", s.user_id)
                        for q in per_user[s.user_id]:
                            await reader_q.put((q, snapshot))
                except Exception as exc:
                    failed("index", getattr(item[0], "session_id", "?"), exc)
                finally:
                    index_q.task_done()

        async def reader_worker():
            while True:
                item = await reader_q.get()
                try:
                    if item is stop:
                        return
                    q, snapshot = item
                    self.event("reader", "start", q.question_id)
                    began = time.monotonic()
                    output = await asyncio.to_thread(self.answer, q, snapshot)
                    if not isinstance(output, dict) or not isinstance(output.get("answer"), str):
                        raise ValueError("reader must return JSON containing answer")
                    result = {"question_id": q.question_id, "user_id": q.user_id,
                              "question": q.text, "output": output}
                    await asyncio.to_thread(
                        save_immutable, self.path / "reader" / (q.question_id + ".json"), result)
                    predictions[q.question_id] = result
                    self.event("reader", "done", q.question_id, time.monotonic() - began)
                except Exception as exc:
                    failed("reader", getattr(item[0], "question_id", "?"), exc)
                finally:
                    reader_q.task_done()

        schedules = [
            (asr_q, [asyncio.create_task(asr_worker()) for _ in range(self.limits.asr)]),
            (writer_q, [asyncio.create_task(writer_worker()) for _ in range(self.limits.writer)]),
            (index_q, [asyncio.create_task(index_worker()) for _ in range(self.limits.index)]),
            (reader_q, [asyncio.create_task(reader_worker()) for _ in range(self.limits.reader)]),
        ]
        began = time.monotonic()
        try:
            for s in sorted(sessions, key=lambda v: (v.user_id, v.date, v.session_id)):
                await asr_q.put(s)
            # Wait for drain without stopping downstream stages until upstream completes.
            for queue, workers in schedules:
                await queue.join()
                for _ in workers:
                    await queue.put(stop)
                await asyncio.gather(*workers)
        finally:
            for _, workers in schedules:
                for t in workers:
                    if not t.done():
                        t.cancel()
            await asyncio.gather(*(t for _, ws in schedules for t in ws),
                                 return_exceptions=True)
        report = {
            "status": "PASS" if len(predictions) == len(questions) and not self.errors else "INCOMPLETE",
            "wall_seconds": round(time.monotonic() - began, 4),
            "sessions": len(sessions),
            "audio_turns": sum(len(s.turns) for s in sessions),
            "qa_expected": len(questions),
            "qa_completed": len(predictions),
            "users_ready": sorted(ready_users),
            "limits": asdict(self.limits),
            "writer_jobs": {state: len(self.outbox.list(status=state))
                            for state in ("pending", "processing", "failed", "committed")},
            "errors": self.errors,
            "events": self.events,
        }
        save_immutable(self.path / "PIPELINE_REPORT.json", report)
        return report
