"""Executable, gold-blind checks for the Methodology's implementation contracts."""
from __future__ import annotations

import json

import numpy as np
import pytest

from voicetrail.adapters.reader import ReaderAdapter, PreparedEvidence
from voicetrail.adapters.writer import WriterAdapter
from voicetrail.memory.affective import classify_affective_quote
from voicetrail.memory.event_identity import stable_event_id
from voicetrail.retrieval.trajectory_selector import select
from voicetrail.scheduler import Question, Session
from voicetrail.streaming import StreamingQuerySession


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
    def raise_for_status(self):
        pass
    def json(self):
        return self.payload


class FakeRequests:
    def __init__(self, payload):
        self.payload = payload
    def post(self, *args, **kwargs):
        return FakeResponse(self.payload)


def write_session(session_id, date, utterance, key, anchor):
    writer = WriterAdapter()
    writer.writer_url = "https://example.invalid"
    writer.writer_model = "fake"
    writer.writer_headers = {}
    payload = {
        "facts": [{"source_turn_id": session_id + ":0000", "text": "The seeker discussed a bus transfer."}],
        "emotions": [{
            "source_turn_id": session_id + ":0000",
            "emotion_quote": utterance,
            "emotion": ["frustration"],
            "emotion_assertion": "EXPERIENCED",
            "time_class": "CURRENT",
            "event_key": key,
            "event_anchor": anchor,
            "event_label": "bus_transfer",
            "aspect": "bus transfer",
            "target": "bus transfer",
        }],
    }
    writer.requests = FakeRequests({
        "choices": [{"message": {"content": json.dumps(payload)}}],
    })
    session = Session("user_1", session_id, date, ())
    result = writer.writer(session, ({
        "turn_index": 0, "role": "seeker", "transcript": utterance,
    },))
    return result["emotions"][0]


def test_cross_session_event_id_reused_only_with_source_attestation():
    first = write_session("S001", "2025-01-01",
                          "I feel frustrated about the late night bus transfer.",
                          "late night bus transfer", "late night bus transfer")
    second = write_session("S008", "2025-05-01",
                           "I feel hopeful about the late night bus transfer.",
                           "late-night bus transfer", "late night bus transfer")
    assert first["event_id"] == second["event_id"]
    assert first["event_id"].startswith("evt:")
    assert first["event_anchor"] == "late night bus transfer"
    assert first["aspect"] == "bus transfer"
    assert first["source_turn_id"] == "S001:0000"

    unrelated = write_session("S010", "2025-08-01",
                              "I feel frustrated about the day bus transfer.",
                              "day bus transfer", "day bus transfer")
    assert unrelated["event_id"] != first["event_id"]
    assert stable_event_id("user_2", "late night bus transfer",
                           "late night bus transfer",
                           ["late night bus transfer"]) != first["event_id"]


def test_unsupported_keys_cannot_silently_merge_sessions():
    first = write_session("S001", "2025-01-01",
                          "I feel frustrated about the bus transfer.",
                          "late night bus transfer", "nonexistent anchor")
    second = write_session("S002", "2025-02-01",
                           "I feel frustrated about the bus transfer.",
                           "late night bus transfer", "nonexistent anchor")
    assert first["event_id"] == "S001:bus_transfer"
    assert second["event_id"] == "S002:bus_transfer"
    assert first["event_key"] is None


@pytest.mark.parametrize("quote,expected", [
    ("I feel worried today", ("EXPERIENCED", "CURRENT")),
    ("I felt worried last month", ("EXPERIENCED", "PAST_RECALL")),
    ("If I fail I would feel afraid", ("HYPOTHETICAL", "UNKNOWN")),
    ("I might feel worried", ("HYPOTHETICAL", "UNKNOWN")),
    ("I will feel nervous", ("INTENDED", "FUTURE_ANTICIPATED")),
])
def test_affective_assertion_and_time_class(quote, expected):
    assert classify_affective_quote(quote, "EXPERIENCED", "CURRENT") == expected


def test_same_id_trajectory_expands_old_and_recent_sessions_before_filling():
    event_a = "evt:identical-key"
    rows = [
        {"obs_id": "middle", "event_id": event_a, "session_id": "S5",
         "record_time": "2025-05-01", "emotion_assertion": "EXPERIENCED",
         "source_turn_id": "t5"},
        {"obs_id": "early", "event_id": event_a, "session_id": "S1",
         "record_time": "2025-01-01", "emotion_assertion": "EXPERIENCED",
         "source_turn_id": "t1"},
        {"obs_id": "late", "event_id": event_a, "session_id": "S9",
         "record_time": "2025-09-01", "emotion_assertion": "EXPERIENCED",
         "source_turn_id": "t9"},
    ]
    scores = {"middle": 0.99, "early": 0.01, "late": 0.02}
    for i in range(17):
        oid = "other_%02d" % i
        rows.append({"obs_id": oid, "event_id": "evt:different_%02d" % i,
                     "session_id": "D%d" % i, "record_time": "2025-06-01",
                     "emotion_assertion": "EXPERIENCED", "source_turn_id": oid})
        scores[oid] = 0.98 - i * 0.03
    result = select("How did feelings change?", rows, scores)
    assert len(result["ids"]) == 15
    assert {"middle", "early", "late"}.issubset(result["ids"])
    assert result["reasons"]["early"].startswith("SAME_EVENT_MISSING_SESSION")
    assert result["reasons"]["late"].startswith("SAME_EVENT_MISSING_SESSION")


def snapshot_entry(sid, date, feeling):
    tid = sid + ":0000"
    fid = sid + ":f001"
    oid = sid + ":o001"
    raw = {"turn_id": tid, "seeker_id": "u", "session_id": sid,
           "record_time": date, "speaker": "seeker", "text": feeling}
    fact = {"fact_id": fid, "seeker_id": "u", "session_id": sid,
            "record_time": date, "source_turn_id": tid,
            "text": "A personal experience was described."}
    obs = {"obs_id": oid, "seeker_id": "u", "session_id": sid,
           "record_time": date, "source_turn_id": tid,
           "emotion": ["worry"], "emotion_quote": feeling,
           "emotion_assertion": "EXPERIENCED", "time_class": "CURRENT",
           "event_id": None, "link_status": None}
    return {"user_id": "u", "session_id": sid, "date": date,
            "raw": [raw], "facts": [fact], "emotions": [obs],
            "vectors": {tid: [1.0, 0.0], fid: [1.0, 0.0], oid: [1.0, 0.0]}}


def test_reader_cutoff_excludes_future_evidence_and_reports_latency():
    class FakeE5:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    reader = ReaderAdapter()
    reader.e5 = FakeE5()
    reader.reader_model_name = "fake"
    data = (snapshot_entry("old", "2025-01-01", "I feel worried."),
            snapshot_entry("new", "2025-09-01", "I feel relieved."))
    prep = reader.prepare_evidence("What emotion did I express?", "u", data,
                                   cutoff_record_date="2025-05-01")
    assert prep.eligible_session_count == 1
    assert "I feel worried" in prep.context
    assert "I feel relieved" not in prep.context
    assert prep.retrieval_latency_ms >= prep.query_embedding_ms >= 0
    absent = reader.prepare_evidence("What emotion did I express?", "u", data,
                                     cutoff_record_date="2024-01-01")
    assert absent.eligible_session_count == 0
    assert reader.answer_prepared(
        Question("u", "q", absent.question, "2024-01-01"),
        absent)["answer"] == "Insufficient information"


def test_streaming_anticipation_revises_candidate_context_on_speech_tail():
    calls = []
    class FakeReader:
        def prepare_evidence(self, query, user, snapshot, *, cutoff_record_date=None):
            calls.append(query)
            return PreparedEvidence(query, user, cutoff_record_date or "2025-10-01",
                                    "Evidence for " + query, {}, 1.0, 0.5, 1)
        def answer_prepared(self, question, prepared):
            return {"answer": prepared.context, "question": question.text}

    session = StreamingQuerySession(FakeReader(), "u", (), "2025-10-01")
    cues = session.listen("How did I feel about the bus")
    assert cues["stage"] == "LISTENING"
    preview = session.anticipate()
    assert "about the bus" in preview.question
    final = session.speech_tail("transfer after it improved?")
    prepared = session.search()
    assert final == prepared.question
    assert prepared.question != preview.question
    assert calls == [preview.question, prepared.question]
    assert session.answer("q1")["answer"] == "Evidence for " + final
    assert len(calls) == 2  # no repeated retrieval for answer generation


def test_streaming_audio_chunks_accept_http_asr_adapter():
    class FakeASR:
        def transcribe_wav_bytes(self, data):
            return {b"first": "How did I feel", b"tail": "last month?"}[data]
    class FakeReader:
        pass
    query = StreamingQuerySession(FakeReader(), "u", ())
    query.listen_audio(b"first", FakeASR())
    assert query.speech_tail_audio(b"tail", FakeASR()) == "How did I feel last month?"
