"""Conservative edge cases for bilingual events and emotion-free sessions."""
import json

import pytest

from voicetrail.adapters.reader import ReaderAdapter
from voicetrail.adapters.writer import WriterAdapter
from voicetrail.memory.event_identity import stable_event_id
from voicetrail.scheduler import Session


def test_chinese_event_key_stays_stable_across_spacing_variants():
    phrase = "深夜公交换乘"
    first = stable_event_id("u", phrase, phrase, ["这次" + phrase + "令人沮丧"])
    second = stable_event_id("u", "深夜 公交换乘", phrase, ["之后" + phrase + "有了进展"])
    assert first is not None
    assert second == first


def test_vague_two_word_event_key_cannot_link():
    assert stable_event_id("u", "bus transfer", "bus transfer",
                           ["I mentioned the bus transfer"]) is None


def test_writer_accepts_session_without_emotion():
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return {"choices": [{"message": {"content": json.dumps({
                "facts": [], "emotions": []})}}]}
    class HTTP:
        def post(self, *args, **kwargs):
            return Response()

    writer = WriterAdapter()
    writer.requests = HTTP()
    writer.writer_url = "https://example.invalid"
    writer.writer_headers = {}
    writer.writer_model = "fake"
    session = Session("user", "session_A", "2025-01-01", ())
    result = writer.writer(session, ({
        "role": "seeker", "turn_index": 0, "transcript": "I went to campus.",
    },))
    assert len(result["raw"]) == 1
    assert result["facts"] == []
    assert result["emotions"] == []


def test_direct_reader_rejects_invalid_record_date():
    reader = ReaderAdapter()
    with pytest.raises(ValueError, match="INVALID_RECORD_DATE"):
        reader.prepare_evidence("What happened?", "u",
                                [{"user_id": "u", "date": "2025-01-01"}],
                                cutoff_record_date="2025-99-99")
