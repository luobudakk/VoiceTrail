"""Tests for HTTP adapters using synthetic transcripts and in-memory services."""
import json
from types import SimpleNamespace

from voicetrail.adapters.asr import ASRAdapter
from voicetrail.adapters.writer import WriterAdapter
from voicetrail.scheduler import AudioTurn, Session, Question


class Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class FakeHTTP:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def post(self, *args, **kwargs):
        self.requests.append((args, kwargs))
        return Response(self.payload)


def test_asr_adapter_transcribes_audio(tmp_path):
    wav = tmp_path / 'turn.wav'
    wav.write_bytes(b'RIFF' + b'\x00' * 128)
    stage = ASRAdapter()
    stage.requests = FakeHTTP({
        'model': 'Qwen2.5-Omni-7B',
        'text': 'I am feeling calmer.',
    })
    stage.asr_url = 'https://example.invalid/asr'
    stage.asr_token = 'synthetic-test-token'
    turn = AudioTurn('example_user', 'session_A', '2025-01-01',
                     0, 'seeker', str(wav))
    assert stage.asr(turn) == 'I am feeling calmer.'
    assert stage.requests.requests[0][1]['headers']['Authorization'] == (
        'Bearer synthetic-test-token')


def test_writer_rejects_third_party_emotion():
    payload = {
        'facts': [{'source_turn_id': 'session_A:0000',
                   'text': 'The speaker discussed an event.'}],
        'emotions': [
            {'source_turn_id': 'session_A:0000',
             'emotion_quote': 'I feel nervous',
             'emotion': ['nervous'], 'event_label': 'new_event'},
            {'source_turn_id': 'session_A:0000',
             'emotion_quote': 'His hands were shaking',
             'emotion': ['fear'], 'event_label': 'other_person'},
        ],
    }
    stage = WriterAdapter()
    stage.requests = FakeHTTP({
        'choices': [{'message': {'content': json.dumps(payload)}}],
        'model': 'example-model',
        'usage': {'total_tokens': 32}
    })
    stage.writer_url = 'https://example.invalid/v1/chat/completions'
    stage.writer_model = 'example-model'
    stage.writer_headers = {}
    session = Session('example_user', 'session_A', '2025-01-01', ())
    rows = ({
        'turn_index': 0, 'role': 'seeker',
        'transcript': 'I feel nervous. His hands were shaking.'
    },)
    result = stage.writer(session, rows)
    assert len(result['facts']) == 1
    assert len(result['emotions']) == 1
    assert len(result['rejected']) == 1
    assert result['emotions'][0]['emotion_quote'] == 'I feel nervous'
    assert result['emotions'][0]['source_turn_id'] == 'session_A:0000'


def test_reader_http_route(monkeypatch):
    import pytest
    np = pytest.importorskip('numpy')
    import voicetrail.retrieval as retrieval
    from voicetrail.adapters.reader import ReaderAdapter

    def fake_assemble(question, **kwargs):
        assert kwargs['user_id'] == 'example_user'
        assert kwargs['schema_enabled'] and kwargs['trail_enabled']
        return SimpleNamespace(
            context='Source: I feel nervous.',
            receipt={
                'pipeline': retrieval.PIPELINE_POLICY,
                'budget_receipt': {'selected_count': 1},
                'trail': {'enabled': True},
            })
    monkeypatch.setattr(retrieval, 'assemble_query', fake_assemble)

    class E5:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0]], dtype=np.float32)

    stage = ReaderAdapter()
    stage.e5 = E5()
    stage.reader_url = 'https://example.invalid/v1/chat/completions'
    stage.reader_model_name = 'example-reader'
    stage.reader_model_path = None
    stage.reader_tokenizer = None
    stage.reader_headers = {}
    stage.requests = FakeHTTP({
        'model': 'example-reader',
        'choices': [{'message': {'content': 'The seeker felt nervous.'}}],
        'usage': {'prompt_tokens': 20, 'completion_tokens': 6},
    })
    snapshot = ({
        'user_id': 'example_user',
        'session_id': 'session_A',
        'date': '2025-01-01',
        'raw': [{'turn_id': 'session_A:0000', 'text': 'I feel nervous.'}],
        'facts': [],
        'emotions': [],
        'vectors': {'session_A:0000': [1.0, 0.0]},
    },)
    result = stage.answer(Question('example_user', 'Q1', 'How?'), snapshot)
    assert result['answer'] == 'The seeker felt nervous.'
    assert result['trail_enabled'] is True
    assert result['total_prompt_tokens'] == 20
