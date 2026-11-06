import asyncio
from pathlib import Path

from voicetrail import AudioTurn, Session, Question, Limits, VoiceTrailPipeline
from voicetrail.adapters.validation import (
    resolve_writer_turn_id, verify_firsthand_emotion_quote)
from voicetrail.retrieval import assemble_query, PIPELINE_POLICY


def test_async_pipeline_and_writer_receipts(tmp_path):
    calls = []

    def transcribe(turn):
        calls.append(('asr', turn.session_id, turn.turn_index))
        return 'I feel uneasy about the change.' if turn.role == 'seeker' else 'I understand.'

    def write_memory(session, turns):
        calls.append(('writer', session.session_id))
        assert len(turns) == 2
        return {
            'raw': [
                {'turn_id': session.session_id + ':0000',
                 'speaker': turns[0]['role'], 'text': turns[0]['transcript'],
                 'record_time': session.date, 'seeker_id': session.user_id},
            ],
            'facts': [{'fact_id': session.session_id + ':f001',
                       'source_turn_id': session.session_id + ':0000',
                       'text': 'The seeker discussed a change.'}],
            'emotions': [{'obs_id': session.session_id + ':o001',
                          'source_turn_id': session.session_id + ':0000',
                          'emotion': ['uneasy'],
                          'emotion_quote': 'I feel uneasy'}],
        }

    def index_memory(session, memory):
        calls.append(('index', session.session_id))
        assert memory['facts']
        return {'user_id': session.user_id, 'session_id': session.session_id,
                'date': session.date, 'memory': memory}

    def answer(question, snapshot):
        calls.append(('reader', question.question_id))
        assert all(row['user_id'] == question.user_id for row in snapshot)
        assert len(snapshot) == 2
        return {'answer': 'The seeker reported feeling uneasy.'}

    sessions = [
        Session('example_user', sid, date, (
            AudioTurn('example_user', sid, date, 0, 'seeker', 'example.wav'),
            AudioTurn('example_user', sid, date, 1, 'supporter', 'example.wav'),
        ))
        for sid, date in [('session_A', '2025-01-01'),
                          ('session_B', '2025-02-01')]
    ]
    questions = [Question('example_user', 'Q-1', 'How did the seeker feel?')]

    pipeline = VoiceTrailPipeline(
        asr=transcribe, writer=write_memory,
        indexer=index_memory, answer=answer,
        out_dir=tmp_path/'private_results',
        limits=Limits(asr=2, writer=2, index=1, reader=1, queue_capacity=2))
    result = asyncio.run(pipeline.run(sessions, questions))

    assert result['status'] == 'PASS'
    assert result['qa_completed'] == 1
    assert result['audio_turns'] == 4
    assert result['writer_jobs']['committed'] == 2
    assert result['writer_jobs']['failed'] == 0
    assert sum(item[0] == 'asr' for item in calls) == 4
    assert (tmp_path/'private_results'/'PIPELINE_REPORT.json').is_file()


def test_source_identity_checks():
    src = {'session_A:0001': {'transcript': 'I feel worried.'}}
    assert resolve_writer_turn_id('turn_0001', 'session_A', src) == 'session_A:0001'
    assert resolve_writer_turn_id('session_B:0001', 'session_A', src) is None
    assert verify_firsthand_emotion_quote('I feel worried', 'I feel worried.') is True
    assert verify_firsthand_emotion_quote('His hands were shaking',
                                           'His hands were shaking.') is False
    assert verify_firsthand_emotion_quote('I feel worried', 'He said something.') is False


def test_valid_pipeline_policy():
    assert isinstance(PIPELINE_POLICY, str)
    assert PIPELINE_POLICY
    assert callable(assemble_query)
