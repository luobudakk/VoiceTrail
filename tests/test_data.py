import json
import wave

import pytest
from voicetrail.data import load_dataset

def test_unified_dataset_loader(tmp_path):
    audio = tmp_path/'speech/audio/example_user/session_A/turn_0000.wav'
    audio.parent.mkdir(parents=True)
    with wave.open(str(audio),'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(b'\x00\x00'*240)

    conversation = {
        'user_id':'example_user', 'session_id':'session_A',
        'date':'2025-01-01',
        'dialogue':[{
            'role':'seeker','content':'I feel better.',
            'speaker':'Example',
            'audio_path':'speech/audio/example_user/session_A/turn_0000.wav'}]}
    q={'question_id':'example_qa','user_id':'example_user',
       'question':'How does the seeker feel?'}
    (tmp_path/'speech/conversations.jsonl').write_text(
        json.dumps(conversation)+'\n',encoding='utf8')
    (tmp_path/'qa').mkdir()
    (tmp_path/'qa/questions.jsonl').write_text(
        json.dumps(q)+'\n',encoding='utf8')

    sessions,questions=load_dataset(tmp_path)
    assert len(sessions)==len(questions)==1
    assert len(sessions[0].turns)==1
    assert sessions[0].turns[0].audio_path==str(audio)

def test_reject_path_escape(tmp_path):
    (tmp_path/'speech').mkdir()
    (tmp_path/'qa').mkdir()
    (tmp_path/'speech/conversations.jsonl').write_text(
        json.dumps({'user_id':'example_user','session_id':'session_A',
            'date':'2025-01-01','dialogue':[{'role':'seeker',
            'content':'Hello','audio_path':'../../outside.wav'}]})+'\n')
    (tmp_path/'qa/questions.jsonl').write_text(
        json.dumps({'question_id':'Q1','user_id':'example_user',
                    'question':'What?'})+'\n')
    with pytest.raises(ValueError,match='inside'):
        load_dataset(tmp_path)
