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

def _make_published_voxemobench(tmp_path):
    release = tmp_path / "PUBLIC_RELEASE_CANDIDATE"
    text_dir = release / "data/text/sessions"
    question_dir = release / "data/questions"
    audio_root = release / "data/audio"
    text_dir.mkdir(parents=True)
    question_dir.mkdir(parents=True)
    audio = audio_root / "example_user/session_A/turn_0000.wav"
    audio.parent.mkdir(parents=True)
    with wave.open(str(audio), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(24000)
        stream.writeframes(b"\x00\x00" * 240)

    conversation = {
        "user_id": "example_user",
        "session_id": "session_A",
        "date": "2025-01-01",
        "dialogue": [{"role": "seeker", "content": "I feel better."}],
    }
    metadata = {
        "user_id": "example_user", "session_id": "session_A",
        "turn_index": 0, "date": "2025-01-01", "role": "seeker",
        "text": "I feel better.",
        "file_name": "example_user/session_A/turn_0000.wav",
    }
    question = {
        "question_id": "example_qa", "user_id": "example_user",
        "group": "ESC", "question_type": "EMOTION_STATE_RECALL",
        "question": "How does the seeker feel?",
    }
    (text_dir / "example_user.jsonl").write_text(json.dumps(conversation) + "\n", encoding="utf8")
    (audio_root / "metadata.jsonl").write_text(json.dumps(metadata) + "\n", encoding="utf8")
    (question_dir / "questions.jsonl").write_text(json.dumps(question) + "\n", encoding="utf8")
    return release, audio


@pytest.mark.parametrize("from_parent", [False, True])
def test_published_voxemobench_unified_layout(tmp_path, from_parent):
    release, wav = _make_published_voxemobench(tmp_path)
    dataset = tmp_path if from_parent else release
    sessions, questions = load_dataset(dataset)
    assert len(sessions) == len(questions) == 1
    assert sessions[0].turns[0].audio_path == str(wav)
    assert sessions[0].turns[0].role == "seeker"
    assert questions[0].question_id == "example_qa"


def test_published_voxemobench_source_text_mismatch(tmp_path):
    release, _ = _make_published_voxemobench(tmp_path)
    metadata_path = release / "data/audio/metadata.jsonl"
    metadata = json.loads(metadata_path.read_text(encoding="utf8"))
    metadata["text"] = "This is a different person's text."
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf8")
    with pytest.raises(ValueError, match="does not match source turn"):
        load_dataset(release)


def test_published_voxemobench_rejects_split_fields(tmp_path):
    release, _ = _make_published_voxemobench(tmp_path)
    q_path = release / "data/questions/questions.jsonl"
    row = json.loads(q_path.read_text(encoding="utf8"))
    row["split"] = "DEV"
    q_path.write_text(json.dumps(row) + "\n", encoding="utf8")
    with pytest.raises(ValueError, match="unified layout"):
        load_dataset(release)


def test_published_voxemobench_audio_path_escape(tmp_path):
    release, _ = _make_published_voxemobench(tmp_path)
    meta_path = release / "data/audio/metadata.jsonl"
    row = json.loads(meta_path.read_text(encoding="utf8"))
    row["file_name"] = "../../../outside.wav"
    meta_path.write_text(json.dumps(row) + "\n", encoding="utf8")
    with pytest.raises(ValueError, match="inside"):
        load_dataset(release)
