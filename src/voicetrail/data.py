'''Load one unified conversation/QA corpus without bundling private data.'''
from __future__ import annotations
import json
from datetime import date
from collections import defaultdict
from pathlib import Path
from .scheduler import AudioTurn, Session, Question

def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def _audio_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Audio path must remain inside the dataset directory')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path

def load_dataset(
    directory: str | Path, *,
    users: set[str] | None = None,
    sessions: set[str] | None = None,
    question_limit: int | None = None,
) -> tuple[list[Session], list[Question]]:
    root = Path(directory).resolve()
    if question_limit is not None and question_limit < 1:
        raise ValueError('question_limit must be positive')

    conversation_rows = read_jsonl(root / 'speech/conversations.jsonl')
    question_rows = read_jsonl(root / 'qa/questions.jsonl')

    loaded = []
    user_set = set()
    keys = set()
    for session in conversation_rows:
        user = str(session['user_id'])
        sid = str(session['session_id'])
        if users is not None and user not in users:
            continue
        if sessions is not None and sid not in sessions:
            continue
        key = (user, sid)
        if key in keys:
            raise ValueError('Duplicate (user, session) in corpus')
        keys.add(key)
        date = str(session['date'])
        turns = []
        for index, turn in enumerate(session['dialogue']):
            audio_path = _audio_path(root, str(turn['audio_path']))
            turns.append(AudioTurn(
                user, sid, date, index, str(turn['role']), str(audio_path)))
        loaded.append(Session(user, sid, date, tuple(turns)))
        user_set.add(user)

    questions = []
    qids = set()
    for row in question_rows:
        if str(row['user_id']) not in user_set:
            continue
        qid = str(row['question_id'])
        if qid in qids:
            raise ValueError('Duplicate question ID')
        qids.add(qid)
        cutoff = row.get('cutoff_record_date', row.get('as_of_record_date'))
        if cutoff is not None:
            cutoff = str(cutoff)
            try:
                if date.fromisoformat(cutoff).isoformat() != cutoff:
                    raise ValueError('invalid date format')
            except ValueError as exc:
                raise ValueError('Invalid question cutoff_record_date: ' + cutoff) from exc
        questions.append(Question(
            str(row['user_id']), qid, str(row['question']), cutoff))
        if question_limit is not None and len(questions) >= question_limit:
            break

    if not loaded or not questions:
        raise ValueError('No matching sessions and questions')
    return loaded, questions
