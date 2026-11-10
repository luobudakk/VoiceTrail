"""Load VoiceTrail input datasets, including the published Voxemobench layout."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .scheduler import AudioTurn, Question, Session


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _audio_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Audio path must remain inside the dataset directory")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _voxemobench_root(directory: Path) -> Path | None:
    """Locate a unified public release, including its local staging directory."""
    candidates = (directory, directory / "PUBLIC_RELEASE_CANDIDATE")
    for candidate in candidates:
        data = candidate / "data"
        if ((data / "questions" / "questions.jsonl").is_file()
                and (data / "audio" / "metadata.jsonl").is_file()
                and (data / "text" / "sessions").is_dir()):
            return candidate
    return None


def _read_voxemobench(root: Path) -> tuple[list[dict], list[dict]]:
    """Join published text conversations to their source-attested WAV index."""
    data = root / "data"
    audio_root = (data / "audio").resolve()
    indexed_audio: dict[tuple[str, str, int], dict] = {}
    for row in read_jsonl(audio_root / "metadata.jsonl"):
        key = (str(row["user_id"]), str(row["session_id"]), int(row["turn_index"]))
        if key in indexed_audio:
            raise ValueError("Duplicate Voxemobench audio metadata key: " + str(key))
        if "split" in row:
            raise ValueError("Voxemobench audio metadata must use the unified layout")
        indexed_audio[key] = {
            "file": str(_audio_path(audio_root, str(row["file_name"]))),
            "text": str(row["text"]),
            "role": str(row["role"]),
            "date": str(row["date"]),
        }

    conversations = []
    observed_turns = set()
    for file in sorted((data / "text" / "sessions").glob("*.jsonl")):
        for session in read_jsonl(file):
            user = str(session["user_id"])
            sid = str(session["session_id"])
            if user != file.stem:
                raise ValueError("User ID does not match conversation filename")
            dialogue = []
            for idx, utterance in enumerate(session["dialogue"]):
                key = (user, sid, idx)
                if key not in indexed_audio:
                    raise ValueError("Missing Voxemobench audio metadata: " + str(key))
                if key in observed_turns:
                    raise ValueError("Duplicate Voxemobench turn identity: " + str(key))
                observed_turns.add(key)
                audio = indexed_audio[key]
                if (audio["text"] != str(utterance["content"])
                        or audio["role"] != str(utterance["role"])
                        or audio["date"] != str(session["date"])):
                    raise ValueError("Audio metadata does not match source turn: " + str(key))
                dialogue.append({
                    "role": utterance["role"],
                    "content": utterance["content"],
                    "audio_path": audio["file"],
                })
            conversations.append({
                "user_id": user,
                "session_id": sid,
                "date": session["date"],
                "dialogue": dialogue,
            })
    if observed_turns != set(indexed_audio):
        raise ValueError("Voxemobench audio index contains unmatched turns")

    questions = read_jsonl(data / "questions" / "questions.jsonl")
    if any("split" in item for item in questions):
        raise ValueError("Voxemobench questions must use the unified layout")
    return conversations, questions


def _question_cutoff(row: dict) -> str | None:
    cutoff = row.get("cutoff_record_date", row.get("as_of_record_date"))
    if cutoff is None:
        return None
    cutoff = str(cutoff)
    try:
        if date.fromisoformat(cutoff).isoformat() != cutoff:
            raise ValueError("invalid date format")
    except ValueError as exc:
        raise ValueError("Invalid question cutoff_record_date: " + cutoff) from exc
    return cutoff


def load_dataset(
    directory: str | Path, *,
    users: set[str] | None = None,
    sessions: set[str] | None = None,
    question_limit: int | None = None,
) -> tuple[list[Session], list[Question]]:
    root = Path(directory).resolve()
    if question_limit is not None and question_limit < 1:
        raise ValueError("question_limit must be positive")

    release_root = _voxemobench_root(root)
    if release_root is not None:
        conversation_rows, question_rows = _read_voxemobench(release_root)
        root = release_root
    else:
        conversation_rows = read_jsonl(root / "speech/conversations.jsonl")
        question_rows = read_jsonl(root / "qa/questions.jsonl")

    loaded = []
    user_set = set()
    keys = set()
    for session in conversation_rows:
        user = str(session["user_id"])
        sid = str(session["session_id"])
        if users is not None and user not in users:
            continue
        if sessions is not None and sid not in sessions:
            continue
        key = (user, sid)
        if key in keys:
            raise ValueError("Duplicate (user, session) in corpus")
        keys.add(key)
        recorded_date = str(session["date"])
        turns = []
        for index, turn in enumerate(session["dialogue"]):
            audio_path = _audio_path(root, str(turn["audio_path"]))
            turns.append(AudioTurn(
                user, sid, recorded_date, index, str(turn["role"]), str(audio_path)))
        loaded.append(Session(user, sid, recorded_date, tuple(turns)))
        user_set.add(user)

    questions = []
    qids = set()
    for row in question_rows:
        if str(row["user_id"]) not in user_set:
            continue
        qid = str(row["question_id"])
        if qid in qids:
            raise ValueError("Duplicate question ID")
        qids.add(qid)
        questions.append(Question(
            str(row["user_id"]), qid, str(row["question"]),
            _question_cutoff(row)))
        if question_limit is not None and len(questions) >= question_limit:
            break

    if not loaded or not questions:
        raise ValueError("No matching sessions and questions")
    return loaded, questions
