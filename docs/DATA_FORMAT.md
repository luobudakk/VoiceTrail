# Unified input format

VoiceTrail loads the published **Voxemobench** dataset directly, and also supports
a generic speech-session / QA JSONL format.

## Voxemobench public release

Use the dataset directory containing `data/`, or its local parent containing
`PUBLIC_RELEASE_CANDIDATE/`:

```text
Voxemobench/
  PUBLIC_RELEASE_CANDIDATE/
    data/
      text/
        sessions/
          user_001.jsonl
          ...
          user_020.jsonl
      questions/
        questions.jsonl
      audio/
        metadata.jsonl
        user_001/
          session_000/
            turn_0000.wav
        ...
```

All **720 QA questions** form a single evaluation set, with no development/test
split in the published schema. The session files provide `user_id`,
`session_id`, `date`, and ordered `dialogue` entries containing
`role` and `content`. The `questions.jsonl` file stores
`question_id`, `user_id`, `group`, `question_type`, and `question`.

Each WAV is linked to its source utterance through
`data/audio/metadata.jsonl` using `(user_id, session_id, turn_index)`.
The audio path is relative to `data/audio/` and follows
`<user_id>/<session_id>/turn_<index>.wav`. VoiceTrail verifies the
audio path, source text, role, date, and turn identity at dataset loading.

Validate the downloaded dataset directly:

```bash
voicetrail-run --dataset /path/to/Voxemobench/PUBLIC_RELEASE_CANDIDATE --preflight
```

## Generic JSONL format

```text
dataset/
  speech/
    conversations.jsonl
    audio/example_user/session_A/turn_0000.wav
  qa/
    questions.jsonl
```

Session object (one JSON object per line):

```json
{"user_id":"example_user","session_id":"session_A","date":"2025-01-01","dialogue":[{"role":"seeker","content":"An example sentence.","speaker":"Example","audio_path":"speech/audio/example_user/session_A/turn_0000.wav"}]}
```

Question object (one JSON object per line):

```json
{"question_id":"example_q_001","user_id":"example_user","question":"What did the person describe?","cutoff_record_date":"2025-01-01"}
```

The optional `cutoff_record_date` (alias: `as_of_record_date`) selects
the latest eligible **conversation record date**, in ISO `YYYY-MM-DD`
format. When omitted, the Reader uses all indexed sessions for that user.

Writer output stores emotional observations with `source_turn_id`,
`emotion_quote`, `emotion_assertion`, `time_class`, and optional
`event_id`, `event_key`, `event_anchor`, `aspect`, and `target`.
Event-conditioned trajectory retrieval groups observations by their exact
stored `event_id`.
