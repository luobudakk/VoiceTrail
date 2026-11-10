# Unified input format

Datasets are kept **outside this repository**.

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

The optional `cutoff_record_date` (alias: `as_of_record_date`) specifies the latest eligible **conversation record date**, as ISO `YYYY-MM-DD`. If omitted, the Reader uses the latest indexed session date. A question whose cutoff precedes every session receives an insufficient-information response.

Writer output stores emotional observations with `source_turn_id`, `emotion_quote`, `emotion_assertion`, `time_class`, and optional `event_id`, `event_key`, `event_anchor`, `aspect`, and `target`. Trajectory linkage depends on exact stored `event_id` equality, not similarity-based identity inference.

The loader validates unique IDs, checks audio existence, and prevents audio paths from escaping the dataset root. No split field is required. Reference answers are scoring-only assets and are not read by the memory pipeline.
