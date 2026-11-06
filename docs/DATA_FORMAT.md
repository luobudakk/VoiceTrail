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
{"question_id":"example_q_001","user_id":"example_user","question":"What did the person describe?"}
```

The loader validates unique IDs, checks audio existence, and prevents audio paths from escaping the dataset root. No split field is required. Reference answers are scoring-only assets and are not read by the memory pipeline.
