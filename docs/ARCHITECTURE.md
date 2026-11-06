# Architecture

## Concurrent stages

1. **ASR** transcribes session audio and preserves source speaker, date, and turn identity.
2. **Writer** extracts source-bound facts and firsthand seeker affective observations.
3. **Indexer** encodes historical raw turns, facts, and affective memory with normalized multilingual E5 embeddings.
4. **Reader** retrieves event-conditioned and schema-routed evidence, respects date and user restrictions, and answers questions.

Bounded asynchronous queues overlap these stages; the Writer uses a durable lease-guarded SQLite job queue. Readers start when their user's required indexed sessions are complete. Intermediate stage results are immutable.

## Callable contracts

```python
asr(audio_turn) -> str
writer(session, transcript_rows) -> dict
indexer(session, memory) -> dict
answer(question, indexed_sessions) -> dict
```

The adapters are independent, allowing model providers and inference devices to change without modifying the scheduler.

## Retrieval and attribution

- Raw turns have source IDs, speaker roles and conversation dates.
- Facts retain their originating turns.
- Seeker emotion observations require firsthand supporting language and a matching seeker role.
- Event-conditioned trails connect supported observations without confusing third-party feelings with the seeker's emotions.
- Dynamic budgets select evidence before the Reader receives context.

All outputs are written into a caller-selected private directory, not committed to source control.
