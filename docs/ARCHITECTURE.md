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
- The Writer attaches exact quotes and classifies first-person assertion and time scope. Optional aspect and target must appear verbatim in their source turn.
- Source-attested event keys produce stable per-user event IDs. Query-time trail expansion groups observations by stored exact-match event IDs; session-local labels stay scoped to their originating session.
- Dynamic budgets select dialogue, facts and affective observations and omit redundant facts tied to already-selected dialogue turns.

## Query-time and streaming preparation

- The `Question` object optionally includes `cutoff_record_date` (`YYYY-MM-DD`); later sessions are excluded before E5 scoring.
- `ReaderAdapter.prepare_evidence` isolates E5 query embedding and deterministic source-aware retrieval from `answer_prepared`, enabling reusable finalized evidence and separate retrieval latency.
- `StreamingQuerySession` coordinates Listening (partial transcript/cues), Anticipation (early evidence), Speech Tail (final segment), and Searching (resolved query and answer). The Qwen2.5-Omni adapter processes short complete WAV chunks; a streaming-capable ASR backend enables native token-level streaming.
- Cross-session continuity uses consistently normalized, source-attested event keys. Semantic similarity ranks evidence while stored event identity determines trajectory membership.
