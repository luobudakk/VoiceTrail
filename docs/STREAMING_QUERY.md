# Streaming-aware query preparation

VoiceTrail supports incremental spoken-query retrieval using `StreamingQuerySession`.
It shares the standard E5 routing, exact event-ID trajectory selection, and
budgeted evidence assembly implemented by the batch Reader.

## Basic flow

```python
from voicetrail.streaming import StreamingQuerySession

# reader = LiveStages(settings)
# indexed_sessions = previously indexed sessions for one user
query = StreamingQuerySession(
    reader=reader, user_id="user_1", snapshot=indexed_sessions,
    cutoff_record_date="2025-10-01",
)
cues = query.listen("How did I feel about the late night bus")
preview = query.anticipate()  # E5 embedding and preliminary candidates
query.speech_tail("transfer after the proposal?")
final_evidence = query.anticipate()  # optional; prepare the resolved query
answer = query.answer("query_42")  # Searching reuses final evidence
print(answer["answer"], answer["retrieval_latency_ms"])
```

A voice frontend can submit updated **cumulative** partial transcripts using
`listen(text)`; each revision invalidates earlier speculative evidence.
`speech_tail(final_segment)` appends the final segment.
If only a partial anticipation is available, Searching retrieves again using
the finalized complete utterance before generating an answer.
`answer_prepared()` generates without repeating the final retrieval.

A frontend can alternatively send self-contained short WAV chunks using
`listen_audio(wav_chunk, asr_adapter)` and
`speech_tail_audio(wav_chunk, asr_adapter)`. The existing HTTP
Qwen2.5-Omni-7B adapter transcribes each chunk separately, so acoustic
continuity, latency, and quality depend on the deployed ASR service.
This interface does **not** imply native token-level ASR streaming.

## Source-grounded event identities

During memory writing, the Writer requests an `event_key` (a specific,
consistent event reference) and an `event_anchor` (a literal substring of
the observation's source turn). `stable_event_id` normalizes the key,
requires distinct informative key tokens supported by the quoted anchor, and
hashes it per user. Only exact nonempty `event_id` equality connects
observations during query-time trajectory expansion. Unverified event keys
fall back to a session-local identifier.

An event key must be assigned consistently across sessions. Literal source
grounding alone cannot determine whether two differently named events are
identical, so the extraction model remains responsible for producing a stable
reference. Evaluate this requirement with longitudinal examples before
claiming full event-linking accuracy.

## Time, assertions, and telemetry

`Question.cutoff_record_date` or
`StreamingQuerySession.cutoff_record_date` accepts a recording-date ISO
value (`YYYY-MM-DD`). The Reader filters indexed sessions after the cutoff
**before** computing relevance. A query with no eligible sessions yields an
insufficient-information answer.

Affective observations retain literal quotations, source IDs, event IDs,
optional aspect/target, an assertion type (`EXPERIENCED`, `HYPOTHETICAL`,
`INTENDED`, `UNCLEAR`), and a time class (`CURRENT`, `PAST_RECALL`,
`FUTURE_ANTICIPATED`, `UNKNOWN`). Keyword safeguards override clearly
conditional or prospective Writer assertions. These labels remain extraction
outputs, not independently verified psychological states.

Reader outputs include `retrieval_latency_ms` (query embedding plus
deterministic evidence assembly) and `query_embedding_ms` separately from
`generation_seconds`. For a speculative partial query, that work is a
**preview**; final-query retrieval latency refers to the final evidence
preparation and does not represent end-to-end speech-to-speech latency.
