# Changelog

## Methodology implementation refinements

- Add source-attested, per-user stable event keys with session-local fallback.
- Extract and validate affective assertion/time scope, plus optional source-backed aspect and target.
- Apply optional per-question conversation-record cutoffs before E5 retrieval.
- Split Reader evidence preparation from answer generation and expose retrieval/embedding latency.
- Add incremental voice query preparation and short-WAV chunk entrypoints.
- Add deterministic synthetic tests for longitudinal event identity, temporal coverage, cutoffs and streaming query resolution.

## 1.0.0 — modular source release

Organized ASR, Writer, Indexer, Retrieval, Reader, Scheduler and Storage into independent modules. Added environment-based configuration, tests, documentation and security checks.
