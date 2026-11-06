# VoiceTrail

**Event-Conditioned Emotion Trajectories for Long-Term Voice Memory**

[简体中文](README.zh-CN.md) · [Architecture](docs/ARCHITECTURE.md) · [Configuration](docs/CONFIGURATION.md) · [Data Format](docs/DATA_FORMAT.md)

VoiceTrail is a modular system for long-term memory in voice conversations. It represents conversational facts and affective observations with temporal, event, and speaker attribution. At question-answering time, VoiceTrail combines semantic retrieval with event-conditioned emotion trajectories to select relevant historical evidence and generate grounded answers.

## Method

VoiceTrail integrates speech processing, memory construction, and evidence-grounded reasoning in a staged pipeline:

```text
Voice Sessions
    |
    v
ASR: Speech Transcription
    |
    v
Writer: Factual and Affective Memory
    |
    v
Indexer: E5 Semantic Embeddings
    |
    v
VoiceTrail Retrieval
    +-- Schema-Aware Routing
    +-- Event-Conditioned Emotion Trail
    +-- Dynamic Evidence Budget
    |
    v
Reader: Evidence-Grounded Longitudinal QA
```

The system has three central components:

- **Joint factual and affective memory:** observations are linked to original turns with speaker, session, and temporal metadata.
- **Event-conditioned emotion trajectories:** cross-session affective changes are organized around events while preserving person attribution.
- **Evidence coordination and dynamic budgeting:** fact, affective, and raw dialogue evidence are routed and selected according to the question.

## Pipeline Scheduling

The ASR, Writer, Indexer, and Reader are implemented as independent stages. Bounded `asyncio.Queue` workers support overlapping execution and backpressure. The Writer uses a persistent SQLite job queue for leases, retries, and commit tracking. Worker concurrency is configurable per stage, and the pipeline records processing receipts.

## Code Structure

| Module | Location | Responsibility |
| --- | --- | --- |
| Scheduler | `src/voicetrail/scheduler.py` | Async stage orchestration |
| ASR | `src/voicetrail/adapters/asr.py` | Speech transcription |
| Writer | `src/voicetrail/adapters/writer.py` | Factual and affective memory extraction |
| Validation | `src/voicetrail/adapters/validation.py` | Source and speaker attribution |
| Indexer | `src/voicetrail/adapters/indexing.py` | E5 embedding construction |
| Retrieval | `src/voicetrail/retrieval/` | Routing, trajectory selection, and evidence budgeting |
| Reader | `src/voicetrail/adapters/reader.py` | Evidence-grounded answer generation |
| Storage | `src/voicetrail/storage/` | Persistent Writer jobs |
| Runtime | `src/voicetrail/config.py`, `src/voicetrail/cli.py` | Configuration and CLI |

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
pip install -e ".[live,test]"
```

Copy `.env.example` to `.env` and configure the ASR, Writer, Reader, and E5 models. See [Configuration](docs/CONFIGURATION.md) for supported options.

## Usage

VoiceTrail accepts a unified speech-session and QA format, including **Voxemobench**. See [Data Format](docs/DATA_FORMAT.md) for the input schema.

Validate a dataset:

```bash
voicetrail-run --dataset /path/to/dataset --preflight
```

Run the pipeline:

```bash
voicetrail-run --dataset /path/to/dataset --output /path/to/output
```

Use `--asr-workers`, `--writer-workers`, `--index-workers`, `--reader-workers`, and `--queue-capacity` to control stage concurrency. The `--users`, `--sessions`, and `--question-limit` options select workloads.

## Testing

```bash
python -m pytest -q
```

The test suite covers scheduling, job receipts, source and speaker attribution, dataset loading, and input path validation.

## License

[Apache License 2.0](LICENSE).
