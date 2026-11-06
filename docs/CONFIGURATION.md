# Configuration

Copy `.env.example` to `.env` and set deployment-specific values.

| Variable | Purpose |
| --- | --- |
| `VOICETRAIL_ASR_URL` | ASR HTTP service base URL |
| `VOICETRAIL_ASR_API_KEY` | ASR bearer token |
| `VOICETRAIL_WRITER_URL` | OpenAI-compatible Writer completion endpoint |
| `VOICETRAIL_WRITER_API_KEY` | Optional Writer authentication token |
| `VOICETRAIL_WRITER_MODEL` | Writer model identifier |
| `VOICETRAIL_READER_URL` | Reader HTTP completion endpoint |
| `VOICETRAIL_READER_API_KEY` | Optional Reader authentication token |
| `VOICETRAIL_READER_MODEL_NAME` | Reader model identifier |
| `VOICETRAIL_READER_MODEL_PATH` | Local Reader Transformer model directory |
| `VOICETRAIL_READER_DEVICE` | Local inference device |
| `VOICETRAIL_READER_GPU_FRACTION` | Fraction of local GPU memory |
| `VOICETRAIL_READER_CPU_THREADS` | Local Reader CPU threads |
| `VOICETRAIL_E5_MODEL` | Multilingual E5 model directory or model identifier |

Use `VOICETRAIL_READER_URL` for an HTTP Reader, or provide `VOICETRAIL_READER_MODEL_PATH` for local inference. ASR accepts base64 WAV JSON and returns a transcription; Writer and Reader use chat-completions JSON.

## Pipeline options

`--asr-workers`, `--writer-workers`, `--index-workers`, `--reader-workers` and `--queue-capacity` independently control concurrency.

All service endpoints, model weights, and credentials are external to this source release.
