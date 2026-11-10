# Security and responsible publication

## Runtime secrets and data

Configure API keys and service endpoints via `.env` or a dedicated secret store. Keep conversation transcripts, audio, identifiable participant information, model assets, and evaluation artifacts in appropriately access-controlled storage. Use placeholders in `.env.example`.

## Publication checks

```bash
python scripts/security_scan.py
python -m pytest -q
git status --short
git diff --cached --name-only
```

The security scanner checks for sensitive files, local paths, private network addresses, participant identifiers, and credentials. Review the Git diff and staged file list before publication.

## Evaluation integrity

The Writer and Reader receive historical dialogue evidence and the question. Scoring references are supplied to the evaluation stage separately, preserving answer independence. Source attribution, user isolation and per-question record-date cutoffs are enforced by the retrieval pipeline.