# Security and GitHub publication

## Repository contents

This repository contains **source code, public documentation, configuration templates, and tests only**.

Excluded: credentials, private-network addresses, user home paths, private or synthetic audio, conversations, Gold answers, datasets, model weights, prediction files, scoring outputs, runtime logs, experimental results, embedding caches and SQLite data.

## Pre-publish checks

```bash
python scripts/security_scan.py
python -m pytest -q
git status --short
git diff --cached --name-only
```

`.gitignore` excludes local configurations and common private artifact formats. The security scanner detects forbidden files, absolute user paths, private IP ranges, participant IDs and literal credential values.

All model tokens and service URLs belong in `.env` or an external secret store. `.env.example` uses placeholders only.

## Privacy and evaluation separation

The model pipeline consumes source conversations and public questions only. Gold answers are never supplied to the Writer or Reader. Evaluation reports and experimental metrics are kept outside the code repository.
