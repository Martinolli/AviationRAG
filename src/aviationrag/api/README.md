# API

Future ownership:

1. Internal Python service interfaces.
2. Bridge-facing request and response models.
3. Structured command handlers for chat, sessions, history, and ingestion.
4. Compatibility layers for worker and HTTP bridge integrations.

Current status: skeleton only. Runtime bridge behavior still lives in
`src/scripts/py_files/aviationai_worker.py`,
`src/scripts/py_files/aviationai_http_bridge.py`, and
`src/utils/server/aviation_api_bridge.ts`.
