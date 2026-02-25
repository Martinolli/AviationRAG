# AviationRAG

AviationRAG is a Retrieval-Augmented Generation (RAG) project for aviation knowledge, certification material, and technical documents.  
It ingests documents, generates embeddings, stores vectors in Astra DB, and provides a chat interface with source-grounded answers.

## Quick Start (5 Minutes)

From the `AviationRAG` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
npm install
copy .env.example .env
```

Edit `.env` and set:

```env
OPENAI_API_KEY=
ASTRA_DB_SECURE_BUNDLE_PATH=
ASTRA_DB_APPLICATION_TOKEN=
ASTRA_DB_KEYSPACE=
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=
APP_AUTH_EMAIL=
APP_AUTH_PASSWORD=
```

Run the pipeline and chat:

```powershell
python src\scripts\py_files\aviationrag_manager.py
python src\scripts\py_files\aviationai.py
```

Quick Astra check:

```powershell
node src/scripts/js_files/check_astra.js
```

## Current Status

- Ingestion pipeline is stable.
- Astra DB auth uses `ASTRA_DB_APPLICATION_TOKEN` (token auth only).
- Document ingestion supports both `.docx` and `.pdf`.
- Chat supports document-grounded responses with citations for document-specific questions.

## Core Features

- Multi-format ingestion: `DOCX` + `PDF`
- Chunking and embedding generation
- Vector storage in Astra DB
- Local FAISS retrieval for chat
- Document-grounded strict mode for queries like:
  - `According to "Introduction Flight Test Engineer"...`
- Citation-friendly context blocks (`filename`, `chunk_id`)
- Session history stored in Astra DB

## Tech Stack

- Python: ingestion, chunking, chat orchestration
- Node.js: Astra DB checks/storage utilities
- OpenAI API: embeddings + answer generation
- Astra DB (Cassandra): vector/chat storage
- FAISS: local semantic retrieval
- Next.js: frontend app scaffold

## Project Layout

```text
AviationRAG/
  data/
    documents/                 # Input files (.docx, .pdf)
    raw/                       # aviation_corpus.pkl
    processed/
      chunked_documents/
      aviation_corpus.json
    embeddings/
      aviation_embeddings.json
    astra_db/
      astra_db_content.json
  logs/
  src/
    scripts/
      py_files/
      js_files/
  docs/
    WEB_APP_REQUIREMENTS_V1.md
```

## Prerequisites

- Python 3.11+ (3.12 works)
- Node.js 20+
- OpenAI API key
- Astra DB serverless database + Secure Connect Bundle

## Setup

From the `AviationRAG` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
npm install
```

## Environment Variables

Copy and edit:

```powershell
copy .env.example .env
```

Required values:

```env
OPENAI_API_KEY=
ASTRA_DB_SECURE_BUNDLE_PATH=
ASTRA_DB_APPLICATION_TOKEN=
ASTRA_DB_KEYSPACE=
NEXTAUTH_URL=http://localhost:3000
NEXTAUTH_SECRET=
APP_AUTH_EMAIL=
APP_AUTH_PASSWORD=
```

Notes:

- `ASTRA_DB_CLIENT_ID` / `ASTRA_DB_CLIENT_SECRET` are no longer used.
- Keep `.env` out of version control.

## Run the Full Pipeline

```powershell
python src\scripts\py_files\aviationrag_manager.py
```

Pipeline steps:

1. Read Documents
2. Chunk Documents
3. Extract PKL to JSON
4. Check PKL Content
5. Generate New Embeddings
6. Check Embeddings
7. Store New Embeddings in AstraDB
8. Check AstraDB Content
9. Check AstraDB Consistency
10. Update Visualizing Data

## Run a Single Step

```powershell
python src\scripts\py_files\aviationrag_manager.py --step "Read Documents"
python src\scripts\py_files\aviationrag_manager.py --step "Chunk Documents"
python src\scripts\py_files\aviationrag_manager.py --step "Generate New Embeddings"
python src\scripts\py_files\aviationrag_manager.py --step "Store New Embeddings in AstraDB"
```

## Astra DB Checks

```powershell
node src/scripts/js_files/check_astra.js
node src/scripts/js_files/check_astradb_content.js
node src/scripts/js_files/check_astradb_consistency.js
```

`check_astradb_content.js` defaults to a lightweight query.  
Use `--full` only when you need full payload export:

```powershell
node src/scripts/js_files/check_astradb_content.js --full
```

## Chat Interface (CLI)

```powershell
python src\scripts\py_files\aviationai.py
```

Tips:

- Use document-specific phrasing for strict grounding, for example:
  - `According to "Introduction Flight Test Engineer" from AGARD...`
- Use `quit` / `exit` to close.

## PDF Ingestion Notes

PDF extraction now uses multiple strategies and keeps extraction metadata:

- `source_type`
- `extraction_method`
- `extraction_quality`
- `needs_manual_review`

If a PDF is scanned/image-heavy, extraction quality may be low and manual/OCR review may still be needed.

## Local Quality Checks

```powershell
.\local_check.ps1
```

This runs:

- Python compile check
- Python unit smoke tests
- JavaScript syntax checks

## Frontend Development

```powershell
npm run dev
```

Open `http://localhost:3000/auth/signin` and sign in with:

- `APP_AUTH_EMAIL`
- `APP_AUTH_PASSWORD`

### Web API Endpoints (MVP)

- `GET /api/health`
- `POST /api/chat/ask`
  - body: `session_id?`, `message`, `strict_mode?`, `target_document?`, `model?`, `store?`
- `GET /api/chat/history/{session_id}?limit=10`
- `GET /api/chat/session?search=&filter=all&limit=50`
- `POST /api/chat/session`
- `PATCH /api/chat/session/{id}`
- `DELETE /api/chat/session/{id}`

Note: All `/api/chat/*` routes require an authenticated session.

Example:

```powershell
curl -X POST http://localhost:3000/api/chat/ask `
  -H "Content-Type: application/json" `
  -d "{\"message\":\"What is SMS in aviation?\",\"store\":false}"
```

For product/UI scope and phased requirements, see:

- [WEB_APP_REQUIREMENTS_V1.md](./docs/WEB_APP_REQUIREMENTS_V1.md)

## Troubleshooting

### `can't open file ... No such file or directory`

You are likely running from the wrong folder.  
Run commands from `AviationRAG` root or use full relative path from parent.

### Astra auth error: deprecated method

Use `ASTRA_DB_APPLICATION_TOKEN` in `.env` and remove old credential usage.

### `OperationTimedOutError` in Astra content check

Use default lightweight mode first:

```powershell
node src/scripts/js_files/check_astradb_content.js
```

### Pipeline says no new chunks/embeddings

This is normal if no new documents were added.  
Add a new file to `data/documents` and rerun `Read Documents`.

## Disclaimer

This project is a research/engineering tool.  
Do not use outputs as the sole authority for operational, legal, or certification decisions.  
Always verify against official FAA/EASA/ICAO documents and approved engineering processes.
