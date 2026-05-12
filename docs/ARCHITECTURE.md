# AviationRAG Architecture

Date: 2026-05-12  
Status: Current baseline documentation  
Scope: Existing system only, with planned architecture clearly marked

## Executive Summary

AviationRAG is a retrieval-augmented generation system for aviation knowledge, certification material, safety material, and technical documents. It ingests local source documents, creates processed text and embeddings, stores vectors and chat data, and exposes a web chat interface that can return source-grounded answers.

The project is past the prototype-only stage but is not yet a compliance-grade RAG platform. The current system has working ingestion, chat, session handling, upload plumbing, API hardening, CI checks, and a dual bridge model for local and Vercel deployments. The main deployment blocker is still external: Vercel production needs a healthy public HTTPS aviation HTTP bridge endpoint.

Current maturity level:

1. Repository hygiene is in place for runtime/generated/private data.
2. Local bridge and local deep health validation work when the local bridge bind env is corrected.
3. Production is configured for HTTP bridge mode, but production deep health still fails `checks.aviation_http_ping`.
4. Retrieval quality, formal response policy enforcement, citation validation, and evaluation harness work are planned but not implemented.

## Current High-Level Architecture

The current system is a hybrid Next.js and Python application.

```text
User
  |
  v
Next.js frontend
  |
  v
Next.js API routes
  |
  v
Bridge adapter: src/utils/server/aviation_api_bridge.ts
  |                         |
  | worker mode             | http mode
  v                         v
Python child process        External HTTP bridge service
aviationai_worker.py        aviationai_http_bridge.py
  |                         |
  +------------+------------+
               v
Python RAG scripts
  |
  +-- FAISS local semantic retrieval over generated embeddings
  +-- OpenAI embeddings and chat completions
  +-- Astra DB chat/vector storage utilities
```

Main components:

1. Next.js frontend:
   - `pages/index.tsx`
   - `components/layout/AppShell.tsx`
   - `components/sidebar/SessionSidebar.tsx`
   - `components/sources/SourceDrawer.tsx`
   - CSS modules under `styles/`
2. API routes:
   - `pages/api/health.ts`
   - `pages/api/auth/[...nextauth].ts`
   - `pages/api/chat/*`
   - document upload/status routes when present in the current branch history
3. Bridge adapter:
   - `src/utils/server/aviation_api_bridge.ts`
   - Supports `AVIATION_API_MODE=worker` and `AVIATION_API_MODE=http`.
4. Python backend scripts:
   - `src/scripts/py_files/aviationrag_manager.py`
   - `src/scripts/py_files/read_documents.py`
   - `src/scripts/py_files/aviation_chunk_saver.py`
   - `src/scripts/py_files/aviationai.py`
   - `src/scripts/py_files/aviationai_worker.py`
   - `src/scripts/py_files/aviationai_http_bridge.py`
5. Storage and retrieval:
   - Local FAISS index built from generated embedding JSON.
   - Astra DB used for vector/chat related storage through Python and Node helper scripts.
6. OpenAI usage:
   - Embedding generation.
   - Answer generation through chat completions.
7. Session/chat storage:
   - Chat exchanges are stored through Astra helper scripts.
   - Runtime session metadata is kept under `chat_id/` and is ignored by Git.

## Current Request Flow

Typical web chat flow:

```text
User submits message
  -> Next.js page
  -> /api/chat/ask
  -> runAviationApiCommand(...)
  -> worker mode or HTTP bridge mode
  -> aviationai_worker.py or aviationai_http_bridge.py
  -> aviationai.answer_query(...)
  -> query embedding generated
  -> FAISS retrieval selects context chunks
  -> OpenAI chat completion generates answer
  -> citations/sources extracted from context
  -> answer returned to API
  -> response rendered in chat UI
```

Session actions follow the same bridge pattern using actions such as `history`, `sessions_list`, `session_upsert`, and `session_delete`.

Health flow:

```text
GET /api/health
  -> reports env/config readiness

GET /api/health?deep=1
  -> in http mode, sends bridge ping to /command
  -> reports checks.aviation_http_ping and deep_check_error
```

## Current Data Flow

Document ingestion flow:

```text
data/documents/
  -> read_documents.py
  -> data/raw/aviation_corpus.pkl
  -> processed text under data/processed/
  -> aviation_chunk_saver.py
  -> chunked documents
  -> JavaScript embedding scripts
  -> data/embeddings/
  -> Astra DB storage utilities
  -> FAISS loads embedding JSON for runtime retrieval
```

Current ingestion characteristics:

1. Source files are local `.pdf` and `.docx` documents.
2. PDF extraction uses multiple extraction paths and records extraction metadata such as method, quality, and manual-review signal.
3. Processed text, chunk outputs, embeddings, Astra exports, logs, and visualizations are generated artifacts.
4. Generated/private data folders are ignored by Git.
5. Only optional tiny sample data under `data/sample_documents/` is intended for public tracking.

Current retrieval and generation:

1. A user query is embedded with OpenAI embeddings.
2. FAISS searches generated local embeddings.
3. The Python answer service builds a context block with source tags.
4. The prompt asks OpenAI for a source-grounded aviation answer.
5. Citations are extracted from source tags, not independently validated.

## Current Deployment Modes

### Local Worker Mode

`AVIATION_API_MODE=worker` lets the Next.js API spawn `aviationai_worker.py` as a local child process. This is suitable for local development where Python and all local data files are available.

Limitations:

1. Not suitable for Vercel serverless production.
2. Requires Python, dependencies, local generated data, and environment variables on the same machine as the Next.js runtime.

### Local HTTP Bridge Mode

`AVIATION_API_MODE=http` with `AVIATION_API_HTTP_URL=http://127.0.0.1:8010` lets Next.js call the local Python bridge service over HTTP.

Typical local sequence:

```powershell
python src\scripts\py_files\aviationai_http_bridge.py
npm run dev
```

Current status:

1. Local bridge health and command ping work when the local bind env is corrected.
2. Local app deep health can report `checks.aviation_http_ping=true`.

### Vercel HTTP Bridge Mode

Vercel must use `AVIATION_API_MODE=http` and call a public HTTPS bridge endpoint.

Required Vercel variables:

```env
AVIATION_API_MODE=http
AVIATION_API_HTTP_URL=https://<public-bridge-url>
AVIATION_API_HTTP_TOKEN=<shared-token>
```

Current status:

1. Production reports `bridge_mode="http"`.
2. Production reports `checks.aviation_http_url_set=true`.
3. Production still reports `checks.aviation_http_ping=false`.
4. The blocker is a missing or unhealthy public HTTPS bridge endpoint.

## Current Repository Structure

```text
components/                 React UI components
config/                     Local configuration assets
data/                       Local source/generated data, mostly ignored
docs/                       Operational and baseline documentation
pages/                      Next.js pages and API routes
public/                     Static web assets
src/interface/              Streamlit and interface experiments
src/scripts/js_files/       Node.js Astra/embedding/check helper scripts
src/scripts/py_files/       Python ingestion, retrieval, chat, bridge scripts
src/aviationrag/            Future backend package skeleton, not runtime source yet
src/utils/server/           Next.js server utilities and bridge adapter
styles/                     CSS modules and global styles
tests/                      Python and Playwright smoke tests
tools/                      Sanitization and bridge check utilities
types/                      Shared TypeScript types
```

Important ignored runtime/generated folders:

```text
chat_id/
chat/
logs/
data/documents/
data/raw/
data/processed/
data/embeddings/
data/astra_db/
assets/pictures/
```

## Known Architectural Debt

1. Backend orchestration is still script-based; `src/aviationrag/` exists only as a package skeleton.
2. Python and Node responsibilities are mixed across ingestion, storage, and bridge flows.
3. Metadata is limited and not yet compliance-grade.
4. There is no document manifest with stable document IDs, file hashes, lifecycle states, or approval status.
5. Retrieval evaluation is not implemented.
6. Citation validation is not implemented.
7. Response modes are not formally enforced.
8. Hybrid dense plus lexical retrieval is not implemented.
9. Audit logging for full answer reconstruction is not implemented.
10. Production depends on an external bridge endpoint that is not yet healthy.

## Planned Architecture Direction

The backend package skeleton from `PLANWORKLOG.md` now exists as `src/aviationrag/`. It is a migration anchor only. Runtime behavior has not moved from the legacy scripts, and the modules below should not be treated as active runtime owners yet.

```text
src/aviationrag/
  __init__.py
  config.py
  ingestion/
    __init__.py
    readers.py
    chunking.py
    manifest.py
  retrieval/
    __init__.py
    faiss_store.py
    hybrid.py
    filters.py
  generation/
    __init__.py
    answer_service.py
    prompts.py
    response_policy.py
    citation_validator.py
  storage/
    __init__.py
    astra.py
    chat_store.py
    document_store.py
  evaluation/
    __init__.py
    retrieval_eval.py
    answer_eval.py
  api/
    __init__.py
  utils/
    __init__.py
```

Planned direction:

1. Keep old script paths as wrappers while moving shared logic into modules.
2. Add explicit document and chunk models.
3. Add document manifest and lifecycle governance.
4. Add retrieval evaluation before retrieval changes.
5. Add citation validation and response policy enforcement before compliance-grade claims.
6. Add audit logs that capture question, mode, retrieved chunks, citations, answer, model, prompt version, latency, and timestamp.
