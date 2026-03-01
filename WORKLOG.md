# WORKLOG

Last Updated: 2026-03-01  
Active Branch: `hardening/sanitize-repo`

## Purpose

Persistent execution log for deployment hardening and product-readiness work so progress is recoverable even if chat or IDE session is interrupted.

## Active Plan

1. `Done` Step 1 repository sanitization baseline.
2. `In Progress` Step 2 deployment hardening:
   - `Done` API rate limiting and request validation.
   - `Done` Deployment routing config cleanup.
   - `Done` CI pipeline for build + smoke tests.
   - `Done` Python bridge architecture split support (`worker` + `http` mode).
   - `In Progress` External aviation command service deployment and cutover.
3. `Done` Step 3 upload workflow (UI + API + ingestion status).
4. `Done` Step 4 formula rendering in chat.
5. `In Progress` Step 5 final production-readiness checklist and release gate.

## Cross-Check Snapshot

### Step 1: Repository Sanitization

1. `Done` Runtime artifact untracking and ignore rules.
2. `Done` Pre-commit sanitization checks and hook wiring.
3. `Done` Sanitization report and repeatable local checks.
4. `Partial` Secret scanning tooling in CI (`gitleaks` / `trufflehog` still missing).
5. `Pending` Git history cleanup for historical large generated blobs.

### Step 2: Deployment Hardening

1. `Done` API rate limiting and request normalization.
2. `Done` Routing cleanup (`vercel.json` no longer rewrites all paths to index).
3. `Done` CI baseline (`sanitize`, `build`, `smoke`).
4. `Done` Auth hardening (attempt limiter + optional password hash mode).
5. `Done` Bridge split support (`worker` and `http` mode in server bridge).
6. `Pending` External aviation command service deployment and production cutover (`AVIATION_API_MODE=http`).
7. `Pending` Real identity/SSO model (current credentials provider is improved but still basic).

### Deferred Issues

1. `In Progress` Historical conversation recovery in web UI:
   - `Done` Identified stale legacy title-only sessions sorting to top without real history.
   - `Done` Applied timestamp fallback fix so stale sessions sort to bottom.
   - `Done` Added explicit UI feedback when a selected session has no stored messages.
   - `Pending` Final user validation in browser real run.

### Step 3: Upload Workflow

1. `Done` Web upload UI and ingestion status tracking.
2. `Done` Upload API + validation + queue/state pipeline.

### Step 4: Formula Rendering

1. `Done` Markdown + math rendering (`remark-math`/`rehype-katex`) in chat responses.
2. `Done` Safety sanitization for rendered Markdown output (`rehype-sanitize` pre-pass + no raw HTML).

### Step 5: Release Gate

1. `Pending` Remaining dependency vulnerability reduction (major upgrades needed for Next/LangChain lines).
2. `Pending` Staging hardening checklist (secrets rotation confirmation, monitoring/alerts, cutover runbook).

## Progress Log

### 2026-02-26

1. Created and pushed sanitization branch `hardening/sanitize-repo`.
2. Added sanitize hooks and checks:
   - `.githooks/pre-commit`
   - `tools/sanitize/precommit-check.mjs`
3. Untracked runtime chat files from repo while keeping local data intact.
4. Added `SANITIZATION_REPORT.md` with findings and follow-up actions.
5. Began Step 2 implementation:
   - Start API security hardening.
   - Start deployment routing cleanup.
   - Start CI workflow creation.
6. Implemented API hardening:
   - Added `src/utils/server/api_security.ts`.
   - Applied in `ask`, `session`, `session/[id]`, `history/[session_id]`.
   - Added per-route in-memory rate limits and normalized input handling.
7. Cleaned deployment routing:
   - Updated `vercel.json` to standard Next.js framework config.
8. Added CI workflow:
   - `.github/workflows/ci.yml` runs sanitize check, build, and smoke test.
9. Revalidated after hardening edits:
   - `npm run sanitize:check:all` passed.
   - `npm run build` passed.
   - `npm run test:smoke` passed.
10. Fixed re-login reliability issue after logout:
    - Updated `pages/auth/signin.tsx` to submit using `FormData` values.
    - Added `name` + `autocomplete` attributes for browser autofill compatibility.
    - Removed dependency on controlled email/password state for auth submission.
    - `npm run build` passed after fix.
11. Hardened authentication configuration:
    - Removed implicit fallback login email for credentials auth.
    - Added in-memory login attempt limiter for credentials provider.
    - Added optional `APP_AUTH_PASSWORD_HASH` (`sha256:<hex>`) verification path.
12. Implemented deployment-ready bridge split:
    - Added HTTP bridge mode in `src/utils/server/aviation_api_bridge.ts`.
    - New env controls: `AVIATION_API_MODE`, `AVIATION_API_HTTP_URL`, `AVIATION_API_HTTP_TOKEN`.
    - Health endpoint now reports bridge mode and HTTP bridge readiness.
13. Added auth hash utility:
    - `tools/auth/hash-password.mjs`
    - `npm run auth:hash -- \"my-password\"`
14. Updated `.env.example` and `README.md` for new auth/bridge settings.
15. Fixed commit workflow blocker for ingestion assets:
    - Updated sanitization size policy in `tools/sanitize/precommit-check.mjs`.
    - Added path-specific limits:
      - `data/documents/*` up to 80 MB
      - `data/raw/*` (`.pkl/.json/.csv`) up to 120 MB
16. Fixed conversation history rendering robustness:
    - Updated `pages/index.tsx` history normalization.
    - Supports multiple payload schemas (`user_query/ai_response`, camelCase, role/content).
17. Verification after fixes:
    - `npm run sanitize:check` passed with new document additions staged.
    - `npm run build` passed.
    - `npm run test:smoke` passed.
18. Added plan cross-check matrix to this file for daily status tracking and handoff continuity.
19. Started Priority 1 cutover track:
    - Added HTTP bridge contract spec:
      - `docs/AVIATION_API_HTTP_BRIDGE_SPEC.md`
    - Added staged cutover checklist:
      - `docs/AVIATION_API_HTTP_BRIDGE_CUTOVER_CHECKLIST.md`
    - Next action: implement/deploy external `/command` service and run checklist in staging.
20. Added reference external HTTP bridge service implementation:
    - `src/scripts/py_files/aviationai_http_bridge.py`
    - Supports `/health` and `/command` with optional bearer token auth.
    - Uses same action handlers as worker mode to preserve payload compatibility.
21. Updated runtime configuration/docs for bridge service:
    - `.env.example` adds `AVIATION_API_HTTP_BIND` and `AVIATION_API_HTTP_PORT`.
    - `README.md` documents running the optional HTTP bridge service.
22. Implemented upload workflow (Step 3):
    - Added upload API: `POST /api/documents/upload` with auth, rate limiting, type/size validation.
    - Added upload status API: `GET /api/documents/status/{id}`.
    - Added server-side upload job store with persisted status in `logs/upload_jobs.json`.
    - Added queued ingestion runner for steps:
      - `Read Documents`
      - `Chunk Documents`
      - `Generate New Embeddings`
      - `Store New Embeddings in AstraDB`
23. Implemented formula rendering (Step 4):
    - Added assistant markdown rendering component with math support:
      - `remark-math`
      - `rehype-katex`
      - `rehype-sanitize`
    - Imported KaTeX CSS in `_app.tsx`.
    - Updated chat styles for markdown blocks and formula display.
24. Added sidebar upload UX:
    - PDF/DOCX upload control.
    - Status and error feedback card.
    - Client polling loop for ingestion state progression.
25. Added configuration/docs for upload pipeline:
    - `.env.example`: `DOCUMENT_UPLOAD_MAX_MB`, `DOCUMENT_UPLOAD_AUTO_INGEST`, `DOCUMENT_UPLOAD_STEP_TIMEOUT_MS`
    - `README.md`: upload env vars + API endpoints + math rendering note.
26. Validation after Step 3/4 implementation:
    - `npm run sanitize:check` passed.
    - `npm run build` passed.
    - `npm run test:smoke` passed.
27. Fixed ingestion detection blocker for newly added large PDFs:
    - Root cause: `read_documents.py` failed on large extracted text (`spaCy` max length) and returned exit code `0`, so pipeline looked successful while silently skipping remaining files.
    - Added robust `read_documents.py` changes:
      - Chunked NLP processing for long texts (`READ_DOC_NLP_CHUNK_CHARS`, default `180000`).
      - Periodic checkpoint persistence to `data/raw/aviation_corpus.pkl` (`READ_DOC_CHECKPOINT_EVERY`, default `1`).
      - Quieter, configurable logging (`READ_DOC_LOG_LEVEL`, default `INFO`) and noisy parser logger suppression.
      - Abbreviation CSV encoding fallback (`utf-8` -> `cp1252` -> `latin-1`).
      - Proper non-zero process exit on fatal errors (`sys.exit(1)`).
28. Re-ran ingestion for newly added documents:
    - `Read Documents` completed and all 6 new PDFs were added to corpus.
    - `Chunk Documents` processed all 6 new PDFs.
    - `Generate New Embeddings` generated `1290` new chunk embeddings.
    - `Store New Embeddings in AstraDB` inserted `1290` embeddings successfully.
29. Staged data refresh artifacts for repository update:
    - Added 6 new source PDFs under `data/documents/`.
    - Updated `data/raw/aviation_corpus.pkl`.
    - Updated visualization outputs in `assets/pictures/`.
30. Adjusted sanitize guardrail for larger valid source PDFs:
    - `tools/sanitize/precommit-check.mjs`
    - `data/documents/*` file-size limit raised from `80 MB` to `120 MB`.
31. Investigated embedding-count inconsistency after llama-parse rollout:
    - Observed report: local `5433` vs Astra `5000`.
    - Verified true DB counts with paginated and count queries:
      - local embeddings: `5433`
      - Astra table rows: `5433`
    - Root cause: `src/scripts/js_files/check_astradb_consistency.js` was reading only one Cassandra page (default first page, commonly `5000` rows).
32. Fixed consistency checker and llama-parse guardrails:
    - `src/scripts/js_files/check_astradb_consistency.js`
      - Added explicit paging loop.
      - Added O(1) chunk-id lookup map for comparisons.
      - Fixed DB embedding buffer conversion for robust numeric comparison.
    - `src/scripts/py_files/read_documents.py`
      - Made llama-parse integration optional and lazy-initialized.
      - Added env toggle: `READ_DOC_ENABLE_LLAMA_PARSE` (default `true`).
      - Added safe fallback when `llama_parse` package or `LLAMA_CLOUD_API_KEY` is missing.
33. Addressed chat history recovery inconsistency:
    - `src/scripts/py_files/chat_db.py`
      - Legacy sessions without index metadata no longer get `updated_at=now`.
      - Added deterministic fallback timestamp (`1970-01-01T00:00:00Z`) so stale sessions sort to bottom.
    - `pages/index.tsx`
      - Added explicit message when selected session has no stored history.
    - Verification:
      - Session listing now prioritizes indexed/recent sessions correctly.
      - `npm run build` passed.

## Session Recovery Procedure

If the chat/session freezes:

1. Re-open this repository and read `WORKLOG.md` + `SANITIZATION_REPORT.md`.
2. Run `git status -b --short`.
3. Continue from the latest `In Progress` item in this file.
4. Commit and push after each completed sub-step.
