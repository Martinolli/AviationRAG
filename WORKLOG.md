# WORKLOG

Last Updated: 2026-07-18
Active Branch: `main`
Source Branch for Historical Entries: `hardening/sanitize-repo`

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
6. `Planned` Step 6 RAG retrieval quality upgrade (post-bridge stabilization):
   - `Planned` 6.1 Structure-aware document parsing for PDF-heavy sources.
   - `Planned` 6.2 Heading/section-aware chunking and table/note chunk separation.
   - `Planned` 6.3 Metadata schema enrichment for compliance filtering.
   - `Planned` 6.4 Hybrid retrieval (dense + lexical + metadata filters).
   - `Planned` 6.5 Mode-specific response templates and evaluation harness.

## Cross-Check Snapshot

### Step 1: Repository Sanitization

1. `Done` Runtime artifact untracking and ignore rules.
2. `Done` Pre-commit sanitization checks and hook wiring.
3. `Done` Sanitization report and repeatable local checks.
4. `Partial` Secret scanning tooling in CI (`gitleaks` added, `trufflehog` still pending if required).
5. `Pending` Git history cleanup for historical large generated blobs.

### Step 2: Deployment Hardening

1. `Done` API rate limiting and request normalization.
2. `Done` Routing cleanup (`vercel.json` no longer rewrites all paths to index).
3. `Done` CI baseline (`sanitize`, `build`, `smoke`).
4. `Done` Auth hardening (attempt limiter + optional password hash mode).
5. `Done` Bridge split support (`worker` and `http` mode in server bridge).
6. `Pending` External aviation command service deployment and production cutover (`AVIATION_API_MODE=http`).
7. `Pending` Real identity/SSO model (current credentials provider is improved but still basic).
8. `Done` Added deep bridge diagnostics in app health endpoint for cutover validation.

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

### Step 6: Retrieval Strategy Upgrade (Planned)

1. `Planned` Parsing upgrade:
   - Move from extractor-only flow to structure-aware parsing for headings/sections/tables.
   - Keep multi-parser fallback model for difficult documents.
2. `Planned` Chunking redesign:
   - Section/subsection aligned chunks.
   - Dedicated chunks for tables, cautions/notes, and exact-citation passages.
3. `Planned` Metadata enrichment:
   - Add authority, doc class, revision/effective date, status, section path, and page spans.
4. `Planned` Retrieval modernization:
   - Keep current semantic layer, add lexical/BM25 branch and metadata filtering.
   - Add deterministic reranking rules for regulatory identifier exact matches.
5. `Planned` Response policy split:
   - Separate compliance, regulatory citation, safety analysis, and internal procedure modes.
6. `Planned` Evaluation harness:
   - Build benchmark set for exact paragraph retrieval, revision correctness, and insufficient-evidence behavior.

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
34. Improved HTTP cutover observability in web API:
    - `pages/api/health.ts`
      - Added deep mode query (`GET /api/health?deep=1`).
      - Added real HTTP bridge `ping` check through `/command`.
      - Added `checks.aviation_http_ping`, `deep_check_requested`, and `deep_check_error`.
35. Added Vercel online deployment runbook and checklist alignment:
    - Added `docs/VERCEL_ONLINE_SETUP.md` with production/preview env var setup and verification.
    - Updated `docs/AVIATION_API_HTTP_BRIDGE_CUTOVER_CHECKLIST.md` with deep health validation.
    - Updated `README.md` to reference Vercel setup doc and deep health usage.

### 2026-03-14

1. Added deployment preflight env validation script for HTTP bridge modes:
    - Added `tools/deploy/check-env.mjs`.
    - New profiles:
      - `local-http` (local app + local bridge)
      - `vercel-http` (Vercel-ready checks, HTTPS/public bridge required)
    - Validates required env vars, mode consistency, URL constraints, and warns on weak auth patterns.
  
2. Added npm scripts for deploy preflight:
    - `npm run deploy:check:local-http`
    - `npm run deploy:check:vercel-http`
3. Updated deployment docs:
    - `README.md` adds "Deployment Env Checks" section.
    - `docs/VERCEL_ONLINE_SETUP.md` includes preflight command before Vercel deploy.
4. Added CI secret scanning gate:
    - Updated `.github/workflows/ci.yml` with `secret-scan` job.
    - Added `gitleaks` working-tree scan (`detect --source . --no-git --redact`).
    - `build-and-smoke` now depends on successful secret scan.
5. Recovered branch alignment after accidental context switch:
    - Confirmed active workline remains `hardening/sanitize-repo` (`c8138858` baseline).
    - Saved temporary uncommitted work from `copilot/deploy-project-on-vercel` into stash for safety.
6. Revalidated authentication hardening path:
    - Confirmed `pages/auth/signin.tsx` uses robust `FormData` submission with explicit `name` + `autocomplete`.
    - Confirmed `src/utils/server/auth_options.ts` uses `APP_AUTH_PASSWORD_HASH` with timing-safe compare.
7. Resolved runtime `fetch failed` in chat area:
    - Root cause: app configured with `AVIATION_API_MODE=http` while local bridge was not running.
    - Verified via `GET /api/health?deep=1` (`deep_check_error: "fetch failed"`).
    - Restored bridge availability and confirmed chat path recovery.
8. Repository hygiene pass:
    - Added `.next_backup_*/` to `.gitignore` to avoid accidental backup artifact noise.
    - Working tree returned to clean state for continued Vercel setup work.
9. Brought `main` branch to Vercel-compatible baseline:
    - Fixed invalid `vercel.json` schema on `main` (`routes[0].pages` removed).
    - `vercel.json` now uses standard Next.js config (`{ "framework": "nextjs" }`).
10. Backported auth hardening to `main` for production login reliability:
    - `src/utils/server/auth_options.ts`
      - Added `APP_AUTH_PASSWORD_HASH` (`sha256:<hex>`) support with timing-safe compare.
      - Kept optional `APP_AUTH_PASSWORD` fallback.
      - Added credentials attempt limiter.
    - `pages/auth/signin.tsx`
      - Restored `FormData` submission path.
      - Added explicit `name` + `autocomplete` attributes.
11. Backported HTTP bridge runtime support to `main`:
    - `src/utils/server/aviation_api_bridge.ts`
      - Added dual bridge mode support (`worker` + `http`).
      - Added HTTP `/command` call path using:
        - `AVIATION_API_MODE=http`
        - `AVIATION_API_HTTP_URL`
        - `AVIATION_API_HTTP_TOKEN`
    - `pages/api/health.ts`
      - Added bridge mode reporting.
      - Added deep check (`GET /api/health?deep=1`) with bridge ping diagnostics.
12. Vercel deployment status checkpoint:
    - App deploy now reaches `bridge_mode: "http"` on hosted health check.
    - Current blocker remains external bridge reachability from Vercel (`deep_check_error: "fetch failed"`).
    - Required next action: set `AVIATION_API_HTTP_URL` to a publicly reachable HTTPS bridge endpoint.
13. Restored missing tracking file on `main`:
    - Re-added `WORKLOG.md` to repository after it disappeared from branch history.
14. Repository hygiene updates on `main`:
    - Added `.next_backup_*/` and `.githooks/` to `.gitignore` to avoid local artifact noise.
15. End-of-day sync state:
    - `main` is synchronized with origin and contains the latest Vercel/auth/bridge fixes.
    - Ready to continue tomorrow from bridge public reachability + final Vercel env validation.
16. Restored missing bridge/docs assets on `main`:
    - Restored from `hardening/sanitize-repo`:
      - `src/scripts/py_files/aviationai_http_bridge.py`
      - `docs/AVIATION_API_HTTP_BRIDGE_CUTOVER_CHECKLIST.md`
      - `docs/AVIATION_API_HTTP_BRIDGE_SPEC.md`
      - `docs/VERCEL_ONLINE_SETUP.md`
    - Root cause: prior updates were selectively backported to `main` instead of full branch merge, so branch-only files were absent on `main`.

### 2026-03-15

1. Verified local bridge/app path is healthy:
    - Local bridge `/health` returns success.
    - Local app `GET /api/health?deep=1` returns `bridge_mode: "http"` and `aviation_http_ping: true`.
2. Confirmed Vercel blocker root cause:
    - Vercel app still shows `deep_check_error: "fetch failed"`.
    - `AVIATION_API_HTTP_URL` value `https://aviation-api-http-bridge.com` does not resolve publicly (DNS failure).
3. Next action checkpoint for tomorrow:
    - Provision a real public HTTPS bridge endpoint (or secure tunnel) for `/health` and `/command`.
    - Update Vercel `AVIATION_API_HTTP_URL` and redeploy.
    - Re-validate `https://aviation-rag.vercel.app/api/health?deep=1` expecting `aviation_http_ping: true`.

### 2026-04-03

1. Reviewed `Brain_Storming_ChatGPT_AviationRAG.txt` and validated feasibility against current codebase.
2. Accepted strategic direction as post-cutover plan:
   - prioritize parsing/chunking/metadata/retrieval upgrades over storage replacement.
3. Added Step 6 planned roadmap to this file with phased execution items.

### 2026-04-03

1. Resumed from Vercel bridge reachability blocker checkpoint.
2. Added repeatable bridge preflight verification utility:
   - `tools/bridge/check-http-bridge.mjs`
   - validates:
     - `GET /health`
     - `POST /command` with `{"action":"ping"}`
   - consumes `--url` / `--token` args (or `AVIATION_API_HTTP_URL` / `AVIATION_API_HTTP_TOKEN` env fallback).
3. Added npm script:
   - `npm run bridge:check`
4. Updated `docs/VERCEL_ONLINE_SETUP.md` with preflight checker usage before Vercel redeploy.
5. Added quick-reference sync guide for local vs Vercel bridge configuration:
   - `docs/RUNBOOK_VERCEL_LOCAL_SYNC.md`
   - includes:
     - env split mapping
     - local startup sequence
     - bridge preflight command
     - Vercel validation sequence
     - error-to-fix table for common failures

### 2026-05-12

1. Started repository sanitation pass on `main`:
   - Local branch was behind `origin/main` by 7 commits.
   - Runtime file `chat_id/session_metadata.json` was modified locally.
2. Removed runtime/generated/private folders from Git tracking using `git rm --cached` while preserving local files on disk:
   - `chat_id/`
   - `chat/`
   - `logs/`
   - `data/documents/`
   - `data/raw/`
   - `data/processed/`
   - `data/embeddings/`
   - `data/astra_db/`
   - `assets/pictures/`
3. Updated `.gitignore` to exclude runtime state, local/private source documents, generated corpora, visualizations, env/secrets, and Python/Node artifacts.
4. Added `data/sample_documents/.gitkeep` as the only optional sample data placeholder.
5. Verified/restored sanitization infrastructure from `origin/hardening/sanitize-repo`:
   - `SANITIZATION_REPORT.md`
   - `.githooks/pre-commit`
   - `tools/sanitize/precommit-check.mjs`
   - `.github/workflows/ci.yml`
6. Added npm sanitation scripts:
   - `hooks:install`
   - `sanitize:check`
   - `sanitize:check:all`
7. Installed local hooks with `npm run hooks:install`.
8. Validation results:
   - `npm run sanitize:check:all` passed.
   - `npm run build` passed.
   - `npm run test:smoke` passed.
   - No validation was skipped.
9. Next action:
   - Pull/rebase latest `origin/main` after committing sanitation changes.

### 2026-05-12

1. Continued `PLANWORKLOG.md` Phase A.1 only: HTTP bridge public reachability and Vercel cutover validation.
2. Local bridge checks:
   - Initial `GET http://127.0.0.1:8010/health` failed: connection refused.
   - Root cause found in local `.env`: `AVIATION_API_HTTP_BIND` is malformed.
   - Retest with process-only override `AVIATION_API_HTTP_BIND=127.0.0.1` passed:
     - `GET http://127.0.0.1:8010/health`: `success=true`.
     - `POST http://127.0.0.1:8010/command` with `{"action":"ping"}`: `success=true`, `action="ping"`.
3. Local app deep health check:
   - Initial `GET http://127.0.0.1:3000/api/health?deep=1` failed because the app was not running.
   - After starting the local bridge with the process-only bind override and starting `npm run dev`, `GET http://127.0.0.1:3000/api/health?deep=1` passed:
     - `success=true`
     - `bridge_mode="http"`
     - `checks.aviation_http_url_set=true`
     - `checks.aviation_http_ping=true`
     - `deep_check_error=null`
4. Public bridge checks:
   - Local `.env` currently points `AVIATION_API_HTTP_URL` to `http://127.0.0.1:8010`; no public bridge URL is configured locally.
   - Previously documented public candidate `https://aviation-api-http-bridge.com` still fails:
     - `GET /health`: DNS failure, `No such host is known`.
     - `POST /command` ping: DNS failure, `No such host is known`.
5. Vercel production deep health check:
   - `GET https://aviation-rag.vercel.app/api/health?deep=1` returned HTTP `200` with:
     - `success=true`
     - `bridge_mode="http"`
     - `checks.aviation_http_url_set=true`
     - `checks.aviation_http_ping=false`
     - `deep_check_requested=true`
     - `deep_check_error="Bridge returned non-JSON response (503)."`
6. Blocker status:
   - Phase A.1 remains `Blocked`.
   - Local bridge/app path is valid when the local bind env var is corrected.
   - Production is configured for HTTP bridge mode, but Vercel is not reaching a healthy aviation bridge `/command` endpoint.
   - Current production failure changed from earlier DNS/fetch failure to a reachable non-JSON `503` response, which indicates the configured Vercel bridge URL is still not a valid/healthy bridge service endpoint.
7. Next action:
   - Fix local `.env` `AVIATION_API_HTTP_BIND` formatting.
   - Provision or identify the real public HTTPS bridge base URL.
   - Validate public `GET /health` and authenticated `POST /command` ping with `npm run bridge:check -- --url <public-bridge-url> --token <token>`.
   - Update Vercel `AVIATION_API_HTTP_URL` to that base URL and redeploy.
   - Re-run `https://aviation-rag.vercel.app/api/health?deep=1` expecting `checks.aviation_http_ping=true`.

### 2026-05-12

1. Continued `PLANWORKLOG.md` Phase A.2 only: production-readiness release gate checklist.
2. Added release gate checklist:
   - `docs/PRODUCTION_RELEASE_GATE.md`
3. Recorded current gate status:
   - Release decision: `Blocked`.
   - Phase A.1 public bridge/Vercel cutover remains the primary release blocker.
   - Upload flow and history/session retrieval still require manual authenticated verification.
   - Secret rotation still requires operator confirmation.
   - Gitleaks CI still requires GitHub Actions confirmation because local `gitleaks` is not installed.
4. Verification run for checklist evidence:
   - `npm run sanitize:check:all` passed.
   - `npm run build` passed.
   - `npm run test:smoke` passed.
   - `git ls-files .env .env.* secure-connect-*.zip` returned only `.env.example`.
5. Known non-blocking smoke-test warning:
   - Next.js reports future `allowedDevOrigins` configuration may be needed for cross-origin dev requests from `127.0.0.1`.
6. Next action:
   - Return to Phase A.1 blocker resolution: provide a real public bridge endpoint, update Vercel env, redeploy, and re-run production deep health.
7. Finalized checklist structure against Phase A.2 requirements:
   - Secrets and environment.
   - Build/test validation.
   - Bridge validation.
   - App validation.
   - Deployment validation.
   - Known limitations.
   - Go / No-Go status.
8. Revalidated after final checklist update:
   - `npm run sanitize:check:all` passed.
   - `npm run build` passed.

### 2026-05-12

1. Continued `PLANWORKLOG.md` Phase B.3 only: documentation baseline.
2. Created foundational baseline docs:
   - `docs/ARCHITECTURE.md`
   - `docs/DATA_GOVERNANCE.md`
   - `docs/EVALUATION_PLAN.md`
   - `docs/RESPONSE_POLICY.md`
3. Updated `PLANWORKLOG.md` B.3 status to show the requested baseline docs were added.
4. No runtime behavior changes were made:
   - No backend refactor.
   - No retrieval logic changes.
   - No prompt changes.
   - No chunking, embedding, metadata, or deployment behavior changes.
5. Remaining blockers:
   - Production bridge remains externally blocked until a real public HTTPS bridge endpoint is provisioned, configured in Vercel, and redeployed.
   - Retrieval evaluation is documented but not implemented.
   - Response policy is documented but not enforced by a validator or structured answer schema.
6. Validation after documentation edits:
   - `npm run sanitize:check:all` passed.
   - `npm run build` passed.

### 2026-05-12

1. Continued `PLANWORKLOG.md` Phase C.1 only: backend package skeleton.
2. Created future-oriented package structure under `src/aviationrag/`:
   - package root with `__init__.py`, `README.md`, and minimal `config.py`
   - `ingestion/`
   - `retrieval/`
   - `generation/`
   - `storage/`
   - `evaluation/`
   - `api/`
   - `utils/`
3. Added README ownership notes for each package area.
4. No runtime logic was moved.
5. No behavior changes were introduced:
   - `src/scripts/py_files/aviationrag_manager.py` remains the active ingestion orchestrator.
   - `src/scripts/py_files/aviationai.py` remains the active answer/retrieval runtime.
   - `src/scripts/py_files/config.py` remains the active runtime config source.
   - Worker and HTTP bridge integration remain unchanged.
6. Updated `PLANWORKLOG.md` C.1 status to show the skeleton exists while migration tasks remain pending.
7. Updated `docs/ARCHITECTURE.md` to clarify that `src/aviationrag/` is now a skeleton only, not active runtime ownership.
8. Remaining blockers:
   - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
   - Retrieval evaluation is not implemented.
   - Response policy is not enforced.
   - Package migration has not started yet.
9. Validation after skeleton creation:
   - `npm run sanitize:check:all` passed.
   - `npm run build` passed.
   - `python -m compileall src` passed.

### 2026-05-13

1. Continued `PLANWORKLOG.md` Phase C.2 only: lightweight core data models.
2. Created `src/aviationrag/models.py` with lightweight standard-library dataclasses:
   - `DocumentRecord`
   - `ChunkRecord`
   - `RetrievedChunk`
   - `AnswerResult`
3. Added minimal `to_dict()` and `from_dict()` helpers for future migration anchors.
4. Added lightweight model tests in `tests/test_models.py`.
5. Updated architecture/package documentation to clarify:
   - Lightweight core models now exist.
   - Runtime migration has not started.
   - Existing scripts still use legacy dictionaries, JSON, pickle, FAISS, Astra, and bridge structures.
6. No runtime migration was performed:
   - No retrieval logic changes.
   - No prompt changes.
   - No chunking changes.
   - No embedding changes.
   - No bridge/API behavior changes.
   - No deployment behavior changes.
7. Validation after model creation:
   - `python tests/test_models.py` passed.
   - `npm run sanitize:check:all` passed.
   - `python -m compileall src` passed.
   - `npm run build` passed.
8. Remaining blockers:
   - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
   - Retrieval evaluation is not implemented.
   - Response policy is not enforced.
   - Metadata migration has not started.

### 2026-05-13

1. Continued `PLANWORKLOG.md` Phase D.1a only: document manifest and metadata schema planning.
2. Created `docs/DOCUMENT_MANIFEST_SCHEMA.md` covering:
   - document identity model
   - manifest storage options
   - proposed manifest and chunk metadata schemas
   - lifecycle and approval states
   - source authority and document type classifications
   - extraction quality fields
   - versioning, traceability, Git/data governance alignment, future phases, and open questions
3. Updated `PLANWORKLOG.md` to mark D.1 schema planning complete while keeping manifest implementation tasks planned/future.
4. No ingestion runtime behavior changed.
5. No documents were reprocessed.
6. No embeddings were regenerated.
7. Existing scripts still use the legacy data flow:
   - `src/scripts/py_files/read_documents.py`
   - `src/scripts/py_files/aviation_chunk_saver.py`
   - embedding/Astra helper scripts
8. Validation after schema planning:
   - `npm run sanitize:check:all` passed.
   - `python -m compileall src` passed.
   - `npm run build` passed.
   - `python tests/test_models.py` passed.
9. Remaining blockers:
   - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
   - Manifest is not implemented yet.
   - Retrieval evaluation is not implemented.
   - Response policy is not enforced.

### 2026-05-13

1. Continued `PLANWORKLOG.md` Phase D.1b only: fake sample manifest fixture.
2. Created `data/sample_documents/sample_manifest.jsonl` with five fake records only:
   - FAA-style advisory circular sample.
   - EASA-style certification specification sample.
   - Fake internal procedure sample for `AeroWorks Example Co.`
   - Fictional accident report sample.
   - Manufacturing quality sample.
3. Created `data/sample_documents/sample_chunks.jsonl` with four short synthetic chunk records tied to fake sample manifest records.
4. Added `tests/test_sample_manifest_fixture.py` to validate JSONL syntax, required fields, unique IDs, sample-only URIs, hash format, and chunk references.
5. Updated `docs/DOCUMENT_MANIFEST_SCHEMA.md` to mention the fake fixture files and clarify they are safe to commit and not used by runtime ingestion.
6. Updated `PLANWORKLOG.md` to mark the fake sample manifest fixture complete while keeping manifest writer/integration tasks planned.
7. No ingestion runtime behavior changed.
8. No real documents were added.
9. No embeddings were regenerated.
10. Validation after fixture creation:
    - `npm run sanitize:check:all` passed.
    - `python -m compileall src` passed.
    - `python tests/test_models.py` passed.
    - `python tests/test_sample_manifest_fixture.py` passed.
    - `npm run build` passed.
11. Remaining blockers:
    - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
    - Manifest writer is not implemented.
    - Manifest is not integrated with ingestion.
    - Retrieval evaluation is not implemented.
    - Response policy is not enforced.

### 2026-05-13

1. Continued `PLANWORKLOG.md` Phase D.1c only: manifest utility module using fake/test data.
2. Created `src/aviationrag/ingestion/manifest.py` with safe JSONL utilities:
   - `read_manifest`
   - `write_manifest`
   - `append_manifest_record`
   - `validate_manifest_record`
   - `document_record_from_dict`
   - `document_record_to_dict`
3. Added `tests/test_manifest_writer.py` using only `data/sample_documents/sample_manifest.jsonl` and temporary directories.
4. Updated `.gitignore` so `data/manifest/` remains ignored as the future local/private manifest path while fake sample fixtures remain allowed.
5. Updated `docs/DOCUMENT_MANIFEST_SCHEMA.md` to clarify:
   - `data/manifest/documents.jsonl` is future local/private storage and must not be committed with real metadata.
   - committed tests use fake sample data from `data/sample_documents/sample_manifest.jsonl`.
6. No ingestion runtime behavior changed.
7. No documents were reprocessed.
8. No embeddings were regenerated.
9. Existing scripts still use the legacy data flow.
10. Validation after manifest utility creation:
    - `npm run sanitize:check:all` passed.
    - `python -m compileall src` passed.
    - `python tests/test_models.py` passed.
    - `python tests/test_sample_manifest_fixture.py` passed.
    - `python tests/test_manifest_writer.py` passed.
    - `npm run build` passed.
11. Remaining blockers:
    - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
    - Manifest is not integrated with ingestion.
    - Retrieval evaluation is not implemented.
    - Response policy is not enforced.

### 2026-05-13

1. Continued `PLANWORKLOG.md` Phase D.1d only: legacy compatibility adapter design using fake/test data.
2. Created `src/aviationrag/ingestion/legacy_adapter.py` with pure conversion helpers for legacy-like dictionaries:
   - filename normalization
   - authority inference
   - document type inference
   - deterministic document ID generation
   - document and chunk record conversion
   - batch document and chunk conversion
3. Added `tests/test_legacy_adapter.py` using inline fake legacy-like records only.
4. Updated `docs/DOCUMENT_MANIFEST_SCHEMA.md` to document the legacy compatibility adapter and clarify it is not integrated with runtime ingestion.
5. No legacy ingestion scripts were modified:
   - `src/scripts/py_files/read_documents.py`
   - `src/scripts/py_files/aviation_chunk_saver.py`
   - `src/scripts/py_files/extract_pkl_to_json.py`
6. No documents were reprocessed.
7. No embeddings were regenerated.
8. Manifest is not integrated with ingestion yet.
9. Validation after compatibility adapter creation:
   - `npm run sanitize:check:all` passed.
   - `python -m compileall src` passed.
   - `python tests/test_models.py` passed.
   - `python tests/test_sample_manifest_fixture.py` passed.
   - `python tests/test_manifest_writer.py` passed.
   - `python tests/test_legacy_adapter.py` passed.
   - `npm run build` passed.
10. Remaining blockers:
    - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
    - Manifest is not integrated with ingestion.
    - Retrieval evaluation is not implemented.
    - Response policy is not enforced.

### 2026-05-13

1. Continued `PLANWORKLOG.md` Phase D.1e only: controlled ingestion manifest integration design.
2. Created `docs/INGESTION_MANIFEST_INTEGRATION_PLAN.md` covering:
   - current legacy ingestion flow
   - proposed manifest-aware ingestion flow
   - planned integration hook points
   - `DocumentRecord` and `ChunkRecord` mapping plans
   - manifest lifecycle write points
   - local/private data handling
   - future Astra/FAISS reset and rebuild strategy
   - migration phases, validation plan, rollback plan, and open questions
3. Did not create an additional dry-run helper because `src/aviationrag/ingestion/legacy_adapter.py` already contains the pure fake-data conversion helpers this phase would call.
4. Updated `docs/DOCUMENT_MANIFEST_SCHEMA.md` and `docs/ARCHITECTURE.md` to link the integration plan while clarifying that runtime integration is not active.
5. No ingestion runtime scripts were modified:
   - `src/scripts/py_files/read_documents.py`
   - `src/scripts/py_files/aviation_chunk_saver.py`
   - `src/scripts/py_files/extract_pkl_to_json.py`
6. No documents were reprocessed.
7. No embeddings were regenerated.
8. No Astra or FAISS reset was performed.
9. Validation after integration planning:
   - `npm run sanitize:check:all` passed.
   - `python -m compileall src` passed.
   - `python tests/test_models.py` passed.
   - `python tests/test_sample_manifest_fixture.py` passed.
   - `python tests/test_manifest_writer.py` passed.
   - `python tests/test_legacy_adapter.py` passed.
   - `npm run build` passed.
10. Remaining blockers:
    - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
    - Manifest is not integrated with ingestion.
    - Retrieval evaluation is not implemented.
    - Response policy is not enforced.
    - Astra/FAISS reset is future only.

### 2026-05-13

1. Continued `PLANWORKLOG.md` Phase D.1f only: fake-data dry-run integration coverage.
2. Created `src/aviationrag/ingestion/dry_run.py` with side-effect-free helpers:
   - `DryRunIngestionPlan`
   - `build_dry_run_ingestion_plan`
   - `summarize_dry_run_plan`
   - `validate_dry_run_plan`
3. Added `tests/test_ingestion_dry_run.py` using inline fake legacy-like records only.
4. Updated `docs/INGESTION_MANIFEST_INTEGRATION_PLAN.md` to document the fake-data dry-run helper and its limits.
5. No legacy ingestion scripts were modified:
   - `src/scripts/py_files/read_documents.py`
   - `src/scripts/py_files/aviation_chunk_saver.py`
   - `src/scripts/py_files/extract_pkl_to_json.py`
6. No real data paths were scanned.
7. No documents were reprocessed.
8. No embeddings were regenerated.
9. No Astra or FAISS reset was performed.
10. Manifest is still not integrated into runtime ingestion.
11. Validation after dry-run coverage:
    - `npm run sanitize:check:all` passed.
    - `python -m compileall src` passed.
    - `python tests/test_models.py` passed.
    - `python tests/test_sample_manifest_fixture.py` passed.
    - `python tests/test_manifest_writer.py` passed.
    - `python tests/test_legacy_adapter.py` passed.
    - `python tests/test_ingestion_dry_run.py` passed.
    - `npm run build` passed.
12. Remaining blockers:
    - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
    - Manifest is not integrated with ingestion.
    - Retrieval evaluation is not implemented.
    - Response policy is not enforced.
    - Real reset/rebuild remains future only.

### 2026-05-14

1. Continued `PLANWORKLOG.md` Phase D.1g only: local-only manifest write dry run.
2. Created `tools/manifest/write-local-sample-manifest.py`.
3. The script writes fake sample records only:
   - input: `data/sample_documents/sample_manifest.jsonl`
   - output: `data/manifest/documents.jsonl`
4. The output path is ignored, private, and local-only.
5. Added `tests/test_local_manifest_dry_run_script.py` to execute the script, verify ignored output, validate JSONL content, and clean up the generated local file.
6. Updated manifest documentation and integration planning docs to describe the local-only dry run and clarify it is not runtime ingestion integration.
7. No real documents were scanned.
8. No ingestion runtime scripts were modified:
   - `src/scripts/py_files/read_documents.py`
   - `src/scripts/py_files/aviation_chunk_saver.py`
   - `src/scripts/py_files/extract_pkl_to_json.py`
9. No documents were reprocessed.
10. No embeddings were regenerated.
11. No Astra or FAISS reset was performed.
12. Manifest is still not integrated into runtime ingestion.
13. Validation after local-only write dry run:
    - `npm run sanitize:check:all` passed.
    - `python -m compileall src` passed.
    - `python tests/test_models.py` passed.
    - `python tests/test_sample_manifest_fixture.py` passed.
    - `python tests/test_manifest_writer.py` passed.
    - `python tests/test_legacy_adapter.py` passed.
    - `python tests/test_ingestion_dry_run.py` passed.
    - `python tests/test_local_manifest_dry_run_script.py` passed.
    - `python tools/manifest/write-local-sample-manifest.py` passed.
    - `git check-ignore data/manifest/documents.jsonl` passed.
    - `npm run build` passed.
14. Remaining blockers:
    - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
    - Manifest is not integrated with ingestion.
    - Retrieval evaluation is not implemented.
    - Response policy is not enforced.
    - Real reset/rebuild remains future only.

### 2026-05-14

1. Continued `PLANWORKLOG.md` Phase D.1h only: gated manifest integration design.
2. Added disabled-by-default manifest integration config helpers in `src/aviationrag/config.py`.
3. Documented future environment variables in `.env.example`:
   - `AVIATIONRAG_ENABLE_MANIFEST_INTEGRATION`
   - `AVIATIONRAG_MANIFEST_DRY_RUN`
   - `AVIATIONRAG_MANIFEST_PATH`
4. Added `tests/test_manifest_config.py` for bool parsing, injected env mappings, default disabled state, dry-run state, and manifest path override behavior.
5. Updated `docs/INGESTION_MANIFEST_INTEGRATION_PLAN.md` to document gated manifest integration controls.
6. Manifest integration remains disabled by default.
7. No legacy ingestion scripts were modified:
   - `src/scripts/py_files/read_documents.py`
   - `src/scripts/py_files/aviation_chunk_saver.py`
   - `src/scripts/py_files/extract_pkl_to_json.py`
8. No real data paths were scanned.
9. No documents were reprocessed.
10. No embeddings were regenerated.
11. No Astra or FAISS reset was performed.
12. Manifest is still not integrated into runtime ingestion.
13. Validation after gated manifest integration settings:
    - `npm run sanitize:check:all` passed.
    - `python -m compileall src` passed.
    - `python tests/test_models.py` passed.
    - `python tests/test_sample_manifest_fixture.py` passed.
    - `python tests/test_manifest_writer.py` passed.
    - `python tests/test_legacy_adapter.py` passed.
    - `python tests/test_ingestion_dry_run.py` passed.
    - `python tests/test_local_manifest_dry_run_script.py` passed.
    - `python tests/test_manifest_config.py` passed.
    - `npm run build` passed.
14. Remaining blockers:
    - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
    - Manifest is not integrated with ingestion.
    - Retrieval evaluation is not implemented.
    - Response policy is not enforced.
    - Real reset/rebuild remains future only.

### 2026-05-14

1. Started and completed Security Sprint S.1 only: dependency audit baseline and safe non-breaking fixes.
2. Created branch `security/dependency-hardening`.
3. Baseline `npm audit` counts:
   - Total: 39
   - Critical: 2
   - High: 21
   - Moderate: 12
   - Low: 4
4. Saved baseline report:
   - `docs/security/npm-audit-baseline.json`
5. Ran safe fix only:
   - `npm audit fix`
   - Did not run `npm audit fix --force`.
6. Safe fix result:
   - `package-lock.json` updated.
   - `package.json` unchanged.
   - Critical findings reduced from 2 to 0.
   - Total findings reduced from 39 to 27.
7. Saved post-fix report:
   - `docs/security/npm-audit-after-safe-fix.json`
8. Created `docs/security/DEPENDENCY_HARDENING_PLAN.md` documenting remaining vulnerabilities, major-upgrade packages, risk classification, upgrade sequence, validation plan, rollback plan, and open questions.
9. Remaining post-fix `npm audit` counts:
   - Total: 27
   - Critical: 0
   - High: 18
   - Moderate: 7
   - Low: 2
10. Remaining vulnerability clusters still require planned follow-up:
    - Next.js / PostCSS major upgrade path.
    - LangChain / LangSmith major upgrade path.
    - Vercel/dev tooling transitive dependency review.
11. No ingestion, retrieval, prompt, embedding, FAISS, Astra, manifest runtime, bridge, or deployment behavior was changed.
12. No ingestion was run.
13. No embeddings were regenerated.
14. No data files were committed.
15. Validation after safe dependency fix:
    - `npm run sanitize:check:all` passed.
    - `npm run build` passed.
    - `npm run test:smoke` passed.
    - `python -m compileall src` passed.
    - `python tests/test_models.py` passed.
    - `python tests/test_sample_manifest_fixture.py` passed.
    - `python tests/test_manifest_writer.py` passed.
    - `python tests/test_legacy_adapter.py` passed.
    - `python tests/test_ingestion_dry_run.py` passed.
    - `python tests/test_local_manifest_dry_run_script.py` passed.
    - `python tests/test_manifest_config.py` passed.
16. Non-blocking validation warnings:
    - `npm run build` and `npm run test:smoke` reported an outdated Browserslist/caniuse-lite warning.
    - Smoke test dev server reported a future Next.js `allowedDevOrigins` warning.
17. Recommendation for next security sprint:
    - Handle Next.js major upgrade first in a separate branch.
    - Then handle LangChain package-family upgrades separately.
    - Keep major dependency upgrades isolated from RAG/ingestion work.

### 2026-05-14

1. Continued `PLANWORKLOG.md` Phase D.1i only: reset/rebuild and retrieval evaluation baseline planning.
2. Created `docs/RESET_REBUILD_AND_EVALUATION_BASELINE.md` covering:
   - reset/rebuild triggers
   - pre-reset checklist
   - backup/export checklist
   - local artifact cleanup plan
   - Astra DB reset plan
   - FAISS/index reset plan
   - re-ingestion/rebuild sequence
   - retrieval evaluation baseline requirements
   - acceptance criteria
   - rollback/recovery plan
   - operational risks
   - go/no-go checklist
   - future implementation phases
3. Updated cross-references in:
   - `docs/INGESTION_MANIFEST_INTEGRATION_PLAN.md`
   - `docs/EVALUATION_PLAN.md`
   - `docs/ARCHITECTURE.md`
4. Updated `PLANWORKLOG.md` to mark D.1i reset/rebuild and retrieval baseline planning complete while keeping actual reset/rebuild and retrieval harness implementation planned.
5. No reset was performed.
6. No Astra or FAISS data was changed.
7. No ingestion runtime scripts were modified.
8. No real data paths were scanned.
9. No documents were reprocessed.
10. No embeddings were regenerated.
11. No `data/manifest/documents.jsonl` was written.
12. Validation after reset/rebuild planning:
    - `npm run sanitize:check:all` passed.
    - `python -m compileall src` passed.
    - `python tests/test_models.py` passed.
    - `python tests/test_sample_manifest_fixture.py` passed.
    - `python tests/test_manifest_writer.py` passed.
    - `python tests/test_legacy_adapter.py` passed.
    - `python tests/test_ingestion_dry_run.py` passed.
    - `python tests/test_local_manifest_dry_run_script.py` passed.
    - `python tests/test_manifest_config.py` passed.
    - `npm run build` passed.
13. Remaining blockers:
    - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
    - Manifest is not integrated with ingestion.
    - Retrieval evaluation harness is not implemented.
    - Response policy is not enforced.
    - Dependency major upgrades remain future work.

### 2026-05-14

1. Continued `PLANWORKLOG.md` Phase E.0 / D.2 only: retrieval evaluation smoke fixture baseline.
2. Created fake/sample-only retrieval evaluation fixture:
   - `data/sample_documents/sample_retrieval_eval.jsonl`
3. Added validation utilities:
   - `src/aviationrag/evaluation/smoke_fixture.py`
4. Added tests:
   - `tests/test_retrieval_smoke_fixture.py`
5. Updated evaluation/reset documentation to describe the smoke fixture baseline and clarify it is not real retrieval execution.
6. Updated `PLANWORKLOG.md` to mark the smoke fixture baseline complete while keeping real retrieval harness execution planned.
7. No real retrieval integration was performed.
8. No Astra or FAISS access occurred.
9. No embeddings were generated.
10. No ingestion runtime scripts were modified.
11. No real data paths were scanned.
12. No documents were reprocessed.
13. Validation after retrieval smoke fixture baseline:
    - `npm run sanitize:check:all` passed.
    - `python -m compileall src` passed.
    - `python tests/test_models.py` passed.
    - `python tests/test_sample_manifest_fixture.py` passed.
    - `python tests/test_manifest_writer.py` passed.
    - `python tests/test_legacy_adapter.py` passed.
    - `python tests/test_ingestion_dry_run.py` passed.
    - `python tests/test_local_manifest_dry_run_script.py` passed.
    - `python tests/test_manifest_config.py` passed.
    - `python tests/test_retrieval_smoke_fixture.py` passed.
    - `npm run build` passed.
14. Remaining blockers:
    - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
    - Manifest is not integrated with ingestion.
    - Retrieval harness is not implemented.
    - Response policy is not enforced.
    - Major dependency upgrades remain future work.

### 2026-05-15

1. Continued `PLANWORKLOG.md` Phase E.1 only: retrieval evaluation harness shell.
2. Added fake/mock retrieval result evaluator:
   - `src/aviationrag/evaluation/retrieval_harness.py`
3. Added tests:
   - `tests/test_retrieval_harness.py`
4. The harness shell evaluates caller-supplied fake/mock results for:
   - expected document match
   - expected chunk match
   - rank/top-k requirement
   - insufficient-evidence behavior
   - out-of-scope rejection behavior
5. No real retrieval integration was performed.
6. No Astra or FAISS access occurred.
7. No embeddings were generated.
8. No ingestion runtime scripts were modified.
9. Updated evaluation/reset documentation and `PLANWORKLOG.md` to keep real retrieval execution future-only.
10. Validation after retrieval harness shell:
    - `npm run sanitize:check:all` passed.
    - `python -m compileall src` passed.
    - `python tests/test_models.py` passed.
    - `python tests/test_sample_manifest_fixture.py` passed.
    - `python tests/test_manifest_writer.py` passed.
    - `python tests/test_legacy_adapter.py` passed.
    - `python tests/test_ingestion_dry_run.py` passed.
    - `python tests/test_local_manifest_dry_run_script.py` passed.
    - `python tests/test_manifest_config.py` passed.
    - `python tests/test_retrieval_smoke_fixture.py` passed.
    - `python tests/test_retrieval_harness.py` passed.
    - `npm run build` passed.
11. Remaining blockers:
    - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
    - Manifest is not integrated with ingestion.
    - Real retrieval harness is not wired to FAISS/Astra.
    - Response policy is not enforced.
    - Major dependency upgrades remain future work.

### 2026-05-15

1. Continued `PLANWORKLOG.md` Phase E.2 only: retrieval evaluation report/export shell.
2. Added fake/mock evaluation reporting utilities:
   - `src/aviationrag/evaluation/reporting.py`
3. Added tests:
   - `tests/test_retrieval_reporting.py`
4. The reporting shell supports:
   - JSON-serializable dictionaries for `EvaluationSummary` and `EvaluationCaseResult`
   - Markdown summary reports
   - explicit JSON report writes
   - explicit Markdown report writes
5. No optional report-generation tool script was created; no local report files were generated.
6. No real retrieval integration was performed.
7. No Astra or FAISS access occurred.
8. No embeddings were generated.
9. No ingestion runtime scripts were modified.
10. Generated reports remain future local/ignored outputs only.
11. Validation after retrieval report/export shell:
    - `npm run sanitize:check:all` passed.
    - `python -m compileall src` passed.
    - `python tests/test_models.py` passed.
    - `python tests/test_sample_manifest_fixture.py` passed.
    - `python tests/test_manifest_writer.py` passed.
    - `python tests/test_legacy_adapter.py` passed.
    - `python tests/test_ingestion_dry_run.py` passed.
    - `python tests/test_local_manifest_dry_run_script.py` passed.
    - `python tests/test_manifest_config.py` passed.
    - `python tests/test_retrieval_smoke_fixture.py` passed.
    - `python tests/test_retrieval_harness.py` passed.
    - `python tests/test_retrieval_reporting.py` passed.
    - `npm run build` passed.
12. Remaining blockers:
    - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
    - Manifest is not integrated with ingestion.
    - Real retrieval harness is not wired to FAISS/Astra.
    - Response policy is not enforced.
    - Major dependency upgrades remain future work.

### 2026-05-15

1. Continued `PLANWORKLOG.md` Phase D.2 only: metadata-rich chunk schema planning.
2. Created `docs/CHUNK_METADATA_SCHEMA.md` covering:
   - chunk identity model
   - required and optional chunk metadata fields
   - chunk type taxonomy
   - page, section, and paragraph traceability
   - extraction quality metadata
   - citation requirements
   - retrieval payload requirements
   - evaluation alignment
   - chunk lifecycle and versioning
   - future Astra payload and FAISS metadata alignment
   - staged migration plan and open questions
3. Updated cross-references in:
   - `docs/DOCUMENT_MANIFEST_SCHEMA.md`
   - `docs/EVALUATION_PLAN.md`
   - `docs/RESET_REBUILD_AND_EVALUATION_BASELINE.md`
4. Updated `PLANWORKLOG.md` to mark D.2 schema planning complete while keeping real chunk migration, re-embedding, and re-indexing planned.
5. No real chunking behavior changed.
6. No ingestion runtime scripts were modified.
7. No documents were reprocessed.
8. No embeddings were generated.
9. No Astra or FAISS changes were made.
10. Validation after chunk metadata schema planning:
    - `npm run sanitize:check:all` passed.
    - `python -m compileall src` passed.
    - `python tests/test_models.py` passed.
    - `python tests/test_sample_manifest_fixture.py` passed.
    - `python tests/test_retrieval_smoke_fixture.py` passed.
    - `python tests/test_retrieval_harness.py` passed.
    - `python tests/test_retrieval_reporting.py` passed.
    - `npm run build` passed.
11. Remaining blockers:
    - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
    - Manifest is not integrated with ingestion.
    - Real chunk migration is not implemented.
    - Real retrieval harness is not wired to FAISS/Astra.
    - Response policy is not enforced.
    - Major dependency upgrades remain future work.

### 2026-05-15

1. Continued `PLANWORKLOG.md` Phase D.2b only: fake chunk fixture expansion.
2. Expanded `data/sample_documents/sample_chunks.jsonl` to 15 fake metadata-rich chunk records.
3. Updated `tests/test_sample_manifest_fixture.py` to validate:
   - sample chunk fixture presence
   - at least 12 chunk records
   - unique `chunk_id` values
   - manifest document linkage
   - required chunk metadata fields
   - controlled chunk type values
   - valid page ranges
   - non-empty fake/synthetic text
   - no obvious real local/private paths
   - no generated `data/manifest/documents.jsonl`
4. Updated `docs/CHUNK_METADATA_SCHEMA.md` to document the expanded fake fixture and clarify it is not used by runtime ingestion.
5. All expanded chunk records use fake/sample-only text and metadata.
6. No real chunking behavior changed.
7. No ingestion runtime scripts were modified.
8. No documents were reprocessed.
9. No embeddings were generated.
10. No Astra or FAISS changes were made.
11. Validation after fake chunk fixture expansion:
    - `npm run sanitize:check:all` passed.
    - `python -m compileall src` passed.
    - `python tests/test_models.py` passed.
    - `python tests/test_sample_manifest_fixture.py` passed.
    - `python tests/test_retrieval_smoke_fixture.py` passed.
    - `python tests/test_retrieval_harness.py` passed.
    - `python tests/test_retrieval_reporting.py` passed.
    - `npm run build` passed.
12. Remaining blockers:
    - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
    - Manifest is not integrated with ingestion.
    - Real chunk migration is not implemented.
    - Real retrieval harness is not wired to FAISS/Astra.
    - Response policy is not enforced.
    - Major dependency upgrades remain future work.

### 2026-05-15

1. Continued `PLANWORKLOG.md` Phase D.2c only: chunk schema validator.
2. Created `src/aviationrag/ingestion/chunk_schema.py` with validation utilities for:
   - metadata-rich chunk dictionaries
   - lightweight `ChunkRecord` objects
   - JSONL chunk fixture loading
   - dataset-level duplicate `chunk_id` and document linkage checks
3. Added `tests/test_chunk_schema.py` using fake/sample fixtures and inline fake negative cases only.
4. Updated `docs/CHUNK_METADATA_SCHEMA.md` to document the validator and clarify it is not integrated with runtime ingestion.
5. Fake/sample data only was used.
6. No real chunking behavior changed.
7. No ingestion runtime scripts were modified.
8. No documents were reprocessed.
9. No embeddings were generated.
10. No Astra or FAISS changes were made.
11. Validation after chunk schema validator:
    - `npm run sanitize:check:all` passed.
    - `python -m compileall src` passed.
    - `python tests/test_models.py` passed.
    - `python tests/test_sample_manifest_fixture.py` passed.
    - `python tests/test_chunk_schema.py` passed.
    - `python tests/test_retrieval_smoke_fixture.py` passed.
    - `python tests/test_retrieval_harness.py` passed.
    - `python tests/test_retrieval_reporting.py` passed.
    - `npm run build` passed.
12. Remaining blockers:
    - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
    - Manifest is not integrated with ingestion.
    - Real chunk migration is not implemented.
    - Real retrieval harness is not wired to FAISS/Astra.
    - Response policy is not enforced.
    - Major dependency upgrades remain future work.

### 2026-05-15

1. Continued `PLANWORKLOG.md` Phase D.2d only: fake chunk payload exporter.
2. Created `src/aviationrag/ingestion/chunk_payload.py` to convert validated fake/sample chunks into future vector payload-shaped dictionaries.
3. Added `tests/test_chunk_payload.py` using fake/sample chunk data only.
4. Created local-only developer export tool:
   - `tools/chunking/export-sample-chunk-payloads.py`
5. The sample export tool writes generated payload JSONL under ignored `logs/chunking/sample_chunk_payloads.jsonl`.
6. Removed the generated local payload output after validation; no generated logs were staged.
7. Updated `docs/CHUNK_METADATA_SCHEMA.md` and `docs/RESET_REBUILD_AND_EVALUATION_BASELINE.md` to document the fake/sample payload exporter and its limits.
8. Fake/sample data only was used.
9. No embeddings were generated.
10. No Astra or FAISS writes occurred.
11. No real chunking behavior changed.
12. No ingestion runtime scripts were modified.
13. No documents were reprocessed.
14. Validation after fake chunk payload exporter:
    - `npm run sanitize:check:all` passed.
    - `python -m compileall src` passed.
    - `python tests/test_models.py` passed.
    - `python tests/test_sample_manifest_fixture.py` passed.
    - `python tests/test_chunk_schema.py` passed.
    - `python tests/test_chunk_payload.py` passed.
    - `python tests/test_retrieval_smoke_fixture.py` passed.
    - `python tests/test_retrieval_harness.py` passed.
    - `python tests/test_retrieval_reporting.py` passed.
    - `python tools/chunking/export-sample-chunk-payloads.py` passed.
    - `git check-ignore logs/chunking/sample_chunk_payloads.jsonl` passed.
    - `npm run build` passed.
15. Remaining blockers:
    - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
    - Manifest is not integrated with ingestion.
    - Real chunk migration is not implemented.
    - Real vector indexing is not implemented.
    - Real retrieval harness is not wired to FAISS/Astra.
    - Response policy is not enforced.
    - Major dependency upgrades remain future work.

### 2026-05-15

1. Continued `PLANWORKLOG.md` Phase D.2e only: gated legacy chunk adapter.
2. Added disabled-by-default chunk migration config flags in `src/aviationrag/config.py`:
   - `AVIATIONRAG_ENABLE_CHUNK_MIGRATION`
   - `AVIATIONRAG_CHUNK_MIGRATION_DRY_RUN`
3. Documented the future chunk migration flags in `.env.example`.
4. Created passive adapter module:
   - `src/aviationrag/ingestion/chunk_legacy_adapter.py`
5. Added `tests/test_chunk_legacy_adapter.py` using fake inline legacy-like chunk dictionaries only.
6. The adapter converts fake legacy-like chunks into `ChunkRecord` objects and optional vector payload-shaped dictionaries for preview only.
7. Updated `docs/CHUNK_METADATA_SCHEMA.md` and `docs/INGESTION_MANIFEST_INTEGRATION_PLAN.md` to document the gated legacy chunk adapter.
8. Fake/sample data only was used.
9. The adapter is not wired into runtime ingestion.
10. No real chunking behavior changed.
11. No ingestion runtime scripts were modified.
12. No documents were reprocessed.
13. No embeddings were generated.
14. No Astra or FAISS writes occurred.
15. Validation after gated legacy chunk adapter:
    - `npm run sanitize:check:all` passed.
    - `python -m compileall src` passed.
    - `python tests/test_models.py` passed.
    - `python tests/test_manifest_config.py` passed.
    - `python tests/test_sample_manifest_fixture.py` passed.
    - `python tests/test_chunk_schema.py` passed.
    - `python tests/test_chunk_payload.py` passed.
    - `python tests/test_chunk_legacy_adapter.py` passed.
    - `python tests/test_retrieval_smoke_fixture.py` passed.
    - `python tests/test_retrieval_harness.py` passed.
    - `python tests/test_retrieval_reporting.py` passed.
    - `npm run build` passed.
16. Remaining blockers:
    - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
    - Manifest is not integrated with ingestion.
    - Real chunk migration is not implemented.
    - Real vector indexing is not implemented.
    - Real retrieval harness is not wired to FAISS/Astra.
    - Response policy is not enforced.
    - Major dependency upgrades remain future work.

### 2026-05-15

1. Continued `PLANWORKLOG.md` Phase D.3 only: real chunk migration design.
2. Created `docs/REAL_CHUNK_MIGRATION_DESIGN.md` covering:
    - current legacy chunking assumptions
    - target metadata-rich chunk model
    - migration prerequisites
    - legacy-to-new mapping strategy
    - deterministic chunk ID strategy
    - chunk type assignment strategy
    - page, section, and paragraph traceability
    - table, figure, warning, caution, and note handling
    - extraction quality and manual review handling
    - vector payload generation strategy
    - evaluation gates
    - Astra/FAISS reset dependency
    - gated rollout, rollback, risks, future phases, and open questions
3. Created `docs/checklists/CHUNK_MIGRATION_GO_NO_GO.md` as a future operational checklist.
4. Updated cross-references in:
    - `docs/CHUNK_METADATA_SCHEMA.md`
    - `docs/INGESTION_MANIFEST_INTEGRATION_PLAN.md`
    - `docs/RESET_REBUILD_AND_EVALUATION_BASELINE.md`
    - `docs/EVALUATION_PLAN.md`
5. No real chunk migration was performed.
6. No ingestion runtime scripts were modified.
7. No documents were reprocessed.
8. No embeddings were generated.
9. No Astra or FAISS reset or writes occurred.
10. Validation after real chunk migration design:
    - `npm run sanitize:check:all` passed.
    - `python -m compileall src` passed.
    - `python tests/test_models.py` passed.
    - `python tests/test_chunk_schema.py` passed.
    - `python tests/test_chunk_payload.py` passed.
    - `python tests/test_chunk_legacy_adapter.py` passed.
    - `python tests/test_retrieval_smoke_fixture.py` passed.
    - `python tests/test_retrieval_harness.py` passed.
    - `python tests/test_retrieval_reporting.py` passed.
    - `npm run build` passed.
11. Remaining blockers:
    - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
    - Manifest is not integrated with ingestion.
    - Real chunk migration is not implemented.
    - Real vector indexing is not implemented.
    - Real retrieval harness is not wired to FAISS/Astra.
    - Response policy is not enforced.
    - Major dependency upgrades remain future work.

### 2026-05-16

1. Continued `PLANWORKLOG.md` Phase D.3b only: read-only legacy chunk format audit.
2. Created read-only audit module:
   - `src/aviationrag/ingestion/chunk_audit.py`
3. Created manual audit tool script:
   - `tools/chunking/audit-legacy-chunks.py`
4. Added tests:
   - `tests/test_chunk_audit.py`
5. Default audit input uses fake/sample chunks from `data/sample_documents/sample_chunks.jsonl`.
6. The audit tool accepts only explicit file paths and does not scan directories by default.
7. The audit summary redacts and summarizes text values by type and length only.
8. No real chunk migration was performed.
9. No ingestion runtime scripts were modified.
10. No documents were reprocessed.
11. No embeddings were generated.
12. No Astra or FAISS changes were made.
13. Validation after read-only legacy chunk format audit:
   - `npm run sanitize:check:all` passed.
   - `python -m compileall src` passed.
   - `python tests/test_models.py` passed.
   - `python tests/test_chunk_schema.py` passed.
   - `python tests/test_chunk_payload.py` passed.
   - `python tests/test_chunk_legacy_adapter.py` passed.
   - `python tests/test_chunk_audit.py` passed.
   - `python tests/test_retrieval_smoke_fixture.py` passed.
   - `python tests/test_retrieval_harness.py` passed.
   - `python tests/test_retrieval_reporting.py` passed.
   - `python tools/chunking/audit-legacy-chunks.py` passed.
   - `git check-ignore logs/chunking/legacy_chunk_audit.json` passed.
   - `npm run build` passed.
14. Remaining blockers:
   - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
   - Manifest is not integrated with ingestion.
   - Real chunk migration is not implemented.
   - Real vector indexing is not implemented.
   - Real retrieval harness is not wired to FAISS/Astra.
   - Response policy is not enforced.
   - Major dependency upgrades remain future work.

### 2026-05-16

1. Continued `PLANWORKLOG.md` Phase D.3c only: fake/local chunk migration dry run.
2. Created dry-run module:
   - `src/aviationrag/ingestion/chunk_migration_dry_run.py`
3. Created manual dry-run tool script:
   - `tools/chunking/run-chunk-migration-dry-run.py`
4. Added tests:
   - `tests/test_chunk_migration_dry_run.py`
5. Default dry run uses fake/sample chunks from `data/sample_documents/sample_chunks.jsonl`.
6. The dry-run report is summary-only and writes to ignored `logs/chunking/chunk_migration_dry_run.json` when the tool is run.
7. No real chunk migration was performed.
8. No ingestion runtime scripts were modified.
9. No documents were reprocessed.
10. No embeddings were generated.
11. No Astra or FAISS changes were made.
12. Generated report remains ignored/local-only and must not be committed.
13. Validation after fake/local chunk migration dry run:
   - `npm run sanitize:check:all` passed.
   - `python -m compileall src` passed.
   - `python tests/test_models.py` passed.
   - `python tests/test_chunk_schema.py` passed.
   - `python tests/test_chunk_payload.py` passed.
   - `python tests/test_chunk_legacy_adapter.py` passed.
   - `python tests/test_chunk_audit.py` passed.
   - `python tests/test_chunk_migration_dry_run.py` passed.
   - `python tests/test_retrieval_smoke_fixture.py` passed.
   - `python tests/test_retrieval_harness.py` passed.
   - `python tests/test_retrieval_reporting.py` passed.
   - `python tools/chunking/run-chunk-migration-dry-run.py` passed.
   - `git check-ignore logs/chunking/chunk_migration_dry_run.json` passed.
   - `npm run build` passed.
14. Remaining blockers:
   - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
   - Manifest is not integrated with ingestion.
   - Real chunk migration is not implemented.
   - Real vector indexing is not implemented.
   - Real retrieval harness is not wired to FAISS/Astra.
   - Response policy is not enforced.
   - Major dependency upgrades remain future work.

### 2026-05-16

1. Continued `PLANWORKLOG.md` Phase D.3d only: gated local chunk conversion writing ignored outputs.
2. Created conversion writer module:
   - `src/aviationrag/ingestion/chunk_conversion_writer.py`
3. Created manual conversion tool script:
   - `tools/chunking/write-local-chunk-conversion.py`
4. Added tests:
   - `tests/test_chunk_conversion_writer.py`
5. Added `.gitignore` coverage for generated local conversion outputs under `data/migration_dry_run/`.
6. The conversion writer requires explicit local-write permission through `--allow-local-write` or the future chunk migration flag.
7. Outputs go to ignored local paths only:
   - `converted_chunks.jsonl`
   - `vector_payloads.jsonl`
   - `conversion_report.json`
8. No real chunk migration was performed.
9. No ingestion runtime scripts were modified.
10. No documents were reprocessed.
11. No embeddings were generated.
12. No Astra or FAISS changes were made.
13. Generated conversion outputs were not committed.
14. Validation after gated local chunk conversion writer:
   - `npm run sanitize:check:all` passed.
   - `python -m compileall src` passed.
   - `python tests/test_models.py` passed.
   - `python tests/test_chunk_schema.py` passed.
   - `python tests/test_chunk_payload.py` passed.
   - `python tests/test_chunk_legacy_adapter.py` passed.
   - `python tests/test_chunk_audit.py` passed.
   - `python tests/test_chunk_migration_dry_run.py` passed.
   - `python tests/test_chunk_conversion_writer.py` passed.
   - `python tests/test_retrieval_smoke_fixture.py` passed.
   - `python tests/test_retrieval_harness.py` passed.
   - `python tests/test_retrieval_reporting.py` passed.
   - `python tools/chunking/write-local-chunk-conversion.py --allow-local-write` passed.
   - `git check-ignore data/migration_dry_run/chunks/converted_chunks.jsonl` passed.
   - `git check-ignore data/migration_dry_run/chunks/vector_payloads.jsonl` passed.
   - `git check-ignore data/migration_dry_run/chunks/conversion_report.json` passed.
   - `npm run build` passed.
15. Remaining blockers:
   - Production bridge remains an external blocker until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
   - Manifest is not integrated with ingestion.
   - Real chunk migration is not implemented.
   - Real vector indexing is not implemented.
   - Real retrieval harness is not wired to FAISS/Astra.
   - Response policy is not enforced.
   - Major dependency upgrades remain future work.

### 2026-07-18

1. Completed maintenance checkpoint M.1/M.2 before starting D.4.
2. Validated the repository `.venv`:
   - Python `3.12.10`.
   - `.venv` `pip check` passed.
   - `.venv` Python compilation of `src` passed.
   - Targeted chunking and retrieval-shell tests passed with `.venv`.
3. Confirmed global Python package conflicts do not affect the validated repository `.venv`.
4. Recorded environment follow-up items:
   - Existing Python dependency declarations are unpinned and only partially reproduce the current `.venv`.
   - `node@22.13.0` is currently declared as an application dependency and needs later controlled correction.
5. Reconciled roadmap numbering so the completed sequence is:
   - D.3 real chunk migration design.
   - D.3b read-only legacy chunk audit.
   - D.3c fake/local chunk migration dry run.
   - D.3d gated local chunk conversion writer.
6. Reconciled the next intended phase as D.4 - page and structure preservation design.
7. No dependency, source, runtime, ingestion, retrieval, embedding, Astra, FAISS, API, prompt, or generated-data changes were made.
8. D.4 remains not started.
9. Remaining blockers:
   - Production bridge remains externally blocked until a real public HTTPS bridge endpoint is provisioned and configured in Vercel.
   - Dependency hardening remains separate future work.
   - Real chunk migration is not implemented.
   - Real embedding regeneration and vector indexing are not implemented.
   - Real retrieval harness is not wired to FAISS/Astra.
   - Response policy is not enforced.

### 2026-07-18

1. Completed D.4 page and structure preservation design as documentation-only work.
2. Added `docs/PAGE_AND_STRUCTURE_PRESERVATION_DESIGN.md`.
3. Added the synthetic design fixture `data/sample_documents/sample_structured_document.json`.
4. Defined future provenance handling for:
   - page numbers and printed page labels
   - section and subsection hierarchy
   - paragraph and clause identifiers
   - table, figure, caption, and equation provenance
   - warning, caution, and note classification
   - appendices, cross-references, source spans, confidence, and citation-ready retrieval metadata
5. Updated roadmap, architecture, chunk schema, reset/rebuild, and migration design docs to reference D.4 as a completed design gate.
6. No dependency, source, runtime, ingestion, retrieval, embedding, Astra, FAISS, API, prompt, or generated-data behavior changes were made.
7. Real parser integration, real document reprocessing, real chunk migration, embedding regeneration, Astra rebuild, FAISS rebuild, real retrieval harness wiring, and response-policy enforcement remain future work.
8. Dependency hardening remains separate future maintenance work.
9. Production bridge remains externally blocked.
10. Validation after D.4:
   - `npm run sanitize:check:all` passed.
   - `.venv` `python -m compileall src` passed.
   - `.venv` JSON syntax check for `data/sample_documents/sample_structured_document.json` passed.
   - `.venv` `tests/test_chunk_migration_dry_run.py` passed.
   - `.venv` `tests/test_chunk_conversion_writer.py` passed.
   - `.venv` `tests/test_retrieval_harness.py` passed.
   - `.venv` `tests/test_retrieval_reporting.py` passed.

## Session Recovery Procedure

If the chat/session freezes:

1. Re-open this repository and read `WORKLOG.md` + `SANITIZATION_REPORT.md`.
2. Run `git status -b --short`.
3. Continue from the latest `In Progress` item in this file.
4. Commit and push after each completed sub-step.
