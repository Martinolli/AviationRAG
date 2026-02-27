# WORKLOG

Last Updated: 2026-02-27  
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
3. `Pending` Step 3 upload workflow (UI + API + ingestion status).
4. `Pending` Step 4 formula rendering in chat.
5. `Pending` Step 5 final production-readiness checklist and release gate.

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

### Step 3: Upload Workflow

1. `Pending` Web upload UI and ingestion status tracking.
2. `Pending` Upload API + validation + queue/state pipeline.

### Step 4: Formula Rendering

1. `Pending` Markdown + math rendering (`remark-math`/`rehype-katex`) in chat responses.
2. `Pending` Safety sanitization for rendered Markdown output.

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

## Session Recovery Procedure

If the chat/session freezes:

1. Re-open this repository and read `WORKLOG.md` + `SANITIZATION_REPORT.md`.
2. Run `git status -b --short`.
3. Continue from the latest `In Progress` item in this file.
4. Commit and push after each completed sub-step.
