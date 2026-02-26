# WORKLOG

Last Updated: 2026-02-26  
Active Branch: `hardening/sanitize-repo`

## Purpose

Persistent execution log for deployment hardening and product-readiness work so progress is recoverable even if chat or IDE session is interrupted.

## Active Plan

1. `Done` Step 1 repository sanitization baseline.
2. `In Progress` Step 2 deployment hardening:
   - `Done` API rate limiting and request validation.
   - `Done` Deployment routing config cleanup.
   - `Done` CI pipeline for build + smoke tests.
   - `Pending` Python worker deployment architecture split/containerization decision.
3. `Pending` Step 3 upload workflow (UI + API + ingestion status).
4. `Pending` Step 4 formula rendering in chat.
5. `Pending` Step 5 final production-readiness checklist and release gate.

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

## Session Recovery Procedure

If the chat/session freezes:

1. Re-open this repository and read `WORKLOG.md` + `SANITIZATION_REPORT.md`.
2. Run `git status -b --short`.
3. Continue from the latest `In Progress` item in this file.
4. Commit and push after each completed sub-step.
