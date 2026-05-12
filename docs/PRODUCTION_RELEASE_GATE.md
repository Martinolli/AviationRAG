# Production Release Gate

Date: 2026-05-12  
Scope: `PLANWORKLOG.md` Phase A.2 only  
Decision: `No-Go`

This checklist records the current production-readiness state. It does not start RAG retrieval work, backend refactoring, response policy work, metadata redesign, or evaluation harness work.

## Status Legend

- `Pass`: Verified in this session or evidenced by the current repository state.
- `Blocked`: Cannot pass until an upstream blocker is resolved.
- `Pending`: Requires operator action, production access, or manual authenticated validation.
- `Waived`: Explicitly accepted for release with a reason.

## Secrets and Environment

| Check | Status | Evidence / Action |
| --- | --- | --- |
| `.env` is not tracked | Pass | `git ls-files .env .env.* secure-connect-*.zip` returns only `.env.example`. |
| Secure Connect bundles are not tracked | Pass | `secure-connect-*.zip` is ignored and no bundle is tracked. |
| Runtime/generated data remains ignored | Pass | Sanitization guardrail passed with `npm run sanitize:check:all`. |
| Production secrets rotated or confirmed safe | Pending | Requires user/operator confirmation outside repository state. |
| Local bridge bind env is valid | Pending | Local `.env` currently has malformed `AVIATION_API_HTTP_BIND`; Phase A.1 validation passed only with process-only override `AVIATION_API_HTTP_BIND=127.0.0.1`. |

## Build/Test Validation

| Check | Status | Evidence / Action |
| --- | --- | --- |
| Sanitization check | Pass | `npm run sanitize:check:all` passed on 2026-05-12. |
| Production build | Pass | `npm run build` passed on 2026-05-12. |
| Smoke test | Pass | `npm run test:smoke` passed on 2026-05-12 in the prior Phase A.2 checklist pass. |
| Secret scan CI | Pending | Local `gitleaks` is not installed; confirm GitHub Actions gitleaks job after push. |

## Bridge Validation

| Check | Status | Evidence / Action |
| --- | --- | --- |
| Local bridge `/health` | Pass | With process-only bind override, `GET http://127.0.0.1:8010/health` returned `success=true`. |
| Local bridge `/command` ping | Pass | With process-only bind override, authenticated `POST http://127.0.0.1:8010/command` with `{"action":"ping"}` returned `success=true`. |
| Local app deep health | Pass | Local `GET http://127.0.0.1:3000/api/health?deep=1` returned `bridge_mode="http"` and `checks.aviation_http_ping=true`. |
| Public HTTPS bridge endpoint | Blocked | No real public HTTPS bridge endpoint is configured locally. Previously documented `https://aviation-api-http-bridge.com` still fails DNS resolution. |
| Vercel production bridge ping | Blocked | `https://aviation-rag.vercel.app/api/health?deep=1` returns `checks.aviation_http_ping=false` and `deep_check_error="Bridge returned non-JSON response (503)."`. |

## App Validation

| Check | Status | Evidence / Action |
| --- | --- | --- |
| App shell smoke test | Pass | Playwright smoke test passed: chat composer remains visible and drawer overlays without shrinking chat. |
| Authenticated chat from production | Blocked | Cannot release until production bridge ping succeeds and production chat can reach the backend bridge. |
| Upload flow with small PDF/DOCX | Pending | Requires manual authenticated verification with a small non-private sample document. |
| Session list/history retrieval | Pending | Requires manual authenticated verification in the intended release environment. |

## Deployment Validation

| Check | Status | Evidence / Action |
| --- | --- | --- |
| Vercel is configured for HTTP bridge mode | Pass | Production deep health reports `bridge_mode="http"` and `checks.aviation_http_url_set=true`. |
| Vercel reaches healthy bridge `/command` | Blocked | Production deep health reports `checks.aviation_http_ping=false`. |
| Vercel redeployed after valid bridge env update | Pending | Requires user/operator action after a real public bridge URL is available. |
| Rollback path documented | Pass | See `docs/VERCEL_ONLINE_SETUP.md` and `docs/RUNBOOK_VERCEL_LOCAL_SYNC.md`. |

## Known Limitations

1. Production release is blocked until a real public HTTPS aviation bridge endpoint is available.
2. Production `/api/health?deep=1` is still failing `checks.aviation_http_ping`.
3. The current Vercel bridge target appears reachable enough to return a `503`, but it is not returning the expected bridge JSON response.
4. Local `.env` needs `AVIATION_API_HTTP_BIND` cleanup for normal local bridge startup.
5. Secret rotation status cannot be proven from repository files.
6. GitHub Actions secret scanning must be checked after pushing.
7. Upload and history/session flows still need authenticated manual verification.

## Required User Action

1. Provision a real public HTTPS endpoint for the aviation HTTP bridge.
2. Validate the public bridge directly:
   - `GET /health`
   - authenticated `POST /command` with `{"action":"ping"}`
   - `npm run bridge:check -- --url <public-bridge-url> --token <token>`
3. Configure Vercel `AVIATION_API_HTTP_URL` to the validated bridge base URL.
4. Ensure Vercel `AVIATION_API_HTTP_TOKEN` matches the bridge token.
5. Redeploy Vercel after the environment update.
6. Re-run `https://aviation-rag.vercel.app/api/health?deep=1` and require `checks.aviation_http_ping=true`.

## Go / No-Go Status

Current status: `No-Go`

Reason: Phase A.1 public bridge/Vercel cutover remains blocked. The release gate cannot move to `Go` until production deep health reports a successful aviation HTTP ping and the pending manual app checks are completed or explicitly waived.
