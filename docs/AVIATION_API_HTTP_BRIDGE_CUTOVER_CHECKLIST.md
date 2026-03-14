# Aviation API HTTP Bridge Cutover Checklist

Date: 2026-02-27  
Status: Pending execution

## Phase A: Service Preparation

1. Implement `/command` endpoint according to `AVIATION_API_HTTP_BRIDGE_SPEC.md`.
   - Reference implementation available at `src/scripts/py_files/aviationai_http_bridge.py`.
2. Implement token auth (`Authorization: Bearer`).
3. Implement server-side request validation.
4. Implement action-level logging with correlation ID (`id`).
5. Implement service health endpoint (`/health`).

## Phase B: Environment Configuration

1. Set Next.js env vars:
   - `AVIATION_API_MODE=http`
   - `AVIATION_API_HTTP_URL=<bridge_base_url>`
   - `AVIATION_API_HTTP_TOKEN=<shared_secret>`
2. Keep fallback config documented:
   - `AVIATION_API_MODE=worker`
3. Verify `/api/health` returns:
   - `bridge_mode: "http"`
   - `checks.aviation_http_url_set: true`
4. Verify deep bridge ping:
   - `GET /api/health?deep=1`
   - `checks.aviation_http_ping: true`
   - no `deep_check_error`

## Phase C: Validation

1. Run app smoke test:
   - `npm run test:smoke`
2. Manual command checks:
   - New chat request (`ask`)
   - Session list and open history
   - Session rename/pin/delete
3. Failure-path checks:
   - Invalid token
   - Timeout simulation
   - Invalid action payload

## Phase D: Rollout

1. Enable in staging first.
2. Run 24h stability window with logs/alerts.
3. Promote to production only if no regression.
4. Keep rollback path:
   - switch `AVIATION_API_MODE=worker`
   - redeploy application

## Phase E: Post-Cutover

1. Document final architecture in README.
2. Remove obsolete local-only assumptions from ops runbooks.
3. Add bridge availability and latency to monitoring dashboard.
