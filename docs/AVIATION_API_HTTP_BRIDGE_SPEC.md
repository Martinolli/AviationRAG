# Aviation API HTTP Bridge Spec

Date: 2026-02-27  
Status: Draft v1  
Owner: Platform hardening track

## Goal

Define a stable HTTP contract for externalizing Aviation command execution so the Next.js app can run in environments where local Python process spawning is not viable.

## Endpoint

- Method: `POST`
- Path: `/command`
- Content-Type: `application/json`
- Auth: `Authorization: Bearer <token>` (recommended for production)

## Request Schema

Base request object:

```json
{
  "id": "req_1730000000000_1",
  "action": "ask",
  "...action_fields": "..."
}
```

Common fields:

1. `id` (string, required): request correlation ID.
2. `action` (string, required): one of:
   - `ping`
   - `ask`
   - `history`
   - `sessions_list`
   - `session_upsert`
   - `session_delete`

Action-specific fields:

1. `ask`
   - `message` (string, required)
   - `session_id` (string, optional)
   - `strict_mode` (boolean, optional)
   - `target_document` (string, optional)
   - `model` (string, optional)
   - `store` (boolean, optional; default `true`)
2. `history`
   - `session_id` (string, required)
   - `limit` (number, optional; `1..50`)
3. `sessions_list`
   - `search` (string, optional)
   - `filter` (string, optional; `all|recent|pinned`)
   - `limit` (number, optional; `1..200`)
4. `session_upsert`
   - `session_id` (string, optional; generated if empty)
   - `title` (string, optional)
   - `pinned` (boolean, optional)
5. `session_delete`
   - `session_id` (string, required)
   - `purge_history` (boolean, optional; default `true`)

## Response Schema

Base response object:

```json
{
  "id": "req_1730000000000_1",
  "success": true,
  "action": "ask"
}
```

Success responses:

1. `ask`
   - `session_id`, `answer`, `strict_mode`, `target_filename`, `citations`, `sources`
2. `history`
   - `session_id`, `messages`
3. `sessions_list`
   - `sessions`
4. `session_upsert`
   - `session`
5. `session_delete`
   - `session_id`, `removed_metadata`, `purged_history`, `deleted_rows`
6. `ping`
   - no extra fields required

Error response:

```json
{
  "id": "req_1730000000000_1",
  "success": false,
  "error": "Human readable message"
}
```

## Status Code Guidance

1. `200`: command handled (including `success=false` domain errors).
2. `400`: invalid payload shape, missing required fields, unknown action.
3. `401`/`403`: auth failure (missing or invalid bearer token).
4. `429`: throttled by bridge service.
5. `500`: internal bridge failure.

## Timeout and Retries

1. Caller timeout configured by `AVIATION_API_TIMEOUT_MS` (default 180000ms).
2. Bridge service should complete within caller timeout budget.
3. If timeout exceeded, caller fails request and does not auto-retry by default.

## Security Requirements

1. Require bearer token in production.
2. Restrict network access to trusted callers only.
3. Log `id`, `action`, duration, and high-level outcome (never log secrets).
4. Apply per-IP and/or per-token rate limiting on bridge endpoint.

## Compatibility Notes

1. Current Next.js integration supports both:
   - `AVIATION_API_MODE=worker` (local child process)
   - `AVIATION_API_MODE=http` (this contract)
2. This spec must remain backward-compatible with the request payload produced by:
   - `src/utils/server/aviation_api_bridge.ts`
