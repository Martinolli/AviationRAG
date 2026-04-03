# Runbook: Local vs Vercel Bridge Config

Date: 2026-04-03  
Status: Active

## Purpose

Quick reference to avoid configuration drift between local app runtime and Vercel runtime.

## 1) Environment Split

Use different `AVIATION_API_HTTP_URL` values per environment.

### Local `.env`

```env
AVIATION_API_MODE=http
AVIATION_API_HTTP_URL=http://127.0.0.1:8010
AVIATION_API_HTTP_TOKEN=<bridge-token>
```

### Vercel Environment Variables (Production/Preview)

```env
AVIATION_API_MODE=http
AVIATION_API_HTTP_URL=https://<public-bridge-url>
AVIATION_API_HTTP_TOKEN=<bridge-token>
```

Notes:

1. Never use `127.0.0.1` or `localhost` in Vercel.
2. `AVIATION_API_HTTP_URL` must be bridge base URL only (no `/command`).
3. Token must match bridge runtime token exactly.

## 2) Local Runtime Sequence

Use two terminals:

1. Start bridge:

```powershell
python src\scripts\py_files\aviationai_http_bridge.py
```

2. Start app:

```powershell
npm run dev
```

3. Validate:

```powershell
Invoke-WebRequest http://127.0.0.1:8010/health
Invoke-WebRequest http://127.0.0.1:3000/api/health?deep=1
```

Expected: `aviation_http_ping: true`

## 3) Bridge Preflight Check

From repo root:

```powershell
npm run bridge:check -- --url https://<public-bridge-url> --token <bridge-token>
```

Expected:

```json
{
  "success": true
}
```

## 4) Vercel Validation Sequence

1. Update env vars in Vercel.
2. Redeploy latest build.
3. Validate:

```powershell
(Invoke-WebRequest "https://aviation-rag.vercel.app/api/health?deep=1").Content
```

Expected fields:

1. `"bridge_mode":"http"`
2. `"aviation_http_url_set":true`
3. `"aviation_http_ping":true`
4. no `deep_check_error`

## 5) Error-to-Fix Map

1. `fetch failed` (local app):
   - Bridge process is down or wrong local URL.
2. `fetch failed` (Vercel):
   - Vercel cannot reach bridge URL (DNS/network/tunnel down).
3. `Bridge returned non-JSON response (404)`:
   - Wrong URL target (often app URL instead of bridge URL).
4. `aviationai_worker.py process error: spawn python ENOENT`:
   - App is in worker mode on Vercel; set `AVIATION_API_MODE=http` and redeploy.
5. `Unauthorized` on `/command`:
   - Token mismatch between caller and bridge runtime.

## 6) Session Checklist Before Stopping Work

1. Save current tunnel/bridge URL used in Vercel.
2. Record latest health check output in `WORKLOG.md`.
3. Confirm branch is clean (`git status -sb`).

