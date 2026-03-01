# Vercel Online Setup Guide

Date: 2026-03-01  
Status: Active

## Why this is required

Vercel serverless functions should not depend on local long-lived Python child processes.  
For Vercel deployment, use:

- `AVIATION_API_MODE=http`
- External Aviation bridge service (`/health`, `/command`)

## 1. Deploy the HTTP bridge service first

Deploy `src/scripts/py_files/aviationai_http_bridge.py` on a host that supports Python runtime and network access to Astra/OpenAI.

Minimum bridge environment variables:

```env
OPENAI_API_KEY=
ASTRA_DB_SECURE_BUNDLE_PATH=
ASTRA_DB_APPLICATION_TOKEN=
ASTRA_DB_KEYSPACE=
AVIATION_API_HTTP_TOKEN=<strong-shared-token>
AVIATION_API_HTTP_BIND=0.0.0.0
AVIATION_API_HTTP_PORT=8010
```

Run command:

```powershell
python src\scripts\py_files\aviationai_http_bridge.py
```

Validate bridge is reachable:

1. `GET https://<bridge-host>/health` returns `success: true`.
2. `POST https://<bridge-host>/command` with bearer token and action `ping` returns `success: true`.

## 2. Configure Vercel project environment variables

In Vercel Dashboard:

1. Open project.
2. Go to `Settings -> Environment Variables`.
3. Add variables for `Production` and `Preview`.

Required app variables:

```env
OPENAI_API_KEY=
ASTRA_DB_SECURE_BUNDLE_PATH=
ASTRA_DB_APPLICATION_TOKEN=
ASTRA_DB_KEYSPACE=
NEXTAUTH_URL=https://<your-vercel-domain>
NEXTAUTH_SECRET=<strong-random-secret>
APP_AUTH_EMAIL=<login-email>
APP_AUTH_PASSWORD_HASH=sha256:<hex_digest>
AVIATION_API_MODE=http
AVIATION_API_HTTP_URL=https://<bridge-host>
AVIATION_API_HTTP_TOKEN=<same-bridge-token>
AVIATION_API_TIMEOUT_MS=180000
DOCUMENT_UPLOAD_MAX_MB=25
DOCUMENT_UPLOAD_AUTO_INGEST=true
DOCUMENT_UPLOAD_STEP_TIMEOUT_MS=1200000
READ_DOC_ENABLE_LLAMA_PARSE=true
LLAMA_CLOUD_API_KEY=
```

Notes:

1. Keep `APP_AUTH_PASSWORD` empty in production when using hash mode.
2. Generate password hash locally:
   - `npm run auth:hash -- "your-password"`
3. If your bridge requires static egress allow-listing, use Vercel documented IP strategy or place bridge behind authenticated public endpoint.

## 3. Trigger Vercel redeploy

After saving environment variables:

1. Redeploy latest commit from Vercel dashboard, or
2. Push a commit to the connected branch.

## 4. Post-deploy validation checklist

From the deployed app domain:

1. `GET /api/health`:
   - `bridge_mode` should be `"http"`.
   - `checks.aviation_http_url_set` should be `true`.
2. `GET /api/health?deep=1`:
   - `checks.aviation_http_ping` should be `true`.
   - `deep_check_error` should be absent.
3. Sign in with configured credentials.
4. Create a new chat and confirm response returns.
5. Upload one document and track status progression.
6. Open an existing session and confirm history appears.

## 5. Rollback strategy

If bridge service is unstable:

1. Set `AVIATION_API_MODE=worker` only for local/non-Vercel deployments.
2. For Vercel production, keep `http` mode and rollback bridge deployment to the last stable version.
3. Re-run `/api/health?deep=1` before restoring traffic.

