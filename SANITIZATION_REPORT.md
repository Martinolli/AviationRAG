# Sanitization Report

Date: 2026-02-26  
Branch: `hardening/sanitize-repo`

## Scope

Step 1 sanitization tasks executed for repository hygiene, basic secret prevention, runtime artifact cleanup, and dependency risk reduction.

## Completed Changes

1. Created sanitize branch: `hardening/sanitize-repo`.
2. Removed runtime chat artifacts from git tracking (files remain local):
   - `chat_id/last_session_id.txt`
   - `chat_id/session_metadata.json`
3. Expanded ignore rules in `.gitignore`:
   - `chat_id/*.json`
4. Added local git hook path installer:
   - `npm run hooks:install`
5. Added pre-commit sanitization hook:
   - `.githooks/pre-commit`
   - `tools/sanitize/precommit-check.mjs`
6. Added npm scripts:
   - `sanitize:check` (staged files)
   - `sanitize:check:all` (full tracked scan)
7. Applied non-breaking dependency fixes:
   - `npm audit fix --omit=dev`

## Scan Results

### Secret Scan

Tool availability:

- `gitleaks`: not installed
- `trufflehog`: not installed

Fallback regex scan findings:

- No exposed live keys detected in tracked files.
- Placeholder variables only found in:
  - `.env.example`
  - `README.md`

### Large File Findings

Current tracked files over 10 MB:

- `data/raw/aviation_corpus.pkl` (46.93 MB)
- `data/documents/2026-02-25_Introduction_Aerospace_Engineering_Coda_v1.pdf` (31.07 MB)

Historical large blobs exist in git history (examples):

- `data/processed/aviation_corpus.json` (~191.43 MB)
- `data/astra_db/astra_db_content.json` (~71.83 MB)
- Multiple `data/embeddings/aviation_embeddings.json` snapshots (~57-71 MB)

Repository pack size:

- `size-pack: 557.99 MiB`

### Dependency Risk Status

Before fix: 12 vulnerabilities (2 critical, 9 high, 1 moderate).  
After `npm audit fix --omit=dev`: 6 vulnerabilities (0 critical, 4 high, 2 moderate).

Remaining blockers require major upgrades:

- `next` (recommended major update path points to 16.x)
- LangChain packages (`@langchain/*`, `langchain`)

## Validation

Executed successfully:

- `npm run sanitize:check`
- `npm run build`
- `npm run test:smoke`

## Pending Actions (Recommended)

1. Install `gitleaks` and `trufflehog` in CI for authoritative secret scanning.
2. Rotate production secrets as a precaution (`OPENAI`, `ASTRA`, `NEXTAUTH_SECRET`, auth password).
3. Decide on git history cleanup with `git filter-repo` for large generated artifacts.
4. Migrate large dataset/versioned binaries to object storage or Git LFS policy.
5. Plan major upgrades for `next` and LangChain dependencies in a dedicated compatibility branch.

## Step 2 Hardening Addendum (Started 2026-02-26)

1. Added persistent progress tracking in `WORKLOG.md`.
2. Added API hardening utility:
   - `src/utils/server/api_security.ts`
3. Applied per-route API rate limiting and input normalization:
   - `pages/api/chat/ask.ts`
   - `pages/api/chat/session.ts`
   - `pages/api/chat/session/[id].ts`
   - `pages/api/chat/history/[session_id].ts`
4. Replaced risky catch-all Vercel route rewrite with standard Next.js framework config in `vercel.json`.
5. Added CI pipeline:
   - `.github/workflows/ci.yml` runs sanitize checks, build, and Playwright smoke test.
6. Added authentication hardening:
   - In-memory credential attempt limiting.
   - Removed fallback credential email for production auth path.
   - Optional hashed password verification via `APP_AUTH_PASSWORD_HASH` (`sha256:<hex>`).
7. Added deployment-ready Python bridge split:
   - `AVIATION_API_MODE=worker|http`
   - `AVIATION_API_HTTP_URL`, `AVIATION_API_HTTP_TOKEN`
   - Health endpoint now reports bridge mode readiness.
