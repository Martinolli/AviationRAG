# AviationRAG Web App Requirements (V1)

## 1) Goal

Build a clean, professional web interface for aviation-domain RAG chat that gives grounded, document-based answers with citations and supports safe growth to uploads and web research.

## 2) Product Scope

### 2.1 V1 (Must Have)

- User login to your app.
- Left sidebar with conversation list and actions.
- Main chat area with clean message layout.
- Source citations in each answer.
- Strict mode for standards/regulatory questions.
- Session history and rename/delete chat.
- Basic settings page.

### 2.2 V1.1 (Next)

- Upload PDF/DOCX from UI and trigger ingestion pipeline.
- Show ingestion status/progress.
- Source viewer panel for cited passages.

### 2.3 V2 (Optional)

- Web research mode (explicit toggle, with external citations).
- Team/shared workspaces.
- Advanced analytics dashboard.

## 3) Users and Access

### 3.1 Primary User Types

- `Engineer`: asks technical questions, needs direct citations.
- `Admin`: manages documents, model settings, and system health.

### 3.2 Authentication (Recommended)

- Use app authentication (email/password or Microsoft/Google).
- Do not require users to log in with OpenAI account directly.

## 4) Credentials Strategy (Important Decision)

### Option A (Recommended for V1)

- Backend uses project-managed OpenAI credentials.
- Users log in only to your app.
- Pros: easier, safer, faster to launch.
- Cons: shared project billing.

### Option B (Later, Optional)

- BYOK: each user can store their own OpenAI API key in profile settings.
- Encrypt keys at rest, never show plain key again after save.
- Pros: per-user billing/control.
- Cons: more security complexity.

Decision for V1:

- `Use Option A` (project-managed credentials).

## 5) UX Requirements

### 5.1 Layout

- Left sidebar:
  - New Chat button.
  - Conversation search.
  - Conversation list with last updated time.
  - Quick filters (Pinned, Recent).
- Main panel:
  - Messages with role separation.
  - Composer with send/stop buttons.
  - Citations under each answer.
- Right panel (toggle):
  - “Sources used” list with filename + chunk id.

### 5.2 Visual Style

- Clean spacing, high readability.
- Professional aviation theme (neutral palette, not flashy).
- Fast interactions, minimal visual noise.

### 5.3 Accessibility

- Keyboard-friendly navigation.
- Good contrast and readable typography.
- Responsive behavior for laptop + mobile.

## 6) AI Behavior Requirements

### 6.1 Response Policy

- If query is document-specific (“according to X”), switch to strict grounded mode.
- For standards/regulations (Part 23/25, FAA Orders, AC, EASA CS), prioritize exact wording and direct evidence.
- If evidence is missing, say: “Not found in provided sources.”

### 6.2 Citation Policy

- Every factual section must contain citations:
  - `[filename | chunk_id]`
- No uncited hard claims in strict mode.

### 6.3 Hallucination Controls

- Lower temperature in strict mode.
- Prefer extractive answer style over generic explanation for compliance queries.

## 7) Upload Requirements (V1.1)

- Accept `.pdf` and `.docx`.
- Validate file type and max size before upload.
- Save upload event to audit log.
- Show ingestion status:
  - `uploaded` -> `processing` -> `embedded` -> `available`.
- If extraction quality is low, mark as `needs review`.

## 8) Web Research Requirements (V2)

- Must be off by default.
- User must explicitly enable for a query.
- External results must be clearly labeled as “Web Sources”.
- Keep internal document citations separated from web citations.

## 9) API Contract (Draft)

- `POST /api/chat/ask`
  - Input: `session_id`, `message`, `strict_mode?`, `target_document?`
  - Output: `answer`, `citations[]`, `used_mode`, `latency_ms`
- `GET /api/chat/history/{session_id}`
- `POST /api/chat/session`
- `PATCH /api/chat/session/{id}`
- `DELETE /api/chat/session/{id}`
- `POST /api/documents/upload` (V1.1)
- `GET /api/documents/status/{id}` (V1.1)
- `GET /api/health`

## 10) Non-Functional Requirements

- Average response target: <= 6 seconds for normal queries.
- Strict mode target: <= 10 seconds.
- Error handling:
  - Friendly message to user.
  - Full technical details in logs.
- Security:
  - Secrets in environment variables only.
  - No secret in frontend code.

## 11) Acceptance Criteria (V1)

- User can create and continue conversations.
- Answers include citations when factual claims are made.
- Document-specific query returns document-grounded answer.
- Regulatory query returns direct/cited response (not generic).
- Chat logs are stored and retrievable.
- UI works on desktop and mobile.

## 12) Phased Delivery Plan

1. `Phase 1`: API hardening + response policy + citations.
2. `Phase 2`: Web UI shell (sidebar/chat/settings).
3. `Phase 3`: Upload and ingestion tracking.
4. `Phase 4`: Optional web research mode.

## 13) Decisions Needed From You

Please confirm these five items before implementation:

1. App auth method:
   - Email/password or Google/Microsoft SSO.
2. V1 credential mode:
   - Project-managed OpenAI key (recommended).
3. V1 pages:
   - Chat, Settings, Admin (yes/no for Admin in V1).
4. Upload in V1 or V1.1:
   - Recommended: V1.1.
5. UI stack:
   - Keep current stack (Next.js) and add API routes first.
