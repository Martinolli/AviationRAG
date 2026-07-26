# Data Governance

Date: 2026-05-12  
Status: Baseline policy  
Scope: Repository data handling, generated artifacts, and future governance direction

## Purpose

This document defines which AviationRAG data belongs in Git, which data must remain local or private, and how generated/runtime artifacts should be handled. The current policy is conservative because the repository is public-facing and aviation documents can be large, copyrighted, proprietary, controlled, or otherwise unsuitable for public distribution.

## Data Categories

| Category | Examples | Current Location | Git Policy |
| --- | --- | --- | --- |
| Source documents | Real `.pdf` and `.docx` aviation references | `data/documents/` | Do not commit by default. Keep local/private. |
| Sample documents | Tiny artificial examples for setup/tests | `data/sample_documents/` | May commit if non-sensitive and intentionally small. |
| Processed text | Extracted/normalized text, expanded text, chunk outputs | `data/processed/` | Do not commit. Generated from source documents. |
| Raw corpora | Pickle/JSON/CSV corpora | `data/raw/` | Do not commit. Generated/private. |
| Embeddings | Embedding JSON, vector files, FAISS inputs | `data/embeddings/` | Do not commit. Generated and potentially source-derived. |
| Astra exports | DB exports, consistency dumps | `data/astra_db/` | Do not commit. Generated and may expose private content. |
| Runtime chat/session state | Session metadata, local chat state | `chat_id/`, `chat/` | Do not commit. Runtime/user state. |
| Logs | App, bridge, ingestion, error, upload logs | `logs/`, `*.log` | Do not commit. Runtime diagnostics. |
| Generated visualizations | Plots, word clouds, analysis graphics | `assets/pictures/` | Do not commit unless deliberately sanitized and approved. |
| Configuration/env | `.env`, `.env.*`, tokens, production URLs | project root, Vercel env | Do not commit except `.env.example`. |
| Secure bundles | Astra Secure Connect bundles | `secure-connect-*.zip` | Do not commit. Treat as sensitive. |

## Git Policy

### Must Not Be Committed

1. `.env` or `.env.*` files, except `.env.example`.
2. API keys, bearer tokens, app auth secrets, password hashes, or production credentials.
3. Astra DB Secure Connect bundles.
4. Real/private/source aviation documents in `data/documents/`.
5. Processed text, corpora, embeddings, Astra exports, or generated chunks.
6. Runtime chat/session files under `chat_id/` or `chat/`.
7. Logs and local diagnostics.
8. Generated visualizations unless explicitly reviewed and approved.

### May Be Committed

1. Source code.
2. Documentation.
3. Sanitization guardrails.
4. `.env.example` with placeholders only.
5. Tiny non-sensitive sample files under `data/sample_documents/`.
6. Tests and fixtures that do not contain private, copyrighted, or controlled source material.

### Requires Git LFS or External Storage

Use Git LFS or external/private storage only after an explicit policy decision for:

1. Large source documents that must be versioned.
2. Large benchmark fixtures.
3. Large generated evaluation artifacts.
4. Approved non-private sample corpora that exceed normal repository size expectations.

For the current public repository posture, real aviation documents should generally stay outside Git and be provided through local/private storage.

## Current `.gitignore` Policy

The repository ignores runtime, generated, and private data folders to prevent accidental publication:

```text
chat_id/
chat/
logs/
data/documents/
data/raw/
data/processed/
data/embeddings/
data/astra_db/
assets/pictures/
.env
.env.*
!.env.example
secure-connect-*.zip
```

Rationale:

1. Source documents can be private, copyrighted, or large.
2. Processed corpora and embeddings can reproduce source content.
3. Chat/session state can contain user questions or sensitive operational context.
4. Logs can contain errors, paths, hostnames, request IDs, or accidental secrets.
5. Generated plots are reproducible outputs, not source-of-truth project assets.

## Sensitive Information Policy

Treat these values as sensitive:

1. `OPENAI_API_KEY`
2. `ASTRA_DB_APPLICATION_TOKEN`
3. `ASTRA_DB_SECURE_BUNDLE_PATH` and the bundle file itself
4. `ASTRA_DB_KEYSPACE` when paired with credentials
5. `NEXTAUTH_SECRET`
6. `APP_AUTH_PASSWORD`, `APP_AUTH_PASSWORD_HASH`
7. `AVIATION_API_HTTP_TOKEN`
8. Production bridge URLs when they imply private infrastructure
9. Vercel project tokens, deployment tokens, or provider credentials

Rules:

1. Store secrets in local `.env` or deployment environment variables only.
2. Never paste secrets into docs, logs, commits, screenshots, issues, or PR comments.
3. Rotate secrets after any suspected exposure.
4. Keep production password mode on `APP_AUTH_PASSWORD_HASH` rather than plaintext password where possible.
5. Use `.env.example` only for placeholder names and safe defaults.

## Public Repository Policy

The public repository should contain code, docs, tests, and tiny safe samples. It should not contain the real operating knowledge base.

Policy:

1. Real aviation documents should generally remain local/private.
2. Only tiny artificial or clearly redistributable sample data should be public.
3. Generated embeddings and corpora should not be public because they can contain source-derived content.
4. Any document proposed for public tracking must be reviewed for license, sensitivity, size, and operational risk.
5. If public benchmark examples are needed, prefer handcrafted minimal text snippets rather than full real documents.

## Runtime and Generated Artifacts

The following are runtime/generated artifacts and should remain ignored:

1. `chat_id/`
   - Local session metadata and indexes.
   - May reveal user workflows or session topics.
2. `chat/`
   - Local chat state and logs.
3. `logs/`
   - Runtime diagnostics from app, bridge, ingestion, and upload jobs.
4. `data/raw/`
   - Corpus files such as `aviation_corpus.pkl`.
5. `data/processed/`
   - Processed text, expanded text, chunk files, and converted corpus files.
6. `data/embeddings/`
   - Embedding JSON and vector data.
7. `data/astra_db/`
   - Astra DB content exports and consistency outputs.
8. `assets/pictures/`
   - Generated visualizations.
9. `.next/`, `.next_backup_*/`, `test-results/`, `playwright-report/`
   - Build and test outputs.

## Future Governance Direction

Planned controls:

1. Document manifest:
   - Stable `document_id`
   - Filename
   - Title
   - Authority
   - Document type
   - Revision/effective date
   - Source URL
   - File hash
   - Ingestion status
2. Document lifecycle:
   - `uploaded`
   - `processing`
   - `embedded`
   - `available`
   - `needs_review`
   - `retired`
   - `error`
3. Approval controls:
   - Uploaded documents should not automatically become trusted in production unless explicitly configured.
   - Admin approval should be required for controlled knowledge bases.
4. Versioning and retirement:
   - Superseded documents should be deactivated rather than silently overwritten.
   - Responses should warn when evidence comes from obsolete or low-quality sources.
5. Extraction quality:
   - Preserve extraction method and quality flags.
   - Route low-quality extraction to manual review.
6. Audit logging:
   - Store question, mode, retrieved chunk IDs, scores, answer, citations, model, prompt version, latency, user/session, and timestamp.
7. Feedback loop:
   - Capture wrong citation, hallucination, obsolete source, and quality feedback for future evaluation.

## Persistence Governance Addendum

D.6 defines the governance boundary between offline persisted-package validation
and any future migration execution.

Persisted-record status governance:

| Status | Controlled rehearsal | Production/indexing |
| --- | --- | --- |
| `valid` | eligible | future approval required |
| `valid_with_warnings` | eligible with approval | blocked pending governance |
| `review_required` | quarantine only | forbidden for indexing/retrieval |
| `rejected` | forbidden | forbidden |

Warning and limitation ownership is role-based. Parser extraction warnings are
owned by `parser_extraction_owner`; adapter/mapping warnings by
`aviationrag_ingestion_owner`; safety-content and table-classification
limitations by `domain_safety_reviewer`; dependency vulnerabilities by
`security_dependency_owner`; and migration authorization by
`migration_authority`.

Approval-scope rules:

1. Candidate-level approvals MUST NOT become document-global automatically.
2. Document-level approvals MUST NOT become corpus-global automatically.
3. Absence of approval fails closed.
4. Approval fixtures MUST NOT contain source text or personal names.

Partial provenance remains disabled by default. It MAY be considered only in a
future controlled pilot with explicit per-record limitation, owner approvals,
`review_required = true`, and indexing/retrieval exclusion. Unknown provenance
is forbidden.

OCR observations do not prove extraction failure, but they also do not establish
page completeness. OCR-affected pages require review or explicit exclusion
before production indexing. D.6 does not authorize OCR execution.

Legacy coexistence for the next phase is shadow mode only: structured records
and legacy chunks remain separate, legacy deletion is forbidden, silent origin
merging is forbidden, and rollback material must be retained.

Controlled rehearsal retention is conservative: no automatic deletion, previous
packages retained until replacement validation succeeds, and production
retention duration remains unresolved.

The reported dependency findings remain a production security gate. They do not
invalidate offline deterministic evidence, but production persistence, indexing,
retrieval, or deployment requires separate security review.

## Controlled Shadow Migration Rehearsal Addendum

D.7 shadow-store origin rules:

1. Structured package records retain `new_structured` origin and remain separate
   from legacy records.
2. Legacy inventory records use `legacy_processed`, `legacy_chunked`, or
   `legacy_unresolved`; filename-only legacy evidence remains
   `legacy_filename_only`.
3. No filename-only or title-only evidence may establish exact identity.
4. Exact source identity requires SHA-256 source checksum match.

Quarantine storage:

1. `review_required` records are written only to quarantine.
2. `valid_with_warnings` records quarantine by default without scoped approval.
3. Quarantine records are not indexing-eligible or retrieval-eligible.
4. Quarantine is accounted for but is not rejection.

Reconciliation records must preserve structured document ID, source filename,
source checksum, reconciliation status, matched legacy keys, exact checksum
match count, filename match count, review requirement, cutover eligibility, and
warning codes.

Legacy snapshot and rollback evidence must record file counts, sizes, and
SHA-256 checksums before and after rehearsal. No destructive overwrite, legacy
deletion, silent origin merge, or legacy chunk mutation is allowed.
