# Reset, Rebuild, and Retrieval Evaluation Baseline

Date: 2026-05-14  
Status: Planning and safety gate only  
Scope: Future controlled reset/rebuild and retrieval baseline requirements

## 1. Purpose

This document defines when and how AviationRAG should perform a controlled reset and rebuild after manifest-driven ingestion and metadata-aware chunks are ready.

The reset/rebuild must not begin until the project has a retrieval evaluation baseline, a documented rollback path, and explicit user approval for the reset window. This document does not execute a reset, reprocess documents, regenerate embeddings, modify Astra, modify FAISS, or change runtime behavior.

## 2. Scope

Included:

1. Astra DB vector data and related vector payload metadata.
2. Local embeddings and embedding-derived files.
3. Local FAISS indexes and FAISS metadata files.
4. Processed text, raw corpus files, and chunked outputs.
5. Future manifest-driven ingestion outputs.
6. Retrieval evaluation baseline requirements before and after rebuild.

Excluded:

1. Actual reset execution.
2. Real document reprocessing.
3. Embedding regeneration.
4. Astra DB table or collection deletion.
5. FAISS index deletion.
6. Production deployment changes.
7. Runtime API, bridge, prompt, retrieval, or ingestion behavior changes.

## 3. Current Status

Current planning state:

1. The document manifest schema exists in `docs/DOCUMENT_MANIFEST_SCHEMA.md`.
2. Fake manifest and chunk fixtures exist under `data/sample_documents/`.
3. Manifest JSONL utilities exist in `src/aviationrag/ingestion/manifest.py`.
4. Legacy compatibility adapter coverage exists in `src/aviationrag/ingestion/legacy_adapter.py`.
5. Fake-data dry-run coverage exists in `src/aviationrag/ingestion/dry_run.py`.
6. A local-only sample manifest writer exists for fake records only.
7. Gated manifest settings exist and are disabled by default.
8. Real ingestion integration is not active.
9. Real `data/manifest/documents.jsonl` is not committed and must remain local/private when created.
10. Retrieval harness and report/export shells exist for fake/mock results only; real retrieval evaluation is not wired yet.
11. The future chunk metadata schema is documented in `docs/CHUNK_METADATA_SCHEMA.md`.
12. Real metadata-rich chunk migration is not implemented.
13. A fake/sample-only chunk payload exporter exists for future vector payload-shaped dictionaries; it does not generate embeddings or write to Astra/FAISS.
14. The real chunk migration design is documented in `docs/REAL_CHUNK_MIGRATION_DESIGN.md`; it is planning-only and no migration has occurred.
15. A fake/local chunk migration dry-run tool exists for summary-only rehearsal reports under ignored `logs/`; it does not write migrated chunks, generate embeddings, or touch Astra/FAISS.
16. A gated local chunk conversion writer exists for explicitly allowed ignored outputs under `data/migration_dry_run/`; it is not production migration and does not generate embeddings or touch Astra/FAISS.
17. Response policy and citation validation are not enforced.
18. Production bridge remains an external blocker.
19. Dependency major upgrades remain future work after Security Sprint S.1.

## 4. Reset/Rebuild Triggers

A controlled reset/rebuild becomes appropriate when one or more of these conditions are true:

1. Manifest integration becomes active for real ingestion.
2. Chunk metadata schema changes.
3. Vector payload schema changes.
4. `document_id` and `chunk_id` traceability becomes mandatory for citations and audit logs.
5. Current Astra vectors lack required manifest or chunk metadata.
6. Retrieval evaluation baseline is ready.
7. Old embeddings no longer match the current chunking strategy.
8. FAISS metadata and Astra payloads no longer share a reliable chunk identity model.
9. Approval/lifecycle rules require filtering active, retired, superseded, or unapproved documents.

The target chunk metadata contract that should be ready before any schema-changing rebuild is documented in `docs/CHUNK_METADATA_SCHEMA.md`.
The future real chunk migration path and go/no-go criteria are documented in `docs/REAL_CHUNK_MIGRATION_DESIGN.md`.

A reset should not be used as a shortcut for unclear metadata. If the target schema, evaluation baseline, or rollback path is incomplete, the decision must remain `No-Go`.

## 5. Pre-Reset Checklist

Before any future reset:

1. Git working tree is clean.
2. Latest `main` is pulled and verified.
3. `npm run sanitize:check:all` passes.
4. `npm run build` passes.
5. Lightweight Python model/manifest tests pass.
6. Retrieval evaluation smoke fixture exists.
7. Current retrieval baseline has been run and saved locally.
8. Real source documents are backed up outside Git.
9. Current local generated artifacts are backed up if they may be needed for rollback.
10. Astra target database, keyspace, collection, or table names are confirmed.
11. Secrets are available locally and stored securely.
12. Manifest integration settings are reviewed.
13. Public/private data boundary is confirmed.
14. Reset window is approved by the user/operator.
15. Rollback path is documented for both local files and Astra data.

## 6. Backup/Export Checklist

Backups must remain outside Git unless they are fake, tiny, and explicitly approved.

Backup/export candidates:

1. Export existing Astra content if it may be useful for rollback, comparison, or audit.
2. Copy `data/raw/` if current corpus files are needed.
3. Copy `data/processed/` if current extracted text or chunks are needed.
4. Copy `data/embeddings/` if current embeddings are needed.
5. Copy existing FAISS index artifacts and metadata files if present.
6. Copy `data/astra_db/` exports if they contain useful local snapshots.
7. Save current `.env` securely outside Git.
8. Save current retrieval evaluation outputs if any exist.
9. Record package, schema, and prompt versions used for the pre-reset baseline.

Never commit backup files that include private document names, extracted text, vectors, credentials, chat history, logs, or real manifest metadata.

## 7. Local Artifact Cleanup Plan

Likely ignored local paths involved in a future cleanup/rebuild:

1. `data/raw/`
2. `data/processed/`
3. `data/embeddings/`
4. `data/astra_db/`
5. `data/manifest/`
6. `assets/pictures/`
7. Local FAISS index artifacts such as `*.index` and matching metadata sidecars.
8. Runtime logs under `logs/`.

Cleanup rules:

1. Do not delete artifacts until backups and reset approval are complete.
2. Prefer moving artifacts to a timestamped local backup folder before deletion.
3. Keep backup folders outside Git-tracked paths when possible.
4. Recreate ignored generated directories only through the approved rebuild sequence.
5. Confirm `git status --short` before and after cleanup to ensure no generated data is staged.

## 8. Astra DB Reset Plan

Future Astra reset sequence:

1. Identify the exact database, keyspace, collection, and table names.
2. Confirm that credentials point to the intended environment.
3. Export current vector/chat content if useful.
4. Stop ingestion and chat jobs that may write to Astra.
5. Confirm the new vector payload schema includes traceability fields.
6. Drop, recreate, or truncate only the approved vector collections/tables.
7. Keep chat/session storage separate unless a chat data reset is explicitly approved.
8. Reinsert vectors only after manifest records and metadata-rich chunks are ready.
9. Run Astra consistency checks after reinsertion.
10. Record reset timestamp, schema version, vector count, and operator decision.

No real credentials should appear in documentation, logs, commits, or issues.

## 9. FAISS/Index Reset Plan

Future FAISS reset sequence:

1. Locate existing FAISS index artifacts and metadata sidecars.
2. Back up old index artifacts if rollback may be required.
3. Remove old indexes only after approval.
4. Rebuild indexes from regenerated embeddings that include the new chunk metadata contract.
5. Verify every FAISS metadata record includes `document_id`, `chunk_id`, filename/title, and source traceability fields where available.
6. Verify FAISS chunk IDs match the same chunk IDs inserted into Astra.
7. Run retrieval evaluation after rebuild.

FAISS and Astra should not be allowed to diverge on chunk identity after the reset.

## 10. Re-Ingestion/Rebuild Sequence

Future controlled sequence:

1. Enable gated manifest integration in dry-run mode.
2. Run fake-data and local validation.
3. Enable manifest integration locally.
4. Generate the document manifest.
5. Extract documents.
6. Generate metadata-rich chunks.
7. Generate embeddings.
8. Insert vectors into Astra with the new payload schema.
9. Rebuild FAISS.
10. Run retrieval evaluation baseline.
11. Validate citations and traceability manually on a small sample.
12. Compare post-rebuild retrieval metrics against pre-reset baseline.
13. Only then consider production deployment changes.

Likely future pipeline stages, not to be executed in this phase:

```text
read documents -> manifest update -> extract text -> chunk text
-> generate embeddings -> insert Astra vectors -> rebuild FAISS
-> run retrieval evaluation -> validate citations/traceability
```

## 11. Retrieval Evaluation Baseline Requirements

A fake/sample-only smoke fixture now exists at `data/sample_documents/sample_retrieval_eval.jsonl`, with validation utilities in `src/aviationrag/evaluation/smoke_fixture.py`. A fake/mock retrieval harness shell exists at `src/aviationrag/evaluation/retrieval_harness.py`, and report/export helpers exist at `src/aviationrag/evaluation/reporting.py`. These utilities define, test, and format the future evaluation shape only; they do not execute real retrieval and are not a substitute for the real pre-reset baseline.

Minimum smoke baseline before reset:

| Category | Initial minimum |
| --- | ---: |
| Regulatory/compliance questions | 5 |
| Document-specific questions | 5 |
| Manufacturing/design questions | 5 |
| SMS/safety questions | 5 |
| Accident analysis questions | 5 |
| Insufficient-evidence questions | 5 |

Later benchmark target:

1. 20 to 50 questions per category.
2. Expected document IDs where possible.
3. Expected page, section, paragraph, or table identifiers where available.
4. Negative examples whose answer should not be found in the controlled source set.

Required metrics:

1. Top-1 hit.
2. Top-3 hit.
3. Top-5 hit.
4. Mean reciprocal rank when practical.
5. Expected document match.
6. Expected section/page match when available.
7. Citation traceability.
8. Not-found correctness.
9. Low-quality extraction exposure rate.
10. Duplicate or near-duplicate chunk rate.

Evaluation outputs should be saved locally as JSON/Markdown and should not be committed if they expose private source names, document text, or operational metadata.

## 12. Acceptance Criteria

A future reset/rebuild is accepted only when:

1. No private data is committed.
2. Manifest records are generated for all active source documents.
3. Chunks have stable `document_id` and `chunk_id`.
4. Astra vector payloads include traceability metadata.
5. FAISS metadata aligns with the same chunk IDs used in Astra.
6. Retrieval evaluation baseline is run before and after rebuild.
7. Evaluation results are saved locally and reviewed.
8. Unsupported or insufficient-evidence queries are handled safely.
9. Sample citations can be traced to document, chunk, filename/title, page/section where available, and source hash.
10. Rollback artifacts are retained until the rebuilt system is accepted.

## 13. Rollback/Recovery Plan

Rollback options:

1. Disable manifest integration flags.
2. Restore previous local generated artifacts from backup.
3. Restore old FAISS indexes and metadata sidecars if they were backed up.
4. Restore prior Astra collection/table from export or switch back to the prior collection/table name.
5. Rebuild from previous local artifacts if embeddings and chunks were preserved.
6. Keep old reset snapshots outside Git until the new system is accepted.
7. Record the rollback reason and validation result in `WORKLOG.md`.

Rollback should be chosen immediately if post-reset retrieval evaluation shows material regression, chunk/vector counts do not align, or traceability metadata is incomplete.

## 14. Operational Risks

| Risk | Impact | Control |
| --- | --- | --- |
| Accidental deletion | Loss of local generated corpus, embeddings, or indexes. | Backup first, require explicit reset approval, verify target paths. |
| Private data leakage | Source names, text, vectors, or metadata committed to Git. | Keep real data under ignored paths, run sanitization checks, inspect staged files. |
| Schema mismatch | Astra, FAISS, manifest, and chunks disagree on IDs. | Define schema before reset, validate counts and IDs after rebuild. |
| Embedding cost | Re-embedding may incur cost and long runtime. | Estimate chunk counts and cost before reset. |
| Long runtime | Full ingestion may take hours or fail mid-run. | Use a reset window, checkpoints, logs, and staged validation. |
| Incomplete PDF extraction | Poor chunks or missing pages can pollute retrieval. | Preserve extraction quality and manual-review flags. |
| Citation mismatch | Answers may cite chunks that do not support claims. | Add citation validation and manual traceability review. |
| Evaluation false confidence | Small benchmark may miss regressions. | Start with smoke set, then expand to 20 to 50 questions per category. |
| Production disruption | Reset could affect live retrieval. | Keep reset local until approved; do not deploy before evaluation gates pass. |

## 15. Go/No-Go Decision Checklist

Use this checklist before any future reset execution:

- [ ] Reset scope is documented.
- [ ] Git working tree is clean.
- [ ] Latest `main` is pulled.
- [ ] Source documents are backed up locally/private.
- [ ] Existing generated artifacts are backed up if rollback is required.
- [ ] Astra target database/keyspace/collection/table is confirmed.
- [ ] Astra export is completed or explicitly waived.
- [ ] Manifest schema and vector payload schema are finalized for this reset.
- [ ] Gated manifest integration has passed dry-run validation.
- [ ] Retrieval evaluation smoke set exists.
- [ ] Pre-reset retrieval baseline has been run.
- [ ] Estimated embedding cost and runtime are accepted.
- [ ] Reset window is approved by the user/operator.
- [ ] Rollback plan is accepted.
- [ ] No private or generated data is staged for Git.

Decision:

- [ ] Go
- [ ] No-Go

## 16. Future Implementation Phases

| Phase | Scope | Runtime impact |
| --- | --- | --- |
| D.1i | Reset/rebuild and retrieval baseline plan only. | None. |
| D.2 | Metadata-rich chunk schema planning. | None. |
| D.2b | Retrieval evaluation smoke fixture. | None to runtime. |
| D.3 | Real chunk migration design. | Completed as planning only; no runtime migration. |
| D.3b | Read-only legacy chunk audit. | Completed with fake/default fixture and explicit-file support only. |
| D.3c | Fake/local chunk migration dry run. | Completed for local rehearsal only; no embeddings, Astra, or FAISS. |
| D.3d | Gated local chunk conversion writer. | Completed for ignored local outputs only; disabled by default. |
| D.4 | Page and structure preservation design. | Planned design only; no document reprocessing or migration. |
| D.5 | Controlled local manifest/chunk integration prototype. | Future gated local work only; disabled by default. |
| D.6 | Controlled local reset/rebuild. | Future work requiring explicit reset window. |
| E.1 | Retrieval evaluation harness shell. | Fake/mock result scoring only; no real retrieval wiring. |
| E.2 | Retrieval report/export shell. | Formats fake/mock evaluation results only; no real retrieval wiring. |
| E.3 | Real retrieval evaluation integration. | Measures retrieval before behavior changes. |
| F.1 | Response and citation validation. | Adds policy enforcement after evidence quality is measurable. |

Actual reset/rebuild work must be performed in a later approved phase and should remain separate from dependency upgrades, prompt changes, retrieval algorithm changes, and deployment bridge changes.
