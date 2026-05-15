# Ingestion Manifest Integration Plan

Date: 2026-05-13  
Status: Planning and dry-run design only  
Scope: Future manifest-aware ingestion integration; no runtime integration

## 1. Purpose

This plan defines how the new manifest utilities should later be integrated into AviationRAG ingestion without breaking current runtime behavior.

The current phase does not modify ingestion scripts, reprocess documents, regenerate embeddings, reset Astra or FAISS, or write real manifest files. It records the intended integration path, dry-run safety rules, reset strategy, and validation gates for a later controlled migration.

The detailed reset/rebuild and retrieval evaluation gate is tracked separately in `docs/RESET_REBUILD_AND_EVALUATION_BASELINE.md`.

## 2. Current Legacy Ingestion Flow

The current ingestion pipeline remains script-based. Conceptually, it follows this flow:

1. Read source documents from the configured local documents directory.
2. Extract text from supported PDF and DOCX files.
3. Save raw corpus data and extracted/processed text outputs under ignored generated data folders.
4. Chunk document text into JSON chunk files.
5. Generate embeddings from chunk records.
6. Store or compare embeddings locally and in Astra helper scripts.
7. Load generated embedding JSON into FAISS for runtime retrieval.

Confirmed legacy anchors:

1. `src/scripts/py_files/config.py` defines `data/documents/`, `data/raw/`, `data/processed/`, `data/embeddings/`, and `data/astra_db/` paths.
2. `src/scripts/py_files/read_documents.py` creates document-like dictionaries with fields such as `filename`, `text`, `tokens`, `section_references`, `metadata`, and `category`.
3. `src/scripts/py_files/aviation_chunk_saver.py` writes chunk JSON containing `filename`, document `metadata`, `category`, and per-chunk `chunk_id`, `text`, and `tokens`.
4. Astra storage helpers currently use payload fields such as `chunk_id`, `filename`, `text`, `tokens`, and `embedding`.
5. Runtime retrieval still loads generated embeddings into FAISS and uses legacy metadata structures.

This plan does not claim that every script always executes in the exact order above; it describes the current conceptual data flow and the observed integration surfaces.

## 3. Proposed Manifest-Aware Ingestion Flow

Future manifest-aware ingestion should preserve the existing pipeline until a controlled migration is approved. The target flow is:

1. Discover a source document.
2. Compute a source file hash.
3. Create or update a `DocumentRecord`.
4. Mark the document as `processing`.
5. Extract text.
6. Record extraction method, quality, warnings, page counts, and manual-review signals.
7. Mark the document as `extracted`.
8. Chunk extracted text.
9. Create `ChunkRecord` metadata with document-level inherited fields.
10. Mark the document as `chunked`.
11. Embed chunks.
12. Store vector payloads with `document_id` and `chunk_id`.
13. Mark the document as `embedded`.
14. Insert vectors into Astra and rebuild or refresh FAISS as appropriate.
15. Mark the document as `available` only when lifecycle and approval rules allow it.

The current code does not perform this flow yet.

## 4. Integration Hook Points

Planned future hook points:

| Hook point | Future purpose | Current status |
| --- | --- | --- |
| After document discovery | Normalize filename, compute file hash, create or update `DocumentRecord` as `discovered` or `processing`. | Planned, not implemented. |
| After extraction | Store extraction method, quality, warnings, review flag, and text availability. | Planned, not implemented. |
| After chunking | Convert chunk outputs into `ChunkRecord` metadata and attach parent `document_id`. | Planned, not implemented. |
| After embedding generation | Mark chunks as embeddable and ensure vector payloads include `document_id` and `chunk_id`. | Planned, not implemented. |
| After Astra insertion | Mark document as `embedded` or `available` depending on approval policy. | Planned, not implemented. |
| After FAISS rebuild | Record index build metadata in a future audit or manifest history field. | Planned, not implemented. |

The first real integration should be gated and reversible. It should write sidecar manifest data without changing retrieval behavior.

## 5. DocumentRecord Mapping Plan

The future integration should map legacy document-like dictionaries into `DocumentRecord` as follows:

| Legacy-like field | Target field | Notes |
| --- | --- | --- |
| `filename` | `filename` | Normalize path separators and preserve the basename. |
| PDF/DOCX metadata title or `title` | `title` / schema `canonical_title` | Prefer extracted title when reliable; otherwise derive from filename. |
| Inferred or explicit authority | `authority` | Use controlled values from the manifest schema. |
| Inferred or classified category | `document_type` | Map legacy `category` into controlled document types where possible. |
| Revision-like filename or metadata token | `revision` | Preserve uncertain detections in `metadata` until reviewed. |
| Effective date-like filename or metadata token | `effective_date` | Use ISO format only when confidence is acceptable. |
| Source URL or source URI | `source_url` | Real local paths should not be committed. |
| Computed SHA-256 | `file_hash` | Required for duplicate detection and traceability. |
| Pipeline status | `ingestion_status` | Use lifecycle states from the manifest schema. |
| Extraction fields | `metadata` | Preserve `source_type`, `extraction_method`, `extraction_quality`, `needs_manual_review`, and warnings. |

The existing `legacy_adapter.py` already provides fake-data-tested conversion helpers. Future ingestion integration should reuse or extend those helpers behind a gate rather than duplicating conversion rules inside legacy scripts.

## 6. ChunkRecord Mapping Plan

The future integration should map chunk-like dictionaries into `ChunkRecord` as follows:

| Legacy-like field | Target field | Notes |
| --- | --- | --- |
| `chunk_id` | `chunk_id` | Preserve existing IDs during compatibility migration; introduce deterministic IDs only during a controlled reset. |
| Parent document manifest record | `document_id` | Required join key for traceability. |
| `filename` | `filename` | Used for backward-compatible display and legacy joins. |
| `text` | `text` | Preserve chunk text exactly as embedded for traceability. |
| `page` / `page_start` / `page_end` | `page_start` / `page_end` | Current chunker may not preserve page spans; future parsing should. |
| Section references or headings | `section_path` | Use an empty list until section-aware parsing exists. |
| `tokens`, `chunk_type`, extraction fields | `metadata` | Preserve legacy fields without changing runtime payloads. |
| Inherited document fields | `metadata` or future vector payload | Include authority, document type, revision, effective date, extraction quality, and source hash when available. |

Future vector records should include both `document_id` and `chunk_id`. Until the reset phase, existing vector payload shape should remain unchanged.

The detailed real chunk migration design is tracked in `docs/REAL_CHUNK_MIGRATION_DESIGN.md`. That document defines the future mapping, chunk ID policy, evaluation gates, and reset dependency; it does not implement runtime ingestion integration.

## 7. Manifest Lifecycle Write Points

Target normal lifecycle:

```text
discovered -> processing -> extracted -> chunked -> embedded -> available
```

Write-point guidance:

| Status | Future write point |
| --- | --- |
| `discovered` | A candidate source is identified before extraction. |
| `processing` | Extraction starts for a discovered or uploaded document. |
| `extracted` | Text extraction succeeds and extraction metadata is available. |
| `chunked` | Chunk JSON or chunk records are created. |
| `embedded` | Embeddings are generated and stored locally or remotely. |
| `available` | Document is eligible for retrieval under lifecycle and approval policy. |
| `needs_review` | Extraction quality, metadata uncertainty, scanned pages, or policy flags require human review. |
| `error` | Extraction, chunking, embedding, or storage fails. |
| `retired` | A document is intentionally removed from normal retrieval. |
| `superseded` | A newer revision replaces the document, but historical traceability remains. |

Approval status should remain separate from ingestion status. A document can be technically `embedded` while still `pending_review`.

## 8. Local and Private Data Handling

The future real local/private manifest path is:

```text
data/manifest/documents.jsonl
```

Governance rules:

1. `data/manifest/` must remain ignored from Git when it contains real document metadata.
2. Committed fixtures stay under `data/sample_documents/`.
3. Tests must use fake fixtures or temporary directories.
4. Real/private filenames, source URLs, local paths, internal titles, proprietary manuals, and source-derived text must not be committed.
5. No integration test should scan `data/documents/` unless a future task explicitly approves local-only migration work.

## 9. Reset and Rebuild Strategy for Astra and FAISS

### Safe Point for Full Index/Database Reset

The detailed go/no-go checklist and retrieval baseline requirements are defined in `docs/RESET_REBUILD_AND_EVALUATION_BASELINE.md`.

A full Astra, FAISS, and embedding reset is recommended when:

1. Manifest-driven ingestion becomes active.
2. Chunk metadata schema changes.
3. Vector payload schema changes.
4. `document_id` and `chunk_id` traceability is required end to end.
5. Retrieval evaluation baseline is ready.
6. Admin approval and document lifecycle behavior are defined for production use.

Future controlled reset sequence:

1. Backup or export useful current data if needed.
2. Stop ingestion and chat jobs.
3. Drop or recreate Astra vector collections or tables.
4. Remove local generated embeddings and index artifacts.
5. Re-run manifest-driven ingestion.
6. Re-chunk with new metadata.
7. Re-embed chunks.
8. Reinsert vectors into Astra.
9. Rebuild the FAISS index.
10. Run the retrieval evaluation baseline.
11. Record reset inputs, schema versions, and evaluation results.

This reset is not part of Phase D.1e.

## 10. Migration Phases

Recommended staged migration:

| Phase | Scope | Runtime impact |
| --- | --- | --- |
| D.1e | Integration plan only. | None. |
| D.1f | Fake-data dry-run adapter coverage. | None. |
| D.1g | Optional local-only manifest write dry run using temporary or ignored paths. | None to production. |
| D.1h | Gated integration behind an environment flag. | Disabled by default. |
| D.1i | Full reset/rebuild and retrieval evaluation baseline. | Controlled migration window required. |

Real ingestion integration should not be mixed with chunking redesign, prompt changes, response policy enforcement, or retrieval behavior changes.

## 11. Fake-Data Dry-Run Coverage

A side-effect-free dry-run helper exists at `src/aviationrag/ingestion/dry_run.py`.

Current scope:

1. Simulate manifest-aware ingestion planning from fake in-memory legacy-like dictionaries.
2. Convert fake document and chunk dictionaries into `DocumentRecord` and `ChunkRecord` objects.
3. Link chunks to documents by normalized filename where possible.
4. Report duplicate document IDs, unknown chunk document references, empty chunk text, and manifest validation issues.

Current limitations:

1. The helper does not write files.
2. It does not scan `data/documents/` or read private local documents.
3. It does not call legacy ingestion scripts, embedding APIs, Astra, or FAISS.
4. It is not integrated with legacy ingestion yet.
5. It is intended to reduce risk before future gated integration.

## 12. Local-Only Manifest Write Dry Run

A manual developer utility exists at `tools/manifest/write-local-sample-manifest.py`.

Current scope:

1. Load fake records from `data/sample_documents/sample_manifest.jsonl`.
2. Write them to the ignored local path `data/manifest/documents.jsonl`.
3. Read the local file back through the manifest utility.
4. Validate record count and required manifest fields.
5. Confirm the generated output path is ignored by Git.

Current limitations:

1. The generated manifest is local-only and must not be committed.
2. The utility refuses to overwrite an existing local manifest unless `--force` is supplied.
3. It does not scan real documents, call legacy ingestion scripts, generate embeddings, access Astra, or build FAISS indexes.
4. It is not runtime ingestion integration.

## 13. Gated Manifest Integration Controls

Future manifest integration settings are defined in `src/aviationrag/config.py`.

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `AVIATIONRAG_ENABLE_MANIFEST_INTEGRATION` | `false` | Future gate for manifest-aware ingestion writes. |
| `AVIATIONRAG_MANIFEST_DRY_RUN` | `false` | Future gate for dry-run behavior when integration is enabled. |
| `AVIATIONRAG_MANIFEST_PATH` | `data/manifest/documents.jsonl` | Local/private manifest path override. |
| `AVIATIONRAG_ENABLE_CHUNK_MIGRATION` | `false` | Future gate for legacy chunk migration utilities. |
| `AVIATIONRAG_CHUNK_MIGRATION_DRY_RUN` | `true` | Future dry-run default for legacy chunk migration previews. |

Current scope:

1. The flags are parsed by side-effect-free package config helpers.
2. Manifest integration remains disabled by default.
3. The default manifest path is local/private and ignored by Git.
4. The flags are not wired into legacy ingestion scripts in this phase.
5. A future integration phase should require these settings before writing manifest records from real ingestion.

A passive legacy chunk adapter now exists at `src/aviationrag/ingestion/chunk_legacy_adapter.py`. It converts fake legacy-like chunk dictionaries into `ChunkRecord` objects and optional vector payload-shaped dictionaries for preview purposes only. It is tested with fake data and is not wired into runtime ingestion, embeddings, Astra, FAISS, API routes, prompts, or deployment behavior.

## 14. Validation Plan

Validation before real integration:

1. Unit tests for model serialization and manifest JSONL utilities.
2. Fake fixture tests for sample manifest and sample chunks.
3. Fake legacy adapter tests for document and chunk conversion.
4. Dry-run tests that prove no files are written unless an explicit temporary path is supplied.
5. Config tests for disabled-by-default manifest integration flags.
6. Sanitization checks confirming no private data, generated data, embeddings, or env files are staged.
7. Compile checks for the Python package.
8. Build checks for the web app.

Validation before reset/rebuild:

1. Retrieval evaluation harness exists.
2. Baseline retrieval metrics are recorded against the current system.
3. New manifest/vector payload schema is documented.
4. Rollback path is documented and tested.
5. Operator confirms whether current Astra data should be backed up.

Validation after reset/rebuild:

1. Manifest record count matches source document count.
2. Chunk record count matches embedded vector count.
3. Every vector has `document_id` and `chunk_id`.
4. Sample citations can trace to document, chunk, page or section where available, source hash, and ingestion timestamp.
5. Retrieval evaluation results are compared against baseline.

## 15. Rollback Plan

Future manifest integration should be reversible:

1. Keep legacy PKL, JSON, chunk, embedding, Astra, and FAISS flows available until migration is accepted.
2. Gate manifest writing behind an environment flag during first integration.
3. If the manifest path causes errors, disable the flag and continue legacy ingestion.
4. If vector payload migration causes retrieval regressions, restore the prior embedding/index artifacts or restore Astra from backup.
5. Preserve old document and chunk identifiers during compatibility phases unless a full reset has been approved.
6. Do not delete superseded manifest records silently; mark them retired or superseded.

## 16. Open Questions

1. What should the final Astra vector schema be?
2. What metadata payload should every chunk/vector carry?
3. Should production `document_id` values be deterministic or UUID-based?
4. Should approval be mandatory before retrieval in every deployment mode?
5. Where should admin approval state live: local manifest, Astra, SQLite, or a separate application table?
6. Should the first integration preserve existing chunk IDs or regenerate them during the reset?
7. How should scanned PDFs and OCR-only documents be represented before manual review?
8. What is the minimum retrieval evaluation baseline required before resetting Astra and FAISS?
9. How should manifest history be versioned if metadata is corrected after ingestion?
10. Which runtime paths should be wrapped first when the migration begins?
