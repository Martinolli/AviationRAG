# Document Manifest Schema

Date: 2026-05-13  
Status: Planning baseline only  
Scope: Future document manifest and metadata model; no runtime migration

## 1. Purpose

The document manifest is intended to make every AviationRAG source document traceable, auditable, version-aware, and suitable for future compliance-grade retrieval.

The manifest should answer these questions for every document:

1. What source file or source URI did this text come from?
2. Which authority, document type, revision, and effective date apply?
3. Which extraction method created the text and how reliable was it?
4. Which chunks and embeddings were derived from the document?
5. Which source version supported a retrieved citation or generated answer?

This is a planning document only. It defines the target schema and migration direction; it does not implement a manifest writer, change ingestion, or change retrieval behavior.

## 2. Scope

Included in scope:

1. Source PDF and DOCX files.
2. Extracted text and normalized corpus records.
3. Chunk records derived from each document.
4. Embeddings derived from chunks.
5. Retrieval citations and source references.
6. Future answer audit logs that need to reconstruct evidence.

Excluded for now:

1. Active ingestion runtime migration.
2. Actual database implementation.
3. Automatic document approval workflow.
4. Runtime retrieval filtering by metadata.
5. Reprocessing existing documents.
6. Regenerating embeddings.

## 3. Current Status

Current state:

1. The active runtime still uses legacy local data files and script-local dictionaries.
2. `src/scripts/py_files/read_documents.py` reads source documents and writes `data/raw/aviation_corpus.pkl`.
3. `src/scripts/py_files/aviation_chunk_saver.py` writes chunk JSON under `data/processed/chunked_documents/`.
4. Embedding scripts write `data/embeddings/aviation_embeddings.json`.
5. Astra helper scripts store vector records with fields such as `chunk_id`, `filename`, `text`, `tokens`, and `embedding`.
6. Real source documents and generated corpora are ignored from Git.
7. Source documents are currently local/private by policy.
8. No document manifest is implemented yet.
9. `src/aviationrag/models.py` provides early dataclass anchors only; it is not used by runtime ingestion or retrieval.

## 4. Document Identity Model

In AviationRAG, a document is a single controlled source item that can produce extracted text, chunks, embeddings, citations, and audit evidence. A document normally corresponds to one source file such as a PDF or DOCX, but the identity model should also support future web or database sources.

Stable identity fields:

| Field | Purpose |
| --- | --- |
| `document_id` | Stable internal identifier used by chunks, embeddings, citations, and audit logs. |
| `file_hash` | Cryptographic hash of the source file bytes, preferably `sha256:<hex>`. |
| `filename` | Original or stored filename. |
| `canonical_title` | Normalized human-readable title. |
| `source_uri` | Original URL, storage URI, or local source reference when available. |
| `authority` | Controlled source authority value such as `FAA`, `EASA`, or `INTERNAL`. |
| `document_type` | Controlled document type value such as `regulation`, `manual`, or `accident_report`. |
| `revision` | Revision, amendment, issue, edition, or version label. |
| `effective_date` | Date the source became effective, when known. |
| `ingestion_batch_id` | Identifier for the ingestion run that produced the manifest entry. |

Recommended near-term `document_id` strategy:

Use a deterministic ID derived from normalized identity inputs:

```text
document_id = "doc_" + sha256(
  normalized_authority + "|" +
  normalized_filename + "|" +
  file_hash
)[0:16]
```

Pros:

1. Re-ingesting the same file produces the same identifier.
2. Duplicate detection is easier across local runs.
3. Chunks can be traced to the same source without a central database.

Cons:

1. Renaming a file changes the ID unless the strategy excludes filename.
2. Correcting authority metadata can change the ID if authority is part of the hash.
3. Deterministic IDs require careful normalization rules.

Alternative strategy:

Generate a UUID once and store it in the manifest.

Pros:

1. The ID remains stable even if filename or metadata is corrected.
2. It avoids accidental ID changes caused by normalization changes.
3. It works well with database-backed lifecycle records.

Cons:

1. A central manifest becomes required to preserve identity.
2. Duplicate detection must rely on `file_hash` and metadata matching.
3. Rebuilding from raw files alone cannot reproduce the same IDs.

Near-term recommendation:

Use deterministic IDs for local JSONL migration prototypes, while preserving `file_hash` as the primary duplicate-detection control. Revisit UUIDs if the manifest moves to a database-first lifecycle service.

## 5. Manifest Storage Options

| Option | Benefits | Tradeoffs |
| --- | --- | --- |
| Local JSONL manifest | Simple, diffable, easy to inspect, good for early migration. | Weak concurrency, no query engine, private metadata must remain ignored. |
| SQLite | Stronger local integrity, indexes, transactional updates. | Adds operational file management and migration scripts. |
| Astra DB table/collection | Centralized production metadata, available to bridge/runtime services. | Requires schema design, credentials, migrations, and deployment coordination. |
| Hybrid local plus database | Local reproducibility with production queryability. | Requires sync rules and conflict handling. |

Near-term recommendation:

1. Use `data/manifest/documents.jsonl` for local development and migration prototypes.
2. Treat `data/manifest/` as private/generated when it contains real document metadata.
3. Commit only tiny fake/sample manifest fixtures under `data/sample_documents/` if needed for tests or docs.
4. Defer SQLite or Astra implementation until the manifest writer and lifecycle rules are designed.

## 6. Proposed Manifest Record Schema

Required or strongly recommended fields:

| Field | Type | Notes |
| --- | --- | --- |
| `document_id` | string | Stable internal ID. |
| `filename` | string | Original/stored filename. |
| `canonical_title` | string or null | Normalized title used for display/search. |
| `authority` | string or null | Controlled value. |
| `document_type` | string or null | Controlled value. |
| `revision` | string or null | Revision, amendment, edition, or version. |
| `effective_date` | string or null | ISO date when known. |
| `source_uri` | string or null | URL or storage URI. |
| `file_hash` | string | `sha256:<hex>` preferred. |
| `ingestion_status` | string | Document pipeline state. |
| `approval_status` | string | Human or policy approval state. |
| `extraction_method` | string or null | Parser or OCR method. |
| `extraction_quality` | string or null | Controlled quality band. |
| `needs_manual_review` | boolean | True when extraction or metadata requires review. |
| `ingestion_batch_id` | string or null | Ingestion run identifier. |
| `created_at` | string | ISO timestamp. |
| `updated_at` | string | ISO timestamp. |
| `metadata` | object | Extra non-contract metadata. |

Example JSONL record:

```json
{
  "document_id": "doc_8d2f6b3a4c91e0ab",
  "filename": "faa_advisory_circular_sample.pdf",
  "canonical_title": "Sample FAA Advisory Circular",
  "authority": "FAA",
  "document_type": "advisory_circular",
  "revision": "1B",
  "effective_date": "YYYY-MM-DD",
  "source_uri": "https://example.invalid/faa/sample-ac.pdf",
  "file_hash": "sha256:0123456789abcdef",
  "ingestion_status": "available",
  "approval_status": "pending_review",
  "extraction_method": "pdfplumber",
  "extraction_quality": "medium",
  "needs_manual_review": false,
  "ingestion_batch_id": "batch_20260513_001",
  "created_at": "2026-05-13T00:00:00Z",
  "updated_at": "2026-05-13T00:00:00Z",
  "metadata": {}
}
```

## 7. Proposed Chunk Metadata Schema

Chunks should inherit document-level metadata needed for retrieval, filtering, display, and citation reconstruction. A chunk should not require loading the full document manifest just to display a basic source citation, but `document_id` should remain the authoritative join key.

Proposed fields:

| Field | Type | Notes |
| --- | --- | --- |
| `chunk_id` | string | Stable chunk identifier. |
| `document_id` | string | Parent document ID. |
| `filename` | string | Parent filename for backward-compatible display. |
| `canonical_title` | string or null | Parent canonical title. |
| `page_start` | integer or null | First source page represented. |
| `page_end` | integer or null | Last source page represented. |
| `section_path` | array of strings | Hierarchical section headings when available. |
| `paragraph_id` | string or null | Paragraph, clause, or section identifier. |
| `chunk_type` | string | `text`, `table`, `definition`, `note`, `warning`, `procedure`, or `other`. |
| `authority` | string or null | Inherited from document. |
| `document_type` | string or null | Inherited from document. |
| `revision` | string or null | Inherited from document. |
| `effective_date` | string or null | Inherited from document. |
| `extraction_quality` | string or null | Inherited or chunk-specific quality band. |
| `source_hash` | string | Parent document file hash or source hash. |
| `text_hash` | string | Hash of normalized chunk text. |
| `metadata` | object | Extra parser- or retrieval-specific metadata. |

Future chunk IDs should be deterministic within a document, for example:

```text
chunk_id = document_id + "_chunk_" + zero_padded_sequence
```

If section-aware chunking later creates stable paragraph IDs, chunk IDs may incorporate paragraph or section identifiers.

## 8. Document Lifecycle States

Pipeline status values:

| State | Meaning |
| --- | --- |
| `discovered` | Source is known but not uploaded or ingested. |
| `uploaded` | Source file is present in the intake location. |
| `processing` | Extraction or ingestion is in progress. |
| `extracted` | Text extraction completed. |
| `chunked` | Chunks were generated. |
| `embedded` | Embeddings were generated and stored locally or remotely. |
| `available` | Document is eligible for retrieval. |
| `needs_review` | Document requires manual review before trust or use. |
| `retired` | Document is no longer active for normal retrieval. |
| `superseded` | Document has been replaced by a newer revision. |
| `error` | Processing failed or metadata is invalid. |

Approval status values:

| State | Meaning |
| --- | --- |
| `pending_review` | Awaiting human or policy review. |
| `approved` | Approved for intended retrieval scope. |
| `rejected` | Not approved for retrieval. |
| `retired` | Previously approved but no longer active. |

Initial implementation should keep lifecycle state and approval state separate. A document can be technically `available` but still `pending_review` in environments where approval is mandatory.

## 9. Source Authority Classification

Controlled values:

1. `FAA`
2. `EASA`
3. `ICAO`
4. `ASTM`
5. `SAE`
6. `ISO`
7. `NASA`
8. `NTSB`
9. `AAIB`
10. `INTERNAL`
11. `MANUFACTURER`
12. `MILITARY`
13. `OTHER`

Rules:

1. Use uppercase authority codes.
2. Use `INTERNAL` for company procedures, local policies, and private operating documents.
3. Use `MANUFACTURER` for OEM manuals, service bulletins, and technical publications.
4. Use `OTHER` only when the authority cannot be mapped after review.
5. Preserve the raw publisher/source name in `metadata.publisher` when available.

## 10. Document Type Classification

Controlled values:

1. `regulation`
2. `advisory_circular`
3. `certification_specification`
4. `standard`
5. `manual`
6. `report`
7. `accident_report`
8. `safety_management`
9. `design_guidance`
10. `manufacturing_quality`
11. `training_material`
12. `paper`
13. `book`
14. `internal_procedure`
15. `other`

Rules:

1. Prefer the most specific type.
2. Classify accident and incident investigation material as `accident_report`, not generic `report`.
3. Classify FAA AC, EASA AMC/GM, and similar advisory material as `advisory_circular` unless a more precise controlled value is introduced later.
4. Use `internal_procedure` only for controlled internal operating or engineering procedures.
5. Preserve current legacy categories in `metadata.legacy_category` during migration.

## 11. Extraction Quality Fields

Required extraction fields:

| Field | Type | Notes |
| --- | --- | --- |
| `extraction_method` | string or null | Example: `docx`, `pypdf2`, `pdfplumber`, `ocr`, `llama_parse`. |
| `extraction_quality` | string or null | Recommended values: `high`, `medium`, `low`, `failed`, `unknown`. |
| `needs_manual_review` | boolean | True when parser quality, missing pages, OCR, or metadata uncertainty needs review. |
| `extraction_warnings` | array of strings | Human-readable parser warnings. |
| `page_count` | integer or null | Source page count when known. |
| `parsed_page_count` | integer or null | Pages with extracted text or parsed content. |
| `table_count` | integer or null | Detected tables. |
| `image_only_pages` | array of integers | Pages likely requiring OCR or manual review. |

Current legacy extraction metadata already includes:

1. `source_type`
2. `extraction_method`
3. `extraction_quality`
4. `needs_manual_review`

Future migration should map the numeric legacy `extraction_quality` score into controlled quality bands while preserving the raw score in `metadata.extraction_quality_score`.

## 12. Versioning and Revision Policy

Superseded documents should not be deleted silently. Aviation and compliance workflows often require knowing which revision was used at the time of an answer, design decision, or review.

Policy:

1. Store each source revision as a distinct manifest record.
2. Preserve `file_hash`, `revision`, `effective_date`, and `source_uri` for every revision.
3. Mark older revisions as `superseded` or `retired` rather than deleting them.
4. Record successor/predecessor relationships in metadata, for example `metadata.superseded_by` or `metadata.supersedes`.
5. Future retrieval should prefer active/current documents by default.
6. Historical documents should remain retrievable when the user explicitly asks for a prior revision, accident-era source, or historical context.
7. Answers should warn when evidence comes from retired, superseded, low-quality, or unapproved documents.

Revision detection options for later implementation:

1. Parse explicit revision/edition fields from the document.
2. Detect revision-like tokens in filenames.
3. Compare source URIs and file hashes.
4. Use administrator-entered metadata when automatic extraction is unreliable.

## 13. Traceability Requirements

Every answer citation should eventually trace to:

1. `document_id`
2. `chunk_id`
3. Filename and canonical title.
4. Page range and section path where available.
5. Paragraph, clause, or table identifier where available.
6. `file_hash` or `source_hash`.
7. Ingestion timestamp and `ingestion_batch_id`.
8. Extraction method and quality.
9. Retrieval score and retrieval method.
10. Answer mode, model, prompt version, and response policy version.

Minimum future citation payload:

```json
{
  "document_id": "doc_8d2f6b3a4c91e0ab",
  "chunk_id": "doc_8d2f6b3a4c91e0ab_chunk_0007",
  "filename": "faa_advisory_circular_sample.pdf",
  "canonical_title": "Sample FAA Advisory Circular",
  "page_start": 12,
  "page_end": 13,
  "section_path": ["Chapter 2", "Inspection"],
  "file_hash": "sha256:0123456789abcdef",
  "retrieval_score": 0.87
}
```

## 14. Git and Data Governance Alignment

This schema follows the existing data governance policy:

1. Real source documents remain ignored under `data/documents/`.
2. Generated raw corpora remain ignored under `data/raw/`.
3. Generated processed text and chunks remain ignored under `data/processed/`.
4. Generated embeddings remain ignored under `data/embeddings/`.
5. Astra exports remain ignored under `data/astra_db/`.
6. Runtime chat/session state remains ignored under `chat/` and `chat_id/`.
7. Generated manifests containing real document metadata should not be committed.
8. Tiny fake/sample manifests may be committed under `data/sample_documents/` when useful for tests or examples.

Recommended future `.gitignore` alignment:

```text
data/manifest/
!data/sample_documents/
```

Do not commit manifests that reveal private filenames, source URIs, document titles, internal procedures, proprietary manuals, operational context, or source-derived chunk text.

## 15. Future Implementation Phases

Phase D.1a: schema document only

1. Add this planning document.
2. Do not change runtime ingestion.
3. Do not create real manifests.

Phase D.1b: sample manifest fixture

1. Add a tiny fake JSONL fixture under `data/sample_documents/`.
2. Keep it free of real private document names and source-derived text.
3. Use it for tests and examples only.

Phase D.1c: manifest writer module

1. Add a new module under `src/aviationrag/ingestion/`.
2. Generate records from explicit inputs.
3. Add unit tests with fake data.
4. Keep legacy ingestion scripts unchanged until integration is approved.

Phase D.1d: ingestion integration

1. Add controlled integration from legacy ingestion outputs to manifest records.
2. Preserve current JSON/PKL outputs while writing the manifest.
3. Do not change chunking or embeddings in the same step.

Phase D.1e: admin approval lifecycle

1. Add approval status storage.
2. Add admin review/update flow.
3. Keep default retrieval behavior unchanged until policy is approved.

Phase D.1f: retrieval filtering by metadata

1. Add filters for authority, document type, lifecycle state, revision, and date.
2. Measure retrieval impact with an evaluation harness before enabling broadly.
3. Prefer active/current approved documents by default only after migration is complete.

## 16. Open Questions

1. Should `document_id` be deterministic or UUID-based?
2. Should the canonical manifest live locally, in Astra DB, SQLite, or both?
3. Should approval be mandatory before any document is retrievable?
4. How should document revisions be detected reliably across FAA, EASA, ICAO, OEM, internal, and scanned sources?
5. How should scanned PDFs be handled: OCR in the ingestion pipeline, manual review first, or both?
6. Which metadata fields are mandatory for production versus optional for local research use?
7. How should source authority be assigned when a document republishes another authority's text?
8. Should chunk IDs be regenerated when section-aware chunking changes chunk boundaries?
9. How should retired/superseded documents appear in the user interface?
10. What audit retention policy should apply to manifest history, answer citations, and retrieval traces?
