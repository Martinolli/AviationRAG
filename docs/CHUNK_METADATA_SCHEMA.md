# Chunk Metadata Schema

Date: 2026-05-15
Status: Planning baseline only
Scope: Future metadata-rich chunk schema; no runtime migration

## 1. Purpose

Chunk metadata is the bridge between source documents, retrieval, citations, evaluation, and compliance-grade answers.

AviationRAG must eventually be able to answer these questions for every retrieved passage:

1. Which controlled document produced this chunk?
2. Which exact page, section, paragraph, table, or figure does it represent?
3. Which extraction method produced it, and how reliable was that extraction?
4. Which vector payload and local index entry correspond to it?
5. Which answer citation or evaluation case depended on it?

This document defines the target chunk metadata contract for future manifest-driven ingestion. It does not change chunking behavior, reprocess documents, regenerate embeddings, modify Astra, modify FAISS, or change runtime retrieval.

## 2. Scope

Included chunk forms:

1. Text chunks.
2. Table chunks.
3. Figure and caption chunks.
4. Warning, caution, and note chunks.
5. Regulatory paragraph chunks.
6. Procedure and checklist chunks.
7. Definition and requirement chunks.
8. Accident finding and safety recommendation chunks.
9. Synthetic sample chunks under `data/sample_documents/`.

Excluded in this phase:

1. Actual re-chunking implementation.
2. Changes to `read_documents.py` or `aviation_chunk_saver.py`.
3. Embedding generation.
4. Astra DB payload migration.
5. FAISS/index rebuild.
6. Database or index reset execution.
7. API, prompt, bridge, or deployment changes.

## 3. Current Status

Current state:

1. Legacy chunk files exist under ignored generated paths.
2. Runtime chunking still uses legacy local dictionaries and JSON output.
3. `src/scripts/py_files/aviation_chunk_saver.py` splits document text by sentences and token limits.
4. Current chunk JSON includes `filename`, document `metadata`, `category`, and per-chunk `chunk_id`, `text`, and `tokens`.
5. Current FAISS metadata currently maps local index positions to fields such as `chunk_id`, `filename`, `text`, and `tokens`.
6. Current metadata is not sufficient for compliance-grade page, section, paragraph, revision, lifecycle, or extraction-quality traceability.
7. `ChunkRecord` exists in `src/aviationrag/models.py` as a lightweight future dataclass anchor.
8. `data/sample_documents/sample_chunks.jsonl` exists and contains fake synthetic chunk records only.
9. Real chunk migration is not implemented yet.

Fixture note:

`data/sample_documents/sample_chunks.jsonl` now contains an expanded fake/sample-only chunk fixture with metadata-rich examples across multiple chunk types. It is intended for future validator and retrieval evaluation development only. It is not used by runtime ingestion, retrieval, embeddings, Astra, FAISS, API routes, prompts, bridge code, or deployment behavior.

`data/sample_documents/sample_structured_document.json` contains a synthetic D.4 structured-document design fixture. It is not parsed from real source material and is not consumed by runtime ingestion, retrieval, embeddings, Astra, FAISS, API routes, prompts, bridge code, or deployment behavior.

## Chunk Schema Validator

A chunk schema validator exists at `src/aviationrag/ingestion/chunk_schema.py`.

Current scope:

1. Validates metadata-rich fake/sample chunk dictionaries and lightweight `ChunkRecord` objects.
2. Checks required fields, allowed chunk types, page ranges, duplicate `chunk_id` values, document linkage, obvious local/private paths, and optional `confidence_score` bounds.
3. Loads JSONL chunk fixtures with line-numbered errors for malformed JSON.
4. Validates `data/sample_documents/sample_chunks.jsonl` using fake/sample data only.

The validator is not integrated with runtime ingestion, real chunking, embeddings, Astra, FAISS, API routes, prompts, bridge code, or deployment behavior.

## Vector Payload Exporter

A fake/sample chunk payload exporter exists at `src/aviationrag/ingestion/chunk_payload.py`.

Current scope:

1. Converts validated fake/sample chunks into vector payload-shaped dictionaries.
2. Adds `payload_schema_version`, `chunk_id`, `document_id`, `text`, and traceability metadata.
3. Preserves optional metadata such as regulatory references, applicability, warning type, table/figure IDs, captions, reviewer notes, language, product type, and confidence score when present.
4. Validates payloads for required traceability fields, duplicate chunk IDs, forbidden embedding/vector fields, and obvious local/private paths.
5. Supports the local-only developer export tool at `tools/chunking/export-sample-chunk-payloads.py`, which writes generated sample payload JSONL under ignored `logs/`.

The exporter does not generate embeddings, call OpenAI or other embedding APIs, connect to Astra, use FAISS, write to vector databases, change real chunking behavior, or integrate with runtime ingestion.

## Legacy Chunk Adapter

A passive legacy chunk adapter exists at `src/aviationrag/ingestion/chunk_legacy_adapter.py`.

Current scope:

1. Converts fake legacy-like chunk dictionaries into lightweight `ChunkRecord` objects.
2. Preserves future metadata-rich fields in `ChunkRecord.metadata` when the core dataclass does not expose them directly.
3. Optionally converts adapted chunks into vector payload-shaped dictionaries through the fake/sample payload exporter.
4. Produces a side-effect-free migration preview with chunks, payloads, issues, warnings, and summary counts.
5. Respects disabled-by-default chunk migration settings from `src/aviationrag/config.py`.

The adapter is tested with fake inline data only. It is not wired into `read_documents.py`, `aviation_chunk_saver.py`, generated chunk files, embeddings, Astra, FAISS, API routes, prompts, bridge code, or deployment behavior.

The real migration path and go/no-go criteria are defined separately in `docs/REAL_CHUNK_MIGRATION_DESIGN.md`. That design remains planning-only and does not activate chunk migration.

## Read-Only Chunk Format Audit

A read-only chunk format audit module exists at `src/aviationrag/ingestion/chunk_audit.py`, with a manual tool at `tools/chunking/audit-legacy-chunks.py`.

Current scope:

1. Audits explicitly provided chunk-like files or the fake sample fixture by default.
2. Summarizes record count, top-level keys, metadata keys, chunk types, missing IDs, page fields, section fields, and redacted sample shapes.
3. Does not scan directories, migrate chunks, write migration outputs, generate embeddings, use FAISS, connect to Astra, or change runtime behavior.
4. Writes optional generated audit reports under ignored `logs/` only when the tool script is run.

## 4. Chunk Identity Model

A chunk is a bounded retrieval unit derived from a parent document. It should represent a coherent passage, table, figure caption, procedure step, definition, requirement, or finding that can be retrieved, cited, evaluated, and audited independently.

Core identity fields:

| Field | Purpose |
| --- | --- |
| `chunk_id` | Stable internal identifier for the chunk. |
| `document_id` | Parent `DocumentRecord` identifier. |
| `text_hash` | Hash of normalized chunk text. |
| `source_hash` | Parent file hash or source hash. |
| `chunk_version` | Schema or generation version for the chunk. |
| `parent_chunk_id` | Optional parent when a chunk is split from a larger source segment. |
| `previous_chunk_id` | Optional prior version when content or boundaries changed. |

Recommended future `chunk_id` strategy:

```text
chunk_id = "chunk_" + sha256(
  document_id + "|" +
  page_start + "-" + page_end + "|" +
  normalized_section_path + "|" +
  paragraph_id + "|" +
  zero_padded_chunk_index + "|" +
  text_hash
)[0:20]
```

Deterministic IDs:

1. Make repeated local ingestion easier to compare.
2. Allow expected chunk IDs in evaluation fixtures to remain stable when inputs are unchanged.
3. Help detect changed text or changed chunk boundaries.

Tradeoffs:

1. IDs change if text normalization, page extraction, or chunk boundaries change.
2. IDs can change when metadata is corrected if metadata is part of the ID formula.
3. Deterministic IDs require a clearly versioned normalization policy.

Generated IDs:

1. Can remain stable across metadata corrections if stored in a canonical manifest or database.
2. Require durable storage and migration rules.
3. Make rebuild-from-source harder without a persisted mapping.

Near-term recommendation:

Use deterministic IDs for fake fixtures and local migration prototypes. Preserve any existing legacy chunk IDs during compatibility phases. Introduce the final deterministic or generated policy only during a controlled reset/rebuild phase.

## 5. Required Chunk Metadata Fields

Target required fields:

| Field | Type | Notes |
| --- | --- | --- |
| `chunk_id` | string | Stable chunk identifier. |
| `document_id` | string | Parent document join key. |
| `filename` | string | Parent filename for display and legacy compatibility. |
| `canonical_title` | string or null | Display title inherited from document manifest. |
| `text` | string | Exact text used for embedding and citation. |
| `text_hash` | string | `sha256:<hex>` hash of normalized chunk text. |
| `chunk_type` | string | Controlled chunk type. |
| `page_start` | integer or null | First source page. |
| `page_end` | integer or null | Last source page. |
| `section_path` | array of strings | Hierarchical section headings. |
| `paragraph_id` | string or null | Clause, paragraph, item, or reference ID. |
| `authority` | string or null | Inherited document authority. |
| `document_type` | string or null | Inherited document type. |
| `revision` | string or null | Inherited revision. |
| `effective_date` | string or null | Inherited effective date. |
| `extraction_quality` | string or null | Chunk-specific or inherited quality band. |
| `created_at` | string or null | ISO timestamp when chunk record was created. |
| `metadata` | object | Extra parser, retrieval, or migration metadata. |

Required fields should be present in future persisted chunk records even when values are `null`. Runtime migration should not begin until the target required field list is versioned and tested with fake data.

## 6. Optional Chunk Metadata Fields

Optional fields:

| Field | Purpose |
| --- | --- |
| `table_id` | Stable table identifier within a document. |
| `figure_id` | Stable figure identifier within a document. |
| `caption` | Figure, table, or image caption. |
| `warning_type` | `warning`, `caution`, `note`, or related warning class. |
| `regulatory_reference` | Citation-like identifier such as a part, section, clause, AMC, GM, or AC reference. |
| `applicability` | Applicability notes for aircraft, product, operation, or condition. |
| `aircraft_category` | Aircraft category when relevant. |
| `product_type` | Product, component, system, or process category. |
| `confidence_score` | Parser or classifier confidence. |
| `language` | Language code when detected. |
| `source_uri` | Source URL or storage URI inherited from document metadata. |
| `source_page_url` | Page-specific source URL when available. |
| `reviewer_notes` | Manual review notes or quality comments. |

Optional fields may live inside `metadata` during early migration. They should be promoted to first-class fields only when retrieval, citation, evaluation, or filtering requires stable access.

## 7. Chunk Type Taxonomy

Controlled `chunk_type` values:

1. `text`
2. `section`
3. `paragraph`
4. `regulatory_paragraph`
5. `table`
6. `figure_caption`
7. `warning`
8. `caution`
9. `note`
10. `definition`
11. `checklist`
12. `procedure`
13. `requirement`
14. `accident_finding`
15. `safety_recommendation`
16. `metadata_only`
17. `other`

Guidance:

1. Use `regulatory_paragraph` when a chunk maps to an identifiable regulation, clause, AMC, GM, AC paragraph, or standard requirement.
2. Use `table` for extracted tabular content, even when serialized as text.
3. Use `metadata_only` for records that support filtering or display but should not be embedded as normal evidence text.
4. Use `other` only when no controlled value fits.
5. Preserve the original parser type or legacy category in `metadata.legacy_chunk_type` when useful.

## 8. Page, Section, and Paragraph Traceability

Every citation-capable chunk should trace back to:

1. `document_id`
2. `filename`
3. `canonical_title`
4. `page_start`
5. `page_end`
6. `section_path`
7. `paragraph_id` or equivalent reference when available
8. `source_hash`
9. `text_hash`

Traceability rules:

1. Page ranges should use source document page numbering when available.
2. If page labels differ from PDF page indexes, preserve both in metadata.
3. `section_path` should be ordered from broadest to narrowest heading.
4. `paragraph_id` should preserve official numbering or clause labels when reliable.
5. Chunks without page or section metadata may still be retrievable but should not be treated as compliance-grade citations until reviewed.

## 8a. Planned D.4 Structural Provenance Extension

`docs/PAGE_AND_STRUCTURE_PRESERVATION_DESIGN.md` defines the D.4 target design
for preserving page and structure before any real document reprocessing or chunk
migration begins.

The following fields are future design targets only. They are not implemented
by the current `ChunkRecord`, validators, local conversion writer, vector
payload writer, runtime ingestion, retrieval, Astra, or FAISS flows:

| Field | Purpose |
| --- | --- |
| `source_block_ids` | Parser block IDs that contributed to the chunk. |
| `source_span` | Page, character, and optional bounding-box provenance. |
| `pdf_page_index_start` / `pdf_page_index_end` | Zero-based source file page indexes. |
| `printed_page_label_start` / `printed_page_label_end` | Printed page labels when they differ from physical page numbers. |
| `section_id`, `section_number`, `section_title` | Normalized section hierarchy references. |
| `clause_id`, `list_path` | Paragraph, clause, and nested list provenance. |
| `table_id`, `figure_id`, `equation_id`, `admonition_id` | Specialized source-object references. |
| `parser_name`, `parser_schema_version` | Parser contract identity for future structured-document outputs. |
| `structure_confidence` | Confidence in extracted structural provenance. |
| `provenance_status` | `structured`, `legacy_partial`, `legacy_unstructured`, or `synthetic_fixture`. |

Future schema implementation must preserve backward compatibility for legacy
chunks that lack these fields. Missing structural provenance must remain visible
instead of being inferred as authoritative.

## 8b. D.4b Structured-Document Validation

D.4b adds an offline validator for the synthetic structured-document contract at
`src/aviationrag/ingestion/structured_document_validator.py`, with a manual CLI
at `tools/chunking/validate-structured-document.py`.

Structured-document validation is a pre-chunk-conversion coherence gate for
future parser output. It does not change the current `ChunkRecord` runtime
contract, does not implement D.4 provenance fields in runtime ingestion, and
does not authorize real migration. D.4 provenance extensions remain planned
until a later controlled implementation phase wires them into chunk conversion,
embedding payloads, Astra, FAISS, retrieval, and response citations.

## 9. Extraction Quality Metadata

Required or strongly recommended extraction metadata:

| Field | Purpose |
| --- | --- |
| `extraction_method` | Parser or OCR method used. |
| `extraction_quality` | Controlled quality band: `high`, `medium`, `low`, `failed`, or `unknown`. |
| `needs_manual_review` | True when extraction or metadata requires review. |
| `extraction_warnings` | Parser warnings, missing pages, unusual layout, OCR concerns. |
| `ocr_used` | Whether OCR was used for this source or chunk. |
| `table_extraction_quality` | Table-specific quality band when applicable. |
| `image_only_source` | Whether source pages appear image-only or mostly image-only. |

Extraction quality should affect future retrieval ranking and display:

1. Low-quality chunks may be penalized unless no better evidence exists.
2. Strict/compliance answers should warn when cited evidence came from low-quality extraction.
3. Chunks requiring manual review should remain distinguishable in retrieval payloads and citations.

## 10. Citation Requirements

A cited answer should eventually show:

1. Document title.
2. Filename.
3. Page or page range.
4. Section path.
5. Paragraph, clause, table, or figure reference when available.
6. Quoted or cited passage.
7. `chunk_id`.
8. `document_id`.

Minimum future citation payload:

```json
{
  "document_id": "doc_sample_faa_ac_001",
  "chunk_id": "doc_sample_faa_ac_001_chunk_0001",
  "canonical_title": "Sample FAA Advisory Circular for Training Only",
  "filename": "sample_faa_advisory_circular.pdf",
  "page_start": 1,
  "page_end": 1,
  "section_path": ["Sample Overview"],
  "paragraph_id": "sample-1",
  "quoted_passage": "Synthetic sample passage.",
  "extraction_quality": "high"
}
```

Future citation validation should reject or warn on citations that cannot be mapped back to a known `chunk_id` and `document_id`.

## 11. Retrieval Payload Requirements

Future Astra payloads and FAISS metadata should include at minimum:

1. `document_id`
2. `chunk_id`
3. `filename`
4. `canonical_title`
5. `authority`
6. `document_type`
7. `revision`
8. `page_start`
9. `page_end`
10. `section_path`
11. `chunk_type`
12. `extraction_quality`

Recommended additional retrieval payload fields:

1. `effective_date`
2. `approval_status`
3. `lifecycle_status`
4. `text_hash`
5. `source_hash`
6. `paragraph_id`
7. `regulatory_reference`
8. `needs_manual_review`

Retrieval payloads should be small enough for vector metadata storage but complete enough to filter, rank, display, and validate citations without reloading large source files.

## 12. Evaluation Alignment

Retrieval evaluation depends on stable chunk identity.

The smoke fixture uses `expected_chunk_ids` to express which chunk should be retrieved for a fake benchmark case. Real retrieval evaluation will require:

1. Stable `chunk_id` values across benchmark runs.
2. Stable `document_id` values across manifest rebuilds.
3. Known relationships between expected cases, documents, chunks, pages, and sections.
4. Local/private benchmark outputs when real source names or retrieved text are exposed.
5. Clear handling when a chunk is intentionally superseded by a new chunk version.

If chunk boundaries change, the benchmark set must be updated or mapped through `previous_chunk_id`/`parent_chunk_id` relationships before comparing metrics.

## 13. Versioning and Lifecycle

Chunk lifecycle values:

| State | Meaning |
| --- | --- |
| `active` | Eligible for normal retrieval. |
| `superseded` | Replaced by a newer chunk version or newer source revision. |
| `retired` | No longer eligible for normal retrieval. |
| `needs_review` | Requires manual review before trust or use. |
| `error` | Chunk generation or metadata creation failed. |

Versioning fields:

1. `chunk_version`
2. `schema_version`
3. `parent_chunk_id`
4. `previous_chunk_id`
5. `superseded_by_chunk_id`
6. `created_at`
7. `updated_at`
8. `ingestion_batch_id`

Policy:

1. Do not silently delete superseded chunks if they supported prior answers or audit records.
2. Prefer active/current chunks for normal retrieval.
3. Allow historical chunks only when the user asks for historical context or a prior revision.
4. Preserve old-to-new chunk relationships during re-chunking when practical.

## 14. Vector Database Payload Plan

Future Astra vector payload shape should include traceability metadata alongside vector content:

```json
{
  "chunk_id": "doc_sample_faa_ac_001_chunk_0001",
  "document_id": "doc_sample_faa_ac_001",
  "text": "Synthetic sample passage.",
  "filename": "sample_faa_advisory_circular.pdf",
  "canonical_title": "Sample FAA Advisory Circular for Training Only",
  "authority": "FAA",
  "document_type": "advisory_circular",
  "revision": "1A",
  "effective_date": "2026-01-15",
  "page_start": 1,
  "page_end": 1,
  "section_path": ["Sample Overview"],
  "paragraph_id": "sample-1",
  "chunk_type": "text",
  "extraction_quality": "high",
  "source_hash": "sha256:...",
  "text_hash": "sha256:...",
  "metadata": {}
}
```

Implementation notes for a future phase:

1. Do not change the Astra payload until manifest and chunk schemas are tested.
2. Do not mix payload migration with prompt changes or response policy enforcement.
3. Reset or rebuild Astra only after the reset/rebuild go/no-go checklist is satisfied.
4. Run retrieval evaluation before and after any vector payload migration.

## 15. FAISS and Local Index Alignment

Future FAISS metadata must map every local index row to the same `chunk_id` and `document_id` used in Astra and the manifest.

Alignment requirements:

1. FAISS metadata must include `chunk_id`.
2. FAISS metadata must include `document_id`.
3. FAISS metadata should include enough display fields for source preview and citation.
4. Local index IDs should not be treated as durable chunk IDs.
5. Any FAISS rebuild must be generated from the same metadata-rich chunk records used for embedding and Astra insertion.
6. Retrieval evaluation should confirm that expected chunk IDs can be found from FAISS result metadata.

## 16. Future Migration Phases

| Phase | Scope | Runtime impact |
| --- | --- | --- |
| D.2 | Chunk metadata schema planning only. | None. |
| D.2b | Fake chunk fixture expansion. | None to runtime. |
| D.2c | Chunk schema validator. | None to runtime. |
| D.2d | Fake chunk payload exporter. | None to runtime. |
| D.2e | Gated legacy chunk adapter. | Disabled by default. |
| D.3 | Real chunk migration design. | Planning only; no runtime migration. |
| D.3b | Read-only legacy chunk format audit. | Completed without runtime ingestion changes. |
| D.3c | Fake/local chunk migration dry run. | Completed for local rehearsal only. |
| D.3d | Gated local chunk conversion writer. | Completed for ignored local outputs only. |
| D.4 | Page and structure preservation design. | Completed as design only; no reprocessing, migration, embeddings, Astra, or FAISS. |
| D.4b | Synthetic structured-provenance validation. | Completed as offline validation only; no parser, runtime ingestion, migration, embeddings, Astra, or FAISS. |
| D.5 | Re-embed/re-index after reset gate. | Future work requiring explicit reset approval. |

Migration rules:

1. Keep legacy chunking behavior unchanged until a future implementation phase is approved.
2. Validate fake fixtures before touching real generated chunks.
3. Add schema validators before writing metadata-rich real chunks.
4. Gate any legacy adapter integration.
5. Do not re-embed or rebuild indexes until the reset/rebuild plan is approved.

## 17. Open Questions

1. What final chunk size and overlap strategy should be used by document type?
2. Should regulatory paragraphs avoid overlap to preserve exact citation boundaries?
3. How should tables be extracted, serialized, embedded, and cited?
4. How should scanned PDFs and OCR confidence be represented?
5. What citation granularity is required for compliance mode: page, paragraph, clause, table cell, or exact sentence?
6. Should `chunk_id` be deterministic, generated once, or stored through a database identity service?
7. How should old benchmark `expected_chunk_ids` be mapped after a chunking redesign?
8. Which metadata fields must be duplicated into vector payloads versus loaded through the manifest at citation time?
9. How should low-quality or unapproved chunks be filtered before retrieval?
10. What UI source viewer fields are mandatory for engineering review?
