# Real Chunk Migration Design

Date: 2026-05-15
Status: Design and go/no-go criteria only
Scope: Future metadata-rich chunk migration; no runtime integration

## 1. Purpose

This design controls the future transition from legacy chunk outputs to metadata-rich chunks suitable for traceable retrieval, retrieval evaluation, and compliance-style citations.

The migration must make every retrievable passage traceable to a document manifest record, source hash, stable chunk identity, page or section location when available, extraction quality signal, and future vector payload. This document does not implement migration code, change chunking behavior, regenerate embeddings, reset Astra or FAISS, or alter runtime retrieval.

## 2. Scope

Included:

1. Legacy chunk outputs and their future assessment.
2. Metadata-rich `ChunkRecord` targets.
3. Future vector payload dictionaries.
4. Future embedding inputs.
5. Future Astra and FAISS metadata requirements.
6. Retrieval evaluation gates for accepting chunk changes.

Excluded:

1. Actual code migration.
2. Real re-chunking or document reprocessing.
3. Embedding generation.
4. Astra or FAISS reset execution.
5. API, prompt, bridge, deployment, or response-policy behavior changes.

## 3. Current Status

Current prepared assets:

1. Fake metadata-rich chunk fixtures exist at `data/sample_documents/sample_chunks.jsonl`.
2. The target chunk metadata contract is documented in `docs/CHUNK_METADATA_SCHEMA.md`.
3. Chunk schema validation utilities exist in `src/aviationrag/ingestion/chunk_schema.py`.
4. Fake/sample vector payload export utilities exist in `src/aviationrag/ingestion/chunk_payload.py`.
5. A passive gated legacy chunk adapter exists in `src/aviationrag/ingestion/chunk_legacy_adapter.py`.
6. Chunk migration settings exist in `src/aviationrag/config.py` and are disabled by default.
7. Retrieval smoke fixtures, a mock harness, and report/export helpers exist for fake/mock data only.

Current limits:

1. Real legacy ingestion and chunking scripts are not modified.
2. No real chunk migration has happened.
3. No metadata-rich real chunks have been written.
4. No embeddings have been regenerated from metadata-rich chunks.
5. No Astra or FAISS payload/index schema has changed.

## 4. Current Legacy Chunking Assumptions

The current legacy chunking path must be audited before migration. Based on code inspection only, current assumptions are:

1. `read_documents.py` creates document-like dictionaries with `filename`, `text`, `tokens`, `section_references`, `metadata`, and `category`.
2. `read_documents.py` stores extraction metadata such as `source_type`, `extraction_method`, numeric `extraction_quality`, and `needs_manual_review`.
3. PDF extraction currently joins page text into a document-level string; page ranges for individual chunks are not guaranteed.
4. `aviation_chunk_saver.py` chunks document text by sentence/token limits and writes per-document JSON files.
5. Current chunk JSON contains document-level `filename`, `metadata`, `category`, and per-chunk `chunk_id`, `text`, and `tokens`.
6. Current generated embeddings include `chunk_id`, `filename`, `metadata`, `text`, `tokens`, and `embedding`.
7. Current Astra insertion stores `chunk_id`, `filename`, `text`, `tokens`, and `embedding`.
8. Current FAISS metadata maps local index positions to `chunk_id`, `filename`, `text`, and `tokens`.

Items that must be verified with a future read-only audit before migration:

1. Actual current chunk file format across all generated chunk files.
2. Whether any page numbers are available in generated chunks or source corpus records.
3. Filename consistency between documents, chunks, embeddings, Astra exports, and FAISS metadata.
4. Whether `section_references` can be safely mapped to `section_path` or only preserved as loose metadata.
5. Whether tables, figures, warnings, cautions, notes, definitions, and regulatory paragraphs are preserved or flattened.
6. Whether extraction quality is consistently present and comparable across PDF and DOCX sources.

This design intentionally does not scan real generated data or source documents.

## 5. Target Metadata-Rich Chunk Model

The target chunk model is defined in `docs/CHUNK_METADATA_SCHEMA.md`. Minimum future persisted chunk fields are:

| Field | Purpose |
| --- | --- |
| `chunk_id` | Stable chunk identity for retrieval, citations, evaluation, and audit logs. |
| `document_id` | Parent manifest record join key. |
| `filename` | Backward-compatible source display and legacy matching. |
| `canonical_title` | Human-readable document title inherited from the manifest. |
| `text` | Exact text used for embedding and citation. |
| `text_hash` | Hash of normalized chunk text. |
| `source_hash` | Parent source file or source record hash. |
| `chunk_type` | Controlled taxonomy value. |
| `page_start` / `page_end` | Source page span when available. |
| `section_path` | Ordered section hierarchy when available. |
| `paragraph_id` | Clause, paragraph, table, figure, or item identifier when available. |
| `authority` | Inherited source authority. |
| `document_type` | Inherited document type. |
| `revision` | Source revision, issue, edition, or version. |
| `effective_date` | Source effective date when known. |
| `extraction_quality` | Controlled extraction quality band. |
| `created_at` | Record creation timestamp. |
| `metadata` | Parser, migration, retrieval, lifecycle, and review metadata. |

The lightweight `ChunkRecord` dataclass remains intentionally small. Future full-schema fields may continue to live in `ChunkRecord.metadata` until a later model migration is approved.

## 6. Migration Prerequisites

Real chunk migration must not begin until these prerequisites are satisfied:

1. Manifest records exist for all candidate source documents.
2. Every active document has a stable `document_id`.
3. Every active document has a computed `file_hash` or source hash.
4. The chunk schema validator passes on fake fixtures.
5. The passive legacy chunk adapter passes fake-data tests.
6. The vector payload exporter passes fake/sample tests.
7. A read-only legacy chunk format audit has been run and reviewed.
8. The retrieval smoke fixture and mock harness remain passing.
9. A real retrieval baseline plan is approved for pre-change and post-change comparison.
10. D.4 page and structure preservation requirements are approved, including the future parser boundary and structured-document provenance contract.
11. Structured-document validation passes for any future parser output or migration input that claims D.4 provenance.
12. The reset/rebuild plan is reviewed and accepted if embeddings, vector payloads, Astra, or FAISS must be rebuilt.
13. The user/operator explicitly approves any deletion, reset, reprocessing, re-embedding, or database write.

D.4 gate:

Real chunk migration must not begin until page, section, paragraph, table,
figure, warning, caution, note, appendix, and source-span provenance rules are
accepted. Legacy conversion must not claim structured provenance unless it is
derived from source evidence or a future validated parser output.

D.4b gate:

Real migration input that claims structured provenance must satisfy the approved
structured-document schema. Validation errors must block migration. Warning
acceptance policy must be explicitly approved before warning-bearing records
enter migration. Unsupported schema versions must not enter migration. Validator
output should be retained or summarized for auditability in a later controlled
phase. D.4b does not authorize real migration.

## Read-Only Legacy Chunk Format Audit

A read-only legacy chunk audit module exists at:

```text
src/aviationrag/ingestion/chunk_audit.py
```

A manual tool script exists at:

```text
tools/chunking/audit-legacy-chunks.py
```

Current scope:

1. Default input uses the fake sample chunk fixture at `data/sample_documents/sample_chunks.jsonl`.
2. An operator may provide an explicit local file path with `--input`.
3. The tool supports explicit `.jsonl`, `.json`, and trusted-local `.pkl`/`.pickle` files.
4. The tool does not scan directories.
5. The tool does not migrate, rewrite, re-chunk, embed, index, or modify data.
6. Sample record shapes summarize keys and value types only.
7. String values, including `text`, are summarized by type and length; full text is not copied into the report.
8. Generated audit reports go to the ignored local path `logs/chunking/legacy_chunk_audit.json` by default.

This audit is intended to reveal legacy schema shape before any future conversion work. It is not runtime ingestion integration.

## Fake/Local Chunk Migration Dry Run

A fake/local chunk migration dry-run module exists at:

```text
src/aviationrag/ingestion/chunk_migration_dry_run.py
```

A manual tool script exists at:

```text
tools/chunking/run-chunk-migration-dry-run.py
```

Current scope:

1. Default input uses the fake sample chunk fixture at `data/sample_documents/sample_chunks.jsonl`.
2. An operator may provide an explicit local chunk-like file with `--input`.
3. The dry run loads one explicit file, audits input structure, converts records through the passive legacy chunk adapter, validates converted `ChunkRecord` outputs, validates vector payload-shaped dictionaries, and returns summary counts, issues, and warnings.
4. Generated reports go to the ignored local path `logs/chunking/chunk_migration_dry_run.json` by default.
5. Reports are intended as local rehearsal artifacts only and should not be committed when they contain private/local metadata.
6. The dry run does not write migrated chunks, modify `data/processed`, generate embeddings, connect to Astra, use FAISS, reset indexes, or change runtime ingestion.

This dry run is a rehearsal for future migration execution. It is not real chunk migration and does not activate chunk migration flags globally.

## Gated Local Chunk Conversion Writer

A gated local conversion writer exists at:

```text
src/aviationrag/ingestion/chunk_conversion_writer.py
```

A manual tool script exists at:

```text
tools/chunking/write-local-chunk-conversion.py
```

Current scope:

1. Default input uses the fake sample chunk fixture at `data/sample_documents/sample_chunks.jsonl`.
2. An operator may provide an explicit local chunk-like file with `--input`.
3. The tool refuses to write unless local writing is explicitly allowed with `--allow-local-write` or the future chunk migration environment flag is enabled.
4. Default output is the ignored local directory `data/migration_dry_run/chunks`.
5. The writer creates:
   - `converted_chunks.jsonl`
   - `vector_payloads.jsonl`
   - `conversion_report.json`
6. The writer reuses the dry-run flow, passive legacy adapter, chunk schema validator, and vector payload validator before writing local outputs.
7. Generated outputs are local rehearsal artifacts and must not be committed when they contain private/local metadata.
8. The writer does not generate embeddings, connect to Astra, use FAISS, write runtime ingestion outputs, reset indexes, or modify legacy ingestion scripts.

This writer is still not production migration. It only proves that explicitly approved input can be converted into ignored local artifacts for review.

## 7. Legacy-to-New Mapping Strategy

Likely mapping rules:

| Legacy field | Target field | Notes |
| --- | --- | --- |
| `filename` | `filename` | Normalize path separators and use the basename. |
| `title`, `document_title`, metadata title | `canonical_title` | Prefer manifest title when available. |
| `text` | `text` | Preserve exact embedded text unless re-chunking is approved. |
| `page`, `page_start`, `page_end` | `page_start`, `page_end` | Use only when reliable; otherwise set null and flag traceability gap. |
| `section`, `section_path`, `section_references` | `section_path` / `paragraph_id` / `metadata` | Promote only validated hierarchical sections; preserve uncertain references in metadata. |
| `metadata` | `metadata` | Preserve raw parser and legacy fields. |
| `category` | `document_type` or `metadata.legacy_category` | Map only when controlled value is clear. |
| `tokens` | `metadata.tokens` | Do not treat token count as citation metadata. |
| `extraction_method` | `metadata.extraction_method` | Inherit from document or chunk source. |
| `extraction_quality` | `extraction_quality` plus `metadata.extraction_quality_score` | Convert numeric scores to bands only with a documented rule. |
| Legacy `chunk_id` | `chunk_id` | Preserve during compatibility previews; replace only during approved reset/rebuild. |
| Manifest `document_id` | `document_id` | Required for real migration. Deterministic fallback is allowed only for dry runs. |

Rules:

1. Manifest metadata is authoritative over filename inference.
2. Legacy fields should be preserved in `metadata` until confidence is high enough to promote them.
3. Missing page/section values should be explicit nulls or empty lists, not invented.
4. Deterministic inference is acceptable for dry-run previews but not sufficient for compliance-grade migration acceptance.

## 8. Chunk ID Strategy

Recommended final strategy:

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

Stable IDs matter because they support:

1. Retrieval evaluation `expected_chunk_ids`.
2. Citation traceability and audit log reconstruction.
3. Re-embedding consistency across rebuilds.
4. Diffability between ingestion runs.
5. Duplicate detection and superseded chunk relationships.

Policy:

1. Preserve current legacy chunk IDs during read-only audit and compatibility preview phases.
2. Introduce deterministic metadata-rich IDs only during an approved migration or reset phase.
3. Store `previous_chunk_id` when replacing a legacy or older metadata-rich chunk.
4. Store `chunk_version` and schema version when the ID policy or chunk boundary policy changes.
5. Do not mix legacy IDs and new IDs in the same active vector index without an explicit compatibility layer.

## 9. Chunk Type Assignment Strategy

Controlled chunk types come from `docs/CHUNK_METADATA_SCHEMA.md`.

Initial assignment rules:

1. Regulatory identifiers, clauses, AMC/GM references, AC paragraphs, standards clauses, or mandatory language map to `regulatory_paragraph` or `requirement`.
2. Extracted table blocks map to `table`.
3. Figure or image captions map to `figure_caption`.
4. Headings beginning with warning, caution, or note map to `warning`, `caution`, or `note`.
5. Checklist markers or step lists map to `checklist`.
6. Procedure headings or ordered procedural steps map to `procedure`.
7. Definition sections or term-definition patterns map to `definition`.
8. Accident findings map to `accident_finding`.
9. Safety recommendations map to `safety_recommendation`.
10. Unknown normal prose falls back to `paragraph` or `text`.
11. Unclassifiable items use `other` with the original parser signal preserved in metadata.

Type assignment should begin as deterministic heuristics plus validator checks. Human review or admin correction can be added later for low-confidence classifications.

## 10. Page/Section/Paragraph Traceability Strategy

Minimum acceptable traceability for future citation-capable chunks:

1. `document_id`
2. `filename`
3. `canonical_title`
4. `page_start` and `page_end` when source page mapping exists
5. `section_path` when headings are available
6. `paragraph_id`, regulatory reference, table ID, figure ID, or item ID when available
7. `source_hash`
8. `text_hash`

Traceability rules:

1. Page spans should use source document page labels when reliable; preserve PDF page indexes separately if they differ.
2. Section paths should be ordered from broadest to narrowest.
3. Paragraph/reference labels should preserve source formatting when reliable.
4. Chunks with missing page or section data may remain searchable, but strict/compliance citation mode should warn or downgrade evidence quality until reviewed.
5. Traceability gaps must be visible in validation output and future admin review screens.

## 11. Table/Figure/Warning/Caution/Note Handling

Special aviation document content should not be flattened silently when structure is recoverable.

Table handling:

1. Preserve each meaningful table as a `table` chunk.
2. Store `table_id`, caption, page range, and table extraction quality.
3. Serialize table text consistently for embedding while preserving enough metadata to show a table citation later.
4. Flag low-confidence table extraction for manual review.

Figure handling:

1. Preserve captions as `figure_caption` chunks.
2. Store `figure_id`, caption, page range, and source reference when available.
3. Do not claim visual facts from images unless OCR/vision extraction is explicitly implemented later.

Warning, caution, and note handling:

1. Keep warning/caution/note blocks atomic where practical.
2. Store `warning_type`.
3. Avoid splitting warning or caution blocks across overlapping chunks unless necessary.
4. Preserve source heading and page metadata for citation display.

## 12. Extraction Quality and Manual Review Handling

Migration should carry extraction quality from document and chunk levels.

Rules:

1. Low-quality OCR, scanned/image-only sources, empty page extraction, table extraction failures, and uncertain structure should set or inherit `needs_manual_review`.
2. Low-quality chunks may remain searchable during local experiments but should be flagged in payload metadata.
3. Future retrieval may deprioritize low-quality chunks unless no better evidence exists.
4. Strict/compliance answers should display warnings when cited chunks came from low-quality extraction.
5. Numeric legacy extraction quality should be preserved in metadata even after mapping to `high`, `medium`, `low`, `failed`, or `unknown`.
6. Manual review should be able to approve, correct, retire, or supersede chunks without deleting audit history.

## 13. Vector Payload Generation Strategy

Future vector payloads should be generated from validated metadata-rich chunks using the shape prototyped by `src/aviationrag/ingestion/chunk_payload.py`.

Future sequence:

1. Validate chunk records.
2. Convert chunks to vector payload-shaped dictionaries.
3. Confirm payloads include `document_id`, `chunk_id`, source display metadata, page/section metadata, chunk type, text hash, source hash, and extraction quality.
4. Generate embeddings from the payload text only after migration and reset gates are approved.
5. Insert vectors into Astra and rebuild FAISS only after payload validation and retrieval evaluation gates are ready.

This phase does not generate embeddings or write vector payloads to any vector database.

## 14. Evaluation Gate Strategy

Chunk migration should be accepted only through staged evaluation:

1. Keep fake fixture tests passing.
2. Keep chunk schema, payload, legacy adapter, retrieval fixture, retrieval harness, and reporting tests passing.
3. Run a read-only legacy chunk audit before converting real chunks.
4. Run fake/local migration previews before writing ignored local outputs.
5. Run a real pre-migration retrieval baseline before changing active embeddings or indexes.
6. Run the same benchmark after migration/rebuild.
7. Compare top-1, top-3, top-5, expected document, expected chunk, citation traceability, and insufficient-evidence behavior.
8. Treat material regression as a no-go or rollback signal.

The fake/mock harness is not a substitute for real retrieval evaluation. It only verifies that the evaluation machinery and expected result schema work before real retrieval is connected.

## 15. Astra/FAISS Reset Dependency

Real metadata-rich chunk migration will likely require a controlled reset/rebuild because:

1. Chunk IDs may change.
2. Vector payload metadata will change.
3. Existing Astra rows do not contain the full `document_id` and chunk traceability contract.
4. Existing FAISS metadata may not align with the target chunk schema.
5. Embeddings must match the exact text and chunk boundaries used by the new chunk records.

Reset/rebuild must follow `docs/RESET_REBUILD_AND_EVALUATION_BASELINE.md` and must not occur until the go/no-go checklist is accepted. A reset should be local first and should not be mixed with prompt changes, response-policy enforcement, or dependency major upgrades.

## 16. Gated Rollout Strategy

Recommended rollout stages:

1. Dry-run preview only with fake data.
2. Local manifest write only with fake/sample data.
3. Read-only audit of legacy chunk formats.
4. Local chunk conversion dry run using explicit, approved local inputs.
5. Local payload export to ignored outputs.
6. Local embedding rebuild after explicit approval.
7. Local FAISS rebuild and evaluation.
8. Astra reset/reinsert only after local baseline review.
9. Retrieval baseline comparison and citation traceability review.
10. Production deployment only after user/operator review.

Feature flags:

1. Manifest integration remains disabled by default.
2. Chunk migration remains disabled by default.
3. Chunk migration dry-run remains enabled by default.
4. No legacy script should start writing metadata-rich chunks until a future gated integration phase explicitly wires the flags.

## 17. Rollback Strategy

Rollback controls:

1. Keep chunk migration and manifest integration flags disabled until activation is approved.
2. Keep old generated artifacts and vector exports until the new baseline is accepted.
3. Back up local `data/raw/`, `data/processed/`, `data/embeddings/`, FAISS artifacts, and Astra exports when needed.
4. Preserve legacy chunk IDs and old-to-new ID mapping during migration where practical.
5. If post-migration evaluation regresses, disable migration flags and restore prior artifacts or Astra/FAISS snapshots.
6. Record rollback decisions and validation results in `WORKLOG.md`.

Do not silently delete superseded chunks that were used in prior retrieval or answer audit records. Mark them `superseded` or `retired`.

## 18. Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Bad PDF extraction | Missing or misleading chunks. | Preserve extraction quality, review low-quality sources, add OCR strategy later. |
| Missing page numbers | Weak citations and evaluation gaps. | Audit page availability first; mark unknown pages explicitly. |
| Unstable chunk IDs | Broken benchmarks and citations. | Version the ID policy and use deterministic inputs. |
| Broken `expected_chunk_ids` | Retrieval evaluation becomes noisy. | Map old/new chunk IDs or update benchmarks only after review. |
| Embedding cost | Unexpected cost and long runtime. | Estimate chunk count before rebuild and require approval. |
| Private data leakage | Source names/text or metadata committed. | Keep real outputs ignored; inspect staged files; run sanitization. |
| Overconfidence in fake tests | False readiness signal. | Require real retrieval baseline before active migration. |
| Schema mismatch | Astra, FAISS, manifest, and chunks disagree. | Validate payloads and counts before indexing. |
| Table parsing errors | Incorrect technical evidence. | Keep table chunks separate and flag low confidence. |
| Citation mismatch | Answers cite irrelevant or unsupported evidence. | Add citation validation after stable chunk IDs exist. |

## 19. Future Implementation Phases

| Phase | Scope | Runtime impact |
| --- | --- | --- |
| D.3 | Real chunk migration design only. | None. |
| D.3b | Legacy chunk format audit script, read-only. | None; no writes or ingestion. |
| D.3c | Fake/local chunk migration dry run. | None to runtime. |
| D.3d | Gated local chunk conversion writing ignored outputs. | Local-only, disabled by default. |
| D.4 | Page and structure preservation design. | Completed as design only; no reprocessing, migration, embeddings, Astra, or FAISS. |
| D.4b | Synthetic structured-provenance validation. | Offline validation only; no parser, migration, embeddings, Astra, or FAISS. |
| E.3 | Connect retrieval harness to local FAISS outputs. | Evaluation only, no answer behavior change. |
| D.5/E.4 | Controlled local re-chunk, reset/rebuild, and evaluation baseline. | Requires go/no-go approval. |

Implementation sequencing rule:

Do not combine real chunk migration with prompt changes, response-policy enforcement, public bridge work, major dependency upgrades, or production deployment changes.

## 20. Open Questions

1. How reliable is current page metadata across existing PDF and DOCX extraction outputs?
2. Should scanned PDFs require OCR before they can become citation-capable?
3. Does table extraction require a separate parser or table-specific review workflow?
4. Should regulatory chunks be paragraph-level, clause-level, or sentence-level?
5. What chunk overlap policy should apply by document type?
6. Should warning/caution/note blocks always be atomic chunks?
7. Should deterministic chunk IDs include chunk index, paragraph reference, or both?
8. How should chunk IDs survive minor metadata corrections?
9. What final Astra table or collection schema should store metadata-rich vector payloads?
10. Which lifecycle states should block retrieval by default?
11. Where should old-to-new chunk ID mappings live after migration?
12. What minimum real retrieval benchmark size is required before approving reset/rebuild?
