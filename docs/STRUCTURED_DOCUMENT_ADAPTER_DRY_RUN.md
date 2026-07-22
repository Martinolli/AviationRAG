# Structured Document Adapter Dry Run

Date: 2026-07-22  
Status: Implemented for synthetic and explicitly provided parser-output artifacts only  
Phase: D.4c

## Purpose

D.4c adds an offline adapter dry run for `techdoc-parser` structured-document
parser outputs. The adapter verifies a structured-document artifact and its
manifest, runs the existing AviationRAG structured-document validator, and maps
validated blocks/entities into review-only chunk candidates.

This is not runtime ingestion. It does not modify `techdoc-parser`, import
`techdoc-parser` modules, parse documents, generate embeddings, connect to
Astra, use FAISS, write runtime chunks, or modify legacy ingestion scripts.

## Files

Implementation:

```text
src/aviationrag/ingestion/structured_document_adapter.py
tools/chunking/run-structured-document-adapter-dry-run.py
```

Committed synthetic fixture:

```text
tests/fixtures/structured_document_adapter/structured_document.json
tests/fixtures/structured_document_adapter/manifest.json
tests/fixtures/structured_document_adapter/source.txt
```

Tests:

```text
tests/test_structured_document_adapter.py
```

## Integrity Gates

The adapter checks these parser-output integrity conditions before candidate
construction:

1. Manifest has exactly one `artifact_type: structured_document` artifact entry.
2. Manifest artifact entry matches artifact schema name, schema version,
   document ID, media type, and artifact path.
3. Manifest `outputs.structured_document`, when present, matches the artifact
   entry path.
4. Manifest `artifact_sha256` is a lowercase raw SHA256 hex digest and matches
   the artifact file bytes.
5. Manifest `source_sha256` is a lowercase raw SHA256 hex digest.
6. Artifact `document.source_hash` matches manifest `source_sha256`.
7. Source file bytes match manifest `source_sha256` when `--source` is provided.

If `--source` is omitted, artifact integrity may still proceed, but the result
is `REVIEW` and candidates use `structured_partial` provenance.

## Validator Policy

The adapter always runs `validate_structured_document` before candidate
construction.

Validation errors block candidate construction and produce `FAIL`.

Validation warnings are fail-closed by default. A warning code can be accepted
for a specific dry run by repeating `--approve-warning <CODE>`, which converts
that warning from `FAIL` to `REVIEW`. `--strict-warnings` treats all warnings as
failing even if approvals are supplied.

## Candidate Policy

The adapter emits `StructuredDocumentChunkCandidate` records, not runtime
`ChunkRecord` objects.

Default candidate behavior:

1. Paragraph, table, figure caption, equation, warning/caution/note, definition,
   procedure step, requirement, table caption, and unknown blocks are eligible.
2. Section and appendix headings are excluded by default and included only with
   `--include-headings`.
3. Admonition root entities produce one candidate and their source body blocks
   are not duplicated.
4. Root table, figure, equation, admonition, and cross-reference entities are
   linked to candidates through declared `source_block_ids`.
5. Raw `text` is copied from the source block or entity body without repair.
6. `normalized_text` is preserved separately when present.
7. Candidate IDs use:

```text
<document_id>:chunk:<percent-encoded-source-entity-id>
```

This keeps IDs deterministic while making unsafe source entity separators
reversible.

## Outcomes

The dry run reports one of three outcomes:

| Outcome | Meaning |
| --- | --- |
| `PASS` | No adapter errors or warnings. |
| `REVIEW` | No errors, but at least one accepted warning or unverified source checksum exists. |
| `FAIL` | Integrity, validator, or candidate-construction errors exist. |

CLI exit codes:

| Outcome | Exit code |
| --- | ---: |
| `PASS` | 0 |
| `REVIEW` | 2 |
| `FAIL` | 1 |

## Manual Fixture Run

```powershell
.\.venv\Scripts\python.exe tools\chunking\run-structured-document-adapter-dry-run.py `
  --artifact tests\fixtures\structured_document_adapter\structured_document.json `
  --manifest tests\fixtures\structured_document_adapter\manifest.json `
  --source tests\fixtures\structured_document_adapter\source.txt
```

Expected fixture result:

```text
Outcome: PASS
Candidate count: 6
Artifact checksum matches: True
Source checksum matches: True
Manifest matches artifact: True
Validator errors: 0
Validator warnings: 0
```

## Optional Local Outputs

Local outputs are disabled unless explicitly allowed:

```powershell
.\.venv\Scripts\python.exe tools\chunking\run-structured-document-adapter-dry-run.py `
  --artifact tests\fixtures\structured_document_adapter\structured_document.json `
  --manifest tests\fixtures\structured_document_adapter\manifest.json `
  --source tests\fixtures\structured_document_adapter\source.txt `
  --allow-local-write
```

Default output directory:

```text
data/migration_dry_run/structured_document_adapter/
```

Generated files:

```text
adapted_chunk_candidates.jsonl
adapter_report.json
artifact_integrity.json
```

`data/migration_dry_run/` is ignored. Generated outputs must not be committed
when they contain local or private metadata.

## Formal Cross-Project Dry Run

D.4c is designed to consume a `techdoc-parser` structured-document artifact and
manifest as files only. AviationRAG must not import `techdoc-parser` runtime
modules.

The formal cross-project dry run should be performed by first generating a
synthetic structured-document export in `techdoc-parser`, then passing the
exported artifact, manifest, and source bytes to the AviationRAG CLI. The
expected acceptance result is `PASS` when hashes and validation are clean.

This proves file-contract compatibility only. It does not approve real corpus
reprocessing or ingestion.

## Remaining Boundaries

D.4c does not:

1. Convert candidates into persisted runtime chunks.
2. Promote D.4 fields into active retrieval payloads.
3. Rebuild embeddings, Astra, or FAISS.
4. Modify `read_documents.py`, `aviation_chunk_saver.py`, or `faiss_indexer.py`.
5. Authorize real source-document migration.

The recommended next phase is a controlled review of adapted candidates and
approval of a future persisted chunk-schema mapping before any reset/rebuild or
real ingestion integration.
