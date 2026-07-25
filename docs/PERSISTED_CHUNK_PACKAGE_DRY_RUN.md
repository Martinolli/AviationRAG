# Persisted Chunk Package Dry Run

Date: 2026-07-25
Status: D.5b implemented; synthetic/local package dry run complete
Scope: Offline persisted-record scaffolding only

## 1. Purpose

D.5b implements the D.5 persisted chunk contract as isolated scaffolding. It
maps D.4c `StructuredDocumentChunkCandidate` records into validated,
storage-neutral `PersistedChunkRecord` records and can write a deterministic
local package for synthetic/sample review.

D.5b does not activate runtime persistence, runtime ingestion, embeddings,
Astra, FAISS, retrieval, prompts, API behavior, deployment, or response policy.

## 2. D.5 Contract Relationship

The approved D.5 design is documented in
`docs/PERSISTED_CHUNK_RECORD_MAPPING_DESIGN.md` and
`docs/persisted_chunk_record_mapping.json`. D.5b implements the synthetic/local
mapper, validator, limitation registry, and package writer for that design.

## 3. D.4c Integration

```text
StructuredDocument artifact + parser manifest
        |
        v
D.4c StructuredDocument adapter
        |
        v
StructuredDocumentChunkCandidate records
        |
        v
D.5b persisted-record mapper
        |
        v
validated local persistence package
```

Adapter `FAIL` blocks package acceptance. Adapter `REVIEW` makes the package
`REVIEW` unless the package has blocking issues.

## 4. Persisted Schema

Implemented in `src/aviationrag/ingestion/persisted_chunk_record.py`:

```text
schema_name: aviationrag-persisted-chunk
schema_version: 0.1.0
mapping specification: aviationrag-persisted-chunk-mapping / 0.1.0
mapper version: 0.1.0
```

`PersistedChunkRecord` is a frozen dataclass and copies mapping values to avoid
input mutation.

## 5. Mapper API

Implemented in `src/aviationrag/ingestion/persisted_chunk_mapper.py`:

- `build_persisted_chunk_id(...)`
- `map_candidate_to_persisted_chunk(...)`
- `PersistedChunkMappingPolicy`
- `PersistedChunkCandidateContext`

Defaults are fail-closed: partial provenance is disabled, headings are
excluded, and unknown content types reject.

## 6. Deterministic ID Policy

`chunk_id` format:

```text
<document_id>:chunk:<first-24-hex-of-sha256>
```

The canonical identity payload includes document ID, persisted schema
identity, content type/subtype, ordered source block IDs, ordered entity IDs,
and candidate sequence key. It does not use filenames, paths, timestamps,
embedding data, Astra IDs, or FAISS positions.

## 7. Provenance Gates

`full_provenance` requires document ID, safe source filename, valid
64-character source checksum, source block IDs, page/PDF provenance, parser
name/version, and no contradictions.

`partial_provenance` rejects by default and requires explicit policy,
approved limitation code, and `review_required=true`.

`legacy_filename_only` and `unknown_provenance` reject for D.5b new-structured
mapping.

## 8. Content-Type Rules

Supported D.4c mappings include paragraph, table, figure caption, equation,
warning, caution, note, procedure, requirement, and definition. Unknown
unsupported content types reject by default.

## 9. Content Policies

Headings reject by default with `HEADING_RECORD_DISABLED`. When explicitly
enabled, heading records require review.

Tables derive from source-block candidates, retain table IDs, and do not
generate rows or cells. Figures retain caption text and figure IDs only.
Equations retain exact equation text and equation IDs. Admonitions preserve
exact body text and admonition IDs. Cross-reference IDs attach to source
chunks; references are not silently resolved.

D.5b does not perform text-only deduplication. Duplicate source block IDs
inside one record reject.

## 10. Accepted Limitation Registry

Implemented in `src/aviationrag/ingestion/persisted_chunk_validator.py` with
registry version `0.1.0`.

| Code | D.5b policy |
| --- | --- |
| `CHUNK_SECTION_CROSSING_REVIEW` | Conditional, review required. |
| `DUPLICATE_TEXT_LINES` | Conditional, text-only deduplication forbidden. |
| `TABLE_CANDIDATE_ONLY` | Conditional for table content, review required. |
| `TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE` | Conditional for table or figure-caption evidence, review required. |

Unknown limitation codes reject.

## 11. Validation and Rejection

Validation states are `valid`, `valid_with_warnings`, `review_required`, and
`rejected`. Rejected candidates are retained in `rejected_candidates.jsonl`
when a package is written. Same-ID collisions use
`PERSISTED_CHUNK_ID_COLLISION` and make the package `FAIL`.

Forbidden persisted fields include embeddings, vectors, Astra IDs, FAISS
positions, timestamps, absolute paths, temporary paths, generic confidence
fields, generated figure descriptions, and inferred revisions.

## 12. Package Artifacts

Implemented in `src/aviationrag/ingestion/persisted_chunk_package.py`.

Package files:

```text
persisted_chunks.jsonl
persistence_manifest.json
persistence_report.json
rejected_candidates.jsonl
warnings.json
```

The manifest records source artifact checksum, source manifest checksum,
mapper version, persisted schema version, counts, file checksums, limitation
registry version, and package digest. The manifest excludes its own exact-byte
checksum to avoid circular checksum dependency.

## 13. Determinism

All JSON files are UTF-8 and end with a final newline. JSONL files use
deterministic sorted keys. The package digest is derived from the deterministic
file checksum map.

## 14. CLI Use

CLI:

```text
tools/chunking/run-persisted-chunk-package-dry-run.py
```

No-write dry run:

```powershell
.\.venv\Scripts\python.exe tools/chunking/run-persisted-chunk-package-dry-run.py `
  --artifact tests/fixtures/structured_document_adapter/structured_document.json `
  --manifest tests/fixtures/structured_document_adapter/manifest.json `
  --source tests/fixtures/structured_document_adapter/source.txt
```

Local write:

```powershell
.\.venv\Scripts\python.exe tools/chunking/run-persisted-chunk-package-dry-run.py `
  --artifact tests/fixtures/structured_document_adapter/structured_document.json `
  --manifest tests/fixtures/structured_document_adapter/manifest.json `
  --source tests/fixtures/structured_document_adapter/source.txt `
  --output-dir data/migration_dry_run/persisted_chunk_package `
  --allow-local-write
```

Exit codes:

```text
0 = PASS
2 = REVIEW
1 = FAIL
```

## 15. Formal Synthetic Dry-Run Result

Formal fixture:

```text
tests/fixtures/structured_document_adapter/structured_document.json
tests/fixtures/structured_document_adapter/manifest.json
tests/fixtures/structured_document_adapter/source.txt
```

Result:

| Field | Value |
| --- | --- |
| Adapter outcome | `PASS` |
| Package outcome | `PASS` |
| Input candidates | 6 |
| Accepted records | 6 |
| Rejected candidates | 0 |
| Warnings | 0 |
| Issues | 0 |
| Validation-status counts | `{"valid": 6}` |
| Provenance counts | `{"full_provenance": 6}` |
| Content-type counts | `{"equation": 1, "figure_caption": 1, "paragraph": 2, "table": 1, "warning": 1}` |
| Limitation counts | `{}` |
| Review-required count | 0 |
| Package digest | `36355a2dbc52c1534ce884fc11d5554dfc9b4c37785054d85b11bc6696a134d9` |

Determinism was verified by writing the package twice into clean ignored local
directories and comparing SHA-256 values for all five package files. The bytes
matched.

## 16. Explicit Exclusions

D.5b excludes runtime ingestion, real corpus processing, production persisted
chunks, embedding generation, Astra, FAISS, retrieval wiring, prompt or
generation changes, API or deployment changes, and `techdoc-parser`
modification or dependency use.

## 17. Preconditions for Later Real Sample Persistence

A later D.5c controlled real parser-output sample persistence phase still
requires approved tiny real sample selection, source-protection checks,
partial-provenance governance, warning-owner approval, package-retention
decision, legacy coexistence policy, and explicit authorization before any
real corpus, embedding, Astra, FAISS, or retrieval work.

## 18. D.5c Controlled Real Parser-Output Validation

D.5c consumed one approved real parser-generated StructuredDocument artifact
for `FAA_Order_4040_26B.pdf` through the unchanged D.5b package writer. The
D.5b synthetic package result remained unchanged, and the storage-neutral
package writer remained isolated from runtime ingestion.

| Field | Value |
| --- | --- |
| Gate outcome | `PASS` |
| Adapter outcome | `PASS` |
| Package outcome | `PASS` |
| Input candidates | 920 |
| Accepted records | 920 |
| Rejected candidates | 0 |
| Warnings | 0 |
| Review-required count | 0 |
| Provenance counts | `{"full_provenance": 920}` |
| Package digest | `d2509f9dbaba886b82cb135b386a7c494aaf0569a8422ad4031cd9c38a26f6a5` |

No runtime integration occurred. The real artifact and generated packages
remained ignored local files, no source PDF was copied into
`AviationRAG/data/documents`, and no embeddings, Astra writes, FAISS writes, or
production migration were authorized.
