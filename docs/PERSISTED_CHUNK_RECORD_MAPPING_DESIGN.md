# Persisted ChunkRecord Mapping Design

Date: 2026-07-25
Status: D.5 design complete; persistence not implemented
Scope: Normative persisted-record contract and governance only

This document defines the planned, versioned contract for converting:

```text
StructuredDocumentChunkCandidate
        |
        v
PersistedChunkRecord
        |
        v
future vector payload
```

D.5 is documentation-only. It does not implement a mapper, write persisted
records, change runtime ingestion, process the real corpus, generate
embeddings, connect to Astra, use FAISS, change retrieval, change prompts, or
modify `techdoc-parser`.

## 1. Purpose

A separate persisted-record contract is required because parser evidence,
adapter candidates, durable chunks, vector payloads, and retrieved chunks have
different ownership and lifecycle requirements.

| Layer | Meaning | Source of truth |
| --- | --- | --- |
| Parser evidence | `techdoc-parser` observations in `techdoc-structured-document / 0.1.0` artifacts. | Parser artifact and manifest. |
| Adapter candidate | AviationRAG D.4c review-only projection of parser evidence. | D.4c dry-run output; not durable runtime data. |
| Persisted chunk | Planned durable, validated, audit-ready retrieval unit. | Future persisted package or storage record. |
| Vector payload | Future storage-neutral embedding/index payload derived from persisted chunks. | Persisted chunk plus later vector policy. |
| Retrieved chunk | Runtime retrieval result returned by FAISS/Astra/hybrid retrieval. | Future vector/index metadata plus persisted source record. |

The persisted record MUST become the durable source of truth for future
embedding, indexing, citation, rollback, and audit. A vector payload MAY derive
selected fields from it, but the persisted record MUST NOT include embeddings,
vectors, Astra-specific IDs, or FAISS positions.

## 2. Scope

In scope:

1. Persisted schema design.
2. Deterministic identity.
3. Provenance requirements.
4. Content policies.
5. Accepted-limitation handling.
6. Validation gates.
7. Local persistence-package design.
8. Rollback and audit requirements.

Out of scope:

1. Real persistence implementation.
2. Real corpus migration.
3. Embedding generation.
4. Astra insertion.
5. FAISS indexing.
6. Retrieval wiring.
7. Response generation.

## 3. Pipeline Boundary

Planned boundary:

```text
techdoc-parser
    |
    v
StructuredDocument / 0.1.0
    |
    v
AviationRAG validator
    |
    v
StructuredDocumentChunkCandidate
    |
    v
PersistedChunkRecord
    |
    v
future vector payload
    |
    v
future embedding/index pipeline
```

Ownership:

| Boundary | Owner | Responsibility |
| --- | --- | --- |
| `techdoc-parser` | Upstream parser | Emit deterministic structured-document artifact and manifest. |
| `StructuredDocument / 0.1.0` | Upstream contract | Preserve parser evidence without RAG runtime dependencies. |
| AviationRAG validator | AviationRAG D.4b | Fail closed on invalid schema/provenance coherence. |
| `StructuredDocumentChunkCandidate` | AviationRAG D.4c | Produce review-only candidates without persistence. |
| `PersistedChunkRecord` | Future AviationRAG mapper | Validate, normalize, and persist durable chunk records. |
| Future vector payload | Future AviationRAG vector phase | Derive embedding/index metadata from persisted records. |

## 4. Schema Identity

The planned persisted schema identity is:

```json
{
  "schema_name": "aviationrag-persisted-chunk",
  "schema_version": "0.1.0"
}
```

Rules:

1. `schema_name` MUST be `aviationrag-persisted-chunk` for this design.
2. `schema_version` MUST use semantic versioning.
3. Additive optional fields MAY use a minor or patch version when old records
   still validate.
4. Required-field changes, identity-policy changes, content-boundary changes,
   validation-state meaning changes, or provenance-status meaning changes MUST
   use a breaking version.
5. Unsupported versions MUST fail closed before persistence.
6. The persisted schema version is distinct from the upstream parser schema
   version.
7. The persisted schema version is distinct from parser version, adapter
   version, mapper version, and embedding-model version.
8. Embedding-model version MUST live in later vector/index metadata, not inside
   the persisted chunk schema identity.

## 5. Conceptual Persisted Record

The conceptual `PersistedChunkRecord` is not implemented in D.5.

Recommended fields:

| Group | Fields |
| --- | --- |
| Schema and identity | `schema_name`, `schema_version`, `chunk_id`, `chunk_index`, `document_id` |
| Document metadata | `source_filename`, `source_checksum`, `document_title`, `document_number`, `document_revision`, `document_issue`, `effective_date` |
| Chunk content | `text`, `normalized_text`, `content_type`, `content_subtype`, `language` |
| Page provenance | `page_start`, `page_end`, `pdf_page_index_start`, `pdf_page_index_end`, `contributing_page_numbers`, `contributing_pdf_page_indexes`, `printed_page_labels` |
| Section and clause provenance | `section_id`, `section_path`, `section_number`, `section_title`, `clause_identifier` |
| Source evidence | `source_block_ids`, `source_span`, `table_ids`, `figure_ids`, `equation_ids`, `admonition_ids`, `cross_reference_ids` |
| Parser and pipeline metadata | `parser_name`, `parser_version`, `structured_document_schema_version`, `adapter_version`, `persistence_mapper_version`, `extraction_method` |
| Governance | `provenance_status`, `accepted_limitation_codes`, `validation_status`, `warning_codes`, `review_required`, `persisted_at` |

`persisted_at` SHOULD remain outside deterministic package core unless a future
execution phase explicitly defines a non-deterministic audit envelope.

## 6. Required, Conditional, Optional, and Forbidden Fields

| Field | Classification | Policy |
| --- | --- | --- |
| `schema_name` | required_for_persistence | MUST equal supported persisted schema name. |
| `schema_version` | required_for_persistence | MUST equal supported persisted schema version. |
| `chunk_id` | required_for_persistence | MUST follow deterministic ID policy. |
| `chunk_index` | required_for_persistence | MUST be deterministic order index. |
| `document_id` | required_for_persistence | MUST match source artifact document ID. |
| `source_filename` | required_for_persistence | MUST be basename or manifest-safe logical name. |
| `source_checksum` | required_for_persistence | MUST be source-byte checksum for structured persistence. |
| `document_title` | optional | MAY be null when unavailable. |
| `document_number` | optional | MUST NOT be inferred from filename alone. |
| `document_revision` | optional | MUST NOT be inferred. |
| `document_issue` | optional | MUST NOT be inferred. |
| `effective_date` | optional | MUST NOT be inferred from file metadata. |
| `text` | required_for_persistence | MUST preserve exact source-derived candidate wording. |
| `normalized_text` | optional | MAY be stored separately; MUST NOT replace `text`. |
| `content_type` | required_for_persistence | MUST use approved persisted taxonomy. |
| `content_subtype` | optional | MAY refine controlled content type. |
| `language` | optional | MAY be absent or null. |
| `page_start` | required_for_persistence | Required for new structured chunks. |
| `page_end` | required_for_persistence | Required for new structured chunks. |
| `pdf_page_index_start` | required_for_persistence | Required for new structured chunks. |
| `pdf_page_index_end` | required_for_persistence | Required for new structured chunks. |
| `contributing_page_numbers` | required_for_persistence | MUST list all contributing source page numbers. |
| `contributing_pdf_page_indexes` | required_for_persistence | MUST list all contributing PDF page indexes. |
| `printed_page_labels` | optional | MAY be empty or null when parser does not supply labels. |
| `section_id` | conditionally_required | Required when source block belongs to a known section. |
| `section_path` | conditionally_required | Required when section evidence exists. |
| `section_number` | optional | MAY be null for unnumbered headings. |
| `section_title` | conditionally_required | Required when `section_id` resolves to a section title. |
| `clause_identifier` | optional | MAY be null when unavailable. |
| `source_block_ids` | required_for_persistence | MUST contain at least one source block ID. |
| `source_span` | required_for_persistence | MUST preserve page/block span evidence. |
| `table_ids` | conditionally_required | Required when chunk represents table evidence. |
| `figure_ids` | conditionally_required | Required when chunk represents figure-caption evidence. |
| `equation_ids` | conditionally_required | Required when chunk represents equation evidence. |
| `admonition_ids` | conditionally_required | Required when chunk represents admonition evidence. |
| `cross_reference_ids` | conditionally_required | Required when source text has mapped cross-references. |
| `parser_name` | required_for_persistence | MUST identify upstream parser. |
| `parser_version` | required_for_persistence | MUST identify upstream parser version. |
| `structured_document_schema_version` | required_for_persistence | MUST identify source schema version. |
| `adapter_version` | required_for_persistence | MUST identify candidate adapter policy. |
| `persistence_mapper_version` | required_for_persistence | MUST identify future mapper policy. |
| `extraction_method` | required_for_persistence | MUST preserve source extraction path. |
| `provenance_status` | required_for_persistence | MUST use approved provenance status. |
| `accepted_limitation_codes` | required_for_persistence | MAY be empty; unknown codes fail. |
| `validation_status` | required_for_persistence | MUST use approved validation state. |
| `warning_codes` | required_for_persistence | MAY be empty; warnings are not erased. |
| `review_required` | required_for_persistence | MUST be explicit boolean. |
| `persisted_at` | optional | SHOULD be outside deterministic package core. |
| `embeddings` | forbidden | MUST NOT appear. |
| `embedding` | forbidden | MUST NOT appear. |
| `vectors` | forbidden | MUST NOT appear. |
| `vector` | forbidden | MUST NOT appear. |
| `absolute_path` | forbidden | MUST NOT appear. |
| `temporary_path` | forbidden | MUST NOT appear. |
| `random_id` | forbidden | MUST NOT appear. |
| `unverified_table_cells` | forbidden | MUST NOT appear as verified structure. |
| `generated_figure_description` | forbidden | MUST NOT appear. |
| `fabricated_confidence` | forbidden | MUST NOT appear. |
| `inferred_revision` | forbidden | MUST NOT appear. |

## 7. Deterministic `chunk_id` Policy

Persisted `chunk_id` MUST be deterministic, traceable, storage-neutral, and
independent of embedding/vector stores.

Candidate readable form:

```text
<document_id>:chunk:<normalized-source-block-key>
```

Pros: easy to inspect and trace. Cons: can become long, leaks source IDs into
primary identity, and needs escaping for multi-block/entity cases.

Candidate hashed form:

```text
<document_id>:chunk:<short-sha256>
```

Pros: compact, stable, handles multi-block/entity records, and keeps detailed
traceability in separate fields. Cons: less readable without source evidence
fields.

Selected D.5 policy:

```text
<document_id>:chunk:<first-24-hex-of-sha256>
```

Canonical digest inputs MUST be serialized in this exact conceptual order:

```text
document_id
schema_name
schema_version
content_type
content_subtype
ordered source_block_ids
ordered entity IDs
chunk_sequence_key
```

Rules:

1. Inputs MUST be UTF-8 strings.
2. Empty optional inputs MUST be represented as empty strings, not omitted.
3. Lists MUST be ordered and joined only through deterministic canonical JSON,
   not ad hoc separators.
4. If a human-readable form is ever emitted for diagnostics, separator
   escaping MUST use percent encoding.
5. Hash algorithm MUST be SHA-256.
6. Truncation SHOULD be 24 lowercase hexadecimal characters for the first
   persisted design because it gives 96 bits of identity space while keeping IDs
   usable in logs and fixtures.
7. A future phase MAY approve 16 hex characters for tiny synthetic packages,
   but real persistence SHOULD use 24 or more.
8. Collision detection MUST compare full canonical identity inputs for any
   duplicate truncated digest in the same package.
9. Any collision MUST reject the package unless a longer truncation policy is
   approved and recorded.
10. Duplicate source text alone MUST NOT collapse chunks.
11. Entity-derived chunks MUST include entity IDs in canonical inputs.
12. Multi-block chunks MUST include the ordered source block ID list.
13. Parser version changes SHOULD NOT change IDs unless source block IDs,
   entity IDs, schema version, content role, or mapper chunk boundaries change.
14. Mapper version changes MAY change IDs only when identity-affecting policy
   changes; the mapper version MUST be recorded in the package.
15. Source changes or mapping-boundary changes are expected to change IDs.
16. Rollback MUST retain old packages and old-to-new mapping evidence when IDs
   change.
17. Random UUIDs, timestamps, absolute paths, embedding content, Astra identity,
   and FAISS positions MUST NOT be used in `chunk_id`.

## 8. `chunk_index` Policy

`chunk_index` is separate from `chunk_id`.

Rules:

1. `chunk_index` MUST be zero-based.
2. It MUST follow final persisted chunk order.
3. It MUST be deterministic for identical input and policy.
4. It MAY change when earlier chunks are inserted, removed, or rejected.
5. It MUST NOT be used as durable primary identity.
6. It MAY be used for ordering, reporting, and audit diffs.

## 9. Provenance Classes

| Provenance status | New StructuredDocument chunks | Legacy migration | Persistence decision |
| --- | --- | --- | --- |
| `full_provenance` | Accepted when document ID, source filename, source checksum, source block IDs, page/PDF provenance, parser name/version, and coherent section evidence are present. | Not used. | Accepted. |
| `partial_provenance` | Accepted only when required document/page/block provenance exists and approved limitation code explains the gap. | Not preferred. | Sample/local only or governed acceptance with `review_required=true`. |
| `legacy_filename_only` | MUST NOT be assigned. | Reserved for migrated legacy chunks. | Migration-only, not structured persistence. |
| `unknown_provenance` | MUST fail. | MUST fail unless future legacy quarantine explicitly allows inspection-only records. | Rejected. |

`partial_provenance` MAY cover missing optional section or printed-label
metadata. It MUST NOT cover missing document ID, source checksum, page
provenance, or source block IDs for new structured chunks.

## 10. Heading Policy

Headings are section metadata by default. They MUST NOT be forced into
standalone semantic chunks.

Rules:

1. Heading-only chunks MAY be created only for document title pages, major
   chapter anchors, appendix/annex anchors, or explicit navigation use.
2. Heading metadata SHOULD be carried through `section_id`, `section_path`,
   `section_number`, and `section_title`.
3. Heading text MAY be rendered as derived retrieval context in a future field,
   but raw chunk `text` MUST NOT be altered.
4. Duplicate headings MUST NOT produce duplicate evidence chunks.
5. Heading-before-body behavior SHOULD attach heading context to the following
   body chunk through metadata.
6. Headings with no body content SHOULD remain metadata unless approved as
   anchors.
7. Appendix and annex headings SHOULD remain first-class section metadata.

## 11. Paragraph and List Policy

Ordinary paragraphs SHOULD map one-to-one from D.4c candidates unless a later
semantic merge policy is explicitly approved.

Rules:

1. Lists SHOULD preserve list context.
2. Nested-list relationships SHOULD remain metadata where available.
3. Unrelated sections MUST NOT be merged.
4. Token-based merging is not part of D.5.
5. D.5 prefers evidence preservation over aggressive merging.

## 12. Table Policy

D.5 separates table source blocks, table entities, and table persisted chunks.

Rules:

1. Root table entities MUST NOT create automatic duplicate persisted chunks.
2. Persisted table chunks MUST derive from source blocks.
3. Table entity IDs MUST attach as metadata.
4. Candidate-level tables remain candidate-level until future review.
5. Row/cell structure MUST remain absent unless explicitly extracted and
   validated.
6. `TABLE_CANDIDATE_ONLY` remains an accepted limitation.
7. `TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE` requires downstream review controls.
8. Candidate table classification alone MUST NOT prove tabular content.
9. Captions MAY attach to table chunks when source evidence links them.
10. Multi-page table chunks MAY exist only with explicit page/PDF ranges.
11. Continuation flags MUST be preserved only when parser evidence supplies
   them.
12. Table deduplication MAY occur only when source block IDs and source spans
   prove duplicate representation.

When table evidence is candidate-level, ambiguous, or affected by accepted
limitations:

```text
content_type: table
review_required: true
```

## 13. Figure-Caption Policy

1. Caption blocks MAY persist as `figure_caption`.
2. Figure IDs MUST attach as metadata when available.
3. Asset-only figure chunks MUST NOT be created.
4. Image descriptions MUST NOT be generated unless source text contains them.
5. Persistence MUST NOT claim image understanding.
6. Caption blocks and figure root entities MUST NOT create duplicate chunks for
   the same source block.
7. Figure content accuracy remains out of scope.

## 14. Equation Policy

1. Equation chunks MUST persist exact raw equation text when available.
2. Normalized representation MAY be separate and optional.
3. Equation labels SHOULD be retained when source supplies them.
4. Source equation blocks MUST be retained in `source_block_ids`.
5. Equation IDs MUST attach as metadata.
6. Persistence MUST NOT solve equations.
7. Persistence MUST NOT convert equations to prose.
8. Persistence MUST NOT fabricate LaTeX.
9. Layout limitations MAY require `review_required=true`.

## 15. Admonition Policy

Admonitions SHOULD produce one persisted chunk per admonition entity when root
entity evidence is available.

Rules:

1. Exact body wording MUST be preserved.
2. Normalized type MUST be stored.
3. Raw label MUST be retained when available.
4. All contributing source block IDs MUST be stored.
5. Page and section provenance MUST be stored.
6. Body block deduplication MUST prevent duplicate paragraph chunks for the
   exact admonition body blocks.
7. Safety-critical wording MUST NOT be altered.

Type mapping:

| Source type | Persisted type |
| --- | --- |
| `WARNING` | `warning` |
| `CAUTION` | `caution` |
| `NOTE` | `note` |
| `IMPORTANT` | `important` |
| `SAFETY_NOTICE` | `safety_notice` |
| `UNKNOWN_ADMONITION` | `unknown_admonition` |

The current D.4c adapter maps `IMPORTANT` to `note`, `SAFETY_NOTICE` to
`warning`, and unknown admonitions to `other`. The persisted policy above is a
planned stricter taxonomy and MUST be implemented only in a later mapper phase.
`unknown_admonition` SHOULD require review.

## 16. Cross-Reference Policy

1. Cross-reference IDs MUST attach to the source chunk.
2. Cross-reference entities are not usually standalone persisted chunks.
3. Raw cross-reference wording MUST remain in source text.
4. Status MUST be preserved as one of `resolved`, `unresolved`, `external`,
   `ambiguous`, or `not_attempted`.
5. AviationRAG MUST NOT silently resolve references during persistence.
6. Resolved target IDs MUST exist.
7. External references MUST remain external.
8. Ambiguous references MUST remain reviewable.

## 17. Duplicate Text Policy

`DUPLICATE_TEXT_LINES` is an accepted upstream limitation and MUST be governed.

Rules:

1. Duplicate source-proxy text is not automatically parser duplication.
2. Identical text from different source locations MUST NOT be automatically
   removed.
3. Deduplication MAY occur only when source block IDs are identical, source
   spans are identical, or an entity relationship proves multi-representation.
4. Text-only deduplication is forbidden.
5. Duplicate warnings and requirements at different source locations MUST
   remain separate.
6. Duplicate-detection findings MUST attach as warnings, not silent deletion.

## 18. Section Crossing Policy

`CHUNK_SECTION_CROSSING_REVIEW` is an accepted upstream limitation and MUST be
governed.

Rules:

1. Persisted chunks MUST NOT cross unrelated sibling sections.
2. Parent/child crossing MAY be reviewable.
3. Current D.4c candidates remain block-preserving, so most crossings should
   be absent.
4. Any future semantic merge MUST enforce this rule.
5. Section-crossing candidates require `review_required=true`, limitation code,
   and explicit validation warning.
6. Unresolved unrelated crossing MUST block persistence.

## 19. Accepted Limitation Governance

Governance fields:

```text
accepted_limitation_codes
warning_codes
review_required
```

Rules:

1. Limitation codes MUST come from an approved registry.
2. Unknown limitation codes MUST fail validation.
3. Accepted limitations MUST NOT erase warnings.
4. A limitation MAY authorize persistence only where policy permits.
5. Blocking findings MUST NOT be converted into accepted limitations silently.
6. Limitation acceptance MUST be recorded in the persistence report.

Initial approved registry:

| Code | Classification | Persistence policy |
| --- | --- | --- |
| `CHUNK_SECTION_CROSSING_REVIEW` | conditionally accepted; review required | Parent/child crossing may persist in local sample packages with warnings; unrelated crossing rejects. |
| `DUPLICATE_TEXT_LINES` | conditionally accepted | No text-only deduplication; preserve separate source locations. |
| `TABLE_CANDIDATE_ONLY` | conditionally accepted; review required for table-specific downstream use | Candidate table chunks may persist locally with review flags; not proof of reconstructed table structure. |
| `TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE` | conditionally accepted; review required | Table classification is not trusted without supporting evidence. |

## 20. Validation Status

Persisted validation states:

| State | Meaning | Future real persistence gate |
| --- | --- | --- |
| `valid` | No warnings or accepted limitations. | Accepted. |
| `valid_with_warnings` | Nonblocking warnings exist. | Accepted with audit record. |
| `review_required` | Review condition or accepted limitation needs disposition. | Sample/local persistence only until governance approval. |
| `rejected` | Required evidence missing or policy violation exists. | Forbidden. |

`rejected` records MUST NOT be persisted. Rejected candidates MUST be reported.

## 21. Content-Type Taxonomy

Planned persisted content types:

| Content type | Eligibility | Required metadata | Review and deduplication |
| --- | --- | --- | --- |
| `paragraph` | Accepted | Source block IDs, page provenance | No text-only deduplication. |
| `list` | Accepted | List context when available | Preserve nested context. |
| `table` | Conditional | Table IDs when available, source blocks | Review required for candidate-only table evidence. |
| `figure_caption` | Accepted | Figure IDs when available | No asset-only chunk. |
| `equation` | Accepted with raw text | Equation IDs, raw text | No solving or fabricated notation. |
| `warning` | Accepted | Admonition ID, raw label, body blocks | Atomic and exact. |
| `caution` | Accepted | Admonition ID, raw label, body blocks | Atomic and exact. |
| `note` | Accepted | Admonition ID when applicable | Atomic when source entity exists. |
| `important` | Accepted | Admonition ID and raw label | Review if source type is ambiguous. |
| `safety_notice` | Accepted | Admonition ID and raw label | Review if source type is ambiguous. |
| `unknown_admonition` | Conditional | Raw label and body | SHOULD require review. |
| `procedure` | Accepted | Source block IDs, section path | Preserve step/list context. |
| `requirement` | Accepted | Clause/section evidence when available | Duplicate locations remain separate. |
| `definition` | Accepted | Term/source block evidence when available | Preserve exact wording. |
| `footnote` | Conditional | Source block/page provenance | Review if detached from body. |
| `reference` | Conditional | Cross-reference IDs/status | Usually metadata, not standalone chunk. |
| `appendix_content` | Accepted | Appendix/annex section evidence | Preserve appendix identity. |
| `mixed` | Conditional | All contributing source block IDs | SHOULD require review. |
| `unknown` | Conditional | Source block IDs and page provenance | SHOULD require review. |

This persisted taxonomy is stricter and more specific than the current
`ALLOWED_CHUNK_TYPES` in `chunk_schema.py`. Implementation MUST include a
separate approval and compatibility plan before changing validators.

## 22. Candidate-to-Persisted Mapping Matrix

| Candidate field | Persisted field | Mapping status | Transformation | Required? | Failure behavior |
| --- | --- | --- | --- | --- | --- |
| `chunk_candidate_id` | `chunk_id` | derive_safely | Deterministic persisted ID policy; candidate ID retained in warning/audit metadata if useful. | Yes | reject on missing canonical inputs |
| `document_id` | `document_id` | direct | Preserve exactly. | Yes | reject |
| `source_filename` | `source_filename` | direct | Require manifest-safe basename/logical name. | Yes | reject |
| `source_checksum` | `source_checksum` | direct | Require verified source checksum for new structured chunks. | Yes | reject |
| `document_title` | `document_title` | direct | Preserve when supplied. | No | null |
| `document_number` | `document_number` | direct | Preserve when supplied. | No | null |
| `document_revision` | `document_revision` | rename | Map to planned `document_revision`. | No | null |
| unavailable | `document_issue` | omit | Not currently available from D.4c candidate. | No | null |
| unavailable | `effective_date` | omit | Not currently available from D.4c candidate. | No | null |
| `text` | `text` | direct | Preserve exact candidate text. | Yes | reject if empty |
| `normalized_text` | `normalized_text` | direct | Preserve separately. | No | null |
| `content_type` | `content_type` | conditional | Map D.4c types to persisted taxonomy. | Yes | reject or review for unsupported |
| unavailable | `content_subtype` | omit | Not currently available. | No | null |
| unavailable | `language` | omit | Not currently available. | No | null |
| `page_start` | `page_start` | direct | Preserve. | Yes | reject |
| `page_end` | `page_end` | direct | Preserve. | Yes | reject |
| `pdf_page_index_start` | `pdf_page_index_start` | direct | Preserve. | Yes | reject |
| `pdf_page_index_end` | `pdf_page_index_end` | direct | Preserve. | Yes | reject |
| `page_start`/`page_end` | `contributing_page_numbers` | derive_safely | Expand page range or use explicit source spans later. | Yes | reject if unavailable |
| `pdf_page_index_start`/`pdf_page_index_end` | `contributing_pdf_page_indexes` | derive_safely | Expand PDF index range or use explicit source spans later. | Yes | reject if unavailable |
| `printed_page_labels` | `printed_page_labels` | direct | Preserve tuple/list. | No | empty list |
| `section_id` | `section_id` | direct | Preserve when supplied. | Conditional | warning or review if missing with section evidence |
| `section_path` | `section_path` | direct | Preserve ordered list. | Conditional | warning or review |
| `section_number` | `section_number` | direct | Preserve when supplied. | No | null |
| `section_title` | `section_title` | direct | Preserve when supplied. | Conditional | warning when section ID exists without title |
| `clause_identifier` | `clause_identifier` | direct | Preserve when supplied. | No | null |
| `source_block_ids` | `source_block_ids` | direct | Preserve ordered IDs. | Yes | reject |
| candidate location fields | `source_span` | derive_safely | Build from page/PDF range and source block IDs; do not fabricate offsets/bboxes. | Yes | reject |
| `table_ids` | `table_ids` | direct | Preserve. | Conditional | review if table content lacks table ID |
| `figure_ids` | `figure_ids` | direct | Preserve. | Conditional | warning if figure-caption lacks figure ID |
| `equation_ids` | `equation_ids` | direct | Preserve. | Conditional | review if equation lacks equation ID |
| `admonition_ids` | `admonition_ids` | direct | Preserve. | Conditional | reject if admonition chunk lacks ID |
| `cross_reference_ids` | `cross_reference_ids` | direct | Preserve. | Conditional | empty list allowed |
| `parser_name` | `parser_name` | direct | Preserve. | Yes | reject |
| `parser_version` | `parser_version` | direct | Preserve. | Yes | reject |
| source artifact `schema_version` | `structured_document_schema_version` | derive_safely | Copy from source artifact/report, not candidate field. | Yes | reject |
| D.4c adapter policy | `adapter_version` | derive_safely | Use documented adapter phase/version. | Yes | reject if undefined |
| future mapper policy | `persistence_mapper_version` | derive_safely | Use future mapper version. | Yes | reject if undefined |
| `extraction_method` | `extraction_method` | direct | Preserve. | Yes | reject |
| `provenance_status` | `provenance_status` | conditional | Map `structured` to `full_provenance`; `structured_partial` to governed partial. | Yes | reject unknown |
| unavailable | `accepted_limitation_codes` | derive_safely | Future validator applies approved registry. | Yes | empty list or governed list |
| unavailable | `validation_status` | derive_safely | Future validation result. | Yes | reject if absent after mapper |
| unavailable | `warning_codes` | derive_safely | Future validation warnings. | Yes | empty list allowed |
| unavailable | `review_required` | derive_safely | True when policy/limitations require review. | Yes | reject if absent after mapper |
| unavailable | `persisted_at` | omit | Excluded from deterministic core. | No | not emitted |
| embedding/vector/index fields | none | forbidden | Not available and not allowed. | No | reject if present |

## 23. Persistence Package Design

Future D.5b package structure:

```text
persisted_chunk_package/
  persisted_chunks.jsonl
  persistence_manifest.json
  persistence_report.json
  rejected_candidates.jsonl
  warnings.json
```

Artifacts:

1. `persisted_chunks.jsonl` MUST contain one validated persisted record per
   line and MUST NOT include embeddings or vectors.
2. `persistence_manifest.json` MUST contain package schema identity, source
   StructuredDocument checksum, source manifest checksum, mapper version,
   persisted schema version, record count, accepted/rejected counts, package
   checksum, and limitation registry version.
3. `persistence_report.json` MUST contain mapping summary, validation results,
   warning counts, limitation counts, content-type counts, provenance counts,
   section-crossing findings, duplicate findings, and table/figure/equation/
   admonition counts.
4. `rejected_candidates.jsonl` MUST contain sanitized candidate IDs and
   rejection reasons.
5. `warnings.json` MAY contain detailed warning registry entries.

Rejected records MUST NOT be silently discarded.

## 24. Package Determinism

1. Record order MUST be deterministic.
2. JSON key order MUST be deterministic.
3. Encoding MUST be UTF-8.
4. JSONL files MUST end with a final newline.
5. Deterministic package core MUST NOT contain timestamps.
6. Package core MUST NOT contain absolute paths or random values.
7. Checksums MUST be based on exact bytes.
8. Repeated mapping MUST produce byte-identical package bytes when inputs and
   policy are unchanged.
9. Execution timestamp MAY be stored only in a separate audit envelope if
   required by a later phase.

## 25. Rollback and Audit

Rollback requirements:

1. Source StructuredDocument artifact retained.
2. Source parser manifest retained.
3. Source and artifact checksums retained.
4. Mapping policy version retained.
5. Persisted schema version retained.
6. Rejected candidate list retained.
7. Package checksum retained.
8. Old package retained until replacement validation passes.
9. No destructive overwrite by default.

A rebuild MUST be reproducible from:

```text
source document
+ parser version
+ StructuredDocument artifact
+ adapter version
+ persistence mapper version
+ limitation policy version
```

## 26. Migration Compatibility

StructuredDocument-derived persisted chunks differ from legacy chunks because
they require source block IDs, page/PDF provenance, parser identity, and
structured provenance status.

Migration states:

| State | Meaning |
| --- | --- |
| `new_structured` | Derived from validated StructuredDocument candidates. |
| `legacy_adapted` | Converted from legacy chunk records with explicit migration policy. |
| `legacy_unresolved` | Legacy record lacks required provenance for trusted migration. |

New structured records MUST NOT be mixed silently with legacy records. Every
package and future vector payload MUST identify record origin. No legacy
migration occurs in D.5.

## 27. Vector Payload Boundary

Future vector payloads MAY derive:

1. `chunk_id`.
2. `text`.
3. Selected provenance metadata.
4. Selected content/entity metadata.
5. Embedding vector in a later phase.

Persisted records MUST NOT include:

1. Embedding.
2. Vector.
3. Index-specific ID.
4. Astra-specific metadata.
5. FAISS position.

The persisted record MUST remain storage-neutral.

## 28. Quality Gates Before D.5b

D.5b MUST NOT begin until these gates are accepted:

1. Persisted schema approved.
2. Deterministic chunk ID policy approved.
3. Required provenance approved.
4. Accepted limitation registry approved.
5. Heading policy approved.
6. Table deduplication policy approved.
7. Admonition deduplication policy approved.
8. Cross-reference policy approved.
9. Persistence-package format approved.
10. Rejection and warning behavior approved.
11. Rollback requirements approved.
12. Synthetic fixture plan approved.
13. No embeddings or vector stores involved.

## 29. Upstream Pilot Evidence and Authorization

The accepted upstream parser checkpoint is `techdoc-parser` `main` at
`27c4146`, with P0 pilot outcome `ACCEPTED_WITH_LIMITATIONS`.

Accepted P0 evidence:

1. 32/32 representative pages reviewed.
2. 28 `PASS`, 4 `REVIEW`, 0 `FAIL`.
3. 0 blocking findings.
4. Active accepted limitations:
   - `CHUNK_SECTION_CROSSING_REVIEW`
   - `DUPLICATE_TEXT_LINES`
   - `TABLE_CANDIDATE_ONLY`
5. Confirmed nonblocking issue:
   - `TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE`

Downstream authorization:

| Activity | Authorized |
| --- | --- |
| AviationRAG persisted `ChunkRecord` mapping design | Yes |
| Controlled local sample-persistence dry run | Yes |
| Full corpus ingestion | No |
| Embedding regeneration | No |
| Astra rebuild | No |
| FAISS rebuild | No |
| Production retrieval activation | No |

The P0 pilot supports design and controlled sample dry-run work only. It does
not establish full-document accuracy, OCR accuracy, full-corpus readiness, or
production migration readiness.

## 30. Current Versus Planned Behavior

Currently implemented:

1. Lightweight runtime-anchor `ChunkRecord` dataclass remains unchanged.
2. D.4b validator validates `techdoc-structured-document / 0.1.0` coherence.
3. D.4c adapter emits review-only `StructuredDocumentChunkCandidate` records.
4. Legacy chunk conversion tools are local dry-run or gated ignored-output
   utilities only.

Planned behavior:

1. A future mapper MAY convert approved candidates into
   `PersistedChunkRecord` records.
2. A future D.5b package MAY write deterministic local synthetic persistence
   artifacts.
3. Future vector payloads MAY derive from persisted records only after
   persistence is validated.

Not implemented or authorized in D.5:

1. Persisted-record mapper.
2. Runtime ingestion integration.
3. Real corpus persistence.
4. Embeddings.
5. Astra writes or rebuild.
6. FAISS reads or rebuild.
7. Retrieval or generation behavior changes.
