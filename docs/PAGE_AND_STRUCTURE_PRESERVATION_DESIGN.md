# Page and Structure Preservation Design

Date: 2026-07-18
Status: Design complete; not implemented in runtime ingestion
Phase: D.4

This document defines the target design for preserving page and structural
provenance before any real document reprocessing, chunk migration, embedding
regeneration, Astra rebuild, or FAISS rebuild is attempted.

No code, parser integration, migration execution, embedding generation, or
runtime retrieval behavior is implemented by this document.

## 1. Purpose

The purpose of D.4 is to define how AviationRAG should represent structural
provenance from source documents so future retrieval results can cite the
original page, section, paragraph, table, figure, warning, caution, note, and
source span with enough precision for safety-sensitive review.

The design is intentionally parser-neutral at the AviationRAG boundary. A
future parser such as `techdoc-parse` MAY extract the raw structure, but
AviationRAG MUST own the normalized ingestion contract and chunk provenance
used by retrieval, generation, evaluation, and governance.

## 2. Scope

D.4 covers design for:

- page numbers and printed page labels
- section and subsection hierarchy
- paragraph, clause, list, and requirement identifiers
- table provenance and cell-level traceability
- figure, caption, and equation provenance
- warning, caution, and note classification
- appendices, annexes, and attachments
- source-document traceability
- future chunk metadata fields required for citation-ready retrieval
- compatibility with legacy chunks that lack these fields

D.4 does not cover:

- implementing a parser
- changing existing chunk schemas or validators
- reprocessing source documents
- converting real legacy chunks
- generating embeddings
- rebuilding Astra or FAISS indexes
- wiring real retrieval to new provenance fields
- enforcing response-policy citations in generation

## 3. Pipeline Boundary

The target future pipeline is:

```text
source file
  -> parser output
  -> StructuredDocument
  -> AviationRAG ingestion adapter
  -> ChunkRecord with structural provenance
  -> vector payloads
  -> retrieval results
  -> response citations
```
The parser boundary MUST be explicit. Parser output MAY be richer than the
AviationRAG contract, but the ingestion adapter MUST normalize it before chunk
creation.

The current runtime ingestion path does not implement this pipeline.

## 4. Responsibility Matrix

| Layer | Responsibility | Must not own |
| ----- | -------------- | ------------ |
| Source file | Original content and document control markings | Normalized metadata |
| Parser | Detect pages, blocks, layout, tables, figures, captions, and warnings | Retrieval-specific chunk policy |
| StructuredDocument | Preserve source order and normalized provenance | Embeddings or vector IDs |
| AviationRAG ingestion adapter | Validate and map structured content into chunk-ready records | Low-level PDF parsing |
| Chunk builder | Create chunks with stable text hashes and source span metadata | Source-file mutation |
| Vector writer | Emit embedding/vector payload metadata | Structure extraction |
| Retrieval | Return citation-ready metadata with matched text | Citation fabrication |
| Generation | Use retrieved provenance to cite answers | Inventing page or section references |

## 5. Canonical Structured Document Model

The future canonical model SHOULD contain these top-level entities:

| Entity | Purpose | Required fields |
| ------ | ------- | --------------- |
| `document` | Stable document identity and governance metadata | `document_id`, `filename`, `source_hash`, `parser_schema_version` |
| `pages` | Physical page observations | `page_id`, `pdf_page_index`, `page_number`, `source_span` |
| `sections` | Logical hierarchy | `section_id`, `title`, `level`, `path`, `source_span` |
| `blocks` | Ordered content blocks | `block_id`, `block_type`, `text`, `source_span`, `reading_order` |
| `tables` | Structured tabular content | `table_id`, `caption`, `page_span`, `cells` |
| `figures` | Image, diagram, and caption provenance | `figure_id`, `caption`, `page_span` |
| `admonitions` | Warning, caution, and note blocks | `admonition_id`, `admonition_type`, `text`, `source_span` |
| `relationships` | Cross-references and containment | `from_id`, `to_id`, `relationship_type` |

The model MUST preserve original text where practical. Normalized fields MAY be
added, but they MUST NOT replace the source text needed for audit.

## 6. Page Identity Model

Page provenance MUST distinguish physical PDF position from printed labels:

| Field | Meaning |
| ----- | ------- |
| `pdf_page_index` | Zero-based page index in the source file |
| `page_number` | One-based physical page number in extraction order |
| `printed_page_label` | Visible label printed in the document, if detected |
| `page_title` | Header or running title if detected with confidence |
| `page_role` | `cover`, `toc`, `body`, `appendix`, `index`, or `unknown` |

A chunk spanning multiple pages MUST record both `page_start` and `page_end`.
For citation display, `printed_page_label` SHOULD be preferred when it is
present and reliable; otherwise `page_number` SHOULD be used.

## 7. Section Hierarchy Model

Sections SHOULD be represented as a hierarchy of nodes, not as a single flat
string. Each section node SHOULD include:

- `section_id`
- `level`
- `section_number`
- `title`
- `parent_section_id`
- `path`
- `source_span`
- `confidence`

Example:

```json
{
  "section_id": "sec-2-1",
  "level": 2,
  "section_number": "2.1",
  "title": "Fuel Vent Inspection",
  "parent_section_id": "sec-2",
  "path": ["2 Inspection Procedures", "2.1 Fuel Vent Inspection"],
  "confidence": 0.96
}
```

If a heading is inferred from style or layout rather than explicit numbering,
the parser MUST mark the confidence and SHOULD preserve the raw heading text.

## 8. Paragraph And Clause Identity

Paragraph and clause identifiers SHOULD be preserved when the document provides
them. The future model SHOULD support:

- numbered paragraphs, such as `4.2.1`
- lettered clauses, such as `a`, `b`, and `c`
- nested list paths, such as `4.2.1/a/iii`
- procedure steps
- requirement identifiers
- local generated block IDs where no source identifier exists

Generated IDs MUST be stable for the same source text and source position. They
MUST be clearly distinguishable from identifiers printed in the source.

## 9. Source Span Model

Every block intended for chunking SHOULD carry a source span:

| Field | Meaning |
| ----- | ------- |
| `source_file` | Original file path or manifest reference |
| `source_hash` | Hash of the source file used for processing |
| `page_start` | One-based physical start page |
| `page_end` | One-based physical end page |
| `pdf_page_index_start` | Zero-based start page index |
| `pdf_page_index_end` | Zero-based end page index |
| `char_start` | Character offset within the extracted page or document text when available |
| `char_end` | Character end offset when available |
| `bbox` | Page coordinate bounding box when available |
| `extraction_method` | Parser and extraction mode |

Bounding boxes MAY be absent for plain text formats or low-confidence OCR.
Missing coordinates MUST NOT block text-only provenance.

## 10. Reading Order

The parser SHOULD emit a deterministic `reading_order` value for each block.
The ingestion adapter MUST preserve this order when creating chunks.

When multi-column layouts, sidebars, footnotes, or callouts make reading order
ambiguous, the parser SHOULD emit a confidence score and MAY emit alternate
relationships. The chunk builder MUST prefer deterministic output over layout
guesswork.

## 11. Headers And Footers

Headers and footers SHOULD be detected separately from body text when possible.
The model SHOULD support:

- repeated running headers
- page numbers
- revision banners
- document-control footers
- confidentiality or distribution markings

Repeated headers and footers SHOULD NOT be duplicated into every retrieval
chunk unless they contain unique safety or document-control content. Their
source span SHOULD remain available for audit.

## 12. Table Preservation

Tables require both text retrieval and structural provenance. The future model
SHOULD preserve:

- table ID
- caption or title
- page span
- row and column headers
- cell text
- merged-cell relationships
- footnotes
- units
- reading order

Chunking strategies MAY include one table-level chunk, row-group chunks, or
cell-context chunks. The strategy MUST preserve enough row and column context
that retrieved values are not separated from their meaning.

Synthetic example:

```json
{
  "chunk_type": "table",
  "table_id": "tbl-synth-1",
  "caption": "Synthetic Inspection Interval Matrix",
  "page_start": 12,
  "page_end": 13,
  "table_context": {
    "column_headers": ["Component", "Interval", "Action"],
    "row_range": [1, 3]
  }
}
```

## 13. Figure And Caption Provenance

Figures, diagrams, charts, and images SHOULD be represented even when image
content is not directly embedded in text chunks. The future model SHOULD
preserve:

- `figure_id`
- figure type
- caption text
- referenced section
- page span
- bounding box where available
- OCR text or alt text where available
- confidence score

Caption chunks MUST remain linked to their figure. A generated answer MUST NOT
cite a figure unless the retrieved metadata identifies the figure or caption.

## 14. Equation Provenance

Equations SHOULD be preserved as explicit blocks when detected. The model MAY
carry both source text and normalized notation, but normalized notation MUST NOT
replace the source form.

Equation metadata SHOULD include:

- `equation_id`
- page span
- section path
- source text
- normalized text when available
- nearby paragraph or caption relationship

## 15. Warning, Caution, And Note Classification

Warnings, cautions, and notes MUST be classified explicitly when detected. The
future taxonomy SHOULD use:

- `warning`
- `caution`
- `note`
- `notice`
- `danger`
- `important`
- `unknown_admonition`

Admonition metadata SHOULD include:

- `admonition_id`
- `admonition_type`
- source label text
- normalized severity
- page span
- section path
- related procedure step or paragraph

Synthetic example:

```json
{
  "chunk_type": "warning",
  "admonition_type": "warning",
  "source_label": "WARNING",
  "text": "Synthetic warning text for design validation only.",
  "page_start": 8,
  "section_path": ["3 Maintenance Safety", "3.2 Isolation"]
}
```

## 16. Appendices, Annexes, And Attachments

Appendices and annexes SHOULD be first-class section nodes. The model SHOULD
preserve:

- appendix or annex label
- title
- page span
- parent document relationship
- references from body sections

Appendix content MUST NOT be flattened into body sections without preserving
its appendix identity.

## 17. Cross-References

Cross-references SHOULD be captured when detected. Examples include:

- "see Section 4.2"
- "refer to Table 3"
- "as shown in Figure 5"
- appendix references
- regulatory references

The model SHOULD represent the raw reference text and the resolved target when
resolution is reliable. Unresolved references SHOULD remain as raw references
with an `unresolved` status.

## 18. Revision And Document-Control Provenance

The future model SHOULD preserve document-control metadata when available:

- revision or amendment number
- effective date
- issue date
- supersession status
- approval authority
- source manifest ID
- source hash
- extraction timestamp

The ingestion adapter MUST NOT infer document currency from filename alone.

## 19. Confidence And Uncertainty

Parser-derived structural observations SHOULD include confidence scores. The
model SHOULD distinguish:

- detected from explicit source text
- inferred from layout
- inferred from font or style
- inferred from OCR
- generated by AviationRAG as a stable local ID

Low-confidence structure MUST be preserved as uncertain rather than silently
promoted to authoritative metadata.

## 20. Chunk Provenance Contract

Future chunk records SHOULD retain the existing core fields from
`CHUNK_METADATA_SCHEMA.md` and MAY add D.4 provenance fields after a controlled
schema update.

Future provenance fields SHOULD include:

- `source_block_ids`
- `source_span`
- `pdf_page_index_start`
- `pdf_page_index_end`
- `printed_page_label_start`
- `printed_page_label_end`
- `section_id`
- `section_number`
- `section_title`
- `paragraph_id`
- `clause_id`
- `list_path`
- `table_id`
- `figure_id`
- `equation_id`
- `admonition_id`
- `parser_name`
- `parser_schema_version`
- `structure_confidence`

Current D.3d local conversion output does not provide these fields unless they
already exist in legacy metadata.

## 21. Multi-Page Chunk Rules

A chunk MAY span multiple pages only when splitting it would destroy meaning.
Examples include:

- a table that continues onto the next page
- a warning block broken by a page boundary
- a short procedure step with continuation text

Multi-page chunks MUST record both start and end pages. If a chunk includes
content from non-contiguous pages, it SHOULD record explicit source spans for
each page segment rather than a single broad page range.

## 22. Content-Type Taxonomy

The future content-type taxonomy SHOULD align with chunk types and parser block
types. Recommended values are:

- `paragraph`
- `section_heading`
- `procedure_step`
- `requirement`
- `definition`
- `table`
- `table_caption`
- `figure_caption`
- `equation`
- `warning`
- `caution`
- `note`
- `appendix_heading`
- `metadata`
- `unknown`

Unknown types MUST remain searchable, but they SHOULD be flagged for review.

## 23. Citation-Ready Retrieval Contract

Retrieval results SHOULD expose enough metadata for generation to cite without
inventing provenance. A citation-ready result SHOULD include:

- document title or filename
- source hash or manifest ID
- page label or page number
- section path
- paragraph, table, figure, or warning identifier where available
- matched text excerpt
- chunk ID
- confidence indicators

If page or section metadata is missing, the result MUST say it is missing
rather than synthesizing a citation.

## 24. Compatibility With Legacy Chunks

Legacy chunks are expected to have incomplete structural provenance. Future
schemas SHOULD represent this explicitly with provenance status values:

- `structured`: parser-derived page and structure metadata is present
- `legacy_partial`: some metadata exists, but page/structure is incomplete
- `legacy_unstructured`: only document-level or filename metadata exists
- `synthetic_fixture`: design or test fixture only

Legacy migration MUST NOT claim structured provenance unless it is actually
available from source-derived evidence.

## 25. Parser Output Contract

A future parser output file SHOULD identify its schema and parser identity:

```json
{
  "schema_name": "techdoc-structured-document",
  "schema_version": "0.1.0",
  "parser_name": "techdoc-parse",
  "parser_version": "future",
  "document": {},
  "pages": [],
  "sections": [],
  "blocks": [],
  "relationships": []
}
```

The contract MUST support validation before chunk creation. Invalid parser
output MUST fail before embedding, indexing, Astra write, or FAISS rebuild.

## D.4b Synthetic Structural Validation

D.4b adds an offline validator for synthetic structured-document records:

```text
src/aviationrag/ingestion/structured_document_validator.py
tools/chunking/validate-structured-document.py
```

The validator checks internal structural coherence only. It validates schema
identity, document metadata, page ordering, block references, section hierarchy,
source spans, confidence values, tables, figures, equations, admonitions, and
cross-references. Validation errors make a record invalid. Validation warnings
identify incomplete optional provenance or review concerns without invalidating
an otherwise coherent record.

Unsupported schema names or schema versions are errors. The default supported
contract is:

```text
schema_name: techdoc-structured-document
schema_version: 0.1.0
```

The validator does not mutate the input document. It does not parse PDFs or
DOCX files, run OCR, implement `techdoc-parse`, judge source-document extraction
accuracy, generate chunks, generate embeddings, connect to Astra, use FAISS,
perform migration, or integrate with runtime ingestion.

## 26. Versioning And Backward Compatibility

The structured-document contract SHOULD use explicit schema versions. Breaking
changes MUST require a new schema version and migration notes.

Chunk records SHOULD retain their existing versioned schema behavior. Adding
D.4 fields to runtime validators MUST be done in a separate implementation
phase with regression tests.

## 27. Validation Rules

Future validation SHOULD check:

- every chunk has stable document identity
- every structured chunk has at least one source span
- page ranges are valid and ordered
- section paths reference known section nodes
- table IDs reference known table records
- figure IDs reference known figure records
- admonition types use the approved taxonomy
- source hashes match manifest records
- synthetic fixtures are clearly marked as synthetic

Validation MUST run before vector payload writing.

## 28. Quality Gates

Before real migration or reprocessing, the project SHOULD pass these gates:

- D.4 design approved
- structured-document schema drafted
- synthetic structured fixture validated
- parser adapter design completed
- sample source documents processed in a controlled dry run
- chunk provenance reviewed manually
- evaluation fixtures updated for citation metadata
- no Astra or FAISS rebuild until metadata output is accepted

These gates are future work beyond this document.

## 29. Edge-Case Catalogue

Future implementation SHOULD account for:

- missing page labels
- duplicated printed page labels
- roman numeral front matter
- rotated pages
- scanned pages with OCR uncertainty
- multi-column procedures
- tables split across pages
- captions detached from figures
- warnings spanning page breaks
- appendices with independent numbering
- superseded document revisions
- extraction failures for selected pages
- documents without stable section numbers

The safe default is to preserve text with an explicit low-confidence or missing
metadata marker.

## 30. Open Design Decisions

Open decisions for later phases:

- whether `techdoc-parse` output should be stored permanently or treated as an
  intermediate artifact
- whether table cell coordinates are required for all table chunks or only for
  high-value manuals
- whether FAISS metadata should mirror all structure fields or only citation
  essentials
- how response policy should rank page citations versus section citations when
  both are available
- whether OCR confidence should influence retrieval ranking or only citation
  display
- how to evaluate structural provenance accuracy against source documents

These decisions MUST be resolved before real corpus reprocessing.

## Synthetic Example 1: Paragraph Chunk

```json
{
  "chunk_id": "synth-doc-001:blk-0007",
  "document_id": "synth-doc-001",
  "filename": "synthetic-maintenance-guide.pdf",
  "chunk_type": "paragraph",
  "text": "Synthetic paragraph text for design validation only.",
  "page_start": 4,
  "page_end": 4,
  "section_path": ["2 Synthetic Inspection", "2.1 Visual Check"],
  "paragraph_id": "2.1.a",
  "source_block_ids": ["blk-0007"],
  "source_span": {
    "pdf_page_index_start": 3,
    "pdf_page_index_end": 3,
    "printed_page_label_start": "2-4",
    "printed_page_label_end": "2-4"
  },
  "provenance_status": "synthetic_fixture"
}
```

## Synthetic Example 2: Multi-Page Table Chunk

```json
{
  "chunk_id": "synth-doc-001:tbl-0002",
  "document_id": "synth-doc-001",
  "filename": "synthetic-maintenance-guide.pdf",
  "chunk_type": "table",
  "text": "Synthetic table summary for design validation only.",
  "page_start": 12,
  "page_end": 13,
  "section_path": ["5 Synthetic Limits"],
  "table_id": "tbl-0002",
  "caption": "Synthetic Operating Limit Matrix",
  "table_context": {
    "column_headers": ["Condition", "Limit", "Action"],
    "row_range": [1, 2],
    "continued_from_previous_page": false,
    "continues_to_next_page": true
  },
  "provenance_status": "synthetic_fixture"
}
```

## Synthetic Example 3: Warning Chunk

```json
{
  "chunk_id": "synth-doc-001:adm-0001",
  "document_id": "synth-doc-001",
  "filename": "synthetic-maintenance-guide.pdf",
  "chunk_type": "warning",
  "text": "Synthetic warning text for design validation only.",
  "page_start": 8,
  "page_end": 8,
  "section_path": ["3 Synthetic Safety", "3.2 Isolation"],
  "admonition_id": "adm-0001",
  "admonition_type": "warning",
  "source_label": "WARNING",
  "related_block_ids": ["blk-0019"],
  "provenance_status": "synthetic_fixture"
}
```
