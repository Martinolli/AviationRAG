"""Offline structured-document validation for synthetic provenance fixtures.

This module validates internal consistency for future structured-document
records. It does not parse source documents, judge extraction accuracy, mutate
input records, write files, generate embeddings, connect to Astra, use FAISS,
or integrate with runtime ingestion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Collection, Mapping


DEFAULT_SUPPORTED_SCHEMA_NAMES = {"techdoc-structured-document"}
DEFAULT_SUPPORTED_SCHEMA_VERSIONS = {"0.1.0"}

ISSUE_SEVERITIES = {"error", "warning"}
_SEVERITY_ORDER = {"error": 0, "warning": 1}

ALLOWED_CONTENT_TYPES = {
    "appendix_heading",
    "caution",
    "definition",
    "equation",
    "figure",
    "figure_caption",
    "metadata",
    "note",
    "paragraph",
    "procedure_step",
    "requirement",
    "section_heading",
    "table",
    "table_caption",
    "unknown",
    "warning",
}

ALLOWED_ADMONITION_TYPES = {
    "WARNING",
    "CAUTION",
    "NOTE",
    "IMPORTANT",
    "SAFETY_NOTICE",
    "UNKNOWN_ADMONITION",
}

ALLOWED_CROSS_REFERENCE_STATUSES = {
    "resolved",
    "unresolved",
    "external",
    "ambiguous",
    "not_attempted",
}

CONFIDENCE_FIELDS = {
    "classification_confidence",
    "confidence",
    "extraction_confidence",
    "ocr_confidence",
    "provenance_confidence",
    "structure_confidence",
}


@dataclass(frozen=True)
class ValidationIssue:
    """One deterministic structured-document validation issue."""

    code: str
    severity: str
    message: str
    path: str
    entity_id: str | None = None


@dataclass
class StructuredDocumentValidationResult:
    """Structured-document validation result with deterministic issue order."""

    schema_name: str | None
    schema_version: str | None
    document_id: str | None
    is_valid: bool
    error_count: int
    warning_count: int
    issues: list[ValidationIssue] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ValidationContext:
    issues: list[ValidationIssue] = field(default_factory=list)
    page_numbers: set[int] = field(default_factory=set)
    page_indices: set[int] = field(default_factory=set)
    page_ids: set[str] = field(default_factory=set)
    block_ids: set[str] = field(default_factory=set)
    section_ids: set[str] = field(default_factory=set)
    table_ids: set[str] = field(default_factory=set)
    figure_ids: set[str] = field(default_factory=set)
    equation_ids: set[str] = field(default_factory=set)
    admonition_ids: set[str] = field(default_factory=set)


def load_structured_document(path: str | Path) -> dict[str, Any]:
    """Load a structured-document JSON object from disk."""
    document_path = Path(path)
    with document_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Structured document must be a JSON object: {document_path}")
    return data


def validate_structured_document(
    document: Mapping[str, Any],
    *,
    supported_schema_names: Collection[str] | None = None,
    supported_schema_versions: Collection[str] | None = None,
) -> StructuredDocumentValidationResult:
    """Validate a structured-document mapping without mutating it."""
    schema_names = set(supported_schema_names or DEFAULT_SUPPORTED_SCHEMA_NAMES)
    schema_versions = set(supported_schema_versions or DEFAULT_SUPPORTED_SCHEMA_VERSIONS)
    context = _ValidationContext()

    if not isinstance(document, Mapping):
        _add_issue(
            context,
            "ROOT_NOT_OBJECT",
            "error",
            "Root value must be a JSON object.",
            "$",
        )
        return _build_result(None, None, None, context, {})

    schema_name = _optional_str(document.get("schema_name"))
    schema_version = _optional_str(document.get("schema_version"))
    document_record = document.get("document")
    document_id = _document_id(document)

    _validate_schema_identity(document, schema_names, schema_versions, context)
    _validate_document_metadata(document, context)

    pages = _validate_pages(document, context)
    sections = _validate_sections(document, context)
    blocks = _validate_blocks(document, context)

    tables = _validate_tables(document, blocks, context)
    figures = _validate_figures(document, context)
    equations = _validate_equations(document, context)
    admonitions = _validate_admonitions(document, blocks, context)
    cross_references = _validate_cross_references(document, context)

    _validate_source_spans(document, blocks, tables, figures, equations, admonitions, context)
    _validate_confidence_values(document, context)

    summary = {
        "page_count": len(pages),
        "block_count": len(blocks),
        "section_count": len(sections),
        "table_count": _entity_count(tables, blocks, "table"),
        "figure_count": _entity_count(figures, blocks, "figure"),
        "equation_count": _entity_count(equations, blocks, "equation"),
        "admonition_count": _admonition_count(admonitions, blocks),
        "cross_reference_count": len(cross_references),
    }

    return _build_result(schema_name, schema_version, document_id, context, summary)


def validate_structured_document_file(
    path: str | Path,
    *,
    supported_schema_names: Collection[str] | None = None,
    supported_schema_versions: Collection[str] | None = None,
) -> StructuredDocumentValidationResult:
    """Load and validate one structured-document JSON file."""
    return validate_structured_document(
        load_structured_document(path),
        supported_schema_names=supported_schema_names,
        supported_schema_versions=supported_schema_versions,
    )


def structured_document_validation_result_to_dict(
    result: StructuredDocumentValidationResult,
) -> dict[str, Any]:
    """Return a JSON-serializable validation report dictionary.

    Issues are ordered by severity, path, code, message, and entity ID.
    """
    issues = [_issue_to_dict(issue) for issue in _sort_issues(result.issues)]
    return {
        "schema_name": result.schema_name,
        "schema_version": result.schema_version,
        "document_id": result.document_id,
        "is_valid": result.is_valid,
        "error_count": result.error_count,
        "warning_count": result.warning_count,
        "summary": dict(result.summary),
        "issues": issues,
    }


def _validate_schema_identity(
    document: Mapping[str, Any],
    supported_schema_names: set[str],
    supported_schema_versions: set[str],
    context: _ValidationContext,
) -> None:
    schema_name = document.get("schema_name")
    schema_version = document.get("schema_version")

    if not _non_empty_string(schema_name):
        _add_issue(context, "SCHEMA_NAME_MISSING", "error", "schema_name is required.", "$.schema_name")
    elif schema_name not in supported_schema_names:
        _add_issue(
            context,
            "SCHEMA_NAME_UNSUPPORTED",
            "error",
            f"Unsupported schema_name: {schema_name!r}.",
            "$.schema_name",
        )

    if not _non_empty_string(schema_version):
        _add_issue(
            context,
            "SCHEMA_VERSION_MISSING",
            "error",
            "schema_version is required.",
            "$.schema_version",
        )
    elif schema_version not in supported_schema_versions:
        _add_issue(
            context,
            "SCHEMA_VERSION_UNSUPPORTED",
            "error",
            f"Unsupported schema_version: {schema_version!r}.",
            "$.schema_version",
        )


def _validate_document_metadata(document: Mapping[str, Any], context: _ValidationContext) -> None:
    metadata = document.get("document")
    if metadata is not None and not isinstance(metadata, Mapping):
        _add_issue(context, "DOCUMENT_METADATA_INVALID", "error", "document must be an object.", "$.document")
        metadata = {}
    if not isinstance(metadata, Mapping):
        metadata = {}

    document_id = _first_non_empty(document, metadata, "document_id")
    title = _first_non_empty(document, metadata, "document_title", "canonical_title", "title")
    filename = _first_non_empty(document, metadata, "source_filename", "filename")
    checksum = _first_non_empty(document, metadata, "source_checksum", "source_hash", "file_hash", "checksum")
    parser_version = _first_non_empty(document, metadata, "parser_version")
    page_count = document.get("page_count", metadata.get("page_count"))

    if not document_id:
        _add_issue(context, "DOCUMENT_ID_MISSING", "error", "Document ID is required.", "$.document.document_id")
    if not title:
        _add_issue(context, "DOCUMENT_TITLE_MISSING", "warning", "Document title is missing.", "$.document")
    if not filename:
        _add_issue(context, "SOURCE_FILENAME_MISSING", "error", "Source filename is required.", "$.document.filename")
    if page_count is not None and not _non_negative_int(page_count):
        _add_issue(
            context,
            "PAGE_COUNT_INVALID",
            "error",
            "page_count must be a non-negative integer.",
            "$.document.page_count",
        )
    if checksum is not None and not _non_empty_string(checksum):
        _add_issue(
            context,
            "SOURCE_CHECKSUM_MISSING",
            "warning",
            "Source checksum must be a non-empty string when present.",
            "$.document.source_checksum",
        )
    elif checksum is None:
        _add_issue(context, "SOURCE_CHECKSUM_MISSING", "warning", "Source checksum is missing.", "$.document")
    if not parser_version:
        _add_issue(context, "PARSER_VERSION_MISSING", "warning", "Parser version is missing.", "$.parser_version")


def _validate_pages(document: Mapping[str, Any], context: _ValidationContext) -> list[Mapping[str, Any]]:
    pages_value = document.get("pages")
    if not isinstance(pages_value, list):
        _add_issue(context, "PAGES_NOT_LIST", "error", "pages must be a list.", "$.pages")
        return []

    pages: list[Mapping[str, Any]] = []
    seen_indices: dict[int, str] = {}
    labels: dict[str, int] = {}
    previous_index: int | None = None

    for index, page in enumerate(pages_value):
        path = f"$.pages[{index}]"
        if not isinstance(page, Mapping):
            _add_issue(context, "PAGE_NOT_OBJECT", "error", "Page record must be an object.", path)
            continue
        pages.append(page)

        page_id = _optional_str(page.get("page_id"))
        if page_id:
            context.page_ids.add(page_id)

        pdf_page_index = page.get("pdf_page_index")
        if pdf_page_index is None:
            _add_issue(context, "PAGE_INDEX_MISSING", "error", "pdf_page_index is required.", f"{path}.pdf_page_index")
        elif not _non_negative_int(pdf_page_index):
            _add_issue(
                context,
                "PAGE_INDEX_INVALID",
                "error",
                "pdf_page_index must be a non-negative integer.",
                f"{path}.pdf_page_index",
                page_id,
            )
        else:
            if pdf_page_index in seen_indices:
                _add_issue(
                    context,
                    "PAGE_INDEX_DUPLICATE",
                    "error",
                    f"Duplicate pdf_page_index: {pdf_page_index}.",
                    f"{path}.pdf_page_index",
                    page_id,
                )
            seen_indices[pdf_page_index] = path
            context.page_indices.add(pdf_page_index)
            if previous_index is not None and pdf_page_index <= previous_index:
                _add_issue(
                    context,
                    "PAGE_INDEX_OUT_OF_ORDER",
                    "error",
                    "Page indices must be strictly ordered.",
                    f"{path}.pdf_page_index",
                    page_id,
                )
            previous_index = pdf_page_index

        page_number = page.get("page_number")
        if page_number is not None:
            if not _positive_int(page_number):
                _add_issue(
                    context,
                    "PAGE_NUMBER_INVALID",
                    "error",
                    "page_number must be a positive integer when present.",
                    f"{path}.page_number",
                    page_id,
                )
            else:
                context.page_numbers.add(page_number)

        label = page.get("printed_page_label")
        if label is not None and not isinstance(label, str):
            _add_issue(
                context,
                "PRINTED_PAGE_LABEL_INVALID",
                "warning",
                "printed_page_label should be a string or null.",
                f"{path}.printed_page_label",
                page_id,
            )
        if isinstance(label, str) and label:
            labels[label] = labels.get(label, 0) + 1

    for label, count in sorted(labels.items()):
        if count > 1:
            _add_issue(
                context,
                "PRINTED_PAGE_LABEL_AMBIGUOUS",
                "warning",
                f"Printed page label appears on {count} pages: {label!r}.",
                "$.pages",
            )

    page_count = _declared_page_count(document)
    if page_count is not None and page_count != len(pages):
        _add_issue(
            context,
            "PAGE_COUNT_MISMATCH",
            "error",
            f"Declared page_count {page_count} does not match {len(pages)} page records.",
            "$.document.page_count",
        )

    if _claims_complete_page_sequence(document) and context.page_indices:
        expected = set(range(min(context.page_indices), max(context.page_indices) + 1))
        missing = sorted(expected - context.page_indices)
        if missing:
            _add_issue(
                context,
                "PAGE_INDEX_GAP",
                "warning",
                f"Page index sequence has gaps: {missing}.",
                "$.pages",
            )

    return pages


def _validate_sections(document: Mapping[str, Any], context: _ValidationContext) -> list[Mapping[str, Any]]:
    sections_value = document.get("sections", [])
    if not isinstance(sections_value, list):
        _add_issue(context, "SECTIONS_NOT_LIST", "error", "sections must be a list when present.", "$.sections")
        return []

    sections: list[Mapping[str, Any]] = []
    parents: dict[str, str | None] = {}
    levels: dict[str, int | None] = {}

    for index, section in enumerate(sections_value):
        path = f"$.sections[{index}]"
        if not isinstance(section, Mapping):
            _add_issue(context, "SECTION_NOT_OBJECT", "error", "Section record must be an object.", path)
            continue
        sections.append(section)
        section_id = _optional_str(section.get("section_id"))
        if not section_id:
            _add_issue(context, "SECTION_ID_MISSING", "error", "section_id is required.", f"{path}.section_id")
            continue
        if section_id in context.section_ids:
            _add_issue(context, "SECTION_ID_DUPLICATE", "error", f"Duplicate section_id: {section_id}.", path, section_id)
        context.section_ids.add(section_id)
        parent_id = _optional_str(section.get("parent_section_id"))
        parents[section_id] = parent_id

        level = section.get("level")
        if level is not None and not _positive_int(level):
            _add_issue(
                context,
                "HEADING_LEVEL_INVALID",
                "error",
                "Heading level must be a positive integer when present.",
                f"{path}.level",
                section_id,
            )
            levels[section_id] = None
        else:
            levels[section_id] = level if isinstance(level, int) and not isinstance(level, bool) else None

        section_path = section.get("path")
        if section_path is not None:
            _validate_section_path(section_path, section, path, section_id, context)

    for section_id, parent_id in sorted(parents.items()):
        if parent_id is None:
            continue
        if parent_id == section_id:
            _add_issue(
                context,
                "SECTION_SELF_PARENT",
                "error",
                "Section must not be its own parent.",
                _section_path_by_id(sections, section_id),
                section_id,
            )
        elif parent_id not in parents:
            _add_issue(
                context,
                "SECTION_PARENT_UNKNOWN",
                "error",
                f"Unknown parent_section_id: {parent_id}.",
                _section_path_by_id(sections, section_id),
                section_id,
            )
        elif (
            levels.get(section_id) is not None
            and levels.get(parent_id) is not None
            and levels[section_id] <= levels[parent_id]
        ):
            _add_issue(
                context,
                "HEADING_LEVEL_INCONSISTENT",
                "warning",
                "Child heading level should be greater than parent heading level.",
                _section_path_by_id(sections, section_id),
                section_id,
            )

    _validate_section_cycles(parents, sections, context)
    return sections


def _validate_blocks(document: Mapping[str, Any], context: _ValidationContext) -> list[Mapping[str, Any]]:
    blocks_value = document.get("blocks", [])
    if not isinstance(blocks_value, list):
        _add_issue(context, "BLOCKS_NOT_LIST", "error", "blocks must be a list when present.", "$.blocks")
        return []

    blocks: list[Mapping[str, Any]] = []
    document_indices: dict[int, str] = {}
    page_indices: dict[tuple[Any, int], str] = {}
    previous_document_index: int | None = None

    for index, block in enumerate(blocks_value):
        path = f"$.blocks[{index}]"
        if not isinstance(block, Mapping):
            _add_issue(context, "BLOCK_NOT_OBJECT", "error", "Block record must be an object.", path)
            continue
        blocks.append(block)

        block_id = _optional_str(block.get("block_id"))
        if not block_id:
            _add_issue(context, "BLOCK_ID_MISSING", "error", "block_id is required.", f"{path}.block_id")
        elif block_id in context.block_ids:
            _add_issue(context, "BLOCK_ID_DUPLICATE", "error", f"Duplicate block_id: {block_id}.", path, block_id)
        elif block_id:
            context.block_ids.add(block_id)

        if not _block_has_known_page(block, context):
            _add_issue(context, "BLOCK_PAGE_UNKNOWN", "error", "Block does not reference a known page.", path, block_id)

        content_type = _optional_str(block.get("content_type", block.get("block_type")))
        if content_type not in ALLOWED_CONTENT_TYPES:
            _add_issue(
                context,
                "CONTENT_TYPE_UNSUPPORTED",
                "error",
                f"Unsupported content type: {content_type!r}.",
                f"{path}.block_type",
                block_id,
            )

        text = block.get("text")
        if not _non_empty_string(text) and content_type not in {"blank", "figure", "metadata", "section_heading"}:
            _add_issue(context, "BLOCK_TEXT_MISSING", "error", "Block text is required for textual blocks.", f"{path}.text", block_id)

        order_value = block.get("document_block_index", block.get("reading_order"))
        if order_value is not None:
            if not _non_negative_int(order_value):
                _add_issue(context, "BLOCK_INDEX_INVALID", "error", "Document block index must be non-negative.", path, block_id)
            else:
                if order_value in document_indices:
                    _add_issue(context, "BLOCK_INDEX_DUPLICATE", "error", f"Duplicate document block index: {order_value}.", path, block_id)
                document_indices[order_value] = path
                if previous_document_index is not None and order_value <= previous_document_index:
                    _add_issue(context, "BLOCK_INDEX_OUT_OF_ORDER", "error", "Document block order must be strictly increasing.", path, block_id)
                previous_document_index = order_value
        else:
            _add_issue(context, "BLOCK_INDEX_INVALID", "warning", "Document block index or reading_order is missing.", path, block_id)

        page_block_index = block.get("page_block_index")
        if page_block_index is not None:
            if not _non_negative_int(page_block_index):
                _add_issue(context, "BLOCK_INDEX_INVALID", "error", "page_block_index must be non-negative.", path, block_id)
            else:
                page_key = _block_page_key(block)
                index_key = (page_key, page_block_index)
                if index_key in page_indices:
                    _add_issue(context, "BLOCK_INDEX_DUPLICATE", "error", f"Duplicate page block index: {page_block_index}.", path, block_id)
                page_indices[index_key] = path

        section_id = _optional_str(block.get("section_id"))
        if section_id and section_id not in context.section_ids:
            _add_issue(context, "BLOCK_SECTION_UNKNOWN", "error", f"Unknown block section_id: {section_id}.", path, block_id)

        _validate_offsets(block, path, "CHARACTER_OFFSET_INVALID", context, block_id)
        _validate_bbox(block.get("bbox"), f"{path}.bbox", "BOUNDING_BOX_INVALID", context, block_id)

    return blocks


def _validate_tables(
    document: Mapping[str, Any],
    blocks: list[Mapping[str, Any]],
    context: _ValidationContext,
) -> list[Mapping[str, Any]]:
    tables = _list_entities(document, "tables", "TABLES_NOT_LIST", context)
    table_blocks = [block for block in blocks if block.get("block_type") == "table" or block.get("content_type") == "table"]
    all_tables = tables + table_blocks
    row_ids_by_table: dict[str, set[str]] = {}
    column_ids_by_table: dict[str, set[str]] = {}

    for index, table in enumerate(all_tables):
        path = _entity_path("tables", index, len(tables))
        table_id = _optional_str(table.get("table_id", table.get("block_id")))
        if not table_id:
            _add_issue(context, "TABLE_ID_MISSING", "error", "table_id is required.", path)
            continue
        if table_id in context.table_ids:
            _add_issue(context, "TABLE_ID_DUPLICATE", "error", f"Duplicate table_id: {table_id}.", path, table_id)
        context.table_ids.add(table_id)

        if not _entity_has_known_page(table, context):
            _add_issue(context, "TABLE_PAGE_UNKNOWN", "error", "Table does not reference a known page.", path, table_id)

        columns = _entity_list(table, "columns")
        column_ids = _table_column_ids(columns, table)
        column_ids_by_table[table_id] = column_ids

        rows = _entity_list(table, "rows")
        row_ids: set[str] = set()
        for row_index, row in enumerate(rows):
            row_id = _item_id(row, "row_id", "id")
            if row_id:
                if row_id in row_ids:
                    _add_issue(
                        context,
                        "TABLE_ROW_ID_DUPLICATE",
                        "error",
                        f"Duplicate row_id in table {table_id}: {row_id}.",
                        f"{path}.rows[{row_index}]",
                        table_id,
                    )
                row_ids.add(row_id)
        row_ids_by_table[table_id] = row_ids

        cells = _entity_list(table, "cells")
        for cell_index, cell in enumerate(cells):
            if not isinstance(cell, Mapping):
                _add_issue(context, "TABLE_CELL_CONTEXT_INVALID", "error", "Table cell must be an object.", f"{path}.cells[{cell_index}]", table_id)
                continue
            row_id = _optional_str(cell.get("row_id"))
            column_id = _optional_str(cell.get("column_id"))
            if row_ids and row_id not in row_ids:
                _add_issue(context, "TABLE_ROW_UNKNOWN", "error", f"Unknown table row_id: {row_id}.", f"{path}.cells[{cell_index}]", table_id)
            if column_ids and column_id not in column_ids:
                _add_issue(context, "TABLE_COLUMN_UNKNOWN", "error", f"Unknown table column_id: {column_id}.", f"{path}.cells[{cell_index}]", table_id)
            if (row_ids and row_id is None) or (column_ids and column_id is None):
                _add_issue(context, "TABLE_CELL_CONTEXT_INVALID", "error", "Table cell must retain row and column context.", f"{path}.cells[{cell_index}]", table_id)

        table_context = table.get("table_context")
        if isinstance(table_context, Mapping):
            _validate_continuation_flags(table_context, path, table_id, context)
        _validate_continuation_flags(table, path, table_id, context)
        _validate_table_merges(table.get("merged_cells"), path, table_id, context)

    return tables


def _validate_figures(document: Mapping[str, Any], context: _ValidationContext) -> list[Mapping[str, Any]]:
    figures = _list_entities(document, "figures", "FIGURES_NOT_LIST", context)
    for index, figure in enumerate(figures):
        path = f"$.figures[{index}]"
        figure_id = _optional_str(figure.get("figure_id"))
        if not figure_id:
            _add_issue(context, "FIGURE_ID_MISSING", "error", "figure_id is required.", path)
            continue
        if figure_id in context.figure_ids:
            _add_issue(context, "FIGURE_ID_DUPLICATE", "error", f"Duplicate figure_id: {figure_id}.", path, figure_id)
        context.figure_ids.add(figure_id)
        if not _entity_has_known_page(figure, context):
            _add_issue(context, "FIGURE_PAGE_UNKNOWN", "error", "Figure does not reference a known page.", path, figure_id)
        if not _non_empty_string(figure.get("caption")):
            _add_issue(context, "FIGURE_CAPTION_MISSING", "warning", "Figure caption is missing.", f"{path}.caption", figure_id)
        for key in ("asset_reference", "asset_path", "image_path"):
            if key in figure and figure.get(key) is not None and not isinstance(figure.get(key), str):
                _add_issue(context, "FIGURE_ASSET_REFERENCE_INVALID", "error", f"{key} must be a string when present.", f"{path}.{key}", figure_id)
        _validate_source_block_refs(figure, path, "FIGURE_SOURCE_BLOCK_UNKNOWN", context, figure_id)
    return figures


def _validate_equations(document: Mapping[str, Any], context: _ValidationContext) -> list[Mapping[str, Any]]:
    equations = _list_entities(document, "equations", "EQUATIONS_NOT_LIST", context)
    for index, equation in enumerate(equations):
        path = f"$.equations[{index}]"
        equation_id = _optional_str(equation.get("equation_id"))
        if not equation_id:
            _add_issue(context, "EQUATION_ID_MISSING", "error", "equation_id is required.", path)
            continue
        if equation_id in context.equation_ids:
            _add_issue(context, "EQUATION_ID_DUPLICATE", "error", f"Duplicate equation_id: {equation_id}.", path, equation_id)
        context.equation_ids.add(equation_id)
        if not _entity_has_known_page(equation, context):
            _add_issue(context, "EQUATION_PAGE_UNKNOWN", "error", "Equation does not reference a known page.", path, equation_id)
        if not _first_non_empty(equation, {}, "raw_text", "raw", "text", "source_text"):
            _add_issue(context, "EQUATION_RAW_MISSING", "error", "Equation raw representation is required.", path, equation_id)
        _validate_source_block_refs(equation, path, "EQUATION_SOURCE_BLOCK_UNKNOWN", context, equation_id)
    return equations


def _validate_admonitions(
    document: Mapping[str, Any],
    blocks: list[Mapping[str, Any]],
    context: _ValidationContext,
) -> list[Mapping[str, Any]]:
    admonitions = _list_entities(document, "admonitions", "ADMONITIONS_NOT_LIST", context)
    block_admonitions = [
        block
        for block in blocks
        if block.get("block_type") in {"warning", "caution", "note"}
        or block.get("content_type") in {"warning", "caution", "note"}
    ]

    for index, admonition in enumerate(admonitions + block_admonitions):
        path = _entity_path("admonitions", index, len(admonitions))
        admonition_id = _optional_str(admonition.get("admonition_id", admonition.get("block_id")))
        if not admonition_id:
            _add_issue(context, "ADMONITION_ID_MISSING", "error", "admonition_id is required.", path)
            continue
        if admonition_id in context.admonition_ids:
            _add_issue(context, "ADMONITION_ID_DUPLICATE", "error", f"Duplicate admonition_id: {admonition_id}.", path, admonition_id)
        context.admonition_ids.add(admonition_id)

        admonition_type = _optional_str(admonition.get("normalized_type", admonition.get("admonition_type")))
        if admonition_type not in ALLOWED_ADMONITION_TYPES:
            _add_issue(
                context,
                "ADMONITION_TYPE_UNSUPPORTED",
                "error",
                f"Unsupported admonition type: {admonition_type!r}.",
                path,
                admonition_id,
            )
        if not _non_empty_string(admonition.get("raw_label", admonition.get("source_label"))):
            _add_issue(context, "ADMONITION_RAW_LABEL_MISSING", "warning", "Raw admonition label is missing.", path, admonition_id)
        if not _first_non_empty(admonition, {}, "body_text", "text"):
            _add_issue(context, "ADMONITION_BODY_MISSING", "error", "Admonition body text is required.", path, admonition_id)
        if not _entity_has_known_page(admonition, context):
            _add_issue(context, "ADMONITION_PAGE_UNKNOWN", "error", "Admonition does not reference a known page.", path, admonition_id)
        section_id = _optional_str(admonition.get("section_id"))
        if section_id and section_id not in context.section_ids:
            _add_issue(context, "ADMONITION_SECTION_UNKNOWN", "error", f"Unknown admonition section_id: {section_id}.", path, admonition_id)
        _validate_source_block_refs(admonition, path, "ADMONITION_SOURCE_BLOCK_UNKNOWN", context, admonition_id)

    return admonitions


def _validate_cross_references(document: Mapping[str, Any], context: _ValidationContext) -> list[Mapping[str, Any]]:
    cross_references = _list_entities(document, "cross_references", "CROSS_REFERENCES_NOT_LIST", context)
    seen_ids: set[str] = set()
    known_targets = (
        context.block_ids
        | context.section_ids
        | context.table_ids
        | context.figure_ids
        | context.equation_ids
        | context.admonition_ids
        | context.page_ids
    )
    document_id = _document_id(document)
    if document_id:
        known_targets.add(document_id)

    for index, reference in enumerate(cross_references):
        path = f"$.cross_references[{index}]"
        reference_id = _optional_str(reference.get("reference_id", reference.get("cross_reference_id")))
        if not reference_id:
            _add_issue(context, "CROSS_REFERENCE_ID_MISSING", "error", "reference_id is required.", path)
            continue
        if reference_id in seen_ids:
            _add_issue(context, "CROSS_REFERENCE_ID_DUPLICATE", "error", f"Duplicate reference_id: {reference_id}.", path, reference_id)
        seen_ids.add(reference_id)

        if not _first_non_empty(reference, {}, "raw_text", "text", "reference_text"):
            _add_issue(context, "CROSS_REFERENCE_TEXT_MISSING", "error", "Raw reference text is required.", path, reference_id)
        status = _optional_str(reference.get("resolution_status", reference.get("status")))
        if status not in ALLOWED_CROSS_REFERENCE_STATUSES:
            _add_issue(context, "CROSS_REFERENCE_STATUS_INVALID", "error", f"Invalid resolution status: {status!r}.", path, reference_id)
        target_id = _optional_str(reference.get("target_id", reference.get("resolved_target_id")))
        if status == "resolved":
            if not target_id:
                _add_issue(context, "CROSS_REFERENCE_FALSE_RESOLUTION", "error", "Resolved reference must include a local target.", path, reference_id)
            elif target_id not in known_targets:
                _add_issue(context, "CROSS_REFERENCE_TARGET_UNKNOWN", "error", f"Unknown resolved target: {target_id}.", path, reference_id)
        elif status in {"unresolved", "ambiguous", "not_attempted", "external"}:
            pass
        _validate_source_block_refs(reference, path, "SOURCE_BLOCK_UNKNOWN", context, reference_id)

    return cross_references


def _validate_source_spans(
    document: Mapping[str, Any],
    blocks: list[Mapping[str, Any]],
    tables: list[Mapping[str, Any]],
    figures: list[Mapping[str, Any]],
    equations: list[Mapping[str, Any]],
    admonitions: list[Mapping[str, Any]],
    context: _ValidationContext,
) -> None:
    entities: list[tuple[str, Mapping[str, Any]]] = [("$", document)]
    entities.extend((f"$.blocks[{index}]", item) for index, item in enumerate(blocks))
    entities.extend((f"$.tables[{index}]", item) for index, item in enumerate(tables))
    entities.extend((f"$.figures[{index}]", item) for index, item in enumerate(figures))
    entities.extend((f"$.equations[{index}]", item) for index, item in enumerate(equations))
    entities.extend((f"$.admonitions[{index}]", item) for index, item in enumerate(admonitions))

    for path, entity in entities:
        if "source_span" in entity:
            span = entity.get("source_span")
            if not isinstance(span, Mapping):
                _add_issue(context, "SOURCE_SPAN_INVALID", "error", "source_span must be an object.", f"{path}.source_span", _entity_id(entity))
                continue
            _validate_span_object(span, f"{path}.source_span", context, _entity_id(entity))
        _validate_span_object(entity, path, context, _entity_id(entity), direct=True)


def _validate_span_object(
    span: Mapping[str, Any],
    path: str,
    context: _ValidationContext,
    entity_id: str | None,
    direct: bool = False,
) -> None:
    page_start = span.get("page_start")
    page_end = span.get("page_end", page_start)
    if page_start is not None or page_end is not None:
        if not _valid_page_ref(page_start, context) or not _valid_page_ref(page_end, context):
            _add_issue(context, "SOURCE_PAGE_UNKNOWN", "error", "Source page range references an unknown page.", path, entity_id)
        elif isinstance(page_start, int) and isinstance(page_end, int) and page_start > page_end:
            _add_issue(context, "SOURCE_PAGE_RANGE_INVALID", "error", "page_start must not exceed page_end.", path, entity_id)

    index_start = span.get("pdf_page_index_start")
    index_end = span.get("pdf_page_index_end", index_start)
    if index_start is not None or index_end is not None:
        if not _non_negative_int(index_start) or not _non_negative_int(index_end) or index_start > index_end:
            _add_issue(context, "SOURCE_PAGE_RANGE_INVALID", "error", "PDF page-index range must be non-negative and ordered.", path, entity_id)
        elif index_start not in context.page_indices or index_end not in context.page_indices:
            _add_issue(context, "SOURCE_PAGE_UNKNOWN", "error", "PDF page-index range references an unknown page.", path, entity_id)

    page_span = span.get("page_span")
    if page_span is not None and not _valid_page_span(page_span, context):
        _add_issue(context, "SOURCE_PAGE_RANGE_INVALID", "error", "page_span must reference known ordered pages.", path, entity_id)

    contributing_pages = span.get("contributing_pages", span.get("pages"))
    if not direct and contributing_pages is not None:
        if not isinstance(contributing_pages, list) or any(not _valid_page_ref(item, context) for item in contributing_pages):
            _add_issue(context, "SOURCE_PAGE_UNKNOWN", "error", "Contributing pages must reference known pages.", path, entity_id)

    _validate_source_block_refs(span, path, "SOURCE_BLOCK_UNKNOWN", context, entity_id)
    _validate_offsets(span, path, "SOURCE_OFFSET_INVALID", context, entity_id)
    _validate_bbox(span.get("bbox"), f"{path}.bbox", "SOURCE_BBOX_INVALID", context, entity_id)

    source_hash = span.get("source_hash")
    if source_hash is not None and not _non_empty_string(source_hash):
        _add_issue(context, "SOURCE_HASH_INVALID", "error", "source_hash must be a non-empty string when present.", path, entity_id)


def _validate_confidence_values(value: Any, context: _ValidationContext, path: str = "$") -> None:
    if isinstance(value, Mapping):
        entity_id = _entity_id(value)
        for key, item in value.items():
            item_path = f"{path}.{key}"
            if key in CONFIDENCE_FIELDS:
                if item is None:
                    continue
                if isinstance(item, bool):
                    _add_issue(context, "CONFIDENCE_TYPE_INVALID", "error", "Confidence value must not be boolean.", item_path, entity_id)
                elif not isinstance(item, (int, float)):
                    _add_issue(context, "CONFIDENCE_TYPE_INVALID", "error", "Confidence value must be numeric or null.", item_path, entity_id)
                elif item < 0 or item > 1:
                    _add_issue(context, "CONFIDENCE_OUT_OF_RANGE", "error", "Confidence value must be between 0.0 and 1.0.", item_path, entity_id)
            _validate_confidence_values(item, context, item_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_confidence_values(item, context, f"{path}[{index}]")


def _validate_source_block_refs(
    entity: Mapping[str, Any],
    path: str,
    unknown_code: str,
    context: _ValidationContext,
    entity_id: str | None,
) -> None:
    refs = entity.get("source_block_ids", entity.get("source_blocks"))
    if refs is None:
        return
    if not isinstance(refs, list):
        _add_issue(context, "SOURCE_SPAN_INVALID", "error", "source_block_ids must be a list.", path, entity_id)
        return

    seen: set[str] = set()
    for ref in refs:
        if not _non_empty_string(ref):
            _add_issue(context, unknown_code, "error", "Source block reference must be a non-empty string.", path, entity_id)
            continue
        if ref in seen:
            _add_issue(context, "SOURCE_BLOCK_DUPLICATE", "warning", f"Duplicate source block ID in span: {ref}.", path, entity_id)
        seen.add(ref)
        if ref not in context.block_ids:
            _add_issue(context, unknown_code, "error", f"Unknown source block ID: {ref}.", path, entity_id)


def _validate_offsets(
    entity: Mapping[str, Any],
    path: str,
    code: str,
    context: _ValidationContext,
    entity_id: str | None,
) -> None:
    start = entity.get("char_start", entity.get("character_start"))
    end = entity.get("char_end", entity.get("character_end"))
    if start is None and end is None:
        return
    if not _non_negative_int(start) or not _non_negative_int(end) or start > end:
        _add_issue(context, code, "error", "Character offsets must be non-negative and ordered.", path, entity_id)


def _validate_bbox(
    bbox: Any,
    path: str,
    code: str,
    context: _ValidationContext,
    entity_id: str | None,
) -> None:
    if bbox is None:
        return
    values: list[Any]
    if isinstance(bbox, Mapping):
        values = [bbox.get(key) for key in ("x0", "y0", "x1", "y1")]
    elif isinstance(bbox, list) and len(bbox) == 4:
        values = list(bbox)
    else:
        _add_issue(context, code, "error", "Bounding box must be a four-value list or coordinate object.", path, entity_id)
        return
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in values):
        _add_issue(context, code, "error", "Bounding box coordinates must be numeric.", path, entity_id)
        return
    x0, y0, x1, y1 = values
    if x0 > x1 or y0 > y1:
        _add_issue(context, code, "error", "Bounding box coordinates must be ordered.", path, entity_id)


def _validate_section_path(
    section_path: Any,
    section: Mapping[str, Any],
    path: str,
    section_id: str,
    context: _ValidationContext,
) -> None:
    if not isinstance(section_path, list) or not all(isinstance(item, str) and item.strip() for item in section_path):
        _add_issue(context, "SECTION_PATH_INVALID", "warning", "Section path should be a non-empty string list.", f"{path}.path", section_id)
        return
    title = _optional_str(section.get("title"))
    number = _optional_str(section.get("section_number"))
    expected = " ".join(part for part in (number, title) if part).strip()
    if expected and section_path[-1] not in {title, expected}:
        _add_issue(context, "SECTION_PATH_INVALID", "warning", "Section path should end with the current section.", f"{path}.path", section_id)


def _validate_section_cycles(
    parents: Mapping[str, str | None],
    sections: list[Mapping[str, Any]],
    context: _ValidationContext,
) -> None:
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(section_id: str) -> None:
        if section_id in visited:
            return
        if section_id in visiting:
            _add_issue(
                context,
                "SECTION_CYCLE",
                "error",
                f"Section parent cycle includes {section_id}.",
                _section_path_by_id(sections, section_id),
                section_id,
            )
            return
        visiting.add(section_id)
        parent = parents.get(section_id)
        if parent in parents:
            visit(parent)
        visiting.remove(section_id)
        visited.add(section_id)

    for section_id in sorted(parents):
        visit(section_id)


def _validate_continuation_flags(
    entity: Mapping[str, Any],
    path: str,
    table_id: str,
    context: _ValidationContext,
) -> None:
    for key in ("continued_from_previous_page", "continues_to_next_page", "continuation"):
        if key in entity and entity.get(key) is not None and not isinstance(entity.get(key), bool):
            _add_issue(context, "TABLE_CONTINUATION_INVALID", "error", f"{key} must be boolean when present.", f"{path}.{key}", table_id)


def _validate_table_merges(
    merged_cells: Any,
    path: str,
    table_id: str,
    context: _ValidationContext,
) -> None:
    if merged_cells is None:
        return
    if not isinstance(merged_cells, list):
        _add_issue(context, "TABLE_MERGE_INVALID", "error", "merged_cells must be a list when present.", f"{path}.merged_cells", table_id)
        return
    for index, merge in enumerate(merged_cells):
        if not isinstance(merge, Mapping):
            _add_issue(context, "TABLE_MERGE_INVALID", "error", "Merged-cell entry must be an object.", f"{path}.merged_cells[{index}]", table_id)
            continue
        row_span = merge.get("row_span", 1)
        column_span = merge.get("column_span", 1)
        if not _positive_int(row_span) or not _positive_int(column_span):
            _add_issue(context, "TABLE_MERGE_INVALID", "error", "Merged-cell spans must be positive integers.", f"{path}.merged_cells[{index}]", table_id)


def _table_column_ids(columns: list[Any], table: Mapping[str, Any]) -> set[str]:
    column_ids: set[str] = set()
    for column in columns:
        if isinstance(column, Mapping):
            column_id = _optional_str(column.get("column_id", column.get("id", column.get("label"))))
        else:
            column_id = _optional_str(column)
        if column_id:
            column_ids.add(column_id)

    if not column_ids:
        table_context = table.get("table_context")
        if isinstance(table_context, Mapping):
            headers = table_context.get("column_headers")
            if isinstance(headers, list):
                column_ids.update(str(header) for header in headers if isinstance(header, str) and header)
    return column_ids


def _entity_has_known_page(entity: Mapping[str, Any], context: _ValidationContext) -> bool:
    if _block_has_known_page(entity, context):
        return True
    page_span = entity.get("page_span")
    return page_span is not None and _valid_page_span(page_span, context)


def _block_has_known_page(block: Mapping[str, Any], context: _ValidationContext) -> bool:
    page_id = _optional_str(block.get("page_id"))
    if page_id and page_id in context.page_ids:
        return True
    pdf_page_index = block.get("pdf_page_index")
    if _non_negative_int(pdf_page_index) and pdf_page_index in context.page_indices:
        return True
    page_ref = block.get("page_number", block.get("page", block.get("page_start")))
    if _valid_page_ref(page_ref, context):
        return True
    source_span = block.get("source_span")
    if isinstance(source_span, Mapping):
        span_page = source_span.get("page_start", source_span.get("page_number", source_span.get("page")))
        return _valid_page_ref(span_page, context)
    return False


def _block_page_key(block: Mapping[str, Any]) -> Any:
    source_span = block.get("source_span")
    if isinstance(source_span, Mapping):
        return source_span.get("page_start", source_span.get("pdf_page_index_start"))
    return block.get("page_id", block.get("page_number", block.get("page", block.get("pdf_page_index"))))


def _valid_page_ref(value: Any, context: _ValidationContext) -> bool:
    if isinstance(value, str):
        return value in context.page_ids
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    return value in context.page_numbers or value in context.page_indices or (value - 1) in context.page_indices


def _valid_page_span(value: Any, context: _ValidationContext) -> bool:
    if isinstance(value, Mapping):
        start = value.get("page_start", value.get("start"))
        end = value.get("page_end", value.get("end", start))
    elif isinstance(value, list) and len(value) == 2:
        start, end = value
    else:
        return False
    return _valid_page_ref(start, context) and _valid_page_ref(end, context) and (
        not isinstance(start, int) or not isinstance(end, int) or start <= end
    )


def _entity_count(root_entities: list[Mapping[str, Any]], blocks: list[Mapping[str, Any]], block_type: str) -> int:
    block_ids = {
        _optional_str(block.get(f"{block_type}_id", block.get("block_id")))
        for block in blocks
        if block.get("block_type") == block_type or block.get("content_type") == block_type
    }
    root_ids = {_optional_str(entity.get(f"{block_type}_id")) for entity in root_entities}
    return len([item for item in (block_ids | root_ids) if item])


def _admonition_count(admonitions: list[Mapping[str, Any]], blocks: list[Mapping[str, Any]]) -> int:
    block_ids = {
        _optional_str(block.get("admonition_id", block.get("block_id")))
        for block in blocks
        if block.get("block_type") in {"warning", "caution", "note"}
        or block.get("content_type") in {"warning", "caution", "note"}
    }
    root_ids = {_optional_str(entity.get("admonition_id")) for entity in admonitions}
    return len([item for item in (block_ids | root_ids) if item])


def _list_entities(document: Mapping[str, Any], key: str, code: str, context: _ValidationContext) -> list[Mapping[str, Any]]:
    value = document.get(key, [])
    if value is None:
        return []
    if not isinstance(value, list):
        _add_issue(context, code, "error", f"{key} must be a list when present.", f"$.{key}")
        return []
    entities: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, Mapping):
            entities.append(item)
        else:
            _add_issue(context, f"{key.upper()}_ITEM_INVALID", "error", f"{key} item must be an object.", f"$.{key}[{index}]")
    return entities


def _entity_list(entity: Mapping[str, Any], key: str) -> list[Any]:
    value = entity.get(key)
    return value if isinstance(value, list) else []


def _entity_path(root_name: str, index: int, root_count: int) -> str:
    if index < root_count:
        return f"$.{root_name}[{index}]"
    return f"$.blocks[{index - root_count}]"


def _item_id(item: Any, *keys: str) -> str | None:
    if not isinstance(item, Mapping):
        return None
    for key in keys:
        value = _optional_str(item.get(key))
        if value:
            return value
    return None


def _entity_id(entity: Mapping[str, Any]) -> str | None:
    return _item_id(
        entity,
        "block_id",
        "section_id",
        "table_id",
        "figure_id",
        "equation_id",
        "admonition_id",
        "reference_id",
        "document_id",
    )


def _document_id(document: Mapping[str, Any]) -> str | None:
    metadata = document.get("document")
    if not isinstance(metadata, Mapping):
        metadata = {}
    return _first_non_empty(document, metadata, "document_id")


def _declared_page_count(document: Mapping[str, Any]) -> int | None:
    metadata = document.get("document")
    if not isinstance(metadata, Mapping):
        metadata = {}
    value = document.get("page_count", metadata.get("page_count"))
    return value if _non_negative_int(value) else None


def _claims_complete_page_sequence(document: Mapping[str, Any]) -> bool:
    metadata = document.get("document")
    if not isinstance(metadata, Mapping):
        metadata = {}
    return bool(document.get("complete_page_sequence", metadata.get("complete_page_sequence", False)))


def _section_path_by_id(sections: list[Mapping[str, Any]], section_id: str) -> str:
    for index, section in enumerate(sections):
        if section.get("section_id") == section_id:
            return f"$.sections[{index}]"
    return "$.sections"


def _first_non_empty(root: Mapping[str, Any], nested: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = root.get(key)
        if _non_empty_string(value):
            return value
        value = nested.get(key)
        if _non_empty_string(value):
            return value
    return None


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _add_issue(
    context: _ValidationContext,
    code: str,
    severity: str,
    message: str,
    path: str,
    entity_id: str | None = None,
) -> None:
    if severity not in ISSUE_SEVERITIES:
        raise ValueError(f"Unsupported validation issue severity: {severity}")
    context.issues.append(
        ValidationIssue(
            code=code,
            severity=severity,
            message=message,
            path=path,
            entity_id=entity_id,
        )
    )


def _build_result(
    schema_name: str | None,
    schema_version: str | None,
    document_id: str | None,
    context: _ValidationContext,
    summary: dict[str, Any],
) -> StructuredDocumentValidationResult:
    issues = _sort_issues(context.issues)
    error_count = sum(1 for issue in issues if issue.severity == "error")
    warning_count = sum(1 for issue in issues if issue.severity == "warning")
    return StructuredDocumentValidationResult(
        schema_name=schema_name,
        schema_version=schema_version,
        document_id=document_id,
        is_valid=error_count == 0,
        error_count=error_count,
        warning_count=warning_count,
        issues=issues,
        summary=summary,
    )


def _sort_issues(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    return sorted(
        issues,
        key=lambda issue: (
            _SEVERITY_ORDER.get(issue.severity, 99),
            issue.path,
            issue.code,
            issue.message,
            issue.entity_id or "",
        ),
    )


def _issue_to_dict(issue: ValidationIssue) -> dict[str, Any]:
    return asdict(issue)


__all__ = [
    "ALLOWED_ADMONITION_TYPES",
    "ALLOWED_CONTENT_TYPES",
    "DEFAULT_SUPPORTED_SCHEMA_NAMES",
    "DEFAULT_SUPPORTED_SCHEMA_VERSIONS",
    "ISSUE_SEVERITIES",
    "StructuredDocumentValidationResult",
    "ValidationIssue",
    "load_structured_document",
    "structured_document_validation_result_to_dict",
    "validate_structured_document",
    "validate_structured_document_file",
]
