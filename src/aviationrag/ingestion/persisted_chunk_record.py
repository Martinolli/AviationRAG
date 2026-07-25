"""Storage-neutral persisted chunk record model for D.5b dry runs.

This module defines immutable schema constants and a frozen persisted-record
dataclass. It performs no filesystem access, environment discovery, database
access, embedding work, or runtime ingestion integration at import time.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from types import MappingProxyType
from typing import Any, Mapping


PERSISTED_CHUNK_SCHEMA_NAME = "aviationrag-persisted-chunk"
PERSISTED_CHUNK_SCHEMA_VERSION = "0.1.0"
SUPPORTED_PERSISTED_CHUNK_SCHEMA_VERSIONS = frozenset({"0.1.0"})

PERSISTED_CHUNK_MAPPING_SPECIFICATION_NAME = "aviationrag-persisted-chunk-mapping"
PERSISTED_CHUNK_MAPPING_SPECIFICATION_VERSION = "0.1.0"
PERSISTED_CHUNK_MAPPER_VERSION = "0.1.0"

RECORD_ORIGINS = frozenset({"new_structured", "legacy_adapted", "legacy_unresolved"})
GENERATED_RECORD_ORIGIN = "new_structured"

PROVENANCE_STATUSES = frozenset(
    {"full_provenance", "partial_provenance", "legacy_filename_only", "unknown_provenance"}
)
VALIDATION_STATUSES = frozenset(
    {"valid", "valid_with_warnings", "review_required", "rejected"}
)

PERSISTED_CONTENT_TYPES = frozenset(
    {
        "paragraph",
        "list",
        "table",
        "figure_caption",
        "equation",
        "warning",
        "caution",
        "note",
        "important",
        "safety_notice",
        "unknown_admonition",
        "procedure",
        "requirement",
        "definition",
        "footnote",
        "reference",
        "appendix_content",
        "mixed",
        "unknown",
    }
)

FORBIDDEN_PERSISTED_FIELDS = frozenset(
    {
        "embedding",
        "embeddings",
        "embedding_vector",
        "vector",
        "vectors",
        "values",
        "astra_id",
        "astra_identity",
        "faiss_id",
        "faiss_position",
        "absolute_path",
        "temporary_path",
        "random_id",
        "confidence",
        "confidence_score",
        "generated_figure_description",
        "unverified_table_cells",
        "inferred_revision",
    }
)


@dataclass(frozen=True)
class PersistedChunkRecord:
    """Validated storage-neutral persisted chunk record."""

    schema_name: str
    schema_version: str
    chunk_id: str
    chunk_index: int
    document_id: str
    source_filename: str
    source_checksum: str
    document_title: str | None
    document_number: str | None
    document_revision: str | None
    document_issue: str | None
    effective_date: str | None
    text: str
    normalized_text: str | None
    content_type: str
    content_subtype: str | None
    language: str | None
    page_start: int
    page_end: int
    pdf_page_index_start: int
    pdf_page_index_end: int
    contributing_page_numbers: tuple[int, ...]
    contributing_pdf_page_indexes: tuple[int, ...]
    printed_page_labels: tuple[str, ...]
    section_id: str | None
    section_path: tuple[str, ...]
    section_number: str | None
    section_title: str | None
    clause_identifier: str | None
    source_block_ids: tuple[str, ...]
    source_span: Mapping[str, Any] | None
    table_ids: tuple[str, ...]
    figure_ids: tuple[str, ...]
    equation_ids: tuple[str, ...]
    admonition_ids: tuple[str, ...]
    cross_reference_ids: tuple[str, ...]
    parser_name: str
    parser_version: str
    structured_document_schema_version: str
    adapter_version: str
    persistence_mapper_version: str
    extraction_method: str | None
    record_origin: str
    provenance_status: str
    accepted_limitation_codes: tuple[str, ...] = field(default_factory=tuple)
    validation_status: str = "valid"
    warning_codes: tuple[str, ...] = field(default_factory=tuple)
    review_required: bool = False

    def __post_init__(self) -> None:
        if self.source_span is not None:
            object.__setattr__(self, "source_span", MappingProxyType(_copy_mapping(self.source_span)))

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-serializable dictionary."""
        return {item.name: _plain_value(getattr(self, item.name)) for item in fields(self)}


def persisted_chunk_record_to_dict(record: PersistedChunkRecord) -> dict[str, Any]:
    """Return a JSON-serializable persisted-record dictionary."""
    return record.to_dict()


def persisted_chunk_record_from_dict(data: Mapping[str, Any]) -> PersistedChunkRecord:
    """Create a persisted record from a mapping using copied immutable values."""
    return PersistedChunkRecord(
        schema_name=str(data.get("schema_name") or ""),
        schema_version=str(data.get("schema_version") or ""),
        chunk_id=str(data.get("chunk_id") or ""),
        chunk_index=_int_or_zero(data.get("chunk_index")),
        document_id=str(data.get("document_id") or ""),
        source_filename=str(data.get("source_filename") or ""),
        source_checksum=str(data.get("source_checksum") or ""),
        document_title=_optional_str(data.get("document_title")),
        document_number=_optional_str(data.get("document_number")),
        document_revision=_optional_str(data.get("document_revision")),
        document_issue=_optional_str(data.get("document_issue")),
        effective_date=_optional_str(data.get("effective_date")),
        text=str(data.get("text") or ""),
        normalized_text=_optional_str(data.get("normalized_text")),
        content_type=str(data.get("content_type") or ""),
        content_subtype=_optional_str(data.get("content_subtype")),
        language=_optional_str(data.get("language")),
        page_start=_int_or_zero(data.get("page_start")),
        page_end=_int_or_zero(data.get("page_end")),
        pdf_page_index_start=_int_or_zero(data.get("pdf_page_index_start")),
        pdf_page_index_end=_int_or_zero(data.get("pdf_page_index_end")),
        contributing_page_numbers=_int_tuple(data.get("contributing_page_numbers")),
        contributing_pdf_page_indexes=_int_tuple(data.get("contributing_pdf_page_indexes")),
        printed_page_labels=_str_tuple(data.get("printed_page_labels")),
        section_id=_optional_str(data.get("section_id")),
        section_path=_str_tuple(data.get("section_path")),
        section_number=_optional_str(data.get("section_number")),
        section_title=_optional_str(data.get("section_title")),
        clause_identifier=_optional_str(data.get("clause_identifier")),
        source_block_ids=_str_tuple(data.get("source_block_ids")),
        source_span=data.get("source_span") if isinstance(data.get("source_span"), Mapping) else None,
        table_ids=_str_tuple(data.get("table_ids")),
        figure_ids=_str_tuple(data.get("figure_ids")),
        equation_ids=_str_tuple(data.get("equation_ids")),
        admonition_ids=_str_tuple(data.get("admonition_ids")),
        cross_reference_ids=_str_tuple(data.get("cross_reference_ids")),
        parser_name=str(data.get("parser_name") or ""),
        parser_version=str(data.get("parser_version") or ""),
        structured_document_schema_version=str(data.get("structured_document_schema_version") or ""),
        adapter_version=str(data.get("adapter_version") or ""),
        persistence_mapper_version=str(data.get("persistence_mapper_version") or ""),
        extraction_method=_optional_str(data.get("extraction_method")),
        record_origin=str(data.get("record_origin") or ""),
        provenance_status=str(data.get("provenance_status") or ""),
        accepted_limitation_codes=_str_tuple(data.get("accepted_limitation_codes")),
        validation_status=str(data.get("validation_status") or ""),
        warning_codes=_str_tuple(data.get("warning_codes")),
        review_required=bool(data.get("review_required", False)),
    )


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _plain_value(item) for key, item in value.items()}


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_value(item) for item in value]
    if isinstance(value, list):
        return [_plain_value(item) for item in value]
    return value


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _str_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item.strip())


def _int_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, int) and not isinstance(item, bool))


def _int_or_zero(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


__all__ = [
    "FORBIDDEN_PERSISTED_FIELDS",
    "GENERATED_RECORD_ORIGIN",
    "PERSISTED_CHUNK_MAPPER_VERSION",
    "PERSISTED_CHUNK_MAPPING_SPECIFICATION_NAME",
    "PERSISTED_CHUNK_MAPPING_SPECIFICATION_VERSION",
    "PERSISTED_CHUNK_SCHEMA_NAME",
    "PERSISTED_CHUNK_SCHEMA_VERSION",
    "PERSISTED_CONTENT_TYPES",
    "PROVENANCE_STATUSES",
    "PersistedChunkRecord",
    "RECORD_ORIGINS",
    "SUPPORTED_PERSISTED_CHUNK_SCHEMA_VERSIONS",
    "VALIDATION_STATUSES",
    "persisted_chunk_record_from_dict",
    "persisted_chunk_record_to_dict",
]
