"""Validation and policy constants for D.5b persisted chunk records."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import re
from typing import Any, Mapping

from aviationrag.ingestion.persisted_chunk_record import (
    FORBIDDEN_PERSISTED_FIELDS,
    GENERATED_RECORD_ORIGIN,
    PERSISTED_CHUNK_MAPPER_VERSION,
    PERSISTED_CHUNK_SCHEMA_NAME,
    PERSISTED_CHUNK_SCHEMA_VERSION,
    PERSISTED_CONTENT_TYPES,
    PROVENANCE_STATUSES,
    RECORD_ORIGINS,
    SUPPORTED_PERSISTED_CHUNK_SCHEMA_VERSIONS,
    VALIDATION_STATUSES,
    PersistedChunkRecord,
    persisted_chunk_record_to_dict,
)


PERSISTED_CHUNK_LIMITATION_REGISTRY_VERSION = "0.1.0"

LIMITATION_CHUNK_SECTION_CROSSING_REVIEW = "CHUNK_SECTION_CROSSING_REVIEW"
LIMITATION_DUPLICATE_TEXT_LINES = "DUPLICATE_TEXT_LINES"
LIMITATION_TABLE_CANDIDATE_ONLY = "TABLE_CANDIDATE_ONLY"
LIMITATION_TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE = "TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE"

APPROVED_LIMITATION_CODES = frozenset(
    {
        LIMITATION_CHUNK_SECTION_CROSSING_REVIEW,
        LIMITATION_DUPLICATE_TEXT_LINES,
        LIMITATION_TABLE_CANDIDATE_ONLY,
        LIMITATION_TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE,
    }
)

LIMITATION_WARNING_CODES = {
    LIMITATION_CHUNK_SECTION_CROSSING_REVIEW: "SECTION_CROSSING_REVIEW_REQUIRED",
    LIMITATION_DUPLICATE_TEXT_LINES: "DUPLICATE_TEXT_LINES_RETAINED",
    LIMITATION_TABLE_CANDIDATE_ONLY: "TABLE_STRUCTURE_CANDIDATE_ONLY",
    LIMITATION_TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE: "TABLE_CLASSIFICATION_REVIEW_REQUIRED",
}

LIMITATIONS_REQUIRING_REVIEW = frozenset(
    {
        LIMITATION_CHUNK_SECTION_CROSSING_REVIEW,
        LIMITATION_TABLE_CANDIDATE_ONLY,
        LIMITATION_TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE,
    }
)

LIMITATION_ALLOWED_CONTENT_TYPES = {
    LIMITATION_CHUNK_SECTION_CROSSING_REVIEW: PERSISTED_CONTENT_TYPES,
    LIMITATION_DUPLICATE_TEXT_LINES: PERSISTED_CONTENT_TYPES,
    LIMITATION_TABLE_CANDIDATE_ONLY: frozenset({"table"}),
    LIMITATION_TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE: frozenset({"table", "figure_caption"}),
}

SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PERSISTED_CHUNK_ID_PATTERN = re.compile(r"^.+:chunk:[0-9a-f]{24}$")
PRIVATE_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"^/"),
    re.compile(r"^\\\\"),
    re.compile(r"(^|[\\/])data[\\/](documents|raw|processed|embeddings|manifest)([\\/]|$)"),
    re.compile(r"(^|[\\/])(Users|home)([\\/]|$)", re.IGNORECASE),
    re.compile(r"secure-connect", re.IGNORECASE),
    re.compile(r"\.env", re.IGNORECASE),
)


@dataclass(frozen=True)
class PersistedChunkIssue:
    """One deterministic persisted-chunk issue."""

    code: str
    severity: str
    message: str
    path: str
    candidate_id: str | None = None
    chunk_id: str | None = None


def persisted_chunk_issue_to_dict(issue: PersistedChunkIssue) -> dict[str, Any]:
    """Return a JSON-serializable issue dictionary."""
    return asdict(issue)


def validate_persisted_chunk_record(
    record: PersistedChunkRecord,
    *,
    candidate_id: str | None = None,
) -> tuple[PersistedChunkIssue, ...]:
    """Return deterministic validation issues for one persisted record."""
    data = persisted_chunk_record_to_dict(record)
    issues: list[PersistedChunkIssue] = []

    _require_equal(issues, data, "schema_name", PERSISTED_CHUNK_SCHEMA_NAME, candidate_id, record.chunk_id)
    if data.get("schema_version") not in SUPPORTED_PERSISTED_CHUNK_SCHEMA_VERSIONS:
        _add_issue(
            issues,
            "PERSISTED_SCHEMA_VERSION_UNSUPPORTED",
            "error",
            "Unsupported persisted chunk schema version.",
            "$.schema_version",
            candidate_id,
            record.chunk_id,
        )
    if data.get("schema_version") != PERSISTED_CHUNK_SCHEMA_VERSION:
        _add_issue(
            issues,
            "PERSISTED_SCHEMA_VERSION_MISMATCH",
            "error",
            "Persisted chunk schema version does not match D.5b version.",
            "$.schema_version",
            candidate_id,
            record.chunk_id,
        )

    for field_name in (
        "chunk_id",
        "document_id",
        "source_filename",
        "source_checksum",
        "text",
        "content_type",
        "parser_name",
        "parser_version",
        "structured_document_schema_version",
        "adapter_version",
        "persistence_mapper_version",
        "record_origin",
        "provenance_status",
        "validation_status",
    ):
        if not _non_empty_string(data.get(field_name)):
            _add_issue(
                issues,
                "PERSISTED_REQUIRED_FIELD_MISSING",
                "error",
                f"{field_name} must be a non-empty string.",
                f"$.{field_name}",
                candidate_id,
                record.chunk_id,
            )

    if not PERSISTED_CHUNK_ID_PATTERN.match(record.chunk_id):
        _add_issue(
            issues,
            "PERSISTED_CHUNK_ID_INVALID",
            "error",
            "chunk_id must use <document_id>:chunk:<24 lowercase hex>.",
            "$.chunk_id",
            candidate_id,
            record.chunk_id,
        )
    elif not record.chunk_id.startswith(f"{record.document_id}:chunk:"):
        _add_issue(
            issues,
            "PERSISTED_CHUNK_ID_DOCUMENT_MISMATCH",
            "error",
            "chunk_id must be namespaced by document_id.",
            "$.chunk_id",
            candidate_id,
            record.chunk_id,
        )

    if record.chunk_index < 0:
        _add_issue(
            issues,
            "PERSISTED_CHUNK_INDEX_INVALID",
            "error",
            "chunk_index must be zero or greater.",
            "$.chunk_index",
            candidate_id,
            record.chunk_id,
        )

    if not SHA256_HEX_PATTERN.match(record.source_checksum):
        _add_issue(
            issues,
            "SOURCE_CHECKSUM_INVALID",
            "error",
            "source_checksum must be a lowercase 64-character SHA256 hex digest.",
            "$.source_checksum",
            candidate_id,
            record.chunk_id,
        )

    if _looks_like_private_path(record.source_filename):
        _add_issue(
            issues,
            "SOURCE_FILENAME_PRIVATE_PATH",
            "error",
            "source_filename must not contain a machine-specific path.",
            "$.source_filename",
            candidate_id,
            record.chunk_id,
        )

    if record.content_type not in PERSISTED_CONTENT_TYPES:
        _add_issue(
            issues,
            "CONTENT_TYPE_UNSUPPORTED",
            "error",
            f"Unsupported persisted content_type: {record.content_type!r}.",
            "$.content_type",
            candidate_id,
            record.chunk_id,
        )

    _validate_page_ranges(issues, record, candidate_id)
    _validate_unique_non_empty_tuple(
        issues,
        record.source_block_ids,
        "$.source_block_ids",
        "SOURCE_BLOCK_IDS_INVALID",
        candidate_id,
        record.chunk_id,
        required=True,
    )
    for path, values in (
        ("$.table_ids", record.table_ids),
        ("$.figure_ids", record.figure_ids),
        ("$.equation_ids", record.equation_ids),
        ("$.admonition_ids", record.admonition_ids),
        ("$.cross_reference_ids", record.cross_reference_ids),
    ):
        _validate_unique_non_empty_tuple(
            issues,
            values,
            path,
            "ENTITY_IDS_INVALID",
            candidate_id,
            record.chunk_id,
            required=False,
        )

    if record.record_origin not in RECORD_ORIGINS:
        _add_issue(issues, "RECORD_ORIGIN_UNSUPPORTED", "error", "record_origin is unsupported.", "$.record_origin", candidate_id, record.chunk_id)
    elif record.record_origin != GENERATED_RECORD_ORIGIN:
        _add_issue(issues, "RECORD_ORIGIN_REJECTED", "error", "D.5b can generate only new_structured records.", "$.record_origin", candidate_id, record.chunk_id)

    if record.provenance_status not in PROVENANCE_STATUSES:
        _add_issue(issues, "PROVENANCE_STATUS_UNSUPPORTED", "error", "provenance_status is unsupported.", "$.provenance_status", candidate_id, record.chunk_id)
    elif record.provenance_status in {"legacy_filename_only", "unknown_provenance"}:
        _add_issue(issues, "PROVENANCE_STATUS_REJECTED", "error", "provenance_status is rejected for D.5b new structured records.", "$.provenance_status", candidate_id, record.chunk_id)

    if record.validation_status not in VALIDATION_STATUSES:
        _add_issue(issues, "VALIDATION_STATUS_UNSUPPORTED", "error", "validation_status is unsupported.", "$.validation_status", candidate_id, record.chunk_id)
    if record.validation_status == "rejected":
        _add_issue(issues, "REJECTED_RECORD_PRESENT", "error", "Rejected records must not be persisted.", "$.validation_status", candidate_id, record.chunk_id)

    issues.extend(validate_limitation_codes(record.accepted_limitation_codes, record.content_type, candidate_id, record.chunk_id))
    _validate_status_consistency(issues, record, candidate_id)
    _validate_forbidden_fields(issues, data, candidate_id, record.chunk_id)

    return tuple(sort_persisted_chunk_issues(issues))


def validate_persisted_chunk_records(
    records: list[PersistedChunkRecord],
) -> tuple[PersistedChunkIssue, ...]:
    """Validate a record list and package-level identity/index constraints."""
    issues: list[PersistedChunkIssue] = []
    chunk_ids = [record.chunk_id for record in records]
    for chunk_id, count in sorted(Counter(chunk_ids).items()):
        if count > 1:
            _add_issue(
                issues,
                "PERSISTED_CHUNK_ID_COLLISION",
                "error",
                f"Duplicate persisted chunk_id appears {count} times.",
                "$.records",
                chunk_id=chunk_id,
            )
    expected_indexes = list(range(len(records)))
    actual_indexes = [record.chunk_index for record in records]
    if actual_indexes != expected_indexes:
        _add_issue(
            issues,
            "PERSISTED_CHUNK_INDEX_SEQUENCE_INVALID",
            "error",
            "Accepted persisted records must have contiguous zero-based indexes.",
            "$.records[].chunk_index",
        )
    for record in records:
        issues.extend(validate_persisted_chunk_record(record))
    return tuple(sort_persisted_chunk_issues(issues))


def validate_limitation_codes(
    limitation_codes: tuple[str, ...],
    content_type: str,
    candidate_id: str | None = None,
    chunk_id: str | None = None,
) -> tuple[PersistedChunkIssue, ...]:
    """Validate accepted limitation codes against the approved registry."""
    issues: list[PersistedChunkIssue] = []
    for code, count in sorted(Counter(limitation_codes).items()):
        if count > 1:
            _add_issue(
                issues,
                "LIMITATION_CODE_DUPLICATE",
                "warning",
                f"Duplicate limitation code normalized: {code}.",
                "$.accepted_limitation_codes",
                candidate_id,
                chunk_id,
            )
        if code not in APPROVED_LIMITATION_CODES:
            _add_issue(
                issues,
                "LIMITATION_CODE_UNKNOWN",
                "error",
                f"Unknown accepted limitation code: {code}.",
                "$.accepted_limitation_codes",
                candidate_id,
                chunk_id,
            )
            continue
        allowed_types = LIMITATION_ALLOWED_CONTENT_TYPES[code]
        if content_type not in allowed_types:
            _add_issue(
                issues,
                "LIMITATION_CONTENT_TYPE_NOT_ALLOWED",
                "error",
                f"Limitation {code} is not allowed for content_type {content_type!r}.",
                "$.accepted_limitation_codes",
                candidate_id,
                chunk_id,
            )
    return tuple(sort_persisted_chunk_issues(issues))


def limitation_warning_codes(limitation_codes: tuple[str, ...]) -> tuple[str, ...]:
    """Return deterministic warning codes implied by limitation codes."""
    return tuple(
        LIMITATION_WARNING_CODES[code]
        for code in sorted(set(limitation_codes))
        if code in LIMITATION_WARNING_CODES
    )


def limitations_require_review(limitation_codes: tuple[str, ...]) -> bool:
    """Return whether any limitation code requires review."""
    return any(code in LIMITATIONS_REQUIRING_REVIEW for code in limitation_codes)


def sort_persisted_chunk_issues(issues: list[PersistedChunkIssue]) -> list[PersistedChunkIssue]:
    """Sort issues deterministically."""
    severity_order = {"error": 0, "warning": 1}
    return sorted(
        issues,
        key=lambda issue: (
            severity_order.get(issue.severity, 99),
            issue.path,
            issue.code,
            issue.message,
            issue.candidate_id or "",
            issue.chunk_id or "",
        ),
    )


def _validate_page_ranges(
    issues: list[PersistedChunkIssue],
    record: PersistedChunkRecord,
    candidate_id: str | None,
) -> None:
    if record.page_start < 1 or record.page_end < record.page_start:
        _add_issue(issues, "PAGE_RANGE_INVALID", "error", "page range must be positive and ordered.", "$.page_start", candidate_id, record.chunk_id)
    if record.pdf_page_index_start < 0 or record.pdf_page_index_end < record.pdf_page_index_start:
        _add_issue(issues, "PDF_PAGE_INDEX_RANGE_INVALID", "error", "PDF page-index range must be non-negative and ordered.", "$.pdf_page_index_start", candidate_id, record.chunk_id)
    if not record.contributing_page_numbers:
        _add_issue(issues, "CONTRIBUTING_PAGES_MISSING", "error", "contributing_page_numbers must not be empty.", "$.contributing_page_numbers", candidate_id, record.chunk_id)
    if not record.contributing_pdf_page_indexes:
        _add_issue(issues, "CONTRIBUTING_PDF_INDEXES_MISSING", "error", "contributing_pdf_page_indexes must not be empty.", "$.contributing_pdf_page_indexes", candidate_id, record.chunk_id)


def _validate_unique_non_empty_tuple(
    issues: list[PersistedChunkIssue],
    values: tuple[str, ...],
    path: str,
    code: str,
    candidate_id: str | None,
    chunk_id: str | None,
    *,
    required: bool,
) -> None:
    if required and not values:
        _add_issue(issues, code, "error", "Value list must not be empty.", path, candidate_id, chunk_id)
    if any(not _non_empty_string(value) for value in values):
        _add_issue(issues, code, "error", "Values must be non-empty strings.", path, candidate_id, chunk_id)
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    for duplicate in duplicates:
        _add_issue(issues, code, "error", f"Duplicate value is not allowed: {duplicate}.", path, candidate_id, chunk_id)


def _validate_status_consistency(
    issues: list[PersistedChunkIssue],
    record: PersistedChunkRecord,
    candidate_id: str | None,
) -> None:
    has_warnings = bool(record.warning_codes or record.accepted_limitation_codes)
    if record.validation_status == "valid" and has_warnings:
        _add_issue(issues, "VALIDATION_STATUS_INCONSISTENT", "error", "valid records must not have warnings or limitations.", "$.validation_status", candidate_id, record.chunk_id)
    if record.validation_status == "review_required" and not record.review_required:
        _add_issue(issues, "VALIDATION_STATUS_INCONSISTENT", "error", "review_required status requires review_required=true.", "$.review_required", candidate_id, record.chunk_id)
    if record.provenance_status == "partial_provenance" and not record.review_required:
        _add_issue(issues, "PARTIAL_PROVENANCE_REVIEW_REQUIRED", "error", "partial_provenance requires review_required=true.", "$.provenance_status", candidate_id, record.chunk_id)
    if record.provenance_status == "partial_provenance" and not record.accepted_limitation_codes:
        _add_issue(issues, "PARTIAL_PROVENANCE_LIMITATION_REQUIRED", "error", "partial_provenance requires an accepted limitation code.", "$.accepted_limitation_codes", candidate_id, record.chunk_id)
    if record.persistence_mapper_version != PERSISTED_CHUNK_MAPPER_VERSION:
        _add_issue(issues, "MAPPER_VERSION_MISMATCH", "error", "persistence_mapper_version is unsupported.", "$.persistence_mapper_version", candidate_id, record.chunk_id)


def _validate_forbidden_fields(
    issues: list[PersistedChunkIssue],
    data: Mapping[str, Any],
    candidate_id: str | None,
    chunk_id: str | None,
    path: str = "$",
) -> None:
    for key, value in data.items():
        key_text = str(key)
        key_lower = key_text.lower()
        item_path = f"{path}.{key_text}"
        if key_lower in FORBIDDEN_PERSISTED_FIELDS:
            _add_issue(issues, "FORBIDDEN_FIELD_PRESENT", "error", f"Forbidden persisted field present: {key_text}.", item_path, candidate_id, chunk_id)
        if isinstance(value, Mapping):
            _validate_forbidden_fields(issues, value, candidate_id, chunk_id, item_path)


def _require_equal(
    issues: list[PersistedChunkIssue],
    data: Mapping[str, Any],
    field_name: str,
    expected: str,
    candidate_id: str | None,
    chunk_id: str | None,
) -> None:
    if data.get(field_name) != expected:
        _add_issue(issues, "PERSISTED_REQUIRED_FIELD_INVALID", "error", f"{field_name} must be {expected!r}.", f"$.{field_name}", candidate_id, chunk_id)


def _add_issue(
    issues: list[PersistedChunkIssue],
    code: str,
    severity: str,
    message: str,
    path: str,
    candidate_id: str | None = None,
    chunk_id: str | None = None,
) -> None:
    issues.append(
        PersistedChunkIssue(
            code=code,
            severity=severity,
            message=message,
            path=path,
            candidate_id=candidate_id,
            chunk_id=chunk_id,
        )
    )


def _looks_like_private_path(value: str) -> bool:
    if "://" in value:
        return False
    return any(pattern.search(value) for pattern in PRIVATE_PATH_PATTERNS)


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


__all__ = [
    "APPROVED_LIMITATION_CODES",
    "LIMITATION_ALLOWED_CONTENT_TYPES",
    "LIMITATION_CHUNK_SECTION_CROSSING_REVIEW",
    "LIMITATION_DUPLICATE_TEXT_LINES",
    "LIMITATION_TABLE_CANDIDATE_ONLY",
    "LIMITATION_TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE",
    "LIMITATION_WARNING_CODES",
    "LIMITATIONS_REQUIRING_REVIEW",
    "PERSISTED_CHUNK_LIMITATION_REGISTRY_VERSION",
    "PersistedChunkIssue",
    "limitation_warning_codes",
    "limitations_require_review",
    "persisted_chunk_issue_to_dict",
    "sort_persisted_chunk_issues",
    "validate_limitation_codes",
    "validate_persisted_chunk_record",
    "validate_persisted_chunk_records",
]
