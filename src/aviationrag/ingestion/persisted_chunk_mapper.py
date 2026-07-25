"""Candidate-to-persisted-chunk mapper for D.5b dry runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence

from aviationrag.ingestion.persisted_chunk_record import (
    GENERATED_RECORD_ORIGIN,
    PERSISTED_CHUNK_MAPPER_VERSION,
    PERSISTED_CHUNK_SCHEMA_NAME,
    PERSISTED_CHUNK_SCHEMA_VERSION,
    PERSISTED_CONTENT_TYPES,
    PersistedChunkRecord,
)
from aviationrag.ingestion.persisted_chunk_validator import (
    APPROVED_LIMITATION_CODES,
    LIMITATION_CHUNK_SECTION_CROSSING_REVIEW,
    LIMITATION_DUPLICATE_TEXT_LINES,
    LIMITATION_TABLE_CANDIDATE_ONLY,
    LIMITATION_TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE,
    PersistedChunkIssue,
    limitation_warning_codes,
    limitations_require_review,
    sort_persisted_chunk_issues,
    validate_limitation_codes,
    validate_persisted_chunk_record,
)
from aviationrag.ingestion.structured_document_adapter import (
    SUPPORTED_SCHEMA_VERSION as STRUCTURED_DOCUMENT_SCHEMA_VERSION,
    StructuredDocumentChunkCandidate,
)


ADAPTER_VERSION = "D.4c"

ADAPTER_TO_PERSISTED_CONTENT_TYPES = {
    "caution": "caution",
    "definition": "definition",
    "figure_caption": "figure_caption",
    "note": "note",
    "paragraph": "paragraph",
    "procedure": "procedure",
    "requirement": "requirement",
    "table": "table",
    "warning": "warning",
}

HEADING_CONTENT_TYPES = frozenset({"section", "section_heading", "appendix_heading"})


@dataclass(frozen=True)
class PersistedChunkCandidateContext:
    """Explicit candidate-level governance context."""

    accepted_limitation_codes: tuple[str, ...] = ()
    warning_codes: tuple[str, ...] = ()
    review_required: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "accepted_limitation_codes",
            tuple(sorted(set(self.accepted_limitation_codes))),
        )
        object.__setattr__(self, "warning_codes", tuple(sorted(set(self.warning_codes))))


@dataclass(frozen=True)
class PersistedChunkMappingPolicy:
    """Fail-closed mapping policy for D.5b local packages."""

    allow_partial_provenance: bool = False
    allow_review_required_records: bool = True
    include_heading_records: bool = False
    approved_limitation_codes: frozenset[str] = field(default_factory=lambda: APPROVED_LIMITATION_CODES)
    strict_unknown_content_types: bool = True


@dataclass(frozen=True)
class PersistedChunkMappingResult:
    """Candidate mapping result."""

    candidate_id: str
    record: PersistedChunkRecord | None
    is_accepted: bool
    issues: tuple[PersistedChunkIssue, ...] = field(default_factory=tuple)
    warnings: tuple[PersistedChunkIssue, ...] = field(default_factory=tuple)


def build_persisted_chunk_id(
    *,
    document_id: str,
    content_type: str,
    content_subtype: str | None,
    source_block_ids: Sequence[str],
    table_ids: Sequence[str] = (),
    figure_ids: Sequence[str] = (),
    equation_ids: Sequence[str] = (),
    admonition_ids: Sequence[str] = (),
    cross_reference_ids: Sequence[str] = (),
    sequence_key: str,
) -> str:
    """Build a deterministic persisted chunk ID.

    Canonical payload keys are sorted during JSON serialization. The semantic
    payload contains document identity, persisted schema identity, content
    role, ordered source block IDs, ordered entity IDs, and the candidate
    sequence key. No filename, path, timestamp, embedding, vector, Astra, or
    FAISS data participates.
    """
    payload = {
        "admonition_ids": list(admonition_ids),
        "content_subtype": content_subtype or "",
        "content_type": content_type,
        "cross_reference_ids": list(cross_reference_ids),
        "document_id": document_id,
        "equation_ids": list(equation_ids),
        "figure_ids": list(figure_ids),
        "schema_name": PERSISTED_CHUNK_SCHEMA_NAME,
        "schema_version": PERSISTED_CHUNK_SCHEMA_VERSION,
        "sequence_key": sequence_key,
        "source_block_ids": list(source_block_ids),
        "table_ids": list(table_ids),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = sha256(encoded).hexdigest()[:24]
    return f"{document_id}:chunk:{digest}"


def map_candidate_to_persisted_chunk(
    candidate: StructuredDocumentChunkCandidate,
    *,
    chunk_index: int,
    context: PersistedChunkCandidateContext | None = None,
    policy: PersistedChunkMappingPolicy | None = None,
) -> PersistedChunkMappingResult:
    """Map one D.4c candidate into a validated persisted chunk record."""
    active_context = context or PersistedChunkCandidateContext()
    active_policy = policy or PersistedChunkMappingPolicy()
    issues: list[PersistedChunkIssue] = []
    warnings: list[PersistedChunkIssue] = []
    candidate_id = candidate.chunk_candidate_id

    limitation_codes = tuple(sorted(set(active_context.accepted_limitation_codes)))
    unknown_policy_codes = sorted(set(limitation_codes) - set(active_policy.approved_limitation_codes))
    for code in unknown_policy_codes:
        _add_issue(
            issues,
            "LIMITATION_CODE_UNKNOWN",
            "error",
            f"Unknown accepted limitation code: {code}.",
            "$.accepted_limitation_codes",
            candidate_id,
        )

    content_type = _persisted_content_type(candidate, active_policy, issues)
    if content_type is None:
        return _mapping_result(candidate_id, None, issues, warnings)

    warnings.extend(_warning_issues_for_codes(candidate_id, limitation_warning_codes(limitation_codes)))
    warnings.extend(_warning_issues_for_codes(candidate_id, active_context.warning_codes))
    issues.extend(validate_limitation_codes(limitation_codes, content_type, candidate_id))

    provenance_status = _map_provenance(candidate, limitation_codes, active_context, active_policy, issues)
    if provenance_status is None:
        return _mapping_result(candidate_id, None, issues, warnings)

    _validate_candidate_required_fields(candidate, issues)
    if issues:
        return _mapping_result(candidate_id, None, issues, warnings)

    review_required = bool(
        active_context.review_required
        or limitations_require_review(limitation_codes)
        or provenance_status == "partial_provenance"
        or candidate.content_type in HEADING_CONTENT_TYPES
        or content_type in {"unknown", "mixed", "unknown_admonition"}
    )
    if review_required and not active_policy.allow_review_required_records:
        _add_issue(
            issues,
            "REVIEW_RECORDS_DISABLED",
            "error",
            "Policy does not allow review-required records.",
            "$.review_required",
            candidate_id,
        )
        return _mapping_result(candidate_id, None, issues, warnings)

    warning_codes = tuple(sorted(set(active_context.warning_codes + limitation_warning_codes(limitation_codes))))
    validation_status = _validation_status(review_required, warning_codes, limitation_codes)
    record = PersistedChunkRecord(
        schema_name=PERSISTED_CHUNK_SCHEMA_NAME,
        schema_version=PERSISTED_CHUNK_SCHEMA_VERSION,
        chunk_id=build_persisted_chunk_id(
            document_id=candidate.document_id,
            content_type=content_type,
            content_subtype=None,
            source_block_ids=candidate.source_block_ids,
            table_ids=candidate.table_ids,
            figure_ids=candidate.figure_ids,
            equation_ids=candidate.equation_ids,
            admonition_ids=candidate.admonition_ids,
            cross_reference_ids=candidate.cross_reference_ids,
            sequence_key=candidate.chunk_candidate_id,
        ),
        chunk_index=chunk_index,
        document_id=candidate.document_id,
        source_filename=str(candidate.source_filename or ""),
        source_checksum=str(candidate.source_checksum or ""),
        document_title=candidate.document_title,
        document_number=candidate.document_number,
        document_revision=candidate.document_revision,
        document_issue=None,
        effective_date=None,
        text=candidate.text,
        normalized_text=candidate.normalized_text,
        content_type=content_type,
        content_subtype=None,
        language=None,
        page_start=int(candidate.page_start or 0),
        page_end=int(candidate.page_end or 0),
        pdf_page_index_start=int(candidate.pdf_page_index_start if candidate.pdf_page_index_start is not None else -1),
        pdf_page_index_end=int(candidate.pdf_page_index_end if candidate.pdf_page_index_end is not None else -1),
        contributing_page_numbers=_int_range(candidate.page_start, candidate.page_end),
        contributing_pdf_page_indexes=_int_range(candidate.pdf_page_index_start, candidate.pdf_page_index_end),
        printed_page_labels=tuple(candidate.printed_page_labels),
        section_id=candidate.section_id,
        section_path=tuple(candidate.section_path),
        section_number=candidate.section_number,
        section_title=candidate.section_title,
        clause_identifier=candidate.clause_identifier,
        source_block_ids=tuple(candidate.source_block_ids),
        source_span=_source_span(candidate),
        table_ids=tuple(candidate.table_ids),
        figure_ids=tuple(candidate.figure_ids),
        equation_ids=tuple(candidate.equation_ids),
        admonition_ids=tuple(candidate.admonition_ids),
        cross_reference_ids=tuple(candidate.cross_reference_ids),
        parser_name=str(candidate.parser_name or ""),
        parser_version=str(candidate.parser_version or ""),
        structured_document_schema_version=STRUCTURED_DOCUMENT_SCHEMA_VERSION,
        adapter_version=ADAPTER_VERSION,
        persistence_mapper_version=PERSISTED_CHUNK_MAPPER_VERSION,
        extraction_method=candidate.extraction_method,
        record_origin="new_structured",
        provenance_status=provenance_status,
        accepted_limitation_codes=limitation_codes,
        validation_status=validation_status,
        warning_codes=warning_codes,
        review_required=review_required,
    )
    validation_issues = validate_persisted_chunk_record(record, candidate_id=candidate_id)
    issues.extend(issue for issue in validation_issues if issue.severity == "error")
    warnings.extend(issue for issue in validation_issues if issue.severity == "warning")
    if issues:
        return _mapping_result(candidate_id, None, issues, warnings)
    return _mapping_result(candidate_id, record, issues, warnings)


def _persisted_content_type(
    candidate: StructuredDocumentChunkCandidate,
    policy: PersistedChunkMappingPolicy,
    issues: list[PersistedChunkIssue],
) -> str | None:
    content_type = candidate.content_type
    if content_type in HEADING_CONTENT_TYPES:
        if not policy.include_heading_records:
            _add_issue(
                issues,
                "HEADING_RECORD_DISABLED",
                "error",
                "Heading records are disabled by default.",
                "$.content_type",
                candidate.chunk_candidate_id,
            )
            return None
        return "appendix_content" if content_type == "appendix_heading" else "reference"

    if content_type == "other":
        if candidate.equation_ids:
            return "equation"
        if candidate.admonition_ids:
            return "unknown_admonition"
        if policy.strict_unknown_content_types:
            _add_issue(
                issues,
                "CONTENT_TYPE_UNSUPPORTED",
                "error",
                "D.4c content_type 'other' requires explicit supported entity evidence.",
                "$.content_type",
                candidate.chunk_candidate_id,
            )
            return None
        return "unknown"

    mapped = ADAPTER_TO_PERSISTED_CONTENT_TYPES.get(content_type, content_type)
    if mapped not in PERSISTED_CONTENT_TYPES:
        _add_issue(
            issues,
            "CONTENT_TYPE_UNSUPPORTED",
            "error",
            f"Unsupported persisted content type: {content_type!r}.",
            "$.content_type",
            candidate.chunk_candidate_id,
        )
        return None
    return mapped


def _map_provenance(
    candidate: StructuredDocumentChunkCandidate,
    limitation_codes: tuple[str, ...],
    context: PersistedChunkCandidateContext,
    policy: PersistedChunkMappingPolicy,
    issues: list[PersistedChunkIssue],
) -> str | None:
    if candidate.provenance_status == "structured":
        return "full_provenance"
    if candidate.provenance_status == "structured_partial":
        if not policy.allow_partial_provenance:
            _add_issue(
                issues,
                "PARTIAL_PROVENANCE_DISABLED",
                "error",
                "Partial provenance is disabled by default.",
                "$.provenance_status",
                candidate.chunk_candidate_id,
            )
            return None
        if not limitation_codes:
            _add_issue(
                issues,
                "PARTIAL_PROVENANCE_LIMITATION_REQUIRED",
                "error",
                "Partial provenance requires an explicit approved limitation code.",
                "$.accepted_limitation_codes",
                candidate.chunk_candidate_id,
            )
            return None
        if not context.review_required:
            _add_issue(
                issues,
                "PARTIAL_PROVENANCE_REVIEW_REQUIRED",
                "error",
                "Partial provenance requires review_required=true.",
                "$.review_required",
                candidate.chunk_candidate_id,
            )
            return None
        return "partial_provenance"
    _add_issue(
        issues,
        "PROVENANCE_STATUS_REJECTED",
        "error",
        f"Unsupported candidate provenance_status: {candidate.provenance_status!r}.",
        "$.provenance_status",
        candidate.chunk_candidate_id,
    )
    return None


def _validate_candidate_required_fields(
    candidate: StructuredDocumentChunkCandidate,
    issues: list[PersistedChunkIssue],
) -> None:
    for field_name, value in (
        ("document_id", candidate.document_id),
        ("source_filename", candidate.source_filename),
        ("source_checksum", candidate.source_checksum),
        ("text", candidate.text),
        ("parser_name", candidate.parser_name),
        ("parser_version", candidate.parser_version),
    ):
        if not _non_empty_string(value):
            _add_issue(
                issues,
                "CANDIDATE_REQUIRED_FIELD_MISSING",
                "error",
                f"{field_name} is required for persistence.",
                f"$.{field_name}",
                candidate.chunk_candidate_id,
            )
    if not candidate.source_block_ids:
        _add_issue(issues, "SOURCE_BLOCK_IDS_MISSING", "error", "source_block_ids are required.", "$.source_block_ids", candidate.chunk_candidate_id)
    if len(set(candidate.source_block_ids)) != len(candidate.source_block_ids):
        _add_issue(issues, "SOURCE_BLOCK_IDS_DUPLICATE", "error", "Duplicate source block IDs are not allowed.", "$.source_block_ids", candidate.chunk_candidate_id)
    if candidate.page_start is None or candidate.page_end is None or candidate.page_start < 1 or candidate.page_end < candidate.page_start:
        _add_issue(issues, "PAGE_RANGE_INVALID", "error", "page_start/page_end are required and must be ordered.", "$.page_start", candidate.chunk_candidate_id)
    if (
        candidate.pdf_page_index_start is None
        or candidate.pdf_page_index_end is None
        or candidate.pdf_page_index_start < 0
        or candidate.pdf_page_index_end < candidate.pdf_page_index_start
    ):
        _add_issue(issues, "PDF_PAGE_INDEX_RANGE_INVALID", "error", "PDF page indexes are required and must be ordered.", "$.pdf_page_index_start", candidate.chunk_candidate_id)


def _source_span(candidate: StructuredDocumentChunkCandidate) -> Mapping[str, Any]:
    return {
        "page_end": candidate.page_end,
        "page_start": candidate.page_start,
        "pdf_page_index_end": candidate.pdf_page_index_end,
        "pdf_page_index_start": candidate.pdf_page_index_start,
        "source_block_ids": list(candidate.source_block_ids),
    }


def _int_range(start: int | None, end: int | None) -> tuple[int, ...]:
    if start is None or end is None or end < start:
        return ()
    return tuple(range(start, end + 1))


def _validation_status(
    review_required: bool,
    warning_codes: tuple[str, ...],
    limitation_codes: tuple[str, ...],
) -> str:
    if review_required:
        return "review_required"
    if warning_codes or limitation_codes:
        return "valid_with_warnings"
    return "valid"


def _warning_issues_for_codes(candidate_id: str, codes: tuple[str, ...]) -> list[PersistedChunkIssue]:
    return [
        PersistedChunkIssue(
            code=code,
            severity="warning",
            message=f"Persisted chunk warning: {code}.",
            path="$.warning_codes",
            candidate_id=candidate_id,
        )
        for code in sorted(set(codes))
    ]


def _mapping_result(
    candidate_id: str,
    record: PersistedChunkRecord | None,
    issues: list[PersistedChunkIssue],
    warnings: list[PersistedChunkIssue],
) -> PersistedChunkMappingResult:
    sorted_issues = tuple(sort_persisted_chunk_issues(issues))
    sorted_warnings = tuple(sort_persisted_chunk_issues(warnings))
    return PersistedChunkMappingResult(
        candidate_id=candidate_id,
        record=record,
        is_accepted=record is not None and not sorted_issues,
        issues=sorted_issues,
        warnings=sorted_warnings,
    )


def _add_issue(
    issues: list[PersistedChunkIssue],
    code: str,
    severity: str,
    message: str,
    path: str,
    candidate_id: str,
) -> None:
    issues.append(
        PersistedChunkIssue(
            code=code,
            severity=severity,
            message=message,
            path=path,
            candidate_id=candidate_id,
        )
    )


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


__all__ = [
    "ADAPTER_TO_PERSISTED_CONTENT_TYPES",
    "ADAPTER_VERSION",
    "PersistedChunkCandidateContext",
    "PersistedChunkMappingPolicy",
    "PersistedChunkMappingResult",
    "build_persisted_chunk_id",
    "map_candidate_to_persisted_chunk",
]
