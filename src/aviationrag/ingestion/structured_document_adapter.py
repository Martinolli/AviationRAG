"""Offline adapter for techdoc-parser structured-document artifacts.

This module adapts already-exported structured-document JSON into candidate
chunk records for review. It does not parse documents, repair artifacts,
generate embeddings, connect to Astra, use FAISS, or integrate with runtime
ingestion scripts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

from aviationrag.ingestion.structured_document_validator import (
    StructuredDocumentValidationResult,
    structured_document_validation_result_to_dict,
    validate_structured_document,
)


DEFAULT_ADAPTER_OUTPUT_DIR = Path("data/migration_dry_run/structured_document_adapter")
ADAPTER_CANDIDATES_FILENAME = "adapted_chunk_candidates.jsonl"
ADAPTER_REPORT_FILENAME = "adapter_report.json"
ADAPTER_INTEGRITY_FILENAME = "artifact_integrity.json"

SUPPORTED_SCHEMA_NAME = "techdoc-structured-document"
SUPPORTED_SCHEMA_VERSION = "0.1.0"
STRUCTURED_DOCUMENT_MEDIA_TYPE = "application/json"
SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")

PASS = "PASS"
REVIEW = "REVIEW"
FAIL = "FAIL"

HEADING_BLOCK_TYPES = {"appendix_heading", "section_heading"}
NON_CANDIDATE_BLOCK_TYPES = {"metadata"}
ADAPTER_CANDIDATE_BLOCK_TYPES = {
    "caution",
    "definition",
    "equation",
    "figure_caption",
    "note",
    "paragraph",
    "procedure_step",
    "requirement",
    "table",
    "table_caption",
    "unknown",
    "warning",
}
BLOCK_CONTENT_TYPE_MAP = {
    "appendix_heading": "section",
    "caution": "caution",
    "definition": "definition",
    "equation": "other",
    "figure_caption": "figure_caption",
    "note": "note",
    "paragraph": "paragraph",
    "procedure_step": "procedure",
    "requirement": "requirement",
    "section_heading": "section",
    "table": "table",
    "table_caption": "table",
    "unknown": "other",
    "warning": "warning",
}
ADMONITION_TYPE_MAP = {
    "CAUTION": "caution",
    "IMPORTANT": "note",
    "NOTE": "note",
    "SAFETY_NOTICE": "warning",
    "UNKNOWN_ADMONITION": "other",
    "WARNING": "warning",
}
RESOLVED_REFERENCE_STATUS = "resolved"
UNRESOLVED_REFERENCE_STATUSES = {"ambiguous", "external", "not_attempted", "unresolved"}


@dataclass(frozen=True)
class StructuredDocumentAdapterIssue:
    """One deterministic adapter issue."""

    code: str
    severity: str
    message: str
    path: str
    entity_id: str | None = None


@dataclass(frozen=True)
class StructuredDocumentArtifactIntegrity:
    """Manifest and checksum integrity result for one structured artifact."""

    artifact_path: str
    manifest_path: str
    source_path: str | None
    schema_name: str | None
    schema_version: str | None
    document_id: str | None
    source_sha256_expected: str | None
    source_sha256_actual: str | None
    artifact_sha256_expected: str | None
    artifact_sha256_actual: str | None
    artifact_checksum_matches: bool
    source_checksum_matches: bool | None
    manifest_matches_artifact: bool
    is_valid: bool
    issues: tuple[StructuredDocumentAdapterIssue, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class StructuredDocumentChunkCandidate:
    """Candidate chunk derived from structured provenance, not a runtime chunk."""

    chunk_candidate_id: str
    document_id: str
    source_filename: str | None
    source_checksum: str | None
    document_title: str | None
    document_number: str | None
    document_revision: str | None
    text: str
    normalized_text: str | None
    content_type: str
    page_start: int | None
    page_end: int | None
    pdf_page_index_start: int | None
    pdf_page_index_end: int | None
    printed_page_labels: tuple[str, ...]
    section_id: str | None
    section_path: tuple[str, ...]
    section_number: str | None
    section_title: str | None
    clause_identifier: str | None
    source_block_ids: tuple[str, ...]
    table_ids: tuple[str, ...]
    figure_ids: tuple[str, ...]
    equation_ids: tuple[str, ...]
    admonition_ids: tuple[str, ...]
    cross_reference_ids: tuple[str, ...]
    parser_name: str | None
    parser_version: str | None
    extraction_method: str
    provenance_status: str


@dataclass(frozen=True)
class StructuredDocumentAdapterResult:
    """Complete report-only adapter result."""

    schema_name: str | None
    schema_version: str | None
    document_id: str | None
    artifact_integrity: StructuredDocumentArtifactIntegrity
    validator_result: StructuredDocumentValidationResult
    outcome: str
    candidates: tuple[StructuredDocumentChunkCandidate, ...]
    issues: tuple[StructuredDocumentAdapterIssue, ...]
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class StructuredDocumentAdapterWriteResult:
    """Result for explicitly approved local adapter-output writes."""

    output_dir: str
    candidates_output_path: str
    report_output_path: str
    integrity_output_path: str
    candidate_count: int
    report_written: bool
    candidates_written: bool
    integrity_written: bool


def load_structured_document_artifact(path: str | Path) -> dict[str, Any]:
    """Load a structured-document JSON artifact from disk."""
    artifact_path = Path(path)
    with artifact_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Structured-document artifact must be a JSON object: {artifact_path}")
    return data


def load_techdoc_parser_manifest(path: str | Path) -> dict[str, Any]:
    """Load a techdoc-parser manifest JSON object from disk."""
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"techdoc-parser manifest must be a JSON object: {manifest_path}")
    return data


def run_structured_document_adapter(
    artifact_path: str | Path,
    manifest_path: str | Path,
    *,
    source_path: str | Path | None = None,
    approved_warning_codes: Iterable[str] | None = None,
    include_headings: bool = False,
    strict_warnings: bool = False,
) -> StructuredDocumentAdapterResult:
    """Run the offline adapter against one artifact and manifest."""
    artifact = load_structured_document_artifact(artifact_path)
    manifest = load_techdoc_parser_manifest(manifest_path)
    source = Path(source_path) if source_path is not None else None
    approved_codes = set(approved_warning_codes or ())
    if strict_warnings:
        approved_codes = set()

    integrity = validate_structured_document_artifact_integrity(
        artifact,
        manifest,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        source_path=source,
    )
    validator = validate_structured_document(artifact)
    issues: list[StructuredDocumentAdapterIssue] = list(integrity.issues)
    issues.extend(_validator_issues(validator, approved_codes, strict_warnings))

    candidates: tuple[StructuredDocumentChunkCandidate, ...] = ()
    candidate_issues: list[StructuredDocumentAdapterIssue] = []
    if integrity.is_valid and validator.error_count == 0:
        candidates, candidate_issues = build_structured_document_chunk_candidates(
            artifact,
            source_checksum_verified=integrity.source_checksum_matches is True,
            include_headings=include_headings,
        )
        issues.extend(candidate_issues)

    issues = _sort_issues(issues)
    outcome = _outcome(issues)
    summary = _build_adapter_summary(
        artifact,
        integrity,
        validator,
        candidates,
        issues,
        approved_codes,
        include_headings,
        strict_warnings,
    )

    return StructuredDocumentAdapterResult(
        schema_name=_string_or_none(artifact.get("schema_name")),
        schema_version=_string_or_none(artifact.get("schema_version")),
        document_id=_document_id(artifact),
        artifact_integrity=integrity,
        validator_result=validator,
        outcome=outcome,
        candidates=candidates,
        issues=tuple(issues),
        summary=summary,
    )


def validate_structured_document_artifact_integrity(
    artifact: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    artifact_path: str | Path,
    manifest_path: str | Path,
    source_path: str | Path | None = None,
) -> StructuredDocumentArtifactIntegrity:
    """Validate manifest identity and raw SHA256 checksums without mutation."""
    artifact_file = Path(artifact_path)
    manifest_file = Path(manifest_path)
    source_file = Path(source_path) if source_path is not None else None
    issues: list[StructuredDocumentAdapterIssue] = []

    artifact_sha_expected = None
    source_sha_expected = None
    manifest_matches_artifact = True
    artifact_checksum_matches = False
    source_checksum_matches: bool | None = None

    entries = _structured_artifact_entries(manifest)
    if not entries:
        _add_issue(
            issues,
            "MANIFEST_STRUCTURED_ARTIFACT_MISSING",
            "error",
            "Manifest does not contain a structured_document artifact entry.",
            "$.artifacts",
        )
    elif len(entries) > 1:
        _add_issue(
            issues,
            "MANIFEST_STRUCTURED_ARTIFACT_DUPLICATE",
            "error",
            "Manifest contains multiple structured_document artifact entries.",
            "$.artifacts",
        )

    entry = entries[0] if len(entries) == 1 else {}
    if entry:
        manifest_matches_artifact = _validate_manifest_entry(
            entry,
            manifest,
            artifact,
            artifact_file,
            manifest_file,
            issues,
        )
        artifact_sha_expected = _string_or_none(entry.get("artifact_sha256"))
        source_sha_expected = _string_or_none(entry.get("source_sha256"))

    artifact_sha_actual = _sha256_file(artifact_file)
    artifact_checksum_matches = bool(
        artifact_sha_expected
        and _valid_sha256_hex(artifact_sha_expected)
        and artifact_sha_actual == artifact_sha_expected
    )
    if artifact_sha_expected is None:
        _add_issue(
            issues,
            "ARTIFACT_CHECKSUM_MISSING",
            "error",
            "Manifest structured_document entry is missing artifact_sha256.",
            "$.artifacts[].artifact_sha256",
        )
    elif not _valid_sha256_hex(artifact_sha_expected):
        _add_issue(
            issues,
            "ARTIFACT_CHECKSUM_INVALID",
            "error",
            "artifact_sha256 must be a lowercase 64-character SHA256 hex digest.",
            "$.artifacts[].artifact_sha256",
        )
    elif not artifact_checksum_matches:
        _add_issue(
            issues,
            "ARTIFACT_CHECKSUM_MISMATCH",
            "error",
            "artifact_sha256 does not match the artifact file bytes.",
            "$.artifacts[].artifact_sha256",
        )

    document_source_hash = _document_source_hash(artifact)
    if source_sha_expected is None:
        _add_issue(
            issues,
            "SOURCE_CHECKSUM_MISSING",
            "error",
            "Manifest structured_document entry is missing source_sha256.",
            "$.artifacts[].source_sha256",
        )
    elif not _valid_sha256_hex(source_sha_expected):
        _add_issue(
            issues,
            "SOURCE_CHECKSUM_INVALID",
            "error",
            "source_sha256 must be a lowercase 64-character SHA256 hex digest.",
            "$.artifacts[].source_sha256",
        )

    if document_source_hash and source_sha_expected and document_source_hash != source_sha_expected:
        _add_issue(
            issues,
            "DOCUMENT_SOURCE_CHECKSUM_MISMATCH",
            "error",
            "document.source_hash does not match manifest source_sha256.",
            "$.document.source_hash",
        )

    source_sha_actual = None
    if source_file is None:
        source_checksum_matches = None
        _add_issue(
            issues,
            "SOURCE_CHECKSUM_NOT_VERIFIED",
            "warning",
            "Source bytes were not provided, so source_sha256 was not verified.",
            "$.artifacts[].source_sha256",
        )
    else:
        source_sha_actual = _sha256_file(source_file)
        source_checksum_matches = bool(
            source_sha_expected
            and _valid_sha256_hex(source_sha_expected)
            and source_sha_actual == source_sha_expected
        )
        if not source_checksum_matches:
            _add_issue(
                issues,
                "SOURCE_CHECKSUM_MISMATCH",
                "error",
                "source_sha256 does not match the provided source file bytes.",
                "$.artifacts[].source_sha256",
            )

    sorted_issues = tuple(_sort_issues(issues))
    return StructuredDocumentArtifactIntegrity(
        artifact_path=str(artifact_file),
        manifest_path=str(manifest_file),
        source_path=str(source_file) if source_file is not None else None,
        schema_name=_string_or_none(artifact.get("schema_name")),
        schema_version=_string_or_none(artifact.get("schema_version")),
        document_id=_document_id(artifact),
        source_sha256_expected=source_sha_expected,
        source_sha256_actual=source_sha_actual,
        artifact_sha256_expected=artifact_sha_expected,
        artifact_sha256_actual=artifact_sha_actual,
        artifact_checksum_matches=artifact_checksum_matches,
        source_checksum_matches=source_checksum_matches,
        manifest_matches_artifact=manifest_matches_artifact,
        is_valid=not any(issue.severity == "error" for issue in sorted_issues),
        issues=sorted_issues,
    )


def build_structured_document_chunk_candidates(
    artifact: Mapping[str, Any],
    *,
    source_checksum_verified: bool,
    include_headings: bool = False,
) -> tuple[tuple[StructuredDocumentChunkCandidate, ...], list[StructuredDocumentAdapterIssue]]:
    """Build review-only chunk candidates from a validated structured document."""
    issues: list[StructuredDocumentAdapterIssue] = []
    document = _mapping(artifact.get("document"))
    document_id = _document_id(artifact) or ""
    source_filename = _first_string(document, artifact, "source_filename", "filename")
    source_checksum = _document_source_hash(artifact)
    document_title = _first_string(document, artifact, "document_title", "canonical_title", "title")
    document_number = _first_string(document, artifact, "document_number")
    document_revision = _first_string(document, artifact, "revision", "issue")
    parser_name = _string_or_none(artifact.get("parser_name"))
    parser_version = _string_or_none(artifact.get("parser_version"))

    pages = [_mapping(page) for page in _sequence(artifact.get("pages"))]
    sections = [_mapping(section) for section in _sequence(artifact.get("sections"))]
    blocks = [_mapping(block) for block in _sequence(artifact.get("blocks"))]
    tables = [_mapping(table) for table in _sequence(artifact.get("tables"))]
    figures = [_mapping(figure) for figure in _sequence(artifact.get("figures"))]
    equations = [_mapping(equation) for equation in _sequence(artifact.get("equations"))]
    admonitions = [_mapping(admonition) for admonition in _sequence(artifact.get("admonitions"))]
    cross_references = [_mapping(reference) for reference in _sequence(artifact.get("cross_references"))]

    page_by_number = {
        page.get("page_number"): page for page in pages if isinstance(page.get("page_number"), int)
    }
    section_by_id = {
        str(section.get("section_id")): section
        for section in sections
        if _non_empty_string(section.get("section_id"))
    }
    block_by_id = {
        str(block.get("block_id")): block for block in blocks if _non_empty_string(block.get("block_id"))
    }
    table_ids_by_block = _ids_by_source_block(tables, "table_id", "$.tables", issues)
    figure_ids_by_block = _ids_by_source_block(figures, "figure_id", "$.figures", issues)
    equation_ids_by_block = _ids_by_source_block(equations, "equation_id", "$.equations", issues)
    xref_ids_by_block = _cross_reference_ids_by_block(
        cross_references,
        known_targets=_known_cross_reference_targets(sections, tables, figures, equations),
        issues=issues,
    )
    admonitions_by_block = _admonitions_by_block(admonitions, "$.admonitions", issues)
    admonition_block_ids = set(admonitions_by_block)
    emitted_block_ids: set[str] = set()
    candidates: list[StructuredDocumentChunkCandidate] = []

    for admonition in sorted(admonitions, key=lambda item: str(item.get("admonition_id") or "")):
        source_block_ids = _string_tuple(admonition.get("source_block_ids"))
        if not source_block_ids:
            continue
        known_block_ids = tuple(block_id for block_id in source_block_ids if block_id in block_by_id)
        if not known_block_ids:
            continue
        first_block = block_by_id[known_block_ids[0]]
        candidates.append(
            _candidate_from_admonition(
                artifact=artifact,
                admonition=admonition,
                fallback_block=first_block,
                document_id=document_id,
                source_filename=source_filename,
                source_checksum=source_checksum,
                document_title=document_title,
                document_number=document_number,
                document_revision=document_revision,
                parser_name=parser_name,
                parser_version=parser_version,
                page_by_number=page_by_number,
                section_by_id=section_by_id,
                source_checksum_verified=source_checksum_verified,
                xref_ids_by_block=xref_ids_by_block,
            )
        )
        emitted_block_ids.update(known_block_ids)

    for block in sorted(blocks, key=_block_sort_key):
        block_id = _string_or_none(block.get("block_id"))
        block_type = _string_or_none(block.get("block_type")) or "unknown"
        if block_id is None:
            continue
        if block_id in emitted_block_ids:
            continue
        if block_type in admonition_block_ids:
            continue
        if block_type in NON_CANDIDATE_BLOCK_TYPES:
            continue
        if block_type in HEADING_BLOCK_TYPES and not include_headings:
            continue
        if block_type not in ADAPTER_CANDIDATE_BLOCK_TYPES and block_type not in HEADING_BLOCK_TYPES:
            _add_issue(
                issues,
                "BLOCK_CONTENT_TYPE_SKIPPED",
                "warning",
                f"Block content type is validated but not adapted as a chunk candidate: {block_type!r}.",
                f"$.blocks[{block_id}]",
                block_id,
            )
            continue
        candidates.append(
            _candidate_from_block(
                artifact=artifact,
                block=block,
                document_id=document_id,
                source_filename=source_filename,
                source_checksum=source_checksum,
                document_title=document_title,
                document_number=document_number,
                document_revision=document_revision,
                parser_name=parser_name,
                parser_version=parser_version,
                page_by_number=page_by_number,
                section_by_id=section_by_id,
                source_checksum_verified=source_checksum_verified,
                table_ids=tuple(sorted(table_ids_by_block.get(block_id, set()))),
                figure_ids=tuple(sorted(figure_ids_by_block.get(block_id, set()))),
                equation_ids=tuple(sorted(equation_ids_by_block.get(block_id, set()))),
                admonition_ids=tuple(
                    sorted(
                        str(item.get("admonition_id"))
                        for item in admonitions_by_block.get(block_id, [])
                        if _non_empty_string(item.get("admonition_id"))
                    )
                ),
                cross_reference_ids=tuple(sorted(xref_ids_by_block.get(block_id, set()))),
            )
        )
        emitted_block_ids.add(block_id)

    return tuple(candidates), _sort_issues(issues)


def write_structured_document_adapter_outputs(
    result: StructuredDocumentAdapterResult,
    output_dir: str | Path = DEFAULT_ADAPTER_OUTPUT_DIR,
    *,
    allow_local_write: bool = False,
) -> StructuredDocumentAdapterWriteResult:
    """Write local-only adapter reports when explicitly allowed."""
    if not allow_local_write:
        raise PermissionError(
            "Structured-document adapter writes are disabled. Pass allow_local_write=True "
            "or use the CLI --allow-local-write flag for an explicit local dry run."
        )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    candidates_path = _child_path(output_root, ADAPTER_CANDIDATES_FILENAME)
    report_path = _child_path(output_root, ADAPTER_REPORT_FILENAME)
    integrity_path = _child_path(output_root, ADAPTER_INTEGRITY_FILENAME)

    _write_jsonl(
        candidates_path,
        (
            structured_document_chunk_candidate_to_dict(candidate)
            for candidate in result.candidates
        ),
    )
    report_path.write_text(
        structured_document_adapter_result_to_json(result),
        encoding="utf-8",
    )
    integrity_path.write_text(
        json.dumps(
            structured_document_artifact_integrity_to_dict(result.artifact_integrity),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return StructuredDocumentAdapterWriteResult(
        output_dir=str(output_root),
        candidates_output_path=str(candidates_path),
        report_output_path=str(report_path),
        integrity_output_path=str(integrity_path),
        candidate_count=len(result.candidates),
        report_written=True,
        candidates_written=True,
        integrity_written=True,
    )


def structured_document_adapter_result_to_json(result: StructuredDocumentAdapterResult) -> str:
    """Return a stable JSON report string for an adapter result."""
    return json.dumps(
        structured_document_adapter_result_to_dict(result),
        indent=2,
        sort_keys=True,
    ) + "\n"


def structured_document_adapter_result_to_dict(
    result: StructuredDocumentAdapterResult,
) -> dict[str, Any]:
    """Return a JSON-serializable adapter result dictionary."""
    return {
        "schema_name": result.schema_name,
        "schema_version": result.schema_version,
        "document_id": result.document_id,
        "outcome": result.outcome,
        "artifact_integrity": structured_document_artifact_integrity_to_dict(
            result.artifact_integrity
        ),
        "validator_result": structured_document_validation_result_to_dict(
            result.validator_result
        ),
        "summary": dict(result.summary),
        "candidate_count": len(result.candidates),
        "candidates": [
            structured_document_chunk_candidate_to_dict(candidate)
            for candidate in result.candidates
        ],
        "issues": [structured_document_adapter_issue_to_dict(issue) for issue in result.issues],
    }


def structured_document_artifact_integrity_to_dict(
    integrity: StructuredDocumentArtifactIntegrity,
) -> dict[str, Any]:
    """Return a JSON-serializable artifact-integrity dictionary."""
    data = asdict(integrity)
    data["issues"] = [structured_document_adapter_issue_to_dict(issue) for issue in integrity.issues]
    return data


def structured_document_chunk_candidate_to_dict(
    candidate: StructuredDocumentChunkCandidate,
) -> dict[str, Any]:
    """Return a JSON-serializable chunk-candidate dictionary."""
    return asdict(candidate)


def structured_document_adapter_issue_to_dict(
    issue: StructuredDocumentAdapterIssue,
) -> dict[str, Any]:
    """Return a JSON-serializable adapter issue dictionary."""
    return asdict(issue)


def _validate_manifest_entry(
    entry: Mapping[str, Any],
    manifest: Mapping[str, Any],
    artifact: Mapping[str, Any],
    artifact_path: Path,
    manifest_path: Path,
    issues: list[StructuredDocumentAdapterIssue],
) -> bool:
    matches = True
    if entry.get("artifact_type") != "structured_document":
        matches = False
        _add_issue(
            issues,
            "MANIFEST_ARTIFACT_TYPE_MISMATCH",
            "error",
            "Structured artifact entry must use artifact_type='structured_document'.",
            "$.artifacts[].artifact_type",
        )
    if entry.get("media_type") != STRUCTURED_DOCUMENT_MEDIA_TYPE:
        matches = False
        _add_issue(
            issues,
            "MANIFEST_MEDIA_TYPE_MISMATCH",
            "error",
            "Structured artifact entry must use media_type='application/json'.",
            "$.artifacts[].media_type",
        )
    if entry.get("schema_name") != artifact.get("schema_name"):
        matches = False
        _add_issue(
            issues,
            "MANIFEST_SCHEMA_NAME_MISMATCH",
            "error",
            "Manifest schema_name does not match artifact schema_name.",
            "$.artifacts[].schema_name",
        )
    if entry.get("schema_version") != artifact.get("schema_version"):
        matches = False
        _add_issue(
            issues,
            "MANIFEST_SCHEMA_VERSION_MISMATCH",
            "error",
            "Manifest schema_version does not match artifact schema_version.",
            "$.artifacts[].schema_version",
        )
    if entry.get("document_id") != _document_id(artifact):
        matches = False
        _add_issue(
            issues,
            "MANIFEST_DOCUMENT_ID_MISMATCH",
            "error",
            "Manifest document_id does not match artifact document_id.",
            "$.artifacts[].document_id",
        )

    entry_path = _string_or_none(entry.get("path"))
    if entry_path is None:
        matches = False
        _add_issue(
            issues,
            "MANIFEST_ARTIFACT_PATH_MISSING",
            "error",
            "Structured artifact entry is missing path.",
            "$.artifacts[].path",
        )
    elif not _path_points_to_artifact(entry_path, artifact_path, manifest_path):
        matches = False
        _add_issue(
            issues,
            "MANIFEST_ARTIFACT_PATH_MISMATCH",
            "error",
            "Manifest structured_document path does not identify the provided artifact.",
            "$.artifacts[].path",
        )

    output_path = _structured_output_path(manifest)
    if output_path is not None and entry_path is not None and output_path != entry_path:
        matches = False
        _add_issue(
            issues,
            "MANIFEST_OUTPUT_PATH_MISMATCH",
            "error",
            "Manifest outputs.structured_document does not match the artifact entry path.",
            "$.outputs.structured_document",
        )

    return matches


def _structured_artifact_entries(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes, bytearray)):
        return []
    return [
        artifact
        for artifact in artifacts
        if isinstance(artifact, Mapping) and artifact.get("artifact_type") == "structured_document"
    ]


def _validator_issues(
    validator: StructuredDocumentValidationResult,
    approved_warning_codes: set[str],
    strict_warnings: bool,
) -> list[StructuredDocumentAdapterIssue]:
    issues: list[StructuredDocumentAdapterIssue] = []
    for issue in validator.issues:
        if issue.severity == "error":
            severity = "error"
            code = "VALIDATOR_ERROR"
        elif strict_warnings:
            severity = "error"
            code = "VALIDATOR_WARNING_STRICT"
        elif issue.code in approved_warning_codes:
            severity = "warning"
            code = "VALIDATOR_WARNING_APPROVED"
        else:
            severity = "error"
            code = "VALIDATOR_WARNING_UNAPPROVED"
        _add_issue(
            issues,
            code,
            severity,
            f"{issue.code}: {issue.message}",
            issue.path,
            issue.entity_id,
        )
    return issues


def _candidate_from_block(
    *,
    artifact: Mapping[str, Any],
    block: Mapping[str, Any],
    document_id: str,
    source_filename: str | None,
    source_checksum: str | None,
    document_title: str | None,
    document_number: str | None,
    document_revision: str | None,
    parser_name: str | None,
    parser_version: str | None,
    page_by_number: Mapping[int, Mapping[str, Any]],
    section_by_id: Mapping[str, Mapping[str, Any]],
    source_checksum_verified: bool,
    table_ids: tuple[str, ...],
    figure_ids: tuple[str, ...],
    equation_ids: tuple[str, ...],
    admonition_ids: tuple[str, ...],
    cross_reference_ids: tuple[str, ...],
) -> StructuredDocumentChunkCandidate:
    del artifact
    block_id = str(block.get("block_id") or "")
    block_type = str(block.get("block_type") or "unknown")
    section = section_by_id.get(str(block.get("section_id") or ""), {})
    page_start, page_end, pdf_start, pdf_end = _page_range(block)
    return StructuredDocumentChunkCandidate(
        chunk_candidate_id=_candidate_id(document_id, block_id),
        document_id=document_id,
        source_filename=source_filename,
        source_checksum=source_checksum,
        document_title=document_title,
        document_number=document_number,
        document_revision=document_revision,
        text=str(block.get("text") or ""),
        normalized_text=_string_or_none(block.get("normalized_text")),
        content_type=BLOCK_CONTENT_TYPE_MAP.get(block_type, "other"),
        page_start=page_start,
        page_end=page_end,
        pdf_page_index_start=pdf_start,
        pdf_page_index_end=pdf_end,
        printed_page_labels=_printed_page_labels(page_by_number, page_start, page_end),
        section_id=_string_or_none(block.get("section_id")),
        section_path=_section_path(section),
        section_number=_string_or_none(section.get("section_number")),
        section_title=_string_or_none(section.get("title")),
        clause_identifier=_first_string(block, section, "clause_identifier", "section_number"),
        source_block_ids=(block_id,),
        table_ids=table_ids,
        figure_ids=figure_ids,
        equation_ids=equation_ids,
        admonition_ids=admonition_ids,
        cross_reference_ids=cross_reference_ids,
        parser_name=parser_name,
        parser_version=parser_version,
        extraction_method="techdoc-parser structured-document adapter",
        provenance_status=_provenance_status(source_checksum_verified, page_start, page_end, (block_id,)),
    )


def _candidate_from_admonition(
    *,
    artifact: Mapping[str, Any],
    admonition: Mapping[str, Any],
    fallback_block: Mapping[str, Any],
    document_id: str,
    source_filename: str | None,
    source_checksum: str | None,
    document_title: str | None,
    document_number: str | None,
    document_revision: str | None,
    parser_name: str | None,
    parser_version: str | None,
    page_by_number: Mapping[int, Mapping[str, Any]],
    section_by_id: Mapping[str, Mapping[str, Any]],
    source_checksum_verified: bool,
    xref_ids_by_block: Mapping[str, set[str]],
) -> StructuredDocumentChunkCandidate:
    del artifact
    admonition_id = str(admonition.get("admonition_id") or "")
    source_block_ids = _string_tuple(admonition.get("source_block_ids"))
    section_id = _string_or_none(admonition.get("section_id")) or _string_or_none(
        fallback_block.get("section_id")
    )
    section = section_by_id.get(section_id or "", {})
    page_start, page_end, pdf_start, pdf_end = _page_range(admonition)
    if page_start is None:
        page_start, page_end, pdf_start, pdf_end = _page_range(fallback_block)
    elif pdf_start is None or pdf_end is None:
        _, _, fallback_pdf_start, fallback_pdf_end = _page_range(fallback_block)
        pdf_start = pdf_start if pdf_start is not None else fallback_pdf_start
        pdf_end = pdf_end if pdf_end is not None else fallback_pdf_end
    cross_reference_ids = tuple(
        sorted({xref for block_id in source_block_ids for xref in xref_ids_by_block.get(block_id, set())})
    )
    return StructuredDocumentChunkCandidate(
        chunk_candidate_id=_candidate_id(document_id, admonition_id),
        document_id=document_id,
        source_filename=source_filename,
        source_checksum=source_checksum,
        document_title=document_title,
        document_number=document_number,
        document_revision=document_revision,
        text=str(admonition.get("body_text") or ""),
        normalized_text=_string_or_none(admonition.get("normalized_text")),
        content_type=ADMONITION_TYPE_MAP.get(str(admonition.get("admonition_type") or ""), "other"),
        page_start=page_start,
        page_end=page_end,
        pdf_page_index_start=pdf_start,
        pdf_page_index_end=pdf_end,
        printed_page_labels=_printed_page_labels(page_by_number, page_start, page_end),
        section_id=section_id,
        section_path=_section_path(section),
        section_number=_string_or_none(section.get("section_number")),
        section_title=_string_or_none(section.get("title")),
        clause_identifier=_first_string(admonition, section, "clause_identifier", "section_number"),
        source_block_ids=source_block_ids,
        table_ids=(),
        figure_ids=(),
        equation_ids=(),
        admonition_ids=(admonition_id,),
        cross_reference_ids=cross_reference_ids,
        parser_name=parser_name,
        parser_version=parser_version,
        extraction_method="techdoc-parser structured-document adapter",
        provenance_status=_provenance_status(source_checksum_verified, page_start, page_end, source_block_ids),
    )


def _ids_by_source_block(
    entities: Sequence[Mapping[str, Any]],
    id_field: str,
    path: str,
    issues: list[StructuredDocumentAdapterIssue],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for index, entity in enumerate(entities):
        entity_id = _string_or_none(entity.get(id_field))
        source_block_ids = _string_tuple(entity.get("source_block_ids"))
        if entity_id is None:
            continue
        if not source_block_ids:
            _add_issue(
                issues,
                "ENTITY_SOURCE_BLOCK_MISSING",
                "error",
                f"{id_field} entity does not declare source_block_ids.",
                f"{path}[{index}].source_block_ids",
                entity_id,
            )
            continue
        for block_id in source_block_ids:
            result[block_id].add(entity_id)
    return result


def _admonitions_by_block(
    admonitions: Sequence[Mapping[str, Any]],
    path: str,
    issues: list[StructuredDocumentAdapterIssue],
) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, admonition in enumerate(admonitions):
        admonition_id = _string_or_none(admonition.get("admonition_id"))
        source_block_ids = _string_tuple(admonition.get("source_block_ids"))
        if admonition_id is None:
            continue
        if not source_block_ids:
            _add_issue(
                issues,
                "ADMONITION_SOURCE_BLOCK_MISSING",
                "error",
                "Admonition does not declare source_block_ids.",
                f"{path}[{index}].source_block_ids",
                admonition_id,
            )
            continue
        for block_id in source_block_ids:
            result[block_id].append(admonition)
    return result


def _cross_reference_ids_by_block(
    cross_references: Sequence[Mapping[str, Any]],
    known_targets: set[str],
    issues: list[StructuredDocumentAdapterIssue],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for index, reference in enumerate(cross_references):
        reference_id = _string_or_none(reference.get("reference_id"))
        status = _string_or_none(reference.get("resolution_status"))
        target_id = _string_or_none(reference.get("target_id"))
        if reference_id is None:
            continue
        if status == RESOLVED_REFERENCE_STATUS and target_id not in known_targets:
            _add_issue(
                issues,
                "CROSS_REFERENCE_TARGET_UNKNOWN",
                "error",
                "Resolved cross reference target_id does not exist in known structured targets.",
                f"$.cross_references[{index}].target_id",
                reference_id,
            )
        elif status in UNRESOLVED_REFERENCE_STATUSES and target_id is not None:
            _add_issue(
                issues,
                "CROSS_REFERENCE_UNRESOLVED_TARGET_PRESENT",
                "error",
                "Unresolved or external cross reference must not declare target_id.",
                f"$.cross_references[{index}].target_id",
                reference_id,
            )
        source_block_ids = _string_tuple(reference.get("source_block_ids"))
        for block_id in source_block_ids:
            result[block_id].add(reference_id)
    return result


def _known_cross_reference_targets(
    sections: Sequence[Mapping[str, Any]],
    tables: Sequence[Mapping[str, Any]],
    figures: Sequence[Mapping[str, Any]],
    equations: Sequence[Mapping[str, Any]],
) -> set[str]:
    targets: set[str] = set()
    for collection, key in (
        (sections, "section_id"),
        (tables, "table_id"),
        (figures, "figure_id"),
        (equations, "equation_id"),
    ):
        targets.update(
            str(item.get(key))
            for item in collection
            if _non_empty_string(item.get(key))
        )
    return targets


def _build_adapter_summary(
    artifact: Mapping[str, Any],
    integrity: StructuredDocumentArtifactIntegrity,
    validator: StructuredDocumentValidationResult,
    candidates: Sequence[StructuredDocumentChunkCandidate],
    issues: Sequence[StructuredDocumentAdapterIssue],
    approved_warning_codes: set[str],
    include_headings: bool,
    strict_warnings: bool,
) -> dict[str, Any]:
    content_type_counts = Counter(candidate.content_type for candidate in candidates)
    provenance_counts = Counter(candidate.provenance_status for candidate in candidates)
    blocks = [_mapping(block) for block in _sequence(artifact.get("blocks"))]
    reference_status_counts = Counter(
        str(reference.get("resolution_status") or "missing")
        for reference in _sequence(artifact.get("cross_references"))
        if isinstance(reference, Mapping)
    )
    return {
        "outcome": _outcome(issues),
        "candidate_count": len(candidates),
        "content_type_counts": dict(sorted(content_type_counts.items())),
        "provenance_status_counts": dict(sorted(provenance_counts.items())),
        "page_count": validator.summary.get("page_count", 0),
        "block_count": validator.summary.get("block_count", 0),
        "section_count": validator.summary.get("section_count", 0),
        "validator_table_count": validator.summary.get("table_count", 0),
        "table_entity_count": len(_sequence(artifact.get("tables"))),
        "table_block_count": sum(1 for block in blocks if block.get("block_type") == "table"),
        "table_candidate_count": content_type_counts.get("table", 0),
        "validator_figure_count": validator.summary.get("figure_count", 0),
        "figure_entity_count": len(_sequence(artifact.get("figures"))),
        "figure_block_count": sum(1 for block in blocks if block.get("block_type") == "figure"),
        "figure_caption_candidate_count": content_type_counts.get("figure_caption", 0),
        "validator_equation_count": validator.summary.get("equation_count", 0),
        "equation_entity_count": len(_sequence(artifact.get("equations"))),
        "equation_block_count": sum(1 for block in blocks if block.get("block_type") == "equation"),
        "validator_admonition_count": validator.summary.get("admonition_count", 0),
        "admonition_entity_count": len(_sequence(artifact.get("admonitions"))),
        "cross_reference_count": validator.summary.get("cross_reference_count", 0),
        "reference_status_counts": dict(sorted(reference_status_counts.items())),
        "issue_count": len(issues),
        "error_count": sum(1 for issue in issues if issue.severity == "error"),
        "warning_count": sum(1 for issue in issues if issue.severity == "warning"),
        "validator_error_count": validator.error_count,
        "validator_warning_count": validator.warning_count,
        "approved_warning_codes": sorted(approved_warning_codes),
        "include_headings": include_headings,
        "strict_warnings": strict_warnings,
        "artifact_checksum_matches": integrity.artifact_checksum_matches,
        "source_checksum_matches": integrity.source_checksum_matches,
        "manifest_matches_artifact": integrity.manifest_matches_artifact,
        "outputs_are_local_only": True,
        "embeddings_generated": False,
        "astra_touched": False,
        "faiss_touched": False,
        "runtime_ingestion_modified": False,
    }


def _add_issue(
    issues: list[StructuredDocumentAdapterIssue],
    code: str,
    severity: str,
    message: str,
    path: str,
    entity_id: str | None = None,
) -> None:
    issues.append(
        StructuredDocumentAdapterIssue(
            code=code,
            severity=severity,
            message=message,
            path=path,
            entity_id=entity_id,
        )
    )


def _outcome(issues: Sequence[StructuredDocumentAdapterIssue]) -> str:
    if any(issue.severity == "error" for issue in issues):
        return FAIL
    if any(issue.severity == "warning" for issue in issues):
        return REVIEW
    return PASS


def _sort_issues(
    issues: Iterable[StructuredDocumentAdapterIssue],
) -> list[StructuredDocumentAdapterIssue]:
    severity_order = {"error": 0, "warning": 1}
    return sorted(
        issues,
        key=lambda issue: (
            severity_order.get(issue.severity, 99),
            issue.path,
            issue.code,
            issue.message,
            issue.entity_id or "",
        ),
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha256_hex(value: str) -> bool:
    return bool(SHA256_HEX_PATTERN.fullmatch(value))


def _document_id(artifact: Mapping[str, Any]) -> str | None:
    document = _mapping(artifact.get("document"))
    return _first_string(document, artifact, "document_id")


def _document_source_hash(artifact: Mapping[str, Any]) -> str | None:
    document = _mapping(artifact.get("document"))
    value = _first_string(document, artifact, "source_sha256", "source_hash", "file_hash", "checksum")
    if value and value.startswith("sha256:"):
        return value.removeprefix("sha256:")
    return value


def _structured_output_path(manifest: Mapping[str, Any]) -> str | None:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        return None
    return _string_or_none(outputs.get("structured_document"))


def _path_points_to_artifact(entry_path: str, artifact_path: Path, manifest_path: Path) -> bool:
    entry = Path(entry_path)
    candidates = {
        str(artifact_path),
        artifact_path.name,
    }
    normalized_entry = entry_path.replace("\\", "/").lstrip("./")
    normalized_artifact = str(artifact_path).replace("\\", "/")
    if normalized_artifact.endswith(normalized_entry):
        return True
    try:
        candidates.add(str(artifact_path.resolve()))
        normalized_resolved_artifact = str(artifact_path.resolve()).replace("\\", "/")
        if normalized_resolved_artifact.endswith(normalized_entry):
            return True
    except OSError:
        pass
    if entry.is_absolute():
        candidates.add(str(entry))
        try:
            return entry.resolve() == artifact_path.resolve()
        except OSError:
            return entry_path in candidates
    manifest_relative = (manifest_path.parent / entry).resolve()
    try:
        return entry_path in candidates or manifest_relative == artifact_path.resolve()
    except OSError:
        return entry_path in candidates


def _page_range(value: Mapping[str, Any]) -> tuple[int | None, int | None, int | None, int | None]:
    source_span = _mapping(value.get("source_span"))
    page_start = _int_or_none(value.get("page_start"))
    page_end = _int_or_none(value.get("page_end"))
    pdf_start = _int_or_none(value.get("pdf_page_index_start"))
    pdf_end = _int_or_none(value.get("pdf_page_index_end"))
    if page_start is None:
        page_start = _int_or_none(source_span.get("page_start"))
    if page_end is None:
        page_end = _int_or_none(source_span.get("page_end"))
    if pdf_start is None:
        pdf_start = _int_or_none(source_span.get("pdf_page_index_start"))
    if pdf_end is None:
        pdf_end = _int_or_none(source_span.get("pdf_page_index_end"))
    return page_start, page_end, pdf_start, pdf_end


def _printed_page_labels(
    page_by_number: Mapping[int, Mapping[str, Any]],
    page_start: int | None,
    page_end: int | None,
) -> tuple[str, ...]:
    if page_start is None or page_end is None:
        return ()
    labels: list[str] = []
    for page_number in range(page_start, page_end + 1):
        label = _string_or_none(page_by_number.get(page_number, {}).get("printed_page_label"))
        if label is not None:
            labels.append(label)
    return tuple(labels)


def _section_path(section: Mapping[str, Any]) -> tuple[str, ...]:
    path = section.get("path")
    if isinstance(path, Sequence) and not isinstance(path, (str, bytes, bytearray)):
        return tuple(str(item) for item in path)
    section_number = _string_or_none(section.get("section_number"))
    title = _string_or_none(section.get("title"))
    if section_number and title:
        return (f"{section_number} {title}",)
    if title:
        return (title,)
    return ()


def _provenance_status(
    source_checksum_verified: bool,
    page_start: int | None,
    page_end: int | None,
    source_block_ids: Sequence[str],
) -> str:
    if source_checksum_verified and page_start is not None and page_end is not None and source_block_ids:
        return "structured"
    return "structured_partial"


def _candidate_id(document_id: str, entity_id: str) -> str:
    return f"{document_id}:chunk:{quote(entity_id, safe='._-')}"


def _block_sort_key(block: Mapping[str, Any]) -> tuple[int, str]:
    index = _int_or_none(block.get("document_block_index"))
    return (index if index is not None else 999999, str(block.get("block_id") or ""))


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(value)
    return ()


def _mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(str(item) for item in value if _non_empty_string(item))
    return ()


def _string_or_none(value: Any) -> str | None:
    if _non_empty_string(value):
        return str(value)
    return None


def _first_string(
    first: Mapping[str, Any],
    second: Mapping[str, Any] | str | None = None,
    *keys: str,
) -> str | None:
    mappings: tuple[Mapping[str, Any], ...]
    key_names: tuple[str, ...]
    if isinstance(second, Mapping):
        mappings = (first, second)
        key_names = keys
    else:
        mappings = (first,)
        key_names = tuple(item for item in (second, *keys) if isinstance(item, str))
    for key in key_names:
        for mapping in mappings:
            value = _string_or_none(mapping.get(key))
            if value is not None:
                return value
    return None


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _child_path(output_dir: Path, filename: str) -> Path:
    output_root = output_dir.resolve()
    child = (output_dir / filename).resolve()
    if child.parent != output_root:
        raise ValueError(f"Refusing to write outside output_dir: {child}")
    return child


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")


__all__ = [
    "ADAPTER_CANDIDATES_FILENAME",
    "ADAPTER_INTEGRITY_FILENAME",
    "ADAPTER_REPORT_FILENAME",
    "DEFAULT_ADAPTER_OUTPUT_DIR",
    "FAIL",
    "PASS",
    "REVIEW",
    "StructuredDocumentAdapterIssue",
    "StructuredDocumentAdapterResult",
    "StructuredDocumentAdapterWriteResult",
    "StructuredDocumentArtifactIntegrity",
    "StructuredDocumentChunkCandidate",
    "build_structured_document_chunk_candidates",
    "load_structured_document_artifact",
    "load_techdoc_parser_manifest",
    "run_structured_document_adapter",
    "structured_document_adapter_issue_to_dict",
    "structured_document_adapter_result_to_dict",
    "structured_document_adapter_result_to_json",
    "structured_document_artifact_integrity_to_dict",
    "structured_document_chunk_candidate_to_dict",
    "validate_structured_document_artifact_integrity",
    "write_structured_document_adapter_outputs",
]
