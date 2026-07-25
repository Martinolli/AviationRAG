"""Deterministic local persisted-chunk package builder for D.5b."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from aviationrag.ingestion.persisted_chunk_mapper import (
    PersistedChunkCandidateContext,
    PersistedChunkMappingPolicy,
    map_candidate_to_persisted_chunk,
)
from aviationrag.ingestion.persisted_chunk_record import (
    PERSISTED_CHUNK_MAPPING_SPECIFICATION_NAME,
    PERSISTED_CHUNK_MAPPING_SPECIFICATION_VERSION,
    PERSISTED_CHUNK_SCHEMA_NAME,
    PERSISTED_CHUNK_SCHEMA_VERSION,
    PersistedChunkRecord,
    persisted_chunk_record_to_dict,
)
from aviationrag.ingestion.persisted_chunk_validator import (
    PERSISTED_CHUNK_LIMITATION_REGISTRY_VERSION,
    PersistedChunkIssue,
    persisted_chunk_issue_to_dict,
    sort_persisted_chunk_issues,
    validate_persisted_chunk_records,
)
from aviationrag.ingestion.structured_document_adapter import (
    FAIL,
    PASS,
    REVIEW,
    StructuredDocumentAdapterResult,
    StructuredDocumentChunkCandidate,
    structured_document_adapter_result_to_dict,
)


DEFAULT_PERSISTED_PACKAGE_OUTPUT_DIR = Path("data/migration_dry_run/persisted_chunk_package")
PERSISTED_CHUNKS_FILENAME = "persisted_chunks.jsonl"
PERSISTENCE_MANIFEST_FILENAME = "persistence_manifest.json"
PERSISTENCE_REPORT_FILENAME = "persistence_report.json"
REJECTED_CANDIDATES_FILENAME = "rejected_candidates.jsonl"
WARNINGS_FILENAME = "warnings.json"
PACKAGE_SCHEMA_NAME = "aviationrag-persisted-chunk-package"
PACKAGE_SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True)
class RejectedPersistedChunkCandidate:
    """Sanitized rejected candidate evidence."""

    candidate_id: str
    reason_codes: tuple[str, ...]
    issues: tuple[PersistedChunkIssue, ...]


@dataclass(frozen=True)
class PersistedChunkPackage:
    """In-memory deterministic package."""

    outcome: str
    records: tuple[PersistedChunkRecord, ...]
    rejected_candidates: tuple[RejectedPersistedChunkCandidate, ...]
    warnings: tuple[PersistedChunkIssue, ...]
    issues: tuple[PersistedChunkIssue, ...]
    manifest: Mapping[str, Any]
    report: Mapping[str, Any]
    package_digest: str
    file_sha256: Mapping[str, str]


@dataclass(frozen=True)
class PersistedChunkPackageWriteResult:
    """Result for explicitly permitted local package writes."""

    output_dir: str
    persisted_chunks_output_path: str
    persistence_manifest_output_path: str
    persistence_report_output_path: str
    rejected_candidates_output_path: str
    warnings_output_path: str
    record_count: int
    rejected_count: int
    warning_count: int
    package_digest: str


def build_persisted_chunk_package(
    candidates: Sequence[StructuredDocumentChunkCandidate],
    *,
    candidate_contexts: Mapping[str, PersistedChunkCandidateContext] | None = None,
    policy: PersistedChunkMappingPolicy | None = None,
    source_structured_document_checksum: str | None = None,
    source_manifest_checksum: str | None = None,
    allow_rejected_candidates: bool = False,
    adapter_result: StructuredDocumentAdapterResult | None = None,
) -> PersistedChunkPackage:
    """Build an in-memory package from D.4c candidates without writing files."""
    contexts = dict(candidate_contexts or {})
    active_policy = policy or PersistedChunkMappingPolicy()
    issues: list[PersistedChunkIssue] = []
    warnings: list[PersistedChunkIssue] = []
    rejected: list[RejectedPersistedChunkCandidate] = []
    records: list[PersistedChunkRecord] = []

    candidate_ids = {candidate.chunk_candidate_id for candidate in candidates}
    for unknown_id in sorted(set(contexts) - candidate_ids):
        issue = PersistedChunkIssue(
            code="CONTEXT_CANDIDATE_UNKNOWN",
            severity="error",
            message="Candidate context references an unknown candidate ID.",
            path="$.candidate_contexts",
            candidate_id=unknown_id,
        )
        if not allow_rejected_candidates:
            issues.append(issue)
        rejected.append(
            RejectedPersistedChunkCandidate(
                candidate_id=unknown_id,
                reason_codes=(issue.code,),
                issues=(issue,),
            )
        )

    next_index = 0
    for candidate in candidates:
        result = map_candidate_to_persisted_chunk(
            candidate,
            chunk_index=next_index,
            context=contexts.get(candidate.chunk_candidate_id),
            policy=active_policy,
        )
        warnings.extend(result.warnings)
        if result.is_accepted and result.record is not None:
            records.append(result.record)
            next_index += 1
            continue
        if not allow_rejected_candidates:
            issues.extend(result.issues)
        rejected.append(
            RejectedPersistedChunkCandidate(
                candidate_id=result.candidate_id,
                reason_codes=tuple(issue.code for issue in result.issues),
                issues=result.issues,
            )
        )

    collision_issues = _collision_issues(records)
    issues.extend(collision_issues)
    validation_issues = validate_persisted_chunk_records(records)
    issues.extend(issue for issue in validation_issues if issue.severity == "error")
    warnings.extend(issue for issue in validation_issues if issue.severity == "warning")

    if collision_issues:
        collided_ids = {issue.chunk_id for issue in collision_issues if issue.chunk_id}
        records = [record for record in records if record.chunk_id not in collided_ids]

    if rejected and not allow_rejected_candidates:
        issues.append(
            PersistedChunkIssue(
                code="REJECTED_CANDIDATES_PRESENT",
                severity="error",
                message="Rejected candidates are present and policy does not allow local package continuation.",
                path="$.rejected_candidates",
            )
        )

    issues = sort_persisted_chunk_issues(issues)
    warnings = sort_persisted_chunk_issues(warnings)
    outcome = _package_outcome(issues, warnings, records)
    if outcome == PASS and rejected:
        outcome = REVIEW
    if adapter_result is not None and adapter_result.outcome == FAIL:
        outcome = FAIL
        issues.append(
            PersistedChunkIssue(
                code="ADAPTER_OUTCOME_FAIL",
                severity="error",
                message="Adapter FAIL blocks persisted package acceptance.",
                path="$.adapter_result.outcome",
            )
        )
    elif adapter_result is not None and adapter_result.outcome == REVIEW and outcome == PASS:
        outcome = REVIEW

    record_dicts = [persisted_chunk_record_to_dict(record) for record in records]
    rejected_dicts = [rejected_candidate_to_dict(item) for item in rejected]
    warnings_dict = {
        "schema_name": "aviationrag-persisted-chunk-warnings",
        "schema_version": "0.1.0",
        "warning_count": len(warnings),
        "warnings": [persisted_chunk_issue_to_dict(issue) for issue in warnings],
    }
    report = _build_report(
        outcome=outcome,
        records=records,
        rejected=rejected,
        warnings=warnings,
        issues=issues,
        adapter_result=adapter_result,
    )
    manifest_without_checksum = _build_manifest(
        outcome=outcome,
        records=records,
        rejected=rejected,
        warnings=warnings,
        issues=issues,
        source_structured_document_checksum=source_structured_document_checksum,
        source_manifest_checksum=source_manifest_checksum,
        package_digest="",
        file_sha256={},
    )
    file_bytes = _package_file_bytes(record_dicts, manifest_without_checksum, report, rejected_dicts, warnings_dict)
    file_sha256 = {
        name: sha256(content).hexdigest()
        for name, content in file_bytes.items()
        if name != PERSISTENCE_MANIFEST_FILENAME
    }
    package_digest = _package_digest(file_sha256)
    manifest = _build_manifest(
        outcome=outcome,
        records=records,
        rejected=rejected,
        warnings=warnings,
        issues=issues,
        source_structured_document_checksum=source_structured_document_checksum,
        source_manifest_checksum=source_manifest_checksum,
        package_digest=package_digest,
        file_sha256=file_sha256,
    )
    final_bytes = _package_file_bytes(record_dicts, manifest, report, rejected_dicts, warnings_dict)
    final_file_sha256 = {
        name: sha256(content).hexdigest()
        for name, content in final_bytes.items()
        if name != PERSISTENCE_MANIFEST_FILENAME
    }
    final_package_digest = _package_digest(final_file_sha256)
    manifest = _build_manifest(
        outcome=outcome,
        records=records,
        rejected=rejected,
        warnings=warnings,
        issues=issues,
        source_structured_document_checksum=source_structured_document_checksum,
        source_manifest_checksum=source_manifest_checksum,
        package_digest=final_package_digest,
        file_sha256=final_file_sha256,
    )

    return PersistedChunkPackage(
        outcome=outcome,
        records=tuple(records),
        rejected_candidates=tuple(rejected),
        warnings=tuple(warnings),
        issues=tuple(issues),
        manifest=manifest,
        report=report,
        package_digest=final_package_digest,
        file_sha256=manifest["file_sha256"],
    )


def build_package_from_adapter_result(
    adapter_result: StructuredDocumentAdapterResult,
    *,
    candidate_contexts: Mapping[str, PersistedChunkCandidateContext] | None = None,
    policy: PersistedChunkMappingPolicy | None = None,
    allow_rejected_candidates: bool = False,
) -> PersistedChunkPackage:
    """Build a persisted package from a completed D.4c adapter result."""
    source_checksum = adapter_result.artifact_integrity.artifact_sha256_actual
    manifest_checksum = _sha256_file_if_path(adapter_result.artifact_integrity.manifest_path)
    if adapter_result.outcome == FAIL:
        issue = PersistedChunkIssue(
            code="ADAPTER_OUTCOME_FAIL",
            severity="error",
            message="Adapter FAIL blocks persisted package construction.",
            path="$.adapter_result.outcome",
        )
        manifest = _build_manifest(
            outcome=FAIL,
            records=[],
            rejected=[],
            warnings=[],
            issues=[issue],
            source_structured_document_checksum=source_checksum,
            source_manifest_checksum=manifest_checksum,
            package_digest="",
            file_sha256={},
        )
        report = {"outcome": FAIL, "issue_count": 1, "issues": [persisted_chunk_issue_to_dict(issue)]}
        return PersistedChunkPackage(
            outcome=FAIL,
            records=(),
            rejected_candidates=(),
            warnings=(),
            issues=(issue,),
            manifest=manifest,
            report=report,
            package_digest="",
            file_sha256={},
        )
    return build_persisted_chunk_package(
        adapter_result.candidates,
        candidate_contexts=candidate_contexts,
        policy=policy,
        source_structured_document_checksum=source_checksum,
        source_manifest_checksum=manifest_checksum,
        allow_rejected_candidates=allow_rejected_candidates,
        adapter_result=adapter_result,
    )


def write_persisted_chunk_package(
    package: PersistedChunkPackage,
    output_dir: str | Path,
    *,
    allow_local_write: bool = False,
    overwrite: bool = False,
) -> PersistedChunkPackageWriteResult:
    """Write package artifacts only when explicitly allowed."""
    if not allow_local_write:
        raise PermissionError("Persisted chunk package writes require allow_local_write=True.")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        PERSISTED_CHUNKS_FILENAME: _child_path(root, PERSISTED_CHUNKS_FILENAME),
        PERSISTENCE_MANIFEST_FILENAME: _child_path(root, PERSISTENCE_MANIFEST_FILENAME),
        PERSISTENCE_REPORT_FILENAME: _child_path(root, PERSISTENCE_REPORT_FILENAME),
        REJECTED_CANDIDATES_FILENAME: _child_path(root, REJECTED_CANDIDATES_FILENAME),
        WARNINGS_FILENAME: _child_path(root, WARNINGS_FILENAME),
    }
    if not overwrite:
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError("Persisted package output exists; pass overwrite=True: " + ", ".join(existing))

    records = [persisted_chunk_record_to_dict(record) for record in package.records]
    rejected = [rejected_candidate_to_dict(item) for item in package.rejected_candidates]
    warnings = {
        "schema_name": "aviationrag-persisted-chunk-warnings",
        "schema_version": "0.1.0",
        "warning_count": len(package.warnings),
        "warnings": [persisted_chunk_issue_to_dict(issue) for issue in package.warnings],
    }
    file_bytes = _package_file_bytes(records, package.manifest, package.report, rejected, warnings)
    for name, content in file_bytes.items():
        _write_bytes_atomic(paths[name], content)

    return PersistedChunkPackageWriteResult(
        output_dir=str(root),
        persisted_chunks_output_path=str(paths[PERSISTED_CHUNKS_FILENAME]),
        persistence_manifest_output_path=str(paths[PERSISTENCE_MANIFEST_FILENAME]),
        persistence_report_output_path=str(paths[PERSISTENCE_REPORT_FILENAME]),
        rejected_candidates_output_path=str(paths[REJECTED_CANDIDATES_FILENAME]),
        warnings_output_path=str(paths[WARNINGS_FILENAME]),
        record_count=len(package.records),
        rejected_count=len(package.rejected_candidates),
        warning_count=len(package.warnings),
        package_digest=package.package_digest,
    )


def persisted_chunk_package_to_dict(package: PersistedChunkPackage) -> dict[str, Any]:
    """Return a JSON-serializable package summary."""
    return {
        "file_sha256": dict(package.file_sha256),
        "issue_count": len(package.issues),
        "issues": [persisted_chunk_issue_to_dict(issue) for issue in package.issues],
        "manifest": dict(package.manifest),
        "outcome": package.outcome,
        "package_digest": package.package_digest,
        "record_count": len(package.records),
        "records": [persisted_chunk_record_to_dict(record) for record in package.records],
        "rejected_candidates": [rejected_candidate_to_dict(item) for item in package.rejected_candidates],
        "rejected_count": len(package.rejected_candidates),
        "report": dict(package.report),
        "warning_count": len(package.warnings),
        "warnings": [persisted_chunk_issue_to_dict(issue) for issue in package.warnings],
    }


def persisted_chunk_package_write_result_to_dict(
    result: PersistedChunkPackageWriteResult,
) -> dict[str, Any]:
    """Return a JSON-serializable write result."""
    return asdict(result)


def rejected_candidate_to_dict(item: RejectedPersistedChunkCandidate) -> dict[str, Any]:
    """Return sanitized rejected candidate evidence."""
    return {
        "candidate_id": item.candidate_id,
        "issues": [persisted_chunk_issue_to_dict(issue) for issue in item.issues],
        "reason_codes": list(item.reason_codes),
    }


def _build_manifest(
    *,
    outcome: str,
    records: Sequence[PersistedChunkRecord],
    rejected: Sequence[RejectedPersistedChunkCandidate],
    warnings: Sequence[PersistedChunkIssue],
    issues: Sequence[PersistedChunkIssue],
    source_structured_document_checksum: str | None,
    source_manifest_checksum: str | None,
    package_digest: str,
    file_sha256: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "accepted_count": len(records),
        "file_sha256": dict(sorted(file_sha256.items())),
        "issue_count": len(issues),
        "limitation_registry_version": PERSISTED_CHUNK_LIMITATION_REGISTRY_VERSION,
        "mapper_version": "0.1.0",
        "outcome": outcome,
        "package_checksum": package_digest,
        "package_schema_name": PACKAGE_SCHEMA_NAME,
        "package_schema_version": PACKAGE_SCHEMA_VERSION,
        "persisted_schema_name": PERSISTED_CHUNK_SCHEMA_NAME,
        "persisted_schema_version": PERSISTED_CHUNK_SCHEMA_VERSION,
        "record_count": len(records),
        "rejected_count": len(rejected),
        "source_manifest_checksum": source_manifest_checksum,
        "source_structured_document_checksum": source_structured_document_checksum,
        "specification_name": PERSISTED_CHUNK_MAPPING_SPECIFICATION_NAME,
        "specification_version": PERSISTED_CHUNK_MAPPING_SPECIFICATION_VERSION,
        "warning_count": len(warnings),
    }


def _build_report(
    *,
    outcome: str,
    records: Sequence[PersistedChunkRecord],
    rejected: Sequence[RejectedPersistedChunkCandidate],
    warnings: Sequence[PersistedChunkIssue],
    issues: Sequence[PersistedChunkIssue],
    adapter_result: StructuredDocumentAdapterResult | None,
) -> dict[str, Any]:
    return {
        "accepted_count": len(records),
        "adapter_outcome": adapter_result.outcome if adapter_result is not None else None,
        "content_type_counts": dict(sorted(Counter(record.content_type for record in records).items())),
        "embedding_generation": False,
        "faiss_touched": False,
        "issue_count": len(issues),
        "issues": [persisted_chunk_issue_to_dict(issue) for issue in issues],
        "limitation_counts": dict(
            sorted(Counter(code for record in records for code in record.accepted_limitation_codes).items())
        ),
        "outcome": outcome,
        "provenance_counts": dict(sorted(Counter(record.provenance_status for record in records).items())),
        "real_corpus_processed": False,
        "record_count": len(records),
        "rejected_count": len(rejected),
        "review_required_count": sum(1 for record in records if record.review_required),
        "runtime_ingestion_modified": False,
        "storage": {"astra_touched": False, "faiss_touched": False},
        "validation_status_counts": dict(sorted(Counter(record.validation_status for record in records).items())),
        "warning_count": len(warnings),
        "warning_code_counts": dict(sorted(Counter(issue.code for issue in warnings).items())),
        "warnings": [persisted_chunk_issue_to_dict(issue) for issue in warnings],
    }


def _package_file_bytes(
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    rejected: Sequence[Mapping[str, Any]],
    warnings: Mapping[str, Any],
) -> dict[str, bytes]:
    return {
        PERSISTED_CHUNKS_FILENAME: _jsonl_bytes(records),
        PERSISTENCE_MANIFEST_FILENAME: _json_bytes(manifest),
        PERSISTENCE_REPORT_FILENAME: _json_bytes(report),
        REJECTED_CANDIDATES_FILENAME: _jsonl_bytes(rejected),
        WARNINGS_FILENAME: _json_bytes(warnings),
    }


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    lines = [json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) for record in records]
    return ("\n".join(lines) + ("\n" if lines else "")).encode("utf-8")


def _package_digest(file_sha256: Mapping[str, str]) -> str:
    payload = json.dumps(dict(sorted(file_sha256.items())), separators=(",", ":"), sort_keys=True).encode("utf-8")
    return sha256(payload).hexdigest()


def _collision_issues(records: Sequence[PersistedChunkRecord]) -> list[PersistedChunkIssue]:
    issues: list[PersistedChunkIssue] = []
    for chunk_id, count in sorted(Counter(record.chunk_id for record in records).items()):
        if count > 1:
            issues.append(
                PersistedChunkIssue(
                    code="PERSISTED_CHUNK_ID_COLLISION",
                    severity="error",
                    message=f"Persisted chunk ID collision appears {count} times.",
                    path="$.records[].chunk_id",
                    chunk_id=chunk_id,
                )
            )
    return issues


def _package_outcome(
    issues: Sequence[PersistedChunkIssue],
    warnings: Sequence[PersistedChunkIssue],
    records: Sequence[PersistedChunkRecord],
) -> str:
    if issues or not records:
        return FAIL
    if warnings or any(record.validation_status == "review_required" for record in records):
        return REVIEW
    return PASS


def _child_path(output_dir: Path, filename: str) -> Path:
    output_root = output_dir.resolve()
    child = (output_dir / filename).resolve()
    if child.parent != output_root:
        raise ValueError(f"Refusing to write outside output_dir: {child}")
    return child


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    temp_path = path.with_name(path.name + ".tmp")
    try:
        temp_path.write_bytes(content)
        temp_path.replace(path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _sha256_file_if_path(path: str | None) -> str | None:
    if not path:
        return None
    return sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "DEFAULT_PERSISTED_PACKAGE_OUTPUT_DIR",
    "PACKAGE_SCHEMA_NAME",
    "PACKAGE_SCHEMA_VERSION",
    "PERSISTED_CHUNKS_FILENAME",
    "PERSISTENCE_MANIFEST_FILENAME",
    "PERSISTENCE_REPORT_FILENAME",
    "REJECTED_CANDIDATES_FILENAME",
    "WARNINGS_FILENAME",
    "PersistedChunkPackage",
    "PersistedChunkPackageWriteResult",
    "RejectedPersistedChunkCandidate",
    "build_package_from_adapter_result",
    "build_persisted_chunk_package",
    "persisted_chunk_package_to_dict",
    "persisted_chunk_package_write_result_to_dict",
    "rejected_candidate_to_dict",
    "write_persisted_chunk_package",
]
