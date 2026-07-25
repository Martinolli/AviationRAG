"""D.5c controlled real parser-output sample persistence gate.

This module orchestrates existing D.4c adapter and D.5b package APIs. It does
not invoke techdoc-parser, write files implicitly, access runtime ingestion,
generate embeddings, or touch vector/database systems.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Collection, Mapping

from aviationrag.ingestion.persisted_chunk_mapper import (
    PersistedChunkCandidateContext,
    PersistedChunkMappingPolicy,
)
from aviationrag.ingestion.persisted_chunk_package import build_package_from_adapter_result
from aviationrag.ingestion.persisted_chunk_record import (
    PERSISTED_CHUNK_MAPPER_VERSION,
    PERSISTED_CHUNK_SCHEMA_VERSION,
)
from aviationrag.ingestion.persisted_chunk_validator import (
    PERSISTED_CHUNK_LIMITATION_REGISTRY_VERSION,
    APPROVED_LIMITATION_CODES,
    limitation_warning_codes,
)
from aviationrag.ingestion.structured_document_adapter import (
    FAIL,
    PASS,
    REVIEW,
    SUPPORTED_SCHEMA_NAME,
    SUPPORTED_SCHEMA_VERSION,
    run_structured_document_adapter,
)


REAL_PARSER_SAMPLE_GATE_SCHEMA_NAME = "aviationrag-real-parser-sample-gate"
REAL_PARSER_SAMPLE_GATE_SCHEMA_VERSION = "0.1.0"

AUTHORIZATION = {
    "controlled_real_sample_package": True,
    "additional_small_controlled_profiles": False,
    "full_corpus_ingestion": False,
    "runtime_ingestion": False,
    "embedding_generation": False,
    "astra_rebuild": False,
    "faiss_rebuild": False,
    "production_retrieval_integration": False,
    "production_migration": False,
}


@dataclass(frozen=True)
class RealParserSampleGateResult:
    """Sanitized D.5c gate result."""

    outcome: str
    document_key: str
    source_filename: str
    source_checksum: str
    structured_document_schema_version: str
    parser_name: str
    parser_version: str
    adapter_outcome: str
    package_outcome: str
    input_candidate_count: int
    accepted_record_count: int
    rejected_candidate_count: int
    warning_count: int
    review_required_count: int
    validation_status_counts: Mapping[str, int]
    provenance_counts: Mapping[str, int]
    content_type_counts: Mapping[str, int]
    accepted_limitation_counts: Mapping[str, int]
    package_digest: str | None
    determinism_verified: bool
    blocking_issue_codes: tuple[str, ...]
    authorization: Mapping[str, bool]
    summary: Mapping[str, Any] = field(default_factory=dict)


def run_real_parser_sample_gate(
    *,
    artifact_path: str | Path,
    manifest_path: str | Path,
    source_path: str | Path,
    candidate_contexts: Mapping[str, PersistedChunkCandidateContext] | None = None,
    approved_adapter_warning_codes: Collection[str] = (),
    mapping_policy: PersistedChunkMappingPolicy | None = None,
    allow_review: bool = False,
) -> RealParserSampleGateResult:
    """Run the D.5c in-memory gate without writing files."""
    artifact_file = _required_file(artifact_path, "artifact")
    manifest_file = _required_file(manifest_path, "manifest")
    source_file = _required_file(source_path, "source")
    artifact = _load_json_object(artifact_file, "artifact")
    manifest = _load_json_object(manifest_file, "manifest")
    source_checksum = _sha256_file(source_file)
    artifact_checksum = _sha256_file(artifact_file)
    manifest_checksum = _sha256_file(manifest_file)

    preflight_codes = _preflight_issue_codes(
        artifact=artifact,
        manifest=manifest,
        source_checksum=source_checksum,
        artifact_checksum=artifact_checksum,
    )
    adapter_result = run_structured_document_adapter(
        artifact_file,
        manifest_file,
        source_path=source_file,
        approved_warning_codes=approved_adapter_warning_codes,
        strict_warnings=False,
    )
    policy = mapping_policy or PersistedChunkMappingPolicy(
        allow_partial_provenance=False,
        allow_review_required_records=allow_review,
        include_heading_records=False,
    )
    package = build_package_from_adapter_result(
        adapter_result,
        candidate_contexts=candidate_contexts,
        policy=policy,
        allow_rejected_candidates=False,
    )
    repeated_package = build_package_from_adapter_result(
        adapter_result,
        candidate_contexts=candidate_contexts,
        policy=policy,
        allow_rejected_candidates=False,
    )
    determinism_verified = (
        package.package_digest == repeated_package.package_digest
        and dict(package.file_sha256) == dict(repeated_package.file_sha256)
    )
    blocking_codes = _blocking_issue_codes(
        preflight_codes=preflight_codes,
        adapter_result=adapter_result,
        package=package,
        approved_adapter_warning_codes=set(approved_adapter_warning_codes),
        determinism_verified=determinism_verified,
        allow_review=allow_review,
    )
    outcome = _gate_outcome(blocking_codes, adapter_result.outcome, package.outcome, allow_review)
    document = artifact.get("document") if isinstance(artifact.get("document"), Mapping) else {}
    return RealParserSampleGateResult(
        outcome=outcome,
        document_key=_string(document.get("document_id") or artifact.get("document_id")),
        source_filename=Path(_string(document.get("source_filename"))).name,
        source_checksum=source_checksum,
        structured_document_schema_version=_string(artifact.get("schema_version")),
        parser_name=_string(artifact.get("parser_name")),
        parser_version=_string(artifact.get("parser_version")),
        adapter_outcome=adapter_result.outcome,
        package_outcome=package.outcome,
        input_candidate_count=len(adapter_result.candidates),
        accepted_record_count=len(package.records),
        rejected_candidate_count=len(package.rejected_candidates),
        warning_count=len(package.warnings) + _adapter_warning_count(adapter_result),
        review_required_count=int(package.report.get("review_required_count", 0)),
        validation_status_counts=dict(package.report.get("validation_status_counts", {})),
        provenance_counts=dict(package.report.get("provenance_counts", {})),
        content_type_counts=dict(package.report.get("content_type_counts", {})),
        accepted_limitation_counts=dict(package.report.get("limitation_counts", {})),
        package_digest=package.package_digest or None,
        determinism_verified=determinism_verified,
        blocking_issue_codes=tuple(sorted(set(blocking_codes))),
        authorization=dict(AUTHORIZATION),
        summary={
            "artifact_checksum": artifact_checksum,
            "manifest_checksum": manifest_checksum,
            "artifact_checksum_matches_manifest": adapter_result.artifact_integrity.artifact_checksum_matches,
            "source_checksum_matches_manifest": adapter_result.artifact_integrity.source_checksum_matches is True,
            "manifest_matches_artifact": adapter_result.artifact_integrity.manifest_matches_artifact,
            "page_count": len(artifact.get("pages") or []),
            "block_count": len(artifact.get("blocks") or []),
            "adapter_warning_codes": _adapter_warning_codes(adapter_result),
            "package_warning_codes": tuple(issue.code for issue in package.warnings),
            "package_issue_codes": tuple(issue.code for issue in package.issues),
            "approved_adapter_warning_codes": tuple(sorted(set(approved_adapter_warning_codes))),
            "approved_limitation_registry_codes": tuple(sorted(APPROVED_LIMITATION_CODES)),
            "persisted_mapper_version": PERSISTED_CHUNK_MAPPER_VERSION,
            "persisted_schema_version": PERSISTED_CHUNK_SCHEMA_VERSION,
            "limitation_registry_version": PERSISTED_CHUNK_LIMITATION_REGISTRY_VERSION,
            "runtime_ingestion_modified": False,
            "real_corpus_processed": False,
            "embedding_generation": False,
            "astra_touched": False,
            "faiss_touched": False,
            "production_migration_authorized": False,
        },
    )


def real_parser_sample_gate_result_to_dict(result: RealParserSampleGateResult) -> dict[str, Any]:
    """Return a deterministic sanitized gate-result dictionary."""
    return {
        "accepted_limitation_counts": dict(sorted(result.accepted_limitation_counts.items())),
        "accepted_record_count": result.accepted_record_count,
        "adapter_outcome": result.adapter_outcome,
        "authorization": dict(sorted(result.authorization.items())),
        "blocking_issue_codes": list(result.blocking_issue_codes),
        "content_type_counts": dict(sorted(result.content_type_counts.items())),
        "determinism_verified": result.determinism_verified,
        "document_key": result.document_key,
        "gate_schema_name": REAL_PARSER_SAMPLE_GATE_SCHEMA_NAME,
        "gate_schema_version": REAL_PARSER_SAMPLE_GATE_SCHEMA_VERSION,
        "input_candidate_count": result.input_candidate_count,
        "outcome": result.outcome,
        "package_digest": result.package_digest,
        "package_outcome": result.package_outcome,
        "parser_name": result.parser_name,
        "parser_version": result.parser_version,
        "provenance_counts": dict(sorted(result.provenance_counts.items())),
        "rejected_candidate_count": result.rejected_candidate_count,
        "review_required_count": result.review_required_count,
        "source_checksum": result.source_checksum,
        "source_filename": result.source_filename,
        "structured_document_schema_version": result.structured_document_schema_version,
        "summary": _plain(result.summary),
        "validation_status_counts": dict(sorted(result.validation_status_counts.items())),
        "warning_count": result.warning_count,
    }


def sanitized_gate_report_bytes(result: RealParserSampleGateResult) -> bytes:
    """Serialize a gate result with deterministic JSON and final newline."""
    return (
        json.dumps(
            real_parser_sample_gate_result_to_dict(result),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _preflight_issue_codes(
    *,
    artifact: Mapping[str, Any],
    manifest: Mapping[str, Any],
    source_checksum: str,
    artifact_checksum: str,
) -> list[str]:
    codes: list[str] = []
    if artifact.get("schema_name") != SUPPORTED_SCHEMA_NAME:
        codes.append("STRUCTURED_DOCUMENT_SCHEMA_UNSUPPORTED")
    if artifact.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        codes.append("STRUCTURED_DOCUMENT_SCHEMA_UNSUPPORTED")
    document = artifact.get("document") if isinstance(artifact.get("document"), Mapping) else {}
    if document.get("source_hash") != source_checksum:
        codes.append("SOURCE_ARTIFACT_CHECKSUM_MISMATCH")
    entry = _structured_artifact_entry(manifest)
    if entry is None:
        codes.append("MANIFEST_STRUCTURED_ARTIFACT_MISSING")
        return codes
    if entry.get("source_sha256") != source_checksum:
        codes.append("SOURCE_MANIFEST_CHECKSUM_MISMATCH")
    if entry.get("artifact_sha256") != artifact_checksum:
        codes.append("ARTIFACT_MANIFEST_CHECKSUM_MISMATCH")
    if entry.get("document_id") != document.get("document_id"):
        codes.append("DOCUMENT_IDENTITY_MISMATCH")
    if entry.get("schema_name") != artifact.get("schema_name"):
        codes.append("ARTIFACT_SCHEMA_MISMATCH")
    if entry.get("schema_version") != artifact.get("schema_version"):
        codes.append("ARTIFACT_SCHEMA_MISMATCH")
    return codes


def _blocking_issue_codes(
    *,
    preflight_codes: list[str],
    adapter_result: Any,
    package: Any,
    approved_adapter_warning_codes: set[str],
    determinism_verified: bool,
    allow_review: bool,
) -> list[str]:
    codes = list(preflight_codes)
    adapter_warning_codes = set(_adapter_warning_codes(adapter_result))
    unknown_adapter_warnings = sorted(adapter_warning_codes - approved_adapter_warning_codes)
    codes.extend(f"UNAPPROVED_ADAPTER_WARNING:{code}" for code in unknown_adapter_warnings)
    if adapter_result.outcome == FAIL:
        codes.append("ADAPTER_OUTCOME_FAIL")
    elif adapter_result.outcome == REVIEW and not allow_review:
        codes.append("ADAPTER_REVIEW_NOT_ALLOWED")
    if package.outcome == FAIL:
        codes.append("PACKAGE_OUTCOME_FAIL")
    elif package.outcome == REVIEW and not allow_review:
        codes.append("PACKAGE_REVIEW_NOT_ALLOWED")
    if not adapter_result.candidates:
        codes.append("NO_ADAPTER_CANDIDATES")
    if not package.records:
        codes.append("NO_ACCEPTED_RECORDS")
    if package.rejected_candidates:
        codes.append("REJECTED_CANDIDATES_PRESENT")
    if not determinism_verified:
        codes.append("PACKAGE_NONDETERMINISTIC")
    provenance_counts = Counter(record.provenance_status for record in package.records)
    if package.records and set(provenance_counts) != {"full_provenance"}:
        codes.append("NON_FULL_PROVENANCE_PRESENT")
    for issue in package.issues:
        codes.append(issue.code)
        if issue.code == "LIMITATION_CODE_UNKNOWN":
            codes.append("UNKNOWN_LIMITATION_CODE")
    known_package_warning_codes = _known_package_warning_codes()
    for warning in package.warnings:
        if warning.code not in known_package_warning_codes:
            codes.append(f"UNKNOWN_PACKAGE_WARNING:{warning.code}")
    return codes


def _gate_outcome(
    blocking_codes: list[str],
    adapter_outcome: str,
    package_outcome: str,
    allow_review: bool,
) -> str:
    if blocking_codes:
        return FAIL
    if allow_review and (adapter_outcome == REVIEW or package_outcome == REVIEW):
        return REVIEW
    return PASS


def _required_file(value: str | Path, label: str) -> Path:
    path = Path(value)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} path does not exist or is not a file: {path}")
    return path


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return data


def _structured_artifact_entry(manifest: Mapping[str, Any]) -> Mapping[str, Any] | None:
    entries = [
        item
        for item in manifest.get("artifacts", [])
        if isinstance(item, Mapping) and item.get("artifact_type") == "structured_document"
    ]
    if len(entries) != 1:
        return None
    return entries[0]


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _adapter_warning_codes(adapter_result: Any) -> tuple[str, ...]:
    return tuple(sorted(issue.code for issue in adapter_result.issues if issue.severity == "warning"))


def _adapter_warning_count(adapter_result: Any) -> int:
    return sum(1 for issue in adapter_result.issues if issue.severity == "warning")


def _known_package_warning_codes() -> set[str]:
    codes: set[str] = set()
    for limitation_code in APPROVED_LIMITATION_CODES:
        codes.update(limitation_warning_codes((limitation_code,)))
    return codes


def _string(value: object) -> str:
    return str(value) if value is not None else ""


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "AUTHORIZATION",
    "REAL_PARSER_SAMPLE_GATE_SCHEMA_NAME",
    "REAL_PARSER_SAMPLE_GATE_SCHEMA_VERSION",
    "RealParserSampleGateResult",
    "real_parser_sample_gate_result_to_dict",
    "run_real_parser_sample_gate",
    "sanitized_gate_report_bytes",
]
