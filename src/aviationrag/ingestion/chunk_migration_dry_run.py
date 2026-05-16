"""Fake/local chunk migration dry-run helpers.

This module composes the read-only chunk audit, passive legacy adapter, chunk
schema validator, and vector payload validator into a side-effect-free dry run.
It does not write files, generate embeddings, call external services, connect
to Astra, use FAISS, or integrate with runtime ingestion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from aviationrag.ingestion.chunk_audit import (
    audit_chunk_records,
    chunk_audit_summary_to_dict,
    load_chunk_like_records,
)
from aviationrag.ingestion.chunk_legacy_adapter import preview_legacy_chunk_migration
from aviationrag.ingestion.chunk_payload import validate_vector_payload_dataset
from aviationrag.ingestion.chunk_schema import (
    chunk_record_to_dict,
    validate_chunk_dataset,
)


@dataclass
class ChunkMigrationDryRunResult:
    """Summary-only result for a fake/local chunk migration rehearsal."""

    source_path: str
    audit: dict[str, Any]
    chunk_count: int
    payload_count: int
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def run_chunk_migration_dry_run(
    input_path: str | Path,
    max_records: int | None = None,
) -> ChunkMigrationDryRunResult:
    """Run a side-effect-free chunk migration dry run for one explicit file."""
    path = Path(input_path)
    records = load_chunk_like_records(path, max_records=max_records)

    audit_summary = audit_chunk_records(records, source_path=str(path))
    preview = preview_legacy_chunk_migration(records, include_payloads=True, env={})

    chunk_dicts = [chunk_record_to_dict(chunk) for chunk in preview.chunks]
    chunk_issues = validate_chunk_dataset(chunk_dicts)
    payload_issues = validate_vector_payload_dataset(preview.payloads)

    issues = _dedupe(
        [f"audit: {issue}" for issue in audit_summary.issues]
        + [f"conversion: {issue}" for issue in preview.issues]
        + [f"chunk_validation: {issue}" for issue in chunk_issues]
        + [f"payload_validation: {issue}" for issue in payload_issues]
    )
    warnings = _dedupe(
        [f"audit: {warning}" for warning in audit_summary.warnings]
        + [f"conversion: {warning}" for warning in preview.warnings]
    )

    result = ChunkMigrationDryRunResult(
        source_path=str(path),
        audit=chunk_audit_summary_to_dict(audit_summary),
        chunk_count=len(preview.chunks),
        payload_count=len(preview.payloads),
        issues=issues,
        warnings=warnings,
    )
    result.summary = summarize_chunk_migration_dry_run(result)
    return result


def chunk_migration_dry_run_result_to_dict(
    result: ChunkMigrationDryRunResult,
) -> dict[str, Any]:
    """Return a JSON-serializable dry-run result dictionary."""
    return asdict(result)


def summarize_chunk_migration_dry_run(
    result: ChunkMigrationDryRunResult,
) -> dict[str, Any]:
    """Return compact dry-run counts suitable for CLI output and logs."""
    audit = result.audit
    return {
        "source_path": result.source_path,
        "detected_format": audit.get("detected_format"),
        "input_record_count": audit.get("record_count", 0),
        "chunk_count": result.chunk_count,
        "payload_count": result.payload_count,
        "issue_count": len(result.issues),
        "warning_count": len(result.warnings),
        "missing_text_count": audit.get("missing_text_count", 0),
        "missing_chunk_id_count": audit.get("missing_chunk_id_count", 0),
        "missing_document_id_count": audit.get("missing_document_id_count", 0),
        "chunk_type_counts": dict(audit.get("chunk_type_counts") or {}),
    }


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


__all__ = [
    "ChunkMigrationDryRunResult",
    "chunk_migration_dry_run_result_to_dict",
    "run_chunk_migration_dry_run",
    "summarize_chunk_migration_dry_run",
]
