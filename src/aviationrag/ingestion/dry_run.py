"""Side-effect-free dry-run planning for future manifest-aware ingestion.

This module accepts fake legacy-like dictionaries and returns typed records plus
validation summaries. It does not read source documents, write manifests, call
legacy ingestion scripts, generate embeddings, access Astra, or build FAISS
indexes.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from aviationrag.ingestion.legacy_adapter import (
    legacy_chunk_to_record,
    legacy_document_to_record,
    normalize_legacy_filename,
)
from aviationrag.ingestion.manifest import validate_manifest_record
from aviationrag.models import ChunkRecord, DocumentRecord


@dataclass
class DryRunIngestionPlan:
    """Result of a fake-data manifest-aware ingestion dry run."""

    documents: list[DocumentRecord] = field(default_factory=list)
    chunks: list[ChunkRecord] = field(default_factory=list)
    manifest_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def build_dry_run_ingestion_plan(
    legacy_documents: Iterable[Mapping[str, Any]],
    legacy_chunks: Iterable[Mapping[str, Any]] | None = None,
) -> DryRunIngestionPlan:
    """Build a typed dry-run plan from fake legacy-like records."""
    documents = [legacy_document_to_record(item) for item in legacy_documents]
    document_by_filename = {
        normalize_legacy_filename(document.filename): document for document in documents
    }

    chunks: list[ChunkRecord] = []
    if legacy_chunks is not None:
        for item in legacy_chunks:
            filename = normalize_legacy_filename(_first_value(item, "filename"))
            document = document_by_filename.get(filename)
            chunk = legacy_chunk_to_record(item, document)
            explicit_document_id = _optional_str(_first_value(item, "document_id"))
            if document is None and explicit_document_id:
                chunk.document_id = explicit_document_id
            chunks.append(chunk)

    manifest_issues = _collect_manifest_issues(documents)
    plan = DryRunIngestionPlan(
        documents=documents,
        chunks=chunks,
        manifest_issues=manifest_issues,
    )
    plan.warnings = _collect_warnings(plan)
    plan.summary = summarize_dry_run_plan(plan)
    return plan


def summarize_dry_run_plan(plan: DryRunIngestionPlan) -> dict[str, Any]:
    """Return stable summary counts for a dry-run plan."""
    document_ids = [document.document_id for document in plan.documents]
    chunk_document_ids = [chunk.document_id for chunk in plan.chunks]
    known_document_ids = set(document_ids)

    return {
        "document_count": len(plan.documents),
        "chunk_count": len(plan.chunks),
        "duplicate_document_ids": _duplicates(document_ids),
        "unknown_chunk_document_refs": sorted(
            {document_id for document_id in chunk_document_ids if document_id not in known_document_ids}
        ),
        "authorities": sorted({document.authority for document in plan.documents if document.authority}),
        "document_types": sorted(
            {document.document_type for document in plan.documents if document.document_type}
        ),
        "issue_count": len(validate_dry_run_plan(plan)),
        "warning_count": len(plan.warnings),
    }


def validate_dry_run_plan(plan: DryRunIngestionPlan) -> list[str]:
    """Return human-readable dry-run validation issues."""
    issues: list[str] = []

    if not plan.documents:
        issues.append("No documents supplied.")

    document_ids = [document.document_id for document in plan.documents]
    for document_id in _duplicates(document_ids):
        issues.append(f"Duplicate document_id: {document_id}")

    issues.extend(plan.manifest_issues)

    known_document_ids = set(document_ids)
    for chunk in plan.chunks:
        if chunk.document_id not in known_document_ids:
            issues.append(
                f"Chunk references unknown document_id: {chunk.chunk_id} -> {chunk.document_id}"
            )
        if not chunk.text.strip():
            issues.append(f"Chunk has empty text: {chunk.chunk_id}")

    return issues


def _collect_manifest_issues(documents: Iterable[DocumentRecord]) -> list[str]:
    issues: list[str] = []
    for document in documents:
        for issue in validate_manifest_record(document):
            issues.append(f"{document.document_id}: {issue}")
    return issues


def _collect_warnings(plan: DryRunIngestionPlan) -> list[str]:
    warnings: list[str] = []
    for issue in validate_dry_run_plan(plan):
        if (
            "Duplicate document_id" in issue
            or "unknown document_id" in issue
            or "empty text" in issue
        ):
            warnings.append(issue)
    return warnings


def _duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(value for value in values if value)
    return sorted(value for value, count in counts.items() if count > 1)


def _first_value(data: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return value
    return default


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


__all__ = [
    "DryRunIngestionPlan",
    "build_dry_run_ingestion_plan",
    "summarize_dry_run_plan",
    "validate_dry_run_plan",
]
