"""Passive legacy chunk adapter for future metadata-rich migration.

These helpers convert explicit legacy-like chunk dictionaries into lightweight
``ChunkRecord`` objects and optional vector payload-shaped dictionaries. They
do not read source folders, write files, call external services, generate
embeddings, connect to Astra, use FAISS, or change runtime ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping

from aviationrag.config import get_chunk_migration_settings
from aviationrag.ingestion.chunk_payload import (
    chunk_to_vector_payload,
    validate_vector_payload,
)
from aviationrag.ingestion.chunk_schema import (
    validate_chunk_record,
    validate_chunk_record_dict,
)
from aviationrag.ingestion.legacy_adapter import (
    build_document_id,
    infer_authority_from_filename,
    infer_document_type_from_filename,
)
from aviationrag.models import ChunkRecord


@dataclass
class ChunkMigrationPreview:
    """Side-effect-free preview of future legacy chunk migration output."""

    chunks: list[ChunkRecord] = field(default_factory=list)
    payloads: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def normalize_legacy_chunk_filename(value: str) -> str:
    """Return a stable filename from a legacy path-like value."""
    normalized = str(value or "").strip().replace("\\", "/")
    if not normalized:
        return ""
    return PurePosixPath(normalized).name.strip()


def legacy_chunk_dict_to_chunk_record(
    data: Mapping[str, Any],
    document_id: str | None = None,
) -> ChunkRecord:
    """Convert a fake legacy-like chunk dictionary into a ``ChunkRecord``."""
    filename = normalize_legacy_chunk_filename(_first_value(data, "filename", "file_name", "name"))
    text = _optional_str(_first_value(data, "text", "content")) or ""
    resolved_document_id = (
        document_id
        or _optional_str(_first_value(data, "document_id"))
        or build_document_id(filename)
    )

    page_start = _optional_int(_first_value(data, "page_start", "page"))
    page_end = _optional_int(_first_value(data, "page_end", default=page_start))
    section_path = _section_path(_first_value(data, "section_path", "section"))
    source_metadata = dict(data.get("metadata") or {})

    metadata = _metadata_with_non_core_fields(data)
    metadata.update(source_metadata)
    metadata.update(
        {
            "canonical_title": _canonical_title(data, filename, source_metadata),
            "chunk_type": _optional_str(_first_value(data, "chunk_type", default=source_metadata.get("chunk_type"))) or "text",
            "authority": _optional_str(_first_value(data, "authority", default=source_metadata.get("authority")))
            or infer_authority_from_filename(filename)
            or "OTHER",
            "document_type": _optional_str(
                _first_value(data, "document_type", default=source_metadata.get("document_type"))
            )
            or infer_document_type_from_filename(filename)
            or "other",
            "revision": _optional_str(_first_value(data, "revision", default=source_metadata.get("revision"))),
            "effective_date": _optional_str(
                _first_value(data, "effective_date", default=source_metadata.get("effective_date"))
            ),
            "extraction_quality": _optional_str(
                _first_value(data, "extraction_quality", default=source_metadata.get("extraction_quality"))
            )
            or "unknown",
            "paragraph_id": _optional_str(
                _first_value(data, "paragraph_id", default=source_metadata.get("paragraph_id"))
            ),
            "text_hash": _optional_str(_first_value(data, "text_hash", default=source_metadata.get("text_hash")))
            or _hash_value(text),
            "source_hash": _optional_str(
                _first_value(data, "source_hash", "file_hash", default=source_metadata.get("source_hash"))
            )
            or _hash_value(filename),
            "created_at": _optional_str(_first_value(data, "created_at", default=source_metadata.get("created_at"))),
            "source_legacy": True,
        }
    )

    chunk_id = _optional_str(_first_value(data, "chunk_id")) or _fallback_chunk_id(
        filename,
        page_start,
        text,
    )

    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=resolved_document_id,
        filename=filename,
        text=text,
        page_start=page_start,
        page_end=page_end,
        section_path=section_path,
        metadata=metadata,
    )


def legacy_chunk_dicts_to_chunk_records(
    items: Iterable[Mapping[str, Any]],
    document_ids_by_filename: Mapping[str, str] | None = None,
) -> list[ChunkRecord]:
    """Convert fake legacy-like chunk dictionaries into records."""
    document_map = {
        normalize_legacy_chunk_filename(filename): document_id
        for filename, document_id in (document_ids_by_filename or {}).items()
    }

    records: list[ChunkRecord] = []
    for item in items:
        filename = normalize_legacy_chunk_filename(_first_value(item, "filename", "file_name", "name"))
        records.append(
            legacy_chunk_dict_to_chunk_record(
                item,
                document_id=document_map.get(filename),
            )
        )
    return records


def preview_legacy_chunk_migration(
    items: Iterable[Mapping[str, Any]],
    document_ids_by_filename: Mapping[str, str] | None = None,
    include_payloads: bool = True,
    env: Mapping[str, str] | None = None,
) -> ChunkMigrationPreview:
    """Build a side-effect-free preview for future legacy chunk migration."""
    settings = get_chunk_migration_settings(env)
    warnings: list[str] = []
    if not settings.enabled:
        warnings.append("Chunk migration is disabled; preview only.")
    if settings.dry_run:
        warnings.append("Chunk migration dry-run mode is enabled.")

    chunks = legacy_chunk_dicts_to_chunk_records(items, document_ids_by_filename)
    issues: list[str] = []
    payloads: list[dict[str, Any]] = []

    for chunk in chunks:
        label = chunk.chunk_id or "<missing chunk_id>"
        for issue in validate_chunk_record(chunk):
            issues.append(f"{label}: {issue}")

        if include_payloads:
            payload = chunk_to_vector_payload(chunk)
            payloads.append(payload)
            for issue in validate_vector_payload(payload):
                issues.append(f"{label}: payload: {issue}")

    preview = ChunkMigrationPreview(
        chunks=chunks,
        payloads=payloads,
        issues=issues,
        warnings=warnings,
    )
    preview.summary = summarize_chunk_migration_preview(preview)
    return preview


def summarize_chunk_migration_preview(preview: ChunkMigrationPreview) -> dict[str, Any]:
    """Summarize a chunk migration preview."""
    return {
        "chunk_count": len(preview.chunks),
        "payload_count": len(preview.payloads),
        "issue_count": len(preview.issues),
        "warning_count": len(preview.warnings),
        "chunk_types": sorted(
            {
                str(chunk.metadata.get("chunk_type"))
                for chunk in preview.chunks
                if chunk.metadata.get("chunk_type")
            }
        ),
        "document_ids": sorted(
            {chunk.document_id for chunk in preview.chunks if chunk.document_id}
        ),
    }


def _metadata_with_non_core_fields(data: Mapping[str, Any]) -> dict[str, Any]:
    core = {
        "chunk_id",
        "document_id",
        "filename",
        "file_name",
        "name",
        "text",
        "content",
        "page",
        "page_start",
        "page_end",
        "section",
        "section_path",
        "metadata",
    }
    return {key: value for key, value in data.items() if key not in core}


def _canonical_title(
    data: Mapping[str, Any],
    filename: str,
    metadata: Mapping[str, Any],
) -> str | None:
    title = _optional_str(
        _first_value(
            data,
            "canonical_title",
            "document_title",
            "title",
            default=metadata.get("canonical_title") or metadata.get("document_title") or metadata.get("title"),
        )
    )
    if title:
        return title
    if not filename:
        return None
    return re.sub(r"[_-]+", " ", PurePosixPath(filename).stem).strip() or None


def _fallback_chunk_id(filename: str, page: int | None, text: str) -> str:
    slug_source = PurePosixPath(filename).stem or "chunk"
    slug = re.sub(r"[^a-z0-9]+", "_", slug_source.lower()).strip("_") or "chunk"
    digest = hashlib.sha256(f"{filename}|{page or ''}|{text}".encode("utf-8")).hexdigest()[:12]
    return f"legacy_{slug}_chunk_{digest}"


def _hash_value(value: str) -> str:
    return "sha256:" + hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


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


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _section_path(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable):
        return [str(item) for item in value if item is not None and str(item)]
    return []


__all__ = [
    "ChunkMigrationPreview",
    "legacy_chunk_dict_to_chunk_record",
    "legacy_chunk_dicts_to_chunk_records",
    "normalize_legacy_chunk_filename",
    "preview_legacy_chunk_migration",
    "summarize_chunk_migration_preview",
]
