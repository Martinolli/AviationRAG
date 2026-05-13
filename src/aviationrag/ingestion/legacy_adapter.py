"""Compatibility adapters for legacy-like ingestion records.

These helpers are pure conversion utilities for future migration work. They do
not read source documents, call legacy scripts, write manifests, or change the
runtime ingestion pipeline.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping

from aviationrag.models import ChunkRecord, DocumentRecord


def normalize_legacy_filename(value: str) -> str:
    """Return a stable filename-like value from a legacy path or name."""
    normalized = str(value or "").strip().replace("\\", "/")
    if not normalized:
        return ""
    return PurePosixPath(normalized).name.strip()


def infer_document_type_from_filename(filename: str) -> str | None:
    """Infer a controlled document type from a filename."""
    value = _filename_signal(filename)

    if "ACCIDENT_REPORT" in value:
        return "accident_report"
    if "ADVISORY_CIRCULAR" in value or re.search(r"(^|_)AC(_|$)", value):
        return "advisory_circular"
    if "CERTIFICATION" in value or re.search(r"(^|_)CS(_|[-]?\d|$)", value):
        return "certification_specification"
    if "SAFETY_MANAGEMENT" in value or "SMS" in value:
        return "safety_management"
    if "REGULATION" in value:
        return "regulation"
    if "STANDARD" in value:
        return "standard"
    if "MANUAL" in value:
        return "manual"
    if "BOOK" in value:
        return "book"
    if "PAPER" in value:
        return "paper"
    if "REPORT" in value:
        return "report"
    return "other"


def infer_authority_from_filename(filename: str) -> str | None:
    """Infer a controlled source authority from a filename."""
    value = _filename_signal(filename)
    for token, authority in [
        ("FAA", "FAA"),
        ("EASA", "EASA"),
        ("ICAO", "ICAO"),
        ("NTSB", "NTSB"),
        ("AAIB", "AAIB"),
        ("NASA", "NASA"),
        ("ISO", "ISO"),
    ]:
        if re.search(rf"(^|_){token}(_|$)", value):
            return authority

    if re.search(r"(^|_)(MIL|DOD)(_|$)", value):
        return "MILITARY"
    return None


def build_document_id(filename: str, file_hash: str | None = None) -> str:
    """Build a deterministic, lowercase, filesystem-safe document ID."""
    normalized = normalize_legacy_filename(filename)
    stem = PurePosixPath(normalized).stem or "document"
    slug = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_") or "document"
    digest_source = f"{normalized.lower()}|{file_hash or ''}"
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:12]
    return f"doc_{slug}_{digest}"


def legacy_document_to_record(data: Mapping[str, Any]) -> DocumentRecord:
    """Convert a legacy-like document dictionary into a ``DocumentRecord``."""
    filename = normalize_legacy_filename(_first_value(data, "filename", "file_name", "name"))
    metadata = _metadata_with_fields(
        data,
        [
            "source_type",
            "extraction_method",
            "extraction_quality",
            "needs_manual_review",
            "category",
        ],
    )

    file_hash = _optional_str(_first_value(data, "file_hash", "source_hash"))
    title = _optional_str(
        _first_value(
            data,
            "canonical_title",
            "title",
            default=metadata.get("canonical_title") or metadata.get("title"),
        )
    )
    if not title:
        title = _title_from_filename(filename)

    authority = _optional_str(
        _first_value(data, "authority", default=metadata.get("authority"))
    ) or infer_authority_from_filename(filename)
    document_type = _optional_str(
        _first_value(data, "document_type", default=metadata.get("document_type"))
    ) or infer_document_type_from_filename(filename)

    return DocumentRecord(
        document_id=_optional_str(_first_value(data, "document_id"))
        or build_document_id(filename, file_hash),
        filename=filename,
        title=title,
        authority=authority,
        document_type=document_type,
        revision=_optional_str(_first_value(data, "revision", default=metadata.get("revision"))),
        effective_date=_optional_str(
            _first_value(data, "effective_date", default=metadata.get("effective_date"))
        ),
        source_url=_optional_str(_first_value(data, "source_url", "source_uri")),
        file_hash=file_hash,
        ingestion_status=_optional_str(_first_value(data, "ingestion_status")) or "discovered",
        created_at=_optional_str(_first_value(data, "created_at")),
        metadata=metadata,
    )


def legacy_chunk_to_record(
    data: Mapping[str, Any],
    document: DocumentRecord | None = None,
) -> ChunkRecord:
    """Convert a legacy-like chunk dictionary into a ``ChunkRecord``."""
    filename = normalize_legacy_filename(
        _first_value(data, "filename", default=document.filename if document else "")
    )
    metadata = _metadata_with_fields(
        data,
        [
            "chunk_type",
            "tokens",
            "source_type",
            "extraction_method",
            "extraction_quality",
            "needs_manual_review",
        ],
    )

    document_id = document.document_id if document else build_document_id(filename)
    chunk_id = _optional_str(_first_value(data, "chunk_id")) or _fallback_chunk_id(
        document_id,
        data,
    )
    page_start = _optional_int(_first_value(data, "page_start", "page"))
    page_end = _optional_int(_first_value(data, "page_end", default=page_start))

    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        filename=filename,
        text=_optional_str(_first_value(data, "text")) or "",
        page_start=page_start,
        page_end=page_end,
        section_path=_section_path(data.get("section_path")),
        metadata=metadata,
    )


def legacy_documents_to_records(items: Iterable[Mapping[str, Any]]) -> list[DocumentRecord]:
    """Convert legacy-like document dictionaries into records."""
    return [legacy_document_to_record(item) for item in items]


def legacy_chunks_to_records(
    items: Iterable[Mapping[str, Any]],
    documents: Iterable[DocumentRecord] | None = None,
) -> list[ChunkRecord]:
    """Convert legacy-like chunk dictionaries into records."""
    document_by_filename: dict[str, DocumentRecord] = {}
    if documents is not None:
        document_by_filename = {
            normalize_legacy_filename(document.filename): document for document in documents
        }

    records: list[ChunkRecord] = []
    for item in items:
        filename = normalize_legacy_filename(_first_value(item, "filename"))
        records.append(legacy_chunk_to_record(item, document_by_filename.get(filename)))
    return records


def _filename_signal(filename: str) -> str:
    normalized = normalize_legacy_filename(filename)
    return re.sub(r"[^A-Z0-9]+", "_", normalized.upper()).strip("_")


def _first_value(data: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None and value != "":
            return value
    return default


def _metadata_with_fields(data: Mapping[str, Any], field_names: Iterable[str]) -> dict[str, Any]:
    metadata = dict(data.get("metadata") or {})
    for field_name in field_names:
        value = data.get(field_name)
        if value is not None:
            metadata[field_name] = value
    return metadata


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


def _title_from_filename(filename: str) -> str | None:
    if not filename:
        return None
    stem = PurePosixPath(filename).stem
    title = re.sub(r"[_-]+", " ", stem).strip()
    return title or None


def _fallback_chunk_id(document_id: str, data: Mapping[str, Any]) -> str:
    text = _optional_str(_first_value(data, "text")) or ""
    page = _optional_str(_first_value(data, "page_start", "page", "page_end")) or ""
    digest = hashlib.sha256(f"{document_id}|{page}|{text}".encode("utf-8")).hexdigest()[:12]
    return f"{document_id}_chunk_{digest}"


__all__ = [
    "build_document_id",
    "infer_authority_from_filename",
    "infer_document_type_from_filename",
    "legacy_chunk_to_record",
    "legacy_chunks_to_records",
    "legacy_document_to_record",
    "legacy_documents_to_records",
    "normalize_legacy_filename",
]
