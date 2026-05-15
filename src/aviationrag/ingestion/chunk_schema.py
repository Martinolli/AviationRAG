"""Metadata-rich chunk schema validation helpers.

This module validates explicit chunk dictionaries and lightweight
``ChunkRecord`` objects for future manifest-driven ingestion work. It is not
connected to runtime ingestion, retrieval, embeddings, Astra, or FAISS.
"""

from __future__ import annotations

from collections import Counter
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from aviationrag.models import ChunkRecord


REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "document_id",
    "filename",
    "canonical_title",
    "text",
    "text_hash",
    "source_hash",
    "chunk_type",
    "page_start",
    "page_end",
    "section_path",
    "authority",
    "document_type",
    "extraction_quality",
    "created_at",
    "metadata",
}

OPTIONAL_CHUNK_FIELDS = {
    "paragraph_id",
    "revision",
    "effective_date",
    "table_id",
    "figure_id",
    "caption",
    "warning_type",
    "regulatory_reference",
    "applicability",
    "aircraft_category",
    "product_type",
    "confidence_score",
    "language",
    "source_uri",
    "source_url",
    "source_page_url",
    "reviewer_notes",
    "lifecycle_status",
    "chunk_version",
    "parent_chunk_id",
    "previous_chunk_id",
}

ALLOWED_CHUNK_TYPES = {
    "text",
    "section",
    "paragraph",
    "regulatory_paragraph",
    "table",
    "figure_caption",
    "warning",
    "caution",
    "note",
    "definition",
    "checklist",
    "procedure",
    "requirement",
    "accident_finding",
    "safety_recommendation",
    "metadata_only",
    "other",
}

ALLOWED_LIFECYCLE_STATES = {
    "active",
    "superseded",
    "retired",
    "needs_review",
    "error",
}

_DATACLASS_FIELDS = {
    "chunk_id",
    "document_id",
    "filename",
    "text",
    "page_start",
    "page_end",
    "section_path",
    "metadata",
}

_PATH_PATTERNS = [
    re.compile(r"[A-Za-z]:[\\/]"),
    re.compile(r"^/"),
    re.compile(r"^\\\\"),
    re.compile(r"(^|[\\/])data[\\/](documents|raw|processed|embeddings|manifest)([\\/]|$)"),
    re.compile(r"(^|[\\/])(Users|home)([\\/]|$)", re.IGNORECASE),
    re.compile(r"secure-connect", re.IGNORECASE),
    re.compile(r"\.env", re.IGNORECASE),
]


def validate_chunk_record_dict(data: Mapping[str, Any]) -> list[str]:
    """Return human-readable validation issues for one chunk dictionary."""
    issues: list[str] = []

    for field_name in sorted(REQUIRED_CHUNK_FIELDS):
        if field_name not in data:
            issues.append(f"Missing required field: {field_name}")

    chunk_id = data.get("chunk_id")
    if not _non_empty_string(chunk_id):
        issues.append("chunk_id must be a non-empty string.")

    document_id = data.get("document_id")
    if not _non_empty_string(document_id):
        issues.append("document_id must be a non-empty string.")

    filename = data.get("filename")
    if "filename" in data and not _non_empty_string(filename):
        issues.append("filename must be a non-empty string.")

    text = data.get("text")
    if not _non_empty_string(text):
        issues.append("text must be a non-empty string.")

    chunk_type = data.get("chunk_type")
    if chunk_type not in ALLOWED_CHUNK_TYPES:
        issues.append(f"chunk_type must be one of the allowed values: {chunk_type!r}")

    for hash_field in ("text_hash", "source_hash"):
        if hash_field in data and not _non_empty_string(data.get(hash_field)):
            issues.append(f"{hash_field} must be a non-empty string.")

    issues.extend(_validate_page_range(data))

    if "section_path" in data and not isinstance(data.get("section_path"), list):
        issues.append("section_path must be a list.")

    metadata = data.get("metadata")
    if "metadata" in data and not isinstance(metadata, dict):
        issues.append("metadata must be a dictionary.")

    issues.extend(_validate_confidence_score(data))
    issues.extend(_validate_lifecycle_state(data))
    issues.extend(_validate_private_paths(data))

    return issues


def validate_chunk_record(record: ChunkRecord) -> list[str]:
    """Return validation issues for a lightweight ``ChunkRecord``."""
    return validate_chunk_record_dict(chunk_record_to_dict(record))


def chunk_record_from_dict(data: Mapping[str, Any]) -> ChunkRecord:
    """Convert a metadata-rich chunk dictionary into a lightweight record."""
    metadata = data.get("metadata")
    merged_metadata: dict[str, Any] = dict(metadata) if isinstance(metadata, Mapping) else {}

    for key, value in data.items():
        if key not in _DATACLASS_FIELDS:
            merged_metadata.setdefault(key, value)

    return ChunkRecord(
        chunk_id=str(data.get("chunk_id") or ""),
        document_id=str(data.get("document_id") or ""),
        filename=str(data.get("filename") or ""),
        text=str(data.get("text") or ""),
        page_start=_optional_int(data.get("page_start")),
        page_end=_optional_int(data.get("page_end")),
        section_path=_string_list(data.get("section_path")),
        metadata=merged_metadata,
    )


def chunk_record_to_dict(record: ChunkRecord) -> dict[str, Any]:
    """Convert a lightweight ``ChunkRecord`` into a dictionary.

    Extra metadata-rich schema fields are restored from ``record.metadata`` when
    present. This keeps the core dataclass lightweight while allowing schema
    validation to exercise the future full chunk contract.
    """
    metadata = dict(record.metadata)
    data: dict[str, Any] = {
        "chunk_id": record.chunk_id,
        "document_id": record.document_id,
        "filename": record.filename,
        "text": record.text,
        "page_start": record.page_start,
        "page_end": record.page_end,
        "section_path": list(record.section_path),
        "metadata": metadata,
    }

    for field_name in sorted(REQUIRED_CHUNK_FIELDS | OPTIONAL_CHUNK_FIELDS):
        if field_name not in data and field_name in metadata:
            data[field_name] = metadata[field_name]

    return data


def validate_chunk_dataset(
    chunks: Iterable[Mapping[str, Any]],
    known_document_ids: Iterable[str] | None = None,
) -> list[str]:
    """Return validation issues for a chunk dataset."""
    chunk_list = list(chunks)
    issues: list[str] = []
    known_ids = set(known_document_ids or [])

    chunk_ids = [str(chunk.get("chunk_id")) for chunk in chunk_list if chunk.get("chunk_id")]
    for chunk_id, count in sorted(Counter(chunk_ids).items()):
        if count > 1:
            issues.append(f"Duplicate chunk_id: {chunk_id}")

    for index, chunk in enumerate(chunk_list, start=1):
        label = str(chunk.get("chunk_id") or f"line {index}")
        for issue in validate_chunk_record_dict(chunk):
            issues.append(f"{label}: {issue}")

        document_id = chunk.get("document_id")
        if known_document_ids is not None and document_id and document_id not in known_ids:
            issues.append(f"{label}: Unknown document_id: {document_id}")

    return issues


def load_chunk_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load chunk records from a JSONL file."""
    chunk_path = Path(path)
    chunks: list[dict[str, Any]] = []

    with chunk_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in chunk fixture {chunk_path} at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid JSON object in chunk fixture {chunk_path} at line {line_number}"
                )
            chunks.append(item)

    return chunks


def _validate_page_range(data: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    page_start = data.get("page_start")
    page_end = data.get("page_end")

    for field_name, value in (("page_start", page_start), ("page_end", page_end)):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
            issues.append(f"{field_name} must be a positive integer or null.")

    if (
        isinstance(page_start, int)
        and not isinstance(page_start, bool)
        and isinstance(page_end, int)
        and not isinstance(page_end, bool)
        and page_end < page_start
    ):
        issues.append("page_end must be greater than or equal to page_start.")

    return issues


def _validate_confidence_score(data: Mapping[str, Any]) -> list[str]:
    value = data.get("confidence_score")
    metadata = data.get("metadata")
    if value is None and isinstance(metadata, Mapping):
        value = metadata.get("confidence_score")

    if value is None:
        return []

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return ["confidence_score must be a number between 0 and 1."]
    if value < 0 or value > 1:
        return ["confidence_score must be between 0 and 1."]
    return []


def _validate_lifecycle_state(data: Mapping[str, Any]) -> list[str]:
    value = data.get("lifecycle_status")
    metadata = data.get("metadata")
    if value is None and isinstance(metadata, Mapping):
        value = metadata.get("lifecycle_status")

    if value is None:
        return []
    if value not in ALLOWED_LIFECYCLE_STATES:
        return [f"lifecycle_status must be one of the allowed values: {value!r}"]
    return []


def _validate_private_paths(data: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    fields_to_check = {"filename", "source_uri", "source_url", "source_page_url", "source_path"}

    for field_name in sorted(fields_to_check):
        value = data.get(field_name)
        if isinstance(value, str) and _looks_like_private_path(value):
            issues.append(f"{field_name} appears to contain a local/private path.")

    metadata = data.get("metadata")
    if isinstance(metadata, Mapping):
        for field_name in sorted(fields_to_check):
            value = metadata.get(field_name)
            if isinstance(value, str) and _looks_like_private_path(value):
                issues.append(f"metadata.{field_name} appears to contain a local/private path.")

    return issues


def _looks_like_private_path(value: str) -> bool:
    if "://" in value:
        return False
    return any(pattern.search(value) for pattern in _PATH_PATTERNS)


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


__all__ = [
    "ALLOWED_CHUNK_TYPES",
    "ALLOWED_LIFECYCLE_STATES",
    "OPTIONAL_CHUNK_FIELDS",
    "REQUIRED_CHUNK_FIELDS",
    "chunk_record_from_dict",
    "chunk_record_to_dict",
    "load_chunk_jsonl",
    "validate_chunk_dataset",
    "validate_chunk_record",
    "validate_chunk_record_dict",
]
