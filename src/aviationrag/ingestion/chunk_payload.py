"""Future vector payload-shaped exports for fake/sample chunks.

The helpers in this module convert validated metadata-rich chunks into plain
payload dictionaries that resemble future vector-store metadata records. They
do not create embeddings, call external APIs, connect to Astra, use FAISS, or
write to any vector database.
"""

from __future__ import annotations

from collections import Counter
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from aviationrag.ingestion.chunk_schema import (
    chunk_record_to_dict,
    load_chunk_jsonl,
    validate_chunk_dataset,
)
from aviationrag.models import ChunkRecord


VECTOR_PAYLOAD_SCHEMA_VERSION = "chunk_payload_v1"

PAYLOAD_METADATA_FIELDS = {
    "filename",
    "canonical_title",
    "authority",
    "document_type",
    "revision",
    "effective_date",
    "page_start",
    "page_end",
    "section_path",
    "paragraph_id",
    "chunk_type",
    "text_hash",
    "source_hash",
    "extraction_quality",
    "created_at",
}

OPTIONAL_PAYLOAD_METADATA_FIELDS = {
    "regulatory_reference",
    "applicability",
    "aircraft_category",
    "product_type",
    "confidence_score",
    "language",
    "warning_type",
    "table_id",
    "figure_id",
    "caption",
    "reviewer_notes",
}

FORBIDDEN_VECTOR_FIELDS = {
    "embedding",
    "embeddings",
    "embedding_vector",
    "vector",
    "vectors",
    "values",
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


def chunk_to_vector_payload(chunk: Mapping[str, Any] | ChunkRecord) -> dict[str, Any]:
    """Convert a chunk dictionary or ``ChunkRecord`` into a vector payload shape."""
    data = _chunk_to_mapping(chunk)
    source_metadata = data.get("metadata")
    if not isinstance(source_metadata, Mapping):
        source_metadata = {}

    metadata: dict[str, Any] = {}
    for field_name in sorted(PAYLOAD_METADATA_FIELDS):
        metadata[field_name] = data.get(field_name, source_metadata.get(field_name))

    for field_name in sorted(OPTIONAL_PAYLOAD_METADATA_FIELDS):
        value = data.get(field_name, source_metadata.get(field_name))
        if value is not None:
            metadata[field_name] = value

    return {
        "payload_schema_version": VECTOR_PAYLOAD_SCHEMA_VERSION,
        "chunk_id": data.get("chunk_id"),
        "document_id": data.get("document_id"),
        "text": data.get("text"),
        "metadata": metadata,
    }


def chunks_to_vector_payloads(
    chunks: Iterable[Mapping[str, Any] | ChunkRecord],
) -> list[dict[str, Any]]:
    """Convert multiple chunks into vector payload-shaped dictionaries."""
    return [chunk_to_vector_payload(chunk) for chunk in chunks]


def validate_vector_payload(payload: Mapping[str, Any]) -> list[str]:
    """Return human-readable validation issues for one payload."""
    issues: list[str] = []

    if not _non_empty_string(payload.get("payload_schema_version")):
        issues.append("payload_schema_version must be present.")
    if not _non_empty_string(payload.get("chunk_id")):
        issues.append("chunk_id must be present.")
    if not _non_empty_string(payload.get("document_id")):
        issues.append("document_id must be present.")
    if not _non_empty_string(payload.get("text")):
        issues.append("text must be a non-empty string.")

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        issues.append("metadata must be a dictionary.")
        metadata = {}

    for field_name in sorted(PAYLOAD_METADATA_FIELDS):
        if field_name not in metadata:
            issues.append(f"metadata missing traceability field: {field_name}")

    forbidden_fields = _find_forbidden_vector_fields(payload)
    for field_name in forbidden_fields:
        issues.append(f"Forbidden embedding/vector field present: {field_name}")

    issues.extend(_validate_private_paths(payload))

    return issues


def validate_vector_payload_dataset(
    payloads: Iterable[Mapping[str, Any]],
) -> list[str]:
    """Return validation issues for a payload dataset."""
    payload_list = list(payloads)
    issues: list[str] = []

    chunk_ids = [
        str(payload.get("chunk_id"))
        for payload in payload_list
        if payload.get("chunk_id")
    ]
    for chunk_id, count in sorted(Counter(chunk_ids).items()):
        if count > 1:
            issues.append(f"Duplicate chunk_id: {chunk_id}")

    for index, payload in enumerate(payload_list, start=1):
        label = str(payload.get("chunk_id") or f"payload {index}")
        for issue in validate_vector_payload(payload):
            issues.append(f"{label}: {issue}")

    return issues


def load_sample_chunk_payloads(path: str | Path) -> list[dict[str, Any]]:
    """Load a fake/sample chunk fixture and return validated payloads."""
    chunks = load_chunk_jsonl(path)
    chunk_issues = validate_chunk_dataset(chunks)
    if chunk_issues:
        raise ValueError("Invalid chunk fixture: " + "; ".join(chunk_issues))

    payloads = chunks_to_vector_payloads(chunks)
    payload_issues = validate_vector_payload_dataset(payloads)
    if payload_issues:
        raise ValueError("Invalid vector payloads: " + "; ".join(payload_issues))

    return payloads


def _chunk_to_mapping(chunk: Mapping[str, Any] | ChunkRecord) -> Mapping[str, Any]:
    if isinstance(chunk, ChunkRecord):
        return chunk_record_to_dict(chunk)
    return chunk


def _find_forbidden_vector_fields(payload: Mapping[str, Any]) -> list[str]:
    fields: list[str] = []

    for key in payload:
        if str(key).lower() in FORBIDDEN_VECTOR_FIELDS:
            fields.append(str(key))

    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        for key in metadata:
            if str(key).lower() in FORBIDDEN_VECTOR_FIELDS:
                fields.append(f"metadata.{key}")

    return sorted(fields)


def _validate_private_paths(payload: Mapping[str, Any]) -> list[str]:
    issues: list[str] = []
    metadata = payload.get("metadata")

    values_to_check: list[tuple[str, Any]] = [
        ("filename", payload.get("filename")),
        ("source_uri", payload.get("source_uri")),
        ("source_url", payload.get("source_url")),
        ("source_page_url", payload.get("source_page_url")),
    ]

    if isinstance(metadata, Mapping):
        values_to_check.extend(
            [
                ("metadata.filename", metadata.get("filename")),
                ("metadata.source_uri", metadata.get("source_uri")),
                ("metadata.source_url", metadata.get("source_url")),
                ("metadata.source_page_url", metadata.get("source_page_url")),
            ]
        )

    for field_name, value in values_to_check:
        if isinstance(value, str) and _looks_like_private_path(value):
            issues.append(f"{field_name} appears to contain a local/private path.")

    return issues


def _looks_like_private_path(value: str) -> bool:
    if "://" in value:
        return False
    return any(pattern.search(value) for pattern in _PATH_PATTERNS)


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


__all__ = [
    "OPTIONAL_PAYLOAD_METADATA_FIELDS",
    "PAYLOAD_METADATA_FIELDS",
    "VECTOR_PAYLOAD_SCHEMA_VERSION",
    "chunk_to_vector_payload",
    "chunks_to_vector_payloads",
    "load_sample_chunk_payloads",
    "validate_vector_payload",
    "validate_vector_payload_dataset",
]
