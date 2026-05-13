"""Safe JSONL manifest utilities for future ingestion migration.

This module is intentionally standalone. It is not wired into the legacy
ingestion pipeline and should be exercised only with fake/sample data until a
controlled migration is approved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from aviationrag.models import DocumentRecord


_EXTRA_METADATA_KEY = "_manifest_extra"

_MODEL_FIELDS = {
    "document_id",
    "filename",
    "title",
    "authority",
    "document_type",
    "revision",
    "effective_date",
    "source_url",
    "file_hash",
    "ingestion_status",
    "created_at",
    "metadata",
}

_MANIFEST_ALIASES = {
    "canonical_title": "title",
    "source_uri": "source_url",
}


def read_manifest(path: str | Path) -> list[DocumentRecord]:
    """Read a JSONL manifest file into ``DocumentRecord`` objects."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        return []

    records: list[DocumentRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in manifest {manifest_path} at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(data, dict):
                raise ValueError(
                    f"Invalid JSON object in manifest {manifest_path} at line {line_number}"
                )
            records.append(document_record_from_dict(data))
    return records


def write_manifest(path: str | Path, records: Iterable[DocumentRecord]) -> None:
    """Write records to a JSONL manifest file, replacing any existing file."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_record_json_line(record))


def append_manifest_record(path: str | Path, record: DocumentRecord) -> None:
    """Append one record to a JSONL manifest file."""
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(_record_json_line(record))


def validate_manifest_record(record: DocumentRecord) -> list[str]:
    """Return human-readable validation issues for a manifest record."""
    issues: list[str] = []
    required_values = {
        "document_id": record.document_id,
        "filename": record.filename,
        "title": record.title,
        "authority": record.authority,
        "document_type": record.document_type,
        "file_hash": record.file_hash,
        "ingestion_status": record.ingestion_status,
    }

    for field_name, value in required_values.items():
        if value is None or (isinstance(value, str) and not value.strip()):
            display_name = "canonical_title or title" if field_name == "title" else field_name
            issues.append(f"Missing required field: {display_name}")

    return issues


def document_record_from_dict(data: Mapping[str, Any]) -> DocumentRecord:
    """Create a ``DocumentRecord`` from manifest-shaped data.

    ``models.DocumentRecord`` currently uses ``title`` and ``source_url`` while
    the manifest schema uses ``canonical_title`` and ``source_uri``. This
    function maps those schema names into the lightweight model and preserves
    additional manifest fields in ``metadata['_manifest_extra']`` for round
    trips.
    """
    mapped: dict[str, Any] = {}
    recognized_keys = set(_MODEL_FIELDS) | set(_MANIFEST_ALIASES)

    for key, value in data.items():
        mapped_key = _MANIFEST_ALIASES.get(key, key)
        if mapped_key in _MODEL_FIELDS and mapped_key != "metadata":
            mapped[mapped_key] = value

    metadata = dict(data.get("metadata") or {})
    extras = {key: value for key, value in data.items() if key not in recognized_keys}
    if extras:
        metadata[_EXTRA_METADATA_KEY] = extras

    return DocumentRecord(
        document_id=str(mapped.get("document_id") or ""),
        filename=str(mapped.get("filename") or ""),
        title=_optional_str(mapped.get("title")),
        authority=_optional_str(mapped.get("authority")),
        document_type=_optional_str(mapped.get("document_type")),
        revision=_optional_str(mapped.get("revision")),
        effective_date=_optional_str(mapped.get("effective_date")),
        source_url=_optional_str(mapped.get("source_url")),
        file_hash=_optional_str(mapped.get("file_hash")),
        ingestion_status=_optional_str(mapped.get("ingestion_status")),
        created_at=_optional_str(mapped.get("created_at")),
        metadata=metadata,
    )


def document_record_to_dict(record: DocumentRecord) -> dict[str, Any]:
    """Convert a ``DocumentRecord`` to manifest-shaped data."""
    metadata = dict(record.metadata or {})
    extras = metadata.pop(_EXTRA_METADATA_KEY, {})
    if not isinstance(extras, dict):
        extras = {}

    data: dict[str, Any] = {
        "document_id": record.document_id,
        "filename": record.filename,
        "canonical_title": record.title,
        "authority": record.authority,
        "document_type": record.document_type,
        "revision": record.revision,
        "effective_date": record.effective_date,
        "source_uri": record.source_url,
        "file_hash": record.file_hash,
        "ingestion_status": record.ingestion_status,
        "created_at": record.created_at,
        "metadata": metadata,
    }
    data.update(extras)
    return data


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _record_json_line(record: DocumentRecord) -> str:
    return json.dumps(
        document_record_to_dict(record),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


__all__ = [
    "append_manifest_record",
    "document_record_from_dict",
    "document_record_to_dict",
    "read_manifest",
    "validate_manifest_record",
    "write_manifest",
]
