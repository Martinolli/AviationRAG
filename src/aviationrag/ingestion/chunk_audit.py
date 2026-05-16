"""Read-only legacy chunk format audit helpers.

These utilities inspect explicitly supplied chunk-like files or in-memory
records and summarize schema shape without migrating, rewriting, embedding,
indexing, or scanning directories.

Pickle support is intentionally limited to explicitly provided files and should
only be used with trusted local files because Python pickle can execute code
during loading.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping


SUPPORTED_CHUNK_AUDIT_FORMATS = {"jsonl", "json", "pickle"}
_PAGE_FIELDS = {"page", "page_start", "page_end"}
_SECTION_FIELDS = {"section", "section_path", "section_references"}


@dataclass
class ChunkAuditSummary:
    """Schema-shape summary for chunk-like records."""

    source_path: str
    record_count: int
    detected_format: str
    top_level_keys: dict[str, int] = field(default_factory=dict)
    metadata_keys: dict[str, int] = field(default_factory=dict)
    chunk_type_counts: dict[str, int] = field(default_factory=dict)
    document_id_count: int = 0
    missing_text_count: int = 0
    missing_chunk_id_count: int = 0
    missing_document_id_count: int = 0
    page_field_count: int = 0
    section_field_count: int = 0
    sample_record_shapes: list[dict[str, Any]] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def detect_chunk_file_format(path: str | Path) -> str:
    """Detect a supported chunk-like file format from its suffix."""
    chunk_path = Path(path)
    suffix = chunk_path.suffix.lower()
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix in {".pkl", ".pickle"}:
        return "pickle"
    return "unknown"


def load_chunk_like_records(
    path: str | Path,
    max_records: int | None = None,
) -> list[dict[str, Any]]:
    """Load chunk-like records from one explicit file path.

    This function does not scan directories. Pickle files should only be loaded
    from trusted local paths.
    """
    chunk_path = Path(path)
    if chunk_path.is_dir():
        raise ValueError(f"Chunk audit input must be a file, not a directory: {chunk_path}")

    detected_format = detect_chunk_file_format(chunk_path)
    if detected_format == "jsonl":
        records = _load_jsonl(chunk_path, max_records)
    elif detected_format == "json":
        records = _flatten_loaded_object(_load_json(chunk_path), source_path=str(chunk_path))
    elif detected_format == "pickle":
        records = _flatten_loaded_object(_load_pickle(chunk_path), source_path=str(chunk_path))
    else:
        raise ValueError(f"Unsupported chunk audit file format: {chunk_path.suffix or '<none>'}")

    return _limit_records(records, max_records)


def audit_chunk_records(
    records: Iterable[Mapping[str, Any]],
    source_path: str = "<memory>",
) -> ChunkAuditSummary:
    """Audit chunk-like record structure without including full text values."""
    record_list = [dict(record) for record in records]
    top_level_keys = summarize_key_frequency(record_list)
    metadata_keys = _summarize_metadata_keys(record_list)
    chunk_type_counts = _summarize_chunk_types(record_list)
    issues: list[str] = []
    warnings: list[str] = []

    if not record_list:
        warnings.append("No chunk-like records supplied.")

    summary = ChunkAuditSummary(
        source_path=source_path,
        record_count=len(record_list),
        detected_format=detect_chunk_file_format(source_path),
        top_level_keys=top_level_keys,
        metadata_keys=metadata_keys,
        chunk_type_counts=chunk_type_counts,
        document_id_count=sum(1 for record in record_list if _has_value(record.get("document_id"))),
        missing_text_count=sum(1 for record in record_list if not _has_value(record.get("text"))),
        missing_chunk_id_count=sum(1 for record in record_list if not _has_value(record.get("chunk_id"))),
        missing_document_id_count=sum(1 for record in record_list if not _has_value(record.get("document_id"))),
        page_field_count=sum(1 for record in record_list if any(field in record for field in _PAGE_FIELDS)),
        section_field_count=sum(1 for record in record_list if any(field in record for field in _SECTION_FIELDS)),
        sample_record_shapes=[_record_shape(record) for record in record_list[:5]],
        issues=issues,
        warnings=warnings,
    )

    if summary.missing_text_count:
        issues.append(f"{summary.missing_text_count} record(s) are missing text.")
    if summary.missing_chunk_id_count:
        issues.append(f"{summary.missing_chunk_id_count} record(s) are missing chunk_id.")
    if summary.missing_document_id_count:
        warnings.append(f"{summary.missing_document_id_count} record(s) are missing document_id.")
    if summary.page_field_count == 0 and record_list:
        warnings.append("No page/page_start/page_end fields detected.")
    if summary.section_field_count == 0 and record_list:
        warnings.append("No section/section_path/section_references fields detected.")

    return summary


def summarize_key_frequency(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Return sorted top-level key frequencies."""
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(str(key) for key in record.keys())
    return dict(sorted(counter.items()))


def chunk_audit_summary_to_dict(summary: ChunkAuditSummary) -> dict[str, Any]:
    """Return a JSON-serializable audit summary dictionary."""
    return asdict(summary)


def _load_jsonl(path: Path, max_records: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL in chunk audit input {path} at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(item, dict):
                raise ValueError(
                    f"Invalid JSON object in chunk audit input {path} at line {line_number}"
                )
            records.extend(_flatten_loaded_object(item, source_path=str(path)))
            if max_records is not None and len(records) >= max_records:
                return records[:max_records]
    return records


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in chunk audit input {path} at line {error.lineno}: {error.msg}"
            ) from error


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _flatten_loaded_object(value: Any, source_path: str) -> list[dict[str, Any]]:
    if isinstance(value, list):
        records: list[dict[str, Any]] = []
        for item in value:
            records.extend(_flatten_loaded_object(item, source_path=source_path))
        return records

    if not isinstance(value, Mapping):
        return []

    if isinstance(value.get("chunks"), list):
        parent = dict(value)
        chunks = parent.pop("chunks")
        records = []
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                continue
            merged = dict(chunk)
            for key in ("filename", "metadata", "category", "document_id", "canonical_title", "title"):
                if key in parent and key not in merged:
                    merged[key] = parent[key]
            records.append(merged)
        return records

    return [dict(value)]


def _limit_records(records: list[dict[str, Any]], max_records: int | None) -> list[dict[str, Any]]:
    if max_records is None:
        return records
    return records[:max_records]


def _summarize_metadata_keys(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        metadata = record.get("metadata")
        if isinstance(metadata, Mapping):
            counter.update(str(key) for key in metadata.keys())
    return dict(sorted(counter.items()))


def _summarize_chunk_types(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        chunk_type = record.get("chunk_type")
        metadata = record.get("metadata")
        if chunk_type is None and isinstance(metadata, Mapping):
            chunk_type = metadata.get("chunk_type")
        counter.update([str(chunk_type) if chunk_type is not None else "<missing>"])
    return dict(sorted(counter.items()))


def _record_shape(record: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _value_shape(key, value) for key, value in sorted(record.items())}


def _value_shape(key: Any, value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        shape: dict[str, Any] = {"type": "str", "length": len(value)}
        if str(key) == "text":
            shape["redacted"] = True
        return shape
    if isinstance(value, Mapping):
        return {
            "type": "dict",
            "keys": sorted(str(item) for item in value.keys()),
            "length": len(value),
        }
    if isinstance(value, list):
        return {
            "type": "list",
            "length": len(value),
            "item_types": sorted({type(item).__name__ for item in value}),
        }
    return {"type": type(value).__name__}


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


__all__ = [
    "ChunkAuditSummary",
    "SUPPORTED_CHUNK_AUDIT_FORMATS",
    "audit_chunk_records",
    "chunk_audit_summary_to_dict",
    "detect_chunk_file_format",
    "load_chunk_like_records",
    "summarize_key_frequency",
]
