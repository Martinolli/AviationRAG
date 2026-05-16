"""Gated local chunk conversion writer.

This module writes converted metadata-rich chunks, vector payload-shaped
dictionaries, and a summary report to an explicit local output directory. It is
for fake/sample or explicitly approved local inputs only. It does not generate
embeddings, connect to Astra, use FAISS, write runtime ingestion outputs, or
integrate with legacy ingestion scripts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from aviationrag.config import (
    CHUNK_MIGRATION_DRY_RUN_ENV,
    CHUNK_MIGRATION_ENV,
    get_chunk_migration_settings,
)
from aviationrag.ingestion.chunk_audit import load_chunk_like_records
from aviationrag.ingestion.chunk_legacy_adapter import preview_legacy_chunk_migration
from aviationrag.ingestion.chunk_migration_dry_run import (
    chunk_migration_dry_run_result_to_dict,
    run_chunk_migration_dry_run,
)
from aviationrag.ingestion.chunk_payload import validate_vector_payload_dataset
from aviationrag.ingestion.chunk_schema import (
    chunk_record_to_dict,
    validate_chunk_dataset,
)


CHUNK_OUTPUT_FILENAME = "converted_chunks.jsonl"
PAYLOAD_OUTPUT_FILENAME = "vector_payloads.jsonl"
REPORT_OUTPUT_FILENAME = "conversion_report.json"


@dataclass
class ChunkConversionWriteResult:
    """Result for an explicitly permitted local conversion write."""

    source_path: str
    output_dir: str
    chunk_output_path: str
    payload_output_path: str
    report_output_path: str
    chunk_count: int
    payload_count: int
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write JSONL records with stable key order and a trailing newline."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), sort_keys=True) + "\n")


def run_local_chunk_conversion_write(
    input_path: str | Path,
    output_dir: str | Path,
    allow_local_write: bool = False,
    max_records: int | None = None,
) -> ChunkConversionWriteResult:
    """Write converted chunks and payloads to an explicitly approved local path."""
    settings = get_chunk_migration_settings()
    if not allow_local_write and not settings.enabled:
        raise PermissionError(
            "Local chunk conversion writes are disabled. Pass allow_local_write=True "
            "or set AVIATIONRAG_ENABLE_CHUNK_MIGRATION=true for an explicit local run."
        )

    source_path = Path(input_path)
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    chunk_output_path = _child_path(resolved_output_dir, CHUNK_OUTPUT_FILENAME)
    payload_output_path = _child_path(resolved_output_dir, PAYLOAD_OUTPUT_FILENAME)
    report_output_path = _child_path(resolved_output_dir, REPORT_OUTPUT_FILENAME)

    dry_run = run_chunk_migration_dry_run(source_path, max_records=max_records)
    records = load_chunk_like_records(source_path, max_records=max_records)
    preview = preview_legacy_chunk_migration(
        records,
        include_payloads=True,
        env={
            CHUNK_MIGRATION_ENV: "true",
            CHUNK_MIGRATION_DRY_RUN_ENV: "true",
        },
    )

    converted_chunks = [chunk_record_to_dict(chunk) for chunk in preview.chunks]
    payloads = list(preview.payloads)
    issues = _dedupe(
        list(dry_run.issues)
        + [f"conversion: {issue}" for issue in preview.issues]
        + [f"chunk_validation: {issue}" for issue in validate_chunk_dataset(converted_chunks)]
        + [f"payload_validation: {issue}" for issue in validate_vector_payload_dataset(payloads)]
    )
    warnings = _dedupe(
        list(dry_run.warnings)
        + [f"conversion: {warning}" for warning in preview.warnings]
        + ["Local conversion write was explicitly allowed; outputs are ignored/local-only."]
    )

    report = {
        "dry_run": chunk_migration_dry_run_result_to_dict(dry_run),
        "source_path": str(source_path),
        "output_dir": str(resolved_output_dir),
        "chunk_output_path": str(chunk_output_path),
        "payload_output_path": str(payload_output_path),
        "report_output_path": str(report_output_path),
        "chunk_count": len(converted_chunks),
        "payload_count": len(payloads),
        "issue_count": len(issues),
        "warning_count": len(warnings),
        "outputs_are_local_only": True,
        "embeddings_generated": False,
        "astra_touched": False,
        "faiss_touched": False,
        "runtime_ingestion_modified": False,
    }

    result = ChunkConversionWriteResult(
        source_path=str(source_path),
        output_dir=str(resolved_output_dir),
        chunk_output_path=str(chunk_output_path),
        payload_output_path=str(payload_output_path),
        report_output_path=str(report_output_path),
        chunk_count=len(converted_chunks),
        payload_count=len(payloads),
        issues=issues,
        warnings=warnings,
        summary=report,
    )

    write_jsonl(chunk_output_path, converted_chunks)
    write_jsonl(payload_output_path, payloads)
    report_output_path.write_text(
        json.dumps(chunk_conversion_write_result_to_dict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return result


def chunk_conversion_write_result_to_dict(
    result: ChunkConversionWriteResult,
) -> dict[str, Any]:
    """Return a JSON-serializable conversion write result dictionary."""
    return asdict(result)


def _child_path(output_dir: Path, filename: str) -> Path:
    output_root = output_dir.resolve()
    child = (output_dir / filename).resolve()
    if child.parent != output_root:
        raise ValueError(f"Refusing to write outside output_dir: {child}")
    return child


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
    "CHUNK_OUTPUT_FILENAME",
    "PAYLOAD_OUTPUT_FILENAME",
    "REPORT_OUTPUT_FILENAME",
    "ChunkConversionWriteResult",
    "chunk_conversion_write_result_to_dict",
    "run_local_chunk_conversion_write",
    "write_jsonl",
]
