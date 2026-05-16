#!/usr/bin/env python
"""Run a read-only audit of one explicit chunk-like file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aviationrag.ingestion.chunk_audit import (  # noqa: E402
    audit_chunk_records,
    chunk_audit_summary_to_dict,
    load_chunk_like_records,
)


DEFAULT_INPUT = PROJECT_ROOT / "data" / "sample_documents" / "sample_chunks.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "logs" / "chunking" / "legacy_chunk_audit.json"
SAMPLE_DOCUMENTS_DIR = PROJECT_ROOT / "data" / "sample_documents"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Explicit chunk-like file to audit. Defaults to fake sample chunk fixture.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum number of records to load from the explicit input file.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSON report path. Defaults to ignored logs/chunking/legacy_chunk_audit.json.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    warnings: list[str] = []
    try:
        input_path.resolve().relative_to(SAMPLE_DOCUMENTS_DIR.resolve())
    except ValueError:
        warnings.append(
            "Input path is outside data/sample_documents; ensure it was explicitly approved "
            "and does not contain private text that should be committed."
        )

    records = load_chunk_like_records(input_path, max_records=args.max_records)
    summary = audit_chunk_records(records, source_path=str(input_path))
    summary.warnings.extend(warnings)
    report = chunk_audit_summary_to_dict(summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Legacy chunk audit completed.")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Records: {summary.record_count}")
    print(f"Detected format: {summary.detected_format}")
    print(f"Missing text: {summary.missing_text_count}")
    print(f"Missing chunk_id: {summary.missing_chunk_id_count}")
    print(f"Missing document_id: {summary.missing_document_id_count}")
    print("Report redacts string values and summarizes text by type/length only.")
    if warnings:
        for warning in warnings:
            print(f"Warning: {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
