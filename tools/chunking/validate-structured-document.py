#!/usr/bin/env python
"""Validate a synthetic structured-document fixture offline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aviationrag.ingestion.structured_document_validator import (  # noqa: E402
    structured_document_validation_result_to_dict,
    validate_structured_document_file,
)


DEFAULT_INPUT = PROJECT_ROOT / "data" / "sample_documents" / "sample_structured_document.json"
DEFAULT_REPORT = PROJECT_ROOT / "logs" / "chunking" / "structured_document_validation.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Structured-document JSON file. Defaults to the synthetic D.4 fixture.",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        help="Explicit report path used only with --allow-report-write.",
    )
    parser.add_argument(
        "--allow-report-write",
        action="store_true",
        help="Allow writing the validation report to the requested path.",
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Exit non-zero when validation warnings are present.",
    )
    args = parser.parse_args()

    input_path = _project_path(args.input)
    report_path = _project_path(args.report)

    print("Structured-document validation is synthetic/offline only.")
    print("No parsing, migration, embeddings, Astra, FAISS, or runtime ingestion work is performed.")

    result = validate_structured_document_file(input_path)
    report = structured_document_validation_result_to_dict(result)

    print(f"Source path: {input_path}")
    print(f"Schema name: {result.schema_name}")
    print(f"Schema version: {result.schema_version}")
    print(f"Document ID: {result.document_id}")
    print(f"Valid: {result.is_valid}")
    print(f"Error count: {result.error_count}")
    print(f"Warning count: {result.warning_count}")
    for key, value in result.summary.items():
        print(f"{key}: {value}")

    for issue in result.issues:
        entity = f" entity={issue.entity_id}" if issue.entity_id else ""
        print(f"{issue.severity.upper()} {issue.code} {issue.path}{entity}: {issue.message}")

    if args.allow_report_write:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Report written: {report_path}")
    else:
        print("Report not written. Re-run with --allow-report-write to write the report.")

    if result.error_count:
        return 1
    if args.strict_warnings and result.warning_count:
        return 1
    return 0


def _project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
