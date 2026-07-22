#!/usr/bin/env python
"""Run the structured-document parser-output adapter offline."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aviationrag.ingestion.structured_document_adapter import (  # noqa: E402
    DEFAULT_ADAPTER_OUTPUT_DIR,
    FAIL,
    PASS,
    REVIEW,
    run_structured_document_adapter,
    write_structured_document_adapter_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, help="Structured-document JSON artifact.")
    parser.add_argument("--manifest", required=True, help="techdoc-parser export manifest JSON.")
    parser.add_argument(
        "--source",
        default=None,
        help="Optional original source bytes used to verify manifest source_sha256.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_ADAPTER_OUTPUT_DIR),
        help="Local-only output directory used only with --allow-local-write.",
    )
    parser.add_argument(
        "--allow-local-write",
        action="store_true",
        help="Allow writing local dry-run outputs under --output-dir.",
    )
    parser.add_argument(
        "--approve-warning",
        action="append",
        default=[],
        help="Approve one validator warning code for REVIEW instead of FAIL. Repeatable.",
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Treat every validator warning as a failing adapter issue.",
    )
    parser.add_argument(
        "--include-headings",
        action="store_true",
        help="Include section and appendix heading blocks as section candidates.",
    )
    args = parser.parse_args()

    artifact_path = _project_path(args.artifact)
    manifest_path = _project_path(args.manifest)
    source_path = _project_path(args.source) if args.source else None
    output_dir = _project_path(args.output_dir)

    print("Structured-document adapter dry run is offline/report-only.")
    print("No runtime ingestion, embeddings, Astra, FAISS, or real corpus migration is performed.")

    result = run_structured_document_adapter(
        artifact_path,
        manifest_path,
        source_path=source_path,
        approved_warning_codes=args.approve_warning,
        include_headings=args.include_headings,
        strict_warnings=args.strict_warnings,
    )

    print(f"Artifact: {artifact_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Source: {source_path if source_path is not None else 'not provided'}")
    print(f"Outcome: {result.outcome}")
    print(f"Schema name: {result.schema_name}")
    print(f"Schema version: {result.schema_version}")
    print(f"Document ID: {result.document_id}")
    print(f"Artifact checksum matches: {result.artifact_integrity.artifact_checksum_matches}")
    print(f"Source checksum matches: {result.artifact_integrity.source_checksum_matches}")
    print(f"Manifest matches artifact: {result.artifact_integrity.manifest_matches_artifact}")
    print(f"Validator errors: {result.validator_result.error_count}")
    print(f"Validator warnings: {result.validator_result.warning_count}")
    print(f"Candidate count: {len(result.candidates)}")

    for key in (
        "content_type_counts",
        "provenance_status_counts",
        "reference_status_counts",
        "table_entity_count",
        "table_candidate_count",
        "figure_entity_count",
        "figure_caption_candidate_count",
        "equation_entity_count",
        "admonition_entity_count",
    ):
        print(f"{key}: {result.summary.get(key)}")

    for issue in result.issues:
        entity = f" entity={issue.entity_id}" if issue.entity_id else ""
        print(f"{issue.severity.upper()} {issue.code} {issue.path}{entity}: {issue.message}")

    if args.allow_local_write:
        write_result = write_structured_document_adapter_outputs(
            result,
            output_dir,
            allow_local_write=True,
        )
        print(f"Candidates written: {write_result.candidates_output_path}")
        print(f"Report written: {write_result.report_output_path}")
        print(f"Integrity written: {write_result.integrity_output_path}")
    else:
        print("Adapter outputs not written. Re-run with --allow-local-write to write local dry-run files.")

    if result.outcome == PASS:
        return 0
    if result.outcome == REVIEW:
        return 2
    if result.outcome == FAIL:
        return 1
    return 1


def _project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
