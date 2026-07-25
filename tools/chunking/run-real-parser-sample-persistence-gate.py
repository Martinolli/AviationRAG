#!/usr/bin/env python
"""Run the D.5c controlled real parser-output sample persistence gate."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.ingestion.persisted_chunk_package import (  # noqa: E402
    PERSISTED_CHUNKS_FILENAME,
    PERSISTENCE_MANIFEST_FILENAME,
    PERSISTENCE_REPORT_FILENAME,
    REJECTED_CANDIDATES_FILENAME,
    WARNINGS_FILENAME,
    build_package_from_adapter_result,
    write_persisted_chunk_package,
)
from aviationrag.ingestion.real_parser_sample_gate import (  # noqa: E402
    real_parser_sample_gate_result_to_dict,
    run_real_parser_sample_gate,
)
from aviationrag.ingestion.structured_document_adapter import (  # noqa: E402
    PASS,
    REVIEW,
    run_structured_document_adapter,
)


DEFAULT_OUTPUT_ROOT = Path("data/migration_dry_run/real_parser_sample/faa_order_4040_26b")
PACKAGE_FILENAMES = (
    PERSISTED_CHUNKS_FILENAME,
    PERSISTENCE_MANIFEST_FILENAME,
    PERSISTENCE_REPORT_FILENAME,
    REJECTED_CANDIDATES_FILENAME,
    WARNINGS_FILENAME,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run D.5c controlled real parser-output sample persistence gate.",
    )
    parser.add_argument("--artifact", required=True, help="StructuredDocument artifact path.")
    parser.add_argument("--manifest", required=True, help="Parser manifest path.")
    parser.add_argument("--source", required=True, help="Exact source PDF path.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Ignored local output root for explicit package writes.",
    )
    parser.add_argument("--allow-local-write", action="store_true", help="Write ignored local outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing local gate outputs.")
    parser.add_argument(
        "--approve-adapter-warning",
        action="append",
        default=[],
        help="Explicit adapter warning code approval.",
    )
    parser.add_argument("--allow-review", action="store_true", help="Permit documented REVIEW outcome.")
    parser.add_argument("--strict", action="store_true", help="Fail closed on review outcomes.")
    parser.add_argument(
        "--verify-determinism",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify deterministic package generation.",
    )
    args = parser.parse_args()

    allow_review = bool(args.allow_review and not args.strict)
    result = run_real_parser_sample_gate(
        artifact_path=args.artifact,
        manifest_path=args.manifest,
        source_path=args.source,
        approved_adapter_warning_codes=args.approve_adapter_warning,
        allow_review=allow_review,
    )
    write_summary: dict[str, object] = {"local_write_requested": bool(args.allow_local_write)}
    write_determinism_ok = True
    if args.allow_local_write:
        output_root = Path(args.output_root)
        run_1 = _child_dir(output_root, "run_1")
        run_2 = _child_dir(output_root, "run_2")
        report_path = _child_file(output_root, "local_gate_report.json")
        adapter_result = run_structured_document_adapter(
            args.artifact,
            args.manifest,
            source_path=args.source,
            approved_warning_codes=args.approve_adapter_warning,
            strict_warnings=False,
        )
        package_1 = build_package_from_adapter_result(adapter_result)
        package_2 = build_package_from_adapter_result(adapter_result)
        write_persisted_chunk_package(
            package_1,
            run_1,
            allow_local_write=True,
            overwrite=args.overwrite,
        )
        write_persisted_chunk_package(
            package_2,
            run_2,
            allow_local_write=True,
            overwrite=args.overwrite,
        )
        comparison = _compare_package_dirs(run_1, run_2) if args.verify_determinism else {}
        write_determinism_ok = bool(all(item["bytes_match"] and item["sha256_match"] for item in comparison.values()))
        if args.verify_determinism and package_1.package_digest != package_2.package_digest:
            write_determinism_ok = False
        write_summary = {
            "local_write_requested": True,
            "package_digest_run_1": package_1.package_digest,
            "package_digest_run_2": package_2.package_digest,
            "package_digest_match": package_1.package_digest == package_2.package_digest,
            "determinism_verified": write_determinism_ok,
            "file_comparison": comparison,
        }
        _write_local_gate_report(report_path, result, write_summary, overwrite=args.overwrite)

    _print_summary(result, write_summary)
    if not write_determinism_ok:
        return 1
    if result.outcome == PASS:
        return 0
    if result.outcome == REVIEW:
        return 2
    return 1


def _compare_package_dirs(run_1: Path, run_2: Path) -> dict[str, dict[str, object]]:
    comparison: dict[str, dict[str, object]] = {}
    for filename in PACKAGE_FILENAMES:
        left = run_1 / filename
        right = run_2 / filename
        left_bytes = left.read_bytes()
        right_bytes = right.read_bytes()
        left_hash = sha256(left_bytes).hexdigest()
        right_hash = sha256(right_bytes).hexdigest()
        comparison[filename] = {
            "bytes_match": left_bytes == right_bytes,
            "sha256_match": left_hash == right_hash,
            "sha256": left_hash if left_hash == right_hash else None,
        }
    return comparison


def _write_local_gate_report(
    path: Path,
    result,
    write_summary: dict[str, object],
    *,
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError("Local gate report exists; pass --overwrite.")
    report = real_parser_sample_gate_result_to_dict(result)
    report["local_write"] = write_summary
    content = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(content, encoding="utf-8")


def _child_dir(root: Path, name: str) -> Path:
    child = _child_file(root, name)
    return child


def _child_file(root: Path, name: str) -> Path:
    resolved_root = root.resolve()
    child = (root / name).resolve()
    try:
        child.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to write outside output root: {child}") from exc
    return child


def _print_summary(result, write_summary: dict[str, object]) -> None:
    print("D.5c real parser-output sample persistence gate")
    print("Controlled real parser-output sample only.")
    print("No runtime ingestion.")
    print("No embeddings.")
    print("No Astra.")
    print("No FAISS.")
    print("No production migration authorization.")
    print(f"Gate outcome: {result.outcome}")
    print(f"Document key: {result.document_key}")
    print(f"Source filename: {result.source_filename}")
    print(f"Source checksum: {result.source_checksum}")
    print(f"Parser: {result.parser_name} / {result.parser_version}")
    print(f"StructuredDocument schema: {result.structured_document_schema_version}")
    print(f"Adapter outcome: {result.adapter_outcome}")
    print(f"Package outcome: {result.package_outcome}")
    print(f"Input candidates: {result.input_candidate_count}")
    print(f"Accepted records: {result.accepted_record_count}")
    print(f"Rejected candidates: {result.rejected_candidate_count}")
    print(f"Warnings: {result.warning_count}")
    print(f"Review-required count: {result.review_required_count}")
    print(f"Validation-status counts: {dict(result.validation_status_counts)}")
    print(f"Provenance counts: {dict(result.provenance_counts)}")
    print(f"Content-type counts: {dict(result.content_type_counts)}")
    print(f"Accepted limitation counts: {dict(result.accepted_limitation_counts)}")
    print(f"Package digest: {result.package_digest}")
    print(f"In-memory determinism verified: {str(result.determinism_verified).lower()}")
    print(f"Blocking issue codes: {list(result.blocking_issue_codes)}")
    if write_summary.get("local_write_requested"):
        print(f"Run-1 package digest: {write_summary.get('package_digest_run_1')}")
        print(f"Run-2 package digest: {write_summary.get('package_digest_run_2')}")
        print(f"Byte/hash determinism verified: {str(write_summary.get('determinism_verified')).lower()}")
        print("Local gate report written.")
    else:
        print("Local outputs not written.")


if __name__ == "__main__":
    raise SystemExit(main())
