#!/usr/bin/env python
"""Run a D.5b persisted chunk package dry run from a D.4c artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.ingestion.persisted_chunk_mapper import (  # noqa: E402
    PersistedChunkCandidateContext,
    PersistedChunkMappingPolicy,
)
from aviationrag.ingestion.persisted_chunk_package import (  # noqa: E402
    DEFAULT_PERSISTED_PACKAGE_OUTPUT_DIR,
    build_package_from_adapter_result,
    write_persisted_chunk_package,
)
from aviationrag.ingestion.structured_document_adapter import (  # noqa: E402
    FAIL,
    PASS,
    REVIEW,
    run_structured_document_adapter,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run D.5b synthetic/local persisted chunk package dry run.",
    )
    parser.add_argument("--artifact", required=True, help="StructuredDocument artifact path.")
    parser.add_argument("--manifest", required=True, help="Parser manifest path.")
    parser.add_argument("--source", help="Optional exact source bytes path.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_PERSISTED_PACKAGE_OUTPUT_DIR),
        help="Output directory for explicit local writes.",
    )
    parser.add_argument("--allow-local-write", action="store_true", help="Write ignored local package files.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing package files.")
    parser.add_argument("--approve-warning", action="append", default=[], help="Adapter warning code approval.")
    parser.add_argument("--allow-partial-provenance", action="store_true", help="Allow governed partial provenance.")
    parser.add_argument(
        "--allow-review-records",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow review-required records in local package.",
    )
    parser.add_argument("--allow-rejected-candidates", action="store_true", help="Allow package continuation with rejected candidates.")
    parser.add_argument("--include-headings", action="store_true", help="Include heading candidates in D.4c adapter and persisted mapping.")
    parser.add_argument("--strict", action="store_true", help="Strict adapter warnings and package policy.")
    args = parser.parse_args()

    adapter_result = run_structured_document_adapter(
        args.artifact,
        args.manifest,
        source_path=args.source,
        approved_warning_codes=args.approve_warning,
        include_headings=args.include_headings,
        strict_warnings=args.strict,
    )
    policy = PersistedChunkMappingPolicy(
        allow_partial_provenance=args.allow_partial_provenance,
        allow_review_required_records=args.allow_review_records and not args.strict,
        include_heading_records=args.include_headings,
    )
    default_context = _default_context(adapter_result, args.allow_partial_provenance)
    package = build_package_from_adapter_result(
        adapter_result,
        candidate_contexts=default_context,
        policy=policy,
        allow_rejected_candidates=args.allow_rejected_candidates and not args.strict,
    )

    write_result = None
    if args.allow_local_write:
        write_result = write_persisted_chunk_package(
            package,
            args.output_dir,
            allow_local_write=True,
            overwrite=args.overwrite,
        )

    _print_summary(adapter_result, package, write_result, args.allow_local_write)
    if package.outcome == PASS:
        return 0
    if package.outcome == REVIEW:
        return 2
    return 1


def _default_context(adapter_result, allow_partial_provenance: bool):
    if not allow_partial_provenance:
        return {}
    return {
        candidate.chunk_candidate_id: PersistedChunkCandidateContext(
            accepted_limitation_codes=("CHUNK_SECTION_CROSSING_REVIEW",),
            warning_codes=("PARTIAL_PROVENANCE_REVIEW_REQUIRED",),
            review_required=True,
        )
        for candidate in adapter_result.candidates
        if candidate.provenance_status == "structured_partial"
    }


def _print_summary(adapter_result, package, write_result, local_write_allowed: bool) -> None:
    print("D.5b persisted chunk package dry run")
    print("Local synthetic persistence dry run only.")
    print("Runtime ingestion modified: false")
    print("Real corpus processed: false")
    print("Embeddings generated: false")
    print("Astra touched: false")
    print("FAISS touched: false")
    print(f"Adapter outcome: {adapter_result.outcome}")
    print(f"Package outcome: {package.outcome}")
    print(f"Input candidates: {len(adapter_result.candidates)}")
    print(f"Accepted records: {len(package.records)}")
    print(f"Rejected candidates: {len(package.rejected_candidates)}")
    print(f"Warnings: {len(package.warnings)}")
    print(f"Issues: {len(package.issues)}")
    print(f"Validation-status counts: {package.report.get('validation_status_counts', {})}")
    print(f"Provenance counts: {package.report.get('provenance_counts', {})}")
    print(f"Content-type counts: {package.report.get('content_type_counts', {})}")
    print(f"Limitation counts: {package.report.get('limitation_counts', {})}")
    print(f"Review-required count: {package.report.get('review_required_count', 0)}")
    print(f"Package digest: {package.package_digest}")
    if write_result is None:
        print("Package outputs not written.")
    else:
        print(f"Package outputs written: {write_result.output_dir}")
    if not local_write_allowed:
        print("Use --allow-local-write to write ignored local package artifacts.")


if __name__ == "__main__":
    raise SystemExit(main())
