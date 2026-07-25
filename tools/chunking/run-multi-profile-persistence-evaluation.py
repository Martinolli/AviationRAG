#!/usr/bin/env python
"""Run the D.5d controlled multi-profile parser-output persistence evaluation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.ingestion.multi_profile_persistence_gate import (  # noqa: E402
    ACCEPTED_WITH_LIMITATIONS,
    aggregate_evaluated_profile_packages,
    load_multi_profile_config,
    multi_profile_persistence_result_to_dict,
    sanitized_multi_profile_report_bytes,
    evaluate_profile_packages,
)
from aviationrag.ingestion.persisted_chunk_package import (  # noqa: E402
    PERSISTED_CHUNKS_FILENAME,
    PERSISTENCE_MANIFEST_FILENAME,
    PERSISTENCE_REPORT_FILENAME,
    REJECTED_CANDIDATES_FILENAME,
    WARNINGS_FILENAME,
    write_persisted_chunk_package,
)
from aviationrag.ingestion.real_parser_sample_gate import real_parser_sample_gate_result_to_dict  # noqa: E402
from aviationrag.ingestion.structured_document_adapter import PASS  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("data/migration_dry_run/multi_profile_persistence")
PACKAGE_FILENAMES = (
    PERSISTED_CHUNKS_FILENAME,
    PERSISTENCE_MANIFEST_FILENAME,
    PERSISTENCE_REPORT_FILENAME,
    REJECTED_CANDIDATES_FILENAME,
    WARNINGS_FILENAME,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run D.5d controlled multi-profile parser-output persistence evaluation.",
    )
    parser.add_argument("--config", required=True, help="D.5d multi-profile config JSON.")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Ignored local output root for explicit package/report writes.",
    )
    parser.add_argument("--allow-local-write", action="store_true", help="Write ignored local outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing local evaluation outputs.")
    parser.add_argument(
        "--verify-determinism",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify deterministic package bytes when writing outputs.",
    )
    parser.add_argument(
        "--allow-reviewed-profiles",
        action="store_true",
        help="Permit configured profile REVIEW outcomes to aggregate as ACCEPTED_WITH_LIMITATIONS.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run fail-closed evaluation rules. Reviewed profiles still require --allow-reviewed-profiles.",
    )
    args = parser.parse_args()

    profiles = load_multi_profile_config(args.config)
    evaluated = evaluate_profile_packages(
        profiles,
        allow_reviewed_profiles=bool(args.allow_reviewed_profiles),
    )
    write_summary: dict[str, object] = {"local_write_requested": bool(args.allow_local_write)}
    write_determinism_ok = True
    if args.allow_local_write:
        output_root = Path(args.output_root)
        profile_summaries = {}
        for item in evaluated:
            profile_root = _child_file(output_root, item.profile.profile_key)
            run_1 = _child_file(profile_root, "run_1")
            run_2 = _child_file(profile_root, "run_2")
            write_persisted_chunk_package(
                item.package,
                run_1,
                allow_local_write=True,
                overwrite=args.overwrite,
            )
            write_persisted_chunk_package(
                item.repeated_package,
                run_2,
                allow_local_write=True,
                overwrite=args.overwrite,
            )
            comparison = _compare_package_dirs(run_1, run_2) if args.verify_determinism else {}
            profile_ok = (
                item.package.package_digest == item.repeated_package.package_digest
                and all(entry["bytes_match"] and entry["sha256_match"] for entry in comparison.values())
            )
            if args.verify_determinism and not profile_ok:
                write_determinism_ok = False
            _write_local_gate_report(
                _child_file(profile_root, "local_gate_report.json"),
                item,
                {
                    "package_digest_run_1": item.package.package_digest,
                    "package_digest_run_2": item.repeated_package.package_digest,
                    "package_digest_match": item.package.package_digest == item.repeated_package.package_digest,
                    "determinism_verified": profile_ok,
                    "file_comparison": comparison,
                },
                overwrite=args.overwrite,
            )
            profile_summaries[item.profile.profile_key] = {
                "determinism_verified": profile_ok,
                "package_digest_run_1": item.package.package_digest,
                "package_digest_run_2": item.repeated_package.package_digest,
            }
        write_summary = {
            "local_write_requested": True,
            "determinism_verified": write_determinism_ok,
            "profiles": profile_summaries,
        }

    result = aggregate_evaluated_profile_packages(
        evaluated,
        allow_reviewed_profiles=bool(args.allow_reviewed_profiles),
        write_determinism_verified=write_determinism_ok if args.allow_local_write and args.verify_determinism else None,
    )
    if args.allow_local_write:
        _write_aggregate_report(
            _child_file(Path(args.output_root), "multi_profile_evaluation_report.json"),
            result,
            write_summary,
            overwrite=args.overwrite,
        )

    _print_summary(result, write_summary)
    if result.outcome == PASS:
        return 0
    if result.outcome == ACCEPTED_WITH_LIMITATIONS:
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
    item,
    local_write: dict[str, object],
    *,
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError("Local gate report exists; pass --overwrite.")
    report = real_parser_sample_gate_result_to_dict(item.gate_result)
    report["profile_key"] = item.profile.profile_key
    report["profile_role"] = item.profile.profile_role
    report["candidate_context_ids"] = sorted(item.profile.candidate_contexts)
    report["local_write"] = local_write
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_aggregate_report(
    path: Path,
    result,
    local_write: dict[str, object],
    *,
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError("Aggregate report exists; pass --overwrite.")
    data = multi_profile_persistence_result_to_dict(result)
    data["local_write"] = local_write
    path.write_bytes((json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    # Exercise the public deterministic serializer in the CLI path too.
    sanitized_multi_profile_report_bytes(result)


def _child_file(root: Path, name: str) -> Path:
    resolved_root = root.resolve()
    child = (root / name).resolve()
    try:
        child.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Refusing to write outside output root: {child}") from exc
    return child


def _print_summary(result, write_summary: dict[str, object]) -> None:
    print("D.5d multi-profile parser-output persistence evaluation")
    print("Controlled multi-profile evaluation only.")
    print("No runtime ingestion.")
    print("No production persistence.")
    print("No embeddings.")
    print("No Astra.")
    print("No FAISS.")
    print("No full-corpus migration authorization.")
    print(f"Aggregate outcome: {result.outcome}")
    print(f"Profile count: {result.profile_count}")
    print(f"Total candidates: {result.total_candidate_count}")
    print(f"Total accepted: {result.total_accepted_record_count}")
    print(f"Total rejected: {result.total_rejected_candidate_count}")
    print(f"Total warnings: {result.total_warning_count}")
    print(f"Total review required: {result.total_review_required_count}")
    print(f"Cross-document chunk-ID collisions: {result.cross_document_chunk_id_collision_count}")
    print(f"Schema consistency verified: {str(result.schema_consistency_verified).lower()}")
    print(f"Determinism verified: {str(result.determinism_verified).lower()}")
    print(f"Blocking issue codes: {list(result.blocking_issue_codes)}")
    for key, outcome in sorted(result.profile_outcomes.items()):
        print(f"Profile {key}: {outcome}")
    if write_summary.get("local_write_requested"):
        print(f"Byte/hash determinism verified: {str(write_summary.get('determinism_verified')).lower()}")
        print("Local aggregate report written.")
    else:
        print("Local outputs not written.")


if __name__ == "__main__":
    raise SystemExit(main())
