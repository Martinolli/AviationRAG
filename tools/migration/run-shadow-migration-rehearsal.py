#!/usr/bin/env python
"""Run the D.7 controlled shadow migration rehearsal."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.migration.shadow_migration_rehearsal import (  # noqa: E402
    FAIL,
    PASS,
    PASS_WITH_QUARANTINE,
    load_shadow_migration_rehearsal_config,
    run_shadow_migration_rehearsal,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run D.7 controlled shadow migration rehearsal.")
    parser.add_argument("--config", required=True, help="Local D.7 rehearsal config JSON.")
    parser.add_argument("--output-root", required=True, help="Ignored local shadow rehearsal output root.")
    parser.add_argument("--allow-local-write", action="store_true", help="Permit local shadow output writes.")
    parser.add_argument("--verify-determinism", action="store_true", help="Write run_1 and run_2 and compare bytes.")
    parser.add_argument("--verify-rollback", action="store_true", help="Run local rollback rehearsal.")
    parser.add_argument("--strict", action="store_true", help="Fail closed on blocking reconciliation/accounting issues.")
    args = parser.parse_args()

    if not args.allow_local_write:
        print("Local shadow rehearsal writes require --allow-local-write.")
        _print_exclusions()
        return 1

    try:
        config = load_shadow_migration_rehearsal_config(args.config)
        result = run_shadow_migration_rehearsal(
            config,
            args.output_root,
            allow_local_write=args.allow_local_write,
            verify_determinism=args.verify_determinism,
            verify_rollback=args.verify_rollback,
            strict=args.strict,
        )
    except Exception as error:  # noqa: BLE001 - CLI must fail closed with context.
        print(f"D.7 rehearsal failed: {error}")
        _print_exclusions()
        return 1

    _print_summary(result)
    if result.outcome == PASS:
        return 0
    if result.outcome == PASS_WITH_QUARANTINE:
        return 2
    if result.outcome == FAIL:
        return 1
    return 1


def _print_summary(result) -> None:
    print("D.7 controlled shadow migration rehearsal")
    _print_exclusions()
    print(f"Outcome: {result.outcome}")
    print(f"Structured packages: {result.package_count}")
    print(f"Structured records: {result.structured_record_count}")
    print(f"Shadow eligible: {result.eligible_count}")
    print(f"Quarantined: {result.quarantine_count}")
    print(f"Forbidden: {result.forbidden_count}")
    print(f"Rejected: {result.rejected_count}")
    print(f"Package integrity verified: {str(result.package_integrity_verified).lower()}")
    print(f"Accounting verified: {str(result.accounting_verified).lower()}")
    print(f"Determinism verification: {str(result.determinism_verified).lower()}")
    print(f"Rollback result: {str(result.rollback_verified).lower()}")
    print(f"Legacy unchanged: {str(result.legacy_unchanged).lower()}")
    print(f"Aggregate shadow digest: {result.aggregate_shadow_digest}")
    print(f"Blocking issues: {list(result.blocking_issue_codes)}")
    print("Authorized next activity: D.8 controlled migration pilot readiness review only.")
    print("Controlled migration pilot authorized: false")
    print("Production persistence authorized: false")
    print("Production retrieval authorized: false")


def _print_exclusions() -> None:
    print("Explicit exclusions:")
    print("  No parser execution.")
    print("  No runtime ingestion changes.")
    print("  No legacy deletion or overwrite.")
    print("  No embeddings generated.")
    print("  No Astra access.")
    print("  No FAISS access.")
    print("  No production retrieval activation.")
    print("  No OCR execution.")


if __name__ == "__main__":
    raise SystemExit(main())
