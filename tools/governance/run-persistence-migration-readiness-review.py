#!/usr/bin/env python
"""Run the D.6 persistence migration readiness governance review."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.governance.persistence_migration_readiness import (  # noqa: E402
    CONDITIONALLY_READY_FOR_CONTROLLED_MIGRATION_REHEARSAL,
    NOT_READY,
    READY_FOR_CONTROLLED_MIGRATION_REHEARSAL,
    decision_report_json_bytes,
    decision_report_markdown,
    evaluate_persistence_migration_readiness,
    load_persistence_governance_policy,
    load_persistence_migration_readiness_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run D.6 persistence migration readiness review.")
    parser.add_argument("--evidence", required=True, help="Sanitized D.6 readiness evidence JSON.")
    parser.add_argument("--policy", required=True, help="D.6 governance policy JSON.")
    parser.add_argument("--report-json", help="Optional deterministic JSON report path.")
    parser.add_argument("--report-markdown", help="Optional deterministic Markdown report path.")
    parser.add_argument("--allow-report-write", action="store_true", help="Permit report writes.")
    parser.add_argument("--strict", action="store_true", help="Fail closed on validation errors.")
    args = parser.parse_args()

    if (args.report_json or args.report_markdown) and not args.allow_report_write:
        print("Report writing requires --allow-report-write.")
        _print_exclusions()
        return 1

    evidence = load_persistence_migration_readiness_evidence(args.evidence)
    policy = load_persistence_governance_policy(args.policy)
    decision = evaluate_persistence_migration_readiness(evidence, policy=policy)

    if args.allow_report_write:
        if args.report_json:
            _write_bytes(Path(args.report_json), decision_report_json_bytes(decision))
        if args.report_markdown:
            _write_text(Path(args.report_markdown), decision_report_markdown(decision))

    _print_summary(decision)
    if decision.decision == READY_FOR_CONTROLLED_MIGRATION_REHEARSAL:
        return 0
    if decision.decision == CONDITIONALLY_READY_FOR_CONTROLLED_MIGRATION_REHEARSAL:
        return 2
    if decision.decision == NOT_READY:
        return 1
    return 1


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _print_summary(decision) -> None:
    print("D.6 persistence migration readiness review")
    _print_exclusions()
    print(f"Decision: {decision.decision}")
    print(f"Controlled rehearsal authorized: {str(decision.controlled_rehearsal_authorized).lower()}")
    print(f"Controlled pilot authorized: {str(decision.controlled_pilot_authorized).lower()}")
    print(f"Production persistence authorized: {str(decision.production_persistence_authorized).lower()}")
    print(f"Production indexing authorized: {str(decision.production_indexing_authorized).lower()}")
    print(f"Production retrieval authorized: {str(decision.production_retrieval_authorized).lower()}")
    print(f"Satisfied technical gates: {list(decision.satisfied_gate_codes)}")
    print(f"Conditional governance gates: {list(decision.conditional_gate_codes)}")
    print(f"Blocking gates: {list(decision.blocking_gate_codes)}")
    print(f"Required controls: {list(decision.required_controls)}")
    print("Record-status policy:")
    for status, policy in decision.summary.get("record_status_policy", {}).items():
        print(f"  {status}: {policy}")
    print("Provenance policy:")
    for status, policy in decision.summary.get("provenance_policy", {}).items():
        print(f"  {status}: {policy}")
    print("Authorized activities:")
    print("  controlled_shadow_migration_rehearsal: conditional")
    print("  controlled_local_persistence_package_generation: true")
    print("Prohibited activities:")
    print("  production_persistence: false")
    print("  production_indexing: false")
    print("  production_retrieval: false")
    print("  embeddings: false")
    print("  astra: false")
    print("  faiss: false")


def _print_exclusions() -> None:
    print("Governance review only.")
    print("No migration executed.")
    print("No runtime ingestion changed.")
    print("No embeddings generated.")
    print("No Astra access.")
    print("No FAISS access.")
    print("No production authorization.")


if __name__ == "__main__":
    raise SystemExit(main())
