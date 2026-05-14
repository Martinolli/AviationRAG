"""Write the fake sample manifest to the ignored local manifest path.

This developer utility verifies that the future private manifest location can be
written and read with fake sample records only. It does not scan source
documents, call ingestion scripts, generate embeddings, or access Astra/FAISS.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
SAMPLE_MANIFEST = PROJECT_ROOT / "data" / "sample_documents" / "sample_manifest.jsonl"
LOCAL_MANIFEST = PROJECT_ROOT / "data" / "manifest" / "documents.jsonl"

sys.path.insert(0, str(SRC_DIR))

from aviationrag.ingestion.manifest import (  # noqa: E402
    read_manifest,
    validate_manifest_record,
    write_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write fake sample manifest records to the ignored local manifest path."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing local manifest file. Use only for fake dry-run data.",
    )
    args = parser.parse_args()

    if not SAMPLE_MANIFEST.exists():
        print(f"ERROR: sample manifest fixture missing: {SAMPLE_MANIFEST}", file=sys.stderr)
        return 1

    if LOCAL_MANIFEST.exists() and not args.force:
        print(
            "ERROR: local manifest already exists. Refusing to overwrite without --force: "
            f"{LOCAL_MANIFEST}",
            file=sys.stderr,
        )
        return 1

    records = read_manifest(SAMPLE_MANIFEST)
    if not records:
        print(f"ERROR: no records loaded from sample manifest: {SAMPLE_MANIFEST}", file=sys.stderr)
        return 1

    issues = _validation_issues(records)
    if issues:
        print("ERROR: sample records failed manifest validation:", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        return 1

    write_manifest(LOCAL_MANIFEST, records)
    round_trip = read_manifest(LOCAL_MANIFEST)

    if len(round_trip) != len(records):
        print(
            "ERROR: write/read mismatch: "
            f"loaded={len(records)} read_back={len(round_trip)}",
            file=sys.stderr,
        )
        return 1

    round_trip_issues = _validation_issues(round_trip)
    if round_trip_issues:
        print("ERROR: written records failed manifest validation:", file=sys.stderr)
        for issue in round_trip_issues:
            print(f"- {issue}", file=sys.stderr)
        return 1

    print("Local sample manifest dry run completed.")
    print(f"Records loaded: {len(records)}")
    print(f"Records written: {len(records)}")
    print(f"Records read back: {len(round_trip)}")
    print(f"Input path: {SAMPLE_MANIFEST.relative_to(PROJECT_ROOT)}")
    print(f"Output path: {LOCAL_MANIFEST.relative_to(PROJECT_ROOT)}")
    print("Output is ignored/local-only and must not be committed.")
    print("Data source: fake sample fixture only; runtime ingestion is not integrated.")

    if _is_git_ignored(LOCAL_MANIFEST):
        print("Git ignore check: ignored")
    else:
        print(
            "WARNING: output path is not ignored by Git. Check .gitignore before committing.",
            file=sys.stderr,
        )
        return 1

    return 0


def _validation_issues(records) -> list[str]:
    issues: list[str] = []
    for record in records:
        for issue in validate_manifest_record(record):
            issues.append(f"{record.document_id}: {issue}")
    return issues


def _is_git_ignored(path: Path) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


if __name__ == "__main__":
    raise SystemExit(main())
