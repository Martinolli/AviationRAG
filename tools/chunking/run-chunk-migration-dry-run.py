#!/usr/bin/env python
"""Run a fake/local chunk migration dry run for one explicit chunk-like file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aviationrag.ingestion.chunk_migration_dry_run import (  # noqa: E402
    chunk_migration_dry_run_result_to_dict,
    run_chunk_migration_dry_run,
)


DEFAULT_INPUT = PROJECT_ROOT / "data" / "sample_documents" / "sample_chunks.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "logs" / "chunking" / "chunk_migration_dry_run.json"
SAMPLE_DOCUMENTS_DIR = PROJECT_ROOT / "data" / "sample_documents"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Explicit chunk-like file. Defaults to fake sample chunk fixture.",
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
        help="JSON report path. Defaults to ignored logs/chunking/chunk_migration_dry_run.json.",
    )
    args = parser.parse_args()

    input_path = _project_path(args.input)
    output_path = _project_path(args.output)

    outside_sample_warning = _outside_sample_warning(input_path)
    if outside_sample_warning:
        print(f"Warning: {outside_sample_warning}")

    result = run_chunk_migration_dry_run(input_path, max_records=args.max_records)
    if outside_sample_warning:
        result.warnings.append(outside_sample_warning)
        result.summary = {
            **result.summary,
            "warning_count": len(result.warnings),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(chunk_migration_dry_run_result_to_dict(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Chunk migration dry run completed.")
    print(f"Source path: {input_path}")
    print(f"Chunk count: {result.chunk_count}")
    print(f"Payload count: {result.payload_count}")
    print(f"Issue count: {len(result.issues)}")
    print(f"Warning count: {len(result.warnings)}")
    print(f"Output path: {output_path}")
    print("No migrated chunks, embeddings, vector indexes, Astra writes, or FAISS outputs were created.")
    print("Warning: output is generated, local-only, and ignored by Git.")

    return 0


def _project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _outside_sample_warning(input_path: Path) -> str | None:
    try:
        input_path.resolve().relative_to(SAMPLE_DOCUMENTS_DIR.resolve())
    except ValueError:
        return (
            "Input path is outside data/sample_documents; ensure it was explicitly "
            "approved and keep generated reports out of Git."
        )
    return None


if __name__ == "__main__":
    raise SystemExit(main())
