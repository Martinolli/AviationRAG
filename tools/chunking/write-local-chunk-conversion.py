#!/usr/bin/env python
"""Write gated local chunk conversion outputs from one explicit chunk-like file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aviationrag.config import is_chunk_migration_enabled  # noqa: E402
from aviationrag.ingestion.chunk_conversion_writer import (  # noqa: E402
    run_local_chunk_conversion_write,
)


DEFAULT_INPUT = PROJECT_ROOT / "data" / "sample_documents" / "sample_chunks.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "migration_dry_run" / "chunks"
SAMPLE_DOCUMENTS_DIR = PROJECT_ROOT / "data" / "sample_documents"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Explicit chunk-like file. Defaults to fake sample chunk fixture.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for local ignored conversion files.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum number of input records to convert.",
    )
    parser.add_argument(
        "--allow-local-write",
        action="store_true",
        help="Explicitly allow local ignored conversion outputs to be written.",
    )
    args = parser.parse_args()

    env_enabled = is_chunk_migration_enabled()
    allow_local_write = args.allow_local_write or env_enabled
    if not allow_local_write:
        print(
            "Refusing to write local conversion outputs. Re-run with --allow-local-write "
            "or set AVIATIONRAG_ENABLE_CHUNK_MIGRATION=true.",
            file=sys.stderr,
        )
        return 2

    input_path = _project_path(args.input)
    output_dir = _project_path(args.output_dir)

    print("Warning: local chunk conversion outputs are generated, ignored, and must not be committed.")
    print("Warning: this tool does not generate embeddings, touch Astra, use FAISS, or modify runtime ingestion.")
    outside_sample_warning = _outside_sample_warning(input_path)
    if outside_sample_warning:
        print(f"Warning: {outside_sample_warning}")
    if env_enabled and not args.allow_local_write:
        print("Warning: local write was enabled by AVIATIONRAG_ENABLE_CHUNK_MIGRATION=true.")

    result = run_local_chunk_conversion_write(
        input_path=input_path,
        output_dir=output_dir,
        allow_local_write=allow_local_write,
        max_records=args.max_records,
    )

    print("Local chunk conversion write completed.")
    print(f"Source path: {result.source_path}")
    print(f"Output dir: {result.output_dir}")
    print(f"Chunk count: {result.chunk_count}")
    print(f"Payload count: {result.payload_count}")
    print(f"Issue count: {len(result.issues)}")
    print(f"Warning count: {len(result.warnings)}")
    print(f"Converted chunks: {result.chunk_output_path}")
    print(f"Vector payloads: {result.payload_output_path}")
    print(f"Report: {result.report_output_path}")
    print("Reminder: outputs are local/ignored only and must not be committed.")
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
            "Input path is outside data/sample_documents; use only explicitly "
            "approved local data and inspect outputs before sharing."
        )
    return None


if __name__ == "__main__":
    raise SystemExit(main())
