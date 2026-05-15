"""Export fake sample chunks to future vector payload-shaped JSONL.

This developer utility uses only committed fake/sample data. It does not
generate embeddings, call external APIs, connect to Astra, use FAISS, or write
to any vector database.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.ingestion.chunk_payload import (  # noqa: E402
    chunks_to_vector_payloads,
    validate_vector_payload_dataset,
)
from aviationrag.ingestion.chunk_schema import (  # noqa: E402
    load_chunk_jsonl,
    validate_chunk_dataset,
)


INPUT_PATH = ROOT / "data" / "sample_documents" / "sample_chunks.jsonl"
OUTPUT_PATH = ROOT / "logs" / "chunking" / "sample_chunk_payloads.jsonl"


def main() -> int:
    if not INPUT_PATH.exists():
        print(f"Sample chunk fixture missing: {INPUT_PATH}", file=sys.stderr)
        return 1

    chunks = load_chunk_jsonl(INPUT_PATH)
    chunk_issues = validate_chunk_dataset(chunks)
    if chunk_issues:
        print("Chunk validation failed:", file=sys.stderr)
        for issue in chunk_issues:
            print(f"- {issue}", file=sys.stderr)
        return 1

    payloads = chunks_to_vector_payloads(chunks)
    payload_issues = validate_vector_payload_dataset(payloads)
    if payload_issues:
        print("Payload validation failed:", file=sys.stderr)
        for issue in payload_issues:
            print(f"- {issue}", file=sys.stderr)
        return 1

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    print("Sample chunk payload export complete.")
    print(f"Chunks loaded: {len(chunks)}")
    print(f"Payloads written: {len(payloads)}")
    print(f"Output path: {OUTPUT_PATH}")
    print("Warning: output is generated, local-only, and ignored by Git.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
