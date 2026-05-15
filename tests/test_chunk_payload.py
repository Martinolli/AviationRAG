import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.ingestion.chunk_payload import (  # noqa: E402
    VECTOR_PAYLOAD_SCHEMA_VERSION,
    chunk_to_vector_payload,
    load_sample_chunk_payloads,
    validate_vector_payload,
    validate_vector_payload_dataset,
)
from aviationrag.ingestion.chunk_schema import (  # noqa: E402
    chunk_record_from_dict,
    load_chunk_jsonl,
)


SAMPLE_CHUNKS = ROOT / "data" / "sample_documents" / "sample_chunks.jsonl"


class TestChunkPayload(unittest.TestCase):
    def setUp(self):
        self.sample_chunks = load_chunk_jsonl(SAMPLE_CHUNKS)
        self.sample_chunk = dict(self.sample_chunks[0])

    def test_chunk_to_vector_payload_converts_sample_chunk_dict(self):
        payload = chunk_to_vector_payload(self.sample_chunk)

        self.assertEqual(payload["payload_schema_version"], VECTOR_PAYLOAD_SCHEMA_VERSION)
        self.assertEqual(payload["chunk_id"], self.sample_chunk["chunk_id"])
        self.assertEqual(payload["document_id"], self.sample_chunk["document_id"])
        self.assertEqual(payload["text"], self.sample_chunk["text"])

    def test_chunk_to_vector_payload_converts_chunk_record(self):
        record = chunk_record_from_dict(self.sample_chunk)

        payload = chunk_to_vector_payload(record)

        self.assertEqual(payload["chunk_id"], self.sample_chunk["chunk_id"])
        self.assertEqual(payload["metadata"]["canonical_title"], self.sample_chunk["canonical_title"])

    def test_payload_contains_top_level_shape(self):
        payload = chunk_to_vector_payload(self.sample_chunk)

        self.assertEqual(
            sorted(payload),
            ["chunk_id", "document_id", "metadata", "payload_schema_version", "text"],
        )
        self.assertIsInstance(payload["metadata"], dict)

    def test_metadata_contains_required_traceability_fields(self):
        payload = chunk_to_vector_payload(self.sample_chunk)
        metadata = payload["metadata"]

        for field_name in [
            "filename",
            "canonical_title",
            "authority",
            "document_type",
            "page_start",
            "page_end",
            "section_path",
            "chunk_type",
            "text_hash",
            "source_hash",
        ]:
            self.assertIn(field_name, metadata)

    def test_optional_metadata_is_preserved(self):
        chunk = dict(self.sample_chunks[1])
        payload = chunk_to_vector_payload(chunk)

        self.assertEqual(payload["metadata"]["regulatory_reference"], "FAKE-AC-1A-2A")
        self.assertEqual(payload["metadata"]["applicability"], "sample compliance fixture only")
        self.assertEqual(payload["metadata"]["aircraft_category"], "fictional training category")

    def test_validate_vector_payload_detects_missing_metadata(self):
        payload = chunk_to_vector_payload(self.sample_chunk)
        del payload["metadata"]

        issues = validate_vector_payload(payload)

        self.assertIn("metadata must be a dictionary.", issues)

    def test_validate_vector_payload_detects_embedding_fields(self):
        payload = chunk_to_vector_payload(self.sample_chunk)
        payload["embedding"] = [0.1, 0.2, 0.3]

        issues = validate_vector_payload(payload)

        self.assertTrue(any("Forbidden embedding/vector field present: embedding" in issue for issue in issues))

    def test_validate_vector_payload_dataset_detects_duplicate_chunk_id(self):
        first = chunk_to_vector_payload(self.sample_chunks[0])
        second = chunk_to_vector_payload(self.sample_chunks[1])
        second["chunk_id"] = first["chunk_id"]

        issues = validate_vector_payload_dataset([first, second])

        self.assertIn(f"Duplicate chunk_id: {first['chunk_id']}", issues)

    def test_load_sample_chunk_payloads_returns_valid_payloads(self):
        payloads = load_sample_chunk_payloads(SAMPLE_CHUNKS)

        self.assertEqual(len(payloads), len(self.sample_chunks))
        self.assertEqual(validate_vector_payload_dataset(payloads), [])

    def test_no_astra_faiss_or_embedding_calls_are_required(self):
        payload = chunk_to_vector_payload(self.sample_chunk)

        self.assertNotIn("embedding", payload)
        self.assertNotIn("vector", payload)
        self.assertNotIn("faiss", payload)
        self.assertNotIn("astra", payload)


if __name__ == "__main__":
    unittest.main()
