import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.ingestion.chunk_schema import (  # noqa: E402
    chunk_record_from_dict,
    chunk_record_to_dict,
    load_chunk_jsonl,
    validate_chunk_dataset,
    validate_chunk_record_dict,
)
from aviationrag.models import ChunkRecord  # noqa: E402


SAMPLE_DIR = ROOT / "data" / "sample_documents"
SAMPLE_MANIFEST = SAMPLE_DIR / "sample_manifest.jsonl"
SAMPLE_CHUNKS = SAMPLE_DIR / "sample_chunks.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class TestChunkSchema(unittest.TestCase):
    def setUp(self):
        self.sample_chunks = load_chunk_jsonl(SAMPLE_CHUNKS)
        self.sample_document_ids = {
            item["document_id"] for item in load_jsonl(SAMPLE_MANIFEST)
        }
        self.valid_chunk = dict(self.sample_chunks[0])

    def test_load_chunk_jsonl_loads_sample_fixture(self):
        chunks = load_chunk_jsonl(SAMPLE_CHUNKS)

        self.assertGreaterEqual(len(chunks), 12)
        self.assertEqual(chunks[0]["chunk_id"], "doc_sample_faa_ac_001_chunk_0001")

    def test_expanded_sample_fixture_validates_with_no_issues(self):
        issues = validate_chunk_dataset(
            self.sample_chunks,
            known_document_ids=self.sample_document_ids,
        )

        self.assertEqual(issues, [])

    def test_missing_required_field_detected(self):
        chunk = dict(self.valid_chunk)
        del chunk["filename"]

        issues = validate_chunk_record_dict(chunk)

        self.assertIn("Missing required field: filename", issues)

    def test_invalid_chunk_type_detected(self):
        chunk = dict(self.valid_chunk, chunk_type="not_a_chunk_type")

        issues = validate_chunk_record_dict(chunk)

        self.assertTrue(any("chunk_type" in issue for issue in issues))

    def test_empty_text_detected(self):
        chunk = dict(self.valid_chunk, text="  ")

        issues = validate_chunk_record_dict(chunk)

        self.assertIn("text must be a non-empty string.", issues)

    def test_invalid_page_range_detected(self):
        chunk = dict(self.valid_chunk, page_start=5, page_end=4)

        issues = validate_chunk_record_dict(chunk)

        self.assertIn("page_end must be greater than or equal to page_start.", issues)

    def test_duplicate_chunk_id_detected(self):
        duplicate = dict(self.sample_chunks[1], chunk_id=self.sample_chunks[0]["chunk_id"])

        issues = validate_chunk_dataset([self.sample_chunks[0], duplicate])

        self.assertIn(f"Duplicate chunk_id: {self.sample_chunks[0]['chunk_id']}", issues)

    def test_unknown_document_id_detected_when_known_document_ids_supplied(self):
        chunk = dict(self.valid_chunk, document_id="doc_sample_unknown_001")

        issues = validate_chunk_dataset(
            [chunk],
            known_document_ids=self.sample_document_ids,
        )

        self.assertTrue(any("Unknown document_id: doc_sample_unknown_001" in issue for issue in issues))

    def test_confidence_score_outside_range_detected(self):
        chunk = dict(self.valid_chunk)
        chunk["metadata"] = dict(chunk["metadata"], confidence_score=1.5)

        issues = validate_chunk_record_dict(chunk)

        self.assertIn("confidence_score must be between 0 and 1.", issues)

    def test_malformed_jsonl_raises_value_error_with_line_number(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "chunks.jsonl"
            path.write_text('{"chunk_id":"ok"}\n{bad json}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 2"):
                load_chunk_jsonl(path)

    def test_chunk_record_from_dict_returns_chunk_record(self):
        record = chunk_record_from_dict(self.valid_chunk)

        self.assertIsInstance(record, ChunkRecord)
        self.assertEqual(record.chunk_id, self.valid_chunk["chunk_id"])
        self.assertEqual(record.metadata["canonical_title"], self.valid_chunk["canonical_title"])
        self.assertEqual(record.metadata["chunk_type"], self.valid_chunk["chunk_type"])

    def test_chunk_record_to_dict_returns_dict(self):
        record = chunk_record_from_dict(self.valid_chunk)

        data = chunk_record_to_dict(record)

        self.assertEqual(data["chunk_id"], self.valid_chunk["chunk_id"])
        self.assertEqual(data["canonical_title"], self.valid_chunk["canonical_title"])
        self.assertEqual(data["chunk_type"], self.valid_chunk["chunk_type"])
        self.assertIsInstance(data["metadata"], dict)

    def test_real_data_paths_are_detected_without_accessing_them(self):
        chunk = dict(self.valid_chunk, filename=r"C:\private\data\documents\real.pdf")

        issues = validate_chunk_record_dict(chunk)

        self.assertTrue(any("local/private path" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
