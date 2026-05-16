import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.ingestion.chunk_audit import (  # noqa: E402
    audit_chunk_records,
    chunk_audit_summary_to_dict,
    detect_chunk_file_format,
    load_chunk_like_records,
    summarize_key_frequency,
)


SAMPLE_CHUNKS = PROJECT_ROOT / "data" / "sample_documents" / "sample_chunks.jsonl"


class ChunkAuditTests(unittest.TestCase):
    def test_detect_chunk_file_format_detects_jsonl_and_json(self):
        self.assertEqual(detect_chunk_file_format("sample.jsonl"), "jsonl")
        self.assertEqual(detect_chunk_file_format("sample.json"), "json")
        self.assertEqual(detect_chunk_file_format("sample.pkl"), "pickle")
        self.assertEqual(detect_chunk_file_format("sample.txt"), "unknown")

    def test_load_chunk_like_records_loads_sample_chunks(self):
        records = load_chunk_like_records(SAMPLE_CHUNKS)
        self.assertGreaterEqual(len(records), 12)
        self.assertIn("chunk_id", records[0])

    def test_audit_chunk_records_returns_expected_record_count(self):
        records = load_chunk_like_records(SAMPLE_CHUNKS)
        summary = audit_chunk_records(records, source_path=str(SAMPLE_CHUNKS))
        self.assertEqual(summary.record_count, len(records))

    def test_audit_counts_top_level_keys(self):
        records = [
            {"chunk_id": "one", "text": "Synthetic text", "metadata": {}},
            {"chunk_id": "two", "document_id": "doc", "metadata": {}},
        ]
        self.assertEqual(summarize_key_frequency(records)["chunk_id"], 2)
        summary = audit_chunk_records(records)
        self.assertEqual(summary.top_level_keys["metadata"], 2)

    def test_audit_counts_metadata_keys(self):
        records = [
            {"chunk_id": "one", "text": "Synthetic text", "metadata": {"source_type": "pdf"}},
            {"chunk_id": "two", "text": "Synthetic text", "metadata": {"source_type": "pdf", "tokens": 3}},
        ]
        summary = audit_chunk_records(records)
        self.assertEqual(summary.metadata_keys["source_type"], 2)
        self.assertEqual(summary.metadata_keys["tokens"], 1)

    def test_audit_counts_missing_text_chunk_id_and_document_id(self):
        summary = audit_chunk_records(
            [
                {"chunk_id": "", "text": "", "metadata": {}},
                {"chunk_id": "ok", "text": "Synthetic text", "metadata": {}},
            ]
        )
        self.assertEqual(summary.missing_text_count, 1)
        self.assertEqual(summary.missing_chunk_id_count, 1)
        self.assertEqual(summary.missing_document_id_count, 2)

    def test_sample_record_shapes_do_not_include_full_text(self):
        text = "Synthetic text that should not be copied into audit shape."
        summary = audit_chunk_records([{"chunk_id": "one", "text": text, "metadata": {}}])
        text_shape = summary.sample_record_shapes[0]["text"]
        self.assertEqual(text_shape["type"], "str")
        self.assertEqual(text_shape["length"], len(text))
        self.assertTrue(text_shape["redacted"])
        self.assertNotIn(text, json.dumps(summary.sample_record_shapes))

    def test_chunk_audit_summary_to_dict_is_json_serializable(self):
        summary = audit_chunk_records([{"chunk_id": "one", "text": "Synthetic text", "metadata": {}}])
        json.dumps(chunk_audit_summary_to_dict(summary))

    def test_malformed_jsonl_raises_value_error_with_line_number(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.jsonl"
            path.write_text('{"chunk_id":"ok"}\n{"bad"\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 2"):
                load_chunk_like_records(path)

    def test_no_directory_scanning_occurs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "must be a file"):
                load_chunk_like_records(Path(tmpdir))

    def test_legacy_json_wrapper_is_flattened(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.json"
            path.write_text(
                json.dumps(
                    {
                        "filename": "sample_legacy.pdf",
                        "metadata": {"source_type": "pdf"},
                        "category": "sample",
                        "chunks": [
                            {"chunk_id": "legacy_1", "text": "Synthetic chunk"},
                            {"chunk_id": "legacy_2", "text": "Another synthetic chunk"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            records = load_chunk_like_records(path)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["filename"], "sample_legacy.pdf")
            self.assertEqual(records[0]["metadata"]["source_type"], "pdf")


if __name__ == "__main__":
    unittest.main()
