import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.ingestion.manifest import (  # noqa: E402
    append_manifest_record,
    document_record_from_dict,
    document_record_to_dict,
    read_manifest,
    validate_manifest_record,
    write_manifest,
)
from aviationrag.models import DocumentRecord  # noqa: E402


SAMPLE_MANIFEST = ROOT / "data" / "sample_documents" / "sample_manifest.jsonl"


class TestManifestWriter(unittest.TestCase):
    def test_read_manifest_reads_sample_fixture(self):
        records = read_manifest(SAMPLE_MANIFEST)

        self.assertEqual(len(records), 5)
        self.assertTrue(all(isinstance(record, DocumentRecord) for record in records))
        self.assertEqual(records[0].document_id, "doc_sample_faa_ac_001")
        self.assertEqual(records[0].title, "Sample FAA Advisory Circular for Training Only")
        self.assertEqual(records[0].source_url, "sample://faa/advisory-circular/sample-001")

    def test_document_ids_are_unique(self):
        records = read_manifest(SAMPLE_MANIFEST)
        document_ids = [record.document_id for record in records]

        self.assertEqual(len(document_ids), len(set(document_ids)))

    def test_write_manifest_writes_valid_jsonl(self):
        records = read_manifest(SAMPLE_MANIFEST)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "manifest.jsonl"
            write_manifest(output_path, records)

            self.assertTrue(output_path.exists())
            content = output_path.read_text(encoding="utf-8")
            self.assertTrue(content.endswith("\n"))

            lines = [line for line in content.splitlines() if line.strip()]
            self.assertEqual(len(lines), len(records))
            parsed = [json.loads(line) for line in lines]
            self.assertEqual(parsed[0]["document_id"], records[0].document_id)
            self.assertIn("canonical_title", parsed[0])

    def test_read_after_write_round_trip(self):
        records = read_manifest(SAMPLE_MANIFEST)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "manifest.jsonl"
            write_manifest(output_path, records)
            round_trip = read_manifest(output_path)

        self.assertEqual([record.document_id for record in round_trip], [r.document_id for r in records])
        self.assertEqual(document_record_to_dict(round_trip[0]), document_record_to_dict(records[0]))

    def test_append_manifest_record_appends_one_record(self):
        records = read_manifest(SAMPLE_MANIFEST)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "manifest.jsonl"
            write_manifest(output_path, records[:1])
            append_manifest_record(output_path, records[1])
            appended = read_manifest(output_path)

        self.assertEqual([record.document_id for record in appended], [records[0].document_id, records[1].document_id])

    def test_missing_file_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_path = Path(temp_dir) / "missing.jsonl"

            self.assertEqual(read_manifest(missing_path), [])

    def test_invalid_json_raises_value_error_with_line_number(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_path = Path(temp_dir) / "bad.jsonl"
            bad_path.write_text('{"document_id":"ok"}\n{bad json}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 2"):
                read_manifest(bad_path)

    def test_validate_manifest_record_detects_missing_required_fields(self):
        record = DocumentRecord(document_id="", filename="", metadata={})
        issues = validate_manifest_record(record)

        self.assertIn("Missing required field: document_id", issues)
        self.assertIn("Missing required field: filename", issues)
        self.assertIn("Missing required field: canonical_title or title", issues)
        self.assertIn("Missing required field: authority", issues)
        self.assertIn("Missing required field: document_type", issues)
        self.assertIn("Missing required field: file_hash", issues)
        self.assertIn("Missing required field: ingestion_status", issues)

    def test_document_record_from_dict_accepts_title_aliases(self):
        record = document_record_from_dict(
            {
                "document_id": "doc_sample_alias_001",
                "filename": "sample_alias.pdf",
                "canonical_title": "Canonical Alias",
                "source_uri": "sample://alias/doc",
                "authority": "OTHER",
                "document_type": "other",
                "file_hash": "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                "ingestion_status": "available",
            }
        )

        self.assertEqual(record.title, "Canonical Alias")
        self.assertEqual(record.source_url, "sample://alias/doc")


if __name__ == "__main__":
    unittest.main()
