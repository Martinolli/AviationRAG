import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample_documents"
MANIFEST_PATH = SAMPLE_DIR / "sample_manifest.jsonl"
CHUNKS_PATH = SAMPLE_DIR / "sample_chunks.jsonl"

MANIFEST_REQUIRED_FIELDS = {
    "document_id",
    "filename",
    "canonical_title",
    "authority",
    "document_type",
    "revision",
    "effective_date",
    "source_uri",
    "file_hash",
    "ingestion_status",
    "approval_status",
    "extraction_method",
    "extraction_quality",
    "needs_manual_review",
    "ingestion_batch_id",
    "created_at",
    "updated_at",
    "metadata",
}

CHUNK_REQUIRED_FIELDS = {
    "chunk_id",
    "document_id",
    "filename",
    "canonical_title",
    "page_start",
    "page_end",
    "section_path",
    "paragraph_id",
    "chunk_type",
    "authority",
    "document_type",
    "revision",
    "effective_date",
    "extraction_quality",
    "source_hash",
    "text_hash",
    "text",
    "metadata",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class TestSampleManifestFixture(unittest.TestCase):
    def test_manifest_exists_and_has_valid_records(self):
        self.assertTrue(MANIFEST_PATH.exists())
        records = load_jsonl(MANIFEST_PATH)

        self.assertGreaterEqual(len(records), 3)
        self.assertLessEqual(len(records), 5)

        document_ids = [record["document_id"] for record in records]
        self.assertEqual(len(document_ids), len(set(document_ids)))

        for record in records:
            self.assertTrue(MANIFEST_REQUIRED_FIELDS.issubset(record))
            self.assertTrue(record["document_id"].startswith("doc_sample_"))
            self.assertTrue(record["source_uri"].startswith("sample://"))
            self.assertRegex(record["file_hash"], r"^sha256:[0-9a-f]{64}$")
            self.assertIsInstance(record["metadata"], dict)
            self.assertTrue(record["metadata"].get("fixture"))
            self.assertIn("Fake sample record", record["metadata"].get("note", ""))

    def test_manifest_does_not_reference_local_or_private_paths(self):
        forbidden_patterns = [
            re.compile(r"[A-Za-z]:\\"),
            re.compile(r"(^|/)(data/documents|data/raw|data/processed|data/embeddings)(/|$)"),
            re.compile(r"secure-connect", re.IGNORECASE),
            re.compile(r"\.env", re.IGNORECASE),
        ]

        for path in [MANIFEST_PATH, CHUNKS_PATH]:
            text = path.read_text(encoding="utf-8")
            for pattern in forbidden_patterns:
                self.assertIsNone(pattern.search(text), f"{path} matched {pattern.pattern}")

    def test_chunks_are_valid_and_reference_manifest_documents(self):
        self.assertTrue(CHUNKS_PATH.exists())

        manifest_records = load_jsonl(MANIFEST_PATH)
        chunk_records = load_jsonl(CHUNKS_PATH)

        manifest_ids = {record["document_id"] for record in manifest_records}
        chunk_ids = [record["chunk_id"] for record in chunk_records]

        self.assertGreaterEqual(len(chunk_records), 3)
        self.assertLessEqual(len(chunk_records), 5)
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))

        for record in chunk_records:
            self.assertTrue(CHUNK_REQUIRED_FIELDS.issubset(record))
            self.assertIn(record["document_id"], manifest_ids)
            self.assertTrue(record["chunk_id"].startswith(record["document_id"]))
            self.assertRegex(record["source_hash"], r"^sha256:[0-9a-f]{64}$")
            self.assertRegex(record["text_hash"], r"^sha256:[0-9a-f]{64}$")
            self.assertLessEqual(len(record["text"]), 160)
            self.assertIn("synthetic", record["text"].lower())
            self.assertTrue(record["metadata"].get("fixture"))


if __name__ == "__main__":
    unittest.main()
