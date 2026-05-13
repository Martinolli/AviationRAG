import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.models import AnswerResult, ChunkRecord, DocumentRecord, RetrievedChunk  # noqa: E402


class TestCoreModels(unittest.TestCase):
    def test_document_record_instantiation_and_defaults(self):
        record = DocumentRecord(document_id="doc-1", filename="manual.pdf")

        self.assertEqual(record.document_id, "doc-1")
        self.assertEqual(record.filename, "manual.pdf")
        self.assertIsNone(record.authority)
        self.assertEqual(record.metadata, {})

    def test_chunk_record_to_dict(self):
        record = ChunkRecord(
            chunk_id="chunk-1",
            document_id="doc-1",
            filename="manual.pdf",
            text="Landing gear inspection interval.",
            page_start=10,
            page_end=11,
            section_path=["Chapter 32", "Inspection"],
            metadata={"tokens": 6},
        )

        self.assertEqual(
            record.to_dict(),
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "filename": "manual.pdf",
                "text": "Landing gear inspection interval.",
                "page_start": 10,
                "page_end": 11,
                "section_path": ["Chapter 32", "Inspection"],
                "metadata": {"tokens": 6},
            },
        )

    def test_retrieved_chunk_from_dict_ignores_unknown_keys(self):
        record = RetrievedChunk.from_dict(
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "filename": "manual.pdf",
                "text": "Evidence text.",
                "score": 0.82,
                "source": "faiss",
                "extra": "legacy field",
            }
        )

        self.assertEqual(record.chunk_id, "chunk-1")
        self.assertEqual(record.score, 0.82)
        self.assertEqual(record.metadata, {})
        self.assertFalse(hasattr(record, "extra"))

    def test_answer_result_defaults_are_independent(self):
        first = AnswerResult(answer="Answer one.")
        second = AnswerResult(answer="Answer two.")

        first.citations.append("manual.pdf")
        first.warnings.append("low evidence")

        self.assertEqual(second.citations, [])
        self.assertEqual(second.warnings, [])


if __name__ == "__main__":
    unittest.main()
