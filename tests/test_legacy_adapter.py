import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.ingestion.legacy_adapter import (  # noqa: E402
    build_document_id,
    infer_authority_from_filename,
    infer_document_type_from_filename,
    legacy_chunk_to_record,
    legacy_chunks_to_records,
    legacy_document_to_record,
    legacy_documents_to_records,
    normalize_legacy_filename,
)
from aviationrag.models import ChunkRecord, DocumentRecord  # noqa: E402


class TestLegacyAdapter(unittest.TestCase):
    def test_normalize_legacy_filename_trims_and_removes_path(self):
        self.assertEqual(
            normalize_legacy_filename(r"  C:\fake\sample\2025-01-01_FAA_Sample.pdf  "),
            "2025-01-01_FAA_Sample.pdf",
        )
        self.assertEqual(
            normalize_legacy_filename("folder/subfolder/Sample Manual.docx"),
            "Sample Manual.docx",
        )

    def test_infer_authority_from_filename(self):
        self.assertEqual(infer_authority_from_filename("2025_Regulation_FAA_Sample.pdf"), "FAA")
        self.assertEqual(infer_authority_from_filename("2025_EASA_CS_Sample.pdf"), "EASA")
        self.assertEqual(infer_authority_from_filename("ICAO_Training_Sample.pdf"), "ICAO")
        self.assertEqual(infer_authority_from_filename("NTSB_Accident_Report_Sample.pdf"), "NTSB")
        self.assertEqual(infer_authority_from_filename("MIL_Test_Standard_Sample.pdf"), "MILITARY")
        self.assertEqual(infer_authority_from_filename("DOD_Test_Standard_Sample.pdf"), "MILITARY")
        self.assertIsNone(infer_authority_from_filename("Unknown_Source_Sample.pdf"))

    def test_infer_document_type_from_filename(self):
        cases = {
            "2025_Regulation_FAA_Sample.pdf": "regulation",
            "FAA_Advisory_Circular_Sample.pdf": "advisory_circular",
            "FAA_AC_Sample.pdf": "advisory_circular",
            "EASA_CS_25_Sample.pdf": "certification_specification",
            "ISO_Standard_Sample.pdf": "standard",
            "NTSB_Accident_Report_Sample.pdf": "accident_report",
            "Aircraft_Book_Sample.pdf": "book",
            "Research_Paper_Sample.pdf": "paper",
            "Unknown_Source_Sample.pdf": "other",
        }

        for filename, expected in cases.items():
            with self.subTest(filename=filename):
                self.assertEqual(infer_document_type_from_filename(filename), expected)

    def test_build_document_id_is_deterministic_and_safe(self):
        first = build_document_id("2025 FAA Sample Document.pdf", "sha256:abc")
        second = build_document_id("2025 FAA Sample Document.pdf", "sha256:abc")
        different_hash = build_document_id("2025 FAA Sample Document.pdf", "sha256:def")

        self.assertEqual(first, second)
        self.assertNotEqual(first, different_hash)
        self.assertRegex(first, r"^doc_[a-z0-9_]+_[0-9a-f]{12}$")
        self.assertNotIn(" ", first)

    def test_legacy_document_to_record_returns_document_record(self):
        record = legacy_document_to_record(
            {
                "filename": "2025-01-01_Regulation_FAA_Sample_v1.pdf",
                "title": "Sample FAA Regulation",
                "source_type": "pdf",
                "extraction_method": "pdfplumber",
                "extraction_quality": "medium",
                "needs_manual_review": False,
                "metadata": {"fixture": True},
            }
        )

        self.assertIsInstance(record, DocumentRecord)
        self.assertEqual(record.filename, "2025-01-01_Regulation_FAA_Sample_v1.pdf")
        self.assertEqual(record.title, "Sample FAA Regulation")
        self.assertEqual(record.ingestion_status, "discovered")
        self.assertEqual(record.metadata["source_type"], "pdf")
        self.assertEqual(record.metadata["extraction_method"], "pdfplumber")
        self.assertFalse(record.metadata["needs_manual_review"])

    def test_legacy_document_to_record_fills_inferred_fields(self):
        record = legacy_document_to_record(
            {
                "filename": "2025-02-02_EASA_CS_Sample_v2.pdf",
                "file_hash": "sha256:abc",
                "metadata": {},
            }
        )

        self.assertEqual(record.authority, "EASA")
        self.assertEqual(record.document_type, "certification_specification")
        self.assertEqual(record.file_hash, "sha256:abc")
        self.assertTrue(record.document_id.startswith("doc_2025_02_02_easa_cs_sample_v2_"))

    def test_legacy_chunk_to_record_returns_chunk_record(self):
        record = legacy_chunk_to_record(
            {
                "chunk_id": "chunk_001",
                "filename": "2025-01-01_Regulation_FAA_Sample_v1.pdf",
                "text": "Synthetic sample text only.",
                "page": 3,
                "chunk_type": "section",
                "metadata": {"fixture": True},
            }
        )

        self.assertIsInstance(record, ChunkRecord)
        self.assertEqual(record.chunk_id, "chunk_001")
        self.assertEqual(record.filename, "2025-01-01_Regulation_FAA_Sample_v1.pdf")
        self.assertEqual(record.text, "Synthetic sample text only.")
        self.assertEqual(record.page_start, 3)
        self.assertEqual(record.page_end, 3)
        self.assertEqual(record.metadata["chunk_type"], "section")

    def test_legacy_chunk_to_record_links_to_document_record(self):
        document = legacy_document_to_record(
            {
                "filename": "2025-01-01_Regulation_FAA_Sample_v1.pdf",
                "file_hash": "sha256:abc",
            }
        )
        chunk = legacy_chunk_to_record(
            {
                "filename": "2025-01-01_Regulation_FAA_Sample_v1.pdf",
                "text": "Synthetic sample text only.",
            },
            document=document,
        )

        self.assertEqual(chunk.document_id, document.document_id)
        self.assertTrue(chunk.chunk_id.startswith(document.document_id))

    def test_batch_conversion_returns_expected_counts(self):
        docs = [
            {"filename": "2025-01-01_Regulation_FAA_Sample_v1.pdf"},
            {"filename": "2025-02-02_EASA_CS_Sample_v2.pdf"},
        ]
        chunks = [
            {"filename": "2025-01-01_Regulation_FAA_Sample_v1.pdf", "text": "Synthetic one."},
            {"filename": "2025-02-02_EASA_CS_Sample_v2.pdf", "text": "Synthetic two."},
        ]

        document_records = legacy_documents_to_records(docs)
        chunk_records = legacy_chunks_to_records(chunks, document_records)

        self.assertEqual(len(document_records), 2)
        self.assertEqual(len(chunk_records), 2)
        self.assertEqual(chunk_records[0].document_id, document_records[0].document_id)
        self.assertEqual(chunk_records[1].document_id, document_records[1].document_id)

    def test_fake_records_do_not_require_real_private_filenames(self):
        record = legacy_document_to_record(
            {
                "filename": "Sample_Public_Style_Test_Document.pdf",
                "metadata": {"fixture": True},
            }
        )

        self.assertEqual(record.filename, "Sample_Public_Style_Test_Document.pdf")
        self.assertTrue(record.metadata["fixture"])


if __name__ == "__main__":
    unittest.main()
