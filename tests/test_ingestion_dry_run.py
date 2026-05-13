import builtins
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.ingestion.dry_run import (  # noqa: E402
    DryRunIngestionPlan,
    build_dry_run_ingestion_plan,
    summarize_dry_run_plan,
    validate_dry_run_plan,
)
from aviationrag.models import ChunkRecord, DocumentRecord  # noqa: E402


def fake_document(filename="2026_FAA_Advisory_Circular_Test.pdf", **overrides):
    data = {
        "filename": filename,
        "title": "Synthetic FAA Advisory Test",
        "authority": "FAA",
        "document_type": "advisory_circular",
        "file_hash": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "ingestion_status": "discovered",
        "metadata": {"fixture": True},
    }
    data.update(overrides)
    return data


def fake_chunk(filename="2026_FAA_Advisory_Circular_Test.pdf", **overrides):
    data = {
        "chunk_id": "chunk_test_001",
        "filename": filename,
        "text": "Synthetic sample text only.",
        "page": 1,
        "metadata": {"fixture": True},
    }
    data.update(overrides)
    return data


class TestIngestionDryRun(unittest.TestCase):
    def test_dry_run_converts_fake_documents_to_document_records(self):
        plan = build_dry_run_ingestion_plan([fake_document()])

        self.assertIsInstance(plan, DryRunIngestionPlan)
        self.assertEqual(len(plan.documents), 1)
        self.assertIsInstance(plan.documents[0], DocumentRecord)
        self.assertEqual(plan.documents[0].authority, "FAA")

    def test_dry_run_converts_fake_chunks_to_chunk_records(self):
        plan = build_dry_run_ingestion_plan([fake_document()], [fake_chunk()])

        self.assertEqual(len(plan.chunks), 1)
        self.assertIsInstance(plan.chunks[0], ChunkRecord)
        self.assertEqual(plan.chunks[0].text, "Synthetic sample text only.")

    def test_chunks_link_to_matching_document_id_by_filename(self):
        plan = build_dry_run_ingestion_plan([fake_document()], [fake_chunk()])

        self.assertEqual(plan.chunks[0].document_id, plan.documents[0].document_id)

    def test_duplicate_document_id_detection(self):
        docs = [
            fake_document(document_id="doc_duplicate_sample"),
            fake_document(
                filename="2026_EASA_CS_Test.pdf",
                authority="EASA",
                document_type="certification_specification",
                document_id="doc_duplicate_sample",
            ),
        ]
        plan = build_dry_run_ingestion_plan(docs)
        issues = validate_dry_run_plan(plan)

        self.assertIn("doc_duplicate_sample", plan.summary["duplicate_document_ids"])
        self.assertIn("Duplicate document_id: doc_duplicate_sample", issues)

    def test_unknown_chunk_document_reference_detection(self):
        plan = build_dry_run_ingestion_plan(
            [fake_document()],
            [
                fake_chunk(
                    filename="Unknown_FAA_Test.pdf",
                    document_id="doc_unknown_reference",
                    chunk_id="chunk_unknown_001",
                )
            ],
        )
        issues = validate_dry_run_plan(plan)

        self.assertIn("doc_unknown_reference", plan.summary["unknown_chunk_document_refs"])
        self.assertIn(
            "Chunk references unknown document_id: chunk_unknown_001 -> doc_unknown_reference",
            issues,
        )

    def test_empty_chunk_text_warning_and_issue(self):
        plan = build_dry_run_ingestion_plan(
            [fake_document()],
            [fake_chunk(chunk_id="chunk_empty_001", text="")],
        )
        issues = validate_dry_run_plan(plan)

        self.assertIn("Chunk has empty text: chunk_empty_001", issues)
        self.assertIn("Chunk has empty text: chunk_empty_001", plan.warnings)

    def test_summary_includes_counts_authorities_and_document_types(self):
        plan = build_dry_run_ingestion_plan([fake_document()], [fake_chunk()])
        summary = summarize_dry_run_plan(plan)

        self.assertEqual(summary["document_count"], 1)
        self.assertEqual(summary["chunk_count"], 1)
        self.assertEqual(summary["authorities"], ["FAA"])
        self.assertEqual(summary["document_types"], ["advisory_circular"])
        self.assertEqual(summary["issue_count"], 0)

    def test_no_filesystem_write_occurs(self):
        with patch.object(builtins, "open", side_effect=AssertionError("open should not run")):
            plan = build_dry_run_ingestion_plan([fake_document()], [fake_chunk()])

        self.assertEqual(plan.summary["document_count"], 1)

    def test_missing_file_paths_are_not_accessed(self):
        private_like_path = r"C:\private\not-read\2026_FAA_Advisory_Circular_Test.pdf"
        with patch.object(builtins, "open", side_effect=AssertionError("open should not run")):
            plan = build_dry_run_ingestion_plan(
                [fake_document(filename=private_like_path)],
                [fake_chunk(filename=private_like_path)],
            )

        self.assertEqual(plan.documents[0].filename, "2026_FAA_Advisory_Circular_Test.pdf")
        self.assertEqual(plan.chunks[0].document_id, plan.documents[0].document_id)

    def test_validation_returns_issues_instead_of_crashing(self):
        plan = build_dry_run_ingestion_plan([], [fake_chunk(document_id="doc_missing_parent")])
        issues = validate_dry_run_plan(plan)

        self.assertIn("No documents supplied.", issues)
        self.assertTrue(any("doc_missing_parent" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
