import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.config import (  # noqa: E402
    CHUNK_MIGRATION_ENV,
    get_chunk_migration_settings,
    is_chunk_migration_enabled,
)
from aviationrag.ingestion.chunk_legacy_adapter import (  # noqa: E402
    legacy_chunk_dict_to_chunk_record,
    legacy_chunk_dicts_to_chunk_records,
    normalize_legacy_chunk_filename,
    preview_legacy_chunk_migration,
)
from aviationrag.ingestion.chunk_payload import validate_vector_payload_dataset  # noqa: E402
from aviationrag.models import ChunkRecord  # noqa: E402


class TestChunkLegacyAdapter(unittest.TestCase):
    def setUp(self):
        self.fake_chunk = {
            "chunk_id": "legacy_chunk_001",
            "filename": " sample_faa_advisory_circular.pdf ",
            "text": "Synthetic fake legacy chunk text.",
            "page": 2,
            "title": "Sample FAA Advisory Circular",
            "section": "General",
            "chunk_type": "text",
            "authority": "FAA",
            "document_type": "advisory_circular",
            "revision": "1A",
            "effective_date": "2026-01-15",
            "extraction_quality": "high",
            "created_at": "2026-05-15T00:00:00Z",
            "metadata": {
                "source_type": "pdf",
                "extraction_method": "pdfplumber",
            },
        }

    def test_normalize_legacy_chunk_filename_handles_slashes_and_whitespace(self):
        self.assertEqual(
            normalize_legacy_chunk_filename(r"  C:\fake\folder\sample_chunk.pdf  "),
            "sample_chunk.pdf",
        )
        self.assertEqual(
            normalize_legacy_chunk_filename("folder/subfolder/sample_chunk.pdf"),
            "sample_chunk.pdf",
        )

    def test_legacy_chunk_dict_to_chunk_record_returns_chunk_record(self):
        record = legacy_chunk_dict_to_chunk_record(self.fake_chunk)

        self.assertIsInstance(record, ChunkRecord)
        self.assertEqual(record.chunk_id, "legacy_chunk_001")

    def test_mapping_preserves_filename_text_and_page_metadata(self):
        record = legacy_chunk_dict_to_chunk_record(self.fake_chunk)

        self.assertEqual(record.filename, "sample_faa_advisory_circular.pdf")
        self.assertEqual(record.text, "Synthetic fake legacy chunk text.")
        self.assertEqual(record.page_start, 2)
        self.assertEqual(record.page_end, 2)
        self.assertEqual(record.metadata["canonical_title"], "Sample FAA Advisory Circular")
        self.assertTrue(record.metadata["source_legacy"])

    def test_section_string_becomes_section_path_list(self):
        record = legacy_chunk_dict_to_chunk_record(self.fake_chunk)

        self.assertEqual(record.section_path, ["General"])

    def test_provided_document_id_overrides_fallback(self):
        record = legacy_chunk_dict_to_chunk_record(
            self.fake_chunk,
            document_id="doc_sample_override_001",
        )

        self.assertEqual(record.document_id, "doc_sample_override_001")

    def test_missing_chunk_id_gets_deterministic_fallback(self):
        chunk = dict(self.fake_chunk)
        del chunk["chunk_id"]

        first = legacy_chunk_dict_to_chunk_record(chunk)
        second = legacy_chunk_dict_to_chunk_record(chunk)

        self.assertEqual(first.chunk_id, second.chunk_id)
        self.assertTrue(first.chunk_id.startswith("legacy_sample_faa_advisory_circular_chunk_"))

    def test_missing_document_id_gets_deterministic_fallback(self):
        first = legacy_chunk_dict_to_chunk_record(self.fake_chunk)
        second = legacy_chunk_dict_to_chunk_record(self.fake_chunk)

        self.assertEqual(first.document_id, second.document_id)
        self.assertTrue(first.document_id.startswith("doc_sample_faa_advisory_circular_"))

    def test_batch_conversion_uses_document_ids_by_filename(self):
        records = legacy_chunk_dicts_to_chunk_records(
            [self.fake_chunk],
            {"sample_faa_advisory_circular.pdf": "doc_sample_faa_ac_001"},
        )

        self.assertEqual(records[0].document_id, "doc_sample_faa_ac_001")

    def test_preview_legacy_chunk_migration_returns_chunks_and_payloads(self):
        preview = preview_legacy_chunk_migration(
            [self.fake_chunk],
            {"sample_faa_advisory_circular.pdf": "doc_sample_faa_ac_001"},
            env={},
        )

        self.assertEqual(len(preview.chunks), 1)
        self.assertEqual(len(preview.payloads), 1)
        self.assertEqual(preview.issues, [])
        self.assertEqual(preview.summary["chunk_count"], 1)
        self.assertIn("Chunk migration is disabled; preview only.", preview.warnings)

    def test_preview_collects_validation_issues_for_bad_fake_chunk(self):
        bad_chunk = dict(self.fake_chunk, text="", chunk_type="bad_type")

        preview = preview_legacy_chunk_migration([bad_chunk], env={})

        self.assertGreater(preview.summary["issue_count"], 0)
        self.assertTrue(any("text must be a non-empty string" in issue for issue in preview.issues))
        self.assertTrue(any("chunk_type" in issue for issue in preview.issues))

    def test_payloads_validate_successfully_for_good_fake_chunks(self):
        preview = preview_legacy_chunk_migration(
            [self.fake_chunk],
            {"sample_faa_advisory_circular.pdf": "doc_sample_faa_ac_001"},
            env={CHUNK_MIGRATION_ENV: "true"},
        )

        self.assertEqual(validate_vector_payload_dataset(preview.payloads), [])

    def test_config_defaults_keep_chunk_migration_disabled_and_dry_run_enabled(self):
        settings = get_chunk_migration_settings({})

        self.assertFalse(settings.enabled)
        self.assertTrue(settings.dry_run)

    def test_config_env_override_enables_chunk_migration(self):
        self.assertTrue(is_chunk_migration_enabled({CHUNK_MIGRATION_ENV: "true"}))

    def test_no_real_files_databases_or_external_services_are_used(self):
        preview = preview_legacy_chunk_migration([self.fake_chunk], env={})

        self.assertEqual(len(preview.chunks), 1)
        self.assertFalse((ROOT / "data" / "manifest" / "documents.jsonl").exists())
        self.assertFalse((ROOT / "logs" / "chunking" / "sample_chunk_payloads.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
