import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.ingestion.chunk_migration_dry_run import (  # noqa: E402
    chunk_migration_dry_run_result_to_dict,
    run_chunk_migration_dry_run,
)


SAMPLE_CHUNKS = ROOT / "data" / "sample_documents" / "sample_chunks.jsonl"
TOOL_SCRIPT = ROOT / "tools" / "chunking" / "run-chunk-migration-dry-run.py"


class ChunkMigrationDryRunTests(unittest.TestCase):
    def test_run_chunk_migration_dry_run_works_with_sample_chunks(self):
        result = run_chunk_migration_dry_run(SAMPLE_CHUNKS)

        self.assertEqual(result.source_path, str(SAMPLE_CHUNKS))
        self.assertEqual(result.audit["record_count"], result.chunk_count)

    def test_result_contains_audit_summary(self):
        result = run_chunk_migration_dry_run(SAMPLE_CHUNKS)

        self.assertIn("top_level_keys", result.audit)
        self.assertIn("sample_record_shapes", result.audit)
        self.assertEqual(result.summary["detected_format"], "jsonl")

    def test_chunk_count_is_greater_than_zero(self):
        result = run_chunk_migration_dry_run(SAMPLE_CHUNKS)

        self.assertGreater(result.chunk_count, 0)

    def test_payload_count_is_greater_than_zero(self):
        result = run_chunk_migration_dry_run(SAMPLE_CHUNKS)

        self.assertGreater(result.payload_count, 0)

    def test_result_to_dict_is_json_serializable(self):
        result = run_chunk_migration_dry_run(SAMPLE_CHUNKS)

        json.dumps(chunk_migration_dry_run_result_to_dict(result))

    def test_max_records_limits_input(self):
        result = run_chunk_migration_dry_run(SAMPLE_CHUNKS, max_records=2)

        self.assertEqual(result.audit["record_count"], 2)
        self.assertEqual(result.chunk_count, 2)
        self.assertEqual(result.payload_count, 2)

    def test_malformed_file_propagates_value_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.jsonl"
            path.write_text('{"chunk_id":"ok"}\n{"bad"\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 2"):
                run_chunk_migration_dry_run(path)

    def test_no_generated_runtime_data_paths_are_written(self):
        generated_runtime_paths = [
            ROOT / "data" / "processed" / "chunk_migration_dry_run.json",
            ROOT / "data" / "embeddings" / "chunk_migration_dry_run.json",
            ROOT / "data" / "manifest" / "chunk_migration_dry_run.json",
        ]
        before = {path: path.exists() for path in generated_runtime_paths}

        run_chunk_migration_dry_run(SAMPLE_CHUNKS)

        after = {path: path.exists() for path in generated_runtime_paths}
        self.assertEqual(after, before)

    def test_no_embeddings_astra_or_faiss_imports_are_used(self):
        module_path = ROOT / "src" / "aviationrag" / "ingestion" / "chunk_migration_dry_run.py"
        source = module_path.read_text(encoding="utf-8").lower()

        self.assertNotIn("import faiss", source)
        self.assertNotIn("from faiss", source)
        self.assertNotIn("astradb", source)
        self.assertNotIn("openai.embeddings", source)

    def test_tool_script_exists(self):
        self.assertTrue(TOOL_SCRIPT.exists())

    def test_tool_script_writes_report_to_explicit_tmp_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dry_run_report.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOL_SCRIPT),
                    "--input",
                    str(SAMPLE_CHUNKS),
                    "--max-records",
                    "2",
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["chunk_count"], 2)
            self.assertEqual(report["payload_count"], 2)


if __name__ == "__main__":
    unittest.main()
