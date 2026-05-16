import json
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.ingestion.chunk_conversion_writer import (  # noqa: E402
    CHUNK_OUTPUT_FILENAME,
    PAYLOAD_OUTPUT_FILENAME,
    REPORT_OUTPUT_FILENAME,
    chunk_conversion_write_result_to_dict,
    run_local_chunk_conversion_write,
)


SAMPLE_CHUNKS = ROOT / "data" / "sample_documents" / "sample_chunks.jsonl"
TOOL_SCRIPT = ROOT / "tools" / "chunking" / "write-local-chunk-conversion.py"


class ChunkConversionWriterTests(unittest.TestCase):
    def test_run_local_chunk_conversion_write_requires_permission(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch(
                "aviationrag.ingestion.chunk_conversion_writer.get_chunk_migration_settings",
                return_value=SimpleNamespace(enabled=False),
            ):
                with self.assertRaises(PermissionError):
                    run_local_chunk_conversion_write(SAMPLE_CHUNKS, tmpdir, allow_local_write=False)

    def test_run_local_chunk_conversion_write_writes_files_when_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_local_chunk_conversion_write(
                SAMPLE_CHUNKS,
                tmpdir,
                allow_local_write=True,
            )

            self.assertEqual(result.chunk_count, 15)
            self.assertEqual(result.payload_count, 15)
            self.assertTrue(Path(result.chunk_output_path).exists())
            self.assertTrue(Path(result.payload_output_path).exists())
            self.assertTrue(Path(result.report_output_path).exists())

    def test_converted_chunks_jsonl_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_local_chunk_conversion_write(SAMPLE_CHUNKS, tmpdir, allow_local_write=True)

            self.assertTrue((Path(tmpdir) / CHUNK_OUTPUT_FILENAME).exists())

    def test_vector_payloads_jsonl_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_local_chunk_conversion_write(SAMPLE_CHUNKS, tmpdir, allow_local_write=True)

            self.assertTrue((Path(tmpdir) / PAYLOAD_OUTPUT_FILENAME).exists())

    def test_conversion_report_json_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_local_chunk_conversion_write(SAMPLE_CHUNKS, tmpdir, allow_local_write=True)

            self.assertTrue((Path(tmpdir) / REPORT_OUTPUT_FILENAME).exists())

    def test_jsonl_outputs_are_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_local_chunk_conversion_write(SAMPLE_CHUNKS, tmpdir, allow_local_write=True)

            chunks = _load_jsonl(Path(result.chunk_output_path))
            payloads = _load_jsonl(Path(result.payload_output_path))
            self.assertEqual(len(chunks), result.chunk_count)
            self.assertEqual(len(payloads), result.payload_count)
            self.assertIn("chunk_id", chunks[0])
            self.assertIn("metadata", payloads[0])

    def test_report_is_json_serializable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_local_chunk_conversion_write(SAMPLE_CHUNKS, tmpdir, allow_local_write=True)

            json.dumps(chunk_conversion_write_result_to_dict(result))
            report = json.loads(Path(result.report_output_path).read_text(encoding="utf-8"))
            self.assertEqual(report["chunk_count"], result.chunk_count)

    def test_no_embedding_or_vector_fields_are_present_in_payloads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_local_chunk_conversion_write(SAMPLE_CHUNKS, tmpdir, allow_local_write=True)
            payloads = _load_jsonl(Path(result.payload_output_path))

            serialized = json.dumps(payloads).lower()
            self.assertNotIn('"embedding"', serialized)
            self.assertNotIn('"vector"', serialized)
            self.assertNotIn('"embeddings"', serialized)
            self.assertNotIn('"vectors"', serialized)

    def test_max_records_limits_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_local_chunk_conversion_write(
                SAMPLE_CHUNKS,
                tmpdir,
                allow_local_write=True,
                max_records=2,
            )

            self.assertEqual(result.chunk_count, 2)
            self.assertEqual(result.payload_count, 2)
            self.assertEqual(len(_load_jsonl(Path(result.chunk_output_path))), 2)

    def test_no_runtime_data_paths_are_written(self):
        runtime_paths = [
            ROOT / "data" / "processed" / CHUNK_OUTPUT_FILENAME,
            ROOT / "data" / "embeddings" / PAYLOAD_OUTPUT_FILENAME,
            ROOT / "data" / "manifest" / REPORT_OUTPUT_FILENAME,
        ]
        before = {path: path.exists() for path in runtime_paths}

        with tempfile.TemporaryDirectory() as tmpdir:
            run_local_chunk_conversion_write(SAMPLE_CHUNKS, tmpdir, allow_local_write=True)

        after = {path: path.exists() for path in runtime_paths}
        self.assertEqual(after, before)

    def test_tool_script_exists(self):
        self.assertTrue(TOOL_SCRIPT.exists())

    def test_tool_script_writes_to_explicit_tmp_path_when_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOL_SCRIPT),
                    "--input",
                    str(SAMPLE_CHUNKS),
                    "--output-dir",
                    tmpdir,
                    "--max-records",
                    "2",
                    "--allow-local-write",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((Path(tmpdir) / CHUNK_OUTPUT_FILENAME).exists())
            self.assertTrue((Path(tmpdir) / PAYLOAD_OUTPUT_FILENAME).exists())
            self.assertTrue((Path(tmpdir) / REPORT_OUTPUT_FILENAME).exists())


def _load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    unittest.main()
