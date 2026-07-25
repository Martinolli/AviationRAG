import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.ingestion.persisted_chunk_mapper import (  # noqa: E402
    PersistedChunkCandidateContext,
    PersistedChunkMappingPolicy,
)
from aviationrag.ingestion.persisted_chunk_package import (  # noqa: E402
    PERSISTED_CHUNKS_FILENAME,
    PERSISTENCE_MANIFEST_FILENAME,
    PERSISTENCE_REPORT_FILENAME,
    REJECTED_CANDIDATES_FILENAME,
    WARNINGS_FILENAME,
    build_package_from_adapter_result,
    build_persisted_chunk_package,
    persisted_chunk_package_to_dict,
    write_persisted_chunk_package,
)
from aviationrag.ingestion.persisted_chunk_record import persisted_chunk_record_to_dict  # noqa: E402
from aviationrag.ingestion.structured_document_adapter import (  # noqa: E402
    FAIL,
    PASS,
    REVIEW,
    run_structured_document_adapter,
)


FIXTURE_DIR = ROOT / "tests" / "fixtures" / "structured_document_adapter"
ARTIFACT = FIXTURE_DIR / "structured_document.json"
MANIFEST = FIXTURE_DIR / "manifest.json"
SOURCE = FIXTURE_DIR / "source.txt"
TOOL_SCRIPT = ROOT / "tools" / "chunking" / "run-persisted-chunk-package-dry-run.py"
EXPECTED_DIR = ROOT / "tests" / "fixtures" / "persisted_chunk_mapping"


class PersistedChunkPackageTests(unittest.TestCase):
    def test_adapter_fixture_builds_pass_package(self):
        adapter_result = _adapter_result()
        package = build_package_from_adapter_result(adapter_result)

        self.assertEqual(adapter_result.outcome, PASS)
        self.assertEqual(package.outcome, PASS)
        self.assertEqual(len(package.records), 6)
        self.assertEqual(len(package.rejected_candidates), 0)
        self.assertEqual(len(package.warnings), 0)
        self.assertEqual(package.report["validation_status_counts"], {"valid": 6})
        self.assertEqual(package.report["provenance_counts"], {"full_provenance": 6})
        self.assertEqual(
            package.report["content_type_counts"],
            {"equation": 1, "figure_caption": 1, "paragraph": 2, "table": 1, "warning": 1},
        )
        serialized_records = json.dumps(
            [persisted_chunk_record_to_dict(record) for record in package.records]
        ).lower()
        self.assertNotIn('"embedding"', serialized_records)
        self.assertNotIn('"vector"', serialized_records)
        self.assertFalse(package.report["storage"]["astra_touched"])
        self.assertFalse(package.report["storage"]["faiss_touched"])

    def test_fixture_expectation_json_files_are_valid_and_aligned(self):
        expected_manifest = json.loads((EXPECTED_DIR / "expected_persistence_manifest.json").read_text(encoding="utf-8"))
        expected_report = json.loads((EXPECTED_DIR / "expected_persistence_report.json").read_text(encoding="utf-8"))
        expected_warnings = json.loads((EXPECTED_DIR / "expected_warnings.json").read_text(encoding="utf-8"))
        package = build_package_from_adapter_result(_adapter_result())

        self.assertEqual(expected_manifest["package_schema_name"], package.manifest["package_schema_name"])
        self.assertEqual(expected_manifest["persisted_schema_name"], package.manifest["persisted_schema_name"])
        self.assertEqual(expected_report["content_type_counts"], package.report["content_type_counts"])
        self.assertEqual(expected_warnings["warning_count"], len(package.warnings))

    def test_indexes_are_zero_based_contiguous_and_rejections_do_not_consume_indexes(self):
        adapter_result = _adapter_result()
        candidates = list(adapter_result.candidates)
        bad_candidate = replace(candidates[1], source_checksum="bad")
        package = build_persisted_chunk_package(
            [candidates[0], bad_candidate, candidates[2]],
            allow_rejected_candidates=True,
        )

        self.assertEqual(package.outcome, REVIEW)
        self.assertEqual([record.chunk_index for record in package.records], [0, 1])
        self.assertEqual(len(package.rejected_candidates), 1)

    def test_review_required_record_produces_review_package(self):
        adapter_result = _adapter_result()
        table_candidate = next(candidate for candidate in adapter_result.candidates if candidate.content_type == "table")
        package = build_persisted_chunk_package(
            [table_candidate],
            candidate_contexts={
                table_candidate.chunk_candidate_id: PersistedChunkCandidateContext(
                    accepted_limitation_codes=("TABLE_CANDIDATE_ONLY",),
                    review_required=True,
                )
            },
        )

        self.assertEqual(package.outcome, REVIEW)
        self.assertEqual(package.records[0].validation_status, "review_required")
        self.assertEqual(package.report["limitation_counts"], {"TABLE_CANDIDATE_ONLY": 1})
        self.assertEqual(package.report["review_required_count"], 1)

    def test_rejected_candidate_strict_policy_fails(self):
        bad_candidate = replace(_adapter_result().candidates[0], source_checksum="not-a-sha")
        package = build_persisted_chunk_package([bad_candidate])

        self.assertEqual(package.outcome, FAIL)
        self.assertEqual(len(package.records), 0)
        self.assertEqual(len(package.rejected_candidates), 1)
        self.assertTrue(any(issue.code == "REJECTED_CANDIDATES_PRESENT" for issue in package.issues))

    def test_unknown_candidate_context_is_reported(self):
        package = build_persisted_chunk_package(
            [_adapter_result().candidates[0]],
            candidate_contexts={"missing-candidate": PersistedChunkCandidateContext()},
            allow_rejected_candidates=True,
        )

        self.assertEqual(package.outcome, REVIEW)
        self.assertTrue(any(item.candidate_id == "missing-candidate" for item in package.rejected_candidates))

    def test_chunk_id_collision_fails_package(self):
        first = _adapter_result().candidates[0]
        duplicate_identity = replace(first, text="Changed text but same identity inputs.")
        package = build_persisted_chunk_package([first, duplicate_identity], allow_rejected_candidates=True)

        self.assertEqual(package.outcome, FAIL)
        self.assertTrue(any(issue.code == "PERSISTED_CHUNK_ID_COLLISION" for issue in package.issues))

    def test_adapter_fail_blocks_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.txt"
            source.write_text("changed source bytes", encoding="utf-8")
            adapter_result = run_structured_document_adapter(ARTIFACT, MANIFEST, source_path=source)
            package = build_package_from_adapter_result(adapter_result)

        self.assertEqual(adapter_result.outcome, FAIL)
        self.assertEqual(package.outcome, FAIL)
        self.assertEqual(len(package.records), 0)

    def test_adapter_review_can_be_mapped_with_explicit_partial_policy(self):
        adapter_result = run_structured_document_adapter(ARTIFACT, MANIFEST)
        contexts = {
            candidate.chunk_candidate_id: PersistedChunkCandidateContext(
                accepted_limitation_codes=("CHUNK_SECTION_CROSSING_REVIEW",),
                warning_codes=("PARTIAL_PROVENANCE_REVIEW_REQUIRED",),
                review_required=True,
            )
            for candidate in adapter_result.candidates
        }
        package = build_package_from_adapter_result(
            adapter_result,
            candidate_contexts=contexts,
            policy=PersistedChunkMappingPolicy(allow_partial_provenance=True),
        )

        self.assertEqual(adapter_result.outcome, REVIEW)
        self.assertEqual(package.outcome, REVIEW)
        self.assertEqual(package.report["provenance_counts"], {"partial_provenance": 6})

    def test_write_requires_permission_and_writes_all_files_when_allowed(self):
        package = build_package_from_adapter_result(_adapter_result())
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PermissionError):
                write_persisted_chunk_package(package, tmp)

            result = write_persisted_chunk_package(package, tmp, allow_local_write=True)
            paths = [
                result.persisted_chunks_output_path,
                result.persistence_manifest_output_path,
                result.persistence_report_output_path,
                result.rejected_candidates_output_path,
                result.warnings_output_path,
            ]
            for path in paths:
                self.assertTrue(Path(path).exists(), path)

    def test_jsonl_json_final_newline_and_checksums(self):
        package = build_package_from_adapter_result(_adapter_result())
        with tempfile.TemporaryDirectory() as tmp:
            result = write_persisted_chunk_package(package, tmp, allow_local_write=True)
            persisted_lines = Path(result.persisted_chunks_output_path).read_text(encoding="utf-8").splitlines()
            manifest = json.loads(Path(result.persistence_manifest_output_path).read_text(encoding="utf-8"))
            report = json.loads(Path(result.persistence_report_output_path).read_text(encoding="utf-8"))
            warnings = json.loads(Path(result.warnings_output_path).read_text(encoding="utf-8"))

            self.assertEqual(len(persisted_lines), 6)
            self.assertTrue(Path(result.persisted_chunks_output_path).read_bytes().endswith(b"\n"))
            self.assertTrue(Path(result.persistence_manifest_output_path).read_bytes().endswith(b"\n"))
            self.assertEqual(report["outcome"], PASS)
            self.assertEqual(warnings["warning_count"], 0)
            for filename, digest in manifest["file_sha256"].items():
                self.assertEqual(hashlib.sha256((Path(tmp) / filename).read_bytes()).hexdigest(), digest)

    def test_existing_files_reject_without_overwrite_and_overwrite_works(self):
        package = build_package_from_adapter_result(_adapter_result())
        with tempfile.TemporaryDirectory() as tmp:
            write_persisted_chunk_package(package, tmp, allow_local_write=True)
            with self.assertRaises(FileExistsError):
                write_persisted_chunk_package(package, tmp, allow_local_write=True)
            write_persisted_chunk_package(package, tmp, allow_local_write=True, overwrite=True)

    def test_no_write_outside_output_directory(self):
        package = build_package_from_adapter_result(_adapter_result())
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "package"
            write_persisted_chunk_package(package, output, allow_local_write=True)
            self.assertFalse((Path(tmp) / PERSISTED_CHUNKS_FILENAME).exists())

    def test_package_bytes_are_deterministic_across_clean_directories(self):
        package = build_package_from_adapter_result(_adapter_result())
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            write_persisted_chunk_package(package, first, allow_local_write=True)
            write_persisted_chunk_package(package, second, allow_local_write=True)
            for filename in [
                PERSISTED_CHUNKS_FILENAME,
                PERSISTENCE_MANIFEST_FILENAME,
                PERSISTENCE_REPORT_FILENAME,
                REJECTED_CANDIDATES_FILENAME,
                WARNINGS_FILENAME,
            ]:
                self.assertEqual((Path(first) / filename).read_bytes(), (Path(second) / filename).read_bytes(), filename)

    def test_cli_pass_review_fail_and_no_write_default(self):
        pass_run = _run_cli("--source", str(SOURCE))
        review_run = _run_cli("--allow-partial-provenance")
        fail_run = _run_cli("--source", str(SOURCE), "--include-headings", "--strict")

        self.assertEqual(pass_run.returncode, 0, pass_run.stdout + pass_run.stderr)
        self.assertIn("Package outcome: PASS", pass_run.stdout)
        self.assertIn("Package outputs not written.", pass_run.stdout)
        self.assertEqual(review_run.returncode, 2, review_run.stdout + review_run.stderr)
        self.assertIn("Package outcome: REVIEW", review_run.stdout)
        self.assertEqual(fail_run.returncode, 1)
        self.assertIn("Package outcome: FAIL", fail_run.stdout)

    def test_cli_local_write_requires_permission_and_reports_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            completed = _run_cli("--source", str(SOURCE), "--output-dir", tmp, "--allow-local-write")

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn("Package digest:", completed.stdout)
            self.assertTrue((Path(tmp) / PERSISTED_CHUNKS_FILENAME).exists())

    def test_source_scripts_are_not_referenced_by_package_modules(self):
        package_source = (ROOT / "src" / "aviationrag" / "ingestion" / "persisted_chunk_package.py").read_text(encoding="utf-8")
        mapper_source = (ROOT / "src" / "aviationrag" / "ingestion" / "persisted_chunk_mapper.py").read_text(encoding="utf-8")
        combined = (package_source + mapper_source).lower()

        self.assertNotIn("read_documents", combined)
        self.assertNotIn("aviation_chunk_saver", combined)
        self.assertNotIn("faiss_indexer", combined)
        self.assertNotIn("import faiss", combined)
        self.assertNotIn("astradb", combined)


def _adapter_result():
    return run_structured_document_adapter(ARTIFACT, MANIFEST, source_path=SOURCE)


def _run_cli(*extra_args):
    return subprocess.run(
        [
            sys.executable,
            str(TOOL_SCRIPT),
            "--artifact",
            str(ARTIFACT),
            "--manifest",
            str(MANIFEST),
            *extra_args,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


if __name__ == "__main__":
    unittest.main()
