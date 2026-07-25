import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


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
    write_persisted_chunk_package,
)
from aviationrag.ingestion.real_parser_sample_gate import (  # noqa: E402
    real_parser_sample_gate_result_to_dict,
    run_real_parser_sample_gate,
    sanitized_gate_report_bytes,
)
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
TOOL_SCRIPT = ROOT / "tools" / "chunking" / "run-real-parser-sample-persistence-gate.py"
ACCEPTANCE_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "real_parser_sample"
    / "faa_order_4040_26b_gate_acceptance.json"
)
PACKAGE_FILENAMES = (
    PERSISTED_CHUNKS_FILENAME,
    PERSISTENCE_MANIFEST_FILENAME,
    PERSISTENCE_REPORT_FILENAME,
    REJECTED_CANDIDATES_FILENAME,
    WARNINGS_FILENAME,
)


class RealParserSampleGateTests(unittest.TestCase):
    def test_matching_source_checksum_passes(self):
        result = run_real_parser_sample_gate(
            artifact_path=ARTIFACT,
            manifest_path=MANIFEST,
            source_path=SOURCE,
        )

        self.assertEqual(result.outcome, PASS)
        self.assertEqual(result.adapter_outcome, PASS)
        self.assertEqual(result.package_outcome, PASS)
        self.assertEqual(result.input_candidate_count, 6)
        self.assertEqual(result.accepted_record_count, 6)
        self.assertEqual(result.rejected_candidate_count, 0)
        self.assertEqual(result.warning_count, 0)
        self.assertEqual(result.provenance_counts, {"full_provenance": 6})
        self.assertTrue(result.determinism_verified)

    def test_source_checksum_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, manifest, source = _copy_fixture(tmp)
            source.write_text("changed", encoding="utf-8")

            result = run_real_parser_sample_gate(
                artifact_path=artifact,
                manifest_path=manifest,
                source_path=source,
            )

        self.assertEqual(result.outcome, FAIL)
        self.assertIn("SOURCE_ARTIFACT_CHECKSUM_MISMATCH", result.blocking_issue_codes)
        self.assertIn("SOURCE_MANIFEST_CHECKSUM_MISMATCH", result.blocking_issue_codes)

    def test_artifact_manifest_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, manifest, source = _copy_fixture(tmp)
            data = _read_json(manifest)
            data["artifacts"][0]["artifact_sha256"] = "0" * 64
            _write_json(manifest, data)

            result = run_real_parser_sample_gate(
                artifact_path=artifact,
                manifest_path=manifest,
                source_path=source,
            )

        self.assertEqual(result.outcome, FAIL)
        self.assertIn("ARTIFACT_MANIFEST_CHECKSUM_MISMATCH", result.blocking_issue_codes)

    def test_wrong_document_identity_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, manifest, source = _copy_fixture(tmp)
            data = _read_json(manifest)
            data["artifacts"][0]["document_id"] = "wrong-doc"
            _write_json(manifest, data)

            result = run_real_parser_sample_gate(
                artifact_path=artifact,
                manifest_path=manifest,
                source_path=source,
            )

        self.assertEqual(result.outcome, FAIL)
        self.assertIn("DOCUMENT_IDENTITY_MISMATCH", result.blocking_issue_codes)

    def test_unsupported_schema_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, manifest, source = _copy_fixture(tmp)
            data = _read_json(artifact)
            data["schema_version"] = "9.9.9"
            _write_artifact_and_manifest(artifact, manifest, source, data)

            result = run_real_parser_sample_gate(
                artifact_path=artifact,
                manifest_path=manifest,
                source_path=source,
            )

        self.assertEqual(result.outcome, FAIL)
        self.assertIn("STRUCTURED_DOCUMENT_SCHEMA_UNSUPPORTED", result.blocking_issue_codes)

    def test_d4c_review_blocks_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, manifest, source = _fixture_with_figure_block(tmp)

            result = run_real_parser_sample_gate(
                artifact_path=artifact,
                manifest_path=manifest,
                source_path=source,
            )

        self.assertEqual(result.outcome, FAIL)
        self.assertEqual(result.adapter_outcome, REVIEW)
        self.assertIn("ADAPTER_REVIEW_NOT_ALLOWED", result.blocking_issue_codes)
        self.assertIn("UNAPPROVED_ADAPTER_WARNING:BLOCK_CONTENT_TYPE_SKIPPED", result.blocking_issue_codes)

    def test_approved_d4c_review_may_proceed_when_allowed(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, manifest, source = _fixture_with_figure_block(tmp)

            result = run_real_parser_sample_gate(
                artifact_path=artifact,
                manifest_path=manifest,
                source_path=source,
                approved_adapter_warning_codes=("BLOCK_CONTENT_TYPE_SKIPPED",),
                allow_review=True,
            )

        self.assertEqual(result.outcome, REVIEW)
        self.assertEqual(result.adapter_outcome, REVIEW)
        self.assertEqual(result.rejected_candidate_count, 0)

    def test_d4c_fail_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, manifest, source = _copy_fixture(tmp)
            source.write_text("bad source", encoding="utf-8")

            result = run_real_parser_sample_gate(
                artifact_path=artifact,
                manifest_path=manifest,
                source_path=source,
            )

        self.assertEqual(result.outcome, FAIL)
        self.assertIn("ADAPTER_OUTCOME_FAIL", result.blocking_issue_codes)

    def test_d5b_review_produces_review_when_allowed(self):
        adapter_result = run_structured_document_adapter(ARTIFACT, MANIFEST, source_path=SOURCE)
        table_id = next(candidate.chunk_candidate_id for candidate in adapter_result.candidates if candidate.content_type == "table")

        result = run_real_parser_sample_gate(
            artifact_path=ARTIFACT,
            manifest_path=MANIFEST,
            source_path=SOURCE,
            candidate_contexts={
                table_id: PersistedChunkCandidateContext(
                    accepted_limitation_codes=("TABLE_CANDIDATE_ONLY",),
                    review_required=True,
                )
            },
            mapping_policy=PersistedChunkMappingPolicy(allow_review_required_records=True),
            allow_review=True,
        )

        self.assertEqual(result.outcome, REVIEW)
        self.assertEqual(result.package_outcome, REVIEW)
        self.assertEqual(result.accepted_limitation_counts, {"TABLE_CANDIDATE_ONLY": 1})
        self.assertEqual(result.review_required_count, 1)

    def test_d5b_fail_produces_fail(self):
        adapter_result = run_structured_document_adapter(ARTIFACT, MANIFEST, source_path=SOURCE)
        candidate_id = adapter_result.candidates[0].chunk_candidate_id

        result = run_real_parser_sample_gate(
            artifact_path=ARTIFACT,
            manifest_path=MANIFEST,
            source_path=SOURCE,
            candidate_contexts={
                candidate_id: PersistedChunkCandidateContext(
                    accepted_limitation_codes=("UNKNOWN_LIMITATION",),
                    review_required=True,
                )
            },
        )

        self.assertEqual(result.outcome, FAIL)
        self.assertEqual(result.package_outcome, FAIL)
        self.assertIn("UNKNOWN_LIMITATION_CODE", result.blocking_issue_codes)

    def test_zero_candidates_and_zero_accepted_records_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, manifest, source = _copy_fixture(tmp)
            data = _read_json(artifact)
            for block in data["blocks"]:
                block["block_type"] = "metadata"
            data["tables"] = []
            data["figures"] = []
            data["equations"] = []
            data["admonitions"] = []
            data["cross_references"] = []
            _write_artifact_and_manifest(artifact, manifest, source, data)

            result = run_real_parser_sample_gate(
                artifact_path=artifact,
                manifest_path=manifest,
                source_path=source,
            )

        self.assertEqual(result.outcome, FAIL)
        self.assertIn("NO_ADAPTER_CANDIDATES", result.blocking_issue_codes)
        self.assertIn("NO_ACCEPTED_RECORDS", result.blocking_issue_codes)

    def test_rejected_candidate_fails_strict_gate(self):
        result = run_real_parser_sample_gate(
            artifact_path=ARTIFACT,
            manifest_path=MANIFEST,
            source_path=SOURCE,
            candidate_contexts={
                "missing-candidate": PersistedChunkCandidateContext(
                    accepted_limitation_codes=("TABLE_CANDIDATE_ONLY",),
                    review_required=True,
                )
            },
        )

        self.assertEqual(result.outcome, FAIL)
        self.assertIn("REJECTED_CANDIDATES_PRESENT", result.blocking_issue_codes)

    def test_unknown_adapter_warning_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, manifest, source = _fixture_with_figure_block(tmp)

            result = run_real_parser_sample_gate(
                artifact_path=artifact,
                manifest_path=manifest,
                source_path=source,
                allow_review=True,
            )

        self.assertEqual(result.outcome, FAIL)
        self.assertIn("UNAPPROVED_ADAPTER_WARNING:BLOCK_CONTENT_TYPE_SKIPPED", result.blocking_issue_codes)

    def test_collision_fails_gate(self):
        with patch(
            "aviationrag.ingestion.persisted_chunk_mapper.build_persisted_chunk_id",
            return_value="adapter-fixture-doc:chunk:111111111111111111111111",
        ):
            result = run_real_parser_sample_gate(
                artifact_path=ARTIFACT,
                manifest_path=MANIFEST,
                source_path=SOURCE,
            )

        self.assertEqual(result.outcome, FAIL)
        self.assertIn("PERSISTED_CHUNK_ID_COLLISION", result.blocking_issue_codes)

    def test_missing_provenance_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, manifest, source = _copy_fixture(tmp)
            data = _read_json(artifact)
            first_block = next(block for block in data["blocks"] if block.get("block_id") == "para-1")
            first_block.pop("pdf_page_index", None)
            first_block.pop("pdf_page_index_start", None)
            first_block.pop("pdf_page_index_end", None)
            first_block["source_span"].pop("pdf_page_index_start", None)
            first_block["source_span"].pop("pdf_page_index_end", None)
            _write_artifact_and_manifest(artifact, manifest, source, data)

            result = run_real_parser_sample_gate(
                artifact_path=artifact,
                manifest_path=manifest,
                source_path=source,
            )

        self.assertEqual(result.outcome, FAIL)
        self.assertIn("PACKAGE_OUTCOME_FAIL", result.blocking_issue_codes)

    def test_two_identical_package_writes_match(self):
        adapter_result = run_structured_document_adapter(ARTIFACT, MANIFEST, source_path=SOURCE)
        package = build_package_from_adapter_result(adapter_result)
        with tempfile.TemporaryDirectory() as tmp:
            run_1 = Path(tmp) / "run_1"
            run_2 = Path(tmp) / "run_2"
            write_persisted_chunk_package(package, run_1, allow_local_write=True)
            write_persisted_chunk_package(package, run_2, allow_local_write=True)

            for filename in PACKAGE_FILENAMES:
                self.assertEqual((run_1 / filename).read_bytes(), (run_2 / filename).read_bytes())
                self.assertEqual(_sha256(run_1 / filename), _sha256(run_2 / filename))

    def test_one_byte_difference_fails_determinism_comparison(self):
        from runpy import run_path

        cli_globals = run_path(str(TOOL_SCRIPT))
        compare_package_dirs = cli_globals["_compare_package_dirs"]
        adapter_result = run_structured_document_adapter(ARTIFACT, MANIFEST, source_path=SOURCE)
        package = build_package_from_adapter_result(adapter_result)
        with tempfile.TemporaryDirectory() as tmp:
            run_1 = Path(tmp) / "run_1"
            run_2 = Path(tmp) / "run_2"
            write_persisted_chunk_package(package, run_1, allow_local_write=True)
            write_persisted_chunk_package(package, run_2, allow_local_write=True)
            target = run_2 / PERSISTENCE_REPORT_FILENAME
            target.write_bytes(target.read_bytes() + b" ")

            comparison = compare_package_dirs(run_1, run_2)

        self.assertFalse(comparison[PERSISTENCE_REPORT_FILENAME]["bytes_match"])
        self.assertFalse(comparison[PERSISTENCE_REPORT_FILENAME]["sha256_match"])

    def test_sanitized_result_is_deterministic_private_and_serializable(self):
        result = run_real_parser_sample_gate(
            artifact_path=ARTIFACT,
            manifest_path=MANIFEST,
            source_path=SOURCE,
        )

        first = sanitized_gate_report_bytes(result)
        second = sanitized_gate_report_bytes(result)
        data = json.loads(first.decode("utf-8"))
        payload = first.decode("utf-8")

        self.assertEqual(first, second)
        self.assertTrue(payload.endswith("\n"))
        self.assertNotIn(SOURCE.read_text(encoding="utf-8").strip(), payload)
        self.assertNotIn(str(ROOT), payload)
        self.assertNotIn("Aspire5 15 i7 4G2050", payload)
        self.assertNotIn("migration_dry_run", payload)
        self.assertEqual(data["authorization"]["full_corpus_ingestion"], False)
        self.assertEqual(data["authorization"]["embedding_generation"], False)
        self.assertEqual(data["authorization"]["astra_rebuild"], False)
        self.assertEqual(data["authorization"]["faiss_rebuild"], False)
        json.dumps(real_parser_sample_gate_result_to_dict(result), sort_keys=True)

    def test_cli_pass_review_fail_and_permission_behavior(self):
        self.assertTrue(TOOL_SCRIPT.exists())
        completed = _run_cli(["--artifact", str(ARTIFACT), "--manifest", str(MANIFEST), "--source", str(SOURCE)])
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Gate outcome: PASS", completed.stdout)
        self.assertIn("No runtime ingestion.", completed.stdout)
        self.assertIn("No embeddings.", completed.stdout)
        self.assertIn("No Astra.", completed.stdout)
        self.assertIn("No FAISS.", completed.stdout)
        self.assertIn("Local outputs not written.", completed.stdout)

        with tempfile.TemporaryDirectory() as tmp:
            artifact, manifest, source = _fixture_with_figure_block(tmp)
            review = _run_cli(
                [
                    "--artifact",
                    str(artifact),
                    "--manifest",
                    str(manifest),
                    "--source",
                    str(source),
                    "--approve-adapter-warning",
                    "BLOCK_CONTENT_TYPE_SKIPPED",
                    "--allow-review",
                ]
            )
        self.assertEqual(review.returncode, 2, review.stdout + review.stderr)
        self.assertIn("Gate outcome: REVIEW", review.stdout)

        with tempfile.TemporaryDirectory() as tmp:
            artifact, manifest, source = _copy_fixture(tmp)
            source.write_text("changed", encoding="utf-8")
            fail = _run_cli(["--artifact", str(artifact), "--manifest", str(manifest), "--source", str(source)])
        self.assertEqual(fail.returncode, 1)
        self.assertIn("Gate outcome: FAIL", fail.stdout)

    def test_cli_local_write_reports_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "gate"
            completed = _run_cli(
                [
                    "--artifact",
                    str(ARTIFACT),
                    "--manifest",
                    str(MANIFEST),
                    "--source",
                    str(SOURCE),
                    "--output-root",
                    str(output_root),
                    "--allow-local-write",
                    "--verify-determinism",
                ]
            )
            report = json.loads((output_root / "local_gate_report.json").read_text(encoding="utf-8"))

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Byte/hash determinism verified: true", completed.stdout)
        self.assertEqual(report["local_write"]["determinism_verified"], True)
        self.assertNotIn(str(output_root), json.dumps(report))

    def test_runtime_ingestion_files_remain_unreferenced(self):
        module_text = (SRC_DIR / "aviationrag" / "ingestion" / "real_parser_sample_gate.py").read_text(encoding="utf-8")
        cli_text = TOOL_SCRIPT.read_text(encoding="utf-8")
        combined = module_text + cli_text

        self.assertNotIn("read_documents.py", combined)
        self.assertNotIn("aviation_chunk_saver.py", combined)
        self.assertNotIn("faiss_indexer.py", combined)

    def test_sanitized_acceptance_fixture_shape(self):
        fixture = _read_json(ACCEPTANCE_FIXTURE)

        self.assertEqual(fixture["gate_schema_name"], "aviationrag-real-parser-sample-gate")
        self.assertEqual(fixture["document_key"], "faa_order_4040_26b")
        self.assertEqual(fixture["source_filename"], "FAA_Order_4040_26B.pdf")
        self.assertEqual(fixture["gate_outcome"], PASS)
        self.assertEqual(fixture["candidate_count"], 920)
        self.assertEqual(fixture["accepted_record_count"], 920)
        self.assertEqual(fixture["rejected_candidate_count"], 0)
        self.assertEqual(fixture["warning_count"], 0)
        self.assertEqual(fixture["authorization"]["full_corpus_ingestion"], False)
        fixture_text = ACCEPTANCE_FIXTURE.read_text(encoding="utf-8")
        self.assertNotIn("C:\\", fixture_text)
        self.assertNotIn("migration_dry_run", fixture_text)
        self.assertTrue(fixture_text.endswith("\n"))


def _copy_fixture(tmp: str) -> tuple[Path, Path, Path]:
    tmp_dir = Path(tmp)
    artifact = tmp_dir / "structured_document.json"
    manifest = tmp_dir / "manifest.json"
    source = tmp_dir / "source.txt"
    artifact.write_text(ARTIFACT.read_text(encoding="utf-8"), encoding="utf-8")
    manifest.write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    source.write_text(SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    return artifact, manifest, source


def _fixture_with_figure_block(tmp: str) -> tuple[Path, Path, Path]:
    artifact, manifest, source = _copy_fixture(tmp)
    data = _read_json(artifact)
    data["blocks"][0]["block_type"] = "figure"
    _write_artifact_and_manifest(artifact, manifest, source, data)
    return artifact, manifest, source


def _write_artifact_and_manifest(
    artifact_path: Path,
    manifest_path: Path,
    source_path: Path,
    artifact: dict,
) -> None:
    _write_json(artifact_path, artifact)
    manifest = _read_json(manifest_path)
    manifest["artifacts"][0]["source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest["artifacts"][0]["artifact_sha256"] = hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    manifest["artifacts"][0]["schema_name"] = artifact.get("schema_name")
    manifest["artifacts"][0]["schema_version"] = artifact.get("schema_version")
    manifest["artifacts"][0]["document_id"] = artifact["document"]["document_id"]
    _write_json(manifest_path, manifest)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL_SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
    )


if __name__ == "__main__":
    unittest.main()
