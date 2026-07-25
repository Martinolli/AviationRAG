import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.ingestion.structured_document_adapter import (  # noqa: E402
    FAIL,
    PASS,
    REVIEW,
    load_structured_document_artifact,
    load_techdoc_parser_manifest,
    run_structured_document_adapter,
    structured_document_adapter_result_to_dict,
    structured_document_artifact_integrity_to_dict,
    validate_structured_document_artifact_integrity,
    write_structured_document_adapter_outputs,
)


FIXTURE_DIR = ROOT / "tests" / "fixtures" / "structured_document_adapter"
ARTIFACT = FIXTURE_DIR / "structured_document.json"
MANIFEST = FIXTURE_DIR / "manifest.json"
SOURCE = FIXTURE_DIR / "source.txt"
TOOL_SCRIPT = ROOT / "tools" / "chunking" / "run-structured-document-adapter-dry-run.py"


class StructuredDocumentAdapterTests(unittest.TestCase):
    def test_fixture_passes_with_verified_source(self):
        result = run_structured_document_adapter(ARTIFACT, MANIFEST, source_path=SOURCE)

        self.assertEqual(result.outcome, PASS)
        self.assertEqual(result.document_id, "adapter-fixture-doc")
        self.assertEqual(result.artifact_integrity.artifact_checksum_matches, True)
        self.assertEqual(result.artifact_integrity.source_checksum_matches, True)
        self.assertEqual(result.validator_result.error_count, 0)
        self.assertEqual(result.validator_result.warning_count, 0)
        self.assertEqual(len(result.candidates), 6)
        self.assertEqual(result.summary["content_type_counts"]["table"], 1)
        self.assertEqual(result.summary["content_type_counts"]["warning"], 1)
        self.assertEqual(result.summary["provenance_status_counts"], {"structured": 6})

    def test_candidate_fields_preserve_raw_text_and_links(self):
        result = run_structured_document_adapter(ARTIFACT, MANIFEST, source_path=SOURCE)
        candidates = {candidate.chunk_candidate_id: candidate for candidate in result.candidates}

        paragraph = candidates["adapter-fixture-doc:chunk:para-1"]
        self.assertEqual(paragraph.text, "Synthetic paragraph keeps exact source text.")
        self.assertEqual(paragraph.section_path, ("1 Scope",))
        self.assertEqual(paragraph.printed_page_labels, ("1",))

        table = candidates["adapter-fixture-doc:chunk:table-block-1"]
        self.assertEqual(table.content_type, "table")
        self.assertEqual(table.table_ids, ("table-1",))

        warning = candidates["adapter-fixture-doc:chunk:warning-1"]
        self.assertEqual(warning.content_type, "warning")
        self.assertEqual(warning.source_block_ids, ("warning-block-1",))
        self.assertEqual(warning.pdf_page_index_start, 1)
        self.assertEqual(warning.pdf_page_index_end, 1)
        self.assertNotIn("adapter-fixture-doc:chunk:warning-block-1", candidates)

        reference = candidates["adapter-fixture-doc:chunk:xref-block-1"]
        self.assertEqual(reference.cross_reference_ids, ("xref-external-1", "xref-section-2"))

    def test_no_source_is_review_only_and_partial_provenance(self):
        result = run_structured_document_adapter(ARTIFACT, MANIFEST)

        self.assertEqual(result.outcome, REVIEW)
        self.assertEqual(result.artifact_integrity.source_checksum_matches, None)
        self.assertTrue(any(issue.code == "SOURCE_CHECKSUM_NOT_VERIFIED" for issue in result.issues))
        self.assertEqual(result.summary["provenance_status_counts"], {"structured_partial": 6})

    def test_source_checksum_mismatch_fails_before_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_source = Path(tmp) / "source.txt"
            bad_source.write_text("changed source bytes", encoding="utf-8")

            result = run_structured_document_adapter(ARTIFACT, MANIFEST, source_path=bad_source)

        self.assertEqual(result.outcome, FAIL)
        self.assertEqual(result.candidates, ())
        self.assertTrue(any(issue.code == "SOURCE_CHECKSUM_MISMATCH" for issue in result.issues))

    def test_manifest_artifact_checksum_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            artifact = tmp_dir / "structured_document.json"
            manifest = tmp_dir / "manifest.json"
            source = tmp_dir / "source.txt"
            artifact.write_text(ARTIFACT.read_text(encoding="utf-8"), encoding="utf-8")
            source.write_text(SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
            manifest_data = load_techdoc_parser_manifest(MANIFEST)
            manifest_data["artifacts"][0]["artifact_sha256"] = "0" * 64
            manifest.write_text(json.dumps(manifest_data), encoding="utf-8")

            result = run_structured_document_adapter(artifact, manifest, source_path=source)

        self.assertEqual(result.outcome, FAIL)
        self.assertEqual(result.candidates, ())
        self.assertTrue(any(issue.code == "ARTIFACT_CHECKSUM_MISMATCH" for issue in result.issues))

    def test_unapproved_validator_warning_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, manifest, source = _copy_fixture(tmp)
            data = load_structured_document_artifact(artifact)
            data["pages"][1]["printed_page_label"] = "1"
            _write_artifact_and_manifest(artifact, manifest, source, data)

            result = run_structured_document_adapter(artifact, manifest, source_path=source)

        self.assertEqual(result.outcome, FAIL)
        self.assertTrue(
            any(issue.code == "VALIDATOR_WARNING_UNAPPROVED" for issue in result.issues),
            structured_document_adapter_result_to_dict(result),
        )

    def test_approved_validator_warning_is_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, manifest, source = _copy_fixture(tmp)
            data = load_structured_document_artifact(artifact)
            data["pages"][1]["printed_page_label"] = "1"
            _write_artifact_and_manifest(artifact, manifest, source, data)

            result = run_structured_document_adapter(
                artifact,
                manifest,
                source_path=source,
                approved_warning_codes={"PRINTED_PAGE_LABEL_AMBIGUOUS"},
            )

        self.assertEqual(result.outcome, REVIEW)
        self.assertTrue(any(issue.code == "VALIDATOR_WARNING_APPROVED" for issue in result.issues))

    def test_resolved_cross_reference_can_target_figure_entity(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, manifest, source = _copy_fixture(tmp)
            data = load_structured_document_artifact(artifact)
            data["cross_references"][0]["target_id"] = "figure-1"
            data["cross_references"][0]["reference_type"] = "figure"
            _write_artifact_and_manifest(artifact, manifest, source, data)

            result = run_structured_document_adapter(artifact, manifest, source_path=source)

        self.assertEqual(result.outcome, PASS)
        self.assertFalse(any(issue.code == "CROSS_REFERENCE_TARGET_UNKNOWN" for issue in result.issues))

    def test_metadata_blocks_are_skipped_without_candidate_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact, manifest, source = _copy_fixture(tmp)
            data = load_structured_document_artifact(artifact)
            data["blocks"][0]["block_type"] = "metadata"
            _write_artifact_and_manifest(artifact, manifest, source, data)

            result = run_structured_document_adapter(artifact, manifest, source_path=source)

        self.assertEqual(result.outcome, PASS)
        self.assertFalse(any(issue.code == "BLOCK_CONTENT_TYPE_SKIPPED" for issue in result.issues))

    def test_integrity_result_is_json_serializable_and_does_not_mutate_inputs(self):
        artifact = load_structured_document_artifact(ARTIFACT)
        manifest = load_techdoc_parser_manifest(MANIFEST)
        before_artifact = copy.deepcopy(artifact)
        before_manifest = copy.deepcopy(manifest)

        integrity = validate_structured_document_artifact_integrity(
            artifact,
            manifest,
            artifact_path=ARTIFACT,
            manifest_path=MANIFEST,
            source_path=SOURCE,
        )

        self.assertEqual(artifact, before_artifact)
        self.assertEqual(manifest, before_manifest)
        json.dumps(structured_document_artifact_integrity_to_dict(integrity))

    def test_local_writer_requires_explicit_permission(self):
        result = run_structured_document_adapter(ARTIFACT, MANIFEST, source_path=SOURCE)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(PermissionError):
                write_structured_document_adapter_outputs(result, tmp)

    def test_local_writer_outputs_report_candidate_jsonl_and_integrity(self):
        result = run_structured_document_adapter(ARTIFACT, MANIFEST, source_path=SOURCE)
        with tempfile.TemporaryDirectory() as tmp:
            write_result = write_structured_document_adapter_outputs(
                result,
                tmp,
                allow_local_write=True,
            )

            candidate_lines = Path(write_result.candidates_output_path).read_text(
                encoding="utf-8"
            ).splitlines()
            report = json.loads(Path(write_result.report_output_path).read_text(encoding="utf-8"))
            integrity = json.loads(
                Path(write_result.integrity_output_path).read_text(encoding="utf-8")
            )

        self.assertEqual(len(candidate_lines), 6)
        self.assertEqual(report["outcome"], PASS)
        self.assertEqual(integrity["artifact_checksum_matches"], True)

    def test_cli_passes_fixture_without_writing_outputs(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(TOOL_SCRIPT),
                "--artifact",
                str(ARTIFACT),
                "--manifest",
                str(MANIFEST),
                "--source",
                str(SOURCE),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Outcome: PASS", completed.stdout)
        self.assertIn("Candidate count: 6", completed.stdout)
        self.assertIn("Adapter outputs not written.", completed.stdout)


def _copy_fixture(tmp: str) -> tuple[Path, Path, Path]:
    tmp_dir = Path(tmp)
    artifact = tmp_dir / "structured_document.json"
    manifest = tmp_dir / "manifest.json"
    source = tmp_dir / "source.txt"
    artifact.write_text(ARTIFACT.read_text(encoding="utf-8"), encoding="utf-8")
    manifest.write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    source.write_text(SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    return artifact, manifest, source


def _write_artifact_and_manifest(
    artifact_path: Path,
    manifest_path: Path,
    source_path: Path,
    artifact: dict,
) -> None:
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    manifest = load_techdoc_parser_manifest(manifest_path)
    manifest["artifacts"][0]["source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest["artifacts"][0]["artifact_sha256"] = hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
