import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.ingestion.structured_document_validator import (  # noqa: E402
    load_structured_document,
    structured_document_validation_result_to_dict,
    validate_structured_document,
    validate_structured_document_file,
)


SAMPLE_STRUCTURED_DOCUMENT = ROOT / "data" / "sample_documents" / "sample_structured_document.json"
TOOL_SCRIPT = ROOT / "tools" / "chunking" / "validate-structured-document.py"


class StructuredDocumentValidatorTests(unittest.TestCase):
    def test_valid_minimal_structured_document_passes(self):
        result = validate_structured_document(_valid_document())

        self.assertTrue(result.is_valid)
        self.assertEqual(result.error_count, 0)
        self.assertEqual(result.warning_count, 0)

    def test_d4_sample_fixture_loads_and_validates(self):
        document = load_structured_document(SAMPLE_STRUCTURED_DOCUMENT)
        result = validate_structured_document(document)

        self.assertTrue(result.is_valid)
        self.assertEqual(result.schema_name, "techdoc-structured-document")
        self.assertEqual(result.schema_version, "0.1.0")

    def test_result_is_json_serializable(self):
        result = validate_structured_document(_valid_document())

        json.dumps(structured_document_validation_result_to_dict(result))

    def test_input_object_is_not_mutated(self):
        document = _valid_document()
        before = copy.deepcopy(document)

        validate_structured_document(document)

        self.assertEqual(document, before)

    def test_summary_counts_are_correct(self):
        document = _valid_document()
        document["figures"] = [{"figure_id": "fig-1", "page_start": 1, "caption": "Synthetic figure."}]
        document["equations"] = [{"equation_id": "eq-1", "page_start": 1, "raw_text": "a=b"}]
        document["admonitions"] = [
            {
                "admonition_id": "adm-1",
                "admonition_type": "NOTE",
                "raw_label": "NOTE",
                "body_text": "Synthetic note.",
                "page_start": 1,
            }
        ]
        document["cross_references"] = [
            {
                "reference_id": "xref-1",
                "raw_text": "See Section 1.",
                "resolution_status": "resolved",
                "target_id": "sec-1",
            }
        ]

        result = validate_structured_document(document)

        self.assertEqual(
            result.summary,
            {
                "page_count": 2,
                "block_count": 2,
                "section_count": 1,
                "table_count": 1,
                "figure_count": 1,
                "equation_count": 1,
                "admonition_count": 1,
                "cross_reference_count": 1,
            },
        )

    def test_issue_ordering_is_deterministic(self):
        document = _valid_document()
        document["schema_name"] = ""
        document["schema_version"] = ""
        document["blocks"][0]["char_start"] = 9
        document["blocks"][0]["char_end"] = 1

        first = validate_structured_document(document).issues
        second = validate_structured_document(document).issues

        self.assertEqual(first, second)
        self.assertEqual(first, sorted(first, key=lambda issue: (issue.severity, issue.path, issue.code, issue.message, issue.entity_id or "")))

    def test_root_non_object_is_rejected(self):
        result = validate_structured_document(["not", "an", "object"])  # type: ignore[arg-type]

        self.assert_issue(result, "ROOT_NOT_OBJECT")

    def test_missing_schema_name_is_rejected(self):
        document = _valid_document()
        del document["schema_name"]

        self.assert_issue(validate_structured_document(document), "SCHEMA_NAME_MISSING")

    def test_unsupported_schema_name_is_rejected(self):
        document = _valid_document()
        document["schema_name"] = "unsupported"

        self.assert_issue(validate_structured_document(document), "SCHEMA_NAME_UNSUPPORTED")

    def test_missing_schema_version_is_rejected(self):
        document = _valid_document()
        del document["schema_version"]

        self.assert_issue(validate_structured_document(document), "SCHEMA_VERSION_MISSING")

    def test_unsupported_schema_version_is_rejected(self):
        document = _valid_document()
        document["schema_version"] = "9.9.9"

        self.assert_issue(validate_structured_document(document), "SCHEMA_VERSION_UNSUPPORTED")

    def test_duplicate_page_index_is_rejected(self):
        document = _valid_document()
        document["pages"][1]["pdf_page_index"] = 0

        self.assert_issue(validate_structured_document(document), "PAGE_INDEX_DUPLICATE")

    def test_negative_page_index_is_rejected(self):
        document = _valid_document()
        document["pages"][0]["pdf_page_index"] = -1

        self.assert_issue(validate_structured_document(document), "PAGE_INDEX_INVALID")

    def test_out_of_order_page_index_is_detected(self):
        document = _valid_document()
        document["pages"][0]["pdf_page_index"] = 1
        document["pages"][1]["pdf_page_index"] = 0

        self.assert_issue(validate_structured_document(document), "PAGE_INDEX_OUT_OF_ORDER")

    def test_page_count_mismatch_is_detected(self):
        document = _valid_document()
        document["document"]["page_count"] = 3

        self.assert_issue(validate_structured_document(document), "PAGE_COUNT_MISMATCH")

    def test_missing_printed_page_label_is_allowed(self):
        document = _valid_document()
        document["pages"][0]["printed_page_label"] = None

        self.assert_no_issue(validate_structured_document(document), "PRINTED_PAGE_LABEL_AMBIGUOUS")

    def test_roman_numeral_printed_page_label_is_allowed(self):
        document = _valid_document()
        document["pages"][0]["printed_page_label"] = "iv"

        self.assertTrue(validate_structured_document(document).is_valid)

    def test_duplicate_block_id_is_rejected(self):
        document = _valid_document()
        document["blocks"][1]["block_id"] = "blk-1"

        self.assert_issue(validate_structured_document(document), "BLOCK_ID_DUPLICATE")

    def test_block_referencing_unknown_page_is_rejected(self):
        document = _valid_document()
        document["blocks"][0]["source_span"]["page_start"] = 99
        document["blocks"][0]["source_span"]["page_end"] = 99

        self.assert_issue(validate_structured_document(document), "BLOCK_PAGE_UNKNOWN")

    def test_duplicate_document_block_index_is_rejected(self):
        document = _valid_document()
        document["blocks"][1]["document_block_index"] = 1

        self.assert_issue(validate_structured_document(document), "BLOCK_INDEX_DUPLICATE")

    def test_unsupported_content_type_is_detected(self):
        document = _valid_document()
        document["blocks"][0]["block_type"] = "custom"

        self.assert_issue(validate_structured_document(document), "CONTENT_TYPE_UNSUPPORTED")

    def test_invalid_character_offsets_are_rejected(self):
        document = _valid_document()
        document["blocks"][0]["char_start"] = 10
        document["blocks"][0]["char_end"] = 3

        self.assert_issue(validate_structured_document(document), "CHARACTER_OFFSET_INVALID")

    def test_invalid_bounding_box_is_rejected(self):
        document = _valid_document()
        document["blocks"][0]["bbox"] = [10, 10, 1, 20]

        self.assert_issue(validate_structured_document(document), "BOUNDING_BOX_INVALID")

    def test_unknown_section_parent_is_rejected(self):
        document = _valid_document()
        document["sections"][0]["parent_section_id"] = "missing"

        self.assert_issue(validate_structured_document(document), "SECTION_PARENT_UNKNOWN")

    def test_self_parent_is_rejected(self):
        document = _valid_document()
        document["sections"][0]["parent_section_id"] = "sec-1"

        self.assert_issue(validate_structured_document(document), "SECTION_SELF_PARENT")

    def test_section_cycle_is_rejected(self):
        document = _valid_document()
        document["sections"].append(
            {
                "section_id": "sec-2",
                "level": 2,
                "title": "Synthetic Child",
                "parent_section_id": "sec-1",
                "path": ["1 Synthetic Section", "Synthetic Child"],
            }
        )
        document["sections"][0]["parent_section_id"] = "sec-2"

        self.assert_issue(validate_structured_document(document), "SECTION_CYCLE")

    def test_block_referencing_unknown_section_is_rejected(self):
        document = _valid_document()
        document["blocks"][0]["section_id"] = "missing"

        self.assert_issue(validate_structured_document(document), "BLOCK_SECTION_UNKNOWN")

    def test_unnumbered_heading_is_allowed(self):
        document = _valid_document()
        document["sections"][0]["section_number"] = None
        document["sections"][0]["path"] = ["Synthetic Section"]

        self.assertTrue(validate_structured_document(document).is_valid)

    def test_source_span_page_start_greater_than_page_end_is_rejected(self):
        document = _valid_document()
        document["blocks"][0]["source_span"]["page_start"] = 2
        document["blocks"][0]["source_span"]["page_end"] = 1

        self.assert_issue(validate_structured_document(document), "SOURCE_PAGE_RANGE_INVALID")

    def test_unknown_source_block_id_is_rejected(self):
        document = _valid_document()
        document["blocks"][0]["source_span"]["source_block_ids"] = ["missing"]

        self.assert_issue(validate_structured_document(document), "SOURCE_BLOCK_UNKNOWN")

    def test_duplicate_source_block_id_inside_span_is_detected(self):
        document = _valid_document()
        document["blocks"][0]["source_span"]["source_block_ids"] = ["blk-1", "blk-1"]

        self.assert_issue(validate_structured_document(document), "SOURCE_BLOCK_DUPLICATE", severity="warning")

    def test_missing_optional_offsets_and_bounding_boxes_are_allowed(self):
        document = _valid_document()
        document["blocks"][0]["source_span"].pop("char_start", None)
        document["blocks"][0]["source_span"].pop("bbox", None)

        self.assertTrue(validate_structured_document(document).is_valid)

    def test_confidence_below_zero_is_rejected(self):
        document = _valid_document()
        document["blocks"][0]["structure_confidence"] = -0.1

        self.assert_issue(validate_structured_document(document), "CONFIDENCE_OUT_OF_RANGE")

    def test_confidence_above_one_is_rejected(self):
        document = _valid_document()
        document["blocks"][0]["structure_confidence"] = 1.1

        self.assert_issue(validate_structured_document(document), "CONFIDENCE_OUT_OF_RANGE")

    def test_boolean_confidence_is_rejected(self):
        document = _valid_document()
        document["blocks"][0]["structure_confidence"] = True

        self.assert_issue(validate_structured_document(document), "CONFIDENCE_TYPE_INVALID")

    def test_null_confidence_is_allowed(self):
        document = _valid_document()
        document["blocks"][0]["structure_confidence"] = None

        self.assertTrue(validate_structured_document(document).is_valid)

    def test_duplicate_table_id_is_rejected(self):
        document = _valid_document()
        document["tables"] = [_valid_table(), _valid_table()]

        self.assert_issue(validate_structured_document(document), "TABLE_ID_DUPLICATE")

    def test_table_cell_referencing_unknown_column_is_rejected(self):
        document = _valid_document()
        table = _valid_table()
        table["cells"][0]["column_id"] = "missing"
        document["tables"] = [table]

        self.assert_issue(validate_structured_document(document), "TABLE_COLUMN_UNKNOWN")

    def test_duplicate_row_id_is_rejected(self):
        document = _valid_document()
        table = _valid_table()
        table["rows"].append({"row_id": "row-1"})
        document["tables"] = [table]

        self.assert_issue(validate_structured_document(document), "TABLE_ROW_ID_DUPLICATE")

    def test_invalid_continuation_flag_is_rejected(self):
        document = _valid_document()
        table = _valid_table()
        table["continues_to_next_page"] = "yes"
        document["tables"] = [table]

        self.assert_issue(validate_structured_document(document), "TABLE_CONTINUATION_INVALID")

    def test_figure_referencing_unknown_page_is_rejected(self):
        document = _valid_document()
        document["figures"] = [{"figure_id": "fig-1", "page_start": 99, "caption": "Synthetic figure."}]

        self.assert_issue(validate_structured_document(document), "FIGURE_PAGE_UNKNOWN")

    def test_missing_figure_caption_is_warning_only(self):
        document = _valid_document()
        document["figures"] = [{"figure_id": "fig-1", "page_start": 1}]
        result = validate_structured_document(document)

        self.assertTrue(result.is_valid)
        self.assert_issue(result, "FIGURE_CAPTION_MISSING", severity="warning")

    def test_equation_without_raw_representation_is_rejected(self):
        document = _valid_document()
        document["equations"] = [{"equation_id": "eq-1", "page_start": 1}]

        self.assert_issue(validate_structured_document(document), "EQUATION_RAW_MISSING")

    def test_unsupported_admonition_type_is_rejected(self):
        document = _valid_document()
        admonition = _valid_admonition()
        admonition["admonition_type"] = "ALERT"
        document["admonitions"] = [admonition]

        self.assert_issue(validate_structured_document(document), "ADMONITION_TYPE_UNSUPPORTED")

    def test_empty_admonition_body_is_rejected(self):
        document = _valid_document()
        admonition = _valid_admonition()
        admonition["body_text"] = ""
        document["admonitions"] = [admonition]

        self.assert_issue(validate_structured_document(document), "ADMONITION_BODY_MISSING")

    def test_unknown_admonition_source_block_is_rejected(self):
        document = _valid_document()
        admonition = _valid_admonition()
        admonition["source_block_ids"] = ["missing"]
        document["admonitions"] = [admonition]

        self.assert_issue(validate_structured_document(document), "ADMONITION_SOURCE_BLOCK_UNKNOWN")

    def test_unknown_admonition_type_is_allowed(self):
        document = _valid_document()
        admonition = _valid_admonition()
        admonition["admonition_type"] = "UNKNOWN_ADMONITION"
        document["admonitions"] = [admonition]

        self.assertTrue(validate_structured_document(document).is_valid)

    def test_unresolved_reference_is_allowed(self):
        document = _valid_document()
        document["cross_references"] = [
            {
                "reference_id": "xref-1",
                "raw_text": "See synthetic reference.",
                "resolution_status": "unresolved",
            }
        ]

        self.assertTrue(validate_structured_document(document).is_valid)

    def test_resolved_reference_with_unknown_target_is_rejected(self):
        document = _valid_document()
        document["cross_references"] = [
            {
                "reference_id": "xref-1",
                "raw_text": "See synthetic reference.",
                "resolution_status": "resolved",
                "target_id": "missing",
            }
        ]

        self.assert_issue(validate_structured_document(document), "CROSS_REFERENCE_TARGET_UNKNOWN")

    def test_external_reference_without_local_target_is_allowed(self):
        document = _valid_document()
        document["cross_references"] = [
            {
                "reference_id": "xref-1",
                "raw_text": "See external synthetic reference.",
                "resolution_status": "external",
            }
        ]

        self.assertTrue(validate_structured_document(document).is_valid)

    def test_cli_tool_exists(self):
        self.assertTrue(TOOL_SCRIPT.exists())

    def test_cli_validates_sample_fixture_successfully(self):
        completed = subprocess.run(
            [sys.executable, str(TOOL_SCRIPT)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Valid: True", completed.stdout)

    def test_cli_refuses_report_writing_unless_explicitly_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            completed = subprocess.run(
                [sys.executable, str(TOOL_SCRIPT), "--report", str(report_path)],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse(report_path.exists())

    def test_cli_writes_report_to_tmp_path_when_allowed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(TOOL_SCRIPT),
                    "--report",
                    str(report_path),
                    "--allow-report-write",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertTrue(report["is_valid"])

    def test_strict_warnings_exits_nonzero(self):
        document = _valid_document()
        document["figures"] = [{"figure_id": "fig-1", "page_start": 1}]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "warning_fixture.json"
            input_path.write_text(json.dumps(document), encoding="utf-8")

            completed = subprocess.run(
                [sys.executable, str(TOOL_SCRIPT), "--input", str(input_path), "--strict-warnings"],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 1)
            self.assertIn("Warning count: 1", completed.stdout)

    def assert_issue(self, result, code, severity="error"):
        self.assertTrue(
            any(issue.code == code and issue.severity == severity for issue in result.issues),
            structured_document_validation_result_to_dict(result),
        )

    def assert_no_issue(self, result, code):
        self.assertFalse(
            any(issue.code == code for issue in result.issues),
            structured_document_validation_result_to_dict(result),
        )


def _valid_document():
    return {
        "schema_name": "techdoc-structured-document",
        "schema_version": "0.1.0",
        "parser_name": "synthetic-parser",
        "parser_version": "0.1",
        "document": {
            "document_id": "synthetic-doc-001",
            "canonical_title": "Synthetic Document",
            "filename": "synthetic-document.pdf",
            "source_hash": "sha256:synthetic",
            "page_count": 2,
            "complete_page_sequence": True,
        },
        "pages": [
            {
                "page_id": "page-1",
                "pdf_page_index": 0,
                "page_number": 1,
                "printed_page_label": "i",
            },
            {
                "page_id": "page-2",
                "pdf_page_index": 1,
                "page_number": 2,
                "printed_page_label": "1",
            },
        ],
        "sections": [
            {
                "section_id": "sec-1",
                "level": 1,
                "section_number": "1",
                "title": "Synthetic Section",
                "parent_section_id": None,
                "path": ["1 Synthetic Section"],
            }
        ],
        "blocks": [
            {
                "block_id": "blk-1",
                "block_type": "paragraph",
                "text": "Synthetic paragraph.",
                "document_block_index": 1,
                "page_block_index": 0,
                "section_id": "sec-1",
                "source_span": {
                    "page_start": 1,
                    "page_end": 1,
                    "pdf_page_index_start": 0,
                    "pdf_page_index_end": 0,
                },
            },
            {
                "block_id": "tbl-1",
                "block_type": "table",
                "table_id": "tbl-1",
                "text": "Synthetic table.",
                "document_block_index": 2,
                "page_block_index": 1,
                "section_id": "sec-1",
                "source_span": {
                    "page_start": 1,
                    "page_end": 1,
                    "pdf_page_index_start": 0,
                    "pdf_page_index_end": 0,
                },
                "table_context": {
                    "column_headers": ["Column A", "Column B"],
                    "continues_to_next_page": False,
                },
            },
        ],
    }


def _valid_table():
    return {
        "table_id": "table-root-1",
        "page_start": 1,
        "page_end": 1,
        "columns": [{"column_id": "col-1"}, {"column_id": "col-2"}],
        "rows": [{"row_id": "row-1"}],
        "cells": [{"row_id": "row-1", "column_id": "col-1", "text": "Synthetic cell"}],
        "continues_to_next_page": False,
    }


def _valid_admonition():
    return {
        "admonition_id": "adm-root-1",
        "admonition_type": "WARNING",
        "raw_label": "WARNING",
        "body_text": "Synthetic warning.",
        "page_start": 1,
        "section_id": "sec-1",
        "source_block_ids": ["blk-1"],
    }


if __name__ == "__main__":
    unittest.main()
