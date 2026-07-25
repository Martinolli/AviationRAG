import copy
import json
import re
import sys
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.ingestion.persisted_chunk_mapper import (  # noqa: E402
    PersistedChunkCandidateContext,
    PersistedChunkMappingPolicy,
    build_persisted_chunk_id,
    map_candidate_to_persisted_chunk,
)
from aviationrag.ingestion.persisted_chunk_record import (  # noqa: E402
    FORBIDDEN_PERSISTED_FIELDS,
    PERSISTED_CHUNK_SCHEMA_NAME,
    PERSISTED_CHUNK_SCHEMA_VERSION,
    PERSISTED_CONTENT_TYPES,
    SUPPORTED_PERSISTED_CHUNK_SCHEMA_VERSIONS,
    persisted_chunk_record_to_dict,
)
from aviationrag.ingestion.persisted_chunk_validator import (  # noqa: E402
    LIMITATION_CHUNK_SECTION_CROSSING_REVIEW,
    LIMITATION_DUPLICATE_TEXT_LINES,
    LIMITATION_TABLE_CANDIDATE_ONLY,
    LIMITATION_TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE,
    validate_persisted_chunk_record,
)
from aviationrag.ingestion.structured_document_adapter import (  # noqa: E402
    StructuredDocumentChunkCandidate,
)


class PersistedChunkMapperTests(unittest.TestCase):
    def test_schema_constants_are_exact_and_supported(self):
        self.assertEqual(PERSISTED_CHUNK_SCHEMA_NAME, "aviationrag-persisted-chunk")
        self.assertEqual(PERSISTED_CHUNK_SCHEMA_VERSION, "0.1.0")
        self.assertIn("0.1.0", SUPPORTED_PERSISTED_CHUNK_SCHEMA_VERSIONS)

    def test_chunk_id_is_deterministic_namespaced_and_24_hex(self):
        first = build_persisted_chunk_id(
            document_id="doc-1",
            content_type="paragraph",
            content_subtype=None,
            source_block_ids=("blk-1",),
            sequence_key="candidate-1",
        )
        second = build_persisted_chunk_id(
            document_id="doc-1",
            content_type="paragraph",
            content_subtype=None,
            source_block_ids=("blk-1",),
            sequence_key="candidate-1",
        )

        self.assertEqual(first, second)
        self.assertTrue(first.startswith("doc-1:chunk:"))
        self.assertRegex(first.rsplit(":", 1)[1], r"^[0-9a-f]{24}$")

    def test_identity_changes_for_source_entity_type_and_sequence(self):
        base = dict(
            document_id="doc-1",
            content_type="paragraph",
            content_subtype=None,
            source_block_ids=("blk-1",),
            sequence_key="candidate-1",
        )
        original = build_persisted_chunk_id(**base)
        variants = [
            build_persisted_chunk_id(**{**base, "source_block_ids": ("blk-2",)}),
            build_persisted_chunk_id(**{**base, "table_ids": ("tbl-1",)}),
            build_persisted_chunk_id(**{**base, "content_type": "table"}),
            build_persisted_chunk_id(**{**base, "sequence_key": "candidate-2"}),
        ]

        self.assertEqual(len(set([original, *variants])), 5)

    def test_full_provenance_candidate_maps_without_mutation(self):
        candidate = _candidate()
        before = copy.deepcopy(candidate)

        result = map_candidate_to_persisted_chunk(candidate, chunk_index=0)

        self.assertTrue(result.is_accepted, result.issues)
        self.assertEqual(candidate, before)
        record = result.record
        self.assertIsNotNone(record)
        data = persisted_chunk_record_to_dict(record)
        self.assertEqual(data["schema_name"], PERSISTED_CHUNK_SCHEMA_NAME)
        self.assertEqual(data["chunk_index"], 0)
        self.assertEqual(data["record_origin"], "new_structured")
        self.assertEqual(data["provenance_status"], "full_provenance")
        self.assertEqual(data["validation_status"], "valid")
        self.assertNotIn("embedding", json.dumps(data).lower())
        self.assertFalse(set(data).intersection(FORBIDDEN_PERSISTED_FIELDS))

    def test_chunk_index_is_supplied_and_not_identity(self):
        first = map_candidate_to_persisted_chunk(_candidate(), chunk_index=0).record
        second = map_candidate_to_persisted_chunk(_candidate(), chunk_index=4).record

        self.assertEqual(first.chunk_id, second.chunk_id)
        self.assertEqual(first.chunk_index, 0)
        self.assertEqual(second.chunk_index, 4)

    def test_required_provenance_fail_closed(self):
        cases = [
            replace(_candidate(), document_id=""),
            replace(_candidate(), source_filename=None),
            replace(_candidate(), source_checksum="sha256:not-valid"),
            replace(_candidate(), source_block_ids=()),
            replace(_candidate(), page_start=None),
            replace(_candidate(), pdf_page_index_start=None),
            replace(_candidate(), parser_name=None),
        ]

        for candidate in cases:
            result = map_candidate_to_persisted_chunk(candidate, chunk_index=0)
            self.assertFalse(result.is_accepted)
            self.assertIsNone(result.record)

    def test_partial_provenance_rejects_by_default_and_requires_review(self):
        candidate = replace(_candidate(), provenance_status="structured_partial")

        default_result = map_candidate_to_persisted_chunk(candidate, chunk_index=0)
        self.assertFalse(default_result.is_accepted)
        self.assertTrue(any(issue.code == "PARTIAL_PROVENANCE_DISABLED" for issue in default_result.issues))

        no_review = map_candidate_to_persisted_chunk(
            candidate,
            chunk_index=0,
            context=PersistedChunkCandidateContext(
                accepted_limitation_codes=(LIMITATION_CHUNK_SECTION_CROSSING_REVIEW,),
                review_required=False,
            ),
            policy=PersistedChunkMappingPolicy(allow_partial_provenance=True),
        )
        self.assertFalse(no_review.is_accepted)
        self.assertTrue(any(issue.code == "PARTIAL_PROVENANCE_REVIEW_REQUIRED" for issue in no_review.issues))

        accepted = map_candidate_to_persisted_chunk(
            candidate,
            chunk_index=0,
            context=PersistedChunkCandidateContext(
                accepted_limitation_codes=(LIMITATION_CHUNK_SECTION_CROSSING_REVIEW,),
                warning_codes=("PARTIAL_PROVENANCE_REVIEW_REQUIRED",),
                review_required=True,
            ),
            policy=PersistedChunkMappingPolicy(allow_partial_provenance=True),
        )
        self.assertTrue(accepted.is_accepted, accepted.issues)
        self.assertEqual(accepted.record.provenance_status, "partial_provenance")
        self.assertEqual(accepted.record.validation_status, "review_required")

    def test_unknown_and_legacy_provenance_reject(self):
        for status in ("unknown_provenance", "legacy_filename_only", "legacy_adapted"):
            result = map_candidate_to_persisted_chunk(
                replace(_candidate(), provenance_status=status),
                chunk_index=0,
            )
            self.assertFalse(result.is_accepted)
            self.assertTrue(any(issue.code == "PROVENANCE_STATUS_REJECTED" for issue in result.issues))

    def test_content_type_mapping_and_entity_retention(self):
        cases = [
            ("paragraph", {}, "paragraph"),
            ("table", {"table_ids": ("tbl-1",)}, "table"),
            ("figure_caption", {"figure_ids": ("fig-1",)}, "figure_caption"),
            ("other", {"equation_ids": ("eq-1",)}, "equation"),
            ("warning", {"admonition_ids": ("adm-1",)}, "warning"),
            ("caution", {"admonition_ids": ("adm-1",)}, "caution"),
            ("note", {"admonition_ids": ("adm-1",)}, "note"),
            ("procedure", {}, "procedure"),
            ("requirement", {}, "requirement"),
            ("definition", {}, "definition"),
        ]

        for source_type, kwargs, expected_type in cases:
            candidate = replace(_candidate(), content_type=source_type, **kwargs)
            result = map_candidate_to_persisted_chunk(candidate, chunk_index=0)
            self.assertTrue(result.is_accepted, (source_type, result.issues))
            self.assertEqual(result.record.content_type, expected_type)
            self.assertIn(expected_type, PERSISTED_CONTENT_TYPES)

    def test_unknown_content_type_and_heading_reject_by_default(self):
        unknown = map_candidate_to_persisted_chunk(
            replace(_candidate(), content_type="other"),
            chunk_index=0,
        )
        heading = map_candidate_to_persisted_chunk(
            replace(_candidate(), content_type="section"),
            chunk_index=0,
        )

        self.assertFalse(unknown.is_accepted)
        self.assertTrue(any(issue.code == "CONTENT_TYPE_UNSUPPORTED" for issue in unknown.issues))
        self.assertFalse(heading.is_accepted)
        self.assertTrue(any(issue.code == "HEADING_RECORD_DISABLED" for issue in heading.issues))

    def test_heading_maps_only_when_enabled(self):
        result = map_candidate_to_persisted_chunk(
            replace(_candidate(), content_type="section"),
            chunk_index=0,
            policy=PersistedChunkMappingPolicy(include_heading_records=True),
            context=PersistedChunkCandidateContext(review_required=True),
        )

        self.assertTrue(result.is_accepted, result.issues)
        self.assertEqual(result.record.content_type, "reference")

    def test_limitation_registry_policies(self):
        policies = [
            (LIMITATION_CHUNK_SECTION_CROSSING_REVIEW, "paragraph", True),
            (LIMITATION_DUPLICATE_TEXT_LINES, "paragraph", False),
            (LIMITATION_TABLE_CANDIDATE_ONLY, "table", True),
            (LIMITATION_TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE, "table", True),
        ]
        for code, content_type, review_expected in policies:
            result = map_candidate_to_persisted_chunk(
                replace(_candidate(), content_type=content_type, table_ids=("tbl-1",) if content_type == "table" else ()),
                chunk_index=0,
                context=PersistedChunkCandidateContext(
                    accepted_limitation_codes=(code, code),
                    warning_codes=("SYNTHETIC_WARNING",),
                    review_required=review_expected,
                ),
            )
            self.assertTrue(result.is_accepted, (code, result.issues))
            self.assertIn(code, result.record.accepted_limitation_codes)
            self.assertEqual(result.record.review_required, review_expected)

    def test_unknown_limitation_rejects(self):
        result = map_candidate_to_persisted_chunk(
            _candidate(),
            chunk_index=0,
            context=PersistedChunkCandidateContext(accepted_limitation_codes=("UNKNOWN_LIMIT",)),
        )

        self.assertFalse(result.is_accepted)
        self.assertTrue(any(issue.code == "LIMITATION_CODE_UNKNOWN" for issue in result.issues))

    def test_table_limitation_does_not_create_cells(self):
        result = map_candidate_to_persisted_chunk(
            replace(_candidate(), content_type="table", table_ids=("tbl-1",)),
            chunk_index=0,
            context=PersistedChunkCandidateContext(
                accepted_limitation_codes=(LIMITATION_TABLE_CANDIDATE_ONLY,),
                review_required=True,
            ),
        )

        self.assertTrue(result.is_accepted, result.issues)
        serialized = json.dumps(persisted_chunk_record_to_dict(result.record)).lower()
        self.assertNotIn("cells", serialized)
        self.assertNotIn("rows", serialized)

    def test_duplicate_source_blocks_reject_and_same_text_different_blocks_survives(self):
        duplicate = map_candidate_to_persisted_chunk(
            replace(_candidate(), source_block_ids=("blk-1", "blk-1")),
            chunk_index=0,
        )
        first = map_candidate_to_persisted_chunk(_candidate(), chunk_index=0).record
        second = map_candidate_to_persisted_chunk(
            replace(_candidate(), chunk_candidate_id="candidate-2", source_block_ids=("blk-2",)),
            chunk_index=1,
        ).record

        self.assertFalse(duplicate.is_accepted)
        self.assertTrue(any(issue.code == "SOURCE_BLOCK_IDS_DUPLICATE" for issue in duplicate.issues))
        self.assertNotEqual(first.chunk_id, second.chunk_id)
        self.assertEqual(first.text, second.text)

    def test_record_validator_catches_status_consistency_and_forbidden_fields(self):
        record = map_candidate_to_persisted_chunk(_candidate(), chunk_index=0).record
        bad_status = replace(record, validation_status="valid", warning_codes=("WARNING",))
        bad_origin = replace(record, record_origin="legacy_adapted")

        self.assertTrue(any(issue.code == "VALIDATION_STATUS_INCONSISTENT" for issue in validate_persisted_chunk_record(bad_status)))
        self.assertTrue(any(issue.code == "RECORD_ORIGIN_REJECTED" for issue in validate_persisted_chunk_record(bad_origin)))


def _candidate(**overrides):
    data = dict(
        chunk_candidate_id="adapter-fixture-doc:chunk:para-1",
        document_id="adapter-fixture-doc",
        source_filename="source.txt",
        source_checksum="4df3052b53cb7d8060bde4bd8b4f25764271e26c6847f3d9535d72a02dc247ae",
        document_title="Synthetic Structured Adapter Fixture",
        document_number="SYN-ADAPT-001",
        document_revision="A",
        text="Synthetic paragraph keeps exact source text.",
        normalized_text="Synthetic paragraph keeps exact source text.",
        content_type="paragraph",
        page_start=1,
        page_end=1,
        pdf_page_index_start=0,
        pdf_page_index_end=0,
        printed_page_labels=("1",),
        section_id="sec-1",
        section_path=("1 Scope",),
        section_number="1",
        section_title="Scope",
        clause_identifier="1",
        source_block_ids=("para-1",),
        table_ids=(),
        figure_ids=(),
        equation_ids=(),
        admonition_ids=(),
        cross_reference_ids=(),
        parser_name="techdoc-parser",
        parser_version="0.1.0-test",
        extraction_method="techdoc-parser structured-document adapter",
        provenance_status="structured",
    )
    data.update(overrides)
    return StructuredDocumentChunkCandidate(**data)


if __name__ == "__main__":
    unittest.main()
