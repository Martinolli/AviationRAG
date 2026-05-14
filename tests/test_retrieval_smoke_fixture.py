import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.evaluation.smoke_fixture import (  # noqa: E402
    RetrievalEvaluationCase,
    load_retrieval_evaluation_cases,
    summarize_retrieval_evaluation_cases,
    validate_retrieval_evaluation_case,
    validate_retrieval_evaluation_dataset,
)


FIXTURE_PATH = ROOT / "data" / "sample_documents" / "sample_retrieval_eval.jsonl"


class TestRetrievalSmokeFixture(unittest.TestCase):
    def test_loads_valid_jsonl_fixture(self):
        cases = load_retrieval_evaluation_cases(FIXTURE_PATH)

        self.assertGreaterEqual(len(cases), 10)
        self.assertTrue(all(isinstance(case, RetrievalEvaluationCase) for case in cases))

    def test_valid_case_has_no_issues(self):
        case = RetrievalEvaluationCase(
            evaluation_id="eval_inline_001",
            category="compliance",
            question="Which fake document should be retrieved?",
            expected_document_id="doc_sample_faa_ac_001",
            expected_chunk_ids=["doc_sample_faa_ac_001_chunk_0001"],
            expected_keywords=["fake document"],
            expected_behavior="retrieve_relevant_context",
            minimum_expected_rank=3,
            notes="Fake sample benchmark only.",
        )

        self.assertEqual(validate_retrieval_evaluation_case(case), [])

    def test_duplicate_evaluation_id_detection(self):
        cases = [
            RetrievalEvaluationCase(
                evaluation_id="eval_dup",
                category="unsupported",
                question="First fake question?",
                expected_behavior="insufficient_evidence",
            ),
            RetrievalEvaluationCase(
                evaluation_id="eval_dup",
                category="unsupported",
                question="Second fake question?",
                expected_behavior="insufficient_evidence",
            ),
        ]

        self.assertIn("Duplicate evaluation_id: eval_dup", validate_retrieval_evaluation_dataset(cases))

    def test_invalid_category_detection(self):
        case = RetrievalEvaluationCase(
            evaluation_id="eval_invalid_category",
            category="real_ops",
            question="Fake question?",
            expected_behavior="insufficient_evidence",
        )

        issues = validate_retrieval_evaluation_case(case)

        self.assertTrue(any("Invalid category" in issue for issue in issues))

    def test_invalid_expected_behavior_detection(self):
        case = RetrievalEvaluationCase(
            evaluation_id="eval_invalid_behavior",
            category="unsupported",
            question="Fake question?",
            expected_behavior="answer_directly",
        )

        issues = validate_retrieval_evaluation_case(case)

        self.assertTrue(any("Invalid expected_behavior" in issue for issue in issues))

    def test_invalid_rank_detection(self):
        case = RetrievalEvaluationCase(
            evaluation_id="eval_invalid_rank",
            category="unsupported",
            question="Fake question?",
            expected_behavior="insufficient_evidence",
            minimum_expected_rank=0,
        )

        issues = validate_retrieval_evaluation_case(case)

        self.assertTrue(any("minimum_expected_rank" in issue for issue in issues))

    def test_malformed_jsonl_raises_value_error_with_line_number(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.jsonl"
            path.write_text('{"evaluation_id": "ok"}\n{"bad":\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "line 2"):
                load_retrieval_evaluation_cases(path)

    def test_summary_generation(self):
        cases = load_retrieval_evaluation_cases(FIXTURE_PATH)

        summary = summarize_retrieval_evaluation_cases(cases)

        self.assertGreaterEqual(summary["total_cases"], 10)
        self.assertIn("compliance", summary["categories"])
        self.assertIn("retrieve_relevant_context", summary["behavior_counts"])
        self.assertEqual(summary["duplicate_ids"], [])
        self.assertEqual(summary["invalid_cases"], [])
        self.assertGreaterEqual(summary["max_expected_rank"], 3)

    def test_fixture_contains_at_least_ten_cases(self):
        lines = [line for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines() if line]

        self.assertGreaterEqual(len(lines), 10)

    def test_fixture_is_fake_sample_only(self):
        cases = [json.loads(line) for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines()]

        for case in cases:
            self.assertTrue(case["evaluation_id"].startswith("eval_sample_"))
            self.assertIn("Fake sample benchmark only", case["notes"])
            expected_document_id = case.get("expected_document_id")
            if expected_document_id:
                self.assertTrue(expected_document_id.startswith("doc_sample_"))
            for chunk_id in case.get("expected_chunk_ids", []):
                self.assertIn("_chunk_", chunk_id)
                self.assertTrue(chunk_id.startswith("doc_sample_"))


if __name__ == "__main__":
    unittest.main()
