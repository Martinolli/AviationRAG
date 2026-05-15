import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.evaluation.reporting import (  # noqa: E402
    evaluation_case_result_to_dict,
    evaluation_summary_to_dict,
    render_markdown_report,
    write_json_report,
    write_markdown_report,
)
from aviationrag.evaluation.retrieval_harness import (  # noqa: E402
    EvaluationCaseResult,
    EvaluationSummary,
)


def _summary(issues=None):
    return EvaluationSummary(
        total_cases=2,
        passed_cases=1,
        failed_cases=1,
        pass_rate=0.5,
        category_counts={"compliance": 1, "unsupported": 1},
        behavior_counts={
            "retrieve_relevant_context": 1,
            "insufficient_evidence": 1,
        },
        issues=issues or [],
    )


def _case_result(evaluation_id="eval_inline_001", passed=True, issues=None):
    return EvaluationCaseResult(
        evaluation_id=evaluation_id,
        passed=passed,
        expected_behavior="retrieve_relevant_context",
        matched_document=passed,
        matched_chunk=passed,
        met_rank_requirement=passed,
        issues=issues or [],
        metadata={"sample": True},
    )


class TestRetrievalReporting(unittest.TestCase):
    def test_evaluation_case_result_to_dict_returns_serializable_dict(self):
        result = _case_result()
        data = evaluation_case_result_to_dict(result)

        self.assertEqual(data["evaluation_id"], "eval_inline_001")
        self.assertTrue(data["passed"])
        json.dumps(data)

    def test_evaluation_summary_to_dict_returns_serializable_dict(self):
        data = evaluation_summary_to_dict(_summary())

        self.assertEqual(data["total_cases"], 2)
        self.assertEqual(data["pass_rate"], 0.5)
        json.dumps(data)

    def test_render_markdown_report_includes_core_sections(self):
        markdown = render_markdown_report(
            _summary(),
            [_case_result()],
            title="Fake Retrieval Report",
        )

        self.assertIn("# Fake Retrieval Report", markdown)
        self.assertIn("## Summary", markdown)
        self.assertIn("50.00%", markdown)
        self.assertIn("eval_inline_001", markdown)
        self.assertIn("## Case Results", markdown)

    def test_write_json_report_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "report.json"

            write_json_report(path, _summary(), [_case_result()])

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["summary"]["total_cases"], 2)
            self.assertEqual(
                payload["case_results"][0]["evaluation_id"],
                "eval_inline_001",
            )

    def test_write_markdown_report_writes_markdown(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "report.md"

            write_markdown_report(path, _summary(), [_case_result()])

            markdown = path.read_text(encoding="utf-8")
            self.assertIn("# Retrieval Evaluation Report", markdown)
            self.assertIn("| Total cases | 2 |", markdown)

    def test_report_handles_empty_issues(self):
        markdown = render_markdown_report(_summary(), [_case_result()])

        self.assertIn("No issues reported.", markdown)

    def test_report_handles_failed_cases_with_issues(self):
        markdown = render_markdown_report(
            _summary(issues=["eval_fail: Expected document was not found."]),
            [
                _case_result(
                    evaluation_id="eval_fail",
                    passed=False,
                    issues=["Expected document was not found."],
                )
            ],
        )

        self.assertIn("eval_fail: Expected document was not found.", markdown)
        self.assertIn("Expected document was not found.", markdown)
        self.assertIn("| eval_fail | false |", markdown)

    def test_no_real_retrieval_astra_or_faiss_is_used(self):
        forbidden_modules = {"faiss", "cassandra", "openai"}
        before = set(sys.modules)

        render_markdown_report(_summary(), [_case_result()])

        imported_during_test = set(sys.modules) - before
        self.assertFalse(forbidden_modules & imported_during_test)


if __name__ == "__main__":
    unittest.main()
