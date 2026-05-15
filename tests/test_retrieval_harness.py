import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.evaluation.retrieval_harness import (  # noqa: E402
    RetrievalResult,
    evaluate_case,
    evaluate_cases,
    retrieval_result_from_dict,
    retrieval_result_to_dict,
    summarize_evaluation_results,
)
from aviationrag.evaluation.smoke_fixture import RetrievalEvaluationCase  # noqa: E402


def _case(
    evaluation_id="eval_inline_001",
    expected_document_id="doc_sample_faa_ac_001",
    expected_chunk_ids=None,
    expected_behavior="retrieve_relevant_context",
    minimum_expected_rank=3,
):
    return RetrievalEvaluationCase(
        evaluation_id=evaluation_id,
        category="compliance",
        question="Which fake document should be retrieved?",
        expected_document_id=expected_document_id,
        expected_chunk_ids=expected_chunk_ids or [],
        expected_keywords=["fake"],
        expected_behavior=expected_behavior,
        minimum_expected_rank=minimum_expected_rank,
        notes="Fake sample benchmark only.",
    )


class TestRetrievalHarness(unittest.TestCase):
    def test_evaluate_case_passes_when_expected_document_is_within_rank(self):
        result = evaluate_case(
            _case(),
            [
                RetrievalResult(
                    document_id="doc_sample_faa_ac_001",
                    chunk_id="doc_sample_faa_ac_001_chunk_0001",
                    score=0.91,
                    rank=2,
                    metadata={},
                )
            ],
        )

        self.assertTrue(result.passed)
        self.assertTrue(result.matched_document)
        self.assertTrue(result.met_rank_requirement)

    def test_evaluate_case_fails_when_expected_document_is_outside_rank(self):
        result = evaluate_case(
            _case(minimum_expected_rank=3),
            [
                RetrievalResult(
                    document_id="doc_sample_faa_ac_001",
                    chunk_id="doc_sample_faa_ac_001_chunk_0001",
                    score=0.91,
                    rank=4,
                    metadata={},
                )
            ],
        )

        self.assertFalse(result.passed)
        self.assertFalse(result.matched_document)
        self.assertTrue(any("Expected document" in issue for issue in result.issues))

    def test_evaluate_case_passes_when_expected_chunk_is_within_rank(self):
        result = evaluate_case(
            _case(expected_chunk_ids=["doc_sample_faa_ac_001_chunk_0001"]),
            [
                RetrievalResult(
                    document_id="doc_sample_faa_ac_001",
                    chunk_id="doc_sample_faa_ac_001_chunk_0001",
                    score=0.88,
                    rank=1,
                    metadata={},
                )
            ],
        )

        self.assertTrue(result.passed)
        self.assertTrue(result.matched_chunk)

    def test_evaluate_case_fails_when_expected_chunk_is_missing(self):
        result = evaluate_case(
            _case(expected_chunk_ids=["doc_sample_faa_ac_001_chunk_0001"]),
            [
                RetrievalResult(
                    document_id="doc_sample_faa_ac_001",
                    chunk_id="doc_sample_other_chunk_0001",
                    score=0.88,
                    rank=1,
                    metadata={},
                )
            ],
        )

        self.assertFalse(result.passed)
        self.assertFalse(result.matched_chunk)
        self.assertTrue(any("Expected chunk" in issue for issue in result.issues))

    def test_insufficient_evidence_passes_with_empty_results(self):
        result = evaluate_case(
            _case(
                evaluation_id="eval_unsupported",
                expected_document_id=None,
                expected_behavior="insufficient_evidence",
                minimum_expected_rank=1,
            ),
            [],
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.expected_behavior, "insufficient_evidence")

    def test_reject_out_of_scope_passes_with_rejected_metadata(self):
        result = evaluate_case(
            _case(
                evaluation_id="eval_rejected",
                expected_document_id=None,
                expected_behavior="reject_out_of_scope",
                minimum_expected_rank=1,
            ),
            [
                RetrievalResult(
                    document_id=None,
                    chunk_id=None,
                    score=None,
                    rank=1,
                    metadata={"rejected": True},
                )
            ],
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.expected_behavior, "reject_out_of_scope")

    def test_evaluate_cases_handles_multiple_cases(self):
        cases = [
            _case(evaluation_id="eval_pass"),
            _case(
                evaluation_id="eval_empty",
                expected_document_id=None,
                expected_behavior="insufficient_evidence",
            ),
        ]
        results = evaluate_cases(
            cases,
            {
                "eval_pass": [
                    RetrievalResult(
                        document_id="doc_sample_faa_ac_001",
                        chunk_id=None,
                        score=0.7,
                        rank=1,
                        metadata={},
                    )
                ],
                "eval_empty": [],
            },
        )

        self.assertEqual(len(results), 2)
        self.assertTrue(all(result.passed for result in results))

    def test_summary_counts_pass_and_fail(self):
        cases = [
            _case(evaluation_id="eval_pass"),
            _case(evaluation_id="eval_fail"),
        ]
        case_results = [
            evaluate_case(
                cases[0],
                [
                    RetrievalResult(
                        document_id="doc_sample_faa_ac_001",
                        chunk_id=None,
                        score=0.7,
                        rank=1,
                        metadata={},
                    )
                ],
            ),
            evaluate_case(cases[1], []),
        ]

        summary = summarize_evaluation_results(cases, case_results)

        self.assertEqual(summary.total_cases, 2)
        self.assertEqual(summary.passed_cases, 1)
        self.assertEqual(summary.failed_cases, 1)
        self.assertEqual(summary.pass_rate, 0.5)
        self.assertIn("compliance", summary.category_counts)
        self.assertTrue(summary.issues)

    def test_retrieval_result_from_dict_handles_missing_optional_fields(self):
        result = retrieval_result_from_dict({"rank": "2"})

        self.assertIsNone(result.document_id)
        self.assertIsNone(result.chunk_id)
        self.assertIsNone(result.score)
        self.assertEqual(result.rank, 2)
        self.assertEqual(result.metadata, {})
        self.assertEqual(retrieval_result_to_dict(result)["rank"], 2)

    def test_no_real_retrieval_astra_or_faiss_is_used(self):
        forbidden_modules = {"faiss", "cassandra", "openai"}
        before = set(sys.modules)

        evaluate_case(
            _case(),
            [
                RetrievalResult(
                    document_id="doc_sample_faa_ac_001",
                    chunk_id=None,
                    score=0.7,
                    rank=1,
                    metadata={},
                )
            ],
        )

        imported_during_test = set(sys.modules) - before
        self.assertFalse(forbidden_modules & imported_during_test)


if __name__ == "__main__":
    unittest.main()
