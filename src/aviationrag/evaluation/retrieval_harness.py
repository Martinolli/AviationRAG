"""Fake/mock retrieval evaluation harness shell.

This module evaluates caller-supplied retrieval result objects against the
sample evaluation case schema. It does not run retrieval, connect to vector
stores, generate embeddings, or read source documents.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from aviationrag.evaluation.smoke_fixture import RetrievalEvaluationCase


LOW_SCORE_THRESHOLD = 0.2


@dataclass(frozen=True)
class RetrievalResult:
    """One fake/mock retrieval result for evaluation scoring."""

    document_id: str | None
    chunk_id: str | None
    score: float | None
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationCaseResult:
    """Scored result for one retrieval evaluation case."""

    evaluation_id: str
    passed: bool
    expected_behavior: str
    matched_document: bool
    matched_chunk: bool
    met_rank_requirement: bool
    issues: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationSummary:
    """Aggregate summary for a retrieval evaluation run."""

    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    category_counts: dict[str, int]
    behavior_counts: dict[str, int]
    issues: list[str] = field(default_factory=list)


def evaluate_case(
    case: RetrievalEvaluationCase,
    results: Sequence[RetrievalResult],
) -> EvaluationCaseResult:
    """Evaluate one case against fake/mock retrieval results."""
    issues = _rank_issues(results)

    if case.expected_behavior == "retrieve_relevant_context":
        return _evaluate_retrieval_case(case, results, issues)
    if case.expected_behavior == "insufficient_evidence":
        return _evaluate_insufficient_evidence_case(case, results, issues)
    if case.expected_behavior == "reject_out_of_scope":
        return _evaluate_reject_out_of_scope_case(case, results, issues)

    issues.append(f"Unknown expected_behavior: {case.expected_behavior}")
    return EvaluationCaseResult(
        evaluation_id=case.evaluation_id,
        passed=False,
        expected_behavior=case.expected_behavior,
        matched_document=False,
        matched_chunk=False,
        met_rank_requirement=False,
        issues=issues,
        metadata={"result_count": len(results)},
    )


def evaluate_cases(
    cases: Sequence[RetrievalEvaluationCase],
    results_by_evaluation_id: Mapping[str, Sequence[RetrievalResult]],
) -> list[EvaluationCaseResult]:
    """Evaluate many cases against fake/mock retrieval results."""
    return [
        evaluate_case(case, results_by_evaluation_id.get(case.evaluation_id, ()))
        for case in cases
    ]


def summarize_evaluation_results(
    cases: Sequence[RetrievalEvaluationCase],
    case_results: Sequence[EvaluationCaseResult],
) -> EvaluationSummary:
    """Summarize scored retrieval evaluation results."""
    passed_cases = sum(1 for result in case_results if result.passed)
    total_cases = len(case_results)
    issues = [
        f"{result.evaluation_id}: {issue}"
        for result in case_results
        for issue in result.issues
    ]

    return EvaluationSummary(
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=total_cases - passed_cases,
        pass_rate=passed_cases / total_cases if total_cases else 0.0,
        category_counts=dict(sorted(Counter(case.category for case in cases).items())),
        behavior_counts=dict(
            sorted(Counter(case.expected_behavior for case in cases).items())
        ),
        issues=issues,
    )


def retrieval_result_from_dict(data: Mapping[str, Any]) -> RetrievalResult:
    """Create a retrieval result from a mapping with permissive defaults."""
    metadata = data.get("metadata")

    return RetrievalResult(
        document_id=_optional_str(data.get("document_id")),
        chunk_id=_optional_str(data.get("chunk_id")),
        score=_optional_float(data.get("score")),
        rank=_int_or_default(data.get("rank"), 0),
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
    )


def retrieval_result_to_dict(result: RetrievalResult) -> dict[str, Any]:
    """Convert a retrieval result to a plain dictionary."""
    return asdict(result)


def _evaluate_retrieval_case(
    case: RetrievalEvaluationCase,
    results: Sequence[RetrievalResult],
    issues: list[str],
) -> EvaluationCaseResult:
    if case.minimum_expected_rank < 1:
        issues.append("minimum_expected_rank must be greater than or equal to 1.")

    ranked_results = _results_within_rank(results, case.minimum_expected_rank)
    expected_chunks = set(case.expected_chunk_ids)

    matched_document = bool(case.expected_document_id) and any(
        result.document_id == case.expected_document_id for result in ranked_results
    )
    matched_chunk = (
        not expected_chunks
        or any(result.chunk_id in expected_chunks for result in ranked_results)
    )
    met_rank_requirement = matched_document and matched_chunk

    if not case.expected_document_id:
        issues.append("expected_document_id is required for retrieval cases.")
    if not matched_document:
        issues.append(
            "Expected document was not found within rank "
            f"{case.minimum_expected_rank}."
        )
    if expected_chunks and not matched_chunk:
        issues.append(
            "Expected chunk was not found within rank "
            f"{case.minimum_expected_rank}."
        )

    return EvaluationCaseResult(
        evaluation_id=case.evaluation_id,
        passed=met_rank_requirement and not issues,
        expected_behavior=case.expected_behavior,
        matched_document=matched_document,
        matched_chunk=matched_chunk,
        met_rank_requirement=met_rank_requirement,
        issues=issues,
        metadata={
            "result_count": len(results),
            "rank_limit": case.minimum_expected_rank,
        },
    )


def _evaluate_insufficient_evidence_case(
    case: RetrievalEvaluationCase,
    results: Sequence[RetrievalResult],
    issues: list[str],
) -> EvaluationCaseResult:
    passed = (
        not results
        or all(_is_low_score(result) for result in results)
        or any(_metadata_true(result.metadata, "insufficient_evidence") for result in results)
        or any(result.metadata.get("evidence_level") == "not_found" for result in results)
    )

    if not passed:
        issues.append("Results did not indicate insufficient evidence.")

    return EvaluationCaseResult(
        evaluation_id=case.evaluation_id,
        passed=passed and not issues,
        expected_behavior=case.expected_behavior,
        matched_document=False,
        matched_chunk=False,
        met_rank_requirement=passed,
        issues=issues,
        metadata={
            "result_count": len(results),
            "low_score_threshold": LOW_SCORE_THRESHOLD,
        },
    )


def _evaluate_reject_out_of_scope_case(
    case: RetrievalEvaluationCase,
    results: Sequence[RetrievalResult],
    issues: list[str],
) -> EvaluationCaseResult:
    passed = not results or any(_metadata_true(result.metadata, "rejected") for result in results)

    if not passed:
        issues.append("Results did not indicate out-of-scope rejection.")

    return EvaluationCaseResult(
        evaluation_id=case.evaluation_id,
        passed=passed and not issues,
        expected_behavior=case.expected_behavior,
        matched_document=False,
        matched_chunk=False,
        met_rank_requirement=passed,
        issues=issues,
        metadata={"result_count": len(results)},
    )


def _results_within_rank(
    results: Sequence[RetrievalResult],
    maximum_rank: int,
) -> list[RetrievalResult]:
    if maximum_rank < 1:
        return []
    return [
        result
        for result in results
        if isinstance(result.rank, int) and 1 <= result.rank <= maximum_rank
    ]


def _rank_issues(results: Sequence[RetrievalResult]) -> list[str]:
    return [
        f"Result has invalid rank: {result.rank}"
        for result in results
        if not isinstance(result.rank, int) or result.rank < 1
    ]


def _is_low_score(result: RetrievalResult) -> bool:
    return result.score is None or result.score <= LOW_SCORE_THRESHOLD


def _metadata_true(metadata: Mapping[str, Any], key: str) -> bool:
    return metadata.get(key) is True


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "EvaluationCaseResult",
    "EvaluationSummary",
    "LOW_SCORE_THRESHOLD",
    "RetrievalResult",
    "evaluate_case",
    "evaluate_cases",
    "retrieval_result_from_dict",
    "retrieval_result_to_dict",
    "summarize_evaluation_results",
]
