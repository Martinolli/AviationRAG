"""Fake-data retrieval evaluation smoke fixture utilities.

This module validates committed sample benchmark cases only. It does not run
retrieval, call FAISS, connect to Astra, generate embeddings, or read private
source documents.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


ACCEPTED_CATEGORIES = {
    "compliance",
    "manufacturing",
    "sms",
    "accident_analysis",
    "maintenance",
    "unsupported",
}

ACCEPTED_EXPECTED_BEHAVIORS = {
    "retrieve_relevant_context",
    "insufficient_evidence",
    "reject_out_of_scope",
}


@dataclass(frozen=True)
class RetrievalEvaluationCase:
    """One fake retrieval smoke benchmark case."""

    evaluation_id: str
    category: str
    question: str
    expected_document_id: str | None = None
    expected_chunk_ids: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    expected_behavior: str = ""
    minimum_expected_rank: int = 1
    notes: str | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RetrievalEvaluationCase":
        return cls(
            evaluation_id=_optional_str(data.get("evaluation_id")) or "",
            category=_optional_str(data.get("category")) or "",
            question=_optional_str(data.get("question")) or "",
            expected_document_id=_optional_str(data.get("expected_document_id")),
            expected_chunk_ids=_string_list(data.get("expected_chunk_ids")),
            expected_keywords=_string_list(data.get("expected_keywords")),
            expected_behavior=_optional_str(data.get("expected_behavior")) or "",
            minimum_expected_rank=_int_or_default(data.get("minimum_expected_rank"), 1),
            notes=_optional_str(data.get("notes")),
        )


def load_retrieval_evaluation_cases(path: str | Path) -> list[RetrievalEvaluationCase]:
    """Load retrieval evaluation cases from JSONL."""
    fixture_path = Path(path)
    cases: list[RetrievalEvaluationCase] = []

    with fixture_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in retrieval evaluation fixture {fixture_path} "
                    f"at line {line_number}: {error.msg}"
                ) from error
            if not isinstance(data, dict):
                raise ValueError(
                    f"Invalid JSON object in retrieval evaluation fixture {fixture_path} "
                    f"at line {line_number}"
                )
            cases.append(RetrievalEvaluationCase.from_mapping(data))

    return cases


def validate_retrieval_evaluation_case(case: RetrievalEvaluationCase) -> list[str]:
    """Return human-readable issues for one evaluation case."""
    issues: list[str] = []

    if not case.evaluation_id.strip():
        issues.append("Missing evaluation_id.")
    if not case.question.strip():
        issues.append(f"{_case_label(case)}: Missing question.")
    if not case.expected_behavior.strip():
        issues.append(f"{_case_label(case)}: Missing expected_behavior.")
    elif case.expected_behavior not in ACCEPTED_EXPECTED_BEHAVIORS:
        issues.append(
            f"{_case_label(case)}: Invalid expected_behavior: {case.expected_behavior}"
        )

    if case.category not in ACCEPTED_CATEGORIES:
        issues.append(f"{_case_label(case)}: Invalid category: {case.category}")

    if case.minimum_expected_rank < 1:
        issues.append(
            f"{_case_label(case)}: minimum_expected_rank must be greater than or equal to 1."
        )

    if case.expected_behavior == "retrieve_relevant_context":
        if not case.expected_keywords:
            issues.append(
                f"{_case_label(case)}: expected_keywords must not be empty for retrieval cases."
            )
        if not case.expected_document_id:
            issues.append(
                f"{_case_label(case)}: expected_document_id is required for retrieval cases."
            )

    return issues


def summarize_retrieval_evaluation_cases(
    cases: Iterable[RetrievalEvaluationCase],
) -> dict[str, Any]:
    """Summarize case counts and validation status."""
    case_list = list(cases)
    ids = [case.evaluation_id for case in case_list]
    validation_issues = validate_retrieval_evaluation_dataset(case_list)

    return {
        "total_cases": len(case_list),
        "categories": dict(sorted(Counter(case.category for case in case_list).items())),
        "behavior_counts": dict(
            sorted(Counter(case.expected_behavior for case in case_list).items())
        ),
        "duplicate_ids": _duplicates(ids),
        "invalid_cases": sorted(
            {
                case.evaluation_id or "<missing>"
                for case in case_list
                if validate_retrieval_evaluation_case(case)
            }
        ),
        "max_expected_rank": max(
            (case.minimum_expected_rank for case in case_list),
            default=0,
        ),
        "issue_count": len(validation_issues),
    }


def validate_retrieval_evaluation_dataset(
    cases: Iterable[RetrievalEvaluationCase],
) -> list[str]:
    """Return human-readable validation issues for a dataset."""
    case_list = list(cases)
    issues: list[str] = []

    ids = [case.evaluation_id for case in case_list]
    for evaluation_id in _duplicates(ids):
        issues.append(f"Duplicate evaluation_id: {evaluation_id}")

    for case in case_list:
        issues.extend(validate_retrieval_evaluation_case(case))

    return issues


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Iterable):
        return [str(item) for item in value if item is not None and str(item)]
    return []


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(value for value in values if value)
    return sorted(value for value, count in counts.items() if count > 1)


def _case_label(case: RetrievalEvaluationCase) -> str:
    return case.evaluation_id or "<missing>"


__all__ = [
    "ACCEPTED_CATEGORIES",
    "ACCEPTED_EXPECTED_BEHAVIORS",
    "RetrievalEvaluationCase",
    "load_retrieval_evaluation_cases",
    "summarize_retrieval_evaluation_cases",
    "validate_retrieval_evaluation_case",
    "validate_retrieval_evaluation_dataset",
]
