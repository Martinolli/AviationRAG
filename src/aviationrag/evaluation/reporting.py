"""Report/export helpers for fake/mock retrieval evaluation results.

These helpers format results produced by the retrieval harness shell. They do
not run retrieval, call vector stores, generate embeddings, or read source
documents. Files are written only when an explicit write function is called.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Sequence

from aviationrag.evaluation.retrieval_harness import (
    EvaluationCaseResult,
    EvaluationSummary,
)


def evaluation_case_result_to_dict(result: EvaluationCaseResult) -> dict[str, Any]:
    """Convert one case result into a JSON-serializable dictionary."""
    return asdict(result)


def evaluation_summary_to_dict(summary: EvaluationSummary) -> dict[str, Any]:
    """Convert an evaluation summary into a JSON-serializable dictionary."""
    return asdict(summary)


def render_markdown_report(
    summary: EvaluationSummary,
    case_results: Sequence[EvaluationCaseResult],
    title: str = "Retrieval Evaluation Report",
) -> str:
    """Render a Markdown report for fake/mock retrieval evaluation results."""
    lines = [
        f"# {title}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total cases | {summary.total_cases} |",
        f"| Passed cases | {summary.passed_cases} |",
        f"| Failed cases | {summary.failed_cases} |",
        f"| Pass rate | {_format_pass_rate(summary.pass_rate)} |",
        "",
        "## Category Counts",
        "",
    ]

    lines.extend(_count_table(summary.category_counts, "Category"))
    lines.extend(["", "## Behavior Counts", ""])
    lines.extend(_count_table(summary.behavior_counts, "Behavior"))
    lines.extend(["", "## Issues", ""])

    if summary.issues:
        lines.extend(f"- {_escape_markdown_cell(issue)}" for issue in summary.issues)
    else:
        lines.append("No issues reported.")

    lines.extend(
        [
            "",
            "## Case Results",
            "",
            "| evaluation_id | passed | expected_behavior | matched_document | matched_chunk | met_rank_requirement | issues |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in case_results:
        issue_text = "; ".join(result.issues) if result.issues else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_markdown_cell(result.evaluation_id),
                    _bool_text(result.passed),
                    _escape_markdown_cell(result.expected_behavior),
                    _bool_text(result.matched_document),
                    _bool_text(result.matched_chunk),
                    _bool_text(result.met_rank_requirement),
                    _escape_markdown_cell(issue_text),
                ]
            )
            + " |"
        )

    return "\n".join(lines) + "\n"


def write_json_report(
    path: str | Path,
    summary: EvaluationSummary,
    case_results: Sequence[EvaluationCaseResult],
) -> None:
    """Write a JSON report for fake/mock retrieval evaluation results."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": evaluation_summary_to_dict(summary),
        "case_results": [
            evaluation_case_result_to_dict(result) for result in case_results
        ],
    }
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_markdown_report(
    path: str | Path,
    summary: EvaluationSummary,
    case_results: Sequence[EvaluationCaseResult],
) -> None:
    """Write a Markdown report for fake/mock retrieval evaluation results."""
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_markdown_report(summary, case_results),
        encoding="utf-8",
    )


def _count_table(counts: dict[str, int], label: str) -> list[str]:
    if not counts:
        return [f"No {label.lower()} counts reported."]

    lines = [
        f"| {label} | Count |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {_escape_markdown_cell(name)} | {count} |"
        for name, count in sorted(counts.items())
    )
    return lines


def _format_pass_rate(value: float) -> str:
    return f"{value:.2%}"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _escape_markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


__all__ = [
    "evaluation_case_result_to_dict",
    "evaluation_summary_to_dict",
    "render_markdown_report",
    "write_json_report",
    "write_markdown_report",
]
