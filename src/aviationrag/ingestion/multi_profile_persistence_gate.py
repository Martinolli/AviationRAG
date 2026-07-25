"""D.5d controlled multi-profile parser-output persistence evaluation.

This module aggregates existing D.5c gates and D.5b packages. It does not
invoke techdoc-parser, copy source documents, modify runtime ingestion, generate
embeddings, or touch database/vector systems.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from aviationrag.ingestion.persisted_chunk_mapper import (
    PersistedChunkCandidateContext,
    PersistedChunkMappingPolicy,
)
from aviationrag.ingestion.persisted_chunk_package import (
    PACKAGE_SCHEMA_NAME,
    PACKAGE_SCHEMA_VERSION,
    PersistedChunkPackage,
    build_package_from_adapter_result,
)
from aviationrag.ingestion.persisted_chunk_record import (
    PERSISTED_CHUNK_MAPPER_VERSION,
    PERSISTED_CHUNK_SCHEMA_NAME,
    PERSISTED_CHUNK_SCHEMA_VERSION,
)
from aviationrag.ingestion.persisted_chunk_validator import (
    PERSISTED_CHUNK_LIMITATION_REGISTRY_VERSION,
    APPROVED_LIMITATION_CODES,
)
from aviationrag.ingestion.real_parser_sample_gate import (
    RealParserSampleGateResult,
    real_parser_sample_gate_result_to_dict,
    run_real_parser_sample_gate,
)
from aviationrag.ingestion.structured_document_adapter import (
    FAIL,
    PASS,
    REVIEW,
    run_structured_document_adapter,
)


ACCEPTED_WITH_LIMITATIONS = "ACCEPTED_WITH_LIMITATIONS"
MULTI_PROFILE_GATE_SCHEMA_NAME = "aviationrag-multi-profile-persistence-gate"
MULTI_PROFILE_GATE_SCHEMA_VERSION = "0.1.0"
REQUIRED_PROFILE_COUNT = 3
ALLOWED_PROFILE_ROLES = frozenset(
    {
        "accepted-limitation profile",
        "flight-test publication",
        "formal safety standard",
    }
)
AUTHORIZATION = {
    "d6_persistence_governance_review": True,
    "additional_uncontrolled_document_processing": False,
    "full_corpus_ingestion": False,
    "runtime_ingestion": False,
    "production_persistence": False,
    "embedding_generation": False,
    "astra_rebuild": False,
    "faiss_rebuild": False,
    "production_retrieval_integration": False,
    "production_migration": False,
}


@dataclass(frozen=True)
class MultiProfileDefinition:
    """One approved D.5d profile definition."""

    profile_key: str
    profile_role: str
    source_path: Path
    artifact_path: Path
    manifest_path: Path
    expected_document_id: str
    expected_source_filename: str
    expected_page_count: int | None = None
    approved_adapter_warning_codes: tuple[str, ...] = ()
    candidate_contexts: Mapping[str, PersistedChunkCandidateContext] = field(default_factory=dict)
    allow_review_outcome: bool = False


@dataclass(frozen=True)
class MultiProfilePersistenceEvaluationResult:
    """Sanitized aggregate D.5d result."""

    outcome: str
    profile_count: int
    profile_results: tuple[RealParserSampleGateResult, ...]
    profile_outcomes: Mapping[str, str]
    total_candidate_count: int
    total_accepted_record_count: int
    total_rejected_candidate_count: int
    total_warning_count: int
    total_review_required_count: int
    aggregate_validation_status_counts: Mapping[str, int]
    aggregate_provenance_counts: Mapping[str, int]
    aggregate_content_type_counts: Mapping[str, int]
    aggregate_limitation_counts: Mapping[str, int]
    cross_document_chunk_id_collision_count: int
    schema_consistency_verified: bool
    determinism_verified: bool
    blocking_issue_codes: tuple[str, ...]
    authorization: Mapping[str, bool]
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class EvaluatedProfilePackage:
    """Internal per-profile package evidence used by the CLI."""

    profile: MultiProfileDefinition
    gate_result: RealParserSampleGateResult
    package: PersistedChunkPackage
    repeated_package: PersistedChunkPackage


def load_multi_profile_config(path: str | Path) -> tuple[MultiProfileDefinition, ...]:
    """Load a fail-closed D.5d profile config."""
    config_path = Path(path)
    data = json.loads(config_path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(data, Mapping):
        raise ValueError("Multi-profile config must be a JSON object.")
    _reject_unknown_fields(
        data,
        {"schema_name", "schema_version", "profiles"},
        "$",
    )
    if data.get("schema_name") != MULTI_PROFILE_GATE_SCHEMA_NAME:
        raise ValueError("Unsupported multi-profile config schema_name.")
    if data.get("schema_version") != MULTI_PROFILE_GATE_SCHEMA_VERSION:
        raise ValueError("Unsupported multi-profile config schema_version.")
    profiles_raw = data.get("profiles")
    if not isinstance(profiles_raw, list):
        raise ValueError("$.profiles must be an array.")

    profiles = tuple(_profile_from_config(item, index) for index, item in enumerate(profiles_raw))
    keys = [profile.profile_key for profile in profiles]
    duplicates = sorted(key for key, count in Counter(keys).items() if count > 1)
    if duplicates:
        raise ValueError("Duplicate profile keys are not allowed: " + ", ".join(duplicates))
    return profiles


def evaluate_profile_package(
    profile: MultiProfileDefinition,
    *,
    allow_reviewed_profiles: bool = False,
) -> EvaluatedProfilePackage:
    """Run one D.5c gate and build its deterministic packages in memory."""
    allow_review = bool(profile.allow_review_outcome and allow_reviewed_profiles)
    mapping_policy = PersistedChunkMappingPolicy(
        allow_partial_provenance=False,
        allow_review_required_records=allow_review,
        include_heading_records=False,
    )
    gate_result = run_real_parser_sample_gate(
        artifact_path=profile.artifact_path,
        manifest_path=profile.manifest_path,
        source_path=profile.source_path,
        candidate_contexts=profile.candidate_contexts,
        approved_adapter_warning_codes=profile.approved_adapter_warning_codes,
        mapping_policy=mapping_policy,
        allow_review=allow_review,
    )
    adapter_result = run_structured_document_adapter(
        profile.artifact_path,
        profile.manifest_path,
        source_path=profile.source_path,
        approved_warning_codes=profile.approved_adapter_warning_codes,
        strict_warnings=False,
    )
    package = build_package_from_adapter_result(
        adapter_result,
        candidate_contexts=profile.candidate_contexts,
        policy=mapping_policy,
        allow_rejected_candidates=False,
    )
    repeated_package = build_package_from_adapter_result(
        adapter_result,
        candidate_contexts=profile.candidate_contexts,
        policy=mapping_policy,
        allow_rejected_candidates=False,
    )
    return EvaluatedProfilePackage(
        profile=profile,
        gate_result=gate_result,
        package=package,
        repeated_package=repeated_package,
    )


def evaluate_profile_packages(
    profiles: Sequence[MultiProfileDefinition],
    *,
    allow_reviewed_profiles: bool = False,
) -> tuple[EvaluatedProfilePackage, ...]:
    """Evaluate profiles in deterministic profile-key order."""
    return tuple(
        evaluate_profile_package(profile, allow_reviewed_profiles=allow_reviewed_profiles)
        for profile in sorted(profiles, key=lambda item: item.profile_key)
    )


def run_multi_profile_persistence_evaluation(
    profiles: Sequence[MultiProfileDefinition],
    *,
    allow_reviewed_profiles: bool = False,
) -> MultiProfilePersistenceEvaluationResult:
    """Run the D.5d aggregate evaluation without writing files."""
    evaluated = evaluate_profile_packages(profiles, allow_reviewed_profiles=allow_reviewed_profiles)
    return aggregate_evaluated_profile_packages(evaluated, allow_reviewed_profiles=allow_reviewed_profiles)


def aggregate_evaluated_profile_packages(
    evaluated: Sequence[EvaluatedProfilePackage],
    *,
    allow_reviewed_profiles: bool = False,
    write_determinism_verified: bool | None = None,
) -> MultiProfilePersistenceEvaluationResult:
    """Aggregate completed profile package evidence into a sanitized result."""
    profile_results = tuple(item.gate_result for item in evaluated)
    profile_by_key = {item.profile.profile_key: item for item in evaluated}
    profile_outcomes = {key: item.gate_result.outcome for key, item in sorted(profile_by_key.items())}
    blocking_codes: list[str] = []

    if len(evaluated) != REQUIRED_PROFILE_COUNT:
        blocking_codes.append("PROFILE_COUNT_MISMATCH")
    if len(profile_by_key) != len(evaluated):
        blocking_codes.append("DUPLICATE_PROFILE_KEY")

    for item in evaluated:
        _profile_identity_blocking_codes(item, blocking_codes)
        if item.gate_result.outcome == FAIL:
            blocking_codes.append(f"PROFILE_FAIL:{item.profile.profile_key}")
        elif item.gate_result.outcome == REVIEW and not allow_reviewed_profiles:
            blocking_codes.append(f"PROFILE_REVIEW_NOT_ALLOWED:{item.profile.profile_key}")
        if item.gate_result.accepted_record_count == 0:
            blocking_codes.append(f"PROFILE_ZERO_ACCEPTED:{item.profile.profile_key}")
        if item.gate_result.rejected_candidate_count:
            blocking_codes.append(f"PROFILE_REJECTED_CANDIDATES:{item.profile.profile_key}")
        if not _profile_deterministic(item):
            blocking_codes.append(f"PROFILE_NONDETERMINISTIC:{item.profile.profile_key}")
        blocking_codes.extend(
            f"PROFILE_BLOCKING:{item.profile.profile_key}:{code}" for code in item.gate_result.blocking_issue_codes
        )
        if _profile_has_unknown_limitation(item):
            blocking_codes.append(f"PROFILE_UNKNOWN_LIMITATION:{item.profile.profile_key}")
        if _profile_has_unknown_warning(item):
            blocking_codes.append(f"PROFILE_UNKNOWN_WARNING:{item.profile.profile_key}")
        if _profile_has_uncontexted_review_records(item):
            blocking_codes.append(f"PROFILE_UNCONTEXTED_REVIEW:{item.profile.profile_key}")

    cross_document_collision_count = _cross_document_collision_count(evaluated)
    if cross_document_collision_count:
        blocking_codes.append("CROSS_DOCUMENT_CHUNK_ID_COLLISION")
    schema_consistency = _schema_consistency_verified(evaluated)
    if not schema_consistency:
        blocking_codes.append("SCHEMA_CONSISTENCY_FAILED")
    package_determinism = all(_profile_deterministic(item) for item in evaluated)
    determinism_verified = package_determinism and all(item.gate_result.determinism_verified for item in evaluated)
    if write_determinism_verified is not None:
        determinism_verified = determinism_verified and write_determinism_verified
    if not determinism_verified:
        blocking_codes.append("AGGREGATE_NONDETERMINISTIC")

    aggregate_validation = Counter()
    aggregate_provenance = Counter()
    aggregate_content = Counter()
    aggregate_limitations = Counter()
    for result in profile_results:
        aggregate_validation.update(result.validation_status_counts)
        aggregate_provenance.update(result.provenance_counts)
        aggregate_content.update(result.content_type_counts)
        aggregate_limitations.update(result.accepted_limitation_counts)
    if aggregate_provenance.get("unknown_provenance", 0):
        blocking_codes.append("UNKNOWN_PROVENANCE_PRESENT")

    outcome = _aggregate_outcome(
        blocking_codes,
        profile_results,
        allow_reviewed_profiles=allow_reviewed_profiles,
    )
    return MultiProfilePersistenceEvaluationResult(
        outcome=outcome,
        profile_count=len(evaluated),
        profile_results=profile_results,
        profile_outcomes=profile_outcomes,
        total_candidate_count=sum(result.input_candidate_count for result in profile_results),
        total_accepted_record_count=sum(result.accepted_record_count for result in profile_results),
        total_rejected_candidate_count=sum(result.rejected_candidate_count for result in profile_results),
        total_warning_count=sum(result.warning_count for result in profile_results),
        total_review_required_count=sum(result.review_required_count for result in profile_results),
        aggregate_validation_status_counts=dict(sorted(aggregate_validation.items())),
        aggregate_provenance_counts=dict(sorted(aggregate_provenance.items())),
        aggregate_content_type_counts=dict(sorted(aggregate_content.items())),
        aggregate_limitation_counts=dict(sorted(aggregate_limitations.items())),
        cross_document_chunk_id_collision_count=cross_document_collision_count,
        schema_consistency_verified=schema_consistency,
        determinism_verified=determinism_verified,
        blocking_issue_codes=tuple(sorted(set(blocking_codes))),
        authorization=dict(AUTHORIZATION),
        summary=_aggregate_summary(evaluated),
    )


def multi_profile_persistence_result_to_dict(
    result: MultiProfilePersistenceEvaluationResult,
) -> dict[str, Any]:
    """Return a deterministic sanitized D.5d result dictionary."""
    return {
        "aggregate_content_type_counts": dict(sorted(result.aggregate_content_type_counts.items())),
        "aggregate_limitation_counts": dict(sorted(result.aggregate_limitation_counts.items())),
        "aggregate_provenance_counts": dict(sorted(result.aggregate_provenance_counts.items())),
        "aggregate_validation_status_counts": dict(sorted(result.aggregate_validation_status_counts.items())),
        "authorization": dict(sorted(result.authorization.items())),
        "blocking_issue_codes": list(result.blocking_issue_codes),
        "cross_document_chunk_id_collision_count": result.cross_document_chunk_id_collision_count,
        "determinism_verified": result.determinism_verified,
        "evaluation_schema_name": MULTI_PROFILE_GATE_SCHEMA_NAME,
        "evaluation_schema_version": MULTI_PROFILE_GATE_SCHEMA_VERSION,
        "outcome": result.outcome,
        "profile_count": result.profile_count,
        "profile_outcomes": dict(sorted(result.profile_outcomes.items())),
        "profiles": [
            _profile_result_dict(item)
            for item in sorted(result.summary.get("profiles", []), key=lambda item: item["profile_key"])
        ],
        "schema_consistency_verified": result.schema_consistency_verified,
        "summary": _plain({key: value for key, value in result.summary.items() if key != "profiles"}),
        "total_accepted_record_count": result.total_accepted_record_count,
        "total_candidate_count": result.total_candidate_count,
        "total_rejected_candidate_count": result.total_rejected_candidate_count,
        "total_review_required_count": result.total_review_required_count,
        "total_warning_count": result.total_warning_count,
    }


def sanitized_multi_profile_report_bytes(
    result: MultiProfilePersistenceEvaluationResult,
) -> bytes:
    """Serialize a D.5d report with deterministic JSON and final newline."""
    return (
        json.dumps(
            multi_profile_persistence_result_to_dict(result),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _profile_from_config(raw: Any, index: int) -> MultiProfileDefinition:
    if not isinstance(raw, Mapping):
        raise ValueError(f"$.profiles[{index}] must be an object.")
    _reject_unknown_fields(
        raw,
        {
            "profile_key",
            "profile_role",
            "source_path",
            "artifact_path",
            "manifest_path",
            "expected_document_id",
            "expected_source_filename",
            "expected_page_count",
            "approved_adapter_warning_codes",
            "candidate_contexts",
            "allow_review_outcome",
        },
        f"$.profiles[{index}]",
    )
    role = _required_string(raw, "profile_role", f"$.profiles[{index}]")
    if role not in ALLOWED_PROFILE_ROLES:
        raise ValueError(f"Invalid profile role: {role}")
    source_path = _required_existing_file(raw, "source_path", f"$.profiles[{index}]")
    artifact_path = _required_existing_file(raw, "artifact_path", f"$.profiles[{index}]")
    manifest_path = _required_existing_file(raw, "manifest_path", f"$.profiles[{index}]")
    expected_page_count = raw.get("expected_page_count")
    if expected_page_count is not None and (
        not isinstance(expected_page_count, int) or isinstance(expected_page_count, bool) or expected_page_count < 1
    ):
        raise ValueError(f"$.profiles[{index}].expected_page_count must be a positive integer or null.")
    return MultiProfileDefinition(
        profile_key=_required_string(raw, "profile_key", f"$.profiles[{index}]"),
        profile_role=role,
        source_path=source_path,
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        expected_document_id=_required_string(raw, "expected_document_id", f"$.profiles[{index}]"),
        expected_source_filename=_required_string(raw, "expected_source_filename", f"$.profiles[{index}]"),
        expected_page_count=expected_page_count,
        approved_adapter_warning_codes=_string_tuple(raw.get("approved_adapter_warning_codes", ())),
        candidate_contexts=_candidate_contexts(raw.get("candidate_contexts", {}), f"$.profiles[{index}]"),
        allow_review_outcome=bool(raw.get("allow_review_outcome", False)),
    )


def _candidate_contexts(raw: Any, path: str) -> dict[str, PersistedChunkCandidateContext]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}.candidate_contexts must be an object.")
    contexts: dict[str, PersistedChunkCandidateContext] = {}
    for candidate_id, value in sorted(raw.items()):
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError(f"{path}.candidate_contexts keys must be non-empty strings.")
        if not isinstance(value, Mapping):
            raise ValueError(f"{path}.candidate_contexts.{candidate_id} must be an object.")
        _reject_unknown_fields(
            value,
            {"accepted_limitation_codes", "warning_codes", "review_required"},
            f"{path}.candidate_contexts.{candidate_id}",
        )
        contexts[candidate_id] = PersistedChunkCandidateContext(
            accepted_limitation_codes=_string_tuple(value.get("accepted_limitation_codes", ())),
            warning_codes=_string_tuple(value.get("warning_codes", ())),
            review_required=bool(value.get("review_required", False)),
        )
    return contexts


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    seen: set[str] = set()
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"Duplicate JSON key is not allowed: {key}")
        seen.add(key)
        result[key] = value
    return result


def _reject_unknown_fields(data: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown field(s) at {path}: {', '.join(unknown)}")


def _required_string(data: Mapping[str, Any], key: str, path: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path}.{key} must be a non-empty string.")
    return value


def _required_existing_file(data: Mapping[str, Any], key: str, path: str) -> Path:
    value = _required_string(data, key, path)
    file_path = Path(value)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"{path}.{key} does not exist or is not a file: {file_path}")
    return file_path


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("Expected an array of strings.")
    result = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("Expected an array of non-empty strings.")
        result.append(item)
    return tuple(sorted(set(result)))


def _profile_identity_blocking_codes(item: EvaluatedProfilePackage, codes: list[str]) -> None:
    profile = item.profile
    result = item.gate_result
    if result.document_key != profile.expected_document_id:
        codes.append(f"DOCUMENT_IDENTITY_MISMATCH:{profile.profile_key}")
    if result.source_filename != profile.expected_source_filename:
        codes.append(f"SOURCE_FILENAME_MISMATCH:{profile.profile_key}")
    if profile.expected_page_count is not None:
        page_count = int(result.summary.get("page_count", -1))
        if page_count != profile.expected_page_count:
            codes.append(f"PAGE_COUNT_MISMATCH:{profile.profile_key}")


def _profile_deterministic(item: EvaluatedProfilePackage) -> bool:
    return (
        item.package.package_digest == item.repeated_package.package_digest
        and dict(item.package.file_sha256) == dict(item.repeated_package.file_sha256)
    )


def _profile_has_unknown_limitation(item: EvaluatedProfilePackage) -> bool:
    codes = set(item.gate_result.accepted_limitation_counts)
    return bool(codes - set(APPROVED_LIMITATION_CODES)) or any(
        code in {"UNKNOWN_LIMITATION_CODE", "LIMITATION_CODE_UNKNOWN"}
        for code in item.gate_result.blocking_issue_codes
    )


def _profile_has_unknown_warning(item: EvaluatedProfilePackage) -> bool:
    return any(code.startswith("UNKNOWN_PACKAGE_WARNING") or code.startswith("UNAPPROVED_ADAPTER_WARNING") for code in item.gate_result.blocking_issue_codes)


def _profile_has_uncontexted_review_records(item: EvaluatedProfilePackage) -> bool:
    context_ids = set(item.profile.candidate_contexts)
    if item.gate_result.review_required_count == 0:
        return False
    for warning in item.package.warnings:
        if warning.candidate_id and warning.candidate_id not in context_ids:
            return True
    return any(record.review_required and not set(record.accepted_limitation_codes) for record in item.package.records)


def _cross_document_collision_count(evaluated: Sequence[EvaluatedProfilePackage]) -> int:
    owners: dict[str, set[str]] = {}
    for item in evaluated:
        for record in item.package.records:
            owners.setdefault(record.chunk_id, set()).add(item.profile.profile_key)
    return sum(1 for profiles in owners.values() if len(profiles) > 1)


def _schema_consistency_verified(evaluated: Sequence[EvaluatedProfilePackage]) -> bool:
    if not evaluated:
        return False
    expected = {
        "package_schema_name": PACKAGE_SCHEMA_NAME,
        "package_schema_version": PACKAGE_SCHEMA_VERSION,
        "persisted_schema_name": PERSISTED_CHUNK_SCHEMA_NAME,
        "persisted_schema_version": PERSISTED_CHUNK_SCHEMA_VERSION,
        "mapper_version": PERSISTED_CHUNK_MAPPER_VERSION,
        "limitation_registry_version": PERSISTED_CHUNK_LIMITATION_REGISTRY_VERSION,
    }
    for item in evaluated:
        manifest = item.package.manifest
        for key, value in expected.items():
            if manifest.get(key) != value:
                return False
        for record in item.package.records:
            if record.schema_name != PERSISTED_CHUNK_SCHEMA_NAME:
                return False
            if record.schema_version != PERSISTED_CHUNK_SCHEMA_VERSION:
                return False
            if record.persistence_mapper_version != PERSISTED_CHUNK_MAPPER_VERSION:
                return False
    return True


def _aggregate_outcome(
    blocking_codes: Sequence[str],
    profile_results: Sequence[RealParserSampleGateResult],
    *,
    allow_reviewed_profiles: bool,
) -> str:
    if blocking_codes:
        return FAIL
    profile_outcomes = {result.outcome for result in profile_results}
    if profile_outcomes == {PASS}:
        return PASS
    if allow_reviewed_profiles and profile_outcomes <= {PASS, REVIEW} and REVIEW in profile_outcomes:
        return ACCEPTED_WITH_LIMITATIONS
    return FAIL


def _aggregate_summary(evaluated: Sequence[EvaluatedProfilePackage]) -> dict[str, Any]:
    schema_values = {
        "adapter_versions": sorted({str(record.adapter_version) for item in evaluated for record in item.package.records}),
        "limitation_registry_versions": sorted(
            {str(item.package.manifest.get("limitation_registry_version")) for item in evaluated}
        ),
        "mapper_versions": sorted({str(item.package.manifest.get("mapper_version")) for item in evaluated}),
        "package_schema_versions": sorted({str(item.package.manifest.get("package_schema_version")) for item in evaluated}),
        "persisted_schema_versions": sorted(
            {str(item.package.manifest.get("persisted_schema_version")) for item in evaluated}
        ),
    }
    return {
        "approved_limitation_codes": tuple(sorted(APPROVED_LIMITATION_CODES)),
        "profiles": [_evaluated_profile_summary(item) for item in evaluated],
        "runtime_ingestion_modified": False,
        "production_persistence": False,
        "full_corpus_processed": False,
        "embedding_generation": False,
        "astra_touched": False,
        "faiss_touched": False,
        "schema_values": schema_values,
    }


def _evaluated_profile_summary(item: EvaluatedProfilePackage) -> dict[str, Any]:
    result = item.gate_result
    return {
        "accepted_limitation_counts": dict(sorted(result.accepted_limitation_counts.items())),
        "accepted_record_count": result.accepted_record_count,
        "artifact_checksum": result.summary.get("artifact_checksum"),
        "candidate_context_ids": tuple(sorted(item.profile.candidate_contexts)),
        "content_type_counts": dict(sorted(result.content_type_counts.items())),
        "determinism_verified": _profile_deterministic(item) and result.determinism_verified,
        "document_key": result.document_key,
        "expected_document_id": item.profile.expected_document_id,
        "expected_source_filename": item.profile.expected_source_filename,
        "gate_outcome": result.outcome,
        "input_candidate_count": result.input_candidate_count,
        "manifest_checksum": result.summary.get("manifest_checksum"),
        "package_digest": result.package_digest,
        "package_outcome": result.package_outcome,
        "page_count": result.summary.get("page_count"),
        "parser_name": result.parser_name,
        "parser_version": result.parser_version,
        "profile_key": item.profile.profile_key,
        "profile_role": item.profile.profile_role,
        "provenance_counts": dict(sorted(result.provenance_counts.items())),
        "rejected_candidate_count": result.rejected_candidate_count,
        "review_required_count": result.review_required_count,
        "source_checksum": result.source_checksum,
        "source_filename": result.source_filename,
        "structured_document_schema_version": result.structured_document_schema_version,
        "validation_status_counts": dict(sorted(result.validation_status_counts.items())),
        "warning_count": result.warning_count,
    }


def _profile_result_dict(summary: Mapping[str, Any]) -> dict[str, Any]:
    return _plain(dict(summary))


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


__all__ = [
    "ACCEPTED_WITH_LIMITATIONS",
    "AUTHORIZATION",
    "MULTI_PROFILE_GATE_SCHEMA_NAME",
    "MULTI_PROFILE_GATE_SCHEMA_VERSION",
    "MultiProfileDefinition",
    "MultiProfilePersistenceEvaluationResult",
    "aggregate_evaluated_profile_packages",
    "evaluate_profile_package",
    "evaluate_profile_packages",
    "load_multi_profile_config",
    "multi_profile_persistence_result_to_dict",
    "run_multi_profile_persistence_evaluation",
    "sanitized_multi_profile_report_bytes",
]
