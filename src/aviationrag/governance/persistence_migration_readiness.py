"""D.6 persistence migration readiness governance evaluator.

The evaluator consumes sanitized evidence and policy dictionaries. It performs
no parser execution, migration, runtime ingestion, embedding, database, vector,
or filesystem write work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Any, Mapping

from aviationrag.ingestion.persisted_chunk_validator import APPROVED_LIMITATION_CODES


READINESS_EVIDENCE_SCHEMA_NAME = "aviationrag-persistence-migration-readiness-evidence"
READINESS_EVIDENCE_SCHEMA_VERSION = "0.1.0"
READINESS_DECISION_SCHEMA_NAME = "aviationrag-persistence-migration-readiness-decision"
READINESS_DECISION_SCHEMA_VERSION = "0.1.0"
POLICY_NAME = "aviationrag-persistence-governance"
POLICY_VERSION = "0.1.0"

NOT_READY = "NOT_READY"
CONDITIONALLY_READY_FOR_CONTROLLED_MIGRATION_REHEARSAL = (
    "CONDITIONALLY_READY_FOR_CONTROLLED_MIGRATION_REHEARSAL"
)
READY_FOR_CONTROLLED_MIGRATION_REHEARSAL = "READY_FOR_CONTROLLED_MIGRATION_REHEARSAL"
READINESS_DECISIONS = (
    NOT_READY,
    CONDITIONALLY_READY_FOR_CONTROLLED_MIGRATION_REHEARSAL,
    READY_FOR_CONTROLLED_MIGRATION_REHEARSAL,
)

FUTURE_EXECUTION_PHASES = (
    "controlled_migration_rehearsal",
    "controlled_migration_pilot",
    "production_persistence",
    "production_indexing",
    "production_retrieval",
)
RECORD_STATUSES = ("valid", "valid_with_warnings", "review_required", "rejected")
PROVENANCE_STATUSES = (
    "full_provenance",
    "partial_provenance",
    "legacy_filename_only",
    "unknown_provenance",
)
MIGRATION_DISPOSITIONS = ("eligible", "eligible_with_approval", "quarantine", "forbidden")

REQUIRED_CONTROL_CODES = (
    "SHADOW_MODE_ONLY",
    "NO_DESTRUCTIVE_OVERWRITE",
    "NO_LEGACY_DELETION",
    "FULL_PROVENANCE_REQUIRED",
    "ZERO_REJECTED_RECORDS",
    "REVIEW_REQUIRED_RECORDS_QUARANTINED",
    "NO_EMBEDDINGS",
    "NO_ASTRA",
    "NO_FAISS",
    "NO_PRODUCTION_RETRIEVAL",
    "PACKAGE_DETERMINISM_REQUIRED",
    "ROLLBACK_PACKAGE_RETAINED",
    "WARNING_APPROVALS_SCOPED",
    "OCR_OBSERVATIONS_RETAINED",
    "SECURITY_REVIEW_PENDING",
)

SATISFIED_GATE_CODES = (
    "D5C_PASS",
    "D5D_ACCEPTABLE_OUTCOME",
    "THREE_OR_MORE_PROFILES_EVALUATED",
    "ACCEPTED_RECORDS_PRESENT",
    "ZERO_REJECTED_RECORDS",
    "ZERO_UNKNOWN_PROVENANCE",
    "ZERO_CHUNK_ID_COLLISIONS",
    "ALL_PROFILES_DETERMINISTIC",
    "SCHEMA_CONSISTENCY_VERIFIED",
    "NO_BLOCKING_ISSUES",
    "RUNTIME_INGESTION_UNCHANGED",
    "EMBEDDINGS_UNTOUCHED",
    "ASTRA_UNTOUCHED",
    "FAISS_UNTOUCHED",
)

TECHNICAL_BLOCKING_RULES = (
    ("D5C_NOT_PASS", lambda evidence: evidence.d5c_outcome != "PASS"),
    (
        "D5D_OUTCOME_BLOCKING",
        lambda evidence: evidence.d5d_outcome not in {"PASS", "ACCEPTED_WITH_LIMITATIONS"},
    ),
    ("PROFILE_COUNT_TOO_LOW", lambda evidence: evidence.evaluated_profile_count < 3),
    ("NO_ACCEPTED_RECORDS", lambda evidence: evidence.total_accepted_record_count <= 0),
    ("REJECTED_RECORDS_PRESENT", lambda evidence: evidence.total_rejected_candidate_count != 0),
    ("UNKNOWN_PROVENANCE_PRESENT", lambda evidence: evidence.unknown_provenance_count != 0),
    ("CHUNK_ID_COLLISION_PRESENT", lambda evidence: evidence.chunk_id_collision_count != 0),
    (
        "PROFILE_NONDETERMINISM",
        lambda evidence: evidence.deterministic_profile_count != evidence.expected_deterministic_profile_count,
    ),
    ("SCHEMA_CONSISTENCY_FAILED", lambda evidence: not evidence.schema_consistency_verified),
    ("BLOCKING_ISSUES_PRESENT", lambda evidence: bool(evidence.blocking_issue_codes)),
    ("RUNTIME_INGESTION_CHANGED", lambda evidence: not evidence.runtime_ingestion_unchanged),
    ("EMBEDDINGS_TOUCHED", lambda evidence: not evidence.embeddings_untouched),
    ("ASTRA_TOUCHED", lambda evidence: not evidence.astra_untouched),
    ("FAISS_TOUCHED", lambda evidence: not evidence.faiss_untouched),
)

CONDITIONAL_RULES = (
    (
        "REVIEW_REQUIRED_RECORDS_PRESENT",
        lambda evidence, policy: evidence.total_review_required_count > 0,
    ),
    (
        "APPROVED_LIMITATIONS_PRESENT",
        lambda evidence, policy: bool(evidence.accepted_limitation_codes),
    ),
    ("OCR_REVIEW_REQUIRED", lambda evidence, policy: evidence.ocr_observation_count > 0),
    (
        "SECURITY_DEPENDENCY_REVIEW_REQUIRED",
        lambda evidence, policy: evidence.unresolved_security_findings,
    ),
    (
        "PRODUCTION_RETENTION_DURATION_UNRESOLVED",
        lambda evidence, policy: not bool(policy.retention_policy.get("production_retention_duration_finalized")),
    ),
    (
        "PRODUCTION_WARNING_OWNER_SIGNOFF_REQUIRED",
        lambda evidence, policy: not bool(policy.warning_ownership.get("production_owner_signoff_complete")),
    ),
    (
        "PRODUCTION_LEGACY_CUTOVER_POLICY_REQUIRED",
        lambda evidence, policy: not bool(
            policy.legacy_coexistence_policy.get("production_cutover_policy_complete")
        ),
    ),
)

EVIDENCE_FIELDS = {
    "accepted_limitation_codes",
    "astra_untouched",
    "blocking_issue_codes",
    "chunk_id_collision_count",
    "d5c_outcome",
    "d5d_outcome",
    "deterministic_profile_count",
    "embeddings_untouched",
    "evidence_schema_name",
    "evidence_schema_version",
    "evaluated_profile_count",
    "expected_deterministic_profile_count",
    "faiss_untouched",
    "high_security_finding_count",
    "low_security_finding_count",
    "moderate_security_finding_count",
    "ocr_observation_count",
    "runtime_ingestion_unchanged",
    "schema_consistency_verified",
    "total_accepted_record_count",
    "total_candidate_count",
    "total_rejected_candidate_count",
    "total_review_required_count",
    "total_warning_count",
    "unknown_provenance_count",
    "unresolved_security_findings",
}
POLICY_FIELDS = {
    "decision_authority",
    "legacy_coexistence_policy",
    "limitation_policy",
    "ocr_policy",
    "phase_authorizations",
    "policy_name",
    "policy_version",
    "provenance_policy",
    "record_status_policy",
    "retention_policy",
    "security_gate",
    "table_policy",
    "warning_ownership",
}


@dataclass(frozen=True)
class PersistenceGovernancePolicy:
    """Machine-readable D.6 governance policy."""

    policy_name: str
    policy_version: str
    decision_authority: str
    record_status_policy: Mapping[str, Any]
    provenance_policy: Mapping[str, Any]
    warning_ownership: Mapping[str, Any]
    limitation_policy: Mapping[str, Any]
    table_policy: Mapping[str, Any]
    ocr_policy: Mapping[str, Any]
    legacy_coexistence_policy: Mapping[str, Any]
    retention_policy: Mapping[str, Any]
    security_gate: Mapping[str, Any]
    phase_authorizations: Mapping[str, Any]


@dataclass(frozen=True)
class PersistenceMigrationReadinessEvidence:
    """Sanitized D.6 evidence model."""

    d5c_outcome: str
    d5d_outcome: str
    evaluated_profile_count: int
    total_candidate_count: int
    total_accepted_record_count: int
    total_rejected_candidate_count: int
    total_warning_count: int
    total_review_required_count: int
    unknown_provenance_count: int
    chunk_id_collision_count: int
    deterministic_profile_count: int
    expected_deterministic_profile_count: int
    schema_consistency_verified: bool
    accepted_limitation_codes: tuple[str, ...]
    blocking_issue_codes: tuple[str, ...]
    ocr_observation_count: int
    unresolved_security_findings: bool
    high_security_finding_count: int
    moderate_security_finding_count: int
    low_security_finding_count: int
    runtime_ingestion_unchanged: bool
    embeddings_untouched: bool
    astra_untouched: bool
    faiss_untouched: bool
    evidence_schema_name: str = READINESS_EVIDENCE_SCHEMA_NAME
    evidence_schema_version: str = READINESS_EVIDENCE_SCHEMA_VERSION


@dataclass(frozen=True)
class PersistenceMigrationReadinessDecision:
    """Deterministic D.6 governance decision."""

    decision: str
    controlled_rehearsal_authorized: bool
    controlled_pilot_authorized: bool
    production_persistence_authorized: bool
    production_indexing_authorized: bool
    production_retrieval_authorized: bool
    satisfied_gate_codes: tuple[str, ...]
    conditional_gate_codes: tuple[str, ...]
    blocking_gate_codes: tuple[str, ...]
    required_controls: tuple[str, ...]
    unresolved_findings: tuple[str, ...]
    summary: Mapping[str, Any] = field(default_factory=dict)


def load_persistence_governance_policy(path: str | Path) -> PersistenceGovernancePolicy:
    """Load and validate a D.6 policy JSON file."""
    data = _load_object(path, "policy")
    _reject_unknown_fields(data, POLICY_FIELDS, "$")
    if data.get("policy_name") != POLICY_NAME:
        raise ValueError("Unsupported policy_name.")
    if data.get("policy_version") != POLICY_VERSION:
        raise ValueError("Unsupported policy_version.")
    policy = PersistenceGovernancePolicy(
        policy_name=_string(data, "policy_name"),
        policy_version=_string(data, "policy_version"),
        decision_authority=_string(data, "decision_authority"),
        record_status_policy=_mapping(data, "record_status_policy"),
        provenance_policy=_mapping(data, "provenance_policy"),
        warning_ownership=_mapping(data, "warning_ownership"),
        limitation_policy=_mapping(data, "limitation_policy"),
        table_policy=_mapping(data, "table_policy"),
        ocr_policy=_mapping(data, "ocr_policy"),
        legacy_coexistence_policy=_mapping(data, "legacy_coexistence_policy"),
        retention_policy=_mapping(data, "retention_policy"),
        security_gate=_mapping(data, "security_gate"),
        phase_authorizations=_mapping(data, "phase_authorizations"),
    )
    _validate_policy(policy)
    return policy


def load_persistence_migration_readiness_evidence(
    path: str | Path,
) -> PersistenceMigrationReadinessEvidence:
    """Load and validate sanitized D.6 evidence."""
    data = _load_object(path, "evidence")
    _reject_unknown_fields(data, EVIDENCE_FIELDS, "$")
    _reject_private_or_text_payloads(data)
    if data.get("evidence_schema_name") != READINESS_EVIDENCE_SCHEMA_NAME:
        raise ValueError("Unsupported evidence_schema_name.")
    if data.get("evidence_schema_version") != READINESS_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("Unsupported evidence_schema_version.")
    evidence = PersistenceMigrationReadinessEvidence(
        d5c_outcome=_string(data, "d5c_outcome"),
        d5d_outcome=_string(data, "d5d_outcome"),
        evaluated_profile_count=_count(data, "evaluated_profile_count"),
        total_candidate_count=_count(data, "total_candidate_count"),
        total_accepted_record_count=_count(data, "total_accepted_record_count"),
        total_rejected_candidate_count=_count(data, "total_rejected_candidate_count"),
        total_warning_count=_count(data, "total_warning_count"),
        total_review_required_count=_count(data, "total_review_required_count"),
        unknown_provenance_count=_count(data, "unknown_provenance_count"),
        chunk_id_collision_count=_count(data, "chunk_id_collision_count"),
        deterministic_profile_count=_count(data, "deterministic_profile_count"),
        expected_deterministic_profile_count=_count(data, "expected_deterministic_profile_count"),
        schema_consistency_verified=_bool(data, "schema_consistency_verified"),
        accepted_limitation_codes=_string_tuple(data.get("accepted_limitation_codes", ())),
        blocking_issue_codes=_string_tuple(data.get("blocking_issue_codes", ())),
        ocr_observation_count=_count(data, "ocr_observation_count"),
        unresolved_security_findings=_bool(data, "unresolved_security_findings"),
        high_security_finding_count=_count(data, "high_security_finding_count"),
        moderate_security_finding_count=_count(data, "moderate_security_finding_count"),
        low_security_finding_count=_count(data, "low_security_finding_count"),
        runtime_ingestion_unchanged=_bool(data, "runtime_ingestion_unchanged"),
        embeddings_untouched=_bool(data, "embeddings_untouched"),
        astra_untouched=_bool(data, "astra_untouched"),
        faiss_untouched=_bool(data, "faiss_untouched"),
        evidence_schema_name=_string(data, "evidence_schema_name"),
        evidence_schema_version=_string(data, "evidence_schema_version"),
    )
    _validate_evidence(evidence)
    return evidence


def evaluate_persistence_migration_readiness(
    evidence: PersistenceMigrationReadinessEvidence,
    *,
    policy: PersistenceGovernancePolicy | None = None,
) -> PersistenceMigrationReadinessDecision:
    """Evaluate D.6 readiness from sanitized evidence."""
    active_policy = policy or default_persistence_governance_policy()
    _validate_evidence(evidence)
    _validate_policy(active_policy)
    blocking = tuple(code for code, predicate in TECHNICAL_BLOCKING_RULES if predicate(evidence))
    satisfied = tuple(
        code
        for code, predicate in _satisfied_gate_predicates()
        if predicate(evidence)
    )
    conditional = () if blocking else tuple(
        code for code, predicate in CONDITIONAL_RULES if predicate(evidence, active_policy)
    )
    if blocking:
        decision = NOT_READY
    elif conditional:
        decision = CONDITIONALLY_READY_FOR_CONTROLLED_MIGRATION_REHEARSAL
    else:
        decision = READY_FOR_CONTROLLED_MIGRATION_REHEARSAL

    controlled_rehearsal = decision in {
        CONDITIONALLY_READY_FOR_CONTROLLED_MIGRATION_REHEARSAL,
        READY_FOR_CONTROLLED_MIGRATION_REHEARSAL,
    }
    unresolved = _unresolved_findings(evidence, active_policy, conditional, blocking)
    return PersistenceMigrationReadinessDecision(
        decision=decision,
        controlled_rehearsal_authorized=controlled_rehearsal,
        controlled_pilot_authorized=False,
        production_persistence_authorized=False,
        production_indexing_authorized=False,
        production_retrieval_authorized=False,
        satisfied_gate_codes=tuple(sorted(satisfied)),
        conditional_gate_codes=tuple(sorted(conditional)),
        blocking_gate_codes=tuple(sorted(blocking)),
        required_controls=REQUIRED_CONTROL_CODES,
        unresolved_findings=unresolved,
        summary={
            "policy_name": active_policy.policy_name,
            "policy_version": active_policy.policy_version,
            "record_status_policy": _plain(active_policy.record_status_policy),
            "provenance_policy": _plain(active_policy.provenance_policy),
            "phase_authorizations": _plain(active_policy.phase_authorizations),
            "security_status": {
                "high": evidence.high_security_finding_count,
                "moderate": evidence.moderate_security_finding_count,
                "low": evidence.low_security_finding_count,
                "unresolved_security_findings": evidence.unresolved_security_findings,
            },
            "ocr_status": {
                "ocr_observation_count": evidence.ocr_observation_count,
                "ocr_execution_authorized": False,
            },
            "legacy_coexistence_mode": active_policy.legacy_coexistence_policy.get("mode"),
            "retention_mode": active_policy.retention_policy.get("rehearsal_retention_mode"),
            "runtime_ingestion_modified": False,
            "migration_executed": False,
            "embedding_generation": False,
            "astra_touched": False,
            "faiss_touched": False,
            "production_authorization": False,
        },
    )


def default_persistence_governance_policy() -> PersistenceGovernancePolicy:
    """Return the built-in D.6 policy."""
    return policy_from_dict(default_persistence_governance_policy_dict())


def policy_from_dict(data: Mapping[str, Any]) -> PersistenceGovernancePolicy:
    """Construct and validate a policy from a mapping."""
    _reject_unknown_fields(data, POLICY_FIELDS, "$")
    policy = PersistenceGovernancePolicy(
        policy_name=_string(data, "policy_name"),
        policy_version=_string(data, "policy_version"),
        decision_authority=_string(data, "decision_authority"),
        record_status_policy=_mapping(data, "record_status_policy"),
        provenance_policy=_mapping(data, "provenance_policy"),
        warning_ownership=_mapping(data, "warning_ownership"),
        limitation_policy=_mapping(data, "limitation_policy"),
        table_policy=_mapping(data, "table_policy"),
        ocr_policy=_mapping(data, "ocr_policy"),
        legacy_coexistence_policy=_mapping(data, "legacy_coexistence_policy"),
        retention_policy=_mapping(data, "retention_policy"),
        security_gate=_mapping(data, "security_gate"),
        phase_authorizations=_mapping(data, "phase_authorizations"),
    )
    _validate_policy(policy)
    return policy


def default_persistence_governance_policy_dict() -> dict[str, Any]:
    """Return a deterministic JSON-serializable policy dictionary."""
    return {
        "decision_authority": "migration_governance_review",
        "legacy_coexistence_policy": {
            "mode": "shadow_mode_only",
            "no_automatic_cutover": True,
            "no_destructive_overwrite": True,
            "no_legacy_deletion": True,
            "production_cutover_policy_complete": False,
            "source_checksum_required": True,
            "structured_and_legacy_origins_separate": True,
        },
        "limitation_policy": {
            "approval_scope_required": True,
            "candidate_level_not_document_global": True,
            "corpus_wide_default": False,
            "known_codes": sorted(APPROVED_LIMITATION_CODES),
            "owner_roles": ["domain_safety_reviewer", "aviationrag_ingestion_owner", "migration_authority"],
        },
        "ocr_policy": {
            "controlled_codes": ["OCR_REVIEW_REQUIRED", "OCR_COMPLETENESS_NOT_ESTABLISHED"],
            "ocr_execution_authorized": False,
            "production_indexing_requires_review": True,
            "rehearsal_disposition": "quarantine",
        },
        "phase_authorizations": {
            "astra_operations": False,
            "controlled_local_persistence_package_generation": True,
            "controlled_shadow_migration_rehearsal": "conditional_yes",
            "dependency_remediation": "separate_task",
            "embedding_generation": False,
            "faiss_operations": False,
            "ocr_execution": "separate_task",
            "production_migration": False,
            "production_retrieval": False,
            "review_required_indexing": False,
            "valid_records_in_rehearsal": True,
            "valid_with_warnings_records": "approval_required",
        },
        "policy_name": POLICY_NAME,
        "policy_version": POLICY_VERSION,
        "provenance_policy": {
            "full_provenance": {
                "disposition": "eligible",
                "production": "future_approval_required",
                "rehearsal": "eligible",
            },
            "legacy_filename_only": {
                "disposition": "quarantine",
                "production": "blocked",
                "rehearsal": "legacy_path_only",
            },
            "partial_provenance": {
                "default": "disabled",
                "disposition": "quarantine",
                "indexing": "forbidden",
                "rehearsal": "disabled_by_default",
            },
            "unknown_provenance": {
                "disposition": "forbidden",
                "production": "forbidden",
                "rehearsal": "forbidden",
            },
        },
        "record_status_policy": {
            "rejected": {
                "controlled_pilot": "forbidden",
                "index_retrieval": "forbidden",
                "production_persistence": "forbidden",
                "rehearsal": "forbidden",
            },
            "review_required": {
                "controlled_pilot": "blocked_unless_separately_reviewed",
                "index_retrieval": "forbidden",
                "production_persistence": "blocked",
                "rehearsal": "quarantine",
            },
            "valid": {
                "controlled_pilot": "eligible",
                "index_retrieval": "future_approval_required",
                "production_persistence": "future_approval_required",
                "rehearsal": "eligible",
            },
            "valid_with_warnings": {
                "controlled_pilot": "eligible_with_approval",
                "index_retrieval": "blocked",
                "production_persistence": "blocked_pending_governance",
                "rehearsal": "eligible_with_approval",
            },
        },
        "retention_policy": {
            "no_automatic_deletion": True,
            "previous_package_retained": True,
            "production_retention_duration_finalized": False,
            "rehearsal_retention_mode": "indefinite_local_controlled_rehearsal",
            "replacement_requires_validation": True,
            "rollback_material_retained": True,
        },
        "security_gate": {
            "dependency_remediation_authorized": False,
            "findings_block_production": True,
            "high_findings_require_triage": True,
            "offline_rehearsal_allowed_without_network_or_deployment": True,
            "production_security_review_required": True,
        },
        "table_policy": {
            "TABLE_CANDIDATE_ONLY": {
                "cell_accuracy_claim_allowed": False,
                "disposition": "eligible_with_approval",
                "row_structure_claim_allowed": False,
            },
            "TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE": {
                "candidate_scope_required": True,
                "known_candidate_id": "aircraft_system_safety:chunk:page-52-table-1",
                "record_status": "review_required",
                "rehearsal_disposition": "quarantine",
            },
        },
        "warning_ownership": {
            "categories": {
                "adapter_mapping_warning": {
                    "primary_owner": "aviationrag_ingestion_owner",
                    "required_reviewers": ["repository_maintainer"],
                },
                "dependency_vulnerability": {
                    "primary_owner": "security_dependency_owner",
                    "required_reviewers": ["repository_maintainer", "migration_authority"],
                },
                "migration_authorization": {
                    "primary_owner": "migration_authority",
                    "required_reviewers": ["required_technical_owners"],
                },
                "ocr_required_or_uncertain_page": {
                    "primary_owner": "parser_extraction_owner",
                    "required_reviewers": ["domain_safety_reviewer", "migration_authority"],
                },
                "parser_extraction_warning": {
                    "primary_owner": "parser_extraction_owner",
                    "required_reviewers": ["aviationrag_ingestion_owner"],
                },
                "provenance_limitation": {
                    "primary_owner": "aviationrag_ingestion_owner",
                    "required_reviewers": ["migration_authority"],
                },
                "safety_content_limitation": {
                    "primary_owner": "domain_safety_reviewer",
                    "required_reviewers": ["aviationrag_ingestion_owner", "migration_authority"],
                },
                "table_classification_ambiguity": {
                    "primary_owner": "domain_safety_reviewer",
                    "required_reviewers": ["aviationrag_ingestion_owner"],
                },
            },
            "production_owner_signoff_complete": False,
        },
    }


def evidence_to_dict(evidence: PersistenceMigrationReadinessEvidence) -> dict[str, Any]:
    """Return deterministic evidence JSON."""
    return {
        "accepted_limitation_codes": list(evidence.accepted_limitation_codes),
        "astra_untouched": evidence.astra_untouched,
        "blocking_issue_codes": list(evidence.blocking_issue_codes),
        "chunk_id_collision_count": evidence.chunk_id_collision_count,
        "d5c_outcome": evidence.d5c_outcome,
        "d5d_outcome": evidence.d5d_outcome,
        "deterministic_profile_count": evidence.deterministic_profile_count,
        "embeddings_untouched": evidence.embeddings_untouched,
        "evidence_schema_name": evidence.evidence_schema_name,
        "evidence_schema_version": evidence.evidence_schema_version,
        "evaluated_profile_count": evidence.evaluated_profile_count,
        "expected_deterministic_profile_count": evidence.expected_deterministic_profile_count,
        "faiss_untouched": evidence.faiss_untouched,
        "high_security_finding_count": evidence.high_security_finding_count,
        "low_security_finding_count": evidence.low_security_finding_count,
        "moderate_security_finding_count": evidence.moderate_security_finding_count,
        "ocr_observation_count": evidence.ocr_observation_count,
        "runtime_ingestion_unchanged": evidence.runtime_ingestion_unchanged,
        "schema_consistency_verified": evidence.schema_consistency_verified,
        "total_accepted_record_count": evidence.total_accepted_record_count,
        "total_candidate_count": evidence.total_candidate_count,
        "total_rejected_candidate_count": evidence.total_rejected_candidate_count,
        "total_review_required_count": evidence.total_review_required_count,
        "total_warning_count": evidence.total_warning_count,
        "unknown_provenance_count": evidence.unknown_provenance_count,
        "unresolved_security_findings": evidence.unresolved_security_findings,
    }


def decision_to_dict(decision: PersistenceMigrationReadinessDecision) -> dict[str, Any]:
    """Return deterministic decision JSON."""
    return {
        "authorization": {
            "controlled_migration_pilot": decision.controlled_pilot_authorized,
            "controlled_migration_rehearsal": decision.controlled_rehearsal_authorized,
            "production_indexing": decision.production_indexing_authorized,
            "production_persistence": decision.production_persistence_authorized,
            "production_retrieval": decision.production_retrieval_authorized,
        },
        "blocking_gate_codes": list(decision.blocking_gate_codes),
        "conditional_gate_codes": list(decision.conditional_gate_codes),
        "controlled_pilot_authorized": decision.controlled_pilot_authorized,
        "controlled_rehearsal_authorized": decision.controlled_rehearsal_authorized,
        "decision": decision.decision,
        "decision_schema_name": READINESS_DECISION_SCHEMA_NAME,
        "decision_schema_version": READINESS_DECISION_SCHEMA_VERSION,
        "policy_name": decision.summary.get("policy_name"),
        "policy_version": decision.summary.get("policy_version"),
        "production_indexing_authorized": decision.production_indexing_authorized,
        "production_persistence_authorized": decision.production_persistence_authorized,
        "production_retrieval_authorized": decision.production_retrieval_authorized,
        "required_controls": list(decision.required_controls),
        "satisfied_gate_codes": list(decision.satisfied_gate_codes),
        "summary": _plain(decision.summary),
        "unresolved_findings": list(decision.unresolved_findings),
    }


def decision_report_json_bytes(decision: PersistenceMigrationReadinessDecision) -> bytes:
    """Serialize a deterministic JSON decision report."""
    return _json_bytes(decision_to_dict(decision))


def decision_report_markdown(decision: PersistenceMigrationReadinessDecision) -> str:
    """Serialize a deterministic Markdown decision report."""
    lines = [
        "# Persistence Migration Readiness Decision",
        "",
        f"Decision: `{decision.decision}`",
        "",
        "## Authorization",
        "",
        f"- Controlled rehearsal authorized: `{str(decision.controlled_rehearsal_authorized).lower()}`",
        f"- Controlled pilot authorized: `{str(decision.controlled_pilot_authorized).lower()}`",
        f"- Production persistence authorized: `{str(decision.production_persistence_authorized).lower()}`",
        f"- Production indexing authorized: `{str(decision.production_indexing_authorized).lower()}`",
        f"- Production retrieval authorized: `{str(decision.production_retrieval_authorized).lower()}`",
        "",
        "## Satisfied Gates",
        "",
        *[f"- `{code}`" for code in decision.satisfied_gate_codes],
        "",
        "## Conditional Gates",
        "",
        *[f"- `{code}`" for code in decision.conditional_gate_codes],
        "",
        "## Blocking Gates",
        "",
        *[f"- `{code}`" for code in decision.blocking_gate_codes],
        "",
        "## Required Controls",
        "",
        *[f"- `{code}`" for code in decision.required_controls],
        "",
    ]
    return "\n".join(lines) + "\n"


def _satisfied_gate_predicates():
    return (
        ("D5C_PASS", lambda evidence: evidence.d5c_outcome == "PASS"),
        (
            "D5D_ACCEPTABLE_OUTCOME",
            lambda evidence: evidence.d5d_outcome in {"PASS", "ACCEPTED_WITH_LIMITATIONS"},
        ),
        ("THREE_OR_MORE_PROFILES_EVALUATED", lambda evidence: evidence.evaluated_profile_count >= 3),
        ("ACCEPTED_RECORDS_PRESENT", lambda evidence: evidence.total_accepted_record_count > 0),
        ("ZERO_REJECTED_RECORDS", lambda evidence: evidence.total_rejected_candidate_count == 0),
        ("ZERO_UNKNOWN_PROVENANCE", lambda evidence: evidence.unknown_provenance_count == 0),
        ("ZERO_CHUNK_ID_COLLISIONS", lambda evidence: evidence.chunk_id_collision_count == 0),
        (
            "ALL_PROFILES_DETERMINISTIC",
            lambda evidence: evidence.deterministic_profile_count == evidence.expected_deterministic_profile_count,
        ),
        ("SCHEMA_CONSISTENCY_VERIFIED", lambda evidence: evidence.schema_consistency_verified),
        ("NO_BLOCKING_ISSUES", lambda evidence: not evidence.blocking_issue_codes),
        ("RUNTIME_INGESTION_UNCHANGED", lambda evidence: evidence.runtime_ingestion_unchanged),
        ("EMBEDDINGS_UNTOUCHED", lambda evidence: evidence.embeddings_untouched),
        ("ASTRA_UNTOUCHED", lambda evidence: evidence.astra_untouched),
        ("FAISS_UNTOUCHED", lambda evidence: evidence.faiss_untouched),
    )


def _validate_evidence(evidence: PersistenceMigrationReadinessEvidence) -> None:
    if evidence.d5c_outcome not in {"PASS", "REVIEW", "FAIL"}:
        raise ValueError("Unsupported d5c_outcome.")
    if evidence.d5d_outcome not in {"PASS", "ACCEPTED_WITH_LIMITATIONS", "FAIL"}:
        raise ValueError("Unsupported d5d_outcome.")
    if evidence.total_accepted_record_count > evidence.total_candidate_count:
        raise ValueError("Accepted record count cannot exceed candidate count.")
    if evidence.deterministic_profile_count > evidence.expected_deterministic_profile_count:
        raise ValueError("Deterministic profile count cannot exceed expected count.")
    unknown_limitations = set(evidence.accepted_limitation_codes) - set(APPROVED_LIMITATION_CODES)
    if unknown_limitations:
        raise ValueError("Unknown limitation code(s): " + ", ".join(sorted(unknown_limitations)))


def _validate_policy(policy: PersistenceGovernancePolicy) -> None:
    if policy.policy_name != POLICY_NAME:
        raise ValueError("Unsupported policy name.")
    if policy.policy_version != POLICY_VERSION:
        raise ValueError("Unsupported policy version.")
    if policy.decision_authority != "migration_governance_review":
        raise ValueError("Unsupported decision authority.")
    for status in RECORD_STATUSES:
        if status not in policy.record_status_policy:
            raise ValueError(f"Missing record status policy: {status}")
    for provenance in PROVENANCE_STATUSES:
        if provenance not in policy.provenance_policy:
            raise ValueError(f"Missing provenance policy: {provenance}")
    for phase in FUTURE_EXECUTION_PHASES:
        if phase not in {
            "controlled_migration_rehearsal",
            "controlled_migration_pilot",
            "production_persistence",
            "production_indexing",
            "production_retrieval",
        }:
            raise ValueError(f"Unsupported phase: {phase}")


def _unresolved_findings(
    evidence: PersistenceMigrationReadinessEvidence,
    policy: PersistenceGovernancePolicy,
    conditional: tuple[str, ...],
    blocking: tuple[str, ...],
) -> tuple[str, ...]:
    findings = list(blocking)
    findings.extend(conditional)
    if evidence.high_security_finding_count:
        findings.append("HIGH_SECURITY_FINDINGS_REMAIN")
    if policy.retention_policy.get("production_retention_duration_finalized") is False:
        findings.append("PRODUCTION_RETENTION_DURATION_UNRESOLVED")
    return tuple(sorted(set(findings)))


def _load_object(path: str | Path, label: str) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(data, dict):
        raise ValueError(f"{label} JSON must be an object.")
    return data


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


def _reject_private_or_text_payloads(data: Mapping[str, Any]) -> None:
    forbidden_keys = {"source_text", "chunk_text", "text", "absolute_path", "local_path", "path"}
    for key, value in data.items():
        if key in forbidden_keys:
            raise ValueError(f"Forbidden evidence field: {key}")
        _reject_absolute_path_value(value, key)


def _reject_absolute_path_value(value: Any, path: str) -> None:
    if isinstance(value, str) and _looks_like_absolute_path(value):
        raise ValueError(f"Forbidden absolute path value at {path}")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_absolute_path_value(item, f"{path}[{index}]")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_absolute_path_value(item, f"{path}.{key}")


def _looks_like_absolute_path(value: str) -> bool:
    return bool(re.search(r"[A-Za-z]:[\\/]", value) or value.startswith("/") or value.startswith("\\\\"))


def _mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object.")
    return dict(value)


def _string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string.")
    return value


def _count(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{key} must be a non-negative integer.")
    return value


def _bool(data: Mapping[str, Any], key: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean.")
    return value


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("Expected an array of strings.")
    result = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise ValueError("Expected an array of non-empty strings.")
        result.append(item)
    return tuple(sorted(set(result)))


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(_plain(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


__all__ = [
    "CONDITIONALLY_READY_FOR_CONTROLLED_MIGRATION_REHEARSAL",
    "FUTURE_EXECUTION_PHASES",
    "MIGRATION_DISPOSITIONS",
    "NOT_READY",
    "POLICY_NAME",
    "POLICY_VERSION",
    "PROVENANCE_STATUSES",
    "PersistenceGovernancePolicy",
    "PersistenceMigrationReadinessDecision",
    "PersistenceMigrationReadinessEvidence",
    "READINESS_DECISION_SCHEMA_NAME",
    "READINESS_DECISION_SCHEMA_VERSION",
    "READINESS_EVIDENCE_SCHEMA_NAME",
    "READINESS_EVIDENCE_SCHEMA_VERSION",
    "READINESS_DECISIONS",
    "READY_FOR_CONTROLLED_MIGRATION_REHEARSAL",
    "RECORD_STATUSES",
    "REQUIRED_CONTROL_CODES",
    "decision_report_json_bytes",
    "decision_report_markdown",
    "decision_to_dict",
    "default_persistence_governance_policy",
    "default_persistence_governance_policy_dict",
    "evaluate_persistence_migration_readiness",
    "evidence_to_dict",
    "load_persistence_governance_policy",
    "load_persistence_migration_readiness_evidence",
    "policy_from_dict",
]
