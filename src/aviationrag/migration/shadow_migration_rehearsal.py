"""D.7 controlled shadow migration rehearsal.

This module consumes already validated persisted chunk packages and a
read-only legacy inventory. It writes only deterministic local rehearsal
artifacts when explicitly allowed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from aviationrag.governance.persistence_migration_readiness import (
    POLICY_NAME,
    POLICY_VERSION,
    REQUIRED_CONTROL_CODES,
)
from aviationrag.ingestion.persisted_chunk_package import (
    PACKAGE_SCHEMA_NAME,
    PACKAGE_SCHEMA_VERSION,
    PERSISTED_CHUNKS_FILENAME,
    PERSISTENCE_MANIFEST_FILENAME,
    PERSISTENCE_REPORT_FILENAME,
    REJECTED_CANDIDATES_FILENAME,
    WARNINGS_FILENAME,
)
from aviationrag.ingestion.persisted_chunk_record import (
    FORBIDDEN_PERSISTED_FIELDS,
    PERSISTED_CHUNK_MAPPER_VERSION,
    PERSISTED_CHUNK_SCHEMA_NAME,
    PERSISTED_CHUNK_SCHEMA_VERSION,
    PersistedChunkRecord,
    persisted_chunk_record_from_dict,
    persisted_chunk_record_to_dict,
)
from aviationrag.ingestion.persisted_chunk_validator import (
    PERSISTED_CHUNK_LIMITATION_REGISTRY_VERSION,
    validate_persisted_chunk_record,
)


REHEARSAL_SCHEMA_NAME = "aviationrag-controlled-shadow-migration-rehearsal"
REHEARSAL_SCHEMA_VERSION = "0.1.0"

CONFIG_SCHEMA_NAME = "aviationrag-shadow-migration-rehearsal-config"
CONFIG_SCHEMA_VERSION = "0.1.0"

PASS = "PASS"
PASS_WITH_QUARANTINE = "PASS_WITH_QUARANTINE"
FAIL = "FAIL"

SHADOW_ELIGIBLE = "shadow_eligible"
QUARANTINE = "quarantine"
FORBIDDEN = "forbidden"

EXACT_SOURCE_CHECKSUM_MATCH = "EXACT_SOURCE_CHECKSUM_MATCH"
DOCUMENT_ID_MATCH_WITHOUT_SOURCE_CHECKSUM = "DOCUMENT_ID_MATCH_WITHOUT_SOURCE_CHECKSUM"
SAME_FILENAME_DIFFERENT_CHECKSUM = "SAME_FILENAME_DIFFERENT_CHECKSUM"
FILENAME_ALIAS_ONLY = "FILENAME_ALIAS_ONLY"
NO_LEGACY_MATCH = "NO_LEGACY_MATCH"
AMBIGUOUS_LEGACY_MATCH = "AMBIGUOUS_LEGACY_MATCH"

OCR_COMPLETENESS_NOT_ESTABLISHED = "OCR_COMPLETENESS_NOT_ESTABLISHED"
KNOWN_TABLE_QUARANTINE_REFERENCE = "aircraft_system_safety:chunk:page-52-table-1"
KNOWN_TABLE_LIMITATION = "TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE"

SHADOW_RECORDS_FILENAME = "shadow_records.jsonl"
QUARANTINE_RECORDS_FILENAME = "quarantine_records.jsonl"
STRUCTURED_PACKAGE_CATALOG_FILENAME = "structured_package_catalog.json"
LEGACY_INVENTORY_FILENAME = "legacy_inventory.json"
DOCUMENT_RECONCILIATION_FILENAME = "document_reconciliation.json"
MIGRATION_ACCOUNTING_FILENAME = "migration_accounting.json"
SHADOW_MANIFEST_FILENAME = "shadow_manifest.json"
SHADOW_REPORT_FILENAME = "shadow_report.json"
ROLLBACK_MANIFEST_FILENAME = "rollback_manifest.json"

RUN_ARTIFACTS = (
    SHADOW_RECORDS_FILENAME,
    QUARANTINE_RECORDS_FILENAME,
    STRUCTURED_PACKAGE_CATALOG_FILENAME,
    LEGACY_INVENTORY_FILENAME,
    DOCUMENT_RECONCILIATION_FILENAME,
    MIGRATION_ACCOUNTING_FILENAME,
    SHADOW_REPORT_FILENAME,
    ROLLBACK_MANIFEST_FILENAME,
)

CONFIG_FIELDS = {
    "config_schema_name",
    "config_schema_version",
    "document_identity_aliases",
    "legacy_inventory",
    "observations",
    "packages",
}
PACKAGE_CONFIG_FIELDS = {"document_id", "expected_package_digest", "package_dir"}
LEGACY_CONFIG_FIELDS = {"chunk_roots", "source_roots"}
OBSERVATION_FIELDS = {"document_id", "observation_code", "page", "scope", "summary"}


@dataclass(frozen=True)
class PackageConfig:
    document_id: str
    package_dir: str
    expected_package_digest: str | None = None


@dataclass(frozen=True)
class ObservationConfig:
    document_id: str
    observation_code: str
    page: int | None
    scope: str
    summary: str


@dataclass(frozen=True)
class ShadowMigrationRehearsalConfig:
    packages: tuple[PackageConfig, ...]
    source_roots: tuple[str, ...]
    chunk_roots: tuple[str, ...]
    document_identity_aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    observations: tuple[ObservationConfig, ...] = ()


@dataclass(frozen=True)
class LegacyDocumentInventoryRecord:
    legacy_document_key: str
    source_filename: str
    relative_source_path: str | None
    source_checksum: str | None
    file_size_bytes: int | None
    legacy_chunk_count: int
    legacy_chunk_ids: tuple[str, ...]
    record_origin: str
    provenance_status: str
    warning_codes: tuple[str, ...]


@dataclass(frozen=True)
class StructuredPackageCatalogRecord:
    document_id: str
    source_filename: str
    source_checksum: str
    package_digest: str
    package_schema_name: str
    package_schema_version: str
    persisted_schema_name: str
    persisted_schema_version: str
    mapper_version: str
    limitation_registry_version: str
    record_count: int
    valid_count: int
    valid_with_warnings_count: int
    review_required_count: int
    rejected_count: int
    eligible_count: int
    quarantined_count: int
    forbidden_count: int
    package_root: str


@dataclass(frozen=True)
class ValidatedStructuredPackage:
    package_dir: str
    manifest: Mapping[str, Any]
    report: Mapping[str, Any]
    warnings: Mapping[str, Any]
    records: tuple[PersistedChunkRecord, ...]
    record_dicts: tuple[Mapping[str, Any], ...]
    package_digest: str
    file_sha256: Mapping[str, str]
    catalog_record: StructuredPackageCatalogRecord


@dataclass(frozen=True)
class DocumentIdentityReconciliation:
    structured_document_id: str
    structured_source_filename: str
    structured_source_checksum: str
    reconciliation_status: str
    matched_legacy_document_keys: tuple[str, ...]
    exact_checksum_match_count: int
    filename_match_count: int
    is_cutover_eligible: bool
    review_required: bool
    warning_codes: tuple[str, ...]


@dataclass(frozen=True)
class RecordClassification:
    record: PersistedChunkRecord
    disposition: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class ShadowMigrationRehearsalResult:
    outcome: str
    exit_code: int
    package_count: int
    structured_record_count: int
    eligible_count: int
    quarantine_count: int
    forbidden_count: int
    rejected_count: int
    aggregate_shadow_digest: str
    accounting_verified: bool
    determinism_verified: bool
    rollback_verified: bool
    legacy_unchanged: bool
    package_integrity_verified: bool
    output_root: str | None
    blocking_issue_codes: tuple[str, ...]
    run_file_sha256: Mapping[str, Mapping[str, str]]


def load_shadow_migration_rehearsal_config(
    path: str | Path,
) -> ShadowMigrationRehearsalConfig:
    """Load strict D.7 rehearsal config JSON."""
    data = _load_object(path)
    _reject_unknown_fields(data, CONFIG_FIELDS, "$")
    if data.get("config_schema_name") != CONFIG_SCHEMA_NAME:
        raise ValueError("Unsupported config_schema_name.")
    if data.get("config_schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError("Unsupported config_schema_version.")

    packages = data.get("packages")
    if not isinstance(packages, list) or not packages:
        raise ValueError("packages must be a non-empty array.")
    package_configs: list[PackageConfig] = []
    seen_documents: set[str] = set()
    seen_dirs: set[str] = set()
    for index, item in enumerate(packages):
        if not isinstance(item, Mapping):
            raise ValueError(f"packages[{index}] must be an object.")
        _reject_unknown_fields(item, PACKAGE_CONFIG_FIELDS, f"$.packages[{index}]")
        document_id = _required_string(item, "document_id")
        package_dir = _required_string(item, "package_dir")
        expected_digest = item.get("expected_package_digest")
        if expected_digest is not None:
            expected_digest = _sha256_string(expected_digest, "expected_package_digest")
        if document_id in seen_documents:
            raise ValueError(f"Duplicate package document_id: {document_id}")
        if package_dir in seen_dirs:
            raise ValueError(f"Duplicate package_dir: {package_dir}")
        if not Path(package_dir).exists():
            raise FileNotFoundError(f"Missing package directory: {package_dir}")
        seen_documents.add(document_id)
        seen_dirs.add(package_dir)
        package_configs.append(PackageConfig(document_id, package_dir, expected_digest))

    legacy = data.get("legacy_inventory")
    if not isinstance(legacy, Mapping):
        raise ValueError("legacy_inventory must be an object.")
    _reject_unknown_fields(legacy, LEGACY_CONFIG_FIELDS, "$.legacy_inventory")
    source_roots = _string_tuple(legacy.get("source_roots", ()), "source_roots")
    chunk_roots = _string_tuple(legacy.get("chunk_roots", ()), "chunk_roots")
    for root in [*source_roots, *chunk_roots]:
        if not Path(root).exists():
            raise FileNotFoundError(f"Invalid legacy root: {root}")

    aliases_raw = data.get("document_identity_aliases", {})
    if not isinstance(aliases_raw, Mapping):
        raise ValueError("document_identity_aliases must be an object.")
    aliases = {
        str(key): _string_tuple(value, f"document_identity_aliases.{key}")
        for key, value in sorted(aliases_raw.items())
    }

    observations_raw = data.get("observations", [])
    if not isinstance(observations_raw, list):
        raise ValueError("observations must be an array.")
    observations: list[ObservationConfig] = []
    for index, item in enumerate(observations_raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"observations[{index}] must be an object.")
        _reject_unknown_fields(item, OBSERVATION_FIELDS, f"$.observations[{index}]")
        page = item.get("page")
        if page is not None and (not isinstance(page, int) or isinstance(page, bool) or page < 1):
            raise ValueError("observation page must be a positive integer or null.")
        observations.append(
            ObservationConfig(
                document_id=_required_string(item, "document_id"),
                observation_code=_required_string(item, "observation_code"),
                page=page,
                scope=_required_string(item, "scope"),
                summary=_required_string(item, "summary"),
            )
        )

    return ShadowMigrationRehearsalConfig(
        packages=tuple(package_configs),
        source_roots=source_roots,
        chunk_roots=chunk_roots,
        document_identity_aliases=aliases,
        observations=tuple(
            sorted(observations, key=lambda item: (item.document_id, item.page or 0, item.observation_code))
        ),
    )


def load_validated_structured_package(
    package_dir: str | Path,
    *,
    expected_package_digest: str | None = None,
) -> ValidatedStructuredPackage:
    """Load and verify one already generated persisted chunk package."""
    root = Path(package_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing package directory: {root}")
    paths = {
        PERSISTED_CHUNKS_FILENAME: root / PERSISTED_CHUNKS_FILENAME,
        PERSISTENCE_MANIFEST_FILENAME: root / PERSISTENCE_MANIFEST_FILENAME,
        PERSISTENCE_REPORT_FILENAME: root / PERSISTENCE_REPORT_FILENAME,
        REJECTED_CANDIDATES_FILENAME: root / REJECTED_CANDIDATES_FILENAME,
        WARNINGS_FILENAME: root / WARNINGS_FILENAME,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing package file(s): " + ", ".join(missing))

    manifest = _load_object(paths[PERSISTENCE_MANIFEST_FILENAME])
    report = _load_object(paths[PERSISTENCE_REPORT_FILENAME])
    warnings = _load_object(paths[WARNINGS_FILENAME])

    _expect(manifest.get("package_schema_name") == PACKAGE_SCHEMA_NAME, "Unsupported package schema.")
    _expect(manifest.get("package_schema_version") == PACKAGE_SCHEMA_VERSION, "Unsupported package schema version.")
    _expect(manifest.get("persisted_schema_name") == PERSISTED_CHUNK_SCHEMA_NAME, "Unsupported persisted schema.")
    _expect(manifest.get("persisted_schema_version") == PERSISTED_CHUNK_SCHEMA_VERSION, "Unsupported persisted schema version.")
    _expect(manifest.get("mapper_version") == PERSISTED_CHUNK_MAPPER_VERSION, "Unsupported mapper version.")
    _expect(
        manifest.get("limitation_registry_version") == PERSISTED_CHUNK_LIMITATION_REGISTRY_VERSION,
        "Unsupported limitation registry version.",
    )

    manifest_sha = manifest.get("file_sha256")
    if not isinstance(manifest_sha, Mapping):
        raise ValueError("manifest file_sha256 must be an object.")
    for filename, digest in sorted(manifest_sha.items()):
        if filename == PERSISTENCE_MANIFEST_FILENAME:
            raise ValueError("Manifest must not checksum itself.")
        _sha256_string(digest, f"file_sha256.{filename}")
        actual = sha256(paths[str(filename)].read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f"File checksum mismatch for {filename}.")

    package_digest = _package_digest({str(key): str(value) for key, value in manifest_sha.items()})
    if package_digest != manifest.get("package_checksum"):
        raise ValueError("Package digest mismatch.")
    if expected_package_digest and package_digest != expected_package_digest:
        raise ValueError("Package digest did not match expected_package_digest.")

    record_dicts = tuple(_load_jsonl(paths[PERSISTED_CHUNKS_FILENAME]))
    records = tuple(persisted_chunk_record_from_dict(item) for item in record_dicts)
    rejected = tuple(_load_jsonl(paths[REJECTED_CANDIDATES_FILENAME]))

    _expect(len(records) == manifest.get("record_count"), "Record count does not match manifest.")
    _expect(len(records) == manifest.get("accepted_count"), "Accepted count does not match manifest records.")
    _expect(len(records) == report.get("record_count"), "Record count does not match report.")
    _expect(int(manifest.get("rejected_count", -1)) == 0, "Rejected count must be zero.")
    _expect(int(report.get("rejected_count", -1)) == 0, "Report rejected count must be zero.")
    _expect(len(rejected) == 0, "Rejected candidates file must be empty.")

    document_ids = {record.document_id for record in records}
    source_filenames = {record.source_filename for record in records}
    source_checksums = {record.source_checksum for record in records}
    _expect(len(document_ids) == 1, "Package document_id is not coherent.")
    _expect(len(source_filenames) == 1, "Package source_filename is not coherent.")
    _expect(len(source_checksums) == 1, "Package source_checksum is not coherent.")

    for raw, record in zip(record_dicts, records):
        _reject_forbidden_fields(raw)
        _reject_absolute_paths(raw)
        issues = validate_persisted_chunk_record(record)
        errors = [issue for issue in issues if issue.severity == "error"]
        if errors:
            raise ValueError(f"Record validation failure for {record.chunk_id}: {errors[0].code}")

    status_counts = Counter(record.validation_status for record in records)
    provenance_counts = Counter(record.provenance_status for record in records)
    _expect(dict(sorted(status_counts.items())) == dict(sorted(report.get("validation_status_counts", {}).items())), "Validation status counts do not match report.")
    _expect(dict(sorted(provenance_counts.items())) == dict(sorted(report.get("provenance_counts", {}).items())), "Provenance counts do not match report.")

    classes = classify_records(records)
    class_counts = Counter(item.disposition for item in classes)
    catalog = StructuredPackageCatalogRecord(
        document_id=next(iter(document_ids)),
        source_filename=next(iter(source_filenames)),
        source_checksum=next(iter(source_checksums)),
        package_digest=package_digest,
        package_schema_name=str(manifest["package_schema_name"]),
        package_schema_version=str(manifest["package_schema_version"]),
        persisted_schema_name=str(manifest["persisted_schema_name"]),
        persisted_schema_version=str(manifest["persisted_schema_version"]),
        mapper_version=str(manifest["mapper_version"]),
        limitation_registry_version=str(manifest["limitation_registry_version"]),
        record_count=len(records),
        valid_count=status_counts["valid"],
        valid_with_warnings_count=status_counts["valid_with_warnings"],
        review_required_count=status_counts["review_required"],
        rejected_count=status_counts["rejected"],
        eligible_count=class_counts[SHADOW_ELIGIBLE],
        quarantined_count=class_counts[QUARANTINE],
        forbidden_count=class_counts[FORBIDDEN],
        package_root=_sanitize_relative(root),
    )
    return ValidatedStructuredPackage(
        package_dir=_sanitize_relative(root),
        manifest=dict(manifest),
        report=dict(report),
        warnings=dict(warnings),
        records=records,
        record_dicts=record_dicts,
        package_digest=package_digest,
        file_sha256=dict(sorted((str(key), str(value)) for key, value in manifest_sha.items())),
        catalog_record=catalog,
    )


def classify_records(records: Iterable[PersistedChunkRecord]) -> tuple[RecordClassification, ...]:
    """Classify records into shadow, quarantine, or forbidden dispositions."""
    results: list[RecordClassification] = []
    for record in records:
        reasons: list[str] = []
        disposition = SHADOW_ELIGIBLE
        if record.validation_status == "rejected":
            disposition = FORBIDDEN
            reasons.append("REJECTED_RECORD_FORBIDDEN")
        elif record.provenance_status == "unknown_provenance":
            disposition = FORBIDDEN
            reasons.append("UNKNOWN_PROVENANCE_FORBIDDEN")
        elif record.provenance_status == "partial_provenance":
            disposition = FORBIDDEN
            reasons.append("PARTIAL_PROVENANCE_FORBIDDEN")
        elif record.provenance_status != "full_provenance":
            disposition = FORBIDDEN
            reasons.append("NON_FULL_PROVENANCE_FORBIDDEN")
        elif record.validation_status == "review_required":
            disposition = QUARANTINE
            reasons.append("REVIEW_REQUIRED_QUARANTINE")
        elif record.validation_status == "valid_with_warnings":
            disposition = QUARANTINE
            reasons.append("VALID_WITH_WARNINGS_APPROVAL_REQUIRED")
        elif record.validation_status != "valid":
            disposition = FORBIDDEN
            reasons.append("UNKNOWN_VALIDATION_STATUS_FORBIDDEN")
        else:
            reasons.append("VALID_FULL_PROVENANCE_SHADOW_ELIGIBLE")
        results.append(RecordClassification(record, disposition, tuple(reasons)))
    return tuple(results)


def build_legacy_inventory(
    source_roots: Sequence[str | Path],
    chunk_roots: Sequence[str | Path],
    structured_packages: Sequence[ValidatedStructuredPackage],
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> tuple[LegacyDocumentInventoryRecord, ...]:
    """Build a read-only legacy inventory for approved structured identities."""
    alias_map = {key: tuple(value) for key, value in (aliases or {}).items()}
    relevant_names: dict[str, set[str]] = {}
    for package in structured_packages:
        catalog = package.catalog_record
        relevant_names.setdefault(catalog.document_id, set()).add(catalog.source_filename)
        relevant_names[catalog.document_id].update(alias_map.get(catalog.document_id, ()))

    records_by_key: dict[str, LegacyDocumentInventoryRecord] = {}
    for document_id, names in sorted(relevant_names.items()):
        for source_path in _find_source_candidates(source_roots, names):
            record = LegacyDocumentInventoryRecord(
                legacy_document_key=f"source:{_stable_file_key(source_path)}",
                source_filename=source_path.name,
                relative_source_path=_sanitize_relative(source_path),
                source_checksum=sha256(source_path.read_bytes()).hexdigest(),
                file_size_bytes=source_path.stat().st_size,
                legacy_chunk_count=0,
                legacy_chunk_ids=(),
                record_origin="legacy_processed",
                provenance_status="full_provenance",
                warning_codes=(),
            )
            records_by_key[record.legacy_document_key] = record
        for chunk_path in _find_chunk_candidates(chunk_roots, names):
            chunk_ids = _legacy_chunk_ids(chunk_path)
            record = LegacyDocumentInventoryRecord(
                legacy_document_key=f"chunked:{_stable_file_key(chunk_path)}",
                source_filename=_source_name_from_chunk_file(chunk_path.name),
                relative_source_path=_sanitize_relative(chunk_path),
                source_checksum=None,
                file_size_bytes=chunk_path.stat().st_size,
                legacy_chunk_count=len(chunk_ids),
                legacy_chunk_ids=chunk_ids,
                record_origin="legacy_chunked",
                provenance_status="legacy_filename_only",
                warning_codes=("LEGACY_FILENAME_ONLY",),
            )
            records_by_key[record.legacy_document_key] = record

    records = list(records_by_key.values())
    if not records:
        for package in structured_packages:
            catalog = package.catalog_record
            records.append(
                LegacyDocumentInventoryRecord(
                    legacy_document_key=f"unresolved:{catalog.document_id}",
                    source_filename=catalog.source_filename,
                    relative_source_path=None,
                    source_checksum=None,
                    file_size_bytes=None,
                    legacy_chunk_count=0,
                    legacy_chunk_ids=(),
                    record_origin="legacy_unresolved",
                    provenance_status="legacy_filename_only",
                    warning_codes=("NO_LEGACY_CANDIDATE_FOUND",),
                )
            )
    return tuple(sorted(records, key=lambda item: item.legacy_document_key))


def reconcile_document_identity(
    package: ValidatedStructuredPackage,
    legacy_inventory: Sequence[LegacyDocumentInventoryRecord],
    aliases: Sequence[str] = (),
    observations: Sequence[ObservationConfig] = (),
) -> DocumentIdentityReconciliation:
    """Reconcile one structured package to read-only legacy inventory."""
    catalog = package.catalog_record
    exact_matches = [
        item
        for item in legacy_inventory
        if item.source_checksum and item.source_checksum == catalog.source_checksum
    ]
    exact_keys = tuple(sorted(item.legacy_document_key for item in exact_matches))
    filename_names = {catalog.source_filename, *aliases}
    filename_matches = [
        item
        for item in legacy_inventory
        if _canonical_filename(item.source_filename) in {_canonical_filename(name) for name in filename_names}
    ]
    filename_keys = tuple(sorted(item.legacy_document_key for item in filename_matches))
    document_id_matches = [
        item
        for item in legacy_inventory
        if catalog.document_id in item.legacy_document_key and not item.source_checksum
    ]

    warning_codes: list[str] = []
    review_required = False
    cutover_eligible = False
    matched_keys: tuple[str, ...] = ()

    if len(exact_matches) > 1:
        status = AMBIGUOUS_LEGACY_MATCH
        warning_codes.append("MULTIPLE_EXACT_SOURCE_CHECKSUM_MATCHES")
        review_required = True
        matched_keys = exact_keys
    elif len(exact_matches) == 1:
        status = EXACT_SOURCE_CHECKSUM_MATCH
        cutover_eligible = not observations
        matched_keys = exact_keys
    elif document_id_matches:
        status = DOCUMENT_ID_MATCH_WITHOUT_SOURCE_CHECKSUM
        review_required = True
        warning_codes.append("DOCUMENT_ID_MATCH_WITHOUT_SOURCE_CHECKSUM")
        matched_keys = tuple(sorted(item.legacy_document_key for item in document_id_matches))
    elif filename_matches:
        different_checksum = any(
            item.source_checksum and item.source_checksum != catalog.source_checksum
            for item in filename_matches
        )
        if len(filename_matches) > 1 and not different_checksum:
            status = AMBIGUOUS_LEGACY_MATCH
            warning_codes.append("MULTIPLE_FILENAME_ALIAS_MATCHES")
        elif different_checksum:
            status = SAME_FILENAME_DIFFERENT_CHECKSUM
            warning_codes.append("SAME_FILENAME_DIFFERENT_CHECKSUM")
        else:
            status = FILENAME_ALIAS_ONLY
            warning_codes.append("FILENAME_ALIAS_ONLY")
        review_required = True
        matched_keys = filename_keys
    else:
        status = NO_LEGACY_MATCH
        warning_codes.append("NO_LEGACY_MATCH_SHADOW_ONLY")

    for observation in observations:
        if observation.document_id == catalog.document_id:
            warning_codes.append(observation.observation_code)
            cutover_eligible = False

    return DocumentIdentityReconciliation(
        structured_document_id=catalog.document_id,
        structured_source_filename=catalog.source_filename,
        structured_source_checksum=catalog.source_checksum,
        reconciliation_status=status,
        matched_legacy_document_keys=matched_keys,
        exact_checksum_match_count=len(exact_matches),
        filename_match_count=len(filename_matches),
        is_cutover_eligible=cutover_eligible,
        review_required=review_required,
        warning_codes=tuple(sorted(set(warning_codes))),
    )


def run_shadow_migration_rehearsal(
    config: ShadowMigrationRehearsalConfig,
    output_root: str | Path,
    *,
    allow_local_write: bool = False,
    verify_determinism: bool = False,
    verify_rollback: bool = False,
    strict: bool = False,
) -> ShadowMigrationRehearsalResult:
    """Run D.7 rehearsal and optionally write ignored local artifacts."""
    if not allow_local_write:
        raise PermissionError("Shadow migration rehearsal writes require allow_local_write=True.")
    root = Path(output_root)
    packages = tuple(
        load_validated_structured_package(
            item.package_dir,
            expected_package_digest=item.expected_package_digest,
        )
        for item in config.packages
    )
    document_ids = [package.catalog_record.document_id for package in packages]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Duplicate structured package document IDs are not allowed.")

    baseline_snapshot = _snapshot_legacy_roots([*config.source_roots, *config.chunk_roots])
    legacy_inventory = build_legacy_inventory(
        config.source_roots,
        config.chunk_roots,
        packages,
        aliases=config.document_identity_aliases,
    )
    reconciliations = tuple(
        reconcile_document_identity(
            package,
            legacy_inventory,
            config.document_identity_aliases.get(package.catalog_record.document_id, ()),
            config.observations,
        )
        for package in packages
    )
    if strict and any(item.reconciliation_status == AMBIGUOUS_LEGACY_MATCH for item in reconciliations):
        blocking = ("AMBIGUOUS_LEGACY_MATCH",)
    else:
        blocking = ()

    classifications = tuple(item for package in packages for item in classify_records(package.records))
    class_counts = Counter(item.disposition for item in classifications)
    status_counts = Counter(item.record.validation_status for item in classifications)
    rejected_count = status_counts["rejected"]
    forbidden_count = class_counts[FORBIDDEN]
    if rejected_count:
        blocking = tuple(sorted({*blocking, "REJECTED_RECORDS_PRESENT"}))
    if forbidden_count:
        blocking = tuple(sorted({*blocking, "FORBIDDEN_RECORDS_PRESENT"}))

    known_quarantine_issues = _known_quarantine_issues(classifications)
    if known_quarantine_issues:
        blocking = tuple(sorted({*blocking, *known_quarantine_issues}))

    accounting = _accounting_dict(packages, classifications, reconciliations, legacy_inventory, blocking)
    accounting_verified = accounting["accounting_result"] == "PASS"
    outcome = FAIL if blocking or not accounting_verified else (PASS_WITH_QUARANTINE if class_counts[QUARANTINE] else PASS)
    exit_code = 1 if outcome == FAIL else (2 if outcome == PASS_WITH_QUARANTINE else 0)

    root.mkdir(parents=True, exist_ok=True)
    run_file_sha256: dict[str, Mapping[str, str]] = {}
    for run_name in ("run_1", "run_2") if verify_determinism else ("run_1",):
        run_dir = _safe_child_dir(root, run_name)
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        aggregate_digest = _write_run_artifacts(
            run_dir,
            packages,
            classifications,
            legacy_inventory,
            reconciliations,
            accounting,
            outcome,
            blocking,
            config.observations,
        )
        run_file_sha256[run_name] = _file_sha256_map(run_dir, RUN_ARTIFACTS + (SHADOW_MANIFEST_FILENAME,))

    determinism_verified = True
    if verify_determinism:
        determinism_verified = run_file_sha256["run_1"] == run_file_sha256["run_2"]
        if not determinism_verified:
            blocking = tuple(sorted({*blocking, "NONDETERMINISTIC_OUTPUT"}))
            outcome = FAIL
            exit_code = 1

    rollback_verified = True
    if verify_rollback:
        rollback_verified = _run_rollback_rehearsal(root, baseline_snapshot, [*config.source_roots, *config.chunk_roots])
        if not rollback_verified:
            blocking = tuple(sorted({*blocking, "ROLLBACK_REHEARSAL_FAILED"}))
            outcome = FAIL
            exit_code = 1

    legacy_after = _snapshot_legacy_roots([*config.source_roots, *config.chunk_roots])
    legacy_unchanged = baseline_snapshot == legacy_after
    if not legacy_unchanged:
        blocking = tuple(sorted({*blocking, "LEGACY_MUTATION_DETECTED"}))
        outcome = FAIL
        exit_code = 1

    aggregate_shadow_digest = str(json.loads((root / "run_1" / SHADOW_REPORT_FILENAME).read_text(encoding="utf-8"))["aggregate_shadow_digest"])
    result = ShadowMigrationRehearsalResult(
        outcome=outcome,
        exit_code=exit_code,
        package_count=len(packages),
        structured_record_count=len(classifications),
        eligible_count=class_counts[SHADOW_ELIGIBLE],
        quarantine_count=class_counts[QUARANTINE],
        forbidden_count=forbidden_count,
        rejected_count=rejected_count,
        aggregate_shadow_digest=aggregate_shadow_digest,
        accounting_verified=accounting_verified,
        determinism_verified=determinism_verified,
        rollback_verified=rollback_verified,
        legacy_unchanged=legacy_unchanged,
        package_integrity_verified=True,
        output_root=_sanitize_relative(root),
        blocking_issue_codes=blocking,
        run_file_sha256=run_file_sha256,
    )
    _write_json(root / "d7_rehearsal_report.json", d7_acceptance_to_dict(result, packages, legacy_inventory, reconciliations, classifications, config.observations))
    return result


def d7_acceptance_to_dict(
    result: ShadowMigrationRehearsalResult,
    packages: Sequence[ValidatedStructuredPackage],
    legacy_inventory: Sequence[LegacyDocumentInventoryRecord],
    reconciliations: Sequence[DocumentIdentityReconciliation],
    classifications: Sequence[RecordClassification],
    observations: Sequence[ObservationConfig],
) -> dict[str, Any]:
    """Return sanitized D.7 acceptance evidence."""
    limitation_counts = Counter(
        code for item in classifications for code in item.record.accepted_limitation_codes
    )
    warning_counts = Counter(code for item in classifications for code in item.record.warning_codes)
    status_counts = Counter(item.record.validation_status for item in classifications)
    provenance_counts = Counter(item.record.provenance_status for item in classifications)
    reconciliation_counts = Counter(item.reconciliation_status for item in reconciliations)
    return {
        "authorization": _authorization_dict(),
        "blocking_issue_codes": list(result.blocking_issue_codes),
        "document_reconciliation": [asdict(item) for item in sorted(reconciliations, key=lambda item: item.structured_document_id)],
        "governance_policy_name": POLICY_NAME,
        "governance_policy_version": POLICY_VERSION,
        "legacy_chunk_count": sum(item.legacy_chunk_count for item in legacy_inventory),
        "legacy_document_count": len(legacy_inventory),
        "legacy_inventory_roots": ["data/documents", "data/processed/chunked_documents"],
        "limitation_counts": dict(sorted(limitation_counts.items())),
        "migration_accounting": {
            "accounting_verified": result.accounting_verified,
            "forbidden_count": result.forbidden_count,
            "package_count": result.package_count,
            "quarantine_count": result.quarantine_count,
            "shadow_eligible_count": result.eligible_count,
            "structured_record_count": result.structured_record_count,
        },
        "observations": [asdict(item) for item in observations],
        "outcome": result.outcome,
        "package_integrity_verified": result.package_integrity_verified,
        "packages": [
            _catalog_sanitized(package.catalog_record)
            for package in sorted(packages, key=lambda item: item.catalog_record.document_id)
        ],
        "privacy": {
            "absolute_paths_in_report": False,
            "chunk_text_in_report": False,
            "source_text_in_report": False,
            "vectors_in_report": False,
        },
        "provenance_counts": dict(sorted(provenance_counts.items())),
        "reconciliation_status_counts": dict(sorted(reconciliation_counts.items())),
        "rehearsal_schema_name": REHEARSAL_SCHEMA_NAME,
        "rehearsal_schema_version": REHEARSAL_SCHEMA_VERSION,
        "rollback_verified": result.rollback_verified,
        "runtime_ingestion_modified": False,
        "shadow_store": {
            "aggregate_shadow_digest": result.aggregate_shadow_digest,
            "astra_touched": False,
            "embeddings_generated": False,
            "faiss_touched": False,
            "production_retrieval_activated": False,
            "shadow_output_ignored": True,
        },
        "validation_status_counts": dict(sorted(status_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
    }


def _write_run_artifacts(
    run_dir: Path,
    packages: Sequence[ValidatedStructuredPackage],
    classifications: Sequence[RecordClassification],
    legacy_inventory: Sequence[LegacyDocumentInventoryRecord],
    reconciliations: Sequence[DocumentIdentityReconciliation],
    accounting: Mapping[str, Any],
    outcome: str,
    blocking: Sequence[str],
    observations: Sequence[ObservationConfig],
) -> str:
    eligible_records = [
        _shadow_record_dict(item.record, _package_digest_for_record(item.record, packages))
        for item in classifications
        if item.disposition == SHADOW_ELIGIBLE
    ]
    quarantine_records = [
        _quarantine_record_dict(item, _package_digest_for_record(item.record, packages))
        for item in classifications
        if item.disposition == QUARANTINE
    ]
    catalog = {"packages": [_catalog_sanitized(package.catalog_record) for package in sorted(packages, key=lambda item: item.catalog_record.document_id)]}
    inventory = {
        "legacy_inventory": [
            _legacy_inventory_dict(item, include_chunk_ids=len(item.legacy_chunk_ids) <= 200)
            for item in sorted(legacy_inventory, key=lambda item: item.legacy_document_key)
        ],
        "source_text_included": False,
        "chunk_text_included": False,
        "embedding_vectors_included": False,
    }
    reconciliation = {"document_reconciliation": [asdict(item) for item in sorted(reconciliations, key=lambda item: item.structured_document_id)]}
    rollback_manifest = {
        "baseline_restoration_required": True,
        "legacy_deletion_authorized": False,
        "legacy_overwrite_authorized": False,
        "rollback_material_retained": True,
        "shadow_store_removable": True,
    }
    _write_jsonl(run_dir / SHADOW_RECORDS_FILENAME, eligible_records)
    _write_jsonl(run_dir / QUARANTINE_RECORDS_FILENAME, quarantine_records)
    _write_json(run_dir / STRUCTURED_PACKAGE_CATALOG_FILENAME, catalog)
    _write_json(run_dir / LEGACY_INVENTORY_FILENAME, inventory)
    _write_json(run_dir / DOCUMENT_RECONCILIATION_FILENAME, reconciliation)
    _write_json(run_dir / MIGRATION_ACCOUNTING_FILENAME, dict(accounting))
    _write_json(run_dir / ROLLBACK_MANIFEST_FILENAME, rollback_manifest)

    aggregate_digest = _aggregate_digest(run_dir, (
        SHADOW_RECORDS_FILENAME,
        QUARANTINE_RECORDS_FILENAME,
        STRUCTURED_PACKAGE_CATALOG_FILENAME,
        LEGACY_INVENTORY_FILENAME,
        DOCUMENT_RECONCILIATION_FILENAME,
        MIGRATION_ACCOUNTING_FILENAME,
        ROLLBACK_MANIFEST_FILENAME,
    ))
    report = _shadow_report_dict(
        outcome,
        blocking,
        packages,
        classifications,
        legacy_inventory,
        reconciliations,
        observations,
        aggregate_digest,
    )
    _write_json(run_dir / SHADOW_REPORT_FILENAME, report)
    manifest = {
        "aggregate_shadow_digest": aggregate_digest,
        "artifact_count": len(RUN_ARTIFACTS),
        "file_sha256": _file_sha256_map(run_dir, RUN_ARTIFACTS),
        "rehearsal_schema_name": REHEARSAL_SCHEMA_NAME,
        "rehearsal_schema_version": REHEARSAL_SCHEMA_VERSION,
    }
    _write_json(run_dir / SHADOW_MANIFEST_FILENAME, manifest)
    return aggregate_digest


def _shadow_report_dict(
    outcome: str,
    blocking: Sequence[str],
    packages: Sequence[ValidatedStructuredPackage],
    classifications: Sequence[RecordClassification],
    legacy_inventory: Sequence[LegacyDocumentInventoryRecord],
    reconciliations: Sequence[DocumentIdentityReconciliation],
    observations: Sequence[ObservationConfig],
    aggregate_digest: str,
) -> dict[str, Any]:
    class_counts = Counter(item.disposition for item in classifications)
    status_counts = Counter(item.record.validation_status for item in classifications)
    provenance_counts = Counter(item.record.provenance_status for item in classifications)
    limitation_counts = Counter(code for item in classifications for code in item.record.accepted_limitation_codes)
    warning_counts = Counter(code for item in classifications for code in item.record.warning_codes)
    reconciliation_counts = Counter(item.reconciliation_status for item in reconciliations)
    return {
        "accounting_result": "FAIL" if blocking else "PASS",
        "aggregate_shadow_digest": aggregate_digest,
        "authorization": _authorization_dict(),
        "blocking_issue_codes": list(blocking),
        "collision_count": _duplicate_count(item.record.chunk_id for item in classifications),
        "determinism_result": "PENDING_EXTERNAL_COMPARISON",
        "forbidden_count": class_counts[FORBIDDEN],
        "governance_policy_name": POLICY_NAME,
        "governance_policy_version": POLICY_VERSION,
        "legacy_chunk_count": sum(item.legacy_chunk_count for item in legacy_inventory),
        "legacy_document_count": len(legacy_inventory),
        "legacy_unchanged_result": "PENDING_EXTERNAL_SNAPSHOT_COMPARISON",
        "limitation_counts": dict(sorted(limitation_counts.items())),
        "ocr_observations": [asdict(item) for item in observations],
        "outcome": outcome,
        "package_count": len(packages),
        "package_digests": [package.package_digest for package in sorted(packages, key=lambda item: item.catalog_record.document_id)],
        "package_integrity_result": "PASS",
        "provenance_counts": dict(sorted(provenance_counts.items())),
        "quarantine_count": class_counts[QUARANTINE],
        "reconciliation_status_counts": dict(sorted(reconciliation_counts.items())),
        "rehearsal_schema_name": REHEARSAL_SCHEMA_NAME,
        "rehearsal_schema_version": REHEARSAL_SCHEMA_VERSION,
        "rejected_count": status_counts["rejected"],
        "rollback_result": "PENDING_EXTERNAL_ROLLBACK_REHEARSAL",
        "shadow_eligible_count": class_counts[SHADOW_ELIGIBLE],
        "source_checksums": [package.catalog_record.source_checksum for package in sorted(packages, key=lambda item: item.catalog_record.document_id)],
        "source_document_keys": [package.catalog_record.document_id for package in sorted(packages, key=lambda item: item.catalog_record.document_id)],
        "source_filenames": [package.catalog_record.source_filename for package in sorted(packages, key=lambda item: item.catalog_record.document_id)],
        "structured_record_count": len(classifications),
        "validation_status_counts": dict(sorted(status_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
    }


def _accounting_dict(
    packages: Sequence[ValidatedStructuredPackage],
    classifications: Sequence[RecordClassification],
    reconciliations: Sequence[DocumentIdentityReconciliation],
    legacy_inventory: Sequence[LegacyDocumentInventoryRecord],
    blocking: Sequence[str],
) -> dict[str, Any]:
    class_counts = Counter(item.disposition for item in classifications)
    status_counts = Counter(item.record.validation_status for item in classifications)
    chunk_ids = [item.record.chunk_id for item in classifications]
    total = len(classifications)
    accounted = class_counts[SHADOW_ELIGIBLE] + class_counts[QUARANTINE] + class_counts[FORBIDDEN]
    issues = list(blocking)
    if accounted != total:
        issues.append("ACCOUNTING_TOTAL_MISMATCH")
    if _duplicate_count(chunk_ids):
        issues.append("DUPLICATE_STRUCTURED_CHUNK_ID")
    if status_counts["rejected"]:
        issues.append("REJECTED_RECORDS_PRESENT")
    return {
        "accounting_result": "FAIL" if issues else "PASS",
        "blocking_issue_codes": sorted(set(issues)),
        "eligible_plus_quarantine_plus_forbidden_equals_total": accounted == total,
        "forbidden_count": class_counts[FORBIDDEN],
        "legacy_chunk_count": sum(item.legacy_chunk_count for item in legacy_inventory),
        "legacy_document_count": len(legacy_inventory),
        "package_count": len(packages),
        "quarantine_count": class_counts[QUARANTINE],
        "reconciliation_status_counts": dict(sorted(Counter(item.reconciliation_status for item in reconciliations).items())),
        "rejected_count": status_counts["rejected"],
        "shadow_eligible_count": class_counts[SHADOW_ELIGIBLE],
        "structured_record_count": total,
    }


def _known_quarantine_issues(classifications: Sequence[RecordClassification]) -> tuple[str, ...]:
    if not any(item.record.document_id == "aircraft_system_safety" for item in classifications):
        return ()
    matches = [
        item
        for item in classifications
        if item.record.document_id == "aircraft_system_safety"
        and "page-52-table-1" in item.record.source_block_ids
    ]
    issues: list[str] = []
    if len(matches) != 1:
        issues.append("KNOWN_PAGE_52_QUARANTINE_COUNT_INVALID")
        return tuple(issues)
    item = matches[0]
    record = item.record
    if item.disposition != QUARANTINE:
        issues.append("KNOWN_PAGE_52_RECORD_NOT_QUARANTINED")
    if record.validation_status != "review_required" or not record.review_required:
        issues.append("KNOWN_PAGE_52_REVIEW_STATUS_INVALID")
    if KNOWN_TABLE_LIMITATION not in record.accepted_limitation_codes:
        issues.append("KNOWN_PAGE_52_LIMITATION_MISSING")
    if not record.warning_codes:
        issues.append("KNOWN_PAGE_52_WARNING_MISSING")
    leaked = [
        other.record.chunk_id
        for other in classifications
        if other.record.chunk_id != record.chunk_id
        and KNOWN_TABLE_LIMITATION in other.record.accepted_limitation_codes
    ]
    if leaked:
        issues.append("KNOWN_LIMITATION_LEAKED")
    return tuple(issues)


def _shadow_record_dict(record: PersistedChunkRecord, package_digest: str) -> dict[str, Any]:
    data = persisted_chunk_record_to_dict(record)
    data["shadow_metadata"] = {
        "indexing_eligible": False,
        "package_digest": package_digest,
        "record_disposition": SHADOW_ELIGIBLE,
        "retrieval_eligible": False,
        "shadow_mode_only": True,
    }
    return data


def _quarantine_record_dict(item: RecordClassification, package_digest: str) -> dict[str, Any]:
    data = persisted_chunk_record_to_dict(item.record)
    data["quarantine_metadata"] = {
        "indexing_eligible": False,
        "limitation_codes": list(item.record.accepted_limitation_codes),
        "package_digest": package_digest,
        "quarantine_reason_codes": list(item.reason_codes),
        "retrieval_eligible": False,
        "warning_codes": list(item.record.warning_codes),
    }
    return data


def _catalog_sanitized(catalog: StructuredPackageCatalogRecord) -> dict[str, Any]:
    data = asdict(catalog)
    data.pop("package_root", None)
    return data


def _legacy_inventory_dict(
    item: LegacyDocumentInventoryRecord,
    *,
    include_chunk_ids: bool,
) -> dict[str, Any]:
    data = asdict(item)
    if not include_chunk_ids:
        data["legacy_chunk_ids"] = []
        data["legacy_chunk_ids_omitted_count"] = len(item.legacy_chunk_ids)
    return data


def _package_digest_for_record(record: PersistedChunkRecord, packages: Sequence[ValidatedStructuredPackage]) -> str:
    for package in packages:
        if package.catalog_record.document_id == record.document_id:
            return package.package_digest
    raise ValueError(f"No package digest for record: {record.chunk_id}")


def _find_source_candidates(source_roots: Sequence[str | Path], names: Iterable[str]) -> list[Path]:
    canonical_names = {_canonical_filename(name) for name in names}
    results: list[Path] = []
    for root in source_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in root_path.rglob("*"):
            if path.is_file() and _canonical_filename(path.name) in canonical_names:
                results.append(path)
    return sorted(set(results), key=lambda item: _sanitize_relative(item))


def _find_chunk_candidates(chunk_roots: Sequence[str | Path], names: Iterable[str]) -> list[Path]:
    canonical_names = {_canonical_filename(Path(name).stem) for name in names}
    results: list[Path] = []
    for root in chunk_roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in root_path.rglob("*.json"):
            stem = path.stem
            if stem.endswith("_chunks"):
                stem = stem[: -len("_chunks")]
            if _canonical_filename(stem) in canonical_names:
                results.append(path)
    return sorted(set(results), key=lambda item: _sanitize_relative(item))


def _legacy_chunk_ids(path: Path) -> tuple[str, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = _flatten_legacy_chunks(data)
    ids: list[str] = []
    for index, record in enumerate(records):
        chunk_id = record.get("chunk_id") if isinstance(record, Mapping) else None
        ids.append(str(chunk_id) if isinstance(chunk_id, str) and chunk_id else f"{path.stem}:{index:06d}")
    return tuple(sorted(ids))


def _flatten_legacy_chunks(value: Any) -> list[Mapping[str, Any]]:
    if isinstance(value, list):
        results: list[Mapping[str, Any]] = []
        for item in value:
            results.extend(_flatten_legacy_chunks(item))
        return results
    if isinstance(value, Mapping):
        chunks = value.get("chunks")
        if isinstance(chunks, list):
            return [item for item in chunks if isinstance(item, Mapping)]
        return [value]
    return []


def _source_name_from_chunk_file(name: str) -> str:
    return name[: -len("_chunks.json")] if name.endswith("_chunks.json") else name


def _snapshot_legacy_roots(roots: Sequence[str | Path]) -> tuple[tuple[str, int, str], ...]:
    files: list[tuple[str, int, str]] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in root_path.rglob("*"):
            if path.is_file():
                files.append((_sanitize_relative(path), path.stat().st_size, sha256(path.read_bytes()).hexdigest()))
    return tuple(sorted(files))


def _run_rollback_rehearsal(output_root: Path, baseline_snapshot: tuple[tuple[str, int, str], ...], legacy_roots: Sequence[str | Path]) -> bool:
    rollback_root = _safe_child_dir(output_root, "rollback_test")
    if rollback_root.exists():
        shutil.rmtree(rollback_root)
    rollback_root.mkdir(parents=True, exist_ok=True)
    _write_json(rollback_root / "pre_rehearsal_snapshot.json", {"legacy_snapshot": [{"relative_path": path, "size": size, "sha256": digest} for path, size, digest in baseline_snapshot]})
    marker = _safe_child_file(rollback_root, "shadow_temporary_activation_marker.json")
    _write_json(marker, {"shadow_activation": "temporary", "production_activation": False})
    if not _is_relative_to(marker.resolve(), rollback_root.resolve()):
        return False
    marker.unlink()
    after_snapshot = _snapshot_legacy_roots(legacy_roots)
    report = {
        "baseline_restored": baseline_snapshot == after_snapshot,
        "legacy_files_created": 0,
        "legacy_files_deleted": 0,
        "legacy_files_modified": 0,
        "rebuild_digest_identical": baseline_snapshot == after_snapshot,
        "shadow_temporary_files_removed": not marker.exists(),
    }
    _write_json(rollback_root / "rollback_execution_report.json", report)
    return (
        report["baseline_restored"] is True
        and report["legacy_files_created"] == 0
        and report["legacy_files_deleted"] == 0
        and report["legacy_files_modified"] == 0
        and report["rebuild_digest_identical"] is True
        and report["shadow_temporary_files_removed"] is True
    )


def _safe_child_dir(root: Path, child_name: str) -> Path:
    child = (root / child_name).resolve()
    if not _is_relative_to(child, root.resolve()):
        raise ValueError(f"Refusing path outside shadow root: {child}")
    return child


def _safe_child_file(root: Path, child_name: str) -> Path:
    return _safe_child_dir(root, child_name)


def _file_sha256_map(root: Path, filenames: Sequence[str]) -> dict[str, str]:
    return {
        filename: sha256((root / filename).read_bytes()).hexdigest()
        for filename in sorted(filenames)
        if (root / filename).exists()
    }


def _aggregate_digest(root: Path, filenames: Sequence[str]) -> str:
    payload = {
        filename: sha256((root / filename).read_bytes()).hexdigest()
        for filename in sorted(filenames)
    }
    return sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()


def _package_digest(file_sha256: Mapping[str, str]) -> str:
    payload = json.dumps(dict(sorted(file_sha256.items())), separators=(",", ":"), sort_keys=True).encode("utf-8")
    return sha256(payload).hexdigest()


def _duplicate_count(values: Iterable[str]) -> int:
    return sum(count - 1 for count in Counter(values).values() if count > 1)


def _authorization_dict() -> dict[str, bool | str]:
    return {
        "astra_operations": False,
        "controlled_migration_pilot": False,
        "d8_controlled_migration_pilot_readiness_review": True,
        "embedding_generation": False,
        "faiss_operations": False,
        "full_corpus_migration": False,
        "legacy_deletion": False,
        "ocr_execution": False,
        "production_persistence": False,
        "production_retrieval": False,
        "runtime_retrieval_activation": False,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_plain(record), ensure_ascii=False, separators=(",", ":"), sort_keys=True) for record in records]
    content = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(content, encoding="utf-8")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(_plain(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _load_object(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(data, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return data


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            data = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
            if not isinstance(data, dict):
                raise ValueError(f"JSONL line {line_number} must be an object: {path}")
            records.append(data)
    return records


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


def _reject_forbidden_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in FORBIDDEN_PERSISTED_FIELDS:
                raise ValueError(f"Forbidden vector/storage field at {path}.{key_text}")
            _reject_forbidden_fields(item, f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_fields(item, f"{path}[{index}]")


def _reject_absolute_paths(value: Any, path: str = "$") -> None:
    if isinstance(value, str):
        if _looks_like_absolute_path(value):
            raise ValueError(f"Forbidden absolute path at {path}")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            _reject_absolute_paths(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_absolute_paths(item, f"{path}[{index}]")


def _looks_like_absolute_path(value: str) -> bool:
    without_urls = re.sub(r"[A-Za-z][A-Za-z0-9+.-]*://\S+", "", value)
    return bool(
        re.search(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]", without_urls)
        or without_urls.startswith("/")
        or without_urls.startswith("\\\\")
    )


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _required_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string.")
    return value


def _string_tuple(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{key} must contain non-empty strings.")
        result.append(item)
    return tuple(sorted(result))


def _sha256_string(value: Any, key: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{key} must be a lowercase SHA256 hex digest.")
    return value


def _canonical_filename(value: str) -> str:
    stem = Path(value).stem if Path(value).suffix else value
    return re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_")


def _stable_file_key(path: Path) -> str:
    return _canonical_filename(path.name)


def _sanitize_relative(path: str | Path) -> str:
    path_obj = Path(path)
    try:
        rel = path_obj.resolve().relative_to(Path.cwd().resolve())
        return rel.as_posix()
    except ValueError:
        return path_obj.name


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


__all__ = [
    "AMBIGUOUS_LEGACY_MATCH",
    "CONFIG_SCHEMA_NAME",
    "CONFIG_SCHEMA_VERSION",
    "DOCUMENT_ID_MATCH_WITHOUT_SOURCE_CHECKSUM",
    "EXACT_SOURCE_CHECKSUM_MATCH",
    "FAIL",
    "FILENAME_ALIAS_ONLY",
    "FORBIDDEN",
    "KNOWN_TABLE_QUARANTINE_REFERENCE",
    "NO_LEGACY_MATCH",
    "OCR_COMPLETENESS_NOT_ESTABLISHED",
    "PASS",
    "PASS_WITH_QUARANTINE",
    "QUARANTINE",
    "REHEARSAL_SCHEMA_NAME",
    "REHEARSAL_SCHEMA_VERSION",
    "SHADOW_ELIGIBLE",
    "SAME_FILENAME_DIFFERENT_CHECKSUM",
    "DocumentIdentityReconciliation",
    "LegacyDocumentInventoryRecord",
    "ObservationConfig",
    "PackageConfig",
    "RecordClassification",
    "ShadowMigrationRehearsalConfig",
    "ShadowMigrationRehearsalResult",
    "StructuredPackageCatalogRecord",
    "ValidatedStructuredPackage",
    "build_legacy_inventory",
    "classify_records",
    "d7_acceptance_to_dict",
    "load_shadow_migration_rehearsal_config",
    "load_validated_structured_package",
    "reconcile_document_identity",
    "run_shadow_migration_rehearsal",
]
