import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aviationrag.migration.shadow_migration_rehearsal import (  # noqa: E402
    AMBIGUOUS_LEGACY_MATCH,
    DOCUMENT_ID_MATCH_WITHOUT_SOURCE_CHECKSUM,
    EXACT_SOURCE_CHECKSUM_MATCH,
    FAIL,
    FILENAME_ALIAS_ONLY,
    FORBIDDEN,
    NO_LEGACY_MATCH,
    PASS,
    PASS_WITH_QUARANTINE,
    QUARANTINE,
    SAME_FILENAME_DIFFERENT_CHECKSUM,
    SHADOW_ELIGIBLE,
    LegacyDocumentInventoryRecord,
    ObservationConfig,
    PackageConfig,
    ShadowMigrationRehearsalConfig,
    build_legacy_inventory,
    classify_records,
    load_shadow_migration_rehearsal_config,
    load_validated_structured_package,
    reconcile_document_identity,
    run_shadow_migration_rehearsal,
)
from aviationrag.ingestion.persisted_chunk_record import (  # noqa: E402
    PERSISTED_CHUNK_MAPPER_VERSION,
    PERSISTED_CHUNK_SCHEMA_NAME,
    PERSISTED_CHUNK_SCHEMA_VERSION,
    PersistedChunkRecord,
    persisted_chunk_record_to_dict,
)
from aviationrag.ingestion.persisted_chunk_validator import (  # noqa: E402
    PERSISTED_CHUNK_LIMITATION_REGISTRY_VERSION,
)


TOOL_SCRIPT = ROOT / "tools" / "migration" / "run-shadow-migration-rehearsal.py"
FIXTURE_DIR = ROOT / "tests" / "fixtures" / "shadow_migration_rehearsal"


class ShadowMigrationRehearsalTests(unittest.TestCase):
    def test_config_fixture_json_is_valid_and_private(self):
        config = json.loads((FIXTURE_DIR / "config.template.json").read_text(encoding="utf-8"))
        acceptance = json.loads((FIXTURE_DIR / "d7_shadow_migration_acceptance.json").read_text(encoding="utf-8"))
        payload = json.dumps({"config": config, "acceptance": acceptance}, sort_keys=True)

        self.assertEqual(config["config_schema_name"], "aviationrag-shadow-migration-rehearsal-config")
        self.assertEqual(acceptance["expected_outcome"], PASS_WITH_QUARANTINE)
        self.assertNotIn("C:\\", payload)
        self.assertNotIn("Aspire5 15 i7 4G2050", payload)
        self.assertNotIn("chunk text", payload.lower())
        self.assertNotIn("source text", payload.lower())
        self.assertNotIn("embedding", json.dumps(acceptance.get("privacy", {})).lower())

    def test_config_validation_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pkg = root / "pkg"
            _write_package(pkg, [_record("doc")])
            legacy = root / "legacy"
            legacy.mkdir()
            base = _config_data(pkg, legacy)

            cases = [
                ("valid", {}, None),
                ("unknown", {"unexpected": True}, ValueError),
                ("duplicate_package", {"packages": [base["packages"][0], base["packages"][0]]}, ValueError),
                ("duplicate_dir", {"packages": [base["packages"][0], {**base["packages"][0], "document_id": "doc2"}]}, ValueError),
                ("missing_package", {"packages": [{**base["packages"][0], "package_dir": str(root / "missing")}]}, FileNotFoundError),
                ("bad_digest", {"packages": [{**base["packages"][0], "expected_package_digest": "bad"}]}, ValueError),
                ("bad_legacy", {"legacy_inventory": {"source_roots": [str(root / "missing")], "chunk_roots": []}}, FileNotFoundError),
                ("deterministic_observations", {"observations": [
                    {"document_id": "doc", "observation_code": "B", "page": 2, "scope": "page", "summary": "b"},
                    {"document_id": "doc", "observation_code": "A", "page": 1, "scope": "page", "summary": "a"},
                ]}, None),
            ]
            for name, patch, error in cases:
                with self.subTest(name=name):
                    data = {**base, **patch}
                    path = root / f"{name}.json"
                    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                    if error:
                        with self.assertRaises(error):
                            load_shadow_migration_rehearsal_config(path)
                    else:
                        config = load_shadow_migration_rehearsal_config(path)
                        self.assertEqual(config.packages[0].document_id, "doc")
                        if name == "deterministic_observations":
                            self.assertEqual([item.observation_code for item in config.observations], ["A", "B"])

    def test_package_integrity_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid_dir = root / "valid"
            digest = _write_package(valid_dir, [_record("doc"), _record("doc", chunk_index=1, chunk_id_suffix="1" * 24)])
            loaded = load_validated_structured_package(valid_dir, expected_package_digest=digest)
            self.assertEqual(loaded.package_digest, digest)
            self.assertEqual(loaded.catalog_record.record_count, 2)

            cases = [
                ("missing_manifest", lambda path: (path / "persistence_manifest.json").unlink(), FileNotFoundError),
                ("missing_chunks", lambda path: (path / "persisted_chunks.jsonl").unlink(), FileNotFoundError),
                ("file_checksum_mismatch", lambda path: (path / "warnings.json").write_text("{}\n", encoding="utf-8"), ValueError),
                ("package_digest_mismatch", lambda path: _patch_manifest(path, {"package_checksum": "0" * 64}), ValueError),
                ("unsupported_package_schema", lambda path: _patch_manifest(path, {"package_schema_version": "9.9.9"}), ValueError),
                ("unsupported_persisted_schema", lambda path: _patch_manifest(path, {"persisted_schema_version": "9.9.9"}), ValueError),
                ("unsupported_mapper", lambda path: _patch_manifest(path, {"mapper_version": "9.9.9"}), ValueError),
                ("nonzero_rejected", lambda path: _patch_manifest(path, {"rejected_count": 1}), ValueError),
                ("record_validation_failure", lambda path: _rewrite_records(path, [dict(_record_dict(_record("doc")), source_checksum="bad")]), ValueError),
                ("vector_field", lambda path: _rewrite_records(path, [dict(_record_dict(_record("doc")), embedding=[0.1])]), ValueError),
            ]
            for name, mutate, error in cases:
                with self.subTest(name=name):
                    case_dir = root / name
                    _write_package(case_dir, [_record("doc")])
                    mutate(case_dir)
                    with self.assertRaises(error):
                        load_validated_structured_package(case_dir)

    def test_legacy_inventory_is_read_only_deterministic_and_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_root = root / "sources"
            chunk_root = root / "chunks"
            source_root.mkdir()
            chunk_root.mkdir()
            source = source_root / "Doc.pdf"
            source.write_bytes(b"source bytes")
            chunk_file = chunk_root / "Doc_chunks.json"
            chunk_file.write_text(json.dumps({"chunks": [{"chunk_id": "legacy-1", "text": "do not copy"}]}) + "\n", encoding="utf-8")
            package_dir = root / "pkg"
            _write_package(package_dir, [_record("doc", source_filename="Doc.pdf", source_checksum=hashlib.sha256(b"source bytes").hexdigest())])
            package = load_validated_structured_package(package_dir)
            before = source.read_bytes(), chunk_file.read_bytes()

            first = build_legacy_inventory([source_root], [chunk_root], [package])
            second = build_legacy_inventory([source_root], [chunk_root], [package])

            self.assertEqual(first, second)
            self.assertEqual(before, (source.read_bytes(), chunk_file.read_bytes()))
            self.assertEqual({item.record_origin for item in first}, {"legacy_chunked", "legacy_processed"})
            self.assertTrue(any(item.source_checksum for item in first))
            self.assertTrue(any(item.provenance_status == "legacy_filename_only" for item in first))
            self.assertNotIn("do not copy", json.dumps([item.__dict__ for item in first]))
            self.assertNotIn("embedding", json.dumps([item.__dict__ for item in first]).lower())

    def test_identity_reconciliation_statuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "pkg"
            checksum = "a" * 64
            _write_package(package_dir, [_record("doc", source_filename="Doc.pdf", source_checksum=checksum)])
            package = load_validated_structured_package(package_dir)
            exact = LegacyDocumentInventoryRecord("legacy:exact", "Doc.pdf", "Doc.pdf", checksum, 10, 0, (), "legacy_processed", "full_provenance", ())
            doc_only = LegacyDocumentInventoryRecord("doc", "Doc.pdf", "chunks/Doc.json", None, 10, 1, ("1",), "legacy_chunked", "legacy_filename_only", ())
            same_name_diff = replace(exact, legacy_document_key="legacy:diff", source_checksum="b" * 64)
            alias = replace(doc_only, legacy_document_key="legacy:alias", source_filename="Alias.pdf")

            self.assertEqual(reconcile_document_identity(package, [exact]).reconciliation_status, EXACT_SOURCE_CHECKSUM_MATCH)
            self.assertEqual(reconcile_document_identity(package, [doc_only]).reconciliation_status, DOCUMENT_ID_MATCH_WITHOUT_SOURCE_CHECKSUM)
            self.assertEqual(reconcile_document_identity(package, [same_name_diff]).reconciliation_status, SAME_FILENAME_DIFFERENT_CHECKSUM)
            self.assertEqual(reconcile_document_identity(package, [alias], aliases=["Alias.pdf"]).reconciliation_status, FILENAME_ALIAS_ONLY)
            self.assertEqual(reconcile_document_identity(package, []).reconciliation_status, NO_LEGACY_MATCH)
            self.assertEqual(reconcile_document_identity(package, [exact, replace(exact, legacy_document_key="legacy:exact2")]).reconciliation_status, AMBIGUOUS_LEGACY_MATCH)
            observed = reconcile_document_identity(package, [exact], observations=[ObservationConfig("doc", "OCR_COMPLETENESS_NOT_ESTABLISHED", 2, "page", "obs")])
            self.assertFalse(observed.is_cutover_eligible)
            self.assertIn("OCR_COMPLETENESS_NOT_ESTABLISHED", observed.warning_codes)

    def test_title_alone_is_ignored_for_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            package_dir = Path(tmp) / "pkg"
            _write_package(package_dir, [_record("doc", document_title="Same Title")])
            package = load_validated_structured_package(package_dir)
            legacy = LegacyDocumentInventoryRecord("legacy:title", "Different.pdf", "Different.pdf", None, 10, 0, (), "legacy_chunked", "legacy_filename_only", ())

            self.assertEqual(reconcile_document_identity(package, [legacy]).reconciliation_status, NO_LEGACY_MATCH)

    def test_eligibility_and_known_quarantine_rules(self):
        valid = _record("doc")
        warned = replace(valid, validation_status="valid_with_warnings", warning_codes=("WARN",))
        review = replace(
            valid,
            document_id="aircraft_system_safety",
            chunk_id="aircraft_system_safety:chunk:a37f59de1d352535ac45f326",
            source_block_ids=("page-52-table-1",),
            accepted_limitation_codes=("TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE",),
            warning_codes=("TABLE_CLASSIFICATION_REVIEW_REQUIRED",),
            validation_status="review_required",
            review_required=True,
        )
        rejected = replace(valid, validation_status="rejected")
        partial = replace(valid, provenance_status="partial_provenance", accepted_limitation_codes=("CHUNK_SECTION_CROSSING_REVIEW",), validation_status="review_required", review_required=True)
        unknown = replace(valid, provenance_status="unknown_provenance")

        classes = classify_records([valid, warned, review, rejected, partial, unknown])
        self.assertEqual([item.disposition for item in classes], [SHADOW_ELIGIBLE, QUARANTINE, QUARANTINE, FORBIDDEN, FORBIDDEN, FORBIDDEN])
        self.assertEqual(classes[0].record.chunk_id, valid.chunk_id)
        self.assertNotIn(review.chunk_id, [item.record.chunk_id for item in classes if item.disposition == SHADOW_ELIGIBLE])

    def test_end_to_end_shadow_output_determinism_rollback_and_privacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "legacy"
            legacy.mkdir()
            (legacy / "Doc.pdf").write_bytes(b"source bytes")
            checksum = hashlib.sha256(b"source bytes").hexdigest()
            package_dir = root / "pkg"
            quarantine_package_dir = root / "quarantine_pkg"
            _write_package(
                package_dir,
                [
                    _record("doc", source_filename="Doc.pdf", source_checksum=checksum),
                ],
            )
            _write_package(
                quarantine_package_dir,
                [
                    replace(
                        _record("aircraft_system_safety", chunk_id_suffix="a37f59de1d352535ac45f326"),
                        content_type="table",
                        source_block_ids=("page-52-table-1",),
                        table_ids=("aircraft_system_safety:p51:t0022",),
                        accepted_limitation_codes=("TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE",),
                        warning_codes=("TABLE_CLASSIFICATION_REVIEW_REQUIRED",),
                        validation_status="review_required",
                        review_required=True,
                    ),
                ],
            )
            config = ShadowMigrationRehearsalConfig(
                packages=(PackageConfig("doc", str(package_dir)), PackageConfig("aircraft_system_safety", str(quarantine_package_dir))),
                source_roots=(str(legacy),),
                chunk_roots=(str(legacy),),
                observations=(ObservationConfig("doc", "OCR_COMPLETENESS_NOT_ESTABLISHED", 2, "page", "obs"),),
            )
            result = run_shadow_migration_rehearsal(
                config,
                root / "shadow",
                allow_local_write=True,
                verify_determinism=True,
                verify_rollback=True,
                strict=True,
            )

            self.assertEqual(result.outcome, PASS_WITH_QUARANTINE)
            self.assertEqual(result.exit_code, 2)
            self.assertTrue(result.determinism_verified)
            self.assertTrue(result.rollback_verified)
            self.assertTrue(result.legacy_unchanged)
            for filename in [
                "shadow_records.jsonl",
                "quarantine_records.jsonl",
                "structured_package_catalog.json",
                "legacy_inventory.json",
                "document_reconciliation.json",
                "migration_accounting.json",
                "shadow_manifest.json",
                "shadow_report.json",
                "rollback_manifest.json",
            ]:
                self.assertTrue((root / "shadow" / "run_1" / filename).exists(), filename)
                self.assertEqual(
                    (root / "shadow" / "run_1" / filename).read_bytes(),
                    (root / "shadow" / "run_2" / filename).read_bytes(),
                    filename,
                )
            report_payload = (root / "shadow" / "d7_rehearsal_report.json").read_text(encoding="utf-8")
            self.assertNotIn(str(root), report_payload)
            self.assertNotIn("Aspire5 15 i7 4G2050", report_payload)
            self.assertNotIn('"embedding"', report_payload)

    def test_failure_outcomes_and_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "legacy"
            legacy.mkdir()
            cases = [
                ("rejected", [_record("doc", validation_status="rejected")], ValueError),
            ]
            for name, records, expected in cases:
                with self.subTest(name=name):
                    package_dir = root / name
                    _write_package(package_dir, records, validate_like_real=False)
                    config = ShadowMigrationRehearsalConfig(
                        packages=(PackageConfig("doc", str(package_dir)),),
                        source_roots=(str(legacy),),
                        chunk_roots=(str(legacy),),
                    )
                    if expected is ValueError:
                        with self.assertRaises(ValueError):
                            run_shadow_migration_rehearsal(config, root / f"out-{name}", allow_local_write=True)
                    else:
                        result = run_shadow_migration_rehearsal(config, root / f"out-{name}", allow_local_write=True)
                        self.assertEqual(result.outcome, expected)
                        self.assertEqual(result.exit_code, 1)

    def test_cli_exit_codes_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = root / "legacy"
            legacy.mkdir()
            package_dir = root / "pkg"
            digest = _write_package(package_dir, [_record("doc")])
            config_path = root / "config.json"
            config_path.write_text(json.dumps(_config_data(package_dir, legacy, digest=digest), indent=2, sort_keys=True) + "\n", encoding="utf-8")

            denied = _run_cli("--config", str(config_path), "--output-root", str(root / "shadow"))
            self.assertEqual(denied.returncode, 1)
            self.assertIn("writes require --allow-local-write", denied.stdout)

            passed = _run_cli("--config", str(config_path), "--output-root", str(root / "shadow"), "--allow-local-write", "--verify-determinism", "--verify-rollback", "--strict")
            self.assertEqual(passed.returncode, 0, passed.stdout + passed.stderr)
            self.assertIn("Outcome: PASS", passed.stdout)
            self.assertIn("Determinism verification: true", passed.stdout)
            self.assertIn("Rollback result: true", passed.stdout)
            self.assertIn("No embeddings generated.", passed.stdout)
            self.assertIn("No Astra access.", passed.stdout)
            self.assertIn("No FAISS access.", passed.stdout)

    def test_runtime_ingestion_files_remain_unreferenced(self):
        module_text = (SRC_DIR / "aviationrag" / "migration" / "shadow_migration_rehearsal.py").read_text(encoding="utf-8")
        cli_text = TOOL_SCRIPT.read_text(encoding="utf-8")
        combined = module_text + cli_text

        self.assertNotIn("read_documents.py", combined)
        self.assertNotIn("aviation_chunk_saver.py", combined)
        self.assertNotIn("faiss_indexer.py", combined)
        self.assertNotIn("import faiss", combined)
        self.assertNotIn("astradb", combined)


def _record(
    document_id,
    *,
    chunk_index=0,
    chunk_id_suffix="0" * 24,
    source_filename="Doc.pdf",
    source_checksum="a" * 64,
    document_title=None,
    validation_status="valid",
    provenance_status="full_provenance",
):
    return PersistedChunkRecord(
        schema_name=PERSISTED_CHUNK_SCHEMA_NAME,
        schema_version=PERSISTED_CHUNK_SCHEMA_VERSION,
        chunk_id=f"{document_id}:chunk:{chunk_id_suffix}",
        chunk_index=chunk_index,
        document_id=document_id,
        source_filename=source_filename,
        source_checksum=source_checksum,
        document_title=document_title,
        document_number=None,
        document_revision=None,
        document_issue=None,
        effective_date=None,
        text="synthetic chunk text",
        normalized_text="synthetic chunk text",
        content_type="paragraph",
        content_subtype=None,
        language="en",
        page_start=1,
        page_end=1,
        pdf_page_index_start=0,
        pdf_page_index_end=0,
        contributing_page_numbers=(1,),
        contributing_pdf_page_indexes=(0,),
        printed_page_labels=(),
        section_id="s1",
        section_path=("Section",),
        section_number="1",
        section_title="Section",
        clause_identifier=None,
        source_block_ids=("block-1",),
        source_span={"page_start": 1, "page_end": 1, "pdf_page_index_start": 0, "pdf_page_index_end": 0, "source_block_ids": ["block-1"]},
        table_ids=(),
        figure_ids=(),
        equation_ids=(),
        admonition_ids=(),
        cross_reference_ids=(),
        parser_name="techdoc-parser",
        parser_version="0.1.0",
        structured_document_schema_version="0.1.0",
        adapter_version="D.4c",
        persistence_mapper_version=PERSISTED_CHUNK_MAPPER_VERSION,
        extraction_method="synthetic",
        record_origin="new_structured",
        provenance_status=provenance_status,
        accepted_limitation_codes=(),
        validation_status=validation_status,
        warning_codes=(),
        review_required=False,
    )


def _record_dict(record):
    return persisted_chunk_record_to_dict(record)


def _write_package(path, records, *, validate_like_real=True):
    path.mkdir(parents=True, exist_ok=True)
    record_dicts = [_record_dict(record) for record in records]
    rejected = [] if validate_like_real else ([] if not any(record.validation_status == "rejected" for record in records) else [{"candidate_id": "bad"}])
    warnings_payload = {
        "schema_name": "aviationrag-persisted-chunk-warnings",
        "schema_version": "0.1.0",
        "warning_count": sum(len(record.warning_codes) for record in records),
        "warnings": [],
    }
    report = {
        "content_type_counts": dict(sorted({record.content_type: sum(1 for item in records if item.content_type == record.content_type) for record in records}.items())),
        "outcome": "REVIEW" if any(record.review_required for record in records) else "PASS",
        "provenance_counts": dict(sorted(_counts(record.provenance_status for record in records).items())),
        "record_count": len(records),
        "rejected_count": len(rejected),
        "review_required_count": sum(1 for record in records if record.review_required),
        "validation_status_counts": dict(sorted(_counts(record.validation_status for record in records).items())),
        "warning_count": sum(len(record.warning_codes) for record in records),
    }
    files_without_manifest = {
        "persisted_chunks.jsonl": _jsonl_bytes(record_dicts),
        "persistence_report.json": _json_bytes(report),
        "rejected_candidates.jsonl": _jsonl_bytes(rejected),
        "warnings.json": _json_bytes(warnings_payload),
    }
    file_sha = {name: hashlib.sha256(content).hexdigest() for name, content in files_without_manifest.items()}
    package_digest = hashlib.sha256(json.dumps(dict(sorted(file_sha.items())), separators=(",", ":"), sort_keys=True).encode("utf-8")).hexdigest()
    manifest = {
        "accepted_count": len(records),
        "file_sha256": file_sha,
        "issue_count": 0,
        "limitation_registry_version": PERSISTED_CHUNK_LIMITATION_REGISTRY_VERSION,
        "mapper_version": PERSISTED_CHUNK_MAPPER_VERSION,
        "outcome": report["outcome"],
        "package_checksum": package_digest,
        "package_schema_name": "aviationrag-persisted-chunk-package",
        "package_schema_version": "0.1.0",
        "persisted_schema_name": PERSISTED_CHUNK_SCHEMA_NAME,
        "persisted_schema_version": PERSISTED_CHUNK_SCHEMA_VERSION,
        "record_count": len(records),
        "rejected_count": len(rejected),
        "source_manifest_checksum": "b" * 64,
        "source_structured_document_checksum": "c" * 64,
        "specification_name": "aviationrag-persisted-chunk-mapping",
        "specification_version": "0.1.0",
        "warning_count": report["warning_count"],
    }
    all_files = {**files_without_manifest, "persistence_manifest.json": _json_bytes(manifest)}
    for name, content in all_files.items():
        (path / name).write_bytes(content)
    return package_digest


def _patch_manifest(path, patch):
    manifest_path = path / "persistence_manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data.update(patch)
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_records(path, records):
    (path / "persisted_chunks.jsonl").write_bytes(_jsonl_bytes(records))


def _config_data(package_dir, legacy_root, *, digest=None):
    data = {
        "config_schema_name": "aviationrag-shadow-migration-rehearsal-config",
        "config_schema_version": "0.1.0",
        "document_identity_aliases": {},
        "legacy_inventory": {"source_roots": [str(legacy_root)], "chunk_roots": [str(legacy_root)]},
        "observations": [],
        "packages": [{"document_id": "doc", "package_dir": str(package_dir)}],
    }
    if digest:
        data["packages"][0]["expected_package_digest"] = digest
    return data


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _jsonl_bytes(records):
    return ("\n".join(json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) for record in records) + ("\n" if records else "")).encode("utf-8")


def _counts(values):
    result = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _run_cli(*args):
    return subprocess.run([sys.executable, str(TOOL_SCRIPT), *args], cwd=ROOT, check=False, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
