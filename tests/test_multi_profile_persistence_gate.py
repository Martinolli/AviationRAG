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

from aviationrag.ingestion.multi_profile_persistence_gate import (  # noqa: E402
    ACCEPTED_WITH_LIMITATIONS,
    MULTI_PROFILE_GATE_SCHEMA_NAME,
    MULTI_PROFILE_GATE_SCHEMA_VERSION,
    MultiProfileDefinition,
    aggregate_evaluated_profile_packages,
    evaluate_profile_packages,
    load_multi_profile_config,
    multi_profile_persistence_result_to_dict,
    run_multi_profile_persistence_evaluation,
    sanitized_multi_profile_report_bytes,
)
from aviationrag.ingestion.persisted_chunk_mapper import PersistedChunkCandidateContext  # noqa: E402
from aviationrag.ingestion.structured_document_adapter import FAIL, PASS, REVIEW  # noqa: E402


FIXTURE_DIR = ROOT / "tests" / "fixtures" / "structured_document_adapter"
ARTIFACT = FIXTURE_DIR / "structured_document.json"
MANIFEST = FIXTURE_DIR / "manifest.json"
SOURCE = FIXTURE_DIR / "source.txt"
TOOL_SCRIPT = ROOT / "tools" / "chunking" / "run-multi-profile-persistence-evaluation.py"
TEMPLATE_FIXTURE = ROOT / "tests" / "fixtures" / "multi_profile_persistence" / "profile_config.template.json"
ACCEPTANCE_FIXTURE = (
    ROOT / "tests" / "fixtures" / "multi_profile_persistence" / "multi_profile_gate_acceptance.json"
)


class MultiProfilePersistenceGateTests(unittest.TestCase):
    def test_three_valid_profiles_load_and_contexts_are_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiles = _write_profiles(tmp)
            config = _write_config(tmp, profiles, review_profile=True)

            loaded = load_multi_profile_config(config)

        self.assertEqual(len(loaded), 3)
        self.assertEqual([profile.profile_key for profile in loaded], ["profile_a", "profile_b", "profile_c"])
        context = loaded[2].candidate_contexts["profile_c:chunk:table-block-1"]
        self.assertEqual(context.accepted_limitation_codes, ("TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE",))
        self.assertEqual(context.warning_codes, ("TABLE_CLASSIFICATION_REVIEW_REQUIRED",))
        self.assertTrue(context.review_required)

    def test_config_rejects_duplicates_unknown_fields_missing_paths_and_invalid_role(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiles = _write_profiles(tmp)
            duplicate = [profiles[0], profiles[0], profiles[2]]
            with self.assertRaises(ValueError):
                load_multi_profile_config(_write_config(tmp, duplicate, name="duplicate.json"))

            unknown = _config_payload(profiles)
            unknown["unexpected"] = True
            with self.assertRaises(ValueError):
                load_multi_profile_config(_write_payload(tmp, "unknown.json", unknown))

            missing_source = _config_payload(profiles)
            missing_source["profiles"][0]["source_path"] = str(Path(tmp) / "missing-source.txt")
            with self.assertRaises(FileNotFoundError):
                load_multi_profile_config(_write_payload(tmp, "missing-source.json", missing_source))

            missing_artifact = _config_payload(profiles)
            missing_artifact["profiles"][0]["artifact_path"] = str(Path(tmp) / "missing-artifact-input.json")
            with self.assertRaises(FileNotFoundError):
                load_multi_profile_config(_write_payload(tmp, "missing-artifact.json", missing_artifact))

            missing_manifest = _config_payload(profiles)
            missing_manifest["profiles"][0]["manifest_path"] = str(Path(tmp) / "missing-manifest-input.json")
            with self.assertRaises(FileNotFoundError):
                load_multi_profile_config(_write_payload(tmp, "missing-manifest.json", missing_manifest))

            invalid_role = _config_payload(profiles)
            invalid_role["profiles"][0]["profile_role"] = "uncontrolled corpus"
            with self.assertRaises(ValueError):
                load_multi_profile_config(_write_payload(tmp, "invalid-role.json", invalid_role))

    def test_duplicate_candidate_context_key_rejects(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiles = _write_profiles(tmp)
            payload = json.dumps(_config_payload(profiles), indent=2)
            payload = payload.replace(
                '"candidate_contexts": {}',
                (
                    '"candidate_contexts": {'
                    '"same": {"accepted_limitation_codes": [], "warning_codes": [], "review_required": false}, '
                    '"same": {"accepted_limitation_codes": [], "warning_codes": [], "review_required": false}'
                    "}"
                ),
                1,
            )
            path = Path(tmp) / "duplicate-context.json"
            path.write_text(payload + "\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_multi_profile_config(path)

    def test_three_pass_profiles_produce_pass_and_aggregate_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiles = _write_profiles(tmp)

            result = run_multi_profile_persistence_evaluation(profiles)

        self.assertEqual(result.outcome, PASS)
        self.assertEqual(result.profile_count, 3)
        self.assertEqual(result.profile_outcomes, {"profile_a": PASS, "profile_b": PASS, "profile_c": PASS})
        self.assertEqual(result.total_candidate_count, 18)
        self.assertEqual(result.total_accepted_record_count, 18)
        self.assertEqual(result.total_rejected_candidate_count, 0)
        self.assertEqual(result.total_warning_count, 0)
        self.assertEqual(result.total_review_required_count, 0)
        self.assertEqual(result.aggregate_provenance_counts, {"full_provenance": 18})
        self.assertEqual(result.cross_document_chunk_id_collision_count, 0)
        self.assertTrue(result.schema_consistency_verified)
        self.assertTrue(result.determinism_verified)

    def test_two_pass_plus_approved_review_produces_accepted_with_limitations(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiles = _write_profiles(tmp, review_profile=True)

            result = run_multi_profile_persistence_evaluation(profiles, allow_reviewed_profiles=True)

        self.assertEqual(result.outcome, ACCEPTED_WITH_LIMITATIONS)
        self.assertEqual(result.profile_outcomes["profile_c"], REVIEW)
        self.assertEqual(result.total_warning_count, 2)
        self.assertEqual(result.total_review_required_count, 1)
        self.assertEqual(result.aggregate_limitation_counts, {"TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE": 1})

    def test_unapproved_review_any_fail_missing_profile_zero_records_and_rejections_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            review_profiles = _write_profiles(tmp, review_profile=True)
            unapproved = run_multi_profile_persistence_evaluation(review_profiles)
            self.assertEqual(unapproved.outcome, FAIL)

        with tempfile.TemporaryDirectory() as tmp:
            fail_profiles = _write_profiles(tmp, source_mismatch="profile_b")
            self.assertEqual(run_multi_profile_persistence_evaluation(fail_profiles).outcome, FAIL)

        with tempfile.TemporaryDirectory() as tmp:
            two_profiles = _write_profiles(tmp)[:2]
            self.assertEqual(run_multi_profile_persistence_evaluation(two_profiles).outcome, FAIL)

        with tempfile.TemporaryDirectory() as tmp:
            zero_profiles = _write_profiles(tmp, zero_candidates="profile_b")
            self.assertEqual(run_multi_profile_persistence_evaluation(zero_profiles).outcome, FAIL)

        with tempfile.TemporaryDirectory() as tmp:
            rejected_profiles = _write_profiles(tmp, unknown_context="profile_c")
            self.assertEqual(run_multi_profile_persistence_evaluation(rejected_profiles, allow_reviewed_profiles=True).outcome, FAIL)

    def test_unknown_warning_and_unknown_limitation_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            warning_profiles = _write_profiles(tmp, context_warning="UNKNOWN_WARNING")
            result = run_multi_profile_persistence_evaluation(warning_profiles, allow_reviewed_profiles=True)
            self.assertEqual(result.outcome, FAIL)
            self.assertIn("PROFILE_UNKNOWN_WARNING:profile_c", result.blocking_issue_codes)

        with tempfile.TemporaryDirectory() as tmp:
            limitation_profiles = _write_profiles(tmp, limitation_code="UNKNOWN_LIMITATION")
            result = run_multi_profile_persistence_evaluation(limitation_profiles, allow_reviewed_profiles=True)
            self.assertEqual(result.outcome, FAIL)
            self.assertIn("PROFILE_UNKNOWN_LIMITATION:profile_c", result.blocking_issue_codes)

    def test_limitation_is_profile_specific_candidate_specific_and_does_not_fabricate(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiles = _write_profiles(tmp, review_profile=True)
            result = run_multi_profile_persistence_evaluation(profiles, allow_reviewed_profiles=True)
            profile_summaries = {item["profile_key"]: item for item in result.summary["profiles"]}

        self.assertEqual(profile_summaries["profile_a"]["accepted_limitation_counts"], {})
        self.assertEqual(profile_summaries["profile_b"]["accepted_limitation_counts"], {})
        self.assertEqual(profile_summaries["profile_c"]["accepted_limitation_counts"], {"TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE": 1})
        self.assertEqual(profile_summaries["profile_c"]["candidate_context_ids"], ("profile_c:chunk:table-block-1",))

        with tempfile.TemporaryDirectory() as tmp:
            missing = _write_profiles(tmp, unknown_context="profile_c")
            result = run_multi_profile_persistence_evaluation(missing, allow_reviewed_profiles=True)
            self.assertEqual(result.outcome, FAIL)
            self.assertIn("PROFILE_REJECTED_CANDIDATES:profile_c", result.blocking_issue_codes)

    def test_cross_document_identity_and_schema_consistency_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiles = _write_profiles(tmp)
            evaluated = list(evaluate_profile_packages(profiles))
            same_chunk_id = evaluated[0].package.records[0].chunk_id
            changed_record = replace(evaluated[1].package.records[0], chunk_id=same_chunk_id)
            changed_package = replace(
                evaluated[1].package,
                records=(changed_record, *evaluated[1].package.records[1:]),
            )
            evaluated[1] = replace(evaluated[1], package=changed_package)
            result = aggregate_evaluated_profile_packages(evaluated)
            self.assertEqual(result.outcome, FAIL)
            self.assertEqual(result.cross_document_chunk_id_collision_count, 1)

        with tempfile.TemporaryDirectory() as tmp:
            profiles = _write_profiles(tmp)
            evaluated = list(evaluate_profile_packages(profiles))
            bad_manifest = dict(evaluated[0].package.manifest)
            bad_manifest["persisted_schema_version"] = "9.9.9"
            evaluated[0] = replace(evaluated[0], package=replace(evaluated[0].package, manifest=bad_manifest))
            self.assertFalse(aggregate_evaluated_profile_packages(evaluated).schema_consistency_verified)

        with tempfile.TemporaryDirectory() as tmp:
            profiles = _write_profiles(tmp)
            evaluated = list(evaluate_profile_packages(profiles))
            bad_manifest = dict(evaluated[0].package.manifest)
            bad_manifest["package_schema_version"] = "9.9.9"
            evaluated[0] = replace(evaluated[0], package=replace(evaluated[0].package, manifest=bad_manifest))
            self.assertFalse(aggregate_evaluated_profile_packages(evaluated).schema_consistency_verified)

        with tempfile.TemporaryDirectory() as tmp:
            profiles = _write_profiles(tmp)
            evaluated = list(evaluate_profile_packages(profiles))
            bad_manifest = dict(evaluated[0].package.manifest)
            bad_manifest["mapper_version"] = "9.9.9"
            evaluated[0] = replace(evaluated[0], package=replace(evaluated[0].package, manifest=bad_manifest))
            self.assertFalse(aggregate_evaluated_profile_packages(evaluated).schema_consistency_verified)

    def test_same_text_with_different_document_ids_is_allowed_and_namespaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiles = _write_profiles(tmp)
            evaluated = evaluate_profile_packages(profiles)
            result = aggregate_evaluated_profile_packages(evaluated)

        self.assertEqual(result.outcome, PASS)
        for item in evaluated:
            for record in item.package.records:
                self.assertTrue(record.chunk_id.startswith(f"{record.document_id}:chunk:"))

    def test_determinism_profile_ordering_and_report_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiles = list(reversed(_write_profiles(tmp)))
            evaluated = list(evaluate_profile_packages(profiles))
            result = aggregate_evaluated_profile_packages(evaluated)
            first = sanitized_multi_profile_report_bytes(result)
            second = sanitized_multi_profile_report_bytes(result)

            bad_repeat = replace(evaluated[0].repeated_package, package_digest="0" * 64)
            evaluated[0] = replace(evaluated[0], repeated_package=bad_repeat)
            nondeterministic = aggregate_evaluated_profile_packages(evaluated)

        self.assertEqual(list(result.profile_outcomes), ["profile_a", "profile_b", "profile_c"])
        self.assertEqual(first, second)
        self.assertTrue(first.decode("utf-8").endswith("\n"))
        self.assertEqual(nondeterministic.outcome, FAIL)
        self.assertIn("AGGREGATE_NONDETERMINISTIC", nondeterministic.blocking_issue_codes)

    def test_authorization_and_privacy_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            profiles = _write_profiles(tmp, review_profile=True)
            result = run_multi_profile_persistence_evaluation(profiles, allow_reviewed_profiles=True)
            payload = sanitized_multi_profile_report_bytes(result).decode("utf-8")

        self.assertTrue(result.authorization["d6_persistence_governance_review"])
        self.assertFalse(result.authorization["full_corpus_ingestion"])
        self.assertFalse(result.authorization["embedding_generation"])
        self.assertFalse(result.authorization["astra_rebuild"])
        self.assertFalse(result.authorization["faiss_rebuild"])
        self.assertFalse(result.authorization["production_retrieval_integration"])
        self.assertNotIn(SOURCE.read_text(encoding="utf-8").strip(), payload)
        self.assertNotIn(str(ROOT), payload)
        self.assertNotIn("Aspire5 15 i7 4G2050", payload)
        self.assertNotIn("migration_dry_run", payload)
        json.dumps(multi_profile_persistence_result_to_dict(result), sort_keys=True)

    def test_cli_pass_accepted_with_limitations_fail_and_permission_behavior(self):
        self.assertTrue(TOOL_SCRIPT.exists())
        with tempfile.TemporaryDirectory() as tmp:
            profiles = _write_profiles(tmp)
            config = _write_config(tmp, profiles)
            completed = _run_cli(["--config", str(config)])
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertIn("Aggregate outcome: PASS", completed.stdout)
            self.assertIn("Controlled multi-profile evaluation only.", completed.stdout)
            self.assertIn("No production persistence.", completed.stdout)
            self.assertIn("No embeddings.", completed.stdout)
            self.assertIn("No Astra.", completed.stdout)
            self.assertIn("No FAISS.", completed.stdout)
            self.assertIn("No full-corpus migration authorization.", completed.stdout)
            self.assertFalse((Path(tmp) / "out" / "multi_profile_evaluation_report.json").exists())

        with tempfile.TemporaryDirectory() as tmp:
            profiles = _write_profiles(tmp, review_profile=True)
            config = _write_config(tmp, profiles, review_profile=True)
            denied = _run_cli(["--config", str(config)])
            self.assertEqual(denied.returncode, 1)

            output_root = Path(tmp) / "out"
            accepted = _run_cli(
                [
                    "--config",
                    str(config),
                    "--output-root",
                    str(output_root),
                    "--allow-local-write",
                    "--allow-reviewed-profiles",
                    "--strict",
                ]
            )
            self.assertEqual(accepted.returncode, 2, accepted.stdout + accepted.stderr)
            self.assertIn("Aggregate outcome: ACCEPTED_WITH_LIMITATIONS", accepted.stdout)
            self.assertIn("Byte/hash determinism verified: true", accepted.stdout)
            self.assertTrue((output_root / "multi_profile_evaluation_report.json").exists())

        with tempfile.TemporaryDirectory() as tmp:
            profiles = _write_profiles(tmp, source_mismatch="profile_a")
            config = _write_config(tmp, profiles)
            failed = _run_cli(["--config", str(config)])
            self.assertEqual(failed.returncode, 1)
            self.assertIn("Aggregate outcome: FAIL", failed.stdout)

    def test_committed_template_and_acceptance_fixture_are_sanitized(self):
        self.assertTrue(TEMPLATE_FIXTURE.exists())
        template = json.loads(TEMPLATE_FIXTURE.read_text(encoding="utf-8"))
        self.assertEqual(template["schema_name"], MULTI_PROFILE_GATE_SCHEMA_NAME)
        self.assertEqual(template["schema_version"], MULTI_PROFILE_GATE_SCHEMA_VERSION)
        template_text = TEMPLATE_FIXTURE.read_text(encoding="utf-8")
        self.assertNotIn("C:\\", template_text)
        self.assertNotIn("Aspire5 15 i7 4G2050", template_text)
        self.assertTrue(template_text.endswith("\n"))

        if ACCEPTANCE_FIXTURE.exists():
            acceptance_text = ACCEPTANCE_FIXTURE.read_text(encoding="utf-8")
            acceptance = json.loads(acceptance_text)
            self.assertEqual(acceptance["evaluation_schema_name"], MULTI_PROFILE_GATE_SCHEMA_NAME)
            self.assertNotIn("C:\\", acceptance_text)
            self.assertNotIn("Aspire5 15 i7 4G2050", acceptance_text)
            self.assertNotIn("migration_dry_run", acceptance_text)
            self.assertTrue(acceptance_text.endswith("\n"))

    def test_runtime_ingestion_files_remain_unreferenced(self):
        module_text = (SRC_DIR / "aviationrag" / "ingestion" / "multi_profile_persistence_gate.py").read_text(encoding="utf-8")
        cli_text = TOOL_SCRIPT.read_text(encoding="utf-8")
        combined = module_text + cli_text

        self.assertNotIn("read_documents.py", combined)
        self.assertNotIn("aviation_chunk_saver.py", combined)
        self.assertNotIn("faiss_indexer.py", combined)


def _write_profiles(
    tmp: str,
    *,
    review_profile: bool = False,
    source_mismatch: str | None = None,
    zero_candidates: str | None = None,
    unknown_context: str | None = None,
    context_warning: str = "TABLE_CLASSIFICATION_REVIEW_REQUIRED",
    limitation_code: str = "TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE",
) -> tuple[MultiProfileDefinition, ...]:
    root = Path(tmp)
    specs = (
        ("profile_a", "flight-test publication"),
        ("profile_b", "formal safety standard"),
        ("profile_c", "accepted-limitation profile"),
    )
    profiles = []
    for profile_key, role in specs:
        source, artifact, manifest = _write_profile_fixture(root / profile_key, profile_key)
        if source_mismatch == profile_key:
            source.write_text("changed", encoding="utf-8")
        if zero_candidates == profile_key:
            data = json.loads(artifact.read_text(encoding="utf-8"))
            for block in data["blocks"]:
                block["block_type"] = "metadata"
            data["tables"] = []
            data["figures"] = []
            data["equations"] = []
            data["admonitions"] = []
            data["cross_references"] = []
            _write_artifact_and_manifest(artifact, manifest, source, data)
        contexts = {}
        allow_review = False
        if profile_key == "profile_c" and (
            review_profile
            or unknown_context == profile_key
            or context_warning != "TABLE_CLASSIFICATION_REVIEW_REQUIRED"
            or limitation_code != "TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE"
        ):
            context_id = "missing-candidate" if unknown_context == profile_key else f"{profile_key}:chunk:table-block-1"
            contexts = {
                context_id: PersistedChunkCandidateContext(
                    accepted_limitation_codes=(limitation_code,) if limitation_code else (),
                    warning_codes=(context_warning,) if context_warning else (),
                    review_required=True,
                )
            }
            allow_review = True
        profiles.append(
            MultiProfileDefinition(
                profile_key=profile_key,
                profile_role=role,
                source_path=source,
                artifact_path=artifact,
                manifest_path=manifest,
                expected_document_id=profile_key,
                expected_source_filename=f"{profile_key}.txt",
                expected_page_count=2,
                candidate_contexts=contexts,
                allow_review_outcome=allow_review,
            )
        )
    return tuple(profiles)


def _write_profile_fixture(profile_dir: Path, document_id: str) -> tuple[Path, Path, Path]:
    profile_dir.mkdir(parents=True, exist_ok=True)
    source = profile_dir / f"{document_id}.txt"
    artifact = profile_dir / "structured_document.json"
    manifest = profile_dir / "manifest.json"
    source.write_text(SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    data = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    data["document"]["document_id"] = document_id
    data["document"]["source_filename"] = source.name
    for item in data["tables"]:
        item["table_id"] = item["table_id"].replace("table-1", f"{document_id}:table-1")
    for item in data["figures"]:
        item["figure_id"] = item["figure_id"].replace("figure-1", f"{document_id}:figure-1")
    for item in data["equations"]:
        item["equation_id"] = item["equation_id"].replace("equation-1", f"{document_id}:equation-1")
    for item in data["admonitions"]:
        item["admonition_id"] = item["admonition_id"].replace("warning-1", f"{document_id}:warning-1")
    for item in data["cross_references"]:
        item["reference_id"] = item["reference_id"].replace("xref-", f"{document_id}:xref-")
    _write_artifact_and_manifest(artifact, manifest, source, data)
    return source, artifact, manifest


def _write_artifact_and_manifest(artifact_path: Path, manifest_path: Path, source_path: Path, artifact: dict) -> None:
    artifact["document"]["source_hash"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["source_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    manifest["artifacts"][0]["artifact_sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    manifest["artifacts"][0]["schema_name"] = artifact.get("schema_name")
    manifest["artifacts"][0]["schema_version"] = artifact.get("schema_version")
    manifest["artifacts"][0]["document_id"] = artifact["document"]["document_id"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_config(
    tmp: str,
    profiles: tuple[MultiProfileDefinition, ...],
    *,
    review_profile: bool = False,
    name: str = "profile_config.json",
) -> Path:
    payload = _config_payload(profiles)
    if review_profile:
        profile = payload["profiles"][2]
        profile["candidate_contexts"] = {
            "profile_c:chunk:table-block-1": {
                "accepted_limitation_codes": ["TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE"],
                "warning_codes": ["TABLE_CLASSIFICATION_REVIEW_REQUIRED"],
                "review_required": True,
            }
        }
        profile["allow_review_outcome"] = True
    return _write_payload(tmp, name, payload)


def _config_payload(profiles: tuple[MultiProfileDefinition, ...]) -> dict:
    return {
        "schema_name": MULTI_PROFILE_GATE_SCHEMA_NAME,
        "schema_version": MULTI_PROFILE_GATE_SCHEMA_VERSION,
        "profiles": [
            {
                "profile_key": profile.profile_key,
                "profile_role": profile.profile_role,
                "source_path": str(profile.source_path),
                "artifact_path": str(profile.artifact_path),
                "manifest_path": str(profile.manifest_path),
                "expected_document_id": profile.expected_document_id,
                "expected_source_filename": profile.expected_source_filename,
                "expected_page_count": profile.expected_page_count,
                "approved_adapter_warning_codes": list(profile.approved_adapter_warning_codes),
                "candidate_contexts": {
                    key: {
                        "accepted_limitation_codes": list(value.accepted_limitation_codes),
                        "warning_codes": list(value.warning_codes),
                        "review_required": value.review_required,
                    }
                    for key, value in profile.candidate_contexts.items()
                },
                "allow_review_outcome": profile.allow_review_outcome,
            }
            for profile in profiles
        ],
    }


def _write_payload(tmp: str, name: str, payload: dict) -> Path:
    path = Path(tmp) / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL_SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
    )


if __name__ == "__main__":
    unittest.main()
