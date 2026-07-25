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

from aviationrag.governance.persistence_migration_readiness import (  # noqa: E402
    CONDITIONALLY_READY_FOR_CONTROLLED_MIGRATION_REHEARSAL,
    NOT_READY,
    READY_FOR_CONTROLLED_MIGRATION_REHEARSAL,
    PersistenceMigrationReadinessEvidence,
    decision_report_json_bytes,
    decision_report_markdown,
    decision_to_dict,
    default_persistence_governance_policy,
    default_persistence_governance_policy_dict,
    evaluate_persistence_migration_readiness,
    load_persistence_governance_policy,
    load_persistence_migration_readiness_evidence,
    policy_from_dict,
)


FIXTURE_DIR = ROOT / "tests" / "fixtures" / "persistence_governance"
EVIDENCE_FIXTURE = FIXTURE_DIR / "d6_readiness_evidence.json"
DECISION_FIXTURE = FIXTURE_DIR / "d6_migration_readiness_decision.json"
POLICY_PATH = ROOT / "docs" / "persistence_governance_policy.json"
TOOL_SCRIPT = ROOT / "tools" / "governance" / "run-persistence-migration-readiness-review.py"


class PersistenceMigrationReadinessTests(unittest.TestCase):
    def test_valid_evidence_and_policy_load(self):
        evidence = load_persistence_migration_readiness_evidence(EVIDENCE_FIXTURE)
        policy = load_persistence_governance_policy(POLICY_PATH)

        self.assertEqual(evidence.d5c_outcome, "PASS")
        self.assertEqual(evidence.d5d_outcome, "ACCEPTED_WITH_LIMITATIONS")
        self.assertEqual(policy.policy_name, "aviationrag-persistence-governance")

    def test_evidence_validation_rejects_unknown_bad_counts_and_private_fields(self):
        cases = [
            ("unknown.json", {"unexpected": True}, ValueError),
            ("negative.json", {"total_candidate_count": -1}, ValueError),
            ("accepted_gt_candidate.json", {"total_accepted_record_count": 20000}, ValueError),
            ("det_gt_expected.json", {"deterministic_profile_count": 4}, ValueError),
            ("bad_outcome.json", {"d5d_outcome": "PRODUCTION_READY"}, ValueError),
            ("bad_limitation.json", {"accepted_limitation_codes": ["UNKNOWN_LIMITATION"]}, ValueError),
            ("source_text.json", {"source_text": "forbidden"}, ValueError),
            ("absolute_path.json", {"blocking_issue_codes": ["C:\\\\tmp\\\\x"]}, ValueError),
        ]
        for name, patch, error in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                data = _fixture_data()
                data.update(patch)
                path = Path(tmp) / name
                path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                with self.assertRaises(error):
                    load_persistence_migration_readiness_evidence(path)

    def test_minimum_technical_gates_fail_closed(self):
        cases = [
            ("d5c", {"d5c_outcome": "FAIL"}, "D5C_NOT_PASS"),
            ("d5d", {"d5d_outcome": "FAIL"}, "D5D_OUTCOME_BLOCKING"),
            ("profiles", {"evaluated_profile_count": 2}, "PROFILE_COUNT_TOO_LOW"),
            ("accepted", {"total_accepted_record_count": 0}, "NO_ACCEPTED_RECORDS"),
            ("rejected", {"total_rejected_candidate_count": 1}, "REJECTED_RECORDS_PRESENT"),
            ("unknown_prov", {"unknown_provenance_count": 1}, "UNKNOWN_PROVENANCE_PRESENT"),
            ("collision", {"chunk_id_collision_count": 1}, "CHUNK_ID_COLLISION_PRESENT"),
            ("nondeterminism", {"deterministic_profile_count": 2}, "PROFILE_NONDETERMINISM"),
            ("schema", {"schema_consistency_verified": False}, "SCHEMA_CONSISTENCY_FAILED"),
            ("blocking", {"blocking_issue_codes": ["X"]}, "BLOCKING_ISSUES_PRESENT"),
            ("runtime", {"runtime_ingestion_unchanged": False}, "RUNTIME_INGESTION_CHANGED"),
            ("embedding", {"embeddings_untouched": False}, "EMBEDDINGS_TOUCHED"),
            ("astra", {"astra_untouched": False}, "ASTRA_TOUCHED"),
            ("faiss", {"faiss_untouched": False}, "FAISS_TOUCHED"),
        ]
        for name, patch, code in cases:
            with self.subTest(name=name):
                decision = evaluate_persistence_migration_readiness(_evidence(**patch))
                self.assertEqual(decision.decision, NOT_READY)
                self.assertIn(code, decision.blocking_gate_codes)
                self.assertFalse(decision.controlled_rehearsal_authorized)

    def test_d5d_pass_and_current_conditional_evidence_are_accepted(self):
        pass_decision = evaluate_persistence_migration_readiness(
            _evidence(d5d_outcome="PASS", accepted_limitation_codes=[], total_review_required_count=0)
        )
        self.assertEqual(pass_decision.decision, CONDITIONALLY_READY_FOR_CONTROLLED_MIGRATION_REHEARSAL)

        current = evaluate_persistence_migration_readiness(load_persistence_migration_readiness_evidence(EVIDENCE_FIXTURE))
        self.assertEqual(current.decision, CONDITIONALLY_READY_FOR_CONTROLLED_MIGRATION_REHEARSAL)
        for code in (
            "REVIEW_REQUIRED_RECORDS_PRESENT",
            "APPROVED_LIMITATIONS_PRESENT",
            "OCR_REVIEW_REQUIRED",
            "SECURITY_DEPENDENCY_REVIEW_REQUIRED",
        ):
            self.assertIn(code, current.conditional_gate_codes)

    def test_ready_for_rehearsal_never_authorizes_production_or_vectors(self):
        policy_data = default_persistence_governance_policy_dict()
        policy_data["retention_policy"]["production_retention_duration_finalized"] = True
        policy_data["warning_ownership"]["production_owner_signoff_complete"] = True
        policy_data["legacy_coexistence_policy"]["production_cutover_policy_complete"] = True
        clean = _evidence(
            accepted_limitation_codes=[],
            high_security_finding_count=0,
            low_security_finding_count=0,
            moderate_security_finding_count=0,
            ocr_observation_count=0,
            total_review_required_count=0,
            total_warning_count=0,
            unresolved_security_findings=False,
        )

        decision = evaluate_persistence_migration_readiness(clean, policy=policy_from_dict(policy_data))

        self.assertEqual(decision.decision, READY_FOR_CONTROLLED_MIGRATION_REHEARSAL)
        self.assertTrue(decision.controlled_rehearsal_authorized)
        self.assertFalse(decision.controlled_pilot_authorized)
        self.assertFalse(decision.production_persistence_authorized)
        self.assertFalse(decision.production_indexing_authorized)
        self.assertFalse(decision.production_retrieval_authorized)

    def test_record_status_policy(self):
        policy = default_persistence_governance_policy()
        status = policy.record_status_policy

        self.assertEqual(status["valid"]["rehearsal"], "eligible")
        self.assertEqual(status["valid_with_warnings"]["rehearsal"], "eligible_with_approval")
        self.assertEqual(status["review_required"]["rehearsal"], "quarantine")
        self.assertEqual(status["review_required"]["index_retrieval"], "forbidden")
        self.assertEqual(status["rejected"]["rehearsal"], "forbidden")
        self.assertEqual(status["valid"]["production_persistence"], "future_approval_required")

    def test_provenance_policy(self):
        policy = default_persistence_governance_policy()
        provenance = policy.provenance_policy

        self.assertEqual(provenance["full_provenance"]["rehearsal"], "eligible")
        self.assertEqual(provenance["partial_provenance"]["default"], "disabled")
        self.assertEqual(provenance["partial_provenance"]["indexing"], "forbidden")
        self.assertEqual(provenance["legacy_filename_only"]["disposition"], "quarantine")
        self.assertEqual(provenance["unknown_provenance"]["disposition"], "forbidden")

    def test_limitation_table_ocr_legacy_retention_security_policies(self):
        policy = default_persistence_governance_policy()

        self.assertTrue(policy.limitation_policy["candidate_level_not_document_global"])
        self.assertFalse(policy.limitation_policy["corpus_wide_default"])
        self.assertEqual(
            policy.table_policy["TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE"]["known_candidate_id"],
            "aircraft_system_safety:chunk:page-52-table-1",
        )
        self.assertEqual(policy.table_policy["TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE"]["rehearsal_disposition"], "quarantine")
        self.assertFalse(policy.table_policy["TABLE_CANDIDATE_ONLY"]["row_structure_claim_allowed"])
        self.assertFalse(policy.ocr_policy["ocr_execution_authorized"])
        self.assertTrue(policy.ocr_policy["production_indexing_requires_review"])
        self.assertEqual(policy.legacy_coexistence_policy["mode"], "shadow_mode_only")
        self.assertTrue(policy.legacy_coexistence_policy["no_legacy_deletion"])
        self.assertTrue(policy.retention_policy["previous_package_retained"])
        self.assertTrue(policy.retention_policy["rollback_material_retained"])
        self.assertFalse(policy.retention_policy["production_retention_duration_finalized"])
        self.assertTrue(policy.security_gate["findings_block_production"])
        self.assertFalse(policy.security_gate["dependency_remediation_authorized"])

    def test_decision_and_policy_serialization_are_deterministic_and_private(self):
        decision = evaluate_persistence_migration_readiness(load_persistence_migration_readiness_evidence(EVIDENCE_FIXTURE))
        first = decision_report_json_bytes(decision)
        second = decision_report_json_bytes(decision)
        markdown = decision_report_markdown(decision)
        payload = first.decode("utf-8") + markdown

        self.assertEqual(first, second)
        self.assertTrue(first.decode("utf-8").endswith("\n"))
        self.assertTrue(markdown.endswith("\n"))
        self.assertNotIn("C:\\", payload)
        self.assertNotIn("Aspire5 15 i7 4G2050", payload)
        self.assertNotIn("source text", payload.lower())
        self.assertNotIn("chunk text", payload.lower())
        json.dumps(decision_to_dict(decision), sort_keys=True)
        json.dumps(default_persistence_governance_policy_dict(), sort_keys=True)

    def test_decision_fixture_matches_current_evaluator(self):
        evidence = load_persistence_migration_readiness_evidence(EVIDENCE_FIXTURE)
        policy = load_persistence_governance_policy(POLICY_PATH)
        decision = evaluate_persistence_migration_readiness(evidence, policy=policy)
        fixture = json.loads(DECISION_FIXTURE.read_text(encoding="utf-8"))

        self.assertEqual(fixture, decision_to_dict(decision))
        self.assertEqual(fixture["decision"], CONDITIONALLY_READY_FOR_CONTROLLED_MIGRATION_REHEARSAL)
        self.assertFalse(fixture["authorization"]["production_persistence"])

    def test_cli_exit_codes_report_permission_and_exclusions(self):
        self.assertTrue(TOOL_SCRIPT.exists())
        conditional = _run_cli(["--evidence", str(EVIDENCE_FIXTURE), "--policy", str(POLICY_PATH), "--strict"])
        self.assertEqual(conditional.returncode, 2, conditional.stdout + conditional.stderr)
        self.assertIn("Decision: CONDITIONALLY_READY_FOR_CONTROLLED_MIGRATION_REHEARSAL", conditional.stdout)
        self.assertIn("Governance review only.", conditional.stdout)
        self.assertIn("No migration executed.", conditional.stdout)
        self.assertIn("No embeddings generated.", conditional.stdout)
        self.assertIn("No Astra access.", conditional.stdout)
        self.assertIn("No FAISS access.", conditional.stdout)

        with tempfile.TemporaryDirectory() as tmp:
            ready_evidence = _write_evidence(
                Path(tmp) / "ready.json",
                accepted_limitation_codes=[],
                high_security_finding_count=0,
                low_security_finding_count=0,
                moderate_security_finding_count=0,
                ocr_observation_count=0,
                total_review_required_count=0,
                total_warning_count=0,
                unresolved_security_findings=False,
            )
            policy_data = default_persistence_governance_policy_dict()
            policy_data["retention_policy"]["production_retention_duration_finalized"] = True
            policy_data["warning_ownership"]["production_owner_signoff_complete"] = True
            policy_data["legacy_coexistence_policy"]["production_cutover_policy_complete"] = True
            ready_policy = Path(tmp) / "policy.json"
            ready_policy.write_text(json.dumps(policy_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            ready = _run_cli(["--evidence", str(ready_evidence), "--policy", str(ready_policy), "--strict"])
            self.assertEqual(ready.returncode, 0, ready.stdout + ready.stderr)

            not_ready = _write_evidence(Path(tmp) / "not-ready.json", d5c_outcome="FAIL")
            failed = _run_cli(["--evidence", str(not_ready), "--policy", str(POLICY_PATH), "--strict"])
            self.assertEqual(failed.returncode, 1)

            denied = _run_cli(
                [
                    "--evidence",
                    str(EVIDENCE_FIXTURE),
                    "--policy",
                    str(POLICY_PATH),
                    "--report-json",
                    str(Path(tmp) / "report.json"),
                    "--strict",
                ]
            )
            self.assertEqual(denied.returncode, 1)
            self.assertIn("Report writing requires --allow-report-write.", denied.stdout)

    def test_cli_report_writes_are_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_1 = Path(tmp) / "run_1"
            run_2 = Path(tmp) / "run_2"
            for run in (run_1, run_2):
                completed = _run_cli(
                    [
                        "--evidence",
                        str(EVIDENCE_FIXTURE),
                        "--policy",
                        str(POLICY_PATH),
                        "--report-json",
                        str(run / "decision.json"),
                        "--report-markdown",
                        str(run / "decision.md"),
                        "--allow-report-write",
                        "--strict",
                    ]
                )
                self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)

            self.assertEqual((run_1 / "decision.json").read_bytes(), (run_2 / "decision.json").read_bytes())
            self.assertEqual((run_1 / "decision.md").read_bytes(), (run_2 / "decision.md").read_bytes())

    def test_runtime_ingestion_files_remain_unreferenced(self):
        module_text = (SRC_DIR / "aviationrag" / "governance" / "persistence_migration_readiness.py").read_text(encoding="utf-8")
        cli_text = TOOL_SCRIPT.read_text(encoding="utf-8")
        combined = module_text + cli_text

        self.assertNotIn("read_documents.py", combined)
        self.assertNotIn("aviation_chunk_saver.py", combined)
        self.assertNotIn("faiss_indexer.py", combined)


def _fixture_data() -> dict:
    return json.loads(EVIDENCE_FIXTURE.read_text(encoding="utf-8"))


def _evidence(**patch) -> PersistenceMigrationReadinessEvidence:
    data = _fixture_data()
    data.update(patch)
    return PersistenceMigrationReadinessEvidence(
        accepted_limitation_codes=tuple(data["accepted_limitation_codes"]),
        astra_untouched=data["astra_untouched"],
        blocking_issue_codes=tuple(data["blocking_issue_codes"]),
        chunk_id_collision_count=data["chunk_id_collision_count"],
        d5c_outcome=data["d5c_outcome"],
        d5d_outcome=data["d5d_outcome"],
        deterministic_profile_count=data["deterministic_profile_count"],
        embeddings_untouched=data["embeddings_untouched"],
        evaluated_profile_count=data["evaluated_profile_count"],
        expected_deterministic_profile_count=data["expected_deterministic_profile_count"],
        faiss_untouched=data["faiss_untouched"],
        high_security_finding_count=data["high_security_finding_count"],
        low_security_finding_count=data["low_security_finding_count"],
        moderate_security_finding_count=data["moderate_security_finding_count"],
        ocr_observation_count=data["ocr_observation_count"],
        runtime_ingestion_unchanged=data["runtime_ingestion_unchanged"],
        schema_consistency_verified=data["schema_consistency_verified"],
        total_accepted_record_count=data["total_accepted_record_count"],
        total_candidate_count=data["total_candidate_count"],
        total_rejected_candidate_count=data["total_rejected_candidate_count"],
        total_review_required_count=data["total_review_required_count"],
        total_warning_count=data["total_warning_count"],
        unknown_provenance_count=data["unknown_provenance_count"],
        unresolved_security_findings=data["unresolved_security_findings"],
    )


def _write_evidence(path: Path, **patch) -> Path:
    data = _fixture_data()
    data.update(patch)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
