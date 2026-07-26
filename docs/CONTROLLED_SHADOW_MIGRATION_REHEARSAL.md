# Controlled Shadow Migration Rehearsal

Date: 2026-07-26
Status: D.7 `PASS_WITH_QUARANTINE`
Scope: Local shadow rehearsal only; no cutover, indexing, retrieval, parser run, OCR, Astra, FAISS, or embeddings

## 1. Purpose

D.7 rehearsed migration of the already validated D.5c and D.5d persisted
packages into a deterministic local shadow store. The rehearsal verified package
integrity, read-only legacy inventory, document reconciliation, eligibility
classification, quarantine, accounting, determinism, rollback, and legacy
unchanged evidence.

## 2. D.6 Authorization Basis

D.6 produced
`CONDITIONALLY_READY_FOR_CONTROLLED_MIGRATION_REHEARSAL`. It authorized only a
local controlled shadow migration rehearsal under the required controls:
`SHADOW_MODE_ONLY`, `NO_DESTRUCTIVE_OVERWRITE`, `NO_LEGACY_DELETION`,
`FULL_PROVENANCE_REQUIRED`, `ZERO_REJECTED_RECORDS`,
`REVIEW_REQUIRED_RECORDS_QUARANTINED`, `NO_EMBEDDINGS`, `NO_ASTRA`, `NO_FAISS`,
`NO_PRODUCTION_RETRIEVAL`, `PACKAGE_DETERMINISM_REQUIRED`,
`ROLLBACK_PACKAGE_RETAINED`, `WARNING_APPROVALS_SCOPED`,
`OCR_OBSERVATIONS_RETAINED`, and `SECURITY_REVIEW_PENDING`.

## 3. Scope

Included: four approved local persisted packages, read-only legacy filename and
chunk metadata inventory, deterministic shadow/quarantine outputs, and rollback
evidence.

Excluded: PDF processing, `techdoc-parser` execution, runtime ingestion changes,
legacy rewrites, embeddings, vectors, Astra, FAISS, OCR, production persistence,
production retrieval, and migration pilot authorization.

## 4. Approved Structured Packages

| Document | Records | Package digest | Integrity |
| --- | ---: | --- | --- |
| `faa_order_4040_26b` | 920 | `d2509f9dbaba886b82cb135b386a7c494aaf0569a8422ad4031cd9c38a26f6a5` | PASS |
| `flight_test_rm_ag_300` | 6,187 | `44fc8fd6ab799d3b2bfe6e530c5e8ddc91e01ee834c8e4a190a8675da3717026` | PASS |
| `mil_std_882e` | 2,406 | `f1abf41c7d93d23eec24829181e1496ca63a55869bc78d9d94bb6105e0ae71c1` | PASS |
| `aircraft_system_safety` | 7,741 | `cdac3287b5da537ca47fe7d9f33f6140292bcf288baefca0a3ae285438b39bef` | PASS |

## 5. Legacy Inventory Boundary

The read-only inventory was restricted to `data/documents` and
`data/processed/chunked_documents` candidates relevant to the four structured
document identities and configured aliases. It did not inventory embeddings,
vectors, full source text, or full chunk text.

## 6. Package Integrity Verification

The D.7 loader verified required files, manifest JSON, supported package schema,
persisted schema, mapper version, limitation registry version, file checksums,
package digest, manifest/report counts, zero rejected records, record
validation, forbidden vector fields, and coherent package source identity.

## 7. Document Identity Policy

Exact source checksum is the only exact identity proof. Document-ID-only,
same-filename, and filename-alias matches remain review findings and do not
authorize merge, overwrite, or cutover. Title similarity alone is ignored.

## 8. Exact Checksum Matching

`flight_test_rm_ag_300` matched one legacy source file by exact SHA-256:
`source:2026_02_25_flight_test_safety_risk_management_ad1183573_v1`.

## 9. Filename-Only Limitations

`mil_std_882e` matched a legacy chunk file by filename/document-key evidence
without source checksum. This requires review and is not exact source identity.

## 10. Origin Separation

Structured records and legacy inventory records retain separate origins. No
legacy and structured text was merged or deduplicated.

## 11. Record Eligibility

`valid` plus `full_provenance` records became shadow eligible. `review_required`
records were quarantined. `valid_with_warnings` would quarantine by default
without explicit approval. Rejected, partial provenance, and unknown provenance
records are forbidden.

## 12. Quarantine Policy

Quarantine is not rejection. Quarantined records remain in accounting but are
not indexing-eligible or retrieval-eligible.

## 13. Known Table-Classification Quarantine

`aircraft_system_safety:chunk:page-52-table-1` appeared exactly once through its
source block reference, retained `TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE`, retained
`TABLE_CLASSIFICATION_REVIEW_REQUIRED`, and was written only to quarantine.

## 14. OCR Observation Retention

`flight_test_rm_ag_300` page 2 retained
`OCR_COMPLETENESS_NOT_ESTABLISHED`. No OCR ran, no records were retroactively
mutated, page completeness was not claimed, and production cutover remains
blocked for that document.

## 15. Shadow-Store Layout

Generated local ignored outputs:

```text
data/migration_dry_run/shadow_migration_rehearsal/
  config.local.json
  run_1/
  run_2/
  rollback_test/
  d7_rehearsal_report.json
```

Each run contains shadow records, quarantine records, package catalog, legacy
inventory, document reconciliation, accounting, shadow manifest, shadow report,
and rollback manifest.

## 16. Structured Package Catalog

The catalog records package digests, source filenames, source checksums, schema
names/versions, mapper version, limitation registry version, record counts,
eligible counts, quarantine counts, forbidden counts, and rejected counts.
Committed fixtures omit local package roots.

## 17. Legacy Inventory

The inventory found 3 legacy records and 160 legacy chunks: one exact source
file for the flight-test document, one legacy chunk file for the flight-test
document, and one legacy chunk file for MIL-STD-882E. The report marks
filename-only legacy chunks as `legacy_filename_only`.

## 18. Document Reconciliation

| Structured document | Reconciliation status | Legacy match | Review required | Cutover eligible |
| --- | --- | --- | --- | --- |
| `aircraft_system_safety` | `NO_LEGACY_MATCH` | none | false | false |
| `faa_order_4040_26b` | `NO_LEGACY_MATCH` | none | false | false |
| `flight_test_rm_ag_300` | `EXACT_SOURCE_CHECKSUM_MATCH` | one source checksum match | false | false |
| `mil_std_882e` | `DOCUMENT_ID_MATCH_WITHOUT_SOURCE_CHECKSUM` | one legacy chunk file | true | false |

## 19. Migration Accounting

```text
Structured records: 17,254
Shadow eligible: 17,253
Quarantined: 1
Forbidden: 0
Rejected: 0
Accounting result: PASS
```

Eligible plus quarantine plus forbidden equaled the total structured record
count. Zero rejected records remained.

## 20. Determinism

`run_1` and `run_2` were byte-identical for all generated run artifacts. The
aggregate shadow digest was
`6a8ad070f565e207616abf8e0e104835a60e683f3be6eab193c098762894b02f`.

## 21. Rollback Rehearsal

Rollback evidence showed zero legacy files created, modified, or deleted. The
temporary shadow activation marker was removed, the baseline was restored, and
the rebuild digest was identical.

## 22. Legacy Unchanged Verification

Legacy source and chunk roots were snapshotted before and after D.7 using file
counts, sizes, and SHA-256 checksums. The snapshots matched.

## 23. Privacy and Source Protection

Committed reports and fixtures do not include source text, chunk text, vectors,
absolute machine paths, or usernames. Full shadow/quarantine records remain
under ignored local output only.

## 24. Security Boundary

No dependencies were added and no dependency remediation was performed.
Unresolved dependency findings remain a production security gate.

## 25. Formal D.7 Result

```text
Outcome: PASS_WITH_QUARANTINE
Exit code: 2
Package integrity verified: true
Accounting verified: true
Determinism verified: true
Rollback verified: true
Legacy unchanged: true
```

## 26. What D.7 Established

D.7 established that the four approved packages can be loaded, verified,
classified, reconciled, written into a local deterministic shadow store, and
rolled back without changing legacy data or runtime retrieval.

## 27. What D.7 Did Not Establish

D.7 did not authorize a controlled migration pilot, production persistence,
production indexing, production retrieval, embeddings, Astra operations, FAISS
operations, OCR execution, parser execution, full-corpus processing, or legacy
deletion.

## 28. Remaining Blockers

Remaining findings: the page-52 table-classification quarantine, the flight-test
OCR observation, the MIL-STD-882E document-ID-without-source-checksum legacy
match, no legacy match for FAA Order or Aircraft System Safety, security review,
warning-owner signoff, production retention, and legacy cutover policy.

## 29. Preconditions for a Controlled Migration Pilot

A controlled migration pilot requires D.8 governance review, scoped warning and
limitation disposition, OCR disposition or exclusion, security review, retention
policy, legacy cutover policy, explicit datastore authorization, and continued
proof that embeddings, Astra, FAISS, and runtime retrieval remain unauthorized
until separately approved.

## 30. Final Decision

D.7 succeeded for local shadow rehearsal with one governed quarantine.
Conditions now exist for D.8 controlled migration pilot readiness review only.
No pilot or production readiness is granted.
