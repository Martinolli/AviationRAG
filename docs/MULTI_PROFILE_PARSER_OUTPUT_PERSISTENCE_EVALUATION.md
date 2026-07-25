# Multi-Profile Parser-Output Persistence Evaluation

Date: 2026-07-25
Status: D.5d `ACCEPTED_WITH_LIMITATIONS`
Scope: Three controlled real parser-output profiles only

## 1. Purpose

D.5d evaluates the existing D.4c adapter and D.5b persisted-record package
boundary against three contrasting real `techdoc-parser` StructuredDocument
artifacts.

The evaluation remains offline, deterministic, storage-neutral, and fail-closed.

## 2. Relationship to D.5c

D.5c proved one clean real parser-output sample could pass the D.4c and D.5b
chain. D.5d extends that check to three independent real profiles while keeping
each profile package separate.

```text
Real PDF
    -> techdoc-parser StructuredDocument artifact + manifest
    -> D.4c adapter
    -> StructuredDocumentChunkCandidate
    -> D.5b mapper and validator
    -> independent deterministic package
    -> D.5d aggregate consistency evaluation
```

## 3. Evaluation Scope

Only these profiles were evaluated:

| Profile | Source file | Pages | Role | P0 outcome |
| --- | --- | ---: | --- | --- |
| `flight_test_rm_ag_300` | `Flight_Test_RM_AG_300_V32.pdf` | 210 | flight-test publication | `ACCEPTED` |
| `mil_std_882e` | `MIL-STD-882E.pdf` | 106 | formal safety standard | `ACCEPTED` |
| `aircraft_system_safety` | `Aircraft_System_Safety_Military_Civil_Aeronautical_Applications.pdf` | 367 | accepted-limitation profile | `ACCEPTED_WITH_LIMITATIONS` |

No other document was processed.

## 4. Profile Selection Rationale

The selected profiles exercise flight-test technical publication structure,
formal safety-standard clauses and terminology, and a complex safety publication
with an accepted classification limitation.

## 5. Source Location Policy

All source PDFs remained in `techdoc-parser/input`. They were not moved or
copied into AviationRAG.

## 6. No Copy to data/documents

Nothing was copied into `AviationRAG/data/documents`.

## 7. Parser Artifact Generation/Reuse

No matching full-document StructuredDocument/manifest pair was found by source
SHA-256, so all three artifacts were generated under ignored
`techdoc-parser/output/d5d_multi_profile/`.

Commands:

```powershell
techdoc-parse Flight_Test_RM_AG_300_V32.pdf --output document.json --structured-document-output structured_document.json --structured-document-id flight_test_rm_ag_300 --manifest-output manifest.json --structured-document-overwrite
techdoc-parse MIL-STD-882E.pdf --output document.json --structured-document-output structured_document.json --structured-document-id mil_std_882e --manifest-output manifest.json --structured-document-overwrite
techdoc-parse Aircraft_System_Safety_Military_Civil_Aeronautical_Applications.pdf --output document.json --structured-document-output structured_document.json --structured-document-id aircraft_system_safety --manifest-output manifest.json --structured-document-overwrite
```

Each parser command exited `0`. The flight-test parser command reported that page
2 appeared to have no native text and may require OCR; no OCR experiment or OCR
flag was run.

## 8. Per-Profile Source Checksums

| Profile | Source SHA-256 |
| --- | --- |
| `flight_test_rm_ag_300` | `70bb005d0540836b0d5d5e759c088f32a5b98a094ad973344b11264507ffb98e` |
| `mil_std_882e` | `b041218c488ce448738696eac463fae040db39cd18dd000939d6efe282a9ac14` |
| `aircraft_system_safety` | `ce6bd8f65f6a1737b8538c0709580f043e853b5c407c87f32d651f17c6ec4477` |

## 9. Per-Profile Parser/Schema Identities

| Profile | Parser | StructuredDocument schema | Artifact SHA-256 | Manifest SHA-256 |
| --- | --- | --- | --- | --- |
| `flight_test_rm_ag_300` | `techdoc-parser / 0.1.0` | `techdoc-structured-document / 0.1.0` | `16fcce707d92e99c231483263d08cfc3106c83a2994007d0b3e421e592711223` | `ea8a3009a0a9a4f69a2d277e62e7a474e5c23d33166ecac678b7fe01adcdcd73` |
| `mil_std_882e` | `techdoc-parser / 0.1.0` | `techdoc-structured-document / 0.1.0` | `7d9898fd35302f548b3a87cbca58cf1c4f0e6f379ad67722c70efb3700fc3a25` | `5b36b4c0dcccb6b755d9b2e8380800e94c629eb00e130f61792a6cfea32f699e` |
| `aircraft_system_safety` | `techdoc-parser / 0.1.0` | `techdoc-structured-document / 0.1.0` | `1efcb160ed5d429e73e1fc09c825546d6d41c5738595c978b76ed3ba9cc47df7` | `b93b5506fd95ca551b525f0771e9f474d75a25116ab083b5997aefa8145398dd` |

## 10. Per-Profile Adapter Outcomes

| Profile | Adapter outcome |
| --- | --- |
| `flight_test_rm_ag_300` | `PASS` |
| `mil_std_882e` | `PASS` |
| `aircraft_system_safety` | `PASS` |

## 11. Per-Profile Candidate Counts

| Profile | Candidates |
| --- | ---: |
| `flight_test_rm_ag_300` | 6187 |
| `mil_std_882e` | 2406 |
| `aircraft_system_safety` | 7741 |

## 12. Per-Profile Persistence Outcomes

| Profile | Persisted | Rejected | Warnings | Review required | Package outcome | Gate outcome |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `flight_test_rm_ag_300` | 6187 | 0 | 0 | 0 | `PASS` | `PASS` |
| `mil_std_882e` | 2406 | 0 | 0 | 0 | `PASS` | `PASS` |
| `aircraft_system_safety` | 7741 | 0 | 2 | 1 | `REVIEW` | `REVIEW` |

## 13. Validation-Status Matrix

| Profile | Valid | Review required |
| --- | ---: | ---: |
| `flight_test_rm_ag_300` | 6187 | 0 |
| `mil_std_882e` | 2406 | 0 |
| `aircraft_system_safety` | 7740 | 1 |

## 14. Provenance Matrix

| Profile | Full | Partial | Unknown |
| --- | ---: | ---: | ---: |
| `flight_test_rm_ag_300` | 6187 | 0 | 0 |
| `mil_std_882e` | 2406 | 0 | 0 |
| `aircraft_system_safety` | 7741 | 0 | 0 |

## 15. Content-Type Matrix

| Profile | Paragraph | Table | Figure caption | Note |
| --- | ---: | ---: | ---: | ---: |
| `flight_test_rm_ag_300` | 5976 | 204 | 0 | 7 |
| `mil_std_882e` | 2333 | 69 | 2 | 2 |
| `aircraft_system_safety` | 7494 | 241 | 0 | 6 |

## 16. Warning and Limitation Matrix

| Profile | Warning count | Limitation code | Candidate count | Disposition |
| --- | ---: | --- | ---: | --- |
| `flight_test_rm_ag_300` | 0 | none | 0 | none |
| `mil_std_882e` | 0 | none | 0 | none |
| `aircraft_system_safety` | 2 | `TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE` | 1 | accepted limitation; review required |

The two warnings on the aircraft-safety profile are the explicit configured
table-classification review warning plus the warning implied by the approved
limitation code.

## 17. Known Page-52 Classification Limitation

The upstream P0 accepted limitation is
`TABLE_FALSE_POSITIVE_ON_FIGURE_PAGE` for `aircraft_system_safety`, one-based
page 52.

The generated artifact resolved page 52 to PDF page index 51. The matching
candidate is `aircraft_system_safety:chunk:page-52-table-1`, with table evidence
`aircraft_system_safety:p51:t0022` and source block `page-52-table-1`.

## 18. Candidate-Level Limitation Attachment Method

The limitation was attached through a local ignored candidate context:

```text
data/migration_dry_run/multi_profile_persistence/aircraft_system_safety/candidate_contexts.local.json
```

Only the exact candidate ID received the accepted limitation. No limitation was
attached globally to the document, all page-52 candidates, all table candidates,
or other profiles.

## 19. Rejected-Candidate Results

Rejected candidates were zero for all profiles.

## 20. Per-Profile Package Digests

| Profile | Package digest |
| --- | --- |
| `flight_test_rm_ag_300` | `44fc8fd6ab799d3b2bfe6e530c5e8ddc91e01ee834c8e4a190a8675da3717026` |
| `mil_std_882e` | `f1abf41c7d93d23eec24829181e1496ca63a55869bc78d9d94bb6105e0ae71c1` |
| `aircraft_system_safety` | `cdac3287b5da537ca47fe7d9f33f6140292bcf288baefca0a3ae285438b39bef` |

## 21. Determinism Results

| Profile | Run 1 digest | Run 2 digest | Byte-identical |
| --- | --- | --- | --- |
| `flight_test_rm_ag_300` | `44fc8fd6ab799d3b2bfe6e530c5e8ddc91e01ee834c8e4a190a8675da3717026` | `44fc8fd6ab799d3b2bfe6e530c5e8ddc91e01ee834c8e4a190a8675da3717026` | yes |
| `mil_std_882e` | `f1abf41c7d93d23eec24829181e1496ca63a55869bc78d9d94bb6105e0ae71c1` | `f1abf41c7d93d23eec24829181e1496ca63a55869bc78d9d94bb6105e0ae71c1` | yes |
| `aircraft_system_safety` | `cdac3287b5da537ca47fe7d9f33f6140292bcf288baefca0a3ae285438b39bef` | `cdac3287b5da537ca47fe7d9f33f6140292bcf288baefca0a3ae285438b39bef` | yes |

## 22. Cross-Document Chunk-ID Uniqueness

Cross-document chunk-ID collision count: 0.

Chunk IDs remained namespaced by document ID. No cross-document text-only
deduplication was performed.

## 23. Schema Consistency

Schema consistency passed:

| Field | Value |
| --- | --- |
| Persisted schema | `aviationrag-persisted-chunk / 0.1.0` |
| Mapper version | `0.1.0` |
| Adapter version | `D.4c` |
| Package schema | `aviationrag-persisted-chunk-package / 0.1.0` |
| Limitation registry version | `0.1.0` |

## 24. Rollback and Audit Evidence

Each profile can be reconstructed from its source checksum, parser version,
StructuredDocument schema version, artifact checksum, manifest checksum,
adapter version, mapper version, limitation-registry version, package digest,
and local gate report.

## 25. Local Output and Ignore Protection

Generated D.5d package outputs were written only under ignored:

```text
data/migration_dry_run/multi_profile_persistence/
```

The parser artifacts remained under ignored `techdoc-parser/output/`.

## 26. Privacy/Source-Content Controls

No source PDF, full StructuredDocument artifact, parser manifest, persisted
package, source text, chunk text, table content, figure content, or local
candidate-context file is committed.

## 27. Aggregate D.5d Outcome

```text
Profile count: 3
Total candidates: 16334
Total accepted: 16334
Total rejected: 0
Total warnings: 2
Total review required: 1
Outcome: ACCEPTED_WITH_LIMITATIONS
```

## 28. What D.5d Established

D.5d established that the D.4c and D.5b contract can produce deterministic,
schema-consistent, independent persisted-record packages for exactly three
contrasting real parser-output profiles.

## 29. What D.5d Did Not Establish

D.5d did not validate the full corpus, prove full-document semantic accuracy,
prove OCR accuracy, authorize runtime ingestion, authorize production
persistence, generate embeddings, rebuild Astra, rebuild FAISS, or integrate
retrieval.

## 30. Remaining Blockers

The aircraft-safety page-52 accepted limitation remains deferred parser
refinement. The flight-test parser note about page 2 native text remains a
source/parser-quality observation, not an OCR validation result.

Full-corpus processing, embeddings, Astra, FAISS, and production retrieval
remain unauthorized.

## 31. Preconditions for the Next Phase

The next phase should be:

```text
D.6 — Persistence Governance Decision and Migration Readiness Review
```

D.6 must decide policy before any broader document processing, vector rebuild,
or production migration.
