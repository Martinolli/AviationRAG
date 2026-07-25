# Real Parser Sample Persistence Gate

Date: 2026-07-25
Status: D.5c PASS
Scope: One controlled real parser-output sample only

## 1. Purpose

D.5c verifies that one approved real `techdoc-parser` StructuredDocument
artifact can pass through the offline AviationRAG D.4c adapter and D.5b
persisted package boundary.

This is not production ingestion, full-corpus migration, embedding generation,
Astra work, FAISS work, retrieval integration, deployment, or response-policy
activation.

## 2. Relationship to D.4c, D.5, and D.5b

```text
Real PDF
    -> techdoc-parser StructuredDocument artifact + manifest
    -> D.4c adapter
    -> StructuredDocumentChunkCandidate
    -> D.5b persisted mapper/package
    -> D.5c acceptance gate
```

D.4c remains an offline parser-output adapter. D.5 remains the persisted
contract. D.5b remains the storage-neutral mapper and deterministic local
package writer.

## 3. Selected Document and Reason

Selected document: `FAA_Order_4040_26B.pdf`

Reason: small 39-page native-text FAA document, included in the completed P0
pilot, with document outcome `ACCEPTED` and no active document-level accepted
limitations in the final pilot closure.

## 4. Source Location Policy

The source remained in:

```text
C:\Users\Aspire5 15 i7 4G2050\techdoc-parser\input\FAA_Order_4040_26B.pdf
```

The source SHA-256 is:

```text
92faf3c369cafe243d668cab40000d6c31a2196a1063003504bfffe769d8c0a9
```

## 5. No Copy Into data/documents

Nothing was copied into `AviationRAG/data/documents`, and no source PDF was
added to the AviationRAG repository.

## 6. Parser Artifact Generation

No matching full-document StructuredDocument artifact/manifest pair existed in
the parser output area, so one was generated under ignored
`techdoc-parser/output/`.

Command:

```powershell
& "C:\Users\Aspire5 15 i7 4G2050\techdoc-parser\.venv\Scripts\techdoc-parse.exe" `
  "C:\Users\Aspire5 15 i7 4G2050\techdoc-parser\input\FAA_Order_4040_26B.pdf" `
  --output "C:\Users\Aspire5 15 i7 4G2050\techdoc-parser\output\d5c_real_parser_sample\faa_order_4040_26b\document.json" `
  --structured-document-output "C:\Users\Aspire5 15 i7 4G2050\techdoc-parser\output\d5c_real_parser_sample\faa_order_4040_26b\structured_document.json" `
  --structured-document-id faa_order_4040_26b `
  --document-title "FAA Order 4040.26B" `
  --document-number "FAA Order 4040.26B" `
  --document-revision "B" `
  --manifest-output "C:\Users\Aspire5 15 i7 4G2050\techdoc-parser\output\d5c_real_parser_sample\faa_order_4040_26b\manifest.json" `
  --structured-document-overwrite
```

Parser exit code: `0`

## 7. Source, Artifact, and Manifest Verification

| Item | SHA-256 |
| --- | --- |
| Source PDF | `92faf3c369cafe243d668cab40000d6c31a2196a1063003504bfffe769d8c0a9` |
| StructuredDocument artifact | `fb33be7d2bfce62d813f0c88676f1da2f0ec4f5146e547ae91f04113be2c7d83` |
| Parser manifest | `7f533273193c6f218e71833334f349bade3c5a7ad332b404b016010ff3252f6e` |

The manifest source checksum matched the source PDF. The manifest artifact
checksum matched the StructuredDocument artifact. The manifest registered the
same document ID, schema name, schema version, media type, and artifact path.

## 8. Parser and Schema Identity

| Field | Value |
| --- | --- |
| Parser name | `techdoc-parser` |
| Parser version | `0.1.0` |
| StructuredDocument schema name | `techdoc-structured-document` |
| StructuredDocument schema version | `0.1.0` |
| Document key | `faa_order_4040_26b` |
| Page count | 39 |
| Block count | 940 |

## 9. D.4c Adapter Result

Adapter outcome: `PASS`

Warning codes: none

Blocking issues: none

## 10. Candidate Counts

Input candidates: 920

## 11. D.5b Mapping Result

Package outcome: `PASS`

## 12. Persisted-Record Counts

Accepted records: 920

Rejected candidates: 0

## 13. Validation-Status Counts

```json
{"valid": 920}
```

## 14. Provenance Counts

```json
{"full_provenance": 920}
```

## 15. Content-Type Counts

```json
{"figure_caption": 2, "note": 7, "paragraph": 887, "table": 24}
```

## 16. Warnings and Limitations

Warnings: 0

Accepted limitation counts: `{}`

No warning codes or limitation codes were approved for the formal D.5c run.

## 17. Rejected-Candidate Result

Rejected candidates: 0

## 18. Package Digest

```text
d2509f9dbaba886b82cb135b386a7c494aaf0569a8422ad4031cd9c38a26f6a5
```

## 19. Two-Run Determinism Result

Run 1 digest: `d2509f9dbaba886b82cb135b386a7c494aaf0569a8422ad4031cd9c38a26f6a5`

Run 2 digest: `d2509f9dbaba886b82cb135b386a7c494aaf0569a8422ad4031cd9c38a26f6a5`

All package files were byte-identical and SHA-256-identical across both runs.

## 20. Rollback and Audit Evidence

The sample can be reconstructed from source PDF checksum, parser version,
StructuredDocument schema version, artifact checksum, manifest checksum, D.4c
adapter version, D.5b mapper version, persisted schema version, limitation
registry version, and package digest.

## 21. Local Output and Ignore Protection

Generated D.5c package outputs were written only under ignored:

```text
data/migration_dry_run/real_parser_sample/faa_order_4040_26b/
```

`git check-ignore` confirmed `run_1`, `run_2`, and `local_gate_report.json`
paths are ignored.

## 22. Privacy and Source-Content Controls

No source text, chunk text, table contents, figures, rendered pages, full
StructuredDocument artifact, full manifest, local output package, or source PDF
was committed.

## 23. Full-Document Accuracy Limitation

D.5c did not establish complete document semantic accuracy. It verified one
parser-generated artifact through the offline mapping and package gate.

## 24. No OCR Claim

No OCR accuracy claim is made. The selected document was treated as native-text
parser input, and no OCR experiment was authorized.

## 25. No Full-Corpus Authorization

D.5c validates one document only. It does not validate the full corpus and does
not authorize full-corpus processing.

## 26. No Embedding or Index Authorization

No embeddings were generated. Astra and FAISS were untouched.

## 27. D.5c Decision

Decision: `PASS`

## 28. Preconditions for the Next Phase

The next recommended phase is D.5d controlled multi-profile parser-output
persistence evaluation. It should use only a very small contrasting sample set
and must not automatically authorize full-corpus ingestion, embeddings, Astra,
FAISS, or production retrieval.
