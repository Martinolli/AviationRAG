# Evaluation Plan

Date: 2026-05-12  
Status: Planning baseline only  
Scope: Defines future evaluation approach; no harness is implemented by this document

## Why Evaluation Matters

AviationRAG is intended for aviation engineering, safety, certification, and compliance-support workflows. In this domain, fluent but unsupported answers are dangerous. Evaluation is required before claiming the system is reliable for compliance-grade use.

Primary risks:

1. Hallucinated regulatory or engineering claims.
2. Incorrect source selection for compliance questions.
3. Citations that point to irrelevant or insufficient evidence.
4. Confident answers when the uploaded sources do not contain the answer.
5. Retrieval regressions after parser, chunking, metadata, or model changes.
6. Latency or timeout behavior that masks backend failures.

Evaluation must measure retrieval quality before answer style is improved. A polished answer is not useful if the evidence is wrong.

## Evaluation Categories

| Category | Purpose |
| --- | --- |
| Retrieval accuracy | Verify the right documents and chunks are retrieved. |
| Citation correctness | Verify citations map to retrieved evidence and support the claim. |
| Answer quality | Verify the answer is accurate, concise, and source-grounded. |
| Refusal behavior | Verify the assistant refuses or asks for clarification when needed. |
| Insufficient evidence behavior | Verify missing evidence is explicitly reported. |
| Latency/performance | Track response time and timeout risk. |
| Regression testing | Detect quality drift across ingestion, retrieval, and prompt changes. |

## Benchmark Categories

Initial benchmark sets should cover:

1. Regulatory/compliance:
   - FAA, EASA, ICAO, advisory circulars, airworthiness, certification, standards.
2. Design/manufacturing:
   - Design assurance, conformity, quality management, MRB, production risk.
3. SMS/safety:
   - Safety management, hazards, HFACS, Dirty Dozen, Reason model, organizational risk.
4. Accident analysis:
   - Accident and incident reports, causal factors, investigation findings.
5. General aviation:
   - Conceptual questions and broad aviation explanations.
6. Not-found questions:
   - Questions whose answer is not present in the controlled source set.

## Retrieval Evaluation Concept

Retrieval evaluation should be implemented before retrieval changes are accepted.

Each benchmark item should include:

```json
{
  "id": "reg_001",
  "category": "regulatory_compliance",
  "question": "What does the source say about ...?",
  "expected_documents": ["example.pdf"],
  "expected_sections": ["optional section path"],
  "expected_pages": [1, 2],
  "notes": "Why this evidence is expected"
}
```

Metrics should include:

1. Top-1 hit rate:
   - Did the best result include expected evidence?
2. Top-3 hit rate:
   - Did any of the first three results include expected evidence?
3. Top-5 hit rate:
   - Did any of the first five results include expected evidence?
4. Mean reciprocal rank:
   - How early did the expected evidence appear?
5. Source-type correctness:
   - Did regulatory questions retrieve regulatory/standard sources before accident reports?

Expected documents, sections, and pages should be added where practical. Page/section checks will become more important after the planned metadata upgrade.

## Smoke Retrieval Fixture Baseline

A fake/sample-only smoke fixture now exists at:

```text
data/sample_documents/sample_retrieval_eval.jsonl
```

Validation utilities exist at:

```text
src/aviationrag/evaluation/smoke_fixture.py
```

Current scope:

1. Defines the JSONL shape for future retrieval evaluation cases.
2. Uses fake sample document and chunk IDs only.
3. Covers initial smoke categories: `compliance`, `manufacturing`, `sms`, `accident_analysis`, `maintenance`, and `unsupported`.
4. Validates fixture syntax, required fields, accepted categories, accepted expected behaviors, duplicate IDs, and minimum expected rank.
5. Does not run retrieval, connect to Astra, use FAISS, generate embeddings, or scan real documents.

Intended future use:

1. Use the smoke fixture to test the evaluation harness before connecting it to any real retrieval engine.
2. Add top-k retrieval hit metrics after retrieval integration exists.
3. Track expected document match and expected chunk match once real retrieval results can be compared.
4. Track unsupported query handling for `insufficient_evidence` and `reject_out_of_scope` cases.
5. Keep real/private evaluation datasets local or private if they expose source titles, document text, or operational metadata.

## Retrieval Harness Shell

A fake/mock retrieval result evaluator now exists at:

```text
src/aviationrag/evaluation/retrieval_harness.py
```

Current scope:

1. Loads `RetrievalEvaluationCase` objects from the smoke fixture utilities.
2. Accepts caller-supplied fake/mock `RetrievalResult` objects.
3. Evaluates expected document match, expected chunk match, top-k/rank requirements, insufficient-evidence behavior, and out-of-scope rejection behavior.
4. Produces per-case results and aggregate pass/fail summaries.
5. Does not call real retrieval, connect to Astra, use FAISS, generate embeddings, scan documents, or change runtime behavior.

Future work:

1. Wire the harness to real retrieval outputs after FAISS/Astra/hybrid retrieval integration is explicitly approved.
2. Save real benchmark runs as local/private JSON and Markdown reports.
3. Compare pre-change and post-change retrieval metrics before accepting retrieval, chunking, or metadata changes.

## Retrieval Evaluation Report Shell

Report/export utilities now exist at:

```text
src/aviationrag/evaluation/reporting.py
```

Current scope:

1. Converts fake/mock `EvaluationSummary` and `EvaluationCaseResult` objects into JSON-serializable dictionaries.
2. Renders Markdown summary reports with totals, pass rate, category counts, behavior counts, issues, and per-case results.
3. Writes JSON and Markdown reports only when explicit write functions are called.
4. Works with fake/mock harness outputs only.
5. Does not call real retrieval, connect to Astra, use FAISS, generate embeddings, scan documents, or change runtime behavior.

Generated reports should remain local and ignored by Git unless a future phase intentionally publishes a fake/sample report fixture. Real benchmark reports must remain local/private if they expose source titles, document text, retrieved chunks, or operational metadata.

Future real retrieval integration can reuse this reporting layer after retrieval outputs are explicitly wired to the harness.

## Citation Validation Concept

Citation validation should verify:

1. Every cited `filename` and `chunk_id` exists in the retrieved context.
2. The cited passage supports the associated factual claim.
3. Strict/compliance answers do not contain hard factual claims without citations.
4. Citations are not fabricated by the model.
5. Missing or weak support is surfaced as a warning or insufficient-evidence result.

Initial validation can be deterministic:

1. Check citation IDs against retrieved context.
2. Require at least one citation in strict/compliance answers.
3. Flag citations that are not present in the source list.

Later validation can include claim-level review and human scoring.

## Response Policy Validation Concept

Response policy evaluation should test whether the answer behavior matches the requested mode.

Examples:

1. `strict_document`:
   - Answer only from the requested document.
   - Say not found when evidence is missing.
2. `regulatory_compliance`:
   - Prefer exact wording and citations.
   - Avoid unsupported interpretation.
3. `design_review`:
   - Separate evidence from engineering interpretation.
4. `insufficient_evidence`:
   - Clearly state that the controlled sources do not contain the answer.

Policy tests should include expected behavior, not just expected text.

## Suggested Metrics

Retrieval metrics:

1. Top-1, top-3, top-5 hit rate.
2. Mean reciprocal rank.
3. Expected source-type hit rate.
4. Duplicate chunk rate.
5. Low-quality extraction rate in retrieved chunks.

Citation metrics:

1. Citation presence rate.
2. Valid citation ID rate.
3. Unsupported claim count.
4. Fabricated citation count.

Answer metrics:

1. Accuracy score.
2. Groundedness score.
3. Completeness score.
4. Conciseness score.
5. Insufficient-evidence correctness.
6. Human review escalation correctness.

Operational metrics:

1. End-to-end latency.
2. Retrieval latency.
3. LLM latency.
4. Timeout rate.
5. Bridge failure rate.

## Planned Evaluation Workflow

1. Create small benchmark files under a future `tests/evaluation/` folder.
2. Start with retrieval-only evaluation and no LLM calls.
3. Record baseline retrieval metrics before changing parsing, chunking, metadata, or retrieval logic.
4. Add answer evaluation only after citation validation and response mode outputs are formalized.
5. Save evaluation results as Markdown and JSON.
6. Compare every retrieval change against the baseline.
7. Keep slow or LLM-dependent evaluations manual or scheduled, not mandatory for every fast CI run.

## Reset/Rebuild Baseline Gate

Any future Astra, FAISS, embedding, or manifest-driven ingestion reset must be paired with retrieval baseline measurement. The reset/rebuild safety plan is documented in `docs/RESET_REBUILD_AND_EVALUATION_BASELINE.md`.

Minimum reset gate:

1. Run a pre-reset retrieval smoke baseline before deleting or rebuilding generated artifacts.
2. Run the same baseline after rebuild.
3. Compare top-1, top-3, top-5, expected-document, citation-traceability, and not-found correctness results.
4. Treat material retrieval regression as a rollback or no-go signal.
5. Keep evaluation outputs local/private when they expose real document names, source text, or operational metadata.

## Known Current Gaps

1. The retrieval harness and report/export shell currently operate on fake/mock results only.
2. No real retrieval benchmark execution is wired yet.
3. No citation validator exists yet.
4. No formal response mode classifier exists yet.
5. No answer schema with `evidence_level` exists yet.
6. Page and section metadata are not mature enough for compliance-grade evaluation.
7. Current citations are extracted from context tags, not validated against claims.
8. Current smoke tests validate UI shell behavior, not RAG quality.

## Implementation Guardrail

This document remains a controlled evaluation plan. The current harness and report/export shells only evaluate and format fake/mock retrieval results; real retrieval integration should be implemented later as a separate phase after the production bridge blocker is closed or explicitly paused and after the repository baseline remains stable.
