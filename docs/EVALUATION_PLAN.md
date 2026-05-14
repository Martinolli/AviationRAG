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

1. No retrieval benchmark harness exists yet.
2. No benchmark question set exists yet.
3. No citation validator exists yet.
4. No formal response mode classifier exists yet.
5. No answer schema with `evidence_level` exists yet.
6. Page and section metadata are not mature enough for compliance-grade evaluation.
7. Current citations are extracted from context tags, not validated against claims.
8. Current smoke tests validate UI shell behavior, not RAG quality.

## Implementation Guardrail

This document is planning only. The evaluation harness should be implemented later as a separate phase after the production bridge blocker is closed or explicitly paused and after the repository baseline remains stable.
