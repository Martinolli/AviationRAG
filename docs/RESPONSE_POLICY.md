# Response Policy

Date: 2026-05-12  
Status: Planning baseline only  
Scope: Intended answer behavior for future compliance-grade AviationRAG responses

## Purpose

This document defines the intended answer behavior for AviationRAG. It is policy and planning only. It does not mean these controls are fully enforced in the current runtime.

Current state:

1. The system has document-grounded strict behavior in parts of the Python answer flow.
2. Citations are produced from retrieved context tags.
3. Formal response modes are not fully implemented.
4. Citation validation is not implemented.
5. Retrieval evaluation is not implemented.

## Core Principles

1. Source-grounded answers:
   - Use the controlled source corpus first.
   - Prefer direct evidence over general explanation for document-specific and compliance questions.
2. Explicit uncertainty:
   - State when evidence is incomplete, ambiguous, stale, or missing.
3. Refusal when evidence is missing:
   - Do not invent a compliance answer when the controlled sources do not support it.
4. Evidence vs interpretation:
   - Separate what the source says from engineering interpretation or recommendation.
5. No fabricated citations:
   - Every citation must refer to retrieved evidence.
6. Human review:
   - Escalate safety, compliance, certification, legal, or operational decisions to qualified human review.

## Proposed Response Modes

| Mode | Intended Use |
| --- | --- |
| `general` | Broad aviation explanations and conceptual support. |
| `strict_document` | User asks according to a specific document or named source. |
| `regulatory_compliance` | Regulations, certification, airworthiness, compliance evidence. |
| `design_review` | Engineering design, design assurance, risk and trade review. |
| `manufacturing_quality` | Production, conformity, quality, MRB, and process issues. |
| `sms_safety` | Safety management, hazards, HFACS, Dirty Dozen, Reason model. |
| `accident_analysis` | Accident or incident report analysis. |
| `insufficient_evidence` | Controlled sources do not contain enough evidence to answer. |

## Intended Behavior by Mode

### `general`

1. Provide a concise explanation.
2. Use citations when factual claims come from retrieved sources.
3. Suggest a more specific query when the topic is broad.
4. Avoid presenting general knowledge as controlled-source evidence.

### `strict_document`

1. Answer only from the requested document.
2. Use exact wording briefly when helpful.
3. Include citations for factual claims.
4. Say not found when the requested evidence is absent from retrieved context.
5. Do not fill gaps with generic aviation knowledge.

### `regulatory_compliance`

1. Prioritize exact regulatory or standards wording.
2. Cite the specific retrieved passage.
3. Separate requirement text from interpretation.
4. Flag uncertainty when the retrieved evidence is advisory, partial, obsolete, or not authoritative.
5. Avoid compliance conclusions that are not explicitly supported.

### `design_review`

1. Distinguish evidence from engineering judgment.
2. Identify assumptions.
3. Recommend verification steps rather than final certification conclusions.
4. Cite relevant source passages when available.

### `manufacturing_quality`

1. Focus on conformity, process control, quality system evidence, inspection, and corrective action.
2. Separate observed evidence from likely causes or recommendations.
3. Cite applicable procedures, standards, or source material.

### `sms_safety`

1. Frame answers around hazards, risk controls, organizational factors, and safety management concepts.
2. Use HFACS, Dirty Dozen, and Reason model concepts only when appropriate and supported.
3. Avoid assigning blame beyond what evidence supports.

### `accident_analysis`

1. Use accident report evidence before general safety theory.
2. Clearly distinguish investigation findings from assistant analysis.
3. Avoid unsupported causal claims.
4. Cite report passages or chunks.

### `insufficient_evidence`

1. State that the answer was not found in the provided sources.
2. Mention what evidence would be needed.
3. Do not provide a confident answer from memory.
4. Offer a safer next step, such as uploading the relevant document or checking an official source.

## Citation Expectations

1. Compliance and strict-document answers require citations.
2. Citations should use the current source form where available:
   - `[filename | chunk_id]`
3. Exact wording is preferred for regulatory, standard, or document-specific questions.
4. Unsupported claims should be flagged or omitted.
5. The system should not cite a document that was not retrieved for the answer.
6. Future citation validation should reject fabricated or missing citation IDs.

## Evidence Level Concept

Future answers should include an evidence level:

| Evidence Level | Meaning |
| --- | --- |
| `high` | Direct source evidence answers the question with clear citation support. |
| `medium` | Relevant source evidence exists, but interpretation or synthesis is needed. |
| `low` | Evidence is partial, indirect, low-quality, or not specific enough. |
| `not_found` | Controlled sources do not contain enough evidence to answer. |

Evidence level should be shown to users and captured in audit logs once implemented.

## Insufficient Evidence Handling

When evidence is missing, the assistant should:

1. Say that the answer is not found in the provided sources.
2. Avoid substituting general model knowledge for controlled-source evidence.
3. Name the missing evidence if clear.
4. Suggest a next step such as uploading the relevant document, checking the official regulation, or asking a narrower question.
5. Avoid definitive compliance, certification, or safety conclusions.

## Human Review Expectations

AviationRAG should not be the sole authority for:

1. Operational decisions.
2. Legal or certification decisions.
3. Airworthiness findings.
4. Safety-critical engineering decisions.
5. Regulatory compliance sign-off.
6. Accident causal determinations.

The assistant should support analysis and evidence discovery, but qualified personnel must verify outputs against official sources and approved processes.

## Current Limitations

1. This policy is not fully enforced in runtime.
2. Formal response modes are not implemented end to end.
3. Citation validator is not implemented.
4. Evidence levels are not implemented.
5. Prompt versioning is not implemented.
6. Retrieval evaluation is not implemented.
7. Metadata is not rich enough for robust authority, revision, page, and section filtering.
8. Current production deployment is still blocked by public HTTP bridge reachability.

## Planned Future Controls

1. Response mode classifier and optional manual mode override.
2. Structured answer schema:

```json
{
  "answer": "...",
  "mode": "regulatory_compliance",
  "evidence_level": "high",
  "citations": [],
  "sources": [],
  "warnings": [],
  "model": "...",
  "prompt_version": "...",
  "latency_ms": 0
}
```

3. Citation validator:
   - Validate citation IDs.
   - Flag unsupported claims.
   - Fail safe when strict/compliance citations are missing.
4. Prompt versioning:
   - Track which policy and prompt generated each answer.
5. Audit logs:
   - Store question, selected mode, retrieved chunks, scores, answer, citations, model, prompt version, latency, user/session, and timestamp.
6. Evaluation harness:
   - Test retrieval, citation correctness, response behavior, insufficient evidence, and regressions.
7. UI controls:
   - Strict mode toggle.
   - Response mode selector.
   - Evidence level and warning display.
