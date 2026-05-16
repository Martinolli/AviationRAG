# PLANWORKLOG

Last Updated: 2026-05-16
Repository: `Martinolli/AviationRAG`  
Base Branch: `main`  
Purpose: Refactored execution plan comparing the current `WORKLOG.md` status with the broader AviationRAG product/audit roadmap.

---

## 1. Executive Summary

The project has progressed beyond the initial prototype stage. The current `WORKLOG.md` shows that several items originally expected in the early roadmap are already complete or partially complete, especially around deployment hardening, upload workflow, API security, authentication hardening, CI, and bridge architecture.

The next phase should not restart the project. It should consolidate the existing work and move the focus from **deployment readiness** to **RAG quality, aviation-domain trustworthiness, compliance behavior, evaluation, and maintainable architecture**.

### Current Strategic Status

| Area | Status | Interpretation |
| --- | ---: | --- |
| Repository sanitization | Partial / mostly done | Runtime artifacts and hooks are addressed, but historical cleanup remains. |
| Deployment hardening | Advanced / near cutover | HTTP bridge architecture exists; public endpoint cutover remains the major blocker. |
| Upload workflow | Done | UI/API/status pipeline exists. Needs governance and approval controls later. |
| Formula rendering | Done | Markdown/math rendering is complete with sanitization. |
| Web MVP | Partial / advanced | Chat, auth, health, upload, and history work exist. Needs final validation and source UX polish. |
| RAG quality | Planned | This is now the primary technical risk. |
| Evaluation harness | Planned | Must become a high-priority item before claiming compliance/specialist reliability. |
| Response policy | Planned / partial | Strict mode exists conceptually, but formal response modes and citation validation need implementation. |
| Data governance | Partial | Git ignore and sanitization exist; metadata, document approval, and version controls need strengthening. |

---

## 2. Comparison: Current WORKLOG vs Refactored Audit Roadmap

### 2.1 Items already ahead of the original audit suggestion

The following items are already implemented or substantially advanced according to `WORKLOG.md`:

1. **Repository sanitization baseline**
   - Runtime artifact untracking.
   - Ignore rules.
   - Pre-commit sanitization checks.
   - Sanitization report.
   - CI secret scan with gitleaks.

2. **Deployment hardening**
   - API rate limiting.
   - Request normalization.
   - Vercel routing cleanup.
   - CI pipeline for sanitize/build/smoke.
   - Auth hardening.
   - HTTP bridge mode.
   - Deep health diagnostics.
   - Vercel setup and bridge runbooks.

3. **Upload workflow**
   - Upload UI.
   - Upload API.
   - File validation.
   - Upload status API.
   - Ingestion queue/state pipeline.

4. **Math/formula rendering**
   - Markdown rendering.
   - Math support.
   - Sanitized rendering path.

5. **Operational documentation**
   - HTTP bridge spec.
   - Cutover checklist.
   - Vercel setup.
   - Local/Vercel sync runbook.

### 2.2 Items partially aligned but requiring more work

1. **Repository hygiene**
   - Current status is good for working-tree hygiene.
   - Still pending: historical large/generated blob cleanup and final policy for real aviation documents in a public repo.

2. **Testing**
   - Current smoke/build checks exist.
   - Still pending: RAG-specific regression tests, citation tests, retrieval tests, and insufficient-evidence tests.

3. **Web app**
   - Current implementation is beyond a simple shell.
   - Still pending: production validation, source viewer polish, admin controls, and final external bridge cutover.

4. **Document ingestion**
   - Large PDF processing and fallback handling have improved.
   - Still pending: full structure-aware parsing, metadata schema, document manifest, and approval lifecycle.

5. **Response behavior**
   - Strict/document-grounded behavior exists in the backend logic.
   - Still pending: formal response modes, policy enforcement, citation validation, and evidence-level output.

### 2.3 Items still missing or not yet mature

1. Formal backend package structure.
2. Versioned prompt templates.
3. Retrieval benchmark set.
4. Hybrid retrieval with lexical branch and metadata filters.
5. Section-aware chunking.
6. Citation correctness validation.
7. Compliance-grade answer contract.
8. Document manifest and document lifecycle governance.
9. Admin approval before new documents become active.
10. Audit log design for engineering/compliance traceability.
11. Human feedback loop for hallucinations, wrong citations, and obsolete sources.

---

## 3. Refactored Roadmap

The project should now follow this execution order:

1. **Finish current production cutover blockers.**
2. **Stabilize repository and runtime baseline.**
3. **Refactor backend without changing behavior.**
4. **Upgrade ingestion and metadata.**
5. **Modernize retrieval.**
6. **Formalize response policy and citation validation.**
7. **Build evaluation harness.**
8. **Harden data governance and audit trail.**
9. **Polish web/admin UX.**
10. **Only then consider external web research mode.**

---

## 4. Active Master Plan

### Security Sprint S — Dependency Hardening

**Goal:** Track dependency vulnerability reduction separately from ingestion, retrieval, manifest, bridge, embedding, and prompt work.

**Priority:** High  
**Recommended branch:** `security/dependency-hardening`

#### S.1 Dependency audit baseline and safe fixes

Status: `Completed` / `Major upgrades planned`

Tasks:

- [x] Create npm audit baseline report under `docs/security/`.
- [x] Apply safe `npm audit fix` only.
- [x] Save post-fix npm audit report under `docs/security/`.
- [x] Document remaining dependency hardening plan.
- [x] Validate sanitize, build, smoke, Python compile, and lightweight model/manifest tests.
- [ ] Upgrade Next.js major version if required by audit.
- [ ] Upgrade LangChain package family if required by audit.
- [ ] Review Vercel/dev tooling transitive vulnerabilities.

Acceptance criteria:

- [x] Critical npm audit findings reduced to zero without `npm audit fix --force`.
- [ ] Remaining high/moderate findings are resolved or formally risk accepted.

### Phase A — Close Production Cutover and Release Gate

**Goal:** Finish the already-started deployment hardening track and reach a stable deployed baseline.

**Priority:** Critical  
**Recommended branch:** `release/production-cutover` or continue from current active branch if already clean.

#### A.1 Public HTTP bridge endpoint

Status: `In Progress` / `Blocked`

Tasks:

- [ ] Provision a real public HTTPS endpoint for the aviation HTTP bridge.
- [ ] Verify `GET /health` from outside the local network.
- [ ] Verify `POST /command` with `{ "action": "ping" }`.
- [ ] Run `npm run bridge:check` against the public endpoint.
- [ ] Set Vercel `AVIATION_API_MODE=http`.
- [ ] Set Vercel `AVIATION_API_HTTP_URL` to the real public bridge URL.
- [ ] Set Vercel `AVIATION_API_HTTP_TOKEN`.
- [ ] Redeploy Vercel.
- [ ] Validate `/api/health?deep=1` from production.

Acceptance criteria:

- [ ] Production health endpoint reports `bridge_mode: "http"`.
- [ ] Production deep health reports successful aviation HTTP ping.
- [ ] Chat works from Vercel without `fetch failed`.

#### A.2 Release gate checklist

Status: `In Progress` / `Blocked`

Tasks:

- [ ] Confirm secrets are rotated after any local/public exposure risk.
- [ ] Confirm `.env` and secure bundles are not tracked.
- [ ] Confirm gitleaks CI passes.
- [ ] Confirm `npm run build` passes.
- [ ] Confirm `npm run test:smoke` passes.
- [ ] Confirm local bridge mode still works.
- [ ] Confirm HTTP bridge mode works.
- [ ] Confirm upload flow works with a small PDF/DOCX.
- [ ] Confirm history/session retrieval works.
- [ ] Document known limitations before release.

Acceptance criteria:

- [x] A single release checklist file exists under `docs/`.
- [ ] Every release gate item is checked or explicitly waived with reason.

---

### Phase B — Repository and Baseline Consolidation

**Goal:** Make the project reproducible, safe, and easy for Codex/VSCode work.

**Priority:** High

#### B.1 Dependency reproducibility

Status: `Planned`

Tasks:

- [ ] Pin Python dependencies.
- [ ] Create `requirements.in` and generated `requirements.txt`, or move to `pyproject.toml`.
- [ ] Confirm supported Python version: recommend Python `3.11` or `3.12`, not both as ambiguous runtime targets.
- [ ] Add `.python-version` if using pyenv-compatible workflow.
- [ ] Confirm Node version policy.
- [ ] Avoid declaring `node` as a normal runtime dependency unless truly required.
- [ ] Commit lockfiles.

Acceptance criteria:

- [ ] A fresh clone can install dependencies deterministically.
- [ ] README setup matches real dependency workflow.

#### B.2 Git history and large file policy

Status: `Partial`

Tasks:

- [ ] Run Git history analysis for large generated blobs.
- [ ] Decide whether to purge historical generated artifacts.
- [ ] Decide whether real test PDFs remain in public repository or move to private storage.
- [ ] Define Git LFS policy for large aviation documents if they must stay versioned.
- [ ] Add `docs/DATA_GOVERNANCE.md`.

Acceptance criteria:

- [ ] Clear policy exists for source documents, generated chunks, embeddings, and logs.
- [ ] No generated embeddings or private/sensitive documents are unintentionally tracked.

#### B.3 Documentation baseline

Status: `In Progress` / `Baseline docs added`

Tasks:

- [x] Add or update `docs/ARCHITECTURE.md`.
- [ ] Add or update `docs/ROADMAP.md`.
- [x] Add `docs/DATA_GOVERNANCE.md`.
- [x] Add `docs/EVALUATION_PLAN.md`.
- [x] Add `docs/RESPONSE_POLICY.md`.

Acceptance criteria:

- [ ] A new developer can understand system architecture, data flow, deployment modes, and current limitations from docs alone.

---

### Phase C — Backend Architecture Refactor Without Behavior Change

**Goal:** Convert script-based prototype logic into maintainable backend modules while preserving current behavior.

**Priority:** High

#### C.1 Create backend package structure

Status: `In Progress` / `Package skeleton added`

Target structure:

```text
src/aviationrag/
  __init__.py
  config.py
  ingestion/
    __init__.py
    readers.py
    chunking.py
    manifest.py
  retrieval/
    __init__.py
    faiss_store.py
    hybrid.py
    filters.py
  generation/
    __init__.py
    answer_service.py
    prompts.py
    response_policy.py
    citation_validator.py
  storage/
    __init__.py
    astra.py
    chat_store.py
    document_store.py
  evaluation/
    __init__.py
    retrieval_eval.py
    answer_eval.py
  api/
    __init__.py
  utils/
    __init__.py
```

Tasks:

- [x] Create package skeleton.
- [ ] Move shared config into package.
- [ ] Keep old script paths as wrappers.
- [ ] Avoid breaking `aviationrag_manager.py`.
- [ ] Avoid breaking `aviationai.py` CLI.
- [ ] Add type hints to newly moved functions.

Acceptance criteria:

- [ ] Existing CLI and manager still run.
- [ ] Tests/build still pass.
- [ ] No major behavioral change introduced in this phase.

#### C.2 Define core data models

Status: `Dataclass anchors added` / `Runtime adoption planned`

Recommended models:

```python
@dataclass
class DocumentRecord:
    document_id: str
    filename: str
    title: str | None
    authority: str | None
    document_type: str | None
    revision: str | None
    effective_date: str | None
    source_url: str | None
    file_hash: str
    ingestion_status: str

@dataclass
class ChunkRecord:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    page_start: int | None
    page_end: int | None
    section_path: list[str]
    metadata: dict

@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float
    source: str
    metadata: dict
```

Tasks:

- [x] Add lightweight dataclasses or Pydantic models.
- [ ] Use models inside new modules first.
- [ ] Keep compatibility adapters for existing JSON/PKL format.

Acceptance criteria:

- [ ] New code uses explicit data structures instead of untyped dictionaries where practical.

---

### Phase D — Ingestion, Parsing, and Metadata Upgrade

**Goal:** Make documents traceable and suitable for compliance-grade retrieval.

**Priority:** High

#### D.1 Document manifest

Status: `Schema planning complete` / `Implementation planned`

Tasks:

- [x] Define document manifest and metadata schema in `docs/DOCUMENT_MANIFEST_SCHEMA.md`.
- [x] Add fake sample manifest fixture under `data/sample_documents/`.
- [x] Add safe JSONL manifest writer/reader utilities using fake/sample test data only.
- [x] Add legacy compatibility adapter design using fake legacy-like test records only.
- [x] Add controlled ingestion integration plan and reset strategy without runtime integration.
- [x] Add fake-data dry-run integration coverage without runtime ingestion integration.
- [x] Add local-only manifest write dry run using fake sample records and ignored output path.
- [x] Add gated manifest integration settings disabled by default.
- [x] Define reset/rebuild and retrieval evaluation baseline gates before real migration.
- [ ] Create `data/manifest/documents.jsonl` or database-backed equivalent.
- [ ] Assign stable `document_id` for every source document.
- [ ] Store file hash.
- [ ] Store filename, title, authority, revision, effective date, document type, and ingestion date.
- [ ] Store extraction method and quality.
- [ ] Store `needs_manual_review` flag.

Acceptance criteria:

- [ ] Every chunk can be traced back to a document record and file hash.

#### D.2 Metadata-rich chunk schema planning

Status: `Completed` / `Real chunk migration planned`

Tasks:

- [x] Define metadata-rich chunk schema in `docs/CHUNK_METADATA_SCHEMA.md`.
- [x] Define chunk identity, traceability, citation, retrieval payload, and evaluation alignment requirements.
- [x] Define future vector database and FAISS metadata alignment requirements.
- [x] Expand fake chunk fixtures with metadata-rich synthetic chunk records.
- [x] Add chunk schema validator.
- [x] Add fake chunk payload exporter.
- [x] Add gated legacy chunk adapter.
- [ ] Implement real chunk migration.
- [ ] Re-embed and re-index after reset gate approval.

Acceptance criteria:

- [x] Chunk metadata contract is documented without changing runtime chunking behavior.
- [ ] Real chunks carry stable `document_id`, `chunk_id`, page/section metadata, and extraction quality after future migration.

#### D.3 Real chunk migration design

Status: `Design complete` / `Real migration execution planned`

Tasks:

- [x] Define real chunk migration path in `docs/REAL_CHUNK_MIGRATION_DESIGN.md`.
- [x] Define legacy-to-new mapping strategy, chunk ID policy, evaluation gates, reset dependency, rollout stages, rollback strategy, and go/no-go criteria.
- [x] Add future operational go/no-go checklist for chunk migration.
- [x] Audit legacy chunk format with a read-only script using fake/default fixture and explicit-file support.
- [ ] Run fake/local chunk migration dry run.
- [ ] Implement gated local chunk conversion writing ignored outputs only.
- [ ] Preserve page numbers for PDF extraction in a future real parsing phase.
- [ ] Detect headings and section paths where possible in a future real parsing phase.
- [ ] Separate tables into dedicated table chunks in a future real parsing phase.
- [ ] Preserve notes, cautions, warnings, and definitions as distinct chunk candidates in a future real parsing phase.
- [ ] Keep fallback extraction for difficult PDFs.
- [ ] Keep manual review flag for low-quality extraction.

Acceptance criteria:

- [x] Real chunk migration design and go/no-go criteria are documented without changing runtime chunking behavior.
- [ ] Real migration execution is approved, implemented, and validated in a future phase.
- [ ] A retrieved chunk can show document, page span, and section path.
- [ ] Low-quality extraction is visible to admin/user.

#### D.4 Chunking redesign

Status: `Planned`

Tasks:

- [ ] Add section-aware chunking.
- [ ] Add configurable chunk size by document type.
- [ ] Add special handling for regulatory paragraphs.
- [ ] Add special handling for tables.
- [ ] Add exact citation passage chunks for compliance mode.

Acceptance criteria:

- [ ] Regulatory queries retrieve exact paragraph-like chunks rather than random long text blocks.

---

### Phase E — Retrieval Quality Upgrade

**Goal:** Improve evidence retrieval before improving answer style.

**Priority:** Critical for aviation/compliance usefulness

#### E.0 Retrieval evaluation smoke fixture baseline

Status: `Completed`

Tasks:

- [x] Add fake/sample-only retrieval evaluation JSONL fixture.
- [x] Add smoke fixture validation utilities.
- [x] Add fixture validation tests.
- [x] Connect evaluation cases to a fake/mock retrieval harness shell.
- [ ] Run real retrieval benchmark execution.

Acceptance criteria:

- [x] Smoke fixture format is validated without running retrieval.
- [ ] Retrieval quality is measured against real retrieval results.

#### E.1 Retrieval evaluation harness shell

Status: `Completed` / `Real retrieval integration planned`

Tasks:

- [x] Add fake/mock retrieval result evaluator.
- [x] Evaluate expected document match, expected chunk match, top-k/rank, insufficient evidence, and out-of-scope rejection behavior.
- [x] Add tests using fake/mock data only.
- [ ] Wire harness to real FAISS/Astra/hybrid retrieval outputs.
- [ ] Run real retrieval benchmark execution.

Acceptance criteria:

- [x] Fake/mock retrieval results can be evaluated without running retrieval.
- [ ] Real retrieval quality is measured against real retrieval results.

#### E.2 Retrieval evaluation report/export shell

Status: `Completed` / `Real retrieval integration planned`

Tasks:

- [x] Add JSON-serializable report dictionaries for evaluation summaries and case results.
- [x] Add Markdown report rendering for fake/mock evaluation results.
- [x] Add explicit JSON and Markdown write helpers.
- [x] Add tests using fake/mock data only.
- [ ] Wire reporting to real retrieval benchmark runs.
- [ ] Publish or archive real benchmark reports only under a future data-governance decision.

Acceptance criteria:

- [x] Fake/mock evaluation results can be exported without running retrieval.
- [ ] Real retrieval benchmark reports are generated after real retrieval integration exists.

#### E.3 Hybrid retrieval

Status: `Planned`

Tasks:

- [ ] Keep current semantic/vector retrieval.
- [ ] Add lexical/BM25 retrieval branch.
- [ ] Add metadata filters for authority, document type, status, revision, and date.
- [ ] Merge dense and lexical results.
- [ ] Add deterministic boosts for regulatory identifiers such as `14 CFR`, `Part 23`, `CS-25`, `AC`, `AMC`, `GM`, `FAA Order`, etc.
- [ ] Add source-type filtering for accident, regulation, internal procedure, design, manufacturing, and SMS queries.

Acceptance criteria:

- [ ] Exact regulatory identifiers reliably retrieve exact matching source passages.
- [ ] General conceptual questions do not over-prioritize irrelevant accident reports.

#### E.4 Reranking rules

Status: `Planned`

Tasks:

- [ ] Add lightweight deterministic reranker first.
- [ ] Prioritize exact title/document matches for quoted document requests.
- [ ] Prioritize exact paragraph/section/identifier matches.
- [ ] Penalize low-quality extraction chunks unless no better evidence exists.
- [ ] Consider LLM reranking only after deterministic rules and evaluation harness exist.

Acceptance criteria:

- [ ] Top 5 retrieved chunks are explainable and stable for benchmark questions.

#### E.5 Retrieval evaluation harness integration

Status: `Planned`

Tasks:

- [ ] Create `tests/evaluation/retrieval_questions.jsonl`.
- [ ] Include expected documents and expected section/page where possible.
- [ ] Add script to evaluate top-1, top-3, top-5 hit rate.
- [ ] Add category tags: regulatory, compliance, design, manufacturing, accident, SMS, general, not_found.
- [ ] Generate a Markdown report.

Acceptance criteria:

- [ ] Retrieval quality is measurable before and after changes.
- [ ] No retrieval upgrade is accepted without benchmark comparison.

---

### Phase F — Response Policy, Citation Validation, and Answer Modes

**Goal:** Make answer behavior safe for aircraft engineering and compliance-support use.

**Priority:** Critical

#### F.1 Formal response modes

Status: `Planned`

Recommended modes:

| Mode | Use case |
| --- | --- |
| `general` | Broad aviation explanation. |
| `strict_document` | User asks according to a specific document. |
| `regulatory_compliance` | Regulations, certification, airworthiness, compliance evidence. |
| `design_review` | Engineering design/manufacturing support. |
| `manufacturing_quality` | Production, conformity, process, MRB, quality issues. |
| `sms_safety` | Safety management, hazards, HFACS, Dirty Dozen, Reason model. |
| `accident_analysis` | Accident/incident report analysis. |
| `insufficient_evidence` | Evidence missing from provided sources. |

Tasks:

- [ ] Implement mode classifier.
- [ ] Allow manual mode override from API/UI.
- [ ] Store selected mode in audit logs.
- [ ] Add mode-specific prompt templates.

Acceptance criteria:

- [ ] Regulatory/compliance questions do not use the same generic answer style as general questions.

#### F.2 Citation validation

Status: `Planned`

Tasks:

- [ ] Validate that every returned citation exists in retrieved context.
- [ ] Require citations for strict/compliance factual claims.
- [ ] Add validator output:
  - `valid_citations`
  - `missing_citations`
  - `unsupported_claim_warning`
- [ ] Add fallback behavior when validation fails.

Acceptance criteria:

- [ ] Strict/compliance answer cannot silently return unsupported hard claims.

#### F.3 Evidence-level output

Status: `Planned`

Recommended output schema:

```json
{
  "answer": "...",
  "mode": "regulatory_compliance",
  "evidence_level": "high | medium | low | not_found",
  "citations": [],
  "sources": [],
  "warnings": [],
  "model": "...",
  "prompt_version": "...",
  "latency_ms": 0
}
```

Tasks:

- [ ] Add `evidence_level` to backend answer result.
- [ ] Show evidence level in UI.
- [ ] Add warning when source extraction quality is low.
- [ ] Add explicit statement for advisory interpretation.

Acceptance criteria:

- [ ] User can distinguish exact document evidence from engineering interpretation.

---

### Phase G — Evaluation and Regression Testing

**Goal:** Prevent quality regressions and hallucination-prone behavior.

**Priority:** High

#### G.1 Benchmark datasets

Status: `Planned`

Create benchmark sets:

- [ ] `retrieval_regulatory.jsonl`
- [ ] `retrieval_document_specific.jsonl`
- [ ] `answer_compliance.jsonl`
- [ ] `answer_not_found.jsonl`
- [ ] `answer_sms_accident.jsonl`

Minimum initial benchmark size:

| Category | Initial target |
| --- | ---: |
| Regulatory/compliance | 25 |
| Document-specific | 25 |
| General aviation | 20 |
| Manufacturing/design | 20 |
| SMS/accident | 20 |
| Not found / insufficient evidence | 20 |

Acceptance criteria:

- [ ] Evaluation can be run locally with one command.
- [ ] Results are saved as Markdown/JSON.

#### G.2 Automated tests

Status: `Partial`

Tasks:

- [ ] Unit test ingestion manifest creation.
- [ ] Unit test chunk metadata.
- [ ] Unit test retrieval filters.
- [ ] Unit test citation validator.
- [ ] Unit test response mode classifier.
- [ ] Integration test `/api/chat/ask` with mock backend response.
- [ ] Integration test upload status lifecycle.

Acceptance criteria:

- [ ] CI runs fast tests on every push.
- [ ] RAG evaluation can run manually or nightly, not necessarily on every PR.

---

### Phase H — Audit Trail and Governance

**Goal:** Support traceability expected in engineering/compliance environments.

**Priority:** High

#### H.1 Chat audit log

Status: `Planned`

Tasks:

- [ ] Store question.
- [ ] Store user/session.
- [ ] Store selected mode.
- [ ] Store retrieved chunk IDs and scores.
- [ ] Store answer.
- [ ] Store citations.
- [ ] Store model and prompt version.
- [ ] Store token usage if available.
- [ ] Store latency.
- [ ] Store timestamp.

Acceptance criteria:

- [ ] Any answer can be reconstructed with its evidence package.

#### H.2 Document lifecycle governance

Status: `Planned`

Tasks:

- [ ] Add document states: `uploaded`, `processing`, `embedded`, `available`, `needs_review`, `retired`, `error`.
- [ ] Add admin approval before `available` state.
- [ ] Add document retirement/deactivation.
- [ ] Add source version display in UI.
- [ ] Add warning for obsolete/superseded documents.

Acceptance criteria:

- [ ] Newly uploaded documents are not automatically trusted unless configured and approved.

#### H.3 Human feedback loop

Status: `Planned`

Tasks:

- [ ] Add thumbs up/down or quality rating.
- [ ] Add wrong citation flag.
- [ ] Add hallucination flag.
- [ ] Add obsolete source flag.
- [ ] Store feedback against answer ID.

Acceptance criteria:

- [ ] User feedback can feed evaluation and prioritization.

---

### Phase I — Web UX Refinement

**Goal:** Make the app practical for aviation engineers and admins.

**Priority:** Medium after core RAG quality

#### I.1 Source viewer panel

Status: `Partial / Planned`

Tasks:

- [ ] Show cited source text.
- [ ] Show document title, filename, page, section, and chunk ID.
- [ ] Allow user to expand/collapse sources.
- [ ] Highlight quoted/cited passage.
- [ ] Show extraction quality warning.

Acceptance criteria:

- [ ] Engineer can verify the answer against source text without leaving chat.

#### I.2 Admin document page

Status: `Planned`

Tasks:

- [ ] List uploaded documents.
- [ ] Show ingestion status.
- [ ] Show metadata.
- [ ] Show approval state.
- [ ] Allow deactivate/retire.
- [ ] Show errors and manual review warnings.

Acceptance criteria:

- [ ] Admin can control active knowledge base without editing files manually.

#### I.3 Settings and mode controls

Status: `Planned`

Tasks:

- [ ] Strict mode toggle.
- [ ] Response mode selector.
- [ ] Source type filters.
- [ ] Model selector if allowed.
- [ ] Show active bridge mode and backend health.

Acceptance criteria:

- [ ] User can intentionally choose strict compliance behavior when needed.

---

### Phase J — Optional Web Research Mode

**Goal:** Add external research only after internal RAG is reliable.

**Priority:** Low / Future

Rules:

- [ ] Off by default.
- [ ] Explicit user toggle required.
- [ ] Internal document citations and web citations must be visually separated.
- [ ] Web source date must be displayed.
- [ ] Web result cannot override official uploaded source documents unless clearly explained.
- [ ] External research output must include disclaimer when source is unofficial.

Acceptance criteria:

- [ ] No silent mixing of internal controlled sources and external web sources.

---

## 5. Immediate Next Actions for Codex

Use this exact order.

### Next Action 1 — Finish bridge cutover validation

```text
Read WORKLOG.md, docs/VERCEL_ONLINE_SETUP.md, docs/RUNBOOK_VERCEL_LOCAL_SYNC.md, and docs/AVIATION_API_HTTP_BRIDGE_CUTOVER_CHECKLIST.md.
Continue from the Vercel bridge reachability blocker.
Do not refactor RAG code yet.
Goal: make production /api/health?deep=1 report successful HTTP bridge ping.
```

### Next Action 2 — Add missing architecture/governance docs

```text
Create or update docs/ARCHITECTURE.md, docs/DATA_GOVERNANCE.md, docs/EVALUATION_PLAN.md, and docs/RESPONSE_POLICY.md.
Base the docs on README.md, WORKLOG.md, and PLANWORKLOG.md.
Do not change runtime behavior.
```

### Next Action 3 — Backend package skeleton only

```text
Create src/aviationrag package skeleton with ingestion, retrieval, generation, storage, evaluation, and utils modules.
Do not move major logic yet.
Add __init__.py files and minimal README/comments explaining intended module ownership.
Ensure npm build and current Python checks are unaffected.
```

### Next Action 4 — Retrieval evaluation harness

```text
Create the first retrieval evaluation harness without changing production retrieval.
Add tests/evaluation/retrieval_questions.sample.jsonl and a script that loads questions, runs retrieval, and reports top-1/top-3/top-5 hits.
Use a small sample set first.
Do not call the LLM in this evaluation.
```

### Next Action 5 — Response policy module

```text
Add a response policy module with response modes and output schema.
Do not force all chat paths to use it yet.
Add unit tests for mode classification and citation validation.
```

---

## 6. Definition of Done by Milestone

### Milestone 1 — Production baseline

- [ ] Vercel app can reach public aviation HTTP bridge.
- [ ] `/api/health?deep=1` passes.
- [ ] Chat works in production.
- [ ] Upload still works locally or in intended deployment mode.
- [ ] Release gate checklist complete.

### Milestone 2 — Maintainable architecture baseline

- [ ] Backend package skeleton exists.
- [ ] Architecture/data governance/evaluation/response policy docs exist.
- [ ] Existing scripts still work.
- [ ] CI still passes.

### Milestone 3 — RAG quality baseline

- [ ] Document manifest exists.
- [ ] Chunks include richer metadata.
- [ ] Retrieval evaluation harness exists.
- [ ] Initial benchmark set exists.
- [ ] Current retrieval score is recorded as baseline.

### Milestone 4 — Compliance behavior baseline

- [ ] Response modes exist.
- [ ] Citation validator exists.
- [ ] Strict/compliance mode requires citations.
- [ ] Insufficient-evidence behavior is tested.
- [ ] Evidence level appears in API response.

### Milestone 5 — Engineer-ready internal MVP

- [ ] Source viewer displays source text and metadata.
- [ ] Admin can see document status.
- [ ] Audit log captures answer evidence package.
- [ ] Feedback mechanism exists.
- [ ] Known limitations are documented.

---

## 7. Risks and Controls

| Risk | Impact | Control |
| --- | ---: | --- |
| Public repo contains large or sensitive aviation documents | High | Data governance, history scan, private storage/Git LFS policy. |
| RAG retrieves wrong source but answer sounds confident | Critical | Retrieval evaluation, citation validation, evidence level, strict not-found behavior. |
| Compliance answer lacks exact source support | Critical | Regulatory mode must require citation and quote exact wording when possible. |
| Deployment bridge unreachable | High | Public HTTPS bridge preflight and deep health check. |
| Dependency drift breaks app | Medium | Pin dependencies and document supported versions. |
| Upload automatically trusts poor-quality extraction | High | Extraction quality, manual review flag, admin approval. |
| Mixed internal/web sources confuse users | Medium | Keep web research deferred and clearly separated later. |

---

## 8. Working Rules for Codex

1. Prefer small commits.
2. Do not mix deployment fixes with RAG refactors.
3. Do not change runtime behavior during architecture-only refactors.
4. Add tests before changing retrieval behavior.
5. Preserve backward compatibility with current data files until migration is explicit.
6. Do not commit secrets, secure bundles, generated embeddings, logs, or private documents unless policy explicitly allows.
7. Update this file after each significant milestone.
8. Use `WORKLOG.md` as the chronological execution log.
9. Use `PLANWORKLOG.md` as the strategic plan and checklist.
10. If a chat/IDE session is interrupted, read `WORKLOG.md`, `PLANWORKLOG.md`, and `git status -b --short` before continuing.

---

## 9. Recommended Branching

| Work type | Branch name example |
| --- | --- |
| Production cutover | `release/production-cutover` |
| Documentation baseline | `docs/architecture-governance` |
| Backend skeleton | `refactor/backend-package-skeleton` |
| Ingestion metadata | `feature/document-manifest` |
| Retrieval evaluation | `feature/retrieval-eval-harness` |
| Hybrid retrieval | `feature/hybrid-retrieval` |
| Response policy | `feature/response-policy` |
| Citation validation | `feature/citation-validation` |
| Admin document UX | `feature/admin-document-governance` |

---

## 10. Current Recommendation

Do **not** jump directly to hybrid retrieval or response policy until the production bridge blocker is closed or explicitly paused.

Recommended immediate order:

1. Close Vercel/public bridge cutover.
2. Complete release gate checklist.
3. Add architecture/governance/evaluation docs.
4. Add backend package skeleton without behavior change.
5. Add retrieval evaluation harness.
6. Then start parsing/chunking/metadata upgrade.

This sequence keeps the project stable while moving it toward the real goal: a trustworthy aviation engineering and compliance-support RAG system.
