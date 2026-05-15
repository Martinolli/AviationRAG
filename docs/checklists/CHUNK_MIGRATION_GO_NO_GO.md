# Chunk Migration Go/No-Go Checklist

Date: 2026-05-15
Status: Operational checklist for future use only

Use this checklist before any future real metadata-rich chunk migration, re-chunking, re-embedding, FAISS rebuild, or Astra reset.

## Preconditions

- [ ] Git working tree is clean.
- [ ] Latest `main` is pulled.
- [ ] Real source documents are backed up outside Git.
- [ ] Current generated artifacts are backed up if rollback is needed.
- [ ] Manifest records exist for all candidate documents.
- [ ] Every candidate document has a stable `document_id`.
- [ ] Every candidate document has a source hash or file hash.
- [ ] Chunk schema, payload, and adapter tests pass.
- [ ] Legacy chunk format audit has been completed read-only.
- [ ] Reset/rebuild plan has been reviewed.
- [ ] User/operator has approved any destructive reset or rebuild window.

## Migration Readiness

- [ ] Target chunk metadata schema is versioned for this migration.
- [ ] Chunk ID policy is selected and documented.
- [ ] Page/section/paragraph traceability gaps are known.
- [ ] Table, figure, warning, caution, and note handling rules are accepted.
- [ ] Extraction quality and manual-review rules are accepted.
- [ ] Vector payload shape is validated with fake/sample data.
- [ ] Chunk migration flags remain disabled until the approved run.
- [ ] Dry-run mode is enabled for first local trial.

## Evaluation Gate

- [ ] Fake/sample evaluation tests pass.
- [ ] Pre-migration real retrieval baseline has been run and saved locally.
- [ ] Expected document and chunk matching rules are defined.
- [ ] Insufficient-evidence cases are included.
- [ ] Post-migration evaluation command and report location are known.

## Data Governance

- [ ] No real source documents are staged.
- [ ] No generated chunks, embeddings, manifests, logs, or Astra exports are staged.
- [ ] No `.env` files or secure bundles are staged.
- [ ] `data/manifest/`, `data/processed/`, `data/embeddings/`, and `logs/` remain ignored.
- [ ] Any real evaluation outputs remain local/private.

## Decision

- [ ] Go
- [ ] No-Go

Decision notes:

```text

```
