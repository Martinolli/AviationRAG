# AviationRAG Backend Package

This package is the future home for maintainable AviationRAG backend modules.

Current status: package skeleton only. The active runtime still depends on the
legacy scripts in `src/scripts/py_files` and Node helpers in
`src/scripts/js_files`.

Backward compatibility rule:

1. Existing scripts remain the source of runtime behavior.
2. Existing CLI, ingestion manager, Next.js bridge, and HTTP bridge flows must
   continue to work during migration.
3. Future work should move logic into this package gradually, with legacy script
   paths kept as wrappers until an explicit migration is complete.

Planned ownership areas:

1. `ingestion/` for document reading, extraction, chunking, and manifests.
2. `retrieval/` for vector, metadata, hybrid retrieval, and reranking logic.
3. `generation/` for answer generation, prompts, response policy, and citations.
4. `storage/` for Astra, session, document metadata, and audit persistence.
5. `evaluation/` for benchmark and regression tooling.
6. `api/` for internal Python service interfaces and bridge contracts.
7. `utils/` for shared helpers.
