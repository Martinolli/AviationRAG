# Retrieval

Future ownership:

1. Vector retrieval over local FAISS indexes.
2. Astra DB retrieval interfaces where appropriate.
3. Metadata filters for authority, document type, revision, status, and dates.
4. Hybrid dense plus lexical retrieval.
5. Deterministic regulatory identifier matching and boosting.
6. Reranking logic and source-type prioritization.

Current status: skeleton only. Runtime retrieval still lives primarily in
`src/scripts/py_files/aviationai.py` and `faiss_indexer.py`.
