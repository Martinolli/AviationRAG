# Ingestion

Future ownership:

1. PDF and DOCX readers.
2. Text extraction and fallback extraction strategies.
3. Extraction metadata and quality scoring.
4. Document manifests and stable document IDs.
5. Section/page metadata capture.
6. Chunking coordination after migration from legacy scripts.
7. Manual-review flags for low-quality extraction.

Current status: skeleton only. Runtime ingestion still lives in
`src/scripts/py_files/read_documents.py`, `aviation_chunk_saver.py`, and
`aviationrag_manager.py`.
