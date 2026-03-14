# AviationRAG – Performance and Logic Review

Summary of quick, low-touch refactors to improve responsiveness and stability, plus logical issues observed while scanning the current codebase (line numbers captured from commit `865e714`, March 2026). File paths are relative to the repo root.

## Performance-focused refactors

1) Defer heavy FAISS + OpenAI setup  
   * File: `src/scripts/py_files/aviationai.py` (lines 68–83).  
   * Issue: Embeddings are loaded and the FAISS index is instantiated at import time, so every CLI entrypoint and the Python worker pay the full load cost even for lightweight commands (e.g., listing sessions). If the index is missing/corrupt, the module exits during import.  
   * Refactor: Lazy-load the FAISS index and OpenAI client behind an accessor with a warmup hook; guard failures so non-RAG commands can still run and emit structured errors instead of hard exits.

2) Avoid per-request Cassandra client + Node process churn for chat history  
   * Files: `src/scripts/py_files/chat_db.py` (store/retrieve/delete functions) and `src/scripts/js_files/store_chat.js`.  
   * Issue: Each chat store/retrieve spins up a new Node process, which then creates a new Cassandra client and tears it down (`client.connect()`/`shutdown()` every call). This adds process startup latency, drops connection pooling, and inflates memory/CPU under load.  
   * Refactor: Move chat persistence to a long-lived service (either stay in Python with the Cassandra driver, or keep a singleton Node worker/HTTP microservice) and reuse a pooled `Client` instance. This eliminates cross-process serialization and repeated TLS handshakes.

3) HTTP bridge timeouts and request fan-out  
   * File: `src/utils/server/aviation_api_bridge.ts` (lines 149–198, 200–231).  
   * Issue: The HTTP mode uses a single `fetch` with a fixed timeout and no retry/backoff; slow document ingestion or cold starts can easily hit the default 180s timeout (from `AVIATION_API_TIMEOUT_MS`, default 180000 ms) with no partial progress.  
   * Refactor: Add short connection/read timeouts with limited retries, and stream responses when available. For the worker mode, consider a readiness probe and bounded queue to avoid unbounded `pendingRequests` growth during spikes.

4) Document ingestion / raw source extraction caching  
   * File: `src/scripts/py_files/aviationai.py` (functions `extract_raw_text_from_source_file`, `split_into_passages`).  
   * Issue: Raw PDF/DOCX text is read per-request unless present in the local `RAW_SOURCE_CACHE`, but cache is process-local and rebuilt every worker restart; large PDFs repeatedly pay the parsing cost.  
   * Refactor: Persist a lightweight on-disk cache (e.g., JSON or SQLite keyed by file hash/mtime) and cap `pdfplumber`/`docx` parsing with timeouts to avoid tail latency. Consider pre-chunking during ingestion and using those chunks directly instead of re-reading sources.

5) Streaming + chunk limits for OpenAI calls  
   * File: `src/scripts/py_files/aviationai.py` (functions `build_retrieval_context`, `generate_response`).  
   * Issue: The retrieval stage can accumulate many context entries until the token budget is met, then `generate_response` sends the full concatenated context to the model in one shot. Large contexts increase latency and cost.  
   * Refactor: Stream responses from OpenAI, enforce stricter per-file chunk caps, and short-circuit once a high-quality subset is found (e.g., stop after top-N high overlap chunks). This reduces prompt size and improves first-token latency.

## Logical issues found

1) API responses ignore worker errors on session endpoints  
   * File: `pages/api/chat/session.ts` (lines 27–62).  
   * Problem: Both GET and POST paths return `200/201` with the worker payload even when `success` is `false`, so client code may treat failed worker calls as successful.  
   * Fix: Check `result.success === true`; otherwise return `502/500` with the error to signal failure.

2) Health endpoint reports success even when bridge checks fail  
   * File: `pages/api/health.ts` (lines 137–153).  
   * Problem: The endpoint always returns `success: true` with HTTP 200 even if required env vars or the HTTP bridge ping fail; `deep_check_error` is optional, so monitoring can miss outages.  
   * Fix: Set `success` based on aggregated checks and return non-200 (or at least `success: false`) when critical checks fail.

3) Boolean flags can flip in worker when HTTP callers send strings  
   * File: `src/scripts/py_files/aviationai_worker.py` (lines 54–69).  
   * Problem: `store` and `strict_mode` are read directly from the payload and passed through `bool(...)`; when HTTP callers send `"false"` as a string, `bool("false")` becomes `true`, causing unintended storage and strict-mode behavior.  
   * Fix: Normalize booleans with a dedicated parser (mirroring `parseBoolean` in `aviation_api_bridge.ts`) before use.

4) Cassandra query uses string-interpolated LIMIT  
   * File: `src/scripts/js_files/store_chat.js` (lines 102–118).  
   * Problem: The LIMIT value is interpolated into the query string. Although it is clamped, it still bypasses prepared statements and prevents query plan reuse.  
   * Fix: Use a prepared statement with a bound limit or move the limit logic into application code after fetching a bounded number of rows via paging.
