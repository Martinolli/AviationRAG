# Storage

Future ownership:

1. Astra DB persistence interfaces.
2. Chat and session storage.
3. Document metadata and manifest persistence.
4. Audit logs for answer reconstruction.
5. Document lifecycle state such as uploaded, embedded, available, retired, or
   needs review.

Current status: skeleton only. Runtime storage still uses legacy Python and
Node helpers, including `src/scripts/py_files/chat_db.py` and
`src/scripts/js_files/store_chat.js`.
