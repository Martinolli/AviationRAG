"""Offline migration rehearsal helpers.

The migration package is intentionally disconnected from runtime ingestion,
embeddings, Astra, FAISS, and production retrieval.
"""

__all__ = ["shadow_migration_rehearsal"]
