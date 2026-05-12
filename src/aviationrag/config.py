"""Future centralized configuration module for AviationRAG.

Current runtime configuration still depends on the legacy script-based config
in ``src/scripts/py_files/config.py``. This module exists only as the first
migration anchor and is intentionally not imported by runtime code yet.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Placeholder path container for future package-based configuration.

    PLANWORKLOG.md Phase C is expected to migrate shared configuration here
    gradually. Keep this dataclass side-effect free until runtime callers are
    intentionally moved from the legacy scripts.
    """

    project_root: Path | None = None
    data_dir: Path | None = None
    documents_dir: Path | None = None
    raw_dir: Path | None = None
    processed_dir: Path | None = None
    embeddings_dir: Path | None = None
    logs_dir: Path | None = None


__all__ = ["ProjectPaths"]
