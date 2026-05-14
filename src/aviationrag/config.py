"""Future centralized configuration module for AviationRAG.

Current runtime configuration still depends on the legacy script-based config
in ``src/scripts/py_files/config.py``. This module exists only as the first
migration anchor and is intentionally not imported by runtime code yet.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping


DEFAULT_MANIFEST_PATH = Path("data/manifest/documents.jsonl")
MANIFEST_INTEGRATION_ENV = "AVIATIONRAG_ENABLE_MANIFEST_INTEGRATION"
MANIFEST_DRY_RUN_ENV = "AVIATIONRAG_MANIFEST_DRY_RUN"
MANIFEST_PATH_ENV = "AVIATIONRAG_MANIFEST_PATH"

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


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


@dataclass(frozen=True)
class ManifestIntegrationSettings:
    """Disabled-by-default settings for future manifest-aware ingestion.

    These settings are intentionally not wired into legacy ingestion scripts
    yet. Future gated integration should read this object before writing any
    local/private manifest data.
    """

    enabled: bool
    dry_run: bool
    manifest_path: Path


def _get_env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def parse_bool_env(value: str | None, default: bool = False) -> bool:
    """Parse a permissive boolean environment value.

    Unknown values fall back to ``default`` so malformed optional flags do not
    crash imports or unrelated runtime paths.
    """

    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def get_manifest_path(env: Mapping[str, str] | None = None) -> Path:
    """Return the future local/private manifest path.

    The default path is intentionally relative and ignored by Git. This helper
    performs no filesystem reads or writes.
    """

    active_env = _get_env(env)
    value = active_env.get(MANIFEST_PATH_ENV, "").strip()
    return Path(value) if value else DEFAULT_MANIFEST_PATH


def is_manifest_integration_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether future manifest integration is explicitly enabled."""

    active_env = _get_env(env)
    return parse_bool_env(active_env.get(MANIFEST_INTEGRATION_ENV), default=False)


def is_manifest_dry_run_enabled(env: Mapping[str, str] | None = None) -> bool:
    """Return whether future manifest integration should run in dry-run mode."""

    active_env = _get_env(env)
    return parse_bool_env(active_env.get(MANIFEST_DRY_RUN_ENV), default=False)


def get_manifest_integration_settings(
    env: Mapping[str, str] | None = None,
) -> ManifestIntegrationSettings:
    """Build side-effect-free manifest integration settings."""

    active_env = _get_env(env)
    return ManifestIntegrationSettings(
        enabled=is_manifest_integration_enabled(active_env),
        dry_run=is_manifest_dry_run_enabled(active_env),
        manifest_path=get_manifest_path(active_env),
    )


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "MANIFEST_DRY_RUN_ENV",
    "MANIFEST_INTEGRATION_ENV",
    "MANIFEST_PATH_ENV",
    "ManifestIntegrationSettings",
    "ProjectPaths",
    "get_manifest_integration_settings",
    "get_manifest_path",
    "is_manifest_dry_run_enabled",
    "is_manifest_integration_enabled",
    "parse_bool_env",
]
