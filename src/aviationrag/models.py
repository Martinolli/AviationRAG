"""Lightweight core data models for future AviationRAG migration work.

These dataclasses are migration anchors for the planned backend package. The
active runtime still uses legacy dictionaries, JSON files, pickle files, and
script-local structures under ``src/scripts``.

No runtime migration has occurred yet. These models deliberately avoid heavy
runtime imports, validation frameworks, database logic, and serialization
dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, TypeVar


ModelT = TypeVar("ModelT")


def _from_dict(model_type: type[ModelT], data: dict[str, Any]) -> ModelT:
    """Create a model from known keys only, preserving lightweight behavior."""
    model_fields = {item.name for item in fields(model_type)}
    return model_type(**{key: value for key, value in data.items() if key in model_fields})


@dataclass
class DocumentRecord:
    """Future document manifest record."""

    document_id: str
    filename: str
    title: str | None = None
    authority: str | None = None
    document_type: str | None = None
    revision: str | None = None
    effective_date: str | None = None
    source_url: str | None = None
    file_hash: str | None = None
    ingestion_status: str | None = None
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentRecord":
        return _from_dict(cls, data)


@dataclass
class ChunkRecord:
    """Future normalized chunk record."""

    chunk_id: str
    document_id: str
    filename: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChunkRecord":
        return _from_dict(cls, data)


@dataclass
class RetrievedChunk:
    """Future retrieval result record."""

    chunk_id: str
    document_id: str
    filename: str
    text: str
    score: float | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RetrievedChunk":
        return _from_dict(cls, data)


@dataclass
class AnswerResult:
    """Future structured answer result."""

    answer: str
    mode: str | None = None
    evidence_level: str | None = None
    citations: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnswerResult":
        return _from_dict(cls, data)


__all__ = [
    "AnswerResult",
    "ChunkRecord",
    "DocumentRecord",
    "RetrievedChunk",
]
