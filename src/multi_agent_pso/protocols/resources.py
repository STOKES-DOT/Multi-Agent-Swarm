"""Resource-gating and read-only wiki retrieval ports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath
from typing import AsyncContextManager, Protocol, runtime_checkable

from .agent_runtime import _require_nonempty


@runtime_checkable
class ResourceManager(Protocol):
    """Separately rate-limit agent calls and evaluation work."""

    def agent_slot(self) -> AsyncContextManager[None]: ...

    def evaluation_slot(self) -> AsyncContextManager[None]: ...


@dataclass(frozen=True, slots=True)
class WikiQuery:
    text: str
    max_results: int

    def __post_init__(self) -> None:
        _require_nonempty(self.text, "text")
        if not 1 <= self.max_results <= 100:
            raise ValueError("max_results must be between 1 and 100")

    def to_json(self) -> dict[str, object]:
        return {"text": self.text, "max_results": self.max_results}


@dataclass(frozen=True, slots=True)
class WikiHit:
    relative_path: str
    line_start: int
    line_end: int
    evidence_layer: str
    content: str

    def __post_init__(self) -> None:
        _require_nonempty(self.relative_path, "relative_path")
        if PurePath(self.relative_path).is_absolute():
            raise ValueError("relative_path must be relative")
        if self.line_start < 1:
            raise ValueError("line_start must be at least 1")
        if self.line_end < self.line_start:
            raise ValueError("line_end must not precede line_start")
        _require_nonempty(self.evidence_layer, "evidence_layer")

    def to_json(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "evidence_layer": self.evidence_layer,
            "content": self.content,
        }


@runtime_checkable
class WikiRetriever(Protocol):
    """Synchronous, read-only retrieval from a local evidence base."""

    def search(self, query: WikiQuery) -> tuple[WikiHit, ...]: ...


__all__ = ["ResourceManager", "WikiHit", "WikiQuery", "WikiRetriever"]
