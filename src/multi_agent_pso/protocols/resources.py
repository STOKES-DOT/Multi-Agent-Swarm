"""Resource-gating and read-only wiki retrieval ports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import AsyncContextManager, Protocol, runtime_checkable

from .agent_runtime import _require_nonempty, _require_nonnegative


_WIKI_EVIDENCE_LAYERS = {
    "direct evidence",
    "author interpretation",
    "cross-paper synthesis",
    "open hypothesis",
}


@runtime_checkable
class ResourceManager(Protocol):
    """Separately rate-limit agent calls and evaluation work."""

    def agent_slot(self) -> AsyncContextManager[None]: ...

    def evaluation_slot(self) -> AsyncContextManager[None]: ...


@dataclass(frozen=True, slots=True)
class WikiQuery:
    text: str
    max_results: int
    score_threshold: float = 0.0
    snippet_max_chars: int = 1200

    def __post_init__(self) -> None:
        _require_nonempty(self.text, "text")
        if not self.text.strip():
            raise ValueError("text must not be blank")
        _require_nonnegative(self.max_results, "max_results")
        if not 1 <= self.max_results <= 100:
            raise ValueError("max_results must be between 1 and 100")
        if type(self.score_threshold) not in (int, float):
            raise TypeError("score_threshold must be a number")
        if (
            not math.isfinite(self.score_threshold)
            or not 0 <= self.score_threshold <= 1
        ):
            raise ValueError("score_threshold must be between 0 and 1")
        object.__setattr__(self, "score_threshold", float(self.score_threshold))
        _require_nonnegative(self.snippet_max_chars, "snippet_max_chars")
        if not 64 <= self.snippet_max_chars <= 8192:
            raise ValueError("snippet_max_chars must be between 64 and 8192")

    def to_json(self) -> dict[str, object]:
        return {
            "text": self.text,
            "max_results": self.max_results,
            "score_threshold": self.score_threshold,
            "snippet_max_chars": self.snippet_max_chars,
        }


@dataclass(frozen=True, slots=True)
class WikiHit:
    relative_path: str
    line_start: int
    line_end: int
    evidence_layer: str
    content: str
    linked_raw_path: str | None = None

    def __post_init__(self) -> None:
        _require_relative_posix_path(self.relative_path, "relative_path")
        _require_nonnegative(self.line_start, "line_start")
        _require_nonnegative(self.line_end, "line_end")
        if self.line_start < 1:
            raise ValueError("line_start must be at least 1")
        if self.line_end < self.line_start:
            raise ValueError("line_end must not precede line_start")
        _require_nonempty(self.evidence_layer, "evidence_layer")
        if self.evidence_layer not in _WIKI_EVIDENCE_LAYERS:
            raise ValueError("evidence_layer is not a supported Wiki evidence layer")
        _require_nonempty(self.content, "content")
        if self.linked_raw_path is not None:
            raw_path = _require_relative_posix_path(
                self.linked_raw_path, "linked_raw_path"
            )
            raw_parts = PurePosixPath(raw_path).parts
            if len(raw_parts) < 2 or raw_parts[0] != "raw":
                raise ValueError("linked_raw_path must be under raw/")

    def to_json(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "evidence_layer": self.evidence_layer,
            "content": self.content,
            "linked_raw_path": self.linked_raw_path,
        }


def _require_relative_posix_path(value: object, name: str) -> str:
    value = _require_nonempty(value, name)
    if "\\" in value or "\x00" in value:
        raise ValueError(f"{name} must be a normalized relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in ("", ".", "..") for part in value.split("/"))
        or path.as_posix() != value
    ):
        raise ValueError(f"{name} must be a normalized relative POSIX path")
    return value


@runtime_checkable
class WikiRetriever(Protocol):
    """Synchronous, read-only retrieval from a local evidence base."""

    def search(self, query: WikiQuery) -> tuple[WikiHit, ...]: ...


__all__ = ["ResourceManager", "WikiHit", "WikiQuery", "WikiRetriever"]
