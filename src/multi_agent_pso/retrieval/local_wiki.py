"""Deterministic, read-only retrieval from an OpenChem Markdown Wiki."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from multi_agent_pso.protocols import WikiHit, WikiQuery


_SEARCH_NAMESPACES = ("mocs", "sources", "entities", "syntheses", "questions")
_EVIDENCE_LAYERS = (
    "direct evidence",
    "author interpretation",
    "cross-paper synthesis",
    "open hypothesis",
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_EVIDENCE_MARKER_RE = re.compile(
    r"evidence[ _-]*layer\s*[:=—-]\s*([^\n]+)", re.IGNORECASE
)
_MARKDOWN_LINK_RE = re.compile(r"\]\(([^)]+)\)")
_WIKI_LINK_RE = re.compile(r"\[\[([^]|]+)(?:\|[^]]+)?\]\]")
_CODE_RE = re.compile(r"`([^`\n]+)`")


@dataclass(frozen=True, slots=True)
class _Section:
    relative_path: str
    heading: str
    lines: tuple[str, ...]
    line_start: int
    evidence_layer: str
    linked_raw_path: str | None
    content_tokens: frozenset[str]
    heading_tokens: frozenset[str]


@dataclass(frozen=True, slots=True)
class _RankedSection:
    section: _Section
    score: float
    heading_overlap: int
    source_preference: int


class LocalWikiRetriever:
    """Build an immutable in-memory index over approved Markdown namespaces."""

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path):
            raise TypeError("wiki root must be a Path")
        try:
            canonical_root = root.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ValueError("wiki root must be an existing directory") from error
        if not canonical_root.is_dir():
            raise ValueError("wiki root must be an existing directory")
        self._root = canonical_root
        self._agents = self._read_anchor("AGENTS.md")
        self._index = self._read_anchor("index.md")
        self._sections = tuple(self._build_sections())

    def search(self, query: WikiQuery) -> tuple[WikiHit, ...]:
        if not isinstance(query, WikiQuery):
            raise TypeError("query must be a WikiQuery")
        query_tokens = frozenset(_tokens(query.text))
        if not query_tokens:
            return ()
        ranked: list[_RankedSection] = []
        for section in self._sections:
            overlap = len(query_tokens & section.content_tokens)
            if overlap == 0:
                continue
            score = overlap / len(query_tokens)
            if score < query.score_threshold:
                continue
            heading_overlap = len(query_tokens & section.heading_tokens)
            ranked.append(
                _RankedSection(
                    section,
                    score,
                    heading_overlap,
                    int(section.relative_path.startswith("sources/")),
                )
            )
        ranked.sort(
            key=lambda item: (
                -item.score,
                -item.heading_overlap,
                -item.source_preference,
                item.section.relative_path,
                item.section.line_start,
            )
        )
        return tuple(
            self._hit(item.section, query_tokens, query.snippet_max_chars)
            for item in ranked[: query.max_results]
        )

    def _read_anchor(self, name: str) -> str:
        path = self._root / name
        if not path.exists() or path.is_symlink() or not path.is_file():
            raise ValueError(f"wiki root requires a regular {name}")
        return self._read_text(path)

    def _build_sections(self) -> Iterator[_Section]:
        for namespace in _SEARCH_NAMESPACES:
            directory = self._root / namespace
            if not directory.exists():
                continue
            for path in self._markdown_files(directory):
                relative_path = path.relative_to(self._root).as_posix()
                text = self._read_text(path)
                linked_raw_path = self._linked_raw(relative_path, text)
                yield from _sections(relative_path, text, linked_raw_path)

    def _markdown_files(self, directory: Path) -> tuple[Path, ...]:
        if directory.is_symlink():
            raise ValueError("wiki search namespace must not be a symlink")
        if not directory.is_dir():
            raise ValueError("wiki search namespace must be a directory")
        files: list[Path] = []
        pending = [directory]
        while pending:
            current = pending.pop()
            for child in sorted(current.iterdir(), key=lambda path: path.name):
                if child.is_symlink():
                    raise ValueError("wiki search namespace contains a symlink")
                try:
                    resolved = child.resolve(strict=True)
                except (OSError, RuntimeError) as error:
                    raise ValueError("wiki search path is invalid") from error
                if not resolved.is_relative_to(self._root):
                    raise ValueError("wiki search path escapes the root")
                if resolved.is_dir():
                    pending.append(resolved)
                elif resolved.is_file() and resolved.suffix.casefold() == ".md":
                    files.append(resolved)
        return tuple(
            sorted(
                files,
                key=lambda path: path.relative_to(self._root).as_posix(),
            )
        )

    def _read_text(self, path: Path) -> str:
        try:
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(self._root) or path.is_symlink():
                raise ValueError("wiki path escapes the root or is a symlink")
            return resolved.read_text(encoding="utf-8")
        except UnicodeError as error:
            raise ValueError("wiki Markdown must be valid UTF-8") from error
        except OSError as error:
            raise ValueError("wiki Markdown could not be read") from error

    def _linked_raw(self, relative_path: str, text: str) -> str | None:
        if not relative_path.startswith("sources/"):
            return None
        source_path = self._root / relative_path
        candidates = {
            candidate.strip().split("#", 1)[0]
            for pattern in (_MARKDOWN_LINK_RE, _WIKI_LINK_RE, _CODE_RE)
            for candidate in pattern.findall(text)
        }
        valid: list[str] = []
        raw_root = self._root / "raw"
        for candidate in candidates:
            if candidate.startswith("raw/"):
                path = self._root / candidate
            elif candidate.startswith("../raw/"):
                path = source_path.parent / candidate
            else:
                continue
            if self._safe_raw_file(path, raw_root):
                valid.append(path.resolve(strict=True).relative_to(self._root).as_posix())
        return min(valid) if valid else None

    def _safe_raw_file(self, path: Path, raw_root: Path) -> bool:
        try:
            raw = raw_root.resolve(strict=True)
            resolved = path.resolve(strict=True)
            relative = resolved.relative_to(raw)
        except (OSError, RuntimeError, ValueError):
            return False
        current = raw
        if raw_root.is_symlink():
            return False
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                return False
        return resolved.is_file()

    @staticmethod
    def _hit(
        section: _Section, query_tokens: frozenset[str], max_chars: int
    ) -> WikiHit:
        line_scores = [
            len(query_tokens & frozenset(_tokens(line))) for line in section.lines
        ]
        anchor = max(
            range(len(line_scores)),
            key=lambda index: (line_scores[index], -index),
        )
        selected: list[str] = []
        used = 0
        for line in section.lines[anchor:]:
            separator = 1 if selected else 0
            available = max_chars - used - separator
            if available <= 0:
                break
            fragment = (
                _relevant_line_fragment(line, query_tokens, available)
                if not selected and len(line) > available
                else line[:available]
            )
            selected.append(fragment)
            used += separator + min(len(line), available)
            if len(line) > available:
                break
        content = "\n".join(selected)
        return WikiHit(
            section.relative_path,
            section.line_start + anchor,
            section.line_start + anchor + len(selected) - 1,
            section.evidence_layer,
            content,
            section.linked_raw_path,
        )


def _sections(
    relative_path: str, text: str, linked_raw_path: str | None
) -> Iterator[_Section]:
    lines = text.splitlines()
    starts = [
        (index, len(match.group(1)), match.group(2))
        for index, line in enumerate(lines)
        if (match := _HEADING_RE.match(line)) is not None
    ]
    if not starts and lines:
        starts = [(0, 1, "")]
    evidence_stack: list[tuple[int, str]] = []
    for position, (start, level, heading) in enumerate(starts):
        end = starts[position + 1][0] if position + 1 < len(starts) else len(lines)
        section_lines = tuple(lines[start:end])
        if not section_lines:
            continue
        while evidence_stack and evidence_stack[-1][0] >= level:
            evidence_stack.pop()
        declared = _declared_evidence_layer(heading, section_lines)
        evidence_layer = (
            declared
            if declared is not None
            else evidence_stack[-1][1]
            if evidence_stack
            else "open hypothesis"
        )
        evidence_stack.append((level, evidence_layer))
        yield _Section(
            relative_path,
            heading,
            section_lines,
            start + 1,
            evidence_layer,
            linked_raw_path,
            frozenset(_tokens("\n".join(section_lines))),
            frozenset(_tokens(heading)),
        )


def _declared_evidence_layer(
    heading: str, lines: tuple[str, ...]
) -> str | None:
    marker = _EVIDENCE_MARKER_RE.search("\n".join(lines))
    if marker is not None:
        declared = _normalize_label(marker.group(1))
        return declared if declared in _EVIDENCE_LAYERS else "open hypothesis"
    normalized_heading = _normalize_label(heading)
    heading_segments = {
        _normalize_label(segment)
        for segment in re.split(
            r"\s*(?:—|–|\||:|\s-\s)\s*",
            unicodedata.normalize("NFKC", heading).casefold(),
        )
    }
    for layer in _EVIDENCE_LAYERS:
        if layer in heading_segments:
            return layer
    if normalized_heading.startswith("interpretation"):
        return "author interpretation"
    return None


def _relevant_line_fragment(
    line: str, query_tokens: frozenset[str], max_chars: int
) -> str:
    normalized = unicodedata.normalize("NFKC", line).casefold()
    positions = [
        position
        for token in query_tokens
        if (position := normalized.find(token)) >= 0
    ]
    if not positions:
        return line[:max_chars]
    normalized_start = max(0, min(positions) - max_chars // 3)
    if normalized:
        start = round(normalized_start * len(line) / len(normalized))
    else:
        start = 0
    start = min(start, len(line) - max_chars)
    return line[start : start + max_chars]


def _normalize_label(value: str) -> str:
    return " ".join(_tokens(value))


def _tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))


__all__ = ["LocalWikiRetriever"]
