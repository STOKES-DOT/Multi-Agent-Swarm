"""Deterministic, read-only retrieval from an OpenChem Markdown Wiki."""

from __future__ import annotations

import os
import re
import stat
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

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
    r"^\s*-\s*Evidence\s+layer\s*:\s*(\S.*?)\s*$", re.IGNORECASE
)
_EVIDENCE_BULLET_RE = re.compile(r"^\s*-\s*([^:]+?)\s*:\s*(\S.*?)\s*$")
_RAW_METADATA_RE = re.compile(
    r"^-\s*(?:Raw|Local raw) snapshot:\s*"
    r"(?:`(?P<backtick>raw/[^\s`<>]+)`|"
    r"<(?P<angle>raw/[^\s`<>]+)>|"
    r"(?P<plain>raw/[^\s`<>]+))\s*$"
)
_FENCE_OPEN_RE = re.compile(r"^ {0,3}(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
_FENCE_CLOSE_RE = re.compile(r"^ {0,3}(?P<fence>`+|~+)[ \t]*$")
_WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_DASH_RE = re.compile("[‐‑‒–—―−]")
_READ_CHUNK_BYTES = 64 * 1024
_HAS_REQUIRED_FD_CALLS = (
    os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.stat in os.supports_follow_symlinks
    and os.listdir in os.supports_fd
)


@dataclass(frozen=True, slots=True)
class _TokenSpan:
    value: str
    start: int
    end: int


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
        _require_fd_platform()
        root_fd = _open_root(root)
        with _owned_fd(root_fd):
            self._agents = _read_utf8_at(root_fd, "AGENTS.md", "AGENTS.md")
            self._index = _read_utf8_at(root_fd, "index.md", "index.md")
            self._sections = tuple(self._build_sections(root_fd))

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
            ranked.append(
                _RankedSection(
                    section,
                    score,
                    len(query_tokens & section.heading_tokens),
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

    def _build_sections(self, root_fd: int) -> Iterator[_Section]:
        for namespace in _SEARCH_NAMESPACES:
            namespace_stat = _optional_stat_at(root_fd, namespace)
            if namespace_stat is None:
                continue
            if stat.S_ISLNK(namespace_stat.st_mode):
                raise ValueError("wiki search namespace contains a symlink")
            if not stat.S_ISDIR(namespace_stat.st_mode):
                raise ValueError("wiki search namespace must be a directory")
            namespace_fd = _open_directory_at(
                root_fd, namespace, expected=namespace_stat
            )
            with _owned_fd(namespace_fd):
                for relative_path, text in _markdown_documents(
                    namespace_fd, PurePosixPath(namespace)
                ):
                    semantic_lines = _markdown_semantic_lines(text)
                    linked_raw_path = _linked_raw(
                        root_fd, relative_path, semantic_lines
                    )
                    yield from _sections(
                        relative_path, semantic_lines, linked_raw_path
                    )

    @staticmethod
    def _hit(
        section: _Section, query_tokens: frozenset[str], max_chars: int
    ) -> WikiHit:
        line_matches = [
            tuple(span for span in _token_spans(line) if span.value in query_tokens)
            for line in section.lines
        ]
        line_scores = [len({span.value for span in spans}) for spans in line_matches]
        anchor = max(
            range(len(line_scores)),
            key=lambda index: (line_scores[index], -index),
        )
        selected: list[str] = []
        used = 0
        for offset, line in enumerate(section.lines[anchor:]):
            separator = 1 if selected else 0
            available = max_chars - used - separator
            if available <= 0:
                break
            matches = line_matches[anchor + offset]
            fragment = (
                _line_fragment(line, matches, available)
                if not selected and len(line) > available
                else line[:available]
            )
            selected.append(fragment)
            used += separator + len(fragment)
            if len(line) > available:
                break
        return WikiHit(
            section.relative_path,
            section.line_start + anchor,
            section.line_start + anchor + len(selected) - 1,
            section.evidence_layer,
            "\n".join(selected),
            section.linked_raw_path,
        )


def _require_fd_platform() -> None:
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("platform lacks O_NOFOLLOW/O_DIRECTORY Wiki safety")
    if not _HAS_REQUIRED_FD_CALLS:
        raise RuntimeError("platform lacks required dir_fd Wiki safety")


def _base_flags() -> int:
    return os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _open_root(root: Path) -> int:
    try:
        fd = os.open(root, _base_flags() | os.O_DIRECTORY)
    except OSError as error:
        raise ValueError("wiki root must be an existing non-symlink directory") from error
    try:
        if not stat.S_ISDIR(os.fstat(fd).st_mode):
            raise ValueError("wiki root must be an existing directory")
        return fd
    except BaseException:
        os.close(fd)
        raise


@contextmanager
def _owned_fd(fd: int) -> Iterator[int]:
    primary: BaseException | None = None
    try:
        yield fd
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            os.close(fd)
        except Exception as close_error:
            if primary is None:
                raise
            primary.add_note(
                f"Wiki fd close failed: {type(close_error).__name__}: {close_error}"
            )


def _optional_stat_at(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ValueError("wiki path could not be inspected") from error


def _open_directory_at(
    parent_fd: int,
    name: str,
    *,
    expected: os.stat_result | None = None,
) -> int:
    try:
        fd = os.open(name, _base_flags() | os.O_DIRECTORY, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError("wiki directory is invalid, changed, or a symlink") from error
    try:
        actual = os.fstat(fd)
        if not stat.S_ISDIR(actual.st_mode):
            raise ValueError("wiki directory changed during traversal")
        if expected is not None and _inode(expected) != _inode(actual):
            raise ValueError("wiki directory changed during traversal")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _open_regular_at(
    parent_fd: int,
    name: str,
    *,
    expected: os.stat_result | None = None,
) -> int:
    flags = _base_flags() | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(name, flags, dir_fd=parent_fd)
    except OSError as error:
        raise ValueError("wiki file is invalid, changed, or a symlink") from error
    try:
        actual = os.fstat(fd)
        if not stat.S_ISREG(actual.st_mode):
            raise ValueError("wiki file must be regular")
        if expected is not None and _inode(expected) != _inode(actual):
            raise ValueError("wiki file changed during open")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _read_utf8_at(parent_fd: int, name: str, label: str) -> str:
    before_path = _optional_stat_at(parent_fd, name)
    if before_path is None or stat.S_ISLNK(before_path.st_mode):
        raise ValueError(f"wiki root requires a regular {label}")
    if not stat.S_ISREG(before_path.st_mode):
        raise ValueError(f"wiki root requires a regular {label}")
    fd = _open_regular_at(parent_fd, name, expected=before_path)
    with _owned_fd(fd):
        before = os.fstat(fd)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, _READ_CHUNK_BYTES)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        after_path = _optional_stat_at(parent_fd, name)
        identity = (before.st_dev, before.st_ino)
        metadata = (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if (
            identity != _inode(before_path)
            or metadata
            != (
                before_path.st_size,
                before_path.st_mtime_ns,
                before_path.st_ctime_ns,
            )
            or identity != (after.st_dev, after.st_ino)
            or metadata != (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            or after_path is None
            or stat.S_ISLNK(after_path.st_mode)
            or identity != (after_path.st_dev, after_path.st_ino)
            or (after.st_size, after.st_mtime_ns, after.st_ctime_ns)
            != (
                after_path.st_size,
                after_path.st_mtime_ns,
                after_path.st_ctime_ns,
            )
        ):
            raise ValueError("wiki file changed or became a symlink while reading")
    try:
        return b"".join(chunks).decode("utf-8")
    except UnicodeError as error:
        raise ValueError("wiki Markdown must be valid UTF-8") from error


def _markdown_documents(
    directory_fd: int, prefix: PurePosixPath
) -> Iterator[tuple[str, str]]:
    try:
        names = sorted(os.listdir(directory_fd))
    except OSError as error:
        raise ValueError("wiki directory could not be listed") from error
    for name in names:
        entry_stat = _optional_stat_at(directory_fd, name)
        if entry_stat is None:
            raise ValueError("wiki path changed during traversal")
        if stat.S_ISLNK(entry_stat.st_mode):
            raise ValueError("wiki search namespace contains a symlink")
        relative = prefix / name
        if stat.S_ISDIR(entry_stat.st_mode):
            child_fd = _open_directory_at(
                directory_fd, name, expected=entry_stat
            )
            with _owned_fd(child_fd):
                yield from _markdown_documents(child_fd, relative)
        elif stat.S_ISREG(entry_stat.st_mode) and name.casefold().endswith(".md"):
            yield relative.as_posix(), _read_utf8_at(
                directory_fd, name, relative.as_posix()
            )


def _linked_raw(
    root_fd: int, relative_path: str, lines: tuple[str, ...]
) -> str | None:
    if not relative_path.startswith("sources/"):
        return None
    valid: list[str] = []
    for line in lines:
        match = _RAW_METADATA_RE.fullmatch(line)
        if match is None:
            continue
        candidate = next(value for value in match.groupdict().values() if value)
        if _valid_raw_path(candidate) and _raw_regular_exists(root_fd, candidate):
            valid.append(candidate)
    return min(valid) if valid else None


def _valid_raw_path(value: str) -> bool:
    if "\\" in value or "\x00" in value:
        return False
    parts = value.split("/")
    return (
        len(parts) >= 2
        and parts[0] == "raw"
        and all(part not in ("", ".", "..") for part in parts)
        and PurePosixPath(value).as_posix() == value
    )


def _raw_regular_exists(root_fd: int, relative_path: str) -> bool:
    parts = relative_path.split("/")
    opened: list[tuple[int, str, int]] = []
    parent_fd = root_fd
    try:
        for part in parts[:-1]:
            component_stat = _optional_stat_at(parent_fd, part)
            if (
                component_stat is None
                or stat.S_ISLNK(component_stat.st_mode)
                or not stat.S_ISDIR(component_stat.st_mode)
            ):
                return False
            child_fd = _open_directory_at(
                parent_fd, part, expected=component_stat
            )
            opened.append((parent_fd, part, child_fd))
            parent_fd = child_fd
        final_stat = _optional_stat_at(parent_fd, parts[-1])
        if (
            final_stat is None
            or stat.S_ISLNK(final_stat.st_mode)
            or not stat.S_ISREG(final_stat.st_mode)
        ):
            return False
        final_fd = _open_regular_at(
            parent_fd, parts[-1], expected=final_stat
        )
        try:
            after_final = _optional_stat_at(parent_fd, parts[-1])
            if (
                after_final is None
                or stat.S_ISLNK(after_final.st_mode)
                or _inode(after_final) != _inode(os.fstat(final_fd))
            ):
                return False
            for ancestor_fd, name, child_fd in opened:
                after_component = _optional_stat_at(ancestor_fd, name)
                if (
                    after_component is None
                    or stat.S_ISLNK(after_component.st_mode)
                    or _inode(after_component) != _inode(os.fstat(child_fd))
                ):
                    return False
            return True
        finally:
            os.close(final_fd)
    except (OSError, ValueError):
        return False
    finally:
        for _, _, fd in reversed(opened):
            os.close(fd)


def _sections(
    relative_path: str,
    lines: tuple[str, ...],
    linked_raw_path: str | None,
) -> Iterator[_Section]:
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
        declared = _declared_section_layer(heading, section_lines)
        evidence_layer = (
            declared
            if declared is not None
            else evidence_stack[-1][1]
            if evidence_stack
            else "open hypothesis"
        )
        evidence_stack.append((level, evidence_layer))
        if _normalized_words(heading) == "evidence boundary":
            yield from _evidence_boundary_fragments(
                relative_path,
                section_lines,
                start + 1,
                linked_raw_path,
            )
            continue
        yield _section(
            relative_path,
            heading,
            section_lines,
            start + 1,
            evidence_layer,
            linked_raw_path,
        )


def _evidence_boundary_fragments(
    relative_path: str,
    lines: tuple[str, ...],
    line_start: int,
    linked_raw_path: str | None,
) -> Iterator[_Section]:
    for offset, line in enumerate(lines[1:], start=1):
        if not line.strip():
            continue
        match = _EVIDENCE_BULLET_RE.match(line)
        layer = (
            _canonical_evidence_label(match.group(1))
            if match is not None
            else None
        )
        yield _section(
            relative_path,
            layer or "",
            (line,),
            line_start + offset,
            layer if layer in _EVIDENCE_LAYERS else "open hypothesis",
            linked_raw_path,
        )


def _section(
    relative_path: str,
    heading: str,
    lines: tuple[str, ...],
    line_start: int,
    evidence_layer: str,
    linked_raw_path: str | None,
) -> _Section:
    return _Section(
        relative_path,
        heading,
        lines,
        line_start,
        evidence_layer,
        linked_raw_path,
        frozenset(_tokens("\n".join(lines))),
        frozenset(_tokens(heading)),
    )


def _declared_section_layer(
    heading: str, lines: tuple[str, ...]
) -> str | None:
    for line in lines:
        marker = _EVIDENCE_MARKER_RE.match(line)
        if marker is not None:
            return _canonical_evidence_label(marker.group(1)) or "open hypothesis"
    heading_segments = {
        _canonical_evidence_label(segment)
        for segment in re.split(
            r"\s*(?:—|–|\||:|\s-\s)\s*",
            unicodedata.normalize("NFKC", heading).casefold(),
        )
    }
    for layer in _EVIDENCE_LAYERS:
        if layer in heading_segments:
            return layer
    if _normalized_words(heading).startswith("interpretation"):
        return "author interpretation"
    return None


def _canonical_evidence_label(value: str) -> str | None:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = _DASH_RE.sub("-", normalized)
    normalized = re.sub(r"\s*-\s*", "-", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    if normalized in _EVIDENCE_LAYERS:
        return normalized
    if normalized[-1:] in {".", "。"}:
        without_terminator = normalized[:-1]
        if without_terminator in _EVIDENCE_LAYERS:
            return without_terminator
    return None


def _normalized_words(value: str) -> str:
    return " ".join(_tokens(value))


def _token_spans(value: str) -> tuple[_TokenSpan, ...]:
    spans: list[_TokenSpan] = []
    for match in _WORD_RE.finditer(value):
        normalized = unicodedata.normalize("NFKC", match.group()).casefold()
        spans.extend(
            _TokenSpan(token.group(), match.start(), match.end())
            for token in _WORD_RE.finditer(normalized)
        )
    return tuple(spans)


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(span.value for span in _token_spans(value))


def _line_fragment(
    line: str, matches: tuple[_TokenSpan, ...], max_chars: int
) -> str:
    if not matches:
        return line[:max_chars]
    match = matches[0]
    start = max(0, match.start - max_chars // 3)
    start = max(start, match.end - max_chars)
    start = min(start, len(line) - max_chars)
    return line[start : start + max_chars]


def _markdown_semantic_lines(text: str) -> tuple[str, ...]:
    semantic: list[str] = []
    fence_character: str | None = None
    fence_length = 0
    for line in text.splitlines():
        if fence_character is None:
            opening = _FENCE_OPEN_RE.match(line)
            if opening is None:
                semantic.append(line)
                continue
            fence = opening.group("fence")
            if fence[0] == "`" and "`" in opening.group("info"):
                semantic.append(line)
                continue
            fence_character = fence[0]
            fence_length = len(fence)
            semantic.append("")
            continue
        closing = _FENCE_CLOSE_RE.match(line)
        if (
            closing is not None
            and closing.group("fence")[0] == fence_character
            and len(closing.group("fence")) >= fence_length
        ):
            fence_character = None
            fence_length = 0
        semantic.append("")
    return tuple(semantic)


def _inode(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


__all__ = ["LocalWikiRetriever"]
