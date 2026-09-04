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
    and os.scandir in os.supports_fd
)


def _require_limit(value: object, name: str, *, minimum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class WikiIndexLimits:
    """Hard limits for one immutable Wiki index construction."""

    max_files: int = 10_000
    max_file_bytes: int = 2 * 1024 * 1024
    max_total_bytes: int = 64 * 1024 * 1024
    max_depth: int = 16
    max_sections: int = 100_000
    max_total_tokens: int = 2_000_000
    max_entries: int = 100_000
    max_name_bytes: int = 4 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in (
            "max_files",
            "max_file_bytes",
            "max_total_bytes",
            "max_sections",
            "max_total_tokens",
            "max_entries",
            "max_name_bytes",
        ):
            _require_limit(getattr(self, name), name, minimum=1)
        _require_limit(self.max_depth, "max_depth", minimum=0)


@dataclass(slots=True)
class _IndexBudget:
    limits: WikiIndexLimits
    file_count: int = 0
    total_declared_bytes: int = 0
    section_count: int = 0
    total_tokens: int = 0
    entry_count: int = 0
    total_name_bytes: int = 0

    def reserve_file(self, size: int, label: str) -> None:
        if size > self.limits.max_file_bytes:
            raise ValueError(f"{label} exceeds max_file_bytes")
        if self.file_count >= self.limits.max_files:
            raise ValueError("Wiki index exceeds max_files")
        if self.total_declared_bytes + size > self.limits.max_total_bytes:
            raise ValueError("Wiki index exceeds max_total_bytes")
        self.file_count += 1
        self.total_declared_bytes += size

    def begin_section(self) -> None:
        if self.section_count >= self.limits.max_sections:
            raise ValueError("Wiki index exceeds max_sections")
        self.section_count += 1

    def consume_token(self) -> None:
        if self.total_tokens >= self.limits.max_total_tokens:
            raise ValueError("Wiki index exceeds max_total_tokens")
        self.total_tokens += 1

    def reserve_entry(self, name: object) -> str:
        if not isinstance(name, str) or not name or name in {".", ".."}:
            raise ValueError("Wiki entry name is invalid")
        if "/" in name or "\\" in name or "\x00" in name:
            raise ValueError("Wiki entry name is invalid")
        try:
            encoded = name.encode("utf-8")
        except UnicodeError as error:
            raise ValueError("Wiki entry name must be valid UTF-8") from error
        if self.entry_count >= self.limits.max_entries:
            raise ValueError("Wiki index exceeds max_entries")
        if self.total_name_bytes + len(encoded) > self.limits.max_name_bytes:
            raise ValueError("Wiki index exceeds max_name_bytes")
        self.entry_count += 1
        self.total_name_bytes += len(encoded)
        return name


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

    def __init__(
        self,
        root: Path,
        *,
        limits: WikiIndexLimits = WikiIndexLimits(),
    ) -> None:
        if not isinstance(root, Path):
            raise TypeError("wiki root must be a Path")
        if not isinstance(limits, WikiIndexLimits):
            raise TypeError("limits must be a WikiIndexLimits")
        _require_fd_platform()
        budget = _IndexBudget(limits)
        root_fd = _open_root(root)
        with _owned_fd(root_fd):
            agents_stat = _required_regular_stat(root_fd, "AGENTS.md", "AGENTS.md")
            budget.reserve_file(agents_stat.st_size, "AGENTS.md")
            agents = _read_utf8_at(
                root_fd,
                "AGENTS.md",
                "AGENTS.md",
                max_bytes=limits.max_file_bytes,
                expected=agents_stat,
            )
            index_stat = _required_regular_stat(root_fd, "index.md", "index.md")
            budget.reserve_file(index_stat.st_size, "index.md")
            index = _read_utf8_at(
                root_fd,
                "index.md",
                "index.md",
                max_bytes=limits.max_file_bytes,
                expected=index_stat,
            )
            sections = tuple(self._build_sections(root_fd, budget))
        self._limits = limits
        self._agents = agents
        self._index = index
        self._sections = sections

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

    def _build_sections(
        self, root_fd: int, budget: _IndexBudget
    ) -> Iterator[_Section]:
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
                    namespace_fd,
                    PurePosixPath(namespace),
                    budget,
                    depth=0,
                ):
                    semantic_lines = _markdown_semantic_lines(text)
                    linked_raw_path = _linked_raw(
                        root_fd, relative_path, semantic_lines
                    )
                    yield from _sections(
                        relative_path,
                        semantic_lines,
                        linked_raw_path,
                        budget,
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


def _required_regular_stat(
    parent_fd: int, name: str, label: str
) -> os.stat_result:
    value = _optional_stat_at(parent_fd, name)
    if value is None or stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise ValueError(f"wiki root requires a regular {label}")
    return value


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


def _read_utf8_at(
    parent_fd: int,
    name: str,
    label: str,
    *,
    max_bytes: int,
    expected: os.stat_result | None = None,
) -> str:
    before_path = _required_regular_stat(parent_fd, name, label)
    if expected is not None and (
        _inode(expected) != _inode(before_path)
        or _file_metadata(expected) != _file_metadata(before_path)
    ):
        raise ValueError("wiki file changed before reading")
    if before_path.st_size > max_bytes:
        raise ValueError(f"{label} exceeds max_file_bytes")
    fd = _open_regular_at(parent_fd, name, expected=before_path)
    with _owned_fd(fd):
        before = os.fstat(fd)
        budget_snapshot = expected if expected is not None else before_path
        if (
            _inode(budget_snapshot) != _inode(before)
            or stat.S_IFMT(budget_snapshot.st_mode) != stat.S_IFMT(before.st_mode)
            or _file_metadata(budget_snapshot) != _file_metadata(before)
        ):
            raise ValueError("wiki file changed between budget check and open")
        if before.st_size > max_bytes:
            raise ValueError(f"{label} exceeds max_file_bytes")
        data = bytearray()
        declared_size = before.st_size
        while True:
            remaining = declared_size - len(data)
            chunk = os.read(fd, min(_READ_CHUNK_BYTES, remaining + 1))
            if not chunk:
                break
            if len(data) + len(chunk) > declared_size:
                raise ValueError("wiki file exceeds its declared byte size")
            data.extend(chunk)
        if len(data) != declared_size:
            raise ValueError("wiki file length differs from its declared byte size")
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
        return data.decode("utf-8")
    except UnicodeError as error:
        raise ValueError("wiki Markdown must be valid UTF-8") from error


def _markdown_documents(
    directory_fd: int,
    prefix: PurePosixPath,
    budget: _IndexBudget,
    *,
    depth: int,
) -> Iterator[tuple[str, str]]:
    try:
        with os.scandir(directory_fd) as entries:
            names = [budget.reserve_entry(entry.name) for entry in entries]
    except OSError as error:
        raise ValueError("wiki directory could not be enumerated") from error
    names.sort()
    for name in names:
        entry_stat = _optional_stat_at(directory_fd, name)
        if entry_stat is None:
            raise ValueError("wiki path changed during traversal")
        if stat.S_ISLNK(entry_stat.st_mode):
            raise ValueError("wiki search namespace contains a symlink")
        relative = prefix / name
        if stat.S_ISDIR(entry_stat.st_mode):
            if depth >= budget.limits.max_depth:
                raise ValueError("Wiki index exceeds max_depth")
            child_fd = _open_directory_at(
                directory_fd, name, expected=entry_stat
            )
            with _owned_fd(child_fd):
                yield from _markdown_documents(
                    child_fd,
                    relative,
                    budget,
                    depth=depth + 1,
                )
        elif stat.S_ISREG(entry_stat.st_mode) and name.casefold().endswith(".md"):
            label = relative.as_posix()
            budget.reserve_file(entry_stat.st_size, label)
            yield relative.as_posix(), _read_utf8_at(
                directory_fd,
                name,
                label,
                max_bytes=budget.limits.max_file_bytes,
                expected=entry_stat,
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
    budget: _IndexBudget,
) -> Iterator[_Section]:
    evidence_stack: list[tuple[int, str]] = []
    current: tuple[int, int, str] | None = None
    for index, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if match is None:
            continue
        if current is not None:
            start, level, heading = current
            yield from _emit_markdown_section(
                relative_path,
                heading,
                level,
                lines[start:index],
                start + 1,
                linked_raw_path,
                evidence_stack,
                budget,
            )
        current = (index, len(match.group(1)), match.group(2))
    if current is not None:
        start, level, heading = current
        yield from _emit_markdown_section(
            relative_path,
            heading,
            level,
            lines[start:],
            start + 1,
            linked_raw_path,
            evidence_stack,
            budget,
        )
    elif lines:
        yield from _emit_markdown_section(
            relative_path,
            "",
            1,
            lines,
            1,
            linked_raw_path,
            evidence_stack,
            budget,
        )


def _emit_markdown_section(
    relative_path: str,
    heading: str,
    level: int,
    section_lines: tuple[str, ...],
    line_start: int,
    linked_raw_path: str | None,
    evidence_stack: list[tuple[int, str]],
    budget: _IndexBudget,
) -> Iterator[_Section]:
    if not section_lines:
        return
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
            line_start,
            linked_raw_path,
            budget,
        )
        return
    yield _section(
        relative_path,
        heading,
        section_lines,
        line_start,
        evidence_layer,
        linked_raw_path,
        budget,
    )


def _evidence_boundary_fragments(
    relative_path: str,
    lines: tuple[str, ...],
    line_start: int,
    linked_raw_path: str | None,
    budget: _IndexBudget,
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
            budget,
        )


def _section(
    relative_path: str,
    heading: str,
    lines: tuple[str, ...],
    line_start: int,
    evidence_layer: str,
    linked_raw_path: str | None,
    budget: _IndexBudget,
) -> _Section:
    budget.begin_section()
    content_token_values: set[str] = set()
    for token in _iter_token_spans("\n".join(lines)):
        budget.consume_token()
        content_token_values.add(token.value)
    return _Section(
        relative_path,
        heading,
        lines,
        line_start,
        evidence_layer,
        linked_raw_path,
        frozenset(content_token_values),
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
    return tuple(_iter_token_spans(value))


def _iter_token_spans(value: str) -> Iterator[_TokenSpan]:
    for match in _WORD_RE.finditer(value):
        normalized = unicodedata.normalize("NFKC", match.group()).casefold()
        for token in _WORD_RE.finditer(normalized):
            yield _TokenSpan(token.group(), match.start(), match.end())


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


def _file_metadata(value: os.stat_result) -> tuple[int, int, int]:
    return value.st_size, value.st_mtime_ns, value.st_ctime_ns


__all__ = ["LocalWikiRetriever", "WikiIndexLimits"]
