from __future__ import annotations

import asyncio
import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import multi_agent_pso.retrieval.local_wiki as local_wiki_module
from multi_agent_pso.protocols import WikiHit, WikiRetriever
from multi_agent_pso.retrieval import LocalWikiRetriever, WikiIndexLimits, WikiQuery


FIXTURE_WIKI = Path(__file__).parents[1] / "fixtures" / "wiki"


def _wiki(tmp_path: Path, pages: dict[str, str] | None = None) -> Path:
    root = tmp_path / "wiki"
    root.mkdir()
    (root / "AGENTS.md").write_text(
        "Use direct evidence, author interpretation, cross-paper synthesis, "
        "and open hypothesis.\n",
        encoding="utf-8",
    )
    (root / "index.md").write_text("# Index\n", encoding="utf-8")
    for relative_path, content in (pages or {}).items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def test_retriever_returns_source_locations_in_stable_order() -> None:
    retriever = LocalWikiRetriever(FIXTURE_WIKI)

    first = retriever.search(WikiQuery(text="ＲＥＤ-Absorption", max_results=5))
    second = retriever.search(WikiQuery(text="ＲＥＤ-Absorption", max_results=5))

    assert first == second
    assert first[0].relative_path == "sources/source-red.md"
    assert first[0].line_start == 5
    assert first[0].line_end >= first[0].line_start
    assert first[0].evidence_layer == "direct evidence"
    assert "Red absorption" in first[0].content
    assert first[0].linked_raw_path == "raw/source-red/v1.pdf"
    assert isinstance(retriever, WikiRetriever)


def test_retriever_does_not_modify_wiki() -> None:
    before = {p: p.read_bytes() for p in FIXTURE_WIKI.rglob("*") if p.is_file()}

    retriever = LocalWikiRetriever(FIXTURE_WIKI)
    retriever.search(WikiQuery(text="absorption", max_results=5))

    after = {p: p.read_bytes() for p in FIXTURE_WIKI.rglob("*") if p.is_file()}
    assert before == after
    assert not hasattr(retriever, "write")


@pytest.mark.parametrize("kind", ["missing", "file", "missing-agents", "missing-index"])
def test_root_and_required_anchor_validation(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "wiki"
    if kind == "missing":
        pass
    elif kind == "file":
        root.write_text("not a directory", encoding="utf-8")
    else:
        root.mkdir()
        if kind != "missing-agents":
            (root / "AGENTS.md").write_text("rules", encoding="utf-8")
        if kind != "missing-index":
            (root / "index.md").write_text("index", encoding="utf-8")

    with pytest.raises(ValueError, match="root|AGENTS.md|index.md"):
        LocalWikiRetriever(root)


def test_only_declared_markdown_namespaces_are_indexed(tmp_path: Path) -> None:
    root = _wiki(
        tmp_path,
        {
            "sources/source.md": "# Source — direct evidence\nallowedtoken\n",
            "mocs/moc.md": "# MOC — cross-paper synthesis\nallowedtoken\n",
            "entities/entity.md": "# Entity — author interpretation\nallowedtoken\n",
            "syntheses/synthesis.md": (
                "# Synthesis — cross-paper synthesis\nallowedtoken\n"
            ),
            "questions/question.md": "# Question — open hypothesis\nallowedtoken\n",
            "raw/hidden.md": "# Hidden\nrawonlytoken\n",
            "derived/hidden.md": "# Hidden\nderivedonlytoken\n",
            ".obsidian/hidden.md": "# Hidden\nobsidianonlytoken\n",
            "misc/hidden.md": "# Hidden\nmiscellaneousonlytoken\n",
        },
    )
    retriever = LocalWikiRetriever(root)

    hits = retriever.search(WikiQuery("allowedtoken", 10))

    assert {hit.relative_path.split("/", 1)[0] for hit in hits} == {
        "entities",
        "mocs",
        "questions",
        "sources",
        "syntheses",
    }
    for token in (
        "rawonlytoken",
        "derivedonlytoken",
        "obsidianonlytoken",
        "miscellaneousonlytoken",
    ):
        assert retriever.search(WikiQuery(token, 5)) == ()


def test_rank_is_overlap_then_heading_then_source_then_path(tmp_path: Path) -> None:
    root = _wiki(
        tmp_path,
        {
            "mocs/heading.md": (
                "# Alpha beta — cross-paper synthesis\nalpha beta\n"
            ),
            "sources/body.md": "# Other — direct evidence\nalpha beta\n",
            "sources/a.md": "# Other — direct evidence\nalpha\n",
            "sources/b.md": "# Other — direct evidence\nalpha\n",
            "entities/entity.md": "# Other — author interpretation\nalpha\n",
        },
    )

    hits = LocalWikiRetriever(root).search(WikiQuery("alpha beta", 10))

    assert [hit.relative_path for hit in hits] == [
        "mocs/heading.md",
        "sources/body.md",
        "sources/a.md",
        "sources/b.md",
        "entities/entity.md",
    ]


def test_threshold_no_match_result_and_result_limit(tmp_path: Path) -> None:
    root = _wiki(
        tmp_path,
        {
            "sources/a.md": "# One — direct evidence\nalpha\n",
            "sources/b.md": "# Two — direct evidence\nalpha beta\n",
        },
    )
    retriever = LocalWikiRetriever(root)

    assert retriever.search(WikiQuery("absent", 10)) == ()
    assert retriever.search(WikiQuery("alpha beta", 10, score_threshold=0.75)) == (
        WikiHit("sources/b.md", 2, 2, "direct evidence", "alpha beta"),
    )
    assert len(retriever.search(WikiQuery("alpha", 1))) == 1


def test_snippet_is_bounded_and_reports_exact_lines(tmp_path: Path) -> None:
    root = _wiki(
        tmp_path,
        {
            "sources/long.md": (
                "# Heading — direct evidence\n"
                "unrelated\n"
                + "target "
                + "x" * 200
                + "\ntrailing\n"
            )
        },
    )

    hit = LocalWikiRetriever(root).search(
        WikiQuery("target", 1, snippet_max_chars=64)
    )[0]

    assert len(hit.content) == 64
    assert hit.content.startswith("target ")
    assert (hit.line_start, hit.line_end) == (3, 3)


def test_source_explicit_raw_link_is_validated_but_raw_is_not_indexed(
    tmp_path: Path,
) -> None:
    root = _wiki(
        tmp_path,
        {
            "sources/source.md": (
                "# Signal — direct evidence\n"
                "- Raw snapshot: raw/source/v1.pdf\n"
                "rawsignal\n"
            ),
            "mocs/moc.md": (
                "# Navigation — cross-paper synthesis\n"
                "- Raw snapshot: raw/source/v1.pdf\n"
                "rawsignal\n"
            ),
        },
    )
    raw = root / "raw/source/v1.pdf"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"pdf snapshot rawonlytoken")

    hits = LocalWikiRetriever(root).search(WikiQuery("rawsignal", 5))

    by_path = {hit.relative_path: hit for hit in hits}
    assert by_path["sources/source.md"].linked_raw_path == "raw/source/v1.pdf"
    assert by_path["mocs/moc.md"].linked_raw_path is None
    assert LocalWikiRetriever(root).search(WikiQuery("rawonlytoken", 5)) == ()


def test_unsafe_raw_link_and_raw_symlink_are_not_exposed(tmp_path: Path) -> None:
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"outside")
    root = _wiki(
        tmp_path,
        {
            "sources/escape.md": (
                "# Escape — direct evidence\n"
                "- Raw snapshot: raw/../../outside.pdf\n"
                "- Raw snapshot: raw/link.pdf\n"
                "escapetoken\n"
            )
        },
    )
    raw = root / "raw"
    raw.mkdir()
    (raw / "link.pdf").symlink_to(outside)

    hit = LocalWikiRetriever(root).search(WikiQuery("escapetoken", 1))[0]

    assert hit.linked_raw_path is None


def test_raw_snapshot_rejects_an_intermediate_directory_symlink(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "v1.pdf").write_bytes(b"outside")
    root = _wiki(
        tmp_path,
        {
            "sources/source.md": (
                "# Source — direct evidence\n"
                "- Raw snapshot: raw/linked/v1.pdf\n"
                "rawinnersymlinktoken\n"
            )
        },
    )
    raw = root / "raw"
    raw.mkdir()
    (raw / "linked").symlink_to(outside, target_is_directory=True)

    hit = LocalWikiRetriever(root).search(WikiQuery("rawinnersymlinktoken", 1))[0]

    assert hit.linked_raw_path is None


def test_raw_snapshot_path_swap_after_open_is_not_exposed(
    tmp_path: Path, monkeypatch
) -> None:
    root = _wiki(
        tmp_path,
        {
            "sources/source.md": (
                "# Source — direct evidence\n"
                "- Raw snapshot: raw/source/v1.pdf\n"
                "rawswaptoken\n"
            )
        },
    )
    raw = root / "raw/source/v1.pdf"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"snapshot")
    backup = raw.with_name("original.pdf")
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"outside")
    original_open = os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if dir_fd is None:
            fd = original_open(path, flags, mode)
        else:
            fd = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == "v1.pdf" and not swapped:
            raw.rename(backup)
            raw.symlink_to(outside)
            swapped = True
        return fd

    monkeypatch.setattr(os, "open", swapping_open)

    hit = LocalWikiRetriever(root).search(WikiQuery("rawswaptoken", 1))[0]

    assert hit.linked_raw_path is None


@pytest.mark.parametrize(
    "metadata",
    [
        "- Raw snapshot: raw/source/v1.pdf",
        "- Raw snapshot: `raw/source/v1.pdf`",
        "- Raw snapshot: <raw/source/v1.pdf>",
        "- Local raw snapshot: raw/source/v1.pdf",
        "- Local raw snapshot: `raw/source/v1.pdf`",
        "- Local raw snapshot: <raw/source/v1.pdf>",
    ],
)
def test_raw_snapshot_metadata_accepts_only_standalone_canonical_lines(
    tmp_path: Path, metadata: str
) -> None:
    root = _wiki(
        tmp_path,
        {"sources/source.md": f"# Source — direct evidence\n{metadata}\nrawmetatoken\n"},
    )
    raw = root / "raw/source/v1.pdf"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"snapshot")

    hit = LocalWikiRetriever(root).search(WikiQuery("rawmetatoken", 1))[0]

    assert hit.linked_raw_path == "raw/source/v1.pdf"


def test_raw_snapshot_allows_a_regular_file_directly_under_raw(tmp_path: Path) -> None:
    root = _wiki(
        tmp_path,
        {
            "sources/source.md": (
                "# Source — direct evidence\n"
                "- Raw snapshot: raw/v1.pdf\n"
                "directrawtoken\n"
            )
        },
    )
    raw = root / "raw/v1.pdf"
    raw.parent.mkdir()
    raw.write_bytes(b"snapshot")

    hit = LocalWikiRetriever(root).search(WikiQuery("directrawtoken", 1))[0]

    assert hit.linked_raw_path == "raw/v1.pdf"


@pytest.mark.parametrize(
    "metadata",
    [
        "- Raw snapshot: raw/source/v1.pdf extra prose",
        "  - Raw snapshot: raw/source/v1.pdf",
        "- Local snapshot: `raw/source/v1.pdf`",
        "The Raw snapshot: raw/source/v1.pdf is useful.",
        "[snapshot](raw/source/v1.pdf)",
    ],
)
def test_raw_snapshot_metadata_rejects_noncanonical_prose(
    tmp_path: Path, metadata: str
) -> None:
    root = _wiki(
        tmp_path,
        {
            "sources/source.md": (
                f"# Source — direct evidence\n{metadata}\nrawrejecttoken\n"
            )
        },
    )
    raw = root / "raw/source/v1.pdf"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"snapshot")

    hit = LocalWikiRetriever(root).search(WikiQuery("rawrejecttoken", 1))[0]

    assert hit.linked_raw_path is None


def test_fixture_legacy_local_raw_snapshot_anchor_is_bound() -> None:
    hit = LocalWikiRetriever(FIXTURE_WIKI).search(WikiQuery("parkanchortoken", 1))[0]

    assert hit.relative_path == "sources/source-park-anchor.md"
    assert hit.linked_raw_path == "raw/source-park-anchor/v1.pdf"


@pytest.mark.parametrize("symlink_kind", ["file", "directory"])
def test_index_rejects_symlinks_in_search_namespaces(
    tmp_path: Path, symlink_kind: str
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("# Secret\noutsideescape\n", encoding="utf-8")
    root = _wiki(tmp_path)
    sources = root / "sources"
    sources.mkdir()
    if symlink_kind == "file":
        (sources / "secret.md").symlink_to(outside / "secret.md")
    else:
        (sources / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink|escape"):
        LocalWikiRetriever(root)


def test_invalid_evidence_label_is_demoted_to_open_hypothesis(tmp_path: Path) -> None:
    root = _wiki(
        tmp_path,
        {
            "sources/unsafe.md": (
                "# Unsupported\n"
                "Evidence layer: fabricated certainty\n"
                "unsupportedlabeltoken\n"
            )
        },
    )

    hit = LocalWikiRetriever(root).search(WikiQuery("unsupportedlabeltoken", 1))[0]

    assert hit.evidence_layer == "open hypothesis"


def test_negated_evidence_marker_does_not_promote_section(tmp_path: Path) -> None:
    root = _wiki(
        tmp_path,
        {
            "sources/negated-marker.md": (
                "# Unclassified\n"
                "This is not an Evidence layer: direct evidence marker.\n"
                "negatedmarkertoken\n"
            )
        },
    )

    hit = LocalWikiRetriever(root).search(WikiQuery("negatedmarkertoken", 1))[0]

    assert hit.evidence_layer == "open hypothesis"


def test_evidence_boundary_bullets_are_independent_typed_fragments() -> None:
    hits = LocalWikiRetriever(FIXTURE_WIKI).search(
        WikiQuery("directclaim authorclaim crossclaim openclaim", 10)
    )
    by_token = {
        token: next(hit for hit in hits if token in hit.content)
        for token in ("directclaim", "authorclaim", "crossclaim", "openclaim")
    }

    assert {
        token: (hit.evidence_layer, hit.line_start, hit.line_end)
        for token, hit in by_token.items()
    } == {
        "directclaim": ("direct evidence", 19, 19),
        "authorclaim": ("author interpretation", 20, 20),
        "crossclaim": ("cross-paper synthesis", 21, 21),
        "openclaim": ("open hypothesis", 22, 22),
    }
    assert len({hit.content for hit in by_token.values()}) == 4


def test_fixture_period_terminated_evidence_markers_are_typed() -> None:
    hits = LocalWikiRetriever(FIXTURE_WIKI).search(
        WikiQuery(
            "perioddirecttoken periodauthortoken "
            "periodsynthesistoken periodhypothesistoken",
            10,
        )
    )
    by_token = {
        token: next(hit for hit in hits if token in hit.content)
        for token in (
            "perioddirecttoken",
            "periodauthortoken",
            "periodsynthesistoken",
            "periodhypothesistoken",
        )
    }

    assert {token: hit.evidence_layer for token, hit in by_token.items()} == {
        "perioddirecttoken": "direct evidence",
        "periodauthortoken": "author interpretation",
        "periodsynthesistoken": "cross-paper synthesis",
        "periodhypothesistoken": "open hypothesis",
    }


@pytest.mark.parametrize(
    "layer",
    [
        "direct evidence",
        "author interpretation",
        "cross-paper synthesis",
        "open hypothesis",
    ],
)
@pytest.mark.parametrize("terminator", [".", "。"])
def test_single_supported_evidence_terminator_is_accepted(
    tmp_path: Path, layer: str, terminator: str
) -> None:
    assert local_wiki_module._canonical_evidence_label(layer + terminator) == layer
    root = _wiki(
        tmp_path,
        {
            "sources/label.md": (
                "# Label check\n"
                f"- Evidence layer: {layer}{terminator}\n"
                "terminatedlabeltoken\n"
            )
        },
    )

    hit = LocalWikiRetriever(root).search(WikiQuery("terminatedlabeltoken", 1))[0]

    assert hit.evidence_layer == layer


@pytest.mark.parametrize(
    "declaration",
    [
        "direct evidence because the source says so",
        "direct evidence..",
        "direct evidence。。",
        "direct evidence!",
        "direct evidence. extra prose",
    ],
)
def test_evidence_terminator_does_not_allow_extra_syntax(
    tmp_path: Path, declaration: str
) -> None:
    root = _wiki(
        tmp_path,
        {
            "sources/label.md": (
                "# Label check\n"
                f"- Evidence layer: {declaration}\n"
                "invalidterminatedlabeltoken\n"
            )
        },
    )

    hit = LocalWikiRetriever(root).search(
        WikiQuery("invalidterminatedlabeltoken", 1)
    )[0]

    assert hit.evidence_layer == "open hypothesis"


def test_backtick_fence_excludes_fake_metadata_headings_and_evidence(
    tmp_path: Path,
) -> None:
    root = _wiki(
        tmp_path,
        {
            "sources/fenced.md": (
                "# Real section — open hypothesis\n"
                "```markdown\n"
                "- Raw snapshot: raw/fake/v1.pdf\n"
                "# Fake heading — direct evidence\n"
                "## Evidence boundary\n"
                "- Direct evidence: fencedclaimtoken\n"
                "```\n"
                "visibleclaimtoken\n"
            )
        },
    )
    raw = root / "raw/fake/v1.pdf"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"fake")

    retriever = LocalWikiRetriever(root)
    visible = retriever.search(WikiQuery("visibleclaimtoken", 1))[0]

    assert visible.evidence_layer == "open hypothesis"
    assert visible.linked_raw_path is None
    assert retriever.search(WikiQuery("fencedclaimtoken", 5)) == ()


def test_tilde_fence_requires_same_character_and_sufficient_closing_length(
    tmp_path: Path,
) -> None:
    root = _wiki(
        tmp_path,
        {
            "sources/fenced.md": (
                "# Real section — open hypothesis\n"
                "~~~~ text\n"
                "shortclosetoken\n"
                "~~~\n"
                "differentclosetoken\n"
                "```\n"
                "stillfencedtoken\n"
                "~~~~~\n"
                "afterfencetoken\n"
            )
        },
    )

    retriever = LocalWikiRetriever(root)

    for token in ("shortclosetoken", "differentclosetoken", "stillfencedtoken"):
        assert retriever.search(WikiQuery(token, 5)) == ()
    assert retriever.search(WikiQuery("afterfencetoken", 1))


def test_longer_backtick_fence_closer_ends_fence(tmp_path: Path) -> None:
    root = _wiki(
        tmp_path,
        {
            "sources/fenced.md": (
                "# Real section — open hypothesis\n"
                "```\n"
                "insidefencetoken\n"
                "````\n"
                "outsidefencetoken\n"
            )
        },
    )

    retriever = LocalWikiRetriever(root)

    assert retriever.search(WikiQuery("insidefencetoken", 1)) == ()
    assert retriever.search(WikiQuery("outsidefencetoken", 1))


def test_evidence_phrase_in_non_label_heading_is_not_promoted(tmp_path: Path) -> None:
    root = _wiki(
        tmp_path,
        {"sources/negated.md": "# Not direct evidence\nnegatedlabeltoken\n"},
    )

    hit = LocalWikiRetriever(root).search(WikiQuery("negatedlabeltoken", 1))[0]

    assert hit.evidence_layer == "open hypothesis"


def test_nested_heading_inherits_parent_evidence_layer(tmp_path: Path) -> None:
    root = _wiki(
        tmp_path,
        {
            "sources/nested.md": (
                "# Source note\n"
                "## Measurements — direct evidence\n"
                "### Absorption details\n"
                "nestedmeasurementtoken\n"
            )
        },
    )

    hit = LocalWikiRetriever(root).search(WikiQuery("nestedmeasurementtoken", 1))[0]

    assert hit.evidence_layer == "direct evidence"


def test_long_line_snippet_keeps_the_matching_text(tmp_path: Path) -> None:
    root = _wiki(
        tmp_path,
        {
            "sources/long-match.md": (
                "# Measurements — direct evidence\n"
                + "x" * 180
                + " needletoken trailing\n"
            )
        },
    )

    hit = LocalWikiRetriever(root).search(
        WikiQuery("needletoken", 1, snippet_max_chars=64)
    )[0]

    assert len(hit.content) <= 64
    assert "needletoken" in hit.content
    assert (hit.line_start, hit.line_end) == (2, 2)


def test_snippet_uses_token_span_not_substring_inside_another_token(
    tmp_path: Path,
) -> None:
    root = _wiki(
        tmp_path,
        {
            "sources/span.md": (
                "# Measurement — direct evidence\n"
                + "hundred "
                + "x" * 160
                + " red is the exact token\n"
            )
        },
    )

    hit = LocalWikiRetriever(root).search(
        WikiQuery("red", 1, snippet_max_chars=64)
    )[0]

    assert " red " in hit.content
    assert "hundred" not in hit.content


def test_markdown_is_read_from_open_fd_across_path_swap(
    tmp_path: Path, monkeypatch
) -> None:
    root = _wiki(
        tmp_path,
        {"sources/race.md": "# Safe — direct evidence\nsafetoken\n"},
    )
    victim = root / "sources/race.md"
    backup = root / "sources/race-original.md"
    outside = tmp_path / "outside.md"
    outside.write_text("# Outside — direct evidence\noutsideescape\n", encoding="utf-8")
    swapped = False
    original_read_text = Path.read_text
    original_open = os.open

    def swap_path() -> None:
        nonlocal swapped
        if not swapped:
            victim.rename(backup)
            victim.symlink_to(outside)
            swapped = True

    def swapping_read_text(path: Path, *args, **kwargs):
        if path == victim:
            swap_path()
        return original_read_text(path, *args, **kwargs)

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is None:
            fd = original_open(path, flags, mode)
        else:
            fd = original_open(path, flags, mode, dir_fd=dir_fd)
        if path == "race.md":
            swap_path()
        return fd

    monkeypatch.setattr(Path, "read_text", swapping_read_text)
    monkeypatch.setattr(os, "open", swapping_open)

    with pytest.raises(ValueError, match="changed|symlink|race"):
        LocalWikiRetriever(root)


def test_regular_file_replacement_between_stat_and_open_is_rejected(
    tmp_path: Path, monkeypatch
) -> None:
    root = _wiki(
        tmp_path,
        {"sources/race.md": "# Safe — direct evidence\nsafetoken\n"},
    )
    victim = root / "sources/race.md"
    backup = root / "sources/race-original.md"
    swapped = False
    original_open = os.open

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "race.md" and not swapped:
            victim.rename(backup)
            victim.write_text(
                "# Replacement — direct evidence\noutsideescape\n",
                encoding="utf-8",
            )
            swapped = True
        if dir_fd is None:
            return original_open(path, flags, mode)
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swapping_open)

    with pytest.raises(ValueError, match="changed|race"):
        LocalWikiRetriever(root)


def test_platform_without_nofollow_fails_closed(tmp_path: Path, monkeypatch) -> None:
    root = _wiki(tmp_path)
    monkeypatch.delattr(os, "O_NOFOLLOW")

    with pytest.raises(RuntimeError, match="O_NOFOLLOW|platform"):
        LocalWikiRetriever(root)


@pytest.mark.parametrize(
    "query",
    [
        WikiQuery("valid", 1),
        WikiQuery("valid", 100, score_threshold=1.0, snippet_max_chars=8192),
    ],
)
def test_query_valid_boundaries(query: WikiQuery) -> None:
    assert query.max_results in {1, 100}


@pytest.mark.parametrize(
    "factory",
    [
        lambda: WikiQuery("   ", 1),
        lambda: WikiQuery("x" * (1024 * 1024), 1),
        lambda: WikiQuery(" ".join(f"token{index}" for index in range(300)), 1),
        lambda: WikiQuery("valid", 1, score_threshold=-0.01),
        lambda: WikiQuery("valid", 1, score_threshold=1.01),
        lambda: WikiQuery("valid", 1, score_threshold=float("nan")),
        lambda: WikiQuery("valid", 1, score_threshold=True),
        lambda: WikiQuery("valid", 1, snippet_max_chars=63),
        lambda: WikiQuery("valid", 1, snippet_max_chars=8193),
        lambda: WikiQuery("valid", 1, snippet_max_chars=True),
    ],
)
def test_query_rejects_invalid_score_and_snippet_boundaries(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_query_budget_accepts_normal_chinese_text() -> None:
    query = WikiQuery("红光 吸收 多重共振 发光", 5)

    assert query.text == "红光 吸收 多重共振 发光"


def test_wiki_index_limits_are_frozen_and_have_explicit_defaults() -> None:
    limits = WikiIndexLimits()

    assert limits == WikiIndexLimits(
        max_files=10_000,
        max_file_bytes=2 * 1024 * 1024,
        max_total_bytes=64 * 1024 * 1024,
        max_depth=16,
        max_sections=100_000,
        max_total_tokens=2_000_000,
        max_entries=100_000,
        max_name_bytes=4 * 1024 * 1024,
    )
    with pytest.raises(FrozenInstanceError):
        limits.max_files = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: WikiIndexLimits(max_files=0),
        lambda: WikiIndexLimits(max_file_bytes=True),
        lambda: WikiIndexLimits(max_total_bytes=0),
        lambda: WikiIndexLimits(max_depth=-1),
        lambda: WikiIndexLimits(max_sections=0),
        lambda: WikiIndexLimits(max_total_tokens=0),
        lambda: WikiIndexLimits(max_entries=0),
        lambda: WikiIndexLimits(max_name_bytes=0),
    ],
)
def test_wiki_index_limits_reject_invalid_values(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def _tracked_fds(monkeypatch) -> set[int]:
    opened: set[int] = set()
    original_open = os.open
    original_close = os.close

    def tracked_open(path, flags, mode=0o777, *, dir_fd=None):
        if dir_fd is None:
            fd = original_open(path, flags, mode)
        else:
            fd = original_open(path, flags, mode, dir_fd=dir_fd)
        opened.add(fd)
        return fd

    def tracked_close(fd: int) -> None:
        original_close(fd)
        opened.discard(fd)

    monkeypatch.setattr(os, "open", tracked_open)
    monkeypatch.setattr(os, "close", tracked_close)
    return opened


def _limit_case(tmp_path: Path, case: str) -> tuple[Path, WikiIndexLimits, str]:
    defaults = WikiIndexLimits()
    if case in {"large", "sparse"}:
        root = _wiki(tmp_path)
        page = root / "sources/page.md"
        page.parent.mkdir()
        if case == "large":
            page.write_bytes(b"# Large\n" + b"x" * 512)
        else:
            with page.open("wb") as stream:
                stream.truncate(4096)
        return root, replace(defaults, max_file_bytes=256), "max_file_bytes"
    if case == "total":
        root = _wiki(
            tmp_path,
            {
                "sources/a.md": "# A\n" + "a" * 80,
                "sources/b.md": "# B\n" + "b" * 80,
            },
        )
        declared = sum(path.stat().st_size for path in (root / "sources").iterdir())
        return root, replace(defaults, max_total_bytes=declared - 1), "max_total_bytes"
    if case == "files":
        root = _wiki(
            tmp_path,
            {"sources/a.md": "# A\na", "sources/b.md": "# B\nb"},
        )
        return root, replace(defaults, max_files=1), "max_files"
    if case == "depth":
        root = _wiki(tmp_path, {"sources/a/b/page.md": "# Deep\ndeep"})
        return root, replace(defaults, max_depth=1), "max_depth"
    if case == "sections":
        root = _wiki(tmp_path, {"sources/page.md": "# One\none\n# Two\ntwo"})
        return root, replace(defaults, max_sections=1), "max_sections"
    if case == "tokens":
        root = _wiki(
            tmp_path,
            {"sources/page.md": "# Heading\none two three four five"},
        )
        return root, replace(defaults, max_total_tokens=3), "max_total_tokens"
    raise AssertionError(f"unknown limit case: {case}")


@pytest.mark.parametrize(
    "case", ["large", "sparse", "total", "files", "depth", "sections", "tokens"]
)
def test_index_limits_fail_closed_and_release_all_fds(
    tmp_path: Path, monkeypatch, case: str
) -> None:
    root, limits, match = _limit_case(tmp_path, case)
    opened = _tracked_fds(monkeypatch)

    with pytest.raises(ValueError, match=match):
        LocalWikiRetriever(root, limits=limits)

    assert opened == set()


def test_failed_construction_does_not_publish_partial_index(tmp_path: Path) -> None:
    root, limits, _ = _limit_case(tmp_path, "sections")
    retriever = object.__new__(LocalWikiRetriever)

    with pytest.raises(ValueError, match="max_sections"):
        retriever.__init__(root, limits=limits)

    assert not hasattr(retriever, "_agents")
    assert not hasattr(retriever, "_index")
    assert not hasattr(retriever, "_sections")


def test_section_limit_stops_heading_scan_without_materializing_all_sections(
    tmp_path: Path, monkeypatch
) -> None:
    root = _wiki(
        tmp_path,
        {
            "sources/page.md": "\n".join(
                f"# Heading {index}\nvalue{index}" for index in range(100)
            )
        },
    )
    original = local_wiki_module._HEADING_RE

    class CountingHeadingPattern:
        calls = 0

        @classmethod
        def match(cls, value: str):
            cls.calls += 1
            return original.match(value)

    monkeypatch.setattr(local_wiki_module, "_HEADING_RE", CountingHeadingPattern)
    with pytest.raises(ValueError, match="max_sections"):
        LocalWikiRetriever(
            root,
            limits=replace(WikiIndexLimits(), max_sections=1),
        )

    assert CountingHeadingPattern.calls <= 5


@pytest.mark.parametrize(
    "primary", [RuntimeError("read failed"), asyncio.CancelledError("cancelled")]
)
def test_read_exception_and_cancellation_release_all_fds(
    tmp_path: Path, monkeypatch, primary: BaseException
) -> None:
    root = _wiki(tmp_path, {"sources/page.md": "# Page\ncontent"})
    opened = _tracked_fds(monkeypatch)

    def failing_read(fd: int, size: int) -> bytes:
        raise primary

    monkeypatch.setattr(os, "read", failing_read)
    with pytest.raises(type(primary)) as raised:
        LocalWikiRetriever(root)

    assert raised.value is primary
    assert opened == set()


def test_anchors_count_toward_file_and_total_byte_limits(tmp_path: Path) -> None:
    root = _wiki(tmp_path)
    defaults = WikiIndexLimits()
    with pytest.raises(ValueError, match="max_files"):
        LocalWikiRetriever(root, limits=replace(defaults, max_files=1))
    anchor_bytes = sum((root / name).stat().st_size for name in ("AGENTS.md", "index.md"))
    with pytest.raises(ValueError, match="max_total_bytes"):
        LocalWikiRetriever(
            root,
            limits=replace(defaults, max_total_bytes=anchor_bytes - 1),
        )


def test_entry_limit_stops_scandir_at_limit_plus_one(tmp_path: Path, monkeypatch) -> None:
    root = _wiki(tmp_path)
    sources = root / "sources"
    sources.mkdir()
    for index in range(20):
        (sources / f"ignored-{index}.bin").write_bytes(b"x")
    original_scandir = os.scandir
    source_inode = sources.stat().st_ino
    consumed = 0

    class CountingIterator:
        def __init__(self, inner):
            self.inner = inner

        def __enter__(self):
            self.inner.__enter__()
            return self

        def __exit__(self, *args):
            return self.inner.__exit__(*args)

        def __iter__(self):
            return self

        def __next__(self):
            nonlocal consumed
            value = next(self.inner)
            consumed += 1
            return value

    def counting_scandir(fd):
        iterator = original_scandir(fd)
        return CountingIterator(iterator) if os.fstat(fd).st_ino == source_inode else iterator

    monkeypatch.setattr(os, "scandir", counting_scandir)
    with pytest.raises(ValueError, match="max_entries"):
        LocalWikiRetriever(
            root,
            limits=replace(WikiIndexLimits(), max_entries=2),
        )
    assert consumed == 3


def test_entry_name_byte_budget_counts_nonmarkdown_and_directories(tmp_path: Path) -> None:
    root = _wiki(tmp_path)
    sources = root / "sources"
    sources.mkdir()
    (sources / "directory-name").mkdir()
    (sources / "ignored-name.bin").write_bytes(b"x")
    with pytest.raises(ValueError, match="max_name_bytes"):
        LocalWikiRetriever(
            root,
            limits=replace(WikiIndexLimits(), max_name_bytes=10),
        )


@pytest.mark.parametrize(
    "primary", [RuntimeError("scan failed"), asyncio.CancelledError("cancelled")]
)
def test_scandir_exception_and_cancellation_release_all_fds(
    tmp_path: Path, monkeypatch, primary: BaseException
) -> None:
    root = _wiki(tmp_path)
    (root / "sources").mkdir()
    opened = _tracked_fds(monkeypatch)

    class FailingScandir:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return self

        def __next__(self):
            raise primary

    monkeypatch.setattr(os, "scandir", lambda fd: FailingScandir())
    with pytest.raises(type(primary)) as raised:
        LocalWikiRetriever(root)

    assert raised.value is primary
    assert opened == set()
