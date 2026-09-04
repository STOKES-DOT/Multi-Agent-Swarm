from __future__ import annotations

from pathlib import Path

import pytest

from multi_agent_pso.protocols import WikiHit, WikiRetriever
from multi_agent_pso.retrieval import LocalWikiRetriever, WikiQuery


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
                "rawsignal `raw/source/v1.pdf`\n"
            ),
            "mocs/moc.md": (
                "# Navigation — cross-paper synthesis\n"
                "rawsignal `raw/source/v1.pdf`\n"
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
                "escapetoken `raw/../../outside.pdf`\n"
                "also [raw](../raw/link.pdf)\n"
            )
        },
    )
    raw = root / "raw"
    raw.mkdir()
    (raw / "link.pdf").symlink_to(outside)

    hit = LocalWikiRetriever(root).search(WikiQuery("escapetoken", 1))[0]

    assert hit.linked_raw_path is None


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
