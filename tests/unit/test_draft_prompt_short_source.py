"""A heading/label source must carry a "do not expand" prompt clause.

Regression: a local model expanded a heading ("2.1. Effects") into a whole
paragraph, which the QE length gate then quarantined. The draft builders append
an in-kind constraint for a short, punctuation-free source, and
``draft_source_from_prompt`` must still recover the source span (the clause is
placed after the anchor the parser reads).

A second regression (``pdf_main#b0003``): a 96-char paper title passed the
60-char heuristic untouched, so the model expanded it into a title plus an
abstract summary. The block's own ``heading`` type now fires the clause
regardless of length, up to a cap that still protects a mislabelled paragraph.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from ubt.core.router.prompts import (
    build_hybrid_draft_prompt,
    build_minimal_draft_prompt,
    build_rich_draft_prompt,
    draft_source_from_prompt,
)

pytestmark = pytest.mark.fast

_Builder = Callable[..., tuple[str, str]]
_BUILDERS: tuple[_Builder, ...] = (
    build_minimal_draft_prompt,
    build_hybrid_draft_prompt,
    build_rich_draft_prompt,
)
_SHORT = "2.1. Effects"
_LONG = "Failure. The effects a component installs reach outside the context that tracks them."
#: A real paper title (96 chars, no terminal punctuation) that the character
#: heuristic misses: the model expanded it into a title-plus-abstract-summary
#: draft, which the QE length gate quarantined.
_TITLE = (
    "DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure "
    "for Effective Agentic Training at Scale"
)


@pytest.mark.parametrize("build", _BUILDERS)
def test_a_short_source_gets_the_no_expand_clause(build: _Builder) -> None:
    system, user = build(source_text=_SHORT)
    assert "equally short" in system + user


@pytest.mark.parametrize("build", _BUILDERS)
def test_a_full_sentence_does_not_get_the_clause(build: _Builder) -> None:
    system, user = build(source_text=_LONG)
    assert "equally short" not in system + user


@pytest.mark.parametrize("build", _BUILDERS)
def test_a_long_title_without_a_type_still_misses_the_clause(build: _Builder) -> None:
    """Documents the gap the block type closes: >60 chars, no punctuation."""
    system, user = build(source_text=_TITLE)
    assert "equally short" not in system + user


@pytest.mark.parametrize("build", _BUILDERS)
def test_a_heading_block_type_gets_the_clause_regardless_of_length(build: _Builder) -> None:
    system, user = build(source_text=_TITLE, block_type="heading")
    assert "equally short" in system + user


@pytest.mark.parametrize("build", _BUILDERS)
def test_a_non_heading_block_type_still_uses_the_length_heuristic(build: _Builder) -> None:
    system, user = build(source_text=_TITLE, block_type="narrative")
    assert "equally short" not in system + user


@pytest.mark.parametrize("build", _BUILDERS)
def test_a_mislabelled_overlong_heading_is_not_told_to_render_short(build: _Builder) -> None:
    overlong = "T " + ("x" * 260)
    system, user = build(source_text=overlong, block_type="heading")
    assert "equally short" not in system + user


@pytest.mark.parametrize("build", _BUILDERS)
@pytest.mark.parametrize("source", [_SHORT, _LONG, _TITLE])
def test_the_source_span_is_recovered_with_the_clause_present(build: _Builder, source: str) -> None:
    for block_type in (None, "heading"):
        system, user = build(source_text=source, block_type=block_type)
        assert draft_source_from_prompt(system + "\n" + user) == source
