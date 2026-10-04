"""A short heading/label source must carry a "do not expand" prompt clause.

Regression: a local model expanded a heading ("2.1. Effects") into a whole
paragraph, which the QE length gate then quarantined. The draft builders now
append an in-kind constraint for a short, punctuation-free source, and
``draft_source_from_prompt`` must still recover the source span (the clause is
placed after the anchor the parser reads).
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


@pytest.mark.parametrize("build", _BUILDERS)
def test_a_short_source_gets_the_no_expand_clause(build: _Builder) -> None:
    system, user = build(source_text=_SHORT)
    assert "equally short" in system + user


@pytest.mark.parametrize("build", _BUILDERS)
def test_a_full_sentence_does_not_get_the_clause(build: _Builder) -> None:
    system, user = build(source_text=_LONG)
    assert "equally short" not in system + user


@pytest.mark.parametrize("build", _BUILDERS)
@pytest.mark.parametrize("source", [_SHORT, _LONG])
def test_the_source_span_is_recovered_with_the_clause_present(build: _Builder, source: str) -> None:
    system, user = build(source_text=source)
    assert draft_source_from_prompt(system + "\n" + user) == source
