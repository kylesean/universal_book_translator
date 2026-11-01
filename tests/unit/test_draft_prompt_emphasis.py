from __future__ import annotations

from collections.abc import Callable

import pytest

from ubt.core.router.prompts import (
    build_hybrid_draft_prompt,
    build_macro_chunk_draft_prompt,
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
_MARKED = "⟦B⟧Effects reach outside.⟦/B⟧"
_CLEAN = "Effects reach outside."


@pytest.mark.parametrize("build", _BUILDERS)
def test_every_builder_states_the_emphasis_rule(build: _Builder) -> None:
    system, user = build(source_text="plain source")
    assert "⟦B⟧" in system + user


@pytest.mark.parametrize("build", _BUILDERS)
def test_the_short_source_hint_ignores_the_markers(build: _Builder) -> None:
    system, user = build(source_text=_MARKED)
    assert "equally short" not in system + user


@pytest.mark.parametrize("build", _BUILDERS)
def test_the_short_source_hint_still_fires_on_a_marked_short_label(build: _Builder) -> None:
    marked = "⟦B⟧" + ("A" * 58) + "⟦/B⟧"
    system, user = build(source_text=marked)
    assert "equally short" in system + user


@pytest.mark.parametrize("build", _BUILDERS)
def test_draft_source_from_prompt_strips_the_markers(build: _Builder) -> None:
    system, user = build(source_text=_MARKED)
    assert draft_source_from_prompt(system + "\n" + user) == _CLEAN


@pytest.mark.parametrize("build", _BUILDERS)
def test_a_few_shot_reference_is_retained_for_a_marked_source(build: _Builder) -> None:
    reference = "REFERENCE_SHOULD_APPEAR"
    marked_system, marked_user = build(source_text=_MARKED, few_shot_reference=reference)
    assert reference in marked_system + marked_user
    plain_system, plain_user = build(source_text=_CLEAN, few_shot_reference=reference)
    assert reference in plain_system + plain_user


def test_the_macro_builder_states_the_emphasis_rule() -> None:
    system, user = build_macro_chunk_draft_prompt([("b1", _MARKED)])
    assert "⟦B⟧" in system + user


def test_the_macro_builder_retains_a_few_shot_reference_for_a_marked_block() -> None:
    reference = "REFERENCE_SHOULD_APPEAR"
    system, user = build_macro_chunk_draft_prompt(
        [("b1", "plain"), ("b2", _MARKED)], few_shot_reference=reference
    )
    assert reference in system + user
