from __future__ import annotations

from collections.abc import Callable

import pytest

from ubt.core.router.prompts import (
    build_hybrid_repair_prompt,
    build_minimal_repair_prompt,
    build_rich_repair_prompt,
    draft_source_from_prompt,
)

pytestmark = pytest.mark.fast

_RepairBuilder = Callable[..., tuple[str, str]]
_BUILDERS: tuple[_RepairBuilder, ...] = (
    build_minimal_repair_prompt,
    build_hybrid_repair_prompt,
    build_rich_repair_prompt,
)
_SOURCE = "where β0 is obtained from Eq. (A.7) using Fth and Ag from Eqs. (A.8), (A.9)."


@pytest.mark.parametrize("builder", _BUILDERS, ids=[b.__name__ for b in _BUILDERS])
def test_the_echo_recovers_the_source_and_nothing_else(builder: _RepairBuilder) -> None:
    _system, user = builder(
        source_text=_SOURCE,
        draft_text=f"[模拟翻译] {_SOURCE}",
        error_flags=["Undelimited math: source carries math symbols"],
        glossary_table="β0 -> beta",
    )

    assert draft_source_from_prompt(user) == _SOURCE


@pytest.mark.parametrize("builder", _BUILDERS, ids=[b.__name__ for b in _BUILDERS])
def test_the_echo_never_replays_the_scaffold(builder: _RepairBuilder) -> None:
    _system, user = builder(
        source_text=_SOURCE,
        draft_text=f"[模拟翻译] {_SOURCE}",
        error_flags=["Undelimited math: source carries math symbols"],
        glossary_table="β0 -> beta",
    )

    echoed = draft_source_from_prompt(user)

    for scaffold in ("###", "Terminology Glossary", "Translation Bible", "Previous Draft"):
        assert scaffold not in echoed, f"{builder.__name__} leaked {scaffold!r}: {echoed[:200]}"


def test_an_unrecognised_prompt_still_falls_back_to_itself() -> None:
    assert draft_source_from_prompt("translate this") == "translate this"
