"""The rehearsal echo must read a *repair* prompt back to its source span.

The ``--dry-run`` provider answers every request through
``draft_source_from_prompt``, repair rounds included. That function only knew
the draft builders' anchors, so a repair prompt fell through to
``prompt.strip()`` and the "repaired" text became the prompt scaffold:

    ### [模拟翻译] Terminology Glossary ... ### Original Source ... ### Previous Draft ...

The QE pass then flagged the block for *prompt scaffold leaked into the target*
-- correctly, but the defect was the rehearsal's, not a model's. Measured on the
corpus as one quarantined block each in `paper-2609-22978` and
`book-chapter-3`.

Every repair builder is covered: the three metaprompt shapes use
``### Original Source`` headings, and the minimal one inlines the source after
``Source <language> text:`` with no heading at all.
"""

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
#: A math-bearing paragraph: the shape whose quarantine exposed this (the mock
#: cannot emit ``$...$``, so it also trips the undelimited-math rule -- but the
#: scaffold leak was a separate, avoidable flag on the same block).
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
    """The whole point: no heading, glossary or critique text survives the echo."""
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
    """The last-resort branch stays: a prompt this module did not build is not
    silently truncated on a guessed heading."""
    assert draft_source_from_prompt("translate this") == "translate this"
