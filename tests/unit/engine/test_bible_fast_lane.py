"""The bible stage builds terminology on the fast lane too.

`is_fast_lane_eligible` is a page-count probe (<= 20 pages, born-digital). It was
applied to *every* bible step, so the short route produced no terminology by any
path: the 0-token miners were skipped, the skeleton extraction was skipped, and
the backfill was skipped. The result was not merely "less consistent" — the
short route had **no** terminology mechanism, and ``deterministic_glossary``
(which the short route turns on) was fed an empty table, so it enforced nothing.
Measured on `vit`: 0 measurable terms on the short route against 28 at 0.9545 on
the long route; after the fix the short route measures 30 terms at 0.9791, at a
lower cost, because its saving comes from the batch API rather than from
skipping the bible.

The same gate also silently discarded an explicit `exec_mode`: the route says
`long`, the stage re-derived `short` from the file probe, and an
`--exec-mode long` run mined nothing.

These tests assert *behaviour* where they can — a source-text assertion cannot
tell a gate on one call from the indentation of the block above it.
"""

from __future__ import annotations

import pytest

from ubt.core.memory.abbreviation_miner import mine_abbreviations_stream
from ubt.core.memory.character_miner import mine_characters_stream

pytestmark = pytest.mark.fast


#: Text with the "Full Name (ACRONYM)" shape the miner keys on.
_EXPANSIONS = [
    "We introduce the Bidirectional Encoder Representations from Transformers (BERT).",
    "Our Multi-Head Attention (MHA) layer differs from the Named Entity Recognition (NER) head.",
]


def test_the_abbreviation_miner_yields_entries_without_a_model() -> None:
    """The miner is the fast lane's own tool: it must run there.

    This pins the *capability* the gate denied. `mine_abbreviations_stream` is
    pure Python over the source text; measured at 0.1-1.4 ms for a whole
    document, so there is nothing for a "fast" lane to save.
    """
    found = mine_abbreviations_stream(_EXPANSIONS)

    assert found, "the 0-token miner must yield entries with no model call"
    aliases = {a for entry in found for a in entry["aliases"]}
    assert {"BERT", "MHA", "NER"} & aliases, aliases


def test_the_character_miner_needs_no_model_either() -> None:
    found = mine_characters_stream(_EXPANSIONS, source_lang="en", allow_bare_tokens=False)
    # A scientific paper has no recurring person names at the default
    # min_freq, so an empty result is correct — the point is that the call
    # completes and returns a list rather than being skipped.
    assert isinstance(found, list)


def _source_of_bible_stage() -> str:
    import inspect

    from ubt.core.engine.stages import bible

    return inspect.getsource(bible)


def test_the_bible_stage_calls_every_terminology_source() -> None:
    """Mining, skeleton extraction and backfill must all run unconditionally.

    Each is a call, not merely an import, and none may sit behind a fast-lane
    guard: the three together are the only path by which the short route can
    obtain a glossary at all.
    """
    source = _source_of_bible_stage()

    assert "= mine_abbreviations_stream(" in source
    assert "= mine_characters_stream(" in source
    assert "= await extract_skeleton_terms_llm(" in source
    assert "= await backfill_abbreviation_translations(" in source
    # No `if not fast_lane:` may remain in the stage body: the lane no longer
    # gates terminology, only the probe's own decision and the cache key.
    assert "if not fast_lane:" not in source


def test_an_explicit_exec_mode_outranks_the_page_probe() -> None:
    r"""`--exec-mode long` must not be re-decided by the fast-lane probe.

    The route already applied the explicit mode (`router_mode.decide` returns
    `long`), and the stage then called `is_fast_lane_eligible` anyway, so a
    `--exec-mode long` run logged "Fast lane enabled" and mined nothing. The
    probe now fills in only the undecided `auto` case.
    """
    source = _source_of_bible_stage()

    assert 'exec_mode == "auto"' in source, (
        "the page-count probe must be limited to the undecided auto case"
    )
