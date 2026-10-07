"""Macro-chunk extraction must not let a duplicate block id replace the real one.

The model emits one ``<block id="...">...</block>`` per micro-block, in document
order. A later duplicate can appear from chunk overlap, a leaked reasoning
draft, or an echo of format documentation in the source; keeping the *first*
emission stops it from silently overwriting the genuine translation.
"""

from __future__ import annotations

import logging

import pytest

from ubt.core.router.extractor import TranslationOutputExtractor

pytestmark = pytest.mark.fast


def test_duplicate_block_id_keeps_the_first_emission() -> None:
    raw = (
        '<block id="b1">first translation</block>'
        '<block id="b2">second</block>'
        '<block id="b1">a later duplicate</block>'
    )
    assert TranslationOutputExtractor.extract_macro_blocks(raw) == {
        "b1": "first translation",
        "b2": "second",
    }


def test_first_non_empty_emission_wins_over_an_empty_earlier_one() -> None:
    raw = '<block id="b1"></block><block id="b1">real</block>'
    assert TranslationOutputExtractor.extract_macro_blocks(raw) == {"b1": "real"}


def test_duplicate_emission_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    raw = '<block id="b1">a</block><block id="b1">b</block>'
    with caplog.at_level(logging.WARNING, logger="ubt.core.router.extractor"):
        TranslationOutputExtractor.extract_macro_blocks(raw)
    assert "duplicate block id" in caplog.text
    assert "b1" in caplog.text


def test_envelope_entities_are_unescaped() -> None:
    raw = '<block id="b1">AT&amp;T &lt;x&gt;</block>'
    assert TranslationOutputExtractor.extract_macro_blocks(raw) == {"b1": "AT&T <x>"}
