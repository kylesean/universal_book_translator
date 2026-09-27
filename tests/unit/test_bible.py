"""Tests for the bible stage's entry split (``ubt/core/engine/stages/bible.py``)."""

from __future__ import annotations

from ubt.core.engine.stages.bible import _split_bible_entries
from ubt.core.memory.bible import BibleEntry


def test_untranslated_person_and_place_are_not_in_the_abbreviation_channel() -> None:
    """A name whose backfill failed must not become an "unchanged abbreviation".

    Regression: the split keyed only on whether ``translation`` was empty, with
    no ``kind`` routing, so an untranslated person/place was rendered by the
    abbreviation channel's "keep abbreviation unchanged" prompt.
    """
    entries = [
        BibleEntry(source="Alvarez", translation="", kind="person", aliases=["Alvarez"]),
        BibleEntry(source="Lyon", translation="", kind="place", aliases=["Lyon"]),
        BibleEntry(source="attention", translation="注意力", kind="term"),
        BibleEntry(source="LSTM", translation="", kind="term", aliases=["LSTM"]),
    ]
    glossary, abbreviations = _split_bible_entries(entries)

    assert [g["source"] for g in glossary] == ["attention"]
    assert [a["source"] for a in abbreviations] == ["LSTM"]
