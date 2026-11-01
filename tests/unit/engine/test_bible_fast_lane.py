from __future__ import annotations

import pytest

from ubt.core.memory.abbreviation_miner import mine_abbreviations_stream
from ubt.core.memory.character_miner import mine_characters_stream

pytestmark = pytest.mark.fast


_EXPANSIONS = [
    "We introduce the Bidirectional Encoder Representations from Transformers (BERT).",
    "Our Multi-Head Attention (MHA) layer differs from the Named Entity Recognition (NER) head.",
]


def test_the_abbreviation_miner_yields_entries_without_a_model() -> None:
    found = mine_abbreviations_stream(_EXPANSIONS)

    assert found, "the 0-token miner must yield entries with no model call"
    aliases = {a for entry in found for a in entry["aliases"]}
    assert {"BERT", "MHA", "NER"} & aliases, aliases


def test_the_character_miner_needs_no_model_either() -> None:
    found = mine_characters_stream(_EXPANSIONS, source_lang="en", allow_bare_tokens=False)
    assert isinstance(found, list)


def _source_of_bible_stage() -> str:
    import inspect

    from ubt.core.engine.stages import bible

    return inspect.getsource(bible)


def test_the_bible_stage_calls_every_terminology_source() -> None:
    source = _source_of_bible_stage()

    assert "= mine_abbreviations_stream(" in source
    assert "= mine_characters_stream(" in source
    assert "= await extract_skeleton_terms_llm(" in source
    assert "= await backfill_abbreviation_translations(" in source
    assert "if not fast_lane:" not in source


def test_an_explicit_exec_mode_outranks_the_page_probe() -> None:
    source = _source_of_bible_stage()

    assert 'exec_mode == "auto"' in source, (
        "the page-count probe must be limited to the undecided auto case"
    )
