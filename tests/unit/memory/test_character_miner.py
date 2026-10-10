"""Unit tests for deterministic 0-token character / proper-noun mining."""

from __future__ import annotations

import pytest

from ubt.core.memory.character_miner import (
    _name_core,
    _strip_cjk_prefixes,
    get_mining_config,
    mine_characters,
    mine_characters_stream,
)

pytestmark = pytest.mark.fast


def test_get_mining_config_resolution() -> None:
    en_cfg = get_mining_config("en")
    assert en_cfg.code == "en"
    assert en_cfg.allow_bare_tokens_default is True

    de_cfg = get_mining_config("de-DE")
    assert de_cfg.code == "de"
    assert de_cfg.allow_bare_tokens_default is False

    zh_cfg = get_mining_config("zh_CN")
    assert zh_cfg.code == "zh"
    assert zh_cfg.allow_bare_tokens_default is False

    ja_cfg = get_mining_config("ja")
    assert ja_cfg.code == "ja"

    ko_cfg = get_mining_config("ko")
    assert ko_cfg.code == "cjk"

    unknown_cfg = get_mining_config("xx_YY")
    assert unknown_cfg.code == "en"


def test_name_core_and_cjk_prefix_stripping() -> None:
    assert _name_core("Mr. Fitzwilliam Darcy", {"de", "von"}) == "Darcy"
    assert _name_core("Baron von Richthofen", {"von", "zu"}) == "Richthofen"
    assert _name_core("Dr. Watson", set()) == "Watson"

    # CJK prefix stripping
    prefixes = ["告诉", "和", "一位"]
    assert _strip_cjk_prefixes("告诉王", prefixes) == "王"
    assert _strip_cjk_prefixes("和林", prefixes) == "林"
    assert _strip_cjk_prefixes("一位李", prefixes) == "李"
    # Kana-only string without ideographs is untouched
    assert _strip_cjk_prefixes("さくら", ["さ"]) == "さくら"


def test_mine_characters_english_honorifics() -> None:
    text = (
        "Mr. Darcy visited the estate. Later, Dr. Watson arrived with Captain Hastings. "
        "Mr. Darcy was seen speaking to Dr. Watson again."
    )
    results = mine_characters(text, source_lang="en", min_freq=2)
    sources = [r["source"] for r in results]

    assert "Mr. Darcy" in sources
    assert "Dr. Watson" in sources
    assert "Captain Hastings" in sources

    # Check frequency backfill: Mr. Darcy appeared twice
    darcy_entry = next(r for r in results if r["source"] == "Mr. Darcy")
    assert darcy_entry["frequency"] >= 2
    assert darcy_entry["aliases"] == ["Darcy"]
    assert darcy_entry["kind"] == "person"


def test_mine_characters_german_suppresses_bare_tokens() -> None:
    text = (
        "Herr Müller ging in das Haus. Frau Schmidt kam auch. "
        "Das Buch und der Tisch waren sehr alt. Herr Müller sprach lange."
    )
    # Even if min_freq is 1, German nouns should not be harvested as bare person tokens
    results = mine_characters(text, source_lang="de", min_freq=1)
    sources = [r["source"] for r in results]

    assert "Herr Müller" in sources
    assert "Frau Schmidt" in sources
    # Common capitalized German nouns must NOT be mined
    assert "Buch" not in sources
    assert "Tisch" not in sources
    assert "Haus" not in sources


def test_mine_characters_chinese_postposed_honorifics() -> None:
    text = (
        "随后，王小明先生来到了北京。旁边的先生并没有说话。见到王小明先生后，"
        "请张老师和李教授正在进行热烈的研讨。"
    )
    results = mine_characters(text, source_lang="zh", min_freq=1)
    sources = [r["source"] for r in results]

    assert "王小明" in sources
    assert "张" in sources
    assert "李" in sources
    # "旁边的先生" ends with 的 before 先生, should be rejected
    assert "旁边" not in sources
    assert "旁边的" not in sources

    wang_entry = next(r for r in results if r["source"] == "王小明")
    assert wang_entry["frequency"] >= 2


def test_mine_characters_japanese_postposed_honorifics() -> None:
    text = (
        "山田さんこんにちは。佐藤様のご案内です。お疲れ様でした。皆様お元気ですか。"
        "さくらちゃんも来ました。"
    )
    results = mine_characters(text, source_lang="ja", min_freq=1)
    sources = [r["source"] for r in results]

    assert "山田" in sources
    assert "佐藤" in sources
    assert "さくら" in sources
    # Generic polite compounds must be rejected via not_names
    assert "お疲れ" not in sources
    assert "皆" not in sources


def test_mine_characters_bare_capitalized_tokens_mid_sentence() -> None:
    # Elizabeth appears mid-sentence 3 times.
    # January is a month (not_names).
    # Chapter is a structure word (not_names).
    # Starter appears only at the start of sentences.
    text = (
        "Starter was the first word. Then Elizabeth smiled warmly. "
        "Starter was repeated here. Later Elizabeth walked home. "
        "Starter began again. Finally Elizabeth said goodbye. "
        "It was in January, as described in Chapter five."
    )
    results = mine_characters(text, source_lang="en", min_freq=3)
    sources = [r["source"] for r in results]

    assert "Elizabeth" in sources
    assert "Starter" not in sources  # Not mid-sentence
    assert "January" not in sources  # in _EN_NOT_NAMES
    assert "Chapter" not in sources  # in _EN_NOT_NAMES


def test_mine_characters_max_entries_and_streaming() -> None:
    blocks = [
        "Mr. Darcy went to Pemberley. ",
        "Dr. Watson stayed at Baker Street. ",
        "Professor Moriarty was scheming. ",
        "Sir Walter Elliot was looking in the mirror. ",
    ]
    stream_results = mine_characters_stream(blocks, source_lang="en", max_entries=2)
    assert len(stream_results) <= 2
    assert stream_results[0]["source"] == "Mr. Darcy"
