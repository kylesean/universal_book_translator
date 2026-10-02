"""The multilingual boilerplate catalog: language-keyed compiled patterns.

The catalog is the single source of publisher-disclaimer, running-header, page
marker and photo-credit patterns. Two facts are load-bearing: lookups are keyed
by the *primary* language subtag (a valid ``zh-Hans`` must not silently fall
back to English), and an unknown language falls back to the English set rather
than returning nothing. Running headers also carry a literal keyword used as a
cheap prefilter, which is why the keyword lookup exists at all.
"""

from __future__ import annotations

import pytest

from ubt.core.cleaners.boilerplate_catalog import BoilerplateCatalog, _primary_lang_code

pytestmark = pytest.mark.fast


# --------------------------------------------------------------------------- #
# _primary_lang_code
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("lang", "expected"),
    [
        ("", "en"),
        ("zh", "zh"),
        ("zh-Hans", "zh"),
        ("zh_CN", "zh"),
        ("en-US", "en"),
        ("  FR  ", "fr"),
        ("DE", "de"),
        ("pt-BR", "pt"),
    ],
)
def test_primary_lang_code(lang: str, expected: str) -> None:
    assert _primary_lang_code(lang) == expected


# --------------------------------------------------------------------------- #
# supported languages / keywords
# --------------------------------------------------------------------------- #


def test_the_catalog_covers_the_declared_languages() -> None:
    assert set(BoilerplateCatalog.get_supported_languages()) >= {
        "en",
        "fr",
        "de",
        "es",
        "ja",
        "ko",
        "zh",
        "ru",
    }


@pytest.mark.parametrize(
    ("lang", "keyword"),
    [
        ("en", "CHAPTER"),
        ("fr", "CHAPITRE"),
        ("de", "KAPITEL"),
        ("ru", "\u0413\u041b\u0410\u0412\u0410"),
    ],
)
def test_running_header_keywords_are_the_literal_chapter_word(lang: str, keyword: str) -> None:
    assert keyword in BoilerplateCatalog.get_running_header_keywords(lang)


def test_spanish_keywords_cover_both_accent_spellings() -> None:
    keywords = BoilerplateCatalog.get_running_header_keywords("es")
    assert "CAP\u00cdTULO" in keywords
    assert "CAPITULO" in keywords


def test_an_unknown_language_falls_back_to_english_keywords() -> None:
    assert BoilerplateCatalog.get_running_header_keywords("xx") == (
        BoilerplateCatalog.get_running_header_keywords("en")
    )


def test_a_region_subtag_uses_the_primary_languages_patterns() -> None:
    regional = BoilerplateCatalog.get_legal_pattern("zh-Hans")
    base = BoilerplateCatalog.get_legal_pattern("zh")
    assert regional is not None
    assert base is not None
    assert regional.pattern == base.pattern


# --------------------------------------------------------------------------- #
# legal disclaimers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("lang", "text"),
    [
        ("en", "Copyright 2020 Cengage Learning. All rights reserved"),
        ("en", "*** START OF THIS PROJECT GUTENBERG EBOOK MOBY DICK ***"),
        ("fr", "Tous droits r\u00e9serv\u00e9s"),
        ("zh", "\u7248\u6743\u6240\u6709\uff0c\u4fb5\u6743\u5fc5\u7a76"),
    ],
)
def test_legal_disclaimers_match(lang: str, text: str) -> None:
    pattern = BoilerplateCatalog.get_legal_pattern(lang)
    assert pattern is not None
    assert pattern.search(text) is not None


def test_ordinary_prose_does_not_match_a_disclaimer() -> None:
    pattern = BoilerplateCatalog.get_legal_pattern("en")
    assert pattern is not None
    assert pattern.search("just ordinary prose about copyright law") is None


@pytest.mark.parametrize("lang", ["xx", ""])
def test_an_unknown_language_falls_back_to_english_disclaimers(lang: str) -> None:
    pattern = BoilerplateCatalog.get_legal_pattern(lang)
    assert pattern is not None
    assert pattern.search("Copyright 2020 Pearson Education") is not None


# --------------------------------------------------------------------------- #
# running headers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text",
    ["CHAPTER 3 Introduction 42", "Overview CHAPTER 5 42", "42 CHAPTER 3 Overview"],
)
def test_running_headers_match(text: str) -> None:
    pattern = BoilerplateCatalog.get_running_header_pattern("en")
    assert pattern is not None
    assert pattern.search(text) is not None


def test_a_chapter_line_without_a_page_number_is_not_a_running_header() -> None:
    # The shape requires the trailing page number; a real chapter title is kept.
    pattern = BoilerplateCatalog.get_running_header_pattern("en")
    assert pattern is not None
    assert pattern.search("CHAPTER 3") is None


def test_chinese_running_headers_match() -> None:
    pattern = BoilerplateCatalog.get_running_header_pattern("zh")
    assert pattern is not None
    assert pattern.search("\u7b2c 3 \u7ae0 \u6982\u8ff0 42") is not None


def test_an_unknown_language_still_gets_a_running_header_pattern() -> None:
    assert BoilerplateCatalog.get_running_header_pattern("xx") is not None


# --------------------------------------------------------------------------- #
# page markers / photo credits
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("lang", "text"),
    [
        ("en", "Page 5"),
        ("zh", "\u7b2c 5 \u9875"),
        ("es", "P\u00e1gina 5"),
        ("ja", "\u30da\u30fc\u30b8 5"),
    ],
)
def test_page_markers_match(lang: str, text: str) -> None:
    pattern = BoilerplateCatalog.get_page_marker_pattern(lang)
    assert pattern is not None
    assert pattern.search(text) is not None


@pytest.mark.parametrize(
    ("lang", "text"),
    [("en", "Bettmann/Corbis"), ("de", "Foto: Max Mustermann")],
)
def test_photo_credits_match(lang: str, text: str) -> None:
    pattern = BoilerplateCatalog.get_photo_credit_pattern(lang)
    assert pattern is not None
    assert pattern.search(text) is not None


# --------------------------------------------------------------------------- #
# caching
# --------------------------------------------------------------------------- #


def test_repeated_lookups_return_the_same_compiled_object() -> None:
    assert BoilerplateCatalog.get_legal_pattern("en") is BoilerplateCatalog.get_legal_pattern("en")
