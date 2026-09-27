"""Declarative multilingual boilerplate and running header catalog.

Provides language-specific patterns for publisher legal disclaimers,
running headers, page markers, and photo credits across mainstream languages
(EN, FR, DE, ES, JA, ZH, RU).
"""

import re
from functools import lru_cache
from typing import NamedTuple


class LanguageBoilerplate(NamedTuple):
    """Patterns for stripping recurring publisher boilerplate in a specific language."""

    code: str
    legal_disclaimers: tuple[str, ...]
    running_headers: tuple[str, ...]
    page_markers: tuple[str, ...]
    photo_credits: tuple[str, ...]
    header_keywords: tuple[str, ...] = ()


# -----------------------------------------------------------------------------
# Declarative Catalog Definitions
# -----------------------------------------------------------------------------
#
# ReDoS hardening: the running-header word runs use DISJOINT
# character classes — ``WORD+`` and horizontal whitespace ``[^\S\n]+`` never
# overlap, so the regex engine cannot thrash exponentially between them on a
# non-matching line. Callers additionally prefilter on the literal chapter
# keyword (see ``header_keywords``) so the regex runs only when it can match.

_CATALOG: dict[str, LanguageBoilerplate] = {
    "en": LanguageBoilerplate(
        code="en",
        legal_disclaimers=(
            r"Copyright\s+\d{4}\s+.*?(?:Cengage|Ce\s*ngage|Lea[rmn]{1,2}ing|Pearson|Wiley|McGraw|All\s+Rights\s+Res).*",
            r"Editorial\s+review\s+has\s+deemed\s+that\s+any\s+suppressed\s+conten.*",
            r"suppressed\s+from\s+the\s+eBook.*",
            r"(?:the\s+)?right\s+to\s+remove\s+additional\s+content.*",
            r"May\s+not\s+be\s+copied,?\s*(?:scanned|duplicated)?.*",
            r"All\s+rights\s+reserved\.\s+No\s+part\s+of\s+this\s+publication.*",
            # Project Gutenberg injects these around every public-domain ebook:
            # the ``*** START/END OF THIS PROJECT GUTENBERG EBOOK <title> ***``
            # banners and the license/front-matter headings. Both are unmistakable
            # (the banner requires the triple-asterisk delimiters *and* the literal
            # "PROJECT GUTENBERG"), so neither can eat ordinary prose the way a
            # looser "Produced by ..." line would.
            r"\*{3}[^\n]*PROJECT\s+GUTENBERG[^\n]*\*{3}",
            r"^\s*(?:THE\s+)?FULL\s+PROJECT\s+GUTENBERG\s+LICENSE[^\n]*",
            r"^\s*Project\s+Gutenberg(?:\s+Trademark)?\s+License[^\n]*",
            r"^\s*Information\s+(?:about|about\s+the)\s+(?:the\s+)?(?:Project\s+)?Gutenberg[-\s][^\n]*",
        ),
        running_headers=(
            # The trailing page number must not be a 4-digit year: a chapter
            # title like "CHAPTER 5 Overview of 2020" otherwise matched the
            # running-header shape and the whole line was deleted.
            r"^\s*(?:[A-Za-z]+(?:[^\S\n]+[A-Za-z]+){0,5}[^\w\s]?[^\S\n]*CHAPTER[^\S\n]+\d+[^\S\n]+(?!(?:1[89]|20)\d{2}\b)\d+|\d+[^\S\n]+CHAPTER[^\S\n]+\d+[^\w\s]?(?:[^\S\n]+[A-Za-z]+){0,5})[^\S\n]*(?=(?-i:[A-Z])|$)",
            r"^\s*CHAPTER[^\S\n]+\d+[^\w\s]?[^\S\n]*(?:[A-Za-z]+(?:[^\S\n]+[A-Za-z]+){0,5})?[^\S\n]*(?!(?:1[89]|20)\d{2}\b)\d+[^\S\n]*",
        ),
        page_markers=(r"^\s*Page\s+\d+\s*\n*",),
        photo_credits=(
            r"\s*(?:Bettmann/Corbis|Fancy\s+Photography/Veer\s+Images|Getty\s+Images|Shutterstock)\s*$",
        ),
        header_keywords=("CHAPTER",),
    ),
    "fr": LanguageBoilerplate(
        code="fr",
        legal_disclaimers=(
            r"Tous\s+droits\s+r[eé]serv[eé]s(?:\s+pour\s+tous\s+pays)?.*",
            r"D[eé]p[oô]t\s+l[eé]gal\s*:?\s*[A-Za-z]+\s+\d{4}.*",
            r"Toute\s+reproduction\s*,?\s*m[eê]me\s+partielle\s*,?\s*est\s+interdite.*",
            r"Ce\s+livre\s+num[eé]rique\s+est\s+prot[eé]g[eé]\s+par\s+le\s+droit\s+d'auteur.*",
            r"Imprim[eé]\s+en\s+[A-Za-zÀ-ÿ]+.*",
        ),
        running_headers=(
            r"^\s*(?:[A-Za-zÀ-ÿ]+(?:[^\S\n]+[A-Za-zÀ-ÿ]+){0,5}[^\w\s]?[^\S\n]*CHAPITRE[^\S\n]+\d+[^\S\n]+\d+|\d+[^\S\n]+CHAPITRE[^\S\n]+\d+[^\w\s]?(?:[^\S\n]+[A-Za-zÀ-ÿ]+){0,5})[^\S\n]*(?=(?-i:[A-ZÀ-ÿ])|$)",
        ),
        page_markers=(r"^\s*Page\s+\d+\s*\n*",),
        photo_credits=(r"\s*(?:Cr[eé]dit\s+photo\s*:?.*?|Photo\s*:?.*?)\s*$",),
        header_keywords=("CHAPITRE",),
    ),
    "de": LanguageBoilerplate(
        code="de",
        legal_disclaimers=(
            r"Alle\s+Rechte\s+vorbehalten(?:\s*,?\s*insbesondere.*)?",
            r"Urheberrechtlich\s+gesch[uü]tztes\s+Material.*",
            r"Das\s+Werk\s+einschlie[sß]lich\s+aller\s+seiner\s+Teile\s+ist\s+urheberrechtlich\s+gesch[uü]tzt.*",
            r"Kein\s+Teil\s+dieses\s+Werkes\s+darf\s+ohne\s+schriftliche\s+Einwilligung.*",
            r"Gedruckt\s+in\s+[A-Za-zÄÖÜäöüß]+.*",
        ),
        running_headers=(
            r"^\s*(?:[A-Za-zÄÖÜäöüß]+(?:[^\S\n]+[A-Za-zÄÖÜäöüß]+){0,5}[^\w\s]?[^\S\n]*KAPITEL[^\S\n]+\d+[^\S\n]+\d+|\d+[^\S\n]+KAPITEL[^\S\n]+\d+[^\w\s]?(?:[^\S\n]+[A-Za-zÄÖÜäöüß]+){0,5})[^\S\n]*(?=(?-i:[A-ZÄÖÜ])|$)",
        ),
        page_markers=(r"^\s*Seite\s+\d+\s*\n*",),
        photo_credits=(r"\s*(?:Bildnachweis\s*:?.*?|Foto\s*:?.*?)\s*$",),
        header_keywords=("KAPITEL",),
    ),
    "es": LanguageBoilerplate(
        code="es",
        legal_disclaimers=(
            r"Todos\s+los\s+derechos\s+reservados(?:\s*,?\s*ninguna\s+parte\s+de\s+esta\s+publicaci[oó]n)?.*",
            r"Dep[oó]sito\s+legal\s*:?\s*[A-Za-z0-9\-]+.*",
            r"Queda\s+(?:rigurosamente\s+)?prohibida\s*,?\s*sin\s+la\s+autorizaci[oó]n\s+escrita.*",
            r"Cualquier\s+forma\s+de\s+reproducci[oó]n\s*,?\s*distribuci[oó]n.*",
            r"Impreso\s+en\s+[A-Za-zÁÉÍÓÚáéíóúñ]+.*",
            r"Edici[oó]n\s+digital\s*:?.*",
        ),
        running_headers=(
            r"^\s*(?:[A-Za-zÁÉÍÓÚáéíóúñ]+(?:[^\S\n]+[A-Za-zÁÉÍÓÚáéíóúñ]+){0,5}[^\w\s]?[^\S\n]*CAP[IÍ]TULO[^\S\n]+\d+[^\S\n]+\d+|\d+[^\S\n]+CAP[IÍ]TULO[^\S\n]+\d+[^\w\s]?(?:[^\S\n]+[A-Za-zÁÉÍÓÚáéíóúñ]+){0,5})[^\S\n]*(?=(?-i:[A-ZÁÉÍÓÚ])|$)",
        ),
        page_markers=(
            r"^\s*P[aá]gina\s+\d+\s*\n*",
            r"^\s*P[aá]g\.\s*\d+\s*\n*",
        ),
        photo_credits=(
            r"\s*(?:Cr[eé]dito\s+fotogr[aá]fico\s*:?.*?|Foto\s*:?.*?|Imagen\s*:?.*?)\s*$",
        ),
        header_keywords=("CAPÍTULO", "CAPITULO"),
    ),
    "ja": LanguageBoilerplate(
        code="ja",
        legal_disclaimers=(
            r"(?:無断転載[・\s]*複製を禁じます|不許複製|禁無断転載).*",
            r"本書の全部または一部を無断で.*(?:禁じます|できません).*",
            r"落丁[・\s]*乱丁本はお取替えいたします.*",
        ),
        running_headers=(
            r"^\s*(?:\d+\s+第\s*[0-9一二三四五六七八九十百]+\s*章|第\s*[0-9一二三四五六七八九十百]+\s*章\s+[^\n]+\s+\d+)\s*",
        ),
        page_markers=(r"^\s*(?:ページ\s*\d+|\d+\s*頁)\s*\n*",),
        photo_credits=(r"\s*(?:写真提供\s*[:：].*?|写真\s*[:：].*?)\s*$",),
        header_keywords=("章",),
    ),
    "ko": LanguageBoilerplate(
        code="ko",
        legal_disclaimers=(
            r"(?:무단\s*전재[・\s]*복제를\s*금합니다|판권\s*소유|무단\s*전재\s*및\s*재배포\s*금지).*",
            r"(?:무단으로\s*복제|무단\s*복제).*?(?:금합니다|없습니다).*",
            r"이\s*책의\s*.*?(?:일부|전부).*?(?:금합니다|없습니다).*",
            r"파본은\s*(?:구입처|구입하신\s*곳)에서\s*교환해\s*드립니다.*",
            r"국립중앙도서관\s*출판시도서목록\s*\(CIP\).*",
            r"인쇄처\s*:?.*",
            r"발행인\s*:?.*",
            r"전자책\s*발행\s*:?.*",
        ),
        running_headers=(
            r"^\s*(?:\d+\s+제\s*[0-9일이삼사오육칠팔구십백]+\s*장|제\s*[0-9일이삼사오육칠팔구십백]+\s*장\s+[^\n]+\s+\d+)\s*",
        ),
        page_markers=(r"^\s*(?:페이지\s*\d+|\d+\s*쪽)\s*\n*",),
        photo_credits=(r"\s*(?:사진\s*제공\s*[:：].*?|사진\s*[:：].*?|출처\s*[:：].*?)\s*$",),
        header_keywords=("장",),
    ),
    "zh": LanguageBoilerplate(
        code="zh",
        legal_disclaimers=(
            r"(?:版权所有[，,\s]*侵权必究|未经许可[，,\s]*不得以任何方式复制.*).*",
            r"图书在版编目\s*（\s*CIP\s*）\s*数据.*",
            r"如有印装质量问题，请与.*?联系调换.*",
            r"内部交流资料.*",
        ),
        running_headers=(
            r"^\s*(?:\d+\s+第\s*[0-9一二三四五六七八九十百]+\s*章|第\s*[0-9一二三四五六七八九十百]+\s*章\s+[^\n]+\s+\d+)\s*",
        ),
        page_markers=(r"^\s*(?:页码\s*[:：]?\s*\d+|第\s*\d+\s*页)\s*\n*",),
        photo_credits=(r"\s*(?:图片来源\s*[:：].*?|摄影\s*[:：].*?)\s*$",),
        header_keywords=("章",),
    ),
    "ru": LanguageBoilerplate(
        code="ru",
        legal_disclaimers=(
            r"Все\s+права\s+защищены(?:\.\s+Никакая\s+часть\s+данной\s+книги)?.*",
            r"Издательство\s+не\s+несет\s+ответственности.*",
            r"Любое\s+использование\s+материалов\s+допускается\s+только\s+с\s+письменного\s+согласия.*",
            r"Отпечатано\s+в\s+.*",
        ),
        running_headers=(
            r"^\s*(?:[А-Яа-я]+(?:[^\S\n]+[А-Яа-я]+){0,5}[^\w\s]?[^\S\n]*ГЛАВА[^\S\n]+\d+[^\S\n]+\d+|\d+[^\S\n]+ГЛАВА[^\S\n]+\d+[^\w\s]?(?:[^\S\n]+[А-Яа-я]+){0,5})[^\S\n]*(?=(?-i:[А-Я])|$)",
        ),
        page_markers=(r"^\s*(?:Стр(?:\.|аница)\s*\d+)\s*\n*",),
        photo_credits=(r"\s*(?:Фото\s*:?.*?|Иллюстрации\s*:?.*?)\s*$",),
        header_keywords=("ГЛАВА",),
    ),
}


def _primary_lang_code(lang: str) -> str:
    """Primary subtag of a BCP-47 tag (``zh-Hans``/``zh_CN`` -> ``zh``).

    ``job_options.LANG_CODE_PATTERN`` accepts region/script subtags, but the
    catalog is keyed by primary language; without this a valid ``zh-Hans``
    silently fell back to the English patterns.
    """
    if not lang:
        return "en"
    return lang.strip().lower().replace("_", "-").split("-", 1)[0]


class BoilerplateCatalog:
    """Access compiled regex patterns for a given ISO language code."""

    @classmethod
    def get_supported_languages(cls) -> tuple[str, ...]:
        """Return tuple of supported ISO language codes."""
        return tuple(_CATALOG.keys())

    @classmethod
    @lru_cache(maxsize=32)
    def get_legal_pattern(cls, lang: str) -> re.Pattern[str] | None:
        """Compile a unified regex pattern matching legal disclaimers for the source language."""
        code = _primary_lang_code(lang)
        entry = _CATALOG.get(code)
        if not entry and code != "en":
            entry = _CATALOG.get("en")
        if not entry or not entry.legal_disclaimers:
            return None

        joined = "|".join(f"(?:{p})" for p in entry.legal_disclaimers)
        return re.compile(joined, flags=re.MULTILINE | re.IGNORECASE)

    @classmethod
    @lru_cache(maxsize=32)
    def get_running_header_pattern(cls, lang: str) -> re.Pattern[str] | None:
        """Compile running header patterns for the source language."""
        code = _primary_lang_code(lang)
        entry = _CATALOG.get(code)
        if not entry and code != "en":
            entry = _CATALOG.get("en")
        if not entry or not entry.running_headers:
            return None

        joined = "|".join(f"(?:{p})" for p in entry.running_headers)
        return re.compile(joined, flags=re.IGNORECASE | re.MULTILINE)

    @classmethod
    @lru_cache(maxsize=32)
    def get_running_header_keywords(cls, lang: str) -> tuple[str, ...]:
        """Return literal keywords that must appear before header patterns can match.

        Used as a cheap prefilter: running-header regexes are only
        executed on text containing one of these literals, which eliminates
        catastrophic-backtracking exposure on adversarial input.
        """
        code = _primary_lang_code(lang)
        entry = _CATALOG.get(code)
        if not entry and code != "en":
            entry = _CATALOG.get("en")
        return entry.header_keywords if entry else ()

    @classmethod
    @lru_cache(maxsize=32)
    def get_page_marker_pattern(cls, lang: str) -> re.Pattern[str] | None:
        """Compile standalone page number/header patterns for the source language."""
        code = _primary_lang_code(lang)
        entry = _CATALOG.get(code)
        if not entry and code != "en":
            entry = _CATALOG.get("en")
        if not entry or not entry.page_markers:
            return None

        joined = "|".join(f"(?:{p})" for p in entry.page_markers)
        return re.compile(joined, flags=re.IGNORECASE | re.MULTILINE)

    @classmethod
    @lru_cache(maxsize=32)
    def get_photo_credit_pattern(cls, lang: str) -> re.Pattern[str] | None:
        """Compile photo/image attribution credit patterns for the source language."""
        code = _primary_lang_code(lang)
        entry = _CATALOG.get(code)
        if not entry and code != "en":
            entry = _CATALOG.get("en")
        if not entry or not entry.photo_credits:
            return None

        joined = "|".join(f"(?:{p})" for p in entry.photo_credits)
        return re.compile(joined, flags=re.IGNORECASE)
