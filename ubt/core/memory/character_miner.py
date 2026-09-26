"""0-token character/proper-noun mining for the Translation Bible.

Extracts person names deterministically (no LLM cost) from two signals:

1. Honorific-anchored names: ``Mr. Darcy``, ``Herr Müller``, ``M. Dupont`` —
   high precision, mined in first-attestation order across supported source languages.
   Postposed-honorific scripts are anchored from the right: Chinese
   ``王小明先生`` and Japanese ``田中さん``.
2. High-frequency bare capitalized names: ``Elizabeth`` — accepted only when
   the token occurs mid-sentence (preceded by lowercase text) at least
   ``min_freq`` times, filtering sentence starters, months, and boilerplate.
   (Automatically suppressed in languages like German where all nouns are
   capitalized, and absent for Chinese/Japanese, where no capitalization signal
   exists so only honorific-anchored names are harvested — zero false-positive
   tolerance: a missed name merely loses glossary protection, while a bogus
   bible entry pollutes every downstream prompt.)

Mined entries enter the bible with ``translation=""`` (kind ``person``) and
get their rendering decided by the bulk backfill channel; the term glossary
then enforces one rendering for the whole book — eliminating drift like
Bingley appearing as both 彬格莱 and 宾利.
"""

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SourceMiningConfig:
    """Linguistic mining rules and closed-class stopword boundaries per source language."""

    code: str
    honorifics: tuple[str, ...]
    particles: tuple[str, ...]
    not_names: frozenset[str]
    allow_bare_tokens_default: bool = True
    honorific_pattern: re.Pattern[str] | None = None
    bare_token_pattern: re.Pattern[str] | None = None
    # Minimum accepted name-core alias length. Latin words need >= 3 letters;
    # CJK names are 1-3 characters, so ZH/JA configs lower this bound and rely
    # on prefix stripping + not_names to keep precision.
    min_core_len: int = 3
    # Function-word prefixes (longest-first is applied by the miner) stripped
    # from a CJK honorific match before the name core is accepted. CJK text has
    # no word segmentation, so conjunctions/particles gluing onto a name
    # ("和林先生", "は田中さん") are shaved off here instead of via regex
    # lookbehind (which would drop every mid-sentence name). Stripping applies
    # only when the match contains a CJK ideograph, so kana-only given names
    # (さくら, はな) are never damaged. Stripped down to nothing => rejected.
    strip_prefixes: tuple[str, ...] = ()


# -----------------------------------------------------------------------------
# English (EN) Configuration
# -----------------------------------------------------------------------------
# "General" is deliberately absent. Unlike the courtesy titles and
# the other ranks it is a productive common adjective, so "General <Capital>"
# is usually a term or an organisation ("General Relativity", "General Motors",
# "General Electric", "General Assembly"), not a person. The honourific path
# stamped those as kind="person" and the backfill prompt then told the model to
# transliterate the invented core ("Relativity") book-wide. A recurring real
# surname ("General Smith") is still picked up by the frequency-gated bare-token
# path; the reverse error has no such net.
_EN_HONORIFICS = (
    "Mr",
    "Mrs",
    "Miss",
    "Ms",
    "Sir",
    "Lady",
    "Dr",
    "Professor",
    "Captain",
    "Colonel",
    "Rev",
)
_EN_PARTICLES = ("de", "von", "van", "di", "da", "der", "la")
_EN_NOT_NAMES = frozenset(
    {
        # Pronouns, determiners, quantifiers (Closed Class)
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "some",
        "any",
        "many",
        "much",
        "more",
        "most",
        "such",
        "all",
        "both",
        "each",
        "every",
        "either",
        "neither",
        "several",
        "few",
        "fewer",
        "fewest",
        "other",
        "another",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "first",
        "second",
        "third",
        "last",
        "next",
        "he",
        "she",
        "it",
        "they",
        "we",
        "you",
        "i",
        "his",
        "her",
        "hers",
        "him",
        "their",
        "theirs",
        "them",
        "my",
        "mine",
        "me",
        "our",
        "ours",
        "us",
        "your",
        "yours",
        "what",
        "whatever",
        "which",
        "whichever",
        "who",
        "whoever",
        "whom",
        "whomever",
        "whose",
        "where",
        "wherever",
        "when",
        "whenever",
        "why",
        "how",
        # Conjunctions, prepositions, adverbs (Closed Class)
        "and",
        "but",
        "or",
        "nor",
        "for",
        "yet",
        "so",
        "as",
        "if",
        "than",
        "because",
        "since",
        "while",
        "although",
        "though",
        "unless",
        "about",
        "above",
        "across",
        "after",
        "against",
        "along",
        "among",
        "around",
        "at",
        "before",
        "behind",
        "below",
        "beneath",
        "beside",
        "between",
        "beyond",
        "by",
        "down",
        "during",
        "except",
        "from",
        "in",
        "inside",
        "into",
        "near",
        "of",
        "off",
        "on",
        "onto",
        "out",
        "outside",
        "over",
        "through",
        "throughout",
        "to",
        "toward",
        "under",
        "underneath",
        "until",
        "up",
        "upon",
        "with",
        "within",
        "without",
        "also",
        "even",
        "only",
        "just",
        "well",
        "now",
        "here",
        "there",
        "then",
        "thus",
        "hence",
        "therefore",
        "moreover",
        "furthermore",
        "meanwhile",
        "instead",
        "indeed",
        "however",
        "yes",
        "no",
        "not",
        # Structural, publication, document nouns
        "chapter",
        "section",
        "part",
        "page",
        "pages",
        "figure",
        "figures",
        "table",
        "tables",
        "chart",
        "diagram",
        "illustration",
        "footnote",
        "appendix",
        "index",
        "volume",
        "edition",
        "preface",
        "introduction",
        "conclusion",
        "summary",
        "review",
        "abstract",
        "reference",
        "references",
        "source",
        "sources",
        "note",
        "notes",
        "copyright",
        "project",
        "gutenberg",
        "internet",
        "archive",
        "press",
        "university",
        "journal",
        "publisher",
        # Common academic / textbook categories that are concepts, not people
        "study",
        "studies",
        "result",
        "results",
        "theory",
        "theories",
        "model",
        "models",
        "data",
        "method",
        "methods",
        "group",
        "groups",
        "system",
        "systems",
        "research",
        "science",
        "nature",
        "participants",
        "subjects",
        "people",
        "humans",
        "animals",
        "children",
        "adults",
        "infants",
        "language",
        "memory",
        "cognitive",
        "cognition",
        "perception",
        "attention",
        "intelligence",
        "neuroscience",
        "brain",
        "mind",
        "psychology",
        "philosophy",
        "biology",
        "physics",
        "chemistry",
        "sociology",
        "anthropology",
        "economics",
        "linguistics",
        "mathematics",
        "statistics",
        "history",
        "geography",
        "medicine",
        "education",
        # Calendar & geography
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "god",
        "english",
        "france",
        "italy",
        "london",
        "paris",
    }
)

_EN_CONFIG = SourceMiningConfig(
    code="en",
    honorifics=_EN_HONORIFICS,
    particles=_EN_PARTICLES,
    not_names=_EN_NOT_NAMES,
    allow_bare_tokens_default=True,
    honorific_pattern=re.compile(
        r"\b(("
        + "|".join(_EN_HONORIFICS)
        # Latin-1 classes, like the DE/FR/ES configs: an ASCII-only name class
        # stopped "Mr. José" at "Mr. Jos" and the miner filed that fragment as a
        # person, ranked first in the global sheet, and rode every block prompt
        # of the book.
        + r")\.?\s+[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+(?:\s+(?:"
        + "|".join(_EN_PARTICLES)
        + r")?\s*[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+){0,2})"
    ),
    bare_token_pattern=re.compile(r"\b[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]{3,}\b"),
)

# -----------------------------------------------------------------------------
# German (DE) Configuration
# Note: Substantivgroßschreibung capitalizes all nouns; allow_bare_tokens_default
# is strictly False to avoid harvesting ordinary vocabulary as persons.
# -----------------------------------------------------------------------------
_DE_HONORIFICS = (
    "Herr",
    "Frau",
    "Fräulein",
    "Dr",
    "Doktor",
    "Prof",
    "Professor",
    "Graf",
    "Baron",
)
_DE_PARTICLES = ("von", "zu", "van", "der", "vom", "im")
_DE_NOT_NAMES = frozenset(
    {
        "der",
        "die",
        "das",
        "den",
        "dem",
        "des",
        "ein",
        "eine",
        "einer",
        "eines",
        "einem",
        "einen",
        "dieser",
        "diese",
        "dieses",
        "diesem",
        "diesen",
        "jener",
        "jene",
        "jenes",
        "jede",
        "jeder",
        "jedes",
        "alle",
        "einige",
        "manche",
        "etwas",
        "nichts",
        "wer",
        "was",
        "wie",
        "wo",
        "wann",
        "warum",
        "und",
        "oder",
        "aber",
        "denn",
        "weil",
        "wenn",
        "dass",
        "in",
        "an",
        "auf",
        "unter",
        "über",
        "vor",
        "nach",
        "bei",
        "mit",
        "von",
        "zu",
        "aus",
        "durch",
        "für",
        "gegen",
        "ohne",
        "um",
        "nicht",
        "ja",
        "nein",
        "auch",
        "nur",
        "schon",
        "noch",
        "hier",
        "da",
        "dort",
        "jetzt",
        "dann",
        "kapitel",
        "seite",
        "tabelle",
        "abbildung",
    }
)

_DE_CONFIG = SourceMiningConfig(
    code="de",
    honorifics=_DE_HONORIFICS,
    particles=_DE_PARTICLES,
    not_names=_DE_NOT_NAMES,
    allow_bare_tokens_default=False,
    honorific_pattern=re.compile(
        r"\b(("
        + "|".join(_DE_HONORIFICS)
        + r")\.?\s+[A-ZÄÖÜ][a-zäöüß]+(?:\s+(?:"
        + "|".join(_DE_PARTICLES)
        + r")?\s*[A-ZÄÖÜ][a-zäöüß]+){0,2})"
    ),
    bare_token_pattern=None,
)

# -----------------------------------------------------------------------------
# French (FR) Configuration
# -----------------------------------------------------------------------------
_FR_HONORIFICS = (
    "M",
    "Mme",
    "Mlle",
    "Monsieur",
    "Madame",
    "Mademoiselle",
    "Docteur",
    "Professeur",
    "Comte",
    "Baron",
)
_FR_PARTICLES = ("de", "du", "des", "d", "la")
_FR_NOT_NAMES = frozenset(
    {
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        "ce",
        "cet",
        "cette",
        "ces",
        "mon",
        "ma",
        "mes",
        "ton",
        "ta",
        "tes",
        "son",
        "sa",
        "ses",
        "notre",
        "nos",
        "votre",
        "vos",
        "leur",
        "leurs",
        "qui",
        "que",
        "quoi",
        "dont",
        "où",
        "quel",
        "quelle",
        "et",
        "ou",
        "mais",
        "donc",
        "or",
        "ni",
        "car",
        "dans",
        "sur",
        "sous",
        "avec",
        "sans",
        "pour",
        "par",
        "chez",
        "vers",
        "de",
        "à",
        "en",
        "ne",
        "pas",
        "plus",
        "tout",
        "tous",
        "toute",
        "toutes",
        "quelques",
        "plusieurs",
        "aucun",
        "chapitre",
        "page",
        "figure",
        "tableau",
    }
)

_FR_CONFIG = SourceMiningConfig(
    code="fr",
    honorifics=_FR_HONORIFICS,
    particles=_FR_PARTICLES,
    not_names=_FR_NOT_NAMES,
    allow_bare_tokens_default=True,
    honorific_pattern=re.compile(
        r"\b(("
        + "|".join(_FR_HONORIFICS)
        + r")\.?\s+[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+(?:\s+(?:"
        + "|".join(_FR_PARTICLES)
        + r")?\s*[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]+){0,2})"
    ),
    bare_token_pattern=re.compile(r"\b[A-ZÀ-ÖØ-ß][a-zà-öø-ÿ]{3,}\b"),
)

# -----------------------------------------------------------------------------
# Spanish (ES) Configuration
# -----------------------------------------------------------------------------
_ES_HONORIFICS = (
    "Sr",
    "Sra",
    "Srta",
    "Señor",
    "Señora",
    "Señorita",
    "Don",
    "Doña",
    "Dr",
    "Doctor",
    "Profesor",
)
_ES_PARTICLES = ("de", "del", "de la", "de los")
_ES_NOT_NAMES = frozenset(
    {
        "el",
        "la",
        "los",
        "las",
        "un",
        "una",
        "unos",
        "unas",
        "este",
        "esta",
        "estos",
        "estas",
        "ese",
        "esa",
        "esos",
        "esas",
        "aquel",
        "aquella",
        "mi",
        "mis",
        "tu",
        "tus",
        "su",
        "sus",
        "nuestro",
        "nuestra",
        "y",
        "e",
        "o",
        "u",
        "pero",
        "sino",
        "en",
        "de",
        "a",
        "para",
        "por",
        "con",
        "sin",
        "sobre",
        "entre",
        "hacia",
        "hasta",
        "no",
        "si",
        "ya",
        "mas",
        "todo",
        "todos",
        "alguno",
        "algunos",
        "capitulo",
        "pagina",
        "figura",
        "tabla",
    }
)

_ES_CONFIG = SourceMiningConfig(
    code="es",
    honorifics=_ES_HONORIFICS,
    particles=_ES_PARTICLES,
    not_names=_ES_NOT_NAMES,
    allow_bare_tokens_default=True,
    honorific_pattern=re.compile(
        r"\b(("
        + "|".join(_ES_HONORIFICS)
        + r")\.?\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+(?:"
        + "|".join(_ES_PARTICLES)
        + r")?\s*[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,2})"
    ),
    bare_token_pattern=re.compile(r"\b[A-ZÁÉÍÓÚÑ][a-záéíóúñ]{3,}\b"),
)

# -----------------------------------------------------------------------------
# CJK (KO) Configuration — placeholder for non-spaced scripts.
# Non-spaced or non-Latin scripts do not use capitalized token heuristics.
# -----------------------------------------------------------------------------
_CJK_CONFIG = SourceMiningConfig(
    code="cjk",
    honorifics=(),
    particles=(),
    not_names=frozenset(),
    allow_bare_tokens_default=False,
    honorific_pattern=None,
    bare_token_pattern=None,
)

# -----------------------------------------------------------------------------
# Chinese (ZH) Configuration
# Postposed honorifics anchor from the right: 王小明先生 / 张老师 / 李教授.
# CJK text is unsegmented, so matches are cleaned up in the miner instead of
# via regex lookbehind (which cannot see word boundaries and would drop every
# mid-sentence name like "和林先生"):
#   1. matches ending in the genitive 的 are rejected outright (names never
#      contain 的 — kills the whole "旁边的/屋里的先生" family);
#   2. strip_prefixes shaves function words off the left ("和林先生" -> "林",
#      "告诉王老师" -> "王", "我的一位先生" -> "" rejected);
#   3. not_names blocks residual generic cores.
# Design trade-off: a missed name only
# loses glossary protection, while a bogus bible entry pollutes every
# downstream prompt — precision is prioritized over coverage. Surname
# homoglyphs (于/沈/许/常/方/石/那/都/曾/应/诸/别) are deliberately kept OUT of
# the strip list for exactly this reason.
# Bare-token mining is disabled: no capitalization signal exists in CJK.
# -----------------------------------------------------------------------------
_ZH_HONORIFICS = ("先生", "女士", "老师", "教授", "博士")
_ZH_NOT_NAMES = frozenset(
    {
        # Residual cores that survive stripping
        "同",  # 同先生 (同学/一同 glue)
        "先生",
        "女士",
        "老师",
        "教授",
        "博士",
        "一个",
        "一些",
        "很多",
        "没有",
        "知道",
        "觉得",
        "认为",
    }
)
# Multi-char function words, then single-char conjunctions/prepositions/
# pronouns/determiners/generic person prefixes / numerals.
_ZH_STRIP_WORDS = (
    "告诉",
    "遇到",
    "遇见",
    "见到",
    "找到",
    "问到",
    "碰到",
    "等到",
    "留给",
    "交给",
    "送给",
    "带给",
    "但是",
    "可是",
    "因为",
    "所以",
    "如果",
    "虽然",
    "而且",
    "或者",
    "还是",
    "就是",
    "也是",
    "然后",
    "于是",
    "只是",
    "还有",
    "其实",
    "不过",
    "然而",
    "尽管",
    "既然",
    "以及",
    "甚至",
    "一位",
    "这位",
    "那位",
    "每个",
    "哪个",
    "这个",
    "那个",
    "什么",
    "怎么",
    "我们",
    "你们",
    "他们",
    "她们",
    "它们",
    "自己",
    "人家",
    "本人",
    "请问",
    "旁边",
    "对面",
    "上面",
    "下面",
    "前面",
    "后面",
    "里面",
    "外面",
    "左边",
    "右边",
    "知道",
    "觉得",
    "认为",
)
_ZH_STRIP_CHARS = "和与及或并对向给被让请叫见找问跟从在是把为以到至了的这我你他她它俺咱某每各数几二两三四五六七八九十百千位个名老小大本该其此之也很最更太又再才就还将只能可得想由替帮迎送访探望催派令使劝邀约候等过"

_ZH_CONFIG = SourceMiningConfig(
    code="zh",
    honorifics=_ZH_HONORIFICS,
    particles=(),
    not_names=_ZH_NOT_NAMES,
    allow_bare_tokens_default=False,
    # Name core = up to 4 CJK chars immediately followed by an honorific.
    # 4 characters covers native single/compound surnames (e.g. 诸葛孔明) and common
    # transliterations; strip_prefixes shaves off any attached function words.
    # Non-greedy core: a greedy ``{1,4}`` spanned across a following honorific
    # ("和先生和先生" -> core "先生和") and manufactured a bogus person entry.
    honorific_pattern=re.compile(r"([一-龥]{1,4}?)(?:" + "|".join(_ZH_HONORIFICS) + r")"),
    bare_token_pattern=None,
    min_core_len=1,
    strip_prefixes=(*_ZH_STRIP_WORDS, *_ZH_STRIP_CHARS),
)

# -----------------------------------------------------------------------------
# Japanese (JA) Configuration
# Postposed honorifics: 田中さん / 佐藤様 / さくらちゃん / 健太くん.
# Same strip-then-validate strategy as ZH. Stripping only applies when the
# match contains a CJK ideograph ("は田中さん" -> "田中"), so kana-only given
# names (さくら, はな) are mined intact. Generic 様 compounds (お疲れ様 /
# お客様 / ご苦労様) are shaved down to cores blocked by not_names.
# -----------------------------------------------------------------------------
_JA_HONORIFICS = ("さん", "様", "さま", "ちゃん", "くん", "君")
_JA_NOT_NAMES = frozenset(
    {
        # Residual cores after stripping generic 様/君 compounds
        "同",  # 同様 (likewise)
        "諸",  # 諸君 (gentlemen)
        "皆",  # 皆様 (everyone)
        "疲れ",  # お疲れ様 (thanks for your work)
        "世話",  # お世話様
        "苦労",  # ご苦労様
        "客",  # お客様 (customer)
        "的",  # 科学的な… (adjectival 的)
        "的な",  # …的なさん shape
        "あなた",
        "わたし",
        "あたし",
        "私",
        "彼",
        "彼女",
        "俺",
        "僕",
        "うち",
        "お前",
        "こいつ",
        "そいつ",
        "あいつ",
        "この",
        "その",
        "あの",
        "こんな",
        "そんな",
        "あんな",
        "こちら",
        "そちら",
        "あちら",
        "どなた",
        "誰",
        "みな",
        "みんな",
        "我々",
        "あ",
        "こ",
        "そ",
        "ど",
    }
)
# Multi-char expressions first, then single kana particles/demonstratives
# (stripped only from ideograph-bearing matches).
_JA_STRIP_WORDS = (
    "お疲れ",
    "お世話",
    "ご苦労",
    "しかし",
    "やはり",
    "やっぱり",
    "ちょっと",
    "とても",
    "本当に",
    "ありがとう",
    "すみません",
    "よろしく",
    "ところ",
    "もちろん",
    "たしかに",
    "なるほど",
    "それで",
    "だから",
    "けれど",
    "または",
    "あなた",
    "わたし",
    "あたし",
    "お前",
    "こいつ",
    "そいつ",
    "あいつ",
    "こちら",
    "そちら",
    "あちら",
    "どなた",
    "みんな",
    "皆さん",
)
_JA_STRIP_CHARS = "はがをでのとにもへやねよなだたてこそあどおごうくしまるりられんけせいすっ"

_JA_CONFIG = SourceMiningConfig(
    code="ja",
    honorifics=_JA_HONORIFICS,
    particles=(),
    not_names=_JA_NOT_NAMES,
    allow_bare_tokens_default=False,
    honorific_pattern=re.compile(r"([一-龥ぁ-んァ-ヶ]{1,4})(?:" + "|".join(_JA_HONORIFICS) + r")"),
    bare_token_pattern=None,
    min_core_len=1,
    strip_prefixes=(*_JA_STRIP_WORDS, *_JA_STRIP_CHARS),
)

# -----------------------------------------------------------------------------
# Portuguese (PT) Configuration
# Reuses the Spanish (es) family conventions: same particle
# list and stopword set, with Portuguese honorifics and PT accent classes.
# -----------------------------------------------------------------------------
_PT_HONORIFICS = (
    "Sr",
    "Sra",
    "Srta",
    "Senhor",
    "Senhora",
    "Dom",
    "Dona",
    "Dr",
    "Doutor",
    "Professor",
)
_PT_UPPER = "A-ZÁÂÃÀÇÉÊÍÓÔÕÚ"
_PT_LOWER = "a-záâãàçéêíóôõú"

_PT_CONFIG = SourceMiningConfig(
    code="pt",
    honorifics=_PT_HONORIFICS,
    particles=_ES_PARTICLES,
    not_names=_ES_NOT_NAMES,
    allow_bare_tokens_default=True,
    honorific_pattern=re.compile(
        r"\b(("
        + "|".join(_PT_HONORIFICS)
        + rf")\.?\s+[{_PT_UPPER}][{_PT_LOWER}]+(?:\s+(?:"
        + "|".join(_ES_PARTICLES)
        + rf")?\s*[{_PT_UPPER}][{_PT_LOWER}]+){{0,2}})"
    ),
    bare_token_pattern=re.compile(rf"\b[{_PT_UPPER}][{_PT_LOWER}]{{3,}}\b"),
)

_REGISTRY: dict[str, SourceMiningConfig] = {
    "en": _EN_CONFIG,
    "de": _DE_CONFIG,
    "fr": _FR_CONFIG,
    "es": _ES_CONFIG,
    "pt": _PT_CONFIG,
    "zh": _ZH_CONFIG,
    "ja": _JA_CONFIG,
    "ko": _CJK_CONFIG,
}


def get_mining_config(lang_code: str) -> SourceMiningConfig:
    """Resolve a SourceMiningConfig by ISO 639-1 code (e.g., 'en', 'de', 'fr')."""
    primary = lang_code.split("-")[0].split("_")[0].strip().lower()
    return _REGISTRY.get(primary, _EN_CONFIG)


def _name_core(full_name: str, core_stop: set[str]) -> str:
    """Last non-particle word of a name, as the prompt/response key."""
    words = full_name.replace(".", " ").split()
    for word in reversed(words[1:]):  # skip the honorific itself
        if word.lower() not in core_stop:
            return word
    return words[-1] if words else full_name


def _has_ideograph(text: str) -> bool:
    """True if the string contains at least one CJK Unified ideograph."""
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _strip_cjk_prefixes(full_name: str, prefixes: list[str]) -> str:
    """Shave function-word prefixes off a CJK honorific match, longest first.

    Only ideograph-bearing matches are stripped, so kana-only given names are
    never damaged; repeated passes handle stacked prefixes ("告诉过王老师" ->
    "王"). An empty result signals rejection (nothing but function words).
    """
    name = full_name
    if not prefixes or not _has_ideograph(name):
        return name
    changed = True
    while changed and name:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
                break
    return name


def mine_characters(
    text: str,
    *,
    source_lang: str = "en",
    max_entries: int = 40,
    min_freq: int = 15,
    allow_bare_tokens: bool | None = None,
) -> list[dict[str, Any]]:
    """Mine person names as bible-entry dicts in first attestation order.

    Adapts deterministically to the source language's linguistic rules:
    - English/French/Spanish/Portuguese: honorific-anchored names plus
      frequent mid-sentence bare capitalized names.
    - German: honorifics only (bare-token mining is disabled so ordinary
      capitalized nouns are not harvested as persons).
    - Chinese/Japanese/Korean: honorific-anchored names only — the postposed
      honorific is matched, function-word prefixes are stripped, and generic
      cores rejected via not_names. Korean has no rules configured and so
      returns an empty list.
    """
    return mine_characters_stream(
        [text],
        source_lang=source_lang,
        max_entries=max_entries,
        min_freq=min_freq,
        allow_bare_tokens=allow_bare_tokens,
    )


def mine_characters_stream(
    blocks: Iterable[str],
    *,
    source_lang: str = "en",
    max_entries: int = 40,
    min_freq: int = 15,
    allow_bare_tokens: bool | None = None,
    chunk_chars: int = 262_144,
    tail_chars: int = 1_024,
) -> list[dict[str, Any]]:
    """Streaming variant of :func:`mine_characters` with bounded memory.

    Blocks are accumulated into ~``chunk_chars`` regions. Honorific and
    mid-sentence detection run on the tail-prefixed region so names adjacent
    to block boundaries are still caught; bare-token frequency counting runs
    on the raw (non-overlapped) region so global counts stay exact — a token
    never spans two blocks because regions are joined with newlines, exactly
    like the historical ``"\\n".join(blocks)`` whole-text scan.
    """
    cfg = get_mining_config(source_lang)
    if cfg.code == "cjk" or not cfg.honorific_pattern:
        return []
    honorific_pattern = cfg.honorific_pattern
    stripped_prefixes = sorted(cfg.strip_prefixes, key=len, reverse=True)

    found: dict[str, dict[str, Any]] = {}
    seen_full_names: set[str] = set()
    core_stop = set(cfg.particles)
    # Attestations per name core across the whole book. The first
    # sighting creates the entry (deduped via seen_full_names), but EVERY
    # sighting must bump the count — otherwise a recurring honorific name
    # ("Mr. Darcy" seen 300 times) keeps the hardcoded frequency=1 and loses
    # prompt slots to one-shot bare tokens in select_terms_for_chunk.
    hon_core_counts: dict[str, int] = {}

    can_bare = cfg.allow_bare_tokens_default and (
        allow_bare_tokens is None or allow_bare_tokens is True
    )
    freq: Counter[str] = Counter()
    # Precompute mid-sentence capitalized words in one linear pass per region
    # to prevent quadratic regex compilation
    mid_sentence_pattern = re.compile(r"[a-zà-öø-ÿ,;]\s+([A-ZÀ-ÖØ-ßÁÉÍÓÚÑ][\w'-]*)")
    mid_sentence_tokens: set[str] = set()

    def _consume(scan_text: str, count_text: str | None, *, skip_upto: int = 0) -> None:
        for match in honorific_pattern.finditer(scan_text):
            # Every region after the first is scanned with the previous chunk's
            # tail glued on, so a name that sits entirely inside that overlap
            # matched a second time. The entry is deduped, but the count is
            # not: each boundary crossed inflated ``frequency`` by one, and
            # frequency ranks the terms that go into every block prompt.
            if match.end() <= skip_upto:
                continue
            full_name = match.group(1).strip()
            # "N的先生" is never a person (Chinese names contain no 的) —
            # reject the whole genitive-modifier family before stripping.
            if full_name.endswith("的") and len(full_name) > 1:
                continue
            full_name = _strip_cjk_prefixes(full_name, stripped_prefixes)
            if not full_name:
                continue
            core = _name_core(full_name, core_stop)
            is_new = full_name.lower() not in seen_full_names
            if is_new and (len(core) < cfg.min_core_len or core.lower() in cfg.not_names):
                continue
            if is_new:
                seen_full_names.add(full_name.lower())
            hon_core_counts[core.lower()] = hon_core_counts.get(core.lower(), 0) + 1
            if is_new:
                found[f"hon:{full_name.lower()}"] = {
                    "source": full_name,
                    "translation": "",
                    "aliases": [core],
                    "kind": "person",
                    "frequency": 1,
                }

        if can_bare and cfg.bare_token_pattern and count_text is not None:
            freq.update(cfg.bare_token_pattern.findall(count_text))
            mid_sentence_tokens.update(w.lower() for w in mid_sentence_pattern.findall(scan_text))

    tail = ""
    buf: list[str] = []
    size = 0
    for text in blocks:
        buf.append(text)
        size += len(text) + 1
        if size >= chunk_chars:
            count_text = "\n".join(buf)
            scan_text = f"{tail}\n{count_text}" if tail else count_text
            _consume(scan_text, count_text, skip_upto=len(tail))
            tail = count_text[-tail_chars:]
            buf, size = [], 0
    if buf:
        count_text = "\n".join(buf)
        scan_text = f"{tail}\n{count_text}" if tail else count_text
        _consume(scan_text, count_text, skip_upto=len(tail))

    # Backfill honorific frequencies from per-core attestation counts
    for key, entry in found.items():
        if not key.startswith("hon:"):
            continue
        aliases = entry.get("aliases") or []
        core = str(aliases[0]).lower() if aliases else ""
        entry["frequency"] = max(int(entry.get("frequency") or 0), hon_core_counts.get(core, 0))

    # Bare-token selection over the global frequency table (unchanged semantics)
    if can_bare and cfg.bare_token_pattern and len(found) < max_entries:
        covered_cores = {str(a).lower() for e in found.values() for a in e.get("aliases", [])}
        for token, count in freq.most_common():
            if len(found) >= max_entries:
                break
            if count < min_freq:
                break  # most_common is frequency-ordered
            if token.lower() in cfg.not_names or token.lower() in covered_cores:
                continue
            # mid-sentence requirement: preceded by lowercase text or punctuation
            if token.lower() not in mid_sentence_tokens:
                continue
            covered_cores.add(token.lower())
            found[f"bare:{token.lower()}"] = {
                "source": token,
                "translation": "",
                "aliases": [token],
                "kind": "person",
                "frequency": count,
            }

    return list(found.values())
