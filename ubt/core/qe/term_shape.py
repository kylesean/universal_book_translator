"""Shared shape predicates for source-verbatim Latin terms and sentence counts.

Used by both the fast-pass structural gate and the omission gate so the
"identifier-shaped" rule (camelCase humps, all-caps acronyms, embedded digits)
has a single definition rather than a byte-identical copy that can drift.

``count_sentences`` is shared for the same reason: it is the omission gate's
abbreviation-masking counter, and the (unwired) MT admission gate imports it
too instead of keeping a second, abbreviation-blind copy that counted
``Eq. (3.11)`` / ``Fig. 3.5`` periods as sentence breaks and therefore
disagreed with the omission gate on the very same text.
"""

import re

from ubt.core.cjk_ranges import CJK_WIDE_CLASS

_CAMEL_HUMP_RE = re.compile(r"[a-z][A-Z]")
_ALL_CAPS_RE = re.compile(r"^[A-Z0-9_]+$")
_HAS_DIGIT_RE = re.compile(r"\d")

# Sentence terminators. Full-width CJK terminators (。！？) are always terminal.
# A Latin . / ! / ? is terminal only when followed by whitespace/EOL (so
# decimals like '3.14' never split). A half-width ! / ? immediately adjacent to
# a CJK character is ALSO terminal: CJK has no whitespace after punctuation, so
# requiring whitespace missed it — but treating !/? as unconditionally terminal
# over-split English (e.g. Alice's "word!word" artifacts) and tripped the
# omission gate. The adjacency test is scoped to CJK/East-Asian ranges, not
# "any non-ASCII byte", because accented Latin (Café!Go) is whitespace-delimited
# and must not split on '!'.
_CJK_ADJ = f"[{CJK_WIDE_CLASS}]"
_SENT_END_RE = re.compile(
    r"[。！？؟؛۔]"
    r"|[.!?]['\"”’\)\]]*(?=\s|$)"
    rf"|(?<={_CJK_ADJ})[!?]"
    rf"|[!?](?={_CJK_ADJ})"
)
# Cross-reference labels whose period is not a sentence break. Without masking,
# 'Eq. (3.11)' / 'Fig. 3.5' inflate the source sentence count while the zh
# translation ('式(3.11)' / '图3.5') carries no Latin period — a systematic
# sentence-ratio false positive on formula-dense narrative. The period is
# masked ONLY when the label introduces a number or citation token, which is
# what separates the label 'No. 5' from the ordinary sentence-final word
# 'no.' / 'lab.' / 'max.': an earlier table matched those on whitespace alone,
# so 'This is a lab. We test the model.' counted two sentences instead of
# three and could drop the source below ``min_source_sentences``, silently
# disarming the omission gate. Masked with a private-use placeholder.
_REF_ABBREV_RE = re.compile(
    r"\b(?:"
    r"Eqs?|Eqns?|Figs?|Secs?|Refs?|Chaps?|Appx?|Vols?|Nos?|pp?|"
    r"Algos?|Tabs?|Thms?|Lems?|Props?|Cors?|Defs?|Ests?|"
    r"Labs?|Mins?|Maxs?|Vars?|Meds?|Diffs?|Accs?|Probs?|Dists?|"
    r"Params?|Freqs?|Precs?|Corrs?|Temps?|Coeffs?|Stats?|Stds?|"
    r"Avgs?|Effs?|Confs?|Procs?|Depts?|Univs?|Intls?|Trans|Socs?|Insts?|"
    r"Reps?|Sers?|Corp|Assn|Approx"
    r")\.(?=\s*[\d\[\(])",
    re.IGNORECASE,
)
# Latin and honorific abbreviations whose period is part of the token itself.
# Masked on whitespace/EOL as before (documented trade-off: a genuine sentence
# break after a trailing 'etc.'/'Dr.' merges; these spellings are not ordinary
# sentence-final words the way 'lab.'/'max.' were).
_LATIN_ABBREV_RE = re.compile(
    r"\b(?:e\.g|i\.e|etc|vs|cf|al|Dr|Prof|St)\.(?=[\s|,\)\]]|$)",
    re.IGNORECASE,
)
# Dotted initialisms (U.S./U.K./D.C.) are masked only mid-sentence — followed
# by a lowercase continuation or punctuation — because unlike 'etc.' they sit
# right next to real sentence boundaries constantly ("...in the U.S. Next
# year..."), and masking those boundary occurrences would erase true sentence
# breaks and under-count the source side of the omission gate.
_INITIALISM_ABBREV_RE = re.compile(
    r"\b(?:U\.S|U\.K|D\.C)\.(?=\s*[a-z(\[]|[,;:\)\]])",
)
_ABBREV_DOT = "\ue000"


# Translatable academic / structural document cross-reference labels.
# These represent document geometry and structure (e.g. FIG. 3.2, TABLE 1, EQ. 4),
# which must be translated into the target language (zh '图', ja '図', fr 'Figure',
# de 'Abbildung', etc.), not preserved as verbatim Latin code identifiers.
# Numeric numbering (e.g. '3.2') is strictly guarded by NumericConsistencyValidator.
_STRUCTURAL_DOCUMENT_LABELS = frozenset(
    {
        "fig",
        "figs",
        "figure",
        "figures",
        "eq",
        "eqs",
        "eqn",
        "eqns",
        "equation",
        "equations",
        "table",
        "tables",
        "tbl",
        "tbls",
        "sec",
        "secs",
        "section",
        "sections",
        "chap",
        "chaps",
        "chapter",
        "chapters",
        "app",
        "apps",
        "appendix",
        "appendices",
        "algo",
        "algos",
        "algorithm",
        "algorithms",
        "ref",
        "refs",
        "reference",
        "references",
        "vol",
        "vols",
        "no",
        "nos",
        "page",
        "pages",
        "pp",
    }
)

_STRUCTURAL_LABEL_WITH_NUM_RE = re.compile(r"^([A-Za-z]+)[._\s-]*\d+(?:[._\-]\d+)*(?:[A-Za-z])?$")


# Common English words that are all-caps-shaped when a heading/emphasis run is
# uppercased. They are prose, not source-verbatim identifiers, so the omission
# gate must not demand they survive translation. Kept deliberately to function
# words and everyday nouns — real acronyms (API/CPU/GPT/LSTM…) are absent.
_COMMON_CAPS_WORDS = frozenset(
    {
        "A",
        "AM",
        "AN",
        "AND",
        "ANY",
        "ARE",
        "AS",
        "AT",
        "BE",
        "BUT",
        "BY",
        "CAN",
        "DID",
        "DO",
        "EU",
        "FEW",
        "FOR",
        "GET",
        "GO",
        "GOT",
        "HAD",
        "HAS",
        "HE",
        "HER",
        "HIM",
        "HIS",
        "HOW",
        "I",
        "IF",
        "IN",
        "IS",
        "IT",
        "ITS",
        "LET",
        "MAY",
        "ME",
        "MEN",
        "MY",
        "NEW",
        "NO",
        "NON",
        "NOT",
        "NOW",
        "OF",
        "OFF",
        "OK",
        "OLD",
        "ON",
        "ONE",
        "OR",
        "OUR",
        "OUT",
        "OWN",
        "PER",
        "PM",
        "PUT",
        "RUN",
        "SAY",
        "SHE",
        "SO",
        "THE",
        "TO",
        "TOO",
        "TOP",
        "TWO",
        "UN",
        "UP",
        "US",
        "USE",
        "VIA",
        "WAS",
        "WAY",
        "WE",
        "WHO",
        "WHY",
        "YES",
        "YET",
        "YOU",
    }
)

_COMMON_HEADING_WORDS = frozenset(
    {
        "ABSTRACT",
        "ACKNOWLEDGEMENTS",
        "ACKNOWLEDGMENTS",
        "APPENDIX",
        "APPENDICES",
        "BACKGROUND",
        "BIBLIOGRAPHY",
        "CHAPTER",
        "CHAPTERS",
        "CONCLUSION",
        "CONCLUSIONS",
        "CONTENTS",
        "DISCUSSION",
        "FOREWORD",
        "INDEX",
        "INTRODUCTION",
        "METHOD",
        "METHODS",
        "METHODOLOGY",
        "NOTES",
        "OVERVIEW",
        "PREFACE",
        "REFERENCES",
        "RESULTS",
        "SECTION",
        "SECTIONS",
        "SUMMARY",
    }
)

_COMMON_CAPS_WORDS = _COMMON_CAPS_WORDS | _COMMON_HEADING_WORDS
_COMMON_CAPS_ACRONYMS = frozenset({"IT", "US", "EU", "AI", "UK"})


_OCR_GLUED_PROSE_PREFIX_RE = re.compile(
    r"^(?:We|In|This|The|That|There|These|Those|Our|It|As|If|When|While|For|With|By|To|From)"
    r"(?:use|uses|used|show|shows|showed|find|finds|found|present|presents|evaluate|evaluates|"
    r"this|that|these|those|is|are|was|were|have|has|had|can|could|will|would|an|a|the)",
    re.IGNORECASE,
)


def is_identifier_shaped(term: str) -> bool:
    """Whether a source-verbatim Latin term must survive translation verbatim.

    camelCase humps (PagedAttention), all-caps acronyms of 3+ characters (MHA,
    LSTM) and terms carrying digits (GQA-8) are identifiers, not translatable
    prose. Translatable structural document labels (FIG, TABLE, EQ, etc.) and
    common all-caps *words* (IT, OR, IF, US, THE in a heading …) are excluded:
    demanding those survive translation verbatim would flag a correct Chinese
    rendering of "the IT department uses OR logic" as "Omission suspected" and
    quarantine it as BLOCKED_HUMAN.
    """
    if term.lower() in _STRUCTURAL_DOCUMENT_LABELS:
        return False
    if _OCR_GLUED_PROSE_PREFIX_RE.match(term):
        return False
    m = _STRUCTURAL_LABEL_WITH_NUM_RE.match(term)
    if m and m.group(1).lower() in _STRUCTURAL_DOCUMENT_LABELS:
        return False
    if _HAS_DIGIT_RE.search(term) or _CAMEL_HUMP_RE.search(term):
        return True
    if "_" in term or "." in term:
        return True
    if _ALL_CAPS_RE.match(term):
        core = term.strip("_")
        # A 1-2 character all-caps run is ordinary prose far more often than a
        # term (IT/OR/IF/US/EU/AI/OK/NO), and a common word that merely happens
        # to be uppercased is not an identifier either.
        return len(core) >= 3 and core not in _COMMON_CAPS_WORDS
    return False


def is_verbatim_carryover(term: str) -> bool:
    """Whether a Latin term may legitimately survive translation verbatim.

    Superset of :func:`is_identifier_shaped`, used by the *target-language
    density* check: any camelCase, all-caps (even 2 letters like IT/US/AI) or
    digit-bearing token a translation keeps verbatim must be stripped from the
    script-density residue. Common English prose and heading words (e.g. THE, AND,
    INTRODUCTION, SUMMARY) are translatable prose and must not be treated as
    verbatim carryover.
    """
    if term.lower() in _STRUCTURAL_DOCUMENT_LABELS:
        return False
    m = _STRUCTURAL_LABEL_WITH_NUM_RE.match(term)
    if m and m.group(1).lower() in _STRUCTURAL_DOCUMENT_LABELS:
        return False
    if _CAMEL_HUMP_RE.search(term) or _HAS_DIGIT_RE.search(term):
        return True
    if _ALL_CAPS_RE.match(term):
        core = term.strip("_")
        return core not in _COMMON_CAPS_WORDS or core in _COMMON_CAPS_ACRONYMS
    return False


_SPACED_DECIMAL_RE = re.compile(r"(?<=\d)\s*\.\s*(?=\d)")


def count_sentences(text: str) -> int:
    """Count sentence fragments split on terminal punctuation (both scripts).

    The ONE sentence counter of the QE subsystem: academic cross-reference
    periods are masked before splitting, so 'Eq. 3.14 shows x. Then y.' is two
    sentences, not three, on every side that compares sentence counts. A
    reference label is only masked when it introduces a number/citation
    ('No. 5', 'Ref. [3]'), so an ordinary word that merely looks abbreviated
    ('lab.', 'max.') still ends its sentence and is counted. Spaced decimals
    from PDF extraction ('19 . 6%') are collapsed to '19.6%' so the dot is not
    falsely matched as terminal punctuation followed by space.
    """
    cleaned = _SPACED_DECIMAL_RE.sub(".", text)
    masked = _REF_ABBREV_RE.sub(lambda m: m.group(0)[:-1] + _ABBREV_DOT, cleaned)
    masked = _LATIN_ABBREV_RE.sub(lambda m: m.group(0)[:-1] + _ABBREV_DOT, masked)
    masked = _INITIALISM_ABBREV_RE.sub(lambda m: m.group(0)[:-1] + _ABBREV_DOT, masked)
    fragments = [f.strip() for f in _SENT_END_RE.split(masked)]
    return sum(1 for f in fragments if f)
