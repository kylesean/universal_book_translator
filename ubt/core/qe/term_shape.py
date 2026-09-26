"""Shared shape predicates for source-verbatim Latin terms and sentence counts.

Used by both the fast-pass structural gate and the omission gate so the
"identifier-shaped" rule (camelCase humps, all-caps acronyms, embedded digits —
F3) has a single definition rather than a byte-identical copy that can drift.

``count_sentences`` is shared for the same reason: it is the omission gate's
abbreviation-masking counter, and the (unwired) MT admission gate imports it
too instead of keeping a second, abbreviation-blind copy that counted
``Eq. (3.11)`` / ``Fig. 3.5`` periods as sentence breaks and therefore
disagreed with the omission gate on the very same text.
"""

import re

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
_CJK_ADJ = (
    r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
    r"\uf900-\ufaff\uac00-\ud7af\uff00-\uffef]"
)
_SENT_END_RE = re.compile(
    r"[。！？]"
    r"|[.!?]['\"”’\)\]]*(?=\s|$)"
    rf"|(?<={_CJK_ADJ})[!?]"
    rf"|[!?](?={_CJK_ADJ})"
)
# Academic abbreviations whose period is not a sentence break. Without this,
# 'Eq. (3.11)' / 'Fig. 3.5' inflate the source sentence count while the zh
# translation ('式(3.11)' / '图3.5') carries no Latin period — a systematic
# sentence-ratio false positive on formula-dense narrative. Masked with a
# private-use placeholder before splitting (trade-off: a real sentence break
# after a trailing 'etc.' is merged; abbreviations mid-sentence dominate).
_ABBREV_RE = re.compile(
    r"\b(?:Eqs?|Figs?|Secs?|Refs?|Chap|App|No|Vol|pp?|e\.g|i\.e|etc|vs|cf|al|Dr|Prof|St)\.(?=\s|$)",
    re.IGNORECASE,
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


def is_identifier_shaped(term: str) -> bool:
    """Whether a source-verbatim Latin term must survive translation verbatim.

    camelCase humps (PagedAttention), all-caps acronyms (MHA, KV) and terms
    carrying digits (GQA-8) are identifiers, not translatable prose.
    Translatable structural document labels (FIG, TABLE, EQ, etc.) are excluded.
    """
    if term.lower() in _STRUCTURAL_DOCUMENT_LABELS:
        return False
    return bool(
        _CAMEL_HUMP_RE.search(term) or _ALL_CAPS_RE.match(term) or _HAS_DIGIT_RE.search(term)
    )


def count_sentences(text: str) -> int:
    """Count sentence fragments split on terminal punctuation (both scripts).

    The ONE sentence counter of the QE subsystem: academic abbreviation
    periods are masked before splitting, so 'Eq. 3.14 shows x. Then y.' is two
    sentences, not three, on every side that compares sentence counts.
    """
    masked = _ABBREV_RE.sub(lambda m: m.group(0)[:-1] + _ABBREV_DOT, text)
    fragments = [f.strip() for f in _SENT_END_RE.split(masked)]
    return sum(1 for f in fragments if f)
