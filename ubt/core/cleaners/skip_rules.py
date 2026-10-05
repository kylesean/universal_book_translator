"""Pre-translation skip classification for untranslatable blocks.

Marks blocks that must bypass the LLM / QE / repair loop entirely and ship
their source text verbatim (status MTQE_PASSED, score 1.0 — see the ingest
stage hook). Every rule here is a precision-first pattern for content where
machine translation is either meaningless or actively harmful:

- social-handle watermarks / running-head chrome (``@techNmak ...``);
- bibliography entries (paper titles must stay searchable in the original
  language; translating them breaks retrievability);
- front-matter bibliographic identity — the author-affiliation/email line and
  the ``∗/†/‡`` symbol legend (institution names, contact addresses and
  cross-reference markers must stay one-to-one with the author list);
- pure symbol / numeric debris (shattered equation fragments, table number
  runs) with no translatable words.

Blocks with real prose — including short citation lines like
``Research anchors: [1], [2]`` — are deliberately NOT skipped; those go
through the LLM and are protected by the QE calibration instead.
"""

from __future__ import annotations

import re
import threading

from ubt.core.memory.cjk_matcher import contains_cjk

# Social handle / watermark chrome at the start of a block.
_HANDLE_RE = re.compile(r"^@[\w.\-]+")
# arXiv identifier anywhere in the block (tight, high-precision bib signal).
_ARXIV_RE = re.compile(r"arXiv:\d{4}\.\d+", re.IGNORECASE)
# Bracket-style reference entry: "[12] A. Author ..." (must start with an
# uppercase letter or opening quote, never a lowercase verb like "[3] yield...").
_BRACKET_REF_RE = re.compile(r"^\[\d+\]\s+[A-Z“\"‘'《]")
_ONLINE_BIB_RE = re.compile(r"\[Online\]\.?\s*Available:\s*https?://", re.IGNORECASE)
# Label-free bibliography entry: segmentation often drops the "[n]"

# label, so the bracket rule above cannot fire. Two precision-first tiers —
#   A: multi-author ("et al." or >=2 author-initial tokens) + year + venue;
#   B: single author + year + strong book cue (publisher / Chapter N).
# Rationale (industry standard, GB/T 7714 practice): foreign-language
# citations are kept in the original language — translating author names,
# venues, or DOIs breaks retrievability and is never publication-grade.
_ETAL_RE = re.compile(r"\bet\.?\s+al\.")
_AUTHOR_RE = re.compile(r"(?<![A-Za-z0-9/])[A-Z]\.-?[A-Z]?\s*[A-Z][a-z]+")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
# Publisher strings shared by the venue and book-title cues — one list so a
# new publisher is added once. The conference-acronym lists in _BIB_VENUE_RE
# and _IN_VENUE_RE below deliberately stay separate: their word-boundary
# contexts differ (ACL needs a \b in one and not the other), so extend those
# two in lockstep instead of sharing a string.
_BIB_PUBLISHERS = r"Pearson|Prentice|Elsevier|Wiley|Springer"
_BIB_VENUE_RE = re.compile(
    r"Technical Digest|Trans\.|Symposium|Conference|Manual|Chapter \d|"
    rf"pp?\.\s*\d|doi\.org|\bdoi:\s*10\.|https?://|University|{_BIB_PUBLISHERS}|"
    r"SISPAD|IEDM|VLSI|IEEE Access|IEEE\b|Journal\b|"
    r"Solid-State|Electron Device|Lett\.|Semicond\.|Nanotech|Microelectron|"
    r"Appl\. Phys|Proc\.|Proceedings of\b|CoRR\b|\bvol\.\s*(?:\d+|abs/)|"
    # Modern ML/Systems venues: the arXiv-era reference list a survey or
    # LLM paper ships. Full-name authors (below) defeat the initial-based
    # author count, so the venue string is the only reliable cue.
    r"Neural Information Processing|Learning Representations|Machine Learning|"
    r"Computational Linguistics|Natural Language Processing|Artificial Intelligence|"
    r"Association for Computing|ACM\b|PMLR|NeurIPS|ICML|ICLR|ACL\b|EMNLP|"
    r"Technical Report|Preprint",
    re.IGNORECASE,
)
_BIB_BOOK_RE = re.compile(
    rf"Chapter \d|{_BIB_PUBLISHERS}|Manual\b|"
    r"University of California|Inc\b|Publishers?\b|Publishing\b|McGraw|"
    r"MIT Press|Addison[- ]?Wesley|O['’]Reilly|Cambridge University Press|"
    r"Oxford University Press|Academic Press|Manning\b|Morgan Kaufmann|CRC Press|"
    r"Princeton University Press|\b\d+(?:st|nd|rd|th)\s+ed\.",
    re.IGNORECASE,
)
_SHORT_ARTICLE_BIB_RE = re.compile(
    r"^[A-Z]\.\s*(?:[A-Z]\.\s*)?[A-Z][A-Za-z'’\-]+,\s*['\"“‘][^'\"”’]{3,120}['\"”’]\.?\s*(?:19|20)\d{2}\.?\s*$"
)
# Generic journal-volume cue: "Lett. 21 (5)", "Integrion 1 (2)" — a
# capitalised venue word followed by a volume and issue. Prose effectively
# never produces this shape while a label-free entry whose venue string was
# OCR-garbled beyond the dictionary still does (chapter-3 p24 b0208).
_JOURNAL_VOL_RE = re.compile(r"\b[A-Z][A-Za-z]{2,}\.?\s+\d{1,4}\s*\(\s*\d{1,3}\s*\)")
# Terminal page range: "245 - 247." / "86-95." — the classic last token of an
# entry whose venue cue was lost (chapter-3 b0207).
_PAGE_RANGE_TAIL_RE = re.compile(r"\b\d{1,4}\s*[-–—]\s*\d{1,4}\s*\.?\s*$")
# Structural reference cues that do NOT depend on the initial-based author
# count. Modern references list full names ("John Yang, Carlos Jimenez, …"),
# which score zero on ``_AUTHOR_RE`` and so never cleared the ``authors >= N``
# gates above — every arXiv/NeurIPS/GitHub entry slipped through to the LLM
# and came back with transliterated author names (arXiv 2609.20519).
_URL_RE = re.compile(r"https?://", re.IGNORECASE)
_IN_VENUE_RE = re.compile(
    r"\bIn\s+(?:the\s+)?(?:\d{4}\s+)?(?:[A-Z][A-Za-z0-9]*\s+)*(?:Proceedings|Proc\.|Conference|Conf\.|Symposium|Symp\.|Workshop|Transactions|Trans\.|Journal|Advances|ACM|IEEE|USENIX|NeurIPS|ICML|ICLR|CVPR|ECCV|ICCV|AAAI|IJCAI|ACL|EMNLP|NAACL|COLING|KDD|WWW|SIGMOD|VLDB|OSDI|SOSP|NSDI|EuroSys|FAST|ASPLOS|MICRO|ISCA|HPCA|ATC)\b"
)
_PAGES_RE = re.compile(r"\bpages?\s+[\d,]+(?:\s*[-–—]\s*[\d,]+)?", re.IGNORECASE)
_VOLUME_RE = re.compile(r"\bvolume\s+\d+", re.IGNORECASE)
_PAGE_RANGE_RE = re.compile(r"\b\d{2,6}\s*[-–—]\s*\d{2,6}\b")
# A full-name author list at the start of a block ("John Yang, Carlos Jimenez,
# …"). The author-independent venue+pages tier below needs this: without an
# author signal it also fires on body prose that merely mentions a venue and a
# page range ("This approach was published by Springer in 2019, spanning pages
# 45-60."), shipping the sentence untranslated.
_NAME_LIST_START_RE = re.compile(r"^[A-Z][A-Za-z'’\-]+\s+[A-Z][A-Za-z'’\-]+\s*[,.]")
# A "real word": two or more consecutive ASCII letters. Single letters
# (K/V/x), digits and punctuation do not count as translatable content.
_WORD_RE = re.compile(r"[A-Za-z]{2,}")

# --- Author byline (a paper's name list under the title) --------------------
# GB/T 7714 and general Chinese academic practice keep foreign author names in
# the original romanization. Transliterating a romanized name back to Chinese
# characters ("Duomin Wang" -> 王多民) is a lossy guess (which 多民? 朵明?), and
# the byline is not prose to translate. UBT's parser tags the byline as plain
# main_text, so detect it by shape: every comma-separated segment is a run of
# Capitalized name tokens optionally trailed by affiliation markers
# (superscript/inline digits, *, †, ‡), with no year/venue/title (which would
# make it a reference instead). Distinct from _is_bib_entry by requiring the
# whole block to be names — prose never matches because its words are lowercase.
# Split on [A-Z][a-z...]+ so camelCase concatenated names from PDF extraction
# ("YoungmokJung") tokenize into constituent given name and surname ("Youngmok", "Jung").
_BYLINE_NAME_RE = re.compile(r"[A-Z][a-z'’\-]+")
_INITIAL_RE = re.compile(r"[A-Z]\.")
_BYLINE_SEGMENT_RE = re.compile(
    r"^(?:and\s+|&\s+)?"  # trailing-list conjunction
    r"(?:[A-Z][A-Za-z'’\-]+|(?:[A-Z]\.)+)"  # first name token or initial(s)
    r"(?:[\s,]+(?:[A-Z][A-Za-z'’\-]+|Jr\.?|Sr\.?|(?:[A-Z]\.)+))*"  # more name tokens / initials
    # Affiliation markers / superscripts. Written as ONE character class, not
    # ``(?:[\s,]*+(?:\d+|¹|ⁿ|…)+)*``: the old nested quantifiers made
    # ``"Aa Aa, Bb " + "1"*n + "!"`` backtrack exponentially (11.7 s at n=18,
    # far worse above) and ``classify_skip`` runs on every ingested block, so a
    # crafted/malformed line could stall the ingest worker. The accepted set is
    # unchanged — the tail is any run of digits, separators or affiliation
    # symbols — and the scan is now linear.
    r"[\s,0-9¹ⁿ∗*†‡§¶]*$"
)
_SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"

# --- Front-matter bibliographic identity ------------------------------------
# The author-affiliation block, its contact email, and the symbol legend
# ("∗ Corresponding author. † DSec project developers. ‡ Tsinghua University.")
# are bibliographic identity matter, not prose: GB/T 7714 and general academic
# practice keep institution names, addresses and the ∗/†/‡ cross-reference
# markers verbatim so the symbols stay one-to-one with the author list.
# Translating only half of the cluster is the failure mode this rule prevents:
# docling labels the legend FOOTNOTE (region BODY) but the author bio
# PAGE_FOOTER (region FOOTER), so a per-item policy translated the legend while
# the FOOTER chrome rule kept the bio — a mixed-language page footer. One shape
# rule keeps the whole front-matter identity cluster together.
#
# Markers are the typographic referential set (∗ U+2217, † U+2020, ‡ U+2021,
# § U+00A7, ¶ U+00B6) plus an ASCII ``*`` only when it stands as its own token
# (so multiplication ``a*b`` and Markdown bullets do not register).
_REFERENTIAL_MARKER_RE = re.compile(r"[∗†‡§¶]|(?<![A-Za-z0-9])\*(?![A-Za-z0-9])")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
#: Institution nouns that mark an affiliation line. Deliberately proper-noun
#: vocabulary ("University", "Institute", …), never a generic word like
#: "research", so ordinary prose that merely mentions an organisation is not
#: swept in — the marker/email requirement below is the second gate.
_INSTITUTION_RE = re.compile(
    r"\b(?:Universit(?:y|ies|ät|é|e)|Institute|Institut|Academy|Laboratory|"
    r"Laboratoire|College|School|Labs?)\b",
    re.IGNORECASE,
)
#: Identity lines are short. A long block that merely mentions a university or
#: prints a contact address is prose (an acknowledgements paragraph, a call for
#: contributions) and must stay translatable.
_IDENTITY_MAX_WORDS = 40

_EXTRA_BIB_VENUE_PATTERNS: list[re.Pattern[str]] = []
_EXTRA_BIB_BOOK_PATTERNS: list[re.Pattern[str]] = []
_PATTERNS_LOCK = threading.Lock()


_NARRATIVE_PROSE_RE = re.compile(
    r"\b(?:according to|we (?:extend|propose|show|evaluate|present|describe|find|observe|use|introduce)|"
    r"our (?:approach|method|model|dataset|benchmark|results|work)|"
    r"this (?:paper|section|work|study|chapter)|"
    r"as (?:shown|demonstrated|noted|described)|"
    r"can be (?:found|downloaded|seen) at|relies on|is available at|was released at)\b",
    re.IGNORECASE,
)


_DOC_CITATION_RE = re.compile(
    r"\b(?:documentation|manual|whitepaper|specification|user\s+guide)\b.*"
    r"\b(?:consulted|accessed|retrieved)\s+(?:[A-Za-z]+\.?\s+)?(?:19|20)\d{2}\b",
    re.IGNORECASE | re.DOTALL,
)


def _is_bib_entry(text: str) -> bool:
    """Label-free bibliography signature (tiers A/B, see above)."""
    if _NARRATIVE_PROSE_RE.search(text):
        return False
    # Narrative discourse guard: 3 or more continuous sentences (>35 words)
    # is narrative prose, never a single bibliography entry.
    sentences = [s.strip() for s in re.split(r"[.!?]\s+[A-Z]", text) if s.strip()]
    if len(sentences) >= 3 and len(text.split()) > 35:
        return False

    if (
        _ONLINE_BIB_RE.search(text)
        or _SHORT_ARTICLE_BIB_RE.match(text)
        or _DOC_CITATION_RE.search(text)
    ):
        return True

    with _PATTERNS_LOCK:
        extra_venues = tuple(_EXTRA_BIB_VENUE_PATTERNS)
        extra_books = tuple(_EXTRA_BIB_BOOK_PATTERNS)

    # In-text parenthetical citations like "(Wang et al., 2023)" are not
    # bibliography author declarations. Filter parenthetical spans when
    # counting et-al occurrences.
    clean_text_no_parens = re.sub(r"\([^)]*\)", "", text)
    clean_etal_hits = len(_ETAL_RE.findall(clean_text_no_parens))
    clean_author_hits = len(_AUTHOR_RE.findall(clean_text_no_parens))

    authors = clean_etal_hits + clean_author_hits
    # ``et al.`` matches cannot authenticate the et-al tier below: body prose
    # cites "(Wang et al., 2023)" far too often (arXiv 2609.32391 shipped its
    # introduction untranslated). Only an author signal independent of
    # ``et al.`` — an initial-based name — may back that tier.
    initial_authors = clean_author_hits
    if not _YEAR_RE.search(text):
        return False
    full_name_start = _NAME_LIST_START_RE.match(text) is not None
    venue_hit = bool(_BIB_VENUE_RE.search(text)) or any(p.search(text) for p in extra_venues)
    # Author-independent structural tiers: full-name references clear none of
    # the ``authors >= N`` gates, so these fire on the citation shape alone.
    if _IN_VENUE_RE.search(text) and (_PAGES_RE.search(text) or _VOLUME_RE.search(text)):
        return True
    # Author-independent tier, so it must still carry a citation start: either
    # an initial-based author (``authors``) or a full-name list ("John Yang,
    # …"). A bare sentence that happens to name a venue and a page range is
    # prose, not an entry.
    if (
        venue_hit
        and (authors >= 1 or _NAME_LIST_START_RE.match(text))
        and (_PAGES_RE.search(text) or _VOLUME_RE.search(text) or _PAGE_RANGE_RE.search(text))
    ):
        return True
    if clean_etal_hits >= 1 and (venue_hit or _IN_VENUE_RE.search(text) or initial_authors >= 1):
        return True
    if _URL_RE.search(text) and (
        venue_hit or _IN_VENUE_RE.search(text) or authors >= 1 or full_name_start
    ):
        return True
    if authors >= 2 and venue_hit:
        return True
    if (
        authors >= 1
        and venue_hit
        and re.search(r"\b(?:in\s+Proceedings|doi:|vol\.\s*abs/)", text, re.IGNORECASE)
    ):
        return True
    if authors >= 1 and _JOURNAL_VOL_RE.search(text):
        return True
    if authors >= 1 and venue_hit and _PAGE_RANGE_TAIL_RE.search(text):
        return True
    book_hit = bool(_BIB_BOOK_RE.search(text)) or any(p.search(text) for p in extra_books)
    return authors >= 1 and book_hit


def _is_author_byline(text: str) -> bool:
    """True when the whole block is a paper's author name list (byline).

    Every comma-separated segment must be Capitalized name tokens (optionally
    trailed by affiliation markers); a single prose word (lowercase, or a verb)
    fails the shape, so ordinary narrative never matches. Requires >=2 names and
    no year (a year would make it a reference, handled by _is_bib_entry).

    A segment also needs at least TWO name tokens: real bylines carry a given
    name and a surname ("Duomin Wang 1", "Jane Doe ¹"), or initials plus a
    surname ("Y.C. Yan"), while a title-case heading list ("Purpose, Scope, and
    Audience", "War and Peace") is one word per segment. Misclassifying those as
    a byline ships an untranslated heading with MTQE_PASSED and no flag.
    """
    if _YEAR_RE.search(text) or contains_cjk(text):
        return False
    # Superscript affiliation digits are stripped so "Li¹²" tokenizes as a name.
    cleaned = "".join(" " if c in _SUPERSCRIPT_DIGITS else c for c in text)
    segments = [s.strip() for s in re.split(r",| and ", cleaned) if s.strip()]
    if len(segments) < 2:
        return False
    for seg in segments:
        if not _BYLINE_SEGMENT_RE.match(seg):
            return False
        full_names = len(_BYLINE_NAME_RE.findall(seg))
        initials = len(_INITIAL_RE.findall(seg))
        if not (full_names >= 2 or (initials >= 1 and full_names >= 1)):
            return False
    if len(segments) < 3:
        return all(len(_BYLINE_NAME_RE.findall(seg)) >= 2 for seg in segments)
    return any(len(_BYLINE_NAME_RE.findall(seg)) >= 2 for seg in segments)


def _is_bibliographic_identity(text: str) -> bool:
    """True for a front-matter author/affiliation/legend line.

    Fires on the two shapes that carry the ∗/†/‡ cross-reference cluster:

    * a referential marker plus an institution cue or a contact email
      (``DeepSeek-AI ‡ Tsinghua University research@deepseek.com``);
    * two or more markers, i.e. a symbol legend
      (``∗ Corresponding author. † … ‡ …``);
    * an institution cue plus a contact email.

    Kept precision-first by three guards: CJK blocks are excluded (already in
    the target script), blocks over ``_IDENTITY_MAX_WORDS`` are excluded (prose
    that merely names a university), and a marker/email must accompany the
    institution word so an acknowledgements sentence is not swept in.
    """
    if contains_cjk(text) or len(text.split()) > _IDENTITY_MAX_WORDS:
        return False
    markers = len(_REFERENTIAL_MARKER_RE.findall(text))
    has_email = _EMAIL_RE.search(text) is not None
    has_institution = _INSTITUTION_RE.search(text) is not None
    if markers >= 2:
        return True
    if markers >= 1 and (has_institution or has_email):
        return True
    return has_email and has_institution


def classify_skip(
    source_text: str, *, is_heading: bool = False, in_bibliography: bool = False
) -> str | None:
    """Return a skip reason when the block must ship verbatim, else None.

    Almost pure function of the source text: CODE / IMAGE / FORMULA blocks are
    already skipped upstream, and this covers the long tail of untranslatable
    residue inside NARRATIVE / HEADING / LIST_ITEM flows. ``is_heading`` is the
    one thing the text cannot say — a heading is structure the author wrote
    ("Purpose, Scope, and Audience"), never a paper's author list.
    """
    text = re.sub(r"\xad\s*", "", (source_text or "")).strip()
    if not text:
        return None
    if _HANDLE_RE.match(text):
        return "social-handle watermark / running-head chrome"
    if in_bibliography and not is_heading and not _NARRATIVE_PROSE_RE.search(text):
        return "bibliography entry (kept verbatim for retrievability)"
    if _ARXIV_RE.search(text) or (
        _BRACKET_REF_RE.match(text)
        and _YEAR_RE.search(text)
        and not _NARRATIVE_PROSE_RE.search(text)
    ):
        return "bibliography entry (kept verbatim for retrievability)"
    if _is_bib_entry(text):
        return "bibliography entry (kept verbatim for retrievability)"
    if not is_heading and _is_author_byline(text):
        return "author byline (foreign names kept in romanization per GB/T 7714)"
    if not is_heading and _is_bibliographic_identity(text):
        return "bibliographic identity (affiliation/legend kept verbatim)"
    if not contains_cjk(text) and _WORD_RE.search(text) is None:
        return "pure symbol/numeric debris (no translatable words)"
    return None
