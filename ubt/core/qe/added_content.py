"""0-token added-content gate — the complement of :mod:`ubt.core.qe.omission`.

The omission gate catches content a translation *lost*. Nothing caught content
it *gained*, and that blind spot let a whole failure class ship unchanged
(chapter-3): the drafter is handed a read-only neighbour excerpt, translates
*that* as well, and prepends it to its own output. The added text is fluent,
on-topic and already in the target language, so every existing gate passed it.
Worse, the omission gate only reads the *low* side of the length band, so the
invented sentences actively masked the real sentence they displaced.

Two deterministic signals, both language-agnostic and free:

1. **Reference containment** — a citation (``[25]``), or an equation / figure /
   table number (``(3.11)``, ``Figs. 3.14 and 3.15``, ``图 3.9``) that the
   target cites but the source never mentions. Reference numbers are the one
   part of a paragraph that must survive translation one-for-one, so an extra
   token is direct evidence of invented content. It is also the cheapest
   hallucination trip-wire available: the echoed text in chapter-3 announced
   itself with a citation (``[2]``) and an equation number (``(2.2)``) that
   appear nowhere in its source.

2. **Prompt-scaffold leakage** — the target opens a markdown heading where the
   source has none. Template labels are the shape this prompt uses to delimit
   its own sections; if the source carries no heading, a heading in the output
   is the model reproducing scaffolding it was shown rather than translating.
   ``pdf_main#b0009`` shipped with a literal ``### 源段落翻译`` line the model
   invented by imitating the prompt.

Why a *check* and not a stronger prompt: contextual echo cannot be prevented by
instruction alone (the injected excerpt is already labelled
``[READ-ONLY ... DO NOT TRANSLATE OR ECHO]`` and was translated anyway). A gate
is the only layer that also covers output that is already in the wild — notably
Translation-Memory entries, which are served verbatim on every later run.

Failure routing is inherited from the fast-pass chain: a failing draft becomes
``REPAIR_PENDING`` with :attr:`AddedContentDecision.reason` surfaced verbatim as
the fix instruction, and a failing TM hit is discarded in favour of a fresh
draft. Both are fail-loud, neither discards content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Reference keyword, source and target language spellings. "Fig. 3.9" and
# "图 3.9" are the same reference written two ways and must collapse to one
# token, or every correctly translated figure callout would look fabricated.
_REF_KEYWORD = (
    r"(?:§+|FIG|Figs?|Figures?|Eqs?|Equations?|Tables?|Tabs?|Secs?|Sections?|Chaps?|Chapters?|"
    r"Apps?|Appendi(?:x|ces)|List(?:ing)?s?|"
    # ``列表`` before ``表``: without the list keyword the single ``表``
    # (table) matched inside ``列表 1`` (Listing 1), fabricating a reference the
    # source's ``List. 1`` never produced — the false positive that quarantined
    # every translated listing callout.
    r"图|式|列表|清单|清單|表|方程|附录|附錄)"
)
# Version-shaped number: chapter-3 uses "3.11", three-level section numbers use
# "3.4.1" and appendices use "A.10". The appendix letter is uppercase by
# convention, so it stays case-sensitive even inside the case-insensitive
# reference patterns; without the letter/third-part alternatives the gate saw
# no token at all for those references and stayed silent about them.
_VERSION_NUM = r"(?:(?-i:[A-Z])\.\d{1,2}(?:\.\d{1,2})?|\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)"
# Docling emits version numbers space-separated ("E q s . \, ( 3 . 7 )"), so the
# digit-dot-digit run is collapsed before matching. Without it a raw docling
# source yields an empty reference set and every correctly translated callout
# looks fabricated.
# The trailing digit is a lookahead, not consumed: a consumed one ends the
# match and the next spaced level ("3 . 4 . 1") is never folded, leaving only
# "3.4" and flagging the correct three-level callout as fabricated.
_VERSION_SPACING_RE = re.compile(r"(\d)(?:\.(?=\d)|\s+\.\s+)(?=\d)")
# Coordinated list: "Figs. 3.14 and 3.15", "Eqs. (3.7), (3.8)", "图 3.14 和图 3.15".
# Missing this is what produced the one false positive in calibration: a source
# reading "Figs. 3.14 and 3.15" yielded only 3.14, so the correct 3.15 looked
# fabricated.
_LIST_SEP = r"(?:\s*(?:,|，|、|和|and|或|or|[-–—~～至到]|to)\s*)"
# A reference number is version-shaped ("3.4.1") or a bare integer ("Chapter 5");
# omitting the bare form made the English path asymmetric with ``_CN_SECTION_RE``.
_REF_NUM = rf"(?:{_VERSION_NUM}|\d{{1,3}})"
# A keyword reference may cite a bare integer ("Chapter 5") as well as a
# version-shaped number ("3.4.1"); see ``_REF_NUM``.
_REF_RUN_RE = re.compile(
    rf"{_REF_KEYWORD}\.?\s*[\(（]?\s*({_REF_NUM}(?:{_LIST_SEP}[\(（]?{_REF_NUM}[\)）]?)*)",
    re.IGNORECASE,
)
# Bare parenthesised reference: "(3.11)" or "（3.11）" with the keyword elided.
_PAREN_REF_RE = re.compile(rf"[\(（]\s*({_VERSION_NUM})\s*[\)）]")
# Chinese section/chapter references put the number BEFORE the keyword
# ("第 3.2 节", "第3章", "见附录 A"): a fabricated echo citing a section carried
# no token and slipped past the gate. English "Ch. 3"/"Sec. 3.2" are already
# covered by the keyword-first _REF_RUN_RE.
_CN_SECTION_RE = re.compile(rf"第\s*({_VERSION_NUM}|\d{{1,3}})\s*(?:章|节|節|部|款|附录|附錄)")
# Any version-shaped number in the raw text, for the parenthesised-number
# false-positive exemption below.
_VERSION_NUMBER_RE = re.compile(_VERSION_NUM, re.IGNORECASE)
# Citation marker: "[25]", "[27,28]", "[25-28]".
_CITATION_SEP = r"(?:\s*(?:[,，]|[-–—~～]|to|至|到)\s*)"
_CITATION_RE = re.compile(rf"\[\s*(\d{{1,3}}(?:{_CITATION_SEP}\d{{1,3}})*)\s*\]")
_CITATION_RANGE_RE = re.compile(r"(\d{1,3})\s*(?:[-–—~～]|to|至|到)\s*(\d{1,3})")
_NUM_RE = re.compile(_REF_NUM, re.IGNORECASE)

# ATX markdown heading, 0-3 leading spaces per CommonMark. Setext underlines
# are deliberately not matched: a two-word source line followed by a rule is
# ordinary prose in these documents.
_ATX_HEADING_RE = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+\S.*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class AddedContentDecision:
    """Outcome of the added-content probe."""

    passed: bool
    reason: str
    #: Reference tokens present in the target but absent from the source.
    fabricated_refs: tuple[str, ...] = ()
    #: Headings present in the target while the source carries none.
    leaked_headings: tuple[str, ...] = ()
    #: Full reference sets, kept for diagnostics and tests.
    source_refs: frozenset[str] = field(default_factory=frozenset)
    target_refs: frozenset[str] = field(default_factory=frozenset)


def _cn_section_numbers(text: str) -> set[str]:
    """Numbers from Chinese ``第 N 章/节/附录`` references."""
    return set(_CN_SECTION_RE.findall(text))


def _reference_token_sets(text: str) -> tuple[frozenset[str], frozenset[str]]:
    """Return ``(all_tokens, anchored_tokens)`` for ``text``.

    *Anchored* tokens are the ones the text writes **as a reference** — attached
    to a reference keyword (``Fig. 3.5``, ``图 3.5``, ``第 3.2 节``) or a
    citation marker (``[25]``). A number that appears only inside bare
    parentheses with the keyword elided (``(3.5)``) is deliberately *not*
    anchored: rendering a plain quantity in parentheses is a legitimate
    translation choice, so the gate may match it against the source's bare
    numbers. An anchored token may not — turning a source quantity into a
    callout the source never had is exactly the fabrication this gate exists
    to catch.
    """
    text = _VERSION_SPACING_RE.sub(r"\1.", text)
    all_tokens: set[str] = set()
    anchored: set[str] = set()
    for run in _REF_RUN_RE.findall(text):
        found = set(_NUM_RE.findall(run))
        all_tokens.update(found)
        anchored.update(found)
    all_tokens.update(_PAREN_REF_RE.findall(text))
    cn_sections = _cn_section_numbers(text)
    all_tokens.update(cn_sections)
    anchored.update(cn_sections)
    for group in _CITATION_RE.findall(text):
        found = set(re.findall(r"\d{1,3}", group))
        for m in _CITATION_RANGE_RE.finditer(group):
            try:
                start_n, end_n = int(m.group(1)), int(m.group(2))
                if 0 <= end_n - start_n <= 50:
                    found.update(str(i) for i in range(start_n, end_n + 1))
            except ValueError:
                pass
        all_tokens.update(found)
        anchored.update(found)
    return frozenset(all_tokens), frozenset(anchored)


def reference_tokens(text: str) -> frozenset[str]:
    """Every citation / equation / figure / table number mentioned in ``text``.

    Returns the bare number (``"3.11"``), never the surrounding phrasing, so the
    same reference written as ``Eq. (3.11)`` in the source and ``式 (3.11)`` in
    the target compares equal.
    """
    return _reference_token_sets(text)[0]


def markdown_headings(text: str) -> tuple[str, ...]:
    """ATX markdown headings in ``text``, normalised and order-preserving."""
    return tuple(ln.strip() for ln in _ATX_HEADING_RE.findall(text))


class AddedContentGate:
    """Deterministic ``0``-token probe for translation-introduced content."""

    def evaluate(self, source_text: str, target_text: str) -> AddedContentDecision:
        """Return whether ``target_text`` stays inside ``source_text``'s content."""
        src = source_text or ""
        tgt = target_text or ""
        src_refs = reference_tokens(src)
        tgt_refs, tgt_anchored = _reference_token_sets(tgt)

        # A number is only "fabricated" when the target writes it as a reference
        # (anchored) the source never had, OR the source does not carry that bare
        # number anywhere. The second clause keeps a correct translation that
        # parenthesises a plain quantity ("3.5 times" -> "(3.5) 倍") from being
        # flagged; the first keeps that exemption from hiding a target-only
        # callout ("3.5 times" -> "见图 3.5") built out of a source quantity.
        # Citation tokens ([25]) are never version-shaped, so they keep flagging
        # unconditionally.
        src_numbers = set(_VERSION_NUMBER_RE.findall(_VERSION_SPACING_RE.sub(r"\1.", src)))
        # Chinese section references ("第 1 章" for "Chapter 1") use a bare
        # integer; exempt one that appears anywhere in the source so a heading
        # translation is not read as a fabricated section citation.
        src_ints = set(re.findall(r"\d{1,3}", src))
        cn_target = _cn_section_numbers(tgt)
        fabricated = tuple(
            sorted(
                t
                for t in (tgt_refs - src_refs)
                if (t in tgt_anchored or t not in src_numbers)
                and not (t in cn_target and t in src_ints)
            )
        )
        if fabricated:
            shown = ", ".join(fabricated[:4])
            return AddedContentDecision(
                passed=False,
                reason=(
                    f"Added reference(s): the target cites {shown} which the "
                    "source does not mention — translate ONLY the text under "
                    "'### Source Paragraph to Translate'; any preceding or "
                    "subsequent context excerpt is read-only reference material "
                    "and its citations and equation numbers must not appear in "
                    "the output; delete the fabricated reference(s)"
                ),
                fabricated_refs=fabricated,
                source_refs=src_refs,
                target_refs=tgt_refs,
            )

        # Only fires when the source has no heading at all: with source headings
        # present, the model is legitimately reproducing structure, and heading
        # drift between the two languages carries no signal.
        src_headings = markdown_headings(src)
        if not src_headings:
            leaked = markdown_headings(tgt)
            if leaked:
                shown = ", ".join(repr(h[:40]) for h in leaked[:3])
                return AddedContentDecision(
                    passed=False,
                    reason=(
                        f"Prompt scaffold leaked into the target: the output "
                        f"opens markdown heading(s) {shown} while the source has "
                        "none — emit the translation as plain prose; headings "
                        "and section labels from the instructions are not part "
                        "of the text to translate"
                    ),
                    leaked_headings=leaked,
                    source_refs=src_refs,
                    target_refs=tgt_refs,
                )

        return AddedContentDecision(
            passed=True,
            reason="No added content detected",
            source_refs=src_refs,
            target_refs=tgt_refs,
        )
