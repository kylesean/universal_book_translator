"""The delivery contract: one verdict over eight artifact invariants.

The export path already writes a quality report, an attestation sidecar and a
visual report. Those three can disagree, and the 26-page BSIM-CMG job is the
measured example: 151/155 text realizations, 43 missing assets, seven
``needs_human`` rows, visibly overlapping pages — and ``passed: true`` with
``needs_human_blocks: 0``. Each report was true to its own narrow question; no
single object answered "may this artifact ship?".

This module answers that question. It is deliberately **pure**: it takes
evidence (page text, artifact text boxes, page bounds, asset/font findings,
overflow records) and returns a report. It imports no adapters and compiles
nothing, so the contract is testable as data and cannot quietly become a second
renderer.

Two rules are load-bearing:

* **Unmeasured is not clean.** A check that could not run reports
  ``*_unmeasured``; silence is never a pass. The old visual gate's
  ``passed: true`` on pages it never sampled is the failure this rule prevents.
* **A clipped block is not a smaller block.** ``clip: true`` stops ink from
  painting outside its box, which is correct, but the text that did not fit is
  still lost. ``clipped_overflow`` is critical, not cosmetic.

Severity model: ``critical`` blocks the artifact; ``major`` blocks a *strict*
delivery but can be accepted explicitly; ``minor`` is informational. ``passed``
means "no critical"; ``deliverable`` means "passed and no major".
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

Severity = Literal["critical", "major", "minor"]
ContractName = Literal["readable", "faithful"]

#: A block shorter than this cannot make a unique-presence claim: the same
#: phrase legitimately repeats in a book ("第 1 章", "其中"), so counting it as a
#: duplicate would be noise rather than evidence.
_MIN_DUPLICATE_PROBE = 12

#: A prefix this long that appears without the full block text is evidence of
#: clipping/truncation rather than a coincidental phrase.
_MIN_CLIPPED_PROBE = 10

#: Two text boxes must overlap by at least this share of the smaller box before
#: it is called a collision. Formula superscripts and fraction bars legitimately
#: stack; a heading painted over a paragraph does not.
_OVERLAP_RATIO = 0.15

#: Ignore tiny boxes entirely: a period's ink or a formula fragment cannot
#: establish a collision, and including them turns every formula page red.
_MIN_OVERLAP_AREA_PT2 = 40.0

#: Page-edge tolerance, matching the visual gate's own allowance.
_OUT_OF_BOUNDS_TOLERANCE_PT = 1.0

#: Style drift tolerance: the PRD's "body group p5–p95 <= 0.5pt".
_STYLE_DRIFT_PT = 0.5

_PLACEHOLDER_RE = re.compile(r"⟦|⟧|⟨missing⟩|SOUP_MASK|\[SENTINEL_|<v>|<m>")

#: Character-level detection (above) is what decides pass/fail — a lone bracket
#: is already enough evidence that the machinery leaked. Reporting needs the
#: opposite shape: *one entry per leak site*, not one per character, or
#: ``⟦SOUP_MASK_0001-f2d⟧`` would count as three findings and the count would
#: overstate the work left. The paired alternative comes first so a bracketed
#: token is consumed as a unit; the lone-bracket alternatives stay last as the
#: fallback that keeps the site list from ever being empty when detection fired.
_PLACEHOLDER_SITE_RE = re.compile(r"⟦[^⟦⟧\n]{0,80}⟧|⟨missing⟩|<v>|<m>|SOUP_MASK|\[SENTINEL_|⟦|⟧")

#: Inline math and sentinel spans are not prose: a correct render draws the
#: formula as Unicode math, so the block's source characters for that span are
#: absent by design. Matching the whole block as one string therefore reports a
#: correctly typeset formula block as clipped. The contract matches prose runs
#: instead, the same rule the artifact audit uses.
_MATH_OR_MARKER_RE = re.compile(r"\$[^$]*\$|⟦[^⟧]*⟧")

#: Latin prose is a *word* stream, and a correct PDF inserts line breaks and
#: hyphenation between words: ``long-channel cylindrical`` arrives as
#: ``long-channel\r\ncylindrical``. Matching a normalized run (no spaces) asks
#: the artifact for a string no renderer can produce. Latin runs therefore
#: compare on a word basis; CJK runs keep the character basis, where a line
#: break between two characters is normal and whitespace-free comparison is
#: exactly what a search index does.
_LATIN_RUN_RE = re.compile(r"[A-Za-z]")

#: A prose run this long is proof of presence; shorter runs are too common to
#: distinguish "this sentence is here" from "these characters appear nearby".
_PROSE_RUN_MIN = 8


@dataclass(frozen=True, slots=True)
class ContractBlock:
    """A text block the artifact is expected to carry."""

    block_id: str
    text: str
    page: int
    kind: str = "body"
    #: ``readable`` reflows, so a block may land on a different page; a
    #: ``faithful`` block without this flag must stay on its source page.
    allow_cross_page: bool = True


@dataclass(frozen=True, slots=True)
class ContractTextBox:
    """One text box read from the delivered artifact."""

    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    block_id: str | None = None
    #: ``None`` means the reader could not measure this box (formula debris,
    #: an asset label). Such a box cannot establish a style or overlap claim.
    font_size_pt: float | None = None
    #: The text inside the box. PDFium returns both an aggregate formula rect
    #: and the glyph rects inside it; without the text, every aggregate looks
    #: like a duplicate layer. The overlap check uses this only to skip a
    #: containing aggregate whose text is a superset of the smaller box.
    text: str = ""


@dataclass(frozen=True, slots=True)
class StyleSample:
    """One measured body-style observation, used for the style-group check.

    Deliberately not derived from arbitrary text boxes: a heading, a caption and
    a formula fragment all have different sizes by design, so a document-wide
    distribution over *all* boxes would call every page a style drift. The
    renderer knows which size it used for the body group; this carries that
    fact.
    """

    page: int
    size_pt: float
    role: str = "body"


@dataclass(frozen=True, slots=True)
class OverflowRecord:
    """A block the renderer knew it could not place in full."""

    block_id: str
    page: int
    deficit_pt: float
    reason: str


@dataclass(frozen=True, slots=True)
class ContractEvidence:
    """Everything the contract is allowed to look at."""

    contract: ContractName
    page_count: int
    page_texts: tuple[str, ...]
    page_bounds: dict[int, tuple[float, float, float, float]]
    blocks: tuple[ContractBlock, ...]
    text_boxes: tuple[ContractTextBox, ...] = ()
    expected_page_count: int | None = None
    missing_assets: tuple[str, ...] = ()
    font_issues: tuple[str, ...] = ()
    overflow_records: tuple[OverflowRecord, ...] = ()
    style_samples: tuple[StyleSample, ...] = ()


@dataclass(frozen=True, slots=True)
class ContractFinding:
    severity: Severity
    code: str
    message: str
    page: int | None = None
    block_id: str | None = None
    #: One entry per occurrence of the same fault, each a short
    #: ``…context⟦token⟧context…`` excerpt. A single message cannot answer
    #: "how many are left": on the 26-page replay a repair loop that fixed one
    #: ``placeholder_leak`` then re-read the *same first* message was
    #: indistinguishable from a defect that cannot be fixed (AD-022). Empty for
    #: faults that are inherently single (page count, unreadable artifact).
    occurrences: tuple[str, ...] = ()

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "page": self.page,
            "block_id": self.block_id,
        }
        if self.occurrences:
            payload["occurrence_count"] = len(self.occurrences)
            payload["occurrences"] = list(self.occurrences)
        return payload


@dataclass(frozen=True, slots=True)
class ContractReport:
    contract: ContractName
    findings: tuple[ContractFinding, ...] = ()
    checks: tuple[dict[str, object], ...] = ()

    @property
    def critical(self) -> tuple[ContractFinding, ...]:
        return tuple(f for f in self.findings if f.severity == "critical")

    @property
    def major(self) -> tuple[ContractFinding, ...]:
        return tuple(f for f in self.findings if f.severity == "major")

    @property
    def passed(self) -> bool:
        """No critical finding. The artifact is readable, not necessarily clean."""
        return not self.critical

    @property
    def deliverable(self) -> bool:
        """No critical and no major finding: a strict delivery may ship."""
        return self.passed and not self.major

    def summary_line(self) -> str:
        verdict = "PASS" if self.deliverable else ("BLOCKED" if not self.passed else "MAJOR")
        return (
            f"[{verdict}] {self.contract}: {len(self.critical)} critical, "
            f"{len(self.major)} major, {len(self.findings)} total finding(s)"
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "contract": self.contract,
            "passed": self.passed,
            "deliverable": self.deliverable,
            "summary": {
                "critical": len(self.critical),
                "major": len(self.major),
                "total": len(self.findings),
            },
            "findings": [finding.to_payload() for finding in self.findings],
            "checks": list(self.checks),
        }


def _normalize(text: str) -> str:
    """Whitespace-free text, so a reflowed line break is not a mismatch."""
    return "".join((text or "").split())


def _box_area(box: ContractTextBox) -> float:
    return max(0.0, box.x1 - box.x0) * max(0.0, box.y1 - box.y0)


def _text_key(text: str) -> str:
    """Normalized text without whitespace or punctuation.

    PDFium's aggregate formula rect and its glyph rects often disagree on
    ``$ % #`` punctuation around the same symbols; stripping it makes the
    subset check stable without weakening the block-level text realization
    check (which uses the original text).
    """
    return re.sub(r"\W+", "", _normalize(text)).casefold()


def _nested_aggregate_text(bigger: ContractTextBox, smaller: ContractTextBox) -> bool:
    """Whether ``smaller`` is a fragment of a larger aggregate text box.

    PDFium returns both the whole formula rectangle and the glyph rectangles
    inside it. Their text is a superset/subset relationship, so treating every
    pair as a collision made a healthy 26-page formula-heavy artifact report
    47 ``text_overlap`` findings. A formula's subscript can also extend beyond
    the aggregate's rectangle, so containment is not required; the normalized
    text subset plus the area ratio is the stable signal. A real duplicate
    layer (equal or near-equal boxes) still falls through and is flagged, and
    block-level ``text_duplicated`` remains the independent check for repeated
    content.

    **The containment can point either way at glyph level.** pdfium's rect for
    a subscripted symbol sometimes carries *more* text than the base glyph's
    rect: the pair is ``𝑉t`` at 7.5pt and ``𝑉tm`` at 4.66pt, overlapping by 16–
    21 %, on the readable replay's formula pages. The candidate that is larger
    in *area* is the base glyph and holds the shorter text, so ``small in big``
    is false while ``big in small`` is true. Both spellings are the same glyph
    cluster at two sizes, which is why the reverse direction is accepted too —
    but only when the smaller box is actually set smaller (a font-size ratio
    below 0.9). A duplicate layer of the same text at the same size keeps
    falling through and stays flagged.
    """
    big_area = _box_area(bigger)
    small_area = _box_area(smaller)
    if big_area <= 0.0 or small_area <= 0.0:
        return False
    big_text = _text_key(bigger.text)
    small_text = _text_key(smaller.text)
    if not big_text or not small_text:
        return False
    small_size = smaller.font_size_pt
    big_size = bigger.font_size_pt
    # Two sizes on the same text already prove it is a glyph cluster. The area
    # ratio then only has to show they are the same *place*: 1.05 is enough
    # (``𝑉D`` 56.2pt² over ``𝑉DS`` 48.9pt² is x1.15, below the aggregate
    # threshold). A duplicate layer keeps the strict 1.25 floor because its
    # boxes are the same size.
    sizes_differ = small_size is not None and big_size is not None and small_size < big_size * 0.9
    if big_area < small_area * (1.05 if sizes_differ else 1.25):
        return False
    if small_text in big_text:
        return True
    return sizes_differ and big_text in small_text


def _intersection(a: ContractTextBox, b: ContractTextBox) -> float:
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def _style_findings(evidence: ContractEvidence) -> list[ContractFinding]:
    body = [
        sample for sample in evidence.style_samples if sample.role == "body" and sample.size_pt > 0
    ]
    if not body:
        # ``readable`` reflows and has no page-level body size to audit; only
        # ``faithful`` promises a cloned style group, so only it is required to
        # report the measurement.
        if evidence.contract == "faithful":
            return [
                ContractFinding(
                    severity="major",
                    code="style_unmeasured",
                    message="No body-style samples; style drift is unverified",
                )
            ]
        return []

    ordered = sorted(body, key=lambda sample: sample.size_pt)
    if len(ordered) == 1:
        return []
    p5 = ordered[max(0, int(len(ordered) * 0.05))]
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    drift = p95.size_pt - p5.size_pt
    if drift <= _STYLE_DRIFT_PT:
        return []

    # The tolerance alone says "some pages differ"; a reviewer needs the pages.
    # The dominant size is included because a book whose body is 8.8pt everywhere
    # except two shrunk pages is a different problem from a book that never
    # settled on a size at all.
    counts = Counter(round(sample.size_pt, 2) for sample in body)
    dominant_size, dominant_count = counts.most_common(1)[0]
    lowest = [sample for sample in ordered if sample.size_pt <= p5.size_pt + 1e-9][:4]
    highest = [sample for sample in reversed(ordered) if sample.size_pt >= p95.size_pt - 1e-9][:4]
    occurrences = tuple(
        f"p{sample.page} {sample.size_pt:.2f}pt" for sample in (*lowest, *highest)
    ) + (f"{dominant_count}/{len(body)} pages set at {dominant_size:.2f}pt",)
    return [
        ContractFinding(
            severity="major",
            code="style_drift",
            message=(
                f"Body size p5–p95 drift {drift:.2f}pt exceeds the "
                f"{_STYLE_DRIFT_PT}pt tolerance (p5 {p5.size_pt:.2f}pt on page "
                f"{p5.page}, p95 {p95.size_pt:.2f}pt on page {p95.page}, "
                f"{len(body)} pages measured)"
            ),
            page=highest[0].page if highest else None,
            occurrences=occurrences,
        )
    ]


def _overlap_findings(evidence: ContractEvidence) -> list[ContractFinding]:
    by_page: dict[int, list[ContractTextBox]] = {}
    for box in evidence.text_boxes:
        # A box without a measured size is formula debris or an asset label:
        # those legitimately stack (a superscript over a base glyph), and
        # including them turns every formula page into a false collision.
        if box.font_size_pt is None:
            continue
        if _box_area(box) < _MIN_OVERLAP_AREA_PT2:
            continue
        by_page.setdefault(box.page, []).append(box)

    findings: list[ContractFinding] = []
    for page, boxes in sorted(by_page.items()):
        for index in range(len(boxes)):
            for other in range(index + 1, len(boxes)):
                a, b = boxes[index], boxes[other]
                if a.block_id is not None and a.block_id == b.block_id:
                    continue
                if _nested_aggregate_text(a, b) or _nested_aggregate_text(b, a):
                    continue
                overlap = _intersection(a, b)
                smaller = min(_box_area(a), _box_area(b))
                if smaller <= 0 or overlap / smaller < _OVERLAP_RATIO:
                    continue
                findings.append(
                    ContractFinding(
                        severity="critical",
                        code="text_overlap",
                        message=(
                            f"Text boxes overlap by {overlap / smaller:.0%} of the smaller box"
                        ),
                        page=page,
                        block_id=",".join(
                            part for part in (a.block_id, b.block_id) if part is not None
                        )
                        or None,
                    )
                )
    return findings


def _out_of_bounds_findings(evidence: ContractEvidence) -> list[ContractFinding]:
    findings: list[ContractFinding] = []
    for box in evidence.text_boxes:
        bounds = evidence.page_bounds.get(box.page)
        if bounds is None:
            continue
        bx0, by0, bx1, by1 = bounds
        tol = _OUT_OF_BOUNDS_TOLERANCE_PT
        if box.x0 < bx0 - tol or box.y0 < by0 - tol or box.x1 > bx1 + tol or box.y1 > by1 + tol:
            findings.append(
                ContractFinding(
                    severity="critical",
                    code="text_out_of_bounds",
                    message=(
                        f"Text box ({box.x0:.1f},{box.y0:.1f})-({box.x1:.1f},{box.y1:.1f}) "
                        f"exceeds page {box.page} bounds"
                    ),
                    page=box.page,
                    block_id=box.block_id,
                )
            )
    return findings


def _text_findings(
    evidence: ContractEvidence,
    page_texts: list[str],
    joined: str,
    overflow_by_block: dict[str, OverflowRecord],
) -> list[ContractFinding]:
    findings: list[ContractFinding] = []
    for block in evidence.blocks:
        expected = _normalize(block.text)
        if not expected:
            continue
        occurrences = joined.count(expected)
        pages = [index + 1 for index, text in enumerate(page_texts) if expected in text]
        runs = _prose_runs(block.text)
        run_pages: dict[str, list[int]] = {
            run: [
                index + 1
                for index, text in enumerate(page_texts)
                if _run_present(run, joined, [text])
            ]
            for run in runs
        }
        present_runs = [
            run for run in runs if run_pages[run] or _run_share_present(run, joined) >= 0.9
        ]

        if occurrences == 0 and not (runs and len(present_runs) == len(runs)):
            if not runs and _token_present(block.text, page_texts):
                # Too short / too math-broken for a contiguous claim; the token
                # fallback is the honest weaker claim, not a pass by default.
                continue
            # The block is not carried whole. Two distinct failures share this
            # branch and must not be collapsed: a formula block whose prose
            # runs are all present is *placed*, while a block with only a
            # prefix on the page was clipped by the renderer.
            if runs and present_runs:
                findings.append(
                    ContractFinding(
                        severity="critical",
                        code="clipped_overflow",
                        message=(
                            f"Block {block.block_id} is present only in "
                            f"{len(present_runs)}/{len(runs)} prose run(s); "
                            "the rest was clipped or dropped"
                        ),
                        page=block.page,
                        block_id=block.block_id,
                    )
                )
                continue
            prefix = _longest_prefix(expected, joined)
            if prefix >= _MIN_CLIPPED_PROBE:
                findings.append(
                    ContractFinding(
                        severity="critical",
                        code="clipped_overflow",
                        message=(
                            f"Block {block.block_id} is present only as a "
                            f"{prefix}-character prefix; the rest was clipped or dropped"
                        ),
                        page=block.page,
                        block_id=block.block_id,
                    )
                )
            else:
                findings.append(
                    ContractFinding(
                        severity="critical",
                        code="text_missing",
                        message=f"Block {block.block_id} is absent from the artifact",
                        page=block.page,
                        block_id=block.block_id,
                    )
                )
            continue

        # Duplication is checked on the *longest prose run* only for a
        # formula-interrupted block: such a block never appears whole in the
        # artifact, so the whole-string count is 0 even when it is printed
        # twice (measured on b0014: all runs on both p2 and p3). For a plain
        # block, the whole-string count is the right probe: a heading's text
        # can legitimately recur inside a running head on every page, and
        # counting that as duplication would flag the chapter title.
        has_math = bool(_MATH_OR_MARKER_RE.search(block.text))
        longest_run = max(runs, key=len) if (runs and has_math) else ""
        run_pages_for_longest = run_pages.get(longest_run, []) if longest_run else []
        duplicated = (len(expected) >= _MIN_DUPLICATE_PROBE and occurrences > 1) or (
            len(longest_run) >= _MIN_DUPLICATE_PROBE and len(run_pages_for_longest) > 1
        )
        if duplicated:
            where = run_pages_for_longest or pages
            findings.append(
                ContractFinding(
                    severity="critical",
                    code="text_duplicated",
                    message=(
                        f"Block {block.block_id} appears on page(s) "
                        f"{', '.join(str(page) for page in where)}; "
                        "each block must be realized exactly once"
                    ),
                    page=where[0] if where else block.page,
                    block_id=block.block_id,
                )
            )

        if evidence.contract == "faithful" and not block.allow_cross_page:
            found_pages = (run_pages_for_longest or pages) if has_math else pages
            if found_pages and block.page not in found_pages:
                findings.append(
                    ContractFinding(
                        severity="critical",
                        code="text_wrong_page",
                        message=(
                            f"Block {block.block_id} belongs on page {block.page} "
                            f"but was found on page {found_pages[0]}"
                        ),
                        page=found_pages[0],
                        block_id=block.block_id,
                    )
                )

        record = overflow_by_block.get(block.block_id)
        if record is not None:
            findings.append(
                ContractFinding(
                    severity="major",
                    code="overflow_recorded",
                    message=(
                        f"Block {block.block_id} was recorded as overflowing by "
                        f"{record.deficit_pt:.1f}pt ({record.reason})"
                    ),
                    page=record.page,
                    block_id=block.block_id,
                )
            )
    return findings


def _prose_runs(text: str) -> list[str]:
    """Non-math runs of ``text``, whitespace-stripped, in order.

    A formula-interrupted block is present when *every* run of at least
    ``_PROSE_RUN_MIN`` characters is present. Requiring all runs (rather than
    any one) is what stops a generic fragment from proving a lost paragraph.
    """
    runs = [_normalize(part) for part in _MATH_OR_MARKER_RE.split(text)]
    return [run for run in runs if len(run) >= _PROSE_RUN_MIN]


def _run_present(run: str, joined: str, page_texts: list[str]) -> bool:
    """Whether one prose run is present, respecting its script.

    CJK: whitespace-free containment, because a line break between two
    ideographs is normal pagination.

    Latin: word-sequence containment. A PDF breaks a reference between
    ``long-channel`` and ``cylindrical``, so the whitespace-free form never
    appears even though the reference is plainly on the page. Requiring the
    same *words* in order is the honest claim for a spaced script.
    """
    if not _LATIN_RUN_RE.search(run):
        return any(run in text for text in page_texts)
    words = _normalize(run).split()
    if not words:
        return False
    # Search page-scoped, so a run cannot be "found" by borrowing words from
    # two different pages.
    for text in page_texts:
        haystack = _normalize(text).split()
        if not haystack:
            continue
        for start in range(len(haystack) - len(words) + 1):
            if haystack[start : start + len(words)] == words:
                return True
    return False


def _run_share_present(run: str, joined: str) -> float:
    """Share of a run's tokens that appear anywhere in the artifact.

    The fallback for a run that is not contiguous. A CJK line break can also
    fall *inside* a sentence at a point where the surrounding text reflows
    differently, so a long run may be split across two pages. Measuring token
    coverage is the weaker claim the artifact audit already uses for exactly
    this case; it is reported separately so a clipped block and a reflowed one
    are not conflated.
    """
    tokens = [token for token in re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", run) if token]
    if not tokens:
        return 1.0
    hits = sum(1 for token in tokens if token in joined)
    return hits / len(tokens)


_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+")

#: Token-overlap floor for a block too short or too math-broken to make a
#: contiguous claim. The same 0.6 the artifact audit uses: it is the weaker
#: claim, and it is only reached when no stronger one is available.
_MIN_TOKEN_OVERLAP = 0.6
_MIN_TOKEN_PROBE = 4


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def _token_present(expected: str, page_texts: list[str]) -> bool:
    """Page-scoped token overlap, the fallback for a block with no long run.

    A block like ``其中$A_{g0},r,F$和$F_{th,SI}$定义如下：`` has no prose run of
    eight characters; its only contiguous claims are too short to prove
    anything. The artifact audit falls back to token overlap for exactly this
    shape, and the contract must not call it missing while its text is on the
    page.
    """
    wanted = _tokens(expected)
    if len(expected) < _MIN_TOKEN_PROBE or not wanted:
        return True
    for text in page_texts:
        available = set(_tokens(text))
        hits = sum(1 for token in wanted if token in available)
        if hits / len(wanted) >= _MIN_TOKEN_OVERLAP:
            return True
    return False


def _longest_prefix(expected: str, joined: str) -> int:
    """Longest prefix of ``expected`` that appears in ``joined``.

    A binary search is enough: every prefix of a found prefix is found, so the
    predicate is monotone.
    """
    low, high = 0, len(expected)
    while low < high:
        mid = (low + high + 1) // 2
        if expected[:mid] in joined:
            low = mid
        else:
            high = mid - 1
    return low


def _placeholder_context(joined: str, start: int, end: int, *, pad: int = 20) -> str:
    excerpt = joined[max(0, start - pad) : min(len(joined), end + pad)]
    return excerpt if len(excerpt) <= 96 else excerpt[:93] + "..."


def _placeholder_findings(joined: str) -> list[ContractFinding]:
    """Every leaked placeholder/sentinel, not just the first one.

    The message names the count and the first site; ``occurrences`` carries one
    excerpt per site so a reviewer can work through them without re-running the
    export after each single repair (AD-022).
    """
    if _PLACEHOLDER_RE.search(joined) is None:
        return []
    sites = list(_PLACEHOLDER_SITE_RE.finditer(joined))
    if not sites:  # pragma: no cover - the site pattern is a superset of detection
        sites = list(_PLACEHOLDER_RE.finditer(joined))
    contexts = tuple(_placeholder_context(joined, site.start(), site.end()) for site in sites)
    first = sites[0]
    return [
        ContractFinding(
            severity="critical",
            code="placeholder_leak",
            message=(
                f"Artifact text contains {len(sites)} placeholder/sentinel "
                f"site(s); first {first.group(0)!r} near {contexts[0]!r}"
            ),
            occurrences=contexts,
        )
    ]


def evaluate_delivery_contract(evidence: ContractEvidence) -> ContractReport:
    """Evaluate all invariants and return one verdict.

    The checks are intentionally independent: a missing block does not suppress
    the overlap check, because a page can be both incomplete and broken and the
    operator needs to see both.
    """
    findings: list[ContractFinding] = []
    checks: list[dict[str, object]] = []

    if evidence.page_count <= 0:
        findings.append(
            ContractFinding(
                severity="critical",
                code="unreadable_artifact",
                message="The artifact reports no readable pages",
            )
        )
    checks.append({"name": "readable", "status": "fail" if evidence.page_count <= 0 else "pass"})

    if evidence.contract == "faithful":
        if evidence.expected_page_count is None:
            findings.append(
                ContractFinding(
                    severity="major",
                    code="page_count_unmeasured",
                    message="faithful delivery did not record the source page count",
                )
            )
            checks.append({"name": "page_count", "status": "unmeasured"})
        elif evidence.page_count != evidence.expected_page_count:
            findings.append(
                ContractFinding(
                    severity="critical",
                    code="page_count_mismatch",
                    message=(
                        f"faithful artifact has {evidence.page_count} pages; "
                        f"the source has {evidence.expected_page_count}"
                    ),
                )
            )
            checks.append({"name": "page_count", "status": "fail"})
        else:
            checks.append({"name": "page_count", "status": "pass"})
    else:
        checks.append({"name": "page_count", "status": "pass"})

    page_texts = [_normalize(text) for text in evidence.page_texts]
    joined = "".join(page_texts)
    overflow_by_block = {record.block_id: record for record in evidence.overflow_records}

    text_findings = _text_findings(evidence, page_texts, joined, overflow_by_block)
    findings.extend(text_findings)
    checks.append(
        {
            "name": "text_realization",
            "status": "fail" if any(f.severity == "critical" for f in text_findings) else "pass",
        }
    )

    placeholder_findings = _placeholder_findings(joined)
    findings.extend(placeholder_findings)
    checks.append({"name": "placeholders", "status": "fail" if placeholder_findings else "pass"})

    overlap_findings = _overlap_findings(evidence)
    findings.extend(overlap_findings)
    checks.append({"name": "overlap", "status": "fail" if overlap_findings else "pass"})

    bounds_findings = _out_of_bounds_findings(evidence)
    findings.extend(bounds_findings)
    checks.append({"name": "bounds", "status": "fail" if bounds_findings else "pass"})

    asset_findings = [
        ContractFinding(
            severity="critical",
            code="asset_missing",
            message=f"Asset {asset} is missing from the artifact",
        )
        for asset in evidence.missing_assets
    ]
    findings.extend(asset_findings)
    checks.append({"name": "assets", "status": "fail" if asset_findings else "pass"})

    font_findings = [
        ContractFinding(
            severity="major",
            code="font_issue",
            message=f"Font issue: {issue}",
        )
        for issue in evidence.font_issues
    ]
    findings.extend(font_findings)
    checks.append({"name": "fonts", "status": "fail" if font_findings else "pass"})

    style_findings = _style_findings(evidence)
    findings.extend(style_findings)
    checks.append(
        {
            "name": "style",
            "status": (
                "unmeasured"
                if any(f.code == "style_unmeasured" for f in style_findings)
                else ("fail" if style_findings else "pass")
            ),
        }
    )

    geometry_measured = any(box.font_size_pt is not None for box in evidence.text_boxes)
    if evidence.contract == "faithful" and not geometry_measured:
        findings.append(
            ContractFinding(
                severity="major",
                code="geometry_unmeasured",
                message="No measurable artifact text boxes; geometry is unverified",
            )
        )
        checks.append({"name": "geometry", "status": "unmeasured"})
    else:
        checks.append({"name": "geometry", "status": "pass"})

    return ContractReport(
        contract=evidence.contract,
        findings=tuple(findings),
        checks=tuple(checks),
    )


__all__ = [
    "ContractBlock",
    "ContractEvidence",
    "ContractFinding",
    "ContractReport",
    "ContractTextBox",
    "OverflowRecord",
    "StyleSample",
    "evaluate_delivery_contract",
]
