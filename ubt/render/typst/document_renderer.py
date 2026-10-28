r"""Typst document renderer: IRBlock sequence → one Typst document → PDF.

This renderer rebuilds the document from the IR and lets Typst lay it out,
which is what makes a uniform body size, native tables and correct math
possible at all (RFC-2026-02 §1.2). It does not draw fragments back onto the
source page: that would inherit every limit of the source geometry (font sizes
shrunk to fit a fixed box, tables left in the source language, a block that
cannot fit quarantined).

Design notes that are load-bearing:

* **Math is not re-implemented.** LaTeX→Typst lives in
  :mod:`ubt.render.typst.math_transpiler` and markup escaping in
  :mod:`ubt.render.typst.escape`; this module only decides *where* they apply.
* **Escaping is structural, not textual.** Prose is escaped, math spans are
  converted, and the two never mix: escaping a ``$`` that opens real math would
  turn the formula into literal text.
* **Tables are rebuilt, not overlaid.** A Markdown grid becomes a Typst
  ``#table``, so the header can be translated and the columns re-measured.
* **A figure carries its caption.** A picture followed by a short paragraph is
  the shape every extractor produces; pairing them keeps the caption with its
  image instead of orphaning it below the following section.
* **Deterministic.** Same blocks in, same bytes out (no timestamps, no set
  iteration in the emitted source), so the artifact is reproducible.
"""

from __future__ import annotations

import hashlib
import re
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ubt.core.content.nodes import ReconstructionRecord

# The wrapper predicates are re-exported under their historical private names:
# existing callers and tests import ``_unwrap_mark`` / ``_is_placeholder`` here.
# The implementation lives in ``ubt.render.delivered_text`` so the companion
# HTML/EPUB views share it (they previously had no wrapper handling at all).
from ubt.render.delivered_text import delivered_text, dropped_page_furniture, is_toc_entry
from ubt.render.delivered_text import is_placeholder as _is_placeholder
from ubt.render.delivered_text import unwrap_mark as _unwrap_mark  # noqa: F401 -- re-exported
from ubt.render.inline_spans import block_runs, locate_runs
from ubt.render.table_compiler import parse_markdown_table, verify_table_reconstruction
from ubt.render.typst.artifact_guard import find_sentinels, has_sentinel
from ubt.render.typst.escape import typst_escape
from ubt.render.typst.math_transpiler import (
    clean_math_fallback_text,
    strip_math_delimiters,
    typstify_math,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from ubt.core.ir.models import InlineRun, IRBlock

#: A display-math or inline-math span in the *source* text. Matched before
#: escaping so the formula is transpiled rather than escaped into literal text.
#: A match is only *candidate* math; ``_is_math_span`` decides.
_MATH_SPAN_RE = re.compile(r"(\$\$[\s\S]*?\$\$|\$[^$\n]+\$)")

#: CJK ideographs and kana/hangul.
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7a3]")

#: Signals that a span is *structure*, not prose: a LaTeX control sequence, a
#: math operator or relation, or a subscript/superscript. A formula that
#: annotates itself in Chinese (``$\mathbf{A} : ( D , N ) \text{ 参数 }$``) is
#: still math; a span made of nothing but Chinese words is not.
_MATH_STRUCTURE_RE = re.compile(r"\\[A-Za-z]+|[_^]|[=+<>≤≥×÷∑∏∈∀∃→←↔±∓⋅·]")

#: A span holding only punctuation is not a formula. A model that emits a stray
#: ``$`` (meaning ``$B$`` but writing ``$$``) leaves a ``$、$`` pair behind, and
#: accepting it as math shifts every later delimiter pairing — measured on mamba
#: ``pdf_main#b0086``, where ``$\Delta$、$$、$C$`` made ``$ 和矩阵 $`` a math
#: span and the surrounding prose was rendered as a formula.
_PUNCTUATION_ONLY_RE = re.compile(r"^[^\w\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+$")


def _is_math_span(span: str) -> bool:
    """True when a matched ``$...$`` span is really math, not prose.

    ``_MATH_SPAN_RE`` pairs delimiters positionally, so an unbalanced ``$`` shifts
    every later pairing and swallows prose into math mode. Measured on the gold
    corpus (alphafold2 ``pdf_main#b0146``): the model returned seven ``$`` for six
    delimiters, the pairing became ``$ 是一个序列，其中 $``, and Typst failed with
    ``unknown variable: 是一个序列`` — losing the entire document.

    Rejecting such a span is enough to repair it: the delimiters are then not
    treated as math, so the prose walk escapes them to literal ``\\$`` and the
    remaining formulas still pair correctly.

    CJK alone is **not** the test, though it is a strong hint. A formula that
    annotates its parts in Chinese is common in a translated document
    (``$\\mathbf{A} : ( D , N ) \\text{ 参数 }$``, ``$x = \\text{总令牌数}$``),
    and rejecting it escaped the whole formula as prose — measured on mamba,
    where ``\\$\\mathbf\\{A\\}\\} \\text\\{ )← 参数 \\}\\$`` shipped as literal
    backslashes. A span is prose only when it has CJK and **no** math
    structure, and a span of pure punctuation is never math at all (it is a
    stray delimiter, and treating it as math misaligns every later pair).
    """
    inner = span.strip().strip("$").strip()
    if not inner:
        return False
    if _PUNCTUATION_ONLY_RE.match(inner):
        return False
    if _CJK_RE.search(inner) is None:
        return True
    return _MATH_STRUCTURE_RE.search(inner) is not None


def _math_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of ``text`` that are genuinely math.

    The one owner of "what counts as a formula", shared by the render walk and
    the styled-run filter, so a run can never be kept against a span the walk
    would reject.
    """
    return [(m.start(), m.end()) for m in _MATH_SPAN_RE.finditer(text) if _is_math_span(m.group(0))]


#: A caption's opening marker (``Figure 1:``, ``图 1：``, ``Table 3.``). Used to
#: recognise the paragraph that belongs to the preceding figure.
_CAPTION_RE = re.compile(
    # ``Figure 1:`` / ``Fig. 2.`` / ``Table 3`` / ``图 1 |`` — the delimiter is a
    # colon, period, or the vertical bar the translators emit
    # (``图 1 | <caption text>``). Missing the bar meant a translated caption
    # failed this match, and one just over the length cap then rendered as a
    # standalone paragraph — the image and its caption split across pages on the
    # PaddleOCR paper.
    r"^\s*(?:Figure|Fig\.?|Table|Listing|Algorithm|图|表|附图|附表)\s*[\d.]+\s*[:：.|｜]",
    re.IGNORECASE,
)

#: Figures narrower than this (in PDF points) render at their own physical width
#: rather than the page-spanning default. A4 text width is ~470pt, so anything
#: below it that is *small* (a logo at ~121pt) keeps its size; a wider figure
#: falls through to the ``figure_width`` cap.
#:
#: The native width is emitted as a *percentage of the container*, not as an
#: absolute ``pt`` length. Absolute points ignore the column: on the two-column
#: academic template the body column measures 228.98pt (470.55pt text width,
#: 1.2em gutter), so a 290pt figure ran ~61pt past its column and over the text
#: beside it — Typst does not clip, it draws. Measuring against A4 text width
#: instead (the old assumption of ~470pt) is what hid the overflow.
_FIGURE_NATIVE_MAX_PT = 300.0

#: A4 text width, the denominator for turning a source figure's point width into
#: the container-relative width Typst actually needs.
_FIGURE_REFERENCE_TEXT_WIDTH_PT = 470.55

#: Longest *unlabelled* paragraph that still binds as a figure's caption. A
#: labelled caption ("图 1 | …") binds at any length; this only catches the
#: caption whose label the extractor dropped, and must stay short enough not to
#: swallow a body paragraph directly under an image.
_PENDING_CAPTION_MAX_CHARS = 320

#: Longest text that may bind as a caption *because it opens with a figure
#: label* ("图 3.1 …"). A caption is a single short line under an image; a
#: sentence of this length that merely mentions a figure is body prose. On the
#: reference chapter a 176-character narrative opening "Fig. 3.1 shows…" was
#: re-laid as a grey caption under the very figure it was discussing.
_CAPTION_LABEL_MAX_CHARS = 160

#: Longest description after a label that still reads as a caption. A caption
#: is a clause under an image ("示意图。"); prose that opens with a figure number
#: runs for several sentences.
_CAPTION_LABEL_BODY_MAX_CHARS = 40

#: Smallest share (in em) a table column keeps when widths are derived from
#: content. Stops a short column (``✓``, ``FPS``, ``3B``) from collapsing to zero
#: beside a long text column, while still letting the text column lead.
#:
#: 3 em is the old floor restated: 6 counted characters were about 3.3 Latin
#: advances or 3 ideographs, and a weight is now a measured em, not a count.
_TABLE_COL_MIN_WEIGHT = 3.0


def escape_markup(text: str) -> str:
    """Escape a run of prose for Typst markup.

    Delegates to :func:`ubt.render.typst.escape.typst_escape` (the single owner
    of the escape set) and adds nothing: a second escape set here would drift
    from that one, and a character escaped in one and not the other is a compile
    failure that only shows on the document that contains it.
    """
    return typst_escape(text)


#: Headings that name a table of contents rather than a section. Matched on the
#: whole text (a genuine section is never called just "Contents"), in any of the
#: languages the deliverable may spell it in.
_TOC_TITLE_TEXTS = frozenset({"contents", "table of contents", "目录", "目錄", "目次"})

#: A leading section label on a heading or TOC row: "1.2.", "3", "A.", "Appendix B".
#: Stripped before matching a TOC row against the headings it names, because the
#: row and the heading can disagree on the trailing dot or the spacing.
_HEADING_LABEL_PREFIX_RE = re.compile(
    r"^\s*(?:appendix\s+)?(?:[0-9]+(?:[.．][0-9]+)*[.．、)）]?|[A-Z][.．])?\s*",
    re.IGNORECASE,
)


def typst_lang_arguments(lang: str | None) -> str:
    """``lang: "zh", region: "CN"`` — the form Typst actually accepts.

    Typst validates ``lang`` as a bare ISO 639-1/2/3 code and rejects a BCP-47
    tag: ``#set text(lang: "zh-CN")`` fails to compile with *"expected two or
    three letter language code"*, and the hint tells you to put the country in
    ``region`` instead. This project stores target languages as BCP-47
    (``zh-CN``, ``pt-BR``), so passing one straight through turned every CJK
    deliverable into a compile failure — caught by the pre-flight check before
    any translation was billed, which is why it surfaced as "render pre-flight
    failed" on an otherwise correct configuration.

    Returns the complete argument fragment, including the leading ``, `` so a
    caller can append it to an existing ``#set text(...)`` call. An empty or
    unparseable tag yields ``""``: no ``lang`` is better than an invalid one,
    because the latter fails the whole document.
    """
    code = (lang or "").strip().replace("_", "-")
    if not code:
        return ""
    parts = [part for part in code.split("-") if part]
    primary = parts[0].lower()
    # Typst wants 2-3 letters; a 4-letter script subtag (``zh-Hans``) is not a
    # language code and must not be sent as one.
    if not (2 <= len(primary) <= 3) or not primary.isalpha():
        return ""
    args = f', lang: "{primary}"'
    # A region is the 2-letter (or 3-digit) subtag, skipping a script subtag.
    for part in parts[1:]:
        if len(part) == 4 and part.isalpha():
            continue  # script: Hans / Hant / Latn — Typst has no slot for it
        if (len(part) in (2, 3) and part.isalpha()) or (len(part) == 3 and part.isdigit()):
            args += f', region: "{part.upper()}"'
            break
    return args


def _is_toc_heading_text(text: str) -> bool:
    """Whether a heading's text names the table of contents itself."""
    return text.strip().lower() in _TOC_TITLE_TEXTS


def _group_toc_runs(blocks: Sequence[IRBlock]) -> list[tuple[int, list[IRBlock]]]:
    """Group consecutive ``toc_entry`` blocks into ``(start, run)`` pairs.

    A single TOC is a *run* of adjacent entries; a document may have more than
    one (a front TOC and an appendix listing), so each maximal run becomes its
    own table. A lone tagged row (a false positive) is still a run of one and is
    rendered compactly rather than promoted to a section heading. The start
    index lets the caller look *backwards* for the run's introducing heading.
    """
    runs: list[tuple[int, list[IRBlock]]] = []
    current: list[IRBlock] = []
    start = 0
    for index, block in enumerate(blocks):
        if is_toc_entry(block):
            if not current:
                start = index
            current.append(block)
        elif current:
            runs.append((start, current))
            current = []
    if current:
        runs.append((start, current))
    return runs


def _toc_title_key(text: str) -> str:
    """A heading's identity for matching a TOC row against the real headings.

    Case-folded, label-stripped and whitespace-collapsed: a TOC row is the
    heading's title with a page number beside it ("1.2. Attention 12"), while
    the heading itself carries no page. Matching the *text* is what separates a
    genuine contents page from five lines the reader happened to tag.
    """
    key = _HEADING_LABEL_PREFIX_RE.sub("", text.strip())
    key = re.sub(r"[\s\u3000]+", " ", key)
    return key.strip(" .:：·—-–").casefold()


def resolve_toc_mode(
    mode: str,
    *,
    layout_template: str,
    has_toc: bool,
    has_explicit_heading: bool = False,
    entry_count: int = 0,
) -> bool:
    """Whether to place the recovered table of contents.

    ``auto`` follows the document kind: short papers without an explicit TOC heading
    or with only a couple isolated entries omit the TOC (preventing spurious fragments).
    However, monographs, long academic reports, or papers with a genuine table of contents
    (possessing an explicit "Contents"/"目录" heading or >= 5 entries) preserve the TOC.
    ``on`` forces it, ``off`` suppresses it. A document with no recovered TOC rows never
    gets an invented one.
    """
    if not has_toc:
        return False
    normalized = str(mode or "auto").strip().lower()
    if normalized in ("off", "false", "0"):
        return False
    if normalized in ("on", "true", "1"):
        return True
    if layout_template != "academic":
        return True
    return bool(has_explicit_heading or entry_count >= 5)


def render_toc(entries: Sequence[tuple[str, str]]) -> str:
    """A table of contents: a heading plus one dot-leadered row per entry.

    ``entries`` are ``(title, page)`` pairs, title already delivered (translated
    or kept) and page the reader's recorded source-pagination destination. The
    whole thing is one full-width block so it neither flows into the body
    columns nor is mistaken for a run of section headings — both were the
    visible defect when TOC rows were emitted as ``#heading``s.

    Typst has no built-in dot leader, so each row is a three-cell grid — title,
    a fill-dotted spacer, page — with a generous row gutter and a two-step size
    hierarchy, so a long entry list reads as a list rather than a wall of text.
    """
    #: An entry title may already carry a section number ("2.1 …"); the indented
    #: look for a numbered subsection is left to the title itself rather than
    #: re-derived here (the reader's number is the source's own, not ours).
    lines = [
        "#block(width: 100%, above: 0.6em, below: 1.4em)[",
        '  #text(size: 14pt, weight: "bold")[目录]',
        "  #v(0.9em)",
        "  #grid(",
        "    columns: (1fr),",
        "    row-gutter: 0.6em,",
        "    align: left,",
    ]
    for title, page in entries:
        body = escape_markup(title.strip())
        number = escape_markup(page.strip())
        # Each row is a nested ``grid`` call. The outer ``#grid(...)`` puts its
        # arguments in *code* mode, so the row is a bare ``grid(...)`` (no ``#``)
        # whose cells are content arguments; ``box`` builds the dot leader and
        # ``repeat`` fills the spacer with dots.
        lines.append(
            "    grid("
            "columns: (auto, 1fr, auto), column-gutter: 0.55em, "
            "align: (left + horizon, left + horizon, right + horizon), "
            f"text(size: 10.5pt)[{body}], "
            "box(width: 100%, height: 0.75em, baseline: 0.35em, "
            "repeat[#text(fill: luma(195))[.]]), "
            f"text(size: 10.5pt)[{number}],"
            "),"
        )
    lines.extend(
        [
            "  )",
            "]",
        ]
    )
    return "\n".join(lines)


def render_prose(
    text: str,
    runs: Sequence[InlineRun] = (),
    *,
    is_math_ok: Callable[[str], bool] | None = None,
) -> str:
    """Render one paragraph: escape prose, transpile math, apply inline styles.

    The span split is the whole point. Escaping first would turn ``$x_i$`` into
    literal ``\\$x_i\\$``; transpiling first and escaping after would escape the
    ``#`` and ``"`` the transpiler emits. So the string is walked once, in
    order, and each half is handled by its own owner.

    ``runs`` are the block's styled spans (bold/italic/superscript/colour). A
    run's ``text`` is its literal characters, so it is located in ``text`` by
    substring — a span the translation rewrote finds no match and is drawn
    plain, which is the documented contract of :class:`InlineRun`. Style markup
    is applied *after* escaping, so the injected ``#strong`` is not escaped into
    literal text.

    ``is_math_ok`` is the compile gate for a converted math body (see
    :class:`~ubt.adapters.pdf.typst_math_probe.TypstMathProbe`). The transpiler
    can emit a body that parses in its own grammar but does not compile in Typst
    (``\\frac{a}{}`` -> ``frac(a, )``), and one such body fails the whole
    document. A body the gate rejects degrades to readable Unicode instead, the
    same fallback an unconvertible span takes. ``None`` skips the gate, which is
    what callers that only inspect the emitted source (tests) want.
    """
    spans = locate_runs(text, runs) if runs else []
    if spans:
        # A run that overlaps a math span would split the formula: the chunk
        # walk below renders each boundary slice on its own, so the ``$`` that
        # opens the span lands in one slice and the ``$`` that closes it in
        # another — each escaped as literal ``\$``. The formula then never
        # enters math mode and Typst reports an unbalanced delimiter, losing the
        # whole document.
        #
        # Any overlap disqualifies the run, not just full containment. A run
        # that merely *straddles* the closing ``$`` (start inside the formula,
        # end outside) is the case that bit: measured on the gold corpus, a
        # superscript run whose fuzzy match reached past the delimiter turned
        # ``$\{ x_k \} \to 0$是一个序列`` into
        # ``\$\\\{ #super[...][x\_k \\} \\to 0\$是]一个序列`` — the formula
        # shipped as literal backslashes and the document failed to compile.
        #
        # A run that *fully contains* the formula is kept: the whole span then
        # stays inside one chunk, so ``_render_math_and_prose`` sees balanced
        # delimiters and the math keeps its own escaping inside the wrapper.
        math_spans = _math_spans(text)
        spans = [
            span
            for span in spans
            if not any(
                span[0] < end and start < span[1] and not (span[0] <= start and end <= span[1])
                for start, end in math_spans
            )
            # A run whose *own* text has an odd number of ``$`` cannot be
            # wrapped: ``#strong[…]`` is emitted around the slice, so one
            # delimiter would sit outside the wrapper and its partner inside,
            # and Typst reads an unbalanced span. Measured on mamba
            # ``pdf_main#b0086``, where the model's bold marker began
            # mid-formula with the text ``$、$C$ 和矩阵
            # $\llbracket A \rrbracket$ 的投影）…`` (five delimiters) and the
            # render emitted ``\$#strong[\$`` — the formula as literal
            # backslashes. Dropping the style is the honest outcome: the run's
            # boundaries are not trustworthy when they fall inside a formula.
            and not _unbalanced_delimiters(text[span[0] : span[1]])
        ]
    if not spans:
        return _render_math_and_prose(text, is_math_ok=is_math_ok)
    boundaries: set[int] = {0, len(text)}
    for start, end, *_style in spans:
        boundaries.update((start, end))
    # Walk the raw text by span boundary so styles and math can overlap: a bold
    # span that contains a formula must bold the prose around it and leave the
    # math to the transpiler.
    out: list[str] = []
    for start, end in zip(sorted(boundaries), sorted(boundaries)[1:], strict=False):
        if end <= start:
            continue
        chunk = text[start:end]
        rendered = _render_math_and_prose(chunk, is_math_ok=is_math_ok)
        style = next((s for s in spans if s[0] <= start and end <= s[1]), None)
        if style is not None:
            rendered = _wrap_style(rendered, style[2], style[3], style[4], style[5])
        out.append(rendered)
    return _break_run_call_chains(out)


def _unbalanced_delimiters(text: str) -> bool:
    """Whether a run's text has an odd number of ``$`` delimiters.

    Such a run cannot be wrapped in ``#strong[…]``/``#emph[…]`` without moving a
    math delimiter across the wrapper boundary, which Typst reads as an
    unbalanced span. See the filter in :func:`render_prose`.
    """
    return text.count("$") % 2 == 1


def _break_run_call_chains(segments: Sequence[str]) -> str:
    """Join segments without letting a styled run absorb what follows it.

    **The rule, not a list.** A styled run ends in ``]`` (``#strong[...]``).
    Typst keeps parsing from that closing bracket, so whatever comes next is
    read *in code mode*: a ``(`` becomes a call on the run's result
    (``#strong[x](y)``), a ``.`` becomes a field access
    (``#super[...][m].installed``), both fail to compile, and the whole
    fragment descends to the source. Measured cases behind this: ``lo#super[...][g](4)``
    on the gold corpus, a citation's paren after a bold run (arXiv 2609.22978),
    and ``…[𝑚].installed𝑚`` on arXiv 2608.25512, which aborted a 92-page
    delivery with ``element super has no method 'installed𝑚'``.

    **What this used to be, and why that was wrong.** It was a blacklist: if the
    next segment began with ``(`` or with ``.`` followed by an identifier, the
    character was re-emitted as ``#text("…")``. That is correct for the two
    cases someone had hit, and it depends on an invariant nothing stated — that
    every *other* follower has already been backslash-escaped by
    :func:`ubt.render.typst.escape.typst_escape`. The invariant happens to hold
    (an exhaustive sweep of all 32 ASCII punctuation characters plus 14 CJK
    marks found no compile failure and no content loss), so the old code was not
    broken — it was correct by coincidence, and a future change to the escaper
    or to what may follow a run would break it silently.

    **The rule.** After every styled run, close the code expression
    unconditionally. ``/**/`` is a Typst comment: it terminates the expression
    with no output, no advance and no glyph, so ``#strong[x]/**/.installed``
    renders exactly as ``x`` followed by ``.installed``. Measured against Typst
    0.15.1: identical glyph positions and identical extracted text versus the
    old ``#text(".")`` emission, and it also neutralises the followers the
    blacklist never considered — ``/``, ``-``, ``*``, ``_``, ``[``, ``{``,
    ``@``, ``<``, ``#``, ``$`` and ``(` — all of which now parse as markup
    instead of continuing the call.

    The cost is one empty comment per styled run. A long paragraph with many
    runs gains a few dozen bytes of Typst source and no visual difference.
    """
    merged: list[str] = []
    for segment in segments:
        # Every segment after a styled run starts a fresh expression. No case
        # analysis: the terminator is valid before any character, including
        # none (a run at the end of a paragraph).
        if merged and merged[-1].endswith("]"):
            merged.append("/**/")
        merged.append(segment)
    return "".join(merged)


#: A masker placeholder (``⟦SOUP_MASK_0001-333⟧`` and the other families) or an
#: emphasis/semantic-break sentinel. One reaching the renderer means the restore
#: pass could not account for it; it must never be transpiled as math. Typst's
#: math mode mangles each family differently, and every one of them is a compile
#: failure that loses the whole document:
#:
#: * ``_`` is the subscript operator, so ``SOUP_MASK_0001`` becomes
#:   ``#"SOUP"_#"MASK"_0001`` — the ``#`` enters code mode and the compiler
#:   fails on ``unknown variable: SOUP``;
#: * ``/`` is the fraction operator, so a leaked ``⟦/B⟧`` fails with
#:   ``unexpected slash`` (mamba, neural-ode);
#: * ``\llbracket`` (how a model may transcribe ``⟦``) is not a Typst name, so
#:   it fails with ``unknown variable: lbracket``.
#:
#: The check lives in :mod:`ubt.render.typst.artifact_guard` so one predicate
#: covers every family the pipeline writes, rather than a regex per family here
#: that a new masker can fall behind.


def _render_math_and_prose(text: str, *, is_math_ok: Callable[[str], bool] | None = None) -> str:
    """Escape prose and transpile math spans, with no style handling.

    A math span the transpiler cannot convert does **not** fall back to escaped
    text: that would put ``\\unknowncmd{x}`` on the page, which is worse than
    the formula being unreadable — it is LaTeX source leaking into a finished
    document. The span degrades to readable Unicode instead
    (:func:`clean_math_fallback_text`), so a reader sees ``\u03d5 + \u03b3 \u2264 \u2211`` rather than
    command names. ``is_math_ok`` extends that fallback to a body the transpiler
    *accepts* but Typst rejects (``frac(a, )``), which would otherwise fail the
    whole compile.

    A pipeline sentinel inside a math span is escaped as prose rather than
    transpiled (see :func:`has_sentinel`): the token is an artifact, not math,
    and Typst's math mode mangles its characters into a compile failure.
    """
    parts = _MATH_SPAN_RE.split(text)
    out: list[str] = []
    for index, part in enumerate(parts):
        # A candidate span the pairing made out of prose (an unbalanced ``$``
        # shifted the delimiters) is escaped as prose instead of transpiled.
        if index % 2 == 1 and _is_math_span(part):
            body = part.strip()
            inner = body[2:-2] if body.startswith("$$") else body[1:-1]
            if has_sentinel(inner):
                # Escape the span's *inner* text as prose, keeping any backslash
                # the model wrote: the sentinel cannot be represented in math
                # mode, and half-transpiling around it would leave the
                # surrounding math in a shape the compiler also rejects. The
                # inner text is used rather than ``clean_math_fallback_text``
                # (which strips backslashes) so a transcribed ``\llbracket``
                # remains visible instead of degrading to ``llbracket``.
                out.append(escape_markup(inner))
                continue
            if body.startswith("$$"):
                converted = typstify_math(inner)
                if converted is not None and is_math_ok is not None and not is_math_ok(converted):
                    converted = None
                out.append(
                    f"\n$ {converted} $\n"
                    if converted
                    else escape_markup(clean_math_fallback_text(part))
                )
            else:
                converted = typstify_math(inner)
                if converted is not None and is_math_ok is not None and not is_math_ok(converted):
                    converted = None
                out.append(
                    f"${converted}$" if converted else escape_markup(clean_math_fallback_text(part))
                )
        else:
            out.append(escape_markup(part))
    return "".join(out)


def _wrap_style(
    segment: str, bold: bool, italic: bool, superscript: bool, color_hex: str | None
) -> str:
    """Wrap an escaped segment in the Typst markup its source style calls for.

    Applied after escaping, so the injected markup is never escaped.
    """
    if superscript:
        segment = f"#super(typographic: false, size: 0.62em)[{segment}]"
    if italic:
        segment = f"#emph[{segment}]"
    if bold:
        segment = f"#strong[{segment}]"
    if color_hex:
        segment = f'#text(fill: rgb("{color_hex}"))[{segment}]'
    return segment


#: Table body text size, as a fraction of the document body size.
#:
#: A Typst ``table`` sized in ``fr`` still cannot shrink a cell below its widest
#: *unbreakable* run. ``(7fr, 46fr, 43fr)`` inside a two-column paper's ~232pt
#: text column could not fit
#: ``// lib/devices/tpm: MmioAddr::new(base,len)``, so the cell overran the
#: column and its text drew on top of the neighbouring one (``p3:rect327``
#: reached 652.9pt on a 595.3pt page; ``p6:rect232`` reached 658.8pt).
#:
#: Shrinking the body is the lossless fix. The first attempt inserted
#: zero-width spaces at code boundaries, which did fit -- but a ZWSP is a real
#: character in the PDF's text layer, so the delivered file carried 67 of them
#: and ``MmioAddr::new`` could no longer be found by search or copied into an
#: editor. Typography must not corrupt the text it typesets.
#:
#: Default table body scale, used when a table's own measured minimum fits at
#: full size but the *emitted* body is deliberately smaller than prose: a table
#: set at 0.84em reads as tabular data rather than as a paragraph. The actual
#: scale for a table that does not fit is computed per table (``table_scale``
#: below) and emitted; this constant is only the "fits at a comfortable size"
#: value, never the value for a table that had to shrink.
#:
#: Measured on the ATC'25 table in a 232pt column: 288.3pt at 9.5pt (overflow),
#: 271.7pt at 8pt, 260.6pt at 7pt.
_TABLE_TEXT_SCALE: str = "0.84"

#: Floor for the computed shrink factor. A table whose columns cannot fit even
#: here is too wide for the page at any readable size; it is laid out at the
#: floor and the visual gate reports the overflow rather than hiding it. 0.60
#: gives a 6.3pt body -- small, but legible.
_TABLE_MIN_TEXT_SCALE: float = 0.60

#: How many lines a column's content may wrap to before the column counts as
#: unreadably thin. This is the readability floor for a column whose only
#: unbreakable run is short: a 7-character CJK header at 3 lines still reads as
#: a label, at 7 lines it reads as a vertical strip.
_TABLE_HEADER_LINES: float = 3.0

#: Smallest usable column width, in em, independent of what the cell happens to
#: contain. A column narrower than one glyph stacks its header one ideograph per
#: line and clips the body to a single character per row, which is what a
#: delivered 9-column comparison table did (columns 3 and 6 had 6.7pt of usable
#: width at a 10.5pt body). Applying it to the fit test -- not the slack phase --
#: makes such a table shrink until every column can hold a character.
_TABLE_MIN_COLUMN_EM: float = 1.0

#: Safety margin applied to the emitted column widths. Absorbs the 0.1pt
#: rounding in the spec plus sub-point differences between the measured advances
#: and Typst's own shaping. Measured case: a run needing 124.4pt in a 124.1pt
#: column crossed the text-block edge by 0.3pt.
_TABLE_WIDTH_SAFETY: float = 0.998

#: Body size the renderer emits (``#set text(size: 10.5pt)``). The em-based
#: table weights are expressed as fractions of this, so a weight of 1.0 means one
#: body em -- 10.5pt at the document size.
TABLE_BASE_PT: float = 10.5

#: Text-block widths the renderer emits, in points, per page setup. A table's
#: verdict is decided against the width it will actually be typeset into rather
#: than against a magic run length: the old ``22 display columns`` trigger came
#: from a hand measurement that the precise em widths contradict (22 Latin
#: characters measure 12.5 em = 131pt, not the 244pt the comment claimed), and
#: it shrank healthy tables while missing a real overflow.
#:
#: A4 is 595.28pt wide. Academic single-column uses ``margin x: 2.5cm`` (453.55pt
#: of text), the dual/两栏 layout uses ``x: 1.8cm`` and splits the 493.2pt text
#: block into two columns (219.5pt each, less the 12pt column gutter), and the
#: book template uses ``x: 2.2cm`` (470.55pt).
_PAGE_TEXT_WIDTH_PT: dict[str, float] = {
    "academic_1col": 453.55,
    "academic_2col": 219.5,
    "book": 470.55,
}
#: Fallback when the caller does not say which layout it is rendering into.
_TABLE_DEFAULT_TEXT_WIDTH_PT = _PAGE_TEXT_WIDTH_PT["academic_1col"]
#: Cell inset (5pt each side) plus a column gutter (0.7em) and rounding slack.
_TABLE_CELL_CHROME_PT = 10.0
_TABLE_GUTTER_EM = 0.7

#: Cell inset on each side, matching the emitted ``inset: 5pt``. A column's
#: width is its content plus twice this.
_TABLE_CELL_INSET_PT: float = 5.0

#: Column count at which a table cannot fit a two-column page's text width at
#: the document size, whatever its cell contents: 232pt / 10 columns leaves ~23pt
#: per column, less than three Latin characters plus the gutter. Such a table is
#: a whole-page exhibit and must be typeset smaller to survive.
_TABLE_MAX_COMFORTABLE_COLUMNS: int = 8


def render_table(
    markup: str,
    *,
    is_math_ok: Callable[[str], bool] | None = None,
    page_columns: int = 1,
    content_width_pt: float | None = None,
    scale_out: list[float] | None = None,
    allow_landscape: bool = False,
    table_scale_factor: float = 1.0,
    is_multipage: bool = False,
    enable_continuation: bool = False,
    table_uid: str = "",
    target_lang: str = "zh",
    continuation_config: object | None = None,
) -> str:
    """Render a Markdown table as a Typst ``#table`` with a three-line frame.

    Rebuilt from the IR rather than drawn over the source: a source-layer table
    cannot change its contents at all, which is why table headers stayed in the
    source language when tables were left on the source page (measured 0.96
    source-term leakage; RFC §9).

    ``is_math_ok`` is the same compile gate :func:`render_prose` takes; a cell's
    ``$\\frac{a}{}$`` reaches the compiler exactly as a paragraph's does.

    ``scale_out``, when given, receives the body scale actually emitted (one
    element). The caller needs it to tell a table set at a comfortable size from
    one squashed onto the 0.60 floor, which is the only thing that can emit the
    ``low_legibility_font`` render-quality flag.
    """
    parsed = parse_markdown_table(markup, escape=False)
    if parsed is None:
        return ""
    headers, rows = parsed
    columns = len(headers)
    if columns == 0:
        return ""

    def cells(values: Sequence[str], *, header: bool) -> str:
        # Cells arrive *unescaped* (``parse_markdown_table(..., escape=False)``)
        # and go through ``render_prose``, which owns escaping and math
        # transpilation — that is how a cell's ``$n^2$`` becomes real typeset
        # math. Pre-escaping them (the default contract, for callers that
        # interpolate cells raw) turned ``$`` into ``\\$``, and inside a ``[...]``
        # content block ``\\`` is a line break, so the math delimiter never
        # closed and Typst failed with ``unclosed delimiter``. Measured on the
        # gold corpus (attention-is-all-you-need ``pdf_main#b0062``, the layer
        # complexity table): the whole document was lost.
        padded = list(values) + [""] * (columns - len(values))
        rendered = []
        for value in padded:
            # ``<br>`` is the in-cell line break (Markdown has none, and Docling
            # stacks a wrapped cell's lines): split before prose rendering so
            # each line is escaped and math-transpiled on its own, then join with
            # Typst's markup line break. Rendering the whole value first would
            # escape the ``<`` and print a literal ``<br>``.
            text = "\\\n".join(
                render_prose(part.strip(), is_math_ok=is_math_ok)
                for part in re.split(r"<br\s*/?>", value.strip())
            )
            rendered.append(f"[*{text}*]" if header and text else f"[{text}]")
        return ", ".join(rendered)

    # Column widths follow content, not an even split. A 9-column benchmark
    # table with an even split crushed the "识别方法" column (wrapped
    # "MonkeyOCR [27]" onto three lines) and squeezed "公式 CDM↑" into one glyph
    # per line. Weight each column by its longest cell (header included), with a
    # floor so a short numeric column still gets room; ``1fr`` weights are
    # relative, so the table stays the text width.
    #
    # The weight is MEASURED IN EM, not counted in characters. Counting was the
    # defect behind a delivered table overflowing by 39.8pt: the column holding
    # ``metadata.openclaw.requires`` was weighted by its 26 characters and given
    # 105pt of room, while the identifier needs 125.03pt at the shrunk 8.82pt
    # size. Typst cannot break an identifier, so the cell ran past the page edge.
    #
    # The per-class advances below were measured against the actual render fonts
    # (Noto Serif CJK SC + Liberation Serif) at 10.5pt and are expressed as a
    # fraction of the 10.5pt em so they hold at any size:
    #
    #     CJK ideograph      10.50pt = 1.000 em
    #     CJK punctuation     5.25pt = 0.500 em
    #     upper-case Latin    7.33pt = 0.698 em
    #     lower-case Latin    5.85pt = 0.557 em
    #     digit               5.76pt = 0.548 em
    #     ASCII punctuation   3.82pt = 0.364 em
    #     space               2.69pt = 0.256 em
    #
    # The old ``2 if ord(c) > 0x2E80 else 1`` put the CJK:Latin ratio at 2.0
    # against a measured 1.79, but the larger error was the unit: one weight
    # step renders as 0.85pt, so a column's weight was never a length at all.
    _LATIN_LOWER_EM = 0.557
    _LATIN_UPPER_EM = 0.698
    _DIGIT_EM = 0.548
    _CJK_EM = 1.0
    _CJK_PUNCT_EM = 0.5
    _ASCII_PUNCT_EM = 0.364
    _SPACE_EM = 0.256

    #: Unicode general categories whose members break between characters in
    #: Typst. An ideograph is a break opportunity on its own; an ASCII letter is
    #: not, which is why a URL or an identifier has to fit a column whole.
    _BREAKABLE_RANGES = (
        (0x2E80, 0x9FFF),  # CJK radicals .. CJK unified ideographs
        (0xF900, 0xFAFF),  # CJK compatibility ideographs
        (0xFF00, 0xFFEF),  # half/full-width forms (full-width punctuation)
        (0x3000, 0x303F),  # CJK symbols and punctuation
        (0x1100, 0x11FF),  # Hangul Jamo
        (0xA960, 0xA97F),  # Hangul Jamo Extended-A
        (0xAC00, 0xD7FF),  # Hangul syllables + Jamo Extended-B
    )

    def _is_cjk(char: str) -> bool:
        code = ord(char)
        return any(low <= code <= high for low, high in _BREAKABLE_RANGES)

    def _is_hangul(char: str) -> bool:
        """Korean syllables and Jamo: full-width, and a break opportunity each."""
        code = ord(char)
        return (
            0x1100 <= code <= 0x11FF  # Jamo
            or 0xA960 <= code <= 0xA97F  # Jamo Extended-A
            or 0xAC00 <= code <= 0xD7FF  # syllables + Jamo Extended-B
        )

    def _is_cjk_punct(char: str) -> bool:
        """Full-width punctuation, which is its own break opportunity.

        ``。SYSTEM_PROMPT_DYNAMIC_BOUNDARY`` is two tokens, not one: the sentence
        ends and an identifier follows. Treating the run as unbreakable gave a
        20.4 em "run" to a column whose real token is 20.4-1 = 19.4 em, and the
        column was promoted to ``auto`` for a token that does not exist.
        """
        code = ord(char)
        return 0x3000 <= code <= 0x303F or 0xFF01 <= code <= 0xFF65

    def _char_em(char: str) -> float:
        """Measured advance of one character, as a fraction of the em."""
        if char.isspace():
            return _SPACE_EM
        if _is_hangul(char):
            # Hangul is full-width: syllables and Jamo advance a whole em, the
            # same as Han. It used to fall through to ``isalpha()`` -> 0.557em,
            # so a Korean target's columns were budgeted 44 % too narrow and
            # the table overflowed on every render.
            return _CJK_EM
        if _is_cjk(char):
            # Full-width ideographs and the full-width punctuation forms share a
            # 1em advance; the narrower CJK punctuation block does not.
            return _CJK_EM if ord(char) >= 0x3040 else _CJK_PUNCT_EM
        if char.isdigit():
            return _DIGIT_EM
        if "A" <= char <= "Z":
            return _LATIN_UPPER_EM
        if char.isalpha():
            return _LATIN_LOWER_EM
        return _ASCII_PUNCT_EM

    def _text_em(text: str) -> float:
        """Measured width of ``text`` in em."""
        return sum(_char_em(char) for char in text)

    def _longest_unbreakable(text: str) -> float:
        """Width in em of the widest run Typst cannot break.

        Three kinds of break opportunity, all verified against the compiler:

        * a space;
        * every CJK character -- Typst breaks between ideographs, so a CJK
          phrase is never one run (treating it as one inflated a column's figure
          from 27 to 89 em);
        * a hyphen -- ``Mini-SWE-Agent`` renders as three lines in a 40pt box.

        A dot is NOT a break opportunity: ``metadata.openclaw.requires`` renders
        as one 126pt run inside a 40pt box, and ``agentskills.io`` as one 61pt
        run. Those two are the strings that overflowed a delivered table.
        """
        widest = 0.0
        run = 0.0
        for char in text:
            # CJK punctuation MUST be tested before ``_is_cjk``: 。and ，live in
            # the U+3000..U+303F block that ``_is_cjk`` also covers, so testing
            # the broader class first let ``。SYSTEM_PROMPT_DYNAMIC_BOUNDARY``
            # stay one 20.4 em run and promoted a prose column to ``auto``.
            if _is_cjk_punct(char):
                widest = max(widest, run)
                run = 0.0
            elif char.isspace() or _is_cjk(char):
                widest = max(widest, run)
                run = _char_em(char) if not char.isspace() else 0.0
            elif char in "-\u2010\u2011\u2013\u2014":
                # Typst breaks after a hyphen and keeps it on the previous line.
                run += _char_em(char)
                widest = max(widest, run)
                run = 0.0
            else:
                run += _char_em(char)
        return max(widest, run)

    # Per-column measured demands: the whole cell (``weights``, for the
    # proportional share) and the longest unbreakable run (``runs``, which sets a
    # hard minimum because it cannot wrap).
    #
    # ``runs`` is floored at one em. A column whose widest unbreakable run is a
    # single ``✓`` or ``3`` used to declare a 1-glyph minimum, so a nine-column
    # table could satisfy the fit test while giving that column 6.7pt of usable
    # width -- less than one ideograph at 10.5pt -- and its CJK header then
    # stacked one character per line. One glyph is the smallest *usable* column,
    # not an optimisation, so it belongs in the fit test rather than in the
    # slack phase: the table must shrink until every column can hold a glyph.
    weights: list[float] = []
    runs: list[float] = []
    widest_unbreakable = 0.0
    for col in range(columns):
        longest = 0.0
        longest_run = 0.0
        for values in (headers, *rows):
            if col < len(values):
                for part in re.split(r"<br\s*/?>", values[col]):
                    longest = max(longest, _text_em(part.strip()))
                    longest_run = max(longest_run, _longest_unbreakable(part))
        widest_unbreakable = max(widest_unbreakable, longest_run)
        weights.append(max(longest, _TABLE_COL_MIN_WEIGHT))
        runs.append(max(longest_run, _TABLE_MIN_COLUMN_EM))

    width_pt = content_width_pt if content_width_pt is not None else _TABLE_DEFAULT_TEXT_WIDTH_PT
    total_weight = sum(weights) or 1.0
    # Only the gutters are taken out here: each column's own inset is folded
    # into its minimum below, so deducting it twice would under-allocate.
    share_available = max(
        width_pt - (columns - 1) * _TABLE_GUTTER_EM * TABLE_BASE_PT,
        1.0,
    )

    # Column widths are allocated the way CSS's ``minmax(min-content, 1fr)``
    # does: every column first gets the width its longest unbreakable run needs,
    # and only the leftover is shared by proportion.
    #
    # Neither of Typst's own options can express that, and using one of them was
    # the defect:
    #
    # * ``fr`` divides by ratio only. The ratio comes from whole-cell widths,
    #   including CJK paragraphs that wrap anyway, so three paragraph columns
    #   (68/99/68 em) took 94% of the width and left a token column 12.6pt for a
    #   55.6pt run. The run cannot wrap, so it ran off the page -- measured
    #   ``metadata.openclaw.requires`` at 39.8pt past the edge.
    # * ``auto`` sizes a column to its WHOLE cell, so promoting a paragraph
    #   column hands it the table: measured x=684 on a 595pt page and a
    #   neighbour squeezed to 8pt (one ideograph per line).
    #
    # Allocating fixed pt widths here removes the guesswork: the renderer knows
    # the container width and the measured advances, so it can solve the layout
    # instead of asking Typst to approximate it.
    def _alloc(scale: float) -> list[float]:
        """Widths in pt: each column's minimum, then proportional slack.

        A column's minimum is its unbreakable run. The *readability* floor --
        the width a header needs to stop stacking one ideograph per line -- is
        applied to the slack phase only, never to the fit test. Applying it to
        the fit test is what would force a 5-column table down to 0.43 (a 4.5pt
        body) just to widen a ``✓`` column, and no reader is served by an
        unreadable table whose header happens to be three lines.

        So: the run decides whether the table fits at all; the header floor
        decides how the leftover is spent.
        """
        # A column's minimum is its run PLUS the cell inset on both sides: the
        # run has to fit the *content* box, and the emitted column width is
        # content + 2*inset. Omitting the inset here left a 120.9pt run in a
        # 123.9pt column whose usable width was 113.9pt, and the string crossed
        # the text-block edge by 5pt on a page that had room.
        run_mins = [run * TABLE_BASE_PT * scale + 2 * _TABLE_CELL_INSET_PT for run in runs]
        total_run_min = sum(run_mins)
        if total_run_min >= share_available:
            # Even the runs do not fit. Squeeze proportionally so the table
            # stays inside the text block: a row that wraps more than intended
            # is readable, a cell past the page edge is not.
            return [m * share_available / total_run_min for m in run_mins]
        # Give each column at least enough to keep its header readable before
        # sharing the rest by proportion.
        floor = [w / _TABLE_HEADER_LINES * TABLE_BASE_PT * scale for w in weights]
        base = [max(m, f) for m, f in zip(run_mins, floor, strict=True)]
        base_total = sum(base)
        if base_total >= share_available:
            return [b * share_available / base_total for b in base]
        slack = share_available - base_total
        return [b + slack * w / total_weight for b, w in zip(base, weights, strict=True)]

    # Choose the largest scale at which every minimum still fits. 1.0 is the
    # common case; a table whose columns cannot all fit shrinks just enough
    # (measured on the skills table: minima total 466.5pt against 391.1pt of
    # room, so 0.84 was 0.8pt short and 0.80 fits).
    #
    # The insets are part of the budget, not an afterthought. ``_alloc`` builds
    # each column's minimum as ``run * BASE * scale + 2 * inset``, so a scale
    # derived from the runs alone is *always* a little too large, and ``_alloc``
    # then falls into its squeeze branch and returns widths below the minimums
    # it was asked to honour. Measured on 2609.00006v1's nine-column comparison
    # table: the scale said "fits at 0.84", the squeeze then cut two columns to
    # 16.7pt total, and with the 10pt cell inset that left 6.7pt of usable width
    # for a CJK header -- one ideograph per line, for the whole table.
    total_run_em = sum(runs)
    inset_budget = 2 * _TABLE_CELL_INSET_PT * columns
    required_scale = (
        1.0
        if total_run_em * TABLE_BASE_PT + inset_budget <= share_available
        else (share_available - inset_budget) / (total_run_em * TABLE_BASE_PT)
    )
    required_scale = max(required_scale, 0.0)
    effective_scale = required_scale * table_scale_factor
    min_scale = _TABLE_MIN_TEXT_SCALE * min(1.0, table_scale_factor)
    table_scale = (
        max(effective_scale, min_scale) if effective_scale < 1.0 else (1.0 * table_scale_factor)
    )
    widths = _alloc(table_scale)

    # (The scale-based ``too_wide`` / ``full_page`` / ``shrink`` trio that used to
    # be computed here was dead: the run-length measurement below recomputes all
    # three and nothing read the earlier values in between. Kept only where the
    # measurement decides, so there is one answer rather than two to keep in
    # sync.)

    # Reserve a hairline of slack: the widths are rounded to 0.1pt for the
    # emitted spec, and a run that needs 124.4pt against a 124.1pt column
    # overflowed by 0.3pt -- enough for ``metadata.openclaw.requires`` to cross
    # the text-block edge. Scaling to 99.8% costs nothing a reader can see.
    slack_ratio = _TABLE_WIDTH_SAFETY
    column_spec = "(" + ", ".join(f"{w * slack_ratio:.1f}pt" for w in widths) + ")"

    # Does this table fit the space it will be typeset into?
    #
    # Decided by measurement, not by a run-length heuristic. Two questions:
    #
    # 1. Does the widest *unbreakable* run fit a single column? A column cannot
    #    be narrower than its longest unbreakable content, so if that run exceeds
    #    the per-column share the table must be laid out differently -- this is
    #    the check that catches ``metadata.openclaw.requires`` (14.8 em = 155pt,
    #    which cannot fit a 105pt column and ran 39.8pt past the page edge).
    # 2. Is the column count itself beyond what the width supports?
    #
    # A two-column page cannot widen one column, so the lever is to let the table
    # span the full page (``#place(..., scope: "parent", float: true)`` measures
    # 535.1pt against the page's 545pt edge). On a single-column page there is no
    # room to gain, so the body shrinks instead.
    width = content_width_pt if content_width_pt is not None else _TABLE_DEFAULT_TEXT_WIDTH_PT
    gutter_pt = _TABLE_GUTTER_EM * TABLE_BASE_PT
    # Space one column can claim: the table width less every cell's inset and
    # every gutter between cells.
    per_column_pt = (width - columns * _TABLE_CELL_CHROME_PT - (columns - 1) * gutter_pt) / max(
        columns, 1
    )
    widest_unbreakable_pt = widest_unbreakable * TABLE_BASE_PT
    too_wide = (
        widest_unbreakable_pt > per_column_pt
        or columns >= _TABLE_MAX_COMFORTABLE_COLUMNS
        or sum(weights) * TABLE_BASE_PT > width
    )
    # Multi-page or tall tables (>18 rows) must not be wrapped in #place floats:
    # Typst floating blocks cannot break across page boundaries.
    full_page = page_columns > 1 and too_wide and not is_multipage and len(rows) <= 18
    landscape = allow_landscape and page_columns == 1 and columns >= _TABLE_MAX_COMFORTABLE_COLUMNS
    shrink = too_wide and not full_page and not landscape

    # When promoting to full-page float across dual columns, re-solve column widths
    # against the full text width (493.2pt) rather than keeping the narrow 219.5pt column spec.
    if full_page:
        full_width_pt = _PAGE_TEXT_WIDTH_PT.get("academic_1col", _TABLE_DEFAULT_TEXT_WIDTH_PT)
        share_available_fp = max(
            full_width_pt - (columns - 1) * _TABLE_GUTTER_EM * TABLE_BASE_PT,
            1.0,
        )

        def _alloc_fp(scale: float) -> list[float]:
            run_mins = [run * TABLE_BASE_PT * scale + 2 * _TABLE_CELL_INSET_PT for run in runs]
            total_run_min = sum(run_mins)
            if total_run_min >= share_available_fp:
                return [m * share_available_fp / total_run_min for m in run_mins]
            floor = [w / _TABLE_HEADER_LINES * TABLE_BASE_PT * scale for w in weights]
            base = [max(m, f) for m, f in zip(run_mins, floor, strict=True)]
            base_total = sum(base)
            if base_total >= share_available_fp:
                return [b * share_available_fp / base_total for b in base]
            slack = share_available_fp - base_total
            return [b + slack * w / total_weight for b, w in zip(base, weights, strict=True)]

        required_scale_fp = (
            1.0
            if total_run_em * TABLE_BASE_PT + inset_budget <= share_available_fp
            else (share_available_fp - inset_budget) / (total_run_em * TABLE_BASE_PT)
        )
        required_scale_fp = max(required_scale_fp, 0.0)
        table_scale_fp = (
            max(required_scale_fp, _TABLE_MIN_TEXT_SCALE) if required_scale_fp < 1.0 else 1.0
        )
        widths = _alloc_fp(table_scale_fp)
        column_spec = "(" + ", ".join(f"{w * slack_ratio:.1f}pt" for w in widths) + ")"
        table_scale = table_scale_fp
    elif landscape:
        # A dense table in single-column layout given landscape permission flips
        # to a dedicated landscape page, measuring against landscape text width.
        # When content_width_pt is specified, landscape width gains the A4 orientation difference
        # (297mm - 210mm = 87mm = 246.6pt); otherwise defaults to 717.0pt (standard A4 book margins).
        ls_width_pt = (content_width_pt + 246.6) if content_width_pt is not None else 717.0
        share_available_ls = max(
            ls_width_pt - (columns - 1) * _TABLE_GUTTER_EM * TABLE_BASE_PT,
            1.0,
        )

        def _alloc_ls(scale: float) -> list[float]:
            run_mins = [run * TABLE_BASE_PT * scale + 2 * _TABLE_CELL_INSET_PT for run in runs]
            total_run_min = sum(run_mins)
            if total_run_min >= share_available_ls:
                return [m * share_available_ls / total_run_min for m in run_mins]
            floor = [w / _TABLE_HEADER_LINES * TABLE_BASE_PT * scale for w in weights]
            base = [max(m, f) for m, f in zip(run_mins, floor, strict=True)]
            base_total = sum(base)
            if base_total >= share_available_ls:
                return [b * share_available_ls / base_total for b in base]
            slack = share_available_ls - base_total
            return [b + slack * w / total_weight for b, w in zip(base, weights, strict=True)]

        required_scale_ls = (
            1.0
            if total_run_em * TABLE_BASE_PT + inset_budget <= share_available_ls
            else (share_available_ls - inset_budget) / (total_run_em * TABLE_BASE_PT)
        )
        required_scale_ls = max(required_scale_ls, 0.0)
        table_scale_ls = (
            max(required_scale_ls, _TABLE_MIN_TEXT_SCALE) if required_scale_ls < 1.0 else 1.0
        )
        widths = _alloc_ls(table_scale_ls)
        column_spec = "(" + ", ".join(f"{w * slack_ratio:.1f}pt" for w in widths) + ")"
        table_scale = table_scale_ls

    indent = "  " if (full_page or landscape) else "    "
    # The body scale the columns were actually sized for. ``_alloc(table_scale)``
    # computes every column's minimum as ``run * TABLE_BASE_PT * table_scale``,
    # so the text must be emitted at that same scale. Emitting the constant
    # 0.84 instead was the bug that made a 9-column table render its header one
    # ideograph per line: the columns were sized for a 0.60 body, the glyphs
    # were set at 0.84, and every cell therefore needed ~40% more width than its
    # column had (measured on 2609.00006v1 p11, whose columns 3 and 6 have
    # 6.7pt of usable width -- less than one CJK glyph at 10.5pt).
    #
    # ``_TABLE_TEXT_SCALE`` remains the *comfortable* tabular size for a table
    # that fits at full size; a table that had to shrink uses the computed
    # factor, and never a value above it.
    emitted_scale = min(table_scale, float(_TABLE_TEXT_SCALE) * table_scale_factor)
    if scale_out is not None:
        scale_out.append(emitted_scale)

    start_anchor = ""
    header_continuation_cell = ""
    footer_continuation_block = ""
    end_anchor = ""

    if enable_continuation and (is_multipage or len(rows) >= 4):
        from ubt.core.tables.models import TableContinuationConfig
        from ubt.core.tables.styler import generate_typst_table_continuation_elements

        t_cfg = (
            continuation_config
            if isinstance(continuation_config, TableContinuationConfig)
            else TableContinuationConfig()
        )
        if len(rows) >= t_cfg.min_rows_for_continuation or is_multipage:
            (
                start_anchor,
                header_continuation_cell,
                footer_continuation_block,
                end_anchor,
            ) = generate_typst_table_continuation_elements(
                columns=columns,
                table_uid=table_uid or f"tbl-{abs(hash(markup)) % 100000}",
                lang=target_lang,
                config=t_cfg,
                indent=indent,
            )

    if full_page:
        lines = ['#place(top + center, scope: "parent", float: true, rect(width: 98%)[']
        if emitted_scale < 1.0:
            lines.append(f"  #set text(size: {emitted_scale:.2f}em)")
        if start_anchor:
            lines.append(f"  {start_anchor}")
        lines.append("  #table(")
    elif landscape:
        lines = ["#page(flipped: true)["]
        if emitted_scale < 1.0:
            lines.append(f"  #set text(size: {emitted_scale:.2f}em)")
        if start_anchor:
            lines.append(f"  {start_anchor}")
        lines.append("  #table(")
    else:
        # ``#[`` opens a *content block*; a bare ``[`` is ordinary text in markup
        # mode. The bare form did not scope anything: ``#set text(size: 0.84em)``
        # inside it leaked into the rest of the document, and the ``[``/``]``
        # themselves were typeset as literal characters. The leak compounded --
        # each shrunken table multiplied every following paragraph's size by
        # 0.84 -- so a paper with 17 such tables ended at 10.5 * 0.84**17 =
        # 0.53pt, which is exactly what the delivered 22-page file measured
        # (0.49pt on the last page). Nothing was lost by the page count: the
        # whole document had been shrunk to fit.
        lines = ["#["]
        if shrink:
            lines.append(f"  #set text(size: {emitted_scale:.2f}em)")
        if start_anchor:
            lines.append(f"  {start_anchor}")
        lines.append("  #table(")
    lines += [
        f"{indent}columns: {column_spec},",
        f"{indent}inset: 5pt,",
        f"{indent}column-gutter: 0.7em,",
        f"{indent}align: (col, row) => if row == 0 {{ center + horizon }} else {{ left + horizon }},",
        f"{indent}stroke: none,",
        f"{indent}table.header(",
        f"{indent}  repeat: true,",
    ]
    if header_continuation_cell:
        lines.append(header_continuation_cell)
    lines += [
        f"{indent}  table.hline(stroke: 1.1pt),",
        f"{indent}  {cells(headers, header=True)},",
        f"{indent}  table.hline(stroke: 0.5pt),",
        f"{indent}),",
    ]
    for row in rows:
        lines.append(f"{indent}{cells(row, header=False)},")
    lines.append(f"{indent}table.hline(stroke: 1.1pt),")
    if footer_continuation_block:
        lines.append(footer_continuation_block)
    lines.append(f"{indent})")
    if end_anchor:
        lines.append(f"  {end_anchor}")
    lines.append("])" if full_page else "]")
    return "\n".join(lines)


def _looks_like_a_label(text: str) -> bool:
    """Whether a label-led line is a caption rather than prose about a figure.

    A caption is a short line under an image: ``FIG. 3.1``, ``图 1：示意图。``,
    ``图 1 | RT-DocLayout 的性能``. A body paragraph can open the same way —
    "图 3.1 展示了双栅 FinFET 的二维横截面，该结构被用作模型推导的参考。"
    — and must stay body prose.

    The delimiter decides first, because it is the translator's own caption
    marker: ``图 1 | …`` and ``Figure 1: …`` are emitted only for a caption, and
    one can run to several clauses (the reference corpus has a 301-character
    one). Only the plain ``.``/``：`` forms are ambiguous, and for those length
    and sentence count are measured *after* the label: a caption's description
    is a clause, while a paragraph that names a figure keeps going.
    """
    stripped = text.strip()
    if len(stripped) > _CAPTION_LABEL_MAX_CHARS * 3:
        return False
    if "|" in stripped[: _label_delimiter_end(stripped)]:
        # The explicit caption marker the translator emits. A caption in that
        # form is a caption however many clauses follow.
        return True
    if len(stripped) > _CAPTION_LABEL_MAX_CHARS:
        return False
    body = _CAPTION_RE.sub("", stripped, count=1).strip()
    if not body:
        # A bare label ("FIG. 3.1") is a caption on its own.
        return True
    from ubt.core.figures.detector import _SUB_MARKER_RE

    if _SUB_MARKER_RE.search(body):
        # A composite caption naming subfigures ("(a) ... (b) ...") is a caption.
        return True
    if len(body) > _CAPTION_LABEL_BODY_MAX_CHARS:
        return False
    # One sentence at most: a caption does not narrate.
    return sum(body.count(ch) for ch in ".。！？!?") <= 1


def _label_delimiter_end(text: str) -> int:
    """Index just past the label's delimiter, or 0 when there is none early."""
    match = _CAPTION_RE.match(text)
    return match.end() if match else 0


def _binds_as_caption(block: IRBlock, text: str, figure: _Figure | None = None) -> bool:
    """Whether this prose block is the caption of the figure just placed.

    Signals, strongest first:

    * ``FlowID.CAPTION`` — the reader saw a caption. Binds however long; the
      descriptive line under ``FIG. 3.11`` is 162 characters and is still a
      caption.
    * An explicit ``Figure N``/``图 N`` label whose text still *is* a label
      (short, one clause) — a reader that classifies by text, and a caption the
      extractor labelled but filed in the main story.
    * ``FlowID.MAIN_STORY`` / ``footnote`` — body prose. Never binds by length:
      on the reference chapter 15 paragraphs were re-laid as grey captions under
      equations because translation compression put them under a character cap.
    * A short unlabelled line after a *picture* — the caption whose label the
      extractor dropped.
    * Never after a formula / text-recovery crop: the paragraph following an
      equation is body prose or the lead-in to the next one.
    """
    if str(getattr(block.block_type, "value", block.block_type) or "") == "heading":
        return False
    flow = str(getattr(block, "flow_id", None) or "")
    if flow == "caption":
        return True
    if _CAPTION_RE.match(text) and _looks_like_a_label(text):
        return True
    if flow in ("main_story", "footnote"):
        return False
    if figure is not None and not figure.is_picture:
        return False
    return len(text) <= _PENDING_CAPTION_MAX_CHARS


@dataclass(slots=True)
class _Figure:
    """A pending image plus the caption that follows it."""

    asset: str
    caption: str = ""
    #: The source bbox's width in PDF points, when known. Used to keep a small
    #: graphic (a logo, a signature) at its own size instead of stretching it to
    #: the page width — measured on the PaddleOCR paper, whose 121pt-wide
    #: PaddleOCR logo was blown up to 88% of an A4 page.
    width_pt: float | None = None
    #: The IR block this figure came from. Carried so a missing asset is
    #: attributed to its *block* in the skip channel (the export stage indexes
    #: that channel by block id); without it the miss was recorded against an
    #: asset path and no consumer could match it.
    block_id: str = ""
    #: Whether the source block was a *picture* (a figure with a caption under
    #: it) rather than a formula / text-recovery crop (an equation, whose
    #: following paragraph is body prose or the lead-in to the next equation).
    #: The extractor already distinguishes them — a ``PICTURE`` item carries
    #: ``FlowID.CAPTION``, a recovered formula carries ``FlowID.MAIN_STORY`` —
    #: so the renderer does not have to guess from text length, which is what
    #: re-laid body paragraphs as captions.
    is_picture: bool = False


@dataclass(slots=True)
class ReflowStats:
    """What the render produced, for the quality report and the tests."""

    blocks_seen: int = 0
    blocks_rendered: int = 0
    tables: int = 0
    #: Tables whose Markdown grid did not parse and were delivered as their
    #: source grid instead (the target did not parse but the source did). Counted
    #: so the quality report can see that the cell text is the source language,
    #: not a silent success.
    tables_fell_back: int = 0
    #: Tables that parsed in neither the target nor the source grid and were
    #: dropped from the page entirely. This is the fail-closed case and must
    #: never be silent — it is surfaced through the render-skip channel.
    tables_dropped: int = 0
    figures: int = 0
    formulas: int = 0
    #: Formulas that compiled as real math are in ``formulas``; these two count
    #: the ones that did not, so a document whose equations silently became
    #: Unicode approximations (or vanished) is visible in the quality report
    #: instead of looking like it rendered cleanly.
    formulas_degraded: int = 0
    formulas_dropped: int = 0
    #: Formulas typeset from the *translated* text (a localized annotation), not
    #: the source. Counted so a model that localizes formula annotations is
    #: visible — before this, the formula branch read only ``source_text`` and
    #: the translated annotation was dropped silently.
    formulas_from_target: int = 0
    #: Pipeline sentinels (``⟦SOUP_MASK…⟧``, ``⟦B⟧``, ``\llbracket…``) that
    #: survived to the renderer. A block carrying one had a restore that could
    #: not account for a span; the renderer escapes it as prose so the document
    #: still compiles, but the leak is a defect upstream and must be visible in
    #: the quality report rather than absorbed silently.
    sentinels_leaked: int = 0
    assets_copied: int = 0
    #: ``(block_id, asset)`` for figures the renderer could not resolve. The
    #: block id is what makes the miss reachable downstream: the export stage's
    #: skip channel is keyed by block id, so an asset *path* there matches no
    #: block and the delivery contract reported ``missing_assets=0`` for a page
    #: whose figure had been replaced by a placeholder.
    assets_missing: list[tuple[str, str]] = field(default_factory=list)


#: The mark-wrapper predicates and the delivered-text rule now live in
#: :mod:`ubt.render.delivered_text` so the companion HTML/EPUB views and this
#: module cannot disagree about them: the HTML view took ``target_text``
#: verbatim and shipped ``<mark class="ubt-failed-draft" title="…">`` (with the
#: provider's exception text) to the reader as escaped markup, while this
#: renderer stripped it. Re-exported here under the names the rest of this module
#: already calls.


def _keeps_source(block: IRBlock) -> bool:
    """Whether the pipeline decided this block stays in the source language.

    Mirrors the draft stage's rule (``skip_translate``, with the policy verdict
    authoritative when it exists) so the renderer and the translator cannot
    disagree about what was supposed to be translated. A bibliography entry or
    an author affiliation is kept verbatim on purpose; re-rendering it through
    the target-text path would silently undo that decision.

    Deliberately NOT ``ubt.core.content.adapt.kept_in_source``: that predicate
    answers a different question ("does the delivery have no translation for
    this block?", so ``BLOCKED_HUMAN`` and fail-closed render-skips count), and
    the renderer must treat a quarantined block differently — it renders the
    rejected *draft* (fail-open) rather than the source, which is the whole
    point of ``_block_text``'s draft fallback. Merging the two would either
    quarantine every rejected-but-translated block or ship the ``<mark>``
    marker; both are wrong. Keep them apart, cross-referenced.
    """
    if block.policy_translate is False:
        return True
    return bool(block.skip_translate)


def _block_text(block: IRBlock) -> str:
    """The text to render for a block: its translation, its draft, else its source.

    Delegates to :func:`ubt.render.delivered_text.delivered_text` — the single
    owner of the rule (wrapper stripping, quarantine fallback, draft fallback),
    shared with the companion HTML and EPUB views. Both used to implement it
    separately, and the HTML view implemented only the first half.

    Named ``_block_text`` for the module's internal callers and for the
    cross-references in ``content.adapt`` / ``docling_crosscheck``; new code
    outside this module should call ``delivered_text`` directly.
    """
    return delivered_text(block)


#: Ordered list markers: ``1.``, ``1)``, ``(1)``, ``１．`` and the compound
#: forms (``2.1.``). Anything else the extractors emit is a bullet.
_ORDERED_MARKER_RE = re.compile(r"^[（(]?[0-9０-９]{1,3}(?:[.．][0-9０-９]{1,3})*[.．、)）]$")


def _list_is_ordered(block: IRBlock) -> bool:
    """Whether a ``LIST_ITEM`` block came from an ordered (numbered) list.

    The marker the extractors store is the source of truth: ``•``/``-``/``*``
    for bullets, ``1.``/``(1)``/``2.1.`` for an enumeration. A block with no
    marker falls back to a bullet, which is what the extractors default to.
    """
    marker = str(getattr(block.element, "marker", "") or "").strip()
    return bool(marker) and _ORDERED_MARKER_RE.match(marker) is not None


def _typst_list_markup(items: Sequence[str], *, ordered: bool) -> str:
    """A Typst list call for one run of consecutive list items.

    ``#list(..)`` / ``#enum(..)`` rather than the ``- `` / ``+ `` markup form:
    the item bodies are already-escaped markup (``render_prose`` output), and
    the markup form would require each body to be a single line with its
    continuation indented, which a translated paragraph with a soft wrap is not.
    A call takes each item as a content block, so line structure inside an item
    is irrelevant.

    Every body is wrapped in ``[...]``. A bare ``#list(...)`` argument is a code
    expression, so an unescaped ``#`` or ``[`` in translated prose would be
    parsed as code; the bracket makes it content again. ``render_prose`` already
    escapes both, so the wrapper is defence for a caller that does not.
    """
    function = "enum" if ordered else "list"
    rendered = ", ".join(f"[{item}]" for item in items)
    return f"#{function}({rendered})"


def _heading_level(block: IRBlock) -> int:
    """Heading depth, clamped to Typst's usable range (1..6)."""
    level = getattr(block.element, "level", 1)
    if not isinstance(level, int) or level < 1:
        return 1
    return min(level, 6)


def _raw_block_markup(text: str, container_width_pt: float = 470.0) -> str:
    """Typeset a code listing as a monospace block that keeps its line breaks and fits container."""
    from ubt.render.typst.container_repair import repair_code_block_markup

    return repair_code_block_markup(text, container_width_pt=container_width_pt)


class TypstDocumentRenderer:
    """Build a Typst document from IRBlocks and compile it to PDF.

    The class owns the emitted source so a caller can inspect or test it without
    running the compiler (``build_source``), which is how the renderer is tested
    without depending on the Typst binary.
    """

    #: Base page setup, shared by every target language.
    _PAGE_SETUP = '#set page(paper: "a4", margin: (x: 2.2cm, y: 2.4cm), numbering: "1")'

    def _table_width_pt(self) -> float:
        """Width the tables will be typeset into, from the layout actually used.

        The verdict has to be taken against the real container: a table that
        fits a two-column page's 219.5pt column and one that needs a 493pt
        page-wide block are different questions, and answering both with one
        constant is how the old heuristic mis-fired.
        """
        template = getattr(self, "layout_template", "book")
        if template == "academic":
            if getattr(self, "columns", 1) > 1:
                return _PAGE_TEXT_WIDTH_PT["academic_2col"]
            return _PAGE_TEXT_WIDTH_PT["academic_1col"]
        return _PAGE_TEXT_WIDTH_PT["book"]

    def preamble(self, target_lang: str = "zh") -> str:
        """The Typst preamble for ``target_lang``.

        The language and text direction are derived from the target, not
        hardcoded: an Arabic or Hebrew book rendered with ``lang: "zh"`` gets
        Chinese line-breaking and left-to-right layout, which is wrong on every
        page. ``typst`` runs UAX #9 itself, so an RTL target only needs the
        direction declared — the text stays in logical order.

        Kept as one function so the template is auditable in one place instead
        of being assembled from f-strings across the builder. Was a
        ``@staticmethod``; the font stack is per-renderer state now, so it
        reads ``self``.
        """
        if getattr(self, "layout_template", "book") == "academic":
            from ubt.render.typst.academic import academic_preamble

            ret = academic_preamble(
                target_lang=target_lang,
                font_stack=self.font_stack,
                paper_title=getattr(self, "paper_title", ""),
                columns=getattr(self, "columns", 1),
            )
            if getattr(self, "enable_decor", False):
                from ubt.core.decor.dropcap import TYPST_DROPCAP_MACRO

                ret = ret + "\n" + TYPST_DROPCAP_MACRO
            return ret

        from ubt.layout.theme import direction_for

        direction = direction_for(target_lang or "zh").value
        lang = (target_lang or "zh").replace("_", "-")
        rtl = "\n#set text(dir: rtl)" if direction == "rtl" else ""
        # Font stack: language-resolved via language_profile (a ja/ko/ar target
        # must not typeset with SC glyphs; zh/en resolve to the historical
        # Latin-first stack, so default rendering is unchanged), with the
        # renderer's explicit stack — the user's config.font_family override —
        # leading when supplied. Headings reverse the order (CJK face first):
        # a Latin heading is unchanged either way, a CJK one gets the real
        # face instead of falling through to the Latin serif.
        from ubt.adapters.pdf.font_probe import resolve_for_render

        if self.font_stack:
            stack: tuple[str, ...] = self.font_stack
        else:
            # No stack supplied: resolve the profile against this machine, so an
            # absent spare face is pruned instead of warning on every compile.
            stack = resolve_for_render(lang).families
        # `covers` routing: in a CJK document the Latin face is scoped to
        # `latin-in-cjk` so shared punctuation (curly quotes, dashes, ellipsis)
        # belongs to the CJK face — the typographic convention, and something a
        # bare priority list gets wrong for every page of every Chinese book.
        from ubt.render.typst.font_routing import typst_font_list

        body = typst_font_list(stack, lang)
        # Headings reverse the order (CJK face first). Routing is applied to the
        # reversed list so a Latin heading face is still scoped inside CJK text.
        cjk_first = typst_font_list(tuple(reversed(stack)), lang)
        # Body metrics come from the per-language convention, not from a
        # literal here: the previous single rule indented *every* target by two
        # characters, which is the Chinese rule only. Japanese and Korean
        # indent one character (一字下げ / 원고지 한 칸), and a Latin target
        # takes a one-em indent. `0` is a real value (no indent), so the
        # `all: true` form is only emitted when there is an indent to apply —
        # `first-line-indent: (amount: 0em, all: true)` would be a no-op with
        # extra ceremony, and dropping the setting entirely is what a block
        # paragraph wants.
        from ubt.layout.theme import convention_for

        convention = convention_for(lang)
        if convention.first_line_indent_em > 0:
            indent_rule = (
                f" first-line-indent: (amount: {convention.first_line_indent_em:g}em, all: true))"
            )
        else:
            indent_rule = ")"
        par_spacing = getattr(self, "par_spacing_em", 0.9)
        base_preamble = "\n".join(
            [
                TypstDocumentRenderer._PAGE_SETUP,
                f"#set text(font: ({body}), size: 10.5pt{typst_lang_arguments(lang)})" + rtl,
                # Leading comes from the same convention owner, not a literal
                # here: this line and ``Theme.leading_em`` disagreed (0.78 vs
                # 0.85), so a maintainer tuning the theme tuned nothing.
                f"#set par(justify: true, leading: {convention.leading_em:g}em, spacing: {par_spacing:g}em,"
                + indent_rule,
                "#show heading.where(level: 1): it => [",
                "  #v(0.6em)",
                f'  #block(text(size: 15pt, weight: "bold", font: ({cjk_first}))[#it.body])',
                "  #v(0.3em)",
                "]",
                "#show heading.where(level: 2): it => [",
                "  #v(0.4em)",
                f'  #block(text(size: 12pt, weight: "bold", font: ({cjk_first}))[#it.body])',
                "  #v(0.2em)",
                "]",
                "#show figure.caption: it => block(text(size: 9pt, fill: luma(60))[#it.body])",
            ]
        )
        if getattr(self, "enable_decor", False):
            from ubt.core.decor.dropcap import TYPST_DROPCAP_MACRO

            return base_preamble + "\n" + TYPST_DROPCAP_MACRO
        return base_preamble

    def __init__(
        self,
        *,
        assets_root: Path | None = None,
        figure_width: str = "88%",
        bilingual: bool = False,
        math_gate: Callable[[str], bool] | None = None,
        math_gate_many: Callable[[Iterable[str]], None] | None = None,
        font_stack: Sequence[str] | None = None,
        layout_template: str = "book",
        columns: int = 1,
        paper_title: str = "",
        abstract: str = "",
        toc_mode: str = "auto",
        allow_landscape_tables: bool = True,
        table_scale_factor: float = 1.0,
        formula_scale: float = 1.0,
        break_long_formulas: bool = False,
        par_spacing_em: float = 0.9,
        enable_kinsoku: bool = False,
        enable_colophon: bool = False,
        colophon: object | None = None,
        enable_index: bool = False,
        index: object | None = None,
        enable_decor: bool = False,
        decor_spec: object | None = None,
        enable_notes: bool = False,
        notes_config: object | None = None,
        enable_backmatter: bool = False,
        backmatter_config: object | None = None,
        enable_grid_figures: bool = False,
        figure_config: object | None = None,
        enable_table_continuation: bool = False,
        table_continuation_config: object | None = None,
        enable_epigraph: bool = False,
        epigraph_config: object | None = None,
        enable_xref: bool = False,
        xref_config: object | None = None,
    ) -> None:
        self.assets_root = Path(assets_root) if assets_root is not None else None
        self.figure_width = figure_width
        self.bilingual = bilingual
        self.layout_template = layout_template
        self.columns = columns
        self.paper_title = paper_title
        self.abstract = abstract
        self.allow_landscape_tables = allow_landscape_tables
        self.table_scale_factor = float(table_scale_factor)
        self.formula_scale = float(formula_scale)
        self.break_long_formulas = bool(break_long_formulas)
        self.par_spacing_em = float(par_spacing_em)
        self.enable_kinsoku = bool(enable_kinsoku)
        self.enable_colophon = bool(enable_colophon)
        self.colophon = colophon
        self.enable_index = bool(enable_index)
        self.index = index
        self.enable_decor = bool(enable_decor)
        self.decor_spec = decor_spec
        self.enable_notes = bool(enable_notes)
        self.notes_config = notes_config
        self.enable_backmatter = bool(enable_backmatter)
        self.backmatter_config = backmatter_config
        self.enable_grid_figures = bool(enable_grid_figures)
        self.figure_config = figure_config
        self.enable_table_continuation = bool(enable_table_continuation)
        self.table_continuation_config = table_continuation_config
        self.enable_epigraph = bool(enable_epigraph)
        self.epigraph_config = epigraph_config
        self.enable_xref = bool(enable_xref)
        self.xref_config = xref_config
        #: ``auto`` | ``on`` | ``off`` for placing a recovered table of contents.
        #: ``auto`` is none for a paper (by convention) and one for a book that
        #: carries TOC rows; see :func:`resolve_toc_mode`.
        self.toc_mode = toc_mode
        #: Body/heading font stack for the preamble. ``None`` keeps the
        #: historical hard-coded Latin+CJK-SC stack; the strategy passes the
        #: language-resolved stack (``resolve_font_config(target_lang)``) so a
        #: Japanese target stops typesetting with Simplified-Chinese glyphs.
        self.font_stack: tuple[str, ...] | None = tuple(font_stack) if font_stack else None
        #: Compile gate for a converted math body (see ``render_prose``). The
        #: publish caller passes a ``TypstMathProbe``; a test that only inspects
        #: the emitted source leaves it ``None`` and pays no compiler.
        self._math_gate = math_gate
        #: Bulk pre-warm for the same cache ``_math_gate`` reads
        #: (``TypstMathProbe.check_many``): one Typst invocation for the whole
        #: document's not-yet-known bodies instead of one per body — measured
        #: 10 distinct bodies: 6.8s per-body vs 0.7s batched. ``None`` (tests,
        #: or a gate without a batch API) keeps the per-body path.
        self._math_gate_many = math_gate_many
        self.stats = ReflowStats()
        #: ``(block_id, reason)`` for blocks this render could not place, drained
        #: by the PDF adapter into its ``last_render_skips`` channel so the
        #: export stage can flag them. A dropped table is the case that matters:
        #: without it the block vanished with no trace (neither the quality
        #: report nor the content contract saw a gap).
        self.last_render_skips: list[tuple[str, str]] = []
        #: ``(block_id, flag)`` for blocks that were *drawn* but not cleanly: a
        #: formula the transpiler could not turn into real Typst math and was
        #: delivered as readable Unicode instead. Drained by the PDF adapter into
        #: its ``last_render_flags`` channel so the export stage can record the
        #: degradation, which is a plain quality flag — never a
        #: ``render_skip:``, because the content did ship.
        self.last_render_flags: list[tuple[str, str]] = []
        #: Per-table reconstruction outcomes, drained by the PDF adapter into
        #: ``last_render_reconstructions`` so the delivery contract can record a
        #: rebuilt table as ``RECONSTRUCTED`` (verified or corrupt) instead of
        #: the existence-only ``PRESERVED_OPAQUE`` it used to assert for every
        #: non-text asset.
        self.table_reconstructions: list[ReconstructionRecord] = []

    def _prose(
        self,
        text: str,
        runs: Sequence[InlineRun] = (),
        skip_anchors: set[str] | None = None,
    ) -> str:
        """``render_prose`` bound to this renderer's compile gate."""
        if not runs:
            from ubt.render.typst.container_repair import inject_soft_breaks

            text = inject_soft_breaks(text)
        if getattr(self, "enable_kinsoku", False):
            from ubt.core.cleaners.kinsoku import (
                apply_kinsoku_word_joiners,
                compress_cjk_punctuations,
            )

            text = compress_cjk_punctuations(text)
            text = apply_kinsoku_word_joiners(text)
        rendered = render_prose(text, runs, is_math_ok=self._math_gate)
        topology = getattr(self, "citation_topology", None)
        if topology is not None:
            from ubt.core.references.linker import inject_hyperlinks

            xref_cfg = getattr(self, "xref_config", None)
            rendered = inject_hyperlinks(
                rendered,
                topology,
                target_format="typst",
                config=xref_cfg,
                skip_anchors=skip_anchors,
            )
        return rendered

    def _prewarm_math_gate(self, blocks: Sequence[IRBlock]) -> None:
        """Resolve every not-yet-known math body in one batched probe.

        ``build_source`` renders block by block, and each formula's gate call
        would otherwise pay a full Typst invocation (~0.7s): a 300-formula book
        spent ~3.5 minutes of process startups on a job whose whole compile is
        seconds. ``check_many`` lays the pending bodies out as one document and
        bisects on failure, so the same verdicts land in the same cache with a
        handful of invocations. Bodies are collected from exactly the two
        places that gate: a text block's ``$…$`` spans and a formula block's
        body.
        """
        if self._math_gate_many is None or self._math_gate is None:
            return
        bodies: list[str] = []
        for block in blocks:
            block_type = str(getattr(block.block_type, "value", block.block_type) or "")
            if block_type == "formula":
                text = _block_text(block)
                if text:
                    converted = typstify_math(strip_math_delimiters(text))
                    if converted:
                        bodies.append(converted)
                continue
            text = _block_text(block) if block_type not in ("image", "table") else ""
            if not text:
                continue
            for span in _MATH_SPAN_RE.finditer(text):
                part = span.group(0)
                if not _is_math_span(part):
                    continue
                inner = part[2:-2] if part.startswith("$$") else part[1:-1]
                if has_sentinel(inner):
                    continue
                converted = typstify_math(inner)
                if converted:
                    bodies.append(converted)
        if bodies:
            self._math_gate_many(bodies)

    def _maybe_pair(
        self, block: IRBlock, rendered: str, *, source_markup: str | None = None
    ) -> str:
        """Add the source line above ``rendered`` when this block has a real pair.

        Three cases must *not* be paired, or the page shows the same sentence
        twice:

        * the run is monolingual (``bilingual`` off);
        * the block was never translated (no target), so its "translation" is
          its source;
        * the policy deliberately kept the source (a bibliography entry), where
          source and target are the same string by construction.

        ``source_markup`` overrides how the source is typeset. Prose goes through
        :func:`render_prose`; a *table* must not, because its source is Markdown
        grid syntax and rendering it as prose puts a literal ``| Method | BLEU |``
        row above the real table.
        """
        if not self.bilingual:
            return rendered
        source = (block.source_text or "").strip()
        target = (block.target_text or "").strip()
        if not source or not target or source == target:
            return rendered
        if _is_placeholder(source) or _is_placeholder(target):
            return rendered
        return self._pair(source, rendered, source_markup=source_markup)

    def _pair(self, source: str, target: str, *, source_markup: str | None = None) -> str:
        """Source above its translation, source visually subordinate.

        The pairing is per block, not per page: a rebuilt document is not
        page-aligned with its source, so pairing pages is impossible. That is
        also the more useful form: the reader never turns a page to find the
        sentence being compared.

        The source keeps its own math handling (a formula must stay a formula,
        not become literal ``\\frac`` text), so by default it goes through
        :func:`render_prose` exactly as the target does. ``source_markup``
        replaces that rendering for a block whose source is not prose — a table,
        whose source is Markdown grid syntax that must be compiled as a table
        rather than pasted as text.
        """
        rendered_source = source_markup if source_markup is not None else self._prose(source)
        return (
            "#block(above: 0.5em, below: 0.25em)[\n"
            '  #text(size: 9pt, fill: luma(115), style: "italic")['
            f"{rendered_source}]\n"
            "]\n"
            f"{target}"
        )

    # -- source building ---------------------------------------------------

    def build_source(
        self,
        blocks: Sequence[IRBlock],
        build_dir: Path,
        target_lang: str = "zh",
    ) -> tuple[str, ReflowStats]:
        """Emit the Typst source for ``blocks``, copying assets into ``build_dir``.

        Assets are copied rather than referenced by absolute path so the build
        directory is self-contained and the compile cannot depend on the
        original extraction cache surviving.
        """
        self.stats = ReflowStats()
        build_dir.mkdir(parents=True, exist_ok=True)

        if getattr(self, "enable_notes", False):
            from ubt.core.notes.extractor import extract_notes_topology
            from ubt.core.notes.models import NotesConfig

            self._notes_cfg = (
                self.notes_config if isinstance(self.notes_config, NotesConfig) else NotesConfig()
            )
            self._notes_topology = extract_notes_topology(blocks, self._notes_cfg)
            self._notes_def_ids = self._notes_topology.def_block_ids
            self._notes_by_source = {}
            for n in self._notes_topology.notes:
                if n.source_block_id:
                    self._notes_by_source.setdefault(n.source_block_id, []).append(n)
            self._notes_by_chapter = {}
            for n in self._notes_topology.notes:
                self._notes_by_chapter.setdefault(n.chapter_index, []).append(n)
            self._notes_emitted_chapters = set()
            self._notes_current_chapter = 1

        if getattr(self, "enable_backmatter", False):
            from ubt.core.backmatter.detector import detect_backmatter_manifest
            from ubt.core.backmatter.models import BackmatterConfig

            self._bm_cfg = (
                self.backmatter_config
                if isinstance(self.backmatter_config, BackmatterConfig)
                else BackmatterConfig()
            )
            self._bm_manifest = detect_backmatter_manifest(blocks, self._bm_cfg)
            self._bm_appendix_switched = False

        if getattr(self, "enable_epigraph", False):
            from ubt.core.epigraph.detector import detect_chapter_epigraphs_and_abstracts
            from ubt.core.epigraph.models import EpigraphConfig

            self._epigraphs_by_id, self._abstracts_by_id = detect_chapter_epigraphs_and_abstracts(
                list(blocks)
            )
            self._epigraph_cfg = (
                self.epigraph_config
                if isinstance(self.epigraph_config, EpigraphConfig)
                else EpigraphConfig()
            )

        self._prewarm_math_gate(blocks)
        from ubt.core.references.linker import build_citation_topology

        self.citation_topology = build_citation_topology(blocks)
        body: list[str] = []

        if getattr(self, "enable_colophon", False) or getattr(self, "colophon", None) is not None:
            from ubt.core.colophon.generator import (
                extract_colophon_from_blocks,
                render_colophon_typst,
            )

            meta = self.colophon or extract_colophon_from_blocks(
                blocks, default_title=self.paper_title
            )
            body.append(render_colophon_typst(meta))

        # Table of contents: the reader records ``toc_entry`` per row so the TOC
        # is not scattered into headings-without-body plus bare page numbers.
        # Group each run of consecutive TOC rows into one full-width block (with
        # the recovered page numbers) and mark them consumed, so they never take
        # the ``#heading`` path below — the defect this fixes emitted every TOC
        # row as a section heading on a near-empty page, dropping the numbers.
        #
        # Placement is *fixed*, not source order: a TOC belongs after the title
        # page and before the body. Emitting it where the reader happened to see
        # it put it on the opening page, above the title banner, when the source
        # spelled the section list before the title block.
        toc_consumed: set[str] = set()
        toc_by_insertion_point: dict[int, list[str]] = {}
        has_toc_heading = False
        # Only entries that *behave* like a TOC row count toward the "this paper
        # really has a contents page" decision. A run of five bare lines that the
        # reader mislabelled ``toc_entry`` (a reference list, a term glossary) is
        # not a table of contents, and counting it invented a 20-page TOC on a
        # short paper — exactly what the old "papers get no TOC" rule prevented.
        # A row qualifies when it carries a page number, or when it names a
        # heading that actually appears in the document.
        heading_keys = {
            _toc_title_key(_block_text(b))
            for b in blocks
            if str(getattr(b.block_type, "value", b.block_type) or "") == "heading"
            # A row mislabelled ``toc_entry`` is still a HEADING block, so it
            # must not vouch for itself: that self-match made every tagged run
            # "name a heading that exists" and defeated this check entirely.
            and not b.provenance.toc_entry
        }
        total_toc_entries = 0
        for run_start, entry_blocks in _group_toc_runs(blocks):
            title_page: list[tuple[str, str]] = []
            for blk in entry_blocks:
                title = _block_text(blk).strip()
                page = str(blk.provenance.toc_page or "").strip()
                if title:
                    title_page.append((title, page))
                    if page or _toc_title_key(title) in heading_keys:
                        total_toc_entries += 1
                toc_consumed.add(blk.id)
                self.stats.blocks_rendered += 1
            # The TOC's own heading ("Contents") is page furniture, not a
            # section: its entries were consumed above, so emitting it leaves an
            # orphan heading with no body. Consume the heading that introduces
            # the run — the nearest preceding block, allowing nothing between it
            # and the rows (a paragraph there means the heading titles prose).
            for earlier in reversed(blocks[:run_start]):
                if earlier.id in toc_consumed:
                    continue
                if str(getattr(earlier.block_type, "value", earlier.block_type) or "") != "heading":
                    break
                if _is_toc_heading_text(_block_text(earlier)):
                    toc_consumed.add(earlier.id)
                    has_toc_heading = True
                break
            if title_page:
                toc_by_insertion_point.setdefault(run_start, []).append(render_toc(title_page))

        should_render_toc = resolve_toc_mode(
            getattr(self, "toc_mode", "auto"),
            layout_template=getattr(self, "layout_template", "book"),
            has_toc=bool(toc_by_insertion_point),
            has_explicit_heading=has_toc_heading,
            entry_count=total_toc_entries,
        )

        if getattr(self, "layout_template", "book") == "academic" and getattr(
            self, "paper_title", ""
        ):
            from ubt.render.typst.academic import academic_title_banner

            banner = academic_title_banner(
                title=self.paper_title,
                abstract=getattr(self, "abstract", ""),
                columns=getattr(self, "columns", 1),
                lang=target_lang,
            )
            if banner:
                # The banner *precedes* the TOC: a paper opens on its title.
                body.insert(0, banner)
        pending: _Figure | None = None
        pending_grid: list[_Figure] = []

        def flush_figure() -> None:
            nonlocal pending, pending_grid
            if pending is not None:
                body.append(self._figure_markup(pending, build_dir))
                pending = None
            if pending_grid:
                body.append(self._figure_grid_markup(pending_grid, build_dir))
                pending_grid = []

        #: Consecutive list items are emitted as ONE list, so the reader gets a
        #: real list rather than a run of paragraphs that happen to start with a
        #: bullet. Grouped up front by id, because the emit loop below has
        #: already ``continue``d on TOC rows and the grouping must agree with
        #: what actually gets rendered.
        list_runs = self._group_list_runs(blocks, toc_consumed)
        chapter_needs_dropcap = False
        last_page_emitted: int | None = None

        for i, block in enumerate(blocks):
            block_span = getattr(getattr(block, "element", None), "span", None)
            block_page = (
                block.bbox.page
                if block.bbox is not None and block.bbox.page is not None
                else getattr(block_span, "page", None)
            )
            if block_page is not None and block_page != last_page_emitted:
                last_page_emitted = block_page
                if getattr(self, "enable_xref", False):
                    body.append(f'[#metadata("page-start") <page-{last_page_emitted}>]')

            if should_render_toc and i in toc_by_insertion_point:
                flush_figure()
                for chunk in toc_by_insertion_point[i]:
                    if body:
                        body.append("#pagebreak()")
                    body.append(chunk)
                    body.append("#pagebreak()")
                del toc_by_insertion_point[i]

            self.stats.blocks_seen += 1
            if block.id in toc_consumed:
                continue
            block_type = str(getattr(block.block_type, "value", block.block_type) or "")

            if block.id in list_runs:
                run = list_runs[block.id]
                if run is None:
                    # Already emitted as part of its run's first item.
                    continue
                flush_figure()
                body.append(run)
                continue

            # A block whose delivered text still carries a pipeline sentinel
            # (``⟦SOUP_MASK…⟧``, ``⟦B⟧``, ``\\llbracket…``) had a restore pass that
            # could not account for a span. The renderer escapes such a span as
            # prose so the document still compiles, but the leak is a defect
            # upstream: attribute it to the block so the quality report and the
            # review verdict can see *which* block leaked rather than a silent
            # count. Checked on the block's own delivered text, so it does not
            # depend on which emit branch below runs.
            if has_sentinel(_block_text(block)):
                self.last_render_flags.append((block.id, "sentinel_leaked"))

            # A block the policy deliberately keeps in the source language (a
            # bibliography entry, an author affiliation) is rendered as its
            # source, never as a "translation". Checked *after* the asset
            # branches below, which have their own handling: a formula is also
            # marked skip-translate but must still be typeset as math, not
            # pasted as literal ``\frac`` source.

            if block_type == "heading":
                text = _block_text(block)
                if dropped_page_furniture(block):
                    continue
                flush_figure()
                level = _heading_level(block)
                anchor_suffix = ""
                topology = getattr(self, "citation_topology", None)
                if topology is not None:
                    from ubt.core.references.extractor import _HEADING_NUMBER_RE
                    from ubt.core.references.models import ReferenceKind

                    m_sec = _HEADING_NUMBER_RE.match(text) or _HEADING_NUMBER_RE.match(
                        block.source_text or ""
                    )
                    if m_sec:
                        sec_key = (m_sec.group(1) or m_sec.group(2) or m_sec.group(3)).lower()
                        if (ReferenceKind.HEADING, sec_key) in topology.key_to_anchor:
                            aid = topology.key_to_anchor[(ReferenceKind.HEADING, sec_key)]
                            anchor_suffix = f" <{aid}>"

                if level == 1 and getattr(self, "enable_notes", False):
                    from ubt.core.notes.manager import render_endnote_section_typst
                    from ubt.core.notes.models import NotePlacement

                    if (
                        self._notes_cfg.placement == NotePlacement.CHAPTER_ENDNOTE
                        and self._notes_current_chapter not in self._notes_emitted_chapters
                        and i > 0
                    ):
                        prev_notes = self._notes_by_chapter.get(self._notes_current_chapter, [])
                        if prev_notes:
                            body.append(render_endnote_section_typst(prev_notes, self._notes_cfg))
                            self._notes_emitted_chapters.add(self._notes_current_chapter)
                        self._notes_current_chapter += 1

                if level == 1 and getattr(self, "enable_backmatter", False):
                    bm_sec = self._bm_manifest.section_for_heading(block.id)
                    if bm_sec is not None:
                        from ubt.core.backmatter.generator import (
                            emit_appendix_switch_typst,
                            render_backmatter_heading_typst,
                        )
                        from ubt.core.backmatter.models import BackmatterKind

                        if (
                            bm_sec.kind == BackmatterKind.APPENDIX
                            and not self._bm_appendix_switched
                        ):
                            body.append(emit_appendix_switch_typst(self._bm_cfg))
                            self._bm_appendix_switched = True

                        aid = anchor_suffix.strip(" <>") if anchor_suffix else None
                        bm_markup = render_backmatter_heading_typst(
                            bm_sec, self._bm_cfg, anchor_id=aid
                        )
                        body.append(self._maybe_pair(block, bm_markup))
                        self.stats.blocks_rendered += 1
                        continue

                if getattr(self, "enable_decor", False) and level == 1:
                    from ubt.core.decor.header import render_chapter_header_typst
                    from ubt.core.decor.models import DecorSpec

                    decor_cfg = (
                        self.decor_spec if isinstance(self.decor_spec, DecorSpec) else DecorSpec()
                    )
                    header_markup = render_chapter_header_typst(text, decor_cfg.header)
                    if anchor_suffix:
                        header_markup = f"{header_markup}\n[#metadata(()) {anchor_suffix.strip()}]"
                    body.append(self._maybe_pair(block, header_markup))
                    chapter_needs_dropcap = True
                    self.stats.blocks_rendered += 1
                    continue

                body.append(
                    self._maybe_pair(
                        block,
                        f"#heading(level: {level})[{self._prose(text, block_runs(block))}]{anchor_suffix}",
                    )
                )
                self.stats.blocks_rendered += 1
                continue

            if block_type == "image":
                # The extractor stores the asset path in the element (``asset_id``)
                # and, for the PDF adapter, repeats it in ``source_text``. Read
                # both so a caller that populates only one still resolves.
                asset = str(getattr(block.element, "asset_id", None) or "") or (
                    block.source_text or ""
                )
                bbox = block.bbox
                fig = _Figure(
                    asset=asset.strip(),
                    width_pt=(bbox.x1 - bbox.x0) if bbox is not None else None,
                    block_id=block.id,
                    is_picture=str(getattr(block, "flow_id", None) or "") == "caption",
                )
                if getattr(self, "enable_grid_figures", False):
                    pending_grid.append(fig)
                else:
                    flush_figure()
                    pending = fig
                continue

            if block_type == "table":
                flush_figure()
                # The source grid is rendered unconditionally: it is the
                # fallback when the target grid does not parse, and the source
                # half of a bilingual pair.
                source_text = (block.source_text or "").strip()
                table_width = self._table_width_pt()
                is_multipage = bool(block.provenance and block.provenance.fused_block_ids)
                source_markup = (
                    render_table(
                        source_text,
                        is_math_ok=self._math_gate,
                        page_columns=self.columns,
                        content_width_pt=table_width,
                        allow_landscape=False,
                        is_multipage=is_multipage,
                        enable_continuation=getattr(self, "enable_table_continuation", False),
                        table_uid=f"src-{block.id}",
                        target_lang=target_lang,
                        continuation_config=getattr(self, "table_continuation_config", None),
                    )
                    or None
                )
                target_text = _block_text(block)
                table_scale: list[float] = []
                # Flipping to landscape is allowed for single-column dense tables
                # when landscape tables are enabled and not in bilingual paired mode
                # (where nested #page inside container blocks is illegal in Typst).
                allow_landscape = self.allow_landscape_tables and not self.bilingual
                markup = render_table(
                    target_text,
                    is_math_ok=self._math_gate,
                    page_columns=self.columns,
                    content_width_pt=table_width,
                    scale_out=table_scale,
                    allow_landscape=allow_landscape,
                    table_scale_factor=getattr(self, "table_scale_factor", 1.0),
                    is_multipage=is_multipage,
                    enable_continuation=getattr(self, "enable_table_continuation", False),
                    table_uid=f"tgt-{block.id}",
                    target_lang=target_lang,
                    continuation_config=getattr(self, "table_continuation_config", None),
                )
                fell_back = False
                if not markup:
                    # The model's translated grid is not a strict pipe table
                    # (e.g. it dropped the header's leading ``|``). Falling back
                    # to the source grid is a *degradation*, not a success, so it
                    # is counted; dropping the block entirely (no source grid to
                    # fall back to either) is fail-closed and must never be
                    # silent — record it so the export stage flags the block
                    # ``render_skip:table_unparseable`` and the quality report
                    # and the content contract both see it.
                    if source_markup:
                        markup = source_markup
                        fell_back = True
                        self.stats.tables_fell_back += 1
                    else:
                        self.stats.tables_dropped += 1
                        self.last_render_skips.append((block.id, "table_unparseable"))
                        continue
                # Record the reconstruction verdict. A table delivered as its
                # *source* grid is a fallback, not a rebuild: verified=False,
                # corrupt=False (the contract reports it UNVERIFIED, not corrupt).
                # A target-grid table is verified only when it kept the source
                # grid's shape; a shape mismatch is a *corrupt* reconstruction the
                # contract must fail on.
                if fell_back:
                    self.table_reconstructions.append(
                        ReconstructionRecord(block_id=block.id, verified=False, corrupt=False)
                    )
                else:
                    verdict = verify_table_reconstruction(source_text, target_text)
                    self.table_reconstructions.append(
                        ReconstructionRecord(
                            block_id=block.id,
                            verified=verdict.ok,
                            corrupt=not verdict.ok,
                            reason=verdict.reason,
                        )
                    )
                # A table laid out at the readability floor was drawn, but at a
                # body size a reader has to squint at. That is a render-quality
                # flag, never a skip: the translated cells shipped.
                if table_scale and table_scale[0] <= _TABLE_MIN_TEXT_SCALE + 1e-6:
                    self.last_render_flags.append((block.id, "low_legibility_font"))

                topology = getattr(self, "citation_topology", None)
                if topology is not None:
                    from ubt.core.references.extractor import _TABLE_CAPTION_RE
                    from ubt.core.references.models import ReferenceKind

                    m_tbl = _TABLE_CAPTION_RE.search(source_text) or _TABLE_CAPTION_RE.search(
                        target_text
                    )
                    if m_tbl:
                        tbl_key = m_tbl.group(1).lower()
                        if (ReferenceKind.TABLE, tbl_key) in topology.key_to_anchor:
                            aid = topology.key_to_anchor[(ReferenceKind.TABLE, tbl_key)]
                            markup = f"{markup} <{aid}>"

                # A bilingual pair whose target fell back to the source grid
                # would show the same grid twice; pair only a genuine
                # translation against the source.
                body.append(
                    self._maybe_pair(
                        block, markup, source_markup=None if fell_back else source_markup
                    )
                )
                self.stats.tables += 1
                self.stats.blocks_rendered += 1
                continue

            if block_type == "formula":
                flush_figure()
                # Prefer the translation: a model that localized a formula's
                # annotations (``\text{parameter}`` -> ``\text{参数}``) puts the
                # result in ``target_text``, and reading only ``source_text``
                # dropped it — the reader got the source-language annotation back
                # with no count. ``_block_text`` applies target→draft→source, so
                # an untranslated formula (the common case) is unchanged.
                source = (block.source_text or "").strip()
                text = _block_text(block)
                from_target = bool(text) and text != source
                converted = typstify_math(strip_math_delimiters(text))
                # The compile gate must run here too, not only in ``render_prose``:
                # a formula block's body skips the prose transpiler path, so a
                # body the transpiler accepts but Typst rejects (``frac(a, )``)
                # used to be emitted verbatim and fail the *whole* document at the
                # single compile. Rejecting it here degrades just this formula to
                # readable Unicode (the ``else`` branch below).
                if (
                    converted is not None
                    and self._math_gate is not None
                    and not self._math_gate(converted)
                ):
                    converted = None
                if converted:
                    if getattr(self, "break_long_formulas", False):
                        from ubt.core.math.breaker import break_equation

                        max_chars = 40 if self.columns > 1 else 65
                        converted = break_equation(converted, target_width=max_chars, align=False)

                    f_scale = getattr(self, "formula_scale", 1.0)
                    formula_markup = (
                        f"#text(size: {10.5 * f_scale:.2f}pt)[$ {converted} $]"
                        if f_scale < 1.0
                        else f"$ {converted} $"
                    )

                    topology = getattr(self, "citation_topology", None)
                    if topology is not None:
                        from ubt.core.references.extractor import _EQUATION_LABEL_RE
                        from ubt.core.references.models import ReferenceKind

                        m_eq = _EQUATION_LABEL_RE.search(source) or _EQUATION_LABEL_RE.search(text)
                        if m_eq:
                            eq_key = (m_eq.group(1) or m_eq.group(2)).lower()
                            if (ReferenceKind.EQUATION, eq_key) in topology.key_to_anchor:
                                aid = topology.key_to_anchor[(ReferenceKind.EQUATION, eq_key)]
                                formula_markup = f"{formula_markup} <{aid}>"

                    body.append(formula_markup)
                    self.stats.formulas += 1
                    self.stats.blocks_rendered += 1
                    if from_target:
                        self.stats.formulas_from_target += 1
                    # A formula transpiled to real Typst math is an interpreted
                    # reconstruction, but there is no round-trip proof here (the
                    # transpiler is not re-parsed), so it is recorded
                    # *unverified* — the contract reports a WARNING, not a silent
                    # "preserved". Only a shape-verified table sets verified=True.
                    self.table_reconstructions.append(
                        ReconstructionRecord(
                            block_id=block.id,
                            verified=False,
                            corrupt=False,
                        )
                    )
                else:
                    # A formula with no convertible body is content, not markup:
                    # dropping it removes an equation from the page with no
                    # trace. Degrade to readable Unicode and count it as a
                    # degradation so the quality report can see it. The one
                    # exception is a bare ``$$`` (the extractor's placeholder for
                    # a formula it could not read), which carries no content to
                    # preserve — the parser recovers those by cropping the source
                    # region before they reach here.
                    fallback = clean_math_fallback_text(source)
                    if fallback:
                        body.append(escape_markup(fallback))
                        self.stats.formulas_degraded += 1
                        self.stats.blocks_rendered += 1
                        # The equation still reads, but as a Unicode
                        # approximation rather than typeset math. Recorded on
                        # the render-quality channel (a drawn block, not a
                        # skip) so the quality report can see the degradation
                        # instead of reading it as a clean render.
                        self.last_render_flags.append((block.id, "formula_degraded"))
                    else:
                        self.stats.formulas_dropped += 1
                        # Nothing to place: the equation is gone from the page.
                        # Fail-closed, like an unparseable table, so the export
                        # stage flags the block and the contract sees the gap
                        # instead of a silently shorter document.
                        self.last_render_skips.append((block.id, "math_unrenderable"))
                continue

            # A code listing the extractor kept as text (short, or its line
            # structure survived) is typeset as code, not prose: Typst folds a
            # markup newline into a space, so the old prose path delivered a
            # listing as one justified paragraph with no line structure and no
            # monospace face. ``raw`` is a string argument, so nothing in the
            # listing can enter code mode (a ``#`` or ``;`` stays literal).
            if block_type == "code":
                flush_figure()
                kept = (block.source_text or "").strip()
                if kept:
                    body.append(_raw_block_markup(kept, container_width_pt=self._table_width_pt()))
                    self.stats.blocks_rendered += 1
                continue

            # A block the policy deliberately keeps in the source language (a
            # bibliography entry, an author affiliation, a code listing) is
            # rendered as its source, never as a "translation". This runs after
            # the asset branches above because a formula is *also* marked
            # skip-translate yet must still be typeset as math rather than
            # pasted as literal ``\frac`` source.
            if _keeps_source(block):
                flush_figure()
                kept = (block.source_text or "").strip()
                if kept:
                    anchor_prefix = ""
                    topology = getattr(self, "citation_topology", None)
                    if topology is not None:
                        from ubt.core.references.extractor import _BIB_ITEM_STARTER_RE
                        from ubt.core.references.models import ReferenceKind

                        m_bib = _BIB_ITEM_STARTER_RE.match(kept)
                        if m_bib:
                            bib_key = (m_bib.group(1) or m_bib.group(2)).lower()
                            if (ReferenceKind.BIBLIOGRAPHY, bib_key) in topology.key_to_anchor:
                                aid = topology.key_to_anchor[(ReferenceKind.BIBLIOGRAPHY, bib_key)]
                                anchor_prefix = f"[#metadata(()) <{aid}>] "

                    body.append(f"{anchor_prefix}{self._prose(kept)}")
                    self.stats.blocks_rendered += 1
                continue

            if (
                block_type in ("footnote", "BlockType.FOOTNOTE")
                or str(getattr(block, "flow_id", None) or "") == "footnote"
            ):
                if getattr(self, "enable_notes", False) and block.id in getattr(
                    self, "_notes_def_ids", set()
                ):
                    continue
                flush_figure()
                text = _block_text(block)
                if text:
                    prose_text = self._prose(text, block_runs(block))
                    markup = f"#block(width: 100%, inset: (top: 0.2em, bottom: 0.2em))[#text(size: 8.5pt, fill: luma(80))[{prose_text}]]"
                    body.append(
                        self._maybe_pair(
                            block,
                            markup,
                        )
                    )
                    self.stats.blocks_rendered += 1
                continue

            # Prose. A short paragraph right after a figure is its caption.
            text = _block_text(block)
            if not text:
                continue
            if getattr(self, "enable_grid_figures", False) and pending_grid:
                if _binds_as_caption(block, text, pending_grid[-1]):
                    body.append(self._figure_grid_markup(pending_grid, build_dir, caption=text))
                    pending_grid = []
                    continue
                flush_figure()
            elif pending is not None:
                # A ``Figure N …`` / ``表 N …`` label binds regardless of length:
                # a long caption is still a caption, and an explicit label is a
                # far stronger signal than the length heuristic. The heuristic
                # stays for an *unlabelled* short paragraph directly under an
                # image (a caption the extractor stripped the "Figure N" from).
                if _binds_as_caption(block, text, pending):
                    pending.caption = text
                    flush_figure()
                    continue
                flush_figure()

            if getattr(self, "enable_epigraph", False):
                if block.id in getattr(self, "_epigraphs_by_id", {}):
                    epi = self._epigraphs_by_id[block.id]
                    from ubt.core.epigraph.generator import render_epigraph_typst

                    epi_markup = render_epigraph_typst(
                        epi,
                        config=self._epigraph_cfg,
                        prose_renderer=lambda t, b=block: self._prose(t, block_runs(b)),
                    )
                    body.append(self._maybe_pair(block, epi_markup))
                    self.stats.blocks_rendered += 1
                    continue
                if block.id in getattr(self, "_abstracts_by_id", {}):
                    ab = self._abstracts_by_id[block.id]
                    from ubt.core.epigraph.generator import render_abstract_typst

                    ab_markup = render_abstract_typst(
                        ab,
                        config=self._epigraph_cfg,
                        prose_renderer=lambda t, b=block: self._prose(t, block_runs(b)),
                    )
                    body.append(self._maybe_pair(block, ab_markup))
                    self.stats.blocks_rendered += 1
                    continue

            rendered_prose = self._prose(text, block_runs(block))
            if getattr(self, "enable_notes", False):
                from ubt.core.notes.manager import (
                    inject_endnote_links_into_text,
                    inject_typst_footnotes_into_text,
                )
                from ubt.core.notes.models import NotePlacement

                notes_for_blk = self._notes_by_source.get(block.id, [])
                if notes_for_blk:
                    if self._notes_cfg.placement == NotePlacement.FOOTNOTE:
                        rendered_prose = inject_typst_footnotes_into_text(
                            rendered_prose, notes_for_blk
                        )
                    else:
                        rendered_prose = inject_endnote_links_into_text(
                            rendered_prose, notes_for_blk, self._notes_cfg.numbering_style
                        )

            if chapter_needs_dropcap and getattr(self, "enable_decor", False):
                from ubt.core.decor.dropcap import apply_dropcap_typst
                from ubt.core.decor.models import DecorSpec

                decor_cfg = (
                    self.decor_spec if isinstance(self.decor_spec, DecorSpec) else DecorSpec()
                )
                rendered_prose = apply_dropcap_typst(rendered_prose, decor_cfg.drop_cap)
                chapter_needs_dropcap = False

            body.append(self._maybe_pair(block, rendered_prose))
            self.stats.blocks_rendered += 1

        flush_figure()

        if getattr(self, "enable_notes", False):
            from ubt.core.notes.manager import render_endnote_section_typst
            from ubt.core.notes.models import NotePlacement

            if self._notes_cfg.placement == NotePlacement.CHAPTER_ENDNOTE:
                if self._notes_current_chapter not in self._notes_emitted_chapters:
                    last_notes = self._notes_by_chapter.get(self._notes_current_chapter, [])
                    if last_notes:
                        body.append(render_endnote_section_typst(last_notes, self._notes_cfg))
                        self._notes_emitted_chapters.add(self._notes_current_chapter)
            elif self._notes_cfg.placement == NotePlacement.BOOK_ENDNOTE:
                all_notes = self._notes_topology.notes
                if all_notes:
                    body.append(render_endnote_section_typst(all_notes, self._notes_cfg))

        if getattr(self, "enable_index", False) or getattr(self, "index", None) is not None:
            from ubt.core.index.extractor import extract_index_from_blocks
            from ubt.core.index.generator import render_index_typst

            idx = self.index or extract_index_from_blocks(blocks)
            if getattr(idx, "entries", None):
                body.append(render_index_typst(idx, columns=3))

        source = self.preamble(target_lang) + "\n\n" + "\n\n".join(body) + "\n"
        # Count the sentinels that reached the page. Escaped prose renders them
        # literally, so they are visible in the text layer; measuring the final
        # source covers every path (prose, tables, captions, formulas) instead
        # of a per-branch increment a new branch could forget.
        self.stats.sentinels_leaked = len(find_sentinels(source))
        return source, self.stats

    def _group_list_runs(
        self, blocks: Sequence[IRBlock], consumed: set[str]
    ) -> dict[str, str | None]:
        """Map each list-item block id to its run's markup (``None`` = already emitted).

        A run is a maximal stretch of consecutive, not-yet-consumed ``LIST_ITEM``
        blocks of the same kind (ordered or bullet). Splitting on kind keeps a
        numbered list from being renumbered as bullets (or the reverse) when two
        lists are adjacent with no paragraph between them.

        The renderer had no list branch at all: a ``LIST_ITEM`` fell through to
        the prose path, so every bullet and every number was dropped and the
        delivered PDF flattened a list into unmarked paragraphs. The marker was
        already on the element (all three readers populate it) and the HTML
        companion already emitted ``<ul><li>``; only the Typst backend lost it.
        """
        runs: dict[str, str | None] = {}
        index = 0
        total = len(blocks)
        while index < total:
            block = blocks[index]
            if block.id in consumed or str(block.block_type) not in (
                "list_item",
                "BlockType.LIST_ITEM",
            ):
                index += 1
                continue
            ordered = _list_is_ordered(block)
            items: list[str] = []
            ids: list[str] = []
            while index < total:
                candidate = blocks[index]
                if candidate.id in consumed:
                    break
                if str(candidate.block_type) not in ("list_item", "BlockType.LIST_ITEM"):
                    break
                if _list_is_ordered(candidate) != ordered:
                    break
                items.append(self._list_item_markup(candidate))
                ids.append(candidate.id)
                index += 1
            if not items:
                index += 1
                continue
            markup = _typst_list_markup(items, ordered=ordered)
            runs[ids[0]] = markup
            for block_id in ids[1:]:
                runs[block_id] = None
            self.stats.blocks_rendered += len(ids)
        return runs

    def _list_item_markup(self, block: IRBlock) -> str:
        """One list item's body, as Typst content (no surrounding brackets)."""
        if _keeps_source(block):
            return self._prose((block.source_text or "").strip())
        return self._maybe_pair(block, self._prose(_block_text(block), block_runs(block)))

    def _figure_markup(self, figure: _Figure, build_dir: Path) -> str:
        """The ``#figure(...)`` call for one image, with its caption if any."""
        resolved = self._resolve_asset(figure.asset)
        if resolved is None:
            self.stats.assets_missing.append((figure.block_id, figure.asset))
            note = escape_markup(f"[missing figure: {Path(figure.asset).name or 'unknown'}]")
            if figure.caption:
                return (
                    f"#block(text(fill: luma(140), size: 9pt)[{note}])\n"
                    f"#block(text(size: 9pt, fill: luma(60))[{self._prose(figure.caption)}])"
                )
            return f"#block(text(fill: luma(140), size: 9pt)[{note}])"
        target = self._copy_asset(resolved, build_dir)
        self.stats.figures += 1
        self.stats.blocks_rendered += 1
        width = self.figure_width
        if figure.width_pt is not None and 0 < figure.width_pt < _FIGURE_NATIVE_MAX_PT:
            ratio = min(figure.width_pt / _FIGURE_REFERENCE_TEXT_WIDTH_PT, 1.0) * 100.0
            width = f"{ratio:.1f}%"

        anchor_suffix = ""
        aid: str | None = None
        topology = getattr(self, "citation_topology", None)
        if topology is not None and figure.caption:
            from ubt.core.references.extractor import _FIGURE_CAPTION_RE
            from ubt.core.references.models import ReferenceKind

            m_fig = _FIGURE_CAPTION_RE.search(figure.caption)
            if m_fig:
                fig_key = m_fig.group(1).lower()
                if (ReferenceKind.FIGURE, fig_key) in topology.key_to_anchor:
                    aid = topology.key_to_anchor[(ReferenceKind.FIGURE, fig_key)]
                    anchor_suffix = f" <{aid}>"

        if figure.caption:
            skip_s = {aid} if aid else None
            lines_fig = [
                "#figure(",
                f'  image("{target.name}", width: {width}),',
                f"  caption: [{self._prose(figure.caption, skip_anchors=skip_s)}],",
                "  numbering: none,",
                f"){anchor_suffix}",
            ]
            return "\n".join(lines_fig)
        return f'#figure(image("{target.name}", width: {width}), numbering: none){anchor_suffix}'

    def _figure_grid_markup(
        self, figures: list[_Figure], build_dir: Path, caption: str = ""
    ) -> str:
        """Render one or more figures as a publication-ready Typst grid panel."""
        from ubt.core.figures.detector import (
            calculate_grid_columns,
            extract_main_and_subcaptions,
        )
        from ubt.core.figures.generator import render_figure_grid_typst
        from ubt.core.figures.models import (
            FigureLayoutConfig,
            FigureLayoutMode,
            FigurePanel,
            SubFigure,
        )

        cfg = (
            self.figure_config
            if isinstance(self.figure_config, FigureLayoutConfig)
            else FigureLayoutConfig()
        )

        main_cap, parsed_subs = extract_main_and_subcaptions(caption) if caption else ("", [])

        if len(figures) == 1 and not parsed_subs:
            single = figures[0]
            single.caption = caption
            return self._figure_markup(single, build_dir)

        subfigs: list[SubFigure] = []
        for i, fig in enumerate(figures):
            resolved = self._resolve_asset(fig.asset)
            if resolved is None:
                self.stats.assets_missing.append((fig.block_id, fig.asset))
                asset_ref = f"[missing figure: {Path(fig.asset).name or 'unknown'}]"
            else:
                target = self._copy_asset(resolved, build_dir)
                self.stats.figures += 1
                self.stats.blocks_rendered += 1
                asset_ref = target.name

            lbl = ""
            sub_cap = ""
            if i < len(parsed_subs):
                lbl, sub_cap = parsed_subs[i]
            elif cfg.auto_subfigures and len(figures) > 1:
                char = chr(ord("a") + (i % 26))
                lbl = f"({char})"

            subfigs.append(
                SubFigure(
                    asset=asset_ref,
                    label=lbl,
                    caption=sub_cap,
                    width_pt=fig.width_pt,
                    block_id=fig.block_id,
                )
            )

        cols = calculate_grid_columns(len(subfigs), cfg.max_columns)
        span_full = any(
            (sf.width_pt is not None and sf.width_pt >= cfg.max_full_width_threshold_pt)
            for sf in subfigs
        )

        anchor_id = ""
        topology = getattr(self, "citation_topology", None)
        if topology is not None and caption:
            from ubt.core.references.extractor import _FIGURE_CAPTION_RE
            from ubt.core.references.models import ReferenceKind

            m_fig = _FIGURE_CAPTION_RE.search(caption)
            if m_fig:
                fig_key = m_fig.group(1).lower()
                if (ReferenceKind.FIGURE, fig_key) in topology.key_to_anchor:
                    anchor_id = topology.key_to_anchor[(ReferenceKind.FIGURE, fig_key)]

        panel = FigurePanel(
            panel_id=f"fig-panel-{len(self.stats.assets_missing) + self.stats.figures}",
            main_caption=main_cap or caption,
            subfigures=subfigs,
            columns=cols,
            gutter_pt=cfg.gutter_pt,
            span_full_width=span_full,
            anchor_id=anchor_id,
            layout_mode=FigureLayoutMode.GRID,
        )

        skip_grid = {anchor_id} if anchor_id else None
        return render_figure_grid_typst(
            panel,
            asset_resolver=lambda a: a,
            prose_renderer=lambda t: self._prose(t, skip_anchors=skip_grid),
            config=cfg,
        )

    def _copy_asset(self, resolved: Path, build_dir: Path) -> Path:
        """Copy one figure into ``build_dir`` under a content-addressed name.

        The build directory is flat and Typst addresses assets by the name in
        ``image("…")``, so two distinct pictures that share a basename (every
        extractor writes ``figure.png`` or ``image1.png``) used to collide: the
        second copy overwrote the first, or — when the sizes matched, because
        the guard compared ``st_size`` — was skipped entirely, and the first
        figure silently rendered the second picture.

        Hashing the bytes makes the name a function of the content, so a
        collision can only happen between identical files, where sharing one
        copy is correct. The stem is kept readable for a human inspecting a
        failed build; the digest is what disambiguates.
        """
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()[:16]
        # A Typst string literal is the boundary here: a stem containing a quote
        # or a backslash closes ``image("…")`` early and the whole document
        # fails to compile. The digest already makes the name unique, so drop
        # everything that is not safe inside a double-quoted literal.
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", resolved.stem).strip("-") or "asset"
        suffix = re.sub(r"[^A-Za-z0-9.]+", "", resolved.suffix)[:10].lower()
        target = build_dir / f"{stem}-{digest}{suffix}"
        if not target.exists():
            shutil.copy2(resolved, target)
            self.stats.assets_copied += 1
        return target

    def _resolve_asset(self, asset: str) -> Path | None:
        """Absolute path of an extracted asset, or ``None`` when unresolvable."""
        if not asset:
            return None
        candidate = Path(asset)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        if self.assets_root is not None:
            rooted = self.assets_root / candidate
            if rooted.exists():
                return rooted
            # Extractors store assets under a content hash; a bare filename may
            # live one level down, so fall back to a bounded search.
            if self.assets_root.exists():
                matches = list(self.assets_root.rglob(candidate.name))
                if matches:
                    return matches[0]
        return candidate if candidate.exists() else None
