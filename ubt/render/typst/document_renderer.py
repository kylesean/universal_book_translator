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

import re
import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ubt.core.content.nodes import ReconstructionRecord
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
    r"^\s*(?:Figure|Fig\.?|Table|Listing|Algorithm|图|表|附图|附表)\s*[\d.]+\s*[:：.]",
    re.IGNORECASE,
)

#: A heading that is really running chrome (an arXiv stamp, a journal line, a
#: submission date). Kept out of the body so the document opens on its title.
_CHROME_HEADING_RE = re.compile(
    r"^(?:arXiv:|Journal of Machine Learning|Submitted \d|Preprint\b|"
    r"Under review\b|Published \d|Revised \d)",
    re.IGNORECASE,
)


def dropped_page_furniture(block: IRBlock) -> bool:
    """Whether the renderer deliberately omits this block from the artifact.

    The single owner of that decision: a heading the renderer recognizes as
    running chrome (an arXiv/journal stamp, a submission date) is dropped so the
    document opens on its title (see ``build_source``). The delivered-artifact
    check must ask *this* function rather than re-test the pattern, or an audit
    would demand a realization the renderer is right not to place — the shape
    that reported a false ``[FAIL] artifact carries 280/281`` on a page-furniture
    heading.

    Kept narrow on purpose: only the chrome-heading drop counts. A block merely
    *kept in the source* (a bibliography entry, an affiliation) is still placed
    verbatim and must still appear, so it is not furniture.
    """
    if str(getattr(block.block_type, "value", block.block_type) or "") != "heading":
        return False
    text = _block_text(block)
    return bool(_CHROME_HEADING_RE.match(text))


def escape_markup(text: str) -> str:
    """Escape a run of prose for Typst markup.

    Delegates to :func:`ubt.render.typst.escape.typst_escape` (the single owner
    of the escape set) and adds nothing: a second escape set here would drift
    from that one, and a character escaped in one and not the other is a compile
    failure that only shows on the document that contains it.
    """
    return typst_escape(text)


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
    """Join segments without letting a styled run absorb a following ``(``.

    A styled run ends in ``]`` (``#strong[...]``). Typst parses a following ``(``
    as another call on that run's *result* — ``#strong[x](y)`` calls the content
    ``x`` as a function — so a citation's opening paren right after a bold run
    (``#strong[Firecracker microVMs ](Agache et al., 2020)``) fails to compile
    and the whole fragment descends to the source. Emit that one literal ``(`` as
    an explicit ``#text("(")``, which renders identically and ends the code
    expression cleanly. A literal ``[`` needs no such guard: ``typst_escape``
    already backslash-escapes it.
    """
    merged: list[str] = []
    for segment in segments:
        if merged and merged[-1].endswith("]") and segment[:1] == "(":
            merged.append('#text("(")')
            merged.append(segment[1:])
        else:
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


def render_table(markup: str, *, is_math_ok: Callable[[str], bool] | None = None) -> str:
    """Render a Markdown table as a Typst ``#table`` with a three-line frame.

    Rebuilt from the IR rather than drawn over the source: a source-layer table
    cannot change its contents at all, which is why table headers stayed in the
    source language when tables were left on the source page (measured 0.96
    source-term leakage; RFC §9).

    ``is_math_ok`` is the same compile gate :func:`render_prose` takes; a cell's
    ``$\\frac{a}{}$`` reaches the compiler exactly as a paragraph's does.
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
            text = render_prose(value.strip(), is_math_ok=is_math_ok)
            rendered.append(f"[*{text}*]" if header and text else f"[{text}]")
        return ", ".join(rendered)

    lines = [
        "#table(",
        f"  columns: {columns},",
        "  inset: 5pt,",
        "  align: (col, row) => if row == 0 { center + horizon } else { left + horizon },",
        "  stroke: none,",
        "  table.hline(stroke: 1.1pt),",
        f"  {cells(headers, header=True)},",
        "  table.hline(stroke: 0.5pt),",
    ]
    for row in rows:
        lines.append(f"  {cells(row, header=False)},")
    lines += ["  table.hline(stroke: 1.1pt),", ")"]
    return "\n".join(lines)


@dataclass(slots=True)
class _Figure:
    """A pending image plus the caption that follows it."""

    asset: str
    caption: str = ""


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
    assets_missing: list[str] = field(default_factory=list)


#: The quarantine placeholder the triage stage writes into ``target_text``
#: (``<mark class="ubt-blocked-human" …>``). It is markup about the block, not
#: the block's content, so it must never reach the rendered page.
_PLACEHOLDER_RE = re.compile(r"^\s*<mark\b", re.IGNORECASE)


def _is_placeholder(text: str) -> bool:
    return bool(_PLACEHOLDER_RE.match(text))


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
    """The text to render for a block: its translation, else its source.

    A quarantined block (``BLOCKED_HUMAN``) carries a ``<mark>`` placeholder as
    its ``target_text``, not a translation, so it is skipped: rendering it would
    put the quarantine HTML on the page. The *draft* is used instead when it
    exists — the draft is real translated prose that a false-positive gate
    rejected, and shipping it is the whole point of the fail-open rule.

    A block with no target and no draft is *not* silently dropped — the reader
    gets the source sentence.
    """
    target = (block.target_text or "").strip()
    if target and not _is_placeholder(target):
        return target
    draft = (block.draft_text or "").strip()
    if draft and not _is_placeholder(draft):
        return draft
    return (block.source_text or "").strip()


def _heading_level(block: IRBlock) -> int:
    """Heading depth, clamped to Typst's usable range (1..6)."""
    level = getattr(block.element, "level", 1)
    if not isinstance(level, int) or level < 1:
        return 1
    return min(level, 6)


def _raw_block_markup(text: str) -> str:
    """Typeset a code listing as a monospace block that keeps its line breaks.

    A markup newline folds into a space and prose escaping would leave the
    listing set in the body face, so the listing is passed to Typst ``raw`` as
    a plain *string* (``raw("...")``): inside that string literal nothing can
    enter markup or code mode, so every ``#``, ``"``, ``;``, ``\\`` and
    newline in the listing stays literal. The only character that must be
    escaped is the double quote itself, so the payload cannot terminate the
    string early.
    """
    payload = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'#block(fill: luma(245), inset: 8pt, radius: 4pt, width: 100%)[#raw("{payload}")]'


class TypstDocumentRenderer:
    """Build a Typst document from IRBlocks and compile it to PDF.

    The class owns the emitted source so a caller can inspect or test it without
    running the compiler (``build_source``), which is how the renderer is tested
    without depending on the Typst binary.
    """

    #: Base page setup, shared by every target language.
    _PAGE_SETUP = '#set page(paper: "a4", margin: (x: 2.2cm, y: 2.4cm), numbering: "1")'

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

            return academic_preamble(
                target_lang=target_lang,
                font_stack=self.font_stack,
                paper_title=getattr(self, "paper_title", ""),
            )

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
        from ubt.core.language_profile import resolve_font_config

        if self.font_stack:
            stack: tuple[str, ...] = self.font_stack
        else:
            stack = tuple(resolve_font_config(lang).typst_fonts)
        body = ", ".join(f'"{face}"' for face in stack)
        cjk_first = ", ".join(f'"{face}"' for face in reversed(stack))
        return "\n".join(
            [
                TypstDocumentRenderer._PAGE_SETUP,
                f'#set text(font: ({body}), size: 10.5pt, lang: "{lang}")' + rtl,
                "#set par(justify: true, leading: 0.78em, spacing: 0.9em,"
                " first-line-indent: (amount: 2em, all: true))",
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
        paper_title: str = "",
        abstract: str = "",
    ) -> None:
        self.assets_root = Path(assets_root) if assets_root is not None else None
        self.figure_width = figure_width
        self.bilingual = bilingual
        self.layout_template = layout_template
        self.paper_title = paper_title
        self.abstract = abstract
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

    def _prose(self, text: str, runs: Sequence[InlineRun] = ()) -> str:
        """``render_prose`` bound to this renderer's compile gate."""
        return render_prose(text, runs, is_math_ok=self._math_gate)

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
        self._prewarm_math_gate(blocks)
        body: list[str] = []
        if getattr(self, "layout_template", "book") == "academic" and getattr(
            self, "paper_title", ""
        ):
            from ubt.render.typst.academic import academic_title_banner

            banner = academic_title_banner(
                title=self.paper_title,
                abstract=getattr(self, "abstract", ""),
            )
            if banner:
                body.append(banner)
        pending: _Figure | None = None

        def flush_figure() -> None:
            nonlocal pending
            if pending is None:
                return
            body.append(self._figure_markup(pending, build_dir))
            pending = None

        for block in blocks:
            self.stats.blocks_seen += 1
            block_type = str(getattr(block.block_type, "value", block.block_type) or "")

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
                body.append(
                    self._maybe_pair(
                        block,
                        f"#heading(level: {level})[{self._prose(text, block_runs(block))}]",
                    )
                )
                self.stats.blocks_rendered += 1
                continue

            if block_type == "image":
                flush_figure()
                # The extractor stores the asset path in the element (``asset_id``)
                # and, for the PDF adapter, repeats it in ``source_text``. Read
                # both so a caller that populates only one still resolves.
                asset = str(getattr(block.element, "asset_id", None) or "") or (
                    block.source_text or ""
                )
                pending = _Figure(asset=asset.strip())
                continue

            if block_type == "table":
                flush_figure()
                # The source grid is rendered unconditionally: it is the
                # fallback when the target grid does not parse, and the source
                # half of a bilingual pair.
                source_text = (block.source_text or "").strip()
                source_markup = render_table(source_text, is_math_ok=self._math_gate) or None
                target_text = _block_text(block)
                markup = render_table(target_text, is_math_ok=self._math_gate)
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
                    body.append(f"$ {converted} $")
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
                    body.append(_raw_block_markup(kept))
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
                    body.append(self._prose(kept))
                    self.stats.blocks_rendered += 1
                continue

            # Prose. A short paragraph right after a figure is its caption.
            text = _block_text(block)
            if not text:
                continue
            if pending is not None:
                if _CAPTION_RE.match(text) or len(text) <= 300:
                    pending.caption = text
                    flush_figure()
                    continue
                flush_figure()
            body.append(self._maybe_pair(block, self._prose(text, block_runs(block))))
            self.stats.blocks_rendered += 1

        flush_figure()
        source = self.preamble(target_lang) + "\n\n" + "\n\n".join(body) + "\n"
        # Count the sentinels that reached the page. Escaped prose renders them
        # literally, so they are visible in the text layer; measuring the final
        # source covers every path (prose, tables, captions, formulas) instead
        # of a per-branch increment a new branch could forget.
        self.stats.sentinels_leaked = len(find_sentinels(source))
        return source, self.stats

    def _figure_markup(self, figure: _Figure, build_dir: Path) -> str:
        """The ``#figure(...)`` call for one image, with its caption if any."""
        resolved = self._resolve_asset(figure.asset)
        if resolved is None:
            self.stats.assets_missing.append(figure.asset)
            # A missing asset must not silently vanish from the page: leave a
            # visible placeholder so the gap is attributable. The caption is
            # rendered too — it is translated prose that the reader needs, and
            # it is the *only* surviving description of what the picture showed,
            # which makes dropping it worse here than anywhere else.
            note = escape_markup(f"[missing figure: {Path(figure.asset).name or 'unknown'}]")
            if figure.caption:
                return (
                    f"#block(text(fill: luma(140), size: 9pt)[{note}])\n"
                    f"#block(text(size: 9pt, fill: luma(60))[{self._prose(figure.caption)}])"
                )
            return f"#block(text(fill: luma(140), size: 9pt)[{note}])"
        target = build_dir / resolved.name
        if not target.exists() or target.stat().st_size != resolved.stat().st_size:
            shutil.copy2(resolved, target)
            self.stats.assets_copied += 1
        self.stats.figures += 1
        self.stats.blocks_rendered += 1
        if figure.caption:
            return (
                "#figure(\n"
                f'  image("{target.name}", width: {self.figure_width}),\n'
                f"  caption: [{self._prose(figure.caption)}],\n"
                "  numbering: none,\n)"
            )
        return f'#figure(image("{target.name}", width: {self.figure_width}), numbering: none)'

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
