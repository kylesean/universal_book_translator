r"""Semantic reflow renderer: IRBlock sequence → one Typst document → PDF.

The overlay path (``LayerCompositor``) draws each translated fragment back onto
its *source* page, so the output inherits every limit of the source geometry:
font sizes shrink to fit a fixed box, tables keep the source language layer,
and a block that cannot fit is quarantined. This renderer takes the other road —
it rebuilds the document from the IR and lets Typst lay it out — which is what
makes a uniform body size, native tables and correct math possible at all
(RFC-2026-02 §1.2).

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ubt.render.table_compiler import parse_markdown_table
from ubt.render.typst.artifact_guard import find_sentinels, has_sentinel
from ubt.render.typst.escape import typst_escape
from ubt.render.typst.math_transpiler import (
    clean_math_fallback_text,
    strip_math_delimiters,
    typstify_math,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ubt.core.ir.models import InlineRun, IRBlock

#: A display-math or inline-math span in the *source* text. Matched before
#: escaping so the formula is transpiled rather than escaped into literal text.
#: A match is only *candidate* math; ``_is_math_span`` decides.
_MATH_SPAN_RE = re.compile(r"(\$\$[\s\S]*?\$\$|\$[^$\n]+\$)")

#: CJK ideographs and kana/hangul. Real math does not contain them; a ``$...$``
#: span that does is prose wearing delimiters.
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7a3]")


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
    """
    inner = span.strip().strip("$").strip()
    if not inner:
        return False
    return _CJK_RE.search(inner) is None


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


def escape_markup(text: str) -> str:
    """Escape a run of prose for Typst markup.

    Delegates to :func:`ubt.render.typst.escape.typst_escape` (the single owner
    of the escape set) and adds nothing: a second escape set here would drift
    from that one, and a character escaped in one and not the other is a compile
    failure that only shows on the document that contains it.
    """
    return typst_escape(text)


def render_prose(text: str, runs: Sequence[InlineRun] = ()) -> str:
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
    """
    spans = _locate_runs(text, runs) if runs else []
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
        ]
    if not spans:
        return _render_math_and_prose(text)
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
        rendered = _render_math_and_prose(chunk)
        style = next((s for s in spans if s[0] <= start and end <= s[1]), None)
        if style is not None:
            rendered = _wrap_style(rendered, style[2], style[3], style[4], style[5])
        out.append(rendered)
    return _break_run_call_chains(out)


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


def _render_math_and_prose(text: str) -> str:
    """Escape prose and transpile math spans, with no style handling.

    A math span the transpiler cannot convert does **not** fall back to escaped
    text: that would put ``\\unknowncmd{x}`` on the page, which is worse than
    the formula being unreadable — it is LaTeX source leaking into a finished
    document. The span degrades to readable Unicode instead
    (:func:`clean_math_fallback_text`), so a reader sees ``\u03d5 + \u03b3 \u2264 \u2211`` rather than
    command names.

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
                out.append(
                    f"\n$ {converted} $\n"
                    if converted
                    else escape_markup(clean_math_fallback_text(part))
                )
            else:
                converted = typstify_math(inner)
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


def _fold_char(ch: str) -> str:
    """Fold one char for tolerant matching.

    Full-width ASCII becomes half-width, the ideographic space a plain space,
    and the unicode dash/quote variants their ASCII forms — so a run text the
    extractor and the target spell differently still matches.
    """
    code = ord(ch)
    if code == 0x3000:
        return " "
    if 0xFF01 <= code <= 0xFF5E:
        return chr(code - 0xFEE0)
    if code == 0x2212 or 0x2010 <= code <= 0x2015:
        return "-"
    if code in (0x2018, 0x2019, 0x02BC, 0x0060, 0x00B4):
        return "'"
    if code in (0x201C, 0x201D):
        return '"'
    return ch


def _fold_for_match(text: str) -> tuple[str, list[int], list[int]]:
    """Fold ``text`` for tolerant matching, with a map back to original indices.

    Drops all whitespace and folds full-width ASCII to half-width, so a run text
    recorded before the target's publishing normalization still matches it: the
    CJK pass inserts a space between Latin and CJK that the draft-time run text
    does not carry.
    """
    folded: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    length = len(text)
    while index < length:
        ch = text[index]
        if ch.isspace():
            index += 1
            continue
        folded.append(_fold_char(ch))
        starts.append(index)
        ends.append(index + 1)
        index += 1
    return "".join(folded), starts, ends


def _folded_find(text: str, needle: str, from_index: int = 0) -> tuple[int, int] | None:
    """Find ``needle`` in ``text`` ignoring whitespace/full-width differences."""
    folded_needle = _fold_for_match(needle)[0].strip()
    if not folded_needle:
        return None
    folded_text, starts, ends = _fold_for_match(text)
    position = 0
    if from_index > 0:
        while position < len(starts) and starts[position] < from_index:
            position += 1
    at = folded_text.find(folded_needle, position)
    if at < 0:
        return None
    return starts[at], ends[at + len(folded_needle) - 1]


def _locate_runs(
    text: str, runs: Sequence[InlineRun]
) -> list[tuple[int, int, bool, bool, bool, str | None]]:
    """Locate each run's literal span in ``text``, in order.

    A run whose text is not present (a span the translation rewrote) is skipped;
    the sequential cursor keeps repeated tokens aligned to their occurrence
    rather than the first one. A span that survives only after the target's CJK
    normalization is found by the tolerant fallback.
    """
    spans: list[tuple[int, int, bool, bool, bool, str | None]] = []
    cursor = 0
    for run in runs:
        if not run.text:
            continue
        index = text.find(run.text, cursor)
        if index < 0:
            index = text.find(run.text)
        if index >= 0:
            spans.append(
                (
                    index,
                    index + len(run.text),
                    run.bold,
                    run.italic,
                    run.superscript,
                    run.color_hex,
                )
            )
            cursor = index + len(run.text)
            continue
        located = _folded_find(text, run.text, cursor)
        if located is None:
            located = _folded_find(text, run.text, 0)
        if located is None:
            continue
        start, end = located
        spans.append((start, end, run.bold, run.italic, run.superscript, run.color_hex))
        cursor = end
    return spans


def block_runs(block: IRBlock) -> tuple[InlineRun, ...]:
    """The block's source- and target-derived inline runs, as renderable runs.

    Source-derived runs (``inline_runs``) come first: they are usually the more
    specific span (a blue citation inside a bold sentence), and the renderer's
    ``next(...)`` lookup lets the first enclosing span win. Target-derived runs
    (``target_runs``, the model's bold markers) follow.
    """
    style = block.style
    if style is None:
        return ()
    return tuple(run for run in (*style.inline_runs, *style.target_runs) if run.text)


def render_table(markup: str) -> str:
    """Render a Markdown table as a Typst ``#table`` with a three-line frame.

    Rebuilt rather than overlaid: the overlay path cannot change a table's
    contents at all, which is why table headers stayed in the source language
    on every overlay delivery (measured 0.96 source-term leakage; RFC §9).
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
            text = render_prose(value.strip())
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
    figures: int = 0
    formulas: int = 0
    #: Formulas that compiled as real math are in ``formulas``; these two count
    #: the ones that did not, so a document whose equations silently became
    #: Unicode approximations (or vanished) is visible in the quality report
    #: instead of looking like it rendered cleanly.
    formulas_degraded: int = 0
    formulas_dropped: int = 0
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


class TypstDocumentRenderer:
    """Build a Typst document from IRBlocks and compile it to PDF.

    The class owns the emitted source so a caller can inspect or test it without
    running the compiler (``build_source``), which is how the renderer is tested
    without depending on the Typst binary.
    """

    #: Base page setup, shared by every target language.
    _PAGE_SETUP = '#set page(paper: "a4", margin: (x: 2.2cm, y: 2.4cm), numbering: "1")'

    @staticmethod
    def preamble(target_lang: str = "zh") -> str:
        """The Typst preamble for ``target_lang``.

        The language and text direction are derived from the target, not
        hardcoded: an Arabic or Hebrew book rendered with ``lang: "zh"`` gets
        Chinese line-breaking and left-to-right layout, which is wrong on every
        page. ``typst`` runs UAX #9 itself, so an RTL target only needs the
        direction declared — the text stays in logical order.

        Kept as one function so the template is auditable in one place instead
        of being assembled from f-strings across the builder.
        """
        from ubt.layout.theme import direction_for

        direction = direction_for(target_lang or "zh").value
        lang = (target_lang or "zh").replace("_", "-")
        rtl = "\n#set text(dir: rtl)" if direction == "rtl" else ""
        return "\n".join(
            [
                TypstDocumentRenderer._PAGE_SETUP,
                f'#set text(font: ("Libertinus Serif", "Noto Serif CJK SC"), size: 10.5pt, lang: "{lang}")'
                + rtl,
                "#set par(justify: true, leading: 0.78em, spacing: 0.9em,"
                " first-line-indent: (amount: 2em, all: true))",
                "#show heading.where(level: 1): it => [",
                "  #v(0.6em)",
                '  #block(text(size: 15pt, weight: "bold",'
                ' font: ("Noto Serif CJK SC", "Libertinus Serif"))[#it.body])',
                "  #v(0.3em)",
                "]",
                "#show heading.where(level: 2): it => [",
                "  #v(0.4em)",
                '  #block(text(size: 12pt, weight: "bold",'
                ' font: ("Noto Serif CJK SC", "Libertinus Serif"))[#it.body])',
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
    ) -> None:
        self.assets_root = Path(assets_root) if assets_root is not None else None
        self.figure_width = figure_width
        self.bilingual = bilingual
        self.stats = ReflowStats()

    def _maybe_pair(self, block: IRBlock, rendered: str) -> str:
        """Add the source line above ``rendered`` when this block has a real pair.

        Three cases must *not* be paired, or the page shows the same sentence
        twice:

        * the run is monolingual (``bilingual`` off);
        * the block was never translated (no target), so its "translation" is
          its source;
        * the policy deliberately kept the source (a bibliography entry), where
          source and target are the same string by construction.
        """
        if not self.bilingual:
            return rendered
        source = (block.source_text or "").strip()
        target = (block.target_text or "").strip()
        if not source or not target or source == target:
            return rendered
        if _is_placeholder(source) or _is_placeholder(target):
            return rendered
        return self._pair(source, rendered)

    def _pair(self, source: str, target: str) -> str:
        """Source above its translation, source visually subordinate.

        The reflow-native equivalent of the overlay path's bilingual compose.
        Overlay paired *pages* (source page, then its composed twin), which only
        works because the composed output is page-aligned with the source; a
        reflowed document has no such alignment, so the pairing is per block.
        That is also the more useful form for a rebuilt document: the reader
        never turns a page to find the sentence being compared.

        The source keeps its own math handling (a formula must stay a formula,
        not become literal ``\\frac`` text), so it goes through
        :func:`render_prose` exactly as the target does.
        """
        return (
            "#block(above: 0.5em, below: 0.25em)[\n"
            '  #text(size: 9pt, fill: luma(115), style: "italic")['
            f"{render_prose(source)}]\n"
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
        body: list[str] = []
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

            # A block the policy deliberately keeps in the source language (a
            # bibliography entry, an author affiliation) is rendered as its
            # source, never as a "translation". Checked *after* the asset
            # branches below, which have their own handling: a formula is also
            # marked skip-translate but must still be typeset as math, not
            # pasted as literal ``\frac`` source.

            if block_type == "heading":
                text = _block_text(block)
                if _CHROME_HEADING_RE.match(text):
                    continue
                flush_figure()
                level = _heading_level(block)
                body.append(
                    self._maybe_pair(
                        block,
                        f"#heading(level: {level})[{render_prose(text, block_runs(block))}]",
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
                markup = render_table(_block_text(block))
                if markup:
                    body.append(self._maybe_pair(block, markup))
                    self.stats.tables += 1
                    self.stats.blocks_rendered += 1
                continue

            if block_type == "formula":
                flush_figure()
                source = (block.source_text or "").strip()
                converted = typstify_math(strip_math_delimiters(source))
                if converted:
                    body.append(f"$ {converted} $")
                    self.stats.formulas += 1
                    self.stats.blocks_rendered += 1
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
                    else:
                        self.stats.formulas_dropped += 1
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
                    body.append(render_prose(kept))
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
            body.append(self._maybe_pair(block, render_prose(text, block_runs(block))))
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
            # visible placeholder so the gap is attributable.
            note = escape_markup(f"[missing figure: {Path(figure.asset).name or 'unknown'}]")
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
                f"  caption: [{render_prose(figure.caption)}],\n"
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
