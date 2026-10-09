"""Lowering translated blocks onto the source page (render compositor).

:class:`LayerCompositor` composes each page from three absolute layers: Layer 0
the source page's own vectors/rasters untouched, Layer 1 a background rectangle
masking the regions being replaced, Layer 2 a typeset fragment (Typst
micro-typeset) placed into each such region. A fragment that will not typeset is
recorded as kept-in-source (no mask is painted), so composition can never lose
content.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar

import pikepdf

from ubt.adapters.pdf import pdf_struct
from ubt.adapters.pdf.stream_strip import shared_form_objgens, strip_page_text_pikepdf
from ubt.adapters.pdf.typst_compile import typst_compile as typst_compile
from ubt.adapters.pdf.typst_math_probe import TypstMathProbe
from ubt.cache.dirs import cache_root
from ubt.core.atomic import atomic_write_path
from ubt.core.ir.bifurcation import bifurcate_blocks
from ubt.core.ir.continuation import find_continuation_runs, join_continuous_text
from ubt.core.ir.models import BlockType, IRBlock
from ubt.core.narrowing import narrow
from ubt.model.span import BBox, CompositeSpan, PhysicalBox
from ubt.render.flow import FlowPlacement, solve_flow


@dataclass(frozen=True, slots=True)
class Placement:
    """One element's position in the artifact: whether it was drawn or kept."""

    element_id: str
    page: int
    drawn: bool
    detail: str = ""
    #: The size (pt) the target was drawn at, when the lowering chose it by
    #: fitting (``None`` for a mask-only part, a math/TOC fragment, or a
    #: source-kept element). Below ``_MIN_FONT_PT`` the drawn fragment is no
    #: longer reading-grade and is reported as a ``low_legibility_font`` defect.
    drawn_pt: float | None = None

    @property
    def kept_source(self) -> bool:
        """True when the lowering kept the source because it cannot draw the target."""
        return not self.drawn


@dataclass(frozen=True, slots=True)
class Composition:
    """A lowered artifact plus, per element, how it was placed."""

    output_path: Path
    placements: tuple[Placement, ...]

    @property
    def by_page(self) -> dict[int, tuple[Placement, ...]]:
        """The placements grouped by source page -- the per-page composition."""
        grouped: dict[int, list[Placement]] = {}
        for placement in self.placements:
            grouped.setdefault(placement.page, []).append(placement)
        return {page: tuple(items) for page, items in grouped.items()}

    @property
    def kept_source_ids(self) -> tuple[str, ...]:
        """Elements the lowering had to keep as source (no drawing for their target)."""
        return tuple(placement.element_id for placement in self.placements if placement.kept_source)

    @property
    def low_legibility_fonts(self) -> tuple[tuple[str, float], ...]:
        """Drawn elements whose fragment had to drop below the readable floor.

        ``(element_id, size_pt)`` pairs for every placement that was drawn at
        less than ``_MIN_FONT_PT`` -- a translation delivered at a size no reader
        can comfortably read. Recorded rather than silent: the adapter turns each
        into a ``low_legibility_font`` defect flag on the block.
        """
        return tuple(
            (placement.element_id, placement.drawn_pt)
            for placement in self.placements
            if placement.drawn
            and placement.drawn_pt is not None
            and placement.drawn_pt < _MIN_FONT_PT
        )


# --------------------------------------------------------------------------- #
# LayerCompositor: three-layer absolute page composition.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class StyledRun:
    """A span of the *target* text to draw with a source-derived style.

    ``text`` is the literal span; the renderer locates it in the fragment body
    (after normalization) and wraps it in Typst markup. Only spans that survive
    translation verbatim are mapped here (see :func:`_map_inline_runs`).
    """

    text: str
    bold: bool = False
    italic: bool = False
    superscript: bool = False
    color_hex: str | None = None


@dataclass(frozen=True, slots=True)
class Overlay:
    """One reconstructed region to draw: where, and the delivered text for it.

    ``boxes`` (optional) is a reading-order chain of physical boxes the text may
    flow across; empty means the single ``bbox`` on ``page``.
    """

    element_id: str
    page: int
    bbox: BBox
    text: str
    boxes: tuple[PhysicalBox, ...] = ()
    #: ``"text"`` typesets prose; ``"math"`` typesets ``text`` as a formula;
    #: ``"toc"`` typesets ``text`` as a TOC title with leaders and ``toc_page``.
    kind: str = "text"
    #: The source text, for an in-place bilingual overlay (target over source).
    #: Empty for a monolingual overlay.
    source: str = ""
    #: For a ``"toc"`` overlay, the page number its dot leaders point at.
    toc_page: str = ""
    #: Whether a ``"toc"`` overlay renders dot leaders (False uses flush-right space).
    toc_leaders: bool = True
    font_size: float | None = None
    is_bold: bool = False
    #: First-line indent (pt) for a body paragraph or a list item's marker
    #: column, from the source's typography. Emitted as Typst
    #: ``par.first-line-indent``; ``None``/0 disables it.
    indent_pt: float | None = None
    #: The source block is a centered heading (symmetric line gaps): the
    #: fragment centers each line within the box instead of hugging the left
    #: edge, reproducing the title's original alignment.
    align_center: bool = False
    #: The *original* source geometry to mask and relocate links against. A
    #: reflowed overlay draws at ``bbox``/``boxes`` but must still erase the
    #: source text where it was, so the mask follows this chain, not the draw box.
    #: Empty means "the draw box is also the source box" (the legacy path).
    mask_boxes: tuple[PhysicalBox, ...] = ()
    #: A reflowed overlay whose box height already equals the fragment's natural
    #: height: the compositor must not add line slack or run the fit search, or
    #: the drawn text would be shorter than its box (a gap) or shrunk to fit it.
    fixed_box: bool = False
    #: The draw size the reflow pass measured the box at. A reflowed box's height
    #: *is* that size's natural height, so the compositor must draw at exactly it:
    #: re-deriving the size from the block's own cap can be a hair larger, wrap
    #: one more line past the box, and ``clip`` hides it -- an ink-less line the
    #: text layer still reports (the visual gate's "text_occluded").
    draw_size_pt: float | None = None
    #: Source-derived styled spans to re-apply to the target (a blue citation, a
    #: raised footnote dagger) where the span survives verbatim.
    runs: tuple[StyledRun, ...] = ()

    @property
    def flow_boxes(self) -> tuple[PhysicalBox, ...]:
        return self.boxes or (PhysicalBox.of(self.page, self.bbox),)

    @property
    def mask_flow_boxes(self) -> tuple[PhysicalBox, ...]:
        """The chain to mask/link against: the source geometry when reflowed."""
        return self.mask_boxes or self.flow_boxes

    def typeset_math(self, latex: str, width_pt: float, height_pt: float) -> Path | None:
        """Typeset a formula into a PDF page exactly ``width`` x ``height`` pt."""
        ...


#: A translated list item that already opens with a marker (the model
#: reproduced the bullet) is left untouched; only a markerless item gets one.
#: Full-width forms (``（1）``, ``１.``, ``（a）``) are matched too, so a model
#: that switches to CJK punctuation does not get a second marker.
_LEADING_MARKER_RE = re.compile(
    r"^\s*(?:"
    r"[•⁃◦▪●*\-]"  # unordered bullet
    r"|[（(]?[0-9０-９]{1,3}[.．、)）]"  # 1. 1) (1) 1、 （1） １.
    r"|[（(]?[a-zA-ZＡ-Ｚａ-ｚ][.．)）]"  # a. a) (a) （a）
    r"|[一二三四五六七八九十百]+[、.．)）]"  # 一、 二）
    r")\s*"
)


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
    CJK pass inserts a space between Latin and CJK (``SoL-Pi通过`` -> ``SoL-Pi
    通过``) that the draft-time run text does not carry. ``starts[i]`` is the
    original index of folded char ``i``; ``ends[i]`` is one past it.
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
    """Find ``needle`` in ``text`` ignoring whitespace/full-width differences.

    Returns the matched ``(start, end)`` in the *original* text, or ``None``.
    """
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
    text: str, runs: Sequence[StyledRun]
) -> list[tuple[int, int, bool, bool, bool, str | None]]:
    """Locate each run's literal span in ``text``, in order.

    A run whose text is not present (a span the translation rewrote) is skipped;
    the sequential cursor keeps repeated tokens (a year, an author) aligned to
    their occurrence rather than the first one on the page. A span that survives
    only after the target's CJK normalization (whitespace collapsed, full-width
    punctuation/digits folded) is found by the tolerant fallback.
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


def _styled_runs(block: IRBlock) -> tuple[StyledRun, ...]:
    """The block's source- and target-derived inline runs, as renderable runs.

    Source-derived runs (``inline_runs``) come first: they are usually the more
    specific span (a blue citation inside a bold sentence), and the renderer's
    ``next(...)`` span lookup lets the first enclosing span win. Target-derived
    runs (``target_runs``, the model's ``⟦B⟧`` markers) follow.
    """
    style = block.style
    if style is None:
        return ()
    return tuple(
        StyledRun(
            text=run.text,
            bold=run.bold,
            italic=run.italic,
            superscript=run.superscript,
            color_hex=run.color_hex,
        )
        for run in (*style.inline_runs, *style.target_runs)
        if run.text
    )


def _with_list_marker(block: IRBlock, text: str) -> str:
    """Restore a dropped list bullet/number onto a translated target.

    The reader stores a ``ListItem``'s marker on the element and strips it from
    the element text, so the compositor -- which draws only the text -- loses the
    bullet once the source glyph is masked away with the region. An ordered
    number the extractor did not keep is unrecoverable here, so a markerless item
    falls back to the unordered bullet, exactly as the retired rigid typesetter
    did. A text that already opens with a marker is left byte-identical.
    """
    if not text or block.block_type is not BlockType.LIST_ITEM:
        return text
    if _LEADING_MARKER_RE.match(text):
        return text
    marker = getattr(block.element, "marker", "") or "•"
    return f"{marker} {text}"


#: Readable font-size floor (pt) for a typeset fragment. Below this the text is
#: not reading-grade: the fragment is still drawn when the box cannot hold the
#: target at this size at all, but the drawn size is recorded as a defect (see
#: `_MIN_DRAW_PT`). The previous 2pt floor traded readability for "delivery
#: contract: something is drawn"; a 2pt glyph satisfies no reader. `_FIT_TOL` is
#: the slack (pt) a fit search allows around the measured height.
_MIN_FONT_PT = 6.0
#: Hard floor (pt) a *monolingual* target's fit search may shrink to before it
#: gives up and keeps the source. A box shorter than the target's line at
#: `_MIN_FONT_PT` -- a source line the extractor split into fragments, a code or
#: formula stub -- can only be replaced by dropping below the readable floor or
#: not at all; between a 4pt glyph and the source text, the glyph still delivers
#: the translation, so it is drawn and flagged (``low_legibility_font``) rather
#: than silently kept. A bilingual pair and a multi-box continuation flow do
#: *not* descend this far: the pair would draw its echo at a fraction of an
#: already-cramped size, and a clipped flow line is an ink-less text layer the
#: visual gate rejects.
_MIN_DRAW_PT = 4.0
_FIT_TOL = 0.5
#: Paragraph leading (extra inter-line space, in em) for every typeset fragment.
#: Typst adds it on top of the font's own line height; the source's own pitch is
#: roughly ``1.2em``. A small positive value opens the drawn lines for readability
#: (``0.0`` left a 1.05em pitch that reads cramped in CJK) while keeping a
#: multi-line fragment close to the source's block height. A reflowed overlay
#: measures its natural height *at this leading*, so the value costs no fit
#: shrink there.
_PAR_LEADING_EM = 0.18
#: Vertical slack (as a fraction of the source font size) added *below* an overlay
#: box before fitting and drawing. The extracted box is the glyph ink, not the
#: line box: it stops at the last line's descender and omits the leading, so a
#: fragment drawn at the source size overflows it and the fit shrinks the whole
#: document. One line's leading is enough to hold the drawn fragment; taking it
#: below the box leaves the top (and so the baseline) fixed.
_LINE_SLACK_RATIO = 0.30
#: Source font size assumed when a block carries none, for the slack above.
_DEFAULT_SLACK_FONT_PT = 10.0
#: Only streams at least this large are deduplicated; below it the win is nil.
_DEDUP_MIN_BYTES = 8192

#: In-place bilingual: the source (secondary) font size as a fraction of the
#: target (primary), the gap between the two blocks (in em of the primary), and
#: the muted fill that marks the source as secondary. The target is fitted; the
#: source follows it down.
_BILINGUAL_RATIO = 0.78
_BILINGUAL_GAP_EM = 0.35
_BILINGUAL_FILL = "#5b5b5b"
#: Bilingual prefetch requests reuse the ``(kind, text, w, h)`` request shape by
#: joining the target and source with this control separator.
_BILINGUAL_SEP = "\x1f"
#: Every fragment's ``#set text`` names an explicit weight. Typst ``#set`` rules
#: apply from their position to the end of the enclosing content, and a batched
#: fragment document is one flat flow: a bold fragment (a section heading) would
#: otherwise leak its weight into every later section whose ``#set text`` omits
#: the parameter, bolding whole pages of body text.
_WEIGHT_REGULAR = ', weight: "regular"'
_WEIGHT_BOLD = ', weight: "bold"'


def _text_weight_line(bold: bool) -> str:
    return _WEIGHT_BOLD if bold else _WEIGHT_REGULAR


def _par_line(leading_em: float) -> str:
    """The shared ``#set par(...)`` rule for every typeset fragment."""
    return f"#set par(leading: {leading_em}em)"


def _indent_hspace(indent_pt: float | None) -> str:
    """A first-line indent as a non-weak horizontal space.

    Typst's ``par.first-line-indent`` is the wrong tool inside a fixed-height
    ``#box``: the indent makes the content a hair taller than the box, which then
    spills to a *second* page, and the compositor -- which copies page 1 -- draws
    nothing at all. A leading ``#h`` is not trimmed at line start when a non-space
    token follows it, so an empty content block trails it; the space shifts the
    first line while the wrapped lines stay at the margin, which is exactly an
    indent. The empty block renders nothing (no glyph, no ink), and the trailing
    space is required: without it a body that opens with ``(`` or ``[``
    ("(1) Rollout ...") turns the empty block into a call (``#[](1)``) and the
    fragment fails to compile, so the item kept its source language. The space
    itself emits no glyph and no text-layer character, and the prefix leaves the
    measured height unchanged, so a box that fits without it still fits with it.
    """
    return f"#h({indent_pt}pt, weak: false)#[] " if indent_pt and indent_pt > 0 else ""


def _align_line(center: bool) -> str:
    """The centering setup line for a fragment source (empty = left-aligned).

    Centering does not change wrapping or natural height, so the measure paths
    omit it — only the drawn fragment carries the flag.
    """
    return "#set align(center)\n" if center else ""


def _line_slack(font_size: float | None) -> float:
    """Points of extra height to add below an overlay box (see ``_LINE_SLACK_RATIO``)."""
    size = font_size if font_size and font_size > 0 else _DEFAULT_SLACK_FONT_PT
    return _LINE_SLACK_RATIO * size


def bilingual_request_text(target: str, source: str) -> str:
    """Join a bilingual prefetch request's target and source for ``prefetch``."""
    return f"{target}{_BILINGUAL_SEP}{source}"


def _strip_math_delimiters(text: str) -> str:
    """The body of a math span, whatever delimiters the extractor used."""
    stripped = text.strip()
    for opening, closing in (("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")):
        if (
            len(stripped) >= len(opening) + len(closing)
            and stripped.startswith(opening)
            and stripped.endswith(closing)
        ):
            return stripped[len(opening) : len(stripped) - len(closing)].strip()
    return stripped


class TypstFragmentTypesetter:
    """Typst as a pure fragment micro-core: one page per box, compiled to PDF.

    This is the role reversal the composition design calls for -- Typst lays out
    a single bounded rectangle (wrapping, hyphenation, CJK/Latin mixing) and
    nothing else; the page furniture, margins and assembly belong to the
    compositor. Fragments are content-addressed by their generated source, so an
    identical box compiles once per run.
    """

    name: ClassVar[str] = "typst-fragment"

    def __init__(
        self,
        *,
        binary: str = "typst",
        size_pt: float = 12.0,
        font: str | Sequence[str] | None = None,
        cache_dir: Path | str | None = None,
        target_lang: str = "zh",
        math_probe: TypstMathProbe | None = None,
        par_leading_em: float = _PAR_LEADING_EM,
    ) -> None:
        self._binary = binary
        self._size_pt = size_pt
        self._font = font
        self._target_lang = target_lang
        self._par_leading_em = par_leading_em
        #: Inline math is emitted only when a probe (the same Typst that will
        #: compile the fragment) validates it; the probe is pure-cache, so a
        #: paragraph measured at several sizes converts once.
        self._math_probe = math_probe if math_probe is not None else TypstMathProbe(binary)
        if cache_dir in (":temp:", "temp"):
            self._work = Path(tempfile.mkdtemp(prefix="ubt-fragment-"))
            self._is_temp = True
        elif cache_dir is not None:
            self._work = Path(cache_dir)
            self._is_temp = False
            self._work.mkdir(parents=True, exist_ok=True)
        else:
            self._work = cache_root() / "fragments"
            self._is_temp = False
            self._work.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._body_cache: dict[Any, str] = {}
        self._measure_cache: dict[tuple[str, float, float], float] = {}
        self._heading_cache: dict[tuple[str, float, float], float] = {}
        self._bilingual_cache: dict[tuple[str, float, float], float] = {}
        #: Measure cache for indent-bearing overlays, keyed with the indent so a
        #: paragraph's indented and unindented wraps do not collide.
        self._indent_cache: dict[Any, float] = {}
        #: Measure cache for the reflow fit, keyed by the full 6-tuple item.
        self._fixed_measure_cache: dict[Any, float] = {}
        #: Side channel for the compositor: the size (pt) the most recent
        #: fragment compile *drew* at, or ``None`` when it drew nothing or the
        #: kind has no fitted size (math, TOC). ``_compile_form`` resets it and
        #: reads it back to record a ``low_legibility_font`` defect; compiles run
        #: one at a time inside ``LayerCompositor.compose``, so a plain attribute
        #: cannot race.
        self.last_drawn_pt: float | None = None

    @property
    def _font_line(self) -> str:
        if not self._font:
            return ""
        if isinstance(self._font, (list, tuple)):
            clean_fonts = [f for f in self._font if f]
            if not clean_fonts:
                return ""
            fonts_quoted = ", ".join(f'"{f}"' for f in clean_fonts)
            return f", font: ({fonts_quoted})"
        return f', font: "{self._font}"'

    @property
    def _dir_line(self) -> str:
        """The text-direction setup line, non-empty only for RTL targets.

        Every fragment source (measure and draw, monolingual and bilingual)
        must carry it or an Arabic/Hebrew book typesets with an LTR base
        direction — wrong bidi ordering at every run boundary.
        """
        from ubt.adapters.pdf.font_probe import needs_rtl

        return "#set text(dir: rtl)\n" if needs_rtl(self._target_lang) else ""

    def _key(self, source: str, tag: str = "") -> str:
        return hashlib.sha256((tag + source).encode("utf-8")).hexdigest()[:16]

    def _compile(self, source: str, *, tag: str = "") -> Path | None:
        """Compile one content-addressed fragment; identical boxes reuse one file."""
        key = self._key(source, tag)
        typ_path = self._work / f"{key}.typ"
        pdf_path = self._work / f"{key}.pdf"
        if pdf_path.exists():
            return pdf_path
        with self._write_lock:
            typ_path.write_text(source, encoding="utf-8")
        ok, _ = typst_compile(str(typ_path), str(pdf_path), self._binary)
        if not ok:
            return None
        return pdf_path

    def _body_markup(self, text: str, runs: tuple[StyledRun, ...] = ()) -> str:
        """The Typst body for a fragment: prose escaped, inline math in math mode.

        Inline ``$...$`` / ``\\(...\\)`` spans the model emitted are converted
        with :func:`typstify_math` and emitted in math mode only when the math
        probe compiles them; a span that does not compile falls back to escaped
        literal text, never worse than the escape-only path. Everything else is
        escaped exactly as before, so a math-free fragment is byte-identical.
        ``runs`` wrap verbatim spans in colour/superscript/weight markup (injected
        after escaping, so the markup itself is not escaped). Cached by
        ``(text, runs)``: one paragraph is measured at several sizes but converted
        once.
        """
        cache_key = (text, runs)
        cached = self._body_cache.get(cache_key)
        if cached is not None:
            return cached
        from ubt.adapters.pdf.overlay_text import prepare_overlay_text, render_overlay_line

        prepared = prepare_overlay_text(text, target_lang=self._target_lang)
        body = render_overlay_line(
            prepared,
            self._math_probe.check,
            target_lang=self._target_lang,
            run_spans=_locate_runs(prepared, runs),
        )
        self._body_cache[cache_key] = body
        return body

    def _measure_source(
        self,
        text: str,
        width_pt: float,
        size_pt: float,
        *,
        kind: str = "text",
        is_bold: bool = False,
        indent_pt: float | None = None,
        runs: tuple[StyledRun, ...] = (),
    ) -> str:
        body = self._body_markup(text, runs)
        font_line = self._font_line
        weight_line = _text_weight_line(kind == "heading" or is_bold)
        return (
            f"#set page(width: {width_pt}pt, height: auto, margin: 0pt)\n"
            f"{_par_line(self._par_leading_em)}\n"
            f'#set text(size: {size_pt}pt{weight_line}, top-edge: "ascender", bottom-edge: "descender"{font_line})\n'
            f"{self._dir_line}"
            f"{_indent_hspace(indent_pt)}{body}\n"
        )

    def _measure_height(
        self,
        text: str,
        width_pt: float,
        size_pt: float,
        *,
        kind: str = "text",
        is_bold: bool = False,
        indent_pt: float | None = None,
        runs: tuple[StyledRun, ...] = (),
    ) -> float:
        # A first-line indent changes wrapping (and so height) only for the
        # indented overlay; it gets its own cache so the shared fit cache keeps
        # its 3-tuple key (and the fit path never passes an indent).
        if indent_pt and indent_pt > 0:
            key: tuple[Any, ...] = (text, width_pt, size_pt, indent_pt, runs)
            cache: dict[Any, float] = self._indent_cache
        elif runs:
            key = (text, width_pt, size_pt, runs)
            cache = self._heading_cache if (kind == "heading" or is_bold) else self._measure_cache
        else:
            key = (text, width_pt, size_pt)
            cache = self._heading_cache if (kind == "heading" or is_bold) else self._measure_cache
        cached = cache.get(key)
        if cached is not None:
            return cached
        height = self._measure_one(
            self._measure_source(
                text, width_pt, size_pt, kind=kind, is_bold=is_bold, indent_pt=indent_pt, runs=runs
            )
        )
        cache[key] = height
        return height

    def _measure_one(self, source: str) -> float:
        """Height of one auto-height measure source; ``inf`` when it cannot build."""
        pdf_path = self._compile(source, tag="measure:")
        if pdf_path is None:
            return float("inf")
        try:
            return float(pdf_struct.page_sizes(pdf_path)[1][1])
        except Exception:
            return float("inf")

    def _batch_measure(
        self,
        items: Sequence[tuple[Any, ...]],
        *,
        build: Callable[..., str] | None = None,
        cache: dict[Any, float] | None = None,
    ) -> None:
        """Height of many items in one Typst invocation.

        A single document renders every not-yet-measured item as an auto-height
        page; the page heights are the natural text heights. Batching is what
        makes a whole document cheap: a Typst invocation costs ~0.7s of startup
        regardless of how many fragments it lays out.

        ``build`` renders one item's measure source (called as ``build(*item)``)
        and ``cache`` stores the heights; both default to the single-text path,
        and the bilingual fit passes its own so one batching implementation
        serves both. The reflow fit passes 6-tuples that carry the indent,
        weight, and kind.
        """
        build = build or self._measure_source
        store = self._measure_cache if cache is None else cache
        todo = [item for item in dict.fromkeys(items) if item not in store]
        if not todo:
            return
        batch = "\n#pagebreak()\n".join(build(*item).rstrip("\n") for item in todo)
        pdf_path = self._compile(batch, tag="measure-batch:")
        heights: list[float] | None = None
        if pdf_path is not None:
            try:
                sizes = pdf_struct.page_sizes(pdf_path)
            except Exception:
                sizes = {}
            if len(sizes) == len(todo):
                heights = [sizes[index][1] for index in range(1, len(todo) + 1)]
        if heights is None:
            # The batch failed (or a page collapsed); measure each item alone so a
            # single bad fragment cannot lose the whole document's fragments.
            for item in todo:
                store[item] = self._measure_one(build(*item))
            return
        for item, height in zip(todo, heights, strict=True):
            store[item] = height

    def _caps_for(self, count: int, max_size_pt: float | Sequence[float] | None) -> list[float]:
        """The per-box starting size cap (the source size, never below the floor).

        A source size below ``_MIN_FONT_PT`` (5pt footnotes) is kept as-is:
        the floor prevents *shrinking* during fit, not inflating small text.
        """
        if isinstance(max_size_pt, (list, tuple)):
            return [float(cap) if cap is not None else self._size_pt for cap in max_size_pt]
        single = max_size_pt if isinstance(max_size_pt, (int, float)) else None
        if single is None:
            return [self._size_pt] * count
        return [float(single)] * count

    def _fit_sizes(
        self,
        items: Sequence[tuple[str, float, float]],
        *,
        kind: str = "text",
        max_size_pt: float | Sequence[float] | None = None,
        is_bold: bool = False,
        indent_pt: float | None = None,
        runs: tuple[StyledRun, ...] = (),
    ) -> list[float | None]:
        """Fit every box at once, one batched measure per correction round.

        Natural height is close to linear in font size, so a proportional step
        lands near the answer and a few rounds absorb the wrapping non-linearity.
        Shrinking rather than clipping keeps the text layer free of ink-less
        lines: a ``clip: true`` box leaves the overflow's text operators in the
        content stream while painting nothing, which the visual gate reads as an
        occluded line.

        ``indent_pt`` is the first-line indent the fragment will be *drawn* with:
        it shortens the first line, so a fit measured without it can wrap one line
        past the box and clip. Indented measures live in their own cache keyed
        with the indent, so an indented and a flush paragraph of the same text
        never share a height.

        The search runs in two stages. The first is the proportional shrink down
        to ``_MIN_FONT_PT`` -- the readable floor, and the size every box that
        *can* read well is drawn at. Only the boxes that still overflow there go
        to the second stage, a bisection in ``[_MIN_DRAW_PT, _MIN_FONT_PT]`` for
        the largest size that fits at all: a source line the extractor split, a
        code stub, a box the translation is a third longer than. Those are drawn
        below the readable floor (the caller flags them ``low_legibility_font``)
        rather than left as source.

        The bisection is what keeps stage two honest. The proportional step is
        not usable below the floor: it scales by ``height / natural``, so a box
        that wraps at 6pt and does not wrap at 5.9pt overshoots straight past the
        answer (6pt -> 4.4pt where 5.9pt would do). Above the floor that overshoot
        was invisible -- the step was clamped at ``_MIN_FONT_PT`` and the next
        round re-measured -- and it must stay invisible, or lowering the floor
        would quietly shrink the boxes that were never a problem.
        """
        results: list[float | None] = [None] * len(items)
        caps = self._caps_for(len(items), max_size_pt)
        sizes: list[float | None] = [
            min(cap, height_pt * 0.85) if width_pt > 0 and height_pt > 0 and text.strip() else None
            for (text, width_pt, height_pt), cap in zip(items, caps, strict=True)
        ]
        initial_sizes: list[float] = [s if s is not None else _MIN_FONT_PT for s in sizes]
        indented = bool(indent_pt and indent_pt > 0)
        cache: dict[Any, float] = (
            self._indent_cache
            if indented
            else (self._heading_cache if (kind == "heading" or is_bold) else self._measure_cache)
        )

        def _measured(text: str, width_pt: float, size_pt: float) -> tuple[Any, ...]:
            if runs:
                return (text, width_pt, size_pt, indent_pt, runs)
            return (text, width_pt, size_pt, indent_pt) if indented else (text, width_pt, size_pt)

        def _build(*item: Any) -> str:
            return self._measure_source(
                item[0],
                item[1],
                item[2],
                kind=kind,
                is_bold=is_bold,
                indent_pt=indent_pt if indented else None,
                runs=runs,
            )

        def _fits(index: int, size_pt: float) -> bool:
            natural = cache[_measured(items[index][0], items[index][1], size_pt)]
            return natural <= items[index][2] + _FIT_TOL

        def _measure_all(pending: Sequence[tuple[int, float]]) -> None:
            self._batch_measure(
                [
                    _measured(items[index][0], items[index][1], size_pt)
                    for index, size_pt in pending
                ],
                build=_build,
                cache=cache,
            )

        active = [index for index, size in enumerate(sizes) if size is not None and size > 0]
        for _ in range(6):
            if not active:
                break
            to_measure: list[tuple[Any, ...]] = []
            for index in active:
                size_pt = narrow(sizes[index], what="fitted font size")
                to_measure.append(_measured(items[index][0], items[index][1], size_pt))
            self._batch_measure(to_measure, build=_build, cache=cache)
            next_active: list[int] = []
            for index in active:
                height_pt = items[index][2]
                size_pt = narrow(sizes[index], what="fitted font size")
                natural = cache[_measured(items[index][0], items[index][1], size_pt)]
                if natural <= height_pt + _FIT_TOL:
                    results[index] = size_pt
                elif size_pt <= _MIN_FONT_PT and size_pt <= initial_sizes[index]:
                    results[index] = None
                else:
                    sizes[index] = max(
                        min(_MIN_FONT_PT, initial_sizes[index]),
                        size_pt * (height_pt / natural) * 0.97,
                    )
                    next_active.append(index)
            active = next_active

        retry = [index for index, size_pt in enumerate(results) if size_pt is None]
        if not retry:
            return results
        # The first stage may have run out of rounds without reaching the floor;
        # measure it outright so a box that does fit at the readable floor keeps
        # it (and its place above the flag).
        _measure_all([(index, _MIN_FONT_PT) for index in retry])
        lo: list[int] = []
        for index in retry:
            if _fits(index, _MIN_FONT_PT):
                results[index] = _MIN_FONT_PT
            else:
                lo.append(index)
        if not lo:
            return results
        # Prove the hard floor out before bisecting between it and the readable
        # one: a box it cannot hold has no drawn size at all and stays source.
        _measure_all([(index, _MIN_DRAW_PT) for index in lo])
        bounds: dict[int, tuple[float, float]] = {}
        for index in lo:
            if _fits(index, _MIN_DRAW_PT):
                results[index] = _MIN_DRAW_PT
                bounds[index] = (_MIN_DRAW_PT, _MIN_FONT_PT)
            else:
                results[index] = None
        for _ in range(5):  # narrows [4, 6] to within 0.0625pt
            if not bounds:
                break
            probes = [(index, (low + high) / 2.0) for index, (low, high) in bounds.items()]
            _measure_all(probes)
            for index, mid in probes:
                low, high = bounds[index]
                if _fits(index, mid):
                    results[index] = mid
                    bounds[index] = (mid, high)
                else:
                    bounds[index] = (low, mid)
        return results

    def _fit_size(
        self,
        text: str,
        width_pt: float,
        height_pt: float,
        *,
        kind: str = "text",
        max_size_pt: float | None = None,
        is_bold: bool = False,
        indent_pt: float | None = None,
        runs: tuple[StyledRun, ...] = (),
    ) -> float | None:
        # Each box draws at the largest size that fits it, capped at its source
        # size (``max_size_pt``). A box that fits at the source size keeps it; only
        # a box the target overflows shrinks. A document-wide shared size was
        # tried and rejected: it dragged every box that *did* fit down to the low
        # percentile of the cramped ones, so a page whose paragraphs did not
        # reflow read as uniformly shrunken next to its neighbours.
        return self._fit_sizes(
            [(text, width_pt, height_pt)],
            kind=kind,
            max_size_pt=max_size_pt,
            is_bold=is_bold,
            indent_pt=indent_pt,
            runs=runs,
        )[0]

    # -- In-place bilingual: target above a smaller, muted source -------------- #

    def _bilingual_key(self, target: str, source: str) -> str:
        """The measure key for a two-text fragment (unit-separator joined)."""
        return f"{target}{_BILINGUAL_SEP}{source}"

    def _bilingual_measure_source(self, key: str, width_pt: float, size_pt: float) -> str:
        target, _, source = key.partition(_BILINGUAL_SEP)
        font_line = self._font_line
        return (
            f"#set page(width: {width_pt}pt, height: auto, margin: 0pt)\n"
            f"#set par(leading: {self._par_leading_em}em)\n"
            f'#set text(size: {size_pt}pt{_WEIGHT_REGULAR}, top-edge: "ascender", bottom-edge: "descender"{font_line})\n'
            f"{self._dir_line}"
            f"{self._body_markup(target)}\n"
            f"#v({_BILINGUAL_GAP_EM}em)\n"
            f'#text(size: {size_pt * _BILINGUAL_RATIO}pt, fill: rgb("{_BILINGUAL_FILL}"))'
            f"[{self._body_markup(source)}]\n"
        )

    def _bilingual_text_source(
        self,
        target: str,
        source: str,
        width_pt: float,
        height_pt: float,
        size_pt: float,
        *,
        align_center: bool = False,
    ) -> str:
        font_line = self._font_line
        return (
            f"#set page(width: {width_pt}pt, height: {height_pt}pt, margin: 0pt)\n"
            f"#set par(leading: {self._par_leading_em}em)\n"
            f'#set text(size: {size_pt}pt{_WEIGHT_REGULAR}, top-edge: "ascender", bottom-edge: "descender"{font_line})\n'
            f"{self._dir_line}"
            f"{_align_line(align_center)}"
            f"#box(width: {width_pt}pt, height: {height_pt}pt, clip: true)[\n"
            f"{self._body_markup(target)}\n"
            f"#v({_BILINGUAL_GAP_EM}em)\n"
            f'#text(size: {size_pt * _BILINGUAL_RATIO}pt, fill: rgb("{_BILINGUAL_FILL}"))'
            f"[{self._body_markup(source)}]\n"
            f"]\n"
        )

    def _fit_bilingual_sizes(
        self, items: Sequence[tuple[str, str, float, float]]
    ) -> list[float | None]:
        """Fit target+source into a box, one batched measure per correction round.

        The primary (target) size is the free variable; the source renders at a
        fixed fraction of it. Same proportional-step search as the monolingual
        fit, sharing the batching, so a whole bilingual book is a handful of
        Typst invocations. Items are ``(target, source, width, height)``.
        """
        results: list[float | None] = [None] * len(items)
        keys = [self._bilingual_key(target, source) for target, source, _, _ in items]
        sizes: list[float | None] = [
            min(self._size_pt, height_pt * 0.82)
            if width_pt > 0 and height_pt > 0 and (target.strip() or source.strip())
            else None
            for target, source, width_pt, height_pt in items
        ]
        initial_sizes: list[float] = [s if s is not None else _MIN_FONT_PT for s in sizes]
        active = [index for index, size in enumerate(sizes) if size is not None and size > 0]
        for _ in range(6):
            if not active:
                break
            to_measure: list[tuple[str, float, float]] = []
            for index in active:
                size_pt = narrow(sizes[index], what="fitted bilingual font size")
                to_measure.append((keys[index], items[index][2], size_pt))
            self._batch_measure(
                to_measure, build=self._bilingual_measure_source, cache=self._bilingual_cache
            )
            next_active: list[int] = []
            for index in active:
                height_pt = items[index][3]
                size_pt = narrow(sizes[index], what="fitted bilingual font size")
                natural = self._bilingual_cache[(keys[index], items[index][2], size_pt)]
                if natural <= height_pt + _FIT_TOL:
                    results[index] = size_pt
                elif size_pt <= _MIN_FONT_PT and size_pt <= initial_sizes[index]:
                    results[index] = None
                else:
                    sizes[index] = max(
                        min(_MIN_FONT_PT, initial_sizes[index]),
                        size_pt * (height_pt / natural) * 0.97,
                    )
                    next_active.append(index)
            active = next_active
        return results

    def fit_bilingual(
        self, target: str, source: str, width_pt: float, height_pt: float
    ) -> float | None:
        """The size ``typeset_bilingual`` would fit this pair at, or None.

        Public so the compositor can size a multi-box run: one paragraph owns
        one size, and it is the smallest fit among the boxes that carry the pair
        (see ``_compile_overlay``).
        """
        return self._fit_bilingual_sizes([(target, source, width_pt, height_pt)])[0]

    def typeset_bilingual(
        self,
        source: str,
        target: str,
        width_pt: float,
        height_pt: float,
        *,
        size_pt: float | None = None,
    ) -> Path | None:
        """Typeset an in-place bilingual fragment: target over a smaller source.

        Both texts share the box; the target is the primary (fitted) size and the
        source a muted fraction of it, so a bilingual page reads top-down target
        then source without one crowding the other out. ``size_pt`` draws the
        pair at exactly that size instead of re-fitting this box, so a
        continuation run's parts share one size (the compositor fits it across
        every box); a per-box re-fit would let one part draw small and its
        neighbour large.
        """
        if width_pt <= 0 or height_pt <= 0 or not (target.strip() or source.strip()):
            return None
        if size_pt is not None:
            self.last_drawn_pt = size_pt
            return self._compile(
                self._bilingual_text_source(target, source, width_pt, height_pt, size_pt),
                tag="bilingual:",
            )
        size_pt = self._fit_bilingual_sizes([(target, source, width_pt, height_pt)])[0]
        if size_pt is not None:
            self.last_drawn_pt = size_pt
            return self._compile(
                self._bilingual_text_source(target, source, width_pt, height_pt, size_pt),
                tag="bilingual:",
            )
        # Target+source will not both fit the box at the readable floor (a short
        # single-line box: a heading, a narrow paragraph). The target is the
        # deliverable and the source echo is best-effort, so drop the echo and
        # typeset the target alone rather than descend to source -- which is why
        # an in-place bilingual page shows headings monolingually. Only when the
        # target alone also misses the floor does the caller keep the source.
        if not target.strip():
            return None
        return self.typeset(target, width_pt, height_pt)

    def _text_source(
        self,
        text: str,
        width_pt: float,
        height_pt: float,
        size_pt: float,
        *,
        kind: str = "text",
        is_bold: bool = False,
        indent_pt: float | None = None,
        align_center: bool = False,
        runs: tuple[StyledRun, ...] = (),
    ) -> str:
        body = self._body_markup(text, runs)
        font_line = self._font_line
        weight_line = _text_weight_line(kind == "heading" or is_bold)
        return (
            f"#set page(width: {width_pt}pt, height: {height_pt}pt, margin: 0pt)\n"
            f"{_par_line(self._par_leading_em)}\n"
            f'#set text(size: {size_pt}pt{weight_line}, top-edge: "ascender", bottom-edge: "descender"{font_line})\n'
            f"{self._dir_line}"
            f"{_align_line(align_center)}"
            f"#box(width: {width_pt}pt, height: {height_pt}pt, clip: true)[{_indent_hspace(indent_pt)}{body}]\n"
        )

    def typeset(
        self,
        text: str,
        width_pt: float,
        height_pt: float,
        *,
        kind: str = "text",
        font_size: float | None = None,
        is_bold: bool = False,
        indent_pt: float | None = None,
        align_center: bool = False,
        runs: tuple[StyledRun, ...] = (),
    ) -> Path | None:
        if width_pt <= 0 or height_pt <= 0 or not text.strip():
            return None
        if font_size is not None and font_size > 0:
            max_size = (
                font_size * 1.05
                if kind == "text"
                else max(font_size, 24.0 if kind == "heading" else font_size)
            )
        else:
            max_size = 24.0 if kind == "heading" else self._size_pt
        size_pt = self._fit_size(
            text,
            width_pt,
            height_pt,
            kind=kind,
            max_size_pt=max_size,
            is_bold=is_bold,
            indent_pt=indent_pt,
            runs=runs,
        )
        if size_pt is None:
            return None
        self.last_drawn_pt = size_pt
        return self._compile(
            self._text_source(
                text,
                width_pt,
                height_pt,
                size_pt,
                kind=kind,
                is_bold=is_bold,
                indent_pt=indent_pt,
                align_center=align_center,
                runs=runs,
            )
        )

    def typeset_fixed(
        self,
        text: str,
        width_pt: float,
        height_pt: float,
        size_pt: float,
        *,
        kind: str = "text",
        is_bold: bool = False,
        indent_pt: float | None = None,
        align_center: bool = False,
        runs: tuple[StyledRun, ...] = (),
    ) -> Path | None:
        """Typeset at an exact size into an exact box (a reflowed overlay).

        The reflow pass already measured the fragment at ``size_pt`` and set the
        box to that natural height, so the fit search would only re-derive (and
        occasionally shrink below) the size it chose. Compiling the source
        directly also matches the string ``prefetch`` warmed, so the cache hits.
        """
        if width_pt <= 0 or height_pt <= 0 or not text.strip() or size_pt <= 0:
            return None
        res = self._compile(
            self._text_source(
                text,
                width_pt,
                height_pt,
                size_pt,
                kind=kind,
                is_bold=is_bold,
                indent_pt=indent_pt,
                align_center=align_center,
                runs=runs,
            )
        )
        if res is not None:
            self.last_drawn_pt = size_pt
        return res

    def measure_fixed(
        self,
        text: str,
        width_pt: float,
        size_pt: float,
        *,
        kind: str = "text",
        is_bold: bool = False,
        indent_pt: float | None = None,
        runs: tuple[StyledRun, ...] = (),
    ) -> float:
        """Natural height of ``text`` at an explicit size and indent."""
        if width_pt <= 0 or not text.strip():
            return 0.0
        return self._measure_height(
            text, width_pt, size_pt, kind=kind, is_bold=is_bold, indent_pt=indent_pt, runs=runs
        )

    def measure_many_fixed(
        self,
        items: Sequence[tuple[str, float, float, float | None, str, bool, tuple[StyledRun, ...]]],
    ) -> list[float]:
        """Natural heights of many reflow items in one batched Typst invocation.

        Items are ``(text, width_pt, size_pt, indent_pt, kind, is_bold, runs)``.
        One auto-height page per item keeps the reflow fit to a single compiler
        call, which is what makes a whole-document reflow affordable.
        """
        keys = [tuple(item) for item in items]
        self._batch_measure(
            keys,
            build=lambda t, w, s, ind, kind, bold, runs: self._measure_source(
                t, w, s, kind=kind, is_bold=bold, indent_pt=ind, runs=runs
            ),
            cache=self._fixed_measure_cache,
        )
        return [self._fixed_measure_cache.get(key, float("inf")) for key in keys]

    def measure(self, text: str, width_pt: float) -> float:
        """Height of the text laid out at ``width_pt`` (Typst ``height: auto``).

        Used by the BreakageSolver to find where a flowed text must break. A
        compile failure returns infinity, i.e. "does not fit any box".
        """
        if width_pt <= 0 or not text.strip():
            return 0.0
        return self._measure_height(text, width_pt, self._size_pt)

    def _toc_source(
        self,
        title: str,
        page: str,
        width_pt: float,
        height_pt: float,
        size_pt: float,
        *,
        is_bold: bool = False,
        has_leaders: bool = True,
    ) -> str:
        from ubt.adapters.pdf.overlay_text import typst_escape

        body = self._body_markup(title)
        weight_line = _text_weight_line(is_bold)
        leader_markup = (
            f"#h(4pt)#box(width: 1fr, repeat(gap: 3.5pt)[.])#h(4pt){typst_escape(page.strip())}"
            if has_leaders
            else f"#h(1fr){typst_escape(page.strip())}"
        )
        return (
            f"#set page(width: {width_pt}pt, height: {height_pt}pt, margin: 0pt)\n"
            f"#set par(leading: {self._par_leading_em}em)\n"
            f'#set text(size: {size_pt}pt{weight_line}, top-edge: "ascender", bottom-edge: "descender"{self._font_line})\n'
            f"#box(width: {width_pt}pt, height: {height_pt}pt, clip: true)["
            f"#box(width: 100%)[{body}{leader_markup}]"
            f"]\n"
        )

    def typeset_toc(
        self,
        title: str,
        page: str,
        width_pt: float,
        height_pt: float,
        *,
        is_bold: bool = False,
        has_leaders: bool = True,
    ) -> Path | None:
        """Typeset one translated TOC row: title, dot leaders, page number.

        Mirrors the retired rigid typesetter's TOC emission. The reader drops
        the source's leader and page-number lines, so the row must be redrawn
        whole or the translation would sit in a title-width slot with a broken
        leader. ``page`` may be empty, in which case only the leaders are drawn.
        """
        if width_pt <= 0 or height_pt <= 0 or not title.strip():
            return None
        size_pt = self._fit_size(
            title, width_pt, height_pt, kind="text", max_size_pt=self._size_pt, is_bold=is_bold
        )
        if size_pt is None:
            return None
        return self._compile(
            self._toc_source(
                title, page, width_pt, height_pt, size_pt, is_bold=is_bold, has_leaders=has_leaders
            ),
            tag="toc:",
        )

    def _math_source(self, latex: str, width_pt: float, height_pt: float) -> str | None:
        if width_pt <= 0 or height_pt <= 0 or not latex.strip():
            return None
        from ubt.adapters.pdf.overlay_text import typstify_math

        converted = typstify_math(_strip_math_delimiters(latex))
        if converted is None:
            return None
        size_pt = max(4.0, min(height_pt * 0.85, 24.0))
        return (
            f"#set page(width: {width_pt}pt, height: {height_pt}pt, margin: 0pt)\n"
            f"#set text(size: {size_pt}pt{_WEIGHT_REGULAR})\n"
            f"#box(width: {width_pt}pt, height: {height_pt}pt, clip: true)[$ {converted} $]\n"
        )

    def typeset_math(self, latex: str, width_pt: float, height_pt: float) -> Path | None:
        """Typeset a formula body as a Typst-math fragment (the design's Layer 2).

        LaTeX is converted with the same ``typstify_math`` the rigid path uses, so
        a formula the extractor protected as LaTeX renders as vector math rather
        than being redrawn as prose. ``None`` means it could not be typeset, and
        the region descends to the source page.
        """
        source = self._math_source(latex, width_pt, height_pt)
        return None if source is None else self._compile(source, tag="math:")

    def _batch_compile(self, pages: Sequence[tuple[str, str]]) -> None:
        """Compile many fragment sources as one Typst document, split by page.

        One invocation lays out every fragment; the pages are then split back
        into the per-fragment PDFs the compositor stamps. A batch failure (or a
        page-count mismatch) falls back to one invocation per fragment so a
        single bad fragment cannot lose all the others.
        """
        pending: list[tuple[str, str, str]] = []
        for source, tag in pages:
            key = self._key(source, tag)
            if not (self._work / f"{key}.pdf").exists():
                pending.append((key, source, tag))
        if not pending:
            return
        batch = "\n#pagebreak()\n".join(source.rstrip("\n") for _, source, _ in pending)
        batch_path = self._compile(batch, tag="fragment-batch:")
        split = False
        if batch_path is not None:
            try:
                with pikepdf.open(batch_path) as pdf:
                    if len(pdf.pages) == len(pending):
                        for index, (key, _, _) in enumerate(pending):
                            one = pikepdf.new()
                            one.pages.append(pdf.pages[index])
                            one.save(str(self._work / f"{key}.pdf"))
                        split = True
            except Exception:
                split = False
        if not split:
            for _, source, tag in pending:
                self._compile(source, tag=tag)

    def _math_bodies(self, texts: Sequence[str]) -> list[str]:
        """The converted Typst math bodies in ``texts``, deduplicated.

        Fitting measures each paragraph through ``_body_markup``, whose math
        probe would otherwise spawn one Typst process per formula. Collecting
        the bodies here lets :meth:`TypstMathProbe.check_many` resolve the whole
        document's math in a handful of compiles before fitting starts.
        """
        from ubt.adapters.pdf.overlay_text import (
            prepare_overlay_text,
            split_math_spans,
            typstify_math_span,
        )

        bodies: list[str] = []
        for text in texts:
            prepared = prepare_overlay_text(text, target_lang=self._target_lang)
            for is_math, content in split_math_spans(prepared):
                if not is_math:
                    continue
                converted = typstify_math_span(content)
                if converted is not None:
                    bodies.append(converted)
        return list(dict.fromkeys(bodies))

    def _cap_for(self, kind: str, font_size: float | None) -> float:
        """The starting size cap for one box, mirroring ``typeset``'s ``max_size``."""
        if font_size is None or font_size <= 0:
            return 24.0 if kind == "heading" else self._size_pt
        if kind == "heading":
            return max(font_size, 24.0)
        return font_size * 1.05

    def cap_size(self, kind: str, font_size: float | None) -> float:
        """The starting size cap for a box: the source size (never below the floor)."""
        return self._cap_for(kind, font_size)

    def prefetch(self, requests: Sequence[tuple[Any, ...]]) -> None:
        """Fit and compile a whole document's fragments in a handful of calls.

        A Typst invocation costs ~0.7s of startup regardless of how many
        fragments it lays out, so one process per fragment dominated wall time.
        This fits every box with batched measure rounds, then renders every
        fragment as a page of one document and splits the pages back out.
        Content addressing keeps it idempotent: a later ``typeset`` for the same
        box finds the split file. A request is ``(kind, text, width_pt,
        height_pt, font_size)``; ``kind == "math"`` selects the math source. A
        non-bilingual text request extends it to ``(..., indent_pt, fixed, is_bold,
        draw_size_pt, align_center, runs)``: ``fixed`` means the box already equals
        the fragment's natural height, so it is compiled at ``draw_size_pt`` (the
        size the reflow measured it at, which is not necessarily this block's own
        cap) instead of being fitted.

        Each box draws at the largest size that fits it, capped at its source
        size, so a box the target does not overflow keeps the source size and only
        a cramped one shrinks. The fitted/``draw_size_pt`` path ignores ``runs``
        and ``align_center`` when *measuring* (they change the drawn markup, not
        the wrap), but the warmed source must still carry them or the draw-time
        content hash differs and the fragment is recompiled one process at a time.
        """
        unique = list(dict.fromkeys(requests))
        if not unique:
            return
        parsed: list[
            tuple[
                str,
                str,
                float,
                float,
                float | None,
                float | None,
                bool,
                bool,
                float | None,
                bool,
                tuple[StyledRun, ...],
            ]
        ] = []
        for request in unique:
            request_kind, text, width_pt, height_pt, font_size, *rest = request
            indent = rest[0] if len(rest) > 0 else None
            fixed = bool(rest[1]) if len(rest) > 1 else False
            is_bold = bool(rest[2]) if len(rest) > 2 else False
            draw_size = rest[3] if len(rest) > 3 else None
            align_center = bool(rest[4]) if len(rest) > 4 else False
            runs: tuple[StyledRun, ...] = rest[5] if len(rest) > 5 else ()
            parsed.append(
                (
                    request_kind,
                    text,
                    width_pt,
                    height_pt,
                    font_size,
                    indent,
                    fixed,
                    is_bold,
                    draw_size,
                    align_center,
                    runs,
                )
            )
        # Resolve every inline-math body up front so fitting's measure calls hit
        # the probe cache instead of spawning a Typst process per formula.
        self._math_probe.check_many(self._math_bodies([text for _kind, text, *_rest in unique]))
        pages: list[tuple[str, str]] = []
        groups: dict[
            tuple[str, float | None, bool],
            list[tuple[str, float, float, float | None, bool, tuple[StyledRun, ...]]],
        ] = {}
        # Fitted boxes: one size per (kind, indent, weight) class so the warmed
        # fragment is the one the draw asks for.
        for (
            request_kind,
            text,
            width_pt,
            height_pt,
            font_size,
            indent,
            fixed,
            is_bold,
            _draw,
            align_center,
            runs,
        ) in parsed:
            if fixed or request_kind not in ("text", "heading"):
                continue
            groups.setdefault((request_kind, indent, is_bold), []).append(
                (text, width_pt, height_pt, font_size, align_center, runs)
            )
        for (kind, indent, is_bold), items in groups.items():
            triples = [(text, width_pt, height_pt) for text, width_pt, height_pt, *_rest in items]
            caps = [self._cap_for(kind, font_size) for _, _, _, font_size, *_rest in items]
            sizes = self._fit_sizes(
                triples, kind=kind, max_size_pt=caps, is_bold=is_bold, indent_pt=indent
            )
            for (text, width_pt, height_pt, _font_size, align_center, runs), size_pt in zip(
                items, sizes, strict=True
            ):
                if size_pt is None:
                    continue
                pages.append(
                    (
                        self._text_source(
                            text,
                            width_pt,
                            height_pt,
                            size_pt,
                            kind=kind,
                            is_bold=is_bold,
                            indent_pt=indent,
                            align_center=align_center,
                            runs=runs,
                        ),
                        "",
                    )
                )

        # Reflowed boxes: exact size, exact height, no fit search.
        for (
            request_kind,
            text,
            width_pt,
            height_pt,
            font_size,
            indent,
            fixed,
            is_bold,
            draw_size,
            align_center,
            runs,
        ) in parsed:
            if not fixed or request_kind not in ("text", "heading"):
                continue
            size_pt = draw_size if draw_size is not None else self._cap_for(request_kind, font_size)
            pages.append(
                (
                    self._text_source(
                        text,
                        width_pt,
                        height_pt,
                        size_pt,
                        kind=request_kind,
                        is_bold=is_bold,
                        indent_pt=indent,
                        align_center=align_center,
                        runs=runs,
                    ),
                    "",
                )
            )

        bilingual_items = [
            (*text.partition(_BILINGUAL_SEP)[::2], width_pt, height_pt)
            for kind, text, width_pt, height_pt, *_rest in unique
            if kind == "bilingual"
        ]
        for (target, source, width_pt, height_pt), size_pt in zip(
            bilingual_items, self._fit_bilingual_sizes(bilingual_items), strict=True
        ):
            if size_pt is not None:
                pages.append(
                    (
                        self._bilingual_text_source(target, source, width_pt, height_pt, size_pt),
                        "bilingual:",
                    )
                )
        for kind, text, width_pt, height_pt, *_rest in unique:
            if kind != "math":
                continue
            math_source = self._math_source(text, width_pt, height_pt)
            if math_source is not None:
                pages.append((math_source, "math:"))
        self._batch_compile(pages)

    def close(self) -> None:
        if getattr(self, "_is_temp", False):
            work = getattr(self, "_work", None)
            if work is not None:
                shutil.rmtree(work, ignore_errors=True)

    def __del__(self) -> None:
        self.close()


#: Historical name for the typesetter surface. ``TypstFragmentTypesetter`` is
#: the production implementation; tests provide duck-typed spies.
FragmentTypesetter = Any


def _dedup_identical_streams(pdf: pikepdf.Pdf) -> None:
    """Point every reference at one copy of a byte-identical stream.

    Each fragment is compiled and copied independently, so a document-wide shared
    resource -- the CJK font subset, a CMap -- is embedded once per fragment. The
    artifact is correct but grows to tens of MB. Streams with the same content
    and the same type are interchangeable, so the duplicates are collapsed; qpdf
    drops the unreferenced copies on write.
    """
    canonical: dict[tuple[str, str, str, str], pikepdf.Object] = {}
    duplicate: dict[tuple[int, int], pikepdf.Object] = {}
    for obj in pdf.objects:
        if not isinstance(obj, pikepdf.Stream):
            continue
        raw = obj.read_raw_bytes()
        if len(raw) < _DEDUP_MIN_BYTES:
            continue
        key = (
            hashlib.sha256(raw).hexdigest(),
            str(obj.get("/Type", "")),
            str(obj.get("/Subtype", "")),
            str(obj.get("/Length1", "")),
        )
        if key in canonical:
            duplicate[obj.objgen] = canonical[key]
        else:
            canonical[key] = obj
    if not duplicate:
        return

    seen: set[tuple[int, int]] = set()

    def rewrite(node: pikepdf.Object) -> None:
        objgen = getattr(node, "objgen", (0, 0))
        if objgen != (0, 0):
            if objgen in seen:
                return
            seen.add(objgen)
        if isinstance(node, pikepdf.Dictionary):
            for name in list(node.keys()):
                value = node[name]
                target = getattr(value, "objgen", (0, 0))
                if target != (0, 0) and target in duplicate:
                    node[name] = duplicate[target]
                else:
                    rewrite(value)
        elif isinstance(node, pikepdf.Array):
            for index, value in enumerate(node):
                target = getattr(value, "objgen", (0, 0))
                if target != (0, 0) and target in duplicate:
                    node[index] = duplicate[target]
                else:
                    rewrite(value)

    for obj in pdf.objects:
        if isinstance(obj, (pikepdf.Dictionary, pikepdf.Array)) and obj.objgen != (0, 0):
            rewrite(obj)


def _clamp_page_boxes(
    boxes: Sequence[PhysicalBox], page_bounds: Mapping[int, tuple[float, float, float, float]]
) -> tuple[PhysicalBox, ...]:
    """Intersect each box with its page mediabox; drop one the clamp collapses.

    Extraction boxes can spill past the page edge by a point or two, and drawing
    a fragment there trips the visual gate's ``block_out_of_bounds``. The clamp
    uses the origin-carrying mediabox (``pdf_struct.page_box``): a page of
    ``[10 10 610 810]`` has the same 600x800 size as ``[0 0 600 800]`` but its
    visible right edge sits at x=610, and clamping to the size alone silently
    dropped every overlay near that edge.
    """
    clamped: list[PhysicalBox] = []
    for box in boxes:
        bounds = page_bounds.get(box.page)
        if bounds is None:
            continue
        x0, y0, x1, y1 = box.bbox
        px0, py0, px1, py1 = bounds
        cx0, cx1 = max(px0, min(x0, px1)), max(px0, min(x1, px1))
        cy0, cy1 = max(py0, min(y0, py1)), max(py0, min(y1, py1))
        if cx1 <= cx0 or cy1 <= cy0:
            continue
        clamped.append(PhysicalBox.of(box.page, (cx0, cy0, cx1, cy1)))
    return tuple(clamped)


@dataclass(frozen=True, slots=True)
class _StampedPart:
    """One compiled fragment waiting to be stamped onto its page."""

    overlay: Overlay
    page_no: int
    bbox: BBox
    #: The fragment to draw, or ``None`` for a mask-only part: a continuation box
    #: the (shorter) target never reached. The box still holds the source tail of
    #: the paragraph being replaced, so it must be erased even though it draws
    #: nothing.
    form: pikepdf.Object | None
    #: The source box to mask. ``bbox`` is expanded below by the line slack so the
    #: fragment fits at the source size, but the mask must not grow into the next
    #: line, so it stays at the extracted box.
    mask_bbox: BBox | None = None
    #: The size (pt) the target in ``form`` was drawn at, when the compile fitted
    #: it (``None`` for a mask-only part or a non-fitted kind). Carried up to the
    #: placement so a below-floor draw becomes a ``low_legibility_font`` defect.
    drawn_pt: float | None = None


class LayerCompositor:
    """Compose pages from L0 source + L1 mask + L2 fragment (prototype).

    Not the default engine. ``typesetter=None`` composes nothing above the floor
    (Layer 0 only, identical to :func:`compose`); supplying a typesetter enables
    L1+L2 for the overlays it can typeset. A region whose fragment fails to
    typeset is left as source -- the mask is painted only after the fragment is
    in hand, so a mask can never hide content that was not replaced.
    """

    name: ClassVar[str] = "layer-compositor"

    def __init__(
        self,
        source_pdf: str | Path,
        *,
        typesetter: FragmentTypesetter | None = None,
        background: tuple[float, float, float] | None = None,
        strip: bool = True,
    ) -> None:
        # ``background`` pins the fill for every micro-mask; ``None`` (the
        # default) samples each masked region from the page instead. A flat
        # white default left a bright scar on every document not printed on
        # white -- cream stock, a tinted band, a shaded row.
        self._source = Path(source_pdf)
        self._typesetter = typesetter
        self._background = background
        # Strip the source text under a region before masking it. A mask alone
        # leaves the source text in the text layer (extractable, and the visual
        # gate flags it as "text_occluded"), so the default removes it; a region
        # whose strip aborts or hits a page-shared form descends instead of being
        # overlaid (never double the text).
        self._strip = strip

    def compose(self, overlays: Sequence[Overlay], output_path: str | Path) -> Composition:
        output = Path(output_path)
        # Layer 0 is the source document itself. Opening it as the working
        # document -- instead of a blank PDF we copy pages into -- keeps the
        # catalog (named destinations, outlines, metadata, AcroForm) that the
        # source's internal GoTo links resolve against; a blank shell left every
        # relocated citation link pointing at a destination that does not
        # exist. The pages are the canvas, so the non-text layers stay intact.
        with pdf_struct.open_pdf(self._source) as composed:
            page_bounds = {
                index: pdf_struct.page_box(page)
                for index, page in enumerate(composed.pages, start=1)
            }
            shared_forms = shared_form_objgens(composed) if self._strip else set()
            # Clamp each region to its page once, then compile every fragment
            # before any strip: the per-fragment subprocess compiles are the whole
            # cost of a large document.
            resolved: list[tuple[Overlay, tuple[PhysicalBox, ...] | None]] = []
            prepared: list[tuple[Overlay, tuple[PhysicalBox, ...]]] = []
            for overlay in overlays:
                boxes = _clamp_page_boxes(overlay.flow_boxes, page_bounds)
                resolved.append((overlay, boxes or None))
                if boxes:
                    prepared.append((overlay, boxes))
            self._prefetch(prepared)
            # Layer 2 is typeset first and Layer 1 strips *once per page*, before
            # any overlay is stamped: a per-overlay strip recursed into the Forms
            # of overlays drawn earlier on the same page and erased them whenever
            # their boxes overlapped.
            stamped: dict[int, list[_StampedPart]] = {}
            space_failed_ids: set[int] = set()
            for overlay, boxes in prepared:
                parts, space_failed = self._compile_overlay(composed, overlay, boxes)
                if space_failed:
                    space_failed_ids.add(id(overlay))
                for item in parts:
                    stamped.setdefault(item.page_no, []).append(item)
            drawn_ids: set[int] = set()
            for page_no in sorted(stamped):
                page = composed.pages[page_no - 1]
                if self._stamp_page(
                    composed, page, page_no, stamped[page_no], shared_forms=shared_forms
                ):
                    drawn_ids.update(id(item.overlay) for item in stamped[page_no])
            # A multi-box overlay draws one fragment per box; the smallest fitted
            # size is the one that decides whether the element is reading-grade,
            # so the placement carries the minimum across its drawn parts.
            drawn_pt_by_id: dict[int, float] = {}
            for items in stamped.values():
                for item in items:
                    if item.form is None or item.drawn_pt is None:
                        continue
                    key = id(item.overlay)
                    drawn_pt_by_id[key] = min(drawn_pt_by_id.get(key, item.drawn_pt), item.drawn_pt)
            placements = tuple(
                self._placement(
                    overlay,
                    boxes,
                    drawn=id(overlay) in drawn_ids,
                    drawn_pt=drawn_pt_by_id.get(id(overlay)),
                    space_failed=id(overlay) in space_failed_ids,
                )
                for overlay, boxes in resolved
            )
            # Each fragment carries its own copy of the shared font/CMap; collapse
            # the duplicates before writing so the artifact is not tens of MB.
            _dedup_identical_streams(composed)
            with atomic_write_path(output) as tmp_path:
                composed.save(str(tmp_path))
        return Composition(output_path=output, placements=placements)

    def _prefetch(self, prepared: Sequence[tuple[Overlay, tuple[PhysicalBox, ...]]]) -> None:
        """Warm the fragment cache for every single-box overlay, concurrently.

        Multi-box overlays are skipped: the flow solver splits their text at draw
        time and its measurement calls drive their own compiles.
        """
        prefetch = getattr(self._typesetter, "prefetch", None)
        if prefetch is None:
            return
        requests: list[tuple[Any, ...]] = []
        for overlay, boxes in prepared:
            if len(boxes) != 1:
                continue
            # A TOC row's fragment depends on its page number, which the prefetch
            # request tuple does not carry; it compiles on demand instead.
            if overlay.kind == "toc":
                continue
            width = boxes[0].bbox[2] - boxes[0].bbox[0]
            # A reflowed box already equals the fragment's natural height, so the
            # line slack (which exists to defeat the ink-box shrink) must not be
            # added; the prefetched source must match the draw-time one.
            slack = 0.0 if overlay.fixed_box else _line_slack(overlay.font_size)
            height = boxes[0].bbox[3] - boxes[0].bbox[1] + slack
            if overlay.source:
                requests.append(
                    (
                        "bilingual",
                        bilingual_request_text(overlay.text, overlay.source),
                        width,
                        height,
                        overlay.font_size,
                    )
                )
            else:
                requests.append(
                    (
                        overlay.kind,
                        overlay.text,
                        width,
                        height,
                        overlay.font_size,
                        overlay.indent_pt,
                        overlay.fixed_box,
                        overlay.is_bold,
                        overlay.draw_size_pt,
                        # The draw-time source carries these too; omitting them made
                        # every styled-run (or centered) fragment miss the cache and
                        # spawn its own Typst process at draw time.
                        overlay.align_center,
                        overlay.runs,
                    )
                )
        prefetch(requests)

    def _flow_plan(
        self,
        overlay: Overlay,
        boxes: tuple[PhysicalBox, ...],
        *,
        text: str | None = None,
    ) -> tuple[tuple[FlowPlacement, ...], float | None]:
        """Split a continuation paragraph across its boxes at ONE consistent size.

        A paragraph that crosses a page/column boundary owns several boxes. The
        whole target is measured at the size it would draw at; if it does not fit
        the boxes' total capacity, the size is scaled down so the *whole*
        paragraph shrinks uniformly -- never one box tiny and its neighbour
        normal. The chosen size is returned so every part draws at exactly it,
        instead of each box re-fitting (and re-shrinking) on its own.
        """
        typesetter = narrow(self._typesetter, what="typesetter for measure")
        body = overlay.text if text is None else text
        # Styled runs describe the *target* text; the source echo has its own
        # shaping, so measuring it with the target's runs would mis-attribute.
        runs = overlay.runs if text is None else ()
        cap = getattr(typesetter, "cap_size", None)
        measure_fixed = getattr(typesetter, "measure_fixed", None)
        if cap is None or measure_fixed is None:
            return solve_flow(body, boxes, typesetter.measure), None
        base = cap(overlay.kind, overlay.font_size)

        # The drawn fragment gets the extracted box *plus* this much height (see
        # ``_LINE_SLACK_RATIO``: the ink box stops at the last line's descender and
        # omits the leading).
        slack = 0.0 if overlay.fixed_box else _line_slack(overlay.font_size)

        def _make_measure(size: float, *, with_slack: bool) -> Callable[[str, float], float]:
            def measure(token: str, width: float) -> float:
                height: float = measure_fixed(
                    token,
                    width,
                    size,
                    kind=overlay.kind,
                    is_bold=overlay.is_bold,
                    indent_pt=overlay.indent_pt,
                    runs=runs,
                )
                return max(0.0, height - slack) if with_slack else height

            return measure

        # Size against the *widest* box: a continuation run repeats one column
        # width (first == max, so this changes nothing there), while a
        # heading's own line chain can mix a narrow first line with a
        # full-width rest — measuring at the narrow first box would inflate
        # the height and shrink the whole heading. The per-box cuts in
        # ``solve_flow`` handle the narrower boxes; the final box receives its
        # remainder at this width, so the widest box is the honest estimate of
        # whether everything fits the capacity below.
        width = max(box.bbox[2] - box.bbox[0] for box in boxes)
        # Match ``solve_flow``'s own capacity (the bare box height, without the
        # line slack the draw adds): sizing against the slack-inflated capacity
        # would let the flow overflow the last box, which then clips its text.
        capacity = sum(box.bbox[3] - box.bbox[1] for box in boxes)
        size = base
        height = measure_fixed(
            body,
            width,
            size,
            kind=overlay.kind,
            is_bold=overlay.is_bold,
            indent_pt=overlay.indent_pt,
            runs=runs,
        )
        if capacity > 0 and height > capacity:
            size = max(min(_MIN_FONT_PT, base), size * capacity / height)
        # The bare ink box is the flow's capacity; a box that would otherwise come
        # out blank falls back to the draw box's (see ``solve_flow``), so the
        # source line it replaces is never erased with nothing drawn over it.
        parts = solve_flow(
            body,
            boxes,
            _make_measure(size, with_slack=False),
            fallback_measure=_make_measure(size, with_slack=True),
        )
        return self._shrink_until_the_flow_fits(
            overlay, body, boxes, parts, size, measure_fixed, _make_measure
        )

    def _shrink_until_the_flow_fits(
        self,
        overlay: Overlay,
        body: str,
        boxes: tuple[PhysicalBox, ...],
        parts: tuple[FlowPlacement, ...],
        size: float,
        measure_fixed: Callable[..., float],
        measure_factory: Callable[..., Callable[[str, float], float]],
    ) -> tuple[tuple[FlowPlacement, ...], float]:
        """Shrink a continuation run until every box actually holds its share.

        The proportional estimate above sizes the *whole* text against the widest
        box and the chain's total height, which assumes the text can be packed
        optimally. It cannot: ``solve_flow`` breaks at punctuation, and a
        candidate that lands well short of a box's capacity pushes the surplus
        into the next box, which then overflows and is clipped -- leaving an
        ink-less line in the text layer (the visual gate's "text_occluded").
        Measuring the flow that was actually produced and shrinking while any box
        overflows keeps every line printed; a box's own line slack counts as its
        capacity because that is what the draw box gets.
        """
        slack = 0.0 if overlay.fixed_box else _line_slack(overlay.font_size)

        def _worst_overflow(current: float, placements: Sequence[FlowPlacement]) -> float:
            worst = 1.0
            for part in placements:
                if not part.text.strip():
                    continue
                width = part.box.bbox[2] - part.box.bbox[0]
                available = (part.box.bbox[3] - part.box.bbox[1]) + slack
                if available <= 0:
                    continue
                needed = measure_fixed(
                    part.text,
                    width,
                    current,
                    kind=overlay.kind,
                    is_bold=overlay.is_bold,
                    indent_pt=overlay.indent_pt,
                )
                if needed <= available + _FIT_TOL:
                    continue
                worst = max(worst, needed / available)
            return worst

        def _flow(current: float) -> tuple[FlowPlacement, ...]:
            return solve_flow(
                body,
                boxes,
                measure_factory(current, with_slack=False),
                fallback_measure=measure_factory(current, with_slack=True),
            )

        parts = _flow(size)
        if _worst_overflow(size, parts) <= 1.0:
            return parts, size
        # The estimate overflowed, so bisect on the size: smaller text fits more
        # into every box, so "the flow fits" is monotone enough for a few rounds.
        low = min(_MIN_FONT_PT, size)
        low_parts = _flow(low)
        if _worst_overflow(low, low_parts) > 1.0:
            # Even the readable floor overflows: drawing it would clip lines
            # into an ink-less text layer (the visual gate's "text_occluded").
            # Empty parts leave the source unmasked -- the element descends.
            return (), low
        high = size
        for _ in range(6):
            mid = (low + high) / 2.0
            if mid - low <= 0.01:
                break
            mid_parts = _flow(mid)
            if _worst_overflow(mid, mid_parts) <= 1.0:
                low, low_parts = mid, mid_parts
            else:
                high = mid
        return low_parts, low

    def _compile_overlay(
        self, composed: pikepdf.Pdf, overlay: Overlay, boxes: tuple[PhysicalBox, ...]
    ) -> tuple[list[_StampedPart], bool]:
        """Compile every box's fragment for one overlay.

        Returns ``(drawable parts, space_failed)``. ``space_failed`` is True only
        for the one provable Axiom-B failure: the flow solver could not place the
        translation in a multi-box run even at the readable floor, so the source
        ships because the target does not *fit*. A fragment that merely fails to
        compile is a different case (a typesetter defect), so it stays False and
        the contract keeps its default WARNING severity.
        """
        typesetter = self._typesetter
        if typesetter is None or not (overlay.text.strip() or overlay.source.strip()):
            return [], False
        slack = 0.0 if overlay.fixed_box else _line_slack(overlay.font_size)
        draw_size: float | None = None
        space_failed = False
        if len(boxes) == 1:
            parts: tuple[FlowPlacement, ...] = (FlowPlacement(boxes[0], overlay.text),)
        else:
            parts, draw_size = self._flow_plan(overlay, boxes)
            # ``_flow_plan`` returns an empty plan exactly when the readable
            # floor still overflows every box: the translation existed and could
            # not be placed. That is the space failure the delivery contract must
            # read as an error, not the silent WARNING it was.
            space_failed = not parts
        # In-place bilingual: flow the source through the same boxes and pair it
        # with the target part sharing each box, so a multi-box paragraph keeps
        # both languages rather than repeating one in every box.
        source_by_box: dict[tuple[int, BBox], str] = {}
        if overlay.source.strip():
            source_parts = (
                (FlowPlacement(boxes[0], overlay.source),)
                if len(boxes) == 1
                else self._flow_plan(overlay, boxes, text=overlay.source)[0]
            )
            if not source_parts and len(boxes) > 1:
                # The source echo (target is shorter, source overflowed at the
                # floor). ``_flow_plan`` returned empty to signal "descend", but
                # the target IS drawn, so dropping the echo silently would break
                # the bilingual pair. Fall back to a greedy fill so the source
                # still appears (the last box clips its tail).
                echo_typesetter = narrow(
                    self._typesetter, what="typesetter for source-echo fallback"
                )
                source_parts = tuple(solve_flow(overlay.source, boxes, echo_typesetter.measure))
            source_by_box = {(part.box.page, part.box.bbox): part.text for part in source_parts}
        stamped: list[_StampedPart] = []
        multi_box = len(boxes) > 1
        # A bilingual run draws at ONE size across its boxes. The pair (target +
        # muted source echo) needs ~1.8x the height of the target alone, so a
        # per-box re-fit drew box 1 at 6.0pt and box 2 at 8.2pt for the same
        # paragraph -- the "one box tiny and its neighbour normal" split
        # ``_flow_plan`` exists to prevent. The run size is the smallest pair
        # fit among the boxes that carry an echo (never above ``draw_size``, the
        # size the target parts were cut at), and every part draws its target at
        # exactly it. A target-carrying box whose pair still misses the readable
        # floor drops its echo and draws the target alone -- the same echo-drop
        # the single-box path documents. A stub typesetter without
        # ``fit_bilingual`` keeps the legacy per-box behavior rather than lose
        # every echo it cannot measure.
        run_bilingual = False
        run_size = draw_size
        pair_fits: dict[tuple[int, BBox], float | None] = {}
        fit_bilingual = getattr(typesetter, "fit_bilingual", None)
        if (
            multi_box
            and overlay.source.strip()
            and draw_size is not None
            and fit_bilingual is not None
        ):
            run_bilingual = True
            for part in parts:
                part_source = source_by_box.get((part.box.page, part.box.bbox), "")
                if not part_source.strip():
                    # No echo here: the target draws alone at the run size, and
                    # a target-alone fit would only over-constrain the run.
                    continue
                x0, y0, x1, y1 = part.box.bbox
                pair_fits[(part.box.page, part.box.bbox)] = fit_bilingual(
                    part.text, part_source, x1 - x0, y1 - y0 + slack
                )
            feasible = [size for size in pair_fits.values() if size is not None]
            if feasible:
                run_size = min(min(feasible), draw_size)
        for index, part in enumerate(parts):
            key = (part.box.page, part.box.bbox)
            source = source_by_box.get(key, "")
            # Mask the *source* geometry: a reflowed overlay draws at its own box
            # but the source text still sits where the block was, so the mask
            # follows ``mask_boxes`` rather than the draw box.
            mask_bbox = (
                overlay.mask_boxes[index].bbox if index < len(overlay.mask_boxes) else part.box.bbox
            )
            if run_bilingual and source.strip() and pair_fits.get(key) is None:
                if not part.text.strip():
                    # An echo-only box (the target flow ended) whose echo cannot
                    # fit even at the floor: leave the source untouched rather
                    # than painting a mask with no drawn replacement.
                    continue
                # Drop the echo (best-effort) and draw the target alone, still
                # at the run size so the paragraph stays one size.
                source = ""
            if not part.text.strip() and not source.strip():
                # A continuation box the (shorter) target did not reach. Its
                # source still sits there -- the tail of the very paragraph we
                # re-rendered in the earlier boxes -- so erase it even though
                # this box draws nothing. Skipping it leaves that tail (e.g. a
                # word spilled to the next page) as stray source text.
                if multi_box:
                    stamped.append(
                        _StampedPart(
                            overlay, part.box.page, part.box.bbox, None, mask_bbox=mask_bbox
                        )
                    )
                continue
            x0, y0, x1, y1 = part.box.bbox
            compiled = self._compile_form(
                composed,
                overlay,
                part,
                source,
                y1 - y0 + slack,
                exact_size=run_size,
                bilingual_size=run_size if run_bilingual and source.strip() else None,
                # Only the paragraph's OWN first line is indented: a continuation
                # box starts mid-paragraph, and indenting there hangs the page's
                # first line as if a new paragraph began at the page break.
                first_part=index == 0,
            )
            if compiled is not None:
                form, drawn_pt = compiled
                draw_bbox = (x0, y0 - slack, x1, y1)
                stamped.append(
                    _StampedPart(
                        overlay,
                        part.box.page,
                        draw_bbox,
                        form,
                        mask_bbox=mask_bbox,
                        drawn_pt=drawn_pt,
                    )
                )
        return stamped, space_failed

    def _compile_form(
        self,
        composed: pikepdf.Pdf,
        overlay: Overlay,
        part: FlowPlacement,
        source: str,
        height: float,
        *,
        exact_size: float | None = None,
        bilingual_size: float | None = None,
        first_part: bool = True,
    ) -> tuple[pikepdf.Object, float | None] | None:
        """Typeset one flowed part and copy it into the artifact as a Form.

        Returns ``(form, drawn_pt)``: the copied Form and the size (pt) the
        fragment was fitted to, or ``None`` for ``drawn_pt`` when the branch
        drew at an already-decided size (reflowed, continuation, bilingual run)
        or a non-fitted kind (math, TOC). ``drawn_pt`` below ``_MIN_FONT_PT`` is
        what the caller turns into a ``low_legibility_font`` defect.

        ``height`` is the draw height: the extracted box plus the line slack, so
        the fragment can be fitted (and drawn) at the source size rather than
        shrunk to the ink box. ``exact_size`` (a continuation run's uniform size)
        draws the part at that size instead of re-fitting it per box, and
        ``bilingual_size`` (a bilingual run's uniform pair size, possibly below
        ``exact_size``) draws the target+source pair at exactly it -- the pair
        branch runs before ``exact_size``, so a per-box re-fit there would
        override the run's one size. ``first_part`` carries the paragraph's
        first-line indent to the box it actually starts in (never to a
        continuation box).
        """
        typesetter = self._typesetter
        if typesetter is None:
            return None
        # A fitted compile publishes its size on this side channel; clear it so a
        # branch that draws without fitting cannot report the previous part's.
        if hasattr(typesetter, "last_drawn_pt"):
            typesetter.last_drawn_pt = None
        x0, _y0, x1, _y1 = part.box.bbox
        width = x1 - x0
        indent_pt = overlay.indent_pt if first_part else None
        if width <= 0 or height <= 0:
            return None
        bilingual = getattr(typesetter, "typeset_bilingual", None)
        toc_typeset = getattr(typesetter, "typeset_toc", None)
        fixed_typeset = getattr(typesetter, "typeset_fixed", None)
        fragment: Path | None
        if overlay.kind == "math":
            fragment = typesetter.typeset_math(part.text, width, height)
        elif overlay.kind == "toc" and toc_typeset is not None:
            fragment = toc_typeset(
                part.text,
                overlay.toc_page,
                width,
                height,
                is_bold=overlay.is_bold,
                has_leaders=overlay.toc_leaders,
            )
        elif source.strip() and bilingual is not None:
            try:
                fragment = bilingual(
                    source,
                    part.text,
                    width,
                    height,
                    align_center=overlay.align_center,
                    size_pt=bilingual_size,
                )
            except TypeError:
                try:
                    # A custom typesetter without the centering flag: fall back
                    # to the positional signature rather than lose the overlay.
                    fragment = bilingual(source, part.text, width, height, size_pt=bilingual_size)
                except TypeError:
                    fragment = bilingual(source, part.text, width, height)
        elif overlay.fixed_box or exact_size is not None:
            # Reflowed (fixed_box) or a continuation part (exact_size): the size is
            # already decided, so compile at exactly it -- no per-box fit shrink.
            cap = getattr(typesetter, "cap_size", None)
            size_pt = exact_size if exact_size is not None else overlay.draw_size_pt
            if size_pt is None and cap is not None:
                size_pt = cap(overlay.kind, overlay.font_size)
            if size_pt is None:
                size_pt = overlay.font_size if overlay.font_size and overlay.font_size > 0 else 10.0
            if fixed_typeset is not None:
                try:
                    fragment = fixed_typeset(
                        part.text,
                        width,
                        height,
                        size_pt,
                        kind=overlay.kind,
                        is_bold=overlay.is_bold,
                        indent_pt=indent_pt,
                        align_center=overlay.align_center,
                        runs=overlay.runs,
                    )
                except TypeError:
                    try:
                        fragment = fixed_typeset(
                            part.text,
                            width,
                            height,
                            size_pt,
                            kind=overlay.kind,
                            is_bold=overlay.is_bold,
                            indent_pt=indent_pt,
                            runs=overlay.runs,
                        )
                    except TypeError:
                        fragment = fixed_typeset(
                            part.text,
                            width,
                            height,
                            size_pt,
                            kind=overlay.kind,
                            is_bold=overlay.is_bold,
                            indent_pt=indent_pt,
                        )
                if fragment is not None and hasattr(typesetter, "last_drawn_pt"):
                    typesetter.last_drawn_pt = size_pt
            else:
                fragment = typesetter.typeset(
                    part.text,
                    width,
                    height,
                    kind=overlay.kind,
                    font_size=overlay.font_size,
                    is_bold=overlay.is_bold,
                )
        else:
            try:
                fragment = typesetter.typeset(
                    part.text,
                    width,
                    height,
                    kind=overlay.kind,
                    font_size=overlay.font_size,
                    is_bold=overlay.is_bold,
                    indent_pt=indent_pt,
                    align_center=overlay.align_center,
                    runs=overlay.runs,
                )
            except TypeError:
                try:
                    fragment = typesetter.typeset(
                        part.text,
                        width,
                        height,
                        kind=overlay.kind,
                        font_size=overlay.font_size,
                        is_bold=overlay.is_bold,
                        runs=overlay.runs,
                    )
                except TypeError:
                    try:
                        fragment = typesetter.typeset(part.text, width, height, kind=overlay.kind)
                    except TypeError:
                        fragment = typesetter.typeset(part.text, width, height)
        if fragment is None:
            return None
        try:
            with pikepdf.open(fragment) as frag:
                if not frag.pages:
                    return None
                form = composed.copy_foreign(frag.pages[0].as_form_xobject())
        except Exception:  # a bad fragment must fall back to source, never crash the run
            return None
        return form, getattr(typesetter, "last_drawn_pt", None)

    def _mask_colours(
        self, page_no: int, rects: Sequence[BBox]
    ) -> list[tuple[float, float, float]]:
        """The fill for each micro-mask rect.

        A caller-pinned ``background`` wins; with none, each rect is sampled
        from the source page (see :mod:`ubt.render.page_background`), and a
        region that cannot be sampled falls back to white -- an approximation
        of the page is always better than source text left legible under the
        translation.
        """
        if self._background is not None:
            return [self._background] * len(rects)
        if not rects:
            return []
        from ubt.render.page_background import (  # noqa: PLC0415
            DEFAULT_MASK_BACKGROUND,
            sample_backgrounds,
        )

        return [
            DEFAULT_MASK_BACKGROUND if sample is None else sample
            for sample in sample_backgrounds(self._source, page_no, rects)
        ]

    def _stamp_page(
        self,
        composed: pikepdf.Pdf,
        page: pikepdf.Page,
        page_no: int,
        items: Sequence[_StampedPart],
        *,
        shared_forms: set[tuple[int, int]],
    ) -> bool:
        apply_micro_mask = False
        if self._strip:
            stats = strip_page_text_pikepdf(
                page,
                [item.mask_bbox or item.bbox for item in items],
                protected_rects=[],
                page_no=page_no,
                shared_forms=shared_forms,
            )
            # A strip that could not remove the source text (a page-shared
            # form left intact, or a form whose resources were unreadable or
            # failed to recurse so its text survives): fall back to defensive
            # micro-masking instead of dropping the page back to untranslated source.
            if stats.aborted or stats.shared_forms_skipped or stats.forms_survived:
                apply_micro_mask = True
        else:
            apply_micro_mask = True

        if apply_micro_mask:
            mask_rects = [item.mask_bbox or item.bbox for item in items]
            rect_ops = []
            colours = self._mask_colours(page_no, mask_rects)
            for mask_rect, colour in zip(mask_rects, colours, strict=True):
                x0, y0, x1, y1 = mask_rect
                red, green, blue = colour
                rect_ops.append(
                    f"{red} {green} {blue} rg {x0:.2f} {y0:.2f} {x1 - x0:.2f} {y1 - y0:.2f} re f"
                )
            if rect_ops:
                # One stream, one colour operator per rect: the masks still
                # coalesce into a single graphics stream, but a page whose
                # regions sit on different backgrounds keeps each region's.
                # The comment is a marker, not decoration: a reader of the
                # artifact (forensics, a test) can tell a fallback mask from
                # the page's own fill, which can look identical.
                combined_mask = f"q % ubt-micro-mask\n{' '.join(rect_ops)} Q".encode("ascii")
                page.contents_add(pikepdf.Stream(composed, combined_mask), prepend=False)
        media = [float(v) for v in page.MediaBox]
        mb_x0, mb_y0, mb_x1, mb_y1 = media[0], media[1], media[2], media[3]
        valid_items = [item for item in items if item.form is not None]
        if not valid_items:
            return True
        if len(valid_items) == 1:
            item = valid_items[0]
            form = narrow(item.form, what="fragment form")
            x0, y0, x1, y1 = item.bbox
            page.add_overlay(
                form,
                pikepdf.Rectangle(max(x0, mb_x0), max(y0, mb_y0), min(x1, mb_x1), min(y1, mb_y1)),
            )
        else:
            # Batch all fragment forms for this page into a single page-level Form XObject.
            # This collapses N separate Form XObjects and repeated stream coalescing into one,
            # drastically reducing PDF resource dictionary bloat and viewer decoding overhead.
            xobjects = pikepdf.Dictionary()
            cs_parts: list[bytes] = []
            for idx, item in enumerate(valid_items):
                form = narrow(item.form, what="fragment form")
                x0, y0, x1, y1 = item.bbox
                rect = pikepdf.Rectangle(
                    max(x0, mb_x0), max(y0, mb_y0), min(x1, mb_x1), min(y1, mb_y1)
                )
                name = pikepdf.Name(f"/Fm{idx}")
                xobjects[name] = form
                cs = page.calc_form_xobject_placement(
                    form,
                    name,
                    rect,
                    invert_transformations=True,
                    allow_shrink=True,
                    allow_expand=False,
                )
                cs_parts.append(cs)
            page_form = pikepdf.Stream(composed, b"".join(cs_parts))
            page_form[pikepdf.Name.Type] = pikepdf.Name.XObject
            page_form[pikepdf.Name.Subtype] = pikepdf.Name.Form
            page_form[pikepdf.Name.BBox] = pikepdf.Array([mb_x0, mb_y0, mb_x1, mb_y1])
            page_form[pikepdf.Name.Resources] = pikepdf.Dictionary(XObject=xobjects)
            page.add_overlay(page_form, pikepdf.Rectangle(mb_x0, mb_y0, mb_x1, mb_y1))
        return True

    @staticmethod
    def _placement(
        overlay: Overlay,
        boxes: tuple[PhysicalBox, ...] | None,
        *,
        drawn: bool,
        drawn_pt: float | None = None,
        space_failed: bool = False,
    ) -> Placement:
        if not boxes:
            return Placement(overlay.element_id, overlay.page, False, "no usable box; source kept")
        if drawn:
            detail = "layer-compositor"
        elif space_failed:
            # The one source-kept reason the delivery contract must read as an
            # Axiom-B error: the translation existed but does not *fit* the
            # boxes even at the readable floor. ``no_fit`` is a
            # ``_SPACE_FAILURE_MARKERS`` token, so the reason reaches
            # ``_is_space_failure`` and the violation escalates from WARNING to
            # ERROR. A generic "no fragment" (a typesetter defect) deliberately
            # does NOT carry a marker and stays a warning.
            detail = "no_fit: source kept (translation does not fit)"
        else:
            detail = "no fragment; source kept"
        return Placement(
            overlay.element_id,
            overlay.page,
            drawn,
            detail,
            drawn_pt=drawn_pt if drawn else None,
        )


def _overlayable(block: IRBlock) -> bool:
    """Whether this block should be drawn above the source (not kept opaque)."""
    box = block.bbox
    if block.skip_translate or box is None or box.page <= 0:
        return False
    # Tables and figures are canvas assets, not flowing text: the compositor
    # cannot reconstruct a grid or a graphic, so they always stay on Layer 0.
    if block.block_type in (BlockType.TABLE, BlockType.IMAGE):
        return False
    return bool((block.target_text or "").strip())


def overlays_from_blocks(
    blocks: Sequence[IRBlock],
    *,
    bilingual: bool = False,
) -> tuple[Overlay, ...]:
    """Build overlays from IR blocks, merging continuation runs into box chains.

    Prose blocks that read as one element (a paragraph crossing a page/column
    boundary) become a *single* overlay whose ``boxes`` are the chain and whose
    text is their targets joined, so the compositor flows it across the boxes.
    Formulas become ``math`` overlays (typeset by the micro-core's math path).
    Everything below the fidelity floor or without geometry is left to Layer 0.

    ``bilingual`` carries each text overlay's source alongside its target, for
    in-place bilingual composition (the compositor draws target over source).
    """
    materialized = bifurcate_blocks(list(blocks))
    runs = {run.block_ids[0]: run for run in find_continuation_runs(materialized)}
    by_id = {block.id: block for block in materialized}
    consumed: set[str] = set()
    overlays: list[Overlay] = []
    # Index single-box overlays by (page, bbox, kind) to merge in-box bifurcated siblings
    by_loc: dict[tuple[int, tuple[float, float, float, float], str], int] = {}
    for block in materialized:
        if block.id in consumed:
            continue
        run = runs.get(block.id)
        if run is not None and all(_overlayable(by_id[block_id]) for block_id in run.block_ids):
            run_blocks = [by_id[block_id] for block_id in run.block_ids]
            # A run reads as one element, so only its first block can carry a
            # list marker (a wrapped list item continues without a second bullet).
            text = join_continuous_text(
                [
                    _with_list_marker(run_block, run_block.target_text or "")
                    if index == 0
                    else (run_block.target_text or "")
                    for index, run_block in enumerate(run_blocks)
                ]
            )
            source = (
                join_continuous_text(
                    [
                        _with_list_marker(run_block, run_block.source_text or "")
                        if index == 0
                        else (run_block.source_text or "")
                        for index, run_block in enumerate(run_blocks)
                    ]
                )
                if bilingual
                else ""
            )
            blk_style = by_id[block.id].style
            font_size = (blk_style.font_size if blk_style is not None else None) or by_id[
                block.id
            ].provenance.font_size
            if font_size is not None and font_size < 4.5:
                font_size = None
            is_bold = bool(
                by_id[block.id].provenance.is_bold
                or (by_id[block.id].block_type == BlockType.HEADING)
            )
            first = run.boxes[0]
            overlays.append(
                Overlay(
                    block.id,
                    first.page,
                    first.bbox,
                    text,
                    boxes=run.boxes,
                    source=source,
                    font_size=font_size,
                    is_bold=is_bold,
                    runs=_styled_runs(by_id[block.id]),
                )
            )
            consumed.update(run.block_ids)
            continue
        if _overlayable(block):
            box = narrow(block.bbox, what=f"overlayable block {block.id!r} bbox")
            if block.provenance.toc_entry:
                # A translated table-of-contents row: the compositor redraws the
                # leaders and the page number the reader stripped.
                kind = "toc"
            elif block.block_type == BlockType.FORMULA:
                kind = "math"
            elif block.block_type == BlockType.HEADING:
                kind = "heading"
            else:
                kind = "text"
            source = (
                _with_list_marker(block, (block.source_text or "").strip())
                if bilingual and kind == "text"
                else ""
            )
            span = block.element.span
            boxes = span.boxes if isinstance(span, CompositeSpan) else ()
            bbox_tuple = (box.x0, box.y0, box.x1, box.y1)
            loc_key = (box.page, bbox_tuple, kind)
            if not boxes and loc_key in by_loc:
                idx = by_loc[loc_key]
                existing = overlays[idx]
                target_chunk = _with_list_marker(block, (block.target_text or "").strip())
                new_text = (
                    f"{existing.text}\n\n{target_chunk}"
                    if existing.text and target_chunk
                    else (existing.text or target_chunk)
                )
                new_source = (
                    f"{existing.source}\n\n{source}"
                    if existing.source and source
                    else (existing.source or source)
                )
                overlays[idx] = replace(existing, text=new_text, source=new_source)
                consumed.add(block.id)
                continue
            cur_style = block.style
            font_size = (
                cur_style.font_size if cur_style is not None else None
            ) or block.provenance.font_size
            if font_size is not None and font_size < 4.5:
                font_size = None
            is_bold = bool(block.provenance.is_bold or (block.block_type == BlockType.HEADING))
            # A first-line indent is a body-paragraph or list-marker attribute;
            # a heading carries alignment instead — a centered title is
            # re-centered within its box, not indented.
            if kind == "text" and block.block_type in (
                BlockType.NARRATIVE,
                BlockType.LIST_ITEM,
            ):
                indent_pt = getattr(block.style, "first_line_indent_pt", None)
            else:
                indent_pt = None
            align_center = kind == "heading" and (
                getattr(block.style, "alignment", None) == "center"
            )
            overlay = Overlay(
                block.id,
                box.page,
                bbox_tuple,
                _with_list_marker(block, (block.target_text or "").strip()),
                boxes=boxes,
                kind=kind,
                source=source,
                toc_page=str(block.provenance.toc_page or ""),
                toc_leaders=True
                if block.provenance.toc_leaders is None
                else bool(block.provenance.toc_leaders),
                font_size=font_size,
                is_bold=is_bold,
                indent_pt=indent_pt,
                align_center=align_center,
                runs=_styled_runs(block),
            )
            overlays.append(overlay)
            if not boxes:
                by_loc[loc_key] = len(overlays) - 1
        consumed.add(block.id)
    return tuple(overlays)


__all__ = [
    "Composition",
    "FragmentTypesetter",
    "LayerCompositor",
    "Overlay",
    "Placement",
    "TypstFragmentTypesetter",
    "overlays_from_blocks",
]
