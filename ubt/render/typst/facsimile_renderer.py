"""1:1 Facsimile (Canvas) Typst Document Renderer.

Renders translated documents with 100% visual and structural isomorphism:
- Background: text-stripped vector PDF (bg.pdf) carrying original diagrams, lines, and colors
- Text: typeset with Typst over original bounding box coordinates using `#place(top + left, dx, dy)`
- Typography: page-level uniform font size and dynamic leading from layout_solver.py

A placed block is drawn at the *page's* uniform body size unless it carries its
own ``font_size_pt``. That exception is what makes a heading look like a heading:
the page solves for one body size, so without it a 16.2pt chapter title was
drawn at ~10pt and the delivered page read as a flat wall of uniform text.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ubt.core.exceptions import DocumentParseError
from ubt.render.typst.escape import typst_escape
from ubt.render.typst.page_locked_renderer import PageLockedPageSpec, PageLockedReflowRenderer

logger = logging.getLogger(__name__)

DEFAULT_FONT_STACK: tuple[str, ...] = (
    "Noto Serif CJK SC",
    "Source Han Serif SC",
    "Songti SC",
    "SimSun",
)

#: Font weights the renderer will pass through to Typst verbatim. Anything else
#: falls back to the block's own bold/regular classification rather than being
#: interpolated into the markup unchecked (a measured ``"semibold condensed"``
#: would have produced an invalid ``weight:`` value and failed the whole page).
_KNOWN_WEIGHTS: frozenset[str] = frozenset({"regular", "medium", "semibold", "bold", "black"})


@dataclass(frozen=True, slots=True)
class FacsimileBlockPlacement:
    """A single placed block on a facsimile page."""

    block_id: str
    page: int
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    is_heading: bool = False
    heading_level: int = 1
    text_color_hex: str = "#000000"
    font_weight: str = "regular"
    align: str = "justify"
    skip_render: bool = False
    #: The block's own measured size, in points. ``None`` means "use the page's
    #: uniform body size". A heading needs this: the page solves for the body.
    font_size_pt: float | None = None
    #: The block is a Markdown table grid and must be typeset as ``#table``,
    #: not escaped and printed as literal pipe text.
    is_table: bool = False
    #: The block is a code listing and must be typeset in a monospace face with
    #: its line structure intact.
    is_code: bool = False


@dataclass(slots=True)
class FacsimilePageSpec:
    """One page's full facsimile configuration."""

    page_no: int
    page_w: float
    page_h: float
    uniform_size_pt: float
    leading_em: float
    tracking_em: float = 0.0
    placements: list[FacsimileBlockPlacement] = None  # type: ignore[assignment]
    tier: str = "L1"  # "L1" (canvas) or "L2" (page-locked reflow)
    l2_spec: PageLockedPageSpec | None = None

    def __post_init__(self) -> None:
        if self.placements is None:
            self.placements = []


def _normalise_weight(font_weight: str, *, is_heading: bool, bold: bool) -> str:
    """The Typst ``weight`` value for a block.

    ``font_weight`` is what the extractor measured; ``bold`` is the block's own
    classification, used when the measurement is not a value Typst accepts.

    The previous expression was ``"bold" if font_weight in ("regular", "")``,
    which inverted the sense: every heading whose measured weight was regular
    was emitted bold, and a genuinely bold one was passed through only by luck.
    """
    candidate = (font_weight or "").strip().lower()
    if candidate in _KNOWN_WEIGHTS:
        return candidate
    if is_heading or bold:
        return "bold"
    return "regular"


class FacsimileRenderer:
    """Produces and compiles 1:1 facsimile Typst documents."""

    def __init__(self, font_stack: Sequence[str] | None = None) -> None:
        self.font_stack = tuple(font_stack) if font_stack else DEFAULT_FONT_STACK

    def _placement_content(
        self,
        pl: FacsimileBlockPlacement,
        *,
        width_pt: float,
        target_lang: str,
    ) -> str:
        """Typst markup for one placement's text, sized and styled by its block type."""
        if pl.is_table:
            from ubt.render.typst.document_renderer import render_table

            table_markup = render_table(
                pl.text,
                page_columns=1,
                content_width_pt=width_pt,
                target_lang=target_lang,
            )
            if table_markup:
                return table_markup
            # An unparseable grid degrades to escaped text rather than vanishing:
            # a table the compiler cannot lay out is still content the reader
            # needs, and an empty block would silently drop a whole exhibit.
            logger.warning(
                "Facsimile: block %s looked like a table but did not parse; delivering it as text",
                pl.block_id,
            )

        if pl.is_code:
            from ubt.render.typst.container_repair import repair_code_block_markup

            return repair_code_block_markup(pl.text.strip(), container_width_pt=width_pt)

        escaped = typst_escape(pl.text.strip())
        size = f"size: {pl.font_size_pt}pt, " if pl.font_size_pt else ""

        if pl.is_heading:
            weight = _normalise_weight(pl.font_weight, is_heading=True, bold=True)
            return (
                f"#set par(justify: false)\n"
                f'    #text({size}weight: "{weight}", '
                f'fill: rgb("{pl.text_color_hex}"))[{escaped}]'
            )

        if size or pl.font_weight not in ("", "regular"):
            weight = _normalise_weight(pl.font_weight, is_heading=False, bold=False)
            fill = f'fill: rgb("{pl.text_color_hex}"), ' if pl.text_color_hex != "#000000" else ""
            return f'#text({size}{fill}weight: "{weight}")[{escaped}]'
        return escaped

    def build_markup(
        self,
        pages: Sequence[FacsimilePageSpec],
        bg_pdf_filename: str,
        *,
        target_lang: str = "zh",
    ) -> str:
        """Generate Typst markup placing translated blocks on top of the stripped background."""
        fonts_formatted = ", ".join(f'"{f}"' for f in self.font_stack)
        lines: list[str] = [
            "// Universal Book Translator - 1:1 Facsimile Canvas Document",
            f'#set text(font: ({fonts_formatted}), lang: "{target_lang}", overhang: true, cjk-latin-spacing: auto)',
            "",
        ]

        l2_renderer = PageLockedReflowRenderer(font_stack=self.font_stack)
        for p in pages:
            if p.tier == "L2" and p.l2_spec is not None:
                lines.append(l2_renderer.build_page_markup(p.l2_spec, target_lang=target_lang))
                continue

            lines.append(
                f"#page(width: {p.page_w}pt, height: {p.page_h}pt, margin: 0pt, "
                f'background: image("{bg_pdf_filename}", page: {p.page_no}, width: 100%, height: 100%))['
            )
            lines.append(f"  #set text(size: {p.uniform_size_pt}pt)")
            lines.append(
                f'  #set par(justify: true, linebreaks: "optimized", leading: {p.leading_em}em)'
            )

            for pl in p.placements:
                if pl.skip_render or not pl.text.strip():
                    continue

                dx = round(pl.x0, 2)
                dy = round(p.page_h - pl.y1, 2)
                width = round(max(10.0, pl.x1 - pl.x0), 2)

                content = self._placement_content(pl, width_pt=width, target_lang=target_lang)

                lines.append(
                    f"  #place(top + left, dx: {dx}pt, dy: {dy}pt, block(width: {width}pt)[\n"
                    f"    {content}\n"
                    f"  ])"
                )

            lines.append("]")
            lines.append("")

        return "\n".join(lines)

    def render_document(
        self,
        pages: Sequence[FacsimilePageSpec],
        bg_pdf_path: Path,
        output_path: Path,
        build_dir: Path,
        *,
        target_lang: str = "zh",
        compile_fn: Callable[[str, str, str], tuple[bool, str]] | None = None,
    ) -> Path:
        """Compile the facsimile pages into the final output PDF."""
        build_dir.mkdir(parents=True, exist_ok=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        local_bg = build_dir / "bg_stripped.pdf"
        if bg_pdf_path.resolve() != local_bg.resolve():
            shutil.copyfile(bg_pdf_path, local_bg)

        markup = self.build_markup(pages, local_bg.name, target_lang=target_lang)
        typ_path = build_dir / "facsimile.typ"
        typ_path.write_text(markup, encoding="utf-8")

        #: One callable type for both branches. Annotating ``compiler`` from
        #: scratch (rather than letting the ``if``/``else`` infer it) matters:
        #: without it mypy pins the variable to the nested function's def type
        #: and then rejects the caller's ``Callable``, which is the same shape.
        compiler: Callable[[str, str, str], tuple[bool, str]]
        if compile_fn is None:
            from ubt.adapters.pdf.typst_compile import typst_compile

            def _default_compile(src: str, dst: str, lang: str) -> tuple[bool, str]:
                return typst_compile(src, dst, target_lang=lang)

            compiler = _default_compile
        else:
            compiler = compile_fn

        ok, stderr = compiler(str(typ_path), str(output_path), target_lang)
        if not ok:
            raise DocumentParseError(
                f"Facsimile compile failed: {stderr.strip() or 'unknown Typst error'} "
                f"(source kept at {typ_path})"
            )
        return output_path


__all__ = [
    "DEFAULT_FONT_STACK",
    "FacsimileBlockPlacement",
    "FacsimilePageSpec",
    "FacsimileRenderer",
]
