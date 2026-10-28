"""1:1 Facsimile (Canvas) Typst Document Renderer.

Renders translated documents with 100% visual and structural isomorphism:
- Background: text-stripped vector PDF (bg.pdf) carrying original diagrams, lines, and colors
- Text: typeset with Typst over original bounding box coordinates using `#place(top + left, dx, dy)`
- Typography: page-level uniform font size and dynamic leading from layout_solver.py
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ubt.core.exceptions import DocumentParseError
from ubt.render.typst.escape import typst_escape

logger = logging.getLogger(__name__)

DEFAULT_FONT_STACK: tuple[str, ...] = (
    "Noto Serif CJK SC",
    "Source Han Serif SC",
    "Songti SC",
    "SimSun",
)


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

    def __post_init__(self) -> None:
        if self.placements is None:
            self.placements = []


class FacsimileRenderer:
    """Produces and compiles 1:1 facsimile Typst documents."""

    def __init__(self, font_stack: Sequence[str] | None = None) -> None:
        self.font_stack = tuple(font_stack) if font_stack else DEFAULT_FONT_STACK

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

        for p in pages:
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
                escaped = typst_escape(pl.text.strip())

                if pl.is_heading:
                    weight = "bold" if pl.font_weight in ("regular", "") else pl.font_weight
                    content = (
                        f"#set par(justify: false)\n"
                        f'    #text(weight: "{weight}", fill: rgb("{pl.text_color_hex}"))[{escaped}]'
                    )
                else:
                    content = escaped

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
