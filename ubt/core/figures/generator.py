"""Code generators for multi-panel publication figures in Typst and HTML."""

from __future__ import annotations

import html
from collections.abc import Callable

from ubt.core.figures.models import FigureLayoutConfig, FigurePanel


def _esc_typst(text: str) -> str:
    """Basic escape for Typst reserved characters when no renderer callback is passed."""
    return (
        text.replace("\\", "\\\\")
        .replace("[", r"\[")
        .replace("]", r"\]")
        .replace("#", r"\#")
        .replace("$", r"\$")
    )


def render_figure_grid_typst(
    panel: FigurePanel,
    asset_resolver: Callable[[str], str] | None = None,
    prose_renderer: Callable[[str], str] | None = None,
    config: FigureLayoutConfig | None = None,
) -> str:
    """Render a FigurePanel into publication-ready Typst #figure(grid(...)) markup."""
    _ = config or FigureLayoutConfig()
    if not panel.subfigures:
        return ""

    render_p = prose_renderer or _esc_typst
    resolve_a = asset_resolver or (lambda a: a)

    # Determine columns representation
    cols = max(1, panel.columns)
    col_spec = f"({', '.join(['1fr'] * cols)})" if cols > 1 else "(1fr)"

    # Build subfigure cells
    cells: list[str] = []
    for sf in panel.subfigures:
        resolved_src = resolve_a(sf.asset)
        if resolved_src.startswith("[missing") or not resolved_src:
            img_code = (
                f"#rect(width: 100%, height: 60pt, stroke: 0.5pt + luma(180))"
                f"[#align(center + horizon)[#text(fill: luma(140), size: 8pt)[{_esc_typst(resolved_src or 'missing')}]]]"
            )
        else:
            img_code = f'#image("{resolved_src}", width: 100%)'

        sub_lines = [
            "    [",
            f"      #align(center + horizon)[{img_code}]",
        ]
        label_text = sf.label.strip()
        caption_text = sf.caption.strip()

        if label_text or caption_text:
            sub_lines.append("      #v(2pt)")
            parts: list[str] = []
            if label_text:
                parts.append(f"*{_esc_typst(label_text)}*")
            if caption_text:
                parts.append(render_p(caption_text))
            sub_lines.append(f"      #align(center)[#text(size: 8.5pt)[{' '.join(parts)}]]")

        sub_lines.append("    ],")
        cells.append("\n".join(sub_lines))

    cells_markup = "\n".join(cells)
    grid_markup = (
        f"  grid(\n    columns: {col_spec},\n    gutter: {panel.gutter_pt}pt,\n{cells_markup}\n  )"
    )

    anchor_suffix = f" <{panel.anchor_id}>" if panel.anchor_id else ""
    scope_arg = ', scope: "parent", placement: auto' if panel.span_full_width else ""

    if panel.main_caption:
        rendered_main = render_p(panel.main_caption)
        return (
            f"#figure(\n"
            f"{grid_markup},\n"
            f"  caption: [{rendered_main}],\n"
            f"  numbering: none{scope_arg},\n"
            f"){anchor_suffix}"
        )

    return f"#figure(\n{grid_markup},\n  numbering: none{scope_arg},\n){anchor_suffix}"


def render_figure_grid_html(
    panel: FigurePanel,
    asset_resolver: Callable[[str], str] | None = None,
    config: FigureLayoutConfig | None = None,
) -> str:
    """Render a FigurePanel into a semantic HTML5 <figure> container with CSS Grid."""
    _ = config or FigureLayoutConfig()
    if not panel.subfigures:
        return ""

    resolve_a = asset_resolver or (lambda a: a)
    cols = max(1, panel.columns)
    anchor_attr = f' id="{html.escape(panel.anchor_id)}"' if panel.anchor_id else ""

    sub_elements: list[str] = []
    for sf in panel.subfigures:
        src = html.escape(resolve_a(sf.asset))
        lbl = html.escape(sf.label.strip())
        cap = html.escape(sf.caption.strip())
        alt_text = f"{lbl} {cap}".strip() or "Figure illustration"

        caption_html = ""
        if lbl or cap:
            lbl_html = f"<strong>{lbl}</strong> " if lbl else ""
            caption_html = (
                f'    <figcaption class="ubt-subcaption" '
                f'style="font-size: 0.85em; margin-top: 4pt; color: #555;">'
                f"{lbl_html}{cap}"
                f"</figcaption>\n"
            )

        sub_elements.append(
            f'  <div class="ubt-subfigure" style="display: flex; flex-direction: column; align-items: center; text-align: center;">\n'
            f'    <img src="{src}" alt="{alt_text}" style="max-width: 100%; height: auto;" />\n'
            f"{caption_html}"
            f"  </div>"
        )

    subs_markup = "\n".join(sub_elements)
    main_caption_html = ""
    if panel.main_caption:
        main_text = html.escape(panel.main_caption)
        main_caption_html = (
            f'\n  <figcaption class="ubt-main-caption" '
            f'style="grid-column: 1 / -1; text-align: center; font-size: 0.95em; margin-top: 8pt; font-weight: 500;">'
            f"{main_text}"
            f"</figcaption>"
        )

    return (
        f'<figure class="ubt-figure-grid"{anchor_attr} '
        f'style="display: grid; grid-template-columns: repeat({cols}, 1fr); gap: {panel.gutter_pt}pt; margin: 1.5em 0;">\n'
        f"{subs_markup}"
        f"{main_caption_html}\n"
        f"</figure>"
    )
