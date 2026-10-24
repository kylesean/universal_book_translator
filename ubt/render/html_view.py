"""Semantic HTML lowering: the delivered blocks as a web view.

It lowers the run's blocks to semantic HTML, using each block's delivered text:
the translation where one was placed, the source slice where the block was kept.
A block's HTML tag follows its typed element, so a heading becomes ``<h2>``, a
list item ``<li>``, and so on. Styled runs (``StyleMeta.inline_runs`` /
``target_runs``) are re-applied as ``<strong>``/``<em>``/``<sup>``/colour
spans -- see :func:`render_fragment`; a paragraph is text-only only when its
source had no recorded styles.
"""

from __future__ import annotations

import base64
import html
import mimetypes
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from ubt.core.atomic import atomic_write_text
from ubt.core.ir.models import IRBlock
from ubt.model.ast import (
    Caption,
    CodeBlock,
    Dialogue,
    Element,
    Figure,
    Formula,
    Heading,
    ListItem,
    Paragraph,
    Table,
)
from ubt.render.delivered_text import (
    delivered_text,
    dropped_page_furniture,
    is_any_mark,
    is_toc_entry,
    unwrap_mark,
)
from ubt.render.inline_spans import SpanStyle, block_runs, styled_chunks
from ubt.render.outputs import Composition, Placement

_TAIL = "</main>\n</body>\n</html>\n"

_MODERN_CSS = """
:root {
  --bg: #ffffff;
  --text: #1f2937;
  --muted: #64748b;
  --border: #e2e8f0;
  --accent: #2563eb;
  --card-bg: #f8fafc;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f172a;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --border: #334155;
    --accent: #3b82f6;
    --card-bg: #1e293b;
  }
}
/* Explicit theme wins over the OS preference so the toolbar toggle works. */
:root[data-theme="dark"] {
  --bg: #0f172a;
  --text: #f1f5f9;
  --muted: #94a3b8;
  --border: #334155;
  --accent: #3b82f6;
  --card-bg: #1e293b;
}
:root[data-theme="light"] {
  --bg: #ffffff;
  --text: #1f2937;
  --muted: #64748b;
  --border: #e2e8f0;
  --accent: #2563eb;
  --card-bg: #f8fafc;
}
.katex-display { overflow-x: auto; overflow-y: hidden; padding: 0.25em 0; }
.katex { font-size: 1.05em; }
body {
  margin: 0;
  padding: 2.5rem 1.5rem;
  background-color: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Source Han Serif SC", "Noto Serif CJK SC", serif;
  font-size: 1.05rem;
  line-height: 1.8;
}
.ubt-article-container {
  max-width: 860px;
  margin: 0 auto;
  transition: max-width 0.2s ease;
}
h1, h2, h3, h4, h5, h6 {
  color: var(--text);
  font-weight: 700;
  line-height: 1.35;
  margin-top: 2em;
  margin-bottom: 0.6em;
}
h1 { font-size: 2.1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.6rem; }
h2 { font-size: 1.55rem; }
h3 { font-size: 1.25rem; }
p { margin-top: 0; margin-bottom: 1.25em; text-align: justify; }
table {
  width: 100%;
  border-collapse: collapse;
  margin: 1.8em 0;
  font-size: 0.95rem;
  background-color: var(--card-bg);
  border-radius: 6px;
  overflow: hidden;
  box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
th, td {
  padding: 0.75rem 1rem;
  border-bottom: 1px solid var(--border);
  text-align: left;
}
th {
  font-weight: 600;
  border-bottom: 2px solid var(--border);
}
pre {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.2rem;
  overflow-x: auto;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 0.9rem;
  line-height: 1.5;
}
figure {
  margin: 2.5em 0;
  text-align: center;
}
figure img {
  max-width: 100%;
  height: auto;
  border-radius: 6px;
  box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
}
figcaption {
  font-size: 0.9rem;
  color: var(--muted);
  margin-top: 0.6em;
}
.formula {
  font-family: "KaTeX_Main", "Times New Roman", serif;
  display: inline-block;
}
/* Reading toolbar */
.ubt-toolbar {
  position: sticky;
  top: 1rem;
  display: flex;
  justify-content: flex-end;
  gap: 0.5rem;
  margin-bottom: 2rem;
  z-index: 100;
}
.ubt-btn {
  background: var(--card-bg);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 0.35rem 0.8rem;
  border-radius: 6px;
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.15s ease;
}
.ubt-btn:hover, .ubt-btn.active {
  background: var(--accent);
  color: #ffffff;
  border-color: var(--accent);
}
/* View modes */
body.mode-stacked .bilingual-pair .source-row {
  display: block;
  color: var(--muted);
  font-size: 0.92em;
  font-style: italic;
  margin-bottom: 0.35em;
}
body.mode-split .ubt-article-container {
  max-width: 1400px;
}
body.mode-split .bilingual-pair {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2.5rem;
  margin-bottom: 1.5em;
}
body.mode-target .source-row {
  display: none;
}
"""


#: Resolves a block's ``asset_id`` to an absolute path (``None`` when the file
#: is not there). The EPUB view supplies one that copies the bytes into the
#: package and returns the in-package href; the HTML view uses the default,
#: which inlines the bytes as a data URI so the single file stays portable.
AssetResolver = Callable[[str], str | None]


def _data_uri(path: Path) -> str:
    """Inline an asset as a ``data:`` URI (self-contained HTML file)."""
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _math_body_for_katex(text: str) -> str:
    """The bare LaTeX body of a formula, with every delimiter form removed.

    ``strip_math_delimiters`` intentionally keeps single ``$``; KaTeX is given
    its own ``$$`` wrapper by the caller, so any surviving delimiter would
    double up (``$$$…$$$``). Strip ``$$``/``$``/``\\(``/``\\)``/``\\[``/``\\]``
    here so the emitted content is exactly one delimited math span.
    """
    body = text.strip()
    for opener, closer in (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$")):
        if (
            len(body) > len(opener) + len(closer)
            and body.startswith(opener)
            and body.endswith(closer)
        ):
            body = body[len(opener) : -len(closer)].strip()
            break
    return body


def _default_asset_resolver(asset: str) -> str | None:
    """Resolve an asset path relative to the CWD; inline it as a data URI."""
    if not asset:
        return None
    candidate = Path(asset)
    if not candidate.is_absolute() or not candidate.exists():
        return None
    return _data_uri(candidate)


def document_head(
    lang: str = "en",
    direction: str = "ltr",
    title: str = "UBT translation",
    bilingual: bool = False,
) -> str:
    """The modern ``<html>`` head for a view, carrying typography, KaTeX and view modes."""
    dir_attr = f' dir="{html.escape(direction)}"' if direction and direction != "ltr" else ""

    # KaTeX renders every ``.math-tex`` span (the formula fallback markup below)
    # into real typeset math, with the LaTeX source kept for copy/zoom. The
    # script is loaded from the same CDN as the stylesheet; the document still
    # degrades to the readable Unicode fallback text if the CDN is unreachable.
    katex_script = (
        '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" crossorigin="anonymous">\n'
        '<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js" crossorigin="anonymous"></script>\n'
        '<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" crossorigin="anonymous"\n'
        '  onload="renderUbtMath()"></script>\n'
    )

    theme_script = (
        "<script>\n"
        "function setTheme(theme) {\n"
        "  document.documentElement.setAttribute('data-theme', theme);\n"
        "  try { localStorage.setItem('ubt_theme', theme); } catch(e){}\n"
        "  var b = document.getElementById('ubt-theme-btn');\n"
        "  if (b) b.textContent = (theme === 'dark') ? '浅色' : '深色';\n"
        "}\n"
        "function toggleTheme() {\n"
        "  var cur = document.documentElement.getAttribute('data-theme');\n"
        "  if (!cur) { cur = (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light'; }\n"
        "  setTheme(cur === 'dark' ? 'light' : 'dark');\n"
        "}\n"
        "function renderUbtMath() {\n"
        "  if (typeof renderMathInElement !== 'function') return;\n"
        "  document.querySelectorAll('.math-tex').forEach(function(el){\n"
        "    try { renderMathInElement(el, { delimiters: [\n"
        "      {left: '$$', right: '$$', display: true},\n"
        "      {left: '$', right: '$', display: false},\n"
        "      {left: '\\\\(', right: '\\\\)', display: false},\n"
        "      {left: '\\\\[', right: '\\\\]', display: true}\n"
        "    ], throwOnError: false }); } catch(e){}\n"
        "  });\n"
        "}\n"
        "(function(){\n"
        "  try {\n"
        "    var savedTheme = localStorage.getItem('ubt_theme');\n"
        "    if (savedTheme) setTheme(savedTheme);\n"
        "  } catch(e){}\n"
        "  document.addEventListener('DOMContentLoaded', renderUbtMath);\n"
        "})();\n"
        "</script>\n"
    )

    toolbar = ""
    if bilingual:
        toolbar = (
            '<div class="ubt-toolbar">\n'
            '  <button class="ubt-btn active" onclick="setMode(\'mode-target\', this)">纯译文</button>\n'
            '  <button class="ubt-btn" onclick="setMode(\'mode-stacked\', this)">双语堆叠</button>\n'
            '  <button class="ubt-btn" onclick="setMode(\'mode-split\', this)">左右对照</button>\n'
            '  <button id="ubt-theme-btn" class="ubt-btn" onclick="toggleTheme()">深色</button>\n'
            "</div>\n"
            "<script>\n"
            "function setMode(mode, btn) {\n"
            "  document.body.className = mode;\n"
            "  document.querySelectorAll('.ubt-btn').forEach(function(b){ if (b.id !== 'ubt-theme-btn') b.classList.remove('active'); });\n"
            "  if (btn) btn.classList.add('active');\n"
            "  try { localStorage.setItem('ubt_view_mode', mode); } catch(e){}\n"
            "}\n"
            "(function(){\n"
            "  try {\n"
            "    var saved = localStorage.getItem('ubt_view_mode');\n"
            "    if (saved) {\n"
            "      var btn = document.querySelector('.ubt-btn[onclick*=\"' + saved + '\"]');\n"
            "      if (btn) setMode(saved, btn);\n"
            "    }\n"
            "  } catch(e){}\n"
            "})();\n"
            "</script>\n"
        )

    return (
        "<!DOCTYPE html>\n"
        f'<html lang="{html.escape(lang)}"{dir_attr}>\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f"<title>{html.escape(title)}</title>\n"
        f"{katex_script}"
        f"<style>{_MODERN_CSS}</style>\n"
        "</head>\n"
        '<body class="mode-target">\n'
        f"{theme_script}"
        f"{toolbar}"
        '<main class="ubt-article-container">\n'
    )


def element_text(block: IRBlock, translations: Mapping[str, str]) -> str:
    """The text this block delivers: its placed translation, else its source.

    A block the delivery keeps in the source shows its source — that is
    :func:`_kept_in_source`'s contract (deliberate keep, fail-closed
    render-skip, or a quarantined ``BLOCKED_HUMAN`` block).

    Otherwise the text goes through
    :func:`ubt.render.delivered_text.delivered_text`, the same rule the PDF
    renderer uses, instead of taking ``target_text`` verbatim. Taking it verbatim
    shipped pipeline markup straight to the reader: a
    ``<mark class="ubt-failed-draft" title="REPAIR ERROR …">`` wrapper around a
    real translation was escaped into the page together with the provider's
    exception text, while the PDF showed the clean sentence.

    The ``translations`` override wins when present: an interactive PE edit is
    the most recent human decision, and it is plain text (no wrapper).

    A wrapper is stripped from the override too. The export stage builds this
    map with ``delivered_text``, but a map that reached the view by any other
    route (an older caller, a ledger row written before that fix, a test
    fixture) must not be able to smuggle pipeline markup onto the page just
    because it arrived through the override branch. The rule is the same one
    :func:`delivered_text` applies, so the two cannot disagree.
    """
    if _kept_in_source(block):
        return block.source_text or ""
    override = (translations.get(block.id) or "").strip()
    if override:
        return unwrap_mark(override).strip() if is_any_mark(override) else override
    return delivered_text(block)


def _kept_in_source(block: IRBlock) -> bool:
    """Whether the delivery keeps this block in the source (deliberate or fail-closed).

    Delegates to the single owner of this decision (``content.adapt``) rather
    than re-listing the skip-flag prefixes: a local copy drifted, so a
    quarantined (``BLOCKED_HUMAN``) block -- whose ``target_text`` is a ``<mark>``
    wrapper around the source, not a translation -- was rendered as its raw
    marker text in the HTML/EPUB view. One predicate, one place to keep correct.
    """
    from ubt.core.content.adapt import kept_in_source

    return kept_in_source(block)


def _render_element(
    element: Element, body: str, asset_href: str | None = None, raw_text: str = ""
) -> str:
    """One element's HTML tag, with ``body`` already escaped.

    ``asset_href`` is the resolved ``src`` for a figure (a data URI in the HTML
    view, an in-package path in the EPUB); ``None`` means the asset could not be
    resolved, and the figure degrades to a labelled placeholder rather than a
    silently empty box. ``raw_text`` is the unescaped delivered text, which the
    table branch parses (a grid must be read before escaping).
    """
    if isinstance(element, Heading):
        level = min(max(element.level, 1), 6)
        return f"<h{level}>{body}</h{level}>"
    if isinstance(element, Paragraph):
        return f"<p>{body}</p>"
    if isinstance(element, Dialogue):
        return f'<p class="dialogue">{body}</p>'
    if isinstance(element, Caption):
        return f"<figcaption>{body}</figcaption>"
    if isinstance(element, CodeBlock):
        return f"<pre><code>{body}</code></pre>"
    if isinstance(element, Formula):
        return f'<span class="formula">{body}</span>'
    if isinstance(element, Table):
        return _render_table(raw_text or body)
    if isinstance(element, Figure):
        # ``data-asset`` carries the BASENAME only: the delivered view is a
        # customer artifact, and the extractor's asset_id is an absolute host
        # path (``/home/<user>/.../pic_p1_1.png``) that has no business
        # shipping. The name is enough to correlate a figure with its block.
        asset_name = html.escape(Path(element.asset_id).name or element.asset_id)
        if asset_href:
            # Self-closed on purpose: this fragment is also lowered into the
            # EPUB's ``text.xhtml`` (``application/xhtml+xml``), where there are
            # no void elements and a bare ``<img>`` is a well-formedness error.
            return (
                f'<figure data-asset="{asset_name}">'
                f'<img src="{html.escape(asset_href, quote=True)}" alt=""/>'
                "</figure>"
            )
        return f'<figure class="figure-missing"><p>[figure: {asset_name}]</p></figure>'
    return f"<p>{body}</p>"


def _render_table(markup: str) -> str:
    """Lower a table block's delivered grid to a real ``<table>``.

    A table block's delivered text is a Markdown pipe grid (the renderer's own
    input). The view used to wrap that grid verbatim in ``<pre class="table">``,
    so a screen reader read the pipes and the text reflowed as code. Parsing the
    grid through the renderer's own parser (one definition of "a real table")
    yields the semantic table; a grid that does not parse — a malformed
    translation the PDF renderer would also fall back on — keeps the ``<pre>``
    form rather than dropping the block.
    """
    from ubt.render.table_compiler import parse_markdown_table

    parsed = parse_markdown_table(markup, escape=False)
    if parsed is None:
        return f'<pre class="table">{html.escape(markup)}</pre>'
    headers, rows = parsed
    head = "".join(f"<th>{html.escape(cell)}</th>" for cell in headers)
    body_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body_rows}</tbody></table>"


def _html_style_wrap(segment: str, style: SpanStyle) -> str:
    """Wrap an escaped segment in the HTML its source style calls for.

    The HTML twin of the renderer's ``_wrap_style``: same inputs, same
    precedence (superscript innermost), a ``<span style>`` carrying the colour.
    Applied after escaping, so the injected tags are never escaped.
    """
    start, _end, bold, italic, superscript, color_hex = style
    if superscript:
        segment = f"<sup>{segment}</sup>"
    if italic:
        segment = f"<em>{segment}</em>"
    if bold:
        segment = f"<strong>{segment}</strong>"
    if color_hex:
        segment = f'<span style="color:{html.escape(color_hex)}">{segment}</span>'
    return segment


def render_fragment(
    blocks: Sequence[IRBlock],
    translations: Mapping[str, str],
    *,
    asset_resolver: AssetResolver | None = None,
    bilingual: bool = False,
    math_renderer: str = "katex",
) -> tuple[str, tuple[Placement, ...]]:
    """The element HTML and every placement, without the document wrapper.

    Shared by the HTML view and the EPUB view, so both lower blocks the same way.

    A block's styled runs (:data:`StyleMeta.inline_runs` for the source spans
    the probe recorded, ``target_runs`` for the model's bold markers) are
    located in the delivered text and wrapped: a heading or paragraph then keeps
    the emphasis and colour its source had, where an ``html.escape`` of the
    whole text dropped every one. A run that survives only after the target's
    CJK normalization still locates (tolerant matching); one the translation
    rewrote draws plain, the same contract the Typst renderer documents.

    Three non-text element classes lower semantically rather than as raw markup
    (2026-10-14 review, W3-7): a table grid becomes a real ``<table>``, a
    formula's LaTeX degrades through the transpiler's own fallback to readable
    Unicode (the same guarantee the PDF renderer's degrade path gives), and a
    figure resolves through ``asset_resolver`` to an actual ``<img>`` instead of
    an empty ``<figure data-asset>`` box.

    ``math_renderer`` selects how a formula is delivered:

    * ``"katex"`` (the HTML companion) emits delimited LaTeX in a ``.math-tex``
      span for the page's KaTeX auto-render to typeset, with the Unicode
      reading in ``title``/``aria-label``.
    * ``"unicode"`` (the EPUB companion) emits the Unicode reading as the
      visible text. An EPUB cannot rely on a script: a reading system is
      frequently offline, and shipping raw ``$$\\frac{a}{b}$$`` meant the
      reader saw LaTeX source in the body. MathML would be the native answer but
      requires a converter the package does not depend on, so the same
      degradation the PDF renderer uses is the honest delivery.
    """
    from ubt.render.typst.document_renderer import _list_is_ordered
    from ubt.render.typst.math_transpiler import clean_math_fallback_text

    if math_renderer not in ("katex", "unicode"):
        raise ValueError(f"unknown math_renderer: {math_renderer!r}")

    resolve = asset_resolver or _default_asset_resolver
    parts: list[str] = []
    placements: list[Placement] = []
    # "" when no list is open, else the open list's tag ("ul"/"ol").
    list_open = ""
    toc_open = False
    for block in sorted(blocks, key=lambda b: b.spine_index):
        element = block.element
        # The PDF renderer's two omissions apply here too. Both were decisions
        # only that renderer made, so the companion views disagreed with it: a
        # running head (an arXiv stamp, a journal author line) opened the HTML
        # body as an <h1>, and a table-of-contents row became a real section
        # heading with no body under it. One predicate each, shared.
        if dropped_page_furniture(block):
            continue
        text = element_text(block, translations)
        if is_toc_entry(block):
            # A TOC row is a reference, not a section: rendering it as a heading
            # invented sections with no body (and an orphaned near-empty page in
            # the PDF) and dropped the recovered page number. Lower it as a list
            # item so the reader keeps the contents listing.
            if list_open:
                parts.append(f"</{list_open}>")
                list_open = ""
            if not toc_open:
                parts.append('<ol class="ubt-toc">\n')
                toc_open = True
            parts.append(f'  <li class="ubt-toc-entry">{html.escape(text)}</li>\n')
            placements.append(
                Placement(
                    block.id,
                    element.span.page,
                    drawn=True,
                    detail="html:toc_entry",
                )
            )
            continue
        if toc_open:
            parts.append("</ol>")
            toc_open = False
        if isinstance(element, Formula):
            # Deliver the LaTeX inside a ``.math-tex`` span so KaTeX renders it
            # to real typeset math. KaTeX's auto-render parses the element's *text
            # content*, so that content must be delimited LaTeX — the readable
            # Unicode fallback goes in the ``title``/``aria-label`` instead, shown
            # only if the KaTeX CDN is unreachable.
            latex_body = _math_body_for_katex(text)
            fallback = clean_math_fallback_text(latex_body)
            if math_renderer == "unicode":
                # EPUB: no script will run, so the Unicode reading must be the
                # visible text. The LaTeX goes in ``title`` for a reader that
                # wants to copy the source.
                parts.append(
                    f'<span class="math-tex math-unicode" '
                    f'title="{html.escape(latex_body, quote=True)}" '
                    f'data-latex="{html.escape(latex_body, quote=True)}">'
                    f"{html.escape(fallback)}</span>"
                )
            else:
                escaped_latex = html.escape(latex_body)
                escaped_fallback = html.escape(fallback, quote=True)
                parts.append(
                    f'<span class="math-tex" title="{escaped_fallback}" '
                    f'aria-label="{escaped_fallback}">$${escaped_latex}$$</span>'
                )
            placements.append(
                Placement(
                    block.id,
                    element.span.page,
                    drawn=not _kept_in_source(block),
                    detail=f"html:{element.kind}",
                )
            )
            continue
        styled = styled_chunks(text, block_runs(block))
        if len(styled) == 1 and styled[0][1] is None:
            body = html.escape(text)
        else:
            body = "".join(
                html.escape(chunk) if style is None else _html_style_wrap(html.escape(chunk), style)
                for chunk, style in styled
            )
        if isinstance(element, ListItem):
            # The marker decides ordered vs bullet — the same single owner the
            # Typst renderer asks — instead of flattening every list to <ul>,
            # which deleted the numbering from the companion HTML/EPUB views
            # while the same job's PDF kept it.
            tag = "ol" if _list_is_ordered(block) else "ul"
            if list_open != tag:
                if list_open:
                    parts.append(f"</{list_open}>")
                parts.append(f"<{tag}>")
                list_open = tag
            parts.append(f"<li>{body}</li>")
        else:
            if list_open:
                parts.append(f"</{list_open}>")
                list_open = ""
            asset_href = resolve(element.asset_id) if isinstance(element, Figure) else None
            rendered_el = _render_element(element, body, asset_href, raw_text=text)
            if (
                bilingual
                and block.source_text
                and block.source_text.strip() != text.strip()
                and not _kept_in_source(block)
            ):
                source_body = html.escape(block.source_text.strip())
                rendered_source = _render_element(
                    element, source_body, asset_href, raw_text=block.source_text
                )
                pair_html = (
                    f'<div class="bilingual-pair" data-block-id="{html.escape(block.id)}">\n'
                    f'  <div class="source-row">{rendered_source}</div>\n'
                    f'  <div class="target-row">{rendered_el}</div>\n'
                    "</div>"
                )
                parts.append(pair_html)
            else:
                parts.append(rendered_el)
        placements.append(
            Placement(
                block.id,
                element.span.page,
                drawn=not _kept_in_source(block),
                detail=f"html:{element.kind}",
            )
        )
    if toc_open:
        parts.append("</ol>")
    if list_open:
        parts.append(f"</{list_open}>")
    return "\n".join(parts), tuple(placements)


def compose_html(
    blocks: Sequence[IRBlock],
    translations: Mapping[str, str],
    output_path: str | Path,
    *,
    lang: str = "en",
    direction: str = "ltr",
    title: str = "UBT translation",
    bilingual: bool = False,
) -> Composition:
    """Lower the delivered blocks to semantic HTML, recording every placement.

    ``lang``/``direction`` describe the *target* language; a right-to-left target
    gets ``dir="rtl"`` on the root element so a browser lays the document out
    right-to-left (the text stays in logical order).
    """
    fragment, placements = render_fragment(blocks, translations, bilingual=bilingual)
    output = Path(output_path)
    atomic_write_text(
        output,
        f"{document_head(lang, direction, title=title, bilingual=bilingual)}{fragment}\n{_TAIL}",
        encoding="utf-8",
    )
    return Composition(output_path=output, placements=placements)


__all__ = ["compose_html", "document_head", "element_text", "render_fragment"]
