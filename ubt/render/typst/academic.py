"""Academic paper layout template and formatting for Typst rendering.

Provides academic paper layout (single or dual column), title/abstract banner,
floating full-width figures/tables, and academic typography rules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ubt.layout.theme import direction_for
from ubt.render.typst.document_renderer import escape_markup, typst_lang_arguments

#: Minimum share of pages the profiler must flag multi-column before the
#: renderer treats the source as two-column. A genuine two-column document has
#: most of its body pages laid out in columns, so it measures well above this;
#: a single-column document now measures ~0.0 (before the page-profiler fix it
#: falsely measured 0.225 on a single-column paper, which is why this floor was
#: briefly set very low — with the false positives gone the bar is a real
#: majority again). Front matter, references, and full-width figure pages are
#: single-column, so the bar is a bare majority, not an absolute.
_MULTICOLUMN_SHARE_FLOOR = 0.5


def is_academic_document(manifest_or_meta: Any) -> bool:
    """Whether a document manifest or metadata indicates academic content styling.

    Signals that the document should use the academic template (title/abstract
    styling, academic headings, academic headers/page numbers).

    NOTE: This does NOT determine column count. Column count is resolved
    independently by `resolve_document_columns` based on the physical source
    layout and explicit user overrides.
    """
    if manifest_or_meta is None:
        return False

    meta: Mapping[str, Any] = {}
    doc_type: str | None = None

    if isinstance(manifest_or_meta, Mapping):
        meta = manifest_or_meta
        doc_type = meta.get("doc_type") or meta.get("layout_template")
    else:
        doc_type = getattr(manifest_or_meta, "doc_type", None)
        if hasattr(manifest_or_meta, "metadata") and isinstance(manifest_or_meta.metadata, Mapping):
            meta = manifest_or_meta.metadata
            if not doc_type:
                doc_type = meta.get("doc_type") or meta.get("layout_template")

    if doc_type and str(doc_type).lower() in (
        "paper",
        "academic",
        "thesis",
        "scientific",
        "journal",
    ):
        return True

    if meta.get("academic") is True or meta.get("is_academic") is True:
        return True

    profile = meta.get("genre_profile")
    return bool(
        profile
        and str(profile).lower()
        in (
            "academic",
            "paper",
            "scientific",
            "journal",
            "thesis",
        )
    )


def resolve_document_columns(manifest_or_meta: Any) -> int:
    """Resolve the physical column count for rendering (1 or 2).

    Fidelity first: the delivered layout mirrors the *source*, and only an
    explicit user override departs from it. The precedence is

    1. **Explicit user override** — ``manifest.run.layout_columns`` (from
       ``--columns`` / ``UBT_LAYOUT_COLUMNS``). This is a run decision, so it
       lives on ``run``, and it wins over everything.
    2. **Physical multi-column evidence in the source** — the PDF page profiler's
       finding, carried on ``manifest.run.route_decision`` as
       ``multicolumn_page_share`` (and ``has_multicolumn`` where a caller already
       reduced it to a bool). A document laid out in two columns keeps two
       columns; this is the whole point of the academic template.
    3. **Metadata keys** (``has_multicolumn`` / ``is_multicolumn`` /
       ``multicolumn_page_share`` / ``columns``) — for callers that hand this a
       plain metadata mapping rather than a manifest.
    4. **Default: single column.** A single-column paper, thesis, report, or
       textbook is never forced into two columns, and a paper with no measured
       multi-column evidence is not assumed to be two-column just because its
       genre profile is academic. ``--profile`` styles the *text*; it does not
       decide the *layout*.

    The previous cut read only ``manifest.metadata``, where the profiler's
    ``multicolumn_page_share`` is never written (it lives under
    ``run.route_decision``), so the function returned 1 on every real two-column
    paper and silently flattened it to a single column — the exact
    "版面坍塌" the template exists to prevent.
    """
    if manifest_or_meta is None:
        return 1

    meta: Mapping[str, Any] = {}
    run: Any = None
    if isinstance(manifest_or_meta, Mapping):
        meta = manifest_or_meta
    else:
        if hasattr(manifest_or_meta, "metadata") and isinstance(manifest_or_meta.metadata, Mapping):
            meta = manifest_or_meta.metadata
        run = getattr(manifest_or_meta, "run", None)

    # 1. Explicit user override (run decision; also accepted on the mapping form).
    override = getattr(run, "layout_columns", None) if run is not None else None
    if override is None:
        override = meta.get("layout_columns")
    if override is not None:
        try:
            col_int = int(override)
            if col_int in (1, 2):
                return col_int
        except (ValueError, TypeError):
            pass

    # 2. Physical multi-column evidence, wherever the run recorded it.
    route = getattr(run, "route_decision", None) if run is not None else None
    if not isinstance(route, Mapping):
        route = meta.get("route_decision")
    if _has_multicolumn_evidence(route):
        return 2

    # 3. Metadata keys (the mapping form; some producers flatten them here).
    if _has_multicolumn_evidence(meta):
        return 2

    # 4. Default to single column (fidelity first).
    return 1


def _has_multicolumn_evidence(source: Any) -> bool:
    """Whether a route-decision/metadata mapping shows a two-column source.

    Accepts either a pre-reduced boolean (``has_multicolumn`` / ``is_multicolumn``)
    or the profiler's ``multicolumn_page_share``. The boolean is preferred when
    present: it is the engine selector's deliberate verdict (``multicolumn_pages
    >= PROBE_MIN_SIGNAL_PAGES``, a handful of sampled pages showing a right-column
    edge), which is the codebase's own notion of "this is a multi-column
    document".

    The share fallback requires a real majority of pages. A paper's front matter
    (title, TOC, author block), its references, and its full-width figure/table
    pages are single-column, so the body must dominate for the document to count
    as two-column. An earlier cut set this bar very low to work around a page
    profiler that mis-measured single-column papers (a rotated margin stamp read
    as a right column, scoring 0.225); with that fixed, a single-column document
    measures ~0.0 and the bar is a plain majority again.
    """
    if not isinstance(source, Mapping):
        return False
    if source.get("has_multicolumn") is True or source.get("is_multicolumn") is True:
        return True
    share = source.get("multicolumn_page_share")
    if share is None:
        return False
    try:
        return float(share) >= _MULTICOLUMN_SHARE_FLOOR
    except (ValueError, TypeError):
        return False


def academic_preamble(
    target_lang: str = "zh",
    font_stack: Sequence[str] | None = None,
    paper_title: str = "",
    columns: int = 1,
) -> str:
    """Generate the Typst preamble for an academic paper (single or dual column)."""
    direction = direction_for(target_lang or "zh").value
    lang = (target_lang or "zh").replace("_", "-")
    rtl = "\n#set text(dir: rtl)" if direction == "rtl" else ""

    from ubt.adapters.pdf.font_probe import resolve_for_render

    if font_stack:
        stack: tuple[str, ...] = tuple(font_stack)
    else:
        # No stack supplied: resolve the profile against this machine, so an
        # absent spare face is pruned instead of warning on every compile.
        stack = resolve_for_render(lang).families

    # Same `covers` routing as the book template: Latin faces inside a CJK
    # document are scoped so CJK punctuation is not claimed by them.
    from ubt.render.typst.font_routing import typst_font_list

    body_fonts = typst_font_list(stack, lang)
    cjk_first = typst_font_list(tuple(reversed(stack)), lang)

    escaped_title = escape_markup(paper_title) if paper_title else ""
    is_zh = bool(lang and lang.strip().lower().startswith("zh"))
    page_display = "第 #counter(page).display() 页" if is_zh else "Page #counter(page).display()"

    header_block = ""
    if escaped_title:
        header_block = (
            "  header: context {\n"
            "    if counter(page).get().first() > 1 [\n"
            "      #set text(size: 8pt, fill: luma(100))\n"
            f"      #grid(columns: (1fr, 1fr), align(left)[{escaped_title}], align(right)[{page_display}])\n"
            "      #v(-0.3em)\n"
            "      #line(length: 100%, stroke: 0.35pt + luma(180))\n"
            "    ]\n"
            "  },"
        )
    else:
        header_block = (
            "  header: context {\n"
            "    if counter(page).get().first() > 1 [\n"
            "      #set text(size: 8pt, fill: luma(100))\n"
            f"      #align(right)[{page_display}]\n"
            "      #v(-0.3em)\n"
            "      #line(length: 100%, stroke: 0.35pt + luma(180))\n"
            "    ]\n"
            "  },"
        )

    is_dual = columns == 2
    col_str = "  columns: 2," if is_dual else "  columns: 1,"
    margin_str = "  margin: (x: 1.8cm, y: 2.0cm)," if is_dual else "  margin: (x: 2.5cm, y: 2.5cm),"
    font_size = "9.5pt" if is_dual else "10.5pt"
    par_rules = (
        "#set par(justify: true, leading: 0.65em, spacing: 0.75em, first-line-indent: (amount: 1.5em, all: true))"
        if is_dual
        else "#set par(justify: true, leading: 0.75em, spacing: 0.85em, first-line-indent: (amount: 2.0em, all: true))"
    )

    lines = [
        "#set page(",
        '  paper: "a4",',
        col_str,
        margin_str,
        header_block,
        '  numbering: "1",',
        ")",
        f"#set text(font: ({body_fonts}), size: {font_size}{typst_lang_arguments(lang)}){rtl}",
        par_rules,
        "#show heading.where(level: 1): it => [",
        "  #v(0.5em)",
        f'  #block(text(size: 13pt, weight: "bold", font: ({cjk_first}))[#it.body])',
        "  #v(0.25em)",
        "]",
        "#show heading.where(level: 2): it => [",
        "  #v(0.35em)",
        f'  #block(text(size: 11pt, weight: "bold", font: ({cjk_first}))[#it.body])',
        "  #v(0.2em)",
        "]",
        "#show figure.caption: it => block(text(size: 8.5pt, fill: luma(60))[#it.body])",
    ]
    return "\n".join(lines)


def academic_title_banner(
    title: str,
    abstract: str = "",
    authors: Sequence[str] = (),
    columns: int = 1,
    lang: str = "zh",
) -> str:
    """Emit a title, author, and abstract block for an academic paper."""
    if not title:
        return ""

    escaped_title = escape_markup(title)
    authors_str = ", ".join(escape_markup(a) for a in authors) if authors else ""

    parts: list[str] = []
    is_dual = columns == 2
    if is_dual:
        parts.append('#place(top, scope: "parent", float: true)[')
    else:
        parts.append("#block(width: 100%)[")

    parts.extend(
        [
            "  #align(center)[",
            f'    #text(size: 16pt, weight: "bold")[{escaped_title}]\\',
        ]
    )

    if authors_str:
        parts.append(f'    #v(0.3em)#text(size: 10pt, style: "italic")[{authors_str}]\\')

    parts.append("    #v(0.6em)")
    parts.append("  ]")

    if abstract:
        escaped_abstract = escape_markup(abstract)
        is_zh = bool(lang and lang.strip().lower().startswith("zh"))
        abstract_label = "摘要：" if is_zh else "Abstract: "
        parts.extend(
            [
                "  #rect(width: 100%, stroke: none, fill: luma(248), inset: (x: 12pt, y: 10pt), radius: 3pt)[",
                f'    #text(weight: "bold")[{abstract_label}]',
                f'    #text(size: 9pt, style: "italic")[{escaped_abstract}]',
                "  ]",
            ]
        )

    parts.extend(["  #v(1.2em)", "]"])
    return "\n".join(parts)
