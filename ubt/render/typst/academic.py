"""Academic paper layout template and formatting for Typst rendering.

Provides two-column academic layout, full-width title/abstract banner,
floating full-width figures/tables, and academic typography rules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ubt.layout.theme import direction_for
from ubt.render.typst.document_renderer import escape_markup


def is_academic_document(manifest_or_meta: Any) -> bool:
    """Whether a document manifest or metadata dictionary designates an academic paper.

    The signal is resolved from the *manifest*, which now carries both the run's
    genre profile (stamped at ingest as ``metadata["genre_profile"]``) and the
    per-page layout classification (``metadata["page_kinds"]`` from the page
    profiler). Before this the predicate read only ``doc_type``/``columns`` keys
    that no producer ever wrote, so it returned ``False`` on every real run and
    the two-column branch was unreachable.
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

    if meta.get("columns") == 2 or meta.get("academic") is True:
        return True

    # Genre profile, stamped at ingest from the run's ``--profile``. The values
    # are the academic-leaning profile names (mirrors ``draft.ACADEMIC_PROFILES``).
    profile = meta.get("genre_profile")
    if profile and str(profile).lower() in (
        "academic",
        "textbook",
        "paper",
        "technical",
        "scientific",
        "journal",
        "thesis",
    ):
        return True

    # Page-layout fallback: a document whose pages are predominantly
    # multi-column/complex (the page profiler's ``MIXED_COMPLEX``) is a paper
    # body, even when no profile was named. Require a clear majority so a book
    # with a handful of two-column figure pages stays on the book template.
    page_kinds = meta.get("page_kinds")
    if isinstance(page_kinds, Mapping) and page_kinds:
        kinds = [str(v).lower() for v in page_kinds.values()]
        complex_share = sum(1 for k in kinds if k == "mixed_complex") / len(kinds)
        if complex_share >= 0.5:
            return True

    return False


def academic_preamble(
    target_lang: str = "zh",
    font_stack: Sequence[str] | None = None,
    paper_title: str = "",
) -> str:
    """Generate the Typst preamble for a two-column academic paper."""
    direction = direction_for(target_lang or "zh").value
    lang = (target_lang or "zh").replace("_", "-")
    rtl = "\n#set text(dir: rtl)" if direction == "rtl" else ""

    from ubt.core.language_profile import resolve_font_config

    if font_stack:
        stack: tuple[str, ...] = tuple(font_stack)
    else:
        stack = tuple(resolve_font_config(lang).typst_fonts)

    body_fonts = ", ".join(f'"{face}"' for face in stack)
    cjk_first = ", ".join(f'"{face}"' for face in reversed(stack))

    escaped_title = escape_markup(paper_title) if paper_title else ""

    header_block = ""
    if escaped_title:
        header_block = (
            "  header: context {\n"
            "    if counter(page).get().first() > 1 [\n"
            "      #set text(size: 8pt, fill: luma(100))\n"
            f"      #grid(columns: (1fr, 1fr), align(left)[{escaped_title}], align(right)[Page #counter(page).display()])\n"
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
            "      #align(right)[Page #counter(page).display()]\n"
            "      #v(-0.3em)\n"
            "      #line(length: 100%, stroke: 0.35pt + luma(180))\n"
            "    ]\n"
            "  },"
        )

    lines = [
        "#set page(",
        '  paper: "a4",',
        "  columns: 2,",
        "  margin: (x: 1.8cm, y: 2.0cm),",
        header_block,
        '  numbering: "1",',
        ")",
        f'#set text(font: ({body_fonts}), size: 9.5pt, lang: "{lang}"){rtl}',
        "#set par(justify: true, leading: 0.65em, spacing: 0.75em, first-line-indent: (amount: 1.5em, all: true))",
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
) -> str:
    """Emit a full-width title, author, and abstract block for page 1 of an academic paper."""
    if not title:
        return ""

    escaped_title = escape_markup(title)
    authors_str = ", ".join(escape_markup(a) for a in authors) if authors else ""

    parts: list[str] = [
        '#place(top, scope: "parent", float: true)[',
        "  #align(center)[",
        f'    #text(size: 16pt, weight: "bold")[{escaped_title}]\\',
    ]

    if authors_str:
        parts.append(f'    #v(0.3em)#text(size: 10pt, style: "italic")[{authors_str}]\\')

    parts.append("    #v(0.6em)")
    parts.append("  ]")

    if abstract:
        escaped_abstract = escape_markup(abstract)
        parts.extend(
            [
                "  #rect(width: 100%, stroke: none, fill: luma(248), inset: (x: 12pt, y: 10pt), radius: 3pt)[",
                '    #text(weight: "bold")[摘要：]',
                f'    #text(size: 9pt, style: "italic")[{escaped_abstract}]',
                "  ]",
            ]
        )

    parts.extend(["  #v(1.2em)", "]"])
    return "\n".join(parts)
