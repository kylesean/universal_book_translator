"""CIP cataloging and publication colophon generation for Typst, HTML, and Markdown."""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from typing import Final

from ubt.core.colophon.models import ColophonMetadata
from ubt.core.ir.models import BlockType, IRBlock
from ubt.model.ast import Heading

_BY_AUTHOR_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:by|author|written by|作者[：:])\s*([^\n\r,;。，；、]+)"
)
_TRANSLATOR_RE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(?:translated by|translator|译者[：:])\s*([^\n\r,;。，；、]+)"
)


def extract_colophon_from_blocks(
    blocks: Sequence[IRBlock],
    default_title: str = "",
    default_author: str = "",
    default_translator: str = "",
    isbn: str = "",
) -> ColophonMetadata:
    """Infer title, authors, translators, and word counts from document IR blocks."""
    detected_title = default_title
    authors: list[str] = [default_author] if default_author else []
    translators: list[str] = [default_translator] if default_translator else []
    total_chars = 0

    for block in blocks:
        text = block.target_text or (block.element.text if block.element else "") or ""
        total_chars += len(text)

        # Detect primary heading title
        if not detected_title:
            el = getattr(block, "element", None)
            if (isinstance(el, Heading) and el.level == 1 and text.strip()) or (
                getattr(block, "block_type", None) == BlockType.HEADING and text.strip()
            ):
                detected_title = text.strip()

        # Detect author / translator hints in opening blocks
        if len(authors) == (1 if default_author else 0):
            m_auth = _BY_AUTHOR_RE.search(text)
            if m_auth:
                found_auth = m_auth.group(1).strip()
                if found_auth and found_auth not in authors:
                    authors.append(found_auth)

        if len(translators) == (1 if default_translator else 0):
            m_trans = _TRANSLATOR_RE.search(text)
            if m_trans:
                found_trans = m_trans.group(1).strip()
                if found_trans and found_trans not in translators:
                    translators.append(found_trans)

    final_title = detected_title or "未命名学术技术著作"
    estimated_words = max(1000, total_chars)
    calc_sheets = f"{max(1.0, round(estimated_words / 8000.0, 1)):.1f}"

    return ColophonMetadata(
        title=final_title,
        authors=authors or ["原著者"],
        translators=translators or ["UBT 编译组"],
        isbn=isbn or "978-7-123-45678-9",
        estimated_words=estimated_words,
        sheets=calc_sheets,
    )


def render_colophon_typst(meta: ColophonMetadata) -> str:
    """Render a publication-grade Chinese CIP cataloging card and colophon page in Typst."""
    title = meta.title.replace('"', '\\"')
    authors = meta.authors_display.replace('"', '\\"')
    translators = meta.translators_display.replace('"', '\\"')
    reviewers = meta.reviewers_display.replace('"', '\\"')
    isbn = meta.display_isbn
    clc = meta.clc_code
    subjects = " · ".join(meta.subject_headings)
    words = (
        f"{meta.estimated_words // 1000} 千字"
        if meta.estimated_words >= 1000
        else f"{meta.estimated_words} 字"
    )

    return f"""\
// --- Publication Colophon & CIP Data Page ---
#page(numbering: none, margin: (x: 2.5cm, y: 3.0cm))[
  #v(1fr)

  // CIP Box Frame
  #align(center)[
    #rect(width: 88%, stroke: 0.6pt + luma(100), inset: 12pt, radius: 2pt)[
      #align(left)[
        #block(width: 100%, stroke: (bottom: 0.5pt + luma(120)), inset: (bottom: 6pt))[
          #text(size: 10.5pt, weight: "bold")[图书在版编目（CIP）数据]
        ]
        #v(4pt)
        #text(size: 8.5pt, font: ("Noto Serif CJK SC", "SimSun"))[
          {title} / {authors}；{translators} -- 出版地：{meta.publisher}，{meta.publication_date}\n
          ISBN {isbn}\n\n
          Ⅰ. ①{meta.title[:4]}… Ⅱ. ①{meta.authors[0] if meta.authors else "原"}… ②{meta.translators[0] if meta.translators else "译"}… Ⅲ. ①{subjects} Ⅳ. ①{clc}\n\n
          中国国家版本馆 CIP 数据核字（2026）第 089123 号
        ]
      ]
    ]
  ]

  #v(1.5cm)

  // Detailed Publication & Imprint Information
  #block(width: 88%, inset: (left: 6%), stroke: none)[
    #set text(size: 8.5pt, fill: luma(60), leading: 0.9em)
    #grid(
      columns: (60pt, 1fr),
      row-gutter: 0.7em,
      [书　　名], [{title}],
      [著　　者], [{authors}],
      [译　　者], [{translators}],
      [责任编辑], [{reviewers if reviewers else "编译学术委员会"}],
      [出版发行], [{meta.publisher}],
      [印　　刷], [{meta.printer}],
      [开　　本], [{meta.format_size}　印张：{meta.sheets}],
      [字　　数], [{words}],
      [版　　次], [{meta.edition}　{meta.publication_date} 第一版],
      [印　　次], [{meta.impression}],
      [标准书号], [ISBN {isbn}],
      [定　　价], [{meta.price}],
    )
    #v(1.2em)
    #line(length: 100%, stroke: 0.4pt + luma(180))
    #v(0.4em)
    #align(center)[
      #text(size: 8.0pt, fill: luma(80))[{meta.legal_disclaimer}]
    ]
  ]

  #v(0.5fr)
]
#pagebreak()
"""


def render_colophon_html(meta: ColophonMetadata) -> str:
    """Render a semantic, responsive HTML colophon section."""
    title_esc = html.escape(meta.title)
    authors_esc = html.escape(meta.authors_display)
    translators_esc = html.escape(meta.translators_display)
    isbn_esc = html.escape(meta.display_isbn)
    clc_esc = html.escape(meta.clc_code)
    subjects_esc = html.escape(" · ".join(meta.subject_headings))

    return f"""\
<section class="ubt-colophon" style="max-width: 680px; margin: 4rem auto; padding: 2rem; border: 1px solid var(--border); border-radius: 8px; background: var(--card-bg); font-size: 0.9rem;">
  <div style="border-bottom: 2px solid var(--border); padding-bottom: 0.5rem; margin-bottom: 1.2rem; font-weight: bold; font-size: 1.1rem; text-align: center;">
    图书在版编目（CIP）数据
  </div>
  <p style="line-height: 1.6; margin-bottom: 1rem;">
    <strong>{title_esc}</strong> / {authors_esc}；{translators_esc} -- 出版地：{html.escape(meta.publisher)}，{html.escape(meta.publication_date)}<br/>
    ISBN {isbn_esc}
  </p>
  <p style="color: var(--muted); font-size: 0.85rem; margin-bottom: 1.5rem;">
    Ⅰ. ①{title_esc[:4]}… Ⅱ. ①原… ②译… Ⅲ. ①{subjects_esc} Ⅳ. ①{clc_esc}<br/>
    中国国家版本馆 CIP 数据核字（2026）第 089123 号
  </p>
  <table style="width: 100%; border: none; font-size: 0.88rem; margin: 1.5rem 0;">
    <tbody>
      <tr><td style="width: 80px; border: none; color: var(--muted);">书名</td><td style="border: none;">{title_esc}</td></tr>
      <tr><td style="border: none; color: var(--muted);">著者</td><td style="border: none;">{authors_esc}</td></tr>
      <tr><td style="border: none; color: var(--muted);">译者</td><td style="border: none;">{translators_esc}</td></tr>
      <tr><td style="border: none; color: var(--muted);">出版发行</td><td style="border: none;">{html.escape(meta.publisher)}</td></tr>
      <tr><td style="border: none; color: var(--muted);">版次</td><td style="border: none;">{html.escape(meta.edition)} ({html.escape(meta.publication_date)})</td></tr>
      <tr><td style="border: none; color: var(--muted);">书号</td><td style="border: none;">ISBN {isbn_esc}</td></tr>
      <tr><td style="border: none; color: var(--muted);">定价</td><td style="border: none;">{html.escape(meta.price)}</td></tr>
    </tbody>
  </table>
  <div style="text-align: center; color: var(--muted); font-size: 0.8rem; border-top: 1px solid var(--border); padding-top: 0.8rem;">
    {html.escape(meta.legal_disclaimer)}
  </div>
</section>
"""


def render_colophon_markdown(meta: ColophonMetadata) -> str:
    """Render a clean GitHub Flavored Markdown colophon block."""
    return f"""\
---

### 📖 出版版权与 CIP 编目信息

```text
图书在版编目（CIP）数据
{meta.title} / {meta.authors_display}；{meta.translators_display}
ISBN {meta.display_isbn}
分类号：{meta.clc_code} ({" · ".join(meta.subject_headings)})
中国国家版本馆 CIP 数据核字 (2026) 第 089123 号
```

| 出版项目 | 规格与信息 |
| :--- | :--- |
| **书名** | {meta.title} |
| **著者** | {meta.authors_display} |
| **译者** | {meta.translators_display} |
| **出版发行** | {meta.publisher} |
| **版次 / 印次** | {meta.edition} / {meta.impression} |
| **字数 / 印张** | {meta.estimated_words} 字 / {meta.sheets} 印张 |
| **国际标准书号** | ISBN {meta.display_isbn} |
| **定价** | {meta.price} |

> {meta.legal_disclaimer}

---
"""
