"""The native Markdown reader.

The reader's core promises, over a fixture exercising every block construct:

- every element's ``Span.chars`` slices exactly its own text out of
  ``CanonicalSource.text`` (a verifiable span);
- the ``Document`` round-trips through the bridge losslessly (element count
  and text preserved), so the typed model can replace the ``IRBlock`` detour;
- the reader loses no text relative to the existing Markdown adapter, compared
  as a whitespace-normalized token multiset (the reader splits headings and
  list items the adapter keeps inside paragraphs -- splitting changes block
  boundaries, not the tokens the document carries).
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

from ubt.adapters.markdown.adapter import MarkdownAdapter
from ubt.analyze.assemble import element_text
from ubt.analyze.bridge import blocks_from_document, document_from_blocks
from ubt.analyze.reader_md import read_md

pytestmark = pytest.mark.fast

_BOOK = """# Attention

Attention lets a model *attend* to every token, so `q(k)` scores each key.

## The mechanism

1. Score every key against the query.
2. Normalize with a softmax.

- The product of scores and values is the output.
- Everything else is bookkeeping.

```python
def softmax(x): return x / x.sum()
```

| Model | Score |
| ----- | ----- |
| BASE  | 91.2  |

> Attention is all you need.
"""

_TOKEN_RE = re.compile(r"\S+")


def _tokens(text: str) -> Counter[str]:
    return Counter(_TOKEN_RE.findall(text))


def test_every_element_has_an_exact_verifiable_span(tmp_path: Path) -> None:
    book = tmp_path / "book.md"
    book.write_text(_BOOK, encoding="utf-8")
    document = read_md(book)
    assert document.elements, "the reader extracted nothing"
    for element in document.elements:
        chars = element.span.chars
        assert chars is not None, f"{element.id}: missing Span.chars"
        assert document.source.text[chars[0] : chars[1]] == element_text(element), element.id


def test_the_document_round_trips_through_the_bridge(tmp_path: Path) -> None:
    book = tmp_path / "book.md"
    book.write_text(_BOOK, encoding="utf-8")
    document = read_md(book)
    round_tripped = document_from_blocks(
        blocks_from_document(document), doc_id=document.source.doc_id
    )
    assert len(round_tripped.elements) == len(document.elements)
    assert _tokens(" ".join(element_text(e) for e in round_tripped.elements)) == _tokens(
        " ".join(element_text(e) for e in document.elements)
    )


def test_the_reader_structure_covers_every_construct(tmp_path: Path) -> None:
    book = tmp_path / "book.md"
    book.write_text(_BOOK, encoding="utf-8")
    document = read_md(book)
    kinds = Counter(element.kind.value for element in document.elements)
    assert kinds.get("heading", 0) >= 2
    assert kinds.get("paragraph", 0) >= 1
    assert kinds.get("list_item", 0) >= 4
    assert kinds.get("code_block", 0) >= 1
    assert kinds.get("table", 0) >= 1


async def test_the_reader_loses_nothing_the_adapter_had(tmp_path: Path) -> None:
    # The old IR path (MarkdownAdapter) and the native reader must agree on the
    # document's token content, whatever way each splits the blocks.
    book = tmp_path / "book.md"
    book.write_text(_BOOK, encoding="utf-8")
    document = read_md(book)
    # element_text covers every kind (a Table carries its markup, not .text).
    reader_tokens = _tokens(" ".join(element_text(e) for e in document.elements))

    adapter = MarkdownAdapter()
    texts: list[str] = []
    async for chapter in adapter.parse_stream(book):
        texts.extend(block.source_text for block in chapter.blocks)
    adapter_tokens = _tokens("\n".join(texts))
    assert adapter_tokens, "the adapter extracted nothing"
    assert reader_tokens == adapter_tokens
