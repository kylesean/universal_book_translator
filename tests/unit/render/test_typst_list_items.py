"""A list item must reach the PDF as a list item.

The Typst backend had no ``LIST_ITEM`` branch: the block fell through to the
prose path, so every bullet and every list number was dropped and a delivered
list flattened into unmarked paragraphs. The marker was already on the element
(all three readers populate it) and the HTML companion already emitted
``<ul><li>`` — only the Typst backend lost it.

Consecutive items of the same kind are emitted as one ``#list(..)``/``#enum(..)``
call. The call form (not the ``- `` markup form) because an item body is already
escaped markup that may contain a soft wrap, and each body is bracketed so a
stray ``#`` cannot be parsed as code.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

from ubt.core.ir.models import BlockStatus, IRBlock
from ubt.model.ast import ListItem, Paragraph
from ubt.model.span import Span
from ubt.render.typst.document_renderer import TypstDocumentRenderer

pytestmark = pytest.mark.fast


def _item(index: int, source: str, target: str, marker: str = "•") -> IRBlock:
    element = ListItem(
        id=f"l{index}", spine_index=index, span=Span(page=1), text=source, marker=marker
    )
    return IRBlock(element=element, target_text=target, status=BlockStatus.MTQE_PASSED)


def _paragraph(index: int, source: str, target: str) -> IRBlock:
    element = Paragraph(id=f"p{index}", spine_index=index, span=Span(page=1), text=source)
    return IRBlock(element=element, target_text=target, status=BlockStatus.MTQE_PASSED)


def _body(blocks: list[IRBlock], tmp_path: pathlib.Path, **kwargs: object) -> str:
    """The emitted body, with the preamble (a single leading block) removed."""
    renderer = TypstDocumentRenderer()
    for key, value in kwargs.items():
        setattr(renderer, key, value)
    source, _stats = renderer.build_source(blocks, tmp_path)
    preamble = renderer.preamble("zh")
    assert source.startswith(preamble)
    return source[len(preamble) :].strip()


def test_bullets_become_one_list(tmp_path: pathlib.Path) -> None:
    body = _body([_item(1, "one", "甲"), _item(2, "two", "乙")], tmp_path)
    assert body == "#list([甲], [乙])"


def test_numbered_items_become_an_enumeration(tmp_path: pathlib.Path) -> None:
    body = _body([_item(1, "one", "甲", "1."), _item(2, "two", "乙", "2.")], tmp_path)
    assert body == "#enum([甲], [乙])"


@pytest.mark.parametrize("marker", ["1.", "(1)", "1)", "１．", "2.1."])
def test_ordered_marker_forms_are_recognised(marker: str, tmp_path: pathlib.Path) -> None:
    body = _body([_item(1, "x", "甲", marker)], tmp_path)
    assert body.startswith("#enum(")


@pytest.mark.parametrize("marker", ["•", "-", "*", ""])
def test_bullet_forms_and_a_missing_marker_are_bullets(marker: str, tmp_path: pathlib.Path) -> None:
    body = _body([_item(1, "x", "甲", marker)], tmp_path)
    assert body.startswith("#list(")


def test_a_paragraph_splits_two_lists(tmp_path: pathlib.Path) -> None:
    blocks = [
        _item(1, "a", "甲"),
        _paragraph(2, "break", "间隔"),
        _item(3, "b", "乙"),
    ]
    body = _body(blocks, tmp_path)
    assert body.count("#list(") == 2


def test_adjacent_lists_of_different_kinds_do_not_merge(tmp_path: pathlib.Path) -> None:
    """A bullet run followed by a numbered run stays two lists."""
    blocks = [_item(1, "a", "甲", "•"), _item(2, "b", "乙", "1."), _item(3, "c", "丙", "•")]
    body = _body(blocks, tmp_path)
    assert "#enum(" in body
    assert body.count("#list(") == 2


def test_every_item_counts_as_rendered(tmp_path: pathlib.Path) -> None:
    renderer = TypstDocumentRenderer()
    _source, stats = renderer.build_source(
        [_item(1, "a", "甲"), _item(2, "b", "乙"), _item(3, "c", "丙")], tmp_path
    )
    assert stats.blocks_rendered == 3


def test_a_source_kept_item_renders_its_source(tmp_path: pathlib.Path) -> None:
    kept = _item(1, "Bibliography entry", "")
    kept.policy_translate = False
    body = _body([kept], tmp_path)
    assert body == "#list([Bibliography entry])"


def test_a_translated_prose_lt_is_escaped_inside_the_item(tmp_path: pathlib.Path) -> None:
    """The bracket wrapper must not let a ``#`` reach Typst as code."""
    body = _body([_item(1, "tag", "使用 #tag 与 [x]")], tmp_path)
    assert "\\#" in body
    assert "\\[" in body


@pytest.mark.skipif(shutil.which("typst") is None, reason="typst not installed")
def test_the_generated_list_compiles_and_keeps_its_markers(tmp_path: pathlib.Path) -> None:
    """End to end: the PDF's text layer carries the bullets and the numbers."""
    blocks = [
        _item(1, "alpha", "甲", "•"),
        _item(2, "beta", "乙", "•"),
        _item(3, "one", "一", "1."),
        _item(4, "two", "二", "2."),
    ]
    renderer = TypstDocumentRenderer()
    source, _stats = renderer.build_source(blocks, tmp_path)
    typ = tmp_path / "list.typ"
    typ.write_text(source, encoding="utf-8")
    compiled = subprocess.run(
        ["typst", "compile", str(typ), str(tmp_path / "list.pdf")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    if shutil.which("pdftotext") is None:
        pytest.skip("pdftotext not installed")
    text = subprocess.run(
        ["pdftotext", str(tmp_path / "list.pdf"), "-"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    assert "• 甲" in text
    assert "1. 一" in text
    assert "2. 二" in text
