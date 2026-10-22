"""A formula in the EPUB companion must be readable, not LaTeX source.

The EPUB view reuses ``render_fragment`` with the HTML companion's KaTeX markup:
delimited LaTeX inside a ``.math-tex`` span, with the Unicode reading only in
``title``. That works in the HTML page, which injects a KaTeX auto-render script.
An EPUB cannot rely on one — reading systems are frequently offline, and the
package ships no script at all — so the reader saw ``$$\\frac{a}{b}$$`` as the
visible text of every equation.

MathML would be the native answer, but it needs a LaTeX→MathML converter the
package does not depend on (``latex2mathml`` is only a transitive dependency of
``docling-core``). The delivery therefore uses the same Unicode degradation the
PDF renderer already applies, with the LaTeX kept in ``title``/``data-latex``
for copying.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from ubt.core.ir.models import BlockStatus, IRBlock
from ubt.model.ast import Formula
from ubt.model.span import Span
from ubt.render.epub_view import compose_epub
from ubt.render.html_view import compose_html, render_fragment

pytestmark = pytest.mark.fast


def _formula(source: str = r"\frac{a}{b}") -> IRBlock:
    element = Formula(id="f1", spine_index=1, span=Span(page=1), source=source)
    return IRBlock(element=element, target_text="", status=BlockStatus.MTQE_PASSED)


def _xhtml(tmp_path: Path, block: IRBlock) -> str:
    output = tmp_path / "view.epub"
    compose_epub([block], {}, output, title="doc")
    with zipfile.ZipFile(output) as archive:
        return archive.read("OEBPS/text.xhtml").decode("utf-8")


def test_the_epub_body_shows_the_unicode_reading(tmp_path: Path) -> None:
    body = _xhtml(tmp_path, _formula())
    assert ">a/b</span>" in body


def test_the_epub_visible_text_carries_no_raw_latex(tmp_path: Path) -> None:
    """The regression: ``$$\\frac{a}{b}$$`` was the *visible* text.

    The LaTeX legitimately survives in ``title``/``data-latex`` — that is the
    copy affordance — so this checks the element's text content.
    """
    import xml.etree.ElementTree as ET

    body = _xhtml(tmp_path, _formula())
    root = ET.fromstring(body)
    visible = "".join(root.itertext())
    assert "$$" not in visible
    assert "\\frac" not in visible
    assert "a/b" in visible


def test_the_epub_keeps_the_latex_available_for_copying(tmp_path: Path) -> None:
    body = _xhtml(tmp_path, _formula())
    assert 'data-latex="\\frac{a}{b}"' in body
    assert 'title="\\frac{a}{b}"' in body


def test_the_epub_ships_no_script(tmp_path: Path) -> None:
    """An EPUB that needed a script to render would fail in most readers."""
    body = _xhtml(tmp_path, _formula())
    assert "<script" not in body


def test_the_html_companion_still_emits_katex_markup(tmp_path: Path) -> None:
    """The change must not regress the page that does have a renderer."""
    output = tmp_path / "view.html"
    compose_html([_formula()], {}, output)
    text = output.read_text(encoding="utf-8")
    assert "$$\\frac{a}{b}$$" in text
    assert "renderMathInElement" in text


def test_the_two_renderers_are_selected_explicitly() -> None:
    """One knob, validated, so a typo cannot silently pick a mode."""
    fragment, _placements = render_fragment([_formula()], {}, math_renderer="unicode")
    assert ">a/b</span>" in fragment
    assert "$$" not in fragment

    fragment, _placements = render_fragment([_formula()], {}, math_renderer="katex")
    assert "$$\\frac{a}{b}$$" in fragment

    with pytest.raises(ValueError, match="math_renderer"):
        render_fragment([_formula()], {}, math_renderer="mathml")


def test_an_unknown_command_degrades_to_the_placeholder_not_the_command() -> None:
    """The visible Unicode reading must not leak a raw command name."""
    import xml.etree.ElementTree as ET

    fragment, _placements = render_fragment(
        [_formula(r"\weirdcmd{x}")], {}, math_renderer="unicode"
    )
    visible = "".join(ET.fromstring(f"<root>{fragment}</root>").itertext())
    assert "\\weirdcmd" not in visible
    assert "⟨math⟩" in visible
