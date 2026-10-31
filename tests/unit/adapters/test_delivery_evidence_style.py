"""The style sample must be the font size, not the ink height.

``_body_size_from_charboxes`` (the ink-height proxy) conflates script with size:
a CJK glyph fills its em box, a Latin one does not. That is why the exact
content-stream reader was added, and these tests pin it: a page set entirely at
10pt must sample as 10.0 even when its glyphs are taller than its x-height, and
a smaller caption must not become the page's body size.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pikepdf
import pytest

from ubt.adapters.pdf.delivery_evidence import _exact_body_sizes, read_artifact_evidence

pytestmark = pytest.mark.fast


def _page(pdf: pikepdf.Pdf, *, body_size: float, caption_size: float) -> None:
    page = pdf.add_blank_page(page_size=(400, 300))
    page.Resources = pikepdf.Dictionary(
        {
            "/Font": pikepdf.Dictionary(
                {
                    "/F1": pikepdf.Dictionary(
                        {
                            "/Type": pikepdf.Name("/Font"),
                            "/Subtype": pikepdf.Name("/Type1"),
                            "/BaseFont": pikepdf.Name("/Helvetica"),
                        }
                    )
                }
            )
        }
    )
    stream = pikepdf.unparse_content_stream(
        cast(
            Any,
            [
                ([], pikepdf.Operator("BT")),
                (["/F1", 1], pikepdf.Operator("Tf")),
                ([body_size, 0, 0, body_size, 50, 240], pikepdf.Operator("Tm")),
                ([pikepdf.String(b"Body text set at one size")], pikepdf.Operator("Tj")),
                ([caption_size, 0, 0, caption_size, 50, 120], pikepdf.Operator("Tm")),
                ([pikepdf.String(b"cap")], pikepdf.Operator("Tj")),
                ([], pikepdf.Operator("ET")),
            ],
        )
    )
    page.Contents = pdf.make_stream(stream)


def test_body_size_is_the_character_weighted_mode_of_the_content_stream(tmp_path: Path) -> None:
    path = tmp_path / "two_pages.pdf"
    with pikepdf.Pdf.new() as pdf:
        _page(pdf, body_size=10.0, caption_size=8.0)
        _page(pdf, body_size=12.0, caption_size=8.0)
        pdf.save(path)

    samples = _exact_body_sizes(path)

    assert [sample.page for sample in samples] == [1, 2]
    assert [sample.size_pt for sample in samples] == pytest.approx([10.0, 12.0])
    # A caption on the page does not become the page's body size: 20 body
    # characters outweigh 3 caption characters.
    assert samples[0].role == "body"


def test_the_reader_prefers_exact_sizes_over_the_ink_proxy(tmp_path: Path) -> None:
    path = tmp_path / "one_page.pdf"
    with pikepdf.Pdf.new() as pdf:
        _page(pdf, body_size=10.0, caption_size=8.0)
        pdf.save(path)

    evidence = read_artifact_evidence(path, contract="faithful", blocks=(), read_geometry=True)

    assert len(evidence.style_samples) == 1
    # The proxy would report an ink height (≈5-8pt for Helvetica); the exact
    # reader reports the size the page was actually set at.
    assert evidence.style_samples[0].size_pt == pytest.approx(10.0)


def test_a_page_without_show_operators_has_no_style_sample(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page(page_size=(400, 300))
        pdf.save(path)

    assert _exact_body_sizes(path) == ()


def test_an_unreadable_artifact_falls_back_without_raising(tmp_path: Path) -> None:
    path = tmp_path / "not_a_pdf.pdf"
    path.write_bytes(b"not a pdf")

    assert _exact_body_sizes(path) == ()
    evidence = read_artifact_evidence(path, contract="faithful", blocks=(), read_geometry=True)
    assert evidence.page_count == 0


def test_a_table_dominated_page_samples_the_body_not_the_grid(tmp_path: Path) -> None:
    """The replay's blocker, as a fixture: a grid whose characters dominate.

    The lmbench page's grid characters (5.3pt) outnumbered the prose three to
    one, the page's size statistic became the *grid's*, and the document read
    as a 4.18pt body drift that blocked the export. A grid is its own style
    group (PRD FR-L4): its long cells text-match the table block, its size
    leaves the vote, and the page samples as the body it also carries.
    """
    body_text = "The body text carries the reading experience across many paragraphs."
    table_rows = "\n".join(f"| row{i} | 0.050 | 0.066 |" for i in range(6))
    path = tmp_path / "table_page.pdf"
    with pikepdf.Pdf.new() as pdf:
        page = pdf.add_blank_page(page_size=(400, 300))
        page.Resources = pikepdf.Dictionary(
            {
                "/Font": pikepdf.Dictionary(
                    {
                        "/F1": pikepdf.Dictionary(
                            {
                                "/Type": pikepdf.Name("/Font"),
                                "/Subtype": pikepdf.Name("/Type1"),
                                "/BaseFont": pikepdf.Name("/Helvetica"),
                            }
                        )
                    }
                )
            }
        )
        ops: list[Any] = [([], pikepdf.Operator("BT"))]
        y = 260.0
        # The grid: more characters than the body, at a smaller size. Each
        # row names itself so the rendered rects text-match the table block.
        for row in range(6):
            ops.append((["/F1", 1], pikepdf.Operator("Tf")))
            ops.append(([8.0, 0, 0, 8.0, 50, y], pikepdf.Operator("Tm")))
            ops.append(
                ([pikepdf.String(f"| row{row} | 0.050 | 0.066 |".encode())], pikepdf.Operator("Tj"))
            )
            y -= 14.0
        # The body prose, at the body size.
        ops.append((["/F1", 1], pikepdf.Operator("Tf")))
        ops.append(([10.0, 0, 0, 10.0, 50, y], pikepdf.Operator("Tm")))
        ops.append(([pikepdf.String(body_text.encode())], pikepdf.Operator("Tj")))
        ops.append(([], pikepdf.Operator("ET")))
        page.Contents = pdf.make_stream(pikepdf.unparse_content_stream(cast(Any, ops)))
        pdf.save(path)

    from ubt.core.engine.delivery_contract import ContractBlock

    blocks = [
        ContractBlock(block_id="t1", text=table_rows, page=1, kind="table"),
        ContractBlock(block_id="b1", text=body_text, page=1, kind="narrative"),
    ]
    evidence = read_artifact_evidence(path, contract="readable", blocks=blocks)

    assert len(evidence.style_samples) == 1
    assert evidence.style_samples[0].size_pt == pytest.approx(10.0)
    from ubt.core.engine.delivery_contract import _style_findings

    assert _style_findings(evidence) == []
