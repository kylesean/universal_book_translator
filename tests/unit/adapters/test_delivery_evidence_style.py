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
    assert samples[0].role == "body"


def test_the_reader_prefers_exact_sizes_over_the_ink_proxy(tmp_path: Path) -> None:
    path = tmp_path / "one_page.pdf"
    with pikepdf.Pdf.new() as pdf:
        _page(pdf, body_size=10.0, caption_size=8.0)
        pdf.save(path)

    evidence = read_artifact_evidence(path, contract="faithful", blocks=(), read_geometry=True)

    assert len(evidence.style_samples) == 1
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
        for row in range(6):
            ops.append((["/F1", 1], pikepdf.Operator("Tf")))
            ops.append(([8.0, 0, 0, 8.0, 50, y], pikepdf.Operator("Tm")))
            ops.append(
                ([pikepdf.String(f"| row{row} | 0.050 | 0.066 |".encode())], pikepdf.Operator("Tj"))
            )
            y -= 14.0
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


def test_the_artifacts_measured_size_beats_a_planned_sample(tmp_path: Path) -> None:
    # The renderer planned 12pt, but the fixed-box fit ladder drew the body at
    # 10pt. Auditing the plan would read the squeezed page as a pass.
    from ubt.core.engine.delivery_contract import StyleSample

    path = tmp_path / "planned_vs_real.pdf"
    with pikepdf.Pdf.new() as pdf:
        _page(pdf, body_size=10.0, caption_size=8.0)
        pdf.save(path)

    planned = (StyleSample(page=1, size_pt=12.0, group_id="group-A"),)
    evidence = read_artifact_evidence(
        path, contract="faithful", blocks=(), read_geometry=True, style_samples=planned
    )

    assert len(evidence.style_samples) == 1
    assert evidence.style_samples[0].size_pt == pytest.approx(10.0)
    # The plan still owns group membership, even though its number lost.
    assert evidence.style_samples[0].group_id == "group-A"


def test_a_planned_sample_is_the_read_failure_fallback(tmp_path: Path) -> None:
    # No measurable text (blank page): the plan is all we have, so keep it
    # rather than reporting a faithful artifact as style_unmeasured.
    from ubt.core.engine.delivery_contract import StyleSample

    path = tmp_path / "blank_with_plan.pdf"
    with pikepdf.Pdf.new() as pdf:
        pdf.add_blank_page(page_size=(400, 300))
        pdf.save(path)

    planned = (StyleSample(page=1, size_pt=11.0, group_id="group-A"),)
    evidence = read_artifact_evidence(
        path, contract="faithful", blocks=(), read_geometry=True, style_samples=planned
    )

    assert len(evidence.style_samples) == 1
    assert evidence.style_samples[0].size_pt == pytest.approx(11.0)
