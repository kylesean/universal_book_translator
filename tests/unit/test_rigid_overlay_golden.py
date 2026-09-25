"""End-to-end golden for the anchored/overlay render path.

The offline book baselines only cover Markdown/EPUB (reflow), so this
provides golden validation for the overlay engine. This builds a tiny source PDF with Typst,
derives paragraph blocks from its extracted lines, renders the overlay, and
pins both the invariants (no block silently dropped) and the layout plan
(font size + line breaks per block) so a layout change is visible in review.

Regenerate the plan with ``UBT_UPDATE_GOLDENS=1`` and read the diff.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from ubt.adapters.pdf.rigid.extract import extract_pages
from ubt.adapters.pdf.rigid.typesetter import RigidTypesetter
from ubt.adapters.pdf.rigid.zones import build_zones
from ubt.adapters.pdf.textgeom import LineBox, extract_lines
from ubt.core.ir.models import BlockType, BookManifest, BoundingBox, FlowID, IRBlock

GOLDEN_UPDATE_ENV = "UBT_UPDATE_GOLDENS"
GOLDEN_PATH = Path(__file__).parent.parent / "baselines" / "rigid" / "plan.golden.json"

SOURCE_TYP = (
    "#set page(width: 420pt, height: 320pt, margin: 36pt)\n"
    '#set text(size: 11pt, font: "Liberation Serif")\n\n'
    "First paragraph: the quick brown fox jumps over the lazy dog and keeps running far "
    "across the meadow without stopping at all.\n\n"
    "Second paragraph: another sentence here with enough words to wrap across a couple "
    "of lines in the source layout.\n"
)
TRANSLATION = "这是用于 overlay 端到端验证的中文译文，包含足够长度以铺满若干行。"


def _deps_available() -> bool:
    if shutil.which("typst") is None:
        return False
    try:
        import pikepdf  # noqa: F401
        import pypdfium2  # noqa: F401

        from ubt.adapters.pdf.font_metrics import resolve_cjk_ttc

        resolve_cjk_ttc()
    except Exception:  # capability probe
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _deps_available(), reason="typst + pikepdf + pypdfium2 + CJK font required"
)


def _paragraphs(lines: list[LineBox]) -> list[list[LineBox]]:
    groups: list[list[LineBox]] = []
    for line in lines:
        if groups:
            prev = groups[-1][-1]
            gap = prev.rect[1] - line.rect[3]  # prev bottom - current top
            height = max(1.0, prev.rect[3] - prev.rect[1])
            if gap <= 0.7 * height:
                groups[-1].append(line)
                continue
        groups.append([line])
    return groups


def _build_source_pdf(tmp_path: Path) -> Path:
    typ = tmp_path / "src.typ"
    pdf = tmp_path / "src.pdf"
    typ.write_text(SOURCE_TYP, encoding="utf-8")
    subprocess.run(["typst", "compile", str(typ), str(pdf)], check=True)
    return pdf


def _blocks_from_pdf(pdf: Path) -> list[IRBlock]:
    lines, _size = extract_lines(pdf, 1)
    blocks: list[IRBlock] = []
    for index, group in enumerate(_paragraphs(lines)):
        blocks.append(
            IRBlock(
                id=f"b{index}",
                flow_id=FlowID.MAIN_STORY,
                spine_index=index + 1,
                block_type=BlockType.NARRATIVE,
                source_text=" ".join(line.text for line in group),
                target_text=TRANSLATION,
                bbox=BoundingBox(
                    page=1,
                    x0=min(line.rect[0] for line in group),
                    y0=min(line.rect[1] for line in group),
                    x1=max(line.rect[2] for line in group),
                    y1=max(line.rect[3] for line in group),
                ),
            )
        )
    return blocks


def _layout_plan(pdf: Path, blocks: list[IRBlock]) -> dict[str, object]:
    typesetter = RigidTypesetter()
    pages = extract_pages(pdf, [1], blocks)
    zone_map = build_zones(pages, blocks)
    paints, _report = typesetter._plan_blocks(  # noqa: SLF001 - golden seam
        blocks, zone_map, {page: facts.height for page, facts in pages.items()}
    )
    plan: dict[str, object] = {}
    for zones in paints.values():
        for entry in zones:
            plan[entry.zone.block_id] = {"size": round(entry.size, 2), "text": entry.text}
    return plan


def test_overlay_renders_every_block_without_failclosed_skips(tmp_path: Path) -> None:
    pdf = _build_source_pdf(tmp_path)
    blocks = _blocks_from_pdf(pdf)
    assert len(blocks) == 2
    out = tmp_path / "out.pdf"

    rendered, report = asyncio.run(
        RigidTypesetter().render(
            BookManifest(doc_id="overlay", title="Overlay", source_path=str(pdf)),
            blocks,
            "zh",
            out,
        )
    )
    assert Path(rendered).exists()
    assert report.skipped == [], f"unexpected fail-closed skips: {report.skipped}"
    assert set(report.rendered_blocks) == {b.id for b in blocks}
    # The translation must actually reach the page, not be clipped away by the
    # zone box: Typst owns the line breaking now, so this is the content guard.
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(rendered))
    page = document[0]
    text = page.get_textpage().get_text_range()
    assert TRANSLATION[:20] in text, f"translation missing from rendered page: {text[:200]!r}"


def _measured_with_noto_metrics() -> bool:
    """Is the width-metrics face the one this golden was measured from?

    The plan pins font sizes and line breaks, which come off the resolved .ttc's
    advance widths. A machine measuring with Microsoft YaHei or PingFang produces a
    legitimately different plan, so a diff there is not a layout regression.
    """
    try:
        from ubt.adapters.pdf.font_metrics import resolve_cjk_ttc

        return "noto" in Path(resolve_cjk_ttc()).name.casefold()
    except Exception:  # module-level skipif already covers a missing font
        return False


@pytest.mark.skipif(
    not _measured_with_noto_metrics(),
    reason="plan.golden.json is Noto-measured; close this gap with a per-OS golden",
)
def test_overlay_layout_plan_matches_golden(tmp_path: Path) -> None:
    pdf = _build_source_pdf(tmp_path)
    plan = _layout_plan(pdf, _blocks_from_pdf(pdf))
    if os.environ.get(GOLDEN_UPDATE_ENV) == "1":
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    assert GOLDEN_PATH.exists(), (
        f"missing golden {GOLDEN_PATH}; regenerate with {GOLDEN_UPDATE_ENV}=1"
    )
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert plan == golden, (
        "overlay layout changed; review the diff and update the golden if intended"
    )
