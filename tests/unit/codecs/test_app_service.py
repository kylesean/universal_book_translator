"""End-to-end tests for Layer 3 SurgicalTranslationService & JobSpec."""

from __future__ import annotations

import re
from pathlib import Path

import pikepdf
import pytest

from ubt.app import JobSpec, SurgicalTranslationService
from ubt.codecs.pdf import HeuristicLayoutDetector, NativePDFCodec
from ubt.codecs.protocol import DeliveryMode, TranslationUnit


def _build_pdf_with_math(path: Path) -> Path:
    pdf = pikepdf.new()
    f_body = pdf.make_indirect(
        pikepdf.Dictionary(
            {
                "/Type": pikepdf.Name("/Font"),
                "/Subtype": pikepdf.Name("/Type1"),
                "/BaseFont": pikepdf.Name("/Helvetica"),
            }
        )
    )
    f_math = pdf.make_indirect(
        pikepdf.Dictionary(
            {
                "/Type": pikepdf.Name("/Font"),
                "/Subtype": pikepdf.Name("/Type1"),
                "/BaseFont": pikepdf.Name("/CMMI10"),
            }
        )
    )
    for p in (1, 2):
        page = pdf.add_blank_page(page_size=(612, 792))
        page["/Resources"] = pikepdf.Dictionary(
            {"/Font": pikepdf.Dictionary({"/F1": f_body, "/FM": f_math})}
        )
        stream = (
            f"BT /F1 11 Tf 1 0 0 1 72 700 Tm (Page {p} prose with ) Tj "
            f"/FM 11 Tf 1 0 0 1 180 700 Tm (x) Tj "
            f"/F1 11 Tf 1 0 0 1 192 700 Tm ( inside.) Tj ET\n"
            f"BT /F1 11 Tf 1 0 0 1 72 670 Tm (Next paragraph blocks borrow below.) Tj ET"
        )
        page["/Contents"] = pikepdf.Stream(pdf, stream.encode("ascii"))
    pdf.save(path)
    pdf.close()
    return path


@pytest.mark.fast
def test_job_spec_rejects_reserved_job_ids_and_defaults_to_mono(tmp_path: Path) -> None:
    pdf = tmp_path / "paper.pdf"
    spec = JobSpec(source_path=pdf)
    assert spec.delivery_mode == DeliveryMode.MONO
    assert spec.resolved_output_path().name == "paper_zh_mono.pdf"

    spec_slice = JobSpec(source_path=pdf, delivery_mode=DeliveryMode.INTERLEAVED, pages=[3, 5])
    assert spec_slice.resolved_output_path().name == "paper_zh_interleaved_p3-5.pdf"

    with pytest.raises(ValueError, match="reserved"):
        JobSpec(source_path=pdf, job_id="tm")
    with pytest.raises(ValueError, match="reserved"):
        JobSpec(source_path=pdf, job_id="job_queue")


@pytest.mark.fast
@pytest.mark.asyncio
async def test_surgical_service_runs_pdf_with_overflow_compaction(tmp_path: Path) -> None:
    src = _build_pdf_with_math(tmp_path / "sample.pdf")
    codec = NativePDFCodec(layout_detector=HeuristicLayoutDetector())
    service = SurgicalTranslationService(codec_resolver=lambda _p: codec)
    tag_re = re.compile(r"(<[^>]+>)")

    async def translate_verbose(unit: TranslationUnit, _src: str, _tgt: str) -> str:
        tags = "".join(tag_re.findall(unit.source_marked))
        # Deliberately produce verbose text to exercise overflow compaction
        return (
            "这是一段故意写得很长很详细的中文学术翻译段落文本内容用于测试溢出压缩闭环" * 4
            + f"{tags}结束。"
        )

    async def compact_llm(_sys: str, user: str) -> str:
        # Extract tags from source section and return concise translation
        src_part = user.split("Source:\n", 1)[1].split("\n\nCurrent translation", 1)[0]
        tags = "".join(tag_re.findall(src_part))
        return f"精简学术译文{tags}。"

    spec = JobSpec(
        source_path=src,
        target_lang="zh",
        delivery_mode=DeliveryMode.MONO,
        pages=[1],
        allow_overflow_compaction=True,
    )
    res = await service.run(spec, translate_unit=translate_verbose, compact_complete=compact_llm)
    assert res.report.output_path.exists()
    assert res.report.page_count_output == 1
    assert res.tag_validation_failures == ()
    assert res.compaction is not None
    assert res.compaction.accepted > 0
    # After compaction, overflows should be strictly reduced!
    assert len(res.report.overflows) == 0
