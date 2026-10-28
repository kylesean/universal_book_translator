"""Unit tests for the Appendix & Backmatter Layout Manager."""

from __future__ import annotations

import pytest

from ubt.core.backmatter.detector import (
    classify_heading,
    detect_backmatter_manifest,
)
from ubt.core.backmatter.generator import (
    emit_appendix_switch_typst,
    render_backmatter_heading_typst,
    render_backmatter_section_html,
    render_backmatter_skeleton_markdown,
)
from ubt.core.backmatter.models import (
    BackmatterConfig,
    BackmatterKind,
    BackmatterManifest,
    BackmatterSection,
)
from ubt.core.ir.models import BlockType, IRBlock, make_element

pytestmark = pytest.mark.fast


def _mk_block(b_id: str, b_type: BlockType, target: str, level: int = 1) -> IRBlock:
    el = make_element(
        id=b_id,
        spine_index=1,
        block_type=b_type,
        source_text=target,
        level=level,
    )
    return IRBlock(
        id=b_id,
        block_type=b_type,
        source_text=target,
        target_text=target,
        element=el,
    )


def test_classify_heading() -> None:
    # Appendices
    res = classify_heading("附录 A：算法正确性证明")
    assert res is not None
    assert res[0] == BackmatterKind.APPENDIX
    assert res[1] == "A"
    assert "证明" in res[2]

    res_en = classify_heading("Appendix B: Benchmarks")
    assert res_en is not None
    assert res_en[0] == BackmatterKind.APPENDIX
    assert res_en[1] == "B"
    assert "Benchmarks" in res_en[2]

    # Chinese numeral appendix
    res_zh = classify_heading("附录 二：数学推导")
    assert res_zh is not None
    assert res_zh[0] == BackmatterKind.APPENDIX
    assert res_zh[1] == "B"

    # Bare appendix
    res_bare = classify_heading("附录")
    assert res_bare is not None
    assert res_bare[0] == BackmatterKind.APPENDIX

    # Bibliography
    assert classify_heading("References") == (BackmatterKind.BIBLIOGRAPHY, "", "References")
    assert classify_heading("参考文献") == (BackmatterKind.BIBLIOGRAPHY, "", "参考文献")

    # Glossary
    assert classify_heading("Glossary") == (BackmatterKind.GLOSSARY, "", "Glossary")
    assert classify_heading("术语表") == (BackmatterKind.GLOSSARY, "", "术语表")

    # Index
    assert classify_heading("Index") == (BackmatterKind.INDEX, "", "Index")
    assert classify_heading("索引") == (BackmatterKind.INDEX, "", "索引")

    # Afterword
    assert classify_heading("后记与致谢") == (BackmatterKind.AFTERWORD, "", "后记与致谢")

    # Normal chapter (not backmatter)
    assert classify_heading("第 1 章：系统概述") is None
    assert classify_heading("Chapter 3: Raft") is None


def test_detect_backmatter_manifest() -> None:
    blocks = [
        _mk_block("h1", BlockType.HEADING, "第 1 章：导论"),
        _mk_block("p1", BlockType.NARRATIVE, "正文第一段。"),
        _mk_block("h2", BlockType.HEADING, "第 2 章：核心架构"),
        _mk_block("p2", BlockType.NARRATIVE, "正文第二段。"),
        _mk_block("app1", BlockType.HEADING, "附录：形式化规范"),
        _mk_block("p_app1", BlockType.NARRATIVE, "附录正文。"),
        _mk_block("app2", BlockType.HEADING, "附录：性能评测数据"),
        _mk_block("p_app2", BlockType.NARRATIVE, "评测正文。"),
        _mk_block("bib", BlockType.HEADING, "参考文献"),
        _mk_block("p_bib", BlockType.NARRATIVE, "[1] Lamport 1998."),
    ]

    manifest = detect_backmatter_manifest(blocks)
    assert manifest.total_sections == 3
    assert manifest.appendix_count == 2
    assert manifest.has_bibliography is True
    assert manifest.first_appendix_block_id == "app1"
    assert manifest.first_backmatter_block_id == "app1"

    # Consecutive appendices receive letters A, B
    s0 = manifest.sections[0]
    assert s0.kind == BackmatterKind.APPENDIX
    assert s0.letter_index == "A"
    assert s0.heading_block_id == "app1"

    s1 = manifest.sections[1]
    assert s1.kind == BackmatterKind.APPENDIX
    assert s1.letter_index == "B"
    assert s1.heading_block_id == "app2"

    s2 = manifest.sections[2]
    assert s2.kind == BackmatterKind.BIBLIOGRAPHY
    assert s2.letter_index == ""


def test_generator_typst_and_html() -> None:
    cfg = BackmatterConfig(appendix_numbering="A.1")
    switch_code = emit_appendix_switch_typst(cfg)
    assert "#counter(heading).update(0)" in switch_code
    assert '#set heading(numbering: "A.1")' in switch_code

    sec_app = BackmatterSection(
        section_id="bm-app-a",
        kind=BackmatterKind.APPENDIX,
        raw_title="附录：数学证明",
        clean_title="数学证明",
        letter_index="A",
    )
    heading_app = render_backmatter_heading_typst(sec_app, cfg)
    assert "#heading(level: 1)[附录 A：数学证明]" in heading_app

    sec_bib = BackmatterSection(
        section_id="bm-bib",
        kind=BackmatterKind.BIBLIOGRAPHY,
        raw_title="参考文献",
        clean_title="参考文献",
    )
    heading_bib = render_backmatter_heading_typst(sec_bib, cfg)
    assert "#heading(level: 1, numbering: none)[参考文献]" in heading_bib

    # HTML
    html_out = render_backmatter_section_html(sec_app, "<p>Proofs...</p>")
    assert (
        '<section class="ubt-backmatter ubt-appendix" id="bm-app-a" role="doc-appendix">'
        in html_out
    )
    assert "<p>Proofs...</p>" in html_out

    # Markdown skeleton
    md_skel = render_backmatter_skeleton_markdown([sec_app, sec_bib])
    assert "# 附录 A：数学证明" in md_skel
    assert "# 参考文献" in md_skel


def test_models_to_dict() -> None:
    cfg = BackmatterConfig()
    d = cfg.to_dict()
    assert d["enabled"] is True
    assert d["appendix_numbering"] == "A.1"

    manifest = BackmatterManifest()
    m_d = manifest.to_dict()
    assert m_d["total_sections"] == 0
    assert m_d["appendix_count"] == 0
