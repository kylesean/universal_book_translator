from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ubt.adapters.pdf.font_probe import resolve_for_render
from ubt.adapters.pdf.typst_compile import typst_compile
from ubt.render.typst.document_renderer import TypstDocumentRenderer
from ubt.render.typst.fonts import BUNDLED_FONTS

_AFFECTED = "：；？"


def _extract_text(pdf: Path) -> str:
    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)


def _require_bundled_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from ubt.render.typst.fonts import DEFAULT_FONT_DIR, bundled_font_dir, font_dir

    candidate = font_dir()
    if bundled_font_dir() is None:
        if candidate == DEFAULT_FONT_DIR:
            pytest.skip(
                f"bundled font set not installed at {DEFAULT_FONT_DIR}; "
                "run `python scripts/fetch_fonts.py`"
            )
        pytest.skip(f"the font set at {candidate} is incomplete")
    monkeypatch.setenv("UBT_FONT_DIR", str(candidate))
    return candidate


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("typst") is None, reason="typst is not installed")
def test_cjk_punctuation_survives_into_the_text_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _require_bundled_set(tmp_path, monkeypatch)

    samples = {
        "zh": "这是测试：包含冒号、分号；以及问号？还有，逗号。",
        "ja": "これはテストです：コロン、セミコロン；疑問符？それと、読点。",
        "ko": "이것은 테스트입니다：콜론、세미콜론；물음표？그리고, 쉼표.",
    }
    for lang, body in samples.items():
        source = TypstDocumentRenderer().preamble(lang) + "\n" + body + "\n"
        typ_path = tmp_path / f"{lang}.typ"
        typ_path.write_text(source, encoding="utf-8")
        typst_compile(str(typ_path), str(tmp_path / f"{lang}.pdf"), target_lang=lang)

        extracted = _extract_text(tmp_path / f"{lang}.pdf")
        assert "\x00" not in extracted, (
            f"{lang}: the artifact's text layer contains NUL — a glyph was drawn "
            f"as CID 0 (the emitted stack was {resolve_for_render(lang).families})"
        )
        missing = sorted({c for c in body if not c.isspace() and c not in extracted})
        assert not missing, f"{lang}: characters drawn but absent from the text layer: {missing}"
        assert directory.is_dir()


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("typst") is None, reason="typst is not installed")
def test_a_mixed_chinese_paragraph_stays_searchable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_bundled_set(tmp_path, monkeypatch)

    body = (
        "研究方法：本文提出一种基于上下文变换（context transformation）的模型；"
        "其核心是“可逆效应”，即每个变换都携带逆操作？我们在第 3.1 节给出定义。"
        "实验表明：准确率达到 94.5%，优于基线（82.3%）。参考文献 [12]、[13]。"
    )
    source = TypstDocumentRenderer().preamble("zh") + "\n" + body + "\n"
    typ_path = tmp_path / "mixed.typ"
    typ_path.write_text(source, encoding="utf-8")
    typst_compile(str(typ_path), str(tmp_path / "mixed.pdf"), target_lang="zh")

    extracted = _extract_text(tmp_path / "mixed.pdf")
    flat = "".join(extracted.split())
    assert "\x00" not in extracted
    for probe in ("准确率达到94.5%", "可逆效应", "参考文献[12]"):
        assert probe in flat, f"a reader could not search for {probe!r} in the delivered PDF"


def _read_cmap(path: str) -> set[int]:
    import struct

    data = Path(path).read_bytes()
    if data[:4] == b"ttcf":
        raise ValueError(f"{path}: a collection needs a face index")
    num_tables = struct.unpack_from(">H", data, 4)[0]
    cmap_offset = None
    for i in range(num_tables):
        rec = 12 + i * 16
        tag = data[rec : rec + 4]
        if tag == b"cmap":
            cmap_offset = struct.unpack_from(">I", data, rec + 8)[0]
            break
    if cmap_offset is None:
        raise ValueError(f"{path}: no cmap table")

    subtables = struct.unpack_from(">H", data, cmap_offset + 2)[0]
    points: set[int] = set()
    for i in range(subtables):
        rec = cmap_offset + 4 + i * 8
        offset = cmap_offset + struct.unpack_from(">I", data, rec + 4)[0]
        fmt = struct.unpack_from(">H", data, offset)[0]
        if fmt == 4:
            seg_x2 = struct.unpack_from(">H", data, offset + 6)[0]
            segs = seg_x2 // 2
            ends = struct.unpack_from(f">{segs}H", data, offset + 14)
            starts = struct.unpack_from(f">{segs}H", data, offset + 16 + seg_x2)
            for start, end in zip(starts, ends, strict=False):
                if start != 0xFFFF:
                    points.update(range(start, end + 1))
        elif fmt == 12:
            groups = struct.unpack_from(">I", data, offset + 12)[0]
            for g in range(groups):
                base = offset + 16 + g * 12
                start, end, _gid = struct.unpack_from(">III", data, base)
                points.update(range(start, end + 1))
    return points


@pytest.mark.fast
def test_the_sc_face_does_not_carry_both_kana_and_hangul() -> None:
    from ubt.render.typst.fonts import bundled_font_dir

    directory = bundled_font_dir()
    if directory is None:
        pytest.skip("bundled font set not installed; run `python scripts/fetch_fonts.py`")

    sc_faces = [f for f in BUNDLED_FONTS if "CJKsc" in f.filename]
    assert sc_faces, "the set must ship at least one Simplified-Chinese face"

    for face in sc_faces:
        cmap = _read_cmap(str(directory / face.filename))
        has_kana = any(cp in cmap for cp in range(0x3041, 0x30FF))
        has_hangul = 0xAC00 in cmap or 0x1100 in cmap
        assert not (has_kana and has_hangul), (
            f"{face.filename} carries both kana and hangul, which is the "
            "combination that makes Typst drop CJK punctuation from the PDF text "
            "layer; the SC cut must leave both to the JP/KR faces"
        )
        han = sum(1 for cp in range(0x4E00, 0xA000) if cp in cmap)
        assert han > 20_000, f"{face.filename} covers only {han} ideographs"
        for cp in (0xFF1A, 0xFF1B, 0xFF1F, 0x3002, 0xFF0C):
            assert cp in cmap, f"{face.filename} is missing U+{cp:04X}"
