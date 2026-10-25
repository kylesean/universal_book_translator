"""The delivered PDF's *text layer* must carry every character the page draws.

Fast-tier tests can verify the font set, the emitted font list and the subset
spec, and all of them passed while the delivered PDF was shipping unsearchable
text. That is the failure this file exists for.

**The defect.** With Typst 0.15.1 and ``--ignore-system-fonts``, a bundled CJK
subset holding ideographs plus *both* kana and hangul (33,624 glyphs) typesets
``：``, ``；`` and ``？`` as CID 0. The page renders correctly — a reader sees
the punctuation — but the PDF's CIDSet declares only CID 0, so those three
characters are absent from the text layer. Copy and search fail on text that is
plainly visible, and the artifact check reports the affected blocks as missing.
Neither block alone triggers it (ideographs+hangul at 32,597 glyphs is clean;
ideographs+kana at 21,799 is clean) and the un-subsetted 65,535-glyph face is
clean, so the fault is in the subsetting path, not the font. The fix is that
the SC cut no longer carries kana/hangul — they belong to the JP/KR faces.

**Why the assertion is on extracted text, not on the PDF's appearance.** Every
earlier check in this suite looked at what was *emitted* (the font list, the
covers routing, the subset spec). The bug is invisible in all of them: the
markup is right, the font is right, the glyphs exist, the page looks right. Only
reading the characters back out of the artifact catches it — which is also what
a user does when they search the delivered book.

Slow tier: it shells out to ``typst`` and reads the PDF back.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ubt.adapters.pdf.font_probe import resolve_for_render
from ubt.adapters.pdf.typst_compile import typst_compile
from ubt.render.typst.document_renderer import TypstDocumentRenderer
from ubt.render.typst.fonts import BUNDLED_FONTS

#: Punctuation that was lost from the text layer, one per affected code point.
_AFFECTED = "：；？"


def _extract_text(pdf: Path) -> str:
    """The artifact's text layer, read with ``pypdf``.

    **Deliberately not the project's own extractor.** The audit's reader
    (``artifact_text._read_all_text_pages``, pdfium) and the line extractor
    (``textgeom``) both turn a NUL glyph into whitespace, so a page whose
    punctuation was drawn as CID 0 reads back as a *space* — present-looking,
    silently wrong. Measured on the broken artifact: pypdf reports three NULs
    where pdfium reports clean text. A regression test that used the audit's own
    reader would therefore pass against the exact defect it exists to catch.

    ``pypdf`` is already a test dependency (``tests/pdf_builders.py`` writes
    fixtures with it), so this adds no requirement.
    """
    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)


def _require_bundled_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the renderer at a real font set, skipping when none is available.

    The suite's autouse ``clean_ubt_env`` deletes every ``UBT_*`` variable, so a
    developer's export cannot leak in; a test that needs the bundled set must ask
    for it explicitly. Skipping (not failing) mirrors the other slow-tier render
    tests: the set is fetched on demand, not committed.
    """
    from ubt.render.typst.fonts import DEFAULT_FONT_DIR, bundled_font_dir, font_dir

    # `font_dir()` honours a caller's UBT_FONT_DIR (an intentionally broken set,
    # used to prove this test bites); `bundled_font_dir()` only answers "is the
    # set complete?" for whatever that resolves to. The suite's autouse
    # ``clean_ubt_env`` has already deleted the variable by the time this runs,
    # so the directory is the default unless the caller restores it below.
    candidate = font_dir()
    if bundled_font_dir() is None:
        if candidate == DEFAULT_FONT_DIR:
            pytest.skip(
                f"bundled font set not installed at {DEFAULT_FONT_DIR}; "
                "run `python scripts/fetch_fonts.py`"
            )
        pytest.skip(f"the font set at {candidate} is incomplete")
    # Re-assert through the environment: `_should_isolate` and `typst_compile`
    # read it independently, and they must agree with what this test renders
    # against or the assertion is about a different set than the caller named.
    monkeypatch.setenv("UBT_FONT_DIR", str(candidate))
    return candidate


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("typst") is None, reason="typst is not installed")
def test_cjk_punctuation_survives_into_the_text_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every character a CJK page draws must come back out of the artifact.

    Runs through the real renderer and the real compile flags, so the font
    stack, the isolation decision and the subset are exercised exactly as a
    delivery uses them. A NUL in the extracted text is the specific signature:
    the CID-0 defect surfaces as ``\\x00`` in place of the glyph, not as a
    dropped character, which is why counting missing characters alone would
    miss it.

    ``UBT_FONT_DIR`` is set here, not inherited: the suite's autouse
    ``clean_ubt_env`` deletes every ``UBT_*`` variable before each test, so no
    export reaches this assertion. That is correct for reproducibility but means
    a hand-built broken set cannot be injected from the outside — which is why
    ``test_the_sc_face_does_not_carry_both_kana_and_hangul`` below asserts the
    trigger directly on the file instead of relying on an external override.
    """
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
    """The realistic case: prose with Latin, math-ish symbols and every mark.

    A translation engine's deliverable is searched, not just read. This pins the
    property a reader depends on — a phrase from the middle of a paragraph can
    be found in the produced PDF — on a paragraph shaped like real content
    (中文 + English + 数字 + 全角标点) rather than a punctuation stress string.
    """
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
    # A PDF text layer carries the layout's own line breaks, and CJK prose
    # breaks between any two characters — that is correct pagination, not loss.
    # Comparing on a whitespace-stripped basis is what a search index does.
    flat = "".join(extracted.split())
    assert "\x00" not in extracted
    for probe in ("准确率达到94.5%", "可逆效应", "参考文献[12]"):
        assert probe in flat, f"a reader could not search for {probe!r} in the delivered PDF"


def _read_cmap(path: str) -> set[int]:
    """Codepoints in an OpenType font's Unicode cmap, using only stdlib.

    ``fontTools`` is not a runtime (or test) dependency here — it is needed only
    by the fetch script — so a guard that reads the shipped files cannot import
    it. This parses just enough of the sfnt table directory and the two cmap
    subtable formats an OTF uses for Unicode: format 4 (BMP) and format 12
    (supplementary planes).
    """
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


# --------------------------------------------------------------------------- #
# The trigger, asserted directly on the shipped files (fast tier)
# --------------------------------------------------------------------------- #


@pytest.mark.fast
def test_the_sc_face_does_not_carry_both_kana_and_hangul() -> None:
    """The shipped SC faces must not carry kana *and* hangul together.

    This is the fast-tier guard for the defect the slow tests above detect. It
    asserts the trigger on the fetched files rather than on a render, because
    the defect depends on the *font contents* and nothing in the render path can
    reveal it: the page looks correct, the font list is correct, and both of the
    project's own PDF readers turn the CID-0 glyph into whitespace.

    Measured on the old set (Typst 0.15.1, ``--ignore-system-fonts``):
    ideographs + kana + hangul = 33,624 glyphs → ``：``, ``；`` and ``？`` render
    as CID 0. Ideographs + hangul alone (32,597) is clean, ideographs + kana
    alone (21,799) is clean, and the un-subsetted 65,535-glyph face is clean.

    The SC cut therefore carries neither block; kana and hangul belong to the JP
    and KR faces, which each own their script. Chinese coverage is unaffected
    (ideographs 100%, CJK punctuation 100%, fullwidth forms 93.8%, ASCII 100%).
    """
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
        # And it must still be a complete Chinese face.
        han = sum(1 for cp in range(0x4E00, 0xA000) if cp in cmap)
        assert han > 20_000, f"{face.filename} covers only {han} ideographs"
        for cp in (0xFF1A, 0xFF1B, 0xFF1F, 0x3002, 0xFF0C):
            assert cp in cmap, f"{face.filename} is missing U+{cp:04X}"
