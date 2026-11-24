"""Unit tests for resolved codec/pipeline limitations:

1. Adaptive chapter splitting in Markdown and DOCX codecs (highest heading level + length chunking).
2. Password-protected PDF decryption via ``UBTConfig.pdf_password`` / ``decrypt_pdf_to_dir``.
3. Cross-format export from Markdown/DOCX codecs directly to Typst PDF.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pikepdf
import pytest
from docx import Document as DocxDocument
from pdf_builders import write_text_pdf
from pydantic import SecretStr

from ubt.adapters.codec_adapter import DocumentCodecAdapter
from ubt.adapters.pdf.encryption import decrypt_pdf_to_dir, reject_encrypted_pdf
from ubt.codecs.docx.codec import DOCXCodec
from ubt.codecs.markdown.codec import MarkdownCodec
from ubt.core.assess import AssessmentError, assess_document_async
from ubt.core.config import UBTConfig
from ubt.core.exceptions import DocumentParseError
from ubt.core.ir.models import BlockStatus, BookManifest, ChapterMeta, IRBlock
from ubt.model.ast import Heading, Paragraph
from ubt.model.span import Span

pytestmark = pytest.mark.fast


def test_markdown_adaptive_chapters_split_on_h2_when_no_h1(tmp_path: Path) -> None:
    md = tmp_path / "h2_only.md"
    md.write_text(
        "## First Section\n\nAlpha paragraph.\n\n## Second Section\n\nBeta paragraph.\n",
        encoding="utf-8",
    )
    _, units = MarkdownCodec().extract(md)
    chapters = list(dict.fromkeys(u.chapter_id for u in units))
    assert chapters == ["ch001", "ch002"]


def test_markdown_adaptive_chapters_auto_chunk_long_headingless_doc(tmp_path: Path) -> None:
    md = tmp_path / "long_plain.txt"
    para = "This is a long prose paragraph without any headings at all. " * 30
    md.write_text("\n\n".join(para for _ in range(16)), encoding="utf-8")
    _, units = MarkdownCodec().extract(md)
    chapters = list(dict.fromkeys(u.chapter_id for u in units))
    assert len(chapters) >= 2


def test_docx_adaptive_chapters_split_on_h2_when_no_h1(tmp_path: Path) -> None:
    path = tmp_path / "h2_only.docx"
    doc = DocxDocument()
    doc.add_heading("Part One", level=2)
    doc.add_paragraph("First body paragraph.")
    doc.add_heading("Part Two", level=2)
    doc.add_paragraph("Second body paragraph.")
    doc.save(str(path))

    _, units = DOCXCodec().extract(path)
    chapters = list(dict.fromkeys(u.chapter_id for u in units))
    assert chapters == ["document_ch001", "document_ch002"]


def test_docx_adaptive_chapters_auto_chunk_long_headingless_doc(tmp_path: Path) -> None:
    path = tmp_path / "long_plain.docx"
    doc = DocxDocument()
    para = "Long document paragraph without Word heading styles. " * 30
    for _ in range(16):
        doc.add_paragraph(para)
    doc.save(str(path))

    _, units = DOCXCodec().extract(path)
    chapters = list(dict.fromkeys(u.chapter_id for u in units))
    assert len(chapters) >= 2


def _make_encrypted_pdf(tmp_path: Path, password: str) -> Path:
    plain = write_text_pdf(
        tmp_path / "plain.pdf",
        [("Encrypted Book", "This page was protected by a user password.")],
    )
    enc = tmp_path / "secret.pdf"
    with pikepdf.open(str(plain)) as pdf:
        pdf.save(
            str(enc),
            encryption=pikepdf.Encryption(user=password, owner="owner-secret", R=6),
        )
    return enc


def test_encrypted_pdf_probe_and_decrypt_helper(tmp_path: Path) -> None:
    enc = _make_encrypted_pdf(tmp_path, "open-sesame")

    with pytest.raises(DocumentParseError, match="--pdf-password"):
        reject_encrypted_pdf(enc)

    with pytest.raises(DocumentParseError, match="Incorrect PDF password"):
        reject_encrypted_pdf(enc, password="wrong")

    reject_encrypted_pdf(enc, password="open-sesame")

    out_dir = tmp_path / "decrypted"
    decrypted = decrypt_pdf_to_dir(enc, "open-sesame", out_dir)
    assert decrypted.exists()
    assert decrypted.name == enc.name
    with pikepdf.open(str(decrypted)) as pdf:
        assert not pdf.is_encrypted
        assert len(pdf.pages) == 1


async def test_assess_supports_pdf_password(tmp_path: Path) -> None:
    enc = _make_encrypted_pdf(tmp_path, "open-sesame")

    base_cfg = UBTConfig().model_copy(update={"db_dir": tmp_path / "ledgers"})
    with pytest.raises(AssessmentError) as exc_info:
        await assess_document_async(enc, base_cfg)
    assert exc_info.value.code == "ENCRYPTED"

    wrong_cfg = base_cfg.model_copy(update={"pdf_password": SecretStr("bad-pw")})
    with pytest.raises(AssessmentError) as bad_info:
        await assess_document_async(enc, wrong_cfg)
    assert bad_info.value.code == "ENCRYPTED"

    good_cfg = base_cfg.model_copy(update={"pdf_password": SecretStr("open-sesame")})
    report = await assess_document_async(enc, good_cfg)
    assert report.document.pages == 1


@pytest.mark.skipif(shutil.which("typst") is None, reason="typst binary not installed")
async def test_markdown_codec_adapter_exports_typst_pdf(tmp_path: Path) -> None:
    md = tmp_path / "sample.md"
    md.write_text("# Hello\n\nWorld paragraph.\n", encoding="utf-8")
    adapter = DocumentCodecAdapter(md)
    assert ".pdf" in adapter.output_suffixes
    assert ".md" in adapter.output_suffixes

    manifest = BookManifest(
        doc_id="md_test",
        title="Sample Book",
        source_path=str(md),
        chapters=[ChapterMeta(chapter_id="ch001", title="ch001", spine_index=0)],
    )
    b1 = IRBlock.from_element(
        Heading(id="md_b00001", spine_index=0, span=Span(page=1), text="Hello", level=1),
    )
    b1.target_text = "你好"
    b1.status = BlockStatus.MTQE_PASSED
    b2 = IRBlock.from_element(
        Paragraph(id="md_b00002", spine_index=1, span=Span(page=1), text="World paragraph."),
    )
    b2.target_text = "世界段落。"
    b2.status = BlockStatus.MTQE_PASSED
    blocks = [b1, b2]
    out_pdf = tmp_path / "sample_zh.pdf"
    written = await adapter.render_blocks(
        manifest, blocks, target_lang="zh", output_path=out_pdf, bilingual_mode="mono"
    )
    assert written == out_pdf
    assert out_pdf.exists()
    assert out_pdf.read_bytes().startswith(b"%PDF-")
