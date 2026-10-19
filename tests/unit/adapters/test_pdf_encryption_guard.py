"""An encrypted PDF fails with an actionable parse error, not engine noise.

Docling surfaced a password-protected file as ``ConversionError: ... could not
load document`` plus a burst of pdf_oxide decryption noise; the operator could
not tell it from a corrupt file. The guard runs at manifest extraction, before
engine dispatch, so both PDF engines and the translate CLI share one message.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pikepdf
import pytest
from pdf_builders import write_text_pdf
from typer.testing import CliRunner

from ubt.adapters.base import BasePDFEngineAdapter
from ubt.adapters.pdf.docling_adapter import DoclingPDFAdapter
from ubt.adapters.pdf.pdfium_adapter import PDFiumAdapter
from ubt.cli.main import app
from ubt.core.exceptions import DocumentParseError

pytestmark = pytest.mark.fast

runner = CliRunner()


def _output(result: object) -> str:
    """stdout+stderr: typer.echo(err=True) lands on stderr under click >= 8.2."""
    text = getattr(result, "output", "") or ""
    return text + (getattr(result, "stderr", "") or "")


def _encrypted_pdf(tmp_path: Path) -> Path:
    source = write_text_pdf(tmp_path / "source.pdf", [["Chapter One", "Body text."]])
    encrypted = tmp_path / "encrypted.pdf"
    with pikepdf.open(str(source)) as pdf:
        pdf.save(
            str(encrypted),
            encryption=pikepdf.Encryption(owner="owner", user="user", R=6),
        )
    return encrypted


@pytest.mark.parametrize("adapter_cls", [PDFiumAdapter, DoclingPDFAdapter])
def test_manifest_rejects_an_encrypted_pdf(
    tmp_path: Path, adapter_cls: type[BasePDFEngineAdapter]
) -> None:
    pdf = _encrypted_pdf(tmp_path)

    with pytest.raises(DocumentParseError) as excinfo:
        asyncio.run(adapter_cls().extract_manifest(pdf))

    message = str(excinfo.value)
    assert "password-protected" in message
    assert "decrypt" in message
    assert str(pdf) in message


@pytest.mark.parametrize("adapter_cls", [PDFiumAdapter, DoclingPDFAdapter])
def test_parse_stream_rejects_an_encrypted_pdf_before_engine_dispatch(
    tmp_path: Path, adapter_cls: type[BasePDFEngineAdapter]
) -> None:
    pdf = _encrypted_pdf(tmp_path)

    async def collect() -> None:
        async for _ in adapter_cls().parse_stream(pdf):
            pass

    with pytest.raises(DocumentParseError, match="password-protected"):
        asyncio.run(collect())


def test_translate_cli_reports_the_encryption_without_conversion_noise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("UBT_DB_DIR", str(tmp_path / "ledgers"))
    monkeypatch.setenv("UBT_CACHE_DIR", str(tmp_path / "cache"))
    pdf = _encrypted_pdf(tmp_path)

    result = runner.invoke(
        app,
        [
            "translate",
            str(pdf),
            "-l",
            "zh",
            "--dry-run",
            "--pdf-engine",
            "pdfium",
            "-o",
            str(tmp_path / "out"),
        ],
    )

    output = _output(result)
    assert result.exit_code == 1
    assert "password-protected" in output
    assert "decrypt" in output
    assert "ConversionError" not in output
    assert "Decryption failed" not in output
    assert "Traceback" not in output
