"""Single-page preview endpoint over a real synthetic PDF.

Renders one page of a job's source PDF with the current ledger text through the
delivery compositor, and rasterizes it. Needs the external ``typst`` compiler
(the same dependency as the synthetic render e2e), so it lives in the slow tier
and skips when typst is absent.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pdf_builders import write_text_pdf
from pydantic import SecretStr

from ubt.adapters.pdf.pdfium_adapter import extract_blocks_with_pdfium
from ubt.api.app import create_app
from ubt.core.config import UBTConfig
from ubt.core.engine.ledger import SQLiteJobLedger
from ubt.core.ir.models import BookManifest, ChapterIR

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        shutil.which("typst") is None,
        reason="the typst compiler is not installed; the render path cannot run",
    ),
]

_API_KEY = "preview-test-key"
_AUTH = {"X-API-Key": _API_KEY}
_JOB = "previewjob0001"


def test_page_preview_returns_png(tmp_path: Path) -> None:
    source = write_text_pdf(
        tmp_path / "book.pdf",
        [
            ["The Attention Machine", "Attention maps queries to values."],
            ["Second page text here."],
        ],
    )
    blocks = extract_blocks_with_pdfium(source, None)
    for block in blocks:
        if block.bbox is not None and block.bbox.page == 1 and block.source_text.strip():
            block.target_text = "ATENCION " + block.source_text[:20]

    config = UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
    )
    config.db_dir.mkdir(parents=True, exist_ok=True)
    with SQLiteJobLedger(config.db_dir / f"{_JOB}.sqlite") as ledger:
        ledger.init_job_from_manifest(
            _JOB,
            BookManifest(doc_id=_JOB, title="t", source_path=str(source), target_lang="es"),
        )
        ledger.append_chapter(
            _JOB,
            ChapterIR(doc_id=_JOB, chapter_id="c1", title="C1", spine_index=0, blocks=blocks),
        )

    client = TestClient(create_app(config))
    res = client.get(f"/jobs/{_JOB}/pages/1/preview?dpi=90", headers=_AUTH)
    assert res.status_code == 200, res.text
    assert res.headers["content-type"] == "image/png"
    assert res.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(res.content) > 1000


def test_page_preview_without_source_is_409(tmp_path: Path) -> None:
    config = UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
    )
    config.db_dir.mkdir(parents=True, exist_ok=True)
    # A ledger row with an empty source_path: nothing to compose onto.
    with SQLiteJobLedger(config.db_dir / f"{_JOB}.sqlite") as ledger:
        ledger.init_job_from_manifest(
            _JOB, BookManifest(doc_id=_JOB, title="t", source_path="", target_lang="es")
        )
    client = TestClient(create_app(config))
    res = client.get(f"/jobs/{_JOB}/pages/1/preview", headers=_AUTH)
    assert res.status_code == 409


def test_source_page_returns_png(tmp_path: Path) -> None:
    # The "before" half of the pixel-witness view: the source PDF page, no
    # composition. Needs only pdf_oxide (not typst).
    source = write_text_pdf(tmp_path / "book.pdf", [["Source page one."], ["Source page two."]])

    config = UBTConfig(
        db_dir=tmp_path / "db",
        allowed_dirs=str(tmp_path),
        service_api_key=SecretStr(_API_KEY),
    )
    config.db_dir.mkdir(parents=True, exist_ok=True)
    with SQLiteJobLedger(config.db_dir / f"{_JOB}.sqlite") as ledger:
        ledger.init_job_from_manifest(
            _JOB, BookManifest(doc_id=_JOB, title="t", source_path=str(source), target_lang="es")
        )

    client = TestClient(create_app(config))
    res = client.get(f"/jobs/{_JOB}/pages/1/source?dpi=90", headers=_AUTH)
    assert res.status_code == 200, res.text
    assert res.headers["content-type"] == "image/png"
    assert res.content[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(res.content) > 1000
