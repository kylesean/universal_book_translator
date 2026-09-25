"""Unit tests for the optional pdf_oxide rasterizer and its fallback legs.

The real ``pdf_oxide`` wheel is never required here: a fake module is injected
into ``sys.modules``, mirroring the lazy in-function ``from pdf_oxide import
PdfDocument`` of :mod:`ubt.adapters.pdf.oxide_render`.
"""

from __future__ import annotations

import io
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from ubt.adapters.pdf import oxide_render

# 8-byte-ish minimal valid PNG placeholder; content never decoded here.
_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"fake-bytes"


class _FakeDoc:
    """Records every render_page call; ``page_count`` fixed; optional failure."""

    calls: list[tuple[int, int]] = []
    constructed = 0
    fail = False

    def __init__(self, path: str) -> None:
        _FakeDoc.constructed += 1
        self.path = path

    @property
    def page_count(self) -> int:
        return 3

    def render_page(self, page: int, dpi: int = 72) -> bytes:
        if _FakeDoc.fail:
            raise RuntimeError("oxide exploded")
        _FakeDoc.calls.append((page, dpi))
        return _FAKE_PNG


@pytest.fixture()
def fake_oxide(monkeypatch: pytest.MonkeyPatch) -> type[_FakeDoc]:
    mod = types.ModuleType("pdf_oxide")
    mod.PdfDocument = _FakeDoc  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pdf_oxide", mod)
    _FakeDoc.calls = []
    _FakeDoc.constructed = 0
    _FakeDoc.fail = False
    return _FakeDoc


def test_render_page_png_maps_to_zero_based(fake_oxide: type[_FakeDoc], tmp_path: Path) -> None:
    data = oxide_render.render_page_png(tmp_path / "x.pdf", 3, 96)
    assert data == _FAKE_PNG
    assert fake_oxide.calls == [(2, 96)]  # 1-based UBT page -> 0-based binding


def test_render_page_png_opens_document_per_call(
    fake_oxide: type[_FakeDoc], tmp_path: Path
) -> None:
    oxide_render.render_page_png(tmp_path / "x.pdf", 1, 72)
    oxide_render.render_page_png(tmp_path / "x.pdf", 2, 72)
    assert fake_oxide.constructed == 2  # &mut self pyclass: never share a doc


def test_render_page_png_out_of_range(fake_oxide: type[_FakeDoc], tmp_path: Path) -> None:
    assert oxide_render.render_page_png(tmp_path / "x.pdf", 0, 72) is None
    assert oxide_render.render_page_png(tmp_path / "x.pdf", 4, 72) is None
    assert fake_oxide.calls == []


def test_render_page_png_swallows_renderer_failure(
    fake_oxide: type[_FakeDoc], tmp_path: Path
) -> None:
    fake_oxide.fail = True
    assert oxide_render.render_page_png(tmp_path / "x.pdf", 1, 72) is None


def test_render_page_png_without_extra_returns_none(tmp_path: Path) -> None:
    assert oxide_render.render_page_png(tmp_path / "x.pdf", 1, 72) is None


def test_write_page_png_persists_bytes(fake_oxide: type[_FakeDoc], tmp_path: Path) -> None:
    out = oxide_render.write_page_png(tmp_path / "x.pdf", 2, 72, tmp_path)
    assert out is not None and out.name == "p2.png"
    assert out.read_bytes() == _FAKE_PNG


def test_render_pages_to_png_contract(fake_oxide: type[_FakeDoc], tmp_path: Path) -> None:
    assert oxide_render.render_pages_to_png(tmp_path / "x.pdf", [], 72) == {}
    out = oxide_render.render_pages_to_png(tmp_path / "x.pdf", [1, 2], 72, work_dir=tmp_path)
    assert sorted(out) == [1, 2]
    assert out[1].name == "p1.png"


def test_render_pages_to_png_cleans_own_tmpdir_when_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_oxide: type[_FakeDoc]
) -> None:
    fake_oxide.fail = True
    import tempfile

    created: list[Path] = []
    real_mkdtemp = tempfile.mkdtemp

    def fake_mkdtemp(*args: str, **kwargs: str) -> str:
        name = str(real_mkdtemp(*args, **kwargs))
        created.append(Path(name))
        return name

    monkeypatch.setattr(tempfile, "mkdtemp", fake_mkdtemp)
    assert oxide_render.render_pages_to_png(tmp_path / "x.pdf", [1], 72) == {}
    assert created and not created[0].exists()


def test_is_available_follows_find_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: object() if name == "pdf_oxide" else real(name, *a, **k),
    )
    assert oxide_render.is_available() is True
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "pdf_oxide" else real(name, *a, **k),
    )
    assert oxide_render.is_available() is False


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# visual_gate: oxide-only rendering (the pdftoppm subprocess leg was retired)
# ---------------------------------------------------------------------------


def test_visual_gate_renders_all_pages_via_oxide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.adapters.pdf import visual_gate as vg

    def fake_write(pdf_path: Path, page: int, dpi: int, out_dir: Path) -> Path:
        p = out_dir / f"p{page}.png"
        p.write_bytes(_FAKE_PNG)
        return p

    def _boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("no subprocess may render pages any more")

    monkeypatch.setattr(oxide_render, "write_page_png", fake_write)
    monkeypatch.setattr(subprocess, "run", _boom)
    out = vg.render_pages_to_png(tmp_path / "x.pdf", [1, 2], dpi=72, work_dir=tmp_path)
    assert set(out) == {1, 2}


def test_visual_gate_render_failure_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ubt.adapters.pdf import visual_gate as vg

    monkeypatch.setattr(oxide_render, "write_page_png", lambda *a, **k: None)
    assert vg.render_pages_to_png(tmp_path / "x.pdf", [1], work_dir=tmp_path) == {}


# ---------------------------------------------------------------------------
# doctor reporting
# ---------------------------------------------------------------------------


def test_doctor_reports_oxide_raster_row(monkeypatch: pytest.MonkeyPatch) -> None:
    from typer.testing import CliRunner

    from ubt.cli.main import app

    monkeypatch.setenv("UBT_LLM_API_KEY", "test-key-12345678")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = CliRunner().invoke(app, ["doctor"])
    # pdf_oxide is a base dependency: the raster row must always be reported.
    assert "PDF raster (oxide)" in result.stdout


# ---------------------------------------------------------------------------
# svg_diagram: oxide-only page raster
# ---------------------------------------------------------------------------


def _real_png_bytes(width: int = 100, height: int = 100) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


def test_svg_diagram_renders_via_oxide_no_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("PIL")
    from ubt.adapters.pdf import svg_diagram

    monkeypatch.setattr(
        oxide_render,
        "render_page_png",
        lambda p, page, dpi: _real_png_bytes(2000, 2500),
    )

    def _boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("pdftoppm must not run — the subprocess leg is gone")

    monkeypatch.setattr(subprocess, "run", _boom)
    out = tmp_path / "crop.png"
    got = svg_diagram.render_diagram_png(
        tmp_path / "x.pdf",
        page_no=1,
        bbox_bottomup=(10.0, 10.0, 100.0, 100.0),
        page_height=600.0,
        out_path=out,
    )
    assert got is not None and out.exists()


def test_svg_diagram_render_failure_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("PIL")
    from ubt.adapters.pdf import svg_diagram

    monkeypatch.setattr(oxide_render, "render_page_png", lambda *a, **k: None)

    def _boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("no subprocess fallback exists any more")

    monkeypatch.setattr(subprocess, "run", _boom)
    got = svg_diagram.render_diagram_png(
        tmp_path / "x.pdf",
        page_no=1,
        bbox_bottomup=(10.0, 10.0, 100.0, 100.0),
        page_height=600.0,
        out_path=tmp_path / "crop.png",
    )
    assert got is None
