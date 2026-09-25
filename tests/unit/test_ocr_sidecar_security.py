"""OCR sidecar hardening: bearer gate and bounded upload."""

from __future__ import annotations

import importlib.util
import io
from pathlib import Path
from types import ModuleType

import pytest
from fastapi.testclient import TestClient
from PIL import Image

_SIDE = Path(__file__).parents[2] / "deploy" / "docker" / "ocr-sidecar" / "server.py"


def _load(monkeypatch: pytest.MonkeyPatch, **env: str) -> ModuleType:
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    spec = importlib.util.spec_from_file_location("ubt_ocr_sidecar", _SIDE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _png_bytes(size: tuple[int, int] = (4, 4)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format="PNG")
    return buf.getvalue()


def test_token_required_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load(monkeypatch, UBT_OCR_SIDECAR_TOKEN="secret-token")
    client = TestClient(module.app)
    assert (
        client.post("/v1/ocr", files={"file": ("x.png", _png_bytes(), "image/png")}).status_code
        == 401
    )
    assert (
        client.post(
            "/v1/ocr",
            files={"file": ("x.png", _png_bytes(), "image/png")},
            headers={"Authorization": "Bearer wrong"},
        ).status_code
        == 401
    )


def test_ocr_requires_loopback_when_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """M5: the no-token posture must be "loopback only", not "open".

    ``require_token`` used to ``return`` when no token was configured, so
    ``/v1/ocr`` served unauthenticated LAN callers while ``/health`` already
    refused them — the asymmetric gate this pins. Both endpoints now run the
    same dependency, so a non-loopback caller gets 403 either way.
    """
    module = _load(monkeypatch)  # no UBT_OCR_SIDECAR_TOKEN: the "open" posture
    # Only the gate is under test here; keep the engine out of it.
    monkeypatch.setattr(module, "get_engine", lambda: ("mock", None))
    remote = TestClient(module.app, client=("203.0.113.9", 41234))

    resp = remote.post("/v1/ocr", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert resp.status_code == 403
    assert "UBT_OCR_SIDECAR_TOKEN" in resp.json()["detail"]
    assert remote.get("/health").status_code == 403

    # Loopback keeps the development-friendly behaviour: no token needed.
    local = TestClient(module.app, client=("127.0.0.1", 50000))
    assert (
        local.post("/v1/ocr", files={"file": ("x.png", _png_bytes(), "image/png")}).status_code
        != 403
    )


def test_health_serves_loopback_without_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loopback fallback must not *block* the probe it exists to allow."""
    module = _load(monkeypatch)
    monkeypatch.setattr(module, "get_engine", lambda: ("rapidocr", None))
    local = TestClient(module.app, client=("::1", 50000))
    resp = local.get("/health")
    assert resp.status_code == 200
    assert resp.json()["engine"] == "rapidocr"


def test_health_still_requires_the_token_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured token gates /health for every caller, loopback included."""
    module = _load(monkeypatch, UBT_OCR_SIDECAR_TOKEN="secret-token")
    monkeypatch.setattr(module, "get_engine", lambda: ("rapidocr", None))
    loopback = TestClient(module.app, client=("127.0.0.1", 50000))
    assert loopback.get("/health").status_code == 401
    assert (
        loopback.get("/health", headers={"Authorization": "Bearer secret-token"}).status_code == 200
    )


def test_invalid_image_rejected_with_token(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load(monkeypatch, UBT_OCR_SIDECAR_TOKEN="secret-token")
    client = TestClient(module.app)
    resp = client.post(
        "/v1/ocr",
        files={"file": ("x.png", b"not an image", "image/png")},
        headers={"Authorization": "Bearer secret-token"},
    )
    assert resp.status_code == 400


def test_oversized_upload_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load(monkeypatch, UBT_OCR_SIDECAR_TOKEN="secret-token", UBT_OCR_MAX_UPLOAD_BYTES="16")
    client = TestClient(module.app)
    resp = client.post(
        "/v1/ocr",
        files={"file": ("x.png", _png_bytes((32, 32)), "image/png")},
        headers={"Authorization": "Bearer secret-token"},
    )
    assert resp.status_code == 413


def test_ocr_offloads_blocking_work_to_threadpool(monkeypatch: pytest.MonkeyPatch) -> None:
    """§10.5: the model load and inference must not run on the event loop.

    Inline they starve /health and let the container HEALTHCHECK kill a busy
    sidecar. Pin that both get_engine and the inference go through
    run_in_threadpool; a revert to a direct call trips this.
    """
    module = _load(monkeypatch)
    real = module.run_in_threadpool
    seen: list[str] = []

    async def spy(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(getattr(fn, "__name__", str(fn)))
        return await real(fn, *args, **kwargs)

    monkeypatch.setattr(module, "run_in_threadpool", spy)
    client = TestClient(module.app)
    resp = client.post("/v1/ocr", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert resp.status_code == 200
    assert "get_engine" in seen
    assert "_run_ocr_inference" in seen


def test_run_ocr_inference_parses_rapidocr_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    """The extracted helper still turns a rapidocr result into line dicts."""
    module = _load(monkeypatch)

    class _FakeRapid:
        def __call__(self, img):  # type: ignore[no-untyped-def]
            box = [[1.0, 2.0], [9.0, 2.0], [9.0, 6.0], [1.0, 6.0]]
            return [(box, "hello", 0.97)], None

    # numpy reaches this env only through the heavy extras; the dev-only
    # main gate must skip, not fail (same invariant as the docling tests).
    np = pytest.importorskip("numpy")

    lines = module._run_ocr_inference("rapidocr", _FakeRapid(), np.zeros((4, 4, 3), dtype="uint8"))
    assert lines == [{"text": "hello", "box": [1.0, 2.0, 9.0, 6.0], "confidence": 0.97}]
