"""The OCR sidecar's engine adapter and coordinate contract.

Three defects, all in the reference sidecar that ships in ``deploy/``:

* It probed only ``rapidocr_onnxruntime``. The package was renamed —
  ``rapidocr>=3`` imports as ``rapidocr``, and that is what the Dockerfile
  installs — so the container reported ``mock`` (503, never adopted) with the
  engine present.
* It passed the PIL image's **RGB** ndarray straight to the engine. rapidocr's
  models are trained on the OpenCV BGR convention and its loader passes an
  ndarray through unchanged, so recognition silently degraded instead of
  failing.
* It parsed only rapidocr 2.x's ``([[box, text, conf], ...], elapse)``. 3.x
  returns a typed result object, so the sidecar returned zero lines on the
  version the project pins.

Plus the coordinate frame: the caller declares the page size and render scale,
and the sidecar now answers in ``pdf_points`` when the declaration agrees with
the bitmap it decoded.

The module lives outside the package, so it is loaded by path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = pytest.mark.fast

_SIDECAR = Path(__file__).resolve().parents[3] / "deploy" / "docker" / "ocr-sidecar" / "server.py"


def _load_sidecar() -> Any:
    spec = importlib.util.spec_from_file_location("ubt_ocr_sidecar_under_test", _SIDECAR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sidecar() -> Any:
    return _load_sidecar()


# --------------------------------------------------------------------------- #
# Engine discovery
# --------------------------------------------------------------------------- #


def test_both_rapidocr_package_names_are_probed(sidecar: Any) -> None:
    """The modern name is what the Dockerfile installs."""
    import inspect

    source = inspect.getsource(sidecar.get_engine)
    assert "rapidocr_onnxruntime" in source
    assert '"rapidocr"' in source or "'rapidocr'" in source


def test_a_missing_engine_still_reports_mock(sidecar: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """The 503 contract ``/health`` depends on."""
    real_import = importlib.import_module

    def only_failing(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("rapidocr", "rapidocr_onnxruntime"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(sidecar, "importlib", SimpleNamespace(import_module=only_failing))
    monkeypatch.setattr(sidecar, "_ocr_engine", None)
    monkeypatch.setattr(sidecar, "_engine_name", "none")
    # PaddleOCR is not installed either, so this walks the whole chain.
    name, engine = sidecar.get_engine()
    assert (name, engine) == ("mock", None)


# --------------------------------------------------------------------------- #
# Output shapes
# --------------------------------------------------------------------------- #


def test_a_2x_tuple_result_is_parsed(sidecar: Any) -> None:
    raw = (
        [
            [[[0, 0], [10, 0], [10, 5], [0, 5]], "hello", 0.9],
            [[[0, 10], [10, 10], [10, 15], [0, 15]], "world", 0.8],
        ],
        {"det": 0.01},
    )
    rows = list(sidecar._iter_rapidocr_predictions(raw))
    assert [row[1] for row in rows] == ["hello", "world"]


def test_a_3x_typed_result_is_parsed(sidecar: Any) -> None:
    """3.x returns an object, not a tuple of lists."""
    result = SimpleNamespace(
        boxes=[[[0, 0], [10, 0], [10, 5], [0, 5]]],
        txts=["hello"],
        scores=[0.9],
    )
    rows = list(sidecar._iter_rapidocr_predictions(result))
    assert len(rows) == 1
    assert rows[0][1] == "hello"
    assert rows[0][2] == 0.9


def test_a_3x_result_without_scores_defaults_to_certain(sidecar: Any) -> None:
    result = SimpleNamespace(boxes=[[[0, 0], [1, 1]]], txts=["x"], scores=None)
    rows = list(sidecar._iter_rapidocr_predictions(result))
    assert rows[0][2] == 1.0


def test_an_empty_result_yields_nothing(sidecar: Any) -> None:
    assert list(sidecar._iter_rapidocr_predictions(None)) == []
    assert list(sidecar._iter_rapidocr_predictions(([], {}))) == []


# --------------------------------------------------------------------------- #
# Channel order
# --------------------------------------------------------------------------- #


def test_the_engine_gets_bgr_not_rgb(sidecar: Any) -> None:
    """A red pixel must reach the model as (0, 0, 255), not (255, 0, 0)."""
    pytest.importorskip("numpy")
    from PIL import Image

    img = Image.new("RGB", (2, 2), (255, 0, 0))
    arr = sidecar._ocr_array(img)
    assert arr.shape == (2, 2, 3)
    assert tuple(int(v) for v in arr[0, 0]) == (0, 0, 255)
    assert arr.flags["C_CONTIGUOUS"], "a negative-stride view would trip cv2"


def test_an_already_native_array_is_passed_through(sidecar: Any) -> None:
    """A caller that hands in a 2-D (grayscale) array must not be sliced."""
    np = pytest.importorskip("numpy")
    arr = np.zeros((4, 4), dtype=np.uint8)
    assert sidecar._ocr_array(arr).shape == (4, 4)


def test_the_inference_path_swaps_channels_before_calling(sidecar: Any) -> None:
    """Pin the wiring, not just the helper: ``_run_ocr_inference`` must use it."""
    import inspect

    source = inspect.getsource(sidecar._run_ocr_inference)
    assert "_ocr_array(" in source
    # PaddleOCR takes RGB and converts internally, so its branch must not.
    paddle_branch = source.split('elif engine_name == "rapidocr"')[0]
    assert "_ocr_array" not in paddle_branch


# --------------------------------------------------------------------------- #
# Coordinate frame
# --------------------------------------------------------------------------- #


class _Engine:
    """A stand-in engine returning one box, in pixels, top-left origin."""

    def __init__(self, box: list[float], text: str = "hello") -> None:
        self.box = box
        self.text = text
        self.seen: Any = None

    def __call__(self, arr: Any) -> tuple[list[Any], dict[str, float]]:
        self.seen = arr
        return ([[self.box, self.text, 0.9]], {})


def test_a_consistent_declaration_reports_pdf_points(sidecar: Any) -> None:
    """595x842pt at scale 2 is a 1190x1684 bitmap; a box at its top-left maps
    to the top-left of the PDF page, in bottom-left points."""
    lines = [{"text": "hello", "box": [0.0, 0.0, 100.0, 40.0], "confidence": 0.9}]
    system = sidecar._to_pdf_points(
        lines,
        width_pt=595.0,
        height_pt=842.0,
        scale=2.0,
        image_width=1190,
        image_height=1684,
    )
    assert system == "pdf_points"
    box = lines[0]["box"]
    assert isinstance(box, list)
    x0, y0, x1, y1 = box
    assert (x0, x1) == pytest.approx((0.0, 50.0))
    # Pixel y 0..40 is PDF y 822..842: the box is at the TOP of the page.
    assert (y0, y1) == pytest.approx((822.0, 842.0))


def test_an_inconsistent_declaration_falls_back_to_image_pixel(sidecar: Any) -> None:
    """A caller that rendered at another scale must not be silently mis-scaled."""
    lines = [{"text": "hello", "box": [0.0, 0.0, 100.0, 40.0], "confidence": 0.9}]
    system = sidecar._to_pdf_points(
        lines,
        width_pt=595.0,
        height_pt=842.0,
        scale=2.0,
        image_width=800,
        image_height=600,
    )
    assert system == "image_pixel"
    assert lines[0]["box"] == pytest.approx([0.0, 0.0, 100.0, 40.0])


def test_one_pixel_of_renderer_rounding_is_tolerated(sidecar: Any) -> None:
    """Renderers ceil or floor the bitmap; a 1px difference is not a mismatch."""
    lines = [{"text": "x", "box": [0.0, 0.0, 10.0, 10.0], "confidence": 0.9}]
    system = sidecar._to_pdf_points(
        lines,
        width_pt=595.0,
        height_pt=842.0,
        scale=2.0,
        image_width=1191,
        image_height=1683,
    )
    assert system == "pdf_points"


def test_a_zero_scale_is_not_treated_as_a_page_declaration(sidecar: Any) -> None:
    lines = [{"text": "x", "box": [0.0, 0.0, 10.0, 10.0], "confidence": 0.9}]
    system = sidecar._to_pdf_points(
        lines, width_pt=595.0, height_pt=842.0, scale=0.0, image_width=1190, image_height=1684
    )
    assert system == "image_pixel"
    assert lines[0]["box"] == [0.0, 0.0, 10.0, 10.0]


def test_the_endpoint_reports_the_helper_verdict(
    sidecar: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wiring check: the endpoint must not keep its own copy of the decision."""
    import inspect

    source = inspect.getsource(sidecar.ocr)
    assert "coord_system = _to_pdf_points(" in source, (
        "the endpoint must take the coordinate system from the helper, not pre-set one of its own"
    )
    assert "px_per_pt" not in source, "the scaling loop moved into the helper"


def test_the_sidecar_rejects_an_unauthenticated_caller(
    sidecar: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    monkeypatch.setenv("UBT_OCR_SIDECAR_TOKEN", "tok")
    resp = TestClient(sidecar.app).get("/health")
    assert resp.status_code == 401
