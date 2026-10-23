"""Reference HTTP OCR Sidecar Service for Universal Book Translator.

Run standalone or in a Docker/Podman container on localhost or GPU server:
    uvicorn server:app --host 0.0.0.0 --port 8765

Exposes:
- GET /health: Heartbeat probe for UBT auto-detection.
- POST /v1/ocr: Multi-engine OCR inference returning normalized line boxes.
"""

from __future__ import annotations

import asyncio
import importlib
import io
import ipaddress
import logging
import os
import secrets
import threading
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from PIL import Image
from starlette.concurrency import run_in_threadpool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ocr-sidecar")

# Upload and decode caps: an unauthenticated sidecar that reads the whole body
# into memory and decodes arbitrary pixel counts is an OOM primitive.
MAX_UPLOAD_BYTES = int(os.environ.get("UBT_OCR_MAX_UPLOAD_BYTES", str(32 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.environ.get("UBT_OCR_MAX_IMAGE_PIXELS", str(80_000_000)))
#: Per-request inference timeout and a bounded number of concurrent OCR jobs.
#: Without these, N stuck requests each hold a shared threadpool worker and
#: their full decoded image in memory (a resource-exhaustion primitive for any
#: token holder / loopback caller).
OCR_TIMEOUT_S = float(os.environ.get("UBT_OCR_TIMEOUT_S", "120"))
MAX_CONCURRENT_OCR = max(1, int(os.environ.get("UBT_OCR_MAX_CONCURRENCY", "2")))
_OCR_SEM = asyncio.Semaphore(MAX_CONCURRENT_OCR)
_CHUNK = 1024 * 1024
# Pillow raises DecompressionBombError above 2x this value, and warns above it.
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

app = FastAPI(title="UBT OCR Sidecar", version="1.0.0")

if not os.environ.get("UBT_OCR_SIDECAR_TOKEN", "").strip():
    logger.warning(
        "UBT_OCR_SIDECAR_TOKEN is not set: the OCR sidecar is unauthenticated. "
        "Set it (and pass 'Authorization: Bearer <token>') before exposing the "
        "port beyond localhost."
    )


def require_token(
    request: Request,
    authorization: str | None = Header(default=None),
) -> None:
    """Shared-secret gate *and* loopback fallback (``UBT_OCR_SIDECAR_TOKEN``).

    Holds the sidecar behind a bearer token when configured, so exposing the
    port beyond localhost does not hand OCR/CPU to any caller. With no token
    configured the gate does not simply open: only loopback callers may
    proceed. Both rules live in this one dependency so ``/health`` and
    ``/v1/ocr`` cannot drift apart again — they used to, and ``/v1/ocr``
    accepted unauthenticated LAN uploads on an "open" sidecar while ``/health``
    already refused them.

    Boot logs a warning when no token is set, so the exposure is never silent.
    """
    expected = os.environ.get("UBT_OCR_SIDECAR_TOKEN", "").strip()
    if expected:
        provided = ""
        if authorization and authorization.lower().startswith("bearer "):
            provided = authorization[7:].strip()
        if not secrets.compare_digest(provided.encode("utf-8"), expected.encode("utf-8")):
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token")
        return
    client_host = request.client.host if request.client else None
    if not _is_loopback(client_host):
        raise HTTPException(
            status_code=403,
            detail="This endpoint requires UBT_OCR_SIDECAR_TOKEN for non-loopback callers",
        )


# Lazy engine loader (PaddleOCR -> RapidOCR fallback)
_ocr_engine: Any = None
_engine_name: str = "none"
# Model construction is multi-second CPU/GPU work; without a lock, concurrent
# /health probes (or an unauthenticated flood) race to build the engine.
_engine_lock = threading.Lock()


def get_engine() -> tuple[str, Any]:
    global _ocr_engine, _engine_name
    with _engine_lock:
        if _ocr_engine is not None:
            return _engine_name, _ocr_engine

        # 1. Try PaddleOCR
        try:
            from paddleocr import PaddleOCR

            _ocr_engine = PaddleOCR(use_angle_cls=True, lang="ch")
            _engine_name = "paddleocr"
            logger.info("Initialized PaddleOCR engine")
            return _engine_name, _ocr_engine
        except Exception as exc:
            logger.debug("PaddleOCR not available: %s", exc)

        # 2. Try RapidOCR (ONNX). Both import paths are probed because the
        #    package was renamed: ``rapidocr>=3`` ships as ``rapidocr``, while
        #    the older ``rapidocr_onnxruntime`` still exists on PyPI. The
        #    Dockerfile pins the modern name, so probing only the legacy one
        #    made the container report "mock" (503, never adopted) even with
        #    the engine installed.
        for module_name in ("rapidocr", "rapidocr_onnxruntime"):
            try:
                module = importlib.import_module(module_name)
                _ocr_engine = module.RapidOCR()
                _engine_name = "rapidocr"
                logger.info("Initialized RapidOCR engine (%s)", module_name)
                return _engine_name, _ocr_engine
            except Exception as exc:
                logger.debug("RapidOCR via %s not available: %s", module_name, exc)

        _engine_name = "mock"
        _ocr_engine = None
        return _engine_name, _ocr_engine


def _is_loopback(host: str | None) -> bool:
    if not host:
        return False
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # "localhost" from a real socket; "testclient" is Starlette's TestClient
        # sentinel host (in-process, never a network caller).
        return host in ("localhost", "testclient")


def _ocr_array(img: Any) -> Any:
    """PIL RGB → the BGR ndarray both engines' ONNX models expect.

    ``rapidocr``'s LoadImage passes an ndarray through unchanged (no channel
    conversion) and its detector/recognizer models are trained on the OpenCV
    BGR convention, so feeding RGB silently degrades recognition rather than
    failing — the driver inside UBT does the same swap (see
    ``ubt/adapters/pdf/vlm/drivers/rapidocr_driver.py``). PaddleOCR takes RGB
    and does its own conversion, so the swap is applied per engine.
    """
    import numpy as np

    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] == 3:
        # ``ascontiguousarray`` keeps the negative-stride view safe for cv2.
        return np.ascontiguousarray(arr[:, :, ::-1])
    return arr


def _iter_rapidocr_predictions(raw: Any) -> Any:
    """Yield ``(box, text, conf)`` across rapidocr output shapes.

    ``rapidocr`` 2.x returns ``([[box, text, conf], ...], elapse)``; 3.x returns
    a typed result object with ``.boxes``/``.txts``/``.scores``. Accepting only
    the first shape made the sidecar return zero lines on the version the
    project's own dependency pins.
    """
    candidate = raw[0] if isinstance(raw, tuple) else raw
    boxes = getattr(candidate, "boxes", None)
    texts = getattr(candidate, "txts", getattr(candidate, "texts", None))
    confs = getattr(candidate, "scores", getattr(candidate, "confs", None))
    if boxes is not None and texts is not None:
        if confs is None:
            confs = [1.0] * len(texts)
        return list(zip(boxes, texts, confs, strict=False))
    return candidate or []


def _run_ocr_inference(engine_name: str, engine: Any, img: Any) -> list[dict[str, Any]]:
    """Synchronous OCR pass. Runs in a worker thread (see the endpoints).

    Both the model call and get_engine()'s lazy import/load block for seconds;
    executed inline in an ``async def`` they stall the event loop, so /health
    stops answering and the container HEALTHCHECK can kill a busy sidecar.
    """
    lines_out: list[dict[str, Any]] = []
    if engine_name == "paddleocr" and engine is not None:
        import numpy as np

        img_np = np.asarray(img)
        result = engine.ocr(img_np, cls=True)
        if result and result[0]:
            for item in result[0]:
                box_pts, (text, conf) = item
                xs = [p[0] for p in box_pts]
                ys = [p[1] for p in box_pts]
                box = [min(xs), min(ys), max(xs), max(ys)]
                lines_out.append({"text": text, "box": box, "confidence": float(conf)})
    elif engine_name == "rapidocr" and engine is not None:
        result = engine(_ocr_array(img))
        for item in _iter_rapidocr_predictions(result):
            try:
                box_pts, text, conf = item
            except (TypeError, ValueError):
                continue
            if not (str(text or "")).strip():
                continue
            try:
                xs = [float(p[0]) for p in box_pts]
                ys = [float(p[1]) for p in box_pts]
            except (TypeError, ValueError, IndexError):
                continue
            if not xs or not ys:
                continue
            box = [min(xs), min(ys), max(xs), max(ys)]
            lines_out.append({"text": str(text), "box": box, "confidence": float(conf)})
    else:
        logger.warning("No OCR models found in sidecar environment; returning empty lines")
    return lines_out


def _to_pdf_points(
    lines_out: list[dict[str, Any]],
    *,
    width_pt: float,
    height_pt: float,
    scale: float,
    image_width: int,
    image_height: int,
) -> str:
    """Rewrite line boxes in place from pixel space to PDF points.

    Returns the coordinate system the boxes are now in. The declared
    page/scale pair is only usable when it agrees with the bitmap actually
    decoded (within 1.5px, the rounding a renderer's ceil/floor can introduce);
    otherwise the boxes are left in ``image_pixel`` and the caller is told so,
    because silently mis-scaling every line is worse than making the caller
    re-derive the scale.

    Pixel space is top-left origin with y increasing downwards; PDF user space
    is bottom-left origin with y increasing upwards, so the y pair flips.
    """
    if not (scale > 0 and width_pt > 0 and height_pt > 0):
        return "image_pixel"
    expected_w = width_pt * scale
    expected_h = height_pt * scale
    if abs(image_width - expected_w) > 1.5 or abs(image_height - expected_h) > 1.5:
        logger.warning(
            "sidecar: declared page %.1fx%.1fpt at scale %.2f implies a %.0fx%.0f "
            "bitmap but the upload is %dx%d; reporting image_pixel instead",
            width_pt,
            height_pt,
            scale,
            expected_w,
            expected_h,
            image_width,
            image_height,
        )
        return "image_pixel"
    for line in lines_out:
        box = line.get("box")
        if not box:
            continue
        x0, y0, x1, y1 = (float(v) / scale for v in box)
        line["box"] = [x0, height_pt - y1, x1, height_pt - y0]
    return "pdf_points"


@app.get("/health", dependencies=[Depends(require_token)])
async def health() -> dict[str, str]:
    """Healthcheck endpoint queried by UBT auto-detection.

    Mock mode (no OCR engine installed) reports 503 not-ready: UBT's
    auto-probe treats HTTP 200 as "adopt this sidecar", so an empty engine
    answering ok used to get selected and silently returned zero lines for
    every scanned page (the one fail-open hole in the OCR chain).

    Auth comes from the shared ``require_token`` dependency: a configured
    ``UBT_OCR_SIDECAR_TOKEN`` is required on every caller, and without a token
    only loopback callers may probe — a non-loopback caller must not be able to
    trigger lazy model initialization (or learn readiness) on an open sidecar.
    ``/v1/ocr`` runs the same dependency, so the two endpoints cannot disagree.
    """
    engine_name, _ = await run_in_threadpool(get_engine)
    if engine_name == "mock":
        raise HTTPException(status_code=503, detail="no OCR engine available (mock mode)")
    return {"status": "ok", "engine": engine_name, "version": "1.0.0"}


@app.post("/v1/ocr", dependencies=[Depends(require_token)])
async def ocr(
    file: UploadFile = File(...),
    width_pt: float = Form(default=595.0),
    height_pt: float = Form(default=842.0),
    scale: float = Form(default=2.0),
) -> dict[str, Any]:
    """Perform OCR on uploaded image and return line text and coordinates.

    ``width_pt``/``height_pt``/``scale`` describe the page the caller rendered
    this image from (page size in PDF points, and the pixels-per-point factor).
    They are used to report boxes in **pdf_points** — the caller's own frame —
    which is unambiguous, where ``image_pixel`` needs the caller to re-derive
    the scale. When they are inconsistent with the uploaded bitmap (a caller
    that rendered at a different scale than it declared), the response falls
    back to ``image_pixel`` and says so, instead of silently mis-scaling.
    """
    # Bounded read: reject before buffering the whole body into memory.
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds {MAX_UPLOAD_BYTES} bytes",
            )
        chunks.append(chunk)
    contents = b"".join(chunks)
    async with _OCR_SEM:
        # Decode *inside* the slot: a decoded RGB frame is ~3 bytes/pixel, so
        # decoding before acquiring would let every queued request hold one,
        # which is the memory primitive the upload/pixel caps only partly close.
        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
        except Image.DecompressionBombError as exc:
            logger.warning("sidecar rejected an oversized image: %s", exc)
            raise HTTPException(status_code=413, detail="Image too large") from exc
        except Exception as exc:
            # Log the internals; never echo them to the caller.
            logger.warning("sidecar rejected an invalid image: %s", exc)
            raise HTTPException(status_code=400, detail="Invalid image format") from exc

        engine_name, engine = await run_in_threadpool(get_engine)
        try:
            lines_out = await asyncio.wait_for(
                run_in_threadpool(_run_ocr_inference, engine_name, engine, img),
                timeout=OCR_TIMEOUT_S,
            )
        except TimeoutError as exc:
            # The caller gets 504 now, but a worker thread is not killable: the
            # inference may finish in the background still holding its frame.
            raise HTTPException(status_code=504, detail="OCR inference timed out") from exc

    coord_system = _to_pdf_points(
        lines_out,
        width_pt=width_pt,
        height_pt=height_pt,
        scale=scale,
        image_width=img.width,
        image_height=img.height,
    )

    return {
        "engine": engine_name,
        "lines": lines_out,
        "coord_system": coord_system,
    }
