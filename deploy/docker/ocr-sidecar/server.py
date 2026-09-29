"""Reference HTTP OCR Sidecar Service for Universal Book Translator.

Run standalone or in a Docker/Podman container on localhost or GPU server:
    uvicorn server:app --host 0.0.0.0 --port 8765

Exposes:
- GET /health: Heartbeat probe for UBT auto-detection.
- POST /v1/ocr: Multi-engine OCR inference returning normalized line boxes.
"""

from __future__ import annotations

import asyncio
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

        # 2. Try RapidOCR (ONNX)
        try:
            from rapidocr_onnxruntime import RapidOCR

            _ocr_engine = RapidOCR()
            _engine_name = "rapidocr"
            logger.info("Initialized RapidOCR (ONNX) engine")
            return _engine_name, _ocr_engine
        except Exception as exc:
            logger.debug("RapidOCR not available: %s", exc)

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
        import numpy as np

        img_np = np.asarray(img)
        result, _ = engine(img_np)
        if result:
            for item in result:
                box_pts, text, conf = item
                xs = [p[0] for p in box_pts]
                ys = [p[1] for p in box_pts]
                box = [min(xs), min(ys), max(xs), max(ys)]
                lines_out.append({"text": text, "box": box, "confidence": float(conf)})
    else:
        logger.warning("No OCR models found in sidecar environment; returning empty lines")
    return lines_out


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
    """Perform OCR on uploaded image and return line text and coordinates."""
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

    return {
        "engine": engine_name,
        "lines": lines_out,
        "coord_system": "image_pixel",
    }
