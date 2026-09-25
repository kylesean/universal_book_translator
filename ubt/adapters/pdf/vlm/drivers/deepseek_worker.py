"""Resident DeepSeek-OCR inference worker: torch runtime behind a process wall.

Launched by ``DeepSeekOcrDriver`` as ``python -m
ubt.adapters.pdf.vlm.drivers.deepseek_worker``. The model loads once and the
session then answers JSON-lines requests ``{"id": N, "image_b64": ...}`` with
``{"id": N, "text": ...}`` (or ``{"id": N, "error": ...}`` for a single failed
page — one bad request must not kill the session). stdin EOF ends the loop:
when the driver dies the pipes close, so this process is never orphaned.

Why a subprocess at all: DeepSeek-OCR runs model-supplied code (gated by
UBT_VLM_TRUST_REMOTE_CODE in the driver before launch), its compat-shimmed
generate path has historically segfaulted through CUDA/native failures, and
the 6.7 GB weights must be releasable by killing one process instead of
taking the whole translation run down with them. stderr is inherited by the
parent so HF download progress and the actionable load-error guidance stay
visible.
"""

from __future__ import annotations

import base64
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


def _load_engine() -> tuple[Any, Any]:
    """Import torch, enforce the VRAM floor, and load the checkpoint once."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    from .deepseek_driver import (
        MIN_FREE_BYTES,
        MODEL_ID,
        _install_cache_compat_shim,
        _install_llama_flash_compat_shim,
    )

    _install_llama_flash_compat_shim()
    _install_cache_compat_shim()
    if not torch.cuda.is_available():
        raise RuntimeError("deepseek-ocr worker needs CUDA (no GPU found)")
    free, _ = torch.cuda.mem_get_info()
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"deepseek-ocr needs ~8 GB free VRAM (have {free / 1024**3:.1f} GB): "
            "stop ollama residents first, e.g. `ollama stop hy-mt2-7b-4k`, "
            "then restart them afterwards"
        )
    model_dir = snapshot_download(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        model_dir, trust_remote_code=True
    )
    # Load straight into bf16 on CUDA: the README recipe (.cuda() then
    # .to(bf16)) peaks through an fp32 shadow copy and OOMs a 12 GB card.
    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        use_safetensors=True,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    torch.cuda.empty_cache()
    return model.eval(), tokenizer


def _infer_text(model: Any, tokenizer: Any, image_b64: str) -> str:
    from .deepseek_driver import FREE_OCR_PROMPT

    with tempfile.TemporaryDirectory() as tmp:
        png = Path(tmp) / "page.png"
        png.write_bytes(base64.b64decode(image_b64))
        # Small preset (no crop): halves vision activations on 12 GB cards.
        # Bump to Gundam (base 1024/crop True) when VRAM allows.
        res = model.infer(
            tokenizer,
            prompt=FREE_OCR_PROMPT,
            image_file=str(png),
            output_path=tmp,
            base_size=640,
            image_size=640,
            crop_mode=False,
            save_results=False,
        )
    return res if isinstance(res, str) else str(res)


def main() -> int:
    try:
        model, tokenizer = _load_engine()
    except Exception as exc:  # the exit code is the channel
        print(f"deepseek worker load failed: {exc}", file=sys.stderr)
        return 3
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id: Any = None
        try:
            request = json.loads(line)
            req_id = request.get("id")
            if request.get("cmd") == "quit":
                return 0
            text = _infer_text(model, tokenizer, str(request.get("image_b64", "")))
            reply: dict[str, Any] = {"id": req_id, "text": text}
        except Exception as exc:  # keep the session alive
            reply = {"id": req_id, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
