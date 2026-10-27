# OCR driver map (post-unification, 2026-10-28)

The six historical `ocr_mode` values collapsed along the axis the anchoring
layer actually consumes: **does the driver return measured geometry?**

| Mode | Driver(s) | Measured boxes | Scans transcribable | Role |
|---|---|---|---|---|
| `local` | `rapidocr` / `sidecar` | yes | yes | **Scan transcription** |
| `vlm` | `unlimited-ocr2` / `CloudOcrDriver` | no | no (fail-closed) | **Proofread upgrade** |
| `off` | — | — | — | Skip OCR entirely |
| `auto` | probe chain | — | — | local first, then VLM |

## Why two tiers, not six

`anchor_transcript` (`ubt/adapters/pdf/vlm/anchor.py:75`) fails closed when a
page has no text layer and the driver cannot produce measured boxes. A
generative VLM emitting text only — no matter how good its OCR — cannot
transcribe a true scan into UBT's geometric IR. Every mode name that collapsed
into `local` shared the measured-box property; every name that collapsed into
`vlm` shared its complement.

## Drivers

### `unlimited-ocr2` (recommended VLM proofread driver)

Baidu's 3.3B MoE document-parsing model, served through a patched llama.cpp
(mainline rejects the `unlimited-ocr2` architecture). Runs as a subprocess
(`ubt/adapters/pdf/vlm/drivers/unlimited_ocr2_worker.py`) that owns the
`llama-mtmd-cli` lifecycle: a native crash kills the worker, not the
translation run.

Configuration:

```bash
export UBT_UNLIMITED_OCR2_MODEL=/path/to/model-dir   # model.gguf + mmproj.gguf
export UBT_UNLIMITED_OCR2_CLI=/path/to/llama-mtmd-cli
export UBT_OCR_MODE=vlm
export UBT_OCR_MODEL=Unlimited-OCR-2                 # billed-model name
export UBT_ALLOW_PAGE_UPLOAD=true                    # the egress gate
```

VRAM floor: 6 GiB free (Q8_0 weights 2.92 GiB + mmproj 0.77 GiB + compute
buffers ~2.4 GiB). A Q4_K_M build fits a smaller card.

### `rapidocr` (the only measured-box in-process engine)

ONNX detector + recognizer, ~58 MB of wheels (opencv-headless + onnxruntime +
shapely + pyclipper), CPU-capable. It is the **only** in-process driver whose
`measured_boxes=True`, which makes it the only driver that can transcribe a
scan without an external service. It is no longer the default recommendation —
`auto` prefers the sidecar when one is running, then falls back here — but the
anchoring contract means it cannot be removed while scans must remain
transcribable offline.

### `sidecar` (rapidocr behind HTTP)

Same engine, isolated process. Useful when the OCR stack lives on a GPU box
or in Docker. `local` prefers it over in-process rapidocr when the health
probe answers.

## The old names

`deepseek-ocr` is gone (485 lines of transformers monkey-patching for an
experimental path that never left the lab). `sidecar`, `rapidocr`, `cloud`,
and `http` are accepted by the driver registry for operator continuity but the
`OcrMode` literal no longer exposes them — use `local` (measured-box tier) or
`vlm` (text-only tier).
