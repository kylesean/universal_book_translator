#!/usr/bin/env python3
"""Compare a PaddleOCR-VL element reading against Docling's own cells, on one crop.

Why this exists: the question "is PaddleOCR-VL worth adopting" cannot be answered
from OmniDocBench numbers. This project's failure modes are concrete — a stacked
table cell fused into one blob, a formula degraded to a screenshot, a caption
bound to the wrong figure — and a model that leads the leaderboard can still miss
exactly those. So both readers are pointed at the same crop and the outputs are
put side by side for a human to judge, with the measurable parts counted.

Usage::

    uv run python scripts/compare_vlm_elements.py --pdf ~/Downloads/2609.20519v1.pdf \\
        --page 7 --box 60,60,540,780 --task table

The box is in PDF points, top-left origin, as ``x0,y0,x1,y1``.

Crop tightly and around ONE structure. Measured: with surrounding prose inside the
crop the model reads the prose into the formula (``Formula Recognition:`` returned
the running text as ``\text{...}``), and a page-sized crop makes it either collapse
several tables into one row or loop one equation until the token cap. This tool
compares element recognition, which is what the model is good at; page layout is
the layout model's job.

Needs ``transformers>=5`` (v5 registered the architecture; on 4.x it raises
``Unrecognized configuration class``). Run it from an environment that has that —
the project's own venv is on 4.x.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TASKS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}


def _crop(pdf: Path, page_no: int, box: tuple[float, float, float, float], dpi: int):
    """Render one page and cut ``box`` out of it.

    ``box`` is in PDF points with a TOP-LEFT origin (what a human reads off the
    page); pdfium renders top-left too, so the box maps straight onto pixels.
    """
    import pypdfium2 as pdfium

    pdf_doc = pdfium.PdfDocument(str(pdf))
    try:
        page = pdf_doc[page_no - 1]
        scale = dpi / 72.0
        x0, y0, x1, y1 = box
        img = page.render(scale=scale).to_pil()
        return img.crop(
            (int(x0 * scale), int(y0 * scale), int(x1 * scale), int(y1 * scale))
        ).convert("RGB")
    finally:
        pdf_doc.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pdf", type=Path, required=True)
    ap.add_argument("--page", type=int, required=True, help="1-based page number")
    ap.add_argument("--box", required=True, help="x0,y0,x1,y1 in PDF points (top-left origin)")
    ap.add_argument("--task", choices=sorted(TASKS), default="table")
    ap.add_argument("--model", default="PaddlePaddle/PaddleOCR-VL-1.6")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--save-crop", type=Path, default=None)
    args = ap.parse_args(argv)

    coords = tuple(float(part) for part in args.box.split(","))
    if len(coords) != 4:
        print("--box needs four comma-separated numbers", file=sys.stderr)
        return 2

    image = _crop(args.pdf, args.page, coords, args.dpi)  # type: ignore[arg-type]
    print(f"crop: {image.width}x{image.height} px at {args.dpi} dpi")
    if args.save_crop:
        image.save(args.save_crop)
        print(f"saved: {args.save_crop}")

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    free, total = torch.cuda.mem_get_info()
    print(f"VRAM before load: {free / 1e9:.2f} / {total / 1e9:.2f} GB free")
    model = (
        AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16)
        .to("cuda")
        .eval()
    )
    processor = AutoProcessor.from_pretrained(args.model)
    free, _ = torch.cuda.mem_get_info()
    print(f"VRAM after load : {free / 1e9:.2f} GB free")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": TASKS[args.task]},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated = output[0][inputs["input_ids"].shape[-1] :]
    text = processor.decode(generated, skip_special_tokens=True)
    free, _ = torch.cuda.mem_get_info()
    print(f"VRAM after run  : {free / 1e9:.2f} GB free\n")
    print("=" * 72)
    print(text)
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
