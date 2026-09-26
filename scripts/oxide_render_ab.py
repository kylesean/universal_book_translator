#!/usr/bin/env python3
"""A/B render equivalence: pdf_oxide ``render_page`` vs poppler ``pdftoppm``.

Gate for stage 2 of the pdf_oxide adoption plan (assessment doc since removed;
`git show 62fcd75^:docs/PDF_OXIDE_ADOPTION_ASSESSMENT_2026-09-19.md` §5.1):
the two rasterizers must agree on output size (proves the ``-r N`` ==
``dpi=N`` scale assumption that svg_diagram's crop math depends on) and be
perceptually close (different rasterization stacks hint/antialias differently,
so a ratio threshold — not pixel equality — is the criterion).

Thresholds below were frozen from the first measured run on this corpus
(docs/design/knob-calibration-protocol.md convention: calibrate once, then pin).

Usage:  uv run python scripts/oxide_render_ab.py [pdf ...]
Exit 0 when every pair passes, 1 on any breach.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# The original real documents (chapter-1-zh.pdf, book2-nistir4653-artifact.pdf)
# were removed in the 2026-09 legal review. The A/B gate now runs on the
# generated, copyright-safe synthetic corpus (scripts/make_sample_corpus.py),
# which tests/conftest.py seeds before collection — so this gate is runnable on
# a fresh checkout without any Release attachment.
DEFAULT_PDFS = [
    REPO_ROOT / "docs" / "synthetic-mono.pdf",
    REPO_ROOT / "docs" / "synthetic-duo.pdf",
]
PAGES = (1, 3, 5)
DPIS = (72, 300)

# Frozen from the 2026-09-20 calibration run on this corpus (see
# docs/design/knob-calibration-protocol.md): sizes matched exactly at every pair;
# mismatch ratios measured 0.008-0.077 @72dpi and 0.003-0.054 @300dpi. The
# heatmap showed the diff is glyph-edge hinting only (a ~1px subpixel shift
# between raster stacks; interiors and background clean) — thresholds carry
# that envelope with headroom, they are not a pixel-equality claim.
MAX_MISMATCH_RATIO_72 = 0.15
MAX_MISMATCH_RATIO_300 = 0.08
PIXEL_EPS = 32  # a difference channel below this is antialias noise, not a mismatch


@dataclass
class PairResult:
    pdf: str
    page: int
    dpi: int
    oxide_size: tuple[int, int]
    pdftoppm_size: tuple[int, int]
    mismatch_ratio: float
    mean_abs_diff: float

    @property
    def size_ok(self) -> bool:
        return all(
            abs(a - b) <= 1 for a, b in zip(self.oxide_size, self.pdftoppm_size, strict=True)
        )

    @property
    def threshold(self) -> float:
        return MAX_MISMATCH_RATIO_72 if self.dpi == 72 else MAX_MISMATCH_RATIO_300

    @property
    def ok(self) -> bool:
        return self.size_ok and self.mismatch_ratio <= self.threshold


def _pdftoppm_png(pdf: Path, page: int, dpi: int, tmp: Path) -> bytes:
    proc = subprocess.run(
        ["pdftoppm", "-png", "-r", str(dpi), "-f", str(page), "-l", str(page), str(pdf), str(tmp)],
        capture_output=True,
        timeout=120,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"pdftoppm rc={proc.returncode}: {proc.stderr[:200]!r}")
    out = sorted(tmp.parent.glob(f"{tmp.name}-*.png"))
    if not out:
        raise RuntimeError("pdftoppm produced no file")
    return out[0].read_bytes()


def compare_pair(pdf: Path, page: int, dpi: int) -> PairResult:
    from PIL import Image, ImageChops, ImageStat

    from ubt.adapters.pdf import oxide_render

    data = oxide_render.render_page_png(pdf, page, dpi)
    if data is None:
        raise RuntimeError(f"oxide failed to render {pdf.name} p{page}@{dpi}")
    with tempfile.TemporaryDirectory(prefix="ubt-ab-") as td:
        ref = _pdftoppm_png(pdf, page, dpi, Path(td) / "p")
    a = Image.open(io.BytesIO(data)).convert("RGB")
    b = Image.open(io.BytesIO(ref)).convert("RGB")
    w = min(a.width, b.width)
    h = min(a.height, b.height)
    diff = ImageChops.difference(a.crop((0, 0, w, h)), b.crop((0, 0, w, h)))
    mask = diff.convert("L").point(lambda v: 255 if v > PIXEL_EPS else 0)
    ratio = ImageStat.Stat(mask).mean[0] / 255.0
    mad = ImageStat.Stat(diff.convert("L")).mean[0]
    return PairResult(
        pdf=pdf.name,
        page=page,
        dpi=dpi,
        oxide_size=a.size,
        pdftoppm_size=b.size,
        mismatch_ratio=ratio,
        mean_abs_diff=mad,
    )


def main(argv: list[str]) -> int:

    if shutil.which("pdftoppm") is None:
        print("pdftoppm unavailable — cannot A/B", file=sys.stderr)
        return 2
    pdfs = [Path(a) for a in argv[1:]] if len(argv) > 1 else list(DEFAULT_PDFS)
    # A missing path must fail the gate, not be silently dropped: filtering it
    # out made a run that compared nothing print "0 failure(s)" and exit 0.
    missing = [p for p in pdfs if not p.exists()]
    if missing:
        for p in missing:
            print(f"missing input PDF: {p}", file=sys.stderr)
        return 2
    if not pdfs:
        print("no input PDFs to compare", file=sys.stderr)
        return 2
    failures = 0
    print(
        f"{'pdf':38} {'pg':>3} {'dpi':>4}  {'oxide':>11} {'pdftoppm':>11}  "
        f"{'mismatch':>8} {'MAD':>6}  verdict"
    )
    for pdf in pdfs:
        try:
            from pdf_oxide import PdfDocument

            n_pages = PdfDocument(str(pdf)).page_count
        except Exception as exc:  # report and skip the file
            print(f"{pdf.name}: unopenable ({exc})", file=sys.stderr)
            failures += 1
            continue
        for page in [p for p in PAGES if p <= n_pages]:
            for dpi in DPIS:
                r = compare_pair(pdf, page, dpi)
                ok = r.ok
                failures += 0 if ok else 1
                print(
                    f"{r.pdf:38} {r.page:>3} {r.dpi:>4}  {r.oxide_size[0]:>5}x{r.oxide_size[1]:<5} "
                    f"{r.pdftoppm_size[0]:>5}x{r.pdftoppm_size[1]:<5}  {r.mismatch_ratio:>7.4f} "
                    f"{r.mean_abs_diff:>6.2f}  {'PASS' if ok else 'FAIL'}"
                )
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
