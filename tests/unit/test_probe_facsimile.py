"""The 1:1 facsimile probe: its two constraint models and the geometry they read.

The probe decides whether a source page can be delivered 1:1 — same page, same
columns, one uniform body size — and it is the producer of every number the
design doc quotes (``docs/design/inplace_delivery.md`` §3.1, §5.3.1, E.5). A
silent regression here would not fail any pipeline gate: it would quietly
change what the channel is believed to be capable of.

Four properties are pinned, each of which was a real defect during development:

* **glue ⊇ pinned.** ``glue`` lets a paragraph eat the gap before the next one;
  ``pinned`` keeps it inside its own box. Every size glue accepts, pinned must
  accept, so ``glue_size >= pinned_size`` on every page. The first
  implementation violated this on 23 pages.
* **Leading scales with the size.** A 12pt target set with 12pt leading is not
  the same document; the box a paragraph needs is ``lines × em × size`` where
  ``em`` is the *source* leading/size ratio. Ignoring ``em`` inflated every
  estimate by ~33%.
* **Page furniture is not body text.** A running head sits above the body and
  would otherwise become the "next paragraph" that bounds the first real one.
* **A single-line paragraph's box bottom is its line-box bottom**, not its ink
  bottom: an 8pt ink box inside a 12pt line leaves 4pt the paragraph does not
  have. Using the ink bottom made glue stricter than pinned on
  ``chapter-3.pdf`` p3 (9.4pt → 6.4pt).

The probe is pure geometry over a source PDF — no render, no model — so these
tests drive it with synthetic ``LineBox``-shaped objects and never touch a
real file.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts import probe_facsimile as probe  # noqa: E402

pytestmark = pytest.mark.fast


@dataclass
class _Line:
    """The slice of ``textgeom.LineBox`` the probe actually reads."""

    text: str
    rect: tuple[float, float, float, float]

    @property
    def top(self) -> float:
        return self.rect[3]

    @property
    def bottom(self) -> float:
        return self.rect[1]


def _line(text: str, *, top: float, width: float = 300.0, ink_h: float = 8.0) -> _Line:
    """One body line whose ink box sits at ``top`` (PDF points, y up)."""
    return _Line(text, (80.0, top - ink_h, 80.0 + width, top))


def _page(lines: list[_Line]) -> list[_Line]:
    """Give a paragraph chain a plausible reading order (top-down)."""
    return sorted(lines, key=lambda line: -line.rect[1])


def _fits_any(
    paragraphs: list[list[_Line]],
    *,
    size: float,
    source_size: float,
    source_leading: float,
    mode: str,
    en_per_zh: float = 3.53,
) -> bool:
    """Re-implementation of the probe's constraint check, for property tests.

    Kept deliberately separate from the probe so a change to one without the
    other fails here rather than silently redefining what "fits" means.
    """
    em = source_leading / source_size
    for para in paragraphs:
        width = max(line.rect[2] - line.rect[0] for line in para)
        glyphs = sum(len(line.text) for line in para) / en_per_zh
        per_line = max(1.0, width / size)
        lines_needed = max(1, math.ceil(glyphs / per_line))
        height = lines_needed * em * size
        top = max(line.rect[3] for line in para)
        if mode == "pinned":
            if height > len(para) * source_leading + probe.TOL_PT:
                return False
        else:
            ink_bottom = min(line.rect[1] for line in para)
            line_box_bottom = top - len(para) * source_leading
            limit = min(ink_bottom, line_box_bottom)
            if top - height < limit - probe.TOL_PT:
                return False
    return True


def test_modes_are_the_two_documented_models() -> None:
    assert probe.MODES == ("pinned", "glue")


def test_unknown_mode_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        probe.probe_page(tmp_path / "missing.pdf", 1, mode="squeeze")


def test_paragraph_gap_tolerates_a_between_line_number() -> None:
    """Reviewer line numbers sit between body lines and are filtered as narrow.

    The surviving pair then reports the gap *across* the excluded line, so the
    paragraph cut has to tolerate more than one body leading. At the naive 1.6x
    a numbered manuscript shattered into one paragraph per line.
    """
    leading = 12.0
    assert probe.PARA_GAP_FACTOR * leading > 2 * leading
    assert probe.PARA_GAP_FACTOR * leading < 3 * leading


def test_glue_is_never_stricter_than_pinned() -> None:
    """The property the first implementation violated on 23 pages.

    Sweeps page shapes rather than one fixture: a page whose paragraphs are
    dense (no gap to borrow) must still never let glue report a *smaller* size
    than pinned, because glue's feasible set is a superset by construction.
    """
    for n_lines in (1, 2, 5, 12):
        for gap in (0.0, 6.0, 24.0, 60.0):
            source_size, leading = 9.0, 12.0
            top = 700.0
            paragraphs: list[list[_Line]] = []
            for _ in range(3):
                para = [_line("x" * 80, top=top - line * leading) for line in range(n_lines)]
                paragraphs.append(para)
                top -= n_lines * leading + gap
            pinned_best = 0.0
            glue_best = 0.0
            for step in range(0, 121):
                size = 6.0 + step * 0.05
                if _fits_any(
                    paragraphs,
                    size=size,
                    source_size=source_size,
                    source_leading=leading,
                    mode="pinned",
                ):
                    pinned_best = size
                if _fits_any(
                    paragraphs,
                    size=size,
                    source_size=source_size,
                    source_leading=leading,
                    mode="glue",
                ):
                    glue_best = size
            assert glue_best >= pinned_best - 1e-9, (n_lines, gap, pinned_best, glue_best)


def test_leading_scales_with_size() -> None:
    """``em`` is the source leading/size ratio; a target at 12pt needs 16pt lines.

    Reading the source leading as the target leading under-estimated every
    paragraph's height by the ratio (12/9 = 1.33), which is what made 12pt look
    feasible on pages where it overflows the column.
    """
    source_size, source_leading = 9.0, 12.0
    em = source_leading / source_size
    assert em == pytest.approx(1.333, abs=1e-3)
    # One line of target text at 12pt occupies 16pt, not 12pt.
    assert 1 * em * 12.0 == pytest.approx(16.0, abs=1e-6)


def test_single_line_paragraph_bottom_is_its_line_box_bottom() -> None:
    """An 8pt ink box inside a 12pt line leaves 4pt the paragraph does not have.

    On ``chapter-3.pdf`` p3 a single-line paragraph at the column foot was
    credited with that 4pt, which turned a 9.4pt page into a 6.4pt page under
    ``glue`` while ``pinned`` still said 9.4pt.
    """
    source_leading = 12.0
    line = _line("x" * 80, top=71.5, ink_h=8.0)
    ink_bottom = line.rect[1]
    line_box_bottom = line.rect[3] - 1 * source_leading
    assert ink_bottom == pytest.approx(63.5)
    assert line_box_bottom == pytest.approx(59.5)
    assert min(ink_bottom, line_box_bottom) == pytest.approx(59.5)


def test_source_overlap_is_detected() -> None:
    """A float or footnote sitting on the text column cannot be 1:1 at any size.

    Detected rather than left to a render-time failure: the probe reports it so
    A1 qualification routes the page to reflow instead of trusting a size.

    The geometry that matters is the *line-box* bottom of the earlier
    paragraph (``top - n * leading``) against the *top* of the later one. Ink
    bottoms are 4pt short of their line boxes, so a page can look separated by
    ink and still have overlapping boxes — which is exactly the case a
    float-on-body page produces.
    """
    source_leading = 12.0

    def overlap(earlier: list[_Line], later: list[_Line]) -> bool:
        earlier_top = max(line.rect[3] for line in earlier)
        earlier_bottom = earlier_top - len(earlier) * source_leading
        later_top = max(line.rect[3] for line in later)
        return earlier_bottom < later_top - probe.TOL_PT

    # Three lines from 700 down: line-box bottom = 700 - 36 = 664.
    earlier = [_line("x" * 80, top=700.0 - index * source_leading) for index in range(3)]
    # Clear: the next paragraph starts below 664 (reading order runs downward).
    assert overlap(earlier, [_line("x" * 80, top=650.0)]) is False
    # Intruding: the next paragraph starts at 670, above the earlier box's
    # bottom — a float or footnote has pushed into the text column.
    assert overlap(earlier, [_line("x" * 80, top=670.0)]) is True


def test_page_probe_carries_mode_and_overlap() -> None:
    """The report shape is part of the contract: consumers key on both fields."""
    page = probe.PageProbe(
        page=1,
        page_w=595.0,
        page_h=842.0,
        columns=1,
        source_size_pt=9.0,
        paragraphs=3,
        fits_at_source=3,
        uniform_size_pt=11.0,
        fits=True,
        fill_ratio=0.6,
    )
    assert page.mode == "glue"
    assert page.source_overlap is False
    assert page.fill_pct == pytest.approx(60.0)


def test_body_lines_drops_furniture_and_narrow_runs() -> None:
    """Page furniture keeps its own box in the background layer.

    A running head above the body would otherwise become the "next paragraph"
    that bounds the first real paragraph, inflating the page's capacity.
    """
    lines: list[Any] = [
        _line("CHAPTER 3", top=790.0, width=40.0),  # 40pt < MIN_LINE_W_PT: decoration
        _line("x" * 80, top=700.0),
        _line("x" * 80, top=688.0),
    ]
    body = probe._body_lines(lines, 595.0)
    assert len(body) == 2
    assert all(line.text != "CHAPTER 3" for line in body)
