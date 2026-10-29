"""The canvas tier end-to-end: a real PDF pair through ``run_visual_gate``.

Split from the synthetic-PNG tests in ``test_visual_gate_canvas.py`` because
these build real PDFs and rasterise them with ``pdf_oxide``, which costs about a
second per page. The point of them is not the metric (that file pins it) but the
*wiring*: the gate must read the canvas page map out of the Typst source,
rasterise the stripped background for exactly those pages, diff each against its
own artifact page, and put the result in ``stats`` — and it must say the tier did
not run when no background, no source, or no paired page was supplied.

**Measured on this fixture at the shipped constants** (one US Letter page, six
12pt Helvetica lines at 16pt leading, one 0.8pt rule per line, 110 dpi). The
artifact page carries the same rules as the background with the text drawn over
them, which is what the facsimile renderer produces. ``pair_identity`` is 1.0000
on every row — the artifact really is drawn on this background. Ratio by where
the rule sits relative to each baseline:

    -10pt (well into the previous line's descender zone) 0.118  fires
    -8pt  (touching the previous line's descenders)      0.022  quiet
    -7pt                                                 0.017  quiet
    -6pt                                                 0.009  quiet
    -5pt  (the empty inter-line band)                    0.000  quiet
    -4pt                                                 0.008  quiet
    -3pt                                                 0.020  quiet
    -2pt  (tight: 2pt under the baseline)                0.019  quiet
    -1pt  (through the descenders)                       0.127  fires
    +0pt  (on the baseline)                              0.129  fires
    +2pt                                                 0.189  fires
    +4pt  (through the x-height)                         0.207  fires
    +6pt  (through the cap height)                       0.141  fires
    +8pt  (above the ascenders)                          0.027  quiet
    +10pt                                                0.010  quiet
    +14pt                                                0.016  quiet

The shape of that table is the honest description of the detector: it fires
where a rule crosses glyph mass — the band from the descender through the cap
height — and stays quiet wherever the rule is clear of the letters, on both
sides. Two consequences are recorded here rather than papered over:

* ``-8pt`` being *quiet* is correct and is a change from the pre-fix numbers: at
  16pt leading, 8pt under a baseline is in the gap before the next line's
  ascenders, and only the previous line's descenders can reach it. The 2.2% it
  measures is under the 3% budget.
* A table with tight cell padding (a rule 2pt under a baseline, ``-2pt``) reads
  1.9% and is **not** reported. The budget buys that silence deliberately.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from ubt.adapters.pdf.visual_gate import CANVAS_COLLISION_RATIO, run_visual_gate

pytestmark = pytest.mark.fast

_WIDTH = 612.0
_HEIGHT = 792.0
_MARGIN = 54.0
_LEADING = 16.0
_FONT_SIZE = 12.0
_LINES = 6

#: Points above a baseline: through the x-height. A rule here is drawn straight
#: across the letters, which is the defect the tier exists for.
_RULE_THROUGH_TEXT = 4.0
#: Points below a baseline: the empty band between one line's descender (-2.5pt
#: for 12pt Helvetica) and the next line's ascender (-7.4pt at 16pt leading).
#: Measured 0.001 — the tier must stay silent here.
_RULE_IN_GAP = -5.0


def _write_page(
    path: Path,
    *,
    text: bool,
    rule_offset: float | None = None,
) -> Path:
    """One US Letter page: optional body text, optional one rule per line.

    Written with ``pypdf`` rather than the engine's own renderer for the same
    reason ``tests/pdf_builders.py`` is: the thing under test must not author its
    own fixture.

    ``rule_offset`` places a rule relative to each line's baseline. Both the
    background and the artifact are built through this function so the artifact
    carries the identical rules *under* its text — that is the facsimile
    relation, and a background-only-rules pair would measure the rules
    themselves as new ink.
    """
    writer = PdfWriter()
    page = writer.add_blank_page(width=_WIDTH, height=_HEIGHT)
    ops: list[str] = []
    if rule_offset is not None:
        for index in range(_LINES):
            y = _HEIGHT - _MARGIN - index * _LEADING + rule_offset
            ops.append(f"q 0 0 0 RG 0.8 w {_MARGIN - 10} {y} m {_WIDTH - _MARGIN + 10} {y} l S Q")
    if text:
        ops.extend(
            [
                "BT",
                f"/F1 {_FONT_SIZE} Tf",
                f"1 0 0 1 {_MARGIN} {_HEIGHT - _MARGIN} Tm",
                f"{_LEADING} TL",
            ]
        )
        ops.extend(
            f"(Line {index}: the quick brown fox jumps over the lazy dog) Tj T*"
            for index in range(_LINES)
        )
        ops.append("ET")
    stream = DecodedStreamObject()
    stream.set_data("\n".join(ops).encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)  # noqa: SLF001
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(  # type: ignore[index]
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): writer._add_object(font)}  # noqa: SLF001
            )
        }
    )
    with path.open("wb") as handle:
        writer.write(handle)
    return path


#: The markup ``FacsimileRenderer`` emits for a single canvas page. The gate
#: reads the artifact-page -> background-page map out of this, so a fixture that
#: omits it is a fixture where the tier correctly declines to run.
_CANVAS_SOURCE = (
    "// Universal Book Translator - 1:1 Facsimile Canvas Document\n"
    "#page(width: 612.0pt, height: 792.0pt, margin: 0pt, "
    'background: image("bg_stripped.pdf", page: 1, width: 100%, height: 100%))[\n'
    "]\n"
)

#: The same document with the page reflowed instead of drawn on the canvas: no
#: ``background:``, which is what an L2 page looks like.
_REFLOW_SOURCE = (
    "#page(width: 612.0pt, height: 792.0pt, "
    "margin: (top: 36pt, bottom: 36pt, left: 36pt, right: 36pt))[\n]\n"
)


def _gate(pdf: Path, background: Path | None, *, typ_text: str | None = _CANVAS_SOURCE):
    return asyncio.run(
        run_visual_gate(
            pdf,
            sample_pages=1,
            canvas_background_pdf=background,
            typ_text=typ_text,
        )
    )


def test_a_rule_crossing_the_text_fires_the_canvas_tier(tmp_path: Path) -> None:
    """**The end-to-end proof that the tier is wired and can fail.**

    Without ``canvas_background_pdf`` the background is never rasterised, so this
    finding cannot exist no matter how bad the page is. The assertion is on the
    delivered report, not on the metric: ``canvas_pages`` proves the diff ran,
    ``canvas_collision_pages`` proves it fired, and ``canvas_worst_ratio``
    (measured 0.193) proves it fired with margin rather than on the rounding.
    """
    background = _write_page(tmp_path / "bg.pdf", text=False, rule_offset=_RULE_THROUGH_TEXT)
    artifact = _write_page(tmp_path / "out.pdf", text=True, rule_offset=_RULE_THROUGH_TEXT)

    result = _gate(artifact, background)

    assert result.stats["canvas_pages"] == 1
    assert result.stats["canvas_collision_pages"] == 1
    assert result.stats["canvas_worst_page"] == 1
    assert result.stats["canvas_worst_ratio"] > 5 * CANVAS_COLLISION_RATIO
    assert [f.page for f in result.findings if f.code == "canvas_ink_collision"] == [1]
    assert "canvas" in result.tiers_ran
    assert "canvas" not in result.tiers_skipped


def test_an_artifact_built_without_the_background_rules_is_not_measured(
    tmp_path: Path,
) -> None:
    """An artifact that dropped the background is refused, not scored.

    This replaces an earlier test that asserted such a pair scored *higher* than
    a matched pair. That was true of the old symmetric diff and it is exactly
    what the one-sided diff exists to stop being true: a page the renderer did
    not draw on the background is not a page with a collision, it is a page with
    **no canvas at all**.

    The pairing guard is what distinguishes them. Measured on this fixture:
    background with rules vs artifact that dropped them reads ``pair_identity``
    0.2076 — far below ``CANVAS_PAIR_IDENTITY_MIN`` — so the page is skipped.
    The old assertion compared two such numbers and pinned a coincidence: the
    identity and the old symmetric ratio were both 0.2068, which is what a
    metric that counts removed ink as collision looks like.
    """
    background = _write_page(tmp_path / "bg.pdf", text=False, rule_offset=_RULE_THROUGH_TEXT)
    artifact = _write_page(tmp_path / "out.pdf", text=True, rule_offset=None)

    result = _gate(artifact, background)

    assert result.stats["canvas_pages"] == 0, "an unpaired page must not be scored"
    assert [f for f in result.findings if f.code == "canvas_ink_collision"] == []
    assert result.stats["canvas_unmeasured_pages"] == 1


def test_a_matched_pair_is_measured_and_reports_its_identity(tmp_path: Path) -> None:
    """The positive half of the guard: a real facsimile pair is not refused.

    Without this, the guard above could be satisfied by a detector that refuses
    every page — which would make the tier useless in the opposite direction.
    Measured identity on this pair: 1.0000.
    """
    background = _write_page(tmp_path / "bg.pdf", text=False, rule_offset=_RULE_THROUGH_TEXT)
    artifact = _write_page(tmp_path / "out.pdf", text=True, rule_offset=_RULE_THROUGH_TEXT)

    result = _gate(artifact, background)

    assert result.stats["canvas_pages"] == 1
    assert result.stats["canvas_collision_pages"] == 1
    assert result.stats["canvas_unmeasured_pages"] == 0


def test_a_rule_in_the_empty_band_between_lines_does_not_fire(tmp_path: Path) -> None:
    """The tier must be quiet where there is genuinely nothing to collide with.

    Measured 0.001 at 5pt below the baseline — the band between one line's
    descender and the next line's ascender. This is the assertion that keeps the
    detector honest in the other direction: a check that fires on everything is
    as useless as one that never fires, and it can never be promoted to
    fail-closed.
    """
    background = _write_page(tmp_path / "bg.pdf", text=False, rule_offset=_RULE_IN_GAP)
    artifact = _write_page(tmp_path / "out.pdf", text=True, rule_offset=_RULE_IN_GAP)

    result = _gate(artifact, background)

    assert result.stats["canvas_pages"] == 1, "the tier ran; it is not merely absent"
    assert result.stats["canvas_collision_pages"] == 0
    assert result.stats["canvas_worst_ratio"] < CANVAS_COLLISION_RATIO
    assert [f for f in result.findings if f.code == "canvas_ink_collision"] == []


def test_no_background_reports_the_tier_as_skipped_not_as_clean(tmp_path: Path) -> None:
    """A reflow delivery must not read as "the canvas tier measured 0".

    ``canvas_pages == 0`` is the machine-readable form of "did not run"; the
    skip note and ``tiers_skipped`` are the human-readable form. Both are
    asserted because the console renders the codes and the report JSON is read
    by the KPI collector.
    """
    artifact = _write_page(tmp_path / "out.pdf", text=True, rule_offset=None)

    result = _gate(artifact, None)

    assert result.stats["canvas_pages"] == 0
    assert "canvas" in result.tiers_skipped
    assert "canvas" not in result.tiers_ran
    assert "canvas_background_absent" in result.skip_codes
    assert "canvas ink-collision check skipped" in result.skipped_reason


def test_a_reflowed_page_is_not_scored_against_the_background(tmp_path: Path) -> None:
    """**The fusion case, end to end. This is what made the tier dangerous.**

    An L2 page is reflowed from the IR and embeds no background. Comparing it
    with the same-numbered background page measures the difference between two
    unrelated documents, and the symmetric diff it used to use called that
    "collision". On a real fusion book 5 of 7 L2 pages were reported at 36–59%.

    Two independent guards must both hold here: the page is absent from the
    canvas map (so it is never paired), and if it were paired anyway the
    identity guard would refuse it.
    """
    background = _write_page(tmp_path / "bg.pdf", text=False, rule_offset=_RULE_THROUGH_TEXT)
    artifact = _write_page(tmp_path / "out.pdf", text=True, rule_offset=None)

    result = _gate(artifact, background, typ_text=_REFLOW_SOURCE)

    assert result.stats["canvas_declared_pages"] == 0
    assert result.stats["canvas_pages"] == 0
    assert [f for f in result.findings if f.code == "canvas_ink_collision"] == []
    assert "canvas" in result.tiers_skipped
    assert "canvas_pages_unknown" in result.skip_codes


def test_a_source_with_no_canvas_page_reports_the_tier_as_skipped(tmp_path: Path) -> None:
    """A facsimile run where every page solved to L2 must say so, not pass silently.

    This is a real renderer outcome (a page whose columns overlap always solves
    to L2), and it is the one case where "the canvas tier did not run" means the
    renderer chose reflow rather than that the wiring is broken. The report must
    distinguish them, so the skip note names the source rather than the
    background.
    """
    background = _write_page(tmp_path / "bg.pdf", text=False, rule_offset=_RULE_THROUGH_TEXT)
    artifact = _write_page(tmp_path / "out.pdf", text=True, rule_offset=_RULE_THROUGH_TEXT)

    result = _gate(artifact, background, typ_text=None)

    assert result.stats["canvas_pages"] == 0
    assert "canvas_pages_unknown" in result.skip_codes
    assert "canvas_background_absent" not in result.skip_codes


def test_a_canvas_page_outside_the_sample_reports_that_rather_than_zero(tmp_path: Path) -> None:
    """``canvas_pages == 0`` must not mean two different things.

    The sample is stratified and capped, so a short book can be told apart from a
    book whose canvas pages all fell outside the sample only if the report says
    which. ``canvas_declared_pages`` is the map's size and
    ``canvas_sampled_declared`` is how much of it the sample covered.
    """
    background = _write_page(tmp_path / "bg.pdf", text=False, rule_offset=_RULE_THROUGH_TEXT)
    artifact = _write_page(tmp_path / "out.pdf", text=True, rule_offset=_RULE_THROUGH_TEXT)
    # Two declared canvas pages, but a single-page artifact: page 2 does not exist,
    # so only page 1 is sampled and declared.
    source = "".join(
        [
            "#page(width: 612pt, height: 792pt, margin: 0pt, "
            'background: image("bg.pdf", page: 1, width: 100%, height: 100%))[\n]\n',
            "#page(width: 612pt, height: 792pt, margin: 0pt, "
            'background: image("bg.pdf", page: 2, width: 100%, height: 100%))[\n]\n',
        ]
    )

    result = _gate(artifact, background, typ_text=source)

    assert result.stats["canvas_declared_pages"] == 2
    assert result.stats["canvas_sampled_declared"] == 1
    assert result.stats["canvas_pages"] == 1


def test_a_sampled_canvas_page_that_measures_nothing_says_so(tmp_path: Path) -> None:
    """**The failure mode that reads as success, and the note that stops it.**

    A canvas page whose artifact adds no ink above the text floor — an untranslated
    page, or a page the renderer drew straight from the background — produces zero
    measurements. Left unsaid, that is indistinguishable in the report from a book
    the gate checked and found clean, and it is exactly the substitution this tier
    exists to prevent. The note names the reason; ``stats`` carries the counts.
    """
    background = _write_page(tmp_path / "bg.pdf", text=False, rule_offset=_RULE_IN_GAP)
    # No text at all: the artifact is a copy of its background.
    artifact = _write_page(tmp_path / "out.pdf", text=False, rule_offset=_RULE_IN_GAP)

    result = _gate(artifact, background)

    assert result.stats["canvas_declared_pages"] == 1
    assert result.stats["canvas_pages"] == 0
    assert "canvas_nothing_measured" in result.skip_codes
    assert "canvas" in result.tiers_skipped
    assert [f for f in result.findings if f.code == "canvas_ink_collision"] == []


def test_an_unreadable_background_is_reported_rather_than_raised(tmp_path: Path) -> None:
    """A stale or corrupt background path must degrade, never break the gate.

    The path comes from ``manifest.metadata``, which survives across a resumed
    job; a run whose build directory was cleaned must still ship, with the tier
    honestly reported as not run.
    """
    artifact = _write_page(tmp_path / "out.pdf", text=True, rule_offset=None)
    broken = tmp_path / "bg.pdf"
    broken.write_bytes(b"%PDF-1.4\nnot really a pdf\n")

    result = _gate(artifact, broken)

    assert result.stats["canvas_pages"] == 0
    assert "canvas_background_render_failed" in result.skip_codes
    assert [f for f in result.findings if f.code == "canvas_ink_collision"] == []
