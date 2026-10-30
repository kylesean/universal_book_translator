r"""The artifact presence check must survive a correctly typeset formula.

:func:`ubt.adapters.pdf.artifact_text.check_artifact` asks "does the delivered
PDF actually carry each text block's realization?" — the check that catches a
paragraph the renderer silently dropped. A contiguous 16-character run is its
proof, and that proof assumes the page carries the block's characters.

A formula breaks that assumption. ``$\psi_s$`` in the ledger draws as ``𝜓𝑠``:
Unicode-math codepoints, not the source characters. Every 16-character window
straddling the formula therefore misses, and the audit called the paragraph lost
while its prose sat plainly on the page. Measured on the controlled
``chapter-3.pdf`` re-export: 5 of 155 realizations flagged missing (``b0039``,
``b0087``, ``b0147``, ``b0169``, ``b0180``) — all five narrative blocks whose
middle is a formula.

That dependency was latent, and it is worth stating out loud because it is the
more important finding: the check was passing *because of* the very defect it
should have been catching. While the render boundary escaped LaTeX as literal
``\psi`` text, those characters really were in the PDF's text layer, so the
whole-string probe hit. The moment math became math, the audit went blind in the
other direction and started inventing losses.

These tests pin both halves of the fix and the asymmetry that keeps it honest:
a formula-interrupted block must locate on its prose runs, while a genuinely
absent block must still be reported missing.
"""

from __future__ import annotations

import pytest

from ubt.adapters.pdf.artifact_text import (
    _contains_math,
    _locate_prose,
    _probes_of,
    _prose_runs,
)

pytestmark = pytest.mark.fast


class TestProseRuns:
    def test_math_spans_are_cut_out_of_the_runs(self) -> None:
        runs = _prose_runs("在表面沿任意点的表面电势 $\\psi_s$ 是通过鳍片表面评估得到的")
        assert runs == ["在表面沿任意点的表面电势", "是通过鳍片表面评估得到的"]

    def test_runs_shorter_than_the_floor_are_dropped(self) -> None:
        # ``其中``/``和`` are 2 characters: no contiguous claim is possible.
        assert _prose_runs("其中 $v_o$ 和 $q_t$ 由") == []

    def test_whitespace_inside_a_run_is_stripped(self) -> None:
        # The page breaks lines; a run that is present with a newline inside it
        # is present. The probe must not mistake layout for loss.
        assert _prose_runs("第一段文字内容\n继续排版 $x_1$ 第二段文字内容足够长") == [
            "第一段文字内容继续排版",
            "第二段文字内容足够长",
        ]

    def test_the_floor_drops_a_run_the_page_cannot_prove(self) -> None:
        # ``第一段文字继续`` is 6 characters: a window that short can be hit by a
        # neighbouring sentence, so it is not allowed to claim presence.
        assert _prose_runs("第一段 文字\n继续 $x_1$ 第二段文字内容") == []

    def test_contains_math_agrees_with_the_render_boundary(self) -> None:
        assert _contains_math("其中 $\\psi_s$ 给出") is True
        assert _contains_math("纯文本，没有公式") is False
        # A stray delimiter that the pairing rejects is not math, and must not
        # buy the block a weaker claim it does not need.
        assert _contains_math("价格 $5 与 $6") is False


class TestLocateProse:
    def test_every_long_run_must_land_on_the_same_page(self) -> None:
        expected = "前半段的中文内容 $x_1$ 后半段的中文内容"
        runs = _prose_runs(expected)
        assert len(runs) == 2
        both = ["", "".join(runs)]
        assert _locate_prose(expected, both) == 2
        split = ["", runs[0], runs[1]]
        assert _locate_prose(expected, split) == 0, "one run on two pages is not presence"

    def test_a_block_with_no_usable_run_gets_no_claim(self) -> None:
        assert _locate_prose("其中 $v_o$ 由 $q_t$ 给出", ["", "其中有由给出"]) == 0

    def test_a_lost_paragraph_is_not_located(self) -> None:
        assert _locate_prose("这段中文从未出现在任何一页上", ["", "无关内容无关内容"]) == 0


class TestProbeBackwardCompatibility:
    """The whole-string windows must stay, in order, for a block with no math."""

    def test_a_math_free_block_keeps_its_five_windows(self) -> None:
        text = "这是一段完全没有公式的中文内容用来验证窗口数量"
        probes = _probes_of(text)
        assert len(probes) == 5, probes
        assert all(len(p) == 16 for p in probes), probes

    def test_a_block_shorter_than_a_probe_yields_nothing(self) -> None:
        assert _probes_of("短") == []

    def test_a_formula_block_adds_no_window_shorter_than_the_floor(self) -> None:
        # 16-char whole-string windows plus the run conjunction are the only
        # claims; nothing under 8 characters may probe.
        for probe in _probes_of("前段中文内容足够长用来形成窗口 $x_1$ 后段中文内容也足够长了"):
            assert len(probe) >= 8, probe
