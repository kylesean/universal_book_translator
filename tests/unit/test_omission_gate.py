"""Omission gate — sentence ratio, proper-noun/number recall, chrF verbatim recall.

All three metrics are 0-token (deterministic). The regression proof: a target
that drops a whole sentence passes every pre-existing fast-pass gate
(length ratio 0.2 floor, script density, numbers, HTML) but must FAIL the
omission gate.
"""

import pytest

from ubt.core.qe.comet_runner import QE_DEFECT_CLASS_LEGEND, HeuristicQERunner
from ubt.core.qe.fast_pass import FastPassFilter
from ubt.core.qe.omission import OmissionGate

# 4-sentence English paragraph. A correct zh translation keeps 4 (or a
# legitimately merged 3); the omission fixture keeps only 2.
_SRC_4_SENT = (
    "The old lighthouse keeper climbed the spiral stairs every evening. "
    "He carried a small brass lantern that had belonged to his father. "
    "From the gallery deck he could see the fishing boats returning home. "
    "Their lights flickered like scattered stars on the dark water."
)
_TGT_4_SENT = (
    "老灯塔守卫每天傍晚都会爬上螺旋楼梯。"
    "他提着一盏属于父亲的黄铜小灯笼。"
    "从环形平台上，他能看到归航的渔船。"
    "它们的灯光像散落在暗色水面上的星星一样闪烁。"
)
_TGT_2_SENT = "老灯塔守卫每天傍晚都会爬上螺旋楼梯。他提着一盏属于父亲的黄铜小灯笼。"

# Two identifier-shaped terms (camelCase + acronym) that must survive verbatim.
_SRC_TERMS = "The PagedAttention kernel talks to the MHA scheduler on every step."
_TGT_TERMS_OK = "PagedAttention 内核在每一步都会与 MHA 调度器通信。"
_TGT_TERMS_LOST = "PagedAttention 内核在每一步都会与调度器通信。"


class TestSentenceRatio:
    def test_dropped_sentences_fail_gate(self) -> None:
        gate = OmissionGate()
        decision = gate.evaluate(_SRC_4_SENT, _TGT_2_SENT)
        assert not decision.passed
        assert "omission" in decision.reason.lower()

    def test_full_translation_passes(self) -> None:
        gate = OmissionGate()
        decision = gate.evaluate(_SRC_4_SENT, _TGT_4_SENT)
        assert decision.passed, decision.reason
        assert decision.metrics.sentence_ratio >= 0.5

    def test_legitimate_merging_passes(self) -> None:
        """4 source sentences rendered as 3 (two merged) is a style choice, not omission."""
        merged = (
            "老灯塔守卫每天傍晚都会爬上螺旋楼梯。"
            "他提着一盏属于父亲的黄铜小灯笼，"
            "从环形平台上，他能看到归航的渔船。"
            "它们的灯光像散落的星星一样闪烁。"
        )
        decision = OmissionGate().evaluate(_SRC_4_SENT, merged)
        assert decision.passed, decision.reason

    def test_halfwidth_cjk_terminators_are_counted(self) -> None:
        """A correct zh translation using half-width !? must not read as omission.

        CJK has no whitespace after punctuation; the counter only recognized
        full-width 。！？, so three correct sentences counted as one and the gate
        quarantined a correct translation.
        """
        src = "Alpha ran fast. Beta ran slow. Gamma won the race."
        tgt = "阿尔法跑得很快!贝塔跑得很慢?伽马赢了比赛。"
        decision = OmissionGate().evaluate(src, tgt)
        assert decision.passed, decision.reason

    def test_short_blocks_skip_sentence_gate(self) -> None:
        """Blocks with fewer than 3 source sentences are too noisy to gate."""
        decision = OmissionGate().evaluate("He climbed the stairs. He fell asleep.", "他爬上楼梯。")
        assert decision.passed, decision.reason

    def test_decimals_do_not_count_as_sentence_breaks(self) -> None:
        src = "The value converged to 3.14159 after several iterations. Then it stabilized. The team celebrated the result."
        tgt = "数值在多次迭代后收敛到 3.14159。随后保持稳定。团队为此庆祝了一番。"
        decision = OmissionGate().evaluate(src, tgt)
        assert decision.passed, decision.reason


class TestProperNounRecall:
    def test_missing_identifier_term_fails(self) -> None:
        decision = OmissionGate().evaluate(_SRC_TERMS, _TGT_TERMS_LOST)
        assert not decision.passed
        assert decision.metrics.proper_noun_recall < 0.75

    def test_all_terms_verbatim_pass(self) -> None:
        decision = OmissionGate().evaluate(_SRC_TERMS, _TGT_TERMS_OK)
        assert decision.passed, decision.reason
        assert decision.metrics.proper_noun_recall == 1.0

    def test_lowercase_words_are_not_identifier_terms(self) -> None:
        """Ordinary lowercase words may be translated; they are not 'proper nouns'."""
        src = "The kernel scheduler runs on every core."
        tgt = "内核调度器在每个核心上运行。"
        decision = OmissionGate().evaluate(src, tgt)
        assert decision.metrics.proper_noun_recall == 1.0
        assert decision.passed, decision.reason

    def test_common_caps_words_are_not_identifier_terms(self) -> None:
        """IT/OR/IF/US are ordinary words, not terms demanded verbatim.

        Regression: every all-caps token counted as an identifier, so a correct
        translation that rendered them in Chinese failed the omission gate and
        the block was quarantined as BLOCKED_HUMAN.
        """
        src = "The IT department uses OR logic in IF statements and US dollars."
        tgt = "信息技术部门在条件语句中使用或逻辑，并以美元计价。"
        decision = OmissionGate().evaluate(src, tgt)
        assert decision.metrics.proper_noun_recall == 1.0
        assert decision.passed, decision.reason

    def test_real_acronyms_remain_identifier_terms(self) -> None:
        """The stoplist must not swallow genuine 3+ character acronyms."""
        src = "The MHA and GQA kernels differ. LSTM layers stack deeply."
        tgt = "MHA 与 GQA 内核不同。LSTM 层可深度堆叠。"
        assert OmissionGate().evaluate(src, tgt).passed


class TestNumberRecall:
    def test_cjk_structural_numeral_keeps_recall(self) -> None:
        """'Chapter 7' -> '第七章' must count as preserved (H11 normalization)."""
        decision = OmissionGate().evaluate(
            "Chapter 7 explains the outcome. It ends with a summary. The next part is exercises.",
            "第七章解释了结果。它以一段总结收尾。接下来是练习。",
        )
        assert decision.metrics.number_recall == 1.0
        assert decision.passed, decision.reason

    def test_truncated_number_kills_recall(self) -> None:
        decision = OmissionGate().evaluate(
            "Revenue reached 1,234,567 dollars. Costs stayed flat. Investors were calm overall.",
            "收入达到了 1,234 美元。成本保持平稳。投资者总体上很镇定。",
        )
        assert decision.metrics.number_recall < 1.0


class TestChrFVerbatimRecall:
    def test_partial_verbatim_loss_fails(self) -> None:
        """Sentence count and identifier terms are fine, but the verbatim residue
        (digits + identifiers) lost most of its character n-grams."""
        src = "In 1984 the archive recorded 1,234,567 visits. The number doubled by 1991. Curators were astonished."
        tgt = "1984 年档案馆记录了 1234567 次访问。到 1991 年这个数字翻了一番。馆长们大为惊讶。"
        ok = OmissionGate().evaluate(src, tgt)
        assert ok.passed, ok.reason

        truncated = "1984 年档案馆记录了 1234 次访问。到 1991 年这个数字翻了一番。馆长们大为惊讶。"
        bad = OmissionGate().evaluate(src, truncated)
        assert not bad.passed
        assert bad.metrics.verbatim_chrf_recall < 0.7

    def test_chrf_recall_perfect_on_verbatim_preserving_translation(self) -> None:
        decision = OmissionGate().evaluate(_SRC_TERMS, _TGT_TERMS_OK)
        assert decision.metrics.verbatim_chrf_recall >= 0.9


class TestDefectClassMapping:
    def test_omission_reason_maps_to_dedicated_band(self) -> None:
        score = HeuristicQERunner.score_from_decision_reason(
            "Omission suspected: target has 2 sentence(s) vs 4 in source"
        )
        assert score == 0.35

    def test_legend_documents_omission_band(self) -> None:
        assert any(value == 0.35 for value, _label in QE_DEFECT_CLASS_LEGEND)


class TestFastPassIntegration:
    # 6 en sentences; the target merges two pairs and silently drops the
    # gulls sentence. Kept length ratio ≈ 0.27 — comfortably inside the
    # Zh length band (0.2–3.0), so every pre-gate passes it.
    _SRC_6_SENT = (
        "Marcus arrived at the harbour before sunrise. "
        "He wanted to watch the fleet prepare for the long voyage. "
        "Sailors moved crates of salted cod along the wet pier. "
        "Gulls screamed above the rigging of the old merchant ships. "
        "A cold wind carried the smell of tar and rope. "
        "Somewhere a bell rang to mark the changing of the watch."
    )
    _TGT_3_SENT = (
        "马库斯在日出之前就抵达了港口，他想看着船队为漫长的航行做准备。"
        "水手们沿着湿漉漉的码头搬运着一箱又一箱的腌鳕鱼，动作熟练而沉默。"
        "冷风里裹挟着焦油和绳索的气味，远处传来报更的钟声。"
    )

    def test_fast_pass_rejects_sentence_omission(self) -> None:
        """Regression: the block below passes every pre-gate."""
        decision = FastPassFilter(source_lang="en", target_lang="zh").evaluate(
            self._SRC_6_SENT, self._TGT_3_SENT
        )
        assert not decision.passed
        assert "omission" in decision.reason.lower()

    def test_fast_pass_attaches_metrics_on_success(self) -> None:
        decision = FastPassFilter(source_lang="en", target_lang="zh").evaluate(
            _SRC_4_SENT, _TGT_4_SENT
        )
        assert decision.passed
        assert decision.omission is not None
        assert decision.omission.sentence_ratio >= 0.5

    def test_fast_pass_passes_clean_translation(self) -> None:
        decision = FastPassFilter(source_lang="en", target_lang="zh").evaluate(
            _SRC_TERMS, _TGT_TERMS_OK
        )
        assert decision.passed, decision.reason


@pytest.mark.parametrize(
    ("src", "tgt"),
    [
        # Markdown table rows: no sentence terminators, both sides stay 1 fragment.
        ("| a | b |\n| c | d |", "| 甲 | 乙 |\n| 丙 | 丁 |"),
        # Code-ish residue: lowercase identifier is translatable.
        (
            "Use the print_buffer helper here. It flushes automatically. Then close the stream.",
            "这里使用 print_buffer 辅助函数。它会自动刷新。然后关闭流。",
        ),
    ],
)
def test_no_false_positive_on_structural_prose(src: str, tgt: str) -> None:
    decision = OmissionGate().evaluate(src, tgt)
    assert decision.passed, decision.reason


class TestAbbreviationAwareSentences:
    def test_eq_fig_abbreviations_do_not_inflate_source_count(self) -> None:
        from ubt.core.qe.omission import count_sentences

        src = (
            "Eq. (3.11) is an implicit equation in beta which must be solved "
            "using numerical methods. Fig. 3.5 shows the surface potential."
        )
        assert count_sentences(src) == 2

    def test_formula_dense_narrative_passes(self) -> None:
        src = (
            "Using this approximation, Eq. (3.13) can be integrated analytically. "
            "Fig. 3.8 shows an example of the drain current from the proposed model."
        )
        tgt = "利用该近似，可对式(3.13)进行解析积分。图3.8给出了所提模型的漏极电流示例。"
        decision = OmissionGate().evaluate(src, tgt)
        assert decision.passed, decision.reason

    def test_real_sentence_drop_still_fails(self) -> None:
        src = (
            "Eq. (3.13) can be integrated analytically. "
            "Fig. 3.8 shows the drain current example. "
            "The model agrees with numerical simulation."
        )
        tgt = "可对式(3.13)进行解析积分。"
        decision = OmissionGate().evaluate(src, tgt)
        assert not decision.passed


class TestIdentifierNoiseTolerance:
    def test_english_plural_renders_singular(self) -> None:
        decision = OmissionGate().evaluate(
            "The FinFETs are robust with Q0 margin for compact modeling.",
            "FinFET 对紧凑建模而言鲁棒且有 Q0 裕量。",
        )
        assert decision.passed, decision.reason

    def test_spaced_subscript_remerged(self) -> None:
        decision = OmissionGate().evaluate(
            "Potentials are set with V ch = V s and V DS = 0 with Q0 margin.",
            "电势设为 Vch = Vs 且 VDS = 0，并留有 Q0 裕量。",
        )
        assert decision.passed, decision.reason

    def test_digit_glued_extraction_noise(self) -> None:
        decision = OmissionGate().evaluate(
            "where Q0 = Qbulk + 5CfinVtm, with Cfin = e ch/Tfin for the device.",
            "其中 Q0 = Qbulk + 5CfinVtm，Cfin = εch/Tfin。",
        )
        assert decision.passed, decision.reason

    def test_camel_glued_words_accept_surviving_parts(self) -> None:
        # 'whereQ0' never appears verbatim in the target; its shaped part Q0 does.
        decision = OmissionGate().evaluate(
            "whereQ0 is defined above with Q0 used throughout this section here.",
            "上文定义中 Q0 在本节通篇使用。",
        )
        assert decision.passed, decision.reason

    def test_hyphen_ghost_excluded(self) -> None:
        src = (
            "The Newton-Raphson method is iterative and robust via the MHA path. "
            "The Newton-Raphson method starts from an initial guess. "
            "A NewtonRaphson step refines the guess further today."
        )
        tgt = "牛顿-拉夫逊法经由 MHA 路径迭代且鲁棒。牛顿-拉夫逊法从初始猜测出发。牛顿-拉夫逊单步可进一步修正猜测。"
        decision = OmissionGate().evaluate(src, tgt)
        assert decision.passed, decision.reason

    def test_letter_glued_impostor_still_fails(self) -> None:
        decision = OmissionGate().evaluate(
            "The Q0 charge and the MOSFET capacitance set the scale. "
            "A third sentence follows here today. A fourth sentence closes the point.",
            "Q0 电荷和 XMOSFET 电容决定了尺度。还有第三句话在这里。第四句话收尾。",
        )
        assert not decision.passed

    def test_fully_dropped_camel_term_still_fails(self) -> None:
        decision = OmissionGate().evaluate(
            "The DataLoader feeds every epoch via the MHA pipeline with samples. "
            "A third sentence follows here today. A fourth sentence closes the point.",
            "数据管线经由 MHA 为每个轮次提供样本。还有第三句话在这里。第四句话收尾。",
        )
        assert not decision.passed

    def test_dropped_shaped_part_still_fails(self) -> None:
        decision = OmissionGate().evaluate(
            "The PowerMOSFET switches fast with Q0 charge here. "
            "A third sentence follows here today. A fourth sentence closes the point.",
            "功率器件在此以 Q0 电荷快速开关。还有第三句话在这里。第四句话收尾。",
        )
        assert not decision.passed

    def test_academic_structural_labels_translatable(self) -> None:
        """Structural labels (FIG, EQ, TABLE, etc.) must translate to target language
        (zh '图', fr 'Figure', de 'Abbildung', ja '図') without false omission."""
        # 2 terms: FIG and FinFET. Translating FIG to '图' leaves 1/2 (50%), which would fail the gate if FIG is tracked.
        src_zh = "FIG. 3.2 illustrates the FinFET operation."
        tgt_zh = "图 3.2 展示了 FinFET 的工作原理。"
        decision_zh = OmissionGate(target_lang="zh").evaluate(src_zh, tgt_zh)
        assert decision_zh.passed, decision_zh.reason

        # French
        src_fr = "FIG. 3.2 illustrates the FinFET operation."
        tgt_fr = "Figure 3.2 illustre le fonctionnement du FinFET."
        assert OmissionGate(target_lang="fr").evaluate(src_fr, tgt_fr).passed

        # German
        src_de = "TABLE 1 summarizes the FinFET parameters."
        tgt_de = "Tabelle 1 fasst die FinFET-Parameter zusammen."
        assert OmissionGate(target_lang="de").evaluate(src_de, tgt_de).passed

    def test_latex_math_subscript_variants_match(self) -> None:
        """Underscore and subscript variables match their standard LaTeX equivalents."""
        src = "Simulation used TFIN = 20 nm, T_CH = 15 nm, and V_DS = 1 V."
        tgt = "仿真采用了 $T_{fin} = 20$ nm、$T_{CH} = 15$ nm 以及 $V_{DS} = 1$ V。"
        decision = OmissionGate(target_lang="zh").evaluate(src, tgt)
        assert decision.passed, decision.reason

    def test_glued_math_extraction_unwound_in_latex(self) -> None:
        """Glued PDF-extraction tokens (whereQ0, withCfin, CfinVtm) match LaTeX-typeset target."""
        src = (
            "whereQ0 = Qbulk +5CfinVtm,withCfin = ε ch/Tfin. "
            "Using this approximation, Eq. (3.13) can be integrated analytically."
        )
        tgt = (
            "其中 $Q_0 = Q_{bulk} + 5 C_{fin} V_{tm}$，而 $C_{fin} = \\varepsilon_{ch}/T_{fin}$。"
            "采用这一近似后，式 (3.13) 可以解析积分。"
        )
        decision = OmissionGate(target_lang="zh").evaluate(src, tgt)
        assert decision.passed, decision.reason
