"""Contract tests for the zero-LLM bilingual-mode advisor (Phase 1 / Phase 2)."""

from __future__ import annotations

import pytest

from ubt.core.config import DualMode
from ubt.core.ir.models import BlockType, BoundingBox, IRBlock, make_element
from ubt.core.policy.bilingual_advisor import (
    _BASE_MODE_PENALTY,
    _DIFFICULTY_DOWNGRADE_RATE,
    _DOWNGRADE_LADDER,
    _OK_CUTOFF,
    _PROFILE_SENSITIVITY,
    _SUGGEST_CUTOFF,
    MODES,
    RENDER_MODE_VALUE,
    RULES,
    SECONDARY_SUFFIX,
    AdvisorRule,
    Advisory,
    DifficultyAssessment,
    DocSignals,
    ModeScore,
    advise_layout,
    assess_difficulty,
    collect_signals,
    downgrade_mode,
    resolve_effective_mode,
    secondary_mode,
)

pytestmark = pytest.mark.fast


_LONG = "word " * 10  # 50 chars, safely above the 25-char fragment cutoff


def _block(
    block_id: str,
    *,
    block_type: BlockType = BlockType.NARRATIVE,
    source: str = _LONG,
    skip: bool = False,
    page: int = 0,
) -> IRBlock:
    bbox = BoundingBox(page=page, x0=0.0, y0=0.0, x1=1.0, y1=1.0) if page else None
    element = make_element(
        id=block_id,
        spine_index=0,
        block_type=block_type,
        source_text=source,
        bbox=bbox,
        skip_translate=skip,
    )
    return IRBlock(element=element)


def _rule(name: str) -> AdvisorRule:
    return next(r for r in RULES if r.name == name)


# --------------------------------------------------------------------------- #
# Constants / small helpers
# --------------------------------------------------------------------------- #


def test_mode_vocabulary_and_cutoffs_are_pinned() -> None:
    assert MODES == ("inline", "alternating", "monolingual")
    assert _OK_CUTOFF == 0.70
    assert _SUGGEST_CUTOFF == 0.40
    assert _DIFFICULTY_DOWNGRADE_RATE == 0.15


def test_profile_sensitivity_defaults() -> None:
    assert _PROFILE_SENSITIVITY == {"paper": 1.0, "textbook": 1.0, "general": 0.8}


def test_base_mode_penalty_penalizes_only_monolingual() -> None:
    assert _BASE_MODE_PENALTY == {"inline": 0.0, "alternating": 0.0, "monolingual": 0.20}


@pytest.mark.parametrize(
    ("mode", "value"),
    [
        ("inline", "bilingual"),
        ("alternating", "alternating"),
        ("monolingual", "monolingual"),
        ("facing", "facing_spread"),
    ],
)
def test_render_mode_value_mapping(mode: DualMode, value: str) -> None:
    assert RENDER_MODE_VALUE[mode] == value


@pytest.mark.parametrize(
    ("mode", "suffix"),
    [("inline", "_mono"), ("alternating", "_mono"), ("facing", "_mono"), ("monolingual", "_dual")],
)
def test_secondary_suffix_mapping(mode: DualMode, suffix: str) -> None:
    assert SECONDARY_SUFFIX[mode] == suffix


@pytest.mark.parametrize(
    ("primary", "secondary"),
    [
        ("inline", "monolingual"),
        ("alternating", "monolingual"),
        ("facing", "monolingual"),
        ("monolingual", "alternating"),
    ],
)
def test_secondary_mode_is_the_complement(primary: DualMode, secondary: DualMode) -> None:
    assert secondary_mode(primary) == secondary


def test_rule_registry_order_and_names() -> None:
    assert [r.name for r in RULES] == [
        "high_interruption_density",
        "medium_interruption_density",
        "figure_heavy",
        "struct_block_heavy",
        "untranslatable_heavy",
        "fragment_heavy",
    ]


# --------------------------------------------------------------------------- #
# Rule predicates (fires)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("value", "fires"),
    [(0.99, False), (1.0, True), (2.5, True)],
)
def test_high_interruption_boundary(value: float, fires: bool) -> None:
    detail = _rule("high_interruption_density").fires(DocSignals(interruption_per_page=value))
    assert (detail is not None) is fires
    if fires:
        assert detail == f"{value:.2f} structural interruptions/page"


@pytest.mark.parametrize(
    ("value", "fires"),
    [(0.49, False), (0.5, True), (0.99, True), (1.0, False)],
)
def test_medium_interruption_boundary(value: float, fires: bool) -> None:
    detail = _rule("medium_interruption_density").fires(DocSignals(interruption_per_page=value))
    assert (detail is not None) is fires


@pytest.mark.parametrize(
    ("value", "fires"),
    [(0.149, False), (0.15, True)],
)
def test_figure_heavy_boundary(value: float, fires: bool) -> None:
    detail = _rule("figure_heavy").fires(DocSignals(figure_page_share=value))
    assert (detail is not None) is fires
    if fires:
        assert detail == f"{value:.0%} of pages carry figures"


@pytest.mark.parametrize(("value", "fires"), [(0.049, False), (0.05, True)])
def test_struct_heavy_boundary(value: float, fires: bool) -> None:
    detail = _rule("struct_block_heavy").fires(DocSignals(struct_block_share=value))
    assert (detail is not None) is fires


@pytest.mark.parametrize(("value", "fires"), [(0.049, False), (0.05, True)])
def test_untranslatable_heavy_boundary(value: float, fires: bool) -> None:
    detail = _rule("untranslatable_heavy").fires(DocSignals(skip_share=value))
    assert (detail is not None) is fires


@pytest.mark.parametrize(("value", "fires"), [(0.199, False), (0.20, True)])
def test_fragment_heavy_boundary(value: float, fires: bool) -> None:
    detail = _rule("fragment_heavy").fires(DocSignals(fragment_share=value))
    assert (detail is not None) is fires


def test_every_rule_penalizes_inline_most_and_monolingual_least() -> None:
    for rule in RULES:
        assert (
            rule.penalties["inline"]
            >= rule.penalties["alternating"]
            >= rule.penalties["monolingual"]
        )


# --------------------------------------------------------------------------- #
# collect_signals
# --------------------------------------------------------------------------- #


def test_empty_blocks_keep_only_page_count() -> None:
    assert collect_signals([], page_count=7) == DocSignals(page_count=7)


def test_page_count_derived_from_bbox_when_unset() -> None:
    blocks = [_block("a", page=3), _block("b", page=5)]
    assert collect_signals(blocks).page_count == 5


def test_page_count_falls_back_to_one_without_geometry() -> None:
    assert collect_signals([_block("a")]).page_count == 1


def test_signals_aggregate_shares() -> None:
    blocks = [
        _block("p", block_type=BlockType.NARRATIVE),
        _block("f", block_type=BlockType.FORMULA),
        _block("t", block_type=BlockType.TABLE),
        _block("i", block_type=BlockType.IMAGE),
    ]
    signals = collect_signals(blocks, page_count=2)
    assert signals.total_blocks == 4
    assert signals.struct_block_share == 0.75
    assert signals.interruption_per_page == 1.5
    assert signals.fragment_share == 0.0
    assert signals.prose_char_share == 0.25  # one narrative of four equal-length blocks


def test_skip_and_fragment_counts() -> None:
    blocks = [
        _block("a", skip=True),
        _block("b", source="tiny"),
        _block("c", source="   "),  # whitespace-only -> fragment
        _block("d"),
    ]
    signals = collect_signals(blocks)
    assert signals.skip_share == 0.25
    assert signals.fragment_share == 0.5


def test_fragment_uses_stripped_length_at_boundary() -> None:
    exact = _block("a", source="x" * 25)  # 25 chars is not a fragment
    under = _block("b", source="x" * 24)
    signals = collect_signals([exact, under])
    assert signals.fragment_share == 0.5


def test_figure_page_share_uses_provided_pages() -> None:
    signals = collect_signals([_block("a")], page_count=4, figure_pages={1, 2})
    assert signals.figure_page_share == 0.5


def test_no_figures_yields_zero_share() -> None:
    assert collect_signals([_block("a")], page_count=4).figure_page_share == 0.0


def test_prose_char_share_zero_when_no_chars() -> None:
    assert collect_signals([_block("a", source="")]).prose_char_share == 0.0


def test_heading_and_dialogue_count_as_prose() -> None:
    blocks = [
        _block("h", block_type=BlockType.HEADING),
        _block("d", block_type=BlockType.DIALOGUE),
        _block("f", block_type=BlockType.FORMULA),
    ]
    signals = collect_signals(blocks)
    assert signals.prose_char_share == pytest.approx(2 / 3)


# --------------------------------------------------------------------------- #
# advise_layout
# --------------------------------------------------------------------------- #


def test_clean_prose_recommends_inline_with_ok_tier() -> None:
    advisory = advise_layout([_block("a"), _block("b")], "inline", page_count=1)
    assert advisory.tier == "ok"
    assert advisory.recommended == "inline"
    assert [m.mode for m in advisory.ranking] == ["inline", "alternating", "monolingual"]
    assert advisory.ranking[0].score == 1.0
    assert advisory.ranking[2].score == 0.8  # monolingual baseline penalty
    assert advisory.reasons == ()


def test_off_vocabulary_request_is_normalized_to_inline() -> None:
    advisory = advise_layout([_block("a")], "facing", page_count=1)
    assert advisory.requested == "inline"
    advisory_auto = advise_layout([_block("a")], "auto", page_count=1)
    assert advisory_auto.requested == "inline"


def _high_interruption_only_blocks() -> list[IRBlock]:
    # 5 structural blocks over 101 total -> 2.5/page but struct share ~0.0495,
    # so only the high-density rule fires.
    structs = [_block(f"s{i}", block_type=BlockType.FORMULA) for i in range(5)]
    prose = [_block(f"p{i}") for i in range(96)]
    return structs + prose


def test_single_high_density_rule_lands_in_suggest() -> None:
    advisory = advise_layout(
        _high_interruption_only_blocks(), "inline", profile="paper", page_count=2
    )
    assert advisory.tier == "suggest"
    assert advisory.ranking[0].mode == "alternating"
    inline = next(m for m in advisory.ranking if m.mode == "inline")
    assert inline.score == 0.5
    assert inline.penalties == ("high_interruption_density",)
    assert advisory.reasons == ("high_interruption_density: 2.50 structural interruptions/page",)


def _chaotic_blocks() -> list[IRBlock]:
    return [
        _block("f1", block_type=BlockType.FORMULA),
        _block("f2", block_type=BlockType.FORMULA),
        _block("t", block_type=BlockType.TABLE),
        _block("i", block_type=BlockType.IMAGE),
        _block("c", block_type=BlockType.CODE),
        _block("p1"),
        _block("p2"),
        _block("p3"),
        _block("p4", skip=True),
        _block("p5", source=""),  # fragment
        _block("p6", source=""),  # fragment
        _block("p7", source=""),  # fragment
    ]


def test_chaotic_document_discourages_inline_and_recommends_monolingual() -> None:
    advisory = advise_layout(
        _chaotic_blocks(), "inline", profile="paper", page_count=2, figure_pages={1, 2}
    )
    assert advisory.tier == "discourage"
    assert advisory.recommended == "monolingual"
    inline = next(m for m in advisory.ranking if m.mode == "inline")
    assert inline.score == 0.0  # clamped after >1.0 of penalties
    assert set(inline.penalties) == {
        "high_interruption_density",
        "figure_heavy",
        "struct_block_heavy",
        "untranslatable_heavy",
        "fragment_heavy",
    }
    # Reasons follow RULES order, not penalty magnitude.
    assert [r.split(":")[0] for r in advisory.reasons] == [
        "high_interruption_density",
        "figure_heavy",
        "struct_block_heavy",
        "untranslatable_heavy",
        "fragment_heavy",
    ]


def test_general_profile_is_lenient_relative_to_paper() -> None:
    blocks = _high_interruption_only_blocks()
    paper = advise_layout(blocks, "inline", profile="paper", page_count=2)
    general = advise_layout(blocks, "inline", profile="general", page_count=2)
    paper_inline = next(m for m in paper.ranking if m.mode == "inline")
    general_inline = next(m for m in general.ranking if m.mode == "inline")
    assert paper_inline.score == 0.5
    assert general_inline.score == 0.6  # 0.5 * 0.8 sensitivity
    assert paper.tier == "suggest"
    assert general.tier == "suggest"


def test_unknown_profile_falls_back_to_general_sensitivity() -> None:
    blocks = _high_interruption_only_blocks()
    unknown = advise_layout(blocks, "inline", profile="nope", page_count=2)
    general = advise_layout(blocks, "inline", profile="general", page_count=2)
    assert unknown.ranking == general.ranking


def test_tier_tracks_the_requested_mode_not_the_ranking_head() -> None:
    blocks = _high_interruption_only_blocks()
    inline = advise_layout(blocks, "inline", profile="paper", page_count=2)
    mono = advise_layout(blocks, "monolingual", profile="paper", page_count=2)
    # Same signals and ranking; only the tiered mode differs.
    assert inline.ranking == mono.ranking
    assert inline.ranking[0].mode == "alternating"
    assert inline.tier == "suggest"  # inline scores 0.5
    assert mono.tier == "ok"  # monolingual scores 0.8


def test_ok_tier_boundary_at_exactly_point_seven() -> None:
    # One figure among 100 blocks over 4 pages: only figure_heavy fires for
    # inline (0.30 penalty), landing the score on exactly 0.70.
    blocks = [_block("fig", block_type=BlockType.IMAGE)] + [_block(f"p{i}") for i in range(99)]
    advisory = advise_layout(blocks, "inline", profile="paper", page_count=4, figure_pages={1})
    inline = next(m for m in advisory.ranking if m.mode == "inline")
    assert inline.score == 0.70
    assert advisory.tier == "ok"


def test_suggest_tier_boundary_at_exactly_point_four() -> None:
    # High interruption (0.50) + fragment-heavy (0.10) -> inline exactly 0.40.
    structs = [_block(f"s{i}", block_type=BlockType.FORMULA) for i in range(5)]
    fragments = [_block(f"f{i}", source="") for i in range(21)]
    prose = [_block(f"p{i}") for i in range(75)]
    advisory = advise_layout(structs + fragments + prose, "inline", profile="paper", page_count=2)
    inline = next(m for m in advisory.ranking if m.mode == "inline")
    assert inline.score == 0.40
    assert advisory.tier == "suggest"


def test_recommended_is_ranking_head() -> None:
    advisory = advise_layout(
        _chaotic_blocks(), "inline", profile="paper", page_count=2, figure_pages={1, 2}
    )
    assert advisory.recommended == advisory.ranking[0].mode


def test_advisory_to_dict_shape_and_rounding() -> None:
    advisory = advise_layout([_block("a")], "inline", page_count=1)
    payload = advisory.to_dict()
    assert payload["requested"] == "inline"
    assert payload["tier"] == "ok"
    assert payload["recommended"] == "inline"
    assert payload["ranking"][0] == {"mode": "inline", "score": 1.0, "penalties": []}
    assert payload["reasons"] == []
    assert payload["signals"] == {
        "total_blocks": 1,
        "page_count": 1,
        "interruption_per_page": 0.0,
        "struct_block_share": 0.0,
        "figure_page_share": 0.0,
        "skip_share": 0.0,
        "fragment_share": 0.0,
        "prose_char_share": 1.0,
    }


# --------------------------------------------------------------------------- #
# assess_difficulty
# --------------------------------------------------------------------------- #


def test_assess_zero_total_is_not_hard() -> None:
    assessment = assess_difficulty(total=0, repaired=0, failed=0)
    assert assessment == DifficultyAssessment(hard=False, repair_share=0.0, reasons=())


def test_assess_below_threshold_is_not_hard() -> None:
    assessment = assess_difficulty(total=100, repaired=14, failed=0)
    assert assessment.hard is False
    assert assessment.repair_share == 0.14
    assert assessment.reasons == ()


def test_assess_at_threshold_is_hard_with_reason() -> None:
    assessment = assess_difficulty(total=100, repaired=10, failed=5)
    assert assessment.hard is True
    assert assessment.repair_share == 0.15
    assert assessment.reasons == ("repair burden 15% >= 15%",)


# --------------------------------------------------------------------------- #
# downgrade_mode / resolve_effective_mode
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("inline", "alternating"),
        ("alternating", "monolingual"),
        ("monolingual", "monolingual"),
        ("facing", "facing"),
        ("auto", "auto"),
    ],
)
def test_downgrade_ladder_and_off_ladder_passthrough(mode: DualMode, expected: DualMode) -> None:
    assert downgrade_mode(mode) == expected


def test_downgrade_ladder_matches_table() -> None:
    assert _DOWNGRADE_LADDER == {
        "inline": "alternating",
        "alternating": "monolingual",
        "monolingual": "monolingual",
    }


def _advisory(tier: str, recommended: DualMode = "alternating") -> Advisory:
    return Advisory(
        requested="inline",
        tier=tier,  # type: ignore[arg-type]
        ranking=(ModeScore(recommended, 0.9), ModeScore("inline", 0.1)),
        reasons=(),
    )


def _hard() -> DifficultyAssessment:
    return DifficultyAssessment(hard=True, repair_share=0.5, reasons=("hard",))


def _easy() -> DifficultyAssessment:
    return DifficultyAssessment(hard=False, repair_share=0.0)


def test_advise_enforcement_always_keeps_requested() -> None:
    assert resolve_effective_mode("inline", _advisory("discourage"), _hard()) == "inline"
    assert (
        resolve_effective_mode("inline", _advisory("discourage"), _hard(), enforcement="advise")
        == "inline"
    )


def test_auto_keeps_requested_when_not_discouraged() -> None:
    assert (
        resolve_effective_mode("inline", _advisory("ok"), _easy(), enforcement="auto") == "inline"
    )
    assert (
        resolve_effective_mode("inline", _advisory("suggest"), _easy(), enforcement="auto")
        == "inline"
    )


def test_auto_switches_to_recommended_when_discouraged() -> None:
    assert (
        resolve_effective_mode(
            "inline", _advisory("discourage", "monolingual"), _easy(), enforcement="auto"
        )
        == "monolingual"
    )


def test_auto_downgrades_one_step_when_hard() -> None:
    assert (
        resolve_effective_mode("inline", _advisory("ok"), _hard(), enforcement="auto")
        == "alternating"
    )


def test_auto_downgrades_after_switching_to_recommended() -> None:
    result = resolve_effective_mode(
        "inline", _advisory("discourage", "monolingual"), _hard(), enforcement="auto"
    )
    assert result == "monolingual"  # monolingual is the ladder floor


def test_auto_hard_requested_off_ladder_passes_through() -> None:
    assert (
        resolve_effective_mode("facing", _advisory("ok"), _hard(), enforcement="auto") == "facing"
    )
