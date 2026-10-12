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
    assert MODES == ("dual", "mono")
    assert _OK_CUTOFF == 0.70
    assert _SUGGEST_CUTOFF == 0.40
    assert _DIFFICULTY_DOWNGRADE_RATE == 0.15


def test_profile_sensitivity_defaults() -> None:
    assert _PROFILE_SENSITIVITY == {"paper": 1.0, "textbook": 1.0, "general": 0.8}


def test_base_mode_penalty_penalizes_only_mono() -> None:
    assert _BASE_MODE_PENALTY == {"dual": 0.0, "mono": 0.20}


@pytest.mark.parametrize(
    ("mode", "value"),
    [("dual", "bilingual"), ("mono", "monolingual")],
)
def test_render_mode_value_mapping(mode: DualMode, value: str) -> None:
    assert RENDER_MODE_VALUE[mode] == value


@pytest.mark.parametrize(("mode", "suffix"), [("dual", "_mono"), ("mono", "_dual")])
def test_secondary_suffix_mapping(mode: DualMode, suffix: str) -> None:
    assert SECONDARY_SUFFIX[mode] == suffix


@pytest.mark.parametrize(("primary", "secondary"), [("dual", "mono"), ("mono", "dual")])
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


def test_every_rule_penalizes_dual_and_never_mono() -> None:
    # Mono is the safe fallback: no rule may push a document toward it except
    # by penalizing the pairing.
    for rule in RULES:
        assert rule.penalties["dual"] >= 0.0
        assert rule.penalties["mono"] == 0.0


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


def test_clean_prose_recommends_dual_with_ok_tier() -> None:
    advisory = advise_layout([_block("a"), _block("b")], "dual", page_count=1)
    assert advisory.tier == "ok"
    assert advisory.recommended == "dual"
    assert [m.mode for m in advisory.ranking] == ["dual", "mono"]
    assert advisory.ranking[0].score == 1.0
    assert advisory.ranking[1].score == 0.8  # mono baseline penalty
    assert advisory.reasons == ()


def test_off_vocabulary_request_is_normalized_to_dual() -> None:
    # ``auto`` and any stale value never reach tiering; they tier as the default.
    advisory = advise_layout([_block("a")], "auto", page_count=1)
    assert advisory.requested == "dual"
    advisory_auto = advise_layout([_block("a")], "auto", page_count=1)
    assert advisory_auto.requested == "dual"


def _high_interruption_only_blocks() -> list[IRBlock]:
    # 5 structural blocks over 101 total -> 2.5/page but struct share ~0.0495,
    # so only the high-density rule fires.
    structs = [_block(f"s{i}", block_type=BlockType.FORMULA) for i in range(5)]
    prose = [_block(f"p{i}") for i in range(96)]
    return structs + prose


def test_single_high_density_rule_lands_in_suggest() -> None:
    advisory = advise_layout(
        _high_interruption_only_blocks(), "dual", profile="paper", page_count=2
    )
    assert advisory.tier == "suggest"
    assert advisory.ranking[0].mode == "mono"  # 0.8 beats dual's 0.5
    dual = next(m for m in advisory.ranking if m.mode == "dual")
    assert dual.score == 0.5
    assert dual.penalties == ("high_interruption_density",)
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


def test_chaotic_document_discourages_dual_and_recommends_mono() -> None:
    advisory = advise_layout(
        _chaotic_blocks(), "dual", profile="paper", page_count=2, figure_pages={1, 2}
    )
    assert advisory.tier == "discourage"
    assert advisory.recommended == "mono"
    dual = next(m for m in advisory.ranking if m.mode == "dual")
    assert dual.score == 0.0  # clamped after >1.0 of penalties
    assert set(dual.penalties) == {
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
    paper = advise_layout(blocks, "dual", profile="paper", page_count=2)
    general = advise_layout(blocks, "dual", profile="general", page_count=2)
    paper_dual = next(m for m in paper.ranking if m.mode == "dual")
    general_dual = next(m for m in general.ranking if m.mode == "dual")
    assert paper_dual.score == 0.5
    assert general_dual.score == 0.6  # 0.5 * 0.8 sensitivity
    assert paper.tier == "suggest"
    assert general.tier == "suggest"


def test_unknown_profile_falls_back_to_general_sensitivity() -> None:
    blocks = _high_interruption_only_blocks()
    unknown = advise_layout(blocks, "dual", profile="nope", page_count=2)
    general = advise_layout(blocks, "dual", profile="general", page_count=2)
    assert unknown.ranking == general.ranking


def test_tier_tracks_the_requested_mode_not_the_ranking_head() -> None:
    blocks = _high_interruption_only_blocks()
    dual = advise_layout(blocks, "dual", profile="paper", page_count=2)
    mono = advise_layout(blocks, "mono", profile="paper", page_count=2)
    # Same signals and ranking; only the tiered mode differs.
    assert dual.ranking == mono.ranking
    assert dual.ranking[0].mode == "mono"
    assert dual.tier == "suggest"  # dual scores 0.5
    assert mono.tier == "ok"  # mono scores 0.8


def test_ok_tier_boundary_at_exactly_point_seven() -> None:
    # One figure among 100 blocks over 4 pages: only figure_heavy fires for
    # dual (0.30 penalty), landing the score on exactly 0.70.
    blocks = [_block("fig", block_type=BlockType.IMAGE)] + [_block(f"p{i}") for i in range(99)]
    advisory = advise_layout(blocks, "dual", profile="paper", page_count=4, figure_pages={1})
    dual = next(m for m in advisory.ranking if m.mode == "dual")
    assert dual.score == 0.70
    assert advisory.tier == "ok"


def test_suggest_tier_boundary_at_exactly_point_four() -> None:
    # High interruption (0.50) + fragment-heavy (0.10) -> dual exactly 0.40.
    structs = [_block(f"s{i}", block_type=BlockType.FORMULA) for i in range(5)]
    fragments = [_block(f"f{i}", source="") for i in range(21)]
    prose = [_block(f"p{i}") for i in range(75)]
    advisory = advise_layout(structs + fragments + prose, "dual", profile="paper", page_count=2)
    dual = next(m for m in advisory.ranking if m.mode == "dual")
    assert dual.score == 0.40
    assert advisory.tier == "suggest"


def test_recommended_is_ranking_head() -> None:
    advisory = advise_layout(
        _chaotic_blocks(), "dual", profile="paper", page_count=2, figure_pages={1, 2}
    )
    assert advisory.recommended == advisory.ranking[0].mode


def test_advisory_to_dict_shape_and_rounding() -> None:
    advisory = advise_layout([_block("a")], "dual", page_count=1)
    payload = advisory.to_dict()
    assert payload["requested"] == "dual"
    assert payload["tier"] == "ok"
    assert payload["recommended"] == "dual"
    assert payload["ranking"][0] == {"mode": "dual", "score": 1.0, "penalties": []}
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
    [("dual", "mono"), ("mono", "mono"), ("auto", "auto")],
)
def test_downgrade_ladder_and_off_ladder_passthrough(mode: DualMode, expected: DualMode) -> None:
    assert downgrade_mode(mode) == expected


def test_downgrade_ladder_matches_table() -> None:
    assert _DOWNGRADE_LADDER == {"dual": "mono", "mono": "mono"}


def _advisory(tier: str, recommended: DualMode = "mono") -> Advisory:
    return Advisory(
        requested="dual",
        tier=tier,  # type: ignore[arg-type]
        ranking=(ModeScore(recommended, 0.9), ModeScore("dual", 0.1)),
        reasons=(),
    )


def _hard() -> DifficultyAssessment:
    return DifficultyAssessment(hard=True, repair_share=0.5, reasons=("hard",))


def _easy() -> DifficultyAssessment:
    return DifficultyAssessment(hard=False, repair_share=0.0)


def test_advise_enforcement_always_keeps_requested() -> None:
    assert resolve_effective_mode("dual", _advisory("discourage"), _hard()) == "dual"
    assert (
        resolve_effective_mode("dual", _advisory("discourage"), _hard(), enforcement="advise")
        == "dual"
    )


def test_auto_keeps_requested_when_not_discouraged() -> None:
    assert resolve_effective_mode("dual", _advisory("ok"), _easy(), enforcement="auto") == "dual"
    assert (
        resolve_effective_mode("dual", _advisory("suggest"), _easy(), enforcement="auto") == "dual"
    )


def test_auto_switches_to_recommended_when_discouraged() -> None:
    assert (
        resolve_effective_mode("dual", _advisory("discourage", "mono"), _easy(), enforcement="auto")
        == "mono"
    )


def test_auto_downgrades_one_step_when_hard() -> None:
    assert resolve_effective_mode("dual", _advisory("ok"), _hard(), enforcement="auto") == "mono"


def test_auto_downgrades_after_switching_to_recommended() -> None:
    result = resolve_effective_mode(
        "dual", _advisory("discourage", "mono"), _hard(), enforcement="auto"
    )
    assert result == "mono"  # mono is the ladder floor


def test_auto_hard_requested_off_ladder_passes_through() -> None:
    # ``auto`` is never tiered, so a hard document leaves it alone.
    assert resolve_effective_mode("auto", _advisory("ok"), _hard(), enforcement="auto") == "auto"


# --------------------------------------------------------------------------- #
# The converged two-value space: ``auto`` and the pairing layout
# --------------------------------------------------------------------------- #


def test_auto_is_never_a_tiered_mode() -> None:
    """``auto`` is resolved before tiering; it is not itself a layout."""
    from ubt.core.policy.bilingual_advisor import MODES, RENDER_MODE_VALUE

    assert "auto" not in MODES
    assert "auto" not in RENDER_MODE_VALUE


def test_dual_is_the_only_pairing_layout() -> None:
    """There is exactly one bilingual layout: the per-block source/translation pair.

    A page-zip against the source cannot exist under reflow (it changes the page
    count), so ``inline``/``alternating``/``facing`` collapsed into one value.
    """
    from ubt.core.policy.bilingual_advisor import RENDER_MODE_VALUE, SECONDARY_SUFFIX

    assert RENDER_MODE_VALUE["dual"] == "bilingual"
    assert set(SECONDARY_SUFFIX) == {"dual", "mono"}


def test_a_structure_heavy_document_is_advised_toward_mono() -> None:
    """The advisor's whole point: pairing degrades on a chaotic document."""
    advisory = advise_layout(
        _chaotic_blocks(), "dual", profile="paper", page_count=2, figure_pages={1, 2}
    )
    assert advisory.tier == "discourage"
    assert advisory.recommended == "mono"
    assert advisory.to_dict()["recommended"] == "mono"


def test_every_advisor_mode_has_an_adapter_mapping() -> None:
    """A mode the advisor can recommend must be renderable."""
    from ubt.core.policy.bilingual_advisor import MODES, RENDER_MODE_VALUE

    for mode in MODES:
        assert mode in RENDER_MODE_VALUE, f"{mode} has no adapter mapping"
