"""Bilingual-mode advisory: pre/post-translation render-mode recommendation.

Industry主流 (BabelDOC ACL 2026, Immersive Translate) does NOT do
suitability detection: they emit dual + mono outputs and let the user pick
(``--use-alternating-pages-dual`` / ``--no-dual`` / ``--no-mono``). This
module fills that gap as a differentiator, but aligned with the mainstream:

- it never blocks a job and never silently overrides an explicit user mode;
- its "discourage inline interleave" verdict promotes the industry-standard
  answer for figure-dense docs (alternating-pages dual), not "no bilingual";
- thresholds are documented as BIoU-calibration targets (see
  ``scripts/biou_score.py`` and the evaluation guide), not ground truth.

Two phases, both zero-LLM:

- Phase 1 (post-ingest, pre-draft): layout suitability from IR structure —
  interruption density, figure coverage, untranslatable share. Decides the
  render-mode ranking and the ``ok / suggest / discourage`` tier for the
  requested mode.
- Phase 2 (post-repair, pre-export): difficulty re-assessment from live
  pipeline signals (repair + failure rate). A "hard" document can only
  downgrade the mode one step (inline -> alternating -> monolingual).

Design rules (elegant / robust / extensible):

- Pure functions of small dataclasses: no I/O, no singletons, trivially
  unit-testable with synthetic block lists (no PDF fixtures needed).
- Rules are data (``RULES`` registry): name, per-mode penalty weights,
  predicate over signals, explain template. New signal = one row + test.
- Assessment never enforces: enforcement (advise / auto / explicit) lives
  in the pipeline/export layer. ``explicit user flag > auto > advise``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ubt.core.config import DualMode
from ubt.core.ir.models import BlockType, IRBlock

AdviseTier = Literal["ok", "suggest", "discourage"]

MODES: tuple[DualMode, ...] = ("inline", "alternating", "monolingual")

# Tier cutoffs on the requested mode's suitability score.
_OK_CUTOFF = 0.70
_SUGGEST_CUTOFF = 0.40

# Profile sensitivity multipliers on rule penalties: technical profiles keep
# full weight (figures/formulas break interleave fast); general prose
# (novels, essays) is lenient. BIoU-calibration targets, see docs.
_PROFILE_SENSITIVITY = {"paper": 1.0, "textbook": 1.0, "general": 0.8}

# Advisor DualMode -> adapter render bilingual_mode value.
# "monolingual" needs no adapter change: it maps to bilingual=False
# (target-only), which the Typst reconstructor already supports.
RENDER_MODE_VALUE: dict[DualMode, str] = {
    "inline": "bilingual",
    "alternating": "alternating",
    "monolingual": "monolingual",
    "facing": "facing_spread",
}

# Secondary artifact suffixes for dual output (BabelDOC no-dual/no-mono style).
SECONDARY_SUFFIX: dict[DualMode, str] = {
    "inline": "_mono",
    "alternating": "_mono",
    "facing": "_mono",
    "monolingual": "_dual",
}


def secondary_mode(primary: DualMode) -> DualMode:
    """Complementary artifact for dual output (mono <-> dual)."""
    return "monolingual" if primary != "monolingual" else "alternating"


# Downgrade ladder for Phase-2 difficulty (one step only, never two).
_DOWNGRADE_LADDER: dict[DualMode, DualMode] = {
    "inline": "alternating",
    "alternating": "monolingual",
    "monolingual": "monolingual",
}
# Phase-2 trigger: repaired + failed share at/above this downgrades one step.
_DIFFICULTY_DOWNGRADE_RATE = 0.15


@dataclass(frozen=True)
class DocSignals:
    """Zero-cost layout signals collected post-ingest (pre-draft)."""

    total_blocks: int = 0
    page_count: int = 0
    interruption_per_page: float = 0.0  # formula+table+image+code blocks / page
    struct_block_share: float = 0.0  # formula+table+image+code / total
    figure_page_share: float = 0.0  # pages carrying figures / page_count
    skip_share: float = 0.0  # F2 skip_translate / total
    fragment_share: float = 0.0  # source chars < 25 / total
    prose_char_share: float = 0.0  # narrative+heading+list chars / chars


@dataclass(frozen=True)
class AdvisorRule:
    """One advisory rule: predicate + per-mode penalty weights + explanation."""

    name: str
    penalties: dict[DualMode, float]
    fires: Callable[[DocSignals], str | None]  # detail string or None


def _rule_high_interruption() -> AdvisorRule:
    return AdvisorRule(
        name="high_interruption_density",
        penalties={"inline": 0.50, "alternating": 0.10, "monolingual": 0.0},
        fires=lambda s: (
            f"{s.interruption_per_page:.2f} structural interruptions/page"
            if s.interruption_per_page >= 1.0
            else None
        ),
    )


def _rule_medium_interruption() -> AdvisorRule:
    return AdvisorRule(
        name="medium_interruption_density",
        penalties={"inline": 0.25, "alternating": 0.05, "monolingual": 0.0},
        fires=lambda s: (
            f"{s.interruption_per_page:.2f} structural interruptions/page"
            if 0.5 <= s.interruption_per_page < 1.0
            else None
        ),
    )


def _rule_figure_heavy() -> AdvisorRule:
    return AdvisorRule(
        name="figure_heavy",
        penalties={"inline": 0.30, "alternating": 0.05, "monolingual": 0.0},
        fires=lambda s: (
            f"{s.figure_page_share:.0%} of pages carry figures"
            if s.figure_page_share >= 0.15
            else None
        ),
    )


def _rule_struct_heavy() -> AdvisorRule:
    return AdvisorRule(
        name="struct_block_heavy",
        penalties={"inline": 0.20, "alternating": 0.05, "monolingual": 0.0},
        fires=lambda s: (
            f"{s.struct_block_share:.0%} of blocks are formula/table/figure/code"
            if s.struct_block_share >= 0.05
            else None
        ),
    )


def _rule_untranslatable_heavy() -> AdvisorRule:
    return AdvisorRule(
        name="untranslatable_heavy",
        penalties={"inline": 0.15, "alternating": 0.05, "monolingual": 0.0},
        fires=lambda s: (
            f"{s.skip_share:.0%} of blocks ship verbatim (interleave stutter)"
            if s.skip_share >= 0.05
            else None
        ),
    )


def _rule_fragment_heavy() -> AdvisorRule:
    return AdvisorRule(
        name="fragment_heavy",
        penalties={"inline": 0.10, "alternating": 0.0, "monolingual": 0.0},
        fires=lambda s: (
            f"{s.fragment_share:.0%} of blocks are sub-sentence fragments"
            if s.fragment_share >= 0.20
            else None
        ),
    )


# Registry order is explanation order (highest typical impact first).
RULES: tuple[AdvisorRule, ...] = (
    _rule_high_interruption(),
    _rule_medium_interruption(),
    _rule_figure_heavy(),
    _rule_struct_heavy(),
    _rule_untranslatable_heavy(),
    _rule_fragment_heavy(),
)


@dataclass(frozen=True)
class ModeScore:
    mode: DualMode
    score: float
    penalties: tuple[str, ...] = ()


@dataclass(frozen=True)
class Advisory:
    """Phase-1 verdict: per-mode scores, tier for the requested mode, ranking."""

    requested: DualMode
    tier: AdviseTier
    ranking: tuple[ModeScore, ...]  # best-first
    reasons: tuple[str, ...]  # fired-rule explanations, impact order
    signals: DocSignals = field(default_factory=DocSignals)

    @property
    def recommended(self) -> DualMode:
        return self.ranking[0].mode

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "tier": self.tier,
            "recommended": self.recommended,
            "ranking": [
                {"mode": m.mode, "score": round(m.score, 3), "penalties": list(m.penalties)}
                for m in self.ranking
            ],
            "reasons": list(self.reasons),
            "signals": {
                "total_blocks": self.signals.total_blocks,
                "page_count": self.signals.page_count,
                "interruption_per_page": round(self.signals.interruption_per_page, 3),
                "struct_block_share": round(self.signals.struct_block_share, 3),
                "figure_page_share": round(self.signals.figure_page_share, 3),
                "skip_share": round(self.signals.skip_share, 3),
                "fragment_share": round(self.signals.fragment_share, 3),
                "prose_char_share": round(self.signals.prose_char_share, 3),
            },
        }


_STRUCT_TYPES = frozenset({BlockType.FORMULA, BlockType.TABLE, BlockType.IMAGE, BlockType.CODE})
_PROSE_TYPES = frozenset(
    {BlockType.NARRATIVE, BlockType.HEADING, BlockType.LIST_ITEM, BlockType.DIALOGUE}
)


def collect_signals(
    blocks: list[IRBlock],
    *,
    page_count: int = 0,
    figure_pages: set[int] | None = None,
) -> DocSignals:
    """Aggregate zero-cost layout signals from post-ingest IR blocks."""
    total = len(blocks)
    if total == 0:
        return DocSignals(page_count=page_count)
    pages = page_count
    if pages <= 0:
        page_nos = [b.bbox.page for b in blocks if b.bbox and b.bbox.page > 0]
        pages = max(page_nos) if page_nos else 1
    struct = sum(1 for b in blocks if b.block_type in _STRUCT_TYPES)
    skipped = sum(1 for b in blocks if b.skip_translate)
    fragments = sum(1 for b in blocks if len((b.source_text or "").strip()) < 25)
    prose_chars = sum(len(b.source_text or "") for b in blocks if b.block_type in _PROSE_TYPES)
    all_chars = sum(len(b.source_text or "") for b in blocks)
    figs = figure_pages or set()
    return DocSignals(
        total_blocks=total,
        page_count=pages,
        interruption_per_page=struct / pages,
        struct_block_share=struct / total,
        figure_page_share=len(figs) / pages if pages else 0.0,
        skip_share=skipped / total,
        fragment_share=fragments / total,
        prose_char_share=prose_chars / all_chars if all_chars else 0.0,
    )


def advise_layout(
    blocks: list[IRBlock],
    requested: DualMode = "inline",
    *,
    profile: str = "general",
    page_count: int = 0,
    figure_pages: set[int] | None = None,
) -> Advisory:
    """Phase-1: score each render mode and tier the requested one."""
    if requested not in MODES:
        requested = "inline"  # e.g. config-level "auto" never reaches tiering
    signals = collect_signals(blocks, page_count=page_count, figure_pages=figure_pages)
    sensitivity = _PROFILE_SENSITIVITY.get(profile, 0.8)
    scored: list[ModeScore] = []
    for mode in MODES:
        score = 1.0
        hits: list[str] = []
        for rule in RULES:
            detail = rule.fires(signals)
            if detail is None:
                continue
            penalty = rule.penalties.get(mode, 0.0) * sensitivity
            if penalty > 0:
                score -= penalty
                hits.append(rule.name)
        scored.append(ModeScore(mode=mode, score=round(max(0.0, score), 3), penalties=tuple(hits)))
    scored.sort(key=lambda m: m.score, reverse=True)
    by_mode = {m.mode: m for m in scored}
    requested_score = by_mode[requested].score
    tier: AdviseTier
    if requested_score >= _OK_CUTOFF:
        tier = "ok"
    elif requested_score >= _SUGGEST_CUTOFF:
        tier = "suggest"
    else:
        tier = "discourage"
    fired_names = {p for m in scored for p in m.penalties}
    reasons = tuple(
        f"{rule.name}: {rule.fires(signals)}" for rule in RULES if rule.name in fired_names
    )
    return Advisory(
        requested=requested, tier=tier, ranking=tuple(scored), reasons=reasons, signals=signals
    )


@dataclass(frozen=True)
class DifficultyAssessment:
    """Phase-2 verdict from live pipeline signals (post-repair, pre-export)."""

    hard: bool
    repair_share: float
    reasons: tuple[str, ...] = ()


def assess_difficulty(
    *,
    total: int,
    repaired: int,
    failed: int,
) -> DifficultyAssessment:
    """Flag documents whose repair burden suggests a render-mode downgrade."""
    share = (repaired + failed) / total if total > 0 else 0.0
    if share >= _DIFFICULTY_DOWNGRADE_RATE:
        return DifficultyAssessment(
            hard=True,
            repair_share=round(share, 3),
            reasons=(f"repair burden {share:.0%} >= {_DIFFICULTY_DOWNGRADE_RATE:.0%}",),
        )
    return DifficultyAssessment(hard=False, repair_share=round(share, 3))


def downgrade_mode(mode: DualMode) -> DualMode:
    """One-step downgrade on the inline -> alternating -> monolingual ladder.

    Modes off the ladder (facing/auto) pass through unchanged instead of
    raising KeyError on a public function.
    """
    return _DOWNGRADE_LADDER.get(mode, mode)


def resolve_effective_mode(
    requested: DualMode,
    advisory: Advisory,
    difficulty: DifficultyAssessment,
    *,
    enforcement: Literal["advise", "auto"] = "advise",
) -> DualMode:
    """Enforcement point: explicit requests always win; auto may downgrade.

    - ``advise`` (default): the requested mode stands; the advisory only
      warns (surfaced in logs / events / quality report).
    - ``auto``: a ``discourage`` tier switches to the recommended mode, and
      a hard Phase-2 difficulty downgrades one further step at most.
    """
    if enforcement != "auto":
        return requested
    effective = advisory.recommended if advisory.tier == "discourage" else requested
    if difficulty.hard:
        effective = downgrade_mode(effective)
    return effective
