"""Hierarchical glossary scope inheritance and context-aware disambiguation arbiter.

Solves the homograph and polysemy challenge in multidisciplinary monograph translation:
Terms such as 'driver' (hardware driver vs. psychological driver vs. golf driver),
'cell' (biological cell vs. telecom cell vs. battery cell), or 'transformer'
(electrical transformer vs. neural architecture) require distinct translations
depending on the active chapter, section, and semantic context.

Architecture:
1. Scope Hierarchy:
   BLOCK (most specific) -> SECTION -> CHAPTER -> DOMAIN -> GLOBAL (*)
2. Inheritance Tree:
   Child scopes inherit parent terms, overriding conflicting parent translations.
3. Context Disambiguation Arbiter:
   When competing entries share the same scope level, evaluates positive context hints
   and negative disqualifying hints against surrounding block text and headings.
4. Conflict Audit:
   Scans glossaries for homographs, verifying whether conflicts are safely resolved
   by scope boundaries or context hints versus unresolvable ambiguities.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from ubt.core.memory.bible import BibleEntry


class ScopeLevel(StrEnum):
    """Precedence hierarchy of glossary definition scopes."""

    BLOCK = "block"
    SECTION = "section"
    CHAPTER = "chapter"
    DOMAIN = "domain"
    GLOBAL = "global"

    @property
    def rank(self) -> int:
        """Numeric rank for scope comparison (higher rank = higher precedence)."""
        ranks = {
            ScopeLevel.BLOCK: 50,
            ScopeLevel.SECTION: 40,
            ScopeLevel.CHAPTER: 30,
            ScopeLevel.DOMAIN: 20,
            ScopeLevel.GLOBAL: 10,
        }
        return ranks[self]


class ScopeContext(BaseModel):
    """Contextual metadata of a chunk/block used for scope matching and disambiguation."""

    model_config = ConfigDict(extra="ignore")

    chapter_index: int | None = None
    chapter_title: str | None = None
    section_index: str | None = None
    section_title: str | None = None
    block_id: str | None = None
    domain: str | None = None
    surrounding_text: str = ""

    def combined_text(self) -> str:
        """Combine all text fields into a unified string for semantic hint matching."""
        parts = [
            self.chapter_title or "",
            self.section_title or "",
            self.surrounding_text or "",
        ]
        return " ".join(filter(None, parts)).lower()


class GlossaryConflictVerdict(StrEnum):
    """Verdict of homograph arbitration."""

    RESOLVED_BY_SCOPE = "resolved_by_scope"
    RESOLVED_BY_CONTEXT = "resolved_by_context"
    RESOLVED_BY_PRIORITY = "resolved_by_priority"
    RESOLVED_BY_FREQUENCY = "resolved_by_frequency"
    UNRESOLVED_AMBIGUOUS = "unresolved_ambiguous"


class GlossaryConflictItem(BaseModel):
    """Details of a discovered homograph conflict and its arbitration verdict."""

    model_config = ConfigDict(extra="ignore")

    source: str
    competing_targets: list[str]
    scopes: list[str]
    verdict: GlossaryConflictVerdict
    resolved_target: str | None = None
    reason: str


class GlossaryAuditSummary(BaseModel):
    """Aggregate audit report for glossary conflict arbitration."""

    model_config = ConfigDict(extra="ignore")

    total_terms: int
    unique_sources: int
    conflicting_sources: int
    resolved_conflicts: int
    unresolved_conflicts: int
    conflicts: list[GlossaryConflictItem] = Field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """True when there are zero unresolved homograph ambiguities."""
        return self.unresolved_conflicts == 0


def parse_scope_spec(scope: str | None) -> tuple[ScopeLevel, str]:
    """Parse a scope string (e.g. 'chapter:2', 'section:3.1', 'domain:semiconductor', '*').

    Returns (ScopeLevel, identifier).
    """
    if not scope or scope.strip() in ("*", "global", ""):
        return ScopeLevel.GLOBAL, "*"

    raw = scope.strip()
    if ":" in raw:
        prefix, ident = raw.split(":", 1)
        prefix = prefix.strip().lower()
        ident = ident.strip()
        if prefix in ("block", "blk"):
            return ScopeLevel.BLOCK, ident
        if prefix in ("section", "sec"):
            return ScopeLevel.SECTION, ident
        if prefix in ("chapter", "ch"):
            return ScopeLevel.CHAPTER, ident
        if prefix in ("domain", "dom"):
            return ScopeLevel.DOMAIN, ident

    # Shorthand chapter matching: e.g. "ch2" or numeric "2"
    if raw.lower().startswith("ch") and raw[2:].isdigit():
        return ScopeLevel.CHAPTER, raw[2:]
    if raw.isdigit():
        return ScopeLevel.CHAPTER, raw

    return ScopeLevel.GLOBAL, raw


def matches_scope(scope_spec: str | None, context: ScopeContext) -> tuple[bool, ScopeLevel]:
    """Check if an entry's scope specification matches the current execution context.

    Returns (is_match, resolved_scope_level).
    """
    level, ident = parse_scope_spec(scope_spec)

    if level == ScopeLevel.GLOBAL:
        return True, ScopeLevel.GLOBAL

    if level == ScopeLevel.BLOCK:
        if context.block_id and context.block_id.lower() == ident.lower():
            return True, ScopeLevel.BLOCK
        return False, ScopeLevel.BLOCK

    if level == ScopeLevel.SECTION:
        if context.section_index and str(context.section_index).lower() == ident.lower():
            return True, ScopeLevel.SECTION
        if context.section_title and ident.lower() in context.section_title.lower():
            return True, ScopeLevel.SECTION
        return False, ScopeLevel.SECTION

    if level == ScopeLevel.CHAPTER:
        if context.chapter_index is not None and str(context.chapter_index) == ident:
            return True, ScopeLevel.CHAPTER
        if context.chapter_title and ident.lower() in context.chapter_title.lower():
            return True, ScopeLevel.CHAPTER
        return False, ScopeLevel.CHAPTER

    if level == ScopeLevel.DOMAIN:
        if context.domain and context.domain.lower() == ident.lower():
            return True, ScopeLevel.DOMAIN
        return False, ScopeLevel.DOMAIN

    return False, ScopeLevel.GLOBAL


def _get_entry_field(entry: BibleEntry | dict[str, Any], field: str, default: Any = None) -> Any:
    """Read a field safely from a BibleEntry object or dictionary."""
    if isinstance(entry, dict):
        return entry.get(field, default)
    return getattr(entry, field, default)


def score_context_affinity(entry: BibleEntry | dict[str, Any], text: str) -> float:
    """Calculate semantic affinity score based on context hints and negative hints.

    Positive hints increase score by 1.0 each.
    Negative disqualifying hints disqualify the entry (score -100.0).
    """
    if not text:
        return 0.0

    context_hints = _get_entry_field(entry, "context_hints", []) or []
    negative_hints = _get_entry_field(entry, "negative_hints", []) or []

    text_lower = text.lower()

    # Disqualification check
    for neg in negative_hints:
        if not neg:
            continue
        neg_clean = str(neg).strip().lower()
        if re.search(r"\b" + re.escape(neg_clean) + r"\b", text_lower) or neg_clean in text_lower:
            return -100.0

    score = 0.0
    for hint in context_hints:
        if not hint:
            continue
        hint_clean = str(hint).strip().lower()
        if re.search(r"\b" + re.escape(hint_clean) + r"\b", text_lower) or hint_clean in text_lower:
            score += 1.0

    return score


def arbitrate_competing_entries(
    entries: Sequence[BibleEntry | dict[str, Any]],
    context: ScopeContext,
) -> tuple[dict[str, Any], GlossaryConflictVerdict, str]:
    """Arbitrate among multiple competing glossary definitions for a single source term.

    Decision hierarchy:
    1. Scope specificity (BLOCK > SECTION > CHAPTER > DOMAIN > GLOBAL)
    2. Context affinity score (matching positive hints without negative hits)
    3. User priority field (higher priority wins)
    4. Attest frequency (higher frequency wins)
    5. Deterministic tie-breaker
    """
    if not entries:
        raise ValueError("Cannot arbitrate empty entries")
    if len(entries) == 1:
        entry = entries[0]
        raw_dict = entry if isinstance(entry, dict) else entry.model_dump()
        return raw_dict, GlossaryConflictVerdict.RESOLVED_BY_SCOPE, "Single candidate"

    # Step 1: Score scope matches
    matching_candidates: list[tuple[int, BibleEntry | dict[str, Any], ScopeLevel]] = []
    for e in entries:
        scope = _get_entry_field(e, "scope", "*")
        is_match, level = matches_scope(scope, context)
        if is_match:
            matching_candidates.append((level.rank, e, level))

    if not matching_candidates:
        # Fall back to global candidates if none matched current scope
        for e in entries:
            scope = _get_entry_field(e, "scope", "*")
            level, _ = parse_scope_spec(scope)
            if level == ScopeLevel.GLOBAL:
                matching_candidates.append((level.rank, e, level))

    if not matching_candidates:
        # Last resort: take all
        matching_candidates = [(0, e, ScopeLevel.GLOBAL) for e in entries]

    # Find highest scope rank among matches
    highest_rank = max(r for r, _, _ in matching_candidates)
    top_scope_candidates = [e for r, e, _ in matching_candidates if r == highest_rank]

    if len(top_scope_candidates) == 1:
        winner = top_scope_candidates[0]
        raw_dict = winner if isinstance(winner, dict) else winner.model_dump()
        scope_str = _get_entry_field(winner, "scope", "*")
        return (
            raw_dict,
            GlossaryConflictVerdict.RESOLVED_BY_SCOPE,
            f"Overridden by narrower scope: '{scope_str}'",
        )

    # Step 2: Context hints affinity
    combined_ctx_text = context.combined_text()
    candidates_with_affinity: list[tuple[float, BibleEntry | dict[str, Any]]] = [
        (score_context_affinity(e, combined_ctx_text), e) for e in top_scope_candidates
    ]

    max_affinity = max(aff for aff, _ in candidates_with_affinity)
    if max_affinity > 0.0:
        affinity_winners = [e for aff, e in candidates_with_affinity if aff == max_affinity]
        if len(affinity_winners) == 1:
            winner = affinity_winners[0]
            raw_dict = winner if isinstance(winner, dict) else winner.model_dump()
            return (
                raw_dict,
                GlossaryConflictVerdict.RESOLVED_BY_CONTEXT,
                f"Resolved by context affinity match (score: {max_affinity})",
            )
        top_scope_candidates = affinity_winners

    # Step 3: Explicit priority
    max_prio = max(int(_get_entry_field(e, "priority", 0) or 0) for e in top_scope_candidates)
    prio_winners = [
        e for e in top_scope_candidates if int(_get_entry_field(e, "priority", 0) or 0) == max_prio
    ]
    if len(prio_winners) == 1 and max_prio > 0:
        winner = prio_winners[0]
        raw_dict = winner if isinstance(winner, dict) else winner.model_dump()
        return (
            raw_dict,
            GlossaryConflictVerdict.RESOLVED_BY_PRIORITY,
            f"Resolved by explicit priority: {max_prio}",
        )
    if prio_winners:
        top_scope_candidates = prio_winners

    # Step 4: Frequency rank
    max_freq = max(int(_get_entry_field(e, "frequency", 0) or 0) for e in top_scope_candidates)
    freq_winners = [
        e for e in top_scope_candidates if int(_get_entry_field(e, "frequency", 0) or 0) == max_freq
    ]
    if len(freq_winners) == 1 and max_freq > 0:
        winner = freq_winners[0]
        raw_dict = winner if isinstance(winner, dict) else winner.model_dump()
        return (
            raw_dict,
            GlossaryConflictVerdict.RESOLVED_BY_FREQUENCY,
            f"Resolved by attest frequency: {max_freq}",
        )

    # Step 5: Unresolved ambiguity (tie-breaker deterministic first)
    winner = top_scope_candidates[0]
    raw_dict = winner if isinstance(winner, dict) else winner.model_dump()
    return (
        raw_dict,
        GlossaryConflictVerdict.UNRESOLVED_AMBIGUOUS,
        "Ambiguous definitions share identical scope and affinity; defaulted to first definition",
    )


def resolve_effective_glossary(
    entries: Sequence[BibleEntry | dict[str, Any]],
    context: ScopeContext | None = None,
) -> list[dict[str, Any]]:
    """Resolve a list of scoped glossary definitions into an unambiguous effective glossary.

    For each unique source term, evaluates active scopes and context hints,
    ensuring the emitted list contains exactly one canonical translation per source.
    """
    ctx = context or ScopeContext()
    grouped_by_source: dict[str, list[BibleEntry | dict[str, Any]]] = {}

    for entry in entries:
        src = str(_get_entry_field(entry, "source", "")).strip()
        tgt = str(
            _get_entry_field(entry, "translation", "") or _get_entry_field(entry, "target", "")
        ).strip()
        if not src or not tgt:
            continue
        key = src.lower()
        grouped_by_source.setdefault(key, []).append(entry)

    effective: list[dict[str, Any]] = []
    for _, candidates in sorted(grouped_by_source.items(), key=lambda kv: kv[0]):
        winner, _, _ = arbitrate_competing_entries(candidates, ctx)
        effective.append(winner)

    return effective


def audit_glossary_conflicts(
    entries: Sequence[BibleEntry | dict[str, Any]],
    contexts: Sequence[ScopeContext] | None = None,
) -> GlossaryAuditSummary:
    """Audit a glossary file for conflicting translations and homographs.

    Evaluates whether conflicting definitions are partitioned by distinct scopes
    or context hints.
    """
    grouped_by_source: dict[str, list[BibleEntry | dict[str, Any]]] = {}
    for entry in entries:
        src = str(_get_entry_field(entry, "source", "")).strip()
        if src:
            grouped_by_source.setdefault(src, []).append(entry)

    conflict_items: list[GlossaryConflictItem] = []
    unresolved_count = 0
    resolved_count = 0

    eval_contexts = list(contexts) if contexts else [ScopeContext()]

    for source, candidates in sorted(grouped_by_source.items(), key=lambda kv: kv[0]):
        # Extract unique translations
        targets = list(
            dict.fromkeys(
                str(
                    _get_entry_field(c, "translation", "") or _get_entry_field(c, "target", "")
                ).strip()
                for c in candidates
                if str(
                    _get_entry_field(c, "translation", "") or _get_entry_field(c, "target", "")
                ).strip()
            )
        )
        if len(targets) <= 1:
            continue

        scopes = [str(_get_entry_field(c, "scope", "*")) for c in candidates]

        # Test arbitration across sample contexts
        verdicts = [arbitrate_competing_entries(candidates, ctx) for ctx in eval_contexts]
        # If all scopes are distinct non-global, or if context affinity resolved them
        has_unresolved = any(
            v == GlossaryConflictVerdict.UNRESOLVED_AMBIGUOUS for _, v, _ in verdicts
        )

        winner_dict, best_verdict, best_reason = verdicts[0]
        resolved_tgt = str(winner_dict.get("translation", "") or winner_dict.get("target", ""))

        if has_unresolved:
            unresolved_count += 1
            conflict_items.append(
                GlossaryConflictItem(
                    source=source,
                    competing_targets=targets,
                    scopes=scopes,
                    verdict=GlossaryConflictVerdict.UNRESOLVED_AMBIGUOUS,
                    resolved_target=resolved_tgt,
                    reason=best_reason,
                )
            )
        else:
            resolved_count += 1
            conflict_items.append(
                GlossaryConflictItem(
                    source=source,
                    competing_targets=targets,
                    scopes=scopes,
                    verdict=best_verdict,
                    resolved_target=resolved_tgt,
                    reason=best_reason,
                )
            )

    return GlossaryAuditSummary(
        total_terms=len(entries),
        unique_sources=len(grouped_by_source),
        conflicting_sources=len(conflict_items),
        resolved_conflicts=resolved_count,
        unresolved_conflicts=unresolved_count,
        conflicts=conflict_items,
    )


__all__ = [
    "GlossaryAuditSummary",
    "GlossaryConflictItem",
    "GlossaryConflictVerdict",
    "ScopeContext",
    "ScopeLevel",
    "arbitrate_competing_entries",
    "audit_glossary_conflicts",
    "matches_scope",
    "parse_scope_spec",
    "resolve_effective_glossary",
    "score_context_affinity",
]
