"""Single detection primitive for glossary terminology drift.

Five implementations used to decide independently whether one glossary term
is drifted in one block:

* :class:`ubt.core.validators.consistency.GlossaryConsistencyValidator` — the
  export / quality-gate verdict;
* :func:`ubt.core.qe.term_metrics.evaluate_terms` — the quality report's
  terminology metrics *and* the consistency stage's repair planner;
* :class:`ubt.core.validators.span_repair.MQMSpanAnnotator` — the terminology
  section of the span annotator;
* :func:`ubt.core.qe.comet_runner.glossary_violation_flag` — a thin wrapper
  over the validator;
* and the read-only scan inside the rewriter itself.

Detection lives here. The rewriter
(:class:`ubt.core.validators.glossary_enforcer.DeterministicGlossaryEnforcer`)
stays independent — its job is to *apply* terms, not to *judge* them — but
every hit in this module comes from the rewriter's own
:func:`~ubt.core.validators.glossary_enforcer.find_term_occurrences`, so
detection and enforcement can never disagree about where a term occurs.

Source-side caliber: ``[source] + aliases`` — the stricter rule, taken from
``consistency.py``. ``evaluate_terms`` used to look only for ``source``, so a
block whose source carried a term *solely as an alias* was invisible to the
consistency-stage planner while the quality gate (via the validator) already
reported that very block as drifted.

Target-side, "the term is rendered" means the canonical rendering occurs
under the same boundary-aware matcher, outside protected spans, folding case
on both sides via ``re.IGNORECASE`` (offsets stay in the original text) — the
validator's historic tolerance: ``finfet器件`` still counts as
``FinFET器件``. The offending-surface scan used by span repair is deliberately
case-**sensitive**: it mirrors the rewriter, which does not fold case, so
repair never flags a surface the exporter would refuse to touch.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from ubt.core.validators.glossary_enforcer import extract_protected_spans, find_term_occurrences

#: Pre-extracted structural spans of one text (HTML/math/URL/masker regions),
#: see :func:`ubt.core.validators.glossary_enforcer.extract_protected_spans`.
ProtectedSpans = list[tuple[int, int]]


@dataclass(frozen=True, slots=True)
class GlossaryTerm:
    """One normalized, auditable glossary entry."""

    source: str
    expected: str
    aliases: tuple[str, ...] = ()
    inflected_variants: tuple[str, ...] = ()

    @property
    def source_surfaces(self) -> tuple[str, ...]:
        """``[source] + aliases``: every surface that makes a block carry this term.

        Deduplicated, alias order preserved — an alias equal to the source
        must not cost a second scan of the same string.
        """
        surfaces = [self.source]
        for alias in self.aliases:
            if alias and alias not in surfaces:
                surfaces.append(alias)
        return tuple(surfaces)


#: What every detector accepts: raw glossary dicts, or entries normalized
#: once by :func:`normalize_glossary` and reused across many blocks.
GlossaryInput = Sequence[Mapping[str, Any]] | Sequence[GlossaryTerm]


@dataclass(frozen=True, slots=True)
class TermDriftFinding:
    """Detection result for one ``(block, glossary entry)`` pair.

    ``source_hits`` / ``expected_hits`` are ``(start, end)`` offsets into the
    *original* (never lower-cased) texts, so a consumer can annotate or
    splice them. ``expected_hits`` is only populated when the source carries
    the term — the target scan is skipped otherwise — which is why
    :attr:`drifted` (not ``expected_hits``) is the drift verdict.
    """

    source: str
    expected: str
    aliases: tuple[str, ...]
    #: Surface found in the source (one of ``[source] + aliases``), ``""``
    #: when the source never carries the term.
    matched_surface: str
    source_hits: tuple[tuple[int, int], ...]
    expected_hits: tuple[tuple[int, int], ...]

    @property
    def occurs_in_source(self) -> bool:
        """Whether the source carries the term at all (source surface or alias)."""
        return bool(self.source_hits)

    @property
    def rendered(self) -> bool:
        """Whether the target carries the canonical rendering (only meaningful when :attr:`occurs_in_source`)."""
        return bool(self.expected_hits)

    @property
    def drifted(self) -> bool:
        """Source carries the term, target does not render it canonically."""
        return self.occurs_in_source and not self.expected_hits


@dataclass(frozen=True, slots=True)
class TargetTermViolation:
    """An offending surface found in the target text.

    ``alias`` — a listed alias standing where the canonical rendering should
    be; ``leak`` — the untranslated source term itself in the target. ``hits``
    are ``(start, end)`` offsets into the original target text.
    """

    source: str
    expected: str
    surface: str
    kind: Literal["alias", "leak"]
    hits: tuple[tuple[int, int], ...]


def normalize_glossary(glossary: Iterable[Mapping[str, Any]]) -> tuple[GlossaryTerm, ...]:
    """Normalize raw glossary dicts into auditable entries.

    Requires a non-empty source **and** a non-empty canonical rendering (an
    entry without one is not checkable — every consumer used to re-implement
    that filter), strips both, and strips/deduplicates the aliases.
    """
    terms: list[GlossaryTerm] = []
    for entry in glossary:
        source = str(entry.get("source", "")).strip()
        expected = str(entry.get("translation", "")).strip()
        if not source or not expected:
            continue
        aliases: list[str] = []
        for alias in entry.get("aliases") or []:
            if not alias:
                continue  # None/"" are absent aliases, not the text "None"
            cleaned = str(alias).strip()
            if cleaned and cleaned not in aliases:
                aliases.append(cleaned)
        inflected: list[str] = []
        for var in entry.get("inflected_variants") or []:
            if not var:
                continue
            cleaned = str(var).strip()
            if cleaned and cleaned not in inflected and cleaned != expected:
                inflected.append(cleaned)
        terms.append(
            GlossaryTerm(
                source=source,
                expected=expected,
                aliases=tuple(aliases),
                inflected_variants=tuple(inflected),
            )
        )
    return tuple(terms)


def _coerce_terms(glossary: GlossaryInput) -> tuple[GlossaryTerm, ...]:
    """Accept raw dicts or pre-normalized entries without re-paying the work."""
    if glossary and isinstance(glossary[0], GlossaryTerm):
        return tuple(cast(Sequence[GlossaryTerm], glossary))
    return normalize_glossary(cast(Sequence[Mapping[str, Any]], glossary))


def detect_term_drift(
    source_text: str,
    target_text: str,
    glossary: GlossaryInput,
    *,
    source_protected: ProtectedSpans | None = None,
    target_protected: ProtectedSpans | None = None,
) -> tuple[TermDriftFinding, ...]:
    """Detect every auditable entry's carrying/rendering state for one block pair.

    Source side is the strict caliber: **any** of ``[source] + aliases``
    occurring (boundary-aware, outside protected spans, case-insensitively)
    means the block carries the term. The target side is only scanned for
    entries the source carries, and asks whether the canonical rendering
    occurs under the same matcher — so ``finding.drifted`` is exactly
    "source carries it, target does not render it canonically".

    ``source_protected`` / ``target_protected`` let a caller match hundreds of
    terms against one block without re-deriving that block's nine structural
    spans per term; each is derived once here when omitted.
    """
    terms = _coerce_terms(glossary)
    if source_protected is None:
        source_protected = extract_protected_spans(source_text)
    if target_protected is None:
        target_protected = extract_protected_spans(target_text)

    findings: list[TermDriftFinding] = []
    for term in terms:
        matched = ""
        source_hits: tuple[tuple[int, int], ...] = ()
        # First matching surface wins (Python's ``any`` short-circuit rule
        # the validator used): whether the term is carried matters, which
        # surface carried it does not change the verdict.
        for surface in term.source_surfaces:
            # Acronym aliases and short uppercase terms (e.g. "BE" for "best-effort",
            # "RL", "LS", "MIG", "OCI", "SDK") must match case-sensitively in source text.
            # Otherwise, an alias like "BE" matches the ubiquitous English verb "be"
            # (as in "can be", "to be"), falsely accusing almost every block of drifting "best-effort".
            is_acronym = surface.isupper() and len(surface) <= 5
            hits = find_term_occurrences(
                source_text, surface, source_protected, case_insensitive=not is_acronym
            )
            if hits:
                matched = surface
                source_hits = tuple(hits)
                break

        expected_hits: tuple[tuple[int, int], ...] = ()
        if source_hits:
            hits = list(
                find_term_occurrences(
                    target_text, term.expected, target_protected, case_insensitive=True
                )
            )
            if not hits and term.inflected_variants:
                for variant in term.inflected_variants:
                    var_hits = find_term_occurrences(
                        target_text, variant, target_protected, case_insensitive=True
                    )
                    if var_hits:
                        hits.extend(var_hits)
            expected_hits = tuple(hits)

        findings.append(
            TermDriftFinding(
                source=term.source,
                expected=term.expected,
                aliases=term.aliases,
                matched_surface=matched,
                source_hits=source_hits,
                expected_hits=expected_hits,
            )
        )
    return tuple(findings)


def detect_target_term_violations(
    target_text: str,
    glossary: GlossaryInput,
    *,
    target_protected: ProtectedSpans | None = None,
) -> tuple[TargetTermViolation, ...]:
    """Scan a target for surfaces that are *not* the canonical rendering.

    Two kinds, mirroring what the rewriter would rewrite:

    * ``alias`` — a listed alias (other than the rendering itself) present in
      the target;
    * ``leak`` — the untranslated source term (>2 chars, distinct from the
      rendering) sitting in the target.

    Matching is case-sensitive on purpose: the rewriter replaces exact
    surfaces only, so a span the repair model is asked to fix must be a span
    the exporter would have fixed too. Order is glossary order and, within an
    entry, alias-then-leak, so downstream span ids stay stable.
    """
    terms = _coerce_terms(glossary)
    if target_protected is None:
        target_protected = extract_protected_spans(target_text)

    # A surface that is another entry's canonical rendering (or inflected
    # variant) is approved globally: the rewriter's own two-pass guard leaves it
    # alone, so flagging it here would ask the repair model to "fix" a term the
    # exporter deliberately accepts.
    globally_approved: set[str] = set()
    for other in terms:
        globally_approved.add(other.expected)
        globally_approved.update(other.inflected_variants)

    violations: list[TargetTermViolation] = []
    for term in terms:
        approved_target = {term.expected, *term.inflected_variants}
        for alias in term.aliases:
            if alias in approved_target or alias in globally_approved:
                continue
            hits = tuple(find_term_occurrences(target_text, alias, target_protected))
            if hits:
                violations.append(
                    TargetTermViolation(
                        source=term.source,
                        expected=term.expected,
                        surface=alias,
                        kind="alias",
                        hits=hits,
                    )
                )
        if (
            term.source not in approved_target
            and term.source not in globally_approved
            and (len(term.source) > 2 or term.source.isupper())
        ):
            hits = tuple(find_term_occurrences(target_text, term.source, target_protected))

            if hits:
                violations.append(
                    TargetTermViolation(
                        source=term.source,
                        expected=term.expected,
                        surface=term.source,
                        kind="leak",
                        hits=hits,
                    )
                )
    return tuple(violations)
