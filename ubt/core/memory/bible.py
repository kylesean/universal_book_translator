import logging
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ubt.core.memory.cjk_matcher import contains_cjk

logger = logging.getLogger(__name__)


class BibleEntry(BaseModel):
    """Single proper name or terminology mapping in the Translation Bible."""

    model_config = ConfigDict(extra="ignore")

    source: str
    translation: str
    aliases: list[str] = Field(default_factory=list)
    #: Alternate *approved* renderings of the same concept: morphological
    #: variants (``current optimal`` / ``currently optimal``) and admitted
    #: synonyms. They are accepted where they appear and never rewritten, but
    #: ``translation`` stays the preferred form the draft is steered to. This is
    #: the terminology-management ``admitted term`` slot next to the
    #: ``preferred term``; :func:`merge_bible_entries` fills it when two entries
    #: for one concept disagree about the rendering.
    inflected_variants: list[str] = Field(default_factory=list)
    kind: str = "term"  # person, place, term
    frequency: int = 0  # Global attest count driving top-N ranking


class BookBible(BaseModel):
    """Whole-book translation bible containing characters, terms, tone, and system prompt."""

    model_config = ConfigDict(extra="ignore")

    doc_id: str
    language: str = "zh"
    tone: str = ""
    glossary: list[BibleEntry] = Field(default_factory=list)
    system_instruction: str = ""


def normalize_glossary_spacing(translation: str, source: str = "") -> str:
    """Insert one half-width space at every CJK<->Latin/digit boundary of a rendering.

    The canonical form the glossary enforces must match the typography the
    pipeline itself ships: the export stage applies Pangu spacing
    (:func:`ubt.core.cleaners.cjk_spacing.apply_pangu_spacing`) to every
    target, so a rendering mined without the space ("切换Transformer") fights
    the document's own convention ("切换 Transformer") for the rest of the
    book — measured on the arXiv 2101.03961 run: the unspaced entry sent three
    blocks into repair loops for "drifting" away from a form the enforcer's
    own exporter would rewrite. Matching is space-tolerant both ways (see
    ``_term_pattern``), so normalizing here only aligns the *canonical
    surface* shown to models and written by the enforcer. Latin-Latin
    spacing ("New York") is never touched; spaces already present at a
    boundary are preserved, not doubled.
    """
    tr = translation or ""
    if not tr or not source:
        return tr
    # Only normalize when the source term is Latin-script: a CJK or Cyrillic
    # source never implies an en->zh typographic convention.
    from ubt.core.validators.glossary_enforcer import is_cjk_char

    if is_cjk_char(source[0]) or is_cjk_char(source[-1]):
        return tr

    out: list[str] = []
    # Boundary pairs the export stage's Pangu rules also space: CJK<->Latin
    # letter/digit, and CJK<->half-width bracket (式(3.1) -> 式 (3.1)). A space
    # is never inserted against full-width punctuation — "多头注意力（MHA）"
    # stays glued, matching apply_pangu_spacing exactly.
    _latin = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    _open_half = "([{"
    _close_half = ")]}"
    for ch in tr:
        prev = out[-1] if out else ""
        if not prev or prev in " \t" or ch in " \t" or is_cjk_char(prev) == is_cjk_char(ch):
            out.append(ch)
            continue
        prev_half = prev in _latin or prev in _close_half
        ch_half = ch in _latin or ch in _open_half
        if prev_half or ch_half:
            out.append(" ")
        out.append(ch)
    return "".join(out)


def clean_bible_entry(
    source: str,
    translation: str,
    aliases: list[str] | None = None,
    kind: str = "term",
) -> BibleEntry | None:
    """Clean extracted entry using 0-token deterministic guardrails."""
    src = source.strip()
    # Strip CJK book quotes (simplified + Japanese style), angle brackets, and quotes
    tr = translation.strip().strip("《》〈〉「」『』\"'“”")

    if not src or not tr:
        return None

    # Guardrail: overly long items are likely slogans/captions, drop them.
    # Whitespace tokenization only works for alphabetic source scripts; CJK
    # sources are length-capped in characters instead.
    if contains_cjk(src):
        if len(src) > 12:
            return None
    elif len(src.split()) > 4:
        return None

    # Canonical typography: the enforcer writes this string into targets, so
    # it must carry the same CJK<->Latin spacing the export stage ships
    # (Pangu). Applied after the length guards so a spaced rendering can
    # never push a marginal entry over the cap retroactively.
    tr = normalize_glossary_spacing(tr, source=src)

    cleaned_aliases: list[str] = []
    if aliases:
        for a in aliases:
            cleaned_a = a.strip()
            if cleaned_a and cleaned_a.lower() != src.lower() and cleaned_a not in cleaned_aliases:
                cleaned_aliases.append(cleaned_a)

    return BibleEntry(source=src, translation=tr, aliases=cleaned_aliases, kind=kind)


def _concat_key(source: str) -> str:
    """Concept key that ignores spacing and hyphenation: ``state of the art``."""
    return re.sub(r"[\s\-_]+", "", source).casefold()


#: An acronym attestation inside a term, with optional material after it:
#: ``Foo (BAR)``, ``Foo (BAR) performance``, ``Foo (BAR)-based``.
#:
#: The trailing-only form was too narrow and let a real duplicate through:
#: measured on arXiv 2606.23344, the LLM extractor supplied
#: ``state-of-the-art (SOTA) performance -> 当前最优（SOTA）性能`` beside the mined
#: ``state-of-the-art -> 当前最先进的``. The acronym sat mid-string, the fold
#: declined, and the two entries fought exactly like the trailing case did
#: (b0002 went to 0.25 with ``repair_structural_failure`` after two rounds).
_ACRONYM_ATTESTATION_RE = re.compile(
    r"^(?P<base>.+?)\s*\(\s*(?P<acronym>[A-Z][A-Z0-9]{1,9})\s*\)(?P<tail>.*)$"
)


def _is_acronym_attested(source: str) -> bool:
    """Whether the source carries a parenthetical acronym the fold can use."""
    return _ACRONYM_ATTESTATION_RE.match(source.strip()) is not None


def concept_key(source: str) -> str:
    """The concept an entry names, with an author-attested acronym folded in.

    Terminology management keeps ONE record per concept: a preferred term plus
    its admitted synonyms and abbreviations. The pipeline had been treating
    "Full Expansion (ACRONYM)" as a *second* concept whenever it arrived from a
    different channel than the bare expansion, which is how one document ended
    up carrying both ``state-of-the-art -> 当前最先进的`` and
    ``state-of-the-art (SOTA) -> 当前最优（SOTA）``. Each entry then judged the
    other's correct rendering as drift, so no translation could satisfy both:

      * b0002 wrote 当前最优（SOTA）, which the first entry called drift (0.25)
      * b0006 wrote 当前最先进的, which the second entry called drift (0.35)

    Folding the acronym requires that it actually spells the base — the same
    word-initial rule the abbreviation miner uses — so ``Transformer (XL)`` and
    ``GPT (4)`` stay distinct concepts while ``state-of-the-art (SOTA)`` collapses
    onto ``state-of-the-art``.

    The acronym may sit mid-term with material after it
    (``state-of-the-art (SOTA) performance``): that is the same concept with a
    qualifier, and declining it left a live duplicate that cost two repair
    rounds and a human-queue verdict on a correct translation.
    """
    from ubt.core.memory.abbreviation_miner import _find_expansion_for

    text = source.strip()
    match = _ACRONYM_ATTESTATION_RE.match(text)
    if match:
        base = match.group("base").strip()
        acronym = match.group("acronym")
        # ``_find_expansion_for`` wants the words *preceding* the acronym and
        # returns the tightest suffix that spells it; requiring it to cover the
        # whole base keeps "Attention (MHA)" a separate concept.
        if base and _find_expansion_for(acronym, base + " ") is not None:
            return _concat_key(base)
    return _concat_key(text)


@dataclass(frozen=True)
class TerminologyConflict:
    """Renderings that disagreed for one concept, and which one won.

    A record is produced only where the pipeline had to *pick* — two entries for
    one concept with different translations and no way to tell which is right.
    :func:`merge_bible_entries` resolves it by precedence (a user glossary
    outranks a mined seed, an attested bare expansion outranks its acronym form),
    and the losing renderings are kept as admitted variants so text already
    written under them keeps passing. This type records the decision so the
    operator can review it, because the pipeline is choosing on *provenance*,
    not on meaning — and provenance can be wrong.

    ``resolved`` says whether the losers are still accepted somewhere:
    ``True`` means they became admitted variants (delivery is unaffected),
    ``False`` means the losing rendering is enforced against — a user glossary
    and a profile seed naming different Chinese for the same English term.
    Neither side is a translation defect; the run is self-contradictory, and the
    operator has to pick one.
    """

    concept: str
    preferred: str
    preferred_source: str
    superseded: tuple[tuple[str, str], ...]  # (rendering, its entry's source)
    resolved: bool

    def as_dict(self) -> dict[str, Any]:
        """JSON-ready shape for the quality report and ``ubt status``."""
        return {
            "concept": self.concept,
            "preferred": self.preferred,
            "preferred_source": self.preferred_source,
            "superseded": [{"rendering": r, "source": s} for r, s in self.superseded],
            "resolved": self.resolved,
        }


def detect_terminology_conflicts(entries: list[BibleEntry]) -> list[TerminologyConflict]:
    """Find concepts carrying more than one rendering BEFORE the merge resolves them.

    This has to run on the *unmerged* entries: :func:`merge_bible_entries` keeps
    one translation per concept by construction, so a scan of its output can
    never find a conflict. An earlier version of this check did exactly that and
    was unreachable code — it reported nothing on the very document it was
    written for (arXiv 2606.23344, which carried two renderings of
    ``state-of-the-art``).

    ``resolved`` mirrors what the merge will actually do, so the record predicts
    the delivery rather than merely noting that two entries existed:

    * entries that differ only by an attested acronym (``Foo`` vs ``Foo (BAR)``)
      fold, and the loser's rendering becomes an admitted variant → resolved;
    * the same source surface carrying two renderings cannot fold — one has to
      win and the other is enforced against → unresolved.
    """

    # Group by concept, keeping which source surface carried each rendering.
    by_concept: dict[str, dict[str, str]] = {}  # concept -> {rendering: source}
    surfaces: dict[str, set[str]] = {}  # concept -> distinct source surfaces
    for entry in entries:
        source = entry.source.strip()
        rendering = entry.translation.strip()
        if not source or not rendering:
            continue
        key = concept_key(source)
        by_concept.setdefault(key, {}).setdefault(rendering, source)
        surfaces.setdefault(key, set()).add(source.casefold())

    conflicts: list[TerminologyConflict] = []
    for key, renderings in sorted(by_concept.items()):
        if len(renderings) < 2:
            continue
        # The merge prefers the bare expansion: an acronym-attested surface is
        # an attestation of the concept, not the term to steer a draft towards.
        preferred_rendering, preferred_source = min(
            renderings.items(),
            key=lambda item: (
                1 if _is_acronym_attested(item[1]) else 0,
                len(item[1]),
            ),
        )
        superseded = tuple(
            (rendering, source)
            for rendering, source in renderings.items()
            if rendering != preferred_rendering
        )
        # A fold only happens when the surfaces differ; identical surfaces with
        # different renderings are a genuine contradiction.
        foldable = len(surfaces.get(key, ())) > 1
        conflicts.append(
            TerminologyConflict(
                concept=key,
                preferred=preferred_rendering,
                preferred_source=preferred_source,
                superseded=superseded,
                resolved=foldable,
            )
        )
    return conflicts


def merge_bible_entries(entries: list[BibleEntry]) -> list[BibleEntry]:
    """Merge entries by concept: preferred rendering wins, others become admitted.

    Two axes, in order:

    1. **Same source surface** (case-insensitive): first translation wins, aliases
       union, max frequency — the historical behaviour.
    2. **Same concept** (:func:`concept_key`): one entry survives and the other's
       rendering becomes an admitted variant on it. Before this, a document could
       carry two contradictory rules for one concept and the gate enforced both,
       which no translation can satisfy (see :func:`concept_key`).

    The surviving rendering is the one the *document* attests more: frequency
    first, then the shorter source surface (the bare expansion is the preferred
    term; ``Foo (BAR)`` is its attestation), then input order. A rendering that
    disagrees is never discarded — it is recorded as an admitted variant, so the
    text already written under it keeps passing and the enforcer leaves it alone.
    """
    merged: dict[str, dict[str, Any]] = {}
    for e in entries:
        key = e.source.lower()
        if key not in merged:
            merged[key] = {
                "source": e.source,
                "translation": e.translation,
                "aliases": set(e.aliases),
                "inflected_variants": set(e.inflected_variants),
                "kind": e.kind,
                "frequency": e.frequency,
                "order": len(merged),
            }
        else:
            merged[key]["aliases"] |= set(e.aliases)
            merged[key]["inflected_variants"] |= set(e.inflected_variants)
            if not merged[key]["translation"] and e.translation:
                merged[key]["translation"] = e.translation
            merged[key]["frequency"] = max(int(merged[key]["frequency"]), e.frequency)

    # Pass 2: fold entries that name the same concept through an attested acronym.
    by_concept: dict[str, list[dict[str, Any]]] = {}
    for record in merged.values():
        by_concept.setdefault(concept_key(str(record["source"])), []).append(record)

    for group in by_concept.values():
        if len(group) < 2:
            continue
        rendered = [r for r in group if str(r["translation"]).strip()]
        if len(rendered) < 2:
            continue
        # Preferred term = the bare expansion. ``Foo (BAR)`` is an *attestation*
        # of the concept, not a term to steer a draft towards — a rendering that
        # embeds "（SOTA）" in every sentence is the wrong thing to prefer across
        # the document, even when the miner happened to see it more often.
        # Frequency then breaks ties between equally bare surfaces.
        winner = min(
            rendered,
            key=lambda r: (
                1 if _is_acronym_attested(str(r["source"])) else 0,
                -int(r["frequency"]),
                len(str(r["source"])),
                int(r["order"]),
            ),
        )
        for loser in group:
            if loser is winner:
                continue
            # Every distinct rendering the loser carried is admitted on the
            # winner, and the loser's source surfaces become aliases so the
            # concept is still recognised in text that spells it the long way.
            losing_translation = str(loser["translation"]).strip()
            if losing_translation and losing_translation != winner["translation"]:
                if losing_translation not in winner["inflected_variants"]:
                    winner["inflected_variants"].add(losing_translation)
                logger.info(
                    "Terminology: %r has two renderings (%r preferred, %r admitted); "
                    "keeping both so neither reads as drift",
                    winner["source"],
                    winner["translation"],
                    losing_translation,
                )
            winner["aliases"] |= {str(loser["source"])} | set(loser["aliases"])
            # The loser's admitted renderings are target-side forms, not source
            # surfaces: merging them into ``aliases`` would teach the source
            # scan to look for Chinese in English text. They join the winner's
            # admitted set instead.
            for admitted in loser["inflected_variants"]:
                if admitted and admitted != winner["translation"]:
                    winner["inflected_variants"].add(admitted)
            winner["frequency"] = max(int(winner["frequency"]), int(loser["frequency"]))
            merged.pop(str(loser["source"]).lower(), None)

    # A concept's own source surface and preferred rendering are never aliases
    # or admitted variants of itself.
    result: list[BibleEntry] = []
    for key in sorted(merged):
        m = merged[key]
        own = {str(m["source"]).casefold()}
        aliases = sorted(a for a in m["aliases"] if a and a.casefold() not in own)
        admitted = sorted(
            v
            for v in m["inflected_variants"]
            if v and v != m["translation"] and v.casefold() not in {a.casefold() for a in aliases}
        )
        result.append(
            BibleEntry(
                source=m["source"],
                translation=m["translation"],
                aliases=aliases,
                inflected_variants=admitted,
                kind=m["kind"],
                frequency=int(m["frequency"]),
            )
        )
    return result
