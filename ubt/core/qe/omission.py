"""0-token omission gate — the blind spot the length-ratio floor can't see.

Three deterministic (0 LLM tokens) omission signals computed per block:

1. **Source/target sentence-count ratio** — deleting a whole sentence keeps the
   character length ratio inside the profile band (zh: 0.2–3.0), so the
   historical length gate is blind to it. Sentences are counted identically on
   both sides (terminator-punctuation splits, decimals excluded), and the gate
   only arms when the source has enough sentences to be statistically
   meaningful (>= ``min_source_sentences``).

2. **Proper-noun (identifier-shaped) recall** — camelCase runs, all-caps
   acronyms and digit-bearing terms must survive translation verbatim (same
   predicate the script-density gate uses for its carry-over whitelist).
   Ordinary lowercase words are translatable and never counted. Recall must
   *exceed* the floor when at least ``min_identifier_terms`` distinct terms
   exist. Metric-only below that population: a single term can be glossary-
   transliterated legitimately, so it cannot gate on its own.

3. **Character n-gram recall over verbatim residue (chrF-style)** — char
   2/3/4-grams extracted from the source's must-carry-over spans (numbers +
   identifier terms) and matched against the normalized target. Catches
   *partial* truncation ("1,234,567" -> "1,234") that exact-token presence
   checks miss.

Number recall is computed (CJK numerals, 万/亿 magnitudes, locale separators
all normalized via :mod:`ubt.core.validators.consistency`) and exposed as a
metric, but NOT gated here: ``NumericConsistencyValidator`` already hard-gates
every missing number earlier in the fast-pass chain, and duplicating that
verdict here would only double-report the same defect.

All floors use "must strictly exceed" semantics: ``ratio <= floor`` fails, so
2-of-4 sentences or 1-of-2 identifier terms is an omission, while legitimate
sentence merging (3-of-4) still passes.
"""

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ubt.core.qe.term_shape import count_sentences as count_sentences
from ubt.core.qe.term_shape import is_identifier_shaped
from ubt.core.validators.consistency import (
    canonicalize_numeric_token,
    denoted_numeric_values,
    magnitude_rewritten,
    normalize_for_numeric_matching,
    scale_equivalent_values,
)

# Sentence terminators and abbreviation masking live in ``term_shape`` next
# to the shared ``count_sentences`` (the MT admission gate imports the same
# one), as does the identifier-shape rule (shared with the fast-pass gate).
_is_identifier_shaped = is_identifier_shaped
# Latin verbatim runs (same shape as the fast-pass carry-over whitelist).
_LATIN_RUN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]*")
# Numeric tokens (same shape as NumericConsistencyValidator's scanner).
_NUM_TOKEN_RE = re.compile(r"\d[\d,.\-–—/]*\d|\d")
_RANGE_DELIMITERS_RE = re.compile(r"[-–—/]")
#: A range's delimiter is notation, not content: "10-50" and "10至50" (Chinese
#: "to") state the same interval. Fold both to "-" before n-gram scoring, or a
#: correct range render shares no delimiter gram with the source and deflates
#: verbatim recall below the omission floor.
_RANGE_FOLD_RE = re.compile(r"(?<=\d)\s*(?:[-–—~～]|至|到)\s*(?=\d)")


_STRICT_LB = r"(?<![A-Za-z0-9])"
_STRICT_RB = r"(?![A-Za-z0-9])"
# Split glued PDF-extraction runs on lower->Upper humps only ('whereQ0' ->
# ['where', 'Q0']); letter->digit is NOT split so subscript.AndDigit groups
# ('Q0', 'Vtm') survive as matchable units.
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")


def singular_variant(term: str) -> str | None:
    """English plural 's' stripped: Chinese never marks plurality, so a source
    'FinFETs' rendered as target 'FinFET' is correct, not an omission."""
    if len(term) > 3 and term.endswith("s") and not term.endswith(("ss", "us", "is")):
        return term[:-1]
    return None


def _digit_glued(term: str, source: str) -> tuple[bool, bool]:
    """Whether the source itself glues digits to the term (extraction dropped
    the subscript space: '+5CfinVtm'). The same gluing is then allowed
    target-side; letter-glued impostors ('XCfinVtm') still fail."""
    left = right = False
    for m in re.finditer(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", source):
        if m.start() > 0 and source[m.start() - 1].isdigit():
            left = True
        if m.end() < len(source) and source[m.end()].isdigit():
            right = True
    return left, right


def _merge_candidates(term: str, source: str) -> list[str]:
    """Source-attested space-merge forms: extraction splits subscripts
    ('V DS') that translation re-merges ('VDS'). Only the exact merge attested
    in source is accepted — a dropped term never matches."""
    cands: list[str] = []
    for m in re.finditer(rf"{_STRICT_LB}{re.escape(term)}{_STRICT_RB}", source):
        left = re.search(r"([A-Za-z0-9]+)\s+$", source[: m.start()])
        if left:
            cands.append(f"{left.group(1)}{term}")
        right = re.match(r"\s*([A-Za-z0-9]+)", source[m.end() :])
        if right:
            cands.append(f"{term}{right.group(1)}")
    return cands


def _is_hyphen_ghost(term: str, freq: "Counter[str]") -> bool:
    """Single unhyphenated run duplicating a repeated hyphenated run: a PDF
    line-break swallows the '-' in one 'Newton-Raphson' occurrence, yielding a
    'NewtonRaphson' token the (correctly transliterated) target can never match
    verbatim. Only fires when the hyphenated form is attested >=2x, so a
    genuinely glued nonce term is never excused."""
    if "-" in term or "_" in term:
        return False
    return any(
        run != term and count >= 2 and run.replace("-", "").replace("_", "") == term
        for run, count in freq.items()
    )


def identifier_terms(source_text: str) -> set[str]:
    """Source-verbatim identifier-shaped Latin runs the omission gate tracks.

    Single source of truth for "which source tokens must survive translation":
    the gate scores recall over exactly this set, and anything that has to
    behave like a competent model (the offline baseline fixtures) reuses it
    instead of re-deriving the rule.
    """
    runs = [
        match.group(0)
        for match in _LATIN_RUN_RE.finditer(source_text.strip())
        if len(match.group(0)) >= 2
    ]
    freq = Counter(runs)
    return {t for t in set(runs) if _is_identifier_shaped(t) and not _is_hyphen_ghost(t, freq)}


def _math_term_variants(term: str) -> list[str]:
    """Plausible LaTeX math expressions for an extracted term/variable."""
    vars: list[str] = []
    if "_" in term:
        prefix, sub = term.split("_", 1)
        vars.append(f"{prefix}_{{{sub}}}")
        vars.append(f"{prefix}_{{{sub.lower()}}}")
        vars.append(f"{prefix}_{sub.lower()}")
        vars.append(f"{prefix.lower()}_{{{sub.lower()}}}")
    m_dig = re.match(r"^([A-Za-z]+)(\d+)$", term)
    if m_dig:
        vars.append(f"{m_dig.group(1)}_{m_dig.group(2)}")
        vars.append(f"{m_dig.group(1)}_{{{m_dig.group(2)}}}")
    m_sub = re.match(r"^([A-Z])([a-z]+|[A-Z]+)$", term)
    if m_sub and len(m_sub.group(2)) >= 2:
        vars.append(f"{m_sub.group(1)}_{{{m_sub.group(2).lower()}}}")
        vars.append(f"{m_sub.group(1)}_{{{m_sub.group(2).upper()}}}")
        vars.append(f"{m_sub.group(1)}_{m_sub.group(2).lower()}")
    return vars


# Deliberate runtime allowlist (introduced in 60c7060 after corpus tuning):
# identifier terms a correct translation may legitimately render other than
# verbatim, mapped to the target surfaces that verify the translation.
# Extend only with corpus-attested cases — every entry here exempts an
# omission in the production gate.
_KNOWN_TRANSLATABLE_TERMS: dict[str, tuple[str, ...]] = {
    "microvm": ("微虚拟机", "虚拟机", "microvm"),
    "microvms": ("微虚拟机", "虚拟机", "microvms"),
    "ext4formatted": ("ext4", "格式化", "ext4 formatted"),
}


def _match_term_surfaces(term: str, target: str, source: str = "") -> list[str] | None:
    """Concrete target surfaces verifying an identifier term, or None.

    Acceptance ladder (first hit wins, all case-sensitive): exact whole-token
    -> plural/singular variant -> LaTeX math variants -> source-attested digit gluing ->
    source-attested space merge -> camel-decomposition parts (every
    identifier-shaped part must verify; when no part is shaped, at least one
    part must — so a fully dropped 'DataLoader' still fails while a
    transliterated 'whereQ0'/'withCfin' passes via its surviving parts).
    Returns matched surfaces so chrF residue scores what was actually
    verified, not the noisy source token.
    """
    esc = re.escape(term)
    if re.search(rf"{_STRICT_LB}{esc}{_STRICT_RB}", target):
        return [term]
    term_lower = term.lower()
    if term_lower in _KNOWN_TRANSLATABLE_TERMS:
        for cand in _KNOWN_TRANSLATABLE_TERMS[term_lower]:
            if cand in target or re.search(
                rf"{_STRICT_LB}{re.escape(cand)}{_STRICT_RB}", target, re.IGNORECASE
            ):
                return [cand]
    variants: list[str] = []
    sing = singular_variant(term)
    if sing is not None:
        variants.append(sing)
    elif len(term) > 2 and not term.endswith("s"):
        variants.append(term + "s")
    variants.extend(_math_term_variants(term))
    for var in variants:
        if re.search(rf"{_STRICT_LB}{re.escape(var)}{_STRICT_RB}", target):
            return [var]
    if not source:
        return None
    left_digit, right_digit = _digit_glued(term, source)
    if left_digit or right_digit:
        lb = r"(?<![A-Za-z])" if left_digit else _STRICT_LB
        rb = r"(?![A-Za-z])" if right_digit else _STRICT_RB
        if re.search(rf"{lb}{esc}{rb}", target):
            return [term]
    for cand in _merge_candidates(term, source):
        if re.search(rf"{_STRICT_LB}{re.escape(cand)}{_STRICT_RB}", target):
            return [cand]
    # A hyphenated qualifier ("QoS-aware", "eBPF-based") is an acronym head
    # plus a prose tail: the head is the part a translation must carry. Camel
    # splitting the whole term shears the acronym instead ("Qo" + "S-aware"),
    # so a target that kept "QoS" verbatim failed the gate and a correct
    # translation was quarantined. Hyphen components are therefore taken whole
    # (no camel split: "eBPF" must not shear into "e" + "BPF").
    if "-" in term:
        parts = [p for p in term.split("-") if len(p) >= 2 or p.isdigit()]
    else:
        parts = [p for p in _CAMEL_SPLIT_RE.split(term) if len(p) >= 2]
    if len(parts) > 1:
        needed = [p for p in parts if _is_identifier_shaped(p) or p.isdigit()]
        if needed:
            hits: list[str] = []
            for p in needed:
                hit = _match_term_surfaces(p, target, source)
                if hit is None:
                    return None
                hits.extend(hit)
            return hits
        for p in parts:
            hit = _match_term_surfaces(p, target, source)
            if hit is not None:
                return hit
    return None


def _char_ngrams(text: str, sizes: tuple[int, ...]) -> Counter[str]:
    grams: Counter[str] = Counter()
    for size in sizes:
        if len(text) < size:
            continue
        for i in range(len(text) - size + 1):
            grams[text[i : i + size]] += 1
    return grams


@dataclass(frozen=True, slots=True)
class OmissionMetrics:
    """Raw 0-token omission signals for one source/target block pair."""

    source_sentences: int
    target_sentences: int
    sentence_ratio: float
    proper_noun_recall: float
    number_recall: float
    verbatim_chrf_recall: float


def _is_table_content(text: str) -> bool:
    """True when text consists of markdown table lines or headers."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    table_lines = sum(1 for ln in lines if ln.count("|") >= 2)
    return table_lines >= max(1, len(lines) // 2)


@dataclass(frozen=True, slots=True)
class OmissionDecision:
    """Outcome of the omission gate; ``metrics`` always carries all signals."""

    passed: bool
    reason: str
    metrics: OmissionMetrics


class OmissionGate:
    """Deterministic omission detector: dropped sentences, missing identifier
    terms, truncated verbatim residue. Pure 0-token — safe to run on every
    block of every draft."""

    def __init__(
        self,
        target_lang: str = "zh",
        *,
        min_sentence_ratio: float = 0.5,
        min_source_sentences: int = 3,
        min_proper_noun_recall: float = 0.5,
        min_identifier_terms: int = 2,
        min_chrf_recall: float = 0.7,
        min_chrf_ngrams: int = 8,
        ngram_sizes: tuple[int, ...] = (2, 3, 4),
        min_length_ratio: float = 0.3,
        glossary: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.target_lang = target_lang
        # The sentence-merge exemption floor must come from the pair's
        # calibrated length band (0.2 for en->zh), not a global constant: a
        # hardcoded 0.3 quarantines faithful compressive translations the pair
        # policy explicitly accepts.
        self.min_length_ratio = min_length_ratio
        self.min_sentence_ratio = min_sentence_ratio
        self.min_source_sentences = min_source_sentences
        self.min_proper_noun_recall = min_proper_noun_recall
        self.min_identifier_terms = min_identifier_terms
        self.min_chrf_recall = min_chrf_recall
        self.min_chrf_ngrams = min_chrf_ngrams
        self.ngram_sizes = ngram_sizes
        self._glossary_surfaces: list[tuple[str, str]] = []
        self.set_glossary(glossary)

    def set_glossary(self, glossary: Sequence[Mapping[str, Any]] | None) -> None:
        """Bind the run's glossary so a term's canonical rendering counts as kept.

        The Bible may assign an identifier-shaped source term a translating
        rendering ("SDK" -> "软件开发工具包", "FnCall" -> "函数调用接口"). The
        glossary validator then *demands* that rendering while the verbatim
        identifier recall here *demands* the source surface — a block cannot
        satisfy both, and the correct translation was quarantined as an
        omission. A tracked term whose glossary rendering occurs in the target
        is therefore verified, not missing.
        """
        surfaces: list[tuple[str, str]] = []
        for entry in glossary or []:
            rendering = str(entry.get("translation") or entry.get("expected") or "").strip()
            if not rendering:
                continue
            sources = [
                str(entry.get("source") or "").strip(),
                *(str(a) for a in entry.get("aliases") or []),
            ]
            for surface in sources:
                if surface:
                    surfaces.append((surface.casefold(), rendering))
        self._glossary_surfaces = surfaces

    def _glossary_verifies(self, term: str, target: str) -> str | None:
        """The glossary rendering present in the target for ``term``, or None.

        Returned surface feeds the residue recall below, which scores what the
        target actually carries: the rendering, not the source term the
        translation was told to replace.
        """
        folded = term.casefold()
        for surface, rendering in self._glossary_surfaces:
            surface_matches = surface == folded or (
                len(folded) >= 3
                and re.search(rf"(?<![a-z0-9]){re.escape(folded)}(?![a-z0-9])", surface)
            )
            if surface_matches and rendering in target:
                return rendering
        return None

    def evaluate(
        self,
        source_text: str,
        target_text: str,
        *,
        is_table: bool | None = None,
        block_type: object = None,
    ) -> OmissionDecision:
        src = source_text.strip()
        tgt = target_text.strip()

        if is_table is None:
            bt = getattr(block_type, "value", block_type or "")
            if str(bt).lower() in ("table", "blocktype.table"):
                is_table = True
            else:
                is_table = _is_table_content(src) or _is_table_content(tgt)

        src_sentences = count_sentences(src)
        tgt_sentences = count_sentences(tgt)
        sentence_ratio = tgt_sentences / max(1, src_sentences)

        tracked_terms = identifier_terms(src)
        surfaces: list[str] = []
        missing_terms = []
        for t in sorted(tracked_terms):
            hit = _match_term_surfaces(t, tgt, src)
            if hit is None:
                rendering = self._glossary_verifies(t, tgt) if self._glossary_surfaces else None
                if rendering is not None:
                    surfaces.append(rendering)
                    continue
                missing_terms.append(t)
            else:
                surfaces.extend(hit)
        proper_noun_recall = 1.0 - len(missing_terms) / max(1, len(tracked_terms))

        number_recall = self._number_recall(src, tgt)
        chrf_recall = self._chrf_recall(src, tgt, surfaces)

        metrics = OmissionMetrics(
            source_sentences=src_sentences,
            target_sentences=tgt_sentences,
            sentence_ratio=sentence_ratio,
            proper_noun_recall=proper_noun_recall,
            number_recall=number_recall,
            verbatim_chrf_recall=chrf_recall,
        )

        if (
            not is_table
            and src_sentences >= self.min_source_sentences
            and (sentence_ratio <= self.min_sentence_ratio)
        ):
            # A translation that preserves 100% of proper nouns, 100% of numbers,
            # and verbatim n-grams with normal target length has not dropped sentences;
            # it merely merged clauses using Chinese comma splices or run-in headings.
            sentence_drop = not (
                proper_noun_recall >= 1.0
                and number_recall >= 1.0
                and chrf_recall >= self.min_chrf_recall
                and len(tgt) >= len(src) * self.min_length_ratio
            )
            if sentence_drop:
                return OmissionDecision(
                    passed=False,
                    reason=(
                        f"Omission suspected: target has {tgt_sentences} sentence(s) "
                        f"vs {src_sentences} in source"
                    ),
                    metrics=metrics,
                )

        if len(tracked_terms) >= self.min_identifier_terms and (
            proper_noun_recall <= self.min_proper_noun_recall
        ):
            return OmissionDecision(
                passed=False,
                reason=(
                    f"Omission suspected: identifier term(s) missing from target: {missing_terms}"
                ),
                metrics=metrics,
            )

        src_grams = self._source_residue_grams(src, surfaces)
        if sum(src_grams.values()) >= self.min_chrf_ngrams and chrf_recall <= self.min_chrf_recall:
            return OmissionDecision(
                passed=False,
                reason=(
                    "Omission suspected: verbatim residue n-gram recall "
                    f"{chrf_recall:.2f} below floor"
                ),
                metrics=metrics,
            )

        return OmissionDecision(passed=True, reason="No omission signals", metrics=metrics)

    def _number_recall(self, src: str, tgt: str) -> float:
        """Fraction of distinct source numbers preserved in the normalized target.

        Mirrors NumericConsistencyValidator's matching: CJK numerals,
        万/亿 magnitudes, full-width digits and locale separators normalize on
        both sides; compound ranges count as preserved when every sub-number
        survives. Scale-word equivalence is applied too, so a correct magnitude
        rendering ("1.5 million" -> "150万") is not read as a dropped number.
        """
        src_nums = {canonicalize_numeric_token(m) for m in _NUM_TOKEN_RE.findall(src)}
        src_nums.discard("")
        if not src_nums:
            return 1.0
        normalized_tgt = normalize_for_numeric_matching(tgt, lang=self.target_lang)
        tgt_values = denoted_numeric_values(tgt) | denoted_numeric_values(normalized_tgt)
        src_scales = scale_equivalent_values(src)
        preserved = 0
        for num in src_nums:
            if num in tgt_values or (src_scales.get(num, set()) & tgt_values):
                preserved += 1
                continue
            if _RANGE_DELIMITERS_RE.search(num):
                sub_parts = [
                    canonicalize_numeric_token(p)
                    for p in _RANGE_DELIMITERS_RE.split(num)
                    if p.strip()
                ]
                if len(sub_parts) > 1 and all(
                    sub in tgt_values or sub in normalized_tgt for sub in sub_parts
                ):
                    preserved += 1
        return preserved / len(src_nums)

    def _source_residue_grams(self, src: str, surfaces: list[str]) -> Counter[str]:
        """Char n-gram multiset of the source's must-carry-over residue.

        Grams are computed per span (each number token, each *verified* term
        surface) and unioned — concatenating spans would create
        cross-boundary grams that no target can ever match, systematically
        deflating recall. Scoring verified surfaces (not noisy source tokens)
        keeps plural/merge/decomposition excuses from deflating recall. A number
        carrying an adjacent scale word is scored at its *value* ("1.5 million"
        -> grams of "1500000"), so a correct "150万" rendering matches.
        """
        grams: Counter[str] = Counter()
        for m in _NUM_TOKEN_RE.findall(_RANGE_FOLD_RE.sub("-", magnitude_rewritten(src))):
            canon = canonicalize_numeric_token(m)
            if canon:
                grams += _char_ngrams(canon, self.ngram_sizes)
        for surface in surfaces:
            grams += _char_ngrams(surface, self.ngram_sizes)
        return grams

    def _chrf_recall(self, src: str, tgt: str, surfaces: list[str]) -> float:
        """Char n-gram recall of the verbatim residue against the whole target."""
        src_grams = self._source_residue_grams(src, surfaces)
        if not src_grams:
            return 1.0
        # Rewrite magnitude-scaled numbers on the target too, so both sides of a
        # correct magnitude rendering share a digit string. Rewrite BEFORE
        # normalizing: the zh normalizer would otherwise read '百万' as '100万'
        # first and lose the magnitude.
        target_view = normalize_for_numeric_matching(
            _RANGE_FOLD_RE.sub("-", magnitude_rewritten(tgt)), lang=self.target_lang
        )
        tgt_grams = _char_ngrams(target_view, self.ngram_sizes)
        overlap = sum(min(cnt, tgt_grams[gram]) for gram, cnt in src_grams.items())
        return overlap / sum(src_grams.values())
