"""0-Token deterministic fast-pass gating for translation blocks.

All language-dependent policy (length-ratio bands, target-language identity
thresholds) comes from the injected :class:`LanguageProfile`; the gating
mechanism itself is language-agnostic. The default profile reproduces the
en→zh behaviour.
"""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ubt.core.cjk_ranges import CJK_SCRIPT_CLASS, HAN_UNIFIED_CLASS
from ubt.core.cleaners.math_masker import extract_math_spans
from ubt.core.language_profile import (
    ZH,
    LanguagePairPolicy,
    LanguageProfile,
    get_pair_policy,
    is_rtl_lang,
)
from ubt.core.policy.layout_policy import PROSE_BLOCK_TYPES
from ubt.core.qe.added_content import AddedContentGate
from ubt.core.qe.defect_taxonomy import (
    ECHO_MARKER,
    NEAR_ECHO_MARKER,
    UNTRANSLATED_RESIDUE_MARKER,
)
from ubt.core.qe.omission import OmissionGate, OmissionMetrics, singular_variant
from ubt.core.qe.term_shape import is_identifier_shaped, is_verbatim_carryover
from ubt.core.validators.consistency import NumericConsistencyValidator
from ubt.core.validators.html_delta import HTMLDeltaValidator
from ubt.core.validators.math_guard import (
    novel_unsupported_latex_commands,
    target_missing_math_delimiters,
)

# Repair instruction appended to both echo rejections. The two phrasings are
# the same defect and are registered in ``defect_taxonomy`` so the score
# classifier and the defect tables can never match only one of them.
_ECHO_REPAIR_HINT = (
    ": the passage was not translated — render it in the target language,"
    " keeping identifiers, numerals and quoted strings verbatim"
)

_REPETITION_PATTERN = re.compile(r"(.{4,20}?)\1{3,}")  # Detect 4+ repetitions of a phrase
# Line-level loop detection: the flat regex above cannot match a loop whose
# separator is the trailing newline (the last repeat lacks it inside the
# group) and caps the unit at 20 chars, so the most common hallucination
# shape — the same sentence repeated line after line — escapes the flat check.
# Count contiguous runs of stripped identical lines instead. A line needs 3+
# letters/CJK (word content bar shared with the flat check, minus digits so
# sparse numeric tables like repeated "| 0 | 0 |" rows don't trip it).
_REPETITION_LINE_RUN = 4
#: A blank line resets the contiguous run (stanzas), but a hallucination loop
#: often separates its repeats with ``\n\n``; a run this long is a loop even
#: with blank gaps.
_REPETITION_BLANK_SEPARATED_RUN = 6
_REPETITION_LINE_MIN_LETTERS = 3
#: How many more identical lines the target may repeat than the source before a
#: line-level repetition counts as a runaway loop rather than a preserved
#: refrain. Mirrors ``_REPETITION_COUNT_SLACK`` for the flat detector.
_REPETITION_LINE_RUN_SLACK = 2
#: Any Unicode letter (``[^\W\d_]`` = word char minus digits/underscore). The
#: ASCII+CJK predecessor left Cyrillic/Greek/Arabic repetition loops invisible.
_LINE_LETTER_RE = re.compile(r"[^\W\d_]")
# URLs survive translation verbatim, so they are stripped before script-density
# measurement (same rationale as HTML tags). Bare domains without a scheme or
# www. prefix are left in place: indistinguishable from ordinary latin tokens.
_URL_RE = re.compile(r"https?://[^\s<>\"]+|www\.[^\s<>\"]+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
# Untranslatable residue stripped before script-density measurement.
# Citation markers, markdown table structure, numerals and isolated
# single letters (K/V/x formula debris) must survive translation verbatim
# (numerals have their own dedicated validator above), so counting them
# against target-script density systematically floors correctly-translated
# short blocks (citation lines, tables, bibliography) at 0.40.
_CITATION_SPAN_RE = re.compile(r"\[[^\[\]]*\]")
_TABLE_SEP_RE = re.compile(r"-{2,}")
_ISOLATED_LETTER_RE = re.compile(r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])")
_DIGIT_RE = re.compile(r"\d")
# Latin term runs (units, acronyms, identifiers) preserved verbatim.
# NOTE: no trailing '.' in the class — a sentence-final period is
# punctuation, not part of the identifier ("FlashAttention." must match
# the target's "FlashAttention").
_VERBATIM_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]*")
# Identifier-shape rule lives in ``term_shape`` (shared with the omission gate).
_is_identifier_shaped = is_identifier_shaped

_COMMON_SENTENCE_STARTERS = frozenset(
    {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "There",
        "Here",
        "When",
        "Where",
        "What",
        "Which",
        "Who",
        "Why",
        "How",
        "Then",
        "Thus",
        "Hence",
        "Also",
        "However",
        "Moreover",
        "Furthermore",
        "Therefore",
        "In",
        "On",
        "At",
        "By",
        "For",
        "With",
        "From",
        "Into",
        "About",
        "After",
        "Before",
        "During",
        "Under",
        "Above",
        "Below",
        "Between",
        "Through",
        "While",
        "Although",
        "Because",
        "Since",
        "Unless",
        "Until",
        "If",
        "As",
        "It",
        "Its",
        "We",
        "Our",
        "You",
        "Your",
        "They",
        "Their",
        "He",
        "His",
        "She",
        "Her",
        "One",
        "All",
        "Some",
        "Many",
        "Most",
        "Each",
        "Every",
        "Both",
        "Few",
        "Other",
        "Another",
        "First",
        "Second",
        "Third",
        "Finally",
        "Next",
        "Last",
        "Note",
        "See",
        "Figure",
        "Table",
        "Section",
        "Chapter",
    }
)


def _repeated_line_runs(text: str) -> list[tuple[str, int]]:
    """Contiguous runs of stripped-identical, letter-bearing lines (line, length).

    A run is a sequence of 4+ stripped-identical non-empty lines carrying 3+
    letters/CJK. Blank lines reset the run: legitimate stanzas / verses
    separated by whitespace stay exempt, and non-contiguous repeats (list
    labels, table headers recurring between sections) never trigger.
    """
    runs: list[tuple[str, int]] = []
    run_line: str | None = None
    run_len = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            run_line = None
            run_len = 0
            continue
        if line == run_line:
            run_len += 1
        else:
            run_line = line
            run_len = 1
        if run_len >= _REPETITION_LINE_RUN and len(_LINE_LETTER_RE.findall(line)) >= (
            _REPETITION_LINE_MIN_LETTERS
        ):
            runs.append((line, run_len))
    return runs


def _detect_line_repetition_loop(text: str) -> str | None:
    """Return the first newline-separated hallucination-loop line, or None."""
    runs = _repeated_line_runs(text)
    return runs[0][0] if runs else None


def _max_repeated_line_run(text: str) -> int:
    """Length of the longest repeated-line run in ``text`` (0 when none).

    The run is compared against the source's: a target run no longer than the
    source's own run is a faithfully preserved refrain, but a runaway loop that
    repeats far more lines than the source did is a hallucination even when the
    source repeats something unrelated.
    """
    return max((run_len for _, run_len in _repeated_line_runs(text)), default=0)


def _has_repeated_line_run(text: str) -> bool:
    """True when ``text`` itself carries a run of identical non-empty lines.

    The loop detector below is only a hallucination signal when the *source*
    does not repeat the same shape: a refrain, a repeated heading, or a table
    of identical rows is faithfully repeated in the target, and quarantining it
    sent a correct block to repair (a flat 0.2 flag score) for nothing.
    """
    return _detect_line_repetition_loop(text) is not None


def _repeated_line_runs_across_blanks(text: str) -> list[tuple[str, int]]:
    """Identical non-empty lines with blank-line gaps tolerated (line, length).

    The contiguous detector resets on a blank line to spare legitimate stanzas,
    but a model loop commonly separates its repeats with ``\n\n`` and slips
    past it. This counts non-empty identical lines ignoring blanks; the caller
    only treats a run long enough to be a runaway loop as one.
    """
    runs: list[tuple[str, int]] = []
    run_line: str | None = None
    run_len = 0
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line == run_line:
            run_len += 1
        else:
            run_line = line
            run_len = 1
        if run_len >= _REPETITION_BLANK_SEPARATED_RUN and len(_LINE_LETTER_RE.findall(line)) >= (
            _REPETITION_LINE_MIN_LETTERS
        ):
            runs.append((line, run_len))
    return runs


def _max_repeated_line_run_across_blanks(text: str) -> int:
    """Longest blank-separated repeated-line run in ``text`` (0 when none)."""
    return max((run_len for _, run_len in _repeated_line_runs_across_blanks(text)), default=0)


# Repetition-exemption tolerances. These are empirical calibration bands, not
# derived values: they bound how much noisier the target repetition may be
# than its source counterpart before the exemption stops applying.
_REPETITION_SRC_ATTEST_COUNT = 3  # source repeats attesting a legitimate shape
_REPETITION_COUNT_SLACK = 2  # target may exceed the attested source count by this
_TABLE_COLUMN_SLACK = 2  # tolerated source/target table column-count difference
_TABLE_CELL_MIN_REPEATS = 3  # cells repeated this often attest column repetition
_TABLE_CELL_REPEAT_SLACK = 1  # source cells may trail the target's top count by this
_REFRAIN_REPEAT_SLACK = 1  # tolerated refrain repeat-count difference
_REFRAIN_EDGE_SLACK_CHARS = 5  # refrain may fall short of the text edges by this many chars
_REFRAIN_COVERAGE_RATIO = 0.3  # refrain must cover this fraction of its text


def _strip_verbatim_term(residue: str, term: str) -> str:
    """Strip a source-verbatim term from the residue without eating into words.

    Plain ``str.replace`` lets the term OR consume the PRI|OR|ITY of an
    untranslated "PRIORITY", inflating script density and hiding a real echo.
    Letter-only terms get a letter-boundary guard (a CJK neighbour still
    matches, so CJK-adjacent carry-overs keep stripping); terms carrying
    digits or punctuation (GQA-8) keep plain replacement.
    """
    if term.isalpha():
        return re.sub(rf"(?<![A-Za-z]){re.escape(term)}(?![A-Za-z])", "", residue)
    return residue.replace(term, "")


def _is_prose_block(block_type: Any) -> bool:
    """Whether the block type gates the prose-only checks (empty/unknown -> prose).

    Shared by the decide and reassess paths so the two gates cannot drift.
    """
    value = getattr(block_type, "value", block_type or "")
    return not value or str(value).lower() in {str(t.value).lower() for t in PROSE_BLOCK_TYPES}


def _is_exempt_repetition(src_clean: str, tgt_clean: str, m: re.Match[str]) -> bool:
    """True when the repeated sequence or pattern is legitimately present in the source.

    Prevents false positives on repeated cell values (e.g. | GPT-5.6 Sol | GPT-5.6 Sol | ...),
    source-attested repeated phrases, refrains, or isomorphic table cell repetitions across columns,
    while correctly rejecting in-cell loops or runaway target hallucinations.
    """
    matched_text = m.group(0)
    repeated_unit = m.group(1)

    # 1. Exact match of the entire repeated sequence in source
    if matched_text in src_clean:
        return True

    # 2. Repeated unit present in source with equal or comparable count (not runaway loop)
    clean_unit = repeated_unit.strip(" |:\t-，。、")
    if len(clean_unit) >= 2:
        src_cnt = src_clean.count(clean_unit)
        tgt_cnt = tgt_clean.count(clean_unit)
        if src_cnt >= _REPETITION_SRC_ATTEST_COUNT and tgt_cnt <= src_cnt + _REPETITION_COUNT_SLACK:
            return True
        if src_cnt > 0 and src_cnt >= tgt_cnt:
            return True

    # 3. Table cell repetition / isomorphic repetition across table columns
    # Find the line in tgt_clean containing the matched repetition
    tgt_lines = tgt_clean.splitlines()
    tgt_line = ""
    char_cursor = 0
    tgt_line_idx = -1
    for idx, ln in enumerate(tgt_lines):
        line_len = len(ln)
        if char_cursor <= m.start() <= char_cursor + line_len:
            tgt_line = ln
            tgt_line_idx = idx
            break
        char_cursor += line_len + 1  # +1 for newline

    if tgt_line and tgt_line.count("|") >= 2:
        tgt_cells = [c.strip() for c in tgt_line.strip().strip("|").split("|")]
        # Only consider repetition across cells (not in-cell runaway text loop)
        tgt_counts = Counter(c for c in tgt_cells if c and not re.match(r"^[-:| ]+$", c))
        if tgt_counts:
            top_tgt_cell, top_tgt_cnt = tgt_counts.most_common(1)[0]
            # Ensure the matched repetition corresponds to this cell repetition across columns
            cell_matches_unit = (
                clean_unit == top_tgt_cell
                or (clean_unit != "" and clean_unit in top_tgt_cell)
                or (top_tgt_cell != "" and top_tgt_cell in repeated_unit)
                or "|" in repeated_unit
            )
            if top_tgt_cnt >= _TABLE_CELL_MIN_REPEATS and cell_matches_unit:
                src_lines = src_clean.splitlines()
                # Check corresponding or compatible source table row
                candidate_src_lines: list[str] = []
                if 0 <= tgt_line_idx < len(src_lines) and src_lines[tgt_line_idx].count("|") >= 2:
                    candidate_src_lines.append(src_lines[tgt_line_idx])
                candidate_src_lines.extend(
                    ln
                    for idx, ln in enumerate(src_lines)
                    if idx != tgt_line_idx and ln.count("|") >= 2
                )
                for s_ln in candidate_src_lines:
                    s_cells = [c.strip() for c in s_ln.strip().strip("|").split("|")]
                    if abs(len(s_cells) - len(tgt_cells)) <= _TABLE_COLUMN_SLACK:
                        s_counts = Counter(
                            c for c in s_cells if c and not re.match(r"^[-:| ]+$", c)
                        )
                        if any(
                            cnt >= _TABLE_CELL_MIN_REPEATS
                            or cnt >= top_tgt_cnt - _TABLE_CELL_REPEAT_SLACK
                            for cnt in s_counts.values()
                        ):
                            return True

    # 4. Source in-line pattern repetition (e.g. poetry, song refrain, in-line repeated phrases)
    # Only exempt when the target's repetition is structurally aligned to the source refrain
    # (matching repeat count, anchored at identical boundaries, and covering comparable span).
    for sm in _REPETITION_PATTERN.finditer(src_clean):
        s_unit = sm.group(1)
        if re.search(f"[\\w{HAN_UNIFIED_CLASS}]", s_unit) and not re.match(
            r"^\|?[\s\-:|]+\|?$", sm.group(0).strip()
        ):
            src_reps = len(sm.group(0)) // max(1, len(s_unit))
            tgt_reps = len(matched_text) // max(1, len(repeated_unit))
            if abs(src_reps - tgt_reps) <= _REFRAIN_REPEAT_SLACK:
                aligned_start = sm.start() == 0 and m.start() == 0
                aligned_end = (
                    abs(len(src_clean) - sm.end()) <= _REFRAIN_EDGE_SLACK_CHARS
                    and abs(len(tgt_clean) - m.end()) <= _REFRAIN_EDGE_SLACK_CHARS
                )
                covers_substantial = (
                    len(sm.group(0)) >= len(src_clean) * _REFRAIN_COVERAGE_RATIO
                    and len(matched_text) >= len(tgt_clean) * _REFRAIN_COVERAGE_RATIO
                )
                if (aligned_start or aligned_end) and covers_substantial:
                    return True

    return False


# A target that repeats the source is not a translation. The script-density
# gate below cannot see this when the pair shares a writing system
# (``language_profile.py`` zeroes ``min_target_ratio`` for Latin->Latin, and
# Japanese kanji clear the Chinese density bar), so an untouched paragraph would
# otherwise clear every gate and be released as MTQE_PASSED.
_ECHO_MIN_LEN = 16
_ECHO_FORMATTING_ONLY_RE = re.compile(r"^[`#*_\s0-9.|\-:=]+$")
_ECHO_WORD_RE = re.compile(r"[^\W\d_]{4}")
_ECHO_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]{2,}")
# Math, inline code, URLs, email addresses and quoted spans are contractually
# verbatim in any translation (the echo repair hint itself tells the model to
# keep "quoted strings verbatim"), so they carry no translation signal: mask
# them before counting retained words. Without the quoted-span alternative a
# faithful translation that preserves a long quotation keeps the quote's words,
# which then dominate the target's Latin tokens and read as a near-echo.
_ECHO_QUOTED_SPAN = (
    r"\"[^\"]*\""
    r"|“[^”]*”"
    r"|„[^“]*“"
    r"|‘[^’]*’"
    r"|«[^»]*»"
    r"|「[^」]*」"
    r"|『[^』]*』"
)
_ECHO_MASK_RE = re.compile(
    r"\$[^$]*\$|\\\[.+?\\\]|\\\(.+?\\\)|`[^`]*`"
    r"|https?://\S+"
    r"|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    rf"|{_ECHO_QUOTED_SPAN}",
    re.S,
)
_ECHO_QUOTED_SPAN_RE = re.compile(_ECHO_QUOTED_SPAN, re.S)
_NEAR_ECHO_MIN_TOKENS = 8
_NEAR_ECHO_RETENTION = 0.9
# A CJK target legitimately carries its Latin proper nouns across verbatim
# (SoL-Pi, EdgeBench, GPT-5, API); those are the only tokens the retention
# test below can see, so a fully translated Chinese paragraph would otherwise
# score a ~0.96 "kept nearly all source words" echo and get quarantined. The
# retention test is only meaningful for a Latin-script target: once the target
# itself is written in a CJK script, the passage is translated. Mirror the
# token floor.
_CJK_SCRIPT_RE = re.compile(f"[{CJK_SCRIPT_CLASS}]")
_NEAR_ECHO_CJK_EXEMPT = _NEAR_ECHO_MIN_TOKENS
# Same exemption for RTL targets: an Arabic/Hebrew passage that carries a few
# Latin proper nouns across verbatim is translated by definition, exactly as a
# CJK passage is.
_RTL_SCRIPT_RE = re.compile(
    r"[\u0590-\u05ff\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff"
    r"\ufb1d-\ufb4f\ufb50-\ufdff\ufe70-\ufeff]"
)


def is_verbatim_echo(source_text: str, target_text: str) -> bool:
    """True when the target repeats the source with no translation attempt.

    Short and wordless blocks are exempt: page numbers, labels (``Fig. 3``),
    DOIs and formatting-only runs legitimately survive the trip unchanged.
    """
    src = source_text.strip()
    if not src or src != target_text.strip():
        return False
    if len(src) < _ECHO_MIN_LEN or _ECHO_FORMATTING_ONLY_RE.match(src):
        return False
    return bool(_ECHO_WORD_RE.search(src))


#: Rehearsal marker emitted by ``--dry-run``'s echo provider. A rehearsal is
#: meant to prove the plumbing reaches a rendered artifact, so its synthetic
#: "translation" (source echoed back behind this marker) must not be quarantined
#: as an untranslated echo. Kept as one constant the dry-run provider imports.
REHEARSAL_MARKER = "[模拟翻译]"

#: A leading rehearsal prefix the PDF renderer strips before it places text, so
#: an artifact audit must compare against the *rendered* text, not the marker a
#: dry run prepends. One definition, shared by the renderer
#: (``render``) and the artifact check (``pipeline.artifact``) --
#: when only one of them knew about the strip, every rehearsal delivery reported a
#: false "text realization missing" (worst on short headings, which cannot absorb
#: the marker's four CJK tokens inside the overlap tolerance).
_REHEARSAL_PREFIX_RE = re.compile(r"^\[(?:模拟翻译|Mock\s*Translation)\]\s*", re.IGNORECASE)


def strip_rehearsal_marker(text: str) -> str:
    """Drop a leading rehearsal prefix; the renderer never places it."""
    return _REHEARSAL_PREFIX_RE.sub("", text)


def _cjk_bigram_retention(source_text: str, target_text: str) -> float | None:
    """Fraction of the target's CJK character bigrams the source also carries.

    Order-sensitive, so shared technical vocabulary ("半導体素子…") does not read
    as an echo while a re-punctuated copy of the source does. ``None`` when the
    target has too few bigrams to judge.
    """

    def _bigrams(text: str) -> list[str]:
        chars = _CJK_SCRIPT_RE.findall(text)
        return [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]

    target_grams = _bigrams(target_text)
    if len(target_grams) < _NEAR_ECHO_MIN_TOKENS:
        return None
    source_counts = Counter(_bigrams(source_text))
    kept = sum(
        min(count, source_counts.get(gram, 0)) for gram, count in Counter(target_grams).items()
    )
    return kept / len(target_grams)


def is_near_verbatim_echo(
    source_text: str,
    target_text: str,
    *,
    target_is_cjk: bool = True,
    source_is_cjk: bool = False,
    target_is_rtl: bool = False,
) -> bool:
    """True when the target keeps almost all source words without translating.

    The exact-equality echo gate is blind to a one-character difference, and
    Latin->Latin pairs zero the script-density bar (language_profile), so
    without this check a lightly retouched source ships as "Flawless"
    MTQE_PASSED prose — a full book of English echoed into German could pass
    the entire gate chain. CJK-heavy text has too few Latin tokens to reach
    the token floor here and stays covered by the exact check.

    That last assumption does not hold for a technical paper: a correct Chinese
    translation still carries eight-plus Latin proper nouns, all of which
    appear in the source, so the Latin-only retention ratio reads ~0.96 and a
    translated paragraph would be quarantined as untranslated. Guard it
    directly: a target written in a CJK script is translated by definition, so
    the Latin-token retention test never runs on it.

    That exemption is only valid when the source is a different script. For a
    same-script CJK pair (zh->ja) a CJK target is *not* translated by definition,
    so the retention test falls back to CJK character bigrams (order-sensitive:
    shared terminology survives, a copied sentence does not).
    """
    if target_is_cjk and source_is_cjk:
        # Same-script pair (zh<->ja/ko): "a CJK target is translated by
        # definition" does not hold, and the Latin-only retention test below is
        # blind to CJK tokens entirely. Judge over character bigrams instead.
        retention = _cjk_bigram_retention(source_text, target_text)
        return retention is not None and retention >= _NEAR_ECHO_RETENTION
    if target_is_cjk and len(_CJK_SCRIPT_RE.findall(target_text)) >= _NEAR_ECHO_CJK_EXEMPT:
        return False
    if target_is_rtl and len(_RTL_SCRIPT_RE.findall(target_text)) >= _NEAR_ECHO_CJK_EXEMPT:
        return False
    if REHEARSAL_MARKER in target_text:
        # A ``--dry-run`` echo is an intentional rehearsal artifact, not an
        # untranslated book: its QE is mocked precisely so the plumbing can be
        # exercised end to end.
        return False
    src_masked = _ECHO_MASK_RE.sub(" ", source_text)
    tgt_masked = _ECHO_MASK_RE.sub(" ", target_text)
    # The quoted-span mask exists so a faithfully preserved quotation does not
    # dominate a real translation's tokens. But when the target is *entirely* a
    # quoted span, masking leaves nothing to judge and the retention test would
    # abstain — so ``"<source sentence>"`` sailed through as Flawless while the
    # bare copy was caught. A target that is only a quoted run (no translation
    # words outside it) is the evasion, not a preserved citation: judge the
    # unwrapped content. The exact-echo check cannot catch it — the added
    # quoting characters make the strings unequal. A quote embedded in real
    # prose keeps the mask: there the outside words are the translation.
    if not _ECHO_TOKEN_RE.findall(tgt_masked):
        # Each quoted-span alternative is one opening delimiter + content + one
        # closing delimiter, so the slice peels exactly the pair.
        unwrapped = _ECHO_QUOTED_SPAN_RE.sub(lambda m: f" {m.group(0)[1:-1]} ", target_text)
        if unwrapped != target_text:
            tgt_masked = _ECHO_MASK_RE.sub(" ", unwrapped)
    src_words = _ECHO_TOKEN_RE.findall(src_masked.lower())
    tgt_words = _ECHO_TOKEN_RE.findall(tgt_masked.lower())
    if len(tgt_words) < _NEAR_ECHO_MIN_TOKENS or len(src_words) < _NEAR_ECHO_MIN_TOKENS:
        return False
    src_set = set(src_words)
    retained = sum(1 for w in tgt_words if w in src_set)
    return retained / len(tgt_words) >= _NEAR_ECHO_RETENTION


def grid_columns(line: str) -> int | None:
    """Column count of a markdown grid row, or None when the line is not one.

    A grid row must *open* with a pipe. Inline pipes in prose (``A | B | C``)
    are not cell separators, so a sentence carrying two of them must not read
    as a one-row table. Once the row is anchored, leading/trailing pipes are
    ignored: ``| a | b |`` and ``| a | b`` are the same two columns, and a
    model that normalizes that style must not be rejected for it.
    """
    stripped = line.strip()
    if not stripped.startswith("|") or stripped.count("|") < 2:
        return None
    stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith(r"\|"):
        stripped = stripped[:-1]
    cells = re.split(r"(?<!\\)\|", stripped)
    return len(cells) if cells else None


def markdown_grid_shape(text: str) -> list[int]:
    """Per-row column counts of a markdown grid, or [] when it is not one.

    A real grid is uniform: the extractor pads every row to the same column
    count (``docling_blocks.table_to_markdown``), so a pipe-bearing prose block
    or a ragged run of pipe lines is not a table and returns ``[]``. Otherwise
    prose that merely contains pipes reads as a grid and a faithful translation
    without pipes is rejected as a dropped table.
    """
    shape = [cols for cols in (grid_columns(ln) for ln in text.splitlines()) if cols]
    if len(shape) < 2 or len(set(shape)) != 1:
        return []
    return shape


@dataclass(frozen=True, slots=True)
class FastPassDecision:
    """Decision outcome of 0-token fast pass filter."""

    passed: bool
    reason: str
    target_ratio: float
    length_ratio: float
    # Omission-gate signals (None when the gate did not run, i.e.
    # early structural failures before the omission check).
    omission: OmissionMetrics | None = None
    #: Non-blocking findings on an otherwise-passing draft (e.g. a numeric order
    #: inversion the translation is entitled to make). Surfaced as review flags,
    #: never as a rejection — the same contract as the HTML delta validator's
    #: formatting warnings.
    advisories: tuple[str, ...] = ()


def _normalize_math_body(span: str) -> str:
    s = span.strip()
    if s.startswith("$$") and s.endswith("$$") and len(s) >= 4:
        return s[2:-2].strip()
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        return s[1:-1].strip()
    if s.startswith(r"\(") and s.endswith(r"\)") and len(s) >= 4:
        return s[2:-2].strip()
    if s.startswith(r"\[") and s.endswith(r"\]") and len(s) >= 4:
        return s[2:-2].strip()
    return s


def _math_spans_equivalent(src_spans: list[str], tgt_spans: list[str]) -> bool:
    if sorted(src_spans) == sorted(tgt_spans):
        return True
    if len(src_spans) != len(tgt_spans):
        return False
    from ubt.core.cleaners.math_text import skeleton_holds

    unmatched = [_normalize_math_body(t) for t in tgt_spans]
    for s_span in src_spans:
        s_norm = _normalize_math_body(s_span)
        match_idx = None
        for idx, t_norm in enumerate(unmatched):
            if s_norm == t_norm or skeleton_holds(s_norm, t_norm):
                match_idx = idx
                break
        if match_idx is not None:
            unmatched.pop(match_idx)
        else:
            return False
    return True


class FastPassFilter:
    """Deterministic 0-token structural gate.

    Passing means the draft has no empty text, prompt leaks, repetition loops,
    HTML delta failures, or language-profile anomalies. It is not a neural
    quality score and must not be recorded as one.
    """

    def __init__(
        self,
        profile: LanguageProfile | LanguagePairPolicy | None = None,
        *,
        policy: LanguagePairPolicy | None = None,
        source_lang: str | None = None,
        target_lang: str | None = None,
        min_target_ratio: float | None = None,
        min_length_ratio: float | None = None,
        max_length_ratio: float | None = None,
    ) -> None:
        if policy is not None:
            active_profile: LanguageProfile | LanguagePairPolicy = policy
        elif source_lang is not None and target_lang is not None:
            active_profile = get_pair_policy(source_lang, target_lang)
        elif profile is not None:
            active_profile = profile
        elif target_lang is not None:
            active_profile = get_pair_policy(source_lang or "en", target_lang)
        else:
            # Fallback when no language is specified. Production paths pass
            # source_lang/target_lang and resolve pair policies via get_pair_policy.
            active_profile = ZH
        self.profile = active_profile
        self.min_target_ratio = (
            self.profile.min_target_ratio if min_target_ratio is None else min_target_ratio
        )
        self.min_length_ratio = (
            self.profile.min_length_ratio if min_length_ratio is None else min_length_ratio
        )
        self.max_length_ratio = (
            self.profile.max_length_ratio if max_length_ratio is None else max_length_ratio
        )
        # Same-script pairs disable the script-density gate; the pair policy
        # carries a lexical (function-word) identity gate in its place.
        self.same_script_gate = getattr(self.profile, "same_script_gate", None)
        self.html_validator = HTMLDeltaValidator()
        self.numeric_validator = NumericConsistencyValidator(profile=self.profile)
        self.omission_gate = OmissionGate(
            target_lang=self.profile.code, min_length_ratio=self.min_length_ratio
        )
        self.added_content_gate = AddedContentGate()

    def set_glossary(self, glossary: Any) -> None:
        """Bind the run's glossary into the omission gate (see :meth:`OmissionGate.set_glossary`).

        Called by the draft stage once the translation Bible exists; every
        later FastPass verdict in the run (repair re-checks, triage) then
        accepts a term's canonical glossary rendering instead of demanding the
        source surface verbatim.
        """
        self.omission_gate.set_glossary(glossary)

    @property
    def policy(self) -> LanguageProfile | LanguagePairPolicy:
        """Active language profile or dynamic language pair policy."""
        return self.profile

    def validate_structural_invariants(
        self,
        source_text: str,
        target_text: str,
        *,
        block_type: Any = None,
        skip_translate: bool = False,
    ) -> FastPassDecision:
        """Validate 0-token deterministic structural integrity: no empty text, no prompt leaks,

        no verbatim source echo, no repetitive loop hallucinations, and lossless HTML tag delta.
        """
        src_clean = source_text.strip()
        tgt_clean = target_text.strip()
        # Non-prose block types (display FORMULA, CODE, tables) and
        # skip-translate blocks keep a source echo by contract, so the prose
        # only gates below share this one computation. (Without the shared
        # computation, formula blocks misroute to repair — which rightly
        # ignores skip_translate — and are left stale-FAILED.)
        _is_prose = _is_prose_block(block_type)

        if not tgt_clean:
            return FastPassDecision(
                passed=False,
                reason="Empty target text",
                target_ratio=0.0,
                length_ratio=0.0,
            )

        # 1. Prompt artifact residue check (highest priority failure)
        if "<issues>" in tgt_clean or "<translation>" in tgt_clean:
            return FastPassDecision(
                passed=False,
                reason="Prompt template XML artifacts leaked into target text",
                target_ratio=0.0,
                length_ratio=0.0,
            )

        # 1b. Verbatim echo. The reason doubles as the repair instruction.
        if not skip_translate and _is_prose and is_verbatim_echo(src_clean, tgt_clean):
            return FastPassDecision(
                passed=False,
                reason=f"{ECHO_MARKER}{_ECHO_REPAIR_HINT}",
                target_ratio=0.0,
                length_ratio=1.0,
            )
        if (
            not skip_translate
            and _is_prose
            and src_clean != tgt_clean
            and is_near_verbatim_echo(
                src_clean,
                tgt_clean,
                target_is_cjk=self.profile.code in ("zh", "ja", "ko"),
                source_is_cjk=getattr(self.profile, "source_code", "") in ("zh", "ja", "ko"),
                target_is_rtl=is_rtl_lang(self.profile.code),
            )
        ):
            return FastPassDecision(
                passed=False,
                reason=f"{NEAR_ECHO_MARKER}{_ECHO_REPAIR_HINT}",
                target_ratio=0.0,
                length_ratio=1.0,
            )

        # 2. Hallucination repetition check (ignoring formatting and markdown table dividers).
        # A source refrain/table may legitimately repeat, but the exemption is
        # per-match and source-aligned (``_is_exempt_repetition``): a blanket
        # "the source repeats anything, so skip every repetition check" let a
        # runaway target loop pass whenever the source happened to carry an
        # unrelated repeated line.
        for m in _REPETITION_PATTERN.finditer(tgt_clean):
            repeated_unit = m.group(1)
            # Ignore pure punctuation/formatting repeats (e.g. markdown table lines |---|---|, dashes, dots)
            if not re.search(f"[\\w{HAN_UNIFIED_CLASS}]", repeated_unit):
                continue
            matched_text = m.group(0)
            if re.match(r"^\|?[\s\-:|]+\|?$", matched_text.strip()):
                continue
            if _is_exempt_repetition(src_clean, tgt_clean, m):
                continue
            return FastPassDecision(
                passed=False,
                reason="Repetitive loop hallucination detected",
                target_ratio=0.0,
                length_ratio=0.0,
            )

        # 2b. Line-level loop detection: catch the newline-separated shape the
        # flat regex above cannot express (repeated sentence lines, the most
        # common hallucination-loop form). A target run is exempt only when the
        # source carries a run at least as long (within slack): a faithful
        # refrain survives, a runaway loop that repeats far more lines than the
        # source did is still caught.
        tgt_line_run = _max_repeated_line_run(tgt_clean)
        if tgt_line_run > _max_repeated_line_run(src_clean) + _REPETITION_LINE_RUN_SLACK:
            return FastPassDecision(
                passed=False,
                reason="Repetitive loop hallucination detected (repeated lines)",
                target_ratio=0.0,
                length_ratio=0.0,
            )

        # 2b-bis. Blank-separated loop: the shape above resets on a blank line
        # (stanzas), so a model repeating a sentence as ``line\n\nline\n\n``
        # slips past it. A run this long is a runaway loop, not a refrain.
        tgt_blank_run = _max_repeated_line_run_across_blanks(tgt_clean)
        if tgt_blank_run >= _REPETITION_BLANK_SEPARATED_RUN and tgt_blank_run > (
            _max_repeated_line_run_across_blanks(src_clean) + _REPETITION_LINE_RUN_SLACK
        ):
            return FastPassDecision(
                passed=False,
                reason="Repetitive loop hallucination detected (blank-separated lines)",
                target_ratio=0.0,
                length_ratio=0.0,
            )

        # 2c. Markdown grid shape. Tables reach us as pipe grids and the emitter
        # maps each row to a Typst table row, so a target that changes the column
        # count ships a mangled table (or fails the compile pages away from the
        # cause). Shape only -- cell text is free to change, and the numeric
        # validator still guards the cells' figures.
        src_grid = markdown_grid_shape(src_clean)
        if src_grid:
            tgt_grid = markdown_grid_shape(tgt_clean)
            if not tgt_grid:
                # A table in the source with no table in the target means the
                # model flattened it to prose — a structural loss, not a shape
                # difference. Checking only when ``tgt_grid`` is non-empty
                # would skip the whole test in exactly this case and fail open.
                return FastPassDecision(
                    passed=False,
                    reason=(
                        "Table dropped: the source held a "
                        f"{src_grid}-row/column grid but the target has none — "
                        "translate the table in place instead of flattening it to prose"
                    ),
                    target_ratio=0.0,
                    length_ratio=0.0,
                )
            if src_grid != tgt_grid:
                return FastPassDecision(
                    passed=False,
                    reason=(
                        "Table grid mismatch: the source rows hold "
                        f"{src_grid} columns, the target holds {tgt_grid} — keep "
                        "the row and column count identical and translate only "
                        "the text inside each cell"
                    ),
                    target_ratio=0.0,
                    length_ratio=0.0,
                )

        # 3. Deterministic HTML delta check
        html_res = self.html_validator.validate(src_clean, tgt_clean)
        if not html_res.is_valid:
            return FastPassDecision(
                passed=False,
                reason=f"HTML delta failure: {html_res.message}",
                target_ratio=0.0,
                length_ratio=0.0,
            )

        # 3b. Math-span preservation check (Gate 4, the XLIFF
        # placeholder-QA paradigm): inline math masked before drafting must be
        # restored verbatim after unmasking. A dropped ⟦MATH_MASK_*⟧ token
        # surfaces here as a span multiset mismatch and routes to repair, where
        # the block is re-masked from source (self-healing closed loop).
        # Currency ($5) is never a span, so prices don't trip this gate.
        src_spans = extract_math_spans(src_clean)
        tgt_spans = extract_math_spans(tgt_clean)
        if src_spans and not _math_spans_equivalent(src_spans, tgt_spans):
            return FastPassDecision(
                passed=False,
                reason=(
                    "Math span mismatch: source carries "
                    f"{len(src_spans)} math span(s), target carries "
                    f"{len(tgt_spans)} ({', '.join(sorted(src_spans)[:3])})"
                ),
                target_ratio=0.0,
                length_ratio=0.0,
            )

        # 3c. Undelimited-math redelimit (docling-flattened counterpart of
        # 3b): the source carries math signals but the target has no $...$
        # span, so the overlay would print formulas as flat prose
        # (``Fth,SI`` instead of $F_{th,SI}$). Routes to repair, whose
        # prompt surfaces this reason verbatim as the fix instruction.
        # Exemptions come from ``_is_prose``/``skip_translate`` above.
        if (
            not skip_translate
            and _is_prose
            and target_missing_math_delimiters(src_clean, tgt_clean)
        ):
            return FastPassDecision(
                passed=False,
                reason=(
                    "Undelimited math: source carries math symbols but the "
                    "target has no $...$ span — re-emit each inline physical "
                    "variable, Greek letter and sub/superscript in standard "
                    "LaTeX inside single dollar signs ($...$, e.g. "
                    "$F_{th,SI}$, $V_{tm}$), translating the prose around them"
                ),
                target_ratio=0.0,
                length_ratio=0.0,
            )

        # 3d. Hallucinated-LaTeX redelimit: the
        # target invents LaTeX control sequences absent from the source
        # AND outside the renderer-supported set — e.g. a repair round
        # "reconstructing" a plain "(2D)" label into $2\mathrm{D}$, which
        # the overlay probe cannot compile and prints literally. Renderable
        # newcomers (\text, Greek, \frac...) pass: they compile downstream,
        # so flagging them would fight the math-reconstruction rule. The
        # reason doubles as the repair instruction (surfaced verbatim).
        if not skip_translate and _is_prose:
            hallucinated = novel_unsupported_latex_commands(src_clean, tgt_clean)
            if hallucinated:
                shown = ", ".join(f"\\{cmd}" for cmd in hallucinated[:3])
                return FastPassDecision(
                    passed=False,
                    reason=(
                        "Hallucinated LaTeX: target introduces LaTeX control "
                        f"sequence(s) absent from the source ({shown}) that "
                        "the renderer cannot compile — delete the fabricated "
                        "commands and reproduce the plain labels verbatim "
                        "(e.g. 2D, not $2\\mathrm{D}$); wrap in $...$ only "
                        "notation already mathematical in the source"
                    ),
                    target_ratio=0.0,
                    length_ratio=0.0,
                )

        # 3e. Added-content probe: every other gate here checks
        # what the translation LOST. Nothing checked what it GAINED, so a
        # drafter that translated the injected read-only neighbour excerpt and
        # prepended it to its own output would pass every gate — the extra text
        # is fluent and on-topic, and because gate 5b only reads the low side
        # of the length band, the invented sentences actively mask the real
        # sentence they displace. The probe is 0-token and language-agnostic;
        # the reason doubles as the repair instruction (surfaced verbatim).
        added = self.added_content_gate.evaluate(src_clean, tgt_clean)
        if not added.passed:
            return FastPassDecision(
                passed=False,
                reason=added.reason,
                target_ratio=0.0,
                length_ratio=0.0,
            )

        return FastPassDecision(
            passed=True,
            reason="Structural invariants passed",
            target_ratio=1.0,
            length_ratio=1.0,
        )

    def evaluate(
        self,
        source_text: str,
        target_text: str,
        *,
        block_type: Any = None,
        skip_translate: bool = False,
    ) -> FastPassDecision:
        """Evaluate whether a translated block qualifies for direct fast-pass release."""
        if REHEARSAL_MARKER in target_text:
            # ``--dry-run`` echo: the rehearsal exists to prove the plumbing
            # reaches a rendered artifact, so its synthetic (untranslated) text
            # must not be held to the deterministic QE gates — otherwise the
            # zero-token end-to-end check quarantines every real-length paragraph
            # and exercises none of the repair/render path it is meant to test.
            return FastPassDecision(
                passed=True,
                reason="rehearsal echo (dry-run): deterministic QE bypassed",
                target_ratio=1.0,
                length_ratio=1.0,
            )
        structural = self.validate_structural_invariants(
            source_text, target_text, block_type=block_type, skip_translate=skip_translate
        )
        if not structural.passed:
            return structural

        src_clean = source_text.strip()
        tgt_clean = target_text.strip()
        _is_prose = _is_prose_block(block_type)

        # 4. Standalone number preservation check
        num_res = self.numeric_validator.validate(src_clean, tgt_clean)
        if not num_res.is_valid:
            return FastPassDecision(
                passed=False,
                reason=f"Numeric fidelity failure: {num_res.message}",
                target_ratio=0.0,
                length_ratio=0.0,
            )
        # A numeric order inversion or elided restatement is not a failure — a
        # translation may legitimately restate quantities in another order — but
        # it is carried forward as a review flag rather than dropped, so the
        # audit's "silent pass" becomes an operator-visible one.
        num_advisories = tuple(num_res.details.get("numeric_advisories", ()))

        # 5. Length ratio check (thresholds are per-language-pair policy)
        length_ratio = len(tgt_clean) / max(1, len(src_clean))
        # Short headings/phrases (e.g. "Introduction" -> "引言", ratio 0.17) in contractive
        # language pairs (e.g. Latin to CJK with min_length_ratio < 0.5) legitimately
        # contract to 2+ characters without terminal sentence punctuation.
        is_short_heading_contraction = (
            self.min_length_ratio < 0.5
            and len(src_clean) <= 25
            and len(tgt_clean) >= 2
            and not any(src_clean.endswith(p) for p in (".", "?", "!", ";", "。", "？", "！", "；"))
        )
        if length_ratio < self.min_length_ratio and not is_short_heading_contraction:
            return FastPassDecision(
                passed=False,
                reason=f"Target text suspiciously truncated (ratio={length_ratio:.2f})",
                target_ratio=0.0,
                length_ratio=length_ratio,
            )
        if length_ratio > self.max_length_ratio:
            return FastPassDecision(
                passed=False,
                reason=f"Target text suspiciously inflated (ratio={length_ratio:.2f})",
                target_ratio=0.0,
                length_ratio=length_ratio,
            )

        omission = self.omission_gate.evaluate(
            src_clean,
            tgt_clean,
            is_table=True if ((not _is_prose) or bool(markdown_grid_shape(src_clean))) else None,
            block_type=block_type,
        )
        if not omission.passed:
            return FastPassDecision(
                passed=False,
                reason=omission.reason,
                target_ratio=0.0,
                length_ratio=length_ratio,
                omission=omission.metrics,
            )

        # 6. Target-language identity check on stripped prose text (profile-driven).
        # Measure density over the *translatable residue* — citation
        # markers, table pipes/separators, numerals and isolated letters are
        # required to survive verbatim and must not dilute the ratio.
        prose_only = re.sub(r"<[^>]+>", "", tgt_clean)
        prose_only = _URL_RE.sub("", prose_only)
        prose_only = _EMAIL_RE.sub("", prose_only)
        residue = _CITATION_SPAN_RE.sub("", prose_only)
        residue = residue.replace("|", "")
        residue = _TABLE_SEP_RE.sub("", residue)
        residue = _DIGIT_RE.sub("", residue)
        residue = _ISOLATED_LETTER_RE.sub("", residue)
        preserved = {
            m.group(0) for m in _VERBATIM_TERM_RE.finditer(src_clean) if len(m.group(0)) >= 2
        }
        # One rule for prose and tables alike: strip only identifier-shaped
        # carry-overs (PagedAttention, MHA, GQA-8). Ordinary Latin words left
        # untranslated stay in the residue, so an untouched table fails here
        # for the same reason untouched prose does. Stripping every
        # source-verbatim term instead (not just identifier-shaped ones) would
        # make an entire untranslated table read as empty residue (ratio 1.0).
        for term in preserved:
            if is_verbatim_carryover(term) or (
                len(term) >= 3
                and term[0].isupper()
                and term[1:].islower()
                and term not in _COMMON_SENTENCE_STARTERS
            ):
                residue = _strip_verbatim_term(residue, term)
                # Plural-tolerant strip: source 'FinFETs' rendered as target
                # 'FinFET' is correct — leaving it in the residue would
                # dilute script density (same rule as the omission gate).
                sing = singular_variant(term)
                if sing is not None:
                    residue = _strip_verbatim_term(residue, sing)
        # Language identity lives in letters: punctuation carries no script
        # signal and only dilutes the ratio (decimal points, parens, slashes
        # dominate tables and formulas).
        residue = re.sub(r"[\W\d_]+", "", residue)
        if residue:
            target_ratio = self.profile.target_script_ratio(residue)
            if target_ratio < self.min_target_ratio:
                return FastPassDecision(
                    passed=False,
                    reason=(
                        f"Insufficient {self.profile.code} script density "
                        f"(ratio={target_ratio:.2f})"
                    ),
                    target_ratio=target_ratio,
                    length_ratio=length_ratio,
                )
        else:
            target_ratio = 1.0

        # 6b. Same-script lexical identity. The script classes above cannot
        # separate a Latin->Latin pair, so the pair policy's function-word gate
        # is the only identity signal. ``None`` means the pair is cross-script
        # (or a language without a fingerprint), so this check is inert.
        if self.same_script_gate is not None and not self.same_script_gate(prose_only):
            source_name = getattr(self.profile, "source_name", "the source language")
            target_name = getattr(self.profile, "target_name", "the target language")
            return FastPassDecision(
                passed=False,
                reason=(
                    f"{UNTRANSLATED_RESIDUE_MARKER}: the target's common words are still "
                    f"{source_name}, not {target_name} — render the passage in "
                    f"{target_name}, keeping identifiers, numerals and quoted strings verbatim"
                ),
                target_ratio=0.0,
                length_ratio=length_ratio,
            )

        return FastPassDecision(
            passed=True,
            reason="Flawless: passed all 0-token deterministic invariants",
            target_ratio=target_ratio,
            length_ratio=length_ratio,
            omission=omission.metrics,
            advisories=num_advisories,
        )
