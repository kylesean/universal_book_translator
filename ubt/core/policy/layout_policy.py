"""Unified policy: every tunable in one view.

Single source of truth for all numeric thresholds, regexes, and categorical
sets across overlay layout, translate pre-filter, continuation, and page
profiling. Import direction is always safe: adapters import this core module,
never the reverse.

Each knob carries a calibration status (see :data:`CALIBRATION`):

- ``PROVEN`` — validated on 2+ documents, synthetic property tests, or PDF
  spec mechanics (not tunable judgment);
- ``SINGLE_DOC`` — calibrated on chapter-1 only; needs the golden corpus
  before anyone "tunes" it for another book;
- ``HYPOTHESIS`` — principled default with no empirical backing yet.

Discipline (see docs/design/knob-calibration-protocol.md): new knobs land HERE with
status + rationale, never inline in a module. Statuses are promoted by evidence,
never by feel.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ubt.core.ir.models import BlockType, FlowID
from ubt.model.ast import RegionKind


class Calibration(StrEnum):
    """Evidence level behind a knob's current value."""

    PROVEN = "proven"
    SINGLE_DOC = "single_doc"
    HYPOTHESIS = "hypothesis"


@dataclass(frozen=True)
class KnobMeta:
    """Why a knob holds its value."""

    status: Calibration
    rationale: str


# ---------------------------------------------------------------------------
# Text normalization (spec/mechanics — shared by align + verdict word scan)
# ---------------------------------------------------------------------------
WS_RE = re.compile(r"\s+")
FOLD_MAP = str.maketrans(
    {
        "\u2019": "'",
        "\u2018": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2010": "-",
        "\u2011": "-",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\u00a0": " ",
    }
)
# Control chars from custom PDF glyphs (pdfium emits \x02 where the font
# substitutes a ligature) — invisible, never present in parsed block text.
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
ASCII_WORD_RE = re.compile(r"[A-Za-z]+")
UNICODE_WORD_RE = re.compile(r"\w+", re.UNICODE)

# ---------------------------------------------------------------------------
# Block vocabulary (semantic categories — stable by design, not tuned)
# ---------------------------------------------------------------------------
PROSE_BLOCK_TYPES = frozenset({BlockType.HEADING, BlockType.NARRATIVE, BlockType.LIST_ITEM})
# Blocks whose text must ship byte-identical. TABLE is deliberately NOT here:
# a table's cells are the book's content, and both the parser
# (``docling_parser`` marks tables ``skip=False``) and the QE layer (which
# fails an untranslated table in FastPass) already assumed they get
# translated -- keeping them here meant no
# run ever sent one to the model, and the verdict's keep stamped them
# MTQE_PASSED/1.0 so a whole book of source-language tables reported a perfect
# pass rate. The rigid engine still leaves them alone (it paints
# ``PROSE_BLOCK_TYPES`` only), which is that engine's stated contract.
NON_TEXT_BLOCK_TYPES = frozenset({BlockType.FORMULA, BlockType.CODE, BlockType.IMAGE})
NON_PROSE_FLOWS = frozenset({FlowID.FOOTNOTE, FlowID.CAPTION, FlowID.TABLE_GRID})
CAPTION_RE = re.compile(r"^(FIG\.|Fig\.|Figure|Table|TAB\.|图|表)\s*[A-Za-z0-9]", re.IGNORECASE)
FOOTER_PATTERNS = re.compile(r"copyright|©|doi\.org|rights reserved|Elsevier", re.IGNORECASE)
# Chrome bands: short text hugging the page edges (points + chars).
HEADER_BAND_PT = 45.0
FOOTER_BAND_PT = 55.0
BAND_TEXT_MAX_LEN = 30

# ---------------------------------------------------------------------------
# Pairing punctuation (docling_adapter prose tails)
# ---------------------------------------------------------------------------
PAIR_TERMINAL_PUNCT = frozenset({".", "。", "!", "！", "?", "？", ":", "：", ";", "；"})

# ---------------------------------------------------------------------------
# Text fitting (text_fit.py)
# ---------------------------------------------------------------------------
# Em advance widths (fractions of em) — conservative upper bounds, shaped
# Typst output only ever shrinks below them.
EM_CJK_ADV = 1.0
EM_ASCII_ADV = 0.55
EM_PUNCT_ADV = 0.62
EM_SPACE_ADV = 0.32


def _env_float(name: str, default: float) -> float:
    """Calibration override: env wins, otherwise the frozen default.

    Lets the knob matrix (docs/design/knob-calibration-protocol.md) sweep values
    without touching source defaults — the default stays the object under
    test. Invalid values fall back silently so a typo never breaks a run.
    """
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


CJK_PUNCT_CHARS = frozenset("，。、；：？！「」『』（）【】《》〈〉…—·")
# Measure words / units that license a preceding Chinese numeral run as a real
# number rather than as part of a word. Consumed by
# ubt/core/validators/consistency.py, which must normalise '为两百元' -> '为200元'
# before the digit-presence check, yet must NOT rewrite word suffixes
# ('统一' -> '统1').
#
# Deliberately conservative: only unambiguous units and counters. Excluded on
# purpose because they form very common idioms with a numeral prefix that are
# NOT numbers — 分 (十分 = "very"), 度 (一度 = "formerly"), 面 (一面…一面…),
# 步 (进一步 = "further"), 部 (一部分 = "a portion"). Adding any of them here
# would reintroduce exactly the fabricated-digit failure this set exists to
# prevent. 万/亿 are included because they are magnitude suffixes of a numeral
# run ('二十万' -> '20万', which the digit check then expands).
CN_MEASURE_WORDS = (
    "个",
    "位",
    "名",
    "只",
    "张",
    "本",
    "册",
    "篇",
    "页",
    "章",
    "节",
    "条",
    "款",
    "项",
    "件",
    "台",
    "套",
    "组",
    "对",
    "双",
    "批",
    "份",
    "种",
    "类",
    "级",
    "层",
    "段",
    "次",
    "轮",
    "期",
    "年",
    "月",
    "日",
    "天",
    "周",
    "秒",
    "米",
    "千米",
    "公里",
    "厘米",
    "毫米",
    "克",
    "千克",
    "公斤",
    "吨",
    "升",
    "毫升",
    "元",
    "角",
    "美元",
    "欧元",
    "万元",
    "亿元",
    "倍",
    "人",
    "家",
    "间",
    "座",
    "辆",
    "艘",
    "架",
    "支",
    "根",
    "颗",
    "粒",
    "束",
    "堆",
    "群",
    "户",
    "所",
    "场",
    "集",
    "首",
    "幅",
    "门",
    "岁",
    "余",
    "多",
    "万",
    "亿",
)
# Kinsoku sets (JIS X 4051 class): a line must not END with an opener nor
# START with a closer. "…—·" are neutral (no adjustment either way).
CJK_OPEN_PUNCT = frozenset("「『（《〈【")
CJK_CLOSE_PUNCT = frozenset("，。、；：？！」』）》〉】")
# Punctuation squeeze: Typst compresses CJK punctuation (edge compression,
# adjacent-punct shrinking), so advance-sum overstates shaped width. The
# discount scales with punctuation density (pure ASCII: zero — no behaviour
# change where no slack provably exists) and caps out at 6%.
PUNCT_SQUEEZE_CAP = _env_float("UBT_PUNCT_SQUEEZE_CAP", 0.06)
PUNCT_SQUEEZE_PER_PUNCT = _env_float("UBT_PUNCT_SQUEEZE_PER_PUNCT", 0.02)


# Fitter defaults: footnote-grade floor (CJK legible), precision matching the
# Typst emit rounding, underfill double-gate (relative + absolute so display
# type is exempt).
# FIT_MIN_FONT_PT is the ``FlowFitter`` DEFAULT and nothing more: the rigid
# typesetter always passes its own floor (RIGID_MIN_FONT_PT) into the fitter.
FIT_MIN_FONT_PT = _env_float("UBT_FIT_MIN_FONT_PT", 6.5)
FIT_PRECISION_PT = 0.1
# Rigid-typesetter floor — the value actually in force for every rigid
# page render. Was an inline literal in typesetter.py, which is exactly what
# This module's registry discipline forbids. Deliberately distinct
# from FIT_MIN_FONT_PT: rigid paint-back shrinks text inside the source
# page's own zones, where the 6.5pt footnote floor reads as damage.
# 7.0: lowered from 7.5 after the coverage sweep showed 7.5 sits
# past the spill cliff on two corpora (single- and two-column). Blocks that need
# <7.5pt now render instead of failing closed to source-visible; see the
# CALIBRATION rationale and scripts/rigid_coverage_sweep.py.
RIGID_MIN_FONT_PT = _env_float("UBT_RIGID_MIN_FONT_PT", 7.0)
# Region-tiered floors: a region the source itself sets in small type may go
# below the body floor, because the block would otherwise fail closed and leave
# the source visible — a still-legible target beats an untranslated line. Both
# mirror FIT_MIN_FONT_PT (6.5, validated on chapter-1 tiny print).
RIGID_CAPTION_MIN_FONT_PT = _env_float("UBT_RIGID_CAPTION_MIN_FONT_PT", 6.5)
RIGID_FOOTNOTE_MIN_FONT_PT = _env_float("UBT_RIGID_FOOTNOTE_MIN_FONT_PT", 6.5)
_RIGID_REGION_FLOORS: dict[str, float] = {
    RegionKind.CAPTION.value: RIGID_CAPTION_MIN_FONT_PT,
    RegionKind.FOOTNOTE.value: RIGID_FOOTNOTE_MIN_FONT_PT,
}


def rigid_min_font_pt_for(region: object, *, default: float = RIGID_MIN_FONT_PT) -> float:
    """Font floor for a block's region.

    Body/title/header regions use the body floor; captions and footnotes — the
    source's own small-print regions — may shrink one step further. A block that
    cannot fit its region floor is dropped to source-visible, so a lower floor
    there only ever adds delivered text.
    """
    return _RIGID_REGION_FLOORS.get(str(getattr(region, "value", region)), default)


# Complex-page nets: row-fragment glue. Thresholds measured on chapter-1
# (good: coverage 0.77-1.0) vs book2 p31 (bad: pairs down to 0.2); gap cap
# sits between word gaps (<10pt) and column gutters.
ROW_MERGE_GAP_PT = _env_float("UBT_ROW_MERGE_GAP_PT", 24.0)
ROW_MERGE_Y_TOL = _env_float("UBT_ROW_MERGE_Y_TOL", 0.5)

# ---------------------------------------------------------------------------
# PDF operator inventory (page profiler / engine selector)
# ---------------------------------------------------------------------------
PDF_PATH_OPS = frozenset({"m", "l", "c", "v", "y", "h", "re"})
PDF_TEXT_OPS = frozenset({"Tj", "TJ", "'", '"'})

# ---------------------------------------------------------------------------
# Background sampling (raster heuristics, rigid scan covers)
# ---------------------------------------------------------------------------
BG_SAMPLE_SCALE = 2.0
# Only all-channels-dark pixels are glyph strokes; saturated backgrounds
# (banner blue has R=0) must survive.
STROKE_BLACK_CUTOFF = 64
# Width-metrics font only: fontTools' TTCollection reads it, so every entry must
# be a .ttc (a bare .ttf/.otf makes TTCollection raise, not degrade). The face is
# never painted — the page renders in whatever the resolved stack says — so any
# CJK collection gives honest advance widths. Noto first keeps Linux unchanged.
CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",  # Microsoft YaHei, ships with Windows
    "/System/Library/Fonts/PingFang.ttc",  # macOS
    "/System/Library/Fonts/Supplemental/Songti.ttc",
)

# ---------------------------------------------------------------------------
# Engine probe (engine_selector.py — whole-book fast path routing)
# ---------------------------------------------------------------------------
PROBE_MIN_CHARS = 32
PROBE_COLUMN_EDGE_RATIO = 0.45
PROBE_COLUMN_SHARE = 0.25
# Justified single-column lines fragment every few glyphs; the gap between
# fragments inside one row stays under ~1% of the page width, while a real
# column gutter exceeds 2%. Between them the split is unambiguous on both
# synthetic corpus papers (mono share 0.000, duo ~0.49).
PROBE_COLUMN_GAP_RATIO = 0.03
PROBE_MIN_ROWS = 6
PROBE_SAMPLE_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
PROBE_FORMULA_SHARE = 0.10
PROSE_SINGLE_LETTERS = frozenset({"a", "A", "I"})

# ---------------------------------------------------------------------------
# Page profiler (page_profiler.py — per-page fact/decision layer)
# ---------------------------------------------------------------------------
PROFILE_VECTOR_PATH_OPS = 200
PROFILE_VECTOR_TEXT_CHARS = 100
# Poster / resume page-kind decision cutoffs (HYPOTHESIS).
# Poster: image-led page with almost no text and few fonts (title + bleed art).
POSTER_MAX_CHARS = 80
POSTER_MAX_FONTS = 2
# Resume: dense single-column prose page (no images, no multi-column tell,
# low formula debris). Document-level single-page guard lives in the router.
RESUME_MIN_CHARS = 2000

# ---------------------------------------------------------------------------
# QA depth policy (visual_gate.py — short-doc parity)
# ---------------------------------------------------------------------------
# <= this many pages: full-page inspection (skill zero-defect-gate parity).
QA_FULL_GATE_MAX_PAGES = 20
# 20..BAND: stratified sampling (~20%, floor 6, cap QA_TIERED_BAND_SAMPLE).
QA_TIERED_BAND_PAGES = 100
QA_TIERED_BAND_SAMPLE = 10
# > BAND: fixed anchor set (flagged + cover/TOC/tail). VLM cap is separate
# (config.visual_max_vlm_pages) and never scales with book size.
QA_LONG_BOOK_SAMPLE = 10

# ---------------------------------------------------------------------------
# Short-chain router (router_mode.py — unified entry, adaptive execution)
# ---------------------------------------------------------------------------
# <= this many born-digital pages: whole-chapter short chain (rewrite +
# reflow + full visual gate). Above: 6-stage long chain. Distinct from
# QA_FULL_GATE_MAX_PAGES (visual-inspection scope), this gates execution.
# Default 30 covers a 26-page textbook chapter (chapter-3实战).
# Display/registry default only: ``router_mode.decide`` resolves the live
# ``UBTConfig.short_max_pages`` at call time, because this constant is frozen at
# import and its ``_env_int`` parse accepts 0/garbage the config field rejects.
SHORT_CHAIN_MAX_PAGES = _env_int("UBT_SHORT_MAX_PAGES", 30)

# ---------------------------------------------------------------------------
# Fast lane (pipeline.py — short-doc parity)
# ---------------------------------------------------------------------------
# Minimum extractable PDF text (chars) for the fast lane: below this the doc
# is scan-like and must take the full path (OCR/VLM), never the shortcut.
FAST_LANE_MIN_TEXT_CHARS = 200

# ---------------------------------------------------------------------------
# Length conservation (export.py — short-doc parity)
# ---------------------------------------------------------------------------
# Overflow skips on length-policy pages (resume_dense/poster_fixed) flip to
# NEEDS_HUMAN so they reach the PE queue instead of shipping source-visible
# with only an error flag. Kill-switch: False restores flag-only behaviour.
LENGTH_OVERFLOW_TO_HUMAN = True
# Page kinds under the length-conservation policy (fit-to-page, no repaginate).
LENGTH_POLICY_PAGE_KINDS = frozenset({"resume_dense", "poster_fixed"})

# ---------------------------------------------------------------------------
# Verdict pre-filter (verdict.py — zero-LLM-cost keep rules)
# ---------------------------------------------------------------------------
HEXDUMP_MIN_TOKENS = 32
HEXDUMP_MIN_SHARE = 0.55
INDEX_MIN_LINES = 12
INDEX_MIN_SINGLE_SHARE = 0.85
SHORT_LABEL_MAX_LEN = 24
BYTE_WORD_RE = re.compile(r"^[0-9a-fA-F]{2,}$")
SINGLE_TOKEN_LINE_RE = re.compile(r"^[^\W_][\w\-/]*$")
SHORT_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _\-/.]*$")
URL_DOI_RE = re.compile(r"^(https?://\S+|doi:\s*\S+)$", re.IGNORECASE)
ISBN_DIGITS_RE = re.compile(r"^(97[89])?\d{9}[0-9xX]$", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Continuation scoring (pair join/break weights)
# ---------------------------------------------------------------------------
JOIN_LOWER_START = 3
JOIN_HYPHEN_TAIL = 4
JOIN_CROSS_NO_TERMINAL = 2
BREAK_TERMINAL = 4
BREAK_CHAPTER_NUMBER = 6
BREAK_UPPER_START = 1
JOIN_THRESHOLD = 4
BREAK_THRESHOLD = 4
CONJUNCTIONS = frozenset(
    {
        "and",
        "or",
        "but",
        "nor",
        "for",
        "yet",
        "so",
        "以及",
        "并且",
        "或者",
        "但是",
        "然而",
    }
)
CONT_SENTENCE_END_RE = re.compile(r"[.!?。！？…]$")
CONT_CHAPTER_NUMBER_RE = re.compile(r"^\d+(\.\d+)+\s")
CONT_UPPER_START_RE = re.compile(r"^[A-Z]")
CONT_LABEL_BARRIER_RE = re.compile(
    r"^((table|figure|fig\.|scheme|equation|eq\.|section|chapter)\b\s*\d)",
    re.IGNORECASE,
)
CONT_LIST_MARKER_RE = re.compile(r"^(\d+[.)、]|[•·\-–—*]|\([a-z0-9]+\))\s+")
# Leading section number on headings ("1.1 ", "2.2.1 ") — universal numbering
# the overlay carries over when the translation dropped it.
HEADING_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+")
# Target already numbered: same dotted pattern, or CJK section word up front.
# ("21世纪" does NOT count — bare digits without dot/space are content.)
TARGET_NUMBERED_RE = re.compile(r"^\d+(?:\.\d+)*\s|^第?[一二三四五六七八九十]+[章节条篇部分]")


def formula_debris_share(text: str) -> float:
    """Share of isolated single-letter tokens among ASCII word tokens.

    Shattered equations shed isolated capitals (``K``, ``V``, ``x``);
    ordinary prose stays below ~0.08, formula pages score 0.16+.
    """
    tokens = ASCII_WORD_RE.findall(text)
    if not tokens:
        return 0.0
    singles = sum(1 for t in tokens if len(t) == 1 and t not in PROSE_SINGLE_LETTERS)
    return singles / len(tokens)


# ---------------------------------------------------------------------------
# VLM page-transcription circuit breaker (adapters/pdf/docling_parser)
# ---------------------------------------------------------------------------
# Stop after this many pages with (nearly) no success: a dead endpoint or a
# revoked key would otherwise burn its provider timeout once per page across a
# whole scanned book.
VLM_CIRCUIT_MIN_TRIES = 3
VLM_CIRCUIT_FAIL_PCT = 50

# ---------------------------------------------------------------------------
# On-disk permissions for artifacts that carry manuscript text (core/fs_perms)
# ---------------------------------------------------------------------------
BOOK_TEXT_DIR_MODE = 0o700
BOOK_TEXT_FILE_MODE = 0o600

# ---------------------------------------------------------------------------
# Concurrent-run ceiling for the MCP entry layer (mcp/server)
# ---------------------------------------------------------------------------
# The REST server already caps live runs at this many; MCP had no ceiling, so a
# single agent turn could spawn unbounded full-pipeline tasks. One value for
# both doors, so they cannot drift apart.
MCP_MAX_RUNNING_JOBS = 8


# ---------------------------------------------------------------------------
# Calibration registry (promoted by evidence)
# ---------------------------------------------------------------------------
P = Calibration.PROVEN
S = Calibration.SINGLE_DOC
H = Calibration.HYPOTHESIS

CALIBRATION: dict[str, KnobMeta] = {
    # Normalization: mechanical, property-tested.
    "WS_RE": KnobMeta(P, "whitespace folding is Unicode-mechanics"),
    "FOLD_MAP": KnobMeta(P, "quote/dash/ligature folds pinned by unit tests"),
    "CONTROL_RE": KnobMeta(P, "pdfium control-char artifacts observed + stripped"),
    "ASCII_WORD_RE": KnobMeta(P, "probe tokenizer, behaviour-tested"),
    "UNICODE_WORD_RE": KnobMeta(P, "verdict placeholder scan, behaviour-tested"),
    # Vocabulary: design decisions, stable.
    "PROSE_BLOCK_TYPES": KnobMeta(P, "flow partition contract, tested"),
    "NON_TEXT_BLOCK_TYPES": KnobMeta(
        P, "verbatim-ship contract (tables excluded: their cells are content)"
    ),
    "NON_PROSE_FLOWS": KnobMeta(P, "verdict keep-origin contract, tested"),
    "CAPTION_RE": KnobMeta(S, "caption prefixes observed in chapter-1 + KV handbook"),
    "FOOTER_PATTERNS": KnobMeta(S, "Elsevier chrome observed in chapter-1"),
    "HEADER_BAND_PT": KnobMeta(S, "45pt band clears chapter-1 running heads"),
    "FOOTER_BAND_PT": KnobMeta(S, "55pt band clears chapter-1 footers"),
    "BAND_TEXT_MAX_LEN": KnobMeta(S, "30-char cut separates chrome from body"),
    "PAIR_TERMINAL_PUNCT": KnobMeta(P, "terminal set, behaviour-tested"),
    # Fitting: mixed evidence.
    "EM_CJK_ADV": KnobMeta(P, "CJK advance == 1em by font design"),
    "EM_ASCII_ADV": KnobMeta(H, "0.55 literature average, uncalibrated vs fontTools"),
    "EM_PUNCT_ADV": KnobMeta(H, "0.62 literature average, uncalibrated vs fontTools"),
    "EM_SPACE_ADV": KnobMeta(H, "0.32 literature average, uncalibrated vs fontTools"),
    "CJK_PUNCT_CHARS": KnobMeta(P, "punctuation inventory, behaviour-tested"),
    "CJK_OPEN_PUNCT": KnobMeta(P, "kinsoku opener set, behaviour-tested"),
    "CJK_CLOSE_PUNCT": KnobMeta(P, "kinsoku closer set, behaviour-tested"),
    "PUNCT_SQUEEZE_CAP": KnobMeta(
        H,
        "6% cap held: matrix 9-cell sweep shows zero delta (corpus too clean "
        "to test, 3 overflows/206 blocks); mechanism unit-tested, needs stress corpus. "
        "Knob sensitivity: broad range (0.04 to 6e4) all green — only cap<=0 flips a test, "
        "and that is the disabled branch (text_fit.py:114), not the 6%",
    ),
    "PUNCT_SQUEEZE_PER_PUNCT": KnobMeta(
        S,
        "2%/punct calibrated: matrix per01==base on books 2+3, rate insensitive. "
        "Sensitivity analysis: ±1.5x invisible, and the first value "
        "anything reacts to is 2e4 (3 tests incl. the rigid overlay golden)",
    ),
    "FIT_MIN_FONT_PT": KnobMeta(
        S,
        "6.5 footnote floor validated on chapter-1 tiny print — FlowFitter default "
        "only; the rigid engine runs RIGID_MIN_FONT_PT (7.0) instead",
    ),
    "FIT_PRECISION_PT": KnobMeta(P, "0.1 matches Typst :.1f emit rounding by construction"),
    "RIGID_MIN_FONT_PT": KnobMeta(
        S,
        "7.0 rigid-engine floor, lowered from 7.5. Two-corpus matrix "
        "sweep by scripts/rigid_coverage_sweep.py: synthetic-mono (251 blocks / "
        "13 pp) latin expansion held ~74% to ratio 1.4 at 7.0 but fell 74%->26% "
        "above 1.15 at 7.5; synthetic-duo pages 1-6 (201 blocks, two-column) held "
        "56% vs 40% at latin ratio 1.2 and 41% vs 32% at cjk ratio 0.6. So 7.5 sat "
        "past the knee; this floor, not margin_reclaim_pt, is the rigid capacity "
        "knob. Blocks needing <7.5pt now render instead of failing closed to "
        "source-visible. Real-corpus (ForMaT) confirmation and the human visual "
        "pass remain open; override per-run with UBT_RIGID_MIN_FONT_PT",
    ),
    "RIGID_CAPTION_MIN_FONT_PT": KnobMeta(
        S,
        "6.5 caption-region floor (region tiering): captions are set in small "
        "type in the source, and one that cannot fit the 7.0 body floor fails "
        "closed and leaves the source visible. Aligned with FIT_MIN_FONT_PT; the "
        "synthetic corpora carry no captions (docling-only classification), so "
        "the value itself stays single-doc",
    ),
    "RIGID_FOOTNOTE_MIN_FONT_PT": KnobMeta(
        S,
        "6.5 footnote-region floor (region tiering), reusing the FIT_MIN_FONT_PT "
        "value validated on chapter-1 tiny print; the body floor would fail "
        "footnotes closed and leave the source visible",
    ),
    "ROW_MERGE_GAP_PT": KnobMeta(
        H,
        "24pt row-glue cap, set between word gaps and gutters; sensitivity sweep: 16–36pt "
        "all pass, only 2.4e7 (fold anything) breaks the side-column zone test — the "
        "placement claim is unmeasured",
    ),
    "ROW_MERGE_Y_TOL": KnobMeta(
        H,
        "0.5 min y-overlap fraction for row bands, two-pass core/small; structural heuristic "
        "for line clustering",
    ),
    # PDF operator inventory (profiler / engine selector).
    "PDF_PATH_OPS": KnobMeta(P, "PDF spec path-construction operators"),
    "PDF_TEXT_OPS": KnobMeta(P, "same set for profiler volume counting"),
    # Sampling: raster heuristics, rigid scan covers.
    "BG_SAMPLE_SCALE": KnobMeta(S, "2x raster balances noise vs cost"),
    "STROKE_BLACK_CUTOFF": KnobMeta(S, "64 fixed the banner-blue nuke (was 128)"),
    "CJK_FONT_CANDIDATES": KnobMeta(
        P,
        "environment config, UBT_CJK_FONT overrides; supports cross-platform fonts "
        "(Linux CJK, Windows YaHei, macOS PingFang/Songti). Entries must be .ttc: "
        "fontTools' TTCollection raises TTLibError on a bare .ttf",
    ),
    # Probe + profiler: single-doc routing.
    "PROBE_MIN_CHARS": KnobMeta(S, "32 chars separates scans from text pages"),
    "PROBE_COLUMN_EDGE_RATIO": KnobMeta(S, "0.45 edge ratio observed on two-column pages"),
    "PROBE_COLUMN_SHARE": KnobMeta(S, "0.25 share observed on two-column pages"),
    "PROBE_COLUMN_GAP_RATIO": KnobMeta(S, "0.03 of page width splits gutter from word gap"),
    "PROBE_MIN_ROWS": KnobMeta(S, "6 rows minimum avoids title-page misfire"),
    "PROBE_SAMPLE_FRACTIONS": KnobMeta(S, "front/quarters/back defeats cover spoofing"),
    "PROBE_FORMULA_SHARE": KnobMeta(S, "0.10 separates formula pages (0.16+) from prose (<0.08)"),
    "PROSE_SINGLE_LETTERS": KnobMeta(P, "a/A/I exclusions, behaviour-tested"),
    "PROFILE_VECTOR_PATH_OPS": KnobMeta(S, "200 ops separates drawings from prose"),
    "PROFILE_VECTOR_TEXT_CHARS": KnobMeta(S, "100 chars separates labels from body"),
    "POSTER_MAX_CHARS": KnobMeta(H, "Image-led poster ceiling, unvalidated"),
    "POSTER_MAX_FONTS": KnobMeta(H, "Title+body font count, unvalidated"),
    "RESUME_MIN_CHARS": KnobMeta(H, "Dense-page floor, unvalidated"),
    "QA_FULL_GATE_MAX_PAGES": KnobMeta(H, "<=20pp full inspection, unvalidated"),
    "QA_TIERED_BAND_PAGES": KnobMeta(H, "20-100pp stratified band, unvalidated"),
    "QA_TIERED_BAND_SAMPLE": KnobMeta(H, "Band sample cap 10, unvalidated"),
    "QA_LONG_BOOK_SAMPLE": KnobMeta(H, ">100pp anchor set 10, unvalidated"),
    "SHORT_CHAIN_MAX_PAGES": KnobMeta(
        H,
        "unified-entry router: <=30pp short chain (covers ~26pp chapters). "
        "Bound calibrated around 20pp (flips short born digital test); 30pp ceiling balances throughput.",
    ),
    "FAST_LANE_MIN_TEXT_CHARS": KnobMeta(H, "200-char scan gate for the fast lane, unvalidated"),
    "LENGTH_OVERFLOW_TO_HUMAN": KnobMeta(
        H, "Overflow skips on length-policy pages go NEEDS_HUMAN, unvalidated"
    ),
    "LENGTH_POLICY_PAGE_KINDS": KnobMeta(
        H, "Resume_dense/poster_fixed under fit-to-page, unvalidated"
    ),
    # Verdict: single-doc keep rules.
    "HEXDUMP_MIN_TOKENS": KnobMeta(S, "32 byte-words observed in hex tables"),
    "HEXDUMP_MIN_SHARE": KnobMeta(S, "0.55 share observed in hex tables"),
    "INDEX_MIN_LINES": KnobMeta(S, "12 lines avoids short-list misfire"),
    "INDEX_MIN_SINGLE_SHARE": KnobMeta(S, "0.85 share observed in index pages"),
    "SHORT_LABEL_MAX_LEN": KnobMeta(S, "24 chars covers figure labels (A, B1)"),
    "BYTE_WORD_RE": KnobMeta(P, "hex byte pattern, behaviour-tested"),
    "SINGLE_TOKEN_LINE_RE": KnobMeta(P, "index-line pattern, behaviour-tested"),
    "SHORT_LABEL_RE": KnobMeta(P, "label pattern, behaviour-tested"),
    "URL_DOI_RE": KnobMeta(P, "identifier patterns, behaviour-tested"),
    "ISBN_DIGITS_RE": KnobMeta(P, "ISBN-10/13 digit rule, behaviour-tested"),
    # Continuation: single-doc weights.
    "JOIN_LOWER_START": KnobMeta(S, "weight 3 from KV断栏 cases"),
    "JOIN_HYPHEN_TAIL": KnobMeta(S, "weight 4, hyphenation is near-certain"),
    "JOIN_CROSS_NO_TERMINAL": KnobMeta(S, "weight 2, weak cross-page signal"),
    "BREAK_TERMINAL": KnobMeta(S, "weight 4 balances hyphen 4"),
    "BREAK_CHAPTER_NUMBER": KnobMeta(S, "weight 6, section numbers never continue"),
    "BREAK_UPPER_START": KnobMeta(S, "weight 1, weak English signal"),
    "JOIN_THRESHOLD": KnobMeta(S, "4 + margin-2 rule from KV cases"),
    "BREAK_THRESHOLD": KnobMeta(S, "4, checked after join"),
    "CONJUNCTIONS": KnobMeta(P, "closed-class word list, behaviour-tested"),
    "CONT_SENTENCE_END_RE": KnobMeta(P, "terminal set, behaviour-tested"),
    "CONT_CHAPTER_NUMBER_RE": KnobMeta(P, "numbering pattern, behaviour-tested"),
    "CONT_UPPER_START_RE": KnobMeta(P, "capital pattern, behaviour-tested"),
    "CONT_LABEL_BARRIER_RE": KnobMeta(S, "caption-prefix barrier tuned on Table/Figure cases"),
    "CONT_LIST_MARKER_RE": KnobMeta(P, "list-marker inventory, behaviour-tested"),
    "HEADING_NUMBER_RE": KnobMeta(P, "dotted section-number pattern, behaviour-tested"),
    "TARGET_NUMBERED_RE": KnobMeta(P, "target-numbered guard incl CJK sections, behaviour-tested"),
    # VLM circuit breaker: principled, never observed against a real dead endpoint.
    "VLM_CIRCUIT_MIN_TRIES": KnobMeta(H, "3 pages before tripping, unvalidated on a live outage"),
    "VLM_CIRCUIT_FAIL_PCT": KnobMeta(H, "50% failure rate, principled not measured"),
    # Permissions: POSIX mechanics, not a tuning judgment (tightest bits that
    # still let the owner read what it just wrote).
    "BOOK_TEXT_DIR_MODE": KnobMeta(P, "0o700 is the tightest mode permitting owner traverse+write"),
    "BOOK_TEXT_FILE_MODE": KnobMeta(
        P, "0o600 is the tightest owner-rw mode; SQLite creates at 0666&~umask = world-readable"
    ),
    # Entry-layer concurrency ceiling: copied from the REST default, never
    # load-tested on either door.
    "MCP_MAX_RUNNING_JOBS": KnobMeta(
        H, "mirrors api/app.py max_running_jobs=8; no measurement behind either"
    ),
}


def calibration_summary() -> dict[str, int]:
    """Count knobs by calibration status."""
    counts = {status.value: 0 for status in Calibration}
    for meta in CALIBRATION.values():
        counts[meta.status.value] += 1
    return counts


_FATAL_LEAK_MARKERS: tuple[str, ...] = (
    "Prompt template XML artifacts",
    "Prompt scaffold",
)


def is_rigid_non_prose_degradable(
    block_type: Any,
    error_flags: Iterable[str],
    *,
    render_engine: str | None = None,
) -> bool:
    """True when a non-prose block (e.g. table) under rigid mode should degrade
    to NEEDS_HUMAN rather than BLOCKED_HUMAN on non-fatal defects.

    In rigid mode, non-prose blocks (tables, code, formulas) are preserved
    verbatim in the rendered PDF via render_skip:non_prose. Unless a defect is
    truly fatal (e.g. prompt template leak), quarantining the block as
    BLOCKED_HUMAN would abort export catastrophically.
    """
    bt = getattr(block_type, "value", block_type or "")
    is_non_prose = str(bt).lower() in ("table", "code", "formula", "image") or (
        str(bt).lower() not in {str(t.value).lower() for t in PROSE_BLOCK_TYPES}
    )
    if not is_non_prose:
        return False

    flags_list = list(error_flags)
    has_render_skip = any("render_skip:non_prose" in f for f in flags_list)
    engine = (render_engine or "").strip().lower()
    is_rigid = engine in ("rigid", "inplace", "hybrid") or has_render_skip
    if not is_rigid:
        return False

    # Truly fatal prompt template leak or security injection must still be quarantined
    return not any(m in flag for flag in flags_list for m in _FATAL_LEAK_MARKERS)


__all__ = [
    "RIGID_MIN_FONT_PT",
    "RIGID_CAPTION_MIN_FONT_PT",
    "RIGID_FOOTNOTE_MIN_FONT_PT",
    "ASCII_WORD_RE",
    "BAND_TEXT_MAX_LEN",
    "BG_SAMPLE_SCALE",
    "BOOK_TEXT_DIR_MODE",
    "BOOK_TEXT_FILE_MODE",
    "BREAK_CHAPTER_NUMBER",
    "BREAK_TERMINAL",
    "BREAK_THRESHOLD",
    "BREAK_UPPER_START",
    "BYTE_WORD_RE",
    "CALIBRATION",
    "CAPTION_RE",
    "CJK_FONT_CANDIDATES",
    "CJK_PUNCT_CHARS",
    "CN_MEASURE_WORDS",
    "CONJUNCTIONS",
    "CONTROL_RE",
    "CONT_CHAPTER_NUMBER_RE",
    "CONT_LABEL_BARRIER_RE",
    "CONT_LIST_MARKER_RE",
    "HEADING_NUMBER_RE",
    "TARGET_NUMBERED_RE",
    "CONT_SENTENCE_END_RE",
    "CONT_UPPER_START_RE",
    "EM_ASCII_ADV",
    "EM_CJK_ADV",
    "EM_PUNCT_ADV",
    "EM_SPACE_ADV",
    "FAST_LANE_MIN_TEXT_CHARS",
    "FIT_MIN_FONT_PT",
    "FIT_PRECISION_PT",
    "FOLD_MAP",
    "FOOTER_BAND_PT",
    "FOOTER_PATTERNS",
    "H",
    "HEADER_BAND_PT",
    "HEXDUMP_MIN_SHARE",
    "HEXDUMP_MIN_TOKENS",
    "INDEX_MIN_LINES",
    "INDEX_MIN_SINGLE_SHARE",
    "ISBN_DIGITS_RE",
    "is_rigid_non_prose_degradable",
    "JOIN_CROSS_NO_TERMINAL",
    "JOIN_HYPHEN_TAIL",
    "JOIN_LOWER_START",
    "JOIN_THRESHOLD",
    "LENGTH_OVERFLOW_TO_HUMAN",
    "LENGTH_POLICY_PAGE_KINDS",
    "MCP_MAX_RUNNING_JOBS",
    "NON_PROSE_FLOWS",
    "NON_TEXT_BLOCK_TYPES",
    "P",
    "PAIR_TERMINAL_PUNCT",
    "PDF_PATH_OPS",
    "PDF_TEXT_OPS",
    "POSTER_MAX_CHARS",
    "POSTER_MAX_FONTS",
    "PROBE_COLUMN_EDGE_RATIO",
    "PROBE_COLUMN_SHARE",
    "PROBE_FORMULA_SHARE",
    "PROBE_MIN_CHARS",
    "PROBE_MIN_ROWS",
    "PROBE_SAMPLE_FRACTIONS",
    "PROFILE_VECTOR_PATH_OPS",
    "PROFILE_VECTOR_TEXT_CHARS",
    "PROSE_BLOCK_TYPES",
    "PROSE_SINGLE_LETTERS",
    "QA_FULL_GATE_MAX_PAGES",
    "QA_LONG_BOOK_SAMPLE",
    "QA_TIERED_BAND_PAGES",
    "QA_TIERED_BAND_SAMPLE",
    "RESUME_MIN_CHARS",
    "S",
    "SHORT_CHAIN_MAX_PAGES",
    "SHORT_LABEL_MAX_LEN",
    "SHORT_LABEL_RE",
    "SINGLE_TOKEN_LINE_RE",
    "STROKE_BLACK_CUTOFF",
    "UNICODE_WORD_RE",
    "URL_DOI_RE",
    "VLM_CIRCUIT_FAIL_PCT",
    "VLM_CIRCUIT_MIN_TRIES",
    "WS_RE",
]
