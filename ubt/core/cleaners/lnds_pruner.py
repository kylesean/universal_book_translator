"""Longest Non-Decreasing Subsequence (LNDS) Page Number Pruning and Calibre Cleaner."""

import bisect
import re
import sys

from ubt.core.cleaners.boilerplate_catalog import BoilerplateCatalog
from ubt.core.cleaners.inline_math import INLINE_DOLLAR_PATTERN, is_math_content
from ubt.core.cleaners.math_masker import _DISPLAY_DOLLAR_PATTERN
from ubt.core.ir.models import BlockStatus, BlockType, IRBlock


def _has_dollar_math(content: str) -> bool:
    """True when text contains display or inline LaTeX math delimited by $."""
    if "$" not in content:
        return False
    if _DISPLAY_DOLLAR_PATTERN.search(content):
        return True
    return any(is_math_content(m.group(1)) for m in INLINE_DOLLAR_PATTERN.finditer(content))


# Extraction-debris control chars: never legitimate prose, not even inside
# formulas (b0030-class ``T si  L g`` junk survived into translation).
# \x00 excluded: math_text uses it as its own SPAN sentinel downstream, and
# stripping a live sentinel would corrupt reassembly catastrophically.
# \t\n\r preserved (whitespace structure).
_CONTROL_DEBRIS_RE = re.compile(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]")
_CMAP_STOPWORDS = frozenset(
    {"a", "an", "the", "add", "mix", "take", "in", "of", "to", "or", "and", "is", "for", "with"}
)
_MEASURE_NOUNS = frozenset(
    {"cup", "share", "inch", "mile", "pound", "ounce", "hour", "page", "part", "tsp", "tbsp"}
)

# Equation-shaped ¼ patterns (CMap '=' corruption). Compiled once so the
# evidence gate and rules 3a/3b below share a single verdict.
_CMAP_EQ_NUM_RE = re.compile(r"([A-Za-z0-9_\)\]\u0370-\u03ff])\s*¼\s*([+-]?\d)")
_CMAP_EQ_VAR_RE = re.compile(
    r"\b([A-Za-z\u0370-\u03ff][A-Za-z0-9_]{0,10}(?:\([^)]*\))?)"
    r"\s*¼\s*([A-Za-z\u0370-\u03ff][A-Za-z0-9_]{0,10})\b"
)


def _is_equation_fraction(left: str, right: str) -> bool:
    """Whether a ¼ flanked by ``left``/``right`` reads as corrupted '='."""
    base_left = re.sub(r"\(.*?\)", "", left).strip().lower()
    return base_left not in _CMAP_STOPWORDS and right.strip().lower() not in _MEASURE_NOUNS


def _has_cmap_equation_evidence(content: str) -> bool:
    """True only when a ¼ sits in equation shape (same verdict as rules 3a/3b)."""
    if _CMAP_EQ_NUM_RE.search(content):
        return True
    return any(
        _is_equation_fraction(m.group(1), m.group(2)) for m in _CMAP_EQ_VAR_RE.finditer(content)
    )


_PAGE_SEQUENCE_MIN_LENGTH = 4
_PAGE_SEQUENCE_MIN_RATIO = 0.5


# Standalone ASCII digit line. str.isdigit() is NOT the right gate: it also
# accepts Unicode digit characters (superscripts '²', full-width '１') whose
# int() conversion can raise ValueError, which crashed the LNDS scan — and a
# crashed cleaner meant the whole chapter failed ingestion. The LNDS sequence
# operates on ASCII page numbers only, so the gate accepts exactly ASCII
# digits: re.ASCII pins \d to [0-9] (Python \d otherwise matches every Nd
# character, the same trap one level below isdigit).
_ASCII_DIGITS_RE = re.compile(r"\d+", re.ASCII)

# int()-from-string is capped at this many digits (4300 by default; 0 when the
# limit is disabled). A longer all-ASCII-digit line passes a naive gate and then
# raised ``ValueError: Exceeds the limit ...`` inside the LNDS scan, crashing
# the cleaner and the whole chapter's ingestion. Page numbers are tiny; rejecting
# over-long lines also keeps the docstring's "int()-safe" contract literally true.
_MAX_INT_DIGITS = sys.get_int_max_str_digits()


def is_ascii_digit_line(line: str) -> bool:
    """True when the stripped line is ASCII digits ([0-9]) that ``int()`` can parse."""
    s = line.strip()
    if not s or (_MAX_INT_DIGITS and len(s) > _MAX_INT_DIGITS):
        return False
    return _ASCII_DIGITS_RE.fullmatch(s) is not None


def detect_page_number_lines(
    lines: list[str],
    min_length: int = _PAGE_SEQUENCE_MIN_LENGTH,
    min_ratio: float = _PAGE_SEQUENCE_MIN_RATIO,
) -> set[int]:
    """Detect standalone-digit lines that form a monotonic page-number sequence via LNDS."""
    digit_indices: list[int] = []
    digit_values: list[int] = []
    for i, line in enumerate(lines):
        if is_ascii_digit_line(line):
            digit_indices.append(i)
            digit_values.append(int(line.strip()))

    n = len(digit_values)
    if n < min_length:
        return set()

    tails: list[int] = []
    tails_idx: list[int] = []
    parents = [-1] * n

    for i, v in enumerate(digit_values):
        pos = bisect.bisect_right(tails, v)
        if pos > 0:
            parents[i] = tails_idx[pos - 1]
        if pos == len(tails):
            tails.append(v)
            tails_idx.append(i)
        else:
            tails[pos] = v
            tails_idx[pos] = i

    lnds: list[int] = []
    cur = tails_idx[-1]
    while cur != -1:
        lnds.append(cur)
        cur = parents[cur]
    lnds.reverse()

    if len(lnds) < min_length:
        return set()
    if len(lnds) / n < min_ratio:
        return set()

    return {digit_indices[i] for i in lnds}


def strip_calibre_markup(content: str) -> str:
    """Strip Calibre pseudo-classes, links, and redundant bold bracket markers."""
    content = re.sub(r"\{\.calibre[^}]*\}", "", content)
    content = re.sub(r"\(#calibre_link-\d+\)", "", content)
    content = re.sub(r"\s*\{#calibre_link-\d+[^}]*\}", "", content)
    return re.sub(r"\[\*\*([^*]+)\*\*\]", r"**\1**", content)


def is_calibre_noise_line(line: str) -> bool:
    """Check whether a single line is standalone Calibre delimiter or callout syntax."""
    s = line.strip()
    if not s:
        return False
    if s.startswith(":::"):
        return True
    return bool(s.endswith(".ct}") or s.endswith(".cn}"))


def is_likely_heading_or_list_item(line: str) -> bool:
    """True if line looks like a title, TOC entry, or numbered item description."""
    s = line.strip()
    if not s:
        return False
    if is_calibre_noise_line(s):
        return False
    # Heading keywords or markdown bullet/number markers
    if s.startswith(("#", "-", "*", "•", "Chapter", "Section", "Part", "Step")):
        return True
    # Short line (< 80 chars) not ending with sentence-terminal punctuation
    return len(s) < 80 and not s.endswith((".", "?", "!", "。", "？", "！", ";", "；"))


def collect_dropped_line_indices(
    lines: list[str], strip_all_page_numbers: bool = False
) -> set[int]:
    """Identify lines that are Calibre structural noise or monotonic LNDS page numbers."""
    page_number_lines = set() if strip_all_page_numbers else detect_page_number_lines(lines)

    # One pass in each direction instead of a linear walk per digit line:
    # page-number-dense chapters (numbers interleaved with blanks) made the
    # per-line scans approach O(n^2).
    prev_nonblank_line: list[str | None] = [None] * len(lines)
    last: str | None = None
    for i, line in enumerate(lines):
        prev_nonblank_line[i] = last
        if line.strip():
            last = line
    next_nonblank_line: list[str | None] = [None] * len(lines)
    nxt: str | None = None
    for i in range(len(lines) - 1, -1, -1):
        next_nonblank_line[i] = nxt
        if lines[i].strip():
            nxt = lines[i]

    dropped: set[int] = set()
    # Last number of a colon-introduced ordered list, so a 4+ item list whose
    # items are full sentences keeps every marker. Without it only the first
    # number is protected (by the ":" rule) and the rest are read as the LNDS
    # page-number sequence: "Steps:\n1\nInstall…\n2\nRun…\n3\nDeploy…" lost
    # 2/3/4 and shipped a mis-numbered list.
    list_run_prev: int | None = None
    for i, line in enumerate(lines):
        if is_calibre_noise_line(line):
            dropped.add(i)
            continue

        if is_ascii_digit_line(line):
            prev = prev_nonblank_line[i]
            nxt = next_nonblank_line[i]
            value = int(line.strip())

            # Protection for TOC entries and ordered list numbering
            # e.g., "1\nIntroduction" or list with introductory colon "Steps:\n1\nFirst"
            if nxt is not None and is_likely_heading_or_list_item(nxt):
                continue
            if prev is not None and prev.strip().endswith(":"):
                list_run_prev = value
                continue
            # The next number of the same colon-introduced list.
            if list_run_prev is not None and value == list_run_prev + 1:
                list_run_prev = value
                continue
            list_run_prev = None

            if strip_all_page_numbers or i in page_number_lines:
                dropped.add(i)
                continue
            if (prev is not None and is_calibre_noise_line(prev)) or (
                nxt is not None and is_calibre_noise_line(nxt)
            ):
                dropped.add(i)
                continue

    return dropped


def normalize_academic_pdf_math(content: str) -> str:
    """Repair publisher font CMap corruption and flattened math in academic PDFs.

    - Elsevier / Academic Press CMap bug: byte 0xBC in math fonts incorrectly mapped
      to Latin-1 U+00BC (¼) instead of '=' when flanked by identifiers and values.
    - PDF stream glyph escapes: /C2 -> ×, /C0 -> -, /uni03BC -> μ.
    - Kerning-split subscripts: 't ox =', 'N ch =', 'V ch =' -> 't_ox =', 'N_ch =', 'V_ch ='.
    - Flattened scientific notation & power subscripts: '1 - 10 15 cm 3' -> '1 × 10^15 cm^-3'.
    - Null byte debris from font glyph extraction.
    """
    if not content:
        return ""
    # 1. Null byte debris (preserve \x00SPAN sentinel if already present)
    if "\x00" in content:
        sentinels = re.findall(r"\x00SPAN\d+\x00", content)
        if sentinels:
            for idx, s_tag in enumerate(sentinels):
                content = content.replace(s_tag, f"__UBT_SPAN_SENTINEL_{idx}__")
            content = content.replace("\x00", "")
            for idx, s_tag in enumerate(sentinels):
                content = content.replace(f"__UBT_SPAN_SENTINEL_{idx}__", s_tag)
        else:
            content = content.replace("\x00", "")

    # 2. PDF stream glyph escapes & soft-hyphen line-break word splits
    # Docling preserves U+00AD (\xad) followed by a space on hyphenated line
    # wraps (e.g. 'compos\xad ability' -> 'compos ability'); rejoin split words.
    content = re.sub(r"([A-Za-z])\xad[ \t]*([a-z])", r"\1\2", content)
    content = content.replace("\xad", "")
    content = content.replace("/C2", "×").replace("/C0", "-").replace("/uni03BC", "μ")

    # Evidence gate for the paren / dollar repairs below: ð/Þ and
    # '$' mappings come from the SAME broken embedded-math CMap that maps '='
    # to '¼'. But a bare "¼ in content" is too coarse: ordinary prose carries
    # genuine vulgar fractions ("¼ cup") next to currency ("paid $5"), and a
    # block-global gate rewrites the currency into "-5". Evidence counts only
    # when a ¼ sits in equation shape — i.e. rules 3a/3b (with their
    # stopword/measure-noun guards) would actually interpret it as '='.
    cmap_equation_evidence = _has_cmap_equation_evidence(content)

    # 3. Academic PDF CMap bug: ¼ -> =
    # 3a. When right-hand side is a number, sign, or operator
    content = re.sub(_CMAP_EQ_NUM_RE, r"\1 = \2", content)

    # 3b. When right-hand side is an identifier or variable (e.g. vG ¼ VG, qm ¼ Qm, Vch(0) ¼ Vs)
    def _replace_var_eq(m: re.Match[str]) -> str:
        left = m.group(1).strip()
        right = m.group(2).strip()
        if not _is_equation_fraction(left, right):
            return m.group(0)
        return f"{left} = {right}"

    content = re.sub(
        _CMAP_EQ_VAR_RE,
        _replace_var_eq,
        content,
    )

    # 3d. Academic math font CMap parentheses corruption: ð -> (, Þ -> )
    # Evidence-gated: requires cmap_equation_evidence to prevent false-positives
    # in Icelandic or Old English prose where ð/Þ are ordinary letters.
    if cmap_equation_evidence:
        content = content.replace("ð", "(").replace("Þ", ")")

    # 3e. Academic math font CMap bug: minus sign '-' mapped to '$'
    # Evidence-gated: without the gate, any currency amount after
    # whitespace ("paid $5") became "-5".
    if cmap_equation_evidence:
        content = re.sub(r"(\s|\(|\[|^)\$([0-9.]+)", r"\1-\2", content)
        content = re.sub(r"\s+\$\s+(?=[a-zA-Z0-9_\(\)])", " - ", content)
        content = re.sub(r"\$([a-zA-Z\u0370-\u03ff])", r"-\1", content)
        content = re.sub(r"(\b[a-zA-Z0-9_]+)\$([0-9]+)\b", r"\1^{-\2}", content)

    # 4. Kerning-split subscripts before math operators. The comma was in the
    # lookahead and matched ordinary prose ("I am," -> "I_am,"); a flattened
    # subscript is followed by an operator, not a sentence comma.
    content = re.sub(r"\b([A-Za-z])\s+([a-z]{1,3}|[A-Z]{2,3})\b(?=\s*[=≈~≤≥<>])", r"\1_\2", content)

    # 5. Flattened scientific notation & powers
    content = re.sub(
        r"(\d+)\s*[-×]\s*10\s*(\d{1,2})\s*cm\s*[-−]?\s*(\d)\b", r"\1 × 10^\2 cm^-\3", content
    )
    content = re.sub(r"(\d+)\s*[-×]\s*10\s*(\d{1,2})\s*cm\s*(\d)\b", r"\1 × 10^\2 cm^-\3", content)
    content = re.sub(r"(\d+)\s*[-×]\s*10\s*(\d{1,2})\s*cm\s*[-−]?3\b", r"\1 × 10^\2 cm^-3", content)
    content = re.sub(r"(\d+)\s*[-×]\s*10\s*(\d{1,2})\b(?=\s*[A-Za-z])", r"\1 × 10^\2", content)

    return content


def strip_textbook_ocr_artifacts(content: str, source_lang: str = "en") -> str:
    """Strip recurring publisher legal boilerplate, OCR running headers, and photo credits from scanned textbooks."""
    if not content:
        return ""

    # 0. Extraction-debris control chars are always stripped first.
    content = _CONTROL_DEBRIS_RE.sub("", content)

    # Layer 3 Invariant: strict immunity for mathematical formulas,
    # LaTeX markup, and code. Checked BEFORE the CMap normalizer: its
    # '/C2' -> '×' / '/C0' -> '-' rewrites would corrupt code fences
    # ('/C2/data') and inline math ('$x /C2 y$'). Only the CMap step is skipped
    # for such content — the boilerplate / page-marker / running-header stripping
    # below must still run (a stray backtick must not disable all of it).
    protect_math = (
        _has_dollar_math(content)
        or "\\begin" in content
        or "```" in content
        or "`" in content
        or "\\frac" in content
    )
    if not protect_math:
        content = normalize_academic_pdf_math(content)

    # 1. Strip publisher legal disclaimers for the source language (via BoilerplateCatalog)
    legal_pat = BoilerplateCatalog.get_legal_pattern(source_lang)
    if legal_pat:
        content = legal_pat.sub("", content)

    # 2. Strip photographer / picture agency attribution credits
    photo_pat = BoilerplateCatalog.get_photo_credit_pattern(source_lang)
    if photo_pat:
        content = photo_pat.sub("", content)

    # 3. Strip standalone page headers like 'Page \d+'
    page_pat = BoilerplateCatalog.get_page_marker_pattern(source_lang)
    if page_pat:
        content = page_pat.sub("", content)

    # 4. Strip textbook OCR running headers glued to the beginning of text
    # (Prefilter on the literal chapter keyword before running the regex
    # so adversarial lines can never trigger catastrophic backtracking)
    hdr_pat = BoilerplateCatalog.get_running_header_pattern(source_lang)
    if hdr_pat:
        keywords = BoilerplateCatalog.get_running_header_keywords(source_lang)
        if not keywords or any(k in content.upper() for k in keywords):
            content = hdr_pat.sub("", content)

    # 5. Safely strip isolated OCR backslash artifacts and stray decorative symbol runs
    # without mutilating LaTeX formulas, paths, or words.
    # Isolated backslashes surrounded by whitespace are stripped, leaving paths and markup intact.
    content = re.sub(r"(?:(?<=\s)|^)\\+(?=\s)", " ", content)
    # Strip decorative '=' / '~' separator lines while preserving in-prose math/symbols.
    content = re.sub(r"(?m)^[ \t]*[=~]{1,20}[ \t]*$", "", content)

    # Collapse horizontal whitespace runs — but never inside fenced code, where
    # leading indentation is semantics (a collapsed fence turns valid Python
    # into an IndentationError). Segments alternate prose / fence on "```".
    segments = content.split("```")
    for i in range(0, len(segments), 2):
        segments[i] = re.sub(r"[ \t]+", " ", segments[i])
    return "```".join(segments).strip()


def clean_calibre_and_lnds_pages(
    content: str, strip_all_page_numbers: bool = False, source_lang: str = "en"
) -> str:
    """Strip Calibre pseudo-classes, textbook OCR artifacts, and monotonic page-number lines while keeping years/headings."""
    content = strip_calibre_markup(content)
    content = strip_textbook_ocr_artifacts(content, source_lang=source_lang)
    lines = content.split("\n")
    dropped = collect_dropped_line_indices(lines, strip_all_page_numbers=strip_all_page_numbers)
    cleaned_lines = [line for i, line in enumerate(lines) if i not in dropped]
    cleaned_content = "\n".join(cleaned_lines)
    return re.sub(r"\n{3,}", "\n\n", cleaned_content)


class LNDSPageCleaner:
    """Stateless cleaner for stripping monotonic page numbers and Calibre artifacts."""

    def __init__(self, strip_all_page_numbers: bool = False, source_lang: str = "en") -> None:
        self.strip_all_page_numbers = strip_all_page_numbers
        self.source_lang = source_lang

    def clean(self, text: str) -> str:
        return clean_calibre_and_lnds_pages(
            text, strip_all_page_numbers=self.strip_all_page_numbers, source_lang=self.source_lang
        )

    def clean_block(self, block: IRBlock) -> IRBlock:
        """Clean Calibre markup and local noise artifacts within an individual block.

        WARNING:
            LNDS monotonic page number detection requires chapter-wide sequence context
            (at least 4 sequential page numbers across lines). Calling this method on an
            isolated single-paragraph block will NOT trigger LNDS sequence stripping.
            For document translation pipelines, always use :meth:`clean_chapter_blocks`.
        """
        cleaned_text = strip_calibre_markup(block.source_text)
        if block.block_type is not BlockType.CODE:
            # CODE blocks ship verbatim, so their whitespace is semantics: the
            # OCR-artifact pass ends in a horizontal-whitespace collapse that
            # turns valid indentation into an IndentationError.
            cleaned_text = strip_textbook_ocr_artifacts(cleaned_text, source_lang=self.source_lang)
        return block.with_source_text(cleaned_text)

    def clean_chapter_blocks(self, blocks: list[IRBlock]) -> list[IRBlock]:
        """Clean Calibre noise and cross-block monotonic page numbers across a chapter."""
        if not blocks:
            return []

        # 1. Flatten all blocks' lines into a global list while tracking mapping
        global_lines: list[str] = []
        mapping: list[tuple[int, int]] = []  # (block_idx, line_idx_within_block)
        pre_cleaned_block_lines: list[list[str]] = []

        for b_idx, b in enumerate(blocks):
            text = strip_calibre_markup(b.source_text)
            if b.block_type is not BlockType.CODE:
                # CODE blocks ship verbatim, so their whitespace is semantics:
                # the OCR-artifact pass ends in a horizontal-whitespace
                # collapse that turns valid indentation into an IndentationError.
                text = strip_textbook_ocr_artifacts(text, source_lang=self.source_lang)
            blines = text.split("\n")
            pre_cleaned_block_lines.append(blines)
            for l_idx, line in enumerate(blines):
                global_lines.append(line)
                mapping.append((b_idx, l_idx))

        # 2. Collect dropped line indices globally across all lines in the chapter
        dropped_global_lines = collect_dropped_line_indices(
            global_lines, strip_all_page_numbers=self.strip_all_page_numbers
        )
        protected_block_indices = {
            b_idx
            for b_idx, b in enumerate(blocks)
            if b.block_type
            in (
                BlockType.HEADING,
                BlockType.LIST_ITEM,
                BlockType.CODE,
                BlockType.FORMULA,
                BlockType.TABLE,
            )
        }
        dropped_coords: set[tuple[int, int]] = {
            mapping[g] for g in dropped_global_lines if mapping[g][0] not in protected_block_indices
        }

        # 3. Reconstruct each block
        result_blocks: list[IRBlock] = []
        for b_idx, (orig_block, blines) in enumerate(
            zip(blocks, pre_cleaned_block_lines, strict=False)
        ):
            kept_lines = [
                line for l_idx, line in enumerate(blines) if (b_idx, l_idx) not in dropped_coords
            ]
            cleaned_text = "\n".join(kept_lines).strip()
            cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

            if not cleaned_text and orig_block.source_text.strip():
                # Block was purely a dropped page number or noise line
                if orig_block.block_type in (
                    BlockType.HEADING,
                    BlockType.LIST_ITEM,
                    BlockType.CODE,
                    BlockType.FORMULA,
                    BlockType.TABLE,
                ):
                    new_block = orig_block
                else:
                    new_block = orig_block.model_copy(
                        update={
                            "target_text": "",
                            "status": BlockStatus.MTQE_PASSED,
                            "mtqe_score": 1.0,
                        }
                    )
                    new_block.set_source_text("")
                    new_block.skip_translate = True
            else:
                new_block = orig_block.with_source_text(cleaned_text)
            result_blocks.append(new_block)

        return result_blocks


_WS_RE = re.compile(r"\s+")
# IoU at/above which two same-text boxes are the same physical region. Set to
# match the visual gate's overlap bar so a deduped pair can never re-surface as
# a block_overlap finding.
_DEDUP_IOU = 0.6


def _norm_text(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip()).casefold()


def _rect(block: IRBlock) -> tuple[float, float, float, float] | None:
    if block.bbox is None:
        return None
    b = block.bbox
    return (b.x0, b.y0, b.x1, b.y1)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _area(rect: tuple[float, float, float, float]) -> float:
    return max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])


def dedup_duplicate_blocks(blocks: list[IRBlock]) -> list[IRBlock]:
    """Drop physically duplicated extraction fragments (same page, same text).

    Docling sometimes emits a figure's sub-panel label twice — two narrative
    blocks with identical text and near-coincident boxes (arXiv 2609.20519
    page 1: ``(a) SoL-Pi: Scaling Auto-Research Loop`` extracted as both
    ``b0008`` and ``b0009``, IoU 0.95). The pair survives into the overlay as
    two ``no_zone`` blocks whose boxes overlap, tripping the visual gate's
    ``block_overlap`` finding and cascading ``needs_human`` onto every block
    that shares the page. Requiring *identical normalized text* keeps
    legitimately overlapping-but-distinct content (a caption over an image, a
    table cell over a rule) intact; only true duplicates are removed, and the
    larger box (the more complete fragment) is kept.
    """
    if len(blocks) < 2:
        return blocks
    dropped: set[int] = set()
    # Bucket by (page, normalized text): duplicates share both.
    by_key: dict[tuple[int, str], list[int]] = {}
    for idx, block in enumerate(blocks):
        if block.bbox is None:
            continue
        key = (block.bbox.page, _norm_text(block.source_text))
        if not key[1]:
            continue
        by_key.setdefault(key, []).append(idx)
    for idxs in by_key.values():
        if len(idxs) < 2:
            continue
        # Keep the largest box; drop the rest that overlap it past the bar.
        keeper = max(idxs, key=lambda i: _area(_rect(blocks[i]) or (0, 0, 0, 0)))
        keeper_rect = _rect(blocks[keeper])
        for i in idxs:
            if i == keeper:
                continue
            other = _rect(blocks[i])
            if (
                other is not None
                and keeper_rect is not None
                and _iou(keeper_rect, other) >= _DEDUP_IOU
            ):
                dropped.add(i)
    if not dropped:
        return blocks
    return [block for idx, block in enumerate(blocks) if idx not in dropped]
