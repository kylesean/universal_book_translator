"""Unit tests for cleaners: LNDS page pruning and code masking."""

import re

import pytest

from ubt.adapters.epub.adapter import BLOCK_TAGS, is_leaf_block
from ubt.adapters.html.adapter import HTMLAdapter
from ubt.core.cleaners.code_masker import CodeMasker
from ubt.core.cleaners.lnds_pruner import (
    LNDSPageCleaner,
    clean_calibre_and_lnds_pages,
    collect_dropped_line_indices,
    dedup_duplicate_blocks,
    detect_page_number_lines,
    is_ascii_digit_line,
    normalize_academic_pdf_math,
)
from ubt.core.ir.models import BoundingBox, IRBlock
from ubt.core.router.extractor import TranslationOutputExtractor

pytestmark = pytest.mark.fast


def test_lnds_detect_monotonic_page_numbers_and_preserve_1984() -> None:
    """Validate that monotonic sequential page numbers are pruned while 1984 is preserved."""
    lines = [
        "Chapter One",
        "It was a bright cold day in April, and the clocks were striking thirteen.",
        "1",  # Page 1
        "Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind.",
        "2",  # Page 2
        "In the year 1984 there was no way of knowing whether you were being watched.",
        "3",  # Page 3
        "The telescreen received and transmitted simultaneously.",
        "4",  # Page 4
        "Any sound that Winston made was picked up by it.",
        "5",  # Page 5
    ]

    dropped_indices = detect_page_number_lines(lines)
    # 1, 2, 3, 4, 5 are line indices 2, 4, 6, 8, 10
    assert 2 in dropped_indices
    assert 4 in dropped_indices
    assert 6 in dropped_indices
    assert 8 in dropped_indices
    assert 10 in dropped_indices

    cleaned = clean_calibre_and_lnds_pages("\n".join(lines))
    # 1984 must be strictly preserved!
    assert "1984" in cleaned
    assert "Chapter One" in cleaned
    # Standalone page digits should be gone
    cleaned_lines = [line.strip() for line in cleaned.split("\n")]
    for p in ["1", "2", "3", "4", "5"]:
        assert p not in cleaned_lines


def test_calibre_noise_stripping() -> None:
    """Validate stripping of Calibre CSS pseudo-classes and bookmark links."""
    raw_markdown = (
        "This is text with {.calibre_1} noise and (#calibre_link-42) links.\n"
        ":::\n"
        "Some callout content {.ct}\n"
        "Normal text paragraph."
    )
    cleaned = clean_calibre_and_lnds_pages(raw_markdown)
    assert "{.calibre_1}" not in cleaned
    assert "(#calibre_link-42)" not in cleaned
    assert ":::" not in cleaned
    assert "Normal text paragraph." in cleaned


def test_superscript_digit_line_does_not_crash_lnds_scan() -> None:
    """A '²' standalone line must not crash the LNDS page-number scan.

    User-visible failure prevented: ``str.isdigit()`` also accepts Unicode
    digit characters (superscript '²', Devanagari '१'), whose ``int()``
    conversion raises ``ValueError``. A scanned page carrying a stray
    superscript crashed the cleaner and took the whole chapter's ingestion
    down with it. The LNDS scan operates on ASCII page numbers only, so
    Unicode digit lines are simply not page-number candidates.
    """
    lines = [
        "Chapter One",
        "First paragraph of prose.",
        "1",  # page numbers 1..5 form the LNDS sequence
        "Second paragraph of prose.",
        "2",
        "Third paragraph of prose.",
        "²",  # isdigit() -> True, but int('²') raised ValueError
        "Fourth paragraph of prose.",
        "3",
        "Fifth paragraph of prose.",
        "4",
        "Sixth paragraph of prose.",
        "5",
    ]
    # Must not raise ValueError.
    dropped = detect_page_number_lines(lines)
    # The ASCII page numbers are still detected as the monotonic sequence.
    assert dropped == {2, 4, 8, 10, 12}
    assert 6 not in dropped  # the superscript line is never a page number


def test_superscript_line_survives_full_clean_and_ascii_gate_is_exact() -> None:
    """clean_calibre_and_lnds_pages keeps '²' prose and the gate matches int()'s domain.

    User-visible failure prevented: the crash in ``detect_page_number_lines``
    propagated through ``collect_dropped_line_indices`` (which runs the same
    scan with the same ``int()`` after the regex gate) and through
    ``clean_calibre_and_lnds_pages``, failing the chapter. Also pins the
    helper's contract: it accepts exactly the lines ``int()`` can parse.
    """
    content = "Area is measured in cm².\n1\nProse A.\n2\nProse B.\n²\nProse C.\n3\nProse D.\n4\n"
    cleaned = clean_calibre_and_lnds_pages("\n".join(content.split("\n")))
    assert "Area is measured in cm²." in cleaned
    assert "Prose C." in cleaned  # the superscript line's neighbor survived

    # The gate accepts exactly what int() accepts, and nothing wider.
    for line in ("1", " 23 ", "0", "9999"):
        assert is_ascii_digit_line(line)
        assert int(line.strip()) >= 0  # int() parses safely by contract
    for line in ("²", "１２", "१२", "1 2", "", "12a", "-5", "1.5"):
        assert not is_ascii_digit_line(line)


def test_collect_dropped_line_indices_superscript_line_no_crash() -> None:
    """``collect_dropped_line_indices`` runs the same scan behind a regex gate."""
    lines = ["Prose.", "²", "Prose again."]
    assert collect_dropped_line_indices(lines) == set()


def test_clean_block_immutability() -> None:
    """Validate that LNDSPageCleaner does not mutate the input IRBlock in-place."""
    block = IRBlock(
        id="b01",
        spine_index=1,
        source_text="Page {.calibre_9}\n42\nReal content.",
    )
    cleaner = LNDSPageCleaner()
    new_block = cleaner.clean_block(block)

    assert new_block is not block
    assert "calibre" in block.source_text  # Original remains intact
    assert "calibre" not in new_block.source_text  # New block is cleaned


def test_code_masker_roundtrip() -> None:
    """Validate that code blocks and inline code are safely masked and restored."""
    text = (
        "Here is a function `process_user_id(user_id)`:\n"
        "```python\n"
        "def process_user_id(uid: int) -> bool:\n"
        "    return uid > 1000\n"
        "```\n"
        "Please use `validate()` before calling."
    )
    masker = CodeMasker()
    masked, mapping = masker.mask(text)

    assert "def process_user_id" not in masked
    assert re.search(r"⟦CODE_MASK_0001-[0-9a-f]{3}⟧", masked)
    assert re.search(r"⟦CODE_MASK_0002-[0-9a-f]{3}⟧", masked)
    assert len(mapping) == 3

    # Simulate translation preserving mask tokens
    simulated_translation = masked.replace("Here is a function", "这是一个函数").replace(
        "Please use", "调用前请使用"
    )

    restored = masker.unmask(simulated_translation, mapping)
    assert "这是一个函数 `process_user_id(user_id)`:" in restored
    assert "def process_user_id(uid: int) -> bool:" in restored
    assert "调用前请使用 `validate()`" in restored


def test_code_masker_renumbered_token_is_not_silently_swapped() -> None:
    """A renumbered token must not restore the wrong code span (checksum binding)."""
    masker = CodeMasker()
    text = "Call `foo()` then `bar()`."
    masked, mapping = masker.mask(text)
    tokens = list(mapping)
    tampered = masked.replace(tokens[0], tokens[0].replace("0001", "0002"))

    unmasked = masker.unmask(tampered, mapping)
    assert "`foo()`" not in unmasked

    report = masker.unmask_checked(tampered, mapping)
    assert not report.clean
    assert report.mismatched == [2]
    assert report.missing == [1]


def test_textbook_ocr_artifacts_stripping() -> None:
    """Validate stripping of Cengage legal boilerplates, OCR running headers, and photo credits."""
    from ubt.core.cleaners.lnds_pruner import strip_textbook_ocr_artifacts

    sample_page = (
        "Page 42\n"
        "Introduction to Cognitive Psychology e CHAPTER 1 13 Criticisms of Behaviorism\n"
        "Behaviorism was challenged on many fronts such as language acquisition, production, and comprehension.\n"
        "Bettmann/Corbis Copyright 2017 Cengage Learning. All Rights Reserved. May not be copied, scanned, or duplicated, in whole or in part.\n"
        "Editorial review has deemed that any suppressed content does not materially affect the overall learning experience."
    )

    cleaned = strip_textbook_ocr_artifacts(sample_page)
    assert "Page 42" not in cleaned
    assert "CHAPTER 1 13" not in cleaned
    assert "Cengage Learning" not in cleaned
    assert "Editorial review has deemed" not in cleaned
    assert "Bettmann/Corbis" not in cleaned
    assert cleaned.startswith("Criticisms of Behaviorism")
    assert "language acquisition, production, and comprehension" in cleaned


def test_code_and_math_immune_to_cmap_glyph_rewrites() -> None:
    """Code fences and $...$ math must reach the CMap normalizer untouched."""
    from ubt.core.cleaners.lnds_pruner import strip_textbook_ocr_artifacts

    code = "```python\npath = '/C2/data'\n```"
    assert "/C2/data" in strip_textbook_ocr_artifacts(code)
    assert "$x /C2 y$" in strip_textbook_ocr_artifacts("$x /C2 y$")
    assert "$x /C0 y$" in strip_textbook_ocr_artifacts("$x /C0 y$")


def test_lnds_preserves_toc_and_numbered_list_items() -> None:
    """TOC section numbers and list numbers must not be pruned as page numbers."""
    toc_lines = [
        "Table of Contents",
        "1",
        "Introduction",
        "2",
        "Background and Prior Art",
        "3",
        "System Architecture",
        "4",
        "Experimental Evaluation",
        "5",
        "Conclusion",
    ]
    cleaned = clean_calibre_and_lnds_pages("\n".join(toc_lines))
    for num in ["1", "2", "3", "4", "5"]:
        assert f"\n{num}\n" in f"\n{cleaned}\n"
    assert "Introduction" in cleaned
    assert "System Architecture" in cleaned

    # Step-by-step list with intro colon
    list_lines = [
        "Follow these steps:",
        "1",
        "Install dependencies",
        "2",
        "Run migrations",
    ]
    cleaned_list = clean_calibre_and_lnds_pages("\n".join(list_lines))
    assert "1" in cleaned_list
    assert "2" in cleaned_list


def test_clean_chapter_blocks_protects_structural_headings_and_list_items() -> None:
    """Structural HEADING and LIST_ITEM blocks should never be blanked as noise."""
    from ubt.core.ir.models import BlockType, FlowID, IRBlock

    cleaner = LNDSPageCleaner(strip_all_page_numbers=True)
    blocks = [
        IRBlock(
            id="h1",
            flow_id=FlowID.MAIN_STORY,
            spine_index=1,
            block_type=BlockType.HEADING,
            source_text="1",
        ),
        IRBlock(
            id="l1",
            flow_id=FlowID.MAIN_STORY,
            spine_index=2,
            block_type=BlockType.LIST_ITEM,
            source_text="First item",
        ),
    ]
    cleaned = cleaner.clean_chapter_blocks(blocks)
    assert len(cleaned) == 2
    assert cleaned[0].source_text == "1"
    assert cleaned[0].skip_translate is False
    assert cleaned[1].source_text == "First item"


def test_html_sanitizer_preserves_math_svg_and_img() -> None:
    """Validate that sanitize_html_fragment preserves math, svg, and img tags with safe attributes."""
    from ubt.core.cleaners.html_sanitizer import sanitize_html_fragment

    fragment = (
        '<p>Equation: <math display="block"><mrow><mi>E</mi><mo>=</mo><mi>m</mi><msup><mi>c</mi><mn>2</mn></msup></mrow></math></p>'
        '<svg width="100" height="100" viewBox="0 0 100 100"><circle cx="50" cy="50" r="40" stroke="green" fill="yellow" /></svg>'
        '<img src="images/fig1.png" alt="Figure 1" width="300" height="200" />'
        '<a href="https://example.com" title="Example">Link</a>'
        '<script>alert("XSS")</script>'
    )
    cleaned = sanitize_html_fragment(fragment)
    # MathML preserved
    assert "<math" in cleaned and "<mi>E</mi>" in cleaned and "</math>" in cleaned
    # SVG preserved
    assert "<svg" in cleaned and "<circle" in cleaned and "</svg>" in cleaned
    # Image preserved
    assert '<img src="images/fig1.png"' in cleaned and 'alt="Figure 1"' in cleaned
    # Script dropped with its content
    assert "<script" not in cleaned
    assert 'alert("XSS")' not in cleaned


def test_html_sanitizer_preserves_table_and_vector_attributes() -> None:
    """Validate that table merge attributes and SVG vector coordinates are preserved."""
    from ubt.core.cleaners.html_sanitizer import sanitize_html_fragment

    table_html = (
        '<table border="1" class="tbl">'
        '<thead><tr><th colspan="2" scope="col">Header</th></tr></thead>'
        '<tbody><tr><td rowspan="3" class="cell">Data</td><td>Sub</td></tr></tbody>'
        "</table>"
    )
    cleaned_tbl = sanitize_html_fragment(table_html)
    assert 'colspan="2"' in cleaned_tbl
    assert 'rowspan="3"' in cleaned_tbl
    assert 'scope="col"' in cleaned_tbl

    svg_html = (
        '<svg viewBox="0 0 100 100" width="100" height="100">'
        '<g transform="translate(10, 20)">'
        '<polygon points="0,0 10,10 0,20" fill="red" />'
        '<text x="15" y="25" font-size="12">Label</text>'
        "</g>"
        "</svg>"
    )
    cleaned_svg = sanitize_html_fragment(svg_html)
    assert 'points="0,0 10,10 0,20"' in cleaned_svg
    assert 'x="15"' in cleaned_svg
    assert 'y="25"' in cleaned_svg
    assert 'transform="translate(10, 20)"' in cleaned_svg


def test_html_sanitizer_never_reconstitutes_markup_from_entities() -> None:
    """Entity-encoded (and double-encoded) tags must stay inert text (XSS)."""
    from ubt.core.cleaners.html_sanitizer import sanitize_html_fragment

    encoded = sanitize_html_fragment("&lt;script&gt;alert(1)&lt;/script&gt;")
    assert "<script" not in encoded
    assert "&lt;script&gt;" in encoded

    double = sanitize_html_fragment("&amp;lt;script&amp;gt;alert(1)&amp;lt;/script&amp;gt;")
    assert "<script" not in double
    assert "&amp;lt;script&amp;gt;" in double

    # A bare ampersand is escaped (renders identically) and is idempotent.
    assert sanitize_html_fragment("AT&T") == "AT&amp;T"
    assert sanitize_html_fragment(sanitize_html_fragment("AT&T")) == "AT&amp;T"


def test_normalize_academic_pdf_math_cmap_and_subscripts() -> None:
    from ubt.core.cleaners.lnds_pruner import normalize_academic_pdf_math

    raw = "N ch ¼ 1 - 10 15 cm \x00 3 , TFIN ¼ 20 nm, t ox ¼ 1 nm, and V ch ¼ 0 V"
    norm = normalize_academic_pdf_math(raw)
    assert "N_ch = 1 × 10^15 cm^-3" in norm
    assert "TFIN = 20 nm" in norm
    assert "t_ox = 1 nm" in norm
    assert "V_ch = 0 V" in norm
    assert "\x00" not in norm
    assert "¼" not in norm

    # Ordinary vulgar fraction in non-equation context preserved
    assert normalize_academic_pdf_math("add ¼ cup of sugar") == "add ¼ cup of sugar"

    # P0: block-global ¼ evidence used to rewrite currency in the same block
    # ("paid $5" -> "paid -5"). Evidence now requires equation-shaped ¼.
    assert (
        normalize_academic_pdf_math("Add ¼ cup sugar, paid $5 for milk")
        == "Add ¼ cup sugar, paid $5 for milk"
    )
    # Genuine CMap corruption still repaired: equation ¼ plus broken minus-$.
    assert normalize_academic_pdf_math("Vch ¼ Vs, I $5 mA") == "Vch = Vs, I -5 mA"


def test_find_soup_spans_protects_physical_parameters() -> None:
    from ubt.core.cleaners.soup_math import SoupMathMasker, find_soup_spans

    text = "Model parameters: N_ch = 1 × 10^15 cm^-3, TFIN = 20 nm, t_ox = 1 nm, and V_ch = 0 V."
    spans = find_soup_spans(text)
    matched_texts = [text[s:e] for s, e in spans]
    assert "N_ch = 1 × 10^15 cm^-3" in matched_texts
    assert "TFIN = 20 nm" in matched_texts
    assert "t_ox = 1 nm" in matched_texts
    assert "V_ch = 0 V" in matched_texts

    # Check that mask / unmask roundtrips cleanly without losing equations
    masker = SoupMathMasker()
    masked, mapping = masker.mask(text)
    assert len(mapping) == 4
    for orig in ["N_ch = 1 × 10^15 cm^-3", "TFIN = 20 nm", "t_ox = 1 nm", "V_ch = 0 V"]:
        assert orig not in masked
    restored = masker.unmask(masked, mapping)
    assert restored == text


def test_code_masker_fuzzy_unmasking() -> None:
    """Fix 7: Verify CodeMasker restores code even if LLM mutates brackets, spacing, or casing."""
    masker = CodeMasker()
    original_text = (
        "Check this: `foo()` and `bar()` and\n```python\ndef test(): pass\n```\nand `baz()`"
    )
    masked, mapping = masker.mask(original_text)
    assert len(mapping) == 4

    # Simulate varied LLM outputs: standard square brackets, extra whitespace, Chinese brackets, lowercase
    mutated_translation = (
        "查看这个: [CODE_MASK_0001] 以及 ⟦ CODE_MASK_0002 ⟧ 以及\n"
        "【CODE_MASK_0003】\n"
        "和 ⟦code_mask_0004⟧"
    )

    unmasked = masker.unmask(mutated_translation, mapping)
    assert "`foo()`" in unmasked
    assert "`bar()`" in unmasked
    assert "def test(): pass" in unmasked
    assert "`baz()`" in unmasked
    assert "CODE_MASK" not in unmasked


def test_code_masker_namespaced_prefix_fuzzy_unmask() -> None:
    """Namespaced token prefix ⟦UBT:CODE:0001-...⟧ must fuzzy-unmask when mutated."""
    masker = CodeMasker(mask_prefix="⟦UBT:CODE:")
    text = "Use `uv run pytest` to run tests."
    masked, mapping = masker.mask(text)
    assert len(mapping) == 1
    tok = next(iter(mapping))
    mutated = tok.replace("⟦", "[").replace("⟧", "]")
    draft = f"运行 {mutated} 进行测试。"
    unmasked = masker.unmask(draft, mapping)
    assert "`uv run pytest`" in unmasked
    assert "[" not in unmasked and "UBT:CODE" not in unmasked


def test_lnds_cleaner_cross_block_monotonic_page_numbers() -> None:
    """Fix 3: Verify LNDSPageCleaner detects monotonic page numbers split across multiple IRBlocks."""
    cleaner = LNDSPageCleaner()

    blocks = [
        IRBlock(id="b1", spine_index=1, source_text="Chapter 1 Opening scene."),
        IRBlock(id="b2", spine_index=2, source_text="1"),  # Page 1
        IRBlock(id="b3", spine_index=3, source_text="Next narrative paragraph."),
        IRBlock(id="b4", spine_index=4, source_text="2"),  # Page 2
        IRBlock(id="b5", spine_index=5, source_text="More story narrative in the year 1984."),
        IRBlock(id="b6", spine_index=6, source_text="3"),  # Page 3
        IRBlock(id="b7", spine_index=7, source_text="Dialogue continuation."),
        IRBlock(id="b8", spine_index=8, source_text="4"),  # Page 4
        IRBlock(id="b9", spine_index=9, source_text="Final thoughts of the chapter."),
        IRBlock(id="b10", spine_index=10, source_text="5"),  # Page 5
    ]

    cleaned_blocks = cleaner.clean_chapter_blocks(blocks)
    assert len(cleaned_blocks) == 10

    # Page number blocks (b2, b4, b6, b8, b10) must be stripped / marked skip
    for idx in [1, 3, 5, 7, 9]:
        assert cleaned_blocks[idx].source_text == ""
        assert cleaned_blocks[idx].skip_translate is True

    # Real story and year 1984 must be strictly preserved
    assert "Chapter 1 Opening scene." in cleaned_blocks[0].source_text
    assert "1984" in cleaned_blocks[4].source_text
    assert cleaned_blocks[4].skip_translate is False


def _bbox(page: int, x0: float, y0: float, x1: float, y1: float) -> BoundingBox:
    return BoundingBox(page=page, x0=x0, y0=y0, x1=x1, y1=y1)


def _narr(id_: str, spine: int, text: str, bbox: BoundingBox) -> IRBlock:
    from ubt.core.ir.models import BlockType, FlowID

    return IRBlock(
        id=id_,
        flow_id=FlowID.MAIN_STORY,
        spine_index=spine,
        block_type=BlockType.NARRATIVE,
        source_text=text,
        bbox=bbox,
    )


def test_dedup_drops_identical_figure_fragments_keeps_larger_box() -> None:
    """Docling's double-extraction of a figure sub-label (arXiv 2609.20519
    page 1) must collapse to one block so the visual gate stops flagging a
    phantom block_overlap. The larger (more complete) box wins."""
    dup = [
        _narr(
            "b1", 1, "(a) SoL-Pi: Scaling Auto-Research Loop", _bbox(1, 68.5, 460.4, 211.9, 468.7)
        ),
        _narr(
            "b2", 2, "(a) SoL-Pi: Scaling Auto-Research Loop", _bbox(1, 68.5, 460.0, 202.8, 467.9)
        ),
    ]
    kept = dedup_duplicate_blocks(dup)
    assert [b.id for b in kept] == ["b1"]  # b1 has the larger area


def test_dedup_keeps_overlapping_blocks_with_different_text() -> None:
    """A caption sitting over an image region overlaps geometrically but is
    distinct content — the normalized-text guard must spare it."""
    pair = [
        _narr("cap", 1, "Figure 1 results", _bbox(1, 0, 0, 100, 40)),
        _narr("body", 2, "Some other prose here", _bbox(1, 0, 5, 100, 45)),
    ]
    assert len(dedup_duplicate_blocks(pair)) == 2


def test_dedup_spares_same_text_in_different_places() -> None:
    """A running head repeated on many pages is not a physical duplicate:
    same normalized text, disjoint boxes, so both survive."""
    pair = [
        _narr("p1", 1, "SoL-Pi", _bbox(1, 0, 0, 40, 10)),
        _narr("p2", 2, "SoL-Pi", _bbox(2, 0, 0, 40, 10)),
    ]
    assert len(dedup_duplicate_blocks(pair)) == 2


def test_kerning_subscript_rule_does_not_corrupt_english_prose() -> None:
    from ubt.core.cleaners.lnds_pruner import strip_textbook_ocr_artifacts

    assert strip_textbook_ocr_artifacts("I am, therefore I think.") == "I am, therefore I think."
    assert (
        strip_textbook_ocr_artifacts("He is a lot, more than before.")
        == "He is a lot, more than before."
    )
    # A genuine flattened subscript before a math operator is still joined.
    assert "x_i" in strip_textbook_ocr_artifacts("The term x i = 3 appears.")


@pytest.mark.fast
def test_ordered_list_numbers_after_an_intro_colon_survive() -> None:
    from ubt.core.cleaners.lnds_pruner import collect_dropped_line_indices

    lines = [
        "Steps:",
        "1",
        "Install the package.",
        "2",
        "Run the tests.",
        "3",
        "Deploy.",
        "4",
        "Verify.",
    ]
    assert collect_dropped_line_indices(lines) == set()


@pytest.mark.fast
def test_normalize_academic_pdf_math_heals_soft_hyphen_word_splits() -> None:
    """Soft hyphens followed by spaces ('transfor\\xad mation', 'compos\\xad ability')
    must be rejoined into whole words."""
    raw = "spatiotemporal compos\xad ability and transfor\xad mation in orches\xad trate"
    cleaned = normalize_academic_pdf_math(raw)
    assert cleaned == "spatiotemporal composability and transformation in orchestrate"


def _a0920_html_leaves(markup: str) -> list[str]:
    from bs4 import BeautifulSoup

    adapter = HTMLAdapter()
    soup = BeautifulSoup(markup, "html.parser")
    return [f"{tag.name}:{tag.get_text(' ', strip=True)}" for tag in adapter._leaf_blocks(soup)]


def test_wrapper_div_is_not_mined_alongside_its_code_block() -> None:
    """A ``<div>`` wrapping ``<pre>`` used to look like a leaf.

    ``is_leaf_block`` searched a hand-written descendant list that omitted
    ``pre``/headings/cells, so the same code text was mined twice: once as
    NARRATIVE (sent to the model and injected into the finished book) and once
    as CODE (verbatim). ``<p>`` nesting already excluded the parent, so the fix
    is one vocabulary for both questions, not a new rule.
    """
    leaves = _a0920_html_leaves("<div><pre>int x = 1;</pre></div>")
    assert leaves == ["pre:int x = 1;"], leaves

    # Headings and definition lists behave the same way (both are BLOCK_TAGS).
    assert _a0920_html_leaves("<div><h2>Results</h2></div>") == ["h2:Results"]
    assert _a0920_html_leaves("<div><dl><dt>Term</dt><dd>Body</dd></dl></div>") == [
        "dt:Term",
        "dd:Body",
    ]


def test_nested_check_uses_the_same_vocabulary_as_the_mining_set() -> None:
    """The leaf predicate and the mining set may never disagree again."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(
        "<div><pre>x</pre><h1>t</h1><table><tr><td>c</td></tr></table></div>", "html.parser"
    )
    block_names = set(BLOCK_TAGS)
    wrapper = soup.find("div")
    assert wrapper is not None
    assert not is_leaf_block(wrapper, block_names)


_r0918_LEAD_INS = [
    ("Translation: 从前有一座山。", "从前有一座山。"),
    ("这是中文翻译：从前有一座山。", "从前有一座山。"),
    ("以下是最终精修中文翻译：从前有一座山。", "从前有一座山。"),
    ("Here is the final translation: Once upon a time.", "Once upon a time."),
]

_r0918_MUST_SURVIVE = [
    "翻译过程中的注意事项：见下文。",
    "Translations of the term appear below.",
    "翻译如下所示。这是一段完整的译文。",
]


@pytest.mark.parametrize(("raw", "expected"), _r0918_LEAD_INS)
def test_conversational_lead_in_is_still_stripped(raw: str, expected: str) -> None:
    assert TranslationOutputExtractor.extract(raw) == expected


@pytest.mark.parametrize("text", _r0918_MUST_SURVIVE)
def test_prose_that_only_resembles_a_lead_in_is_untouched(text: str) -> None:
    # Pre-fix, the optional colon let these match and lose their first words.
    assert TranslationOutputExtractor.extract(text) == text


def test_leading_blockquote_marker_is_preserved() -> None:
    assert TranslationOutputExtractor.extract("> 引用的原文块。") == "> 引用的原文块。"
