"""Tests for dynamic boilerplate harvester, script family transitions, and heuristic QE runner."""

import pytest

from ubt.core.cleaners.dynamic_boilerplate import (
    BoilerplateFingerprint,
    DynamicBoilerplateHarvester,
)
from ubt.core.language_profile import (
    get_pair_policy,
    resolve_script_family_bounds,
)
from ubt.core.qe.comet_runner import HeuristicQERunner


def test_dynamic_boilerplate_consensus_harvesting() -> None:
    disclaimer = (
        "Copyright 2024 Academic Publisher. All Rights Reserved. "
        "May not be copied, scanned, or duplicated, in whole or in part."
    )
    pages = [
        f"Chapter 1: The introduction to deep learning and neural networks. {disclaimer}",
        f"Chapter 2: Optimization techniques, gradient descent, and Adam. {disclaimer}",
        f"Chapter 3: Convolutional networks and spatial invariance principles. {disclaimer}",
        f"Chapter 4: Recurrent networks and sequential processing. {disclaimer}",
        f"Chapter 5: Transformer architectures and attention mechanisms. {disclaimer}",
    ]

    harvester = DynamicBoilerplateHarvester(min_sample_length=30, min_match_size=30)
    fp = harvester.harvest(pages)

    assert len(fp.footer_disclaimers) >= 1
    assert "Copyright 2024 Academic Publisher" in fp.footer_disclaimers[0]

    # Test cleaning on dirty page
    dirty_text = f"Page 10\nCHAPTER 1 10 Deep Learning Overview. Neural nets work well. Bettmann/Corbis {disclaimer}"
    cleaned = fp.clean(dirty_text)
    assert disclaimer not in cleaned
    assert "Bettmann/Corbis" not in cleaned
    assert "Page 10" not in cleaned
    assert "Deep Learning Overview. Neural nets work well." in cleaned


def test_boilerplate_fingerprint_safe_on_clean_text() -> None:
    fp = BoilerplateFingerprint(footer_disclaimers=("Random Publisher All Rights Reserved",))
    clean_text = "This is an untainted authorial sentence explaining cognitive memory."
    assert fp.clean(clean_text) == clean_text


def test_header_patterns_reserved_and_clean_head_uses_fixed_patterns() -> None:
    """Consensus header harvesting is not implemented; clean_head uses fixed regexes."""
    harvester = DynamicBoilerplateHarvester(min_sample_length=30, min_match_size=30)
    fp = harvester.harvest(
        [
            "Chapter one " + "alpha " * 40,
            "Chapter two " + "beta " * 40,
            "Chapter three " + "gamma " * 40,
        ]
    )
    assert fp.header_patterns == ()

    cleaned, header = fp.clean_head("Page 10\n正文内容")
    assert cleaned == "正文内容"
    assert "Page 10" in header


def test_script_family_transitions() -> None:
    # Latin to CJK (contractive)
    bounds_it_zh = resolve_script_family_bounds("it", "zh", (0.2, 4.0))
    assert bounds_it_zh == (0.18, 1.8)

    # CJK to Latin (expansive)
    bounds_zh_fr = resolve_script_family_bounds("zh", "fr", (0.2, 4.0))
    assert bounds_zh_fr == (0.8, 5.0)

    # Latin to Latin (stable)
    bounds_fr_de = resolve_script_family_bounds("fr", "de", (0.2, 4.0))
    assert bounds_fr_de == (0.5, 2.0)

    policy = get_pair_policy("it", "zh")
    assert policy.min_length_ratio == 0.18
    assert policy.max_length_ratio == 1.8
    assert policy.min_target_ratio == 0.25  # CJK script density active


@pytest.mark.asyncio
async def test_heuristic_qe_runner_scoring() -> None:
    qe = HeuristicQERunner(target_lang="zh", source_lang="en")

    # 1. Flawless translation
    scores = await qe.score_pairs(
        [{"src": "Deep learning models require data.", "mt": "深度学习模型需要数据。"}]
    )
    assert scores[0] >= 0.90

    # 2. Leaked prompt artifact
    scores = await qe.score_pairs([{"src": "Hello world", "mt": "<issues>None</issues> 你好世界"}])
    assert scores[0] <= 0.20

    # 3. Broken HTML tags (unescaped quotes corrupting img attribute)
    scores = await qe.score_pairs(
        [
            {
                "src": '<img src="cat.jpg" alt="A cute cat"/>',
                "mt": '<img src="cat.jpg" alt="一只可爱的"小猫""/>',
            }
        ]
    )
    assert scores[0] <= 0.35

    # 4. Corrupted numbers
    scores = await qe.score_pairs(
        [{"src": "In 1998, 45 patients were treated.", "mt": "在当年，许多患者得到了治疗。"}]
    )
    assert scores[0] <= 0.60

    # 5. Untranslated prose residue
    scores = await qe.score_pairs(
        [
            {
                "src": "This is completely untranslated prose.",
                "mt": "This is completely untranslated prose.",
            }
        ]
    )
    assert scores[0] <= 0.20


def test_page_marker_cleaning_never_eats_a_paragraphs_own_leading_number() -> None:
    """The marker is a *line*, not a prefix of content.

    The EPUB adapter applies this cleaner per block, so the unanchored pattern
    deleted the list number from "3. Preheat the oven…", the year from "1812
    was…", and the numeral from "IV. On the Origin of Species" — and the
    damaged text was written into the delivered XHTML.
    """
    fp = BoilerplateFingerprint()
    for text in (
        "3. Preheat the oven and mix the dry flour into the wet.",
        "1812 was the year of the treaty.",
        "IV. On the Origin of Species",
    ):
        assert fp.clean(text) == text, text
    # A real marker still goes.
    assert fp.clean("42\nChapter title") == "Chapter title"
    assert fp.clean("Page 42\nBody text") == "Body text"
    # A block that is only a page number is left alone (nothing to strip to).
    assert fp.clean("233") == "233"


def test_leading_page_marker_does_not_eat_english_words() -> None:
    from ubt.core.cleaners.dynamic_boilerplate import BoilerplateFingerprint

    fp = BoilerplateFingerprint()
    for word in ("Mild", "Civil", "Dim", "Mix"):
        text = f"{word}\nBody sentence follows here."
        assert fp.clean_head(text)[0] == text, word
    # Real page markers are still stripped.
    assert fp.clean_head("xiv\nChapter body")[0] == "Chapter body"
    assert fp.clean_head("Page 42\nBody")[0] == "Body"


def test_clean_head_keeps_a_leading_year() -> None:
    """A leading year is content, not a page marker.

    Regression: the bare-number page-marker alternative matched a 4-digit year,
    so a chapter headed "2024" or a dated front-matter line lost its first line.
    """
    from ubt.core.cleaners.dynamic_boilerplate import BoilerplateFingerprint

    cleaner = BoilerplateFingerprint()
    kept, _ = cleaner.clean_head("2024\nAnnual Report of the Society")
    assert kept.startswith("2024")
    # A real page number (and a non-year 4-digit number) is still stripped.
    for page in ("42\nAnnual Report of the Society", "1234\nAnnual Report of the Society"):
        cleaned, _ = cleaner.clean_head(page)
        assert cleaned.startswith("Annual Report"), (page, cleaned)


def test_footer_disclaimer_prefix_in_body_prose_is_not_stripped() -> None:
    """A sentence that merely opens with the disclaimer's first words is prose.

    Regression: the "anchor on leading 30 chars" branch stripped from a bare
    prefix match, so a paragraph mentioning the copyright line had its tail
    deleted and written back into the delivered file.
    """
    from ubt.core.cleaners.dynamic_boilerplate import BoilerplateFingerprint

    disc = "Copyright 2024 Academic Publisher. All Rights Reserved. May not be copied."
    fp = BoilerplateFingerprint(footer_disclaimers=(disc,))

    body = "Please read the Copyright 2024 Academic Publisher guidelines before continuing."
    cleaned, stripped = fp.clean_tail(body)
    assert cleaned == body, cleaned
    assert stripped == ""

    # A genuine disclaimer whose tail varies is still stripped.
    variant = disc.replace("copied.", "copied!")
    cleaned2, stripped2 = fp.clean_tail(f"Some prose text. {variant}")
    assert cleaned2 == "Some prose text", cleaned2
    assert variant in stripped2
