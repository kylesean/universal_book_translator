from __future__ import annotations

import time

import pytest

from ubt.core.cleaners.skip_rules import (
    _is_author_byline,
    _is_bibliographic_identity,
    classify_skip,
)

pytestmark = pytest.mark.fast

HANDLE = "social-handle watermark / running-head chrome"
BIB = "bibliography entry (kept verbatim for retrievability)"
BYLINE = "author byline (foreign names kept in romanization per GB/T 7714)"
IDENTITY = "bibliographic identity (affiliation/legend kept verbatim)"
DEBRIS = "pure symbol/numeric debris (no translatable words)"


@pytest.mark.parametrize("text", ["", "   ", "\n\t ", "\u00ad"])
def test_blank_or_whitespace_is_never_skipped(text: str) -> None:
    assert classify_skip(text) is None


@pytest.mark.parametrize(
    "text",
    ["@techNmak 2024 follow", "@a.b-c_1 hello", "@handle"],
)
def test_social_handle_at_block_start_is_skipped(text: str) -> None:
    assert classify_skip(text) == HANDLE


def test_handle_soft_hyphen_is_stripped_before_matching() -> None:
    assert classify_skip("@techNmak\u00ad stuff") == HANDLE


def test_handle_only_matches_at_the_start_of_the_block() -> None:
    assert classify_skip("see @techNmak for details") is None


@pytest.mark.parametrize(
    "text",
    [
        "arXiv:2609.20519",
        "see arxiv:1234.5678",
        "We propose a method, see arXiv:2609.20519 for details.",
    ],
)
def test_arxiv_identifier_is_always_skipped(text: str) -> None:
    assert classify_skip(text) == BIB


@pytest.mark.parametrize(
    "text",
    ["[12] A. Author, Journal of Things, 2019", '[12] "A Quoted Title", 2019'],
)
def test_bracket_reference_with_year_is_skipped(text: str) -> None:
    assert classify_skip(text) == BIB


@pytest.mark.parametrize(
    "text",
    [
        "[3] yield the following results in 2019",
        "[12] A. Author, Journal",
        "[12] We propose a method in 2019.",
    ],
)
def test_bracket_reference_requires_uppercase_start_year_and_no_prose(text: str) -> None:
    assert classify_skip(text) is None


def test_bibliography_flag_skips_non_narrative_block() -> None:
    assert classify_skip("Some line without narrative", in_bibliography=True) == BIB


def test_bibliography_flag_defers_narrative_prose_to_the_llm() -> None:
    assert classify_skip("we propose a new method", in_bibliography=True) is None


def test_bibliography_flag_defers_headings() -> None:
    assert classify_skip("Purpose, Scope", in_bibliography=True, is_heading=True) is None


@pytest.mark.parametrize(
    "text",
    [
        "Foo. [Online]. Available: https://example.com/x",
        'J. Smith, "A Great Paper". 2019.',
        "See the user guide consulted 2019 for details",
        "Smith et al. IEEE, 2019",
        "J. Doe. https://example.com/paper, NeurIPS 2020.",
        "J. Smith, A. B. Jones. IEEE Access, 2019.",
        "J. Smith, Lett. 21 (5), 245-247, 2019.",
        "J. Smith. IEEE, 245-247, 2019.",
        "J. Smith. MIT Press, 2019.",
        "John Yang, Carlos Jimenez. Springer, pages 45-60, 2019.",
        "In NeurIPS, volume 5, 2019.",
        "J. Smith. doi:10.1000/xyz, 2019.",
        "J. Smith. In Proceedings of ACL, 2019.",
    ],
)
def test_label_free_bibliography_entries_are_skipped(text: str) -> None:
    assert classify_skip(text) == BIB


@pytest.mark.parametrize(
    "text",
    [
        "J. Smith, Lett. 21 (5), 245-247.",
        "In NeurIPS, pages 1-10",
        "https://example.com/paper",
        "This was published by Springer in 2019, spanning pages 45-60.",
        "Springer, pages 45-60, 2019.",
    ],
)
def test_bibliography_near_misses_stay_translatable(text: str) -> None:
    assert classify_skip(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "We propose a method. J. Smith. IEEE, 2019.",
        "This paper cites J. Smith. IEEE, 2019.",
        "Our method beats J. Smith. IEEE, 2019.",
    ],
)
def test_narrative_prose_overrides_a_bibliography_shape(text: str) -> None:
    assert classify_skip(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "The tool was released in 2024; see https://example.com/docs for details.",
        "See the docs at https://example.com/guide published in 2021.",
        "The 2024 survey (https://example.org) covers this well.",
        "In 2023 we published our code at https://github.com/foo/bar.",
        "Our implementation is available at https://github.com/foo/bar since 2023.",
    ],
)
def test_prose_containing_a_url_and_year_is_not_a_bibliography_entry(text: str) -> None:
    assert classify_skip(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "Wang, Y. (2017). Some title. Retrieved from https://example.com/paper.pdf",
        "Y. Zhang, L. Wang. A study. https://arxiv.org/abs/2001.00001",
        "Foo. [Online]. Available: https://example.com/x",
        "John Yang, Carlos Jimenez. SWE-bench. 2024. https://swebench.com",
    ],
)
def test_real_url_bearing_references_are_still_skipped(text: str) -> None:
    assert classify_skip(text) == BIB


@pytest.mark.parametrize(
    "text",
    [
        "This **important** step and **that** step must be followed.",
        "Use **foo** and **bar** to configure it.",
        "The **first** and **second** options are best.",
    ],
)
def test_markdown_emphasis_is_not_a_bibliographic_identity(text: str) -> None:
    assert classify_skip(text) is None


def test_inline_citation_prose_is_not_a_bibliography_entry() -> None:
    assert (
        classify_skip(
            "A continual-learning agent is a system comprising a memory layer that "
            "retains accumulated experience (Wang et al., 2023; Park et al., 2023), "
            "and model training that refines its behavior across sessions."
        )
        is None
    )


def test_et_al_with_an_independent_author_signal_still_skips() -> None:
    assert classify_skip("A. Wang et al. 2019") == BIB
    assert classify_skip("Smith et al. IEEE, 2019") == BIB


@pytest.mark.parametrize(
    "text",
    [
        "Duomin Wang, Jane Doe",
        "Duomin Wang, Jane Doe, John Smith",
        "Duomin Wang and Jane Doe",
        "Duomin Wang 1, Jane Doe 2",
        "Li\u00b9\u00b2 Zhang, Wang\u00b3 Li",
        "Yifan Shi\u00b9\u02d2\u00b2 , Wei Zhang\u00b9 , Tianyi Cui\u00b2",
        "John Smith Jr., Jane Doe",
        "Duomin Wang, Jane Doe,",
        "Duomin Wang\u2020, Jane Doe\u2021",
        "YoungmokJung , Sirajul Salekin , Henry Tran",
        "Duomin Wang, Y.C. Yan, Jane Doe",
    ],
)
def test_author_byline_is_skipped(text: str) -> None:
    assert classify_skip(text) == BYLINE


@pytest.mark.parametrize(
    "text",
    [
        "Duomin Wang",
        "Smith, Jones",
        "J. Smith, A. Jones",
        "Purpose, Scope, and Audience",
        "Duomin Wang, Scope",
        "Duomin Wang, Jane Doe 2020",
        "duomin wang, jane doe",
    ],
)
def test_byline_near_misses_stay_translatable(text: str) -> None:
    assert classify_skip(text) is None


def test_heading_is_never_a_byline() -> None:
    assert classify_skip("Duomin Wang, Jane Doe", is_heading=True) is None


def test_cjk_block_is_never_a_byline() -> None:
    assert classify_skip("\u738b\u591a\u6c11, \u674e\u56db") is None


def test_byline_scan_is_linear_on_an_adversarial_tail() -> None:
    text = "Aa Aa, Bb " + "1" * 200 + "!"
    started = time.perf_counter()
    result = _is_author_byline(text)
    elapsed = time.perf_counter() - started
    assert result is False
    assert elapsed < 1.0


@pytest.mark.parametrize(
    "text",
    [
        "DeepSeek-AI \u2021 Tsinghua University research@deepseek.com",
        "\u2217 Corresponding author. \u2020 DSec project developers. \u2021 Tsinghua University.",
        "DeepSeek-AI, Tsinghua University, research@deepseek.com",
        "\u2217 Equal contribution. \u2020 Work done while at DeepSeek-AI.",
        "* Equal contribution. \u2020 Corresponding author.",
    ],
)
def test_bibliographic_identity_is_skipped(text: str) -> None:
    assert classify_skip(text) == IDENTITY


@pytest.mark.parametrize(
    "text",
    [
        "The dataset was released by Tsinghua University.",
        "Contact us at research@deepseek.com for access to the dataset.",
        "The dataset was released by Tsinghua University. \u2020 "
        + "This section describes how we collected the corpus and cleaned it. " * 4,
        "The product a*b is bounded, and Stanford University publishes the report.",
        "\u6e05\u534e\u5927\u5b66 \u2021 research@deepseek.com",
        "DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for "
        "Effective Agentic Training at Scale",
    ],
)
def test_identity_near_misses_stay_translatable(text: str) -> None:
    assert classify_skip(text) is None


def test_heading_is_never_bibliographic_identity() -> None:
    text = "Affiliations \u2020 Tsinghua University"
    assert classify_skip(text, is_heading=True) is None
    assert classify_skip(text) == IDENTITY


def test_identity_word_cap_boundary() -> None:
    assert _is_bibliographic_identity("Tsinghua University \u2020 " + "word " * 35) is True
    assert _is_bibliographic_identity("Tsinghua University \u2020 " + "word " * 40) is False


@pytest.mark.parametrize(
    "text",
    ["123 --- 456 +++ ==", "K / V / x", "1 2 3 4 5", "\u2014\u2014\u2014"],
)
def test_symbol_or_numeric_debris_is_skipped(text: str) -> None:
    assert classify_skip(text) == DEBRIS


@pytest.mark.parametrize(
    "text",
    [
        "ab",
        "--- ab ---",
        "\u4e2d\u6587 --- 123",
        "x \u4e2d\u6587",
    ],
)
def test_debris_rule_does_not_fire_on_real_content(text: str) -> None:
    assert classify_skip(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "This is an ordinary sentence.",
        "\u8fd9\u662f\u4e00\u53e5\u666e\u901a\u7684\u4e2d\u6587\u3002",
        "Research anchors: [1], [2]",
    ],
)
def test_ordinary_prose_is_translated(text: str) -> None:
    assert classify_skip(text) is None


def test_narrative_with_in_text_citations_is_not_bib_entry() -> None:
    p1 = (
        "The agentic training pipeline encompasses environment and data construction, "
        "RL rollouts, reward computation, policy updates, and periodic evaluation. "
        "In RL (Guo et al., 2025; Ouyang et al., 2022), training proceeds as a feedback "
        "loop with three stages."
    )
    p2 = (
        "Base image and workspace storage. The sandbox runtime uses 3FS as a shared backing store "
        "for base images and workspace images. MicroVM disk images use an OverlayBD (Li et al., 2020) "
        "format over the same storage."
    )
    p3 = (
        "Existing on-demand image distribution systems often combine a container registry with "
        "peer-to-peer delivery to prevent the registry from becoming a bottleneck (Wang et al., 2021). "
        "However, 3FS exhibits highly asymmetric I/O. This asymmetry dictates our design."
    )
    p4 = (
        "3FS deployment. Each 3FS (An et al., 2024) storage server is equipped with 20 × 15TB SSDs "
        "and 2 × 400Gbps RDMA NICs. CPU nodes access 3FS through its FUSE-based client."
    )
    assert classify_skip(p1) is None
    assert classify_skip(p2) is None
    assert classify_skip(p3) is None
    assert classify_skip(p4) is None
