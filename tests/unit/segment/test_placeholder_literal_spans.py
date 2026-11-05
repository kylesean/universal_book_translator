from __future__ import annotations

import pytest

from ubt.segment.placeholders import default_placeholder_engine

pytestmark = pytest.mark.fast


def test_dropped_email_and_url_are_reported_missing() -> None:
    engine = default_placeholder_engine()
    source = "Contact alice@example.com or visit https://example.org/release."

    missing = engine.missing_literal_spans(source, "请联系获取发布信息。")

    assert ("email", "alice@example.com") in missing
    assert ("email", "https://example.org/release") in missing


def test_kept_literal_spans_are_not_reported() -> None:
    engine = default_placeholder_engine()
    source = "Contact alice@example.com or visit https://example.org/release."
    target = "请联系 alice@example.com 或访问 https://example.org/release。"

    assert engine.missing_literal_spans(source, target) == ()


def test_lost_code_span_is_reported() -> None:
    engine = default_placeholder_engine()
    source = "Call `build_index()` before parsing."
    target = "解析前调用 build_index()。"

    missing = engine.missing_literal_spans(source, target)

    assert ("code", "`build_index()`") in missing


def test_lost_soup_span_is_reported() -> None:
    engine = default_placeholder_engine()
    source = "The term e−ψ appears."
    assert engine.mask(source).soup_map, "the fixture must actually carry a soup span"

    missing = engine.missing_literal_spans(source, "该项出现。")

    assert ("soup", "e−ψ") in missing


def test_citations_are_counted_not_value_matched() -> None:
    engine = default_placeholder_engine()
    source = "See [1] and [2]."
    assert engine.mask(source).cite_map

    # Renumbering is legitimate.
    assert engine.missing_literal_spans(source, "见 [3] 与 [4]。") == ()

    # A net deletion is not.
    missing = engine.missing_literal_spans(source, "见 [3]。")
    assert [label for label, _ in missing] == ["citation"]


def test_adjacent_citations_merged_into_one_group_are_not_a_loss() -> None:
    engine = default_placeholder_engine()
    source = "See [1][2] for the trial."

    assert engine.missing_literal_spans(source, "见[1,2]关于该试验。") == ()
    assert engine.missing_literal_spans("See [1,2] for it.", "见[1][2]关于它。") == ()


def test_a_fullwidth_localized_citation_is_not_a_loss() -> None:
    engine = default_placeholder_engine()
    source = "See (Smith, 2020) for the review."

    assert engine.missing_literal_spans(source, "见（Smith，2020）关于该综述。") == ()


def test_net_deletion_of_a_citation_group_is_still_reported() -> None:
    engine = default_placeholder_engine()
    source = "See [1][2] for the trial."

    missing = engine.missing_literal_spans(source, "见[1]关于该试验。")

    assert [label for label, _ in missing] == ["citation"]


def test_renumbering_a_citation_is_not_a_loss() -> None:
    engine = default_placeholder_engine()

    assert engine.missing_literal_spans("See [1] for it.", "见[2]关于它。") == ()


def test_math_and_citations_are_not_literal_span_checks() -> None:
    engine = default_placeholder_engine()
    source = "The value $x_{i}$ is defined in [1]."

    # FastPass owns the math-span comparison; citations are count-checked
    # above, not value-checked, so a same-count renumbering is not a loss.
    assert engine.missing_literal_spans(source, "该值在文献 [2] 中定义。") == ()
