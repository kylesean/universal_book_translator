"""Contract tests for the HTML/Markdown structural-delta validator.

Structural image deltas are hard errors (RETRY); emphasis-tag drift and dropped
anchor hrefs are advisory warnings that must never fail the gate.
"""

from __future__ import annotations

import pytest

from ubt.core.validators.html_delta import (
    _VALID_ATTR_NAME_RE,
    HTMLDeltaValidator,
)

pytestmark = pytest.mark.fast


@pytest.fixture
def validator() -> HTMLDeltaValidator:
    return HTMLDeltaValidator()


def test_identical_plain_text_passes(validator: HTMLDeltaValidator) -> None:
    result = validator.validate("plain text", "纯文本")
    assert result.is_valid
    assert result.details == {}


def test_missing_html_image_is_a_hard_error(validator: HTMLDeltaValidator) -> None:
    result = validator.validate('<img src="a.png">', "no img")
    assert not result.is_valid
    assert result.error_code == "HTML_DELTA_MISMATCH"
    assert result.suggested_action == "RETRY"
    assert result.details["image_diff"]["missing_html"] == [("a.png", 1)]


def test_extra_html_image_is_a_hard_error(validator: HTMLDeltaValidator) -> None:
    result = validator.validate("plain", '<img src="b.png">')
    assert not result.is_valid
    assert result.details["image_diff"]["extra_html"] == [("b.png", 1)]


def test_swapped_html_image_reports_both_sides(validator: HTMLDeltaValidator) -> None:
    result = validator.validate('<img src="a.png">', '<img src="b.png">')
    assert not result.is_valid
    diff = result.details["image_diff"]
    assert diff["missing_html"] == [("a.png", 1)]
    assert diff["extra_html"] == [("b.png", 1)]


def test_duplicated_html_image_counts_per_occurrence(validator: HTMLDeltaValidator) -> None:
    result = validator.validate('<img src="a.png"><img src="a.png">', '<img src="a.png">')
    assert not result.is_valid
    assert result.details["image_diff"]["missing_html"] == [("a.png", 1)]


def test_missing_markdown_image_is_a_hard_error(validator: HTMLDeltaValidator) -> None:
    result = validator.validate("![](a.png)", "plain")
    assert not result.is_valid
    assert result.details["image_diff"]["missing_md"] == [("a.png", 1)]


def test_extra_markdown_image_is_a_hard_error(validator: HTMLDeltaValidator) -> None:
    result = validator.validate("plain", "![](a.png)")
    assert not result.is_valid
    assert result.details["image_diff"]["extra_md"] == [("a.png", 1)]


def test_an_escaped_markdown_image_is_not_counted(validator: HTMLDeltaValidator) -> None:
    result = validator.validate(r"\![](a.png)", "plain")
    assert result.is_valid
    assert result.details == {}


def test_emphasis_drift_is_only_a_warning(validator: HTMLDeltaValidator) -> None:
    dropped = validator.validate("<b>x</b>", "x")
    assert dropped.is_valid
    assert dropped.error_code is None
    assert dropped.details["formatting_warnings"] == [
        "formatting tag drift: <b> x1 in source, x0 in target"
    ]

    added = validator.validate("x", "<em>x</em>")
    assert added.is_valid
    assert added.details["formatting_warnings"] == [
        "formatting tag drift: <em> x0 in source, x1 in target"
    ]


def test_a_dropped_anchor_href_is_only_a_warning(validator: HTMLDeltaValidator) -> None:
    result = validator.validate('<a href="u">x</a>', "x")
    assert result.is_valid
    assert result.details["formatting_warnings"] == ["anchor href dropped: u"]


def test_an_added_anchor_href_is_not_a_warning(validator: HTMLDeltaValidator) -> None:
    result = validator.validate("x", '<a href="u">x</a>')
    assert result.is_valid
    assert result.details == {}


def test_a_faithful_translation_of_a_dirty_source_cancels(validator: HTMLDeltaValidator) -> None:
    # Both sides carry the same malformed <img>; the per-tag delta is zero.
    dirty = '<img src="a" alt="say "hi"">'
    result = validator.validate(dirty, dirty)
    assert result.is_valid
    assert result.details == {}


def test_a_newly_malformed_img_tag_is_a_hard_error(validator: HTMLDeltaValidator) -> None:
    result = validator.validate('<img src="a" alt="clean">', '<img src="a" alt="bad "quote"">')
    assert not result.is_valid
    assert result.error_code == "HTML_DELTA_MISMATCH"
    raw_tag, attr_name = result.details["malformed_tags"][0]
    assert raw_tag == '<img src="a" alt="bad "quote"">'
    assert attr_name == 'quote""'


def test_warnings_accompany_a_structural_error(validator: HTMLDeltaValidator) -> None:
    result = validator.validate('<b><img src="a.png"></b>', "no img")
    assert not result.is_valid
    assert result.details["formatting_warnings"] == [
        "formatting tag drift: <b> x1 in source, x0 in target"
    ]


def test_self_closing_img_tags_are_scanned(validator: HTMLDeltaValidator) -> None:
    assert validator.validate('<img src="a.png" />', '<img src="a.png" />').is_valid
    result = validator.validate('<img src="a.png" />', "no img")
    assert not result.is_valid
    assert result.details["image_diff"]["missing_html"] == [("a.png", 1)]


def test_img_tag_matching_is_case_insensitive(validator: HTMLDeltaValidator) -> None:
    result = validator.validate('<IMG SRC="a.png">', "no")
    assert not result.is_valid
    assert result.details["image_diff"]["missing_html"] == [("a.png", 1)]


def test_added_attributes_do_not_change_the_image_set(validator: HTMLDeltaValidator) -> None:
    assert validator.validate('<img src="a.png">', '<img src="a.png" alt="z">').is_valid


def test_scan_img_tags_counts_sources_and_flags_bad_attrs(
    validator: HTMLDeltaValidator,
) -> None:
    counts, bad = validator._scan_img_tags('<img src="a"><img src="a">')
    assert counts == {"a": 2}
    assert bad == []


def test_scan_img_tags_short_circuits_without_a_tag(validator: HTMLDeltaValidator) -> None:
    assert validator._scan_img_tags("no tags") == ({}, [])


@pytest.mark.parametrize(
    ("name", "valid"),
    [
        ("src", True),
        ("data-x", True),
        ("a:b.c-d", True),
        ("1bad", False),
        ("", False),
    ],
)
def test_valid_attr_name(name: str, valid: bool) -> None:
    assert bool(_VALID_ATTR_NAME_RE.match(name)) is valid
