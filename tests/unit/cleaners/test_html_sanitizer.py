from __future__ import annotations

import re

import pytest

from ubt.core.cleaners.html_sanitizer import (
    sanitize_html_fragment,
    sanitize_inline_html,
    scrub_css_text,
    scrub_source_document,
    strip_html_mark_tags,
)

pytestmark = pytest.mark.fast


def _has_active_danger(html: str) -> bool:
    lowered = html.lower()
    return (
        "<script" in lowered
        or "<iframe" in lowered
        or "<object" in lowered
        or "javascript:" in lowered
        or re.search(r"\son[a-z]+\s*=", lowered) is not None
    )


@pytest.mark.parametrize(
    "payload",
    [
        "<script>alert(1)</script>",
        "<style>body{}</style>",
        "<iframe src='http://evil'></iframe>",
        "<object data='x'>inner</object>",
        "<template><b>x</b></template>",
        "<noscript><img src=x></noscript>",
    ],
)
def test_fragment_drops_dangerous_containers_with_their_content(payload: str) -> None:
    assert sanitize_html_fragment(payload) == ""


def test_fragment_keeps_surrounding_text_when_dropping_a_script() -> None:
    assert sanitize_html_fragment("<p>a</p><script>alert(1)</script><p>b</p>") == "<p>a</p><p>b</p>"


@pytest.mark.parametrize(
    "payload",
    [
        '<em onmouseover="evil()">t</em>',
        '<img src="a.png" onerror="evil()">',
    ],
)
def test_fragment_strips_event_handlers(payload: str) -> None:
    result = sanitize_html_fragment(payload)
    assert not _has_active_danger(result)
    assert "evil()" not in result


def test_fragment_strips_javascript_and_data_navigation() -> None:
    assert sanitize_html_fragment('<a href="javascript:alert(1)">x</a>') == "<a>x</a>"
    assert sanitize_html_fragment('<a href="data:text/html,<script>x</script>">x</a>') == "<a>x</a>"


def test_fragment_keeps_safe_links_and_attributes() -> None:
    html = '<a href="https://ok/x" title="t">x</a>'
    assert sanitize_html_fragment(html) == html


def test_fragment_allows_data_urls_for_embedded_images() -> None:
    html = '<img src="data:image/png;base64,AAAA" alt="a">'
    assert sanitize_html_fragment(html) == html


def test_fragment_escapes_decoded_angle_brackets() -> None:
    result = sanitize_html_fragment("&lt;script&gt;alert(1)&lt;/script&gt;")
    assert "<script" not in result
    assert result == "&lt;script&gt;alert(1)&lt;/script&gt;"


def test_fragment_preserves_technical_prose() -> None:
    assert sanitize_html_fragment("List<T> and a<b") == "List&lt;T&gt; and a&lt;b"


def test_fragment_strips_svg_event_handlers_but_keeps_the_shape() -> None:
    result = sanitize_html_fragment('<svg onload="evil()"><rect/></svg>')
    assert "<svg" in result
    assert "onload" not in result


def test_fragment_drops_comments() -> None:
    result = sanitize_html_fragment("a<!-- secret -->b")
    assert "<!--" not in result


def test_fragment_is_idempotent() -> None:
    once = sanitize_html_fragment('<b onclick="x">t</b>')
    assert sanitize_html_fragment(once) == once


def test_inline_drops_a_script_with_its_body() -> None:
    assert sanitize_inline_html("<script>x</script>text") == "text"
    assert sanitize_inline_html("<style>p{color:red}</style>text") == "text"


def test_inline_drops_every_script_body_not_just_the_first() -> None:
    assert sanitize_inline_html("<script>a</script>mid<script>b</script>end") == "midend"


def test_inline_leaves_ordinary_markup_alone() -> None:
    assert sanitize_inline_html("<b>bold</b> x") == "<b>bold</b> x"
    assert sanitize_inline_html("a<b>c") == "a<b>c"


def test_inline_rewrites_attributes_to_the_allowlist() -> None:
    assert sanitize_inline_html('<b onclick="x">t</b>') == "<b>t</b>"


def test_inline_leaves_plain_text_and_latex_untouched() -> None:
    text = "a < b and $x&y$ and List<T>"
    assert sanitize_inline_html(text) == text


def test_inline_drops_comments() -> None:
    assert sanitize_inline_html("a<!-- c -->b") == "ab"


def test_inline_is_idempotent() -> None:
    once = sanitize_inline_html('<a href="javascript:x">y</a>')
    assert sanitize_inline_html(once) == once


def test_scrub_clean_document_is_byte_exact() -> None:
    clean = '<p class="x">Hello <b>world</b></p>'
    assert scrub_source_document(clean) == clean


def test_scrub_preserves_xml_tag_case_when_unchanged() -> None:
    xml = "<TextBlock>hi</TextBlock>"
    assert scrub_source_document(xml) == xml


def test_scrub_drops_script_with_content() -> None:
    assert scrub_source_document("<p>a</p><script>x()</script>") == "<p>a</p>"


def test_scrub_strips_event_handlers() -> None:
    assert scrub_source_document('<img src="a.png" onload="evil()">') == '<img src="a.png">'


def test_scrub_strips_javascript_href() -> None:
    assert scrub_source_document('<a href="javascript:x">y</a>') == "<a>y</a>"


def test_scrub_is_idempotent() -> None:
    once = scrub_source_document('<a href="javascript:x">y</a>')
    assert scrub_source_document(once) == once


@pytest.mark.parametrize(
    "markup",
    [
        "<style><script>alert(1)</script></style>",
        '<textarea><img src="x" onerror="alert(1)"></textarea>',
        "<svg><title><script>alert(1)</script></title></svg>",
    ],
)
def test_a_rawtext_body_cannot_carry_a_live_element(markup: str) -> None:
    import xml.etree.ElementTree as ET

    out = scrub_source_document(markup)
    root = ET.fromstring(f'<root xmlns="http://www.w3.org/1999/xhtml">{out}</root>')
    tags = [element.tag.split("}")[-1] for element in root.iter()]
    assert "script" not in tags, out
    assert "img" not in tags, out


def test_a_clean_rawtext_body_is_byte_exact() -> None:
    for markup in (
        "<style>p{color:red}</style>",
        "<title>Chapter 1</title>",
        '<textarea placeholder="notes"></textarea>',
    ):
        assert scrub_source_document(markup) == markup


def test_a_style_body_still_gets_the_css_gate_after_neutralising() -> None:
    out = scrub_source_document("<style>@import url(https://evil/x.css);p{}</style>")
    assert "@import" not in out


@pytest.mark.parametrize(
    "markup",
    [
        '<img src="https://evil/t.png"/>',
        '<video poster="https://evil/p.jpg"></video>',
        '<form action="https://evil/steal"></form>',
        '<object data="https://evil/x"></object>',
        '<p style="background:url(https://evil/x.png)">t</p>',
        '<img src="//evil/t.png"/>',
    ],
)
def test_a_remote_loading_url_is_removed(markup: str) -> None:
    out = scrub_source_document(markup)
    assert "evil" not in out, out


@pytest.mark.parametrize(
    "markup",
    [
        '<img src="images/0001.png"/>',
        '<video poster="images/p.jpg"></video>',
        '<p style="background:url(images/bg.png)">t</p>',
        '<img src="data:image/png;base64,AAAA"/>',
    ],
)
def test_an_in_package_resource_is_kept(markup: str) -> None:
    assert scrub_source_document(markup) == markup


def test_an_external_hyperlink_is_kept() -> None:
    markup = '<a href="https://example.com/ref">ref</a>'
    assert scrub_source_document(markup) == markup


def test_a_remote_css_url_is_neutralised() -> None:
    assert "evil" not in scrub_css_text("p{background:url(https://evil/x.png)}")
    assert "images/x.png" in scrub_css_text("p{background:url(images/x.png)}")


def test_a_prose_lt_before_a_close_tag_does_not_eat_the_close() -> None:
    assert scrub_source_document("<b on the other hand</p>") == "&lt;b on the other hand</p>"
    assert (
        scrub_source_document("prose <b on the other hand</p> more")
        == "prose &lt;b on the other hand</p> more"
    )


def test_a_real_tag_with_lt_inside_a_quoted_value_is_untouched() -> None:
    markup = '<img alt="a < b">'
    assert scrub_source_document(markup) == markup
    assert scrub_source_document("<div title='x < y'>z</div>") == "<div title='x < y'>z</div>"


def test_an_unterminated_lt_escapes_the_remainder_instead_of_dropping_it() -> None:
    out = scrub_source_document("<p>unterminated <b bold")
    assert out == "<p>unterminated &lt;b bold"
    assert "bold" in out


def test_scrub_css_drops_remote_import_and_keeps_relative() -> None:
    assert "@import" not in scrub_css_text('@import url("http://evil/x.css"); body{}')
    assert "@import url(" in scrub_css_text('@import url("book.css"); body{}')


def test_scrub_css_neutralizes_expression_and_javascript_urls() -> None:
    assert "expression(" not in scrub_css_text("a{width:expression(alert(1))}")
    assert "javascript:" not in scrub_css_text("a{background:url(javascript:alert(1))}")


def test_scrub_css_leaves_safe_css_untouched() -> None:
    css = "body { color: red; }"
    assert scrub_css_text(css) == css


def test_strip_html_mark_tags_keeps_inner_text() -> None:
    assert strip_html_mark_tags('a <mark class="q">hit</mark> b') == "a hit b"


def test_strip_html_mark_tags_is_a_noop_without_mark() -> None:
    assert strip_html_mark_tags("plain text") == "plain text"
