"""The HTML sanitizer: stored-XSS defense for untrusted markup.

Two halves, one allowlist:

- :func:`sanitize_html_fragment` / :func:`sanitize_inline_html` guard
  *LLM-produced* target text (a prompt-injected response can carry ``<script>``
  or an inline event handler into a stored EPUB/Markdown artifact);
- :func:`scrub_source_document` guards *source* markup that is copied into the
  deliverable largely verbatim.

Both must remove executable constructs while preserving legitimate bilingual
markup and plain technical prose (``List<T>``), and both must be idempotent. A
clean document must come back byte-for-byte.
"""

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


# --------------------------------------------------------------------------- #
# Fragment sanitizer: LLM-produced target text.
# --------------------------------------------------------------------------- #


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
    # HTMLParser decodes character references before handle_data; if the text
    # were emitted verbatim, `&lt;script&gt;` would reconstitute a real element.
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


# --------------------------------------------------------------------------- #
# Inline sanitizer: tags only, text left verbatim (Markdown / LaTeX survive).
# --------------------------------------------------------------------------- #


def test_inline_drops_a_script_but_keeps_the_text() -> None:
    assert sanitize_inline_html("<script>x</script>text") == "xtext"


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


# --------------------------------------------------------------------------- #
# Source-document scrubber.
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# CSS and <mark> handling.
# --------------------------------------------------------------------------- #


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
