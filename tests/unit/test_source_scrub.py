"""``scrub_source_document`` — the L-4 guard for markup the run did not write.

EPUB/HTML sources are copied into the deliverable largely verbatim, so they are
scrubbed for executable constructs (``<script>``, ``on*`` handlers,
``javascript:`` URLs) before export. The tests below pin the two failure
directions of that scrub: a payload must not survive, and a clean book must be
returned byte-identical (tag case included — EPUB members and SVG are XML).
"""

from __future__ import annotations

import pytest

from ubt.core.cleaners.html_sanitizer import scrub_source_document


def test_script_element_is_dropped_with_its_content() -> None:
    out = scrub_source_document("<p>keep</p><script>alert('stolen')</script><p>also</p>")
    assert "alert" not in out
    assert "<script" not in out.lower()
    assert "<p>keep</p>" in out and "<p>also</p>" in out


def test_iframe_object_embed_applet_drop_with_content() -> None:
    for tag in ("iframe", "object", "embed", "applet"):
        out = scrub_source_document(f"<div>a<{tag}>payload</{tag}>b</div>")
        assert "payload" not in out, tag


def test_style_template_noscript_containers_survive() -> None:
    src = "<style>p{color:red}</style><template>t</template><noscript>n</noscript>"
    assert scrub_source_document(src) == src


def test_script_smuggled_inside_a_kept_container_is_still_scrubbed() -> None:
    out = scrub_source_document("<noscript><script>steal()</script></noscript>")
    assert "steal" not in out
    assert "<noscript>" in out


def test_event_handler_attribute_removed_other_attributes_kept() -> None:
    out = scrub_source_document('<img src="pic.png" onload="steal()" alt="ok">')
    assert "onload" not in out and "steal" not in out
    assert 'src="pic.png"' in out and 'alt="ok"' in out


def test_javascript_url_attribute_dropped() -> None:
    out = scrub_source_document('<a href="javascript:steal()">x</a>')
    assert "javascript" not in out
    assert ">x</a>" in out


def test_safe_and_relative_urls_kept() -> None:
    src = '<a href="chapter2.xhtml#s3">x</a><a href="https://example.com/">y</a>'
    assert scrub_source_document(src) == src


def test_clean_xml_member_is_byte_exact_including_tag_case() -> None:
    src = (
        '<?xml version="1.0"?>\n<Message><SenderName>ACME</SenderName>'
        "<Note>keep &amp; pass</Note></Message>"
    )
    assert scrub_source_document(src) == src


def test_comments_and_doctype_pass_through_verbatim() -> None:
    src = "<!DOCTYPE html><!-- <script>inert</script> --><html><body>b</body></html>"
    assert scrub_source_document(src) == src


def test_unterminated_dangerous_tag_fails_closed() -> None:
    out = scrub_source_document("<p>ok</p><script>unclosed")
    assert "unclosed" not in out
    assert "<p>ok</p>" in out


def test_scrub_is_idempotent() -> None:
    src = '<div onload="a()"><script>b()</script><img src="x" onerror="c()" /></div>'
    once = scrub_source_document(src)
    assert scrub_source_document(once) == once


def test_scrubbed_self_closing_tag_stays_well_formed_xml() -> None:
    """EPUB members are XHTML: ``<br>`` is not interchangeable with ``<br/>``.

    A tag whose attributes had to go is re-rendered; the void-element form must
    survive the re-render or the reading system rejects the whole member.
    """
    out = scrub_source_document('<img src="pic.png" onload="steal()" />')
    assert out.endswith("/>")
    assert "onload" not in out


def test_empty_input_returns_empty() -> None:
    assert scrub_source_document("") == ""


def test_case_sensitive_xml_attributes_survive_a_re_render() -> None:
    """SVG/XML attribute names are case-sensitive: ``viewBox`` != ``viewbox``.

    Removing a neighbouring handler forces the tag to be re-rendered, and the
    parser's lowercased names would otherwise be written back — silently
    breaking the graphic while looking like a successful scrub.
    """
    out = scrub_source_document(
        '<svg viewBox="0 0 10 10" preserveAspectRatio="xMidYMid" onload="steal()">'
        '<path d="M0 0"/></svg>'
    )
    assert "onload" not in out and "steal" not in out
    assert 'viewBox="0 0 10 10"' in out
    assert 'preserveAspectRatio="xMidYMid"' in out


@pytest.mark.parametrize("attr", ["poster", "action", "formaction", "data", "src"])
def test_url_bearing_attributes_are_scheme_gated(attr: str) -> None:
    out = scrub_source_document(f'<div {attr}="javascript:x()">t</div>')
    assert "javascript" not in out, attr
