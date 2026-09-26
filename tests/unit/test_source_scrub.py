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


def test_benign_paired_tags_are_dropped_not_escaped() -> None:
    import html

    from ubt.core.cleaners.html_sanitizer import sanitize_html_fragment

    assert sanitize_html_fragment("<div>hi</div>") == "hi"
    assert sanitize_html_fragment('<font color="red">hi</font>') == "hi"
    # Unpaired pseudo-tags must still survive as literal text (escaped in the
    # raw string, which is the sanitizer's contract; they render as-is).
    assert "List<T>" in html.unescape(sanitize_html_fragment("List<T>"))
    assert "<stdio.h>" in html.unescape(sanitize_html_fragment("<stdio.h>"))
    # A dangerous container is still dropped together with its content.
    dropped = sanitize_html_fragment("<div>ok<script>alert(1)</script></div>")
    assert "ok" in dropped and "alert" not in dropped


@pytest.mark.fast
def test_html_sanitizer_cdata_and_processing_instructions() -> None:
    """_SourceTagScrubber.scan must support CDATA ending at ]]> and PI ending at ?>."""
    # 1. CDATA containing internal '>' and tags that shouldn't be parsed as HTML elements
    cdata_doc = (
        '<root><![CDATA[ <div title="a">x > y</div> <script>test</script> ]]><p>Hello</p></root>'
    )
    scrubbed = scrub_source_document(cdata_doc)
    assert '<![CDATA[ <div title="a">x > y</div> <script>test</script> ]]>' in scrubbed
    assert "<p>Hello</p>" in scrubbed

    # 2. Processing instruction containing '>'
    pi_doc = '<root><?custom-pi note="a > b" ?><p>Body</p></root>'
    scrubbed_pi = scrub_source_document(pi_doc)
    assert '<?custom-pi note="a > b" ?>' in scrubbed_pi
    assert "<p>Body</p>" in scrubbed_pi
