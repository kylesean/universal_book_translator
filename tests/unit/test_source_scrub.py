"""``scrub_source_document`` — the L-4 guard for markup the run did not write.

EPUB/HTML sources are copied into the deliverable largely verbatim, so they are
scrubbed for executable constructs (``<script>``, ``on*`` handlers,
``javascript:`` URLs) before export. The tests below pin the two failure
directions of that scrub: a payload must not survive, and a clean book must be
returned byte-identical (tag case included — EPUB members and SVG are XML).
"""

from __future__ import annotations

import pytest

from ubt.core.cleaners.html_sanitizer import sanitize_html_fragment, scrub_source_document


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


def test_unpaired_dangerous_tag_is_escaped_but_unterminated_still_fails_closed() -> None:
    """An unpaired ``<script>`` is escaped to inert text, not dropped-with-content.

    Superseded contract (explicit product decision, 2026-09): a dangerous name
    with no matching close tag is prose-like (``<embed src=…>`` in a caption,
    ``<script>`` mentioned in a CS book), so escaping it preserves the rest of
    the member instead of deleting it. Escaped text cannot execute, so the XSS
    guarantee is unchanged. A tag whose own ``>`` never arrives still drops the
    remainder (fail closed).
    """
    out = scrub_source_document("<p>ok</p><script>unclosed")
    assert "unclosed" in out
    assert "<script" not in out.lower()
    assert "<p>ok</p>" in out
    # No ``>`` for the tag itself: still fail closed, dropping the remainder.
    out_unterminated = scrub_source_document("<p>ok</p><script unclosed")
    assert "unclosed" not in out_unterminated
    assert "<p>ok</p>" in out_unterminated


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


def test_xlink_href_javascript_is_dropped() -> None:
    """SVG links use ``xlink:href``; it is the standard activatable link attr."""
    out = scrub_source_document('<a xlink:href="javascript:alert(1)">x</a>')
    assert "javascript" not in out
    assert ">x</a>" in out


def test_srcset_with_a_javascript_candidate_is_dropped() -> None:
    out = scrub_source_document('<img srcset="a.png 1x, javascript:alert(1) 2x" alt="ok">')
    assert "javascript" not in out
    assert 'alt="ok"' in out


def test_base_refresh_meta_and_remote_link_are_dropped() -> None:
    """``<base>`` hijacks every relative link; ``<meta refresh>``/remote ``<link>`` fetch."""
    out = scrub_source_document(
        '<head><base href="https://evil.example/"><meta http-equiv="refresh" '
        'content="0;url=https://evil.example/"><link rel="stylesheet" '
        'href="https://evil.example/x.css"></head><body>b</body>'
    )
    assert "evil.example" not in out
    assert "<base" not in out.lower()
    assert "http-equiv" not in out.lower()
    assert "<link" not in out.lower()
    assert "<body>b</body>" in out


def test_style_import_is_stripped_but_css_kept() -> None:
    out = scrub_source_document(
        '<style>@import url("https://evil.example/x.css");p{color:red}</style>'
    )
    assert "evil.example" not in out
    assert "p{color:red}" in out


def test_legitimate_link_and_style_survive() -> None:
    src = '<link rel="record" href="onix.xml"><style>p{background:url(fig.png)}</style>'
    assert scrub_source_document(src) == src


def test_local_stylesheet_link_and_import_survive() -> None:
    """Only *remote* resource links/imports are egress; relative ones must stay.

    The EPUB renderer emits its own ``<link rel="stylesheet" href="ubt-bilingual.css">``
    into every chapter head — a blanket resource-rel drop removed it.
    """
    src = (
        '<link rel="stylesheet" href="ubt-bilingual.css">'
        '<style>@import "chapter.css"; p{color:red}</style>'
    )
    assert scrub_source_document(src) == src


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


_r0918_CONTROL_CHAR_URLS = [
    '<a href="java\tscript:alert(1)">x</a>',
    '<a href="java\nscript:alert(1)">x</a>',
    '<a href="HTM\x00L:alert(1)">x</a>',
    '<a href="vbscript:msgbox(1)">x</a>',
]


@pytest.mark.parametrize("fragment", _r0918_CONTROL_CHAR_URLS)
def test_scheme_split_by_control_characters_is_dropped(fragment: str) -> None:
    # Pre-fix the scheme never matched the anchored regex, so the URL was
    # treated as a relative link and emitted verbatim.
    assert "href" not in sanitize_html_fragment(fragment)


def test_data_url_is_refused_for_navigation_but_allowed_for_images() -> None:
    assert "href" not in sanitize_html_fragment('<a href="data:text/html,<b>x</b>">x</a>')
    img = '<img src="data:image/png;base64,iVBORw0KGgo=" alt="i">'
    assert "data:image/png" in sanitize_html_fragment(img)


def test_legitimate_relative_and_absolute_links_survive() -> None:
    for href in ("chapter1.xhtml", "#anchor", "../img/a.png", "https://ok.example/x"):
        assert f'href="{href}"' in sanitize_html_fragment(f'<a href="{href}">x</a>')


def test_dangerous_tag_names_in_prose_do_not_truncate_the_fragment() -> None:
    """A translated fragment *about* a dangerous tag is prose, not a live node.

    Regression: an unpaired ``<script>`` stayed markup, so the allowlist parser
    dropped everything after it — a technical-book sentence such as
    ``The <script> tag loads JavaScript into the page.`` was silently cut to
    ``The ``. Only a *paired* container (a real close tag) may drop content.
    """
    for sentence in (
        "The <script> tag loads JavaScript into the page.",
        "Apply styles with a <style> block, then test.",
        "Use the <embed> element to embed external content.",
        "An <object> can host a plugin.",
        "Wrap the fallback in a <noscript> element.",
    ):
        out = sanitize_html_fragment(sentence)
        assert out.rstrip().endswith("."), (sentence, out)
        assert "<script" not in out.lower()
        assert "<embed" not in out.lower()
    # A real, paired container still loses its content (XSS guarantee intact).
    assert sanitize_html_fragment("a<script>alert(1)</script>b") == "ab"


def test_source_unpaired_dangerous_tag_keeps_the_document_tail() -> None:
    """``<embed>`` is void: it has no close tag, so hunting one deleted the tail.

    Regression: ``_skip_element`` looked for ``</embed>``, never found it, and
    returned end-of-input, dropping the rest of the member. An unpaired
    dangerous name is now escaped to inert text; a paired one still drops its
    content.
    """
    out = scrub_source_document('<p>Before</p><embed src="movie.swf"><p>AFTER BODY</p>')
    assert "AFTER BODY" in out
    assert "<embed" not in out.lower()
    assert "payload" not in scrub_source_document("<div>a<embed>payload</embed>b</div>")


def test_smil_animation_elements_are_dropped() -> None:
    """SMIL retargets a parent's attribute to a scheme the URL gate never sees.

    Regression: ``<animate attributeName="href" values="javascript:…">``
    survived ``scrub_source_document`` verbatim, so a source could animate an
    already-approved ``href`` into an executable URL after the scheme scrubber
    had passed it.
    """
    src = (
        '<svg xmlns="http://www.w3.org/2000/svg">'
        '<a id="l" href="https://ok.example/"><text>click</text>'
        '<animate attributeName="href" values="javascript:alert(1)" dur="0s" fill="freeze"/>'
        "</a></svg>"
    )
    out = scrub_source_document(src)
    assert "javascript:alert(1)" not in out
    assert "<animate" not in out.lower()
    # The approved link itself is untouched.
    assert 'href="https://ok.example/"' in out
    # A paired SMIL element is dropped too.
    paired = scrub_source_document('<set to="javascript:alert(2)"></set><p>x</p>')
    assert "javascript:alert(2)" not in paired
    assert "<p>x</p>" in paired
