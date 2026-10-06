"""Unit tests for canonical markup_cleanup module."""

from __future__ import annotations

import pytest

from ubt.core.cleaners.markup_cleanup import (
    clean_model_repair_text,
    normalize_escaped_entities,
    strip_protocol_tags,
)

pytestmark = pytest.mark.fast


def test_strip_protocol_tags_removes_error_span_and_corrections() -> None:
    raw = '<error_span id="1" severity="major">broken</error_span> and <correction id="1">fixed</correction>'
    assert strip_protocol_tags(raw) == "broken and fixed"


def test_strip_protocol_tags_handles_final_translation() -> None:
    raw = "<final_translation>This is the final text.</final_translation>"
    assert strip_protocol_tags(raw) == "This is the final text."


def test_clean_model_repair_text_strips_tags_and_decodes_entities() -> None:
    raw = "<final_translation>\n  AT&amp;T &lt;Bell Labs&gt; &quot;Innovations&quot;  \n</final_translation>"
    assert clean_model_repair_text(raw) == 'AT&T <Bell Labs> "Innovations"'


def test_normalize_escaped_entities() -> None:
    assert normalize_escaped_entities("Tom &amp; Jerry&#39;s house") == "Tom & Jerry's house"
