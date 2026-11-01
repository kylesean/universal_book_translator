from __future__ import annotations

import pytest

from ubt.render.styler import (
    generate_epub_css,
    generate_html_interactivity_script,
    generate_modern_html_css,
)

pytestmark = pytest.mark.fast


def test_modern_html_css_contains_responsive_and_mobile_queries() -> None:
    css = generate_modern_html_css()
    assert ":root" in css
    assert "--bg:" in css
    assert "--text:" in css
    assert "--accent:" in css
    assert 'data-theme="dark"' in css
    assert 'data-theme="sepia"' in css
    assert "@media (max-width: 768px)" in css
    assert "body.mode-split .bilingual-pair" in css
    assert ".table-responsive" in css
    assert "overflow-x: auto" in css
    assert ".ubt-copy-btn" in css


def test_epub_css_contains_eink_and_cjk_rules() -> None:
    css = generate_epub_css()
    assert '@charset "utf-8";' in css
    assert "font-family:" in css
    assert "text-align: justify;" in css
    assert "text-justify: inter-ideograph;" in css
    assert "word-break: break-all;" in css
    assert ".bilingual-pair" in css
    assert ".source-row" in css
    assert ".target-row" in css
    assert "@media (prefers-color-scheme: dark)" in css


def test_html_interactivity_script_contains_shortcuts_and_helpers() -> None:
    script = generate_html_interactivity_script()
    assert "function setTheme" in script
    assert "function cycleTheme" in script
    assert "function toggleTheme" in script
    assert "function setMode" in script
    assert "function copyCode" in script
    assert "function initCodeBlocks" in script
    assert "function renderUbtMath" in script
    assert "addEventListener('keydown'" in script
    assert "localStorage" in script
