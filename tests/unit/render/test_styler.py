"""Unit tests for the responsive EPUB/HTML styler engine."""

from __future__ import annotations

import pytest

from ubt.render.styler import (
    BilingualMode,
    DeviceProfile,
    ReadingTheme,
    generate_epub_css,
    generate_html_interactivity_script,
    generate_modern_html_css,
)

pytestmark = pytest.mark.fast


def test_device_profile_and_theme_enums() -> None:
    assert DeviceProfile.MOBILE == "mobile"
    assert DeviceProfile.TABLET == "tablet"
    assert DeviceProfile.DESKTOP == "desktop"
    assert DeviceProfile.E_READER == "e_reader"

    assert ReadingTheme.LIGHT == "light"
    assert ReadingTheme.DARK == "dark"
    assert ReadingTheme.SEPIA == "sepia"
    assert ReadingTheme.EINK == "eink"

    assert BilingualMode.TARGET_ONLY == "mode-target"
    assert BilingualMode.STACKED == "mode-stacked"
    assert BilingualMode.SPLIT == "mode-split"
    assert BilingualMode.SOURCE_ONLY == "mode-source"


def test_modern_html_css_contains_responsive_and_mobile_queries() -> None:
    css = generate_modern_html_css()
    assert ":root" in css
    assert "--bg:" in css
    assert "--text:" in css
    assert "--accent:" in css
    assert 'data-theme="dark"' in css
    assert 'data-theme="sepia"' in css
    # Mobile breakpoint
    assert "@media (max-width: 768px)" in css
    # Mobile split collapse to stacked
    assert "body.mode-split .bilingual-pair" in css
    # Table responsive wrapper
    assert ".table-responsive" in css
    assert "overflow-x: auto" in css
    # Code block copy button
    assert ".ubt-copy-btn" in css


def test_epub_css_contains_eink_and_cjk_rules() -> None:
    css = generate_epub_css()
    assert '@charset "utf-8";' in css
    assert "font-family:" in css
    assert "text-align: justify;" in css
    assert "text-justify: inter-ideograph;" in css
    assert "word-break: break-all;" in css
    # Bilingual pairs in EPUB
    assert ".bilingual-pair" in css
    assert ".source-row" in css
    assert ".target-row" in css
    # Night/dark mode in e-readers
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
