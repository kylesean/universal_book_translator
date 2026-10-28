"""Responsive styling and layout adapter for EPUB 3 and modern Web HTML views.

Provides multi-viewport responsive media queries, E-ink reader contrast tuning,
bilingual stacked/split viewports, and mobile-friendly table/code containers.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class DeviceProfile(StrEnum):
    """Target reader hardware and viewport profiles."""

    MOBILE = "mobile"  # Smartphones (<768px)
    TABLET = "tablet"  # Tablets and foldable screens (768px-1024px)
    DESKTOP = "desktop"  # Desktop browsers (>1024px)
    E_READER = "e_reader"  # E-ink devices (Kindle, Kobo)


class ReadingTheme(StrEnum):
    """Palette and contrast themes for electronic reading."""

    LIGHT = "light"
    DARK = "dark"
    SEPIA = "sepia"
    EINK = "eink"


class BilingualMode(StrEnum):
    """Layout modes for bilingual translated texts."""

    TARGET_ONLY = "mode-target"
    STACKED = "mode-stacked"
    SPLIT = "mode-split"
    SOURCE_ONLY = "mode-source"


#: Modern responsive CSS for standalone HTML reading view
MODERN_HTML_CSS: Final[str] = """
:root {
  --bg: #ffffff;
  --bg-rgb: 255, 255, 255;
  --text: #1f2937;
  --muted: #64748b;
  --border: #e2e8f0;
  --accent: #2563eb;
  --card-bg: #f8fafc;
  --code-bg: #f1f5f9;
  --source-border: #94a3b8;
  --font-base: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Source Han Serif SC", "Noto Serif CJK SC", serif;
  --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f172a;
    --bg-rgb: 15, 23, 42;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --border: #334155;
    --accent: #3b82f6;
    --card-bg: #1e293b;
    --code-bg: #111827;
    --source-border: #475569;
  }
}

:root[data-theme="dark"] {
  --bg: #0f172a;
  --bg-rgb: 15, 23, 42;
  --text: #f1f5f9;
  --muted: #94a3b8;
  --border: #334155;
  --accent: #3b82f6;
  --card-bg: #1e293b;
  --code-bg: #111827;
  --source-border: #475569;
}

:root[data-theme="light"] {
  --bg: #ffffff;
  --bg-rgb: 255, 255, 255;
  --text: #1f2937;
  --muted: #64748b;
  --border: #e2e8f0;
  --accent: #2563eb;
  --card-bg: #f8fafc;
  --code-bg: #f1f5f9;
  --source-border: #94a3b8;
}

:root[data-theme="sepia"] {
  --bg: #fbf0d9;
  --bg-rgb: 251, 240, 217;
  --text: #5f4b32;
  --muted: #8f7d66;
  --border: #e8d8be;
  --accent: #b06222;
  --card-bg: #f5e6cc;
  --code-bg: #ece0c8;
  --source-border: #c4b094;
}

html {
  box-sizing: border-box;
  scroll-behavior: smooth;
}

*, *:before, *:after {
  box-sizing: inherit;
}

body {
  margin: 0;
  padding: 2rem 1rem;
  background-color: var(--bg);
  color: var(--text);
  font-family: var(--font-base);
  font-size: clamp(1rem, 0.95rem + 0.35vw, 1.15rem);
  line-height: 1.8;
  text-align: justify;
  text-justify: inter-ideograph;
  word-break: break-word;
  transition: background-color 0.2s ease, color 0.2s ease;
}

.ubt-article-container {
  max-width: 860px;
  margin: 0 auto;
  transition: max-width 0.25s ease;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
  color: var(--text);
  font-weight: 700;
  line-height: 1.35;
  margin-top: 2em;
  margin-bottom: 0.6em;
  page-break-after: avoid;
}

h1 { font-size: 2.1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.6rem; }
h2 { font-size: 1.55rem; }
h3 { font-size: 1.25rem; }
p { margin-top: 0; margin-bottom: 1.25em; }

/* Sticky Reading Toolbar */
.ubt-toolbar {
  position: sticky;
  top: 1rem;
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 0.5rem;
  margin-bottom: 2rem;
  z-index: 100;
  background: rgba(var(--bg-rgb), 0.85);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  padding: 0.5rem 0.75rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

.ubt-btn {
  background: var(--card-bg);
  border: 1px solid var(--border);
  color: var(--text);
  padding: 0.35rem 0.8rem;
  border-radius: 6px;
  font-size: 0.85rem;
  font-family: inherit;
  cursor: pointer;
  transition: all 0.15s ease;
  user-select: none;
  min-height: 32px;
}

.ubt-btn:hover, .ubt-btn.active {
  background: var(--accent);
  color: #ffffff;
  border-color: var(--accent);
}

/* Responsive Table Wrapper */
.table-responsive {
  width: 100%;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  margin: 1.8em 0;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--card-bg);
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.95rem;
  min-width: 480px;
}

th, td {
  padding: 0.75rem 1rem;
  border-bottom: 1px solid var(--border);
  text-align: left;
}

th {
  font-weight: 600;
  background: rgba(0, 0, 0, 0.03);
  border-bottom: 2px solid var(--border);
}

/* Code Blocks */
pre {
  position: relative;
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1.2rem;
  overflow-x: auto;
  font-family: var(--font-mono);
  font-size: 0.9rem;
  line-height: 1.5;
  tab-size: 2;
}

code {
  font-family: var(--font-mono);
  font-size: 0.9em;
  background: rgba(125, 125, 125, 0.12);
  padding: 0.15em 0.35em;
  border-radius: 4px;
}

pre code {
  background: none;
  padding: 0;
}

.ubt-copy-btn {
  position: absolute;
  top: 0.5rem;
  right: 0.5rem;
  opacity: 0;
  transition: opacity 0.2s ease;
  background: var(--card-bg);
  border: 1px solid var(--border);
  color: var(--text);
  font-size: 0.75rem;
  padding: 0.25rem 0.5rem;
  border-radius: 4px;
  cursor: pointer;
}

pre:hover .ubt-copy-btn {
  opacity: 0.9;
}

/* Figures */
figure {
  margin: 2.5em 0;
  text-align: center;
}

figure img {
  max-width: 100%;
  height: auto;
  border-radius: 6px;
  box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.1);
}

figcaption {
  font-size: 0.9rem;
  color: var(--muted);
  margin-top: 0.6em;
}

/* KaTeX Math Container */
.katex-display {
  overflow-x: auto;
  overflow-y: hidden;
  padding: 0.35em 0;
}

.katex {
  font-size: 1.05em;
}

.formula {
  font-family: "KaTeX_Main", "Times New Roman", serif;
  display: inline-block;
}

/* Bilingual Modes */
body.mode-stacked .bilingual-pair .source-row {
  display: block;
  color: var(--muted);
  font-size: 0.92em;
  font-style: italic;
  border-left: 3px solid var(--source-border);
  padding-left: 0.75rem;
  margin-bottom: 0.45em;
}

body.mode-stacked .bilingual-pair .target-row {
  display: block;
}

body.mode-split .ubt-article-container {
  max-width: 1400px;
}

body.mode-split .bilingual-pair {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2.5rem;
  margin-bottom: 1.5em;
}

body.mode-split .bilingual-pair .source-row {
  color: var(--muted);
  font-size: 0.95em;
}

body.mode-target .source-row {
  display: none;
}

body.mode-source .target-row {
  display: none;
}

/* Responsive Mobile Viewport Adaptations (<768px) */
@media (max-width: 768px) {
  body {
    padding: 1.25rem 0.75rem;
  }

  .ubt-toolbar {
    justify-content: center;
    gap: 0.35rem;
    padding: 0.4rem;
  }

  .ubt-btn {
    font-size: 0.78rem;
    padding: 0.3rem 0.6rem;
  }

  /* Split view automatically collapses to stacked layout on mobile */
  body.mode-split .bilingual-pair {
    display: block;
    margin-bottom: 1.75em;
  }

  body.mode-split .bilingual-pair .source-row {
    display: block;
    color: var(--muted);
    font-size: 0.9em;
    font-style: italic;
    border-left: 3px solid var(--accent);
    padding-left: 0.6rem;
    margin-bottom: 0.4em;
  }

  body.mode-split .bilingual-pair .target-row {
    display: block;
  }
}
"""


#: Publication-grade EPUB 3 reflowable stylesheet for E-ink, tablets, and phones
EPUB_CSS: Final[str] = """\
@charset "utf-8";

body {
  margin: 0 4%;
  line-height: 1.75;
  font-family: "Source Han Serif SC", "Noto Serif CJK SC", "PingFang SC", "Songti SC", "SimSun", serif;
  font-size: 1.0em;
  text-align: justify;
  text-justify: inter-ideograph;
  word-break: break-all;
  -webkit-hyphens: none;
  hyphens: none;
}

p {
  margin: 0 0 1.0em;
  text-indent: 0;
}

h1, h2, h3, h4, h5, h6 {
  line-height: 1.3;
  margin: 1.4em 0 0.6em;
  page-break-after: avoid;
  break-after: avoid;
  font-weight: bold;
}

h1 { font-size: 1.55em; border-bottom: 1px solid #ccc; padding-bottom: 0.3em; }
h2 { font-size: 1.35em; }
h3 { font-size: 1.18em; }

figure {
  margin: 1.5em 0;
  text-align: center;
  page-break-inside: avoid;
  break-inside: avoid;
}

figure img {
  max-width: 100%;
  height: auto;
  display: block;
  margin: 0 auto;
}

figcaption {
  font-size: 0.88em;
  color: #444444;
  text-align: center;
  margin-top: 0.5em;
}

table {
  border-collapse: collapse;
  width: 100%;
  font-size: 0.9em;
  margin: 1.2em 0;
  page-break-inside: auto;
  break-inside: auto;
}

th, td {
  border: 1px solid #888888;
  padding: 0.4em 0.6em;
  vertical-align: top;
}

th {
  background-color: #f2f2f2;
  font-weight: bold;
}

thead {
  display: table-header-group;
}

pre {
  overflow-x: auto;
  font-family: monospace, Courier, "Courier New";
  font-size: 0.85em;
  line-height: 1.45;
  background-color: #f6f8fa;
  border: 1px solid #d0d7de;
  border-radius: 4px;
  padding: 0.8em;
  white-space: pre-wrap;
  word-break: break-all;
}

code {
  font-family: monospace, Courier, "Courier New";
  font-size: 0.9em;
  background-color: #f0f2f5;
  padding: 0.1em 0.3em;
  border-radius: 3px;
}

blockquote {
  margin: 1em 0 1em 1.2em;
  padding-left: 0.8em;
  border-left: 3px solid #aaaaaa;
  color: #444444;
}

/* Bilingual Presentation in EPUB */
.bilingual-pair {
  margin-bottom: 1.4em;
  page-break-inside: avoid;
  break-inside: avoid;
}

.source-row {
  color: #555555;
  font-size: 0.88em;
  font-style: italic;
  border-left: 2px solid #888888;
  padding-left: 0.7em;
  margin-bottom: 0.35em;
}

.target-row {
  color: #000000;
  font-size: 1.0em;
}

/* Night / Dark reading mode support */
@media (prefers-color-scheme: dark) {
  body {
    background-color: #121212;
    color: #e4e4e4;
  }

  th {
    background-color: #222222;
  }

  th, td {
    border-color: #555555;
  }

  figcaption {
    color: #aaaaaa;
  }

  pre {
    background-color: #1e1e1e;
    border-color: #333333;
    color: #d4d4d4;
  }

  code {
    background-color: #2a2a2a;
    color: #e0e0e0;
  }

  .source-row {
    color: #888888;
    border-left-color: #666666;
  }

  .target-row {
    color: #ffffff;
  }
}
"""


def generate_modern_html_css() -> str:
    """Return the modern responsive stylesheet for HTML book viewing."""
    return MODERN_HTML_CSS


def generate_epub_css() -> str:
    """Return the high-fidelity EPUB 3 publication stylesheet."""
    return EPUB_CSS


def generate_html_interactivity_script() -> str:
    """Return the client-side JavaScript for themes, modes, shortcuts, and copy buttons."""
    return """\
<script>
function setTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  try { localStorage.setItem('ubt_theme', theme); } catch(e){}
  var b = document.getElementById('ubt-theme-btn');
  if (b) {
    if (theme === 'dark') b.textContent = '浅色';
    else if (theme === 'sepia') b.textContent = '深色';
    else b.textContent = '护眼';
  }
}

function cycleTheme() {
  var cur = document.documentElement.getAttribute('data-theme') || 'light';
  if (cur === 'light') setTheme('sepia');
  else if (cur === 'sepia') setTheme('dark');
  else setTheme('light');
}

function toggleTheme() {
  cycleTheme();
}

function setMode(mode, btn) {
  document.body.className = mode;
  document.querySelectorAll('.ubt-btn').forEach(function(b){
    if (b.id !== 'ubt-theme-btn') b.classList.remove('active');
  });
  if (btn) btn.classList.add('active');
  try { localStorage.setItem('ubt_view_mode', mode); } catch(e){}
}

function copyCode(btn) {
  var pre = btn.parentElement;
  var code = pre.querySelector('code');
  var text = code ? code.innerText : pre.innerText;
  navigator.clipboard.writeText(text).then(function() {
    var orig = btn.textContent;
    btn.textContent = '已复制!';
    setTimeout(function() { btn.textContent = orig; }, 1800);
  });
}

function initCodeBlocks() {
  document.querySelectorAll('pre').forEach(function(pre) {
    if (!pre.querySelector('.ubt-copy-btn')) {
      var btn = document.createElement('button');
      btn.className = 'ubt-copy-btn';
      btn.textContent = '复制';
      btn.onclick = function() { copyCode(btn); };
      pre.appendChild(btn);
    }
  });
}

function renderUbtMath() {
  if (typeof renderMathInElement !== 'function') return;
  document.querySelectorAll('.math-tex').forEach(function(el){
    try {
      renderMathInElement(el, {
        delimiters: [
          {left: '$$', right: '$$', display: true},
          {left: '$', right: '$', display: false},
          {left: '\\\\(', right: '\\\\)', display: false},
          {left: '\\\\[', right: '\\\\]', display: true}
        ],
        throwOnError: false
      });
    } catch(e){}
  });
}

document.addEventListener('keydown', function(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 't' || e.key === 'T') cycleTheme();
  else if (e.key === '1') {
    var b1 = document.querySelector('.ubt-btn[onclick*="mode-target"]');
    if (b1) setMode('mode-target', b1);
  } else if (e.key === '2') {
    var b2 = document.querySelector('.ubt-btn[onclick*="mode-stacked"]');
    if (b2) setMode('mode-stacked', b2);
  } else if (e.key === '3') {
    var b3 = document.querySelector('.ubt-btn[onclick*="mode-split"]');
    if (b3) setMode('mode-split', b3);
  }
});

(function(){
  try {
    var savedTheme = localStorage.getItem('ubt_theme');
    if (savedTheme) setTheme(savedTheme);
    var savedMode = localStorage.getItem('ubt_view_mode');
    if (savedMode) {
      var btn = document.querySelector('.ubt-btn[onclick*="' + savedMode + '"]');
      if (btn) setMode(savedMode, btn);
    }
  } catch(e){}
  document.addEventListener('DOMContentLoaded', function() {
    renderUbtMath();
    initCodeBlocks();
  });
})();
</script>
"""
