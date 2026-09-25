# MathJax formula renderer (optional)

Display-formula backend for UBT. Renders the OCR LaTeX to an
SVG vector with MathJax and rasterizes it for the formula witness; Typst
embeds the SVG. Pinned dependencies only.

```bash
cd scripts/mathjax && npm install   # mathjax-full + @resvg/resvg-js
```

The pipeline probes this directory automatically and falls back to the legacy
Typst converter (`UBT_MATH_BACKEND=typst`) when Node or these packages are
missing. `node_modules/` is git-ignored; `package-lock.json` is committed for
reproducibility. Protocol: one JSON request per line on stdin, one response
per line on stdout (`{"id","latex","tag","display","png_width"}`).
