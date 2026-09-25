# dual-column-paper

Academic arXiv dual-column paper baseline for Universal Book Translator.

- File: `dual_column_paper.md`
- Source: Synthetic benchmark modelled on top-tier arXiv NLP/ML publications.
- Why this baseline: Fixed-layout scientific prose, LaTeX inline and display mathematics, Python algorithm implementations, and academic evaluation tables.

## Testing Purpose
1. Verify math formula preservation: Mathematical expressions $\mathcal{D}$ and display equations must not be translated or corrupted.
2. Verify code block integrity: Algorithmic code blocks must remain untranslated with exact indentation and syntax.
3. Verify table formatting: Benchmark metric tables must retain pipe alignments and numerical values.
