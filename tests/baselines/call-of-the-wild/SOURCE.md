# call-of-the-wild

Full-length novel baseline input for Universal Book Translator.

- File: `call_of_the_wild.epub`
- Source: Project Gutenberg eBook #215, *The Call of the Wild* by Jack London
  (d. 1916). Text is in the public domain in the United States and
  essentially worldwide; retrieved 2026-09-19 from
  `https://www.gutenberg.org/ebooks/215.epub.noimages`.
- Why this baseline: Long-form narrative prose with chapter progression
  (5 chapters across two books), recurring named entities (Buck, Thornton,
  Hal, the "law of club and fang"), and continuous narrative flow — the same
  properties the retired animal-farm fixture provided, from a corpus that is
  unambiguously redistributable. (The previous fixture's claimed Gutenberg
  provenance was false: Orwell's novel is still under copyright.)

## Testing Purpose
1. Verify streaming chapter cursor processing and non-destructive checkpointing across large novel chapters.
2. Verify cross-chapter entity consistency via Translation Bible (e.g. "Buck" -> "巴克", "Thornton" -> "桑顿", "call of the wild" -> "野性的呼唤").
3. Verify memory isolation and low resident memory footprint ($O(1)$) during sustained pipeline execution.
