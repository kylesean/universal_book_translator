/** Whether text visibly carries Markdown / GFM / TeX structure. */
export function hasStructuredMarkup(text: string): boolean {
  return (
    /(^|\n)\s{0,3}(#{1,6}|[-*+]|\d+\.)\s|`|\*\*|\[[^\]]+\]\([^)]+\)|\$[^$]+\$/.test(
      text
    ) ||
    // GFM tables: a row of pipes plus the delimiter row. Without this a
    // source block that is *only* a table (no heading, list or code fence
    // elsewhere in the text) failed the gate and rendered as raw pipe text.
    /\n\s*\|[-:| ]+\|\s*\n/.test(text)
  )
}
