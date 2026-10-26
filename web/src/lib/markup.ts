/** Whether text visibly carries Markdown / GFM / TeX structure. */
export function hasStructuredMarkup(text: string): boolean {
  return /(^|\n)\s{0,3}(#{1,6}|[-*+]|\d+\.)\s|`|\*\*|\[[^\]]+\]\([^)]+\)|\$[^$]+\$/.test(
    text
  )
}
