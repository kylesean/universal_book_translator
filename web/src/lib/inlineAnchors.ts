/**
 * Helpers for UBT v2 native PDF inline anchors (`<f1/>`, `<n1/>`, `<x1/>`)
 * and paired style spans (`<b1>...</b1>`, `<i1>...</i1>`, `<c1>...</c1>`, etc.).
 */

const RIGID_ANCHOR_RE = /<([fnx])(\d+)\s*\/>/gi
const PAIRED_TAG_RE = /<(\/?)([bicars])(\d+)>/gi
const BOLD_PAIR_RE = /<b(\d+)>([\s\S]*?)<\/b\1>/gi
const ITALIC_PAIR_RE = /<i(\d+)>([\s\S]*?)<\/i\1>/gi
const PLAIN_PAIR_RE = /<([cars])(\d+)>([\s\S]*?)<\/\1\2>/gi
const ANY_INLINE_ANCHOR_RE = /<(?:[fnx]\d+\s*\/|\/?[bicars]\d+)>/i

const ANCHOR_TOKEN_PREFIX = '⟦UBT_ANCHOR:'
const ANCHOR_TOKEN_SUFFIX = '⟧'
const ANCHOR_TOKEN_EXACT_RE = /^⟦UBT_ANCHOR:([fnx])(\d+)⟧$/i

export interface InlineAnchorCheck {
  missingRigid: string[]
  unpairedTags: string[]
}

export function hasInlineAnchors(text: string): boolean {
  return ANY_INLINE_ANCHOR_RE.test(text)
}

/**
 * Compare rigid inline anchors in `source` against `target` and verify paired
 * style tags in `target` are balanced.
 */
export function checkInlineAnchors(source: string, target: string): InlineAnchorCheck {
  const sourceRigid: string[] = []
  const sourceCounts = new Map<string, number>()
  for (const match of source.matchAll(RIGID_ANCHOR_RE)) {
    const tag = `<${match[1].toLowerCase()}${match[2]}/>`
    if (!sourceCounts.has(tag)) sourceRigid.push(tag)
    sourceCounts.set(tag, (sourceCounts.get(tag) ?? 0) + 1)
  }

  const targetCounts = new Map<string, number>()
  for (const match of target.matchAll(RIGID_ANCHOR_RE)) {
    const tag = `<${match[1].toLowerCase()}${match[2]}/>`
    targetCounts.set(tag, (targetCounts.get(tag) ?? 0) + 1)
  }

  const missingRigid = sourceRigid.filter(
    (tag) => (targetCounts.get(tag) ?? 0) < (sourceCounts.get(tag) ?? 0),
  )

  const openCounts = new Map<string, number>()
  const closeCounts = new Map<string, number>()
  for (const match of target.matchAll(PAIRED_TAG_RE)) {
    const isClose = match[1] === '/'
    const key = `${match[2].toLowerCase()}${match[3]}`
    if (isClose) {
      closeCounts.set(key, (closeCounts.get(key) ?? 0) + 1)
    } else {
      openCounts.set(key, (openCounts.get(key) ?? 0) + 1)
    }
  }

  const unpairedTags: string[] = []
  const allPairedKeys = new Set([...openCounts.keys(), ...closeCounts.keys()])
  for (const key of allPairedKeys) {
    if ((openCounts.get(key) ?? 0) !== (closeCounts.get(key) ?? 0)) {
      unpairedTags.push(`<${key}>`)
    }
  }

  return { missingRigid, unpairedTags }
}

/**
 * Transform UBT inline anchors into Markdown-safe tokens so `ReactMarkdown`
 * renders bold/italic styling cleanly and never silently drops `<f1/>` tags
 * as raw HTML.
 */
export function prepareInlineAnchorsForMarkdown(text: string): string {
  if (!hasInlineAnchors(text)) return text
  let out = text
  // Iteratively unwrap nested style tags (e.g. `<b1><i2>...</i2></b1>`).
  for (let pass = 0; pass < 4; pass++) {
    const next = out
      .replace(BOLD_PAIR_RE, '**$2**')
      .replace(ITALIC_PAIR_RE, '*$2*')
      .replace(PLAIN_PAIR_RE, '$3')
    if (next === out) break
    out = next
  }
  // Convert rigid anchors into inline code sentinels rendered as badges in RichText.
  out = out.replace(RIGID_ANCHOR_RE, (_full, kind: string, num: string) => {
    return `\`${ANCHOR_TOKEN_PREFIX}${kind.toLowerCase()}${num}${ANCHOR_TOKEN_SUFFIX}\``
  })
  return out
}

export function parseAnchorBadgeToken(
  raw: string,
): { kind: 'f' | 'n' | 'x'; id: string; rawTag: string } | null {
  const m = ANCHOR_TOKEN_EXACT_RE.exec(raw.trim())
  if (!m) return null
  const kind = m[1].toLowerCase() as 'f' | 'n' | 'x'
  const num = m[2]
  return {
    kind,
    id: `${kind}${num}`,
    rawTag: `<${kind}${num}/>`,
  }
}
