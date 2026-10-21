/**
 * Read the `ubt-failed-draft` wrapper the repair stage writes around a block.
 *
 * When a repair call fails, the engine keeps the block's text instead of losing
 * it, wrapped in a marker that records why:
 *
 *     <mark class="ubt-failed-draft" title="Repair error: ...">译文正文</mark>
 *
 * The workbench used to render `target_text` as a plain string, so an operator
 * saw the raw markup — attribute and all — instead of the translation. On the
 * reference job that was an 828px line inside a 446px column, which pushed the
 * whole page into horizontal scroll.
 *
 * This splits the two apart so the column can render the *translation* and show
 * the reason as a separate, wrapped notice.
 */

export interface FailedDraft {
  /** The translation carried inside the mark, or the input with no wrapper. */
  text: string
  /** Why the repair failed, as the engine summarized it. */
  reason: string | null
  /** True when the input carried the marker at all. */
  failed: boolean
}

// `[^>]*` is safe here: the engine escapes quotes inside the title, so the
// attribute cannot contain a `>` that would end the tag early.
const MARK = /^\s*<mark\s+class="ubt-failed-draft"[^>]*title="([^"]*)"[^>]*>([\s\S]*?)<\/mark>\s*$/

/** Decode the entities the engine escaped into the attribute. */
function decode(value: string): string {
  return value
    .replace(/&quot;/g, '"')
    .replace(/&#x27;|&#39;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    // `&amp;` last: decoding it first would turn `&amp;quot;` into a quote.
    .replace(/&amp;/g, '&')
}

/**
 * How much of a reason the pane shows before it is shortened.
 *
 * Deliberately short. A notice that reads "500: unspecific error: upstream
 * command exited prematurely" says everything an operator can act on, while a
 * 155-character envelope with `src`, `type`, `param` and `code` fields is noise
 * in a review pane — and it is what the engine used to write, so blocks already
 * on disk still carry it.
 */
const MAX_REASON_CHARS = 90

/**
 * Shorten a reason already stored on disk.
 *
 * Blocks written before the engine learned to summarize carry the whole provider
 * envelope in their title — measured at ~700 characters, which is what broke the
 * layout. Fixing the writer does nothing for those already on disk, so the
 * reader shortens them too: one long line of escaped JSON identifies nothing a
 * reader can act on, while the first sentence keeps the actual cause.
 */
function condense(reason: string): string {
  const flat = reason.replace(/\s+/g, ' ').trim()
  if (flat.length <= MAX_REASON_CHARS) return flat
  // Prefer the provider's own sentence: llama-swap nests it under "message".
  const message = /"message"\s*:\s*"([^"]+)"/.exec(flat)
  const status = /\b(\d{3})\b/.exec(flat)
  const parts = [
    status ? status[1] : null,
    message ? message[1] : null,
  ].filter(Boolean)
  if (parts.length > 0) return parts.join(' ')
  return flat.slice(0, MAX_REASON_CHARS - 1) + '…'
}

export function parseFailedDraft(raw: string | null | undefined): FailedDraft {
  const value = raw ?? ''
  const match = MARK.exec(value)
  if (!match) return { text: value, reason: null, failed: false }
  // Decode *before* condensing: the attribute is HTML-escaped, so
  // `"message"` arrives as `&quot;message&quot;` and a pattern looking for the
  // raw form matches nothing — which silently returned the whole 620-character
  // envelope and left the pane overflowing.
  return {
    text: match[2],
    reason: condense(decode(match[1])),
    failed: true,
  }
}
