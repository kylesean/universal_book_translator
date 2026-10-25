import type { DeliverableItem } from '@/api/client'
import type { useI18n } from '@/i18n/useI18n'

/** The `t` object's type, shared by the gate's helper modules. */
export type Translation = ReturnType<typeof useI18n>['t']

/**
 * How a deliverable can be inspected inside the console.
 *
 * - `embed` — the browser renders it itself (PDF viewer / HTML document) in an
 *   iframe pointed at the `?inline=1` download URL.
 * - `xliff` — parsed client-side into a source/target segment table.
 * - `json`  — fetched and rendered as a structured digest (plus the raw JSON
 *   behind a disclosure), never as a raw blob dump.
 * - `null`  — nothing the browser can show; download only (EPUB).
 */
export type PreviewKind = 'embed' | 'xliff' | 'json' | null

/** What (if anything) the console can preview for this deliverable. */
export function previewKind(item: DeliverableItem): PreviewKind {
  if (item.key === 'xliff') return 'xliff'
  if (item.key === 'contract' || item.key === 'quality_report' || item.key === 'visual_report' || item.key === 'metrics')
    return 'json'
  if (item.key === 'html') return 'embed'
  if (item.key === 'epub') return null
  // primary/secondary follow the primary artifact's suffix: a PDF job's
  // "Translated document" previews as a PDF; anything else is download-only.
  const ext = item.filename.split('.').pop()?.toLowerCase() ?? ''
  if (ext === 'pdf' || ext === 'html' || ext === 'htm') return 'embed'
  return null
}

export type DeliverableGroup = 'reading' | 'l10n' | 'audit'

/**
 * Who each deliverable is *for*. The flat grid of eight equal buttons read as
 * one undifferentiated pile of downloads — the reader-facing book, the
 * translator hand-off file and the engine's audit JSONs carry entirely
 * different intents, and the section now names those intents instead of
 * leaving the operator to decode file formats.
 */
export function deliverableGroup(key: string): DeliverableGroup {
  if (key === 'xliff') return 'l10n'
  if (key === 'contract' || key === 'quality_report' || key === 'visual_report' || key === 'metrics')
    return 'audit'
  return 'reading'
}

/** The friendly display name; falls back to the backend label, then the key. */
export function deliverableName(key: string, fallback: string, t: Translation): string {
  switch (key) {
    case 'primary':
      return t.quality.targetPdf
    case 'secondary':
      return t.quality.secondaryDoc
    case 'epub':
      return t.quality.epub
    case 'contract':
      return t.quality.contractJson
    case 'quality_report':
      return t.quality.qualityReportJson
    case 'visual_report':
      return t.quality.visualReportJson
    case 'metrics':
      return t.quality.metricsJson
    case 'html':
      return t.quality.htmlView
    case 'xliff':
      return t.quality.xliffDoc
    default:
      return fallback
  }
}

/** One plain-language line: what this file is, and who would open it. */
export function deliverableDesc(key: string, t: Translation): string {
  switch (key) {
    case 'primary':
      return t.quality.descPrimary
    case 'secondary':
      return t.quality.descSecondary
    case 'epub':
      return t.quality.descEpub
    case 'html':
      return t.quality.descHtml
    case 'xliff':
      return t.quality.descXliff
    case 'contract':
      return t.quality.descContract
    case 'quality_report':
      return t.quality.descQualityReport
    case 'visual_report':
      return t.quality.descVisualReport
    case 'metrics':
      return t.quality.descMetrics
    default:
      return ''
  }
}

/** "1.2 MB" / "843 KB" — the size tells the reader what they are about to get. */
export function fmtSize(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
