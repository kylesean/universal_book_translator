import React from 'react'
import { useI18n } from '@/i18n/useI18n'

export type LogLevel = 'ALL' | 'INFO' | 'WARN' | 'ERROR'

export interface LogEntry {
  text: string
  level: 'INFO' | 'WARN' | 'ERROR'
  at?: number
  eventType?: string
}

export const FAITHFUL_DELIVERY_MODES = new Set(['faithful'])
export const READER_PRODUCT_KEYS = new Set(['primary', 'secondary', 'epub', 'html'])

export const getFormatBadge = (filename?: string): string => {
  if (!filename) return 'FILE'
  const ext = filename.split('.').pop()?.toLowerCase() ?? ''
  switch (ext) {
    case 'pdf': return 'PDF'
    case 'epub': return 'EPUB'
    case 'docx': return 'DOCX'
    case 'md':
    case 'markdown': return 'MD'
    case 'html':
    case 'htm': return 'HTML'
    case 'xliff': return 'XLIFF'
    case 'json': return 'JSON'
    default: return ext.toUpperCase() || 'FILE'
  }
}

export const formatBytes = (bytes?: number): string => {
  if (!bytes || bytes <= 0 || isNaN(bytes)) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  return `${(bytes / Math.pow(1024, i)).toFixed(i > 0 ? 1 : 0)} ${units[i]}`
}

export function displayFileName(rawName?: string | null): string {
  if (!rawName) return '—'
  const match = rawName.match(/^(?:\d{8}-\d{6}-[a-f0-9]{8}-)?(.+)$/)
  return match ? match[1] : rawName
}

export function compactJobId(jobId: string): string {
  const separator = jobId.indexOf('_')
  if (separator < 0) return jobId
  return `${jobId.slice(0, separator + 1)}…${jobId.slice(-6)}`
}

export function formatDuration(startStr?: string | null, endStr?: string | null): string | null {
  if (!startStr || !endStr) return null
  const start = new Date(startStr.replace(' ', 'T')).getTime()
  const end = new Date(endStr.replace(' ', 'T')).getTime()
  if (isNaN(start) || isNaN(end) || end <= start) return null
  const sec = Math.round((end - start) / 1000)
  const m = Math.floor(sec / 60)
  const s = sec % 60
  if (m === 0) return `${s}s`
  if (m < 60) return `${m}m ${s}s`
  const h = Math.floor(m / 60)
  const rm = m % 60
  return `${h}h ${rm}m`
}

export function formatEngineName(raw: string): string {
  if (!raw) return ''
  let name = raw.replace(/^dm/, '')
  if (name.includes('gemma')) name = name.replace(/4e4b/i, ' 4B').replace(/27b/i, ' 27B')
  else if (name.includes('qwen')) name = name.replace(/25/i, ' 2.5')
  else if (name.includes('deepseek')) name = name.replace(/v3/i, ' V3').replace(/r1/i, ' R1')
  return name.charAt(0).toUpperCase() + name.slice(1)
}

export function parseJobIdMeta(jobId: string) {
  const parts = jobId.split('_')
  let targetLang = ''
  let preset = ''
  let engine = ''
  if (parts.length >= 3) {
    targetLang = parts[2].toUpperCase()
  }
  for (const part of parts) {
    if (part.startsWith('pr')) {
      preset = part.slice(2)
    } else if (part.startsWith('eng')) {
      const engMatch = part.match(/^eng([a-z0-9]+?)(?:rm[a-z0-9]+)?(?:ps[a-z0-9]+)?$/i)
      engine = engMatch ? engMatch[1] : part.slice(3)
    }
  }
  return { targetLang, preset, engine: formatEngineName(engine) }
}

export function renderHighlightedText(text: string, query: string): React.ReactNode {
  if (!query.trim()) return text
  const escaped = query.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const parts = text.split(new RegExp(`(${escaped})`, 'gi'))
  if (parts.length === 1) return text
  return parts.map((part, i) =>
    part.toLowerCase() === query.toLowerCase()
      ? React.createElement(
          'mark',
          {
            key: i,
            className: 'bg-amber-400/35 text-amber-100 rounded-xs px-0.5 font-semibold',
          },
          part
        )
      : part
  )
}

export const liveProgressPct = (finalized: number, drafted: number, total: number): number => {
  if (total <= 0) return 0
  const finalizedWork = Math.max(0, Math.min(total, finalized))
  const draftedWork = Math.max(0, Math.min(total, drafted))
  if (finalizedWork >= total) return 100
  return Math.min(99.9, Math.round((Math.max(finalizedWork, draftedWork) / total) * 1000) / 10)
}

export const jobStatusLabel = (t: ReturnType<typeof useI18n>['t'], status: string): string =>
  t.mission.jobStatuses[status] ?? status.toUpperCase()

export const statusBadgeClass = (status: string): string =>
  status === 'completed'
    ? 'border-emerald-600/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
    : status === 'failed' || status === 'cancelled'
      ? 'border-destructive/25 bg-destructive/10 text-destructive'
      : status === 'needs_human' || status === 'blocked_human'
        ? 'border-amber-600/25 bg-amber-500/10 text-amber-700 dark:text-amber-400'
        : 'border-sky-600/25 bg-sky-500/10 text-sky-700 dark:text-sky-400'

export const stageLabel = (t: ReturnType<typeof useI18n>['t'], stage: string): string =>
  t.mission.stageNames[stage] ?? t.mission.jobStatuses[stage] ?? stage.toUpperCase()

export const deliveryModeLabel = (
  t: ReturnType<typeof useI18n>['t'],
  mode: string | null | undefined,
): string =>
  !mode
    ? ''
    : FAITHFUL_DELIVERY_MODES.has(mode)
      ? t.mission.deliveryModeBadgeFaithful
      : t.mission.deliveryModeBadgeReadable

export const degradeReasonText = (template: string, params: Record<string, unknown>): string =>
  template.replace(/\{(\w+)\}/g, (whole, key: string) => {
    const value = params[key]
    if (value === undefined || value === null) return whole
    if (key === 'text_trust' || key === 'threshold') {
      const numeric = Number(value)
      return Number.isFinite(numeric) ? numeric.toFixed(2) : String(value)
    }
    return String(value)
  })

export const failureSummary = (
  t: ReturnType<typeof useI18n>['t'],
  error: string,
): string => {
  if (/transport_unreachable|ConnectError|All connection attempts failed/i.test(error)) {
    return t.mission.failureProviderUnreachable
  }
  if (/ModelProviderError|ProviderConfigError|status_code|no router/i.test(error)) {
    return t.mission.failureConfig
  }
  if (/IntegrityViolationError|completion ratio|completion floor/i.test(error)) {
    return t.mission.failureIntegrity
  }
  if (/DocumentParseError|UnsupportedDocumentFormatError|Document .* produced 0/i.test(error)) {
    return t.mission.failureDocument
  }
  if (/timeout|timed out|connection|503|502|504/i.test(error)) {
    return t.mission.failureProvider
  }
  return t.mission.failureGeneric
}

export function formatTimestamp(value: string | null | undefined): string {
  if (!value) return '—'
  const normalized = value.includes('T') ? value : value.replace(' ', 'T')
  const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/.test(normalized)
  const parsed = new Date(hasZone ? normalized : `${normalized}Z`)
  if (Number.isNaN(parsed.getTime())) return value
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(parsed.getMonth() + 1)}-${pad(parsed.getDate())} ${pad(parsed.getHours())}:${pad(parsed.getMinutes())}`
}

export function formatClock(value: number): string {
  const parsed = new Date(value)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${pad(parsed.getHours())}:${pad(parsed.getMinutes())}:${pad(parsed.getSeconds())}`
}
