import React, { useEffect, useState } from 'react'
import { createPortal } from 'react-dom'
import { X, Download, LoaderCircle, FileText, AlertTriangle } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import {
  deliverableDownloadUrl,
  triggerBrowserDownload,
  fetchDeliverableText,
  getJobReport,
  getVisualReport,
  type DeliverableItem,
} from '@/api/client'
import { useI18n } from '@/i18n/useI18n'
import { deliverableName, fmtSize, previewKind } from './artifactMeta'

/**
 * In-console preview for one deliverable.
 *
 * The download section used to be a wall of identical buttons — every artifact,
 * from the reader-facing PDF to the engine's metrics JSON, could only "be
 * downloaded". A reader who just wants to check the translation should not
 * leave the browser for a local PDF viewer, and a JSON sidecar is meaningless
 * as a raw file: the drawer renders each kind the way its reader thinks —
 * PDF/HTML in an iframe, XLIFF as a source/target table, JSON as a digest with
 * the raw payload behind a disclosure.
 *
 * Embeds point at the `?inline=1` download URL, which the server serves under
 * a CSP `sandbox` directive: engine-generated HTML is built from untrusted
 * book text, so the preview runs scriptless in an opaque origin.
 */

type Status = 'loading' | 'ready' | 'error'

/** One flattened top-level row of a JSON sidecar. */
function flattenRows(data: Record<string, unknown>): Array<[string, string]> {
  return Object.entries(data).map(([key, value]) => {
    if (value === null || value === undefined) return [key, '—'] as [string, string]
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean')
      return [key, String(value)] as [string, string]
    const text = JSON.stringify(value)
    return [key, text.length > 140 ? `${text.slice(0, 140)}…` : text] as [string, string]
  })
}

function RawJson({ data }: { data: unknown }) {
  const { t } = useI18n()
  return (
    <details className="mt-3">
      <summary className="text-xs text-[var(--ink-muted)] cursor-pointer select-none hover:text-[var(--ink-secondary)]">
        {t.quality.previewRawJson}
      </summary>
      <pre className="mt-2 text-[11px] leading-relaxed font-mono bg-[var(--paper-subsurface)] border border-[var(--paper-border)] rounded p-2.5 overflow-auto max-h-72">
        {JSON.stringify(data, null, 2)}
      </pre>
    </details>
  )
}

/** A digest of the quality report — the numbers a human would look up first. */
function QualityReportDigest({ data }: { data: Record<string, any> }) {
  const { t } = useI18n()
  const summary = data.summary ?? {}
  const placeholder = data.placeholder ?? {}
  const terminology = data.terminology ?? {}
  const entity = data.entity_consistency ?? {}
  const coverage = data.render_coverage ?? {}
  const verdict: string | null = data.delivery_verdict ?? null
  const pct = (v: unknown) =>
    typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : '—'
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Badge
          variant={verdict === 'passed' ? 'success' : verdict === 'warning' ? 'warning' : 'destructive'}
          dot
        >
          {verdict === 'passed'
            ? t.quality.verdictPass
            : verdict === 'warning'
              ? t.quality.verdictWarning
              : verdict === 'blocked'
                ? t.quality.verdictBlocked
                : '—'}
        </Badge>
        <span className="text-xs font-mono text-[var(--ink-muted)]">
          {t.quality.passRate.replace('{value}', pct(summary.pass_rate))}
        </span>
      </div>
      <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1 text-xs font-mono">
        {[
          [t.quality.pillar1Title, `${placeholder.masked_spans ?? 0} / ${pct(placeholder.retention_rate)}`],
          [t.quality.pillar2Title, `${pct(terminology.term_recall)} · ${entity.terms_with_drift ?? 0}/${entity.terms_audited ?? 0}`],
          [t.quality.pillar4Title, `${pct(coverage.render_coverage)} · fail-closed ${coverage.fail_closed_blocks ?? 0}`],
          [
            t.quality.delivCostLabel,
            summary.estimated_cost_usd === null || summary.estimated_cost_usd === undefined
              ? t.quality.costNotPriced
              : `$${Number(summary.estimated_cost_usd).toFixed(2)}`,
          ],
        ].map(([label, value]) => (
          <div key={String(label)} className="flex justify-between gap-3 border-b border-[var(--paper-border)] pb-1">
            <dt className="text-[var(--ink-muted)] shrink-0">{label}</dt>
            <dd className="text-right">{value}</dd>
          </div>
        ))}
      </dl>
      <RawJson data={data} />
    </div>
  )
}

/** A digest of the visual gate report — what ran, on which pages, what it saw. */
function VisualReportDigest({ data }: { data: Record<string, any> }) {
  const { t } = useI18n()
  const stats = data.stats ?? {}
  const findings: Array<Record<string, any>> = data.findings ?? []
  const tierLabel = (tier: string): string =>
    (t.quality.visualTierNames as Record<string, string>)[tier] ?? tier
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Badge variant={data.passed ? 'success' : 'destructive'} dot>
          {data.passed ? t.quality.verdictPass : t.quality.verdictBlocked}
        </Badge>
        <span className="text-xs font-mono text-[var(--ink-muted)]">
          {Array.isArray(data.tiers_ran) ? data.tiers_ran.map(tierLabel).join(' · ') : ''}
        </span>
      </div>
      <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1 text-xs font-mono">
        {[
          [t.quality.delivPagesScanned, `${stats.sampled ?? 0} / ${stats.total_pages ?? 0}`],
          [t.quality.delivOverlapPairs, stats.overlap_pairs ?? 0],
          [t.quality.delivTruncations, stats.truncation_count ?? 0],
          [t.quality.delivTypeIssues, (stats.type_illegible_pages ?? 0) + (stats.type_collapsed_pages ?? 0)],
          [t.quality.flaggedPages, findings.length],
        ].map(([label, value]) => (
          <div key={String(label)} className="flex justify-between gap-3 border-b border-[var(--paper-border)] pb-1">
            <dt className="text-[var(--ink-muted)] shrink-0">{label}</dt>
            <dd className="text-right">{value}</dd>
          </div>
        ))}
      </dl>
      {findings.length > 0 && (
        <ul className="space-y-1 text-xs max-h-56 overflow-y-auto">
          {findings.slice(0, 30).map((finding, index) => (
            <li key={index} className="flex gap-2">
              <span className="font-mono text-[var(--ink-muted)] shrink-0">
                {finding.page != null ? `P${finding.page}` : '—'}
              </span>
              <span className="text-[var(--ink-secondary)]">
                <span className="font-mono text-[#b45309]">{finding.code}</span> {finding.message}
              </span>
            </li>
          ))}
        </ul>
      )}
      <RawJson data={data} />
    </div>
  )
}

/** A flat key/value read of the contract or metrics sidecar. */
function FlatJsonDigest({ data }: { data: Record<string, unknown> }) {
  return (
    <div>
      <dl className="grid grid-cols-1 gap-x-6 gap-y-1 text-xs font-mono">
        {flattenRows(data).map(([key, value]) => (
          <div key={key} className="flex justify-between gap-3 border-b border-[var(--paper-border)] pb-1">
            <dt className="text-[var(--ink-muted)] shrink-0">{key}</dt>
            <dd className="text-right break-all">{value}</dd>
          </div>
        ))}
      </dl>
      <RawJson data={data} />
    </div>
  )
}

/** XLIFF rendered as the two-column table its reader (a translator) expects. */
function XliffTable({ text }: { text: string }) {
  const { t } = useI18n()
  const pairs = React.useMemo(() => {
    const doc = new DOMParser().parseFromString(text, 'application/xml')
    const out: Array<{ source: string; target: string }> = []
    const segs = doc.getElementsByTagName('segment')
    if (segs.length > 0) {
      // XLIFF 2.x: <unit><segment><source/><target/></segment></unit>
      for (const seg of Array.from(segs)) {
        const source = seg.getElementsByTagName('source')[0]?.textContent?.trim() ?? ''
        const target = seg.getElementsByTagName('target')[0]?.textContent?.trim() ?? ''
        if (source || target) out.push({ source, target })
      }
    } else {
      // XLIFF 1.2: <body><trans-unit><source/><target/></trans-unit></body>
      const units = doc.getElementsByTagName('trans-unit')
      for (const unit of Array.from(units)) {
        const source = unit.getElementsByTagName('source')[0]?.textContent?.trim() ?? ''
        const target = unit.getElementsByTagName('target')[0]?.textContent?.trim() ?? ''
        if (source || target) out.push({ source, target })
      }
    }
    return out
  }, [text])
  const shown = pairs.slice(0, 200)
  return (
    <div>
      <div className="text-xs text-[var(--ink-muted)] mb-2">
        {t.quality.previewSegmentsTotal
          .replace('{total}', String(pairs.length))
          .replace('{shown}', String(shown.length))}
      </div>
      <table className="w-full text-xs text-left border-collapse">
        <thead className="text-[var(--ink-muted)]">
          <tr>
            <th className="py-1 pr-3 font-semibold w-1/2">{t.assets.colSource}</th>
            <th className="py-1 font-semibold w-1/2">{t.assets.colTarget}</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--paper-border)] align-top">
          {shown.map((pair, index) => (
            <tr key={index}>
              <td className="py-1.5 pr-3 text-[var(--ink-secondary)] leading-relaxed">{pair.source}</td>
              <td className="py-1.5 text-[var(--ink-primary)] leading-relaxed">{pair.target}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function PreviewDrawer({
  jobId,
  item,
  onClose,
}: {
  jobId: string
  item: DeliverableItem
  onClose: () => void
}) {
  const { t } = useI18n()
  const kind = previewKind(item)
  const [status, setStatus] = useState<Status>('loading')
  const [error, setError] = useState<string | null>(null)
  const [frameLoaded, setFrameLoaded] = useState(false)
  const [jsonText, setJsonText] = useState<string | null>(null)

  // Parsed payloads load only for the kinds that need them; an embed streams
  // straight from the server into the iframe. The parent mounts this drawer
  // with `key={item.key}`, so every open starts from the initial states and
  // the effect only ever runs for one fresh item.
  useEffect(() => {
    if (kind !== 'xliff' && kind !== 'json') return
    let cancelled = false
    ;(async () => {
      try {
        if (kind === 'xliff') {
          const text = await fetchDeliverableText(jobId, item.key)
          if (!cancelled) {
            setJsonText(text)
            setStatus('ready')
          }
          return
        }
        // The two reports have dedicated endpoints that return the exact file
        // contents; everything else comes off the download route as text.
        if (item.key === 'quality_report') {
          const data = await getJobReport(jobId)
          if (!cancelled) {
            setJsonText(JSON.stringify(data))
            setStatus('ready')
          }
          return
        }
        if (item.key === 'visual_report') {
          const data = await getVisualReport(jobId)
          if (!cancelled) {
            setJsonText(JSON.stringify(data))
            setStatus('ready')
          }
          return
        }
        const text = await fetchDeliverableText(jobId, item.key)
        if (!cancelled) {
          setJsonText(text)
          setStatus('ready')
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Preview failed')
          setStatus('error')
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [jobId, item.key, kind])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKey)
    const prevOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = prevOverflow
    }
  }, [onClose])

  if (kind === null) {
    // The button is hidden for non-previewable items; reaching here means a
    // stale caller. Say so rather than rendering an empty shell.
    return null
  }

  const name = deliverableName(item.key, item.label, t)

  let body: React.ReactNode
  if (kind === 'embed') {
    body = (
      <div className="relative flex-1 min-h-0 bg-white">
        {!frameLoaded && (
          <div className="absolute inset-0 flex items-center justify-center gap-2 text-xs text-[var(--ink-muted)]">
            <LoaderCircle className="h-4 w-4 animate-spin" />
            {t.quality.previewLoading}
          </div>
        )}
        <iframe
          src={deliverableDownloadUrl(jobId, item.key, { inline: true })}
          title={`${name} · ${t.quality.previewAction}`}
          className="w-full h-full border-0"
          onLoad={() => setFrameLoaded(true)}
        />
      </div>
    )
  } else if (status === 'error') {
    body = (
      <div className="flex-1 flex flex-col items-center justify-center gap-3 p-8 text-center">
        <AlertTriangle className="h-6 w-6 text-[#b45309]" />
        <p className="text-xs text-[var(--ink-secondary)] max-w-sm">{error}</p>
        <Button variant="secondary" size="sm" onClick={() => triggerBrowserDownload(deliverableDownloadUrl(jobId, item.key), item.filename)}>
          <Download className="h-3.5 w-3.5" />
          {t.quality.previewDownloadInstead}
        </Button>
      </div>
    )
  } else if (status === 'loading') {
    body = (
      <div className="flex-1 flex items-center justify-center gap-2 text-xs text-[var(--ink-muted)]">
        <LoaderCircle className="h-4 w-4 animate-spin" />
        {t.quality.previewLoading}
      </div>
    )
  } else if (kind === 'xliff' && jsonText !== null) {
    body = (
      <div className="flex-1 overflow-auto p-4">
        <XliffTable text={jsonText} />
      </div>
    )
  } else if (jsonText !== null) {
    let parsed: Record<string, unknown>
    try {
      parsed = JSON.parse(jsonText) as Record<string, unknown>
    } catch {
      parsed = { content: jsonText }
    }
    body = (
      <div className="flex-1 overflow-auto p-4">
        {item.key === 'quality_report' && <QualityReportDigest data={parsed as Record<string, any>} />}
        {item.key === 'visual_report' && <VisualReportDigest data={parsed as Record<string, any>} />}
        {item.key !== 'quality_report' && item.key !== 'visual_report' && <FlatJsonDigest data={parsed} />}
      </div>
    )
  } else {
    body = null
  }

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4 sm:p-8"
      role="dialog"
      aria-modal="true"
      aria-label={name}
    >
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div className="relative flex flex-col w-full max-w-4xl h-[85vh] rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] shadow-xl overflow-hidden">
        <div className="flex items-center justify-between gap-3 px-4 py-2.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)]">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-sm font-semibold text-[var(--ink-primary)]">
              <FileText className="h-4 w-4 text-[var(--ink-secondary)] shrink-0" />
              <span className="truncate">{name}</span>
              <span className="text-xs font-mono text-[var(--ink-muted)] font-normal shrink-0">
                {item.filename} · {fmtSize(item.size_bytes)}
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => triggerBrowserDownload(deliverableDownloadUrl(jobId, item.key), item.filename)}
            >
              <Download className="h-3.5 w-3.5" />
              {t.quality.downloadAction}
            </Button>
            <button
              onClick={onClose}
              aria-label={t.quality.previewClose}
              autoFocus
              className="h-7 w-7 inline-flex items-center justify-center rounded-[5px] text-[var(--ink-secondary)] hover:bg-[var(--paper-surface)] hover:text-[var(--ink-primary)] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--ink-primary)]"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
        {body}
      </div>
    </div>,
    document.body
  )
}
