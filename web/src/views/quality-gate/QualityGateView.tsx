import React, { useEffect, useState } from 'react'
import {
  ShieldCheck,
  FileCheck2,
  Download,
  ArrowUpRight,
  CheckCircle2,
  AlertTriangle,
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import {
  getJobReport,
  listDeliverables,
  deliverableDownloadUrl,
  type DeliverableItem,
} from '@/api/client'
import { useI18n } from '@/i18n/I18nContext'

interface QualityGateViewProps {
  jobId: string | null
  onOpenReview: (jobId: string) => void
}

interface Pillar {
  title: string
  desc: string
  metric: string
  status: 'pass' | 'warn' | 'fail'
}

/** i18n labels for the deliverable keys the backend serves. */
function deliverableLabel(key: string, fallback: string, t: ReturnType<typeof useI18n>['t']): string {
  switch (key) {
    case 'primary':
      return t.quality.targetPdf
    case 'epub':
      return t.quality.epub
    case 'contract':
      return t.quality.contractJson
    default:
      return fallback
  }
}

export function QualityGateView({ jobId, onOpenReview }: QualityGateViewProps) {
  const { t } = useI18n()
  const [report, setReport] = useState<Record<string, any> | null>(null)
  const [reportError, setReportError] = useState<string | null>(null)
  const [deliverables, setDeliverables] = useState<DeliverableItem[]>([])

  useEffect(() => {
    if (!jobId) return
    let isMounted = true

    getJobReport(jobId)
      .then((rep) => {
        if (isMounted) {
          setReport(rep)
          setReportError(null)
        }
      })
      .catch((err) => {
        if (isMounted) setReportError(err instanceof Error ? err.message : 'Report unavailable')
      })

    listDeliverables(jobId)
      .then((items) => {
        if (isMounted) setDeliverables(items)
      })
      .catch(() => {
        if (isMounted) setDeliverables([])
      })

    return () => {
      isMounted = false
    }
  }, [jobId])

  if (!jobId) {
    return (
      <div className="flex-1 flex items-center justify-center p-8 text-center text-[var(--ink-muted)]">
        <div className="max-w-sm space-y-2">
          <ShieldCheck className="h-8 w-8 mx-auto text-[var(--paper-border-hover)]" />
          <h2 className="text-sm font-semibold text-[var(--ink-primary)]">{t.quality.noJobTitle}</h2>
          <p className="text-xs text-[var(--ink-secondary)] leading-relaxed">{t.quality.noJobDesc}</p>
        </div>
      </div>
    )
  }

  const summary = report?.summary ?? {}
  const score = report?.score_metrics ?? {}
  const placeholder = report?.placeholder ?? {}
  const terminology = report?.terminology ?? {}
  const entity = report?.entity_consistency ?? {}
  const fidelity = report?.fidelity ?? {}
  const coverage = report?.render_coverage ?? {}

  const failedBlocks = summary.failed_blocks ?? 0
  const unsuitable = report?.delivery_status === 'UNSUITABLE_FOR_DELIVERY'
  const isPassed = !unsuitable && failedBlocks === 0
  const fidelityScore = Math.round((summary.pass_rate ?? 0) * 1000) / 10

  const pct = (value: number | undefined, digits = 1) =>
    value === undefined ? '—' : `${(value * 100).toFixed(digits)}%`

  const pillars: Pillar[] = [
    {
      title: t.quality.pillar1Title,
      desc: t.quality.pillar1Desc,
      metric: `${placeholder.corrupt_spans ?? 0} corrupt / ${placeholder.masked_spans ?? 0} masked (retention ${pct(placeholder.retention_rate)})`,
      status: (placeholder.corrupt_spans ?? 0) > 0 ? 'fail' : 'pass',
    },
    {
      title: t.quality.pillar2Title,
      desc: t.quality.pillar2Desc,
      metric: `recall ${pct(terminology.term_recall)} · drift ${entity.terms_with_drift ?? 0}/${entity.terms_audited ?? 0}`,
      status: (entity.terms_with_drift ?? 0) > 0 ? 'warn' : 'pass',
    },
    {
      title: t.quality.pillar3Title,
      desc: t.quality.pillar3Desc,
      metric: `${fidelity.pages_measured ?? 0} pages · residual ${pct(fidelity.non_text_residual, 2)}`,
      status: (fidelity.non_text_residual ?? 0) > 0.02 ? 'warn' : 'pass',
    },
    {
      title: t.quality.pillar4Title,
      desc: t.quality.pillar4Desc,
      metric: `fail-closed ${coverage.fail_closed_blocks ?? 0} · coverage ${pct(coverage.render_coverage)}`,
      status: (coverage.fail_closed_blocks ?? 0) > 0 ? 'fail' : 'pass',
    },
    {
      title: t.quality.pillar5Title,
      desc: t.quality.pillar5Desc,
      metric:
        summary.estimated_cost_usd === null || summary.estimated_cost_usd === undefined
          ? 'cost not priced'
          : `$${Number(summary.estimated_cost_usd).toFixed(2)} · cache ${pct(summary.cache_hit_rate)}`,
      status: 'pass',
    },
  ]

  const handleDownload = (key: string) => {
    window.open(deliverableDownloadUrl(jobId, key), '_blank')
  }

  const statusBadge = (status: Pillar['status']) =>
    status === 'pass' ? 'success' : status === 'warn' ? 'warning' : 'destructive'

  return (
    <div className="flex-1 overflow-y-auto px-8 py-7 space-y-7 max-w-5xl mx-auto w-full">
      {/* View Header */}
      <div className="flex items-center justify-between border-b border-[var(--paper-border)] pb-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[var(--ink-primary)]">
            {t.quality.title}
          </h1>
          <p className="text-xs text-[var(--ink-secondary)] mt-1">{t.quality.subtitle}</p>
        </div>

        <Button onClick={() => onOpenReview(jobId)} variant="secondary" size="sm">
          {t.mission.openWorkbench}
          <ArrowUpRight className="h-3.5 w-3.5 ml-1 text-[var(--ink-secondary)]" />
        </Button>
      </div>

      {reportError && !report && (
        <div className="p-3 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] text-xs text-[var(--ink-secondary)] flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-[#b45309] shrink-0" />
          {reportError}
        </div>
      )}

      {/* Decision Banner */}
      <div className="p-5 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] flex items-center justify-between shadow-2xs">
        <div className="flex items-center gap-3.5">
          <div
            className={`h-10 w-10 rounded-[6px] bg-[var(--paper-subsurface)] border border-[var(--paper-border)] flex items-center justify-center ${
              isPassed ? 'text-[#15803d]' : 'text-[#b91c1c]'
            }`}
          >
            {isPassed ? <CheckCircle2 className="h-6 w-6" /> : <AlertTriangle className="h-6 w-6" />}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-sm font-bold text-[var(--ink-primary)]">
                {isPassed ? t.quality.decisionReady : t.quality.decisionBlocked}
              </span>
              <Badge variant={isPassed ? 'success' : 'destructive'} dot>
                {isPassed ? 'PASS' : 'BLOCKED'}
              </Badge>
            </div>
            <p className="text-xs text-[var(--ink-secondary)] mt-0.5">
              {isPassed ? t.quality.allPassedDesc : t.quality.blockedDesc}
            </p>
          </div>
        </div>

        <div className="text-right">
          <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
            {t.quality.fidelityScore}
          </div>
          <div className="text-2xl font-bold font-mono text-[var(--ink-primary)] mt-0.5">
            {report ? fidelityScore.toFixed(1) : '—'}{' '}
            <span className="text-xs text-[var(--ink-muted)] font-normal">/ 100</span>
          </div>
          {report && (
            <div className="text-[10px] font-mono text-[var(--ink-muted)] mt-0.5">
              pass rate {pct(summary.pass_rate)} · avg QE {Number(score.avg_qe ?? 0).toFixed(3)}
            </div>
          )}
        </div>
      </div>

      {/* Formal Verification Checklist */}
      <div className="rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] overflow-hidden shadow-2xs">
        <div className="px-4 py-2.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
          FORMAL VERIFICATION GATEWAYS (5 PILLARS)
        </div>
        <div className="divide-y divide-[var(--paper-border)]">
          {pillars.map((item, idx) => (
            <div
              key={idx}
              className="p-4 flex items-center justify-between hover:bg-[var(--paper-subsurface)]/60 transition-colors"
            >
              <div>
                <div className="text-xs font-semibold text-[var(--ink-primary)]">{item.title}</div>
                <div className="text-[11px] text-[var(--ink-secondary)] mt-0.5 leading-relaxed">
                  {item.desc}
                </div>
              </div>
              <div className="text-right flex items-center gap-3">
                <span className="font-mono text-xs text-[var(--ink-primary)] font-medium">
                  {item.metric}
                </span>
                <Badge variant={statusBadge(item.status)} dot>
                  {item.status.toUpperCase()}
                </Badge>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Deliverable Download Section */}
      <div className="rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] p-5 space-y-4 shadow-2xs">
        <div>
          <h3 className="text-xs font-bold text-[var(--ink-primary)] tracking-tight">
            {t.quality.deliverablesTitle}
          </h3>
          <p className="text-[11px] text-[var(--ink-secondary)] mt-0.5">
            {t.quality.deliverablesSubtitle}
          </p>
        </div>

        {deliverables.length === 0 ? (
          <div className="text-xs text-[var(--ink-muted)] py-2">
            No deliverables are available for this job yet.
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
            {deliverables.map((item) => (
              <Button
                key={item.key}
                onClick={() => handleDownload(item.key)}
                variant="secondary"
                className="h-10 justify-start px-3 text-xs"
              >
                {item.key === 'contract' ? (
                  <FileCheck2 className="h-3.5 w-3.5 mr-2 text-[#15803d]" />
                ) : (
                  <Download className="h-3.5 w-3.5 mr-2 text-[var(--ink-secondary)]" />
                )}
                <span className="truncate">{deliverableLabel(item.key, item.label, t)}</span>
              </Button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
