import React, { useEffect, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import {
  ShieldCheck,
  FileCheck2,
  Download,
  ArrowUpRight,
  CheckCircle2,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import {
  getJobReport,
  getVisualReport,
  listDeliverables,
  deliverableDownloadUrl,
  pagePreviewUrl,
  type DeliverableItem,
} from '@/api/client'
import { useI18n } from '@/i18n/useI18n'

type PillarStatus = 'pass' | 'warn' | 'fail'

interface Pillar {
  title: string
  desc: string
  metric: string
  status: PillarStatus
  /** 0-100 axis value for the radar. */
  score: number
  detail: React.ReactNode
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

/** A five-axis radar drawn as inline SVG (no chart dependency). */
function RadarChart({ axes }: { axes: { label: string; value: number }[] }) {
  const size = 200
  const cx = size / 2
  const cy = size / 2
  const radius = 70
  const point = (index: number, ratio: number): [number, number] => {
    const angle = ((-90 + index * (360 / axes.length)) * Math.PI) / 180
    return [cx + radius * ratio * Math.cos(angle), cy + radius * ratio * Math.sin(angle)]
  }
  const polygon = axes
    .map((axis, index) => point(index, Math.min(1, Math.max(0, axis.value / 100))))
    .map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`)
    .join(' ')

  return (
    <svg viewBox={`0 0 ${size} ${size}`} className="w-52 h-52 shrink-0" role="img">
      {[0.25, 0.5, 0.75, 1].map((ratio) => (
        <polygon
          key={ratio}
          points={axes
            .map((_, index) => point(index, ratio))
            .map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`)
            .join(' ')}
          fill="none"
          stroke="var(--paper-border)"
          strokeWidth={0.8}
        />
      ))}
      {axes.map((_, index) => {
        const [x, y] = point(index, 1)
        return (
          <line
            key={index}
            x1={cx}
            y1={cy}
            x2={x}
            y2={y}
            stroke="var(--paper-border)"
            strokeWidth={0.8}
          />
        )
      })}
      <polygon
        points={polygon}
        fill="rgba(21,128,61,0.18)"
        stroke="#15803d"
        strokeWidth={1.4}
      />
      {axes.map((axis, index) => {
        const [x, y] = point(index, 1.22)
        return (
          <text
            key={axis.label}
            x={x}
            y={y}
            textAnchor="middle"
            dominantBaseline="middle"
            fontSize={9}
            fontFamily="monospace"
            fill="var(--ink-muted)"
          >
            {axis.label} {Math.round(axis.value)}
          </text>
        )
      })}
    </svg>
  )
}

export function QualityGateView() {
  const { t } = useI18n()
  const navigate = useNavigate()
  const { jobId: routeJobId } = useParams<{ jobId: string }>()
  const jobId = routeJobId ?? null
  const onOpenReview = (id: string) => navigate(`/jobs/${id}/review`)
  const [report, setReport] = useState<Record<string, any> | null>(null)
  const [reportError, setReportError] = useState<string | null>(null)
  const [deliverables, setDeliverables] = useState<DeliverableItem[]>([])
  const [visual, setVisual] = useState<Record<string, any> | null>(null)
  const [expanded, setExpanded] = useState<number | null>(null)

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

    getVisualReport(jobId)
      .then((rep) => {
        if (isMounted) setVisual(rep)
      })
      .catch(() => {
        if (isMounted) setVisual(null)
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
  const repair = report?.repair_breakdown ?? {}
  const config = report?.config_snapshot ?? {}

  const failedBlocks = summary.failed_blocks ?? 0
  const unsuitable = report?.delivery_status === 'UNSUITABLE_FOR_DELIVERY'
  const isPassed = !unsuitable && failedBlocks === 0
  const fidelityScore = Math.round((summary.pass_rate ?? 0) * 1000) / 10

  const pct = (value: number | undefined, digits = 1) =>
    value === undefined ? '—' : `${(value * 100).toFixed(digits)}%`

  const clampScore = (value: number | undefined, fallback = 100) =>
    value === undefined || Number.isNaN(value) ? fallback : Math.min(100, Math.max(0, value * 100))

  const topDrifted: Array<Record<string, any>> = entity.top_drifted ?? []
  const visualFindings: Array<Record<string, any>> = visual?.findings ?? []
  const flaggedPages = Array.from(
    new Set(
      visualFindings
        .map((finding) => finding.page)
        .filter((page): page is number => typeof page === 'number')
    )
  ).sort((a, b) => a - b)

  const pillarStatus = (bad: boolean, warn = false): PillarStatus =>
    bad ? 'fail' : warn ? 'warn' : 'pass'

  const pillars: Pillar[] = [
    {
      title: t.quality.pillar1Title,
      desc: t.quality.pillar1Desc,
      metric: `${placeholder.corrupt_spans ?? 0} corrupt / ${placeholder.masked_spans ?? 0} masked (retention ${pct(placeholder.retention_rate)})`,
      status: (placeholder.corrupt_spans ?? 0) > 0 ? 'fail' : 'pass',
      score: clampScore(placeholder.retention_rate),
      detail: (
        <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-[11px] font-mono">
          <div className="flex justify-between">
            <dt className="text-[var(--ink-muted)]">masked_spans</dt>
            <dd>{placeholder.masked_spans ?? 0}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-[var(--ink-muted)]">corrupt_spans</dt>
            <dd>{placeholder.corrupt_spans ?? 0}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-[var(--ink-muted)]">masked_blocks</dt>
            <dd>{placeholder.masked_blocks ?? 0}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-[var(--ink-muted)]">corrupt_blocks</dt>
            <dd>{placeholder.corrupt_blocks ?? 0}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-[var(--ink-muted)]">formula_blocks</dt>
            <dd>{report?.formula_blocks ?? 0}</dd>
          </div>
        </dl>
      ),
    },
    {
      title: t.quality.pillar2Title,
      desc: t.quality.pillar2Desc,
      metric: `recall ${pct(terminology.term_recall)} · drift ${entity.terms_with_drift ?? 0}/${entity.terms_audited ?? 0}`,
      status: (entity.terms_with_drift ?? 0) > 0 ? 'warn' : 'pass',
      score: clampScore(terminology.term_recall),
      detail:
        topDrifted.length === 0 ? (
          <div className="text-[11px] text-[var(--ink-muted)]">{t.quality.noDrift}</div>
        ) : (
          <div className="space-y-1.5">
            <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
              {t.quality.termDriftTitle}
            </div>
            <table className="w-full text-[11px] text-left font-mono">
              <thead className="text-[var(--ink-muted)]">
                <tr>
                  <th className="py-1 pr-3 font-semibold">{t.quality.colTerm}</th>
                  <th className="py-1 pr-3 font-semibold">{t.quality.colExpected}</th>
                  <th className="py-1 pr-3 font-semibold">{t.quality.colOccurrences}</th>
                  <th className="py-1 font-semibold">{t.quality.colDriftRate}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--paper-border)]">
                {topDrifted.map((row) => (
                  <tr key={`${row.source}-${row.expected}`}>
                    <td className="py-1 pr-3 text-[var(--ink-primary)]">{row.source}</td>
                    <td className="py-1 pr-3 text-[#15803d]">{row.expected}</td>
                    <td className="py-1 pr-3">{row.occurrences ?? 0}</td>
                    <td className="py-1 text-[#b45309]">
                      {typeof row.drift_rate === 'number'
                        ? `${(row.drift_rate * 100).toFixed(0)}%`
                        : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ),
    },
    {
      title: t.quality.pillar3Title,
      desc: t.quality.pillar3Desc,
      metric: `${fidelity.pages_measured ?? 0} pages · residual ${pct(fidelity.non_text_residual, 2)} · ${flaggedPages.length} ${t.quality.flaggedPages}`,
      status: pillarStatus(
        flaggedPages.length > 0,
        (fidelity.non_text_residual ?? 0) > 0.02
      ),
      score:
        (fidelity.pages_measured ?? 0) > 0
          ? clampScore(1 - (fidelity.non_text_residual ?? 0))
          : flaggedPages.length > 0
            ? 60
            : 100,
      detail: (
        <div className="space-y-2">
          {visualFindings.length === 0 ? (
            <div className="text-[11px] text-[var(--ink-muted)]">{t.quality.noVisualFindings}</div>
          ) : (
            <>
              <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
                {t.quality.visualFindingsTitle}
              </div>
              <ul className="space-y-1 text-[11px]">
                {visualFindings.slice(0, 8).map((finding, index) => (
                  <li key={index} className="flex gap-2">
                    <span className="font-mono text-[var(--ink-muted)] shrink-0">
                      {finding.page != null ? `P${finding.page}` : '—'}
                    </span>
                    <span className="text-[var(--ink-secondary)]">
                      <span className="font-mono text-[#b45309]">{finding.code}</span>{' '}
                      {finding.message}
                    </span>
                  </li>
                ))}
              </ul>
            </>
          )}
          {flaggedPages.length > 0 && (
            <div className="grid grid-cols-4 gap-2 pt-1">
              {flaggedPages.slice(0, 8).map((page) => (
                <div
                  key={page}
                  className="rounded-[4px] border border-[var(--paper-border)] overflow-hidden bg-white"
                >
                  <img
                    src={pagePreviewUrl(jobId, page, { dpi: 70 })}
                    alt={`Page ${page}`}
                    className="w-full h-24 object-contain"
                    onError={(e) => {
                      e.currentTarget.style.display = 'none'
                    }}
                  />
                  <div className="text-center text-[10px] font-mono text-[var(--ink-muted)] py-0.5 border-t border-[var(--paper-border)]">
                    P{page}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      ),
    },
    {
      title: t.quality.pillar4Title,
      desc: t.quality.pillar4Desc,
      metric: `fail-closed ${coverage.fail_closed_blocks ?? 0} · coverage ${pct(coverage.render_coverage)}`,
      status: (coverage.fail_closed_blocks ?? 0) > 0 ? 'fail' : 'pass',
      score: clampScore(coverage.render_coverage),
      detail: (
        <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-[11px] font-mono">
          {[
            ['rendered_blocks', coverage.rendered_blocks ?? 0],
            ['skipped_blocks', coverage.skipped_blocks ?? 0],
            ['fail_closed_blocks', coverage.fail_closed_blocks ?? 0],
            ['preserved_blocks', coverage.preserved_blocks ?? 0],
          ].map(([label, value]) => (
            <div key={String(label)} className="flex justify-between">
              <dt className="text-[var(--ink-muted)]">{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      ),
    },
    {
      title: t.quality.pillar5Title,
      desc: t.quality.pillar5Desc,
      metric:
        summary.estimated_cost_usd === null || summary.estimated_cost_usd === undefined
          ? 'cost not priced'
          : `$${Number(summary.estimated_cost_usd).toFixed(2)} · cache ${pct(summary.cache_hit_rate)}`,
      status: 'pass',
      score: 100,
      detail: (
        <dl className="grid grid-cols-2 gap-x-6 gap-y-1 text-[11px] font-mono">
          {[
            [t.quality.auditDraftModel, config.draft_model ?? '—'],
            [t.quality.auditRepairModel, config.repair_model ?? '—'],
            [t.quality.auditQeEngine, report?.qe_score_source ?? 'heuristic'],
            [
              t.quality.auditRetries,
              `${repair.direct_pass_count ?? 0}/${repair.round_1_repaired_count ?? 0}/${repair.round_2_repaired_count ?? 0}/${repair.exhausted_count ?? 0}`,
            ],
            [t.quality.auditCacheHit, pct(summary.cache_hit_rate)],
            [t.quality.auditTypst, report?.typst_version ?? '—'],
          ].map(([label, value]) => (
            <div key={String(label)} className="flex justify-between gap-3">
              <dt className="text-[var(--ink-muted)] shrink-0">{label}</dt>
              <dd className="truncate text-right">{value}</dd>
            </div>
          ))}
        </dl>
      ),
    },
  ]

  const radarAxes = [
    { label: t.quality.axisMath, value: pillars[0].score },
    { label: t.quality.axisTerminology, value: pillars[1].score },
    { label: t.quality.axisVisual, value: pillars[2].score },
    { label: t.quality.axisCompleteness, value: pillars[3].score },
    { label: t.quality.axisDelivered, value: clampScore(summary.pass_rate) },
  ]

  const handleDownload = (key: string) => {
    window.open(deliverableDownloadUrl(jobId, key), '_blank')
  }

  const statusBadge = (status: PillarStatus) =>
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

      {/* Decision Banner + Radar */}
      <div className="p-5 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] flex items-center justify-between gap-6 shadow-2xs">
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
            <div className="text-[10px] font-mono text-[var(--ink-muted)] mt-1.5">
              pass rate {pct(summary.pass_rate)} · avg QE {Number(score.avg_qe ?? 0).toFixed(3)}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <div className="text-right">
            <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
              {t.quality.fidelityScore}
            </div>
            <div className="text-2xl font-bold font-mono text-[var(--ink-primary)] mt-0.5">
              {report ? fidelityScore.toFixed(1) : '—'}{' '}
              <span className="text-xs text-[var(--ink-muted)] font-normal">/ 100</span>
            </div>
          </div>
          {report && <RadarChart axes={radarAxes} />}
        </div>
      </div>

      {/* Formal Verification Checklist */}
      <div className="rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] overflow-hidden shadow-2xs">
        <div className="px-4 py-2.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
          FORMAL VERIFICATION GATEWAYS (5 PILLARS)
        </div>
        <div className="divide-y divide-[var(--paper-border)]">
          {pillars.map((item, idx) => (
            <div key={idx}>
              <button
                onClick={() => setExpanded(expanded === idx ? null : idx)}
                className="w-full p-4 flex items-center justify-between hover:bg-[var(--paper-subsurface)]/60 transition-colors text-left"
              >
                <div className="flex items-start gap-2.5">
                  {expanded === idx ? (
                    <ChevronDown className="h-4 w-4 mt-0.5 text-[var(--ink-muted)] shrink-0" />
                  ) : (
                    <ChevronRight className="h-4 w-4 mt-0.5 text-[var(--ink-muted)] shrink-0" />
                  )}
                  <div>
                    <div className="text-xs font-semibold text-[var(--ink-primary)]">
                      {item.title}
                    </div>
                    <div className="text-[11px] text-[var(--ink-secondary)] mt-0.5 leading-relaxed">
                      {item.desc}
                    </div>
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
              </button>
              {expanded === idx && (
                <div className="px-4 pb-4 pl-11 text-[var(--ink-primary)]">{item.detail}</div>
              )}
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
