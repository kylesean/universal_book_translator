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
  Info,
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import {
  getJobReport,
  getVisualReport,
  listDeliverables,
  deliverableDownloadUrl,
  reExportJob,
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

/** The checklist's per-pillar verdict, translated. */
function pillarStatusLabel(
  t: ReturnType<typeof useI18n>['t'],
  status: 'pass' | 'warn' | 'fail'
): string {
  if (status === 'pass') return t.quality.pillarStatusPass
  if (status === 'warn') return t.quality.pillarStatusWarn
  return t.quality.pillarStatusFail
}

/** A five-axis radar drawn as inline SVG (no chart dependency). */
function RadarChart({ axes }: { axes: { label: string; value: number }[] }) {
  const cx = 110
  const cy = 100
  const radius = 60
  const point = (index: number, ratio: number): [number, number] => {
    const angle = ((-90 + index * (360 / axes.length)) * Math.PI) / 180
    return [cx + radius * ratio * Math.cos(angle), cy + radius * ratio * Math.sin(angle)]
  }
  const polygon = axes
    .map((axis, index) => point(index, Math.min(1, Math.max(0, axis.value / 100))))
    .map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`)
    .join(' ')

  return (
    <svg viewBox="0 0 220 200" className="w-48 h-44 sm:w-56 sm:h-52 shrink-0 max-w-full" role="img">
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
        const [x, y] = point(index, 1.25)
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
  // True when the files predate the ledger text. The download endpoint
  // refuses such a job with 409, and `window.open` would show the operator
  // a page of raw JSON instead of a file.
  const [deliveryStale, setDeliveryStale] = useState(false)
  const [downloadBusy, setDownloadBusy] = useState(false)
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const [visual, setVisual] = useState<Record<string, any> | null>(null)
  // Whether the visual report itself is absent (gate disabled, or the
  // deliverable is not a PDF — the endpoint then answers 404/400). Distinct
  // from a loaded report: an absent report is no attestation either way, and
  // must not read as a clean gate.
  const [visualMissing, setVisualMissing] = useState(false)
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
      .then(({ items, stale }) => {
        if (isMounted) {
          setDeliverables(items)
          setDeliveryStale(stale)
        }
      })
      .catch(() => {
        if (isMounted) setDeliverables([])
      })

    getVisualReport(jobId)
      .then((rep) => {
        if (isMounted) {
          setVisual(rep)
          setVisualMissing(false)
        }
      })
      .catch(() => {
        if (isMounted) {
          setVisual(null)
          setVisualMissing(true)
        }
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
  // `report.fidelity` no longer exists — the overlay engine it described was
  // deleted, and nothing in the report schema carries `pages_measured` or
  // `non_text_residual` any more. Reading it produced a permanent "0 pages ·
  // residual —" with a 100 score and a PASS badge: a full-marks pillar for a
  // measurement that had not been taken since the field was removed.
  //
  // The real pixel verdict lives on the *visual report* payload, which the
  // route already serves beside it.
  const visualPassed: boolean | null =
    typeof visual?.passed === 'boolean' ? visual.passed : null
  const visualSkipped: string | null =
    typeof visual?.skipped_reason === 'string' && visual.skipped_reason
      ? visual.skipped_reason
      : null
  // The report now carries the tier facts as lists and skip reasons as stable
  // codes; ``skipped_reason`` is the English sentence built from them, kept for
  // logs and the CLI. Rendering that sentence here printed raw English inside a
  // Chinese panel — and, because it listed every skipped tier, it was long
  // enough to squeeze the status badge into a two-character column.
  const tiersRan: string[] = Array.isArray(visual?.tiers_ran) ? visual.tiers_ran : []
  const tiersSkipped: string[] = Array.isArray(visual?.tiers_skipped) ? visual.tiers_skipped : []
  const skipCodes: string[] = Array.isArray(visual?.skip_codes) ? visual.skip_codes : []
  /** One translated phrase per skipped tier, e.g. "未运行：语义视觉判定". */
  const tierLabel = (tier: string): string =>
    (t.quality.visualTierNames as Record<string, string>)[tier] ?? tier
  const skipLabel = (code: string): string =>
    (t.quality.visualSkipCodes as Record<string, string>)[code] ?? code
  const coverage = report?.render_coverage ?? {}
  const repair = report?.repair_breakdown ?? {}
  const config = report?.config_snapshot ?? {}

  const failedBlocks = summary.failed_blocks ?? 0
  // The gate decision is the shared three-state verdict the backend computes
  // (`job_options.delivery_verdict_from_summary` — the same rule the CLI banner
  // applies), falling back to failed-blocks-only for reports served before the
  // field existed. Two report fields this screen used to read
  // (`delivery_status` / `delivery_warning`) were removed: no engine code ever
  // assigned them, so the advisory banner they fed could never appear.
  const verdict: string | null = report?.delivery_verdict ?? null
  // A rehearsal's artifact is echo text, not a translation. Every structural
  // gate passes it by construction, so without this flag the screen showed
  // "PRODUCTION READY" over `[模拟翻译]` text.
  const rehearsal: boolean = report?.rehearsal === true
  // No report means NO VERDICT — never a pass.
  //
  // `summary` defaults to `{}`, so `failed_blocks ?? 0` is 0 when the report
  // failed to load, and the old `verdict ? ... : failedBlocks === 0` fallback
  // rendered a green check, "PRODUCTION READY", a PASS badge, "All formal
  // checks passed" and five passing pillars for a job whose report had never
  // been written. The report fetch legitimately 409s while a job is still
  // running and 404s before the reporter writes it, so this was reachable by
  // simply opening the gate too early — an attestation of nothing, on the one
  // screen whose entire purpose is attestation.
  const hasReport = report !== null
  const isPassed = hasReport && (verdict ? verdict === 'passed' : failedBlocks === 0)
  const isWarning = hasReport && verdict === 'warning'
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
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1 text-xs font-mono">
          <div className="flex justify-between">
            <dt className="text-[var(--ink-muted)]">{t.quality.metricMaskedSpans}</dt>
            <dd>{placeholder.masked_spans ?? 0}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-[var(--ink-muted)]">{t.quality.metricCorruptSpans}</dt>
            <dd>{placeholder.corrupt_spans ?? 0}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-[var(--ink-muted)]">{t.quality.metricMaskedBlocks}</dt>
            <dd>{placeholder.masked_blocks ?? 0}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-[var(--ink-muted)]">{t.quality.metricCorruptBlocks}</dt>
            <dd>{placeholder.corrupt_blocks ?? 0}</dd>
          </div>
          <div className="flex justify-between">
            <dt className="text-[var(--ink-muted)]">{t.quality.metricFormulaBlocks}</dt>
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
          <div className="text-xs text-[var(--ink-muted)]">{t.quality.noDrift}</div>
        ) : (
          <div className="space-y-1.5">
            <div className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
              {t.quality.termDriftTitle}
            </div>
            <div className="overflow-x-auto min-w-0">
              <table className="w-full min-w-[340px] text-xs text-left font-mono">
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
          </div>
        ),
    },
    {
      title: t.quality.pillar3Title,
      desc: t.quality.pillar3Desc,
      // A short, translated summary. The full tier breakdown and the reason live
      // in the expanded detail — the row header must stay one line so the badge
      // beside it is not squeezed.
      metric: visualMissing
        ? t.quality.visualNoReport
        : visualSkipped
          ? skipCodes.length > 0
            ? `${t.quality.visualGateSkipped} · ${skipCodes.map(skipLabel).join('、')}`
            : t.quality.visualGateSkipped
          : `${flaggedPages.length} ${t.quality.flaggedPages}`,
      // A skipped gate is not a passing gate, and neither is an absent one:
      // `visual_gate` reports `skipped_reason` when the tooling or the artifact
      // type prevents the check (e.g. "not a rendered pdf"), and the visual
      // *report itself* is missing entirely when the gate was disabled or the
      // deliverable is not a PDF — this view used to render both as a green
      // PASS because it only ever looked at `findings`, which are empty when
      // the gate never ran.
      status:
        visualMissing || visualSkipped
          ? 'warn'
          : pillarStatus(flaggedPages.length > 0 || visualPassed === false, false),
      score: visualMissing
        ? 0
        : visualSkipped
          ? 0
          : visualPassed === false
            ? 55
            : flaggedPages.length > 0
              ? 60
              : 100,
      detail: (
        <div className="space-y-2">
          {visualMissing && (
            <div className="text-xs text-[var(--ink-secondary)]">
              {t.quality.visualNoReportDesc}
            </div>
          )}
          {/* Which inspection tiers ran. This is the full statement the row
              header summarizes; it wraps freely here, so it can afford the
              names instead of a comma-separated code list. */}
          {tiersRan.length > 0 && (
            <div className="text-xs text-[var(--ink-secondary)]">
              <span className="text-[var(--ink-muted)]">{t.quality.visualTiersRan}: </span>
              {tiersRan.map(tierLabel).join(' · ')}
              {tiersSkipped.length > 0 && (
                <>
                  <span className="text-[var(--ink-muted)]">
                    {'  ·  '}
                    {t.quality.visualTiersSkipped}:{' '}
                  </span>
                  {tiersSkipped.map(tierLabel).join(' · ')}
                </>
              )}
            </div>
          )}
          {visualFindings.length === 0 ? (
            <div className="text-xs text-[var(--ink-muted)]">{t.quality.noVisualFindings}</div>
          ) : (
            <>
              <div className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
                {t.quality.visualFindingsTitle}
              </div>
              <ul className="space-y-1 text-xs max-h-56 overflow-y-auto">
                {visualFindings.map((finding, index) => (
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
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 max-h-72 overflow-y-auto">
              {flaggedPages.map((page) => (
                <div
                  key={page}
                  className="rounded-[4px] border border-[var(--paper-border)] overflow-hidden bg-white"
                >
                  <img
                    src={pagePreviewUrl(jobId, page, { dpi: 70 })}
                    alt={t.quality.pageAlt.replace('{n}', String(page))}
                    className="w-full h-24 object-contain"
                    onError={(e) => {
                      e.currentTarget.style.display = 'none'
                    }}
                  />
                  <div className="text-center text-xs font-mono text-[var(--ink-muted)] py-0.5 border-t border-[var(--paper-border)]">
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
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1 text-xs font-mono">
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
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-1 text-xs font-mono">
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

  // A stale delivery is re-rendered before the download, the same way Mission
  // Control does it: the backend guard exists so a customer never receives the
  // pre-edit file, and the console's job is to make that guard invisible rather
  // than to hand the operator a 409 they cannot act on.
  const handleDownload = async (key: string) => {
    if (downloadBusy) return
    setDownloadBusy(true)
    // The tab is opened inside the click gesture, before any await: a stale
    // delivery re-renders first (whole-book, easily past the browser's ~5 s
    // transient-activation window), and a `window.open` after that wait is
    // silently swallowed by the popup blocker — the click "did nothing".
    // Mission Control fixed the same bug the same way.
    const tab = window.open('', '_blank')
    try {
      if (deliveryStale) {
        await reExportJob(jobId)
        const refreshed = await listDeliverables(jobId)
        setDeliverables(refreshed.items)
        setDeliveryStale(refreshed.stale)
      }
      if (tab) tab.location.href = deliverableDownloadUrl(jobId, key)
      else window.open(deliverableDownloadUrl(jobId, key), '_blank')
    } catch (err) {
      tab?.close()
      setDownloadError(err instanceof Error ? err.message : t.quality.downloadFailed)
    } finally {
      setDownloadBusy(false)
    }
  }

  const statusBadge = (status: PillarStatus) =>
    status === 'pass' ? 'success' : status === 'warn' ? 'warning' : 'destructive'

  return (
    <div className="flex-1 overflow-y-auto px-4 sm:px-8 py-5 sm:py-7 space-y-5 sm:space-y-7 max-w-5xl mx-auto w-full">
      {/* View Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[var(--paper-border)] pb-4">
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

      {/* Rehearsal watermark. A dry run's artifact is the source text behind
          `[模拟翻译]`, and every structural gate passes it by construction —
          so without this banner the screen's "PRODUCTION READY / PASS" is a
          statement about echo text. */}
      {rehearsal && (
        <div className="p-3 rounded-lg border border-[#1d4ed8]/30 bg-[#1d4ed8]/5 text-xs flex items-start gap-2">
          <Info className="h-4 w-4 text-[#1d4ed8] shrink-0 mt-0.5" />
          <div className="space-y-1">
            <div className="font-mono uppercase tracking-wider text-xs font-semibold text-[#1d4ed8]">
              {t.quality.rehearsalBadge}
            </div>
            <p className="text-[var(--ink-secondary)] leading-relaxed">{t.quality.rehearsalBody}</p>
          </div>
        </div>
      )}

      {/* No report yet: say so, and do not render a verdict block at all.
          The pillars below read from the same absent payload, so they would
          otherwise all render as passing. */}
      {!hasReport && (
        <div className="p-5 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] flex items-start gap-3 shadow-2xs">
          <AlertTriangle className="h-5 w-5 text-[#b45309] shrink-0 mt-0.5" />
          <div className="space-y-1">
            <div className="text-sm font-semibold text-[var(--ink-primary)]">
              {t.quality.noVerdictTitle}
            </div>
            <p className="text-xs text-[var(--ink-secondary)] leading-relaxed">
              {t.quality.noVerdictBody}
            </p>
          </div>
        </div>
      )}

      {/* Decision Banner + Radar */}
      {hasReport && (
      <div className="p-4 sm:p-5 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] flex flex-col md:flex-row items-start md:items-center justify-between gap-5 md:gap-6 shadow-2xs">
        <div className="flex items-start sm:items-center gap-3.5">
          <div
            className={`h-10 w-10 rounded-[6px] bg-[var(--paper-subsurface)] border border-[var(--paper-border)] flex items-center justify-center shrink-0 ${
              isPassed ? 'text-[#15803d]' : isWarning ? 'text-[#b45309]' : 'text-[#b91c1c]'
            }`}
          >
            {isPassed ? <CheckCircle2 className="h-6 w-6" /> : <AlertTriangle className="h-6 w-6" />}
          </div>
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm font-bold text-[var(--ink-primary)]">
                {isPassed
                  ? t.quality.decisionReady
                  : isWarning
                    ? t.quality.decisionWarning
                    : t.quality.decisionBlocked}
              </span>
              <Badge
                variant={isPassed ? 'success' : isWarning ? 'warning' : 'destructive'}
                dot
              >
                {isPassed
                  ? t.quality.verdictPass
                  : isWarning
                    ? t.quality.verdictWarning
                    : t.quality.verdictBlocked}
              </Badge>
            </div>
            <p className="text-xs text-[var(--ink-secondary)] mt-0.5">
              {isPassed
                ? t.quality.allPassedDesc
                : isWarning
                  ? t.quality.warningDesc
                  : t.quality.blockedDesc}
            </p>
            <div className="text-xs font-mono text-[var(--ink-muted)] mt-1.5">
              {t.quality.passRate.replace('{value}', pct(summary.pass_rate))} ·{' '}
              {score.qe_measured ? (
                <span>
                  {t.quality.avgQeValue.replace('{score}', Number(score.avg_qe ?? 0).toFixed(3))}
                  {' · '}
                  {t.quality.qeMeasuredOn.replace(
                    '{count}',
                    String(score.scored_blocks ?? 0)
                  )}
                </span>
              ) : (
                <span className="uppercase" title={t.quality.qeNotMeasuredHint}>
                  {t.quality.qeNotMeasured}
                </span>
              )}
            </div>
            {/* "QE not measured" on its own is a dead end: the reader cannot
                tell whether something broke, what it would take to measure, or
                whether it matters. Both causes are on the report — the engine
                that ran and whether the judge was enabled — so state the reason
                and the remedy instead of the bare status. */}
            {!score.qe_measured && hasReport && (
              <p className="text-xs text-[var(--ink-secondary)] mt-1.5 leading-relaxed max-w-xl">
                {config.qe_judge_enabled
                  ? t.quality.qeNotMeasuredWhyJudge
                  : t.quality.qeNotMeasuredWhyHeuristic}{' '}
                <span className="text-[var(--ink-muted)]">{t.quality.qeNotMeasuredHow}</span>
              </p>
            )}
          </div>
        </div>

        <div className="flex flex-col sm:flex-row items-center justify-center md:justify-end gap-4 w-full md:w-auto pt-3 md:pt-0 border-t md:border-t-0 border-[var(--paper-border)]">
          <div className="text-center sm:text-right">
            <div className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
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
      )}

      {/* Formal Verification Checklist — only with a report: every pillar is
          derived from it, so without one they would all render as passing. */}
      {hasReport && (
      <div className="rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] overflow-hidden shadow-2xs">
        <div className="px-4 py-2.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
          {t.quality.pillarsTitle}
        </div>
        <div className="divide-y divide-[var(--paper-border)]">
          {pillars.map((item, idx) => (
            <div key={idx}>
              <button
                onClick={() => setExpanded(expanded === idx ? null : idx)}
                className="w-full p-3.5 sm:p-4 flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 sm:gap-4 hover:bg-[var(--paper-subsurface)] transition-colors text-left"
              >
                <div className="flex items-start gap-2.5 min-w-0 flex-1">
                  {expanded === idx ? (
                    <ChevronDown className="h-4 w-4 mt-0.5 text-[var(--ink-muted)] shrink-0" />
                  ) : (
                    <ChevronRight className="h-4 w-4 mt-0.5 text-[var(--ink-muted)] shrink-0" />
                  )}
                  <div className="min-w-0">
                    <div className="text-xs font-semibold text-[var(--ink-primary)]">
                      {item.title}
                    </div>
                    <div className="text-xs text-[var(--ink-secondary)] mt-0.5 leading-relaxed">
                      {item.desc}
                    </div>
                  </div>
                </div>
                {/* The metric is a *sentence* for some pillars — pillar 3 says the
                    pixel check was skipped and lists the tiers that did run — so it
                    must wrap and truncate rather than push the status badge into a
                    column three characters wide. It shrinks before the title does:
                    the title identifies the row, the metric explains it. */}
                <div className="flex items-center gap-3 shrink-0 w-full sm:w-auto sm:max-w-[45%] justify-between sm:justify-end pl-6 sm:pl-0">
                  <span className="font-mono text-xs text-[var(--ink-primary)] font-medium text-left sm:text-right leading-snug break-words">
                    {item.metric}
                  </span>
                  <Badge variant={statusBadge(item.status)} dot className="shrink-0 whitespace-nowrap">
                    {pillarStatusLabel(t, item.status)}
                  </Badge>
                </div>
              </button>
              {expanded === idx && (
                <div className="px-4 pb-4 pl-8 sm:pl-11 text-[var(--ink-primary)] overflow-x-auto">{item.detail}</div>
              )}
            </div>
          ))}
        </div>
      </div>
      )}

      {/* Deliverable Download Section */}
      <div className="rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] p-4 sm:p-5 space-y-4 shadow-2xs">
        <div>
          <h3 className="text-xs font-bold text-[var(--ink-primary)] tracking-tight">
            {t.quality.deliverablesTitle}
          </h3>
          <p className="text-xs text-[var(--ink-secondary)] mt-0.5">
            {t.quality.deliverablesSubtitle}
          </p>
        </div>

        {deliveryStale && (
          <div className="rounded border border-[#b45309]/30 bg-[#b45309]/5 px-2.5 py-1.5 text-xs text-[#b45309]">
            {t.quality.deliveryStaleNote}
          </div>
        )}
        {downloadError && (
          <div className="rounded border border-[#b91c1c]/30 bg-[#b91c1c]/5 px-2.5 py-1.5 text-xs text-[#b91c1c]">
            {downloadError}
          </div>
        )}
        {deliverables.length === 0 ? (
          <div className="text-xs text-[var(--ink-muted)] py-2">
            {t.quality.noDeliverables}
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-2.5 sm:gap-3">
            {deliverables.map((item) => (
              <Button
                key={item.key}
                onClick={() => void handleDownload(item.key)}
                disabled={downloadBusy}
                variant="secondary"
                className="h-10 justify-start px-3 text-xs w-full"
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
