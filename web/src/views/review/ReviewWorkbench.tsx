import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  SplitSquareVertical,
  Save,
  Eye,
  Loader2,
  AlertTriangle,
  ChevronUp,
  ChevronDown,
  CheckCircle2,
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { listSegments, listIssues, editSegment, type Segment, type IssuesReport } from '@/api/client'
import { useI18n } from '@/i18n/I18nContext'

interface ReviewWorkbenchProps {
  jobId: string | null
}

type Filter = 'issues' | 'all' | 'needs_human'

//: Human labels for the backend's issue kinds (ribbon + card chips).
const ISSUE_LABELS: Record<string, string> = {
  terminology: 'Terminology',
  formula: 'Formula',
  numeric: 'Numeric',
  omission: 'Omission',
  fabrication: 'Fabrication',
  repetition: 'Repetition',
  structure: 'Structure',
  echo: 'Echo',
  review: 'Needs review',
  critical: 'Critical',
  render: 'Render/overflow',
}

const PAGE_LIMIT = 200

export function ReviewWorkbench({ jobId }: ReviewWorkbenchProps) {
  const { t } = useI18n()
  const [viewMode, setViewMode] = useState<'segments' | 'visual_witness'>('segments')
  const [filter, setFilter] = useState<Filter>('issues')

  const [segments, setSegments] = useState<Segment[]>([])
  const [total, setTotal] = useState(0)
  const [issues, setIssues] = useState<IssuesReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState<Record<string, boolean>>({})
  const [saved, setSaved] = useState<Record<string, boolean>>({})
  const [issueCursor, setIssueCursor] = useState(0)

  const cardRefs = useRef<Record<string, HTMLDivElement | null>>({})

  const load = useCallback(async () => {
    if (!jobId) return
    setLoading(true)
    setError(null)
    try {
      const [segmentData, issueData] = await Promise.all([
        listSegments(jobId, { status: filter, limit: PAGE_LIMIT }),
        listIssues(jobId),
      ])
      setSegments(segmentData.segments)
      setTotal(segmentData.total)
      setIssues(issueData)
      setDrafts({})
      setSaved({})
      setIssueCursor(0)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load workbench')
    } finally {
      setLoading(false)
    }
  }, [jobId, filter])

  useEffect(() => {
    void load()
  }, [load])

  const issueSegments = useMemo(
    () => segments.filter((s) => s.issues.length > 0 || s.status === 'needs_human' || s.status === 'blocked_human'),
    [segments]
  )

  const jumpToIssue = (delta: number) => {
    if (issueSegments.length === 0) return
    const next = (issueCursor + delta + issueSegments.length) % issueSegments.length
    setIssueCursor(next)
    const target = cardRefs.current[issueSegments[next].block_id]
    target?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  const handleSave = async (segment: Segment) => {
    if (!jobId) return
    const draft = drafts[segment.block_id]
    if (draft === undefined || draft === segment.target_text) return
    setSaving((prev) => ({ ...prev, [segment.block_id]: true }))
    try {
      const res = await editSegment(jobId, segment.block_id, draft)
      if (res.segment) {
        const updated = res.segment
        setSegments((prev) => prev.map((s) => (s.block_id === updated.block_id ? updated : s)))
      }
      setSaved((prev) => ({ ...prev, [segment.block_id]: true }))
      setDrafts((prev) => {
        const next = { ...prev }
        delete next[segment.block_id]
        return next
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save revision')
    } finally {
      setSaving((prev) => ({ ...prev, [segment.block_id]: false }))
    }
  }

  if (!jobId) {
    return (
      <div className="flex-1 flex items-center justify-center p-8 text-center text-[var(--ink-muted)]">
        <div className="max-w-sm space-y-2">
          <SplitSquareVertical className="h-8 w-8 mx-auto text-[var(--paper-border-hover)]" />
          <h2 className="text-sm font-semibold text-[var(--ink-primary)]">{t.quality.noJobTitle}</h2>
          <p className="text-xs text-[var(--ink-secondary)] leading-relaxed">{t.quality.noJobDesc}</p>
        </div>
      </div>
    )
  }

  const ribbonCounts = issues
    ? Object.entries(issues.counts).filter(([, value]) => value > 0)
    : []

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-hidden bg-[var(--paper-bg)] text-[var(--ink-primary)]">
      {/* Precision Ribbon */}
      <div className="min-h-11 px-6 py-2 border-b border-[var(--paper-border)] bg-[var(--paper-surface)] flex flex-wrap items-center justify-between gap-2 shrink-0 shadow-2xs">
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-1.5 text-xs font-bold text-[var(--ink-primary)]">
            <SplitSquareVertical className="h-4 w-4 text-[#15803d]" />
            <span>{t.review.title}</span>
          </div>
          <div className="h-3 w-px bg-[var(--paper-border)]" />
          <div className="flex items-center gap-2 text-[11px] font-mono flex-wrap">
            {ribbonCounts.length === 0 ? (
              <span className="text-[#15803d] flex items-center gap-1 font-medium">
                <CheckCircle2 className="h-3.5 w-3.5" /> 0 issues
              </span>
            ) : (
              ribbonCounts.map(([kind, value]) => (
                <span
                  key={kind}
                  className="px-1.5 py-0.5 rounded border border-[var(--paper-border)] bg-[var(--paper-subsurface)] text-[var(--ink-secondary)]"
                >
                  {ISSUE_LABELS[kind] ?? kind} {value}
                </span>
              ))
            )}
            {(issues?.status.needs_human ?? 0) > 0 && (
              <span className="text-[#b45309] font-medium">
                needs human {issues?.status.needs_human}
              </span>
            )}
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => jumpToIssue(-1)}
              title={t.review.prevIssue}
              className="p-1 rounded hover:bg-[var(--paper-subsurface)] text-[var(--ink-muted)]"
            >
              <ChevronUp className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={() => jumpToIssue(1)}
              title={t.review.nextIssue}
              className="p-1 rounded hover:bg-[var(--paper-subsurface)] text-[var(--ink-muted)]"
            >
              <ChevronDown className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <div className="flex bg-[var(--paper-subsurface)] p-0.5 rounded-[4px] border border-[var(--paper-border)]">
            {(
              [
                ['issues', t.review.filterIssues],
                ['needs_human', t.review.filterNeedsHuman],
                ['all', t.review.filterAll],
              ] as [Filter, string][]
            ).map(([id, label]) => (
              <button
                key={id}
                onClick={() => setFilter(id)}
                className={`px-2.5 py-1 text-xs rounded-[3px] transition-colors font-medium ${
                  filter === id
                    ? 'bg-[var(--paper-surface)] text-[var(--ink-primary)] font-bold shadow-2xs'
                    : 'text-[var(--ink-muted)] hover:text-[var(--ink-primary)]'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="flex bg-[var(--paper-subsurface)] p-0.5 rounded-[4px] border border-[var(--paper-border)]">
            <button
              onClick={() => setViewMode('segments')}
              className={`px-3 py-1 text-xs rounded-[3px] transition-colors font-medium ${
                viewMode === 'segments'
                  ? 'bg-[var(--paper-surface)] text-[var(--ink-primary)] font-bold shadow-2xs'
                  : 'text-[var(--ink-muted)] hover:text-[var(--ink-primary)]'
              }`}
            >
              {t.review.modeSegments}
            </button>
            <button
              onClick={() => setViewMode('visual_witness')}
              className={`px-3 py-1 text-xs rounded-[3px] transition-colors font-medium ${
                viewMode === 'visual_witness'
                  ? 'bg-[var(--paper-surface)] text-[var(--ink-primary)] font-bold shadow-2xs'
                  : 'text-[var(--ink-muted)] hover:text-[var(--ink-primary)]'
              }`}
            >
              {t.review.modeVisual}
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="px-6 py-2 text-xs text-[var(--ink-secondary)] bg-[#b45309]/10 border-b border-[var(--paper-border)] flex items-center gap-2">
          <AlertTriangle className="h-3.5 w-3.5 text-[#b45309] shrink-0" />
          {error}
        </div>
      )}

      {viewMode === 'segments' ? (
        <div className="flex-1 overflow-y-auto px-8 py-6 space-y-4 max-w-5xl mx-auto w-full">
          {loading ? (
            <div className="py-12 text-center text-[var(--ink-muted)]">
              <Loader2 className="h-5 w-5 animate-spin mx-auto" />
            </div>
          ) : segments.length === 0 ? (
            <div className="py-12 text-center text-xs text-[var(--ink-muted)]">
              {t.review.noSegments}
            </div>
          ) : (
            <>
              {segments.map((seg) => {
                const draft = drafts[seg.block_id] ?? seg.target_text
                const dirty = draft !== seg.target_text
                return (
                  <div
                    key={seg.block_id}
                    ref={(el) => {
                      cardRefs.current[seg.block_id] = el
                    }}
                    className={`rounded-lg border transition-colors shadow-2xs bg-[var(--paper-surface)] ${
                      seg.issues.length > 0
                        ? 'border-[#b45309]/50'
                        : 'border-[var(--paper-border)]'
                    }`}
                  >
                    <div className="h-8 px-3.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] flex items-center justify-between text-xs font-mono text-[var(--ink-muted)]">
                      <div className="flex items-center gap-2">
                        <span className="text-[var(--ink-primary)] font-bold">{seg.block_id}</span>
                        {seg.page !== null && <span>Page {seg.page}</span>}
                        <span className="text-[10px] px-1.5 py-0.2 rounded bg-[var(--paper-border)] text-[var(--ink-secondary)] font-semibold">
                          {seg.block_type}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        {seg.issues.map((kind) => (
                          <span
                            key={kind}
                            className="text-[10px] px-1.5 py-0.5 rounded bg-[#b45309]/15 text-[#b45309] font-semibold"
                          >
                            {ISSUE_LABELS[kind] ?? kind}
                          </span>
                        ))}
                        {seg.mtqe_score !== null && (
                          <span className="font-semibold text-[var(--ink-primary)]">
                            QE {(seg.mtqe_score * 100).toFixed(0)}%
                          </span>
                        )}
                        {seg.human_verified ? (
                          <Badge variant="success" dot>
                            {t.review.humanVerified}
                          </Badge>
                        ) : (
                          <Badge
                            variant={seg.issues.length > 0 ? 'warning' : 'outline'}
                            dot
                          >
                            {seg.status.toUpperCase()}
                          </Badge>
                        )}
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-[var(--paper-border)] text-xs">
                      <div className="p-4 text-[var(--ink-secondary)] font-mono leading-relaxed select-text bg-[var(--paper-subsurface)]/20">
                        <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] mb-1 font-semibold">
                          {t.review.sourceSegment}
                        </div>
                        {seg.source_text}
                      </div>

                      <div className="p-4 flex flex-col justify-between bg-[var(--paper-surface)]">
                        <div>
                          <div className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] mb-1.5 flex items-center justify-between font-semibold">
                            <span>{t.review.targetSegment}</span>
                            {saved[seg.block_id] && (
                              <span className="text-[#15803d] font-sans font-medium">
                                {t.review.saved}
                              </span>
                            )}
                          </div>
                          <textarea
                            value={draft}
                            onChange={(e) =>
                              setDrafts((prev) => ({ ...prev, [seg.block_id]: e.target.value }))
                            }
                            rows={3}
                            className="w-full bg-transparent text-[var(--ink-primary)] text-xs resize-none focus:outline-none font-mono leading-relaxed placeholder:text-[var(--ink-muted)]"
                          />
                        </div>

                        <div className="flex items-center justify-end gap-2 pt-2.5 border-t border-[var(--paper-border)] mt-3">
                          <Button
                            onClick={() => handleSave(seg)}
                            variant="primary"
                            size="sm"
                            disabled={!dirty || saving[seg.block_id]}
                            className="text-[11px] h-6 px-2.5 font-bold"
                          >
                            {saving[seg.block_id] ? (
                              <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                            ) : (
                              <Save className="h-3 w-3 mr-1" />
                            )}
                            {saving[seg.block_id] ? t.review.saving : t.review.saveFeedback}
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>
                )
              })}

              {total > segments.length && (
                <div className="text-center text-[11px] text-[var(--ink-muted)] pt-2">
                  {segments.length} / {total}
                </div>
              )}
            </>
          )}
        </div>
      ) : (
        /* Visual witness mode — backend support (page rasterization + single-page
           incremental re-render) is not built yet; show an honest placeholder
           rather than a fake diff. */
        <div className="flex-1 flex items-center justify-center p-8 text-center">
          <div className="max-w-md space-y-2">
            <Eye className="h-8 w-8 mx-auto text-[var(--paper-border-hover)]" />
            <p className="text-xs text-[var(--ink-secondary)] leading-relaxed">
              {t.review.visualPending}
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
