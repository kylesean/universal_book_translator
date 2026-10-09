import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useVirtualizer } from '@tanstack/react-virtual'
import {
  SplitSquareVertical,
  Save,
  Eye,
  Loader2,
  AlertTriangle,
  ChevronUp,
  ChevronDown,
  CheckCircle2,
  ImageOff,
  Wand2,
  RefreshCw,
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import {
  listSegments,
  listIssues,
  editSegment,
  getBlockTerms,
  propagateTerm,
  pagePreviewUrl,
  sourcePageUrl,
  type Segment,
  type IssuesReport,
  type TermViolation,
} from '@/api/client'
import { useI18n } from '@/i18n/useI18n'

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

//: Segments fetched per request. The backend caps a single page at 500, so this
//: is also the hard ceiling on one request; the list pages past it via
//: ``offset`` and the "load more" affordance rather than silently dropping the
//: tail of a long book (which is what a fixed 500-row fetch used to do — every
//: fault past the cut was invisible to the F8 fault walk).
const PAGE_LIMIT = 200

export function ReviewWorkbench() {
  const { t } = useI18n()
  const { jobId: routeJobId } = useParams<{ jobId: string }>()
  const jobId = routeJobId ?? null
  const [viewMode, setViewMode] = useState<'segments' | 'visual_witness'>('segments')
  const [filter, setFilter] = useState<Filter>('issues')

  const [segments, setSegments] = useState<Segment[]>([])
  const [segmentTotal, setSegmentTotal] = useState(0)
  const [issues, setIssues] = useState<IssuesReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [editing, setEditing] = useState<Record<string, boolean>>({})
  const [saving, setSaving] = useState<Record<string, boolean>>({})
  const [saved, setSaved] = useState<Record<string, boolean>>({})
  const [issueCursor, setIssueCursor] = useState(0)
  const [openPreviews, setOpenPreviews] = useState<Record<number, boolean>>({})
  const [previewBust, setPreviewBust] = useState(0)
  const [termNotice, setTermNotice] = useState<Record<string, string>>({})
  const [diffBlend, setDiffBlend] = useState(false)
  const [selectedBlockId, setSelectedBlockId] = useState<string | null>(null)
  const [inspectorOpen, setInspectorOpen] = useState(false)

  const scrollRef = useRef<HTMLDivElement>(null)

  const load = useCallback(async () => {
    if (!jobId) return
    setLoading(true)
    setError(null)
    try {
      const [segmentData, issueData] = await Promise.all([
        listSegments(jobId, { status: filter, limit: PAGE_LIMIT, offset: 0 }),
        listIssues(jobId),
      ])
      setSegments(segmentData.segments)
      setSegmentTotal(segmentData.total)
      setIssues(issueData)
      setSaved({})
      setIssueCursor(0)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load workbench')
    } finally {
      setLoading(false)
    }
  }, [jobId, filter])

  // Append the next window. The backend returns ``total`` with every page, so
  // the "load more" button can state how much is left instead of guessing.
  const loadMore = useCallback(async () => {
    if (!jobId || loadingMore) return
    setLoadingMore(true)
    setError(null)
    try {
      const data = await listSegments(jobId, {
        status: filter,
        limit: PAGE_LIMIT,
        offset: segments.length,
      })
      setSegments((prev) => {
        const seen = new Set(prev.map((segment) => segment.block_id))
        return [...prev, ...data.segments.filter((segment) => !seen.has(segment.block_id))]
      })
      setSegmentTotal(data.total)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load more segments')
    } finally {
      setLoadingMore(false)
    }
  }, [jobId, filter, segments.length, loadingMore])

  useEffect(() => {
    void load()
  }, [load])

  // Virtualize the segment list: a 10k-segment book keeps a constant DOM node
  // count (PRD §9 risk 1) instead of one card per segment.
  // @tanstack/react-virtual returns non-memoizable values by design; the compiler
  // skipping this hook's output is its documented behaviour, not a defect here.
  // oxlint-disable-next-line react/incompatible-library
  const virtualizer = useVirtualizer({
    count: segments.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 210,
    overscan: 6,
  })

  // The visual-witness view stacks one two-up page comparison per page. Without
  // virtualization a 400-page book mounts 800 <img> nodes and fires as many
  // rasterization requests at once — exactly the OOM the PRD forbids (PRD §9.1:
  // "never render unbounded DOM"). Windowed the same way as the segment grid.
  const previewPages = useMemo(
    () =>
      Array.from(
        new Set(
          segments.map((segment) => segment.page).filter((page): page is number => page !== null)
        )
      ),
    [segments]
  )
  const witnessScrollRef = useRef<HTMLDivElement>(null)
  // oxlint-disable-next-line react/incompatible-library
  const witnessVirtualizer = useVirtualizer({
    count: previewPages.length,
    getScrollElement: () => witnessScrollRef.current,
    estimateSize: () => 760,
    overscan: 2,
  })

  const issueIndices = useMemo(
    () =>
      segments
        .map((segment, index) => ({ segment, index }))
        .filter(
          ({ segment }) =>
            segment.issues.length > 0 ||
            segment.status === 'needs_human' ||
            segment.status === 'blocked_human'
        )
        .map(({ index }) => index),
    [segments]
  )

  const jumpToIssue = (delta: number) => {
    if (issueIndices.length === 0) return
    const next = (issueCursor + delta + issueIndices.length) % issueIndices.length
    setIssueCursor(next)
    virtualizer.scrollToIndex(issueIndices[next], { align: 'center' })
  }

  // F8 / Shift+F8 walk the fault list (PRD §5.1).
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'F8') return
      e.preventDefault()
      jumpToIssue(e.shiftKey ? -1 : 1)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  })

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
      // The edit changed the ledger text: any open page preview is now stale.
      if (segment.page !== null && openPreviews[segment.page]) {
        setPreviewBust(Date.now())
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save revision')
    } finally {
      setSaving((prev) => ({ ...prev, [segment.block_id]: false }))
    }
  }

  const togglePreview = (page: number | null) => {
    if (page === null) return
    setOpenPreviews((prev) => ({ ...prev, [page]: !prev[page] }))
  }

  const handleTermReplaced = (blockId: string, page: number | null, replacements: number) => {
    setTermNotice((prev) => ({
      ...prev,
      [blockId]: t.review.cascadeDone.replace('{n}', String(replacements)),
    }))
    if (page !== null && openPreviews[page]) {
      setPreviewBust(Date.now())
    }
    void load()
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

  const inspectorSegment = selectedBlockId
    ? (segments.find((segment) => segment.block_id === selectedBlockId) ?? null)
    : null

  const hasMore = segments.length < segmentTotal

  const renderPreview = (page: number) => (
    <div className="mt-3 rounded-[6px] border border-[var(--paper-border)] bg-[var(--paper-subsurface)] overflow-hidden">
      <div className="h-7 px-3 flex items-center justify-between text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] border-b border-[var(--paper-border)]">
        <span>
          {t.review.previewPage} · P{page}
        </span>
        <button
          type="button"
          onClick={() => setPreviewBust(Date.now())}
          title={t.review.rerenderHint}
          className="flex items-center gap-1 text-[var(--ink-secondary)] hover:text-[var(--ink-primary)] font-mono transition-colors cursor-pointer"
        >
          <RefreshCw className="h-3 w-3" />
          <span>{t.review.rerenderPage}</span>
        </button>
      </div>
      <img
        src={pagePreviewUrl(jobId, page, { dpi: 110, cacheBust: previewBust })}
        alt={t.review.targetCanvasAlt.replace('{n}', String(page))}
        className="w-full max-h-[560px] object-contain bg-white"
        onError={(e) => {
          const target = e.currentTarget
          target.style.display = 'none'
          const sibling = target.nextElementSibling as HTMLElement | null
          if (sibling) sibling.style.display = 'flex'
        }}
      />
      <div className="hidden items-center gap-2 p-4 text-xs text-[var(--ink-secondary)]">
        <ImageOff className="h-4 w-4 shrink-0" />
        {t.review.previewFailed}
      </div>
    </div>
  )

  return (
    <div className="flex-1 flex min-h-0 overflow-hidden bg-[var(--paper-bg)] text-[var(--ink-primary)]">
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
      {/* Precision Ribbon */}
      <div className="min-h-11 px-6 py-2 border-b border-[var(--paper-border)] bg-[var(--paper-surface)] flex flex-wrap items-center justify-between gap-2 shrink-0 shadow-2xs">
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-1.5 text-xs font-bold text-[var(--ink-primary)]">
            <SplitSquareVertical className="h-4 w-4 text-[#15803d]" />
            <span>{t.review.title}</span>
          </div>
          <div className="h-3 w-px bg-[var(--paper-border)]" />
          <div className="flex items-center gap-2 text-xs font-mono flex-wrap">
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
        loading ? (
          <div className="flex-1 flex items-center justify-center text-[var(--ink-muted)]">
            <Loader2 className="h-5 w-5 animate-spin" />
          </div>
        ) : segments.length === 0 ? (
          <div className="flex-1 flex items-center justify-center text-xs text-[var(--ink-muted)]">
            {t.review.noSegments}
          </div>
        ) : (
          <div ref={scrollRef} className="flex-1 overflow-y-auto">
            <div
              className="relative w-full max-w-5xl mx-auto py-6"
              style={{ height: virtualizer.getTotalSize() }}
            >
              {virtualizer.getVirtualItems().map((virtualRow) => {
                const seg = segments[virtualRow.index]
                const draft = drafts[seg.block_id] ?? seg.target_text
                const dirty = draft !== seg.target_text
                const previewOpen = seg.page !== null && openPreviews[seg.page]
                return (
                  <div
                    key={seg.block_id}
                    data-index={virtualRow.index}
                    ref={virtualizer.measureElement}
                    className="absolute top-0 left-0 w-full px-8"
                    style={{ transform: `translateY(${virtualRow.start}px)` }}
                  >
                    <div className="pb-4">
                      <div
                        onClick={() => {
                          setSelectedBlockId(seg.block_id)
                          setInspectorOpen(true)
                        }}
                        className={`rounded-lg border transition-colors shadow-2xs bg-[var(--paper-surface)] cursor-pointer ${
                          selectedBlockId === seg.block_id
                            ? 'border-[var(--ink-primary)]'
                            : seg.issues.length > 0
                              ? 'border-[#b45309]/50'
                              : 'border-[var(--paper-border)]'
                        }`}
                      >
                        <div className="h-8 px-3.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] flex items-center justify-between text-xs font-mono text-[var(--ink-muted)]">
                          <div className="flex items-center gap-2">
                            <span className="text-[var(--ink-primary)] font-bold">
                              {seg.block_id}
                            </span>
                            {seg.page !== null && (
                              <button
                                onClick={() => togglePreview(seg.page)}
                                title={t.review.previewPage}
                                className={`px-1.5 py-0.5 rounded text-xs font-semibold transition-colors ${
                                  previewOpen
                                    ? 'bg-[var(--ink-primary)] text-[var(--paper-bg)]'
                                    : 'bg-[var(--paper-border)] text-[var(--ink-secondary)] hover:text-[var(--ink-primary)]'
                                }`}
                              >
                                Page {seg.page}
                              </button>
                            )}
                            <span className="text-xs px-1.5 py-0.2 rounded bg-[var(--paper-border)] text-[var(--ink-secondary)] font-semibold">
                              {seg.block_type}
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            {seg.issues.map((kind) => (
                              <span
                                key={kind}
                                className="text-xs px-1.5 py-0.5 rounded bg-[#b45309]/15 text-[#b45309] font-semibold"
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
                              <Badge variant={seg.issues.length > 0 ? 'warning' : 'outline'} dot>
                                {seg.status.toUpperCase()}
                              </Badge>
                            )}
                          </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-[var(--paper-border)] text-xs">
                          <div className="p-4 text-[var(--ink-secondary)] font-mono leading-relaxed select-text bg-[var(--paper-subsurface)]">
                            <div className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] mb-1 font-semibold">
                              {t.review.sourceSegment}
                            </div>
                            {seg.source_text}
                          </div>

                          <div className="p-4 flex flex-col justify-between bg-[var(--paper-surface)]">
                            <div>
                              <div className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] mb-1.5 flex items-center justify-between font-semibold">
                                <span>{t.review.targetSegment}</span>
                                {saved[seg.block_id] && (
                                  <span className="text-[#15803d] font-sans font-medium">
                                    {t.review.saved}
                                  </span>
                                )}
                              </div>
                              {editing[seg.block_id] ? (
                                <textarea
                                  autoFocus
                                  value={draft}
                                  onChange={(e) =>
                                    setDrafts((prev) => ({
                                      ...prev,
                                      [seg.block_id]: e.target.value,
                                    }))
                                  }
                                  onKeyDown={(e) => {
                                    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                                      e.preventDefault()
                                      void handleSave(seg)
                                      setEditing((prev) => ({ ...prev, [seg.block_id]: false }))
                                    } else if (e.key === 'Escape') {
                                      e.preventDefault()
                                      setDrafts((prev) => {
                                        const next = { ...prev }
                                        delete next[seg.block_id]
                                        return next
                                      })
                                      setEditing((prev) => ({ ...prev, [seg.block_id]: false }))
                                    }
                                  }}
                                  rows={3}
                                  className="w-full bg-transparent text-[var(--ink-primary)] text-xs resize-none focus:outline-none font-mono leading-relaxed placeholder:text-[var(--ink-muted)]"
                                />
                              ) : (
                                <div
                                  onDoubleClick={() =>
                                    setEditing((prev) => ({ ...prev, [seg.block_id]: true }))
                                  }
                                  title={t.review.editHint}
                                  className="min-h-[3rem] text-[var(--ink-primary)] text-xs font-mono leading-relaxed whitespace-pre-wrap cursor-text"
                                >
                                  {seg.target_text || '—'}
                                </div>
                              )}
                              {seg.issues.includes('terminology') && (
                                <TermPanel
                                  jobId={jobId}
                                  segment={seg}
                                  onReplaced={(replacements) =>
                                    handleTermReplaced(seg.block_id, seg.page, replacements)
                                  }
                                />
                              )}
                            </div>

                            <div className="flex items-center justify-end gap-2 pt-2.5 border-t border-[var(--paper-border)] mt-3">
                              {termNotice[seg.block_id] && (
                                <span className="text-[#15803d] text-xs font-medium mr-auto">
                                  {termNotice[seg.block_id]}
                                </span>
                              )}
                              {editing[seg.block_id] && (
                                <Button
                                  onClick={() => {
                                    setDrafts((prev) => {
                                      const next = { ...prev }
                                      delete next[seg.block_id]
                                      return next
                                    })
                                    setEditing((prev) => ({ ...prev, [seg.block_id]: false }))
                                  }}
                                  variant="secondary"
                                  size="sm"
                                  className="text-xs h-6 px-2.5"
                                >
                                  {t.review.cancelEdit}
                                </Button>
                              )}
                              <Button
                                onClick={() => handleSave(seg)}
                                variant="primary"
                                size="sm"
                                disabled={!editing[seg.block_id] || !dirty || saving[seg.block_id]}
                                className="text-xs h-6 px-2.5 font-bold"
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

                      {previewOpen && seg.page !== null && renderPreview(seg.page)}
                    </div>
                  </div>
                )
              })}
            </div>
            {hasMore && (
              <div className="flex justify-center pb-6">
                <Button
                  onClick={() => void loadMore()}
                  variant="secondary"
                  size="sm"
                  disabled={loadingMore}
                >
                  {loadingMore ? (
                    <Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />
                  ) : (
                    <ChevronDown className="h-3.5 w-3.5 mr-1.5" />
                  )}
                  {t.review.loadMore} ({segmentTotal - segments.length})
                </Button>
              </div>
            )}
          </div>
        )
      ) : (
        /* Visual witness: source page vs the composed translation (PRD §5.1 mode B). */
        <div ref={witnessScrollRef} className="flex-1 overflow-y-auto">
          <div className="max-w-6xl mx-auto w-full px-8 py-6">
            <div className="flex items-center justify-end">
              <label className="flex items-center gap-1.5 text-xs text-[var(--ink-secondary)] cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={diffBlend}
                  onChange={(e) => setDiffBlend(e.target.checked)}
                />
                {t.review.diffBlend}
              </label>
            </div>
            {previewPages.length === 0 ? (
              <div className="py-12 text-center text-xs text-[var(--ink-muted)]">
                {t.review.noSegments}
              </div>
            ) : (
              <div
                className="relative w-full mt-4"
                style={{ height: witnessVirtualizer.getTotalSize() }}
              >
                {witnessVirtualizer.getVirtualItems().map((virtualRow) => {
                  const page = previewPages[virtualRow.index]
                  return (
                    <div
                      key={page}
                      data-index={virtualRow.index}
                      ref={witnessVirtualizer.measureElement}
                      className="absolute top-0 left-0 w-full pb-4"
                      style={{ transform: `translateY(${virtualRow.start}px)` }}
                    >
                      <div className="rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] shadow-2xs overflow-hidden">
                        <div className="h-8 px-3.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] flex items-center justify-between text-xs font-mono text-[var(--ink-muted)]">
                          <span className="flex items-center gap-1.5">
                            <Eye className="h-3.5 w-3.5" /> Page {page}
                          </span>
                          <button
                            type="button"
                            onClick={() => setPreviewBust(Date.now())}
                            title={t.review.rerenderHint}
                            className="flex items-center gap-1 text-xs text-[var(--ink-secondary)] hover:text-[var(--ink-primary)] font-mono transition-colors cursor-pointer"
                          >
                            <RefreshCw className="h-3 w-3" />
                            <span>{t.review.rerenderPage}</span>
                          </button>
                        </div>
                        {diffBlend ? (
                          <div className="relative bg-white">
                            <img
                              src={sourcePageUrl(jobId, page, { dpi: 110 })}
                              alt={t.review.sourceCanvasAlt.replace('{n}', String(page))}
                              className="w-full max-h-[720px] object-contain"
                              onError={(e) => {
                                e.currentTarget.style.display = 'none'
                              }}
                            />
                            <img
                              src={pagePreviewUrl(jobId, page, { dpi: 110, cacheBust: previewBust })}
                              alt={t.review.targetCanvasAlt.replace('{n}', String(page))}
                              className="absolute inset-0 w-full h-full object-contain opacity-50 mix-blend-multiply"
                              onError={(e) => {
                                e.currentTarget.style.display = 'none'
                              }}
                            />
                          </div>
                        ) : (
                          <div className="grid grid-cols-2 divide-x divide-[var(--paper-border)]">
                            <div className="bg-white">
                              <div className="px-3 py-1.5 text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] border-b border-[var(--paper-border)]">
                                {t.review.sourceCanvas}
                              </div>
                              <img
                                src={sourcePageUrl(jobId, page, { dpi: 110 })}
                                alt={t.review.sourceCanvasAlt.replace('{n}', String(page))}
                                className="w-full max-h-[640px] object-contain"
                                onError={(e) => {
                                  e.currentTarget.style.display = 'none'
                                }}
                              />
                            </div>
                            <div className="bg-white">
                              <div className="px-3 py-1.5 text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] border-b border-[var(--paper-border)]">
                                {t.review.targetCanvas}
                              </div>
                              <img
                                src={pagePreviewUrl(jobId, page, { dpi: 110, cacheBust: previewBust })}
                                alt={t.review.targetCanvasAlt.replace('{n}', String(page))}
                                className="w-full max-h-[640px] object-contain"
                                onError={(e) => {
                                  const target = e.currentTarget
                                  target.style.display = 'none'
                                  const sibling = target.nextElementSibling as HTMLElement | null
                                  if (sibling) sibling.style.display = 'flex'
                                }}
                              />
                              <div className="hidden items-center gap-2 p-4 text-xs text-[var(--ink-secondary)]">
                                <ImageOff className="h-4 w-4 shrink-0" />
                                {t.review.previewFailed}
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>
      )}
      </div>

      {inspectorOpen && inspectorSegment && (
        <aside className="w-80 border-l border-[var(--paper-border)] bg-[var(--paper-surface)] overflow-y-auto shrink-0 select-text">
          <div className="h-11 px-4 border-b border-[var(--paper-border)] flex items-center justify-between">
            <span className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
              {t.review.inspectorTitle}
            </span>
            <button
              onClick={() => setInspectorOpen(false)}
              className="text-xs font-mono text-[var(--ink-muted)] hover:text-[var(--ink-primary)]"
            >
              ✕
            </button>
          </div>
          <div className="p-4 space-y-3 text-xs">
            <div>
              <div className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
                {inspectorSegment.block_id}
              </div>
              <div className="text-xs text-[var(--ink-secondary)] font-mono mt-0.5">
                {inspectorSegment.block_type} · P{inspectorSegment.page ?? '—'}
              </div>
            </div>

            <dl className="divide-y divide-[var(--paper-border)] text-xs font-mono">
              <div className="py-1.5 flex justify-between">
                <dt className="text-[var(--ink-muted)]">{t.review.inspectorQe}</dt>
                <dd className="font-bold">
                  {inspectorSegment.mtqe_score === null
                    ? '—'
                    : `${(inspectorSegment.mtqe_score * 100).toFixed(0)}%`}
                </dd>
              </div>
              <div className="py-1.5 flex justify-between">
                <dt className="text-[var(--ink-muted)]">{t.review.inspectorTmHit}</dt>
                <dd>{inspectorSegment.tm_hit ? '✓' : '—'}</dd>
              </div>
              <div className="py-1.5 flex justify-between">
                <dt className="text-[var(--ink-muted)]">{t.review.inspectorRepairs}</dt>
                <dd>{inspectorSegment.repair_rounds}</dd>
              </div>
              <div className="py-1.5 flex justify-between">
                <dt className="text-[var(--ink-muted)]">{t.review.inspectorSeverity}</dt>
                <dd>{inspectorSegment.mqm_severity ?? '—'}</dd>
              </div>
            </dl>

            {inspectorSegment.glossary_hits.length > 0 && (
              <div>
                <div className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold mb-1">
                  {t.review.inspectorGlossary}
                </div>
                <div className="flex flex-wrap gap-1">
                  {inspectorSegment.glossary_hits.map((term) => (
                    <span
                      key={term}
                      className="px-1.5 py-0.5 rounded bg-[#15803d]/10 text-[#15803d] font-mono text-xs"
                    >
                      {term}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {inspectorSegment.error_flags.length > 0 && (
              <div>
                <div className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold mb-1">
                  error_flags
                </div>
                <ul className="space-y-0.5">
                  {inspectorSegment.error_flags.map((flag, index) => (
                    <li key={index} className="text-xs font-mono text-[var(--ink-secondary)] break-all">
                      {flag}
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {Object.keys(inspectorSegment.provenance).length > 0 && (
              <div>
                <div className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold mb-1">
                  {t.review.inspectorProvenance}
                </div>
                <pre className="text-xs font-mono text-[var(--ink-secondary)] whitespace-pre-wrap break-all bg-[var(--paper-subsurface)] rounded p-2">
                  {JSON.stringify(inspectorSegment.provenance, null, 1)}
                </pre>
              </div>
            )}
          </div>
        </aside>
      )}
    </div>
  )
}

interface TermPanelProps {
  jobId: string
  segment: Segment
  onReplaced: (replacements: number) => void
}

/**
 * Terminology recommendations for one segment (PRD §5.2.2). Fetches lazily on
 * mount so only the virtualizer's visible window hits the backend, and shows the
 * "fix all N" cascade checkbox with the book-wide count.
 */
function TermPanel({ jobId, segment, onReplaced }: TermPanelProps) {
  const { t } = useI18n()
  const [violations, setViolations] = useState<TermViolation[] | null>(null)
  const [cascade, setCascade] = useState(false)
  const [confirmViolation, setConfirmViolation] = useState<TermViolation | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let alive = true
    getBlockTerms(jobId, segment.block_id)
      .then((report) => {
        if (alive) setViolations(report.violations)
      })
      .catch(() => {
        if (alive) setViolations([])
      })
    return () => {
      alive = false
    }
  }, [jobId, segment.block_id, segment.target_text])

  const handleTriggerReplace = (violation: TermViolation) => {
    if (cascade && violation.cascade_all > 0) {
      setConfirmViolation(violation)
    } else {
      void executeReplace(violation)
    }
  }

  const executeReplace = async (violation: TermViolation) => {
    setBusy(true)
    setError(null)
    setConfirmViolation(null)
    try {
      const res = await propagateTerm(jobId, {
        block_id: segment.block_id,
        surface: violation.surface,
        expected: violation.expected,
        scope: cascade ? 'all' : 'block',
      })
      onReplaced(res.replacements)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to propagate term')
    } finally {
      setBusy(false)
    }
  }

  if (violations === null || violations.length === 0) return null

  return (
    <div className="mt-3 rounded-[6px] border border-[#b45309]/40 bg-[#b45309]/5 divide-y divide-[#b45309]/20">
      {violations.map((violation) => {
        const isConfirming = confirmViolation === violation
        return (
          <div
            key={`${violation.surface}:${violation.expected}`}
            className="px-2.5 py-2 flex flex-col gap-2 text-xs"
          >
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-mono text-[var(--ink-muted)]">{t.review.recommendedTerm}</span>
              <span className="line-through text-[#b45309] font-mono">{violation.surface}</span>
              <span className="text-[var(--ink-muted)]">→</span>
              <span className="font-semibold text-[#15803d] font-mono">{violation.expected}</span>
              {violation.cascade_all > 0 && (
                <label className="flex items-center gap-1 text-[var(--ink-secondary)] cursor-pointer ml-auto select-none">
                  <input
                    type="checkbox"
                    checked={cascade}
                    onChange={(e) => {
                      setCascade(e.target.checked)
                      if (!e.target.checked) setConfirmViolation(null)
                    }}
                  />
                  {t.review.cascadeFix.replace('{n}', String(violation.cascade_all))}
                </label>
              )}
              <Button
                onClick={() => handleTriggerReplace(violation)}
                variant="primary"
                size="sm"
                disabled={busy}
                className="text-xs h-6 px-2.5 font-bold"
              >
                {busy ? (
                  <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                ) : (
                  <Wand2 className="h-3 w-3 mr-1" />
                )}
                {busy ? t.review.cascadeApplying : t.review.replaceTerm}
              </Button>
            </div>

            {isConfirming && (
              <div className="w-full mt-1 p-2 rounded-[4px] bg-[#b45309]/10 border border-[#b45309]/30 space-y-1.5 animate-in fade-in duration-150">
                <div className="font-semibold text-[var(--ink-primary)]">
                  {t.review.cascadeConfirmTitle}
                </div>
                <div className="text-[var(--ink-secondary)] text-[10.5px] leading-relaxed">
                  {t.review.cascadeConfirmDesc
                    .replace('{n}', String(violation.cascade_all))
                    .replace('{surface}', violation.surface)
                    .replace('{expected}', violation.expected)}
                </div>
                <div className="flex items-center gap-2 pt-0.5">
                  <Button
                    onClick={() => executeReplace(violation)}
                    variant="primary"
                    size="sm"
                    disabled={busy}
                    className="text-xs h-5 px-2 bg-[#b45309] hover:bg-[#92400e] text-white font-bold"
                  >
                    {busy ? <Loader2 className="h-3 w-3 animate-spin mr-1" /> : null}
                    {t.review.cascadeConfirmProceed}
                  </Button>
                  <Button
                    onClick={() => setConfirmViolation(null)}
                    variant="secondary"
                    size="sm"
                    disabled={busy}
                    className="text-xs h-5 px-2"
                  >
                    {t.review.cascadeConfirmCancel}
                  </Button>
                </div>
              </div>
            )}
          </div>
        )
      })}
      {error && <div className="px-2.5 py-1.5 text-xs text-[#b45309]">{error}</div>}
    </div>
  )
}
