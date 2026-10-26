import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { useVirtualizer, type VirtualItem } from '@tanstack/react-virtual'
import {
  SplitSquareVertical,
  Save,
  Eye,
  EyeOff,
  Loader2,
  AlertTriangle,
  ChevronUp,
  ChevronDown,
  CheckCircle2,
  ImageOff,
  Wand2,
  RefreshCw,
  Sparkles,
  Filter,
  Edit3,
  X,
  ArrowLeft,
  Info,
  ListFilter,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from '@/components/ui/resizable'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import { Switch } from '@/components/ui/switch'
import { useIsMobile } from '@/hooks/use-mobile'
import {
  ApiError,
  listSegments,
  listIssues,
  listDeliverables,
  editSegment,
  reExportJob,
  getBlockTerms,
  propagateTerm,
  pagePreviewUrl,
  sourcePageUrl,
  type Segment,
  type IssuesReport,
  type TermViolation,
} from '@/api/client'
import { useI18n } from '@/i18n/useI18n'

type Filter = 'issues' | 'all' | 'needs_human' | string
type Scope = 'issues' | 'all'

function scopeLabel(t: ReturnType<typeof useI18n>['t'], scope: Scope): string {
  return scope === 'issues' ? t.review.scopeToReview : t.review.scopeEverything
}

function segmentLocation(t: ReturnType<typeof useI18n>['t'], segment: Segment): string {
  const parts: string[] = []
  if (segment.page !== null) {
    parts.push(t.review.pageLabel.replace('{n}', String(segment.page)))
  }
  parts.push(blockTypeLabel(t, segment.block_type))
  parts.push(t.review.blockPosition.replace('{n}', String(segment.spine_index + 1)))
  return parts.join(' · ')
}

//: Fallback label for an issue kind the dictionary does not name yet. The
//: backend can add a kind before the console learns its translation; showing
//: the raw key beats showing nothing. The known kinds live in the i18n tables
//: (``t.review.issueKinds``) so the ribbon follows the selected language.
const issueLabel = (t: ReturnType<typeof useI18n>['t'], kind: string): string =>
  t.review.issueKinds[kind] ?? kind

//: Same idea for block types and statuses, which arrive as engine identifiers.
const blockTypeLabel = (t: ReturnType<typeof useI18n>['t'], kind: string): string =>
  t.review.blockTypes[kind] ?? kind

const blockStatusLabel = (t: ReturnType<typeof useI18n>['t'], status: string): string =>
  t.review.blockStatuses[status] ?? status.toUpperCase()

//: MQM triage tier, or an em dash when the block was never triaged.
const mqmSeverityLabel = (t: ReturnType<typeof useI18n>['t'], severity: string | null): string =>
  severity ? (t.review.mqmSeverities[severity] ?? severity) : '—'

// Markdown, GFM and KaTeX are only needed once a segment is visible. Loading
// that renderer with the workbench would put the whole parser and its fonts in
// the route chunk even when the operator only opens the evidence panel.
const RichText = React.lazy(() =>
  import('@/components/content/RichText').then((module) => ({
    default: module.RichText,
  }))
)

function RichTextFallback() {
  return (
    <div className="space-y-2 py-1" aria-busy="true">
      <div className="h-4 w-[92%] animate-pulse rounded bg-muted" />
      <div className="h-4 w-[78%] animate-pulse rounded bg-muted" />
      <div className="h-4 w-[64%] animate-pulse rounded bg-muted" />
    </div>
  )
}

function findingLabel(t: ReturnType<typeof useI18n>['t'], kind: string): string {
  const labels: Record<string, string> = {
    review: t.review.flagHumanReview,
    critical: t.review.flagCritical,
    omission: t.review.flagOmission,
    numeric: t.review.flagNumeric,
    terminology: t.review.flagTerminology,
    formula: t.review.flagFormula,
    structure: t.review.flagStructure,
    repetition: t.review.flagRepetition,
    fabrication: t.review.flagFabrication,
    echo: t.review.flagEcho,
    render: t.review.flagRender,
  }
  return labels[kind] ?? t.review.flagGeneric
}

function findingKindForFlag(flag: string): string {
  const value = flag.toLowerCase()
  if (value.includes('needs_human_review')) return 'review'
  if (value.includes('mqm_critical_blocked')) return 'critical'
  if (value.includes('omission') || value.includes('empty target') || value.includes('table dropped')) {
    return 'omission'
  }
  if (value.includes('numeric')) return 'numeric'
  if (value.includes('glossary') || value.includes('consistency:')) return 'terminology'
  if (value.includes('formula') || value.includes('math') || value.includes('latex')) return 'formula'
  if (value.includes('repetition') || value.includes('loop hallucination')) return 'repetition'
  if (value.includes('fabrication') || value.includes('added reference') || value.includes('prompt')) {
    return 'fabrication'
  }
  if (value.includes('render_skip')) return 'render'
  if (value.includes('echo')) return 'echo'
  return 'structure'
}

/**
 * Uncommitted review edits, mirrored to `sessionStorage`.
 *
 * `drafts` lived only in component state, so leaving the workbench — to check
 * the quality gate, to look at another block, or simply a reload — discarded
 * every edit the reviewer had typed but not yet saved, with no warning. On a
 * 100-page book that is a real amount of work, and nothing in the UI said it
 * was at risk.
 *
 * `sessionStorage`, not `localStorage`: a draft belongs to this editing session
 * and this job; keeping it past the tab would resurrect half-finished text days
 * later. Keyed by job so two jobs never mix.
 */
const _DRAFT_PREFIX = 'ubt_review_drafts:'

function readDrafts(jobId: string | null): Record<string, string> {
  if (!jobId) return {}
  try {
    const raw = sessionStorage.getItem(_DRAFT_PREFIX + jobId)
    if (!raw) return {}
    const parsed: unknown = JSON.parse(raw)
    if (parsed === null || typeof parsed !== 'object') return {}
    const out: Record<string, string> = {}
    for (const [k, v] of Object.entries(parsed as Record<string, unknown>)) {
      if (typeof v === 'string') out[k] = v
    }
    return out
  } catch {
    return {}   // private mode, or a corrupt entry: start clean rather than crash
  }
}

function writeDrafts(jobId: string | null, drafts: Record<string, string>): void {
  if (!jobId) return
  try {
    if (Object.keys(drafts).length === 0) {
      sessionStorage.removeItem(_DRAFT_PREFIX + jobId)
    } else {
      sessionStorage.setItem(_DRAFT_PREFIX + jobId, JSON.stringify(drafts))
    }
  } catch {
    // Storage refused: the in-memory copy still carries the session.
  }
}

//: Segments fetched per request. The backend caps a single page at 500, so this
//: is also the hard ceiling on one request; the list pages past it via
//: ``offset`` and the "load more" affordance rather than silently dropping the
//: tail of a long book (which is what a fixed 500-row fetch used to do — every
//: fault past the cut was invisible to the F8 fault walk).
const PAGE_LIMIT = 200

export function ReviewWorkbench() {
  const { t } = useI18n()
  const navigate = useNavigate()
  const { jobId: routeJobId } = useParams<{ jobId: string }>()
  const [searchParams] = useSearchParams()
  const jobId = routeJobId ?? null
  const [viewMode, setViewMode] = useState<'segments' | 'reading_flow'>(
    searchParams.get('view') === 'flow' ? 'reading_flow' : 'segments'
  )
  const [filter, setFilter] = useState<Filter>('issues')

  const [segments, setSegments] = useState<Segment[]>([])
  const [segmentTotal, setSegmentTotal] = useState(0)
  const [issues, setIssues] = useState<IssuesReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [drafts, setDrafts] = useState<Record<string, string>>(() => readDrafts(jobId))
  const [editing, setEditing] = useState<Record<string, boolean>>({})
  const [saving, setSaving] = useState<Record<string, boolean>>({})
  const [saved, setSaved] = useState<Record<string, boolean>>({})
  const [reExportNotice, setReExportNotice] = useState<string | null>(null)
  const [reExporting, setReExporting] = useState(false)
  const [reExportPending, setReExportPending] = useState(false)
  const [issueCursor, setIssueCursor] = useState(0)
  const [previewPage, setPreviewPage] = useState<number | null>(null)
  const [previewBust, setPreviewBust] = useState(0)
  const [termNotice, setTermNotice] = useState<Record<string, string>>({})
  const [diffBlend, setDiffBlend] = useState(false)
  const [selectedBlockId, setSelectedBlockId] = useState<string | null>(null)
  const [inspectorOpen, setInspectorOpen] = useState(false)
  const [desktopEvidenceOpen, setDesktopEvidenceOpen] = useState(true)
  const isMobile = useIsMobile()

  const scrollRef = useRef<HTMLDivElement>(null)
  // Guards `load` against a stale response clobbering a newer filter's result:
  // the ribbon switches filters quickly, and the slow request for the old
  // filter can resolve after the new one. Same pattern as the wizard's assess
  // guard.
  const loadSeqRef = useRef(0)

  const load = useCallback(async () => {
    if (!jobId) return
    const seq = ++loadSeqRef.current
    setLoading(true)
    setError(null)
    try {
      const [segmentData, issueData, listing] = await Promise.all([
        listSegments(jobId, { status: filter, limit: PAGE_LIMIT, offset: 0 }),
        listIssues(jobId),
        // The delivered file's freshness, from the ledger — not from this tab's
        // memory. `reExportPending` used to be set only by a save in this
        // session, so reopening the workbench (or another reviewer's browser)
        // showed a clean "re-render" button while the delivered files were
        // still stale. The backend already answers this on the deliverables
        // endpoint; the review pane just never asked.
        listDeliverables(jobId).catch(() => null),
      ])
      if (seq !== loadSeqRef.current) return // stale response
      setSegments(segmentData.segments)
      setSegmentTotal(segmentData.total)
      setIssues(issueData)
      setSaved({})
      setIssueCursor(0)
      if (listing) {
        setReExportPending(listing.stale)
      }
    } catch (err) {
      if (seq !== loadSeqRef.current) return // stale failure
      setError(err instanceof Error ? err.message : t.review.loadFailed)
    } finally {
      if (seq === loadSeqRef.current) setLoading(false)
    }
  }, [jobId, filter, t])

  // Append the next window. The backend returns ``total`` with every page, so
  // the "load more" button can state how much is left instead of guessing.
  // Same staleness guard as `load`: a slow page for the old filter/job that
  // lands after a switch would splice foreign blocks into the new list (the
  // block_id dedup cannot see that) and inflate the next offset.
  const loadMore = useCallback(async () => {
    if (!jobId || loadingMore) return
    const seq = loadSeqRef.current
    setLoadingMore(true)
    setError(null)
    try {
      const data = await listSegments(jobId, {
        status: filter,
        limit: PAGE_LIMIT,
        offset: segments.length,
      })
      if (seq !== loadSeqRef.current) return // stale response
      setSegments((prev) => {
        const seen = new Set(prev.map((segment) => segment.block_id))
        return [...prev, ...data.segments.filter((segment) => !seen.has(segment.block_id))]
      })
      setSegmentTotal(data.total)
    } catch (err) {
      if (seq !== loadSeqRef.current) return // stale failure
      setError(err instanceof Error ? err.message : t.review.loadMoreFailed)
    } finally {
      if (seq === loadSeqRef.current) setLoadingMore(false)
    }
  }, [jobId, filter, segments.length, loadingMore, t])

  useEffect(() => {
    void load()
  }, [load])

  // Mirror the drafts to sessionStorage on every change, and re-read them when
  // the job changes: switching from job A to job B inside this view does not
  // remount it. The two effects run in declaration order on the same commit,
  // so on a job switch the write must NOT fire with the OLD drafts under the
  // NEW key — the ref marks which job the in-memory drafts belong to, and the
  // write skips until the read effect has swapped them in. Without it, the
  // "protection" persisted A's drafts into B's storage key and read them
  // straight back.
  const draftsJobRef = useRef<string | null>(jobId)
  useEffect(() => {
    if (draftsJobRef.current === jobId) writeDrafts(jobId, drafts)
  }, [jobId, drafts])

  useEffect(() => {
    draftsJobRef.current = jobId
    setDrafts(readDrafts(jobId))
  }, [jobId])

  // Snapshots of measured item heights to survive tab switches and eliminate layout shift
  const flowMeasurementsCacheRef = useRef<VirtualItem[]>([])
  const getSegmentKey = useCallback(
    (index: number) => segments[index]?.block_id ?? index,
    [segments]
  )

  // Dynamic estimateSize calculation based on source and target character counts
  const estimateFlowSize = useCallback(
    (index: number) => {
      const seg = segments[index]
      if (!seg) return 130
      const sourceLen = seg.source_text?.length ?? 0
      const draftText = drafts[seg.block_id] ?? seg.delivered_text ?? ''
      const targetLen = draftText.length
      // In dual-column reading flow:
      // Left column: source text (monospace ~40 chars/line)
      // Right column: target text (prose ~30 chars/line)
      const sourceLines = Math.ceil(sourceLen / 40)
      const targetLines = Math.ceil(targetLen / 30)
      const maxLines = Math.max(1, Math.max(sourceLines, targetLines))
      // Metadata header + p-4 card padding + bottom spacing = 68px. Line height = 24px.
      const base = 68 + maxLines * 24
      const hasIssues = seg.issues.length > 0 || seg.quarantined || seg.repair_failed
      return Math.min(850, Math.max(90, base + (hasIssues ? 24 : 0)))
    },
    [segments, drafts]
  )

  // The list is a master-detail workspace: one focused segment at a time. It
  // is still paged at PAGE_LIMIT, but rendering every visible row as a virtual
  // card made selecting a row move the target editor and left the page looking
  // like a stack of unrelated documents. Keep the list bounded by the backend
  // page size and use native document flow here.
  const virtualizer = {
    scrollToIndex: (index: number) => {
      document
        .getElementById(`review-segment-${index}`)
        ?.scrollIntoView({ behavior: 'smooth', block: 'center' })
    },
  }

  const flowScrollRef = useRef<HTMLDivElement>(null)
  // Virtualize Reading Flow with dynamic height measurement and persistent cache snapshot
  // oxlint-disable-next-line react/incompatible-library
  const flowVirtualizer = useVirtualizer({
    count: segments.length,
    getScrollElement: () => flowScrollRef.current,
    estimateSize: estimateFlowSize,
    getItemKey: getSegmentKey,
    initialMeasurementsCache: flowMeasurementsCacheRef.current,
    onChange: (instance) => {
      flowMeasurementsCacheRef.current = instance.takeSnapshot()
    },
    overscan: 6,
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

  // Start on the first fault so the reviewer immediately sees the sentence
  // that needs attention instead of an unexplained blank inspector.
  useEffect(() => {
    if (segments.length === 0 || selectedBlockId) return
    // oxlint-disable-next-line react/set-state-in-effect
    setSelectedBlockId(segments[issueIndices[0] ?? 0]?.block_id ?? null)
  }, [segments, issueIndices, selectedBlockId])

  const selectedIndex = selectedBlockId
    ? segments.findIndex((segment) => segment.block_id === selectedBlockId)
    : -1

  // Opening a segment from a narrow-screen issue queue surfaces the same
  // evidence that is always visible beside the editor on desktop.
  const selectSegment = (index: number) => {
    const segment = segments[index]
    if (!segment) return
    setSelectedBlockId(segment.block_id)
    setIssueCursor(Math.max(0, issueIndices.indexOf(index)))
  }

  const selectAndReveal = (index: number) => {
    selectSegment(index)
    if (isMobile) setInspectorOpen(true)
  }

  const moveSegment = (delta: number): number => {
    if (segments.length === 0) return -1
    const current = selectedIndex >= 0 ? selectedIndex : delta > 0 ? -1 : 1
    const next = Math.min(segments.length - 1, Math.max(0, current + delta))
    selectSegment(next)
    return next
  }

  const scrollToSelected = (index = selectedIndex) => {
    if (index < 0) return
    if (viewMode === 'reading_flow') {
      flowVirtualizer.scrollToIndex(index, { align: 'center' })
    } else if (viewMode === 'segments') {
      virtualizer.scrollToIndex(index)
    }
  }

  const jumpToIssue = (delta: number) => {
    if (issueIndices.length === 0) return
    const next = (issueCursor + delta + issueIndices.length) % issueIndices.length
    setIssueCursor(next)
    setSelectedBlockId(segments[issueIndices[next]]?.block_id ?? null)
    virtualizer.scrollToIndex(issueIndices[next])
  }

  // F8 / Shift+F8 walk the fault list (PRD §5.1). J / K navigate segments. Escape closes the inspector.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      const active = document.activeElement
      const isInput = active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA')

      if (e.key === 'Escape' && inspectorOpen) {
        if (isInput) return
        setInspectorOpen(false)
        return
      }
      if (e.key === 'F8') {
        e.preventDefault()
        jumpToIssue(e.shiftKey ? -1 : 1)
        return
      }
      if (!isInput && !e.metaKey && !e.ctrlKey && !e.altKey && segments.length > 0) {
        if (e.key === 'j') {
          e.preventDefault()
          moveSegment(1)
        } else if (e.key === 'k') {
          e.preventDefault()
          moveSegment(-1)
        }
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  })

  const handleSave = async (segment: Segment) => {
    if (!jobId) return
    const draft = drafts[segment.block_id]
    if (draft === undefined || draft === segment.delivered_text) return
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
      // The edit changed the ledger text: an open page preview is now stale.
      if (segment.page !== null && previewPage === segment.page) {
        setPreviewBust(Date.now())
      }
      // The ledger text changed, but the *delivered file* was rendered before
      // the edit and would otherwise keep the machine draft the reviewer just
      // corrected. Re-rendering is NOT triggered here: it rebuilds the whole
      // book, so firing it per save paid a full render for every keystroke
      // batch a reviewer committed. It is now an explicit action (the
      // "re-render" button) so a review session re-renders once, at the end.
      setReExportPending(true)
    } catch (err) {
      // A 409 is not "the save failed" — it means this job's ledger is held by
      // another writer (a running resume, another reviewer's edit), or the run
      // started again underneath the workbench. Retrying blindly would lose the
      // draft, so say which case it is and keep the text in `drafts` for a
      // retry once the conflict clears.
      const conflict = err instanceof ApiError && err.status === 409
      setError(
        conflict
          ? `${t.review.saveConflict} ${err instanceof Error ? err.message : ''}`.trim()
          : err instanceof Error
            ? err.message
            : t.review.saveFailed
      )
    } finally {
      setSaving((prev) => ({ ...prev, [segment.block_id]: false }))
    }
  }

  const handleReExport = async () => {
    if (!jobId || reExporting) return
    setReExporting(true)
    setReExportNotice(null)
    try {
      const out = await reExportJob(jobId)
      setReExportPending(false)
      setReExportNotice(t.review.reExportDone.replace('{path}', out.artifact_path ?? ''))
    } catch (err) {
      setReExportNotice(
        t.review.reExportFailed.replace(
          '{msg}',
          err instanceof Error ? err.message : 'unknown error'
        )
      )
    } finally {
      setReExporting(false)
    }
  }

  const openLayoutPreview = (page: number | null) => {
    if (page === null) return
    setPreviewPage(page)
  }

  const handleTermReplaced = (blockId: string, page: number | null, replacements: number) => {
    setTermNotice((prev) => ({
      ...prev,
      [blockId]: t.review.cascadeDone.replace('{n}', String(replacements)),
    }))
    if (page !== null && previewPage === page) {
      setPreviewBust(Date.now())
    }
    // Refresh only the first page and splice it in front of the paged-in tail:
    // a full `load()` here resets the segment list back to page one, throwing
    // away every "load more" window the reviewer had paged through. There is
    // no single-segment endpoint, so a replaced block deep in the tail keeps
    // its pre-cascade text until the next explicit reload — the notice above
    // still explains what happened.
    if (!jobId) return
    const seq = loadSeqRef.current
    void listSegments(jobId, { status: filter, limit: PAGE_LIMIT, offset: 0 })
      .then((data) => {
        if (seq !== loadSeqRef.current) return // stale response
        const firstPageIds = new Set(data.segments.map((segment) => segment.block_id))
        setSegments((prev) => [
          ...data.segments,
          ...prev.filter((segment) => !firstPageIds.has(segment.block_id)),
        ])
        setSegmentTotal(data.total)
      })
      .catch(() => {
        /* the notice above already explains the outcome; a silent refresh
           failure must not surface as a load error */
      })
  }

  if (!jobId) {
    return (
      <div className="flex-1 flex items-center justify-center p-8 text-center text-muted-foreground">
        <div className="max-w-sm space-y-2">
          <SplitSquareVertical className="h-8 w-8 mx-auto text-muted-foreground/40" />
          <h2 className="text-sm font-semibold text-foreground">{t.quality.noJobTitle}</h2>
          <p className="text-xs text-muted-foreground leading-relaxed">{t.quality.noJobDesc}</p>
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

  const scope: Scope =
    filter === 'all' ? 'all' : filter === 'needs_human' ? 'issues' : filter === 'issues' ? 'issues' : 'issues'
  const hasMore = segments.length < segmentTotal

  const saveAndContinue = async (segment: Segment) => {
    await handleSave(segment)
    setEditing((prev) => ({ ...prev, [segment.block_id]: false }))
    const next = moveSegment(1)
    requestAnimationFrame(() => scrollToSelected(next))
  }

  const evidencePanel = inspectorSegment ? (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-b border-border/70 px-4 py-3">
        <div className="flex items-center gap-2">
          <Info className="size-4 text-muted-foreground" />
          <h2 className="text-sm font-semibold text-foreground">{t.review.evidenceTitle}</h2>
        </div>
        <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
          {t.review.evidenceHint}
        </p>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
        <div className="mb-5 rounded-lg border border-border/80 bg-muted/25 p-3">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0 text-xs font-semibold text-foreground">
              {segmentLocation(t, inspectorSegment)}
            </div>
            <Badge
              variant="outline"
              className={cn(
                'shrink-0 font-normal',
                inspectorSegment.mqm_severity === 'critical' &&
                  'border-rose-600/25 bg-rose-500/10 text-rose-700 dark:text-rose-400'
              )}
            >
              {mqmSeverityLabel(t, inspectorSegment.mqm_severity)}
            </Badge>
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
            <span>{blockTypeLabel(t, inspectorSegment.block_type)}</span>
            {inspectorSegment.page !== null && (
              <>
                <span>·</span>
                <span>{t.review.pageLabel.replace('{n}', String(inspectorSegment.page))}</span>
              </>
            )}
          </div>
        </div>
        <div className="mb-5">
          <div className="mb-2 text-xs font-medium text-muted-foreground">
            {t.review.findingsTitle}
          </div>
          {inspectorSegment.error_flags.length > 0 || inspectorSegment.issues.length > 0 ? (
            <div className="space-y-1.5">
              {(inspectorSegment.issues.length > 0
                ? inspectorSegment.issues
                : inspectorSegment.error_flags.map(findingKindForFlag)
              ).map((kind) => (
                <div key={kind} className="flex items-start gap-2.5 rounded-md bg-muted/40 px-3 py-2.5">
                  <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-amber-600" />
                  <div className="min-w-0">
                    <div className="text-xs font-medium text-foreground">
                      {issueLabel(t, kind)}
                    </div>
                    <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">
                      {findingLabel(t, kind)}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs leading-relaxed text-muted-foreground">
              {t.review.noFindings}
            </p>
          )}
        </div>
        <dl className="mb-5 grid grid-cols-2 gap-2">
          {[
            [t.review.inspectorQe, inspectorSegment.mtqe_score === null ? '—' : `${Math.round(inspectorSegment.mtqe_score * 100)}%`],
            [t.review.inspectorRepairs, String(inspectorSegment.repair_rounds)],
            [t.review.inspectorTmHit, inspectorSegment.tm_hit ? t.common.yes : '—'],
          ].map(([label, value], index) => (
            <div
              key={label}
              className={cn(
                'rounded-lg border border-border/70 bg-card px-3 py-2.5',
                index === 0 && 'col-span-2'
              )}
            >
              <dt className="text-[11px] text-muted-foreground">{label}</dt>
              <dd className="mt-1 font-mono text-sm font-medium text-foreground">{value}</dd>
            </div>
          ))}
        </dl>
        {inspectorSegment.glossary_hits.length > 0 && (
          <div className="mb-4">
            <div className="mb-2 text-xs font-medium text-muted-foreground">
              {t.review.inspectorGlossary}
            </div>
            <div className="flex flex-wrap gap-1.5">
              {inspectorSegment.glossary_hits.map((term) => (
                <Badge key={term} variant="outline" className="font-mono text-xs">
                  {term}
                </Badge>
              ))}
            </div>
          </div>
        )}
          <Accordion type="single" collapsible className="rounded-lg border border-border/80 px-3">
          <AccordionItem value="raw" className="border-b-0">
            <AccordionTrigger className="py-3 text-xs hover:no-underline">
              {t.review.rawEvidence}
            </AccordionTrigger>
            <AccordionContent className="space-y-3 pb-3">
              <p className="text-xs leading-relaxed text-muted-foreground">
                {t.review.rawEvidenceHint}
              </p>
              {inspectorSegment.error_flags.length > 0 && (
                <div className="space-y-1.5">
                  <div className="text-xs font-medium text-foreground">{t.review.errorFlags}</div>
                  {inspectorSegment.error_flags.map((flag) => (
                    <div key={flag} className="break-all rounded-md bg-muted/50 p-2 font-mono text-xs text-muted-foreground">
                      {flag}
                    </div>
                  ))}
                </div>
              )}
              {Object.keys(inspectorSegment.provenance).length > 0 && (
                <div>
                  <div className="mb-1.5 text-xs font-medium text-foreground">
                    {t.review.inspectorProvenance}
                  </div>
                  <pre className="max-h-52 overflow-auto whitespace-pre-wrap break-all rounded-md bg-muted/50 p-2 font-mono text-xs text-muted-foreground">
                    {JSON.stringify(inspectorSegment.provenance, null, 2)}
                  </pre>
                </div>
              )}
              <div>
                <div className="mb-1.5 text-xs font-medium text-foreground">
                  {t.review.blockTechnicalId}
                </div>
                <button
                  type="button"
                  className="flex w-full items-center justify-between gap-3 rounded-md bg-muted/50 px-2.5 py-2 text-left font-mono text-xs text-muted-foreground transition-colors hover:text-foreground"
                  title={t.mission.copy}
                  onClick={() => {
                    void navigator.clipboard?.writeText(inspectorSegment.block_id)
                  }}
                >
                  <span className="break-all">{inspectorSegment.block_id}</span>
                  <span className="shrink-0 text-[11px]">{t.mission.copy}</span>
                </button>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </div>
    </div>
  ) : null

  return (
    <div className="flex-1 flex min-h-0 overflow-hidden bg-background text-foreground">
      <div className="flex-1 flex flex-col min-h-0 overflow-hidden">
      {/* Precision Top Toolbar: Single-line fixed height 56px (h-14) */}
      <div className="flex min-h-14 shrink-0 flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b border-border/80 bg-card px-4 py-2 shadow-2xs md:px-5">
        {/* Left: Workbench Title + Count + Issue Navigation */}
        <div className="flex min-w-0 flex-[1_1_18rem] flex-wrap items-center gap-3">
          {jobId && (
            <Button
              variant="ghost"
              size="icon"
              onClick={() => navigate(`/jobs/${jobId}`)}
              className="-ml-2 size-8 shrink-0"
              title={t.quality.backToJob}
              aria-label={t.quality.backToJob}
            >
              <ArrowLeft />
            </Button>
          )}
          <div className="flex min-w-0 items-center gap-2 whitespace-nowrap text-sm font-semibold tracking-[-0.01em] text-foreground">
            <SplitSquareVertical className="size-4 text-primary" />
            <span>{t.review.title}</span>
          </div>
          <Badge variant="secondary" className="font-mono text-xs font-normal shrink-0 whitespace-nowrap">
            {t.review.segmentCount.replace('{count}', String(segmentTotal))}
          </Badge>
          {selectedIndex >= 0 && (
            <span className="text-xs tabular-nums text-muted-foreground">
              {t.review.segmentPosition
                .replace('{current}', String(selectedIndex + 1))
                .replace('{total}', String(segments.length))}
            </span>
          )}
          <Button
            variant="ghost"
            size="sm"
            className="hidden h-7 px-2 text-xs lg:inline-flex"
            onClick={() => setDesktopEvidenceOpen((open) => !open)}
          >
            {desktopEvidenceOpen ? <EyeOff /> : <Info />}
            {desktopEvidenceOpen ? t.review.hideEvidence : t.review.showEvidence}
          </Button>
          <div className="h-4 w-px bg-border hidden sm:block" />
          <div className="hidden items-center rounded-md border border-input bg-background/50 shadow-2xs sm:flex">
            <button
              type="button"
              onClick={() => {
                const next = moveSegment(-1)
                requestAnimationFrame(() => scrollToSelected(next))
              }}
              title={t.review.prevIssue}
              className="p-1.5 hover:bg-muted text-muted-foreground hover:text-foreground rounded-l-md transition-colors cursor-pointer"
            >
              <ChevronUp className="h-3.5 w-3.5" />
            </button>
            <div className="h-3 w-px bg-border" />
            <button
              type="button"
              onClick={() => {
                const next = moveSegment(1)
                requestAnimationFrame(() => scrollToSelected(next))
              }}
              title={t.review.nextIssue}
              className="p-1.5 hover:bg-muted text-muted-foreground hover:text-foreground rounded-r-md transition-colors cursor-pointer"
            >
              <ChevronDown className="h-3.5 w-3.5" />
            </button>
          </div>
        </div>

        {/* Only two independent choices belong in persistent chrome:
            the review scope and the reading view. Everything else is a
            contextual action inside the selected segment. */}
        <div className="flex min-w-0 flex-[2_1_34rem] flex-wrap items-center justify-start gap-2 sm:justify-end">
          <p className="text-xs text-muted-foreground">{t.review.scopeLabel}</p>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="outline" size="sm" className="h-8 gap-1.5 px-2.5 text-xs">
                <ListFilter />
                {scopeLabel(t, scope)}
                <ChevronDown className="opacity-60" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-48">
              <DropdownMenuLabel className="text-xs font-medium text-muted-foreground">
                {t.review.scopeLabel}
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuRadioGroup
                value={scope}
                onValueChange={(value) => setFilter(value === 'all' ? 'all' : 'issues')}
              >
                <DropdownMenuRadioItem value="issues" className="text-xs">
                  {t.review.scopeToReview}
                </DropdownMenuRadioItem>
                <DropdownMenuRadioItem value="all" className="text-xs">
                  {t.review.scopeEverything}
                </DropdownMenuRadioItem>
              </DropdownMenuRadioGroup>
            </DropdownMenuContent>
          </DropdownMenu>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                className="hidden h-8 gap-1.5 px-2.5 text-xs sm:inline-flex"
              >
                {viewMode === 'segments' ? t.review.modeSegments : t.review.modeFlow}
                <ChevronDown className="opacity-60" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuLabel className="text-xs font-medium text-muted-foreground">
                {t.review.viewLabel}
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuRadioGroup
                value={viewMode}
                onValueChange={(value) => setViewMode(value as typeof viewMode)}
              >
                <DropdownMenuRadioItem value="segments" className="text-xs">
                  <span>{t.review.modeSegments}</span>
                  <span className="block text-[11px] font-normal text-muted-foreground">
                    {t.review.modeSegmentsDesc}
                  </span>
                </DropdownMenuRadioItem>
                <DropdownMenuRadioItem value="reading_flow" className="text-xs">
                  <span>{t.review.modeFlow}</span>
                  <span className="block text-[11px] font-normal text-muted-foreground">
                    {t.review.modeFlowDesc}
                  </span>
                </DropdownMenuRadioItem>
              </DropdownMenuRadioGroup>
            </DropdownMenuContent>
          </DropdownMenu>

          {/* Re-export indicator button if needed */}
          {reExportPending && (
            <Button
              variant="outline"
              size="sm"
              onClick={() => void handleReExport()}
              disabled={reExporting}
              className="h-8 text-xs border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-400 hover:bg-amber-500/20"
              title={t.review.reExportHelp}
            >
              {reExporting ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin mr-1" />
              ) : (
                <RefreshCw className="h-3.5 w-3.5 mr-1" />
              )}
              {reExporting ? t.review.reExporting : t.review.reExportStale}
            </Button>
          )}
        </div>
      </div>

      {/* Issue Sub-Toolbar: Filter pills that never wrap or break the top header */}
      <div className="h-10 px-4 md:px-6 border-b border-border bg-muted/20 flex items-center gap-2 overflow-x-auto shrink-0 text-xs">
        <div className="flex items-center gap-1.5 text-muted-foreground shrink-0 font-medium">
          <Filter className="h-3.5 w-3.5" />
          <span>{t.review.filterByIssueKind}:</span>
        </div>

        {ribbonCounts.length === 0 ? (
          <span className="text-emerald-600 dark:text-emerald-400 flex items-center gap-1 font-medium shrink-0">
            <CheckCircle2 className="h-3.5 w-3.5" /> {t.review.noIssues}
          </span>
        ) : (
          ribbonCounts.map(([kind, value]) => {
            const isActive = filter === kind
            return (
              <button
                key={kind}
                type="button"
                onClick={() => setFilter(isActive ? 'issues' : kind)}
                className={cn(
                  "h-6 px-2.5 rounded-full border transition-all text-xs flex items-center gap-1.5 shrink-0 cursor-pointer",
                  isActive
                    ? "bg-primary text-primary-foreground border-primary shadow-xs font-medium"
                    : "bg-background border-border text-muted-foreground hover:text-foreground hover:bg-accent"
                )}
                title={isActive ? t.review.clearFilter : `${t.review.filterByIssueKind}: ${issueLabel(t, kind)}`}
              >
                <span>{issueLabel(t, kind)}</span>
                <span className={cn(
                  "text-xs font-mono px-1 rounded-full",
                  isActive ? "bg-primary-foreground/20 text-primary-foreground font-bold" : "bg-muted text-foreground"
                )}>
                  {value}
                </span>
                {isActive && <X className="h-3 w-3 ml-0.5 opacity-80" />}
              </button>
            )
          })
        )}

        {filter !== 'issues' && filter !== 'needs_human' && filter !== 'all' && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setFilter('issues')}
            className="h-6 px-2 text-xs text-muted-foreground hover:text-foreground shrink-0"
          >
            <X className="h-3 w-3 mr-1" />
            {t.review.clearFilter}
          </Button>
        )}
      </div>
      <div className="flex shrink-0 items-center gap-2 border-b border-border/60 bg-card px-4 py-2 text-xs md:hidden">
        <ListFilter className="size-3.5 text-muted-foreground" />
        <span className="text-muted-foreground">{t.review.issueQueueHint}</span>
        {inspectorSegment && (
          <Button
            variant="ghost"
            size="sm"
            className="ml-auto h-7 px-2 text-xs"
            onClick={() => setInspectorOpen(true)}
          >
            {t.review.openEvidence}
          </Button>
        )}
      </div>

      {reExportNotice && (
        <div className="px-6 py-2 text-xs text-muted-foreground bg-muted/40 border-b border-border">
          {reExportNotice}
        </div>
      )}

      {error && (
        <div className="px-6 py-2 text-xs text-muted-foreground bg-amber-500/10 border-b border-border flex items-center gap-2">
          <AlertTriangle className="h-3.5 w-3.5 text-amber-700 dark:text-amber-400 shrink-0" />
          {error}
        </div>
      )}

      {viewMode === 'segments' ? (
        loading ? (
          <div className="flex-1 flex items-center justify-center text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin" />
          </div>
        ) : segments.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center gap-3 p-8 text-xs text-muted-foreground">
            <p>{t.review.noSegments}</p>
            {filter !== 'all' && (
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setFilter('all')}
              >
                {t.review.filterAll}
              </Button>
            )}
          </div>
        ) : (
          <div ref={scrollRef} className="flex-1 overflow-y-auto">
            <div className="mx-auto w-full max-w-[1500px] px-4 py-5 md:px-6">
              {segments.map((seg, index) => {
                const draft = drafts[seg.block_id] ?? seg.delivered_text
                const dirty = draft !== seg.delivered_text
                const isSelected = selectedBlockId === seg.block_id
                const isEditing = editing[seg.block_id] === true
                return (
                  <article
                    key={seg.block_id}
                    id={`review-segment-${index}`}
                    className={cn(
                      'relative mb-4 rounded-xl border bg-card px-4 py-4 transition-colors sm:px-5',
                      isSelected
                        ? 'border-primary/35 bg-primary/[0.025] shadow-sm ring-1 ring-primary/15'
                        : 'border-border/70 hover:border-border'
                    )}
                    onClick={() => selectAndReveal(index)}
                  >
                    <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                      <div className="flex min-w-0 flex-wrap items-center gap-2">
                        <span className="text-xs font-medium text-foreground">
                          {segmentLocation(t, seg)}
                        </span>
                      </div>
                      <div className="flex flex-wrap items-center gap-1.5">
                        {seg.page !== null && (
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-7 gap-1.5 px-2 text-xs text-muted-foreground"
                            title={t.review.layoutPreviewHint}
                            onClick={(event) => {
                              event.stopPropagation()
                              openLayoutPreview(seg.page)
                            }}
                          >
                            <Eye />
                            {t.review.layoutPreview}
                          </Button>
                        )}
                        {seg.issues.slice(0, 2).map((kind) => (
                          <Badge key={kind} variant="outline" className="font-normal">
                            {issueLabel(t, kind)}
                          </Badge>
                        ))}
                        {seg.issues.length > 2 && (
                          <span className="text-xs text-muted-foreground">+{seg.issues.length - 2}</span>
                        )}
                        {seg.human_verified ? (
                          <Badge variant="outline" className="border-emerald-600/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400">
                            {t.review.humanVerified}
                          </Badge>
                        ) : (
                          <Badge variant="outline" className="font-normal">
                            {blockStatusLabel(t, seg.status)}
                          </Badge>
                        )}
                      </div>
                    </div>

                    {!isMobile && (
                      <ResizablePanelGroup
                        orientation="horizontal"
                        className="min-h-40"
                      >
                        <ResizablePanel defaultSize="50%" minSize="30%">
                          <div className="h-full bg-muted/20 p-4 sm:p-5">
                            <div className="mb-2 text-xs font-medium text-muted-foreground">
                              {t.review.sourceSegment}
                            </div>
                            <div className="font-serif text-[15px] leading-7 text-foreground">
                              <React.Suspense fallback={<RichTextFallback />}>
                                <RichText text={seg.source_text} source />
                              </React.Suspense>
                            </div>
                          </div>
                        </ResizablePanel>
                        <ResizableHandle withHandle />
                        <ResizablePanel defaultSize="50%" minSize="30%">
                        <div className="flex h-full min-w-0 flex-col p-4 sm:p-5">
                          <div className="mb-2 flex items-center justify-between gap-2">
                            <span className="text-xs font-medium text-muted-foreground">
                              {t.review.targetSegment}
                            </span>
                            {dirty ? (
                              <Badge variant="outline" className="border-amber-600/25 bg-amber-500/10 text-amber-700 dark:text-amber-400">
                                {t.review.unsavedDraft}
                              </Badge>
                            ) : saved[seg.block_id] ? (
                              <span className="text-xs text-emerald-700 dark:text-emerald-400">
                                {t.review.saved}
                              </span>
                            ) : null}
                          </div>
                          {isEditing ? (
                            <div className="flex flex-1 flex-col gap-3" onClick={(event) => event.stopPropagation()}>
                              <AutoGrowTextarea
                                autoFocus
                                value={draft}
                                onChange={(event) =>
                                  setDrafts((prev) => ({ ...prev, [seg.block_id]: event.target.value }))
                                }
                                onKeyDown={(event) => {
                                  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
                                    event.preventDefault()
                                    void saveAndContinue(seg)
                                  } else if (event.key === 'Escape') {
                                    event.preventDefault()
                                    setDrafts((prev) => {
                                      const next = { ...prev }
                                      delete next[seg.block_id]
                                      return next
                                    })
                                    setEditing((prev) => ({ ...prev, [seg.block_id]: false }))
                                  }
                                }}
                                className="w-full resize-none rounded-md border border-input bg-background p-3 font-sans text-[15px] leading-7 text-foreground shadow-xs focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                              />
                              <div className="flex flex-wrap items-center justify-end gap-2">
                                <span className="mr-auto text-xs text-muted-foreground">
                                  {t.review.editShortcutHint}
                                </span>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => {
                                    setDrafts((prev) => {
                                      const next = { ...prev }
                                      delete next[seg.block_id]
                                      return next
                                    })
                                    setEditing((prev) => ({ ...prev, [seg.block_id]: false }))
                                  }}
                                >
                                  {t.review.cancelEdit}
                                </Button>
                                <Button
                                  size="sm"
                                  disabled={saving[seg.block_id] || !dirty}
                                  onClick={() => void saveAndContinue(seg)}
                                >
                                  {saving[seg.block_id] ? (
                                    <Loader2 className="animate-spin" />
                                  ) : (
                                    <Save />
                                  )}
                                  {t.review.saveAndNext}
                                </Button>
                              </div>
                            </div>
                          ) : (
                            <div className="flex flex-1 flex-col">
                              <div
                                role="button"
                                tabIndex={0}
                                title={t.review.editHint}
                                onClick={(event) => event.stopPropagation()}
                                onDoubleClick={() =>
                                  setEditing((prev) => ({ ...prev, [seg.block_id]: true }))
                                }
                                onKeyDown={(event) => {
                                  if (event.key === 'Enter') {
                                    event.preventDefault()
                                    setEditing((prev) => ({ ...prev, [seg.block_id]: true }))
                                  }
                                }}
                                className="flex-1 cursor-text rounded-md font-serif text-[15px] leading-7 text-foreground"
                              >
                                <React.Suspense fallback={<RichTextFallback />}>
                                  <RichText text={seg.delivered_text} />
                                </React.Suspense>
                              </div>
                              {seg.repair_failed && (
                                <p className="mt-3 rounded-md bg-amber-500/10 px-2.5 py-1.5 text-xs text-amber-700 dark:text-amber-400">
                                  {t.review.failedDraftNotice}: {seg.repair_error}
                                </p>
                              )}
                              {seg.quarantined && (
                                <p className="mt-3 rounded-md bg-muted px-2.5 py-1.5 text-xs text-muted-foreground">
                                  {t.review.quarantinedNotice}
                                </p>
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
                              <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-border/70 pt-3">
                                <span className="text-xs text-muted-foreground">
                                  {termNotice[seg.block_id] ?? t.review.editHint}
                                </span>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={(event) => {
                                    event.stopPropagation()
                                    setEditing((prev) => ({ ...prev, [seg.block_id]: true }))
                                  }}
                                >
                                  <Edit3 />
                                  {t.review.startEditing}
                                </Button>
                              </div>
                            </div>
                          )}
                        </div>
                        </ResizablePanel>
                      </ResizablePanelGroup>
                    )}
                    {isMobile && (
                    <div className="space-y-4">
                      <div className="rounded-lg bg-muted/20 p-4">
                        <div className="mb-2 text-xs font-medium text-muted-foreground">
                          {t.review.sourceSegment}
                        </div>
                        <div className="font-serif text-[15px] leading-7 text-foreground">
                          <React.Suspense fallback={<RichTextFallback />}>
                            <RichText text={seg.source_text} source />
                          </React.Suspense>
                        </div>
                      </div>
                      <div className="rounded-lg border border-border/70 p-4">
                        <div className="mb-2 flex items-center justify-between gap-2">
                          <span className="text-xs font-medium text-muted-foreground">
                            {t.review.targetSegment}
                          </span>
                          {dirty && (
                            <Badge variant="outline" className="border-amber-600/25 bg-amber-500/10 text-amber-700 dark:text-amber-400">
                              {t.review.unsavedShort}
                            </Badge>
                          )}
                        </div>
                        {isEditing ? (
                          <div className="space-y-2">
                            <AutoGrowTextarea
                              autoFocus
                              value={draft}
                              onChange={(event) =>
                                setDrafts((prev) => ({ ...prev, [seg.block_id]: event.target.value }))
                              }
                              onKeyDown={(event) => {
                                if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
                                  event.preventDefault()
                                  void saveAndContinue(seg)
                                }
                              }}
                              className="w-full resize-none rounded-md border border-input bg-background p-3 text-[15px] leading-7"
                            />
                            <div className="flex justify-end gap-2">
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => setEditing((prev) => ({ ...prev, [seg.block_id]: false }))}
                              >
                                {t.review.cancelEdit}
                              </Button>
                              <Button size="sm" disabled={saving[seg.block_id] || !dirty} onClick={() => void saveAndContinue(seg)}>
                                <Save />
                                {t.review.saveAndNext}
                              </Button>
                            </div>
                          </div>
                        ) : (
                          <>
                            <div
                              role="button"
                              tabIndex={0}
                              onClick={(event) => event.stopPropagation()}
                              onDoubleClick={() => setEditing((prev) => ({ ...prev, [seg.block_id]: true }))}
                              onKeyDown={(event) => {
                                if (event.key === 'Enter') {
                                  event.preventDefault()
                                  setEditing((prev) => ({ ...prev, [seg.block_id]: true }))
                                }
                              }}
                              className="cursor-text font-serif text-[15px] leading-7 text-foreground"
                            >
                              <React.Suspense fallback={<RichTextFallback />}>
                                <RichText text={seg.delivered_text} />
                              </React.Suspense>
                            </div>
                            <div className="mt-3 flex justify-end border-t border-border/70 pt-3">
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={(event) => {
                                  event.stopPropagation()
                                  setEditing((prev) => ({ ...prev, [seg.block_id]: true }))
                                }}
                              >
                                <Edit3 />
                                {t.review.startEditing}
                              </Button>
                            </div>
                          </>
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
                    </div>
                    )}

                  </article>
                )
              })}
              {hasMore && (
                <div className="flex justify-center py-6">
                  <Button
                    onClick={() => void loadMore()}
                    variant="secondary"
                    size="sm"
                    disabled={loadingMore}
                  >
                    {loadingMore ? <Loader2 className="animate-spin" /> : <ChevronDown />}
                    {t.review.loadMore} ({segmentTotal - segments.length})
                  </Button>
                </div>
              )}
            </div>
          </div>
        )
      ) : viewMode === 'reading_flow' ? (
        /* Reading Flow View: continuous natural layout with dual-column typography */
        <div ref={flowScrollRef} className="flex-1 overflow-y-auto px-4 md:px-10 py-6 bg-background">
          <div
            className="relative w-full max-w-5xl mx-auto"
            style={{ height: flowVirtualizer.getTotalSize() }}
          >
            {flowVirtualizer.getVirtualItems().map((virtualRow) => {
              const seg = segments[virtualRow.index]
              const draft = drafts[seg.block_id] ?? seg.delivered_text ?? ''
              const hasIssues = seg.issues.length > 0 || seg.quarantined || seg.repair_failed
              const isSelected = selectedBlockId === seg.block_id
              const isEditingThis = editing[seg.block_id]
              return (
                <div
                  key={seg.block_id}
                  data-index={virtualRow.index}
                  ref={flowVirtualizer.measureElement}
                  className="absolute top-0 left-0 w-full pb-4"
                  style={{ transform: `translateY(${virtualRow.start}px)` }}
                >
                    <div
                      onClick={() => {
                        setSelectedBlockId(seg.block_id)
                        setInspectorOpen(true)
                      }}
                      className={cn(
                        "reading-segment grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6 p-5 rounded-xl transition-all cursor-pointer border shadow-xs",
                        isSelected
                          ? "bg-card border-primary ring-2 ring-primary/20 shadow-sm"
                          : hasIssues
                            ? "bg-amber-500/[0.045] hover:bg-amber-500/[0.075] border-amber-500/45"
                            : "border-border hover:border-border/80 hover:bg-card/80 bg-card"
                      )}
                    >
                      {/* Left: Source text */}
                      <div className="min-w-0 break-words [overflow-wrap:anywhere] text-sm font-sans leading-relaxed text-muted-foreground select-text md:pr-4 max-w-[65ch] [text-wrap:pretty]">
                        <div className="mb-1 flex items-center justify-between text-xs text-muted-foreground">
                          <span>{segmentLocation(t, seg)}</span>
                          {seg.page !== null && (
                            <button
                              type="button"
                              className="rounded px-1.5 py-0.5 transition-colors hover:bg-muted hover:text-foreground"
                              title={t.review.layoutPreviewHint}
                              onClick={(event) => {
                                event.stopPropagation()
                                openLayoutPreview(seg.page)
                              }}
                            >
                              {t.review.layoutPreview}
                            </button>
                          )}
                        </div>
                        <div className="whitespace-pre-wrap">{seg.source_text}</div>
                      </div>

                      {/* Right: Target translation */}
                      <div className="min-w-0 break-words [overflow-wrap:anywhere] text-sm font-sans leading-relaxed text-foreground md:pl-2 max-w-[65ch] [text-wrap:pretty]">
                        {isEditingThis ? (
                          <div className="space-y-2" onClick={(e) => e.stopPropagation()}>
                            <AutoGrowTextarea
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
                                  setDrafts((prev) => ({ ...prev, [seg.block_id]: seg.target_text }))
                                  setEditing((prev) => ({ ...prev, [seg.block_id]: false }))
                                }
                              }}
                              className="w-full text-sm font-sans bg-background border border-input p-3 rounded-md focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring resize-none leading-relaxed text-foreground shadow-xs"
                            />
                            <div className="flex items-center justify-between text-xs">
                              <span className="text-muted-foreground text-xs">{t.review.editShortcutHint}</span>
                              <Button
                                size="sm"
                                className="text-xs h-7 px-2.5 font-medium"
                                onClick={() => {
                                  void handleSave(seg)
                                  setEditing((prev) => ({ ...prev, [seg.block_id]: false }))
                                }}
                                disabled={saving[seg.block_id]}
                              >
                                {saving[seg.block_id] ? <Loader2 className="h-3 w-3 animate-spin mr-1" /> : <Save className="h-3 w-3 mr-1" />}
                                {t.common.confirm}
                              </Button>
                            </div>
                          </div>
                        ) : (
                          <div
                            role="button"
                            tabIndex={0}
                            onDoubleClick={(e) => {
                              e.stopPropagation()
                              setEditing((prev) => ({ ...prev, [seg.block_id]: true }))
                            }}
                            onKeyDown={(e) => {
                              if (e.key === 'Enter' || e.key === ' ') {
                                e.preventDefault()
                                e.stopPropagation()
                                setEditing((prev) => ({ ...prev, [seg.block_id]: true }))
                              }
                            }}
                            className="group relative cursor-pointer focus:outline-hidden focus:ring-1 focus:ring-primary rounded p-1 -m-1 hover:bg-muted/20 transition-colors"
                          >
                            <div className="select-text font-serif text-[15px] leading-7 text-foreground">
                              <React.Suspense fallback={<RichTextFallback />}>
                                <RichText text={draft} />
                              </React.Suspense>
                            </div>
                            <div className="mt-2 flex items-center gap-1.5 flex-wrap">
                              {seg.mtqe_score !== null && (() => {
                                const pct = Math.round(seg.mtqe_score * 100)
                                const colorClass = pct >= 85
                                  ? 'text-emerald-700 dark:text-emerald-400 bg-emerald-500/10 border-emerald-500/20'
                                  : pct >= 70
                                    ? 'text-sky-700 dark:text-sky-400 bg-sky-500/10 border-sky-500/20'
                                    : pct >= 55
                                      ? 'text-amber-700 dark:text-amber-400 bg-amber-500/10 border-amber-500/20'
                                      : 'text-rose-700 dark:text-rose-400 bg-rose-500/10 border-rose-500/20'
                                return (
                                  <span className={`text-xs font-mono font-bold px-1.5 py-0.5 rounded border ${colorClass}`}>
                                    QE {pct}%
                                  </span>
                                )
                              })()}
                              {hasIssues &&
                                seg.issues.map((kind) => (
                                  <span
                                    key={kind}
                                    className="text-xs px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-700 dark:text-amber-400 font-mono font-medium"
                                  >
                                    {issueLabel(t, kind)}
                                  </span>
                                ))}
                            </div>
                          </div>
                        )}
                      </div>
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
      ) : null}
      </div>
      <Sheet
        open={previewPage !== null}
        onOpenChange={(open) => {
          if (!open) setPreviewPage(null)
        }}
      >
        <SheetContent
          side="right"
          className="flex w-[min(96vw,72rem)] flex-col gap-0 p-0"
        >
          <SheetHeader className="shrink-0 border-b border-border/70 px-5 py-4 pr-12 text-left">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <SheetTitle className="text-base">
                  {t.review.previewPage}
                  {previewPage !== null && (
                    <span className="ml-2 font-normal text-muted-foreground">
                      {t.review.pageLabel.replace('{n}', String(previewPage))}
                    </span>
                  )}
                </SheetTitle>
                <SheetDescription className="mt-1 text-xs">
                  {t.review.layoutPreviewHint}
                </SheetDescription>
              </div>
              {previewPage !== null && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPreviewBust(Date.now())}
                  title={t.review.rerenderHint}
                >
                  <RefreshCw />
                  {t.review.rerenderPage}
                </Button>
              )}
            </div>
          </SheetHeader>
          <div className="flex shrink-0 items-center justify-between gap-3 border-b border-border/60 bg-muted/20 px-5 py-2.5">
            <p className="text-xs text-muted-foreground">{t.review.diffBlendHint}</p>
            <label className="flex shrink-0 items-center gap-2 text-xs text-foreground">
              {t.review.diffBlend}
              <Switch checked={diffBlend} onCheckedChange={setDiffBlend} />
            </label>
          </div>
          <div className="min-h-0 flex-1 overflow-auto bg-muted/20 p-4 md:p-6">
            {previewPage !== null &&
              (diffBlend ? (
                <div className="relative mx-auto max-w-5xl overflow-hidden rounded-lg border bg-white">
                  <img
                    src={sourcePageUrl(jobId, previewPage, { dpi: 110 })}
                    alt={t.review.sourceCanvasAlt.replace('{n}', String(previewPage))}
                    className="w-full object-contain"
                  />
                  <img
                    src={pagePreviewUrl(jobId, previewPage, { dpi: 110, cacheBust: previewBust })}
                    alt={t.review.targetCanvasAlt.replace('{n}', String(previewPage))}
                    className="absolute inset-0 h-full w-full object-contain opacity-50 mix-blend-multiply"
                  />
                </div>
              ) : (
                <div className="mx-auto grid max-w-6xl gap-4 md:grid-cols-2">
                  {[
                    {
                      title: t.review.sourceCanvas,
                      src: sourcePageUrl(jobId, previewPage, { dpi: 110 }),
                      alt: t.review.sourceCanvasAlt.replace('{n}', String(previewPage)),
                    },
                    {
                      title: t.review.targetCanvas,
                      src: pagePreviewUrl(jobId, previewPage, {
                        dpi: 110,
                        cacheBust: previewBust,
                      }),
                      alt: t.review.targetCanvasAlt.replace('{n}', String(previewPage)),
                    },
                  ].map((page) => (
                    <figure
                      key={page.title}
                      className="overflow-hidden rounded-lg border bg-card"
                    >
                      <figcaption className="border-b border-border/70 px-3 py-2 text-xs font-medium text-muted-foreground">
                        {page.title}
                      </figcaption>
                      <img
                        src={page.src}
                        alt={page.alt}
                        className="w-full bg-white object-contain"
                        onError={(event) => {
                          event.currentTarget.style.display = 'none'
                          const fallback = event.currentTarget.nextElementSibling
                          if (fallback instanceof HTMLElement) fallback.hidden = false
                        }}
                      />
                      <div
                        hidden
                        className="flex items-center gap-2 p-4 text-xs text-muted-foreground"
                      >
                        <ImageOff className="size-4 shrink-0" />
                        {t.review.previewFailed}
                      </div>
                    </figure>
                  ))}
                </div>
              ))}
          </div>
        </SheetContent>
      </Sheet>

      {isMobile ? (
        <Sheet open={inspectorOpen && inspectorSegment !== null} onOpenChange={setInspectorOpen}>
          <SheetContent side="right" className="w-[min(92vw,24rem)] p-0">
            <SheetHeader className="sr-only">
              <SheetTitle>{t.review.evidenceTitle}</SheetTitle>
              <SheetDescription>{t.review.evidenceHint}</SheetDescription>
            </SheetHeader>
            {evidencePanel}
          </SheetContent>
        </Sheet>
      ) : (
        inspectorSegment && desktopEvidenceOpen && (
          <aside className="hidden h-full w-80 shrink-0 border-l border-border/70 bg-card md:block">
            {evidencePanel}
          </aside>
        )
      )}

    </div>
  )
}

/**
 * A textarea that is exactly as tall as the text it holds.
 *
 * The review pane sized its editor with `rows={3}`. Measured on the reference
 * job, the translation of `pdf_main#b0005` occupies 351px beside 718 characters
 * of source, but opening the editor replaced it with a 58.5px box showing three
 * lines — so reviewing a paragraph meant scrolling inside a sliver that showed
 * a sixth of the text it was editing. The read view and the edit view of the
 * same block must be the same size, or the pane appears to shrink the moment
 * the operator starts working.
 *
 * Height is re-measured whenever the value changes (including the mount commit,
 * where the node and the text arrive together). `height` is reset to `auto`
 * first: without it a textarea that grew once can never shrink back after the
 * operator deletes a line.
 */
function AutoGrowTextarea({
  className,
  value,
  ...props
}: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  const localRef = useRef<HTMLTextAreaElement | null>(null)

  const fit = useCallback(() => {
    const el = localRef.current
    if (!el) return
    el.style.height = 'auto'
    // +2px: the border box, so a fractional last line cannot clip its descenders.
    el.style.height = `${el.scrollHeight + 2}px`
  }, [])

  useEffect(fit, [fit, value])

  // A width change re-wraps the text, so the height measured for the old width
  // no longer covers it: opening the inspector narrows the target column from
  // 446px to 286px at 1280px, and the same text needs 566px instead of 351px.
  // Without this the textarea keeps the pre-resize height and hides its tail.
  //
  // Only a *width* change re-fits. This element is its own observer target and
  // fitting changes its height, so reacting to height would re-enter on every
  // adjustment and spin (the browser reports it as an undelivered-notification
  // loop). Width is driven by the layout, not by this effect, so it cannot.
  useEffect(() => {
    const el = localRef.current
    if (!el || typeof ResizeObserver === 'undefined') return
    let lastWidth = el.clientWidth
    const observer = new ResizeObserver(() => {
      if (el.clientWidth === lastWidth) return
      lastWidth = el.clientWidth
      fit()
    })
    observer.observe(el)
    return () => observer.disconnect()
  }, [fit])

  return <textarea {...props} ref={localRef} value={value} className={className} />
}

type TermScope = 'block' | 'subsequent' | 'all'

interface TermPanelProps {
  jobId: string
  segment: Segment
  onReplaced: (replacements: number) => void
}

/**
 * Terminology recommendations for one segment (PRD §5.2.2). Fetches lazily on
 * mount so only the virtualizer's visible window hits the backend, and provides
 * a 3-way Scope Selector ('block' | 'subsequent' | 'all') with diff preview and
 * TM feedback indicators.
 */
function TermPanel({ jobId, segment, onReplaced }: TermPanelProps) {
  const { t } = useI18n()
  const [violations, setViolations] = useState<TermViolation[] | null>(null)
  const [scopes, setScopes] = useState<Record<string, TermScope>>({})
  const [confirmViolation, setConfirmViolation] = useState<{
    violation: TermViolation
    scope: TermScope
  } | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const reloadTerms = useCallback(() => {
    getBlockTerms(jobId, segment.block_id)
      .then((report) => setViolations(report.violations))
      .catch(() => setViolations([]))
  }, [jobId, segment.block_id])

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
  }, [jobId, segment.block_id, segment.delivered_text])

  const getScope = (v: TermViolation): TermScope => {
    const key = `${v.surface}:${v.expected}`
    if (scopes[key]) return scopes[key]
    if (v.cascade_subsequent > 1) return 'subsequent'
    if (v.cascade_all > 1) return 'all'
    return 'block'
  }

  const setScopeFor = (v: TermViolation, s: TermScope) => {
    const key = `${v.surface}:${v.expected}`
    setScopes((prev) => ({ ...prev, [key]: s }))
    if (confirmViolation?.violation === v) {
      setConfirmViolation({ violation: v, scope: s })
    }
  }

  const handleTriggerReplace = (violation: TermViolation) => {
    const currentScope = getScope(violation)
    if (currentScope === 'block') {
      void executeReplace(violation, 'block')
    } else {
      setConfirmViolation({ violation, scope: currentScope })
    }
  }

  const executeReplace = async (violation: TermViolation, scope: TermScope) => {
    setBusy(true)
    setError(null)
    setConfirmViolation(null)
    try {
      const res = await propagateTerm(jobId, {
        block_id: segment.block_id,
        surface: violation.surface,
        expected: violation.expected,
        scope,
      })
      onReplaced(res.replacements)
      reloadTerms()
    } catch (err) {
      setError(err instanceof Error ? err.message : t.review.propagateFailed)
    } finally {
      setBusy(false)
    }
  }

  if (violations === null || violations.length === 0) return null

  return (
    <div className="mt-3 divide-y divide-amber-600/20 rounded-lg border border-amber-600/35 bg-amber-500/5">
      {violations.map((violation) => {
        const key = `${violation.surface}:${violation.expected}`
        const currentScope = getScope(violation)
        const isConfirming = confirmViolation?.violation === violation
        const countAll = Math.max(1, violation.cascade_all)
        const countSubsequent = Math.max(1, violation.cascade_subsequent)
        const targetCount =
          currentScope === 'all'
            ? countAll
            : currentScope === 'subsequent'
              ? countSubsequent
              : 1

        return (
          <div key={key} className="px-2.5 py-2 flex flex-col gap-2 text-xs">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-mono text-muted-foreground">{t.review.recommendedTerm}</span>
              <span className="line-through text-amber-700 dark:text-amber-400 font-mono">{violation.surface}</span>
              <span className="text-muted-foreground">→</span>
              <span className="font-semibold text-emerald-700 dark:text-emerald-400 font-mono">{violation.expected}</span>

              {violation.cascade_all > 1 && (
                <div className="flex items-center gap-0.5 bg-muted/40 p-0.5 rounded border border-border ml-auto">
                  <button
                    type="button"
                    onClick={() => setScopeFor(violation, 'block')}
                    className={`px-1.5 py-0.5 rounded text-xs transition-colors cursor-pointer ${
                      currentScope === 'block'
                        ? 'bg-card text-foreground font-bold shadow-xs'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                    title={t.review.scopeBlock}
                  >
                    {t.review.scopeBlock} (1)
                  </button>
                  {violation.cascade_subsequent > 1 && (
                    <button
                      type="button"
                      onClick={() => setScopeFor(violation, 'subsequent')}
                      className={`px-1.5 py-0.5 rounded text-xs transition-colors cursor-pointer ${
                        currentScope === 'subsequent'
                          ? 'bg-card text-foreground font-bold shadow-xs'
                          : 'text-muted-foreground hover:text-foreground'
                      }`}
                      title={t.review.scopeSubsequent}
                    >
                      {t.review.scopeSubsequent} ({countSubsequent})
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => setScopeFor(violation, 'all')}
                    className={`px-1.5 py-0.5 rounded text-xs transition-colors cursor-pointer ${
                      currentScope === 'all'
                        ? 'bg-card text-foreground font-bold shadow-xs'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                    title={t.review.scopeAll}
                  >
                    {t.review.scopeAll} ({countAll})
                  </button>
                </div>
              )}

              <Button
                onClick={() => handleTriggerReplace(violation)}
                variant="default"
                size="sm"
                disabled={busy}
                className={`text-xs min-h-[28px] md:min-h-0 h-7 md:h-6 px-2.5 font-bold ${violation.cascade_all <= 1 ? 'ml-auto' : ''}`}
              >
                {busy ? (
                  <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                ) : (
                  <Wand2 className="h-3 w-3 mr-1" />
                )}
                {busy
                  ? t.review.cascadeApplying
                  : currentScope === 'block'
                    ? t.review.replaceTerm
                    : `${t.review.replaceTerm} (${targetCount})`}
              </Button>
            </div>

            {isConfirming && (
              <div className="w-full mt-1 p-2.5 rounded-md bg-amber-500/10 border border-amber-600/25 space-y-2 animate-in fade-in duration-150">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-foreground text-xs">
                    {currentScope === 'all'
                      ? t.review.cascadeConfirmTitle
                      : `${t.review.scopeSubsequent} · ${t.review.cascadeConfirmTitle}`}
                  </span>
                  <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-700 dark:text-amber-400 font-bold">
                    {t.review.scopeCount.replace('{n}', String(targetCount))}
                  </span>
                </div>

                <div className="flex items-center gap-2 p-1.5 rounded bg-card border border-border font-mono text-xs">
                  <span className="line-through text-amber-700 dark:text-amber-400 font-medium bg-amber-500/10 px-1.5 py-0.5 rounded">
                    {violation.surface}
                  </span>
                  <span className="text-muted-foreground">➔</span>
                  <span className="text-emerald-700 dark:text-emerald-400 font-bold bg-emerald-500/10 px-1.5 py-0.5 rounded">
                    {violation.expected}
                  </span>
                </div>

                <div className="text-muted-foreground text-xs leading-relaxed">
                  {t.review.cascadeConfirmDesc
                    .replace('{n}', String(targetCount))
                    .replace('{surface}', violation.surface)
                    .replace('{expected}', violation.expected)}
                </div>

                <div className="flex items-center gap-1.5 text-muted-foreground text-xs pt-0.5">
                  <Sparkles className="h-3 w-3 text-amber-500 shrink-0" />
                  <span>{t.review.cascadeTmSync}</span>
                </div>

                <div className="flex items-center gap-2 pt-1 border-t border-amber-600/20">
                  <Button
                    onClick={() => executeReplace(violation, currentScope)}
                    variant="default"
                    size="sm"
                    disabled={busy}
                    className="h-7 min-h-[28px] bg-amber-600 px-2.5 text-xs font-bold text-white hover:bg-amber-700 md:h-6 md:min-h-0"
                  >
                    {busy ? (
                      <Loader2 className="h-3 w-3 animate-spin mr-1" />
                    ) : (
                      <Wand2 className="h-3 w-3 mr-1" />
                    )}
                    {t.review.cascadeConfirmProceed} ({targetCount})
                  </Button>
                  <Button
                    onClick={() => setConfirmViolation(null)}
                    variant="secondary"
                    size="sm"
                    disabled={busy}
                    className="text-xs min-h-[28px] md:min-h-0 h-7 md:h-6 px-2.5"
                  >
                    {t.review.cascadeConfirmCancel}
                  </Button>
                </div>
              </div>
            )}
          </div>
        )
      })}
      {error && <div className="px-2.5 py-1.5 text-xs text-amber-700 dark:text-amber-400">{error}</div>}
    </div>
  )
}
