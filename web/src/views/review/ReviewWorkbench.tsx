import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useParams } from 'react-router-dom'
import { useVirtualizer, type VirtualItem } from '@tanstack/react-virtual'
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
  Sparkles,
  Download,
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { ConfirmDialog } from '@/components/ui/ConfirmDialog'
import { useToast } from '@/components/ui/useToast'
import {
  ApiError,
  listSegments,
  listIssues,
  listDeliverables,
  deliverableDownloadUrl,
  triggerBrowserDownload,
  editSegment,
  reExportJob,
  getBlockTerms,
  propagateTerm,
  pagePreviewUrl,
  sourcePageUrl,
  type Segment,
  type IssuesReport,
  type TermViolation,
  type DeliverableItem,
  type DeliverableListing,
} from '@/api/client'
import { useI18n } from '@/i18n/useI18n'
import { deliverableName } from '@/views/quality-gate/artifactMeta'

type Filter = 'issues' | 'all' | 'needs_human' | string

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

const getFormatBadge = (filename?: string): string => {
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
    default: return ext.toUpperCase() || 'FILE'
  }
}

const formatBytes = (bytes?: number): string => {
  if (!bytes || bytes <= 0 || isNaN(bytes)) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(1024))
  return `${(bytes / Math.pow(1024, i)).toFixed(i > 0 ? 1 : 0)} ${units[i]}`
}

export function ReviewWorkbench() {
  const { t } = useI18n()
  const { jobId: routeJobId } = useParams<{ jobId: string }>()
  const jobId = routeJobId ?? null
  const [viewMode, setViewMode] = useState<'segments' | 'reading_flow' | 'visual_witness'>('segments')
  const [mobileInspectorExpanded, setMobileInspectorExpanded] = useState(false)
  const [filter, setFilter] = useState<Filter>('issues')

  const [segments, setSegments] = useState<Segment[]>([])
  const [segmentTotal, setSegmentTotal] = useState(0)
  const [issues, setIssues] = useState<IssuesReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [loadingMore, setLoadingMore] = useState(false)
  const toast = useToast()
  const [error, setError] = useState<string | null>(null)
  const [drafts, setDrafts] = useState<Record<string, string>>(() => readDrafts(jobId))
  const [editing, setEditing] = useState<Record<string, boolean>>({})
  const [saving, setSaving] = useState<Record<string, boolean>>({})
  const [saved, setSaved] = useState<Record<string, boolean>>({})
  const [reExportNotice, setReExportNotice] = useState<string | null>(null)
  const [reExporting, setReExporting] = useState(false)
  const [reExportPending, setReExportPending] = useState(false)
  const [deliverables, setDeliverables] = useState<DeliverableItem[]>([])
  const [deliveryVerdict, setDeliveryVerdict] = useState<DeliverableListing['deliveryVerdict']>(null)
  const [exporting, setExporting] = useState(false)
  const [exportStatusText, setExportStatusText] = useState<string | null>(null)
  const [exportMenuOpen, setExportMenuOpen] = useState(false)
  const [pendingBlockedExport, setPendingBlockedExport] = useState<string | null>(null)
  const exportContainerRef = useRef<HTMLDivElement>(null)
  const [issueCursor, setIssueCursor] = useState(0)
  const [openPreviews, setOpenPreviews] = useState<Record<number, boolean>>({})
  const [previewBust, setPreviewBust] = useState(0)
  const [termNotice, setTermNotice] = useState<Record<string, string>>({})
  const [diffBlend, setDiffBlend] = useState(false)
  const [selectedBlockId, setSelectedBlockId] = useState<string | null>(null)
  const [inspectorOpen, setInspectorOpen] = useState(false)

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
        setDeliverables(listing.items)
        setDeliveryVerdict(listing.deliveryVerdict)
        setReExportPending(listing.stale)
      }
    } catch (err) {
      if (seq !== loadSeqRef.current) return // stale failure
      setError(err instanceof Error ? err.message : t.review.loadFailed)
    } finally {
      if (seq === loadSeqRef.current) setLoading(false)
    }
  }, [jobId, filter, t])

  // Close export dropdown on outside click
  useEffect(() => {
    if (!exportMenuOpen) return
    const handleClickOutside = (e: MouseEvent) => {
      if (exportContainerRef.current && !exportContainerRef.current.contains(e.target as Node)) {
        setExportMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [exportMenuOpen])

  // Filter reading-facing books / documents from internal inspection JSONs
  const readingDeliverables = useMemo(() => {
    return deliverables.filter((d) => {
      const ext = d.filename.split('.').pop()?.toLowerCase() ?? ''
      return (
        d.key === 'primary' ||
        d.key === 'secondary' ||
        d.key === 'epub' ||
        d.key === 'html' ||
        d.key === 'xliff' ||
        ext === 'pdf' ||
        ext === 'epub' ||
        ext === 'docx' ||
        ext === 'md' ||
        ext === 'html' ||
        ext === 'xliff'
      )
    })
  }, [deliverables])

  const primaryDeliverable = useMemo(() => {
    return (
      readingDeliverables.find((d) => d.key === 'primary') ??
      readingDeliverables[0] ??
      deliverables[0] ??
      null
    )
  }, [readingDeliverables, deliverables])

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
  const segmentMeasurementsCacheRef = useRef<VirtualItem[]>([])

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

  const estimateSegmentSize = useCallback(
    (index: number) => {
      const seg = segments[index]
      if (!seg) return 210
      const sourceLen = seg.source_text?.length ?? 0
      const draftText = drafts[seg.block_id] ?? seg.delivered_text ?? ''
      const targetLen = draftText.length
      const sourceLines = Math.ceil(sourceLen / 38)
      const targetLines = Math.ceil(targetLen / 30)
      const maxLines = Math.max(1, Math.max(sourceLines, targetLines))
      // Header (34px) + action footer (38px) + padding (32px) + bottom pb-4 (16px) = 120px
      const base = 120 + maxLines * 24
      const previewExtra = seg.page !== null && openPreviews[seg.page] ? 380 : 0
      const failExtra = seg.repair_failed ? 36 : 0
      const termExtra = seg.issues.includes('terminology') ? 60 : 0
      return Math.min(1000, Math.max(160, base + previewExtra + failExtra + termExtra))
    },
    [segments, drafts, openPreviews]
  )

  // Virtualize the segment list with dynamic size measurement and key-based cache
  // oxlint-disable-next-line react/incompatible-library
  const virtualizer = useVirtualizer({
    count: segments.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: estimateSegmentSize,
    getItemKey: getSegmentKey,
    initialMeasurementsCache: segmentMeasurementsCacheRef.current,
    onChange: (instance) => {
      segmentMeasurementsCacheRef.current = instance.takeSnapshot()
    },
    overscan: 6,
  })

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

  // F8 / Shift+F8 walk the fault list (PRD §5.1). Escape closes the inspector.
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && inspectorOpen) {
        const active = document.activeElement
        if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA')) {
          return
        }
        setInspectorOpen(false)
        return
      }
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
      // The edit changed the ledger text: any open page preview is now stale.
      if (segment.page !== null && openPreviews[segment.page]) {
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

  const issueWorkbenchDownload = async (key: string, force = false) => {
    if (!jobId || exporting) return
    setExporting(true)
    setExportMenuOpen(false)
    setExportStatusText(t.review.exportPreparing)
    try {
      // 1. Flush any uncommitted in-memory edits to the ledger before exporting
      const unsavedEntries = Object.entries(drafts).filter(([blockId, text]) => {
        const seg = segments.find((s) => s.block_id === blockId)
        return seg && text !== seg.delivered_text
      })
      let reExportNeeded = reExportPending
      if (unsavedEntries.length > 0) {
        setExportStatusText(t.review.exportSavingDrafts)
        for (const [blockId, draftText] of unsavedEntries) {
          try {
            const res = await editSegment(jobId, blockId, draftText)
            if (res.segment) {
              const updated = res.segment
              setSegments((prev) => prev.map((s) => (s.block_id === updated.block_id ? updated : s)))
            }
            setSaved((prev) => ({ ...prev, [blockId]: true }))
            setEditing((prev) => ({ ...prev, [blockId]: false }))
            setDrafts((prev) => {
              const next = { ...prev }
              delete next[blockId]
              return next
            })
          } catch (err) {
            console.error('Failed to auto-save draft for block', blockId, err)
          }
        }
        reExportNeeded = true
      }

      // 2. Fetch fresh deliverable listing to check freshness and gate status
      let currentItems = deliverables
      let listing: DeliverableListing | null = null
      try {
        listing = await listDeliverables(jobId)
        setDeliverables(listing.items)
        setDeliveryVerdict(listing.deliveryVerdict)
        currentItems = listing.items
        if (listing.stale) {
          reExportNeeded = true
        }
      } catch {
        // Fall back to currentItems if listDeliverables fails
      }

      // 3. Re-export if needed due to edits, pending changes, or stale deliverables
      if (reExportNeeded) {
        setExportStatusText(t.review.exportRerendering)
        toast.push(t.review.exportRerendering, 'info')
        await reExportJob(jobId)
        setReExportPending(false)
        const refreshed = await listDeliverables(jobId).catch(() => null)
        if (refreshed) {
          setDeliverables(refreshed.items)
          setDeliveryVerdict(refreshed.deliveryVerdict)
          currentItems = refreshed.items
          if (refreshed.deliveryVerdict === 'blocked' && !force) {
            setPendingBlockedExport(key)
            return
          }
        }
      } else if (listing && listing.deliveryVerdict === 'blocked' && !force) {
        setPendingBlockedExport(key)
        return
      }

      // 4. If current items are empty, try fetching one last time
      if (currentItems.length === 0) {
        const refreshed = await listDeliverables(jobId).catch(() => null)
        if (refreshed) {
          setDeliverables(refreshed.items)
          setDeliveryVerdict(refreshed.deliveryVerdict)
          currentItems = refreshed.items
        }
      }

      // 5. Trigger download
      const target = currentItems.find((d) => d.key === key) ?? currentItems[0]
      if (target) {
        const url = deliverableDownloadUrl(jobId, target.key, { force })
        triggerBrowserDownload(url, target.filename)
        toast.push(t.review.exportSuccess, 'success')
      } else {
        toast.push(t.mission.noDeliverables, 'error')
      }
    } catch (err) {
      toast.push(err instanceof Error ? err.message : t.review.exportFailed, 'error')
    } finally {
      setExporting(false)
      setExportStatusText(null)
    }
  }

  const handleOneClickExport = async (key?: string) => {
    if (!jobId || exporting) return
    const targetKey = key ?? primaryDeliverable?.key ?? 'primary'
    if (deliveryVerdict === 'blocked') {
      setPendingBlockedExport(targetKey)
      return
    }
    await issueWorkbenchDownload(targetKey, false)
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
      <div className="h-8 px-3 flex items-center justify-between text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] border-b border-[var(--paper-border)]">
        <span>
          {t.review.previewPage} · P{page}
        </span>
        <button
          type="button"
          onClick={() => setPreviewBust(Date.now())}
          title={t.review.rerenderHint}
          className="min-h-[28px] md:min-h-0 flex items-center gap-1.5 px-2 py-1 md:py-0.5 rounded text-xs text-[var(--ink-secondary)] hover:text-[var(--ink-primary)] font-mono transition-colors cursor-pointer hover:bg-[var(--paper-surface)]"
        >
          <RefreshCw className="h-3.5 w-3.5 md:h-3 md:w-3" />
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
      <div className="min-h-11 px-3 md:px-6 py-2 border-b border-[var(--paper-border)] bg-[var(--paper-surface)] flex flex-wrap items-center justify-between gap-2 shrink-0 shadow-2xs">
        <div className="flex items-center gap-2 md:gap-3 flex-wrap">
          <div className="flex items-center gap-1.5 text-xs font-bold text-[var(--ink-primary)]">
            <SplitSquareVertical className="h-4 w-4 text-[#15803d]" />
            <span>{t.review.title}</span>
          </div>
          <div className="h-3 w-px bg-[var(--paper-border)] hidden sm:block" />
          <div className="flex items-center gap-1.5 md:gap-2 text-xs font-mono flex-wrap">
            {ribbonCounts.length === 0 ? (
              <span className="text-[#15803d] flex items-center gap-1 font-medium">
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
                    className={`min-h-[30px] md:min-h-0 px-2 py-1 md:py-0.5 rounded border transition-colors cursor-pointer text-xs font-mono flex items-center gap-1 ${
                      isActive
                        ? 'border-[#b45309] bg-[#b45309]/20 text-[#b45309] font-bold shadow-xs'
                        : 'border-[var(--paper-border)] bg-[var(--paper-subsurface)] text-[var(--ink-secondary)] hover:border-[var(--paper-border-hover)] hover:text-[var(--ink-primary)]'
                    }`}
                    title={isActive ? t.review.clearFilter : `${t.review.filterByIssueKind}: ${issueLabel(t, kind)}`}
                  >
                    <span>{issueLabel(t, kind)}</span>
                    <span className="opacity-80 font-bold">{value}</span>
                    {isActive && <span className="ml-0.5 text-[10px] opacity-70">✕</span>}
                  </button>
                )
              })
            )}
            {(issues?.status.needs_human ?? 0) > 0 && (
              <button
                type="button"
                onClick={() => setFilter(filter === 'needs_human' ? 'issues' : 'needs_human')}
                className={`min-h-[30px] md:min-h-0 px-2 py-1 md:py-0.5 rounded border transition-colors cursor-pointer text-xs font-mono flex items-center gap-1 ${
                  filter === 'needs_human'
                    ? 'border-[#b45309] bg-[#b45309]/20 text-[#b45309] font-bold shadow-xs'
                    : 'border-[#b45309]/40 bg-[#b45309]/10 text-[#b45309] hover:bg-[#b45309]/20'
                }`}
              >
                {blockStatusLabel(t, 'needs_human')} {issues?.status.needs_human}
              </button>
            )}
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={() => jumpToIssue(-1)}
              title={t.review.prevIssue}
              className="p-1.5 md:p-1 min-h-[30px] min-w-[30px] md:min-h-0 md:min-w-0 flex items-center justify-center rounded hover:bg-[var(--paper-subsurface)] text-[var(--ink-muted)] cursor-pointer"
            >
              <ChevronUp className="h-4 w-4 md:h-3.5 md:w-3.5" />
            </button>
            <button
              onClick={() => jumpToIssue(1)}
              title={t.review.nextIssue}
              className="p-1.5 md:p-1 min-h-[30px] min-w-[30px] md:min-h-0 md:min-w-0 flex items-center justify-center rounded hover:bg-[var(--paper-subsurface)] text-[var(--ink-muted)] cursor-pointer"
            >
              <ChevronDown className="h-4 w-4 md:h-3.5 md:w-3.5" />
            </button>
          </div>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {filter !== 'issues' && filter !== 'needs_human' && filter !== 'all' && (
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-[4px] bg-[#b45309]/15 border border-[#b45309]/40 text-[#b45309] text-xs font-mono">
              <span className="font-semibold">{issueLabel(t, filter)}</span>
              <button
                type="button"
                onClick={() => setFilter('issues')}
                title={t.review.clearFilter}
                className="hover:opacity-100 opacity-60 text-xs font-bold cursor-pointer p-0.5 min-h-[24px] min-w-[24px] flex items-center justify-center"
              >
                ✕
              </button>
            </div>
          )}
          <div className="flex bg-[var(--paper-subsurface)] p-0.5 rounded-lg border border-[var(--paper-border)] shadow-xs">
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
                className={`btn-tactile min-h-[30px] md:min-h-0 px-2.5 py-1 text-xs rounded-md transition-all font-medium cursor-pointer ${
                  filter === id
                    ? 'bg-[var(--paper-surface)] text-[var(--ink-primary)] font-bold shadow-xs'
                    : 'text-[var(--ink-muted)] hover:text-[var(--ink-primary)]'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="flex bg-[var(--paper-subsurface)] p-0.5 rounded-lg border border-[var(--paper-border)] shadow-xs">
            <button
              onClick={() => setViewMode('segments')}
              className={`btn-tactile min-h-[30px] md:min-h-0 px-3 py-1 text-xs rounded-md transition-all font-medium cursor-pointer ${
                viewMode === 'segments'
                  ? 'bg-[var(--paper-surface)] text-[var(--ink-primary)] font-bold shadow-xs'
                  : 'text-[var(--ink-muted)] hover:text-[var(--ink-primary)]'
              }`}
            >
              {t.review.modeSegments}
            </button>
            <button
              onClick={() => setViewMode('reading_flow')}
              className={`btn-tactile min-h-[30px] md:min-h-0 px-3 py-1 text-xs rounded-md transition-all font-medium cursor-pointer ${
                viewMode === 'reading_flow'
                  ? 'bg-[var(--paper-surface)] text-[var(--ink-primary)] font-bold shadow-xs'
                  : 'text-[var(--ink-muted)] hover:text-[var(--ink-primary)]'
              }`}
            >
              {t.review.modeFlow}
            </button>
            <button
              onClick={() => setViewMode('visual_witness')}
              className={`btn-tactile min-h-[30px] md:min-h-0 px-3 py-1 text-xs rounded-md transition-all font-medium cursor-pointer ${
                viewMode === 'visual_witness'
                  ? 'bg-[var(--paper-surface)] text-[var(--ink-primary)] font-bold shadow-xs'
                  : 'text-[var(--ink-muted)] hover:text-[var(--ink-primary)]'
              }`}
            >
              {t.review.modeVisual}
            </button>
          </div>

          {/* Explicit re-render: it rebuilds the whole delivered file, so it
              runs once per review session rather than on every save. */}
          <button
            onClick={() => void handleReExport()}
            disabled={reExporting || exporting}
            className={`btn-tactile min-h-[30px] md:min-h-0 px-3 py-1 text-xs rounded-lg border transition-all font-medium cursor-pointer disabled:cursor-wait ${
              reExportPending
                ? 'border-[#b45309]/40 bg-[#b45309]/10 text-[#b45309] font-semibold shadow-xs'
                : 'border-[var(--paper-border)] bg-[var(--paper-surface)] text-[var(--ink-secondary)] hover:text-[var(--ink-primary)] shadow-xs'
            }`}
            title={t.review.reExportHelp}
          >
            {reExporting
              ? t.review.reExporting
              : reExportPending
                ? t.review.reExportStale
                : t.review.reExportAction}
          </button>

          {/* Prominent One-Click Export & Deliver Primary Action */}
          <div ref={exportContainerRef} className="relative inline-flex items-center shadow-xs">
            <Button
              onClick={() => void handleOneClickExport()}
              disabled={exporting}
              variant="primary"
              size="sm"
              className={`h-9 sm:h-7 px-3 text-xs font-bold transition-all flex items-center gap-1.5 ${
                readingDeliverables.length > 1 ? 'rounded-r-none pr-2 border-r border-white/20' : ''
              }`}
              title={t.review.exportQuickHelp}
            >
              {exporting ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Download className="h-3.5 w-3.5" />
              )}
              <span>{exporting && exportStatusText ? exportStatusText : t.review.oneClickExport}</span>
              {primaryDeliverable && !exporting && (
                <span className="hidden sm:inline-block text-[10px] font-mono opacity-80 uppercase px-1 py-0.5 bg-black/15 rounded">
                  {getFormatBadge(primaryDeliverable.filename)}
                </span>
              )}
            </Button>
            {readingDeliverables.length > 1 && (
              <button
                type="button"
                onClick={() => setExportMenuOpen((prev) => !prev)}
                disabled={exporting}
                title={t.review.exportFormats}
                className="h-9 sm:h-7 px-1.5 text-xs bg-[var(--btn-bg)] text-[var(--btn-fg)] hover:bg-[var(--btn-hover)] rounded-r transition-all flex items-center justify-center cursor-pointer border-l border-white/20 disabled:opacity-60"
              >
                <ChevronDown className={`h-3.5 w-3.5 transition-transform ${exportMenuOpen ? 'rotate-180' : ''}`} />
              </button>
            )}

            {/* Format Dropdown Menu */}
            {exportMenuOpen && (
              <div
                className="absolute right-0 top-full mt-1.5 w-64 bg-[var(--paper-surface)] border border-[var(--paper-border)] rounded-md shadow-lg py-1 z-50 text-xs"
                onClick={(e) => e.stopPropagation()}
              >
                <div className="px-3 py-1.5 text-[11px] font-semibold text-[var(--ink-muted)] border-b border-[var(--paper-border)] uppercase tracking-wider">
                  {t.review.exportFormats}
                </div>
                <div className="max-h-60 overflow-y-auto py-1">
                  {readingDeliverables.map((item) => {
                    const isPrimary = item.key === 'primary' || item === primaryDeliverable
                    const badge = getFormatBadge(item.filename)
                    return (
                      <button
                        key={item.key}
                        type="button"
                        onClick={() => void handleOneClickExport(item.key)}
                        className="w-full px-3 py-2 text-left hover:bg-[var(--paper-subsurface)] flex items-center justify-between gap-2 transition-colors cursor-pointer"
                      >
                        <div className="min-w-0 flex-1">
                          <div className="flex items-center gap-1.5">
                            <span className="font-mono text-[10px] font-bold px-1.5 py-0.5 rounded bg-[var(--paper-border)] text-[var(--ink-primary)]">
                              {badge}
                            </span>
                            <span className="font-medium text-[var(--ink-primary)] truncate">
                              {deliverableName(item.key, item.label || item.filename, t)}
                            </span>
                          </div>
                          <div className="text-[10px] text-[var(--ink-muted)] font-mono mt-0.5 truncate">
                            {item.filename} · {formatBytes(item.size_bytes)}
                          </div>
                        </div>
                        {isPrimary && (
                          <span className="shrink-0 text-[10px] px-1 py-0.5 rounded bg-[#15803d]/15 text-[#15803d] font-semibold font-mono">
                            {t.review.exportPrimaryBadge}
                          </span>
                        )}
                      </button>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {reExportNotice && (
        <div className="px-6 py-2 text-xs text-[var(--ink-secondary)] bg-[var(--paper-subsurface)] border-b border-[var(--paper-border)]">
          {reExportNotice}
        </div>
      )}

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
          <div className="flex-1 flex flex-col items-center justify-center gap-3 p-8 text-xs text-[var(--ink-muted)]">
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
            <div
              className="relative w-full max-w-5xl mx-auto py-6"
              style={{ height: virtualizer.getTotalSize() }}
            >
              {virtualizer.getVirtualItems().map((virtualRow) => {
                const seg = segments[virtualRow.index]
                const draft = drafts[seg.block_id] ?? seg.delivered_text
                const dirty = draft !== seg.delivered_text
                // The engine's wrapper semantics live server-side (one owner);
                // the segment arrives with the delivered text, the repair
                // failure flag and its decoded reason already split apart.
                const failed = {
                  text: seg.delivered_text,
                  reason: seg.repair_error,
                  failed: seg.repair_failed,
                }
                const previewOpen = seg.page !== null && openPreviews[seg.page]
                return (
                  <div
                    key={seg.block_id}
                    data-index={virtualRow.index}
                    ref={virtualizer.measureElement}
                    className="absolute top-0 left-0 w-full px-3 md:px-8"
                    style={{ transform: `translateY(${virtualRow.start}px)` }}
                  >
                    <div className="pb-4">
                      <div
                        onClick={() => {
                          setSelectedBlockId(seg.block_id)
                          setInspectorOpen(true)
                        }}
                        className={`card-tactile reading-segment rounded-xl border transition-all shadow-[var(--shadow-card)] bg-[var(--paper-surface)] cursor-pointer ${
                          selectedBlockId === seg.block_id
                            ? 'border-[var(--ink-primary)] ring-1 ring-[var(--paper-border-hover)]'
                            : seg.issues.length > 0
                              ? 'border-[#b45309]/50 hover:border-[#b45309]'
                              : 'border-[var(--paper-border)] hover:border-[var(--paper-border-hover)]'
                        }`}
                      >
                        <div className="h-8.5 px-3.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] flex items-center justify-between text-xs font-mono text-[var(--ink-muted)]">
                          <div className="flex items-center gap-2">
                            <span className="text-[var(--ink-primary)] font-bold">
                              {seg.block_id}
                            </span>
                            {seg.page !== null && (
                              <button
                                onClick={(e) => {
                                  e.stopPropagation()
                                  togglePreview(seg.page)
                                }}
                                title={t.review.previewPage}
                                className={`btn-tactile min-h-[28px] md:min-h-0 px-2 py-1 md:py-0.5 rounded text-xs font-semibold transition-colors cursor-pointer ${
                                  previewOpen
                                    ? 'bg-[var(--ink-primary)] text-[var(--paper-bg)]'
                                    : 'bg-[var(--paper-border)] text-[var(--ink-secondary)] hover:text-[var(--ink-primary)]'
                                }`}
                              >
                                {t.review.pageLabel.replace('{n}', String(seg.page))}
                              </button>
                            )}
                            <span className="text-xs px-1.5 py-0.5 rounded bg-[var(--paper-border)] text-[var(--ink-secondary)] font-semibold">
                              {blockTypeLabel(t, seg.block_type)}
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            {seg.issues.map((kind) => (
                              <span
                                key={kind}
                                className="text-xs px-1.5 py-0.5 rounded bg-[#b45309]/15 text-[#b45309] font-semibold"
                              >
                                {issueLabel(t, kind)}
                              </span>
                            ))}
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
                                <span className={`text-[11px] font-mono font-bold px-2 py-0.5 rounded-full border ${colorClass}`}>
                                  QE {pct}%
                                </span>
                              )
                            })()}
                            {seg.human_verified ? (
                              <Badge variant="success" dot>
                                {t.review.humanVerified}
                              </Badge>
                            ) : (
                              <Badge
                                variant={
                                  seg.status === 'mtqe_passed'
                                    ? 'success'
                                    : seg.status === 'needs_human' || seg.issues.length > 0
                                      ? 'warning'
                                      : seg.status === 'blocked_human' || seg.status === 'failed'
                                        ? 'destructive'
                                        : 'outline'
                                }
                                dot
                              >
                                {blockStatusLabel(t, seg.status)}
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
                                {/* A draft restored from a previous session is
                                    indistinguishable from the saved text
                                    otherwise: the reviewer would not know which
                                    block they had left unfinished. */}
                                {!saved[seg.block_id] &&
                                  drafts[seg.block_id] !== undefined &&
                                  drafts[seg.block_id] !== seg.delivered_text && (
                                    <span className="text-[#b45309] font-sans font-medium">
                                      {t.review.unsavedDraft}
                                    </span>
                                  )}
                              </div>
                              {editing[seg.block_id] ? (
                                <AutoGrowTextarea
                                  autoFocus
                                  // Editing starts from the *translation*, not
                                  // from the <mark> markup: saving what was on
                                  // screen would have written the engine's
                                  // wrapper back into the block as content.
                                  value={
                                    drafts[seg.block_id] !== undefined
                                      ? draft
                                      : failed.text
                                  }
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
                                      e.stopPropagation()
                                      setDrafts((prev) => {
                                        const next = { ...prev }
                                        delete next[seg.block_id]
                                        return next
                                      })
                                      setEditing((prev) => ({ ...prev, [seg.block_id]: false }))
                                    }
                                  }}
                                  // No `rows`: the height is measured from the
                                  // content, so one block's editor is never a
                                  // three-line window onto its own paragraph.
                                  className="w-full bg-transparent text-[var(--ink-primary)] text-xs resize-none focus:outline-none font-mono leading-relaxed placeholder:text-[var(--ink-muted)]"
                                />
                              ) : (
                                // A fragment, not a bare element: the pane needs
                                // two siblings (the translation and the failure
                                // notice), and a ternary arms only one.
                                <>
                                  <div
                                    onDoubleClick={() =>
                                      setEditing((prev) => ({ ...prev, [seg.block_id]: true }))
                                    }
                                    // Post-editing was reachable ONLY by double-click,
                                    // and the Save button stays disabled until an edit
                                    // is open — so a keyboard user could browse, filter
                                    // and read the whole book but could not fix a single
                                    // block. role+tabIndex+Enter make the pane a control;
                                    // the double-click stays as a mouse shortcut.
                                    role="button"
                                    tabIndex={0}
                                    onKeyDown={(event) => {
                                      if (event.key === 'Enter') {
                                        event.preventDefault()
                                        setEditing((prev) => ({ ...prev, [seg.block_id]: true }))
                                      }
                                    }}
                                    aria-label={t.review.startEditing}
                                    // The card's own onClick opens the inspector,
                                    // which narrows this column from 446px to
                                    // 286px at 1280px. A double-click fires a
                                    // click first, so the element moved out from
                                    // under the second click and the editor never
                                    // opened (measured: 0 of 2 attempts at 1280
                                    // and 1366, 2 of 2 at 1600). Opening the
                                    // editor is this node's own gesture; it must
                                    // not also reflow the card underneath it.
                                    onClick={(event) => event.stopPropagation()}
                                    title={t.review.editHint}
                                    className="min-h-[3rem] text-[var(--ink-primary)] text-xs font-mono leading-relaxed whitespace-pre-wrap break-words cursor-text"
                                  >
                                    {failed.text || '—'}
                                  </div>
                                  {/* Why the repair failed, decoded and wrapped.
                                      It used to sit inside the markup, where it
                                      rendered as 828px of escaped JSON in a 446px
                                      column. `break-words` + a bounded box keeps
                                      a long reason inside the pane. */}
                                  {failed.failed && (
                                    <div className="mt-1.5 rounded border border-[#b45309]/30 bg-[#b45309]/5 px-2 py-1 text-[10px] font-mono text-[#b45309] break-words max-h-24 overflow-y-auto">
                                      {t.review.failedDraftNotice}: {failed.reason}
                                    </div>
                                  )}
                                  {seg.quarantined && (
                                    <div className="mt-1.5 rounded border border-[var(--paper-border)] bg-[var(--paper-subsurface)] px-2 py-1 text-[10px] font-mono text-[var(--ink-secondary)] break-words">
                                      {t.review.quarantinedNotice}
                                    </div>
                                  )}
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
                                  className="btn-tactile text-xs h-7 md:h-6 px-2.5"
                                >
                                  {t.review.cancelEdit}
                                </Button>
                              )}
                              <Button
                                onClick={() => handleSave(seg)}
                                variant="primary"
                                size="sm"
                                disabled={!editing[seg.block_id] || !dirty || saving[seg.block_id]}
                                className="btn-tactile text-xs h-7 md:h-6 px-2.5 font-bold"
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
      ) : viewMode === 'reading_flow' ? (
        /* Reading Flow View: continuous natural layout with dual-column typography */
        <div ref={flowScrollRef} className="flex-1 overflow-y-auto px-4 md:px-10 py-6 bg-[var(--paper-bg)]">
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
                      className={`reading-segment reading-spine-line card-tactile grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6 p-4 rounded-xl transition-all cursor-pointer border ${
                        isSelected
                          ? 'bg-[var(--paper-surface)] shadow-[var(--shadow-card)] border-[var(--ink-primary)] ring-1 ring-[var(--paper-border-hover)]'
                          : hasIssues
                            ? 'bg-[#b45309]/5 hover:bg-[#b45309]/10 border-l-[3px] border-l-[#b45309] border-y-[var(--paper-border)] border-r-[var(--paper-border)]'
                            : 'border-[var(--paper-border)] hover:border-[var(--paper-border-hover)] hover:bg-[var(--paper-surface)] bg-[var(--paper-subsurface)]'
                      }`}
                    >
                      {/* Left: Source text */}
                      <div className="min-w-0 break-words [overflow-wrap:anywhere] text-xs md:text-sm font-mono leading-relaxed text-[var(--ink-secondary)] select-text md:pr-4 max-w-[65ch] [text-wrap:pretty]">
                        <div className="text-[10px] font-mono uppercase text-[var(--ink-muted)] mb-1 flex items-center justify-between">
                          <span>{seg.block_id} · P{seg.page ?? '—'}</span>
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-[var(--paper-border)] text-[var(--ink-secondary)] font-semibold">
                            {blockTypeLabel(t, seg.block_type)}
                          </span>
                        </div>
                        <div className="whitespace-pre-wrap">{seg.source_text}</div>
                      </div>

                      {/* Right: Target translation */}
                      <div className="min-w-0 break-words [overflow-wrap:anywhere] text-xs md:text-sm font-sans leading-relaxed text-[var(--ink-primary)] md:pl-2 max-w-[65ch] [text-wrap:pretty]">
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
                              className="w-full text-xs md:text-sm font-sans bg-[var(--paper-surface)] border border-[var(--paper-border)] p-2.5 rounded-lg focus:border-[var(--ink-primary)] focus:outline-none resize-none leading-relaxed text-[var(--ink-primary)] shadow-xs"
                            />
                            <div className="flex items-center justify-between text-xs">
                              <span className="text-[var(--ink-muted)] font-mono text-[10px]">{t.review.editShortcutHint}</span>
                              <Button
                                size="sm"
                                className="btn-tactile text-xs h-7 px-2.5 font-bold"
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
                            className="group relative cursor-pointer focus:outline-hidden focus:ring-1 focus:ring-[var(--ink-primary)] rounded"
                          >
                            <p className="select-text whitespace-pre-wrap leading-relaxed">{draft}</p>
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
                                  <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded border ${colorClass}`}>
                                    QE {pct}%
                                  </span>
                                )
                              })()}
                              {hasIssues &&
                                seg.issues.map((kind) => (
                                  <span
                                    key={kind}
                                    className="text-[10px] px-1.5 py-0.5 rounded bg-[#b45309]/15 text-[#b45309] font-mono font-medium"
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
      ) : (
        /* Visual witness: source page vs the composed translation (PRD §5.1 mode B). */
        <div ref={witnessScrollRef} className="flex-1 overflow-y-auto">
          <div className="max-w-6xl mx-auto w-full px-3 md:px-8 py-4 md:py-6">
            <div className="flex items-center justify-end">
              <label className="flex items-center gap-1.5 text-xs text-[var(--ink-secondary)] cursor-pointer select-none py-1">
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
                            <Eye className="h-3.5 w-3.5" /> {t.review.pageLabel.replace('{n}', String(page))}
                          </span>
                          <button
                            type="button"
                            onClick={() => setPreviewBust(Date.now())}
                            title={t.review.rerenderHint}
                            className="min-h-[28px] md:min-h-0 flex items-center gap-1.5 px-2 py-1 md:py-0.5 rounded text-xs text-[var(--ink-secondary)] hover:text-[var(--ink-primary)] font-mono transition-colors cursor-pointer hover:bg-[var(--paper-surface)]"
                          >
                            <RefreshCw className="h-3.5 w-3.5 md:h-3 md:w-3" />
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
                          <div className="grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-[var(--paper-border)]">
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
        <>
          {/* Mobile Overlay Backdrop (< md) */}
          <div
            className="fixed inset-0 z-40 bg-black/40 backdrop-blur-xs md:hidden"
            onClick={() => setInspectorOpen(false)}
            aria-hidden="true"
          />

          {/* Inspector Panel: Fixed Bottom Sheet on Mobile (< md), Side-by-Side Column on Desktop (md+) */}
          <aside className={`fixed inset-x-0 bottom-0 z-50 transition-all duration-200 ${mobileInspectorExpanded ? 'max-h-[85vh]' : 'max-h-[42vh]'} rounded-t-xl border-t border-[var(--paper-border)] shadow-2xl bg-[var(--paper-surface)] flex flex-col overflow-hidden overscroll-contain shrink-0 select-text pb-[env(safe-area-inset-bottom,0px)] md:static md:z-auto md:max-h-none md:h-full md:rounded-none md:border-t-0 md:w-80 md:border-l md:shadow-none md:pb-0`}>
            {/* Grab Handle Indicator (Mobile Only) */}
            <div
              className="md:hidden flex justify-center pt-2 pb-1 shrink-0 cursor-pointer"
              onClick={() => setMobileInspectorExpanded(!mobileInspectorExpanded)}
              title={mobileInspectorExpanded ? t.review.collapseInspector : t.review.expandInspector}
            >
              <div className="w-10 h-1.5 rounded-full bg-[var(--paper-border)] hover:bg-[var(--ink-muted)] transition-colors" />
            </div>

            <div className="h-11 px-4 border-b border-[var(--paper-border)] flex items-center justify-between shrink-0">
              <span className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
                {t.review.inspectorTitle}
              </span>
              <button
                type="button"
                onClick={() => setInspectorOpen(false)}
                className="w-8 h-8 min-h-[32px] min-w-[32px] flex items-center justify-center rounded text-xs font-mono text-[var(--ink-muted)] hover:text-[var(--ink-primary)] hover:bg-[var(--paper-subsurface)] transition-colors cursor-pointer"
                aria-label={t.common.dismiss}
              >
                ✕
              </button>
            </div>
            <div className="p-4 space-y-3 text-xs overflow-y-auto flex-1">
              <div>
                <div className="text-xs font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
                  {inspectorSegment.block_id}
                </div>
                <div className="text-xs text-[var(--ink-secondary)] font-mono mt-0.5">
                  {blockTypeLabel(t, inspectorSegment.block_type)} · P{inspectorSegment.page ?? '—'}
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
                  <dd>{mqmSeverityLabel(t, inspectorSegment.mqm_severity)}</dd>
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
                    {t.review.errorFlags}
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
        </>
      )}

      <ConfirmDialog
        open={pendingBlockedExport !== null}
        title={t.review.exportBlockedTitle}
        body={t.review.exportBlockedBody}
        confirmLabel={t.review.exportBlockedConfirm}
        cancelLabel={t.review.exportBlockedCancel}
        danger
        onConfirm={() => {
          const key = pendingBlockedExport ?? 'primary'
          setPendingBlockedExport(null)
          void issueWorkbenchDownload(key, true)
        }}
        onCancel={() => setPendingBlockedExport(null)}
      />
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
    <div className="mt-3 rounded-[6px] border border-[#b45309]/40 bg-[#b45309]/5 divide-y divide-[#b45309]/20">
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
              <span className="font-mono text-[var(--ink-muted)]">{t.review.recommendedTerm}</span>
              <span className="line-through text-[#b45309] font-mono">{violation.surface}</span>
              <span className="text-[var(--ink-muted)]">→</span>
              <span className="font-semibold text-[#15803d] font-mono">{violation.expected}</span>

              {violation.cascade_all > 1 && (
                <div className="flex items-center gap-0.5 bg-[var(--paper-subsurface)] p-0.5 rounded border border-[var(--paper-border)] ml-auto">
                  <button
                    type="button"
                    onClick={() => setScopeFor(violation, 'block')}
                    className={`px-1.5 py-0.5 rounded text-[10.5px] transition-colors cursor-pointer ${
                      currentScope === 'block'
                        ? 'bg-[var(--paper-surface)] text-[var(--ink-primary)] font-bold shadow-xs'
                        : 'text-[var(--ink-muted)] hover:text-[var(--ink-primary)]'
                    }`}
                    title={t.review.scopeBlock}
                  >
                    {t.review.scopeBlock} (1)
                  </button>
                  {violation.cascade_subsequent > 1 && (
                    <button
                      type="button"
                      onClick={() => setScopeFor(violation, 'subsequent')}
                      className={`px-1.5 py-0.5 rounded text-[10.5px] transition-colors cursor-pointer ${
                        currentScope === 'subsequent'
                          ? 'bg-[var(--paper-surface)] text-[var(--ink-primary)] font-bold shadow-xs'
                          : 'text-[var(--ink-muted)] hover:text-[var(--ink-primary)]'
                      }`}
                      title={t.review.scopeSubsequent}
                    >
                      {t.review.scopeSubsequent} ({countSubsequent})
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => setScopeFor(violation, 'all')}
                    className={`px-1.5 py-0.5 rounded text-[10.5px] transition-colors cursor-pointer ${
                      currentScope === 'all'
                        ? 'bg-[var(--paper-surface)] text-[var(--ink-primary)] font-bold shadow-xs'
                        : 'text-[var(--ink-muted)] hover:text-[var(--ink-primary)]'
                    }`}
                    title={t.review.scopeAll}
                  >
                    {t.review.scopeAll} ({countAll})
                  </button>
                </div>
              )}

              <Button
                onClick={() => handleTriggerReplace(violation)}
                variant="primary"
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
              <div className="w-full mt-1 p-2.5 rounded-[4px] bg-[#b45309]/10 border border-[#b45309]/30 space-y-2 animate-in fade-in duration-150">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-[var(--ink-primary)] text-xs">
                    {currentScope === 'all'
                      ? t.review.cascadeConfirmTitle
                      : `${t.review.scopeSubsequent} · ${t.review.cascadeConfirmTitle}`}
                  </span>
                  <span className="text-[10.5px] font-mono px-1.5 py-0.5 rounded bg-[#b45309]/20 text-[#b45309] font-bold">
                    {t.review.scopeCount.replace('{n}', String(targetCount))}
                  </span>
                </div>

                <div className="flex items-center gap-2 p-1.5 rounded bg-[var(--paper-surface)] border border-[var(--paper-border)] font-mono text-xs">
                  <span className="line-through text-[#b45309] font-medium bg-[#b45309]/10 px-1.5 py-0.5 rounded">
                    {violation.surface}
                  </span>
                  <span className="text-[var(--ink-muted)]">➔</span>
                  <span className="text-[#15803d] font-bold bg-[#15803d]/10 px-1.5 py-0.5 rounded">
                    {violation.expected}
                  </span>
                </div>

                <div className="text-[var(--ink-secondary)] text-[10.5px] leading-relaxed">
                  {t.review.cascadeConfirmDesc
                    .replace('{n}', String(targetCount))
                    .replace('{surface}', violation.surface)
                    .replace('{expected}', violation.expected)}
                </div>

                <div className="flex items-center gap-1.5 text-[var(--ink-secondary)] text-[11px] pt-0.5">
                  <Sparkles className="h-3 w-3 text-amber-500 shrink-0" />
                  <span>{t.review.cascadeTmSync}</span>
                </div>

                <div className="flex items-center gap-2 pt-1 border-t border-[#b45309]/20">
                  <Button
                    onClick={() => executeReplace(violation, currentScope)}
                    variant="primary"
                    size="sm"
                    disabled={busy}
                    className="text-xs min-h-[28px] md:min-h-0 h-7 md:h-6 px-2.5 bg-[#b45309] hover:bg-[#92400e] text-white font-bold"
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
      {error && <div className="px-2.5 py-1.5 text-xs text-[#b45309]">{error}</div>}
    </div>
  )
}
