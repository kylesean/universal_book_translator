import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  CheckCheck,
  Edit3,
  ExternalLink,
  Filter,
  LoaderCircle,
  RefreshCw,
  RotateCcw,
  Save,
  Search,
  X,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Card } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { useToast } from '@/components/ui/useToast'
import { cn } from '@/lib/utils'
import {
  editSegment,
  listSegments,
  reExportJob,
  type Segment,
} from '@/api/client'
import { useI18n } from '@/i18n/useI18n'

export function PostEditDrawer({
  jobId,
  onClose,
  onReExportComplete,
}: {
  jobId: string
  onClose: () => void
  onReExportComplete?: () => void
}) {
  const { t } = useI18n()
  const toast = useToast()
  const navigate = useNavigate()

  const [segments, setSegments] = useState<Segment[]>([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState<'all' | 'issues'>('issues')
  const [search, setSearch] = useState('')
  const [edits, setEdits] = useState<Record<string, string>>({})
  const [savingId, setSavingId] = useState<string | null>(null)
  const [reExporting, setReExporting] = useState(false)

  const fetchSegments = useCallback(async () => {
    setLoading(true)
    try {
      const res = await listSegments(jobId, {
        status: filter === 'issues' ? 'issues' : undefined,
        limit: 300,
      })
      setSegments(res.segments)
    } catch {
      toast.push('Failed to load segments', 'error')
    } finally {
      setLoading(false)
    }
  }, [jobId, filter, toast])

  useEffect(() => {
    void fetchSegments()
  }, [fetchSegments])

  const filtered = useMemo(() => {
    if (!search.trim()) return segments
    const q = search.toLowerCase()
    return segments.filter(
      (s) =>
        s.block_id.toLowerCase().includes(q) ||
        s.source_text.toLowerCase().includes(q) ||
        (edits[s.block_id] ?? s.target_text).toLowerCase().includes(q)
    )
  }, [segments, search, edits])

  const handleTextChange = (blockId: string, val: string) => {
    setEdits((prev) => ({ ...prev, [blockId]: val }))
  }

  const handleReset = (blockId: string) => {
    setEdits((prev) => {
      const next = { ...prev }
      delete next[blockId]
      return next
    })
  }

  const handleSave = async (block: Segment) => {
    const textToSave = edits[block.block_id]
    if (textToSave === undefined || textToSave === block.target_text) return

    setSavingId(block.block_id)
    try {
      const res = await editSegment(jobId, block.block_id, textToSave)
      toast.push(t.quality.postEditSavedToast, 'success')
      setSegments((prev) =>
        prev.map((s) =>
          s.block_id === block.block_id
            ? {
                ...s,
                target_text: textToSave,
                delivered_text: textToSave,
                human_verified: true,
                status: 'repaired',
                ...(res.segment ? res.segment : {}),
              }
            : s
        )
      )
      setEdits((prev) => {
        const next = { ...prev }
        delete next[block.block_id]
        return next
      })
    } catch {
      toast.push('Failed to save segment revision', 'error')
    } finally {
      setSavingId(null)
    }
  }

  const handleReExport = async () => {
    setReExporting(true)
    try {
      await reExportJob(jobId)
      toast.push(t.quality.postEditReExportSuccess, 'success')
      onReExportComplete?.()
    } catch {
      toast.push('Re-export failed', 'error')
    } finally {
      setReExporting(false)
    }
  }

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent
        showCloseButton={false}
        className="flex h-[min(92vh,1000px)] max-w-6xl flex-col gap-0 overflow-hidden p-0"
      >
        {/* Header */}
        <DialogHeader className="flex-row items-center justify-between gap-4 space-y-0 border-b border-border/70 px-5 py-3.5">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <DialogTitle className="text-base font-semibold">
                {t.quality.postEditTitle}
              </DialogTitle>
              <Badge variant="secondary" className="font-mono text-xs">
                {t.quality.postEditTotalSegments.replace('{count}', String(segments.length))}
              </Badge>
            </div>
            <DialogDescription className="mt-0.5 truncate text-xs text-muted-foreground">
              {t.quality.postEditSubtitle}
            </DialogDescription>
          </div>

          <div className="flex shrink-0 items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                onClose()
                navigate(`/jobs/${jobId}/review`)
              }}
            >
              <ExternalLink className="size-3.5" />
              <span className="hidden sm:inline">{t.quality.postEditOpenFullWorkbench}</span>
            </Button>
            <Button
              size="sm"
              onClick={() => void handleReExport()}
              disabled={reExporting}
              className="bg-primary text-primary-foreground"
            >
              <RefreshCw className={cn('size-3.5', reExporting && 'animate-spin')} />
              <span>{reExporting ? t.quality.postEditReExporting : t.quality.postEditReExportBtn}</span>
            </Button>
            <DialogClose asChild>
              <Button variant="ghost" size="icon" className="size-8" aria-label="Close">
                <X className="size-4" />
              </Button>
            </DialogClose>
          </div>
        </DialogHeader>

        {/* Toolbar: Filters & Search */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border/70 bg-muted/30 px-5 py-2.5">
          <div className="flex items-center gap-1.5">
            <Button
              variant={filter === 'issues' ? 'secondary' : 'ghost'}
              size="sm"
              onClick={() => setFilter('issues')}
              className="h-8 text-xs font-medium"
            >
              <Filter className="size-3" />
              {t.quality.postEditScopeIssues}
            </Button>
            <Button
              variant={filter === 'all' ? 'secondary' : 'ghost'}
              size="sm"
              onClick={() => setFilter('all')}
              className="h-8 text-xs font-medium"
            >
              {t.quality.postEditScopeAll}
            </Button>
          </div>

          <div className="relative w-full sm:w-72">
            <Search className="absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder={t.quality.postEditSearchPlaceholder}
              className="h-8 pl-8 text-xs"
            />
          </div>
        </div>

        {/* Content Area: Bilingual Rows */}
        <div className="min-h-0 flex-1 overflow-y-auto p-5">
          {loading ? (
            <div className="flex h-64 items-center justify-center gap-2 text-sm text-muted-foreground">
              <LoaderCircle className="size-4 animate-spin" />
              <span>Loading segments…</span>
            </div>
          ) : filtered.length === 0 ? (
            <div className="flex h-64 flex-col items-center justify-center text-center">
              <p className="text-sm font-medium text-foreground">{t.quality.postEditNoSegmentsFound}</p>
              <p className="mt-1 text-xs text-muted-foreground">
                {filter === 'issues'
                  ? 'No outstanding defects or review flags found.'
                  : 'Try adjusting your search criteria.'}
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {filtered.map((seg) => {
                const currentVal = edits[seg.block_id] ?? seg.target_text
                const isDirty = currentVal !== seg.target_text
                const isSaving = savingId === seg.block_id

                return (
                  <Card
                    key={seg.block_id}
                    className={cn(
                      'overflow-hidden border border-border/80 transition-all',
                      isDirty && 'ring-1 ring-primary/40'
                    )}
                  >
                    {/* Segment Meta Ribbon */}
                    <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border/60 bg-muted/40 px-4 py-2 text-xs">
                      <div className="flex items-center gap-2">
                        <span className="font-mono font-medium text-muted-foreground">
                          {seg.page !== null ? `P${seg.page}` : '—'} · #{seg.spine_index + 1}
                        </span>
                        <Badge variant="outline" className="text-[10px] uppercase">
                          {seg.block_type}
                        </Badge>
                        <Badge
                          variant="secondary"
                          className={cn(
                            'text-[10px] font-medium',
                            seg.status === 'needs_human' &&
                              'border-amber-500/30 bg-amber-500/10 text-amber-700 dark:text-amber-400',
                            seg.status === 'mtqe_passed' &&
                              'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400',
                            seg.human_verified &&
                              'border-blue-500/30 bg-blue-500/10 text-blue-700 dark:text-blue-400'
                          )}
                        >
                          {seg.human_verified ? 'Human Edited' : seg.status.toUpperCase()}
                        </Badge>
                        {seg.issues?.map((iss) => (
                          <Badge
                            key={iss}
                            variant="destructive"
                            className="bg-destructive/10 text-[10px] text-destructive"
                          >
                            {iss}
                          </Badge>
                        ))}
                      </div>

                      <div className="flex items-center gap-2 font-mono text-[11px] text-muted-foreground">
                        {seg.mtqe_score !== null && (
                          <span>QE: {Number(seg.mtqe_score).toFixed(2)}</span>
                        )}
                        {isDirty && (
                          <span className="flex items-center gap-1 font-sans text-amber-600 dark:text-amber-400">
                            <span className="size-1.5 rounded-full bg-amber-500 animate-pulse" />
                            {t.quality.postEditDirtyIndicator}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Bilingual Content Comparison */}
                    <div className="grid grid-cols-1 divide-y divide-border/60 md:grid-cols-2 md:divide-x md:divide-y-0">
                      {/* Left: Source */}
                      <div className="flex flex-col bg-muted/15 p-4">
                        <span className="mb-1 text-[11px] font-medium text-muted-foreground">
                          {t.quality.postEditColSource}
                        </span>
                        <div className="flex-1 whitespace-pre-wrap font-sans text-sm leading-relaxed text-foreground select-text">
                          {seg.source_text}
                        </div>
                      </div>

                      {/* Right: Target Editor */}
                      <div className="flex flex-col p-4">
                        <div className="mb-1 flex items-center justify-between">
                          <span className="text-[11px] font-medium text-muted-foreground">
                            {t.quality.postEditColTarget}
                          </span>
                          <span className="font-mono text-[10px] text-muted-foreground">
                            {currentVal.length} chars
                          </span>
                        </div>
                        <Textarea
                          value={currentVal}
                          onChange={(e) => handleTextChange(seg.block_id, e.target.value)}
                          rows={Math.max(3, Math.min(8, Math.ceil(currentVal.length / 45)))}
                          className="min-h-[80px] resize-y font-sans text-sm leading-relaxed"
                        />
                        <div className="mt-2.5 flex items-center justify-end gap-2">
                          {isDirty && (
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleReset(seg.block_id)}
                              className="h-7 text-xs text-muted-foreground"
                            >
                              <RotateCcw className="size-3" />
                              {t.quality.postEditResetBtn}
                            </Button>
                          )}
                          <Button
                            size="sm"
                            onClick={() => void handleSave(seg)}
                            disabled={!isDirty || isSaving}
                            className="h-7 text-xs"
                          >
                            {isSaving ? (
                              <LoaderCircle className="size-3 animate-spin" />
                            ) : (
                              <Save className="size-3" />
                            )}
                            {t.quality.postEditSaveBtn}
                          </Button>
                        </div>
                      </div>
                    </div>
                  </Card>
                )
              })}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
