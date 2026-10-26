import React, { useCallback, useEffect, useRef, useState } from 'react'
import {
  AlertCircle,
  AlertTriangle,
  ArrowDownToLine,
  ChevronLeft,
  ChevronRight,
  FileText,
  FilterX,
  Loader2,
  Maximize2,
  MoreHorizontal,
  Plus,
  Search,
  Trash2,
  Upload,
} from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/Alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { ConfirmDialog } from '@/components/ui/ConfirmDialog'
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from '@/components/ui/empty'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'
import { useToast } from '@/components/ui/useToast'
import { RichText } from '@/components/content/RichText'
import {
  ApiError,
  deleteGlossaryTerm,
  enableGlossary,
  evictMatchingTm,
  evictTm,
  getGlossary,
  getGlossaryConflicts,
  importGlossary,
  importTm,
  listTm,
  upsertGlossaryTerm,
  type GlossaryConflict,
  type GlossaryTerm,
  type TmEntry,
} from '@/api/client'
import { useI18n } from '@/i18n/useI18n'
import { cn } from '@/lib/utils'

type AssetTab = 'glossary' | 'tm'
type TmImportFormat = 'tmx' | 'json'
type TmProvenanceFilter = 'all' | 'machine' | 'human_pe'
type TmUsageFilter = 'all' | 'never' | 'used'
type TmSort = 'newest' | 'oldest' | 'most_used'

const TM_PAGE_SIZES = [20, 50, 100] as const
type TmPageSize = (typeof TM_PAGE_SIZES)[number]

/** A count that stays readable without turning into another chrome badge. */
function AssetCount({ value, label }: { value: number; label: string }) {
  return (
    <span className="tabular-nums text-xs text-muted-foreground">
      <span className="font-medium text-foreground">{value}</span> {label}
    </span>
  )
}

/** Keep long TM passages scannable without hiding the full entry workflow. */
function TmPassage({
  text,
  source,
}: {
  text: string
  source: boolean
}) {
  return (
    <div className="line-clamp-4 w-full text-left">
      <RichText text={text} source={source} density="compact" />
    </div>
  )
}

function ToolbarSearch({
  value,
  onChange,
  placeholder,
}: {
  value: string
  onChange: (value: string) => void
  placeholder: string
}) {
  return (
    <div className="relative w-full sm:w-64">
      <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
      <Input
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        aria-label={placeholder}
        className="h-9 bg-background pl-9"
      />
    </div>
  )
}

function TableLoading({ columns }: { columns: number }) {
  return (
    <>
      {[0, 1, 2].map((row) => (
        <TableRow key={row}>
          {Array.from({ length: columns }, (_, column) => (
            <TableCell key={column} className="py-4">
              <Skeleton className={cn('h-4', column === columns - 1 ? 'ml-auto w-8' : 'w-3/4')} />
            </TableCell>
          ))}
        </TableRow>
      ))}
    </>
  )
}

function StackedLoading({ rows = 3 }: { rows?: number }) {
  return (
    <div className="divide-y divide-border/70 md:hidden" aria-hidden="true">
      {Array.from({ length: rows }, (_, row) => (
        <div key={row} className="space-y-4 px-4 py-5">
          <div className="space-y-2">
            <Skeleton className="h-3 w-12" />
            <Skeleton className="h-4 w-4/5" />
          </div>
          <div className="space-y-2">
            <Skeleton className="h-3 w-10" />
            <Skeleton className="h-4 w-3/5" />
          </div>
        </div>
      ))}
    </div>
  )
}

function AssetEmpty({
  title,
  description,
  action,
}: {
  title: string
  description: string
  action?: React.ReactNode
}) {
  return (
    <div className="border-t border-border/70 px-6 py-12">
      <Empty className="gap-3 p-0">
        <EmptyHeader className="max-w-md">
          <EmptyTitle className="text-base">{title}</EmptyTitle>
          <EmptyDescription>{description}</EmptyDescription>
        </EmptyHeader>
        {action ? <div>{action}</div> : null}
      </Empty>
    </div>
  )
}

function FilterSelect<T extends string>({
  value,
  onValueChange,
  label,
  options,
}: {
  value: T
  onValueChange: (value: T) => void
  label: string
  options: { value: T; label: string }[]
}) {
  return (
    <Select value={value} onValueChange={(next) => onValueChange(next as T)}>
      <SelectTrigger className="h-9 w-full bg-background sm:w-auto" aria-label={label}>
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        {options.map((option) => (
          <SelectItem key={option.value} value={option.value}>
            {option.label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

/** Row actions stay visible: a destructive action must never hide under a hover. */
function RowActions({
  label,
  deleteLabel,
  onDelete,
  inspectLabel,
  onInspect,
}: {
  label: string
  deleteLabel: string
  onDelete: () => void
  inspectLabel?: string
  onInspect?: () => void
}) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="size-8 text-muted-foreground"
          aria-label={label}
        >
          <MoreHorizontal />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-44">
        {inspectLabel && onInspect ? (
          <DropdownMenuItem onSelect={onInspect}>
            <Maximize2 />
            {inspectLabel}
          </DropdownMenuItem>
        ) : null}
        <DropdownMenuItem
          onSelect={onDelete}
          className="text-destructive focus:bg-destructive/10 focus:text-destructive"
        >
          <Trash2 />
          {deleteLabel}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export function LanguageAssetsView() {
  const { language, t } = useI18n()
  const toast = useToast()
  const [activeTab, setActiveTab] = useState<AssetTab>('glossary')

  const [terms, setTerms] = useState<GlossaryTerm[]>([])
  const [conflicts, setConflicts] = useState<GlossaryConflict[]>([])
  const [glossaryConfigured, setGlossaryConfigured] = useState(true)
  const [glossaryLoading, setGlossaryLoading] = useState(true)
  const [glossaryError, setGlossaryError] = useState<string | null>(null)
  const [enablingGlossary, setEnablingGlossary] = useState(false)
  const [glossarySearch, setGlossarySearch] = useState('')
  const [pendingDeleteTerm, setPendingDeleteTerm] = useState<string | null>(null)
  const [deletingTerm, setDeletingTerm] = useState(false)

  const [newSource, setNewSource] = useState('')
  const [newTarget, setNewTarget] = useState('')
  const [savingTerm, setSavingTerm] = useState(false)
  const [glossaryImportOpen, setGlossaryImportOpen] = useState(false)
  const [glossaryImporting, setGlossaryImporting] = useState(false)
  const [glossaryImportFile, setGlossaryImportFile] = useState<File | null>(null)
  const glossaryFileRef = useRef<HTMLInputElement>(null)

  const [tmEntries, setTmEntries] = useState<TmEntry[]>([])
  const [tmTotal, setTmTotal] = useState(0)
  const [tmMachineCount, setTmMachineCount] = useState(0)
  const [tmHumanCount, setTmHumanCount] = useState(0)
  const [tmNeverUsedCount, setTmNeverUsedCount] = useState(0)
  const [tmLoading, setTmLoading] = useState(false)
  const [tmError, setTmError] = useState<string | null>(null)
  const [tmSearch, setTmSearch] = useState('')
  const [tmQuery, setTmQuery] = useState('')
  const [tmProvenance, setTmProvenance] = useState<TmProvenanceFilter>('all')
  const [tmUsage, setTmUsage] = useState<TmUsageFilter>('all')
  const [tmSort, setTmSort] = useState<TmSort>('newest')
  const [tmPage, setTmPage] = useState(1)
  const [tmPageSize, setTmPageSize] = useState<TmPageSize>(20)
  const [tmLoadedOnce, setTmLoadedOnce] = useState(false)
  const [pendingEvict, setPendingEvict] = useState<TmEntry | null>(null)
  const [inspectTm, setInspectTm] = useState<TmEntry | null>(null)
  const [evicting, setEvicting] = useState(false)
  const [selectedTmIds, setSelectedTmIds] = useState<Set<number>>(new Set())
  const [bulkDeleteOpen, setBulkDeleteOpen] = useState(false)
  const [deleteWholeScope, setDeleteWholeScope] = useState(false)
  const [bulkDeleting, setBulkDeleting] = useState(false)

  const [tmImportOpen, setTmImportOpen] = useState(false)
  const [tmImportFormat, setTmImportFormat] = useState<TmImportFormat>('tmx')
  const [tmImportContent, setTmImportContent] = useState('')
  const [tmImportSourceLang, setTmImportSourceLang] = useState('en')
  const [tmImportTargetLang, setTmImportTargetLang] = useState('zh')
  const [tmImporting, setTmImporting] = useState(false)
  const tmFileRef = useRef<HTMLInputElement>(null)

  const loadGlossary = useCallback(async () => {
    setGlossaryLoading(true)
    setGlossaryError(null)
    try {
      const listing = await getGlossary()
      setTerms(listing.terms)
      setGlossaryConfigured(listing.configured)
      setConflicts(
        listing.configured
          ? await getGlossaryConflicts().catch(() => [] as GlossaryConflict[])
          : []
      )
    } catch (err) {
      setGlossaryError(err instanceof Error ? err.message : t.assets.loadGlossaryFailed)
    } finally {
      setGlossaryLoading(false)
    }
  }, [t.assets.loadGlossaryFailed])

  const loadTm = useCallback(async (page = tmPage) => {
    setTmLoading(true)
    setTmError(null)
    try {
      const data = await listTm({
        limit: tmPageSize,
        offset: (page - 1) * tmPageSize,
        query: tmQuery || undefined,
        provenance: tmProvenance === 'all' ? undefined : tmProvenance,
        usage: tmUsage === 'all' ? undefined : tmUsage,
        sort: tmSort,
      })
      setTmEntries(data.entries)
      setTmTotal(data.total)
      setTmMachineCount(data.machine_count)
      setTmHumanCount(data.human_count)
      setTmNeverUsedCount(data.never_used_count)
      setTmLoadedOnce(true)
      setSelectedTmIds(new Set())
      setDeleteWholeScope(false)
      const lastPage = Math.max(1, Math.ceil(data.total / tmPageSize))
      if (page > lastPage) {
        setTmPage(lastPage)
      }
    } catch (err) {
      setTmError(err instanceof Error ? err.message : t.assets.loadTmFailed)
    } finally {
      setTmLoading(false)
    }
  }, [
    t.assets.loadTmFailed,
    tmPage,
    tmPageSize,
    tmProvenance,
    tmQuery,
    tmSort,
    tmUsage,
  ])

  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect
    void loadGlossary()
  }, [loadGlossary])

  useEffect(() => {
    if (activeTab !== 'tm') return
    // oxlint-disable-next-line react/set-state-in-effect
    void loadTm()
  }, [activeTab, loadTm])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setTmQuery(tmSearch.trim())
      setTmPage(1)
    }, 250)
    return () => window.clearTimeout(timer)
  }, [tmSearch])

  const enableSharedGlossary = async () => {
    setEnablingGlossary(true)
    setGlossaryError(null)
    try {
      const created = await enableGlossary()
      setTerms(created)
      setGlossaryConfigured(true)
      setConflicts([])
      toast.push(t.assets.glossaryEnabled, 'success')
    } catch (err) {
      setGlossaryError(err instanceof Error ? err.message : t.assets.enableGlossaryFailed)
    } finally {
      setEnablingGlossary(false)
    }
  }

  const addTerm = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!newSource.trim() || !newTarget.trim() || savingTerm) return
    setSavingTerm(true)
    setGlossaryError(null)
    try {
      setTerms(await upsertGlossaryTerm(newSource.trim(), newTarget.trim()))
      setConflicts(await getGlossaryConflicts().catch(() => conflicts))
      setNewSource('')
      setNewTarget('')
      toast.push(t.assets.termSaved, 'success')
    } catch (err) {
      setGlossaryError(err instanceof Error ? err.message : t.assets.saveTermFailed)
    } finally {
      setSavingTerm(false)
    }
  }

  const deleteTerm = async () => {
    const source = pendingDeleteTerm
    if (!source || deletingTerm) return
    setDeletingTerm(true)
    setGlossaryError(null)
    try {
      setTerms(await deleteGlossaryTerm(source))
      setConflicts(await getGlossaryConflicts().catch(() => conflicts))
      setPendingDeleteTerm(null)
      toast.push(t.assets.termDeleted, 'success')
    } catch (err) {
      setGlossaryError(err instanceof Error ? err.message : t.assets.deleteTermFailed)
    } finally {
      setDeletingTerm(false)
    }
  }

  const importGlossaryFile = async () => {
    const file = glossaryImportFile
    if (!file || glossaryImporting) return
    const extension = file.name.toLowerCase().split('.').pop()
    const format = extension === 'json' ? 'json' : extension === 'tsv' ? 'tsv' : 'csv'
    setGlossaryImporting(true)
    setGlossaryError(null)
    try {
      const result = await importGlossary({ format, content: await file.text() })
      setGlossaryImportOpen(false)
      setGlossaryImportFile(null)
      await loadGlossary()
      toast.push(t.assets.glossaryImportDone.replace('{n}', String(result.imported)), 'success')
    } catch (err) {
      setGlossaryError(err instanceof Error ? err.message : t.assets.importGlossaryFailed)
    } finally {
      setGlossaryImporting(false)
    }
  }

  const importTranslationMemory = async () => {
    if (!tmImportContent.trim() || tmImporting) return
    setTmImporting(true)
    setTmError(null)
    try {
      const result = await importTm({
        format: tmImportFormat,
        content: tmImportContent,
        src_lang: tmImportSourceLang,
        tgt_lang: tmImportTargetLang,
      })
      setTmImportOpen(false)
      setTmImportContent('')
      await loadTm()
      toast.push(t.assets.importDone.replace('{n}', String(result.imported)), 'success')
    } catch (err) {
      setTmError(err instanceof Error ? err.message : t.assets.importTmFailed)
    } finally {
      setTmImporting(false)
    }
  }

  const evictTranslation = async () => {
    const entry = pendingEvict
    if (!entry || evicting) return
    setEvicting(true)
    setTmError(null)
    try {
      await evictTm([entry.id])
      setTmEntries((previous) => previous.filter((candidate) => candidate.id !== entry.id))
      setTmTotal((previous) => Math.max(0, previous - 1))
      setTmMachineCount((previous) =>
        Math.max(0, previous - (entry.provenance === 'machine' ? 1 : 0))
      )
      setTmHumanCount((previous) =>
        Math.max(0, previous - (entry.provenance === 'human_pe' ? 1 : 0))
      )
      setTmNeverUsedCount((previous) =>
        Math.max(0, previous - (entry.use_count === 0 ? 1 : 0))
      )
      setPendingEvict(null)
      toast.push(t.assets.tmDeleted, 'success')
      if (tmEntries.length === 1 && tmPage > 1) {
        setTmPage((previous) => Math.max(1, previous - 1))
      }
    } catch (err) {
      setTmError(err instanceof Error ? err.message : t.assets.deleteTmFailed)
    } finally {
      setEvicting(false)
    }
  }

  const evictSelected = async () => {
    const ids = [...selectedTmIds]
    if ((ids.length === 0 && !deleteWholeScope) || bulkDeleting) return
    setBulkDeleting(true)
    setTmError(null)
    try {
      const removed = deleteWholeScope
        ? await evictMatchingTm({
            expected_total: tmTotal,
            query: tmQuery || undefined,
            provenance: tmProvenance === 'all' ? undefined : tmProvenance,
            usage: tmUsage === 'all' ? undefined : tmUsage,
          })
        : await evictTm(ids)
      setBulkDeleteOpen(false)
      setDeleteWholeScope(false)
      setSelectedTmIds(new Set())
      await loadTm(tmPage)
      toast.push(t.assets.tmDeletedSelected.replace('{count}', String(removed)), 'success')
    } catch (err) {
      if (err instanceof ApiError && err.status === 409) {
        setBulkDeleteOpen(false)
        setDeleteWholeScope(false)
        setSelectedTmIds(new Set())
        setTmError(t.assets.tmScopeChanged)
        await loadTm(tmPage)
      } else {
        setTmError(err instanceof Error ? err.message : t.assets.deleteTmFailed)
      }
    } finally {
      setBulkDeleting(false)
    }
  }

  const readTmFile = async (file: File) => {
    const extension = file.name.toLowerCase().split('.').pop()
    setTmImportFormat(extension === 'json' ? 'json' : 'tmx')
    setTmImportContent(await file.text())
  }

  const normalizedGlossarySearch = glossarySearch.trim().toLocaleLowerCase()
  const filteredTerms = terms.filter((term) => {
    if (!normalizedGlossarySearch) return true
    return (
      term.source.toLocaleLowerCase().includes(normalizedGlossarySearch) ||
      term.target.toLocaleLowerCase().includes(normalizedGlossarySearch)
    )
  })

  const glossaryHasSearch = normalizedGlossarySearch.length > 0
  const tmHasFilters =
    tmQuery.length > 0 || tmProvenance !== 'all' || tmUsage !== 'all'
  const tmPageCount = Math.max(1, Math.ceil(tmTotal / tmPageSize))
  const tmRangeStart = tmTotal === 0 ? 0 : (tmPage - 1) * tmPageSize + 1
  const tmRangeEnd = Math.min(tmPage * tmPageSize, tmTotal)
  const tmPageIds = tmEntries.map((entry) => entry.id)
  const tmPageAllSelected =
    tmPageIds.length > 0 && tmPageIds.every((id) => selectedTmIds.has(id))
  const tmPageSomeSelected =
    !tmPageAllSelected && tmPageIds.some((id) => selectedTmIds.has(id))

  const toggleTmPage = (checked: boolean) => {
    setDeleteWholeScope(false)
    setSelectedTmIds((previous) => {
      const next = new Set(previous)
      for (const id of tmPageIds) {
        if (checked) next.add(id)
        else next.delete(id)
      }
      return next
    })
  }

  const toggleTmEntry = (id: number, checked: boolean) => {
    setDeleteWholeScope(false)
    setSelectedTmIds((previous) => {
      const next = new Set(previous)
      if (checked) next.add(id)
      else next.delete(id)
      return next
    })
  }

  const clearTmFilters = () => {
    setTmSearch('')
    setTmQuery('')
    setTmProvenance('all')
    setTmUsage('all')
    setTmPage(1)
    setDeleteWholeScope(false)
    setSelectedTmIds(new Set())
  }

  const formatTmDate = (value: string) => {
    if (!value) return ''
    const parsed = new Date(`${value.replace(' ', 'T')}Z`)
    if (Number.isNaN(parsed.getTime())) return value
    return new Intl.DateTimeFormat(language === 'zh' ? 'zh-CN' : 'en', {
      dateStyle: 'medium',
    }).format(parsed)
  }

  return (
    <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 overflow-y-auto px-4 py-6 sm:px-7">
      <header className="flex flex-col justify-between gap-4 border-b border-border/80 pb-5 sm:flex-row sm:items-end">
        <div className="space-y-1.5">
          <h1 className="text-2xl font-semibold tracking-[-0.025em] text-foreground">
            {t.assets.title}
          </h1>
          <p className="max-w-2xl text-sm leading-relaxed text-muted-foreground">
            {t.assets.subtitle}
          </p>
        </div>
        <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as AssetTab)}>
          <TabsList className="h-9 w-full sm:w-auto">
            <TabsTrigger value="glossary" className="min-w-0 gap-1.5 px-3">
              <span className="truncate">{t.assets.tabGlossary}</span>
              <span className="shrink-0 tabular-nums text-xs text-muted-foreground">
                {glossaryConfigured ? terms.length : '—'}
              </span>
            </TabsTrigger>
            <TabsTrigger value="tm" className="min-w-0 gap-1.5 px-3">
              <span className="truncate">{t.assets.tabTm}</span>
              <span className="shrink-0 tabular-nums text-xs text-muted-foreground">
                {tmLoadedOnce ? tmTotal : '—'}
              </span>
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </header>

      {activeTab === 'glossary' ? (
        <section className="min-w-0 space-y-4" aria-labelledby="glossary-heading">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <h2 id="glossary-heading" className="text-base font-semibold text-foreground">
                {t.assets.tabGlossary}
              </h2>
              <p className="mt-1 max-w-2xl text-sm leading-relaxed text-muted-foreground">
                {t.assets.glossaryDescription}
              </p>
            </div>
            {glossaryConfigured ? (
              <Button
                variant="outline"
                size="sm"
                onClick={() => setGlossaryImportOpen(true)}
              >
                <Upload />
                {t.assets.importAction}
              </Button>
            ) : null}
          </div>

          {glossaryError ? (
            <Alert variant="destructive">
              <AlertCircle />
              <AlertTitle>{t.assets.actionFailed}</AlertTitle>
              <AlertDescription>{glossaryError}</AlertDescription>
            </Alert>
          ) : null}

          {conflicts.length > 0 ? (
            <Alert variant="warning">
              <AlertTriangle />
              <AlertTitle>{t.assets.conflictsTitle}</AlertTitle>
              <AlertDescription>
                <ul className="space-y-1">
                  {conflicts.map((conflict) => (
                    <li key={conflict.source} className="break-words">
                      <span className="font-medium text-foreground">{conflict.source}</span>
                      <span className="mx-1.5 text-muted-foreground">→</span>
                      <span>{conflict.targets.join(' / ')}</span>
                    </li>
                  ))}
                </ul>
              </AlertDescription>
            </Alert>
          ) : null}

          {!glossaryConfigured ? (
            <div className="rounded-xl border border-dashed border-border bg-card/60 px-6 py-10">
              <Empty className="gap-4 p-0">
                <EmptyHeader className="max-w-md">
                  <EmptyTitle className="text-base">{t.assets.glossaryNotConfiguredTitle}</EmptyTitle>
                  <EmptyDescription>{t.assets.glossaryNotConfigured}</EmptyDescription>
                </EmptyHeader>
                <Button onClick={() => void enableSharedGlossary()} disabled={enablingGlossary}>
                  {enablingGlossary ? <Loader2 className="animate-spin" /> : <Plus />}
                  {enablingGlossary ? t.assets.enablingGlossary : t.assets.enableGlossary}
                </Button>
              </Empty>
            </div>
          ) : (
            <>
              <form
                onSubmit={addTerm}
                className="grid grid-cols-1 gap-2.5 rounded-xl border border-border/80 bg-card p-3 shadow-card md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto]"
              >
                <div className="space-y-1.5">
                  <Label htmlFor="new-term-source" className="text-xs text-muted-foreground">
                    {t.assets.sourceTerm}
                  </Label>
                  <Input
                    id="new-term-source"
                    value={newSource}
                    onChange={(event) => setNewSource(event.target.value)}
                    placeholder={t.assets.sourceTermPlaceholder}
                    className="h-9 bg-background"
                  />
                </div>
                <div className="space-y-1.5">
                  <Label htmlFor="new-term-target" className="text-xs text-muted-foreground">
                    {t.assets.targetTerm}
                  </Label>
                  <Input
                    id="new-term-target"
                    value={newTarget}
                    onChange={(event) => setNewTarget(event.target.value)}
                    placeholder={t.assets.targetTermPlaceholder}
                    className="h-9 bg-background"
                  />
                </div>
                <Button
                  type="submit"
                  className="h-9 self-end"
                  disabled={!newSource.trim() || !newTarget.trim() || savingTerm}
                >
                  {savingTerm ? <Loader2 className="animate-spin" /> : <Plus />}
                  {t.assets.submitTerm}
                </Button>
              </form>

              <div className="overflow-hidden rounded-xl border border-border/80 bg-card shadow-card">
                <div className="flex flex-col gap-3 border-b border-border/70 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
                  <ToolbarSearch
                    value={glossarySearch}
                    onChange={setGlossarySearch}
                    placeholder={t.assets.searchTerms}
                  />
                  <AssetCount
                    value={filteredTerms.length}
                    label={
                      glossaryHasSearch
                        ? t.assets.matchingTerms
                        : t.assets.glossaryEntries
                    }
                  />
                </div>

                {glossaryLoading ? (
                  <>
                    <StackedLoading />
                    <div className="hidden md:block">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>{t.assets.colSource}</TableHead>
                            <TableHead>{t.assets.colTarget}</TableHead>
                            <TableHead className="w-14">
                              <span className="sr-only">{t.assets.colAction}</span>
                            </TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          <TableLoading columns={3} />
                        </TableBody>
                      </Table>
                    </div>
                  </>
                ) : filteredTerms.length === 0 ? (
                  <AssetEmpty
                    title={glossaryHasSearch ? t.assets.noMatchingTerms : t.assets.noTerms}
                    description={
                      glossaryHasSearch
                        ? t.assets.noMatchingTermsDescription
                        : t.assets.noTermsDescription
                    }
                  />
                ) : (
                  <>
                    <ul className="divide-y divide-border/70 md:hidden">
                      {filteredTerms.map((term) => (
                        <li
                          key={term.source}
                          className="flex items-start gap-3 px-4 py-4"
                        >
                          <dl className="min-w-0 flex-1 space-y-3">
                            <div>
                              <dt className="text-xs font-medium text-muted-foreground">
                                {t.assets.colSource}
                              </dt>
                              <dd className="mt-1 break-words font-medium text-foreground">
                                {term.source}
                              </dd>
                            </div>
                            <div>
                              <dt className="text-xs font-medium text-muted-foreground">
                                {t.assets.colTarget}
                              </dt>
                              <dd className="mt-1 break-words text-foreground">{term.target}</dd>
                            </div>
                          </dl>
                          <RowActions
                            label={t.assets.termActions.replace('{term}', term.source)}
                            deleteLabel={t.assets.deleteTermAction}
                            onDelete={() => setPendingDeleteTerm(term.source)}
                          />
                        </li>
                      ))}
                    </ul>
                    <div className="hidden md:block">
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead className="w-[46%]">{t.assets.colSource}</TableHead>
                            <TableHead>{t.assets.colTarget}</TableHead>
                            <TableHead className="w-14">
                              <span className="sr-only">{t.assets.colAction}</span>
                            </TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {filteredTerms.map((term) => (
                            <TableRow key={term.source}>
                              <TableCell className="font-medium text-foreground">
                                <span className="block break-words">{term.source}</span>
                              </TableCell>
                              <TableCell className="text-foreground">
                                <span className="block break-words">{term.target}</span>
                              </TableCell>
                              <TableCell className="text-right">
                                <RowActions
                                  label={t.assets.termActions.replace('{term}', term.source)}
                                  deleteLabel={t.assets.deleteTermAction}
                                  onDelete={() => setPendingDeleteTerm(term.source)}
                                />
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </div>
                  </>
                )}
              </div>
            </>
          )}
        </section>
      ) : (
        <section className="min-w-0 space-y-4" aria-labelledby="tm-heading">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <h2 id="tm-heading" className="text-base font-semibold text-foreground">
                {t.assets.tabTm}
              </h2>
              <p className="mt-1 max-w-2xl text-sm leading-relaxed text-muted-foreground">
                {t.assets.tmDescription}
              </p>
            </div>
            <Button variant="outline" size="sm" onClick={() => setTmImportOpen(true)}>
              <ArrowDownToLine />
              {t.assets.importAction}
            </Button>
          </div>

          {tmError ? (
            <Alert variant="destructive">
              <AlertCircle />
              <AlertTitle>{t.assets.actionFailed}</AlertTitle>
              <AlertDescription>{tmError}</AlertDescription>
            </Alert>
          ) : null}

          <div className="overflow-hidden rounded-xl border border-border/80 bg-card shadow-card">
            <div className="space-y-3 border-b border-border/70 px-4 py-4">
              <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                <ToolbarSearch
                  value={tmSearch}
                  onChange={setTmSearch}
                  placeholder={t.assets.searchTm}
                />
                {tmLoadedOnce ? (
                  <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                    <span className="tabular-nums text-foreground">
                      {t.assets.tmSummary
                        .replace('{machine}', String(tmMachineCount))
                        .replace('{human}', String(tmHumanCount))}
                    </span>
                    <span className="text-border">·</span>
                    {tmNeverUsedCount > 0 ? (
                      <button
                        type="button"
                        className={cn(
                          'rounded-full border border-border/80 px-2 py-0.5 tabular-nums transition-colors hover:border-foreground/30 hover:text-foreground',
                          tmUsage === 'never' &&
                            'border-foreground/30 bg-muted/50 text-foreground'
                        )}
                        aria-pressed={tmUsage === 'never'}
                        onClick={() => {
                          setTmUsage((previous) =>
                            previous === 'never' ? 'all' : 'never'
                          )
                          setTmPage(1)
                        }}
                      >
                        {t.assets.tmNeverUsedSummary.replace(
                          '{count}',
                          String(tmNeverUsedCount)
                        )}
                      </button>
                    ) : null}
                  </div>
                ) : null}
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <FilterSelect
                  value={tmProvenance}
                  onValueChange={(value) => {
                    setTmProvenance(value)
                    setTmPage(1)
                  }}
                  label={t.assets.tmFilterProvenance}
                  options={[
                    { value: 'all', label: `${t.assets.tmFilterProvenance}：${t.assets.tmFilterAll}` },
                    { value: 'machine', label: t.assets.provenanceMachine },
                    { value: 'human_pe', label: t.assets.provenanceHuman },
                  ]}
                />
                <FilterSelect
                  value={tmUsage}
                  onValueChange={(value) => {
                    setTmUsage(value)
                    setTmPage(1)
                  }}
                  label={t.assets.tmFilterUsage}
                  options={[
                    { value: 'all', label: `${t.assets.tmFilterUsage}：${t.assets.tmFilterAll}` },
                    { value: 'never', label: t.assets.tmUsageNever },
                    { value: 'used', label: t.assets.tmUsageUsed },
                  ]}
                />
                <FilterSelect
                  value={tmSort}
                  onValueChange={setTmSort}
                  label={t.assets.tmSort}
                  options={[
                    { value: 'newest', label: t.assets.tmSortNewest },
                    { value: 'oldest', label: t.assets.tmSortOldest },
                    { value: 'most_used', label: t.assets.tmSortMostUsed },
                  ]}
                />
                {tmHasFilters ? (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="justify-start text-muted-foreground sm:justify-center"
                    onClick={clearTmFilters}
                  >
                    <FilterX />
                    {t.assets.tmClearFilters}
                  </Button>
                ) : null}
                {tmLoadedOnce && tmPageIds.length > 0 ? (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="justify-start text-muted-foreground lg:ml-auto"
                    onClick={() => toggleTmPage(!tmPageAllSelected)}
                  >
                    <Checkbox
                      checked={
                        tmPageAllSelected
                          ? true
                          : tmPageSomeSelected
                            ? 'indeterminate'
                            : false
                      }
                      aria-hidden="true"
                      tabIndex={-1}
                    />
                    {tmPageAllSelected
                      ? t.assets.tmSelected.replace(
                          '{count}',
                          String(selectedTmIds.size)
                        )
                      : t.assets.tmSelectPage}
                  </Button>
                ) : null}
              </div>

              {selectedTmIds.size > 0 || deleteWholeScope ? (
                <div className="flex flex-col gap-2 rounded-lg bg-muted/45 px-3 py-2 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
                    <span className="font-medium text-foreground">
                      {deleteWholeScope
                        ? t.assets.tmAllFilteredSelected.replace(
                            '{count}',
                            String(tmTotal)
                          )
                        : t.assets.tmSelected.replace(
                            '{count}',
                            String(selectedTmIds.size)
                          )}
                    </span>
                    {!deleteWholeScope && tmTotal > tmPageIds.length ? (
                      <Button
                        variant="link"
                        size="sm"
                        className="h-auto p-0 text-xs"
                        onClick={() => setDeleteWholeScope(true)}
                      >
                        {t.assets.tmSelectAllFiltered.replace(
                          '{count}',
                          String(tmTotal)
                        )}
                      </Button>
                    ) : null}
                    <Button
                      variant="link"
                      size="sm"
                      className="h-auto p-0 text-xs text-muted-foreground"
                      onClick={() => {
                        setSelectedTmIds(new Set())
                        setDeleteWholeScope(false)
                      }}
                    >
                      {t.assets.tmClearSelection}
                    </Button>
                  </div>
                  <Button
                    variant="destructive"
                    size="sm"
                    onClick={() => setBulkDeleteOpen(true)}
                  >
                    <Trash2 />
                    {t.assets.tmDeleteSelected}
                  </Button>
                </div>
              ) : null}
            </div>

            {tmLoading ? (
              <>
                <StackedLoading />
                <div className="hidden md:block">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>{t.assets.colSource}</TableHead>
                        <TableHead>{t.assets.colTarget}</TableHead>
                        <TableHead className="w-28">{t.assets.colProvenance}</TableHead>
                        <TableHead className="w-14">
                          <span className="sr-only">{t.assets.colAction}</span>
                        </TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      <TableLoading columns={4} />
                    </TableBody>
                  </Table>
                </div>
              </>
            ) : tmEntries.length === 0 ? (
              <AssetEmpty
                title={tmHasFilters ? t.assets.tmFilterNoResults : t.assets.noTm}
                description={
                  tmHasFilters
                    ? t.assets.tmFilterNoResultsDescription
                    : t.assets.noTmDescription
                }
                action={
                  !tmHasFilters ? (
                    <Button variant="outline" size="sm" onClick={() => setTmImportOpen(true)}>
                      <ArrowDownToLine />
                      {t.assets.importAction}
                    </Button>
                  ) : undefined
                }
              />
            ) : (
              <>
                <ul className="divide-y divide-border/70 md:hidden">
                  {tmEntries.map((entry) => (
                    <li key={entry.id} className="px-4 py-4">
                      <div className="mb-3 flex items-center justify-between gap-3">
                        <label className="flex items-center gap-2 text-xs text-muted-foreground">
                          <Checkbox
                            checked={selectedTmIds.has(entry.id)}
                            onCheckedChange={(checked) =>
                              toggleTmEntry(entry.id, checked === true)
                            }
                            aria-label={t.assets.tmSelectEntry}
                          />
                          {t.assets.tmSelectEntry}
                        </label>
                        <span className="tabular-nums text-xs text-muted-foreground">
                          {entry.use_count > 0
                            ? t.assets.tmReuseCount.replace(
                                '{count}',
                                String(entry.use_count)
                              )
                            : t.assets.tmNeverReused}
                        </span>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => setInspectTm(entry)}
                          className="h-8 px-2 text-xs"
                        >
                          <Maximize2 />
                          {t.assets.tmInspectEntry}
                        </Button>
                      </div>
                      <dl className="space-y-3">
                        <div>
                          <dt className="text-xs font-medium text-muted-foreground">
                            {t.assets.colSource}
                          </dt>
                          <dd className="mt-1">
                              <TmPassage
                                text={entry.source_text}
                                source
                              />
                          </dd>
                        </div>
                        <div>
                          <dt className="text-xs font-medium text-muted-foreground">
                            {t.assets.colTarget}
                          </dt>
                          <dd className="mt-1">
                              <TmPassage
                                text={entry.target_text}
                                source={false}
                              />
                          </dd>
                        </div>
                      </dl>
                      <div className="mt-3 flex items-center justify-between gap-3 border-t border-border/60 pt-3">
                        <div className="flex min-w-0 items-center gap-2">
                          <span className="text-xs font-medium text-muted-foreground">
                            {t.assets.colProvenance}
                          </span>
                          <Badge
                            variant="outline"
                            className="font-normal text-muted-foreground"
                          >
                            {entry.provenance === 'human_pe'
                              ? t.assets.provenanceHuman
                              : t.assets.provenanceMachine}
                          </Badge>
                          <span className="text-xs text-muted-foreground">
                            {entry.updated_at
                              ? t.assets.tmUpdatedAt.replace(
                                  '{date}',
                                  formatTmDate(entry.updated_at)
                                )
                              : null}
                          </span>
                        </div>
                        <RowActions
                          label={t.assets.tmRowActions}
                          inspectLabel={t.assets.tmInspectEntry}
                          onInspect={() => setInspectTm(entry)}
                          deleteLabel={t.assets.deleteTmAction}
                          onDelete={() => setPendingEvict(entry)}
                        />
                      </div>
                    </li>
                  ))}
                </ul>
                <div className="hidden md:block">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-10">
                          <Checkbox
                            checked={
                              tmPageAllSelected
                                ? true
                                : tmPageSomeSelected
                                  ? 'indeterminate'
                                  : false
                            }
                            onCheckedChange={(checked) =>
                              toggleTmPage(checked === true)
                            }
                            aria-label={t.assets.tmSelectPage}
                          />
                        </TableHead>
                        <TableHead className="w-[38%]">{t.assets.colSource}</TableHead>
                        <TableHead className="w-[34%]">{t.assets.colTarget}</TableHead>
                        <TableHead className="w-36">{t.assets.colProvenance}</TableHead>
                        <TableHead className="w-14">
                          <span className="sr-only">{t.assets.colAction}</span>
                        </TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {tmEntries.map((entry) => (
                        <TableRow key={entry.id}>
                          <TableCell>
                            <Checkbox
                              checked={selectedTmIds.has(entry.id)}
                              onCheckedChange={(checked) =>
                                toggleTmEntry(entry.id, checked === true)
                              }
                              aria-label={t.assets.tmSelectEntry}
                            />
                          </TableCell>
                          <TableCell className="align-top py-2.5">
                            <TmPassage
                              text={entry.source_text}
                              source
                            />
                          </TableCell>
                          <TableCell className="align-top py-2.5">
                            <TmPassage
                              text={entry.target_text}
                              source={false}
                            />
                          </TableCell>
                          <TableCell className="align-top py-2.5">
                            <div className="space-y-1.5">
                              <Badge
                                variant="outline"
                                className="font-normal text-muted-foreground"
                              >
                                {entry.provenance === 'human_pe'
                                  ? t.assets.provenanceHuman
                                  : t.assets.provenanceMachine}
                              </Badge>
                              <div className="text-xs tabular-nums text-muted-foreground">
                                {entry.use_count > 0
                                  ? t.assets.tmReuseCount.replace(
                                      '{count}',
                                      String(entry.use_count)
                                    )
                                  : t.assets.tmNeverReused}
                              </div>
                              {entry.updated_at ? (
                                <div className="text-xs text-muted-foreground">
                                  {t.assets.tmUpdatedAt.replace(
                                    '{date}',
                                    formatTmDate(entry.updated_at)
                                  )}
                                </div>
                              ) : null}
                            </div>
                          </TableCell>
                          <TableCell className="text-right">
                            <RowActions
                              label={t.assets.tmRowActions}
                              inspectLabel={t.assets.tmInspectEntry}
                              onInspect={() => setInspectTm(entry)}
                              deleteLabel={t.assets.deleteTmAction}
                              onDelete={() => setPendingEvict(entry)}
                            />
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </>
            )}

            {tmLoadedOnce && tmTotal > 0 ? (
              <div className="flex flex-col gap-3 border-t border-border/70 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
                <span className="text-xs tabular-nums text-muted-foreground">
                  {t.assets.tmResultRange
                    .replace('{from}', String(tmRangeStart))
                    .replace('{to}', String(tmRangeEnd))
                    .replace('{total}', String(tmTotal))}
                </span>
                <div className="flex flex-wrap items-center justify-between gap-2 sm:justify-end">
                  <Select
                    value={String(tmPageSize)}
                    onValueChange={(value) => {
                      setTmPageSize(Number(value) as TmPageSize)
                      setTmPage(1)
                    }}
                  >
                    <SelectTrigger
                      className="h-8 w-auto bg-background text-xs"
                      aria-label={t.assets.tmRowsPerPage.replace(
                        '{count}',
                        String(tmPageSize)
                      )}
                    >
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {TM_PAGE_SIZES.map((size) => (
                        <SelectItem key={size} value={String(size)}>
                          {t.assets.tmRowsPerPage.replace('{count}', String(size))}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setTmPage((previous) => Math.max(1, previous - 1))}
                    disabled={tmPage <= 1 || tmLoading}
                  >
                    <ChevronLeft />
                    {t.assets.tmPagePrev}
                  </Button>
                  <span className="min-w-24 text-center text-xs tabular-nums text-muted-foreground">
                    {t.assets.tmPageOf
                      .replace('{page}', String(tmPage))
                      .replace('{pages}', String(tmPageCount))}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() =>
                      setTmPage((previous) => Math.min(tmPageCount, previous + 1))
                    }
                    disabled={tmPage >= tmPageCount || tmLoading}
                  >
                    {t.assets.tmPageNext}
                    <ChevronRight />
                  </Button>
                </div>
              </div>
            ) : null}
          </div>
        </section>
      )}

      <Dialog open={glossaryImportOpen} onOpenChange={setGlossaryImportOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t.assets.glossaryImportTitle}</DialogTitle>
            <DialogDescription>{t.assets.glossaryImportHint}</DialogDescription>
          </DialogHeader>
          <div className="space-y-3">
            <input
              ref={glossaryFileRef}
              type="file"
              accept=".csv,.tsv,.json"
              className="sr-only"
              aria-label={t.assets.glossaryImportTitle}
              onChange={(event) => {
                setGlossaryImportFile(event.target.files?.[0] ?? null)
                event.target.value = ''
              }}
            />
            <button
              type="button"
              onClick={() => glossaryFileRef.current?.click()}
              className="flex w-full items-center gap-3 rounded-lg border border-dashed border-border bg-muted/20 px-4 py-5 text-left transition-colors hover:bg-muted/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-background text-muted-foreground">
                <FileText className="size-4" />
              </span>
              <span className="min-w-0">
                <span className="block truncate text-sm font-medium text-foreground">
                  {glossaryImportFile?.name ?? t.assets.glossaryChooseFile}
                </span>
                <span className="mt-0.5 block text-xs text-muted-foreground">
                  {glossaryImportFile
                    ? t.assets.glossaryReplaceFile
                    : t.assets.glossaryFileFormats}
                </span>
              </span>
            </button>
          </div>
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="ghost">{t.common.cancel}</Button>
            </DialogClose>
            <Button
              onClick={() => void importGlossaryFile()}
              disabled={!glossaryImportFile || glossaryImporting}
            >
              {glossaryImporting ? <Loader2 className="animate-spin" /> : <Upload />}
              {glossaryImporting ? t.assets.importing : t.assets.importAction}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={tmImportOpen} onOpenChange={setTmImportOpen}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] max-w-xl overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{t.assets.importTitle}</DialogTitle>
            <DialogDescription>{t.assets.importTmDescription}</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <input
              ref={tmFileRef}
              type="file"
              accept=".tmx,.xml,.json"
              className="sr-only"
              aria-label={t.assets.tmChooseFile}
              onChange={(event) => {
                const file = event.target.files?.[0]
                if (file) void readTmFile(file)
                event.target.value = ''
              }}
            />
            <div className="grid grid-cols-2 items-end gap-3">
              <div className="space-y-1.5">
                <Label className="text-xs text-muted-foreground">
                  {t.assets.tmChooseFile}
                </Label>
                <Button
                  variant="outline"
                  className="h-9 w-full justify-start bg-background"
                  onClick={() => tmFileRef.current?.click()}
                >
                  <FileText />
                  {t.assets.tmChooseFile}
                </Button>
              </div>
              <div className="space-y-1.5">
                <Label htmlFor="tm-import-format" className="text-xs text-muted-foreground">
                  {t.assets.importFormat}
                </Label>
                <Select
                  value={tmImportFormat}
                  onValueChange={(value) => setTmImportFormat(value as TmImportFormat)}
                >
                  <SelectTrigger id="tm-import-format" className="h-9 w-full bg-background">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="tmx">TMX</SelectItem>
                    <SelectItem value="json">JSON</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
            <div className="grid grid-cols-[1fr_auto_1fr] items-end gap-2">
              <div className="space-y-1.5">
                <Label htmlFor="tm-source-lang" className="text-xs text-muted-foreground">
                  {t.assets.importSourceLang}
                </Label>
                <Input
                  id="tm-source-lang"
                  value={tmImportSourceLang}
                  onChange={(event) => setTmImportSourceLang(event.target.value)}
                  className="font-mono"
                />
              </div>
              <span className="pb-2.5 text-sm text-muted-foreground">→</span>
              <div className="space-y-1.5">
                <Label htmlFor="tm-target-lang" className="text-xs text-muted-foreground">
                  {t.assets.importTargetLang}
                </Label>
                <Input
                  id="tm-target-lang"
                  value={tmImportTargetLang}
                  onChange={(event) => setTmImportTargetLang(event.target.value)}
                  className="font-mono"
                />
              </div>
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="tm-import-content" className="text-xs text-muted-foreground">
                {t.assets.importContent}
              </Label>
              <Textarea
                id="tm-import-content"
                value={tmImportContent}
                onChange={(event) => setTmImportContent(event.target.value)}
                placeholder={t.assets.importContentPlaceholder}
                rows={8}
                className="resize-y font-mono text-xs"
              />
            </div>
          </div>
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="ghost">{t.common.cancel}</Button>
            </DialogClose>
            <Button
              onClick={() => void importTranslationMemory()}
              disabled={!tmImportContent.trim() || tmImporting}
            >
              {tmImporting ? <Loader2 className="animate-spin" /> : <ArrowDownToLine />}
              {tmImporting ? t.assets.importing : t.assets.importBtn}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <ConfirmDialog
        open={pendingDeleteTerm !== null}
        title={t.assets.deleteTermTitle}
        body={t.assets.deleteTermBody.replace('{term}', pendingDeleteTerm ?? '')}
        confirmLabel={t.assets.deleteTermAction}
        cancelLabel={t.common.cancel}
        danger
        onConfirm={() => void deleteTerm()}
        onCancel={() => setPendingDeleteTerm(null)}
      />
      <ConfirmDialog
        open={pendingEvict !== null}
        title={t.assets.evictTmTitle}
        body={
          <div className="space-y-2">
            <p>{t.assets.evictTmBody}</p>
            {pendingEvict ? (
              <p className="rounded-md bg-muted/50 px-3 py-2 text-xs text-foreground">
                {pendingEvict.source_text}
              </p>
            ) : null}
          </div>
        }
        confirmLabel={t.assets.deleteTmAction}
        cancelLabel={t.common.cancel}
        danger
        onConfirm={() => void evictTranslation()}
        onCancel={() => setPendingEvict(null)}
      />
      <ConfirmDialog
        open={bulkDeleteOpen}
        title={
          deleteWholeScope
            ? t.assets.tmDeleteScopeTitle
            : t.assets.tmDeleteSelectedTitle
        }
        body={(deleteWholeScope
          ? t.assets.tmDeleteScopeBody
          : t.assets.tmDeleteSelectedBody
        ).replace('{count}', String(deleteWholeScope ? tmTotal : selectedTmIds.size))}
        confirmLabel={(deleteWholeScope
          ? t.assets.tmDeleteScopeConfirm
          : t.assets.tmDeleteSelectedConfirm
        ).replace('{count}', String(deleteWholeScope ? tmTotal : selectedTmIds.size))}
        cancelLabel={t.common.cancel}
        danger
        onConfirm={() => void evictSelected()}
        onCancel={() => {
          if (!bulkDeleting) setBulkDeleteOpen(false)
        }}
      />
      <Dialog open={inspectTm !== null} onOpenChange={(open) => !open && setInspectTm(null)}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] max-w-3xl overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{t.assets.tmInspectTitle}</DialogTitle>
            <DialogDescription>{t.assets.tmInspectDescription}</DialogDescription>
          </DialogHeader>
          {inspectTm ? (
            <div className="space-y-4">
              <section className="rounded-lg border border-border/70 bg-card">
                <h3 className="px-4 pt-3 text-xs font-medium text-muted-foreground">
                  {t.assets.colSource}
                </h3>
                <div className="px-4 pb-4 text-sm leading-7 text-foreground">
                  <RichText text={inspectTm.source_text} source />
                </div>
              </section>
              <section className="rounded-lg border border-border/70 bg-card">
                <h3 className="px-4 pt-3 text-xs font-medium text-muted-foreground">
                  {t.assets.colTarget}
                </h3>
                <div className="px-4 pb-4 text-sm leading-7 text-foreground">
                  <RichText text={inspectTm.target_text} />
                </div>
              </section>
              <div className="flex flex-wrap items-center gap-x-3 gap-y-2 border-t border-border/70 pt-4 text-xs text-muted-foreground">
                <Badge variant="outline" className="font-normal text-muted-foreground">
                  {inspectTm.provenance === 'human_pe'
                    ? t.assets.provenanceHuman
                    : t.assets.provenanceMachine}
                </Badge>
                <span>
                  {inspectTm.use_count > 0
                    ? t.assets.tmReuseCount.replace(
                        '{count}',
                        String(inspectTm.use_count)
                      )
                    : t.assets.tmNeverReused}
                </span>
                {inspectTm.updated_at ? (
                  <span>
                    {t.assets.tmUpdatedAt.replace(
                      '{date}',
                      formatTmDate(inspectTm.updated_at)
                    )}
                  </span>
                ) : null}
              </div>
            </div>
          ) : null}
          <DialogFooter>
            <DialogClose asChild>
              <Button variant="outline">{t.common.close}</Button>
            </DialogClose>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
