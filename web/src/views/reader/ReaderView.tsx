import React, { useEffect, useState, useRef } from 'react'
import {
  Upload,
  BookOpen,
  FileText,
  Download,
  ArrowLeft,
  CheckCircle2,
  Clock,
  Sparkles,
  Edit3,
  Check,
  ChevronDown,
  Image as ImageIcon,
} from 'lucide-react'
import {
  listJobs,
  getJobStatus,
  listSegments,
  editSegment,
  uploadSourceDocument,
  submitJob,
  listDeliverables,
  type JobSummary,
  type Segment,
  type DeliverableListing,
} from '@/api/client'
import { useI18n } from '@/i18n/useI18n'
import { useToast } from '@/components/ui/useToast'

export function ReaderView({ initialJobId }: { initialJobId?: string }) {
  const { t, language } = useI18n()
  const { push: showToast } = useToast()

  // Navigation / View State
  const [activeJobId, setActiveJobId] = useState<string | null>(initialJobId ?? null)
  const [jobs, setJobs] = useState<JobSummary[]>([])
  const [loadingJobs, setLoadingJobs] = useState(false)

  // Upload & Translation State
  const [file, setFile] = useState<File | null>(null)
  const [manualPath, setManualPath] = useState('')
  const [targetLang, setTargetLang] = useState('zh-Hans')
  const [stylePreset, setStylePreset] = useState<'academic' | 'literary' | 'technical'>('academic')
  const [submitting, setSubmitting] = useState(false)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  // Active Job Reading State
  const [segments, setSegments] = useState<Segment[]>([])
  const [loadingSegments, setLoadingSegments] = useState(false)
  const [deliverables, setDeliverables] = useState<DeliverableListing | null>(null)
  const [bilingualView, setBilingualView] = useState<'bilingual' | 'target_only' | 'source_only'>('bilingual')
  const [editingBlockId, setEditingBlockId] = useState<string | null>(null)
  const [draftText, setDraftText] = useState('')
  const [savingEdit, setSavingEdit] = useState(false)
  const [exportMenuOpen, setExportMenuOpen] = useState(false)

  // Active Job Progress
  const [activeJobStatus, setActiveJobStatus] = useState<string>('unknown')
  const [activeJobProgress, setActiveJobProgress] = useState<{
    percent: number
    stage: string
    current: number
    total: number
  }>({ percent: 0, stage: '', current: 0, total: 0 })

  // Refresh recent jobs
  const refreshJobs = async () => {
    setLoadingJobs(true)
    try {
      const list = await listJobs(20)
      setJobs(list)
    } catch {
      // Ignore initial failure if server offline
    } finally {
      setLoadingJobs(false)
    }
  }

  useEffect(() => {
    refreshJobs()
  }, [])

  // When active job changes, load status & segments
  useEffect(() => {
    if (!activeJobId) {
      setSegments([])
      setDeliverables(null)
      return
    }

    let isMounted = true

    const loadJobData = async () => {
      setLoadingSegments(true)
      try {
        const [statusRes, segsRes, delivRes] = await Promise.allSettled([
          getJobStatus(activeJobId),
          listSegments(activeJobId, { limit: 500 }),
          listDeliverables(activeJobId),
        ])

        if (!isMounted) return

        if (statusRes.status === 'fulfilled') {
          setActiveJobStatus(statusRes.value.status)
          const total = statusRes.value.total_blocks ?? 0
          if (total > 0) {
            const completed = statusRes.value.completed_blocks ?? 0
            const pct = Math.round((completed / total) * 100)
            setActiveJobProgress({
              percent: pct,
              stage: statusRes.value.stage ?? '',
              current: completed,
              total,
            })
          }
        }

        if (segsRes.status === 'fulfilled') {
          setSegments(segsRes.value.segments)
        }

        if (delivRes.status === 'fulfilled') {
          setDeliverables(delivRes.value)
        }
      } catch {
        showToast('加载书籍内容失败', 'error')
      } finally {
        if (isMounted) setLoadingSegments(false)
      }
    }

    loadJobData()

    // Poll if job is running
    const interval = setInterval(() => {
      if (activeJobStatus === 'running' || activeJobStatus === 'pending') {
        loadJobData()
      }
    }, 3000)

    return () => {
      isMounted = false
      clearInterval(interval)
    }
  }, [activeJobId, activeJobStatus])

  // Handle file select
  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault()
    if (e.dataTransfer.files?.[0]) {
      setFile(e.dataTransfer.files[0])
      setManualPath('')
    }
  }

  // Handle start translation
  const handleStartTranslation = async () => {
    if (!file && !manualPath.trim()) {
      showToast('请先选择或输入要翻译的文件', 'error')
      return
    }

    setSubmitting(true)
    try {
      let resolvedPath = manualPath.trim()

      if (file) {
        showToast('正在上传源文件...', 'info')
        const uploadRes = await uploadSourceDocument(file)
        resolvedPath = uploadRes.file_path
      }

      const presetMap: Record<'academic' | 'literary' | 'technical', 'high_fidelity' | 'standard' | 'fast'> = {
        academic: 'high_fidelity',
        literary: 'standard',
        technical: 'fast',
      }

      showToast('正在启动翻译任务...', 'info')
      const submitRes = await submitJob({
        input_path: resolvedPath,
        target_lang: targetLang,
        source_lang: 'auto',
        profile: 'general',
        preset: presetMap[stylePreset],
        priority: 0,
        dry_run: false,
      })

      showToast('任务已启动，正在解析排版！', 'success')
      setActiveJobId(submitRes.job_id)
      setActiveJobStatus('running')
      refreshJobs()
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      showToast(`启动失败: ${msg}`, 'error')
    } finally {
      setSubmitting(false)
    }
  }

  // Handle saving paragraph edit
  const handleSaveParagraph = async (blockId: string) => {
    if (!activeJobId) return
    setSavingEdit(true)
    try {
      await editSegment(activeJobId, blockId, draftText)
      setSegments((prev) =>
        prev.map((s) => (s.block_id === blockId ? { ...s, target_text: draftText } : s))
      )
      setEditingBlockId(null)
      showToast('修改已保存并同步', 'success')
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      showToast(`保存失败: ${msg}`, 'error')
    } finally {
      setSavingEdit(false)
    }
  }

  // Handle deliverable download
  const handleDownload = (format?: string) => {
    if (!activeJobId) return
    const url = format
      ? `/api/v1/jobs/${activeJobId}/deliverable?format=${encodeURIComponent(format)}`
      : `/api/v1/jobs/${activeJobId}/deliverable`
    window.open(url, '_blank')
    setExportMenuOpen(false)
  }

  // Active book name helper
  const getActiveBookTitle = () => {
    const job = jobs.find((j) => j.job_id === activeJobId)
    if (job?.file_name) {
      const raw = job.file_name.split(/[/\\]/).pop() ?? job.file_name
      return raw.replace(/^\d{8}-\d{6}-[a-f0-9]{8}-/, '')
    }
    return activeJobId ?? 'Book'
  }

  // --------------------------------------------------------------------------
  // RENDER: Immersive Book Reading View
  // --------------------------------------------------------------------------
  if (activeJobId) {
    const isRunning = activeJobStatus === 'running' || activeJobStatus === 'pending'

    return (
      <div className="max-w-5xl mx-auto px-4 py-6 md:py-8 space-y-6">
        {/* Top Sticky Header */}
        <div className="sticky top-0 z-20 backdrop-blur-md bg-[var(--paper-surface)] border-b border-[var(--paper-border)] pb-4 pt-2 -mt-2 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setActiveJobId(null)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold border border-[var(--paper-border)] hover:bg-[var(--paper-subsurface)] text-[var(--ink-secondary)] transition-colors cursor-pointer"
            >
              <ArrowLeft className="h-4 w-4" />
              <span>{t.reader.backToShelf}</span>
            </button>
            <div className="truncate max-w-[260px] md:max-w-md">
              <h2 className="text-sm md:text-base font-bold text-[var(--ink-primary)] truncate">
                {getActiveBookTitle()}
              </h2>
            </div>
            {isRunning ? (
              <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-500/20 animate-pulse">
                <Clock className="h-3 w-3" />
                {t.reader.translating}
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border border-emerald-500/20">
                <CheckCircle2 className="h-3 w-3" />
                {t.reader.stepReady}
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            {/* View Mode Toggle */}
            <div className="inline-flex p-0.5 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-subsurface)] text-xs font-medium">
              <button
                onClick={() => setBilingualView('bilingual')}
                className={`px-2.5 py-1 rounded-md transition-all cursor-pointer ${
                  bilingualView === 'bilingual'
                    ? 'bg-[var(--paper-surface)] text-[var(--ink-primary)] shadow-2xs font-semibold'
                    : 'text-[var(--ink-muted)] hover:text-[var(--ink-primary)]'
                }`}
              >
                {t.reader.bilingualMode}
              </button>
              <button
                onClick={() => setBilingualView('target_only')}
                className={`px-2.5 py-1 rounded-md transition-all cursor-pointer ${
                  bilingualView === 'target_only'
                    ? 'bg-[var(--paper-surface)] text-[var(--ink-primary)] shadow-2xs font-semibold'
                    : 'text-[var(--ink-muted)] hover:text-[var(--ink-primary)]'
                }`}
              >
                {t.reader.targetOnlyMode}
              </button>
              <button
                onClick={() => setBilingualView('source_only')}
                className={`px-2.5 py-1 rounded-md transition-all cursor-pointer ${
                  bilingualView === 'source_only'
                    ? 'bg-[var(--paper-surface)] text-[var(--ink-primary)] shadow-2xs font-semibold'
                    : 'text-[var(--ink-muted)] hover:text-[var(--ink-primary)]'
                }`}
              >
                {t.reader.sourceOnlyMode}
              </button>
            </div>

            {/* Export Dropdown */}
            <div className="relative">
              <button
                onClick={() => setExportMenuOpen(!exportMenuOpen)}
                className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-[var(--ink-primary)] text-[var(--paper-bg)] hover:opacity-90 shadow-xs transition-opacity cursor-pointer"
              >
                <Download className="h-3.5 w-3.5" />
                <span>{t.reader.exportDeliverable}</span>
                <ChevronDown className="h-3 w-3 opacity-70" />
              </button>

              {exportMenuOpen && (
                <div
                  className="absolute right-0 mt-1 w-44 rounded-lg shadow-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] py-1 z-30 animate-in fade-in zoom-in-95 duration-100"
                  onClick={() => setExportMenuOpen(false)}
                >
                  <button
                    onClick={() => handleDownload()}
                    className="w-full text-left px-3 py-2 text-xs text-[var(--ink-primary)] hover:bg-[var(--paper-subsurface)] font-medium flex items-center justify-between"
                  >
                    <span>默认完整版</span>
                    <span className="text-[10px] text-[var(--ink-muted)] uppercase">推荐</span>
                  </button>
                  <div className="border-t border-[var(--paper-border)] my-1" />
                  <button
                    onClick={() => handleDownload('pdf')}
                    className="w-full text-left px-3 py-1.5 text-xs text-[var(--ink-primary)] hover:bg-[var(--paper-subsurface)] flex items-center justify-between"
                  >
                    <span>PDF 排版电子书</span>
                    <span className="text-[10px] text-[var(--ink-muted)] font-mono">.pdf</span>
                  </button>
                  <button
                    onClick={() => handleDownload('epub')}
                    className="w-full text-left px-3 py-1.5 text-xs text-[var(--ink-primary)] hover:bg-[var(--paper-subsurface)] flex items-center justify-between"
                  >
                    <span>EPUB 移动电子书</span>
                    <span className="text-[10px] text-[var(--ink-muted)] font-mono">.epub</span>
                  </button>
                  <button
                    onClick={() => handleDownload('docx')}
                    className="w-full text-left px-3 py-1.5 text-xs text-[var(--ink-primary)] hover:bg-[var(--paper-subsurface)] flex items-center justify-between"
                  >
                    <span>Word 交付文档</span>
                    <span className="text-[10px] text-[var(--ink-muted)] font-mono">.docx</span>
                  </button>
                  <button
                    onClick={() => handleDownload('markdown')}
                    className="w-full text-left px-3 py-1.5 text-xs text-[var(--ink-primary)] hover:bg-[var(--paper-subsurface)] flex items-center justify-between"
                  >
                    <span>纯文本 Markdown</span>
                    <span className="text-[10px] text-[var(--ink-muted)] font-mono">.md</span>
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Translation In-Progress Progress Banner */}
        {isRunning && (
          <div className="p-4 md:p-6 rounded-xl border border-amber-500/20 bg-amber-500/5 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-[var(--ink-primary)] flex items-center gap-2">
                  <Sparkles className="h-4 w-4 text-amber-600 dark:text-amber-400" />
                  {t.reader.translating}
                </h3>
                {t.reader.translatingDesc ? (
                  <p className="text-xs text-[var(--ink-muted)] mt-0.5">{t.reader.translatingDesc}</p>
                ) : null}
              </div>
              <div className="text-right">
                <span className="text-lg font-bold font-mono text-[var(--ink-primary)]">
                  {activeJobProgress.percent}%
                </span>
                {activeJobProgress.total > 0 && (
                  <div className="text-[11px] text-[var(--ink-muted)] font-mono">
                    {activeJobProgress.current} / {activeJobProgress.total} 段
                  </div>
                )}
              </div>
            </div>

            {/* Visual Step Progress */}
            <div className="grid grid-cols-4 gap-2 text-center text-xs">
              <div
                className={`py-2 px-1 rounded-lg border font-medium ${
                  activeJobProgress.percent >= 10
                    ? 'bg-[var(--paper-surface)] text-[var(--ink-primary)] border-emerald-500/40'
                    : 'text-[var(--ink-muted)] border-transparent'
                }`}
              >
                ① {t.reader.stepParse}
              </div>
              <div
                className={`py-2 px-1 rounded-lg border font-medium ${
                  activeJobProgress.percent >= 25
                    ? 'bg-[var(--paper-surface)] text-[var(--ink-primary)] border-emerald-500/40'
                    : 'text-[var(--ink-muted)] border-transparent'
                }`}
              >
                ② {t.reader.stepTranslate}
              </div>
              <div
                className={`py-2 px-1 rounded-lg border font-medium ${
                  activeJobProgress.percent >= 85
                    ? 'bg-[var(--paper-surface)] text-[var(--ink-primary)] border-emerald-500/40'
                    : 'text-[var(--ink-muted)] border-transparent'
                }`}
              >
                ③ {t.reader.stepRender}
              </div>
              <div
                className={`py-2 px-1 rounded-lg border font-medium ${
                  activeJobProgress.percent >= 100
                    ? 'bg-[var(--paper-surface)] text-[var(--ink-primary)] border-emerald-500/40'
                    : 'text-[var(--ink-muted)] border-transparent'
                }`}
              >
                ④ {t.reader.stepReady}
              </div>
            </div>

            {/* Progress Track */}
            <div className="h-1.5 w-full bg-[var(--paper-subsurface)] rounded-full overflow-hidden">
              <div
                className="h-full bg-amber-500 transition-all duration-500 rounded-full"
                style={{ width: `${Math.max(activeJobProgress.percent, 8)}%` }}
              />
            </div>
          </div>
        )}

        {/* Reading Flow Body */}
        <div className="space-y-4">
          {loadingSegments && segments.length === 0 ? (
            <div className="py-20 text-center space-y-3">
              <div className="h-6 w-6 border-2 border-[var(--paper-border)] border-t-[var(--ink-primary)] rounded-full animate-spin mx-auto" />
              <p className="text-xs text-[var(--ink-muted)]">正在加载书籍段落...</p>
            </div>
          ) : segments.length === 0 ? (
            <div className="py-20 text-center rounded-xl border border-[var(--paper-border)] bg-[var(--paper-subsurface)]">
              <p className="text-sm text-[var(--ink-secondary)]">该任务正在解析或暂未提取到文本段落</p>
            </div>
          ) : (
            <div className="space-y-4 pb-36">
              {segments.map((seg) => {
                const isEditing = editingBlockId === seg.block_id

                // Detect raw image asset paths or image block types
                const isImage =
                  seg.block_type === 'image' ||
                  seg.source_text.includes('/docling_cache/assets/') ||
                  (seg.source_text.startsWith('/') &&
                    /\.(png|jpg|jpeg|webp|svg|gif)$/i.test(seg.source_text))

                if (isImage) {
                  return (
                    <div
                      key={seg.block_id}
                      className="my-3 py-3.5 px-4 rounded-xl border border-dashed border-[var(--paper-border)] bg-[var(--paper-subsurface)] flex items-center justify-center gap-2.5 text-xs text-[var(--ink-muted)] select-none"
                    >
                      <ImageIcon className="h-4 w-4 text-[var(--ink-secondary)] shrink-0" />
                      <span className="font-medium text-[var(--ink-secondary)]">原书插图 / 视觉图表</span>
                      {seg.page !== null && (
                        <span className="text-[10px] font-mono text-[var(--ink-muted)]">
                          · P{seg.page}
                        </span>
                      )}
                    </div>
                  )
                }

                const isHeading = seg.block_type === 'heading'

                return (
                  <div
                    key={seg.block_id}
                    className={`rounded-xl border border-transparent hover:border-[var(--paper-border)] p-4 transition-all duration-150 ${
                      isEditing
                        ? 'bg-[var(--paper-surface)] shadow-md border-[var(--paper-border-hover)]'
                        : isHeading
                          ? 'bg-[var(--paper-subsurface)] font-semibold'
                          : 'hover:bg-[var(--paper-subsurface)]'
                    }`}
                  >
                    {bilingualView === 'bilingual' ? (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-8 items-start">
                        {/* Source column */}
                        <div
                          className={`min-w-0 break-words [overflow-wrap:anywhere] select-text ${
                            isHeading
                              ? 'text-sm md:text-base font-bold font-serif leading-snug text-[var(--ink-primary)]'
                              : 'text-xs md:text-sm font-serif leading-relaxed text-[var(--ink-secondary)] opacity-90'
                          }`}
                        >
                          {seg.source_text}
                        </div>

                        {/* Target column */}
                        <div
                          className={`min-w-0 break-words [overflow-wrap:anywhere] ${
                            isHeading
                              ? 'text-sm md:text-base font-bold font-sans leading-snug text-[var(--ink-primary)]'
                              : 'text-xs md:text-sm font-sans leading-relaxed text-[var(--ink-primary)]'
                          }`}
                        >
                          {isEditing ? (
                            <div className="space-y-2">
                              <textarea
                                autoFocus
                                value={draftText}
                                onChange={(e) => setDraftText(e.target.value)}
                                onKeyDown={(e) => {
                                  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                                    e.preventDefault()
                                    handleSaveParagraph(seg.block_id)
                                  } else if (e.key === 'Escape') {
                                    setEditingBlockId(null)
                                  }
                                }}
                                rows={Math.max(3, draftText.split('\n').length + 1)}
                                className="w-full text-xs md:text-sm p-3 rounded-lg border border-[var(--paper-border-hover)] bg-[var(--paper-bg)] text-[var(--ink-primary)] focus:outline-hidden focus:ring-1 focus:ring-[var(--ink-primary)] leading-relaxed resize-none"
                              />
                              <div className="flex items-center gap-2 justify-end">
                                <button
                                  onClick={() => setEditingBlockId(null)}
                                  className="px-2.5 py-1 text-xs text-[var(--ink-muted)] hover:text-[var(--ink-primary)] rounded cursor-pointer"
                                >
                                  {t.reader.cancelEdit}
                                </button>
                                <button
                                  disabled={savingEdit}
                                  onClick={() => handleSaveParagraph(seg.block_id)}
                                  className="inline-flex items-center gap-1 px-3 py-1 text-xs font-medium rounded bg-[var(--ink-primary)] text-[var(--paper-bg)] hover:opacity-90 transition-opacity cursor-pointer"
                                >
                                  <Check className="h-3 w-3" />
                                  <span>{t.reader.saveEdit}</span>
                                </button>
                              </div>
                            </div>
                          ) : (
                            <div
                              onClick={() => {
                                setEditingBlockId(seg.block_id)
                                setDraftText(seg.target_text || '')
                              }}
                              className="group relative cursor-pointer min-h-[24px]"
                              title={t.reader.editParagraph}
                            >
                              <div>{seg.target_text || <span className="italic text-[var(--ink-muted)]">（等待翻译...）</span>}</div>
                              <span className="hidden group-hover:inline-flex items-center gap-0.5 text-[10px] text-[var(--ink-muted)] mt-1 opacity-60">
                                <Edit3 className="h-2.5 w-2.5" />
                                <span>{t.reader.editParagraph}</span>
                              </span>
                            </div>
                          )}
                        </div>
                      </div>
                    ) : bilingualView === 'target_only' ? (
                      <div className="text-sm md:text-base font-sans leading-relaxed text-[var(--ink-primary)] max-w-2xl mx-auto min-w-0 break-words [overflow-wrap:anywhere]">
                        {isEditing ? (
                          <div className="space-y-2">
                            <textarea
                              autoFocus
                              value={draftText}
                              onChange={(e) => setDraftText(e.target.value)}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
                                  e.preventDefault()
                                  handleSaveParagraph(seg.block_id)
                                } else if (e.key === 'Escape') {
                                  setEditingBlockId(null)
                                }
                              }}
                              rows={Math.max(3, draftText.split('\n').length + 1)}
                              className="w-full text-sm p-3 rounded-lg border border-[var(--paper-border-hover)] bg-[var(--paper-bg)] text-[var(--ink-primary)] focus:outline-hidden focus:ring-1 focus:ring-[var(--ink-primary)] leading-relaxed resize-none"
                            />
                            <div className="flex items-center gap-2 justify-end">
                              <button
                                onClick={() => setEditingBlockId(null)}
                                className="px-2.5 py-1 text-xs text-[var(--ink-muted)] hover:text-[var(--ink-primary)] rounded cursor-pointer"
                              >
                                {t.reader.cancelEdit}
                              </button>
                              <button
                                disabled={savingEdit}
                                onClick={() => handleSaveParagraph(seg.block_id)}
                                className="inline-flex items-center gap-1 px-3 py-1 text-xs font-medium rounded bg-[var(--ink-primary)] text-[var(--paper-bg)] cursor-pointer"
                              >
                                <Check className="h-3 w-3" />
                                <span>{t.reader.saveEdit}</span>
                              </button>
                            </div>
                          </div>
                        ) : (
                          <div
                            onClick={() => {
                              setEditingBlockId(seg.block_id)
                              setDraftText(seg.target_text || '')
                            }}
                            className="cursor-pointer"
                          >
                            {seg.target_text || <span className="italic text-[var(--ink-muted)]">（等待翻译...）</span>}
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="text-sm md:text-base font-serif leading-relaxed text-[var(--ink-secondary)] max-w-2xl mx-auto select-text min-w-0 break-words [overflow-wrap:anywhere]">
                        {seg.source_text}
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    )
  }

  // --------------------------------------------------------------------------
  // RENDER: Reader Home / Bookshelf & Minimalist Upload
  // --------------------------------------------------------------------------
  return (
    <div className="max-w-4xl mx-auto px-4 py-8 md:py-12 space-y-12">
      {/* Hero Header */}
      <div className="text-center space-y-2">
        <h1 className="text-2xl md:text-3xl font-extrabold tracking-tight text-[var(--ink-primary)] font-serif">
          {t.reader.title}
        </h1>
        {t.reader.subtitle ? (
          <p className="text-xs md:text-sm text-[var(--ink-secondary)] max-w-xl mx-auto leading-relaxed">
            {t.reader.subtitle}
          </p>
        ) : null}
      </div>

      {/* Minimalist Upload & Settings Card */}
      <div className="p-6 md:p-8 rounded-2xl border border-[var(--paper-border)] bg-[var(--paper-surface)] shadow-sm space-y-6">
        {/* Dropzone */}
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleFileDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-6 md:p-8 text-center transition-all cursor-pointer ${
            file
              ? 'border-emerald-500/50 bg-emerald-500/5'
              : 'border-[var(--paper-border)] hover:border-[var(--ink-primary)] bg-[var(--paper-subsurface)] hover:bg-[var(--paper-surface)]'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.epub,.txt,.md,.markdown"
            className="hidden"
            onChange={(e) => {
              if (e.target.files?.[0]) {
                setFile(e.target.files[0])
                setManualPath('')
              }
            }}
          />

          {file ? (
            <div className="space-y-2">
              <div className="h-11 w-11 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 flex items-center justify-center mx-auto">
                <FileText className="h-5 w-5" />
              </div>
              <div className="font-semibold text-sm text-[var(--ink-primary)]">{file.name}</div>
              <div className="text-xs text-[var(--ink-muted)]">
                {(file.size / (1024 * 1024)).toFixed(2)} MB · 点击可更换文件
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="h-10 w-10 rounded-full bg-[var(--paper-surface)] text-[var(--ink-secondary)] flex items-center justify-center mx-auto border border-[var(--paper-border)]">
                <Upload className="h-5 w-5" />
              </div>
              <div className="text-sm font-semibold text-[var(--ink-primary)]">
                {t.reader.dropzoneTitle}
              </div>
              {t.reader.dropzoneSubtitle ? (
                <div className="text-xs font-mono text-[var(--ink-muted)]">
                  {t.reader.dropzoneSubtitle}
                </div>
              ) : null}
            </div>
          )}
        </div>

        {/* Manual path alternative (unobtrusive) */}
        {!file && (
          <div>
            <input
              type="text"
              value={manualPath}
              onChange={(e) => setManualPath(e.target.value)}
              placeholder={t.reader.orEnterPath}
              className="w-full text-xs px-3.5 py-2 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-bg)] text-[var(--ink-primary)] focus:outline-hidden focus:border-[var(--ink-primary)] font-mono"
            />
          </div>
        )}

        {/* Translation Settings Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
          {/* Target Language */}
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-[var(--ink-secondary)]">
              {t.reader.targetLanguage}
            </label>
            <select
              value={targetLang}
              onChange={(e) => setTargetLang(e.target.value)}
              className="w-full text-xs px-3 py-2 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-bg)] text-[var(--ink-primary)] focus:outline-hidden focus:border-[var(--ink-primary)] cursor-pointer"
            >
              <option value="zh-Hans">简体中文 (zh-Hans)</option>
              <option value="en">English (en)</option>
              <option value="ja">日本語 (ja)</option>
              <option value="fr">Français (fr)</option>
              <option value="de">Deutsch (de)</option>
              <option value="es">Español (es)</option>
              <option value="ru">Русский (ru)</option>
            </select>
          </div>

          {/* Style Presets */}
          <div className="space-y-1.5">
            <label className="text-xs font-semibold text-[var(--ink-secondary)]">
              {t.reader.stylePreset}
            </label>
            <select
              value={stylePreset}
              onChange={(e) => setStylePreset(e.target.value as 'academic' | 'literary' | 'technical')}
              className="w-full text-xs px-3 py-2 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-bg)] text-[var(--ink-primary)] focus:outline-hidden focus:border-[var(--ink-primary)] cursor-pointer"
            >
              <option value="academic">{t.reader.styleAcademic}</option>
              <option value="literary">{t.reader.styleLiterary}</option>
              <option value="technical">{t.reader.styleTechnical}</option>
            </select>
          </div>
        </div>

        {/* Start Translation CTA */}
        <button
          disabled={submitting || (!file && !manualPath.trim())}
          onClick={handleStartTranslation}
          className="w-full py-3 px-4 rounded-xl text-sm font-bold bg-[var(--ink-primary)] text-[var(--paper-bg)] hover:opacity-90 disabled:opacity-40 disabled:cursor-not-allowed shadow-md transition-all cursor-pointer flex items-center justify-center gap-2"
        >
          {submitting ? (
            <>
              <div className="h-4 w-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
              <span>正在处理并启动...</span>
            </>
          ) : (
            <>
              <Sparkles className="h-4 w-4" />
              <span>{t.reader.startTranslation}</span>
            </>
          )}
        </button>
      </div>

      {/* Bookshelf Section */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-bold text-[var(--ink-primary)] flex items-center gap-2">
            <BookOpen className="h-4 w-4 text-[var(--ink-muted)]" />
            <span>{t.reader.recentBooks}</span>
          </h2>
          <button
            onClick={refreshJobs}
            className="text-xs text-[var(--ink-muted)] hover:text-[var(--ink-primary)] transition-colors cursor-pointer"
          >
            刷新列表
          </button>
        </div>

        {loadingJobs && jobs.length === 0 ? (
          <div className="py-12 text-center text-xs text-[var(--ink-muted)]">正在加载书籍列表...</div>
        ) : jobs.length === 0 ? (
          <div className="py-12 text-center rounded-xl border border-[var(--paper-border)] bg-[var(--paper-subsurface)] text-xs text-[var(--ink-muted)]">
            {t.reader.noBooksYet}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {jobs.map((job) => {
              const rawName = job.file_name ? (job.file_name.split(/[/\\]/).pop() ?? job.file_name) : job.job_id
              const fileName = rawName.replace(/^\d{8}-\d{6}-[a-f0-9]{8}-/, '')
              const isCompleted = job.status === 'completed'
              const isRunning = job.status === 'running' || job.status === 'pending'

              return (
                <div
                  key={job.job_id}
                  onClick={() => setActiveJobId(job.job_id)}
                  className="p-4 rounded-xl border border-[var(--paper-border)] bg-[var(--paper-surface)] hover:border-[var(--paper-border-hover)] hover:shadow-sm transition-all cursor-pointer flex flex-col justify-between space-y-3 group"
                >
                  <div className="space-y-1">
                    <div className="flex items-start justify-between gap-2">
                      <div className="font-semibold text-xs md:text-sm text-[var(--ink-primary)] group-hover:text-blue-600 dark:group-hover:text-blue-400 transition-colors line-clamp-2">
                        {fileName}
                      </div>
                      <span
                        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold shrink-0 ${
                          isCompleted
                            ? 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
                            : isRunning
                              ? 'bg-amber-500/10 text-amber-700 dark:text-amber-400 animate-pulse'
                              : 'bg-red-500/10 text-red-700 dark:text-red-400'
                        }`}
                      >
                        {isCompleted ? '已完成' : isRunning ? '翻译中' : '失败'}
                      </span>
                    </div>
                    <div className="text-[11px] text-[var(--ink-muted)]">
                      目标: {job.target_lang ?? 'zh'} · 进度: {job.progress_percent ?? (isCompleted ? 100 : 0)}%
                    </div>
                  </div>

                  <div className="pt-2 border-t border-[var(--paper-border)] flex items-center justify-between text-xs font-medium text-[var(--ink-secondary)] group-hover:text-[var(--ink-primary)]">
                    <span className="text-[10px] text-[var(--ink-muted)] font-mono truncate max-w-[150px]">
                      {job.job_id}
                    </span>
                    <span className="inline-flex items-center gap-1 text-blue-600 dark:text-blue-400">
                      <span>{t.reader.readAndExport}</span>
                      <span>→</span>
                    </span>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
