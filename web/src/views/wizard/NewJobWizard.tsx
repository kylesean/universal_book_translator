import React, { useState, useRef, useEffect, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  UploadCloud,
  FileText,
  Sparkles,
  ArrowRight,
  AlertTriangle,
  Loader2,
  Sliders,
  ChevronDown,
  ChevronRight,
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { InfoTip } from '@/components/ui/InfoTip'
import {
  assessJob,
  getRuntimeConfig,
  submitJob,
  uploadSourceDocument,
  type JobAssessResponse,
  type JobSubmitRequest,
  type RuntimeConfig,
} from '@/api/client'
import { useI18n } from '@/i18n/useI18n'

/** Human-readable duration from a seconds estimate (e.g. "3m", "1.2h"). */
function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  return `${(seconds / 3600).toFixed(1)}h`
}

/**
 * Substitute ``{key}`` placeholders; a missing key is left visible, not blank.
 * A list value joins with the locale's separator rather than JS's default comma.
 */
function interpolate(
  template: string,
  params: Record<string, unknown>,
  listSeparator: string,
): string {
  return template.replace(/\{(\w+)\}/g, (whole, key: string) => {
    if (!(key in params)) return whole
    const value = params[key]
    return Array.isArray(value) ? value.join(listSeparator) : String(value)
  })
}

/**
 * Localize a pre-flight warning by its stable engine code.
 *
 * The engine's ``detail_zh`` is Chinese only, so rendering it directly showed
 * Chinese copy in the English console. The dictionary is keyed on the same
 * codes the engine switches on; an unknown code falls back to the engine copy
 * (Chinese, but honest) rather than printing a raw key.
 */
function warningText(
  warning: JobAssessResponse['warnings'][number],
  t: ReturnType<typeof useI18n>['t'],
): string {
  const template = t.wizard.warningCodes[warning.code]
  if (!template) return warning.detail_zh
  const params: Record<string, unknown> = { ...(warning.params ?? {}) }
  // FONT_RESIDUE_RISK interpolates a remedy clause, which the engine picks
  // from a scanned / not-scanned branch, plus the at-risk page count when the
  // witness found suspect pages it could not confirm.
  if (warning.code === 'FONT_RESIDUE_RISK') {
    params.remedy = params.is_scanned ? t.wizard.warningRemedyRescan : t.wizard.warningRemedyOcr
    const atRisk = Number(params.at_risk_pages ?? 0)
    params.at_risk_clause =
      atRisk > 0 ? t.wizard.warningAtRiskClause.replace('{count}', String(atRisk)) : ''
  }
  return interpolate(template, params, t.wizard.listSeparator)
}

/**
 * The languages the engine has a profile for (`ubt/core/language_profile.py`),
 * plus the two Chinese script variants the target validator accepts. Labels are
 * the language's own endonym, which needs no translation.
 *
 * The old picker offered four source / three target languages while the engine
 * supported ten, so ja→ko or en→ru were unreachable from the console. A test
 * (tests/unit/api/test_console_contract.py) pins this list to the engine's.
 */
const LANGUAGES: ReadonlyArray<{ code: string; label: string }> = [
  { code: 'en', label: 'English (en)' },
  { code: 'zh', label: '简体中文 (zh)' },
  { code: 'zh-tw', label: '繁體中文 (zh-tw)' },
  { code: 'ja', label: '日本語 (ja)' },
  { code: 'ko', label: '한국어 (ko)' },
  { code: 'de', label: 'Deutsch (de)' },
  { code: 'fr', label: 'Français (fr)' },
  { code: 'es', label: 'Español (es)' },
  { code: 'ru', label: 'Русский (ru)' },
  { code: 'ar', label: 'العربية (ar)' },
  { code: 'he', label: 'עברית (he)' },
]

export function NewJobWizard() {
  const { t } = useI18n()
  const navigate = useNavigate()
  const [filePath, setFilePath] = useState('')
  const [fileName, setFileName] = useState('')
  const [fileSize, setFileSize] = useState<string>('')
  const [isDragOver, setIsDragOver] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [targetLang, setTargetLang] = useState('zh')
  const [sourceLang, setSourceLang] = useState('en')
  const [preset, setPreset] = useState<'publication' | 'standard' | 'publication_math' | 'preview' | 'fast' | null>(null)
  const [budgetUsd, setBudgetUsd] = useState<number | null>(null)
  // The raw field text is separate from ``budgetUsd``: while the user is
  // mid-typing ("" or "0.") the numeric value is not yet usable, but the input
  // must not snap back to a seeded default.
  const [budgetInput, setBudgetInput] = useState('')

  // Advanced compiler & layout options
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false)
  const [domainProfile, setDomainProfile] = useState('auto')
  const [dualMode, setDualMode] = useState<'auto' | 'dual' | 'mono'>('auto')
  // 'auto' submits as null, which lets the engine align the delivered layout to
  // the source. The engine accepts only 1 or 2, so the sentinel never reaches it.
  const [layoutColumns, setLayoutColumns] = useState<'auto' | '1' | '2'>('auto')
  const [translateChrome, setTranslateChrome] = useState<'auto' | 'on' | 'off'>('auto')
  // 'auto' submits as null so the operator's UBT_OCR_MODE still applies.
  const [ocrMode, setOcrMode] = useState<
    'auto' | 'rapidocr' | 'sidecar' | 'cloud' | 'vlm' | 'off'
  >('auto')
  const [pageRange, setPageRange] = useState('')
  const [glossaryPath, setGlossaryPath] = useState('')
  const [dryRun, setDryRun] = useState(false)
  // Delivery shape and spend knobs the REST model has always accepted but no
  // control reached. Only the ones a user can actually reason about are here;
  // ``draft_model``/``pdf_engine``/``batch_limit`` stay operator territory.
  const [emitBoth, setEmitBoth] = useState<'auto' | 'on' | 'off'>('auto')
  const [visualJudge, setVisualJudge] = useState<'auto' | 'on' | 'off'>('auto')
  // Empty = follow the operator's UBTConfig.max_concurrency.
  const [concurrency, setConcurrency] = useState('')
  const [domainHint, setDomainHint] = useState('')
  // Chapter window (1-based start + count). Empty = the whole book.
  const [chapterStart, setChapterStart] = useState('')
  const [chapterCount, setChapterCount] = useState('')

  const [isAssessing, setIsAssessing] = useState(false)
  const [isReassessing, setIsReassessing] = useState(false)
  const [assessment, setAssessment] = useState<JobAssessResponse | null>(null)
  const [assessError, setAssessError] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)

  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)
  // Set when the submit matched an existing job instead of starting one, so the
  // console can say so rather than presenting a run it did not start.
  const [reusedJobId, setReusedJobId] = useState<string | null>(null)
  // Whether a real translation is possible at all. ``null`` = not loaded yet, so
  // the banner stays hidden rather than flashing a false warning on mount.
  const [runtime, setRuntime] = useState<RuntimeConfig | null>(null)
  // Which declared provider block this job runs against. Empty means "use
  // whatever the operator configured as active" — a real default, not a sentinel
  // the server has to interpret.
  const [provider, setProvider] = useState('')
  // The operator can proceed knowingly, but only after seeing the banner. A hard
  // block would be wrong (a rehearsal is a legitimate feature); a silent one is
  // what produced a mock deliverable nobody asked for.
  const [rehearsalAcknowledged, setRehearsalAcknowledged] = useState(false)

  // One read on mount: whether a provider/model is configured decides whether a
  // submit can be real. A failure is treated as "unknown" (no banner) rather than
  // "unconfigured", so a transient error cannot block a working setup.
  useEffect(() => {
    let cancelled = false
    void getRuntimeConfig()
      .then((loaded) => {
        if (!cancelled) setRuntime(loaded)
      })
      .catch(() => {
        if (!cancelled) setRuntime(null)
      })
    return () => {
      cancelled = true
    }
  }, [])

  // The server's own verdict, not a client-side guess. The rule is "no model, or
  // no real credential" — an operator using a built-in protocol with
  // UBT_LLM_API_KEY set has no declared provider block and still runs for real, so
  // inferring this from ``providers.some(p => p.declared)`` would warn wrongly.
  const rehearsalOnly = runtime !== null && !runtime.active.can_translate

  // Only blocks the operator declared. Built-in wire protocols are usable names
  // but carry no endpoint of their own, so listing them here would offer a choice
  // that changes nothing visible.
  const declaredProviders = (runtime?.providers ?? []).filter((p) => p.declared)
  const selectedProviderDetail = declaredProviders.find((p) => p.name === provider)

  const lastAssessedFileRef = useRef<string>('')
  const hasUserSelectedPresetRef = useRef<boolean>(false)
  const hasUserSelectedDomainRef = useRef<boolean>(false)
  const hasUserSelectedDualModeRef = useRef<boolean>(false)
  const assessSeqRef = useRef(0)

  const runAssess = useCallback(
    async (
      path: string,
      opts?: {
        overridePreset?: 'publication' | 'standard' | 'publication_math' | 'preview' | 'fast'
        overridePages?: string
        overrideTarget?: string
        overrideSource?: string
        silent?: boolean
      }
    ) => {
      if (!path.trim()) return
      const isSilent = opts?.silent ?? false
      if (!isSilent) {
        setIsAssessing(true)
        setAssessment(null)
      } else {
        setIsReassessing(true)
      }
      setAssessError(null)

      const activePreset = opts?.overridePreset ?? preset ?? undefined
      const activePages = opts?.overridePages !== undefined ? opts.overridePages : pageRange
      const activeTarget = opts?.overrideTarget ?? targetLang
      const activeSource = opts?.overrideSource ?? sourceLang

      const seq = ++assessSeqRef.current
      try {
        const res = await assessJob({
          input_path: path.trim(),
          target_lang: activeTarget,
          source_lang: activeSource,
          preset: activePreset,
          pages: activePages.trim() ? activePages.trim() : null,
          // The quote must price what this form will actually submit. Without
          // these the server priced ``[defaults]``'s provider while the
          // dropdown above selected another — and a cloud OCR choice added a
          // vision line the quote never saw.
          provider: provider || null,
          ocr_mode: ocrMode === 'auto' ? null : ocrMode,
          visual_judge_enabled: visualJudge === 'auto' ? null : visualJudge === 'on',
          max_concurrency: concurrency.trim() ? Number(concurrency) : null,
          deep: !isSilent,
        })
        if (seq !== assessSeqRef.current) return // stale response
        setAssessment(res)

        // Auto-feed recommendations when a new document is assessed
        if (lastAssessedFileRef.current !== path.trim()) {
          lastAssessedFileRef.current = path.trim()
          if (!hasUserSelectedPresetRef.current && res.route?.recommended_preset) {
            const recP = res.route.recommended_preset.toLowerCase() as any
            if (['publication', 'standard', 'publication_math', 'preview', 'fast'].includes(recP)) {
              setPreset(recP)
            }
          }
          if (!hasUserSelectedDomainRef.current && res.route?.recommended_profile) {
            const recDom = res.route.recommended_profile.toLowerCase()
            if (['auto', 'general', 'textbook', 'paper', 'fiction', 'humanities', 'semiconductor'].includes(recDom)) {
              setDomainProfile(recDom)
            }
          }
          if (!hasUserSelectedDualModeRef.current && res.route?.recommended_dual_mode) {
            const recDual = res.route.recommended_dual_mode.toLowerCase() as any
            if (['auto', 'dual', 'mono'].includes(recDual)) {
              setDualMode(recDual)
            }
          }
        }
      } catch (err) {
        if (seq !== assessSeqRef.current) return // stale response
        if (!isSilent) {
          setAssessError(err instanceof Error ? err.message : 'Assessment failed')
        }
        setAssessment(null) // don't leave a stale quote on error
      } finally {
        if (seq === assessSeqRef.current) {
          setIsAssessing(false)
          setIsReassessing(false)
        }
      }
    },
    [preset, pageRange, targetLang, sourceLang, provider, ocrMode, visualJudge, concurrency]
  )

  // Reactive re-assessment on preset / page slice / lang tweaks
  useEffect(() => {
    if (!filePath.trim()) return
    if (lastAssessedFileRef.current === filePath.trim()) {
      const timer = setTimeout(() => {
        void runAssess(filePath, {
          overridePreset: preset ?? undefined,
          overridePages: pageRange,
          overrideTarget: targetLang,
          overrideSource: sourceLang,
          silent: true,
        })
      }, 350)
      return () => clearTimeout(timer)
    }
  }, [
    filePath,
    preset,
    pageRange,
    targetLang,
    sourceLang,
    provider,
    ocrMode,
    visualJudge,
    concurrency,
    runAssess,
  ])

  // Browsers do not expose a real filesystem path (`File.path` exists only in
  // Electron), so the picked file is uploaded to the server and the staged
  // path it returns becomes the assess/submit input. A previously staged file
  // stays valid — the manual path field still allows server-side paths too.
  const handleFile = async (file: File) => {
    setFileName(file.name)
    setFileSize((file.size / (1024 * 1024)).toFixed(1) + ' MB')
    setIsUploading(true)
    setAssessError(null)
    setAssessment(null)
    lastAssessedFileRef.current = ''
    hasUserSelectedPresetRef.current = false
    hasUserSelectedDomainRef.current = false
    hasUserSelectedDualModeRef.current = false
    try {
      const staged = await uploadSourceDocument(file)
      setFilePath(staged.file_path)
      void runAssess(staged.file_path, { silent: false })
    } catch (err) {
      setAssessError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setIsUploading(false)
    }
  }

  const handleFileDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    if (isUploading) return
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      void handleFile(e.dataTransfer.files[0])
    }
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      void handleFile(e.target.files[0])
      e.target.value = '' // allow re-picking the same file after an edit
    }
  }

  /**
   * Whether the glossary field would be silently ignored by the engine.
   *
   * ``load_external_glossary`` logs a warning and returns ``[]`` for a path that
   * does not exist, so a typo'd path starts a full run with no glossary and no
   * error. We cannot stat the server's filesystem from here, but we can reject
   * the shapes that cannot possibly load: a relative path, or an extension the
   * loader does not parse.
   */
  const glossaryError = (() => {
    const p = glossaryPath.trim()
    if (!p) return null
    if (!p.startsWith('/')) return t.wizard.glossaryPathInvalid
    if (!/\.(csv|tsv|json)$/i.test(p)) return t.wizard.glossaryPathInvalid
    return null
  })()
  /**
   * A cap is enabled but no usable positive number was entered.
   *
   * ``JobSubmitRequest.budget_usd`` is ``Field(gt=0)``, so sending 0 (or an
   * empty field coerced to 0) is a 422 on every submission. Blocking it here
   * keeps the failure visible at the field instead of at the API boundary.
   */
  const budgetInvalid = budgetUsd === null && budgetInput.trim() !== ''
  /**
   * OCR modes that ship rendered page images off this machine.
   *
   * The engine refuses these unless UBT_ALLOW_PAGE_UPLOAD=true, so picking
   * one here without that flag is a submit-time error rather than a silent
   * privacy change. 'auto' can also reach the cloud, but only as a last
   * resort after local engines fail, and it warns in the log.
   */
  const ocrNeedsEgress = ocrMode === 'cloud' || ocrMode === 'vlm'
  /**
   * Page egress is closed on this deployment, so neither the visual judge nor a
   * cloud/vlm OCR mode can ship a page anywhere. The judge would skip every page
   * and bill nothing, which is worse than not offering it: the switch would
   * promise a quality gate the run cannot perform.
   */
  const pageEgressClosed = runtime !== null && !runtime.active.allow_page_upload

  /**
   * The submit payload, built once so the idempotency fingerprint and the
   * request cannot drift apart.
   */
  const buildPayload = (): JobSubmitRequest => ({
    input_path: filePath.trim(),
    target_lang: targetLang,
    source_lang: sourceLang,
    profile: domainProfile,
    // A provider *name*, validated server-side against the declared blocks.
    // Endpoint and credentials still come from that block.
    provider: provider || null,
    preset: preset,
    dual_mode: dualMode === 'auto' ? null : dualMode,
    // null = "align to the source layout"; the engine rejects anything but 1/2.
    layout_columns: layoutColumns === 'auto' ? null : Number(layoutColumns),
    // null = follow UBTConfig's operator default, not a hard-coded false.
    translate_chrome: translateChrome === 'auto' ? null : translateChrome === 'on',
    // null = follow UBT_OCR_MODE / config default.
    ocr_mode: ocrMode === 'auto' ? null : ocrMode,
    emit_both: emitBoth === 'auto' ? null : emitBoth === 'on',
    // A judge with egress closed never runs, so do not claim it will.
    visual_judge_enabled:
      visualJudge === 'auto' || pageEgressClosed ? null : visualJudge === 'on',
    // Empty = operator default; the engine rejects anything below 1.
    max_concurrency: concurrency.trim() ? Number(concurrency) : null,
    domain: domainHint.trim() ? domainHint.trim() : null,
    start_chapter: chapterStart.trim() ? Number(chapterStart) : null,
    max_chapters: chapterCount.trim() ? Number(chapterCount) : null,
    pages: pageRange.trim() ? pageRange.trim() : null,
    glossary: glossaryPath.trim() ? glossaryPath.trim() : null,
    priority: 0,
    dry_run: dryRun,
    budget_usd: budgetUsd,
  })

  /**
   * A job id derived from the payload, so a retried submit is idempotent.
   *
   * ``POST /jobs/submit`` already returns the existing job for a repeated
   * ``job_id``; the console simply never sent one, so a double click or a
   * network retry started — and billed — a second identical run. The key is
   * keyed on the payload itself: *changing* a field mints a new id, because
   * reusing the old one would hand back the previous job and silently ignore
   * the edit.
   */
  const submitKeyRef = useRef<{ fingerprint: string; jobId: string } | null>(null)
  const idempotencyKeyFor = (payload: JobSubmitRequest): string => {
    const fingerprint = JSON.stringify(payload)
    const memo = submitKeyRef.current
    if (memo !== null && memo.fingerprint === fingerprint) return memo.jobId
    const rand =
      typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
        ? crypto.randomUUID().replace(/-/g, '').slice(0, 12)
        : Math.random().toString(36).slice(2, 14)
    const jobId = `web_${Date.now().toString(36)}_${rand}`
    submitKeyRef.current = { fingerprint, jobId }
    return jobId
  }

  const handleSubmit = async () => {
    if (!filePath.trim() || isSubmitting || glossaryError || budgetInvalid) return
    if (rehearsalOnly && !rehearsalAcknowledged) {
      // The banner is the only path to a mock delivery: pressing compile without
      // acknowledging it must not silently produce one.
      setSubmitError(t.wizard.rehearsalBannerBody)
      return
    }
    setIsSubmitting(true)
    setSubmitError(null)
    setReusedJobId(null)
    try {
      const payload = buildPayload()
      const res = await submitJob({ ...payload, job_id: idempotencyKeyFor(payload) })
      if (res.reused) {
        // Nothing was started: this id already had a job. Jumping to a progress
        // page would present it as a run this submit began.
        setReusedJobId(res.job_id)
        return
      }
      navigate(`/jobs/${res.job_id}`)
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : 'Submission failed')
    } finally {
      setIsSubmitting(false)
    }
  }

  // Keyboard shortcut: ⌘⏎ or Ctrl+Enter to compile
  const handleSubmitRef = useRef(handleSubmit)
  useEffect(() => {
    // Latest-ref pattern: the listener below is bound once and must call the
    // current `handleSubmit`, so the ref is refreshed after each commit rather
    // than written during render (which React forbids).
    handleSubmitRef.current = handleSubmit
  })
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        if (filePath.trim() && !isSubmitting) {
          e.preventDefault()
          handleSubmitRef.current()
        }
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [filePath, isSubmitting])

  // The badge names the engine parameters that actually tell one preset apart
  // from another (ubt/core/presets.py): the PDF parser and the prompt depth.
  // It deliberately does NOT name a model (a preset never picks one — the
  // router does) and does NOT claim a quality "gate" (a preset never changes
  // one; presets.py says so in its header). It also does not claim formula
  // vectorisation outside publication_math: ``auto`` formula enrichment crops
  // the source region, so publication and standard render identical math.
  const presetOptions = [
    {
      id: 'publication' as const,
      name: t.wizard.presetPublication,
      desc: t.wizard.presetPublicationDesc,
      badge: t.wizard.presetBadgeRich,
    },
    {
      id: 'standard' as const,
      name: t.wizard.presetStandard,
      desc: t.wizard.presetStandardDesc,
      badge: t.wizard.presetBadgeAdaptive,
    },
    {
      id: 'publication_math' as const,
      name: t.wizard.presetPublicationMath,
      desc: t.wizard.presetPublicationMathDesc,
      badge: t.wizard.presetBadgeVectorMath,
      // The only preset that pays for VLM transcription; flagged so the cost
      // jump is visible before submission rather than after.
      slow: true,
    },
    {
      id: 'preview' as const,
      name: t.wizard.presetPreview,
      desc: t.wizard.presetPreviewDesc,
      badge: t.wizard.presetBadgeFormulaImage,
    },
    {
      id: 'fast' as const,
      name: t.wizard.presetFast,
      desc: t.wizard.presetFastDesc,
      badge: t.wizard.presetBadgeShortChain,
    },
  ]

  return (
    <div className="flex-1 overflow-y-auto px-4 sm:px-8 py-5 sm:py-7 space-y-5 sm:space-y-7 max-w-6xl mx-auto w-full pb-24 lg:pb-7">
      {/* View Title */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-2 border-b border-[var(--paper-border)] pb-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[var(--ink-primary)]">
            {t.wizard.title}
          </h1>
          <p className="text-xs text-[var(--ink-secondary)] mt-1">
            {t.wizard.subtitle}
          </p>
        </div>
        <div className="text-xs font-mono text-[var(--ink-muted)]">
          {t.wizard.walLedger}: <span className="text-[#15803d] font-semibold">{t.common.online}</span>
        </div>
      </div>

      {/* Main 2-Column Split */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8 items-start">
        {/* Left Column: Input, Preset, Budget */}
        <div className="lg:col-span-7 space-y-6">
          {/* File Dropzone */}
          <div>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileInputChange}
              accept=".pdf,.epub,.docx,.md,.markdown,.txt,.html,.htm"
              className="hidden"
            />
            {/* A div with onClick is unreachable by keyboard, and the real
                <input type="file"> above is `hidden` (hence unfocusable), so a
                keyboard-only operator could not attach a document at all — the
                wizard's very first required step. `role="button"` + tabIndex +
                Enter/Space make the dropzone a control, without changing how
                drag-and-drop behaves for a mouse. */}
            <div
              role="button"
              tabIndex={0}
              aria-label={t.wizard.uploadAria}
              onDragOver={(e) => {
                e.preventDefault()
                setIsDragOver(true)
              }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={handleFileDrop}
              onClick={() => fileInputRef.current?.click()}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  fileInputRef.current?.click()
                }
              }}
              className={`rounded-lg border-2 border-dashed p-5 sm:p-7 transition-all cursor-pointer text-center ${
                isDragOver
                  ? 'border-[var(--ink-primary)] bg-[var(--paper-subsurface)]'
                  : 'border-[var(--paper-border)] hover:border-[var(--paper-border-hover)] bg-[var(--paper-surface)] hover:bg-[var(--paper-subsurface)]'
              }`}
            >
              <div className="h-10 w-10 mx-auto rounded-full bg-[var(--paper-subsurface)] border border-[var(--paper-border)] flex items-center justify-center text-[var(--ink-secondary)] mb-2.5 shadow-2xs">
                {isUploading ? (
                  <Loader2 className="h-5 w-5 animate-spin" />
                ) : (
                  <UploadCloud className="h-5 w-5" />
                )}
              </div>
              <div className="text-xs font-semibold text-[var(--ink-primary)]">
                {isUploading
                  ? t.wizard.uploading
                  : isDragOver
                    ? t.wizard.dropzoneActive
                    : t.wizard.dropzoneTitle}
              </div>
              <div className="text-xs text-[var(--ink-secondary)] mt-1">
                {t.wizard.dropzoneSubtitle}
              </div>
              <div className="mt-3">
                <span className="inline-block py-2 px-3 rounded text-xs font-mono text-[var(--ink-primary)] underline decoration-[var(--paper-border-hover)] hover:text-black font-medium">
                  {t.wizard.orBrowse}
                </span>
              </div>
            </div>

            {/* Manual Path Fallback */}
            <div className="mt-2.5 flex gap-2">
              <input
                type="text"
                value={filePath}
                onChange={(e) => setFilePath(e.target.value)}
                placeholder={t.wizard.manualPathPlaceholder}
                className="flex-1 min-w-0 h-9 sm:h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-3 text-xs text-[var(--ink-primary)] focus:border-[var(--ink-primary)] focus:outline-none font-mono placeholder:text-[var(--ink-muted)] shadow-2xs"
              />
              <Button
                onClick={() => {
                  lastAssessedFileRef.current = ''
                  hasUserSelectedPresetRef.current = false
                  hasUserSelectedDomainRef.current = false
                  hasUserSelectedDualModeRef.current = false
                  void runAssess(filePath, { silent: false })
                }}
                disabled={!filePath.trim() || isAssessing}
                variant="secondary"
                size="sm"
                className="shrink-0 h-9 sm:h-8 px-3"
              >
                {isAssessing ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin mr-1.5" />
                ) : (
                  <Sparkles className="h-3.5 w-3.5 mr-1.5 text-[var(--ink-primary)]" />
                )}
                {t.wizard.assessBtn}
              </Button>
            </div>

            {assessError && (
              <div className="mt-2.5 p-2.5 rounded-[5px] bg-[#b91c1c]/10 border border-[#b91c1c]/25 text-[#b91c1c] text-xs flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 shrink-0" />
                <span>{assessError}</span>
              </div>
            )}
          </div>

          {/* Languages & Preset */}
          <div className="space-y-4 pt-4 border-t border-[var(--paper-border)]">
            {/* Provider: which declared block does the work. A job may name one;
                the server validates it against the operator's ubt.toml. Hidden
                when nothing is declared, because then there is nothing to choose
                and the first-run banner already says what to do. */}
            {declaredProviders.length > 0 ? (
              <div>
                <div className="flex items-center gap-1 mb-1">
                  <label className="type-label">{t.wizard.provider}</label>
                  <InfoTip label={t.wizard.providerHelp}>{t.wizard.providerHelp}</InfoTip>
                </div>
                <select
                  value={provider}
                  onChange={(e) => setProvider(e.target.value)}
                  className="type-value w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs"
                >
                  <option value="">
                    {runtime?.active.provider
                      ? `${t.wizard.providerDefault} (${runtime.active.provider})`
                      : t.wizard.providerDefault}
                  </option>
                  {declaredProviders.map((p) => (
                    <option key={p.name} value={p.name}>
                      {p.name}
                      {p.draft_model ? ` — ${p.draft_model}` : ''}
                    </option>
                  ))}
                </select>
                {selectedProviderDetail ? (
                  <p className="type-help text-[var(--ink-muted)] mt-1">
                    {selectedProviderDetail.base_url ?? '—'}
                    {selectedProviderDetail.api_mode
                      ? ` · ${selectedProviderDetail.api_mode}`
                      : ''}
                    {selectedProviderDetail.draft_model
                      ? ` · ${selectedProviderDetail.draft_model}`
                      : ''}
                  </p>
                ) : null}
              </div>
            ) : null}

            {/* Language Pair */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="type-label block mb-1">
                  {t.wizard.sourceLang}
                </label>
                <select
                  value={sourceLang}
                  onChange={(e) => setSourceLang(e.target.value)}
                  className="type-value w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs"
                >
                  {LANGUAGES.map((lang) => (
                    <option key={lang.code} value={lang.code}>
                      {lang.label}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="type-label block mb-1">
                  {t.wizard.targetLang}
                </label>
                <select
                  value={targetLang}
                  onChange={(e) => setTargetLang(e.target.value)}
                  className="type-value w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs"
                >
                  {LANGUAGES.map((lang) => (
                    <option key={lang.code} value={lang.code}>
                      {lang.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Presets List */}
            <div>
              <label className="type-label block mb-2">
                {t.wizard.presets}
              </label>
              <div className="space-y-2">
                {presetOptions.map((p) => {
                  const isSelected = preset === p.id
                  const isRecommended =
                    assessment?.route?.recommended_preset?.toLowerCase() === p.id
                  return (
                    <div
                      key={p.id}
                      onClick={() => {
                        hasUserSelectedPresetRef.current = true
                        setPreset(p.id)
                      }}
                      className={`p-3 rounded-[6px] border cursor-pointer transition-all ${
                        isSelected
                          ? 'border-[var(--ink-primary)] bg-[var(--paper-surface)] shadow-xs'
                          : 'border-[var(--paper-border)] bg-[var(--paper-subsurface)] hover:bg-[var(--paper-surface)] hover:border-[var(--paper-border-hover)]'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span
                            className={`type-value ${
                              isSelected ? 'font-semibold' : 'font-medium'
                            }`}
                          >
                            {p.name}
                          </span>
                          {isRecommended && (
                            <span className="inline-flex items-center font-mono text-[10px] font-medium text-[#15803d] bg-[#15803d]/10 px-1.5 py-0.5 rounded border border-[#15803d]/20">
                              {t.wizard.engineRecommended}
                            </span>
                          )}
                          {p.slow && (
                            <span className="inline-flex items-center font-mono text-[10px] font-medium text-[#b45309] bg-[#b45309]/10 px-1.5 py-0.5 rounded border border-[#b45309]/20">
                              {t.wizard.presetSlowBadge}
                            </span>
                          )}
                        </div>
                        <span className="font-mono text-[10px] text-[var(--ink-muted)] uppercase tracking-wider shrink-0">
                          {p.badge}
                        </span>
                      </div>
                      <div className="type-help mt-1">{p.desc}</div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Budget Cap.
                A slider could not express this: its range was 0-50 USD, so a
                cloud run on a large book was uncappable, and 0 meant "no cap"
                rather than "zero" — the exact opposite of what a user sliding
                to the left intends. An explicit switch plus a numeric field
                removes both traps, and keeps the default (no cap) visible
                instead of hidden at one end of a track. */}
            <div className="pt-2">
              <div className="flex items-center justify-between text-xs mb-1.5">
                <span className="type-label">{t.wizard.budgetCeiling}</span>
                <label className="flex items-center gap-1.5 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={budgetUsd === null}
                    onChange={(e) => {
                      if (e.target.checked) {
                        // "No cap": clear both the numeric value and the text.
                        setBudgetUsd(null)
                        setBudgetInput('')
                        return
                      }
                      // Turning the cap on seeds a value so the field is never
                      // enabled-but-empty, which would submit budget_usd=0 and
                      // be rejected by the API's gt=0 bound.
                      const seeded = 10
                      setBudgetInput(String(seeded))
                      setBudgetUsd(seeded)
                    }}
                    className="h-3.5 w-3.5 rounded border-[var(--paper-border)] accent-[var(--ink-primary)] cursor-pointer"
                  />
                  <span className="text-xs text-[var(--ink-secondary)]">
                    {t.wizard.budgetUnlimitedToggle}
                  </span>
                </label>
              </div>

              {budgetUsd === null ? (
                <p className="type-help">{t.wizard.budgetUnlimited}</p>
              ) : (
                <div>
                  <div className="relative">
                    <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-xs font-mono text-[var(--ink-muted)]">
                      $
                    </span>
                    <input
                      type="number"
                      min="0.01"
                      step="0.01"
                      value={budgetInput}
                      onChange={(e) => {
                        const raw = e.target.value
                        setBudgetInput(raw)
                        const parsed = parseFloat(raw)
                        // The API rejects budget_usd <= 0 (Field(gt=0)), so an
                        // empty or non-positive entry must not reach it.
                        setBudgetUsd(Number.isFinite(parsed) && parsed > 0 ? parsed : null)
                      }}
                      placeholder={t.wizard.budgetPlaceholder}
                      className={`w-full h-8 rounded-[5px] border bg-[var(--paper-surface)] pl-6 pr-2.5 text-xs font-mono text-[var(--ink-primary)] focus:outline-none shadow-2xs ${
                        budgetInvalid
                          ? 'border-[#b91c1c] focus:border-[#b91c1c]'
                          : 'border-[var(--paper-border)] focus:border-[var(--ink-primary)]'
                      }`}
                    />
                  </div>
                  {budgetInvalid ? (
                    <p className="type-help mt-1 text-[#b91c1c]">{t.wizard.budgetInvalid}</p>
                  ) : assessment?.cost.total_cost_usd != null &&
                    assessment.cost.total_cost_usd > budgetUsd ? (
                    <p className="type-help mt-1 text-[#b45309]">
                      {t.wizard.budgetExceedsQuote} (${assessment.cost.total_cost_usd.toFixed(2)})
                    </p>
                  ) : (
                    <p className="type-help mt-1">{t.wizard.budgetHelp}</p>
                  )}
                </div>
              )}
            </div>

            {/* Advanced Compiler & Layout Options Drawer */}
            <div className="pt-3 border-t border-[var(--paper-border)]">
              <button
                type="button"
                onClick={() => setIsAdvancedOpen(!isAdvancedOpen)}
                className="w-full flex items-center justify-between py-1 text-left type-value font-semibold text-[var(--ink-primary)] hover:text-black transition-colors focus:outline-none group cursor-pointer"
              >
                <div className="flex items-center gap-2">
                  <Sliders className="h-3.5 w-3.5 text-[var(--ink-secondary)] group-hover:text-[var(--ink-primary)]" />
                  <span>{t.wizard.advancedOptionsTitle}</span>
                  <Badge variant="outline" className="text-xs font-mono font-normal uppercase tracking-wider py-0 px-1.5">
                    {t.wizard.advancedOptionsTag}
                  </Badge>
                </div>
                {isAdvancedOpen ? (
                  <ChevronDown className="h-4 w-4 text-[var(--ink-muted)] group-hover:text-[var(--ink-primary)] transition-transform" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-[var(--ink-muted)] group-hover:text-[var(--ink-primary)] transition-transform" />
                )}
              </button>
              <p className="type-help mt-0.5 mb-2">{t.wizard.advancedOptionsSubtitle}</p>

              {isAdvancedOpen && (
                <div className="space-y-4 pt-3 pb-1 border-t border-dashed border-[var(--paper-border)] mt-2">
                  {/* Grouped by intent: "what the reader gets" vs "what reads
                      the source". Seven flat controls with no headings made the
                      drawer one undifferentiated block. */}
                  <div className="type-group -mb-1">{t.wizard.groupLayout}</div>

                  {/* Row 1: Profile & Dual Mode */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 items-start">
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <label className="type-label">{t.wizard.domainProfile}</label>
                        <InfoTip label={t.wizard.domainProfile}>{t.wizard.domainProfileHelp}</InfoTip>
                      </div>
                      <select
                        value={domainProfile}
                        onChange={(e) => {
                          hasUserSelectedDomainRef.current = true
                          setDomainProfile(e.target.value)
                        }}
                        className="type-value w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs"
                      >
                        <option value="auto">{t.wizard.domainAuto}</option>
                        <option value="general">{t.wizard.domainGeneral}</option>
                        <option value="textbook">{t.wizard.domainTextbook}</option>
                        <option value="paper">{t.wizard.domainPaper}</option>
                        <option value="fiction">{t.wizard.domainFiction}</option>
                        <option value="humanities">{t.wizard.domainHumanities}</option>
                        <option value="semiconductor">{t.wizard.domainSemiconductor}</option>
                      </select>
                    </div>

                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <label className="type-label">{t.wizard.dualMode}</label>
                        <InfoTip label={t.wizard.dualMode}>{t.wizard.dualModeHelp}</InfoTip>
                      </div>
                      <select
                        value={dualMode}
                        onChange={(e) => {
                          hasUserSelectedDualModeRef.current = true
                          setDualMode(e.target.value as any)
                        }}
                        className="type-value w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs"
                      >
                        <option value="auto">{t.wizard.dualModeAuto}</option>
                        <option value="dual">{t.wizard.dualModeDual}</option>
                        <option value="mono">{t.wizard.dualModeMono}</option>
                      </select>
                    </div>
                  </div>

                  {/* Row 2: Layout columns & page furniture */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 items-start">
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <label className="type-label">{t.wizard.layoutColumns}</label>
                        <InfoTip label={t.wizard.layoutColumns}>
                          {t.wizard.layoutColumnsHelp}
                        </InfoTip>
                      </div>
                      <select
                        value={layoutColumns}
                        onChange={(e) => setLayoutColumns(e.target.value as any)}
                        className="type-value w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs"
                      >
                        <option value="auto">{t.wizard.layoutColumnsAuto}</option>
                        <option value="1">{t.wizard.layoutColumnsOne}</option>
                        <option value="2">{t.wizard.layoutColumnsTwo}</option>
                      </select>
                    </div>

                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <label className="type-label">{t.wizard.translateChrome}</label>
                        <InfoTip label={t.wizard.translateChrome}>
                          {t.wizard.translateChromeHelp}
                        </InfoTip>
                      </div>
                      <select
                        value={translateChrome}
                        onChange={(e) => setTranslateChrome(e.target.value as any)}
                        className="type-value w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs"
                      >
                        <option value="auto">{t.wizard.translateChromeAuto}</option>
                        <option value="on">{t.wizard.translateChromeOn}</option>
                        <option value="off">{t.wizard.translateChromeOff}</option>
                      </select>
                    </div>
                  </div>

                  <div className="type-group -mb-1 pt-1">{t.wizard.groupEngines}</div>

                  {/* Row 2b: OCR engine.
                      Reachable from the CLI (--ocr) and REST (ocr_mode) since
                      the beginning, but not from the console — so the operator
                      reading a FONT_RESIDUE_RISK warning had no way to act on
                      its own advice. 'auto' is the engine default and is left
                      as the null sentinel so an operator's UBT_OCR_MODE still
                      wins. */}
                  <div>
                    <div className="flex items-center gap-1 mb-1">
                      <label className="type-label">{t.wizard.ocrMode}</label>
                      <InfoTip label={t.wizard.ocrMode}>{t.wizard.ocrModeHelp}</InfoTip>
                    </div>
                    <select
                      value={ocrMode}
                      onChange={(e) => setOcrMode(e.target.value as any)}
                      className="type-value w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs"
                    >
                      <option value="auto">{t.wizard.ocrModeAuto}</option>
                      <option value="rapidocr">{t.wizard.ocrModeRapidocr}</option>
                      <option value="sidecar">{t.wizard.ocrModeSidecar}</option>
                      <option value="cloud">{t.wizard.ocrModeCloud}</option>
                      <option value="vlm">{t.wizard.ocrModeVlm}</option>
                      <option value="off">{t.wizard.ocrModeOff}</option>
                    </select>
                    {/* Dynamic, not static: only shown when the choice has a
                        consequence the operator must know before submitting. */}
                    {ocrNeedsEgress && (
                      <p className="type-help mt-1 text-[#b45309]">
                        {t.wizard.ocrModeEgressWarning}
                      </p>
                    )}
                  </div>

                  {/* Row 3: Pages Range Filter */}
                  <div>
                    <div className="flex items-center justify-between mb-1">
                      <div className="flex items-center gap-1">
                        <label className="type-label">{t.wizard.pageFilter}</label>
                        <InfoTip label={t.wizard.pageFilter}>
                          {t.wizard.pageFilterHelp}
                        </InfoTip>
                      </div>
                      <button
                        type="button"
                        onClick={() => setPageRange('1-5')}
                        className="type-help underline hover:text-[var(--ink-primary)] cursor-pointer"
                      >
                        {t.wizard.sliceSampleQuickBtn}
                      </button>
                    </div>
                    <input
                      type="text"
                      value={pageRange}
                      onChange={(e) => setPageRange(e.target.value)}
                      placeholder={t.wizard.pageFilterPlaceholder}
                      className="type-value w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 font-mono focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs placeholder:text-[var(--ink-muted)]"
                    />
                  </div>

                  {/* Row 4: External Glossary Path */}
                  <div>
                    <div className="flex items-center gap-1 mb-1">
                      <label className="type-label">{t.wizard.glossaryPath}</label>
                      <InfoTip label={t.wizard.glossaryPath}>
                        {t.wizard.glossaryPathHelp}
                      </InfoTip>
                    </div>
                    <input
                      type="text"
                      value={glossaryPath}
                      onChange={(e) => setGlossaryPath(e.target.value)}
                      placeholder={t.wizard.glossaryPathPlaceholder}
                      className={`type-value w-full h-8 rounded-[5px] border bg-[var(--paper-surface)] px-2.5 font-mono focus:outline-none shadow-2xs placeholder:text-[var(--ink-muted)] ${
                        glossaryError
                          ? 'border-[#b91c1c] focus:border-[#b91c1c]'
                          : 'border-[var(--paper-border)] focus:border-[var(--ink-primary)]'
                      }`}
                    />
                    {/* Only rendered when invalid — the format rules live in
                        the InfoTip, so a valid path costs no vertical space. */}
                    {glossaryError && (
                      <p className="type-help mt-1 text-[#b91c1c]">{glossaryError}</p>
                    )}
                  </div>

                  {/* What the reader receives. Both of these were REST-only:
                      the engine emits them, the API accepts them, and no
                      control ever reached them. */}
                  <div className="type-group -mb-1 pt-1">{t.wizard.groupDelivery}</div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 items-start">
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <label className="type-label">{t.wizard.emitBoth}</label>
                        <InfoTip label={t.wizard.emitBoth}>{t.wizard.emitBothHelp}</InfoTip>
                      </div>
                      <select
                        value={emitBoth}
                        onChange={(e) => setEmitBoth(e.target.value as any)}
                        className="type-value w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs"
                      >
                        <option value="auto">{t.wizard.toggleAuto}</option>
                        <option value="on">{t.wizard.toggleOn}</option>
                        <option value="off">{t.wizard.toggleOff}</option>
                      </select>
                    </div>

                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <label className="type-label">{t.wizard.visualJudge}</label>
                        <InfoTip label={t.wizard.visualJudge}>
                          {t.wizard.visualJudgeHelp}
                        </InfoTip>
                      </div>
                      <select
                        value={visualJudge}
                        onChange={(e) => setVisualJudge(e.target.value as any)}
                        disabled={pageEgressClosed}
                        className="type-value w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs disabled:opacity-60 disabled:cursor-not-allowed"
                      >
                        <option value="auto">{t.wizard.toggleAuto}</option>
                        <option value="on">{t.wizard.toggleOn}</option>
                        <option value="off">{t.wizard.toggleOff}</option>
                      </select>
                      {/* Stated, not guessed: with egress closed the judge
                          skips every page and bills nothing, so offering the
                          switch would promise a gate the run cannot perform. */}
                      {pageEgressClosed && (
                        <p className="type-help mt-1 text-[var(--ink-muted)]">
                          {t.wizard.visualJudgeBlocked}
                        </p>
                      )}
                    </div>
                  </div>

                  {/* Subject hint: a free-text descriptor, distinct from the
                      profile dropdown above (which picks a term directory). */}
                  <div>
                    <div className="flex items-center gap-1 mb-1">
                      <label className="type-label">{t.wizard.domainHint}</label>
                      <InfoTip label={t.wizard.domainHint}>{t.wizard.domainHintHelp}</InfoTip>
                    </div>
                    <input
                      type="text"
                      value={domainHint}
                      onChange={(e) => setDomainHint(e.target.value)}
                      placeholder={t.wizard.domainHintPlaceholder}
                      className="type-value w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs placeholder:text-[var(--ink-muted)]"
                    />
                  </div>

                  {/* Time and cost. Both feed the quote directly: the runtime
                      estimate scales with concurrency, and the chapter window
                      decides how much of the book is billed. */}
                  <div className="type-group -mb-1 pt-1">{t.wizard.groupBudget}</div>

                  <div>
                    <div className="flex items-center gap-1 mb-1">
                      <label className="type-label">{t.wizard.concurrency}</label>
                      <InfoTip label={t.wizard.concurrency}>
                        {t.wizard.concurrencyHelp}
                      </InfoTip>
                    </div>
                    <input
                      type="number"
                      min={1}
                      value={concurrency}
                      onChange={(e) => setConcurrency(e.target.value)}
                      placeholder={t.wizard.concurrencyAuto}
                      className="type-value w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 font-mono focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs placeholder:text-[var(--ink-muted)]"
                    />
                  </div>

                  <div>
                    <div className="flex items-center gap-1 mb-1">
                      <label className="type-label">{t.wizard.chapterWindow}</label>
                      <InfoTip label={t.wizard.chapterWindow}>
                        {t.wizard.chapterWindowHelp}
                      </InfoTip>
                    </div>
                    <div className="grid grid-cols-2 gap-3.5">
                      <input
                        type="number"
                        min={1}
                        value={chapterStart}
                        onChange={(e) => setChapterStart(e.target.value)}
                        placeholder={t.wizard.chapterStart}
                        aria-label={t.wizard.chapterStart}
                        className="type-value w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 font-mono focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs placeholder:text-[var(--ink-muted)]"
                      />
                      <input
                        type="number"
                        min={1}
                        value={chapterCount}
                        onChange={(e) => setChapterCount(e.target.value)}
                        placeholder={t.wizard.chapterCount}
                        aria-label={t.wizard.chapterCount}
                        className="type-value w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 font-mono focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs placeholder:text-[var(--ink-muted)]"
                      />
                    </div>
                  </div>

                  {/* Rehearsal: zero-token dry run through the same pipeline */}
                  <div className="pt-3 border-t border-[var(--paper-border)]">
                    <label className="flex items-start gap-2 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={dryRun}
                        onChange={(e) => setDryRun(e.target.checked)}
                        className="mt-0.5 h-3.5 w-3.5 rounded border-[var(--paper-border)] accent-[var(--ink-primary)] cursor-pointer"
                      />
                      <span>
                        <span className="type-value font-semibold">{t.wizard.rehearsalLabel}</span>
                        <span className="type-help block mt-0.5">{t.wizard.rehearsalHelp}</span>
                      </span>
                    </label>
                    {dryRun && (
                      <div className="mt-2 rounded-[5px] border border-[#b45309]/30 bg-[#b45309]/5 p-2 type-help text-[#b45309]">
                        {t.wizard.rehearsalActiveWarning}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right Column: Pre-flight Assessment Radar */}
        <div className="lg:col-span-5 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] flex flex-col justify-between overflow-hidden shadow-[0_1px_4px_rgba(0,0,0,0.03)]">
          <div className="p-5 space-y-5">
            <div className="flex items-center justify-between border-b border-[var(--paper-border)] pb-3">
              <div>
                <h3 className="type-value font-bold text-[var(--ink-primary)] tracking-tight">
                  {t.wizard.preflightTitle}
                </h3>
                <p className="type-help mt-0.5">{t.wizard.preflightSubtitle}</p>
              </div>
              {isReassessing ? (
                <Badge variant="outline" className="animate-pulse text-xs font-mono">
                  <Loader2 className="h-3 w-3 animate-spin mr-1 inline" />
                  UPDATING...
                </Badge>
              ) : assessment ? (
                <div className="flex items-center gap-2 shrink-0">
                  {/* One provenance badge for the whole panel, not one per row.
                      Every scale figure below (chars, tokens, blocks, cost) comes
                      from the same assess call, so they share one source. The
                      silent re-assess on a preset change runs with deep=false,
                      which turns an exact block count into a sampled estimate;
                      marking each row separately said the same thing four times
                      and still left the user wondering why the character count
                      moved. */}
                  <Badge
                    variant="outline"
                    className={`text-xs font-mono ${
                      assessment.cost.billable_blocks_is_exact
                        ? 'text-[#15803d] border-[#15803d]/30'
                        : 'text-[#b45309] border-[#b45309]/30'
                    }`}
                  >
                    {assessment.cost.billable_blocks_is_exact
                      ? t.wizard.estimateExact
                      : t.wizard.estimateApproximate}
                  </Badge>
                  <Badge variant="success" dot>
                    {t.common.ready}
                  </Badge>
                </div>
              ) : (
                <Badge variant="outline">{t.common.idle}</Badge>
              )}
            </div>

            {assessment ? (
              <div className="space-y-4">
                {/* Which provider this quote was priced against. The dropdown
                    selects one; the figure below used to come from whatever the
                    operator's file named, so a user comparing blocks saw two
                    identical totals and no way to tell which was priced. */}
                {typeof assessment.meta?.provider === 'string' &&
                  assessment.meta.provider !== '' && (
                    <p className="type-help text-[var(--ink-muted)] font-mono">
                      {t.wizard.quoteProviderNote.replace(
                        '{provider}',
                        assessment.meta.provider
                      )}
                    </p>
                  )}
                {/* Book File Header Card */}
                {fileName && (
                  <div className="p-3 rounded-[5px] bg-[var(--paper-subsurface)] border border-[var(--paper-border)] flex items-center justify-between">
                    <div className="flex items-center gap-2.5 truncate">
                      <FileText className="h-4 w-4 text-[var(--ink-secondary)] shrink-0" />
                      <div className="truncate">
                        <div className="text-xs font-semibold text-[var(--ink-primary)] truncate font-mono">
                          {fileName}
                        </div>
                        <div className="text-xs text-[var(--ink-secondary)] font-mono">{fileSize}</div>
                      </div>
                    </div>
                    <Badge variant="outline" className="uppercase shrink-0 text-xs font-mono">
                      {assessment.document.format_ext.replace('.', '').toUpperCase() ||
                        t.wizard.formatUnknown}
                    </Badge>
                  </div>
                )}

                {/* Metric Rows — real nested assess fields */}
                <div className="divide-y divide-[var(--paper-border)]">
                  <div className="py-2.5 flex items-center justify-between">
                    <span className="type-help">{t.wizard.pageCount}</span>
                    <span className="font-mono font-bold text-[var(--ink-primary)]">
                      {assessment.document.selected_pages != null ? (
                        <>
                          {assessment.document.selected_pages}
                          <span className="text-[var(--ink-muted)] font-normal">
                            {' '}
                            / {assessment.document.pages} ({t.wizard.sliceTestingTag})
                          </span>
                        </>
                      ) : (
                        assessment.document.pages
                      )}
                      <span className="text-[var(--ink-muted)] font-normal">
                        {' '}
                        · {assessment.document.chapters} {t.wizard.chapters}
                      </span>
                    </span>
                  </div>
                  <div className="py-2.5 flex items-center justify-between">
                    <span className="type-help">{t.wizard.sourceChars}</span>
                    <span className="font-mono font-bold text-[var(--ink-primary)]">
                      {assessment.document.source_chars.toLocaleString()}
                    </span>
                  </div>
                  <div className="py-2.5 flex items-start justify-between gap-3">
                    <span className="text-[var(--ink-secondary)] shrink-0">
                      {t.wizard.mathDensity}
                    </span>
                    <span className="text-right">
                      <span
                        className={`font-mono font-bold ${
                          assessment.document.has_formulas ? 'text-[#b45309]' : 'text-[#15803d]'
                        }`}
                      >
                        {assessment.document.math_density}
                      </span>
                      {/* "high" and "no formula objects" are both true and not a
                          contradiction: the density comes from a text sample,
                          while has_formulas comes from the per-page layout
                          census. Saying which is which stops the pair reading
                          as a bug. */}
                      <span className="block text-xs font-normal text-[var(--ink-muted)] leading-relaxed">
                        {assessment.document.has_formulas
                          ? t.wizard.mathDensityWithObjects
                          : t.wizard.mathDensityTextOnly}
                      </span>
                    </span>
                  </div>
                  <div className="py-2.5 flex items-center justify-between">
                    <span className="type-help">{t.wizard.scannedRatio}</span>
                    <span
                      className={`font-mono font-bold ${
                        (assessment.document.scan_page_share ?? 0) > 0.2
                          ? 'text-[#b45309]'
                          : 'text-[#15803d]'
                      }`}
                    >
                      {assessment.document.scan_page_share !== null &&
                      assessment.document.scan_page_share !== undefined
                        ? `${(assessment.document.scan_page_share * 100).toFixed(0)}%`
                        : assessment.document.is_scanned
                          ? 'yes'
                          : 'no'}
                    </span>
                  </div>
                  <div className="py-2.5 flex items-start justify-between gap-3">
                    <span className="text-[var(--ink-secondary)] shrink-0">
                      {t.wizard.estTokens}
                    </span>
                    <span className="text-right">
                      {/* ``estimated_tokens`` is the *source* size only. The
                          engine bills prompt (prefix per call + source) and
                          completion (source × pair fertility) separately, so
                          showing the source figure alone under-quoted the real
                          spend by ~2.6x on the reference document. Both
                          components are already on the assess payload. */}
                      <span className="font-mono font-bold text-[var(--ink-primary)]">
                        {(assessment.cost.prompt_tokens + assessment.cost.completion_tokens).toLocaleString()}
                      </span>
                      <span className="block text-xs font-normal text-[var(--ink-muted)] leading-relaxed">
                        {t.wizard.tokensSource}{' '}
                        {assessment.document.estimated_tokens.toLocaleString()} ·{' '}
                        {t.wizard.tokensPrompt}{' '}
                        {assessment.cost.prompt_tokens.toLocaleString()} ·{' '}
                        {t.wizard.tokensCompletion}{' '}
                        {assessment.cost.completion_tokens.toLocaleString()}
                      </span>
                    </span>
                  </div>
                  <div className="py-2.5 flex items-center justify-between">
                    <span className="type-help">{t.wizard.estCost}</span>
                    <span className="font-mono font-bold">
                      {assessment.cost.total_cost_usd === null ||
                      assessment.cost.total_cost_usd === undefined ? (
                        <span className="text-[var(--ink-muted)]">{t.wizard.notPriced}</span>
                      ) : assessment.cost.total_cost_usd === 0 ? (
                        <span className="text-[#15803d]">
                          $0.00{' '}
                          <span className="text-xs font-normal text-[var(--ink-secondary)]">
                            ({t.wizard.localFreeLabel})
                          </span>
                        </span>
                      ) : (
                        <span className="text-[#b45309]">
                          ${assessment.cost.total_cost_usd.toFixed(2)}
                        </span>
                      )}
                    </span>
                  </div>
                  <div className="py-2.5 flex items-center justify-between">
                    <span className="type-help">{t.wizard.estTime}</span>
                    <span className="font-mono font-bold text-[var(--ink-primary)]">
                      {formatDuration(assessment.runtime.est_seconds_low)} –{' '}
                      {formatDuration(assessment.runtime.est_seconds_high)}
                    </span>
                  </div>
                  <div className="py-2.5 flex items-start justify-between gap-3">
                    <span className="text-[var(--ink-secondary)] shrink-0">
                      {t.wizard.recommendation}
                    </span>
                    <span className="font-mono text-[#15803d] text-xs font-semibold text-right">
                      {assessment.route.recommended_dual_mode} · {assessment.route.recommended_preset}
                      <span className="text-[var(--ink-muted)] font-normal">
                        {' '}
                        ({t.wizard.confidence} {(assessment.route.confidence * 100).toFixed(0)}%)
                      </span>
                    </span>
                  </div>

                  {/* One disclaimer for the whole estimate block: the numbers
                      above are a character heuristic, and no heuristic can be
                      exact across model families. */}
                  <p className="pt-2 text-xs text-[var(--ink-muted)] leading-relaxed border-t border-[var(--paper-border)]">
                    {t.wizard.tokensEstimateNote}
                  </p>
                </div>

                {assessment.warnings.length > 0 && (
                  <div className="rounded-[5px] border border-[#b45309]/30 bg-[#b45309]/5 p-2.5 space-y-1">
                    <div className="text-xs font-mono uppercase tracking-wider text-[#b45309] font-semibold">
                      {t.wizard.probeWarnings} ({assessment.warnings.length})
                    </div>
                    <div className="space-y-1 max-h-40 overflow-y-auto">
                      {assessment.warnings.map((warning) => (
                        <div key={warning.code} className="text-xs text-[var(--ink-secondary)]">
                          <span className="font-mono text-[var(--ink-muted)]">{warning.code}</span>{' '}
                          · {warningText(warning, t)}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="py-16 text-center text-[var(--ink-muted)] text-xs space-y-2">
                <p className="max-w-xs mx-auto leading-relaxed">
                  {t.wizard.waitingFile}
                </p>
              </div>
            )}
          </div>

          {/* Rehearsal banner: the server downgrades a keyless submit to a
              zero-token rehearsal, which delivers mock output. Saying it here —
              with a link to the configuration form — is the difference between a
              labelled feature and a silent substitution. */}
          {rehearsalOnly ? (
            <div className="mx-4 mb-3 p-3 rounded-[5px] bg-[#b45309]/10 border border-[#b45309]/25">
              <div className="type-value font-semibold text-[#b45309] mb-1">
                {t.wizard.rehearsalBannerTitle}
              </div>
              <p className="type-help text-[var(--ink-secondary)] leading-relaxed mb-2">
                {t.wizard.rehearsalBannerBody}
              </p>
              <div className="flex items-center gap-2">
                <Button size="sm" onClick={() => navigate('/system')}>
                  {t.wizard.rehearsalBannerCta}
                </Button>
                {!rehearsalAcknowledged ? (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setRehearsalAcknowledged(true)}
                  >
                    {t.wizard.rehearsalBannerAck}
                  </Button>
                ) : null}
              </div>
            </div>
          ) : null}

          {/* Action Footer */}
          <div className="p-4 border-t border-[var(--paper-border)] bg-[var(--paper-subsurface)]">
            {submitError && (
              <div className="mb-3 p-2 rounded-[5px] bg-[#b91c1c]/10 border border-[#b91c1c]/25 text-[#b91c1c] text-xs">
                {submitError}
              </div>
            )}
            {/* The submit matched an existing job, so nothing new started. Say
                that plainly and offer the way in — navigating straight to the
                job page would present an old run as this click's result. */}
            {reusedJobId && (
              <div className="mb-3 p-2 rounded-[5px] bg-[#b45309]/10 border border-[#b45309]/25 text-[#b45309] text-xs flex items-start justify-between gap-2">
                <span>{t.wizard.submitReusedNote}</span>
                <button
                  type="button"
                  onClick={() => navigate(`/jobs/${reusedJobId}`)}
                  className="shrink-0 underline hover:no-underline font-semibold cursor-pointer"
                >
                  {reusedJobId}
                </button>
              </div>
            )}
            <Button
              onClick={handleSubmit}
              disabled={!filePath.trim() || isSubmitting || glossaryError !== null || budgetInvalid}
              variant="primary"
              className="w-full h-10 text-xs font-bold tracking-tight shadow-sm"
              shortcut="⌘⏎"
            >
              {isSubmitting ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  {t.wizard.compiling}
                </>
              ) : (
                <>
                  <span>{t.wizard.startCompile}</span>
                  <ArrowRight className="h-4 w-4 ml-1.5" />
                </>
              )}
            </Button>
          </div>
        </div>
      </div>

      {/* Mobile Submit Error Alert (Phase 2) */}
      {submitError && (
        <div className="lg:hidden fixed bottom-[calc(4.75rem+env(safe-area-inset-bottom,0px))] left-0 md:left-64 right-0 mx-3 p-2.5 rounded-[5px] bg-[#b91c1c] text-white text-xs z-30 shadow-md flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 shrink-0 text-white" />
          <span className="truncate flex-1">{submitError}</span>
        </div>
      )}

      {/* Mobile Sticky Bottom Action Bar (Phase 2) */}
      <div className="lg:hidden fixed bottom-0 left-0 md:left-64 right-0 p-3 bg-[var(--paper-surface)] border-t border-[var(--paper-border)] shadow-lg z-30 pb-[calc(0.75rem+env(safe-area-inset-bottom,0px))] flex items-center justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="text-xs font-semibold text-[var(--ink-primary)] truncate font-mono">
            {fileName || (filePath ? filePath.split('/').pop() : t.wizard.waitingFile)}
          </div>
          <div className="text-xs font-mono text-[var(--ink-secondary)] flex items-center gap-2">
            {assessment?.cost.total_cost_usd !== null && assessment?.cost.total_cost_usd !== undefined ? (
              <span className="text-[#b45309] font-bold">
                ${assessment.cost.total_cost_usd.toFixed(2)}
              </span>
            ) : assessment ? (
              <span className="text-[#15803d] font-semibold">{t.common.ready}</span>
            ) : null}
            {isAssessing && (
              <span className="text-[var(--ink-muted)] flex items-center gap-1">
                <Loader2 className="h-3 w-3 animate-spin" />
                {t.wizard.compiling}
              </span>
            )}
          </div>
        </div>
        <Button
          onClick={handleSubmit}
          disabled={!filePath.trim() || isSubmitting || glossaryError !== null || budgetInvalid}
          variant="primary"
          className="h-10 px-4 text-xs font-bold tracking-tight shadow-sm shrink-0"
        >
          {isSubmitting ? (
            <>
              <Loader2 className="h-4 w-4 mr-1.5 animate-spin" />
              {t.wizard.compiling}
            </>
          ) : (
            <>
              <span>{t.wizard.startCompile}</span>
              <ArrowRight className="h-4 w-4 ml-1" />
            </>
          )}
        </Button>
      </div>
    </div>
  )
}
