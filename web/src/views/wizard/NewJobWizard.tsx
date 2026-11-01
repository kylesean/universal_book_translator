import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  UploadCloud,
  FileText,
  RefreshCw,
  ArrowRight,
  AlertTriangle,
  Loader2,
  ChevronDown,
  X,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { Alert, AlertDescription } from '@/components/ui/Alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { InfoTip } from '@/components/ui/InfoTip'
import { ConfirmDialog } from '@/components/ui/ConfirmDialog'
import { cn } from '@/lib/utils'
import {
  assessJob,
  getRuntimeConfig,
  saveEgressConfig,
  submitJob,
  uploadGlossaryFile,
  uploadSourceDocument,
  type JobAssessResponse,
  type JobSubmitRequest,
  type RuntimeConfig,
} from '@/api/client'
import { useI18n } from '@/i18n/useI18n'
import { PreflightPanel } from './PreflightPanel'

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

/**
 * Whether a numeric field holds a value the server would 422 on. Pure and
 * module-scoped: ``min={1}`` only constrains the stepper, a hand-typed 0 (or a
 * half-finished edit) still enters state, and both the assess payload and the
 * submit guard must agree on the verdict.
 */
function numericFieldInvalid(raw: string, min: number): boolean {
  if (!raw.trim()) return false // empty = follow the operator's default
  const value = Number(raw)
  return !Number.isFinite(value) || value < min
}

function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

interface ChoiceSelectProps {
  id: string
  value: string
  onValueChange: (value: string) => void
  options: ReadonlyArray<{ value: string; label: string }>
  placeholder?: string
  disabled?: boolean
}

/** A thin composition of the official shadcn Select primitives. */
function ChoiceSelect({
  id,
  value,
  onValueChange,
  options,
  placeholder,
  disabled,
}: ChoiceSelectProps) {
  return (
    <Select value={value} onValueChange={onValueChange} disabled={disabled}>
      <SelectTrigger id={id}>
        <SelectValue placeholder={placeholder} />
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

function FieldLabel({
  htmlFor,
  children,
  className,
}: {
  htmlFor?: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <Label
      htmlFor={htmlFor}
      className={cn('text-sm font-medium leading-none text-foreground', className)}
    >
      {children}
    </Label>
  )
}

function WizardSection({
  step,
  title,
  meta,
  children,
  className,
}: {
  step: number
  title: string
  meta?: React.ReactNode
  children: React.ReactNode
  className?: string
}) {
  return (
    <section
      className={cn(
        'border-t border-border/70 pt-6 first:border-t-0 first:pt-0',
        className,
      )}
    >
      <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
        <div className="flex min-w-0 items-baseline gap-3">
          <span className="font-mono text-xs font-medium text-muted-foreground tabular-nums">
            {String(step).padStart(2, '0')}
          </span>
          <h2 className="text-base font-semibold tracking-tight text-foreground">{title}</h2>
        </div>
        {meta ? <div className="max-w-full text-xs text-muted-foreground">{meta}</div> : null}
      </div>
      {children}
    </section>
  )
}

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
  const [preset, setPreset] = useState<'high_fidelity' | 'standard' | 'vector_math' | 'preview' | 'fast' | null>('standard')
  const [budgetUsd, setBudgetUsd] = useState<number | null>(null)
  // The raw field text is separate from ``budgetUsd``: while the user is
  // mid-typing ("" or "0.") the numeric value is not yet usable, but the input
  // must not snap back to a seeded default.
  const [budgetInput, setBudgetInput] = useState('')

  // Advanced compiler & layout options
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false)
  // The server-path input is a secondary affordance for files already on the
  // host; the browser picker remains the primary intake path.
  const [showServerPath, setShowServerPath] = useState(false)
  // 无有效报价时的提交确认（见 requestSubmit）。
  const [quoteRiskOpen, setQuoteRiskOpen] = useState(false)
  // 已确认过的报价指纹：同一配置确认一次即可。
  const quoteRiskAckRef = useRef('')
  const [domainProfile, setDomainProfile] = useState('auto')
  const [deliveryMode, setDeliveryMode] = useState<'auto' | 'readable' | 'faithful'>('auto')
  const [allowDegrade, setAllowDegrade] = useState(false)
  const [enableCels, setEnableCels] = useState(true)
  const [dualMode, setDualMode] = useState<'auto' | 'dual' | 'mono'>('auto')
  // 'auto' submits as null, which lets the engine align the delivered layout to
  // the source. The engine accepts only 1 or 2, so the sentinel never reaches it.
  const [layoutColumns, setLayoutColumns] = useState<'auto' | '1' | '2'>('auto')
  const [translateChrome, setTranslateChrome] = useState<'auto' | 'on' | 'off'>('auto')
  // 'auto' submits as null so the operator's UBT_OCR_MODE still applies.
  const [ocrMode, setOcrMode] = useState<'auto' | 'local' | 'cloud' | 'off'>('auto')
  const [pageRange, setPageRange] = useState('')
  // Per-book glossary: a staged upload, never a typed server path. Browsers do
  // not expose filesystem paths, so the old input asked the operator to type a
  // location they could not know; the console stages bytes like it stages the
  // source document, and the server answers with the path the job accepts.
  const [glossaryFile, setGlossaryFile] = useState<{ name: string; path: string } | null>(null)
  const [glossaryUploading, setGlossaryUploading] = useState(false)
  // Set only when a pick or an upload fails; a working flow renders no prose.
  const [glossaryError, setGlossaryError] = useState<string | null>(null)
  const glossaryFileRef = useRef<HTMLInputElement>(null)
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
  // The repair tier can run on a different block entirely — its own endpoint,
  // protocol and credential. Empty follows the primary provider.
  const [repairProvider, setRepairProvider] = useState('')
  // The operator can proceed knowingly, but only after seeing the banner. A hard
  // block would be wrong (a rehearsal is a legitimate feature); a silent one is
  // what produced a mock deliverable nobody asked for.
  const [rehearsalAcknowledged, setRehearsalAcknowledged] = useState(false)

  // One read on mount, re-read after a settings change (enabling page egress
  // must unlock the QC controls without a manual reload): whether a provider is
  // configured decides whether a submit can be real. A failure is treated as
  // "unknown" (no banner) rather than "unconfigured", so a transient error
  // cannot block a working setup.
  const refreshRuntime = useCallback(async () => {
    try {
      setRuntime(await getRuntimeConfig())
    } catch {
      setRuntime(null)
    }
  }, [])
  useEffect(() => {
    // Fetch-on-mount: the runtime verdict comes from the server, so syncing
    // with it is exactly what this effect is for.
    // oxlint-disable-next-line react/set-state-in-effect
    void refreshRuntime()
  }, [refreshRuntime])

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
  // Fingerprint of the params the LAST COMPLETED assessment priced. The
  // reactive re-assess compares the form's live fingerprint against it, so a
  // change made while a (slow, deep) assessment was in flight is re-priced
  // when things settle — the old filePath-only gate skipped those forever and
  // the operator could submit a differently-priced payload than the quote
  // shown.
  const assessedFingerprintRef = useRef<string>('')
  // The live fingerprint, kept current for completion-time rechecks (the
  // callback closing over the render that started it would see stale params).
  const formFingerprintRef = useRef<string>('')
  // True while an explicit (deep) assess owns the next fingerprint: the
  // reactive effect must not schedule a shallow re-price that would race it.
  const deepInFlightRef = useRef(false)
  const hasUserSelectedPresetRef = useRef<boolean>(false)
  const hasUserSelectedDomainRef = useRef<boolean>(false)
  const hasUserSelectedDualModeRef = useRef<boolean>(false)
  const assessSeqRef = useRef(0)
  const [assessStale, setAssessStale] = useState(false)

  const runAssess = useCallback(
    async (
      path: string,
      opts?: {
        overridePreset?: 'high_fidelity' | 'standard' | 'vector_math' | 'preview' | 'fast'
        overridePages?: string
        overrideTarget?: string
        overrideSource?: string
        silent?: boolean
      }
    ) => {
      if (!path.trim()) return
      const isSilent = opts?.silent ?? false
      const wasDeep = !isSilent
      if (wasDeep) deepInFlightRef.current = true
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
          repair_provider: repairProvider || null,
          ocr_mode: ocrMode === 'auto' ? null : ocrMode,
          visual_judge_enabled: visualJudge === 'auto' ? null : visualJudge === 'on',
          // An invalid value prices as unset — the field-level guard stops the
          // submit; the quote must not 422 on the way.
          max_concurrency:
            concurrency.trim() && !numericFieldInvalid(concurrency, 1)
              ? Number(concurrency)
              : null,
          deep: !isSilent,
        })
        if (seq !== assessSeqRef.current) return // stale response
        setAssessment(res)
        setAssessStale(false)
        // What this response actually priced — the reactive effect and the
        // completion-time recheck both compare against it.
        assessedFingerprintRef.current = [
          path.trim(),
          activePreset ?? '',
          activePages.trim(),
          activeTarget,
          activeSource,
          provider,
          repairProvider,
          ocrMode,
          visualJudge,
          concurrency.trim(),
        ].join('\u0000')

        // Auto-feed recommendations when a new document is assessed
        if (lastAssessedFileRef.current !== path.trim()) {
          lastAssessedFileRef.current = path.trim()
          if (res.document?.file_name) {
            setFileName((prev) => prev || res.document?.file_name || '')
          }
          if (res.document?.file_size_bytes) {
            setFileSize((prev) => prev || formatFileSize(res.document?.file_size_bytes ?? 0))
          }
          if (!hasUserSelectedPresetRef.current && res.route?.recommended_preset) {
            const recP = res.route.recommended_preset.toLowerCase() as any
            if (['high_fidelity', 'standard', 'vector_math', 'preview', 'fast'].includes(recP)) {
              setPreset(recP)
            }
          }
          if (!hasUserSelectedDomainRef.current && res.route?.recommended_profile) {
            const recDom = res.route.recommended_profile.toLowerCase()
            // 'semiconductor' is intentionally absent: it is a subject, not a
            // genre, and 'auto' already resolves to it server-side (the probe
            // re-runs on submit), where its built-in seed glossary applies.
            if (['auto', 'general', 'textbook', 'paper', 'fiction', 'humanities'].includes(recDom)) {
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
          setAssessment(null) // don't leave a stale quote on error
        } else {
          // A silent re-price failure must not blank the panel: the last quote
          // stays, visibly flagged as possibly outdated. Record the attempted
          // fingerprint so the drift check does not loop on the same failure.
          setAssessStale(true)
          assessedFingerprintRef.current = [
            path.trim(),
            activePreset ?? '',
            activePages.trim(),
            activeTarget,
            activeSource,
            provider,
            ocrMode,
            visualJudge,
            concurrency.trim(),
          ].join('\u0000')
        }
      } finally {
        if (seq === assessSeqRef.current) {
          setIsAssessing(false)
          setIsReassessing(false)
          if (wasDeep) deepInFlightRef.current = false
          // Params moved while this assessment was in flight: the quote that
          // just landed prices the old ones, so re-price with the current form.
          if (
            formFingerprintRef.current &&
            assessedFingerprintRef.current !== formFingerprintRef.current
          ) {
            setTimeout(() => void runAssessRef.current(path.trim(), { silent: true }), 350)
          }
        }
      }
    },
    [preset, pageRange, targetLang, sourceLang, provider, repairProvider, ocrMode, visualJudge, concurrency]
  )

  // The live fingerprint and the latest callback, read at completion time.
  const assessFingerprint = [
    filePath.trim(),
    preset ?? '',
    pageRange.trim(),
    targetLang,
    sourceLang,
    provider,
    repairProvider,
    ocrMode,
    visualJudge,
    concurrency.trim(),
  ].join('\u0000')
  const runAssessRef = useRef(runAssess)
  useEffect(() => {
    formFingerprintRef.current = assessFingerprint
    runAssessRef.current = runAssess
  })

  // Reactive re-assessment: whenever the form's pricing fingerprint drifts
  // from what the last completed assessment priced, re-price (debounced). A
  // deep assessment in flight owns the fingerprint — racing it with a shallow
  // call would discard the deep result — so drift during a deep flight is
  // handled by the completion-time recheck above instead.
  useEffect(() => {
    if (!filePath.trim()) return
    if (deepInFlightRef.current) return
    if (assessedFingerprintRef.current === assessFingerprint) return
    const timer = setTimeout(() => {
      const isFirstAssess = !lastAssessedFileRef.current || lastAssessedFileRef.current !== filePath.trim()
      void runAssessRef.current(filePath, { silent: !isFirstAssess })
    }, 450)
    return () => clearTimeout(timer)
  }, [assessFingerprint, filePath])

  // Browsers do not expose a real filesystem path (`File.path` exists only in
  // Electron), so the picked file is uploaded to the server and the staged
  // path it returns becomes the assess/submit input. A previously staged file
  // stays valid — the manual path field still allows server-side paths too.
  const handleFile = async (file: File) => {
    setFileName(file.name)
    setFileSize(formatFileSize(file.size))
    setIsUploading(true)
    setAssessError(null)
    setAssessment(null)
    setAssessStale(false)
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
    // Same guard as the drop path: a second pick while an upload is in flight
    // would race the first (last-picked name, last-finished path).
    if (isUploading) {
      e.target.value = ''
      return
    }
    if (e.target.files && e.target.files.length > 0) {
      void handleFile(e.target.files[0])
      e.target.value = '' // allow re-picking the same file after an edit
    }
  }

  // The file picker already filters by extension; this second check catches a
  // drag-typed name and keeps the honest error at the control, not the API.
  const handleGlossaryFile = async (file: File) => {
    setGlossaryError(null)
    if (!/\.(csv|tsv|json)$/i.test(file.name)) {
      setGlossaryError(t.wizard.glossaryPathInvalid)
      return
    }
    setGlossaryUploading(true)
    try {
      const staged = await uploadGlossaryFile(file)
      setGlossaryFile({ name: file.name, path: staged.path })
    } catch (err) {
      setGlossaryError(err instanceof Error ? err.message : 'Upload failed')
    } finally {
      setGlossaryUploading(false)
    }
  }

  const handleGlossaryInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (file) void handleGlossaryFile(file)
  }

  /**
   * A cap is enabled but no usable positive number was entered.
   *
   * ``JobSubmitRequest.budget_usd`` is ``Field(gt=0)``, so sending 0 (or an
   * empty field coerced to 0) is a 422 on every submission. Blocking it here
   * keeps the failure visible at the field instead of at the API boundary.
   */
  const budgetInvalid = budgetUsd === null && budgetInput.trim() !== ''
  /**
   * Same trap as the budget field, for the other numeric fields: ``min={1}``
   * only constrains the stepper, a hand-typed 0 (or a half-finished edit)
   * still enters state, and the server answers 422 for
   * ``max_concurrency<=0`` and ``start_chapter``/``max_chapters`` below 1 —
   * after the submit was already allowed.
   */
  const concurrencyInvalid = numericFieldInvalid(concurrency, 1)
  const chapterStartInvalid = numericFieldInvalid(chapterStart, 1)
  const chapterCountInvalid = numericFieldInvalid(chapterCount, 1)
  const advancedNumericInvalid =
    concurrencyInvalid || chapterStartInvalid || chapterCountInvalid
  const canSubmit =
    filePath.trim() !== '' &&
    !isSubmitting &&
    glossaryError === null &&
    !budgetInvalid &&
    !advancedNumericInvalid
  const submitBlockedReason = !filePath.trim()
    ? null
    : glossaryError ??
      (budgetInvalid
        ? t.wizard.budgetInvalid
        : advancedNumericInvalid
          ? t.wizard.fieldMinOne
          : null)
  /**
   * 报价是否存在且与当前表单一致——「先验后花」承诺的守门条件。
   *
   * 禁用态不另设文字说明：没文件时拖拽区就是说明，预算/数值无效时错误
   * 就在字段旁；唯一不可见的场景是错误藏在收起的高级抽屉里——那个信号
   * 放在抽屉的开关行上（下方高级选项的琥珀标记），而不是按钮底下。
   */
  const quoteCost = assessment?.cost.total_cost_usd ?? null
  const quoteRiskPending = assessment === null || assessStale
  /**
   * OCR modes that ship rendered page images off this machine.
   *
   * The engine refuses these unless UBT_ALLOW_PAGE_UPLOAD=true, so picking
   * one here without that flag is a submit-time error rather than a silent
   * privacy change. 'auto' can also reach the cloud, but only as a last
   * resort after local engines fail, and it warns in the log.
   */
  const ocrNeedsEgress = ocrMode === 'cloud'
  /**
   * Page egress is closed on this deployment, so neither the visual judge nor a
   * cloud/vlm OCR mode can ship a page anywhere. The judge would skip every page
   * and bill nothing, which is worse than not offering it: the switch would
   * promise a quality gate the run cannot perform.
   */
  const pageEgressClosed = runtime !== null && !runtime.active.allow_page_upload

  // The egress gate is server state, but the person who hits it is here: when
  // a control that needs page images is blocked, the note carries its own
  // enable action (confirmed once) instead of pointing at an env var. The
  // choice is a runtime override on the server — in force immediately, gone at
  // restart — and the dialog says exactly that.
  const [egressConfirmOpen, setEgressConfirmOpen] = useState(false)
  const [egressError, setEgressError] = useState<string | null>(null)

  const enablePageEgress = async () => {
    setEgressError(null)
    try {
      await saveEgressConfig({ allow_page_upload: true })
      await refreshRuntime()
      setEgressConfirmOpen(false)
    } catch (err) {
      setEgressError(err instanceof Error ? err.message : t.wizard.egressEnableFailed)
    }
  }

  const openEgressConfirm = () => {
    setEgressError(null)
    setEgressConfirmOpen(true)
  }

  /**
   * "The other edition" is only concrete once the primary form is: emit_both
   * renders the *complementary* form next to whatever dual_mode picks. The
   * help text therefore follows the delivery-mode select instead of describing
   * an abstraction.
   */
  const emitBothHelpText =
    dualMode === 'dual'
      ? t.wizard.emitBothHelpDual
      : dualMode === 'mono'
        ? t.wizard.emitBothHelpMono
        : t.wizard.emitBothHelpAuto

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
    repair_provider: repairProvider || null,
    preset: preset,
    delivery_mode: deliveryMode,
    allow_degrade: deliveryMode === 'faithful' ? allowDegrade : null,
    enable_cels: deliveryMode === 'faithful' ? enableCels : null,
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
    glossary: glossaryFile?.path ?? null,
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
    if (
      !filePath.trim() ||
      isSubmitting ||
      glossaryError ||
      budgetInvalid ||
      advancedNumericInvalid
    )
      return
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

  /**
   * 提交的守门入口：没有有效报价（体检未完成/失败，或重估价失败导致的
   * 过期报价）时先确认一次。同一个报价指纹只问一次——确认过之后，同一
   * 配置下的重复提交不再打扰；指纹变化（换文件、换配置又失败）会重新
   * 要求确认。专家快路径（⌘⏎）走同一入口，护栏不因快捷键而失效。
   */
  const requestSubmit = () => {
    if (quoteRiskPending && quoteRiskAckRef.current !== assessFingerprint) {
      setQuoteRiskOpen(true)
      return
    }
    void handleSubmit()
  }

  const handleSubmitRef = useRef(handleSubmit)
  const requestSubmitRef = useRef(requestSubmit)
  useEffect(() => {
    // Latest-ref pattern: the listener below is bound once and must call the
    // current handler, so the ref is refreshed after each commit rather
    // than written during render (which React forbids).
    handleSubmitRef.current = handleSubmit
    requestSubmitRef.current = requestSubmit
  })
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        if (filePath.trim() && !isSubmitting) {
          e.preventDefault()
          requestSubmitRef.current()
        }
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [filePath, isSubmitting])

  // Presets are user-facing quality tiers. Keep implementation choices
  // (parser, prompt depth, engine names) out of the console.
  const presetOptions = [
    {
      id: 'high_fidelity' as const,
      name: t.wizard.presetHighFidelity,
      desc: t.wizard.presetHighFidelityDesc,
      badge: t.wizard.presetBadgeBest,
      detail: t.wizard.presetHFDetail,
    },
    {
      id: 'standard' as const,
      name: t.wizard.presetStandard,
      desc: t.wizard.presetStandardDesc,
      badge: t.wizard.presetBadgeBalanced,
      detail: t.wizard.presetStandardDetail,
    },
    {
      id: 'vector_math' as const,
      name: t.wizard.presetVectorMath,
      desc: t.wizard.presetVectorMathDesc,
      badge: t.wizard.presetBadgeMath,
      detail: t.wizard.presetVectorDetail,
    },
    {
      id: 'preview' as const,
      name: t.wizard.presetPreview,
      desc: t.wizard.presetPreviewDesc,
      badge: t.wizard.presetBadgeProof,
      detail: t.wizard.presetPreviewDetail,
    },
    {
      id: 'fast' as const,
      name: t.wizard.presetFast,
      desc: t.wizard.presetFastDesc,
      badge: t.wizard.presetBadgeFast,
      detail: t.wizard.presetFastDetail,
    },
  ]

  const deliveryOptions = [
    {
      value: 'auto' as const,
      name: t.wizard.deliveryModeAuto,
      desc: t.wizard.deliveryDescAuto,
    },
    {
      value: 'faithful' as const,
      name: t.wizard.deliveryModeFaithful,
      desc: t.wizard.deliveryDescFaithful,
    },
    {
      value: 'readable' as const,
      name: t.wizard.deliveryModeReadable,
      desc: t.wizard.deliveryDescReadable,
    },
  ]

  return (
    <div
      className={cn(
        'mx-auto w-full max-w-[1500px] flex-1 overflow-y-auto px-4 py-6 sm:px-7 lg:pb-8',
        filePath.trim() && 'pb-24',
      )}
    >
      <header className="border-b border-border/70 pb-4">
        <h1 className="text-xl font-semibold tracking-tight text-foreground sm:text-2xl">
          {t.wizard.title}
        </h1>
        <p className="mt-1 max-w-2xl text-xs sm:text-sm text-muted-foreground">
          {t.wizard.subtitle}
        </p>
      </header>

      <div className="mt-6 grid grid-cols-1 items-start gap-7 xl:grid-cols-[minmax(0,1fr)_390px]">
        <div className="min-w-0 space-y-7">
          <WizardSection
            step={1}
            title={t.wizard.stepFile}
            meta={<span className="hidden font-mono sm:inline">PDF · EPUB · DOCX · MD · TXT</span>}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileInputChange}
              accept=".pdf,.epub,.docx,.md,.markdown,.txt,.html,.htm"
              aria-label={t.wizard.uploadAria}
              className="hidden"
            />

            {fileName || (filePath && filePath.trim()) ? (
              <div className="flex flex-col gap-3 rounded-xl border border-border bg-card p-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-semibold text-foreground">
                    {fileName || filePath.split('/').pop()}
                  </p>
                  <div className="mt-1 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-xs text-muted-foreground">
                    {fileSize ? <span className="font-mono">{fileSize}</span> : null}
                    {assessment?.document?.format_ext ? (
                      <span className="font-mono uppercase">
                        {assessment.document.format_ext.replace('.', '')}
                      </span>
                    ) : null}
                    <span
                      className={cn(
                        'inline-flex items-center gap-1.5 font-medium',
                        isUploading || isAssessing
                          ? 'text-muted-foreground'
                          : 'text-emerald-700 dark:text-emerald-400',
                      )}
                    >
                      {isUploading || isAssessing ? (
                        <Loader2 className="size-3 animate-spin" />
                      ) : null}
                      {isUploading
                        ? t.wizard.uploading
                        : isAssessing
                          ? t.wizard.reassessing
                          : t.common.ready}
                    </span>
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isUploading}
                  >
                    {isUploading ? (
                      <Loader2 className="mr-1.5 size-3.5 animate-spin" />
                    ) : (
                      <UploadCloud className="mr-1.5 size-3.5" />
                    )}
                    {t.wizard.orBrowse}
                  </Button>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={() => {
                      setFilePath('')
                      setFileName('')
                      setFileSize('')
                      setAssessment(null)
                      setAssessError(null)
                      setAssessStale(false)
                      lastAssessedFileRef.current = ''
                    }}
                    className="size-8 text-muted-foreground hover:text-foreground"
                    title={t.wizard.clearFile}
                    aria-label={t.wizard.clearFile}
                  >
                    <X className="size-4" />
                  </Button>
                </div>
              </div>
            ) : (
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
                onClick={() => {
                  if (!isUploading) fileInputRef.current?.click()
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault()
                    if (!isUploading) fileInputRef.current?.click()
                  }
                }}
                className={cn(
                  'group flex cursor-pointer flex-col gap-4 rounded-xl border border-dashed p-6 outline-none transition-colors focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 sm:flex-row sm:items-center sm:justify-between',
                  isDragOver
                    ? 'border-primary bg-primary/[0.04]'
                    : 'border-border bg-card hover:border-border-hover hover:bg-muted/20',
                )}
              >
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-foreground">
                    {isUploading
                      ? t.wizard.uploading
                      : isDragOver
                        ? t.wizard.dropzoneActive
                        : t.wizard.dropzoneTitle}
                  </p>
                  <p className="mt-1 max-w-xl text-xs leading-relaxed text-muted-foreground">
                    {t.wizard.dropzoneSubtitle}
                  </p>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="shrink-0"
                  disabled={isUploading}
                  onClick={(event) => {
                    event.stopPropagation()
                    fileInputRef.current?.click()
                  }}
                >
                  {isUploading ? (
                    <Loader2 className="mr-1.5 size-3.5 animate-spin" />
                  ) : (
                    <UploadCloud className="mr-1.5 size-3.5" />
                  )}
                  {t.wizard.orBrowse}
                </Button>
              </div>
            )}

            <div className="mt-2">
              <button
                type="button"
                className="inline-flex items-center gap-1.5 text-xs font-medium text-muted-foreground underline underline-offset-4 transition-colors hover:text-foreground"
                aria-expanded={showServerPath}
                onClick={() => setShowServerPath((open) => !open)}
              >
                {t.wizard.serverPathToggle}
                <ChevronDown
                  className={cn(
                    'size-3.5 transition-transform',
                    showServerPath && 'rotate-180',
                  )}
                />
              </button>
            </div>

            {showServerPath ? (
              <div className="mt-3 flex flex-col gap-2 sm:flex-row">
                <Input
                  type="text"
                  value={filePath}
                  onChange={(e) => setFilePath(e.target.value)}
                  placeholder={t.wizard.manualPathPlaceholder}
                  aria-label={t.wizard.manualPathPlaceholder}
                  className="min-w-0 flex-1"
                />
                <Button
                  type="button"
                  onClick={() => {
                    hasUserSelectedPresetRef.current = false
                    hasUserSelectedDomainRef.current = false
                    hasUserSelectedDualModeRef.current = false
                    void runAssess(filePath, { silent: false })
                  }}
                  disabled={!filePath.trim() || isAssessing}
                  variant="secondary"
                  className="shrink-0"
                >
                  {isAssessing ? (
                    <Loader2 className="mr-1.5 size-3.5 animate-spin" />
                  ) : (
                    <RefreshCw className="mr-1.5 size-3.5" />
                  )}
                  {t.wizard.assessBtn}
                </Button>
              </div>
            ) : null}

            {assessError ? (
              <Alert variant="destructive" className="mt-3">
                <AlertTriangle className="size-4" />
                <AlertDescription>{assessError}</AlertDescription>
              </Alert>
            ) : null}
          </WizardSection>

          {/* Step 2: Language & output */}
          <WizardSection
            step={2}
            title={t.wizard.stepDelivery}
            meta={
              <span className="font-mono">
                {sourceLang.toUpperCase()} → {targetLang.toUpperCase()}
              </span>
            }
          >
            <div className="space-y-3">
              <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                <div className="space-y-1.5">
                  <FieldLabel htmlFor="wizard-source-lang">{t.wizard.sourceLang}</FieldLabel>
                  <ChoiceSelect
                    id="wizard-source-lang"
                    value={sourceLang}
                    onValueChange={setSourceLang}
                    options={LANGUAGES.map((lang) => ({
                      value: lang.code,
                      label: lang.label,
                    }))}
                  />
                </div>
                <div className="space-y-1.5">
                  <FieldLabel htmlFor="wizard-target-lang">{t.wizard.targetLang}</FieldLabel>
                  <ChoiceSelect
                    id="wizard-target-lang"
                    value={targetLang}
                    onValueChange={setTargetLang}
                    options={LANGUAGES.map((lang) => ({
                      value: lang.code,
                      label: lang.label,
                    }))}
                  />
                </div>
                {declaredProviders.length > 0 ? (
                  <div className="space-y-1.5 sm:col-span-2 xl:col-span-1">
                    <div className="flex items-center gap-1.5">
                      <FieldLabel htmlFor="wizard-provider">{t.wizard.provider}</FieldLabel>
                      <InfoTip label={t.wizard.providerHelp}>{t.wizard.providerHelp}</InfoTip>
                    </div>
                    <ChoiceSelect
                      id="wizard-provider"
                      value={provider || '__default__'}
                      onValueChange={(value) =>
                        setProvider(value === '__default__' ? '' : value)
                      }
                      options={[
                        {
                          value: '__default__',
                          label: runtime?.active.provider
                            ? `${t.wizard.providerDefault} (${runtime.active.provider})`
                            : t.wizard.providerDefault,
                        },
                        ...declaredProviders.map((p) => ({
                          value: p.name,
                          label: p.name + (p.draft_model ? ` — ${p.draft_model}` : ''),
                        })),
                      ]}
                    />
                    {selectedProviderDetail ? (
                      <p className="truncate font-mono text-xs text-muted-foreground">
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
              </div>
              {runtime !== null && !runtime.active.can_translate ? (
                <p className="text-xs leading-relaxed text-amber-700 dark:text-amber-400">
                  {t.wizard.providerUnavailable} {t.wizard.providerUnavailableHelp}
                </p>
              ) : null}
            </div>

            <RadioGroup
              value={deliveryMode}
              onValueChange={(value) => {
                setDeliveryMode(value as typeof deliveryMode)
              }}
              className="mt-5 grid gap-2 md:grid-cols-3"
            >
              {deliveryOptions.map((opt) => {
                const isSelected = deliveryMode === opt.value
                return (
                  <label
                    key={opt.value}
                    htmlFor={`delivery-${opt.value}`}
                    className={cn(
                      'flex cursor-pointer flex-col gap-2 rounded-xl border p-3.5 transition-colors',
                      isSelected
                        ? 'border-primary/35 bg-primary/[0.045]'
                        : 'border-border bg-card hover:border-border-hover hover:bg-muted/20',
                    )}
                  >
                    <div className="flex items-start gap-2">
                      <RadioGroupItem
                        id={`delivery-${opt.value}`}
                        value={opt.value}
                        className="mt-0.5 shrink-0"
                      />
                      <span className="text-sm font-semibold leading-tight text-foreground">
                        {opt.name}
                      </span>
                    </div>
                    <p className="pl-6 text-xs leading-relaxed text-muted-foreground">
                      {opt.desc}
                    </p>
                  </label>
                )
              })}
            </RadioGroup>

            {deliveryMode === 'faithful' ? (
              <div className="mt-3 rounded-xl border border-primary/20 bg-primary/[0.02] p-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h3 className="text-sm font-semibold text-foreground">
                    {t.wizard.faithfulShowcaseTitle}
                  </h3>
                  <div className="flex flex-wrap gap-1.5">
                    {[
                      t.wizard.faithfulFeatureStretch,
                      t.wizard.faithfulFeatureBooktabs,
                      t.wizard.faithfulFeatureFlow,
                      t.wizard.faithfulFeatureAnchor,
                    ].map((feat) => (
                      <Badge key={feat} variant="secondary" className="text-[11px] font-normal">
                        {feat}
                      </Badge>
                    ))}
                  </div>
                </div>

                <div className="mt-3 grid gap-3 border-t border-border/60 pt-3 sm:grid-cols-2">
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-1.5 min-w-0">
                      <Label
                        htmlFor="inline-allow-degrade"
                        className="cursor-pointer text-xs font-medium text-foreground truncate"
                      >
                        {t.wizard.allowDegrade}
                      </Label>
                      <InfoTip label={t.wizard.allowDegrade}>
                        {t.wizard.allowDegradeHelp}
                      </InfoTip>
                    </div>
                    <Switch
                      id="inline-allow-degrade"
                      checked={allowDegrade}
                      onCheckedChange={setAllowDegrade}
                    />
                  </div>

                  <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-1.5 min-w-0">
                      <Label
                        htmlFor="inline-enable-cels"
                        className="cursor-pointer text-xs font-medium text-foreground truncate"
                      >
                        {t.wizard.enableCels}
                      </Label>
                      <InfoTip label={t.wizard.enableCels}>{t.wizard.enableCelsHelp}</InfoTip>
                    </div>
                    <Switch
                      id="inline-enable-cels"
                      checked={enableCels}
                      onCheckedChange={setEnableCels}
                    />
                  </div>
                </div>

                {preset !== 'high_fidelity' ? (
                  <div className="mt-3 flex items-center justify-between gap-2 rounded-lg bg-amber-500/10 px-3 py-2 text-xs text-amber-900 dark:text-amber-100">
                    <span className="leading-tight">
                      {t.wizard.faithfulPresetRecommendation}
                    </span>
                    <Button
                      type="button"
                      size="sm"
                      variant="link"
                      onClick={() => {
                        hasUserSelectedPresetRef.current = true
                        setPreset('high_fidelity')
                      }}
                      className="h-auto shrink-0 p-0 text-xs font-semibold text-amber-900 underline underline-offset-2 dark:text-amber-100"
                    >
                      {t.wizard.faithfulPresetSwitchAction}
                    </Button>
                  </div>
                ) : null}
              </div>
            ) : null}
          </WizardSection>

          {/* Step 3: Quality presets */}
          <WizardSection step={3} title={t.wizard.stepQuality}>
            <RadioGroup
              value={preset ?? undefined}
              onValueChange={(value) => {
                hasUserSelectedPresetRef.current = true
                setPreset(value as typeof preset)
              }}
              className="grid gap-2.5 sm:grid-cols-2 xl:grid-cols-3"
            >
              {presetOptions.map((p) => {
                const isSelected = preset === p.id
                const isRecommended =
                  assessment?.route?.recommended_preset?.toLowerCase() === p.id
                return (
                  <label
                    key={p.id}
                    htmlFor={`preset-${p.id}`}
                    className={cn(
                      'relative flex cursor-pointer flex-col justify-between rounded-xl border p-3.5 transition-colors',
                      isSelected
                        ? 'border-primary/45 bg-primary/[0.045] shadow-xs'
                        : 'border-border bg-card hover:border-border-hover hover:bg-muted/20',
                    )}
                  >
                    <div>
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-center gap-2 min-w-0">
                          <RadioGroupItem
                            id={`preset-${p.id}`}
                            value={p.id}
                            className="shrink-0"
                          />
                          <span className="text-sm font-semibold text-foreground truncate">
                            {p.name}
                          </span>
                        </div>
                        {isRecommended ? (
                          <Badge
                            variant="outline"
                            className="shrink-0 border-emerald-600/25 bg-emerald-500/10 text-[11px] font-normal text-emerald-700 dark:text-emerald-400"
                          >
                            {t.wizard.engineRecommended}
                          </Badge>
                        ) : (
                          <Badge variant="secondary" className="shrink-0 text-[11px] font-normal">
                            {p.badge}
                          </Badge>
                        )}
                      </div>
                      <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                        {p.desc}
                      </p>
                    </div>
                  </label>
                )
              })}
            </RadioGroup>
          </WizardSection>

          {/* Step 4: Budget */}
          <WizardSection step={4} title={t.wizard.stepBudget}>
            <div className="rounded-xl border border-border bg-card p-3.5 sm:p-4">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2">
                    <p className="text-sm font-semibold text-foreground">{t.wizard.budgetCeiling}</p>
                    <span className="text-xs text-muted-foreground">
                      · {budgetUsd === null ? t.wizard.budgetUnlimited : t.wizard.budgetHelp}
                    </span>
                  </div>
                </div>

                <div className="flex shrink-0 items-center gap-3">
                  {budgetUsd === null ? (
                    <div className="flex items-center gap-2">
                      {quoteCost !== null && quoteCost > 0 ? (
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          onClick={() => {
                            const value = Math.ceil(quoteCost * 1.5 * 100) / 100
                            setBudgetInput(value.toFixed(2))
                            setBudgetUsd(value)
                          }}
                          className="h-8 font-mono text-xs"
                        >
                          {t.wizard.budgetSuggest.replace(
                            '{usd}',
                            (Math.ceil(quoteCost * 1.5 * 100) / 100).toFixed(2),
                          )}
                        </Button>
                      ) : null}
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          const seeded = 10
                          setBudgetInput(String(seeded))
                          setBudgetUsd(seeded)
                        }}
                        className="h-8 text-xs font-medium"
                      >
                        设置限额
                      </Button>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2">
                      <div className="relative w-28">
                        <span className="absolute left-2.5 top-1/2 -translate-y-1/2 font-mono text-xs text-muted-foreground">
                          $
                        </span>
                        <Input
                          type="number"
                          min="0.01"
                          step="0.01"
                          value={budgetInput}
                          onChange={(event) => {
                            const raw = event.target.value
                            setBudgetInput(raw)
                            const parsed = parseFloat(raw)
                            setBudgetUsd(Number.isFinite(parsed) && parsed > 0 ? parsed : null)
                          }}
                          placeholder={t.wizard.budgetPlaceholder}
                          className={cn(
                            'h-8 pl-6 pr-2 text-xs font-mono',
                            budgetInvalid && 'border-destructive focus-visible:ring-destructive',
                          )}
                        />
                      </div>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          setBudgetUsd(null)
                          setBudgetInput('')
                        }}
                        className="h-8 text-xs text-muted-foreground hover:text-foreground"
                      >
                        {t.wizard.budgetUnlimitedToggle}
                      </Button>
                    </div>
                  )}
                </div>
              </div>

              {budgetInvalid ? (
                <p className="mt-2 text-xs text-destructive">{t.wizard.budgetInvalid}</p>
              ) : budgetUsd !== null && assessment?.cost.total_cost_usd != null &&
                assessment.cost.total_cost_usd > budgetUsd ? (
                <p className="mt-2 text-xs leading-relaxed text-amber-700 dark:text-amber-400">
                  {t.wizard.budgetExceedsQuote} ($
                  {assessment.cost.total_cost_usd.toFixed(2)})
                </p>
              ) : null}
            </div>
          </WizardSection>

          {/* Optional controls, grouped so the default path stays quiet. */}
          <Collapsible open={isAdvancedOpen} onOpenChange={setIsAdvancedOpen}>
            <div className="rounded-xl border border-border bg-card px-4">
              <CollapsibleTrigger asChild>
                <button
                  type="button"
                  className="group flex w-full items-center justify-between py-3.5 text-left"
                >
                  <div className="flex flex-wrap items-center gap-2.5">
                    <span className="text-sm font-semibold text-foreground">
                      {t.wizard.advancedOptionsTitle}
                    </span>
                    <span className="text-xs text-muted-foreground">
                      {t.wizard.advancedOptionsSubtitle}
                    </span>
                    {(advancedNumericInvalid || glossaryError) && (
                      <Badge variant="destructive" className="gap-1">
                        <AlertTriangle className="size-3" />
                        {t.wizard.advancedErrorTag}
                      </Badge>
                    )}
                  </div>
                  <ChevronDown
                    className={cn(
                      'size-4 text-muted-foreground transition-transform',
                      isAdvancedOpen && 'rotate-180'
                    )}
                  />
                </button>
              </CollapsibleTrigger>

              <CollapsibleContent className="border-t border-border py-4">
                <Tabs defaultValue="delivery">
                  <TabsList className="grid h-auto w-full grid-cols-2 gap-1 p-1 sm:grid-cols-4">
                    <TabsTrigger value="delivery">{t.wizard.groupDelivery}</TabsTrigger>
                    <TabsTrigger value="content">{t.wizard.groupContent}</TabsTrigger>
                    <TabsTrigger value="scope">{t.wizard.groupScope}</TabsTrigger>
                    <TabsTrigger value="run">{t.wizard.groupExec}</TabsTrigger>
                  </TabsList>

                  <TabsContent value="delivery" className="space-y-4 pt-3">
                    <div className="grid gap-4 sm:grid-cols-2">
                      <div className="space-y-1.5">
                        <div className="flex items-center gap-1.5">
                          <FieldLabel htmlFor="wizard-dual-mode">{t.wizard.dualMode}</FieldLabel>
                          <InfoTip label={t.wizard.dualMode}>{t.wizard.dualModeHelp}</InfoTip>
                        </div>
                        <ChoiceSelect
                          id="wizard-dual-mode"
                          value={dualMode}
                          onValueChange={(value) => {
                            hasUserSelectedDualModeRef.current = true
                            setDualMode(value as typeof dualMode)
                          }}
                          options={[
                            { value: 'auto', label: t.wizard.dualModeAuto },
                            { value: 'dual', label: t.wizard.dualModeDual },
                            { value: 'mono', label: t.wizard.dualModeMono },
                          ]}
                        />
                      </div>
                      <div className="space-y-1.5">
                        <div className="flex items-center gap-1.5">
                          <FieldLabel htmlFor="wizard-emit-both">{t.wizard.emitBoth}</FieldLabel>
                          <InfoTip label={t.wizard.emitBoth}>{emitBothHelpText}</InfoTip>
                        </div>
                        <ChoiceSelect
                          id="wizard-emit-both"
                          value={emitBoth}
                          onValueChange={(value) => setEmitBoth(value as typeof emitBoth)}
                          options={[
                            { value: 'auto', label: t.wizard.emitBothAuto },
                            { value: 'on', label: t.wizard.emitBothOn },
                            { value: 'off', label: t.wizard.emitBothOff },
                          ]}
                        />
                      </div>
                      <div className="space-y-1.5">
                        <div className="flex items-center gap-1.5">
                          <FieldLabel htmlFor="wizard-layout-columns">{t.wizard.layoutColumns}</FieldLabel>
                          <InfoTip label={t.wizard.layoutColumns}>{t.wizard.layoutColumnsHelp}</InfoTip>
                        </div>
                        <ChoiceSelect
                          id="wizard-layout-columns"
                          value={layoutColumns}
                          onValueChange={(value) =>
                            setLayoutColumns(value as typeof layoutColumns)
                          }
                          options={[
                            { value: 'auto', label: t.wizard.layoutColumnsAuto },
                            { value: '1', label: t.wizard.layoutColumnsOne },
                            { value: '2', label: t.wizard.layoutColumnsTwo },
                          ]}
                        />
                      </div>
                      <div className="space-y-1.5">
                        <div className="flex items-center gap-1.5">
                          <FieldLabel htmlFor="wizard-translate-chrome">{t.wizard.translateChrome}</FieldLabel>
                          <InfoTip label={t.wizard.translateChrome}>{t.wizard.translateChromeHelp}</InfoTip>
                        </div>
                        <ChoiceSelect
                          id="wizard-translate-chrome"
                          value={translateChrome}
                          onValueChange={(value) =>
                            setTranslateChrome(value as typeof translateChrome)
                          }
                          options={[
                            { value: 'auto', label: t.wizard.translateChromeAuto },
                            { value: 'on', label: t.wizard.translateChromeOn },
                            { value: 'off', label: t.wizard.translateChromeOff },
                          ]}
                        />
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="content" className="space-y-4 pt-3">
                    <div className="grid gap-4 sm:grid-cols-2">
                      <div className="space-y-1.5">
                        <div className="flex items-center gap-1.5">
                          <FieldLabel htmlFor="wizard-domain-profile">{t.wizard.domainProfile}</FieldLabel>
                          <InfoTip label={t.wizard.domainProfile}>{t.wizard.domainProfileHelp}</InfoTip>
                        </div>
                        <ChoiceSelect
                          id="wizard-domain-profile"
                          value={domainProfile}
                          onValueChange={(value) => {
                            hasUserSelectedDomainRef.current = true
                            setDomainProfile(value)
                          }}
                          options={[
                            { value: 'auto', label: t.wizard.domainAuto },
                            { value: 'general', label: t.wizard.domainGeneral },
                            { value: 'textbook', label: t.wizard.domainTextbook },
                            { value: 'paper', label: t.wizard.domainPaper },
                            { value: 'fiction', label: t.wizard.domainFiction },
                            { value: 'humanities', label: t.wizard.domainHumanities },
                          ]}
                        />
                      </div>
                      <div className="space-y-1.5">
                        <div className="flex items-center gap-1.5">
                          <FieldLabel htmlFor="wizard-domain-hint">{t.wizard.domainHint}</FieldLabel>
                          <InfoTip label={t.wizard.domainHint}>{t.wizard.domainHintHelp}</InfoTip>
                        </div>
                        <Input
                          id="wizard-domain-hint"
                          value={domainHint}
                          onChange={(event) => setDomainHint(event.target.value)}
                        />
                      </div>
                      <div className="space-y-1.5">
                        <div className="flex items-center gap-1.5">
                          <FieldLabel htmlFor="wizard-ocr-mode">{t.wizard.ocrMode}</FieldLabel>
                          <InfoTip label={t.wizard.ocrMode}>{t.wizard.ocrModeHelp}</InfoTip>
                        </div>
                        <ChoiceSelect
                          id="wizard-ocr-mode"
                          value={ocrMode}
                          onValueChange={(value) => setOcrMode(value as typeof ocrMode)}
                          options={[
                            { value: 'auto', label: t.wizard.ocrModeAuto },
                            { value: 'local', label: t.wizard.ocrModeLocal },
                            { value: 'cloud', label: t.wizard.ocrModeCloud },
                            { value: 'off', label: t.wizard.ocrModeOff },
                          ]}
                        />
                        {ocrNeedsEgress && (
                          <p className="text-xs text-amber-700 dark:text-amber-400">
                            {t.wizard.ocrModeEgressWarning}
                          </p>
                        )}
                        {ocrNeedsEgress && pageEgressClosed && (
                          <Button
                            type="button"
                            variant="link"
                            size="sm"
                            onClick={openEgressConfirm}
                            className="h-auto text-xs"
                          >
                            {t.wizard.egressEnableBtn}
                          </Button>
                        )}
                      </div>
                      <div className="space-y-1.5">
                        <div className="flex items-center gap-1.5">
                          <FieldLabel>{t.wizard.glossaryPath}</FieldLabel>
                          <InfoTip label={t.wizard.glossaryPath}>{t.wizard.glossaryPathHelp}</InfoTip>
                        </div>
                        <input
                          ref={glossaryFileRef}
                          type="file"
                          accept=".csv,.tsv,.json"
                          className="hidden"
                          aria-label={t.wizard.glossaryPath}
                          onChange={handleGlossaryInputChange}
                        />
                        {glossaryFile ? (
                          <div className="flex items-center gap-2">
                            <div className="flex h-9 min-w-0 flex-1 items-center gap-2 rounded-lg border border-border bg-muted/35 px-3 text-xs">
                              <FileText className="size-3.5 shrink-0 text-muted-foreground" />
                              <span className="truncate">{glossaryFile.name}</span>
                            </div>
                            <Button
                              type="button"
                              variant="outline"
                              size="icon"
                              onClick={() => setGlossaryFile(null)}
                              aria-label={t.wizard.glossaryRemove}
                            >
                              <X />
                            </Button>
                          </div>
                        ) : (
                          <div className="flex flex-wrap items-center gap-3">
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              disabled={glossaryUploading}
                              onClick={() => glossaryFileRef.current?.click()}
                            >
                              {glossaryUploading ? (
                                <Loader2 className="animate-spin" />
                              ) : (
                                <UploadCloud />
                              )}
                              {t.wizard.glossaryUploadBtn}
                            </Button>
                            <Link
                              to="/assets"
                              className="text-xs text-muted-foreground underline underline-offset-2 hover:text-foreground"
                            >
                              {t.wizard.glossaryManageLink}
                            </Link>
                          </div>
                        )}
                        {glossaryError && (
                          <p className="text-xs text-destructive">{glossaryError}</p>
                        )}
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="scope" className="space-y-4 pt-3">
                    <div className="grid gap-4 sm:grid-cols-2">
                      <div className="space-y-1.5">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-1.5">
                            <FieldLabel htmlFor="wizard-page-range">{t.wizard.pageFilter}</FieldLabel>
                            <InfoTip label={t.wizard.pageFilter}>{t.wizard.pageFilterHelp}</InfoTip>
                          </div>
                          <Button
                            type="button"
                            variant="link"
                            size="sm"
                            onClick={() => setPageRange('1-5')}
                            className="h-auto text-xs"
                          >
                            {t.wizard.sliceSampleQuickBtn}
                          </Button>
                        </div>
                        <Input
                          id="wizard-page-range"
                          value={pageRange}
                          onChange={(event) => setPageRange(event.target.value)}
                          placeholder={t.wizard.pageFilterPlaceholder}
                        />
                      </div>
                      <div className="space-y-1.5">
                        <div className="flex items-center gap-1.5">
                          <FieldLabel>{t.wizard.chapterWindow}</FieldLabel>
                          <InfoTip label={t.wizard.chapterWindow}>{t.wizard.chapterWindowHelp}</InfoTip>
                        </div>
                        <div className="grid grid-cols-2 gap-2">
                          <Input
                            type="number"
                            min={1}
                            value={chapterStart}
                            onChange={(event) => setChapterStart(event.target.value)}
                            placeholder={t.wizard.chapterStart}
                            aria-label={t.wizard.chapterStart}
                          />
                          <Input
                            type="number"
                            min={1}
                            value={chapterCount}
                            onChange={(event) => setChapterCount(event.target.value)}
                            placeholder={t.wizard.chapterCount}
                            aria-label={t.wizard.chapterCount}
                          />
                        </div>
                        {(chapterStartInvalid || chapterCountInvalid) && (
                          <p className="text-xs text-destructive">{t.wizard.fieldMinOne}</p>
                        )}
                      </div>
                    </div>
                    <div className="flex items-center justify-between rounded-lg border border-border bg-muted/20 px-3 py-2.5">
                      <div className="flex items-center gap-1.5">
                        <FieldLabel htmlFor="wizard-dry-run" className="cursor-pointer">
                          {t.wizard.rehearsalLabel}
                        </FieldLabel>
                        <InfoTip label={t.wizard.rehearsalLabel}>{t.wizard.rehearsalHelp}</InfoTip>
                      </div>
                      <Switch id="wizard-dry-run" checked={dryRun} onCheckedChange={setDryRun} />
                    </div>
                    {dryRun && (
                      <Alert variant="warning">
                        <AlertTriangle className="size-4" />
                        <AlertDescription>{t.wizard.rehearsalActiveWarning}</AlertDescription>
                      </Alert>
                    )}
                  </TabsContent>

                  <TabsContent value="run" className="space-y-4 pt-3">
                    <div className="grid gap-4 sm:grid-cols-2">
                      {declaredProviders.length > 0 ? (
                        <div className="space-y-1.5">
                          <div className="flex items-center gap-1.5">
                            <FieldLabel htmlFor="wizard-repair-provider">
                              {t.wizard.repairProvider}
                            </FieldLabel>
                            <InfoTip label={t.wizard.repairProvider}>
                              {t.wizard.repairProviderHelp}
                            </InfoTip>
                          </div>
                          <ChoiceSelect
                            id="wizard-repair-provider"
                            value={repairProvider || '__follow__'}
                            onValueChange={(value) =>
                              setRepairProvider(value === '__follow__' ? '' : value)
                            }
                            options={[
                              {
                                value: '__follow__',
                                label: t.wizard.repairProviderDefault,
                              },
                              ...declaredProviders.map((p) => ({
                                value: p.name,
                                label: p.name + (p.draft_model ? ` — ${p.draft_model}` : ''),
                              })),
                            ]}
                          />
                        </div>
                      ) : null}
                      <div className="space-y-1.5">
                        <div className="flex items-center gap-1.5">
                          <FieldLabel>{t.wizard.concurrency}</FieldLabel>
                          <InfoTip label={t.wizard.concurrency}>{t.wizard.concurrencyHelp}</InfoTip>
                        </div>
                        <Input
                          type="number"
                          min={1}
                          value={concurrency}
                          onChange={(event) => setConcurrency(event.target.value)}
                          placeholder={t.wizard.concurrencyAuto}
                        />
                        {concurrencyInvalid && (
                          <p className="text-xs text-destructive">{t.wizard.fieldMinOne}</p>
                        )}
                      </div>
                      <div className="space-y-1.5">
                        <div className="flex items-center gap-1.5">
                          <FieldLabel htmlFor="wizard-visual-judge">{t.wizard.visualJudge}</FieldLabel>
                          <InfoTip label={t.wizard.visualJudge}>{t.wizard.visualJudgeHelp}</InfoTip>
                        </div>
                        <ChoiceSelect
                          id="wizard-visual-judge"
                          value={visualJudge}
                          onValueChange={(value) => setVisualJudge(value as typeof visualJudge)}
                          disabled={pageEgressClosed}
                          options={[
                            { value: 'auto', label: t.wizard.toggleAuto },
                            { value: 'on', label: t.wizard.toggleOn },
                            { value: 'off', label: t.wizard.toggleOff },
                          ]}
                        />
                        {pageEgressClosed && (
                          <div className="flex items-center justify-between gap-2 rounded-lg border border-border bg-muted/30 px-3 py-2 text-xs">
                            <span className="text-muted-foreground">
                              {t.wizard.visualJudgeBlocked}
                            </span>
                            <Button
                              type="button"
                              variant="link"
                              size="sm"
                              onClick={openEgressConfirm}
                              className="h-auto shrink-0 text-xs"
                            >
                              {t.wizard.egressEnableBtn}
                            </Button>
                          </div>
                        )}
                      </div>
                    </div>
                  </TabsContent>
                </Tabs>
              </CollapsibleContent>
            </div>
          </Collapsible>
        </div>

        <PreflightPanel
          assessment={assessment}
          assessStale={assessStale}
          canSubmit={canSubmit}
          deliveryMode={deliveryMode}
          isReassessing={isReassessing}
          isSubmitting={isSubmitting}
          onAcknowledgeRehearsal={() => setRehearsalAcknowledged(true)}
          onOpenReusedJob={() => {
            if (reusedJobId) navigate(`/jobs/${reusedJobId}`)
          }}
          onOpenSystem={() => navigate('/system')}
          onRequestSubmit={requestSubmit}
          rehearsalAcknowledged={rehearsalAcknowledged}
          rehearsalOnly={rehearsalOnly}
          reusedJobId={reusedJobId}
          submitBlockedReason={submitBlockedReason}
          submitError={submitError}
        />
      </div>

      {/* Submit guard dialog */}
      <ConfirmDialog
        open={quoteRiskOpen}
        title={t.wizard.quoteRiskTitle}
        body={
          <span className="block space-y-1.5">
            <span className="block">
              {assessStale ? t.wizard.quoteRiskBodyStale : t.wizard.quoteRiskBodyNoQuote}
            </span>
            {budgetUsd === null && quoteCost !== null && quoteCost > 0 && (
              <span className="block font-medium">
                {t.wizard.quoteRiskUnlimited.replace('{usd}', quoteCost.toFixed(2))}
              </span>
            )}
          </span>
        }
        confirmLabel={t.wizard.quoteRiskConfirm}
        cancelLabel={t.wizard.quoteRiskCancel}
        onConfirm={() => {
          quoteRiskAckRef.current = assessFingerprint
          setQuoteRiskOpen(false)
          void handleSubmit()
        }}
        onCancel={() => setQuoteRiskOpen(false)}
      />

      {/* Page egress dialog */}
      <ConfirmDialog
        open={egressConfirmOpen}
        title={t.wizard.egressConfirmTitle}
        body={
          <span className="block space-y-1.5">
            <span className="block">{t.wizard.egressConfirmBody}</span>
            {egressError && <span className="block text-destructive">{egressError}</span>}
          </span>
        }
        confirmLabel={t.wizard.egressConfirmOn}
        cancelLabel={t.wizard.egressConfirmCancel}
        onConfirm={() => void enablePageEgress()}
        onCancel={() => setEgressConfirmOpen(false)}
      />

      {/* Mobile Submit Error Alert */}
      {submitError && (
        <div className="lg:hidden fixed bottom-[calc(4.75rem+env(safe-area-inset-bottom,0px))] left-0 md:left-64 right-0 mx-3 z-30 shadow-md">
          <Alert variant="destructive">
            <AlertTriangle className="size-4 shrink-0" />
            <AlertDescription className="truncate">{submitError}</AlertDescription>
          </Alert>
        </div>
      )}

      {/* Mobile sticky action appears only once there is something to submit. */}
      {filePath.trim() ? (
      <div className="fixed inset-x-0 bottom-0 z-30 flex items-center justify-between gap-3 border-t border-border bg-card p-3 pb-[calc(0.75rem+env(safe-area-inset-bottom,0px))] shadow-lg lg:hidden">
        <div className="min-w-0 flex-1">
          <div className="text-xs font-semibold text-foreground truncate">
            {fileName || (filePath ? filePath.split('/').pop() : t.wizard.waitingFile)}
          </div>
          <div className="text-xs font-mono text-muted-foreground flex items-center gap-2">
            {assessment?.cost.total_cost_usd !== null && assessment?.cost.total_cost_usd !== undefined ? (
              <span className="text-amber-600 dark:text-amber-500 font-bold">
                ${assessment.cost.total_cost_usd.toFixed(2)}
              </span>
            ) : assessment ? (
              <span className="text-emerald-600 dark:text-emerald-400 font-semibold">{t.common.ready}</span>
            ) : null}
            {isAssessing && (
              <span className="text-muted-foreground flex items-center gap-1">
                <Loader2 className="size-3 animate-spin" />
                {t.wizard.compiling}
              </span>
            )}
          </div>
        </div>
        <Button
          onClick={requestSubmit}
          disabled={!canSubmit}
          variant="default"
          className="h-10 px-4 text-xs font-semibold shadow-xs shrink-0"
        >
          {isSubmitting ? (
            <>
              <Loader2 className="size-4 mr-1.5 animate-spin" />
              {t.wizard.compiling}
            </>
          ) : (
            <>
              <span>{t.wizard.startCompile}</span>
              <ArrowRight className="size-4 ml-1" />
            </>
          )}
        </Button>
      </div>
      ) : null}
    </div>
  )
}
