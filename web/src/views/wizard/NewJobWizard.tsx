import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import {
  UploadCloud,
  FileText,
  RefreshCw,
  ArrowRight,
  AlertTriangle,
  Loader2,
  Sliders,
  Globe,
  ChevronDown,
  ChevronRight,
  X,
} from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { InfoTip } from '@/components/ui/InfoTip'
import { ConfirmDialog } from '@/components/ui/ConfirmDialog'
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
 * The route badge's short localized labels. The engine emits raw enums
 * (`recommended_dual_mode` mono/dual, `recommended_preset`
 * high_fidelity/standard); an unknown value still shows itself so a future
 * preset renders as its own name rather than "undefined".
 */
function routeBadge(
  dual: string,
  preset: string,
  t: ReturnType<typeof useI18n>['t'],
): string {
  const duals: Record<string, string> = { mono: t.wizard.routeMono, dual: t.wizard.routeDual }
  const presets: Record<string, string> = {
    high_fidelity: t.wizard.routeHighFidelity,
    standard: t.wizard.routeStandard,
  }
  return `${duals[dual] ?? dual} · ${presets[preset] ?? preset}`
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
  const [preset, setPreset] = useState<'high_fidelity' | 'standard' | 'vector_math' | 'preview' | 'fast' | null>(null)
  const [budgetUsd, setBudgetUsd] = useState<number | null>(null)
  // The raw field text is separate from ``budgetUsd``: while the user is
  // mid-typing ("" or "0.") the numeric value is not yet usable, but the input
  // must not snap back to a seeded default.
  const [budgetInput, setBudgetInput] = useState('')

  // Advanced compiler & layout options
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false)
  // 提供商与语言：默认值已经足够好（默认提供商、en→zh），折叠不影响
  // 正确性——摘要徽章让收起态仍有信息量。
  const [basicsOpen, setBasicsOpen] = useState(false)
  // 无有效报价时的提交确认（见 requestSubmit）。
  const [quoteRiskOpen, setQuoteRiskOpen] = useState(false)
  // 已确认过的报价指纹：同一配置确认一次即可。
  const quoteRiskAckRef = useRef('')
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
          ocrMode,
          visualJudge,
          concurrency.trim(),
        ].join('\u0000')

        // Auto-feed recommendations when a new document is assessed
        if (lastAssessedFileRef.current !== path.trim()) {
          lastAssessedFileRef.current = path.trim()
          if (res.document?.file_name && !fileName) {
            setFileName(res.document.file_name)
          }
          if (res.document?.file_size_bytes && !fileSize) {
            const sz = res.document.file_size_bytes
            setFileSize(sz > 1024 * 1024 ? `${(sz / (1024 * 1024)).toFixed(1)} MB` : `${(sz / 1024).toFixed(0)} KB`)
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
    [preset, pageRange, targetLang, sourceLang, provider, ocrMode, visualJudge, concurrency]
  )

  // The live fingerprint and the latest callback, read at completion time.
  const assessFingerprint = [
    filePath.trim(),
    preset ?? '',
    pageRange.trim(),
    targetLang,
    sourceLang,
    provider,
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
    setFileSize((file.size / (1024 * 1024)).toFixed(1) + ' MB')
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
  const ocrNeedsEgress = ocrMode === 'cloud' || ocrMode === 'vlm'
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

  // Keyboard shortcut: ⌘⏎ or Ctrl+Enter to compile
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

  // The badge names the engine parameters that actually tell one preset apart
  // from another (ubt/core/presets.py): the PDF parser and the prompt depth.
  // It deliberately does NOT name a model (a preset never picks one — the
  // router does) and does NOT claim a quality "gate" (a preset never changes
  // one; presets.py says so in its header). It also does not claim formula
  // vectorisation outside vector_math: ``auto`` formula enrichment crops
  // the source region, so high_fidelity and standard render identical math.
  const presetOptions = [
    {
      id: 'high_fidelity' as const,
      name: t.wizard.presetHighFidelity,
      desc: t.wizard.presetHighFidelityDesc,
      badge: t.wizard.presetBadgeRich,
      detail: t.wizard.presetHFDetail,
    },
    {
      id: 'standard' as const,
      name: t.wizard.presetStandard,
      desc: t.wizard.presetStandardDesc,
      badge: t.wizard.presetBadgeAdaptive,
      detail: t.wizard.presetStandardDetail,
    },
    {
      id: 'vector_math' as const,
      name: t.wizard.presetVectorMath,
      desc: t.wizard.presetVectorMathDesc,
      badge: t.wizard.presetBadgeVectorMath,
      detail: t.wizard.presetVectorDetail,
      // The only preset that pays for VLM transcription; flagged so the cost
      // jump is visible before submission rather than after.
      slow: true,
    },
    {
      id: 'preview' as const,
      name: t.wizard.presetPreview,
      desc: t.wizard.presetPreviewDesc,
      badge: t.wizard.presetBadgeFormulaImage,
      detail: t.wizard.presetPreviewDetail,
    },
    {
      id: 'fast' as const,
      name: t.wizard.presetFast,
      desc: t.wizard.presetFastDesc,
      badge: t.wizard.presetBadgeShortChain,
      detail: t.wizard.presetFastDetail,
    },
  ]

  // 工作台页（双栏：表单 + 报告）限宽 1500px，比全站阅读页（max-w-5xl/6xl）
  // 宽：宽屏上的多余空间应该喂给表单列与报告列，而不是留成两侧空白；字段的
  // 行宽仍由列宽（约 840px）自然约束。
  return (
    <div className="flex-1 overflow-y-auto px-4 sm:px-8 py-5 sm:py-7 space-y-5 sm:space-y-7 max-w-[1500px] mx-auto w-full pb-24 lg:pb-7">
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
            <div className="type-step">{t.wizard.stepFile}</div>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileInputChange}
              accept=".pdf,.epub,.docx,.md,.markdown,.txt,.html,.htm"
              aria-label={t.wizard.uploadAria}
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
              onClick={() => {
                // Guarded like the drop path: re-opening the picker mid-upload
                // races two staged files (name from the last pick, path from
                // the last upload to finish).
                if (!isUploading) fileInputRef.current?.click()
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault()
                  if (!isUploading) fileInputRef.current?.click()
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
                aria-label={t.wizard.manualPathPlaceholder}
                className="flex-1 min-w-0 h-9 sm:h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-3 text-xs text-[var(--ink-primary)] focus:border-[var(--ink-primary)] focus:outline-none font-mono placeholder:text-[var(--ink-muted)] shadow-2xs"
              />
              <Button
                onClick={() => {
                  hasUserSelectedPresetRef.current = false
                  hasUserSelectedDomainRef.current = false
                  hasUserSelectedDualModeRef.current = false
                  void runAssess(filePath, { silent: false })
                }}
                disabled={!filePath.trim() || isAssessing}
                variant="secondary"
                size="icon"
                className="shrink-0 h-9 sm:h-8 w-9 sm:w-8"
                aria-label={t.wizard.assessBtn}
                title={t.wizard.assessBtn}
              >
                {isAssessing ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin text-[var(--ink-primary)]" />
                ) : (
                  <RefreshCw className="h-3.5 w-3.5 text-[var(--ink-secondary)]" />
                )}
              </Button>
            </div>

            {assessError && (
              <div className="mt-2.5 p-2.5 rounded-[5px] bg-[#b91c1c]/10 border border-[#b91c1c]/25 text-[#b91c1c] text-xs flex items-center gap-2">
                <AlertTriangle className="h-4 w-4 shrink-0" />
                <span>{assessError}</span>
              </div>
            )}
          </div>

            {/* Presets List */}
            <div>
              <div className="type-step">{t.wizard.stepQuality}</div>
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
                      <div className="flex items-center gap-2 flex-wrap">
                        <span
                          className={`type-value ${
                            isSelected ? 'font-semibold' : 'font-medium'
                          }`}
                        >
                          {p.name}
                        </span>
                        <InfoTip label={p.name}>{p.detail}</InfoTip>
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
                      <div className="type-help mt-1">{p.desc}</div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* 提供商与语言：默认值已经足够好（默认提供商、en→zh 覆盖大多数
                场景），折叠掉；摘要徽章让收起态仍有信息量。语言对影响报价的
                输出膨胀率，所以它包含在自动体检的指纹里——展开改动会触发重报。 */}
            <div className="pt-2">
              <button
                type="button"
                onClick={() => setBasicsOpen(!basicsOpen)}
                className="w-full flex items-center justify-between py-1 text-left type-value font-semibold text-[var(--ink-primary)] hover:text-black transition-colors focus:outline-none group cursor-pointer"
              >
                <div className="flex items-center gap-2 flex-wrap">
                  <Globe className="h-3.5 w-3.5 text-[var(--ink-secondary)] group-hover:text-[var(--ink-primary)]" />
                  <span>{t.wizard.basicsTitle}</span>
                  <Badge variant="outline" className="text-xs font-mono font-normal py-0 px-1.5 normal-case tracking-normal">
                    {(provider || t.wizard.providerDefault) + ` · ` + sourceLang + `→` + targetLang}
                  </Badge>
                </div>
                {basicsOpen ? (
                  <ChevronDown className="h-4 w-4 text-[var(--ink-muted)] group-hover:text-[var(--ink-primary)] transition-transform" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-[var(--ink-muted)] group-hover:text-[var(--ink-primary)] transition-transform" />
                )}
              </button>
              {basicsOpen && (
                <div className="space-y-4 pt-3 mt-2 border-t border-dashed border-[var(--paper-border)]">
            {/* Provider: which declared block does the work. A job may name one;
                the server validates it against the operator's ubt.toml. Hidden
                when nothing is declared, because then there is nothing to choose
                and the first-run banner already says what to do. */}
            {declaredProviders.length > 0 ? (
              <div>
                <div className="flex items-center gap-1 mb-1">
                  <label htmlFor="wizard-provider" className="type-label">{t.wizard.provider}</label>
                  <InfoTip label={t.wizard.providerHelp}>{t.wizard.providerHelp}</InfoTip>
                </div>
                <select
                  id="wizard-provider"
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
                <label htmlFor="wizard-source-lang" className="type-label block mb-1">
                  {t.wizard.sourceLang}
                </label>
                <select
                  id="wizard-source-lang"
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
                <label htmlFor="wizard-target-lang" className="type-label block mb-1">
                  {t.wizard.targetLang}
                </label>
                <select
                  id="wizard-target-lang"
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
                </div>
              )}
            </div>

            {/* Budget Cap.
                A slider could not express this: its range was 0-50 USD, so a
                cloud run on a large book was uncappable, and 0 meant "no cap"
                rather than "zero" — the exact opposite of what a user sliding
                to the left intends. An explicit switch plus a numeric field
                removes both traps, and keeps the default (no cap) visible
                instead of hidden at one end of a track. */}
            <div className="pt-2">
              <div className="type-step">{t.wizard.stepBudget}</div>
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
                <div className="flex items-center justify-between gap-2">
                  <p className="type-help">{t.wizard.budgetUnlimited}</p>
                  {/* 报价已知且 > 0 时给一键安全上限：不强迫，但把
                      「随手设个上限」的成本降到一次点击。 */}
                  {quoteCost !== null && quoteCost > 0 && (
                    <button
                      type="button"
                      onClick={() => {
                        const v = Math.ceil(quoteCost * 1.5 * 100) / 100
                        setBudgetInput(v.toFixed(2))
                        setBudgetUsd(v)
                      }}
                      className="shrink-0 text-[11px] font-mono text-[var(--ink-secondary)] underline decoration-[var(--paper-border-hover)] hover:text-[var(--ink-primary)]"
                    >
                      {t.wizard.budgetSuggest.replace(
                        '{usd}',
                        (Math.ceil(quoteCost * 1.5 * 100) / 100).toFixed(2)
                      )}
                    </button>
                  )}
                </div>
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
                <div className="flex items-center gap-2 flex-wrap">
                  <Sliders className="h-3.5 w-3.5 text-[var(--ink-secondary)] group-hover:text-[var(--ink-primary)]" />
                  <span>{t.wizard.advancedOptionsTitle}</span>
                  <Badge variant="outline" className="text-xs font-mono font-normal uppercase tracking-wider py-0 px-1.5">
                    {t.wizard.advancedOptionsTag}
                  </Badge>
                  {(advancedNumericInvalid || glossaryError) && (
                    <span className="inline-flex items-center gap-1 font-mono text-[10px] font-medium text-[#b45309] bg-[#b45309]/10 px-1.5 py-0.5 rounded border border-[#b45309]/20">
                      <AlertTriangle className="h-3 w-3" />
                      {t.wizard.advancedErrorTag}
                    </span>
                  )}
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
                  {/* Four groups, one line per control. The label names the
                      thing, the InfoTip carries the depth, and visible prose is
                      reserved for consequences the operator must act on (a
                      refused upload, a closed egress gate). Defaults are sound;
                      nothing here is required. */}
                  <div className="type-group -mb-1">{t.wizard.groupDelivery}</div>

                  {/* Delivery shape pairs with its companion artifact; layout
                      and page furniture pair as the second row. */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 items-start">
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <label htmlFor="wizard-dual-mode" className="type-label">{t.wizard.dualMode}</label>
                        <InfoTip label={t.wizard.dualMode}>{t.wizard.dualModeHelp}</InfoTip>
                      </div>
                      <select
                        id="wizard-dual-mode"
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

                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <label htmlFor="wizard-emit-both" className="type-label">{t.wizard.emitBoth}</label>
                        <InfoTip label={t.wizard.emitBoth}>{emitBothHelpText}</InfoTip>
                      </div>
                      <select
                        id="wizard-emit-both"
                        value={emitBoth}
                        onChange={(e) => setEmitBoth(e.target.value as any)}
                        className="type-value w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs"
                      >
                        <option value="auto">{t.wizard.emitBothAuto}</option>
                        <option value="on">{t.wizard.emitBothOn}</option>
                        <option value="off">{t.wizard.emitBothOff}</option>
                      </select>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 items-start">
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <label htmlFor="wizard-layout-columns" className="type-label">{t.wizard.layoutColumns}</label>
                        <InfoTip label={t.wizard.layoutColumns}>
                          {t.wizard.layoutColumnsHelp}
                        </InfoTip>
                      </div>
                      <select
                        id="wizard-layout-columns"
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
                        <label htmlFor="wizard-translate-chrome" className="type-label">{t.wizard.translateChrome}</label>
                        <InfoTip label={t.wizard.translateChrome}>
                          {t.wizard.translateChromeHelp}
                        </InfoTip>
                      </div>
                      <select
                        id="wizard-translate-chrome"
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

                  <div className="type-group -mb-1 pt-1">{t.wizard.groupContent}</div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 items-start">
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <label htmlFor="wizard-domain-profile" className="type-label">{t.wizard.domainProfile}</label>
                        <InfoTip label={t.wizard.domainProfile}>{t.wizard.domainProfileHelp}</InfoTip>
                      </div>
                      <select
                        id="wizard-domain-profile"
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
                      </select>
                    </div>

                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <label htmlFor="wizard-domain-hint" className="type-label">{t.wizard.domainHint}</label>
                        <InfoTip label={t.wizard.domainHint}>{t.wizard.domainHintHelp}</InfoTip>
                      </div>
                      <input
                        id="wizard-domain-hint"
                        type="text"
                        value={domainHint}
                        onChange={(e) => setDomainHint(e.target.value)}
                        className="type-value w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs"
                      />
                    </div>
                  </div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 items-start">
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <label htmlFor="wizard-ocr-mode" className="type-label">{t.wizard.ocrMode}</label>
                        <InfoTip label={t.wizard.ocrMode}>{t.wizard.ocrModeHelp}</InfoTip>
                      </div>
                      <select
                        id="wizard-ocr-mode"
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
                      {ocrNeedsEgress && pageEgressClosed && (
                        <button
                          type="button"
                          onClick={openEgressConfirm}
                          className="type-help mt-1 underline underline-offset-2 text-[var(--ink-secondary)] hover:text-[var(--ink-primary)] cursor-pointer"
                        >
                          {t.wizard.egressEnableBtn}
                        </button>
                      )}
                    </div>

                    {/* A staged upload, not a typed path: browsers do not
                        expose filesystem paths, and a server-side location was
                        never something the operator could know. The standing
                        glossary in /assets always applies; this adds one file
                        for this book. */}
                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <label className="type-label">{t.wizard.glossaryPath}</label>
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
                        <div className="flex items-center gap-1.5">
                          <span
                            className="type-value inline-flex items-center gap-1.5 h-8 min-w-0 px-2.5 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] text-xs"
                            title={glossaryFile.name}
                          >
                            <FileText className="h-3.5 w-3.5 text-[var(--ink-muted)] shrink-0" />
                            <span className="truncate">{glossaryFile.name}</span>
                          </span>
                          <button
                            type="button"
                            onClick={() => setGlossaryFile(null)}
                            aria-label={t.wizard.glossaryRemove}
                            title={t.wizard.glossaryRemove}
                            className="h-8 w-8 shrink-0 inline-flex items-center justify-center rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] text-[var(--ink-muted)] hover:text-[var(--ink-primary)] hover:border-[var(--ink-secondary)] transition-colors cursor-pointer"
                          >
                            <X className="h-3.5 w-3.5" />
                          </button>
                        </div>
                      ) : (
                        <button
                          type="button"
                          disabled={glossaryUploading}
                          onClick={() => glossaryFileRef.current?.click()}
                          className="type-value inline-flex items-center gap-1.5 h-8 px-2.5 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] text-[var(--ink-secondary)] hover:text-[var(--ink-primary)] hover:border-[var(--ink-secondary)] transition-colors disabled:opacity-60 disabled:cursor-not-allowed cursor-pointer"
                        >
                          {glossaryUploading ? (
                            <Loader2 className="h-3.5 w-3.5 animate-spin" />
                          ) : (
                            <UploadCloud className="h-3.5 w-3.5" />
                          )}
                          {t.wizard.glossaryUploadBtn}
                        </button>
                      )}
                      {glossaryError && (
                        <p className="type-help mt-1 text-[#b91c1c]">{glossaryError}</p>
                      )}
                      <Link
                        to="/assets"
                        className="type-help mt-1.5 inline-block text-[var(--ink-secondary)] underline underline-offset-2 hover:text-[var(--ink-primary)]"
                      >
                        {t.wizard.glossaryManageLink}
                      </Link>
                    </div>
                  </div>

                  <div className="type-group -mb-1 pt-1">{t.wizard.groupScope}</div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 items-start">
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center gap-1">
                          <label htmlFor="wizard-page-range" className="type-label">{t.wizard.pageFilter}</label>
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
                        id="wizard-page-range"
                        type="text"
                        value={pageRange}
                        onChange={(e) => setPageRange(e.target.value)}
                        placeholder={t.wizard.pageFilterPlaceholder}
                        aria-label={t.wizard.pageFilter}
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
                      <div className="grid grid-cols-2 gap-2.5">
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
                      {(chapterStartInvalid || chapterCountInvalid) && (
                        <p className="type-help mt-1 text-[#b91c1c]">{t.wizard.fieldMinOne}</p>
                      )}
                    </div>
                  </div>

                  {/* Rehearsal: one checkbox row; the explanation lives in the
                      InfoTip and the honest warning only appears when it runs. */}
                  <div className="pt-1">
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        id="wizard-dry-run"
                        checked={dryRun}
                        onChange={(e) => setDryRun(e.target.checked)}
                        className="h-3.5 w-3.5 rounded border-[var(--paper-border)] accent-[var(--ink-primary)] cursor-pointer"
                      />
                      <label htmlFor="wizard-dry-run" className="type-label cursor-pointer">
                        {t.wizard.rehearsalLabel}
                      </label>
                      <InfoTip label={t.wizard.rehearsalLabel}>{t.wizard.rehearsalHelp}</InfoTip>
                    </div>
                    {dryRun && (
                      <div className="mt-2 rounded-[5px] border border-[#b45309]/30 bg-[#b45309]/5 p-2 type-help text-[#b45309]">
                        {t.wizard.rehearsalActiveWarning}
                      </div>
                    )}
                  </div>

                  <div className="type-group -mb-1 pt-1">{t.wizard.groupExec}</div>

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5 items-start">
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
                      {concurrencyInvalid && (
                        <p className="type-help mt-1 text-[#b91c1c]">{t.wizard.fieldMinOne}</p>
                      )}
                    </div>

                    <div>
                      <div className="flex items-center gap-1 mb-1">
                        <label htmlFor="wizard-visual-judge" className="type-label">{t.wizard.visualJudge}</label>
                        <InfoTip label={t.wizard.visualJudge}>
                          {t.wizard.visualJudgeHelp}
                        </InfoTip>
                      </div>
                      <select
                        id="wizard-visual-judge"
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
                        <>
                          <p className="type-help mt-1 text-[var(--ink-muted)]">
                            {t.wizard.visualJudgeBlocked}
                          </p>
                          <button
                            type="button"
                            onClick={openEgressConfirm}
                            className="type-help mt-1 underline underline-offset-2 text-[var(--ink-secondary)] hover:text-[var(--ink-primary)] cursor-pointer"
                          >
                            {t.wizard.egressEnableBtn}
                          </button>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

        {/* Right Column: Pre-flight Assessment Radar */}
        <div className="lg:col-span-5 lg:sticky lg:top-6 rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] flex flex-col justify-between overflow-hidden shadow-[0_1px_4px_rgba(0,0,0,0.03)]">
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
                  {t.wizard.reassessing}
                </Badge>
              ) : assessStale && assessment ? (
                <Badge variant="warning" className="text-xs">
                  {t.wizard.assessStale}
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
                {(fileName || assessment.document?.file_name) && (
                  <div className="p-3 rounded-[5px] bg-[var(--paper-subsurface)] border border-[var(--paper-border)] flex items-center justify-between">
                    <div className="flex items-center gap-2.5 truncate">
                      <FileText className="h-4 w-4 text-[var(--ink-secondary)] shrink-0" />
                      <div className="truncate">
                        <div className="text-xs font-semibold text-[var(--ink-primary)] truncate font-mono">
                          {fileName || assessment.document.file_name}
                        </div>
                        <div className="text-xs text-[var(--ink-secondary)] font-mono">
                          {fileSize ||
                            (assessment.document.file_size_bytes
                              ? assessment.document.file_size_bytes > 1024 * 1024
                                ? `${(assessment.document.file_size_bytes / (1024 * 1024)).toFixed(1)} MB`
                                : `${(assessment.document.file_size_bytes / 1024).toFixed(0)} KB`
                              : '')}
                        </div>
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
                          assessment.document.math_density === 'high'
                            ? 'text-[#b45309]'
                            : assessment.document.math_density === 'low'
                              ? 'text-[#15803d]'
                              : 'text-[var(--ink-secondary)]'
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
                      {/* The engine bills prompt = (per-call prefix × calls) +
                          source and completion = source × pair fertility. The
                          honest additive decomposition is therefore source +
                          prefix-overhead + completion — three figures that SUM
                          to the total. Showing the raw prompt beside its own
                          source term read as three additive parts and
                          over-summed by the source share (~28% on the
                          reference document). The overhead is derived here as
                          prompt_tokens − estimated_tokens, which the engine
                          guarantees is exactly the prefix fan-out
                          (estimate_draft_cost_from_totals: prompt = calls ×
                          prefix + source, with the same source_tokens). */}
                      <span className="font-mono font-bold text-[var(--ink-primary)]">
                        {(assessment.cost.prompt_tokens + assessment.cost.completion_tokens).toLocaleString()}
                      </span>
                      <span className="block text-xs font-normal text-[var(--ink-muted)] leading-relaxed">
                        {t.wizard.tokensSource}{' '}
                        {assessment.document.estimated_tokens.toLocaleString()} ·{' '}
                        {t.wizard.tokensPrompt}{' '}
                        {Math.max(
                          0,
                          assessment.cost.prompt_tokens - assessment.document.estimated_tokens
                        ).toLocaleString()} ·{' '}
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
                    <span className="type-help inline-flex items-center gap-1">
                      {t.wizard.estTime}
                      {/* ``runtime.heuristic`` is always true — no per-stage
                          timing is persisted anywhere — so carry the engine's
                          own basis text rather than let a bare minute range
                          read as a measurement. */}
                      <InfoTip label={t.wizard.estTime}>
                        <span className="block space-y-1">
                          <span className="block">{t.wizard.estTimeHeuristicNote}</span>
                          {assessment.runtime.basis && (
                            <span className="block text-[var(--ink-muted)]">
                              {assessment.runtime.basis}
                            </span>
                          )}
                        </span>
                      </InfoTip>
                    </span>
                    <span className="font-mono font-bold text-[var(--ink-primary)]">
                      {formatDuration(assessment.runtime.est_seconds_low)} –{' '}
                      {formatDuration(assessment.runtime.est_seconds_high)}
                      <span className="text-[var(--ink-muted)] font-normal">
                        {' '}
                        ({t.wizard.heuristicTag})
                      </span>
                    </span>
                  </div>
                  <div className="py-2.5 flex items-start justify-between gap-3">
                    <span className="text-[var(--ink-secondary)] shrink-0">
                      {t.wizard.recommendation}
                    </span>
                    <span className="font-mono text-[#15803d] text-xs font-semibold text-right">
                      {routeBadge(
                        assessment.route.recommended_dual_mode,
                        assessment.route.recommended_preset,
                        t,
                      )}
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
              onClick={requestSubmit}
              disabled={
                !filePath.trim() ||
                isSubmitting ||
                glossaryError !== null ||
                budgetInvalid ||
                advancedNumericInvalid
              }
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

      {/* 提交护栏：无有效报价（或报价已过期）时确认一次。 */}
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

      {/* Enabling page egress ships rendered manuscript pages off this machine,
          so the action carries its scope in the dialog: what it unlocks, what
          it costs in privacy, and that it reverts at restart. */}
      <ConfirmDialog
        open={egressConfirmOpen}
        title={t.wizard.egressConfirmTitle}
        body={
          <span className="block space-y-1.5">
            <span className="block">{t.wizard.egressConfirmBody}</span>
            {egressError && <span className="block text-[#b91c1c]">{egressError}</span>}
          </span>
        }
        confirmLabel={t.wizard.egressConfirmOn}
        cancelLabel={t.wizard.egressConfirmCancel}
        onConfirm={() => void enablePageEgress()}
        onCancel={() => setEgressConfirmOpen(false)}
      />

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
          onClick={requestSubmit}
          disabled={
            !filePath.trim() ||
            isSubmitting ||
            glossaryError !== null ||
            budgetInvalid ||
            advancedNumericInvalid
          }
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
