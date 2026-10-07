import React, { useState, useRef, useEffect } from 'react'
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
import { assessJob, submitJob, uploadSourceDocument, type JobAssessResponse } from '@/api/client'
import { useI18n } from '@/i18n/I18nContext'

/** Human-readable duration from a seconds estimate (e.g. "3m", "1.2h"). */
function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  return `${(seconds / 3600).toFixed(1)}h`
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
  const [preset, setPreset] = useState<'publication' | 'standard' | 'preview' | 'fast'>('publication')
  const [budgetUsd, setBudgetUsd] = useState<number>(10.0)

  // Advanced compiler & layout options
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false)
  const [domainProfile, setDomainProfile] = useState('general')
  const [dualMode, setDualMode] = useState<'auto' | 'inline' | 'facing' | 'alternating' | 'monolingual'>('auto')
  const [pageRange, setPageRange] = useState('')
  const [glossaryPath, setGlossaryPath] = useState('')
  const [execMode, setExecMode] = useState<'auto' | 'short' | 'long'>('auto')
  const [formulaMode, setFormulaMode] = useState<'readable' | 'strict'>('strict')

  const [isAssessing, setIsAssessing] = useState(false)
  const [isReassessing, setIsReassessing] = useState(false)
  const [assessment, setAssessment] = useState<JobAssessResponse | null>(null)
  const [assessError, setAssessError] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)

  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitError, setSubmitError] = useState<string | null>(null)

  const lastAssessedFileRef = useRef<string>('')
  const hasUserSelectedPresetRef = useRef<boolean>(false)
  const hasUserSelectedDomainRef = useRef<boolean>(false)

  const runAssess = async (
    path: string,
    opts?: {
      overridePreset?: 'publication' | 'standard' | 'preview' | 'fast'
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

    const activePreset = opts?.overridePreset ?? preset
    const activePages = opts?.overridePages !== undefined ? opts.overridePages : pageRange
    const activeTarget = opts?.overrideTarget ?? targetLang
    const activeSource = opts?.overrideSource ?? sourceLang

    try {
      const res = await assessJob({
        input_path: path.trim(),
        target_lang: activeTarget,
        source_lang: activeSource,
        preset: activePreset,
        pages: activePages.trim() ? activePages.trim() : null,
        deep: true,
      })
      setAssessment(res)

      // Auto-feed recommendations when a new document is assessed
      if (lastAssessedFileRef.current !== path.trim()) {
        lastAssessedFileRef.current = path.trim()
        if (!hasUserSelectedPresetRef.current && res.route?.recommended_preset) {
          const recP = res.route.recommended_preset.toLowerCase() as any
          if (['publication', 'standard', 'preview', 'fast'].includes(recP)) {
            setPreset(recP)
          }
        }
        if (!hasUserSelectedDomainRef.current && res.route?.recommended_profile) {
          const recDom = res.route.recommended_profile.toLowerCase()
          if (['general', 'textbook', 'paper', 'fiction', 'humanities', 'semiconductor'].includes(recDom)) {
            setDomainProfile(recDom)
          }
        }
      }
    } catch (err) {
      if (!isSilent) {
        setAssessError(err instanceof Error ? err.message : 'Assessment failed')
      }
    } finally {
      setIsAssessing(false)
      setIsReassessing(false)
    }
  }

  // Reactive re-assessment on preset / page slice / lang tweaks
  useEffect(() => {
    if (!filePath.trim()) return
    if (lastAssessedFileRef.current === filePath.trim()) {
      const timer = setTimeout(() => {
        void runAssess(filePath, {
          overridePreset: preset,
          overridePages: pageRange,
          overrideTarget: targetLang,
          overrideSource: sourceLang,
          silent: true,
        })
      }, 350)
      return () => clearTimeout(timer)
    }
  }, [filePath, preset, pageRange, targetLang, sourceLang])

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

  const handleSubmit = async () => {
    if (!filePath.trim() || isSubmitting) return
    setIsSubmitting(true)
    setSubmitError(null)
    try {
      const res = await submitJob({
        input_path: filePath.trim(),
        target_lang: targetLang,
        source_lang: sourceLang,
        profile: domainProfile,
        preset: preset,
        dual_mode: dualMode === 'auto' ? null : dualMode,
        pages: pageRange.trim() ? pageRange.trim() : null,
        glossary: glossaryPath.trim() ? glossaryPath.trim() : null,
        exec_mode: execMode === 'auto' ? null : execMode,
        formula_mode: formulaMode,
        priority: 0,
        dry_run: false,
        budget_usd: budgetUsd,
      })
      navigate(`/jobs/${res.job_id}`)
    } catch (err) {
      setSubmitError(err instanceof Error ? err.message : 'Submission failed')
    } finally {
      setIsSubmitting(false)
    }
  }

  // Keyboard shortcut: ⌘⏎ or Ctrl+Enter to compile
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        if (filePath.trim() && !isSubmitting) {
          e.preventDefault()
          handleSubmit()
        }
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [filePath, isSubmitting])

  const presetOptions = [
    {
      id: 'publication' as const,
      name: t.wizard.presetPublication,
      desc: t.wizard.presetPublicationDesc,
      gate: 'CLAUDE 3.5 + STRICT GATE',
    },
    {
      id: 'standard' as const,
      name: t.wizard.presetStandard,
      desc: t.wizard.presetStandardDesc,
      gate: 'FAST PASS + DEEP PROOF',
    },
    {
      id: 'preview' as const,
      name: t.wizard.presetPreview,
      desc: t.wizard.presetPreviewDesc,
      gate: 'STRUCTURE & LAYOUT PROOF',
    },
    {
      id: 'fast' as const,
      name: t.wizard.presetFast,
      desc: t.wizard.presetFastDesc,
      gate: 'ECONOMY TOKEN BUDGET',
    },
  ]

  return (
    <div className="flex-1 overflow-y-auto px-8 py-7 space-y-7 max-w-6xl mx-auto w-full">
      {/* View Title */}
      <div className="flex items-end justify-between border-b border-[var(--paper-border)] pb-4">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-[var(--ink-primary)]">
            {t.wizard.title}
          </h1>
          <p className="text-xs text-[var(--ink-secondary)] mt-1">
            {t.wizard.subtitle}
          </p>
        </div>
        <div className="text-[11px] font-mono text-[var(--ink-muted)]">
          WAL LEDGER: <span className="text-[#15803d] font-semibold">ONLINE</span>
        </div>
      </div>

      {/* Main 2-Column Split */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
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
            <div
              onDragOver={(e) => {
                e.preventDefault()
                setIsDragOver(true)
              }}
              onDragLeave={() => setIsDragOver(false)}
              onDrop={handleFileDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`rounded-lg border-2 border-dashed p-7 transition-all cursor-pointer text-center ${
                isDragOver
                  ? 'border-[var(--ink-primary)] bg-[var(--paper-subsurface)]'
                  : 'border-[var(--paper-border)] hover:border-[var(--paper-border-hover)] bg-[var(--paper-surface)] hover:bg-[#fdfdfc]'
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
              <div className="text-[11px] text-[var(--ink-secondary)] mt-1">
                {t.wizard.dropzoneSubtitle}
              </div>
              <div className="mt-3">
                <span className="text-[11px] font-mono text-[var(--ink-primary)] underline decoration-[var(--paper-border-hover)] hover:text-black font-medium">
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
                className="flex-1 h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-3 text-xs text-[var(--ink-primary)] focus:border-[var(--ink-primary)] focus:outline-none font-mono placeholder:text-[var(--ink-muted)] shadow-2xs"
              />
              <Button
                onClick={() => {
                  lastAssessedFileRef.current = ''
                  hasUserSelectedPresetRef.current = false
                  hasUserSelectedDomainRef.current = false
                  void runAssess(filePath, { silent: false })
                }}
                disabled={!filePath.trim() || isAssessing}
                variant="secondary"
                size="sm"
                className="shrink-0"
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
            {/* Language Pair */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] block mb-1 font-semibold">
                  {t.wizard.sourceLang}
                </label>
                <select
                  value={sourceLang}
                  onChange={(e) => setSourceLang(e.target.value)}
                  className="w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 text-xs text-[var(--ink-primary)] focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs font-medium"
                >
                  <option value="en">English (en)</option>
                  <option value="ja">Japanese (ja)</option>
                  <option value="de">German (de)</option>
                  <option value="fr">French (fr)</option>
                </select>
              </div>
              <div>
                <label className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] block mb-1 font-semibold">
                  {t.wizard.targetLang}
                </label>
                <select
                  value={targetLang}
                  onChange={(e) => setTargetLang(e.target.value)}
                  className="w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 text-xs text-[var(--ink-primary)] focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs font-medium"
                >
                  <option value="zh">简体中文 (zh)</option>
                  <option value="zh-tw">繁體中文 (zh-tw)</option>
                  <option value="ja">日本語 (ja)</option>
                </select>
              </div>
            </div>

            {/* Presets List */}
            <div>
              <label className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] block mb-2 font-semibold">
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
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <span
                            className={`text-xs ${
                              isSelected
                                ? 'font-bold text-[var(--ink-primary)]'
                                : 'font-medium text-[var(--ink-primary)]'
                            }`}
                          >
                            {p.name}
                          </span>
                          {isRecommended && (
                            <span className="inline-flex items-center text-[10px] font-mono font-medium text-[#15803d] bg-[#15803d]/10 px-1.5 py-0.5 rounded border border-[#15803d]/20">
                              {t.wizard.engineRecommended}
                            </span>
                          )}
                        </div>
                        <span className="font-mono text-[10px] text-[var(--ink-muted)] uppercase tracking-wider">
                          {p.gate}
                        </span>
                      </div>
                      <div className="text-[11px] text-[var(--ink-secondary)] mt-1 leading-relaxed">
                        {p.desc}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>

            {/* Budget Cap */}
            <div className="pt-2">
              <div className="flex items-center justify-between text-xs mb-1.5">
                <span className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
                  {t.wizard.budgetCeiling}
                </span>
                <span className="font-mono text-xs font-bold text-[#15803d]">
                  ${budgetUsd.toFixed(2)} USD (HARD FUSE)
                </span>
              </div>
              <input
                type="range"
                min="1"
                max="50"
                step="1"
                value={budgetUsd}
                onChange={(e) => setBudgetUsd(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-[var(--paper-border)] rounded-lg appearance-none cursor-pointer accent-[var(--ink-primary)]"
              />
            </div>

            {/* Advanced Compiler & Layout Options Drawer */}
            <div className="pt-3 border-t border-[var(--paper-border)]">
              <button
                type="button"
                onClick={() => setIsAdvancedOpen(!isAdvancedOpen)}
                className="w-full flex items-center justify-between py-1 text-left text-xs font-semibold text-[var(--ink-primary)] hover:text-black transition-colors focus:outline-none group cursor-pointer"
              >
                <div className="flex items-center gap-2">
                  <Sliders className="h-3.5 w-3.5 text-[var(--ink-secondary)] group-hover:text-[var(--ink-primary)]" />
                  <span>{t.wizard.advancedOptionsTitle}</span>
                  <Badge variant="outline" className="text-[9px] font-mono font-normal uppercase tracking-wider py-0 px-1.5">
                    {t.wizard.advancedOptionsTag}
                  </Badge>
                </div>
                {isAdvancedOpen ? (
                  <ChevronDown className="h-4 w-4 text-[var(--ink-muted)] group-hover:text-[var(--ink-primary)] transition-transform" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-[var(--ink-muted)] group-hover:text-[var(--ink-primary)] transition-transform" />
                )}
              </button>
              <p className="text-[11px] text-[var(--ink-muted)] mt-0.5 mb-2 leading-relaxed">
                {t.wizard.advancedOptionsSubtitle}
              </p>

              {isAdvancedOpen && (
                <div className="space-y-4 pt-3 pb-1 border-t border-dashed border-[var(--paper-border)] mt-2">
                  {/* Row 1: Profile & Dual Mode */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
                    <div>
                      <label className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] block mb-1 font-semibold">
                        {t.wizard.domainProfile}
                      </label>
                      <select
                        value={domainProfile}
                        onChange={(e) => {
                          hasUserSelectedDomainRef.current = true
                          setDomainProfile(e.target.value)
                        }}
                        className="w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 text-xs text-[var(--ink-primary)] focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs font-medium"
                      >
                        <option value="general">{t.wizard.domainGeneral}</option>
                        <option value="textbook">{t.wizard.domainTextbook}</option>
                        <option value="paper">{t.wizard.domainPaper}</option>
                        <option value="fiction">{t.wizard.domainFiction}</option>
                        <option value="humanities">{t.wizard.domainHumanities}</option>
                        <option value="semiconductor">{t.wizard.domainSemiconductor}</option>
                      </select>
                    </div>

                    <div>
                      <label className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] block mb-1 font-semibold">
                        {t.wizard.dualMode}
                      </label>
                      <select
                        value={dualMode}
                        onChange={(e) => setDualMode(e.target.value as any)}
                        className="w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 text-xs text-[var(--ink-primary)] focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs font-medium"
                      >
                        <option value="auto">{t.wizard.dualModeAuto}</option>
                        <option value="inline">{t.wizard.dualModeInline}</option>
                        <option value="facing">{t.wizard.dualModeFacing}</option>
                        <option value="alternating">{t.wizard.dualModeAlternating}</option>
                        <option value="monolingual">{t.wizard.dualModeMonolingual}</option>
                      </select>
                    </div>
                  </div>

                  {/* Row 2: Formula Mode */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
                    <div>
                      <label className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] block mb-1 font-semibold">
                        {t.wizard.formulaMode}
                      </label>
                      <select
                        value={formulaMode}
                        onChange={(e) => setFormulaMode(e.target.value as any)}
                        className="w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 text-xs text-[var(--ink-primary)] focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs font-medium"
                      >
                        <option value="readable">{t.wizard.formulaModeReadable}</option>
                        <option value="strict">{t.wizard.formulaModeStrict}</option>
                      </select>
                    </div>
                  </div>

                  {/* Row 3: Pages Range Filter & Execution Mode */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <label className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] font-semibold">
                          {t.wizard.pageFilter}
                        </label>
                        <button
                          type="button"
                          onClick={() => setPageRange('1-5')}
                          className="text-[10px] font-mono text-[var(--ink-secondary)] hover:text-[var(--ink-primary)] underline cursor-pointer"
                        >
                          {t.wizard.sliceSampleQuickBtn}
                        </button>
                      </div>
                      <input
                        type="text"
                        value={pageRange}
                        onChange={(e) => setPageRange(e.target.value)}
                        placeholder={t.wizard.pageFilterPlaceholder}
                        className="w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 text-xs text-[var(--ink-primary)] focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs font-mono placeholder:text-[var(--ink-muted)]"
                      />
                      <p className="text-[10px] text-[var(--ink-muted)] mt-1">
                        {t.wizard.pageFilterHelp}
                      </p>
                    </div>

                    <div>
                      <label className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] block mb-1 font-semibold">
                        {t.wizard.execMode}
                      </label>
                      <select
                        value={execMode}
                        onChange={(e) => setExecMode(e.target.value as any)}
                        className="w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 text-xs text-[var(--ink-primary)] focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs font-medium"
                      >
                        <option value="auto">{t.wizard.execModeAuto}</option>
                        <option value="short">{t.wizard.execModeShort}</option>
                        <option value="long">{t.wizard.execModeLong}</option>
                      </select>
                    </div>
                  </div>

                  {/* Row 4: External Glossary Path */}
                  <div>
                    <label className="text-[10px] font-mono uppercase tracking-wider text-[var(--ink-muted)] block mb-1 font-semibold">
                      {t.wizard.glossaryPath}
                    </label>
                    <input
                      type="text"
                      value={glossaryPath}
                      onChange={(e) => setGlossaryPath(e.target.value)}
                      placeholder={t.wizard.glossaryPathPlaceholder}
                      className="w-full h-8 rounded-[5px] border border-[var(--paper-border)] bg-[var(--paper-surface)] px-2.5 text-xs text-[var(--ink-primary)] focus:border-[var(--ink-primary)] focus:outline-none shadow-2xs font-mono placeholder:text-[var(--ink-muted)]"
                    />
                    <p className="text-[10px] text-[var(--ink-muted)] mt-1">
                      {t.wizard.glossaryPathHelp}
                    </p>
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
                <h3 className="text-xs font-bold text-[var(--ink-primary)] tracking-tight">
                  {t.wizard.preflightTitle}
                </h3>
                <p className="text-[11px] text-[var(--ink-secondary)] mt-0.5">
                  {t.wizard.preflightSubtitle}
                </p>
              </div>
              {isReassessing ? (
                <Badge variant="outline" className="animate-pulse text-[10px] font-mono">
                  <Loader2 className="h-3 w-3 animate-spin mr-1 inline" />
                  UPDATING...
                </Badge>
              ) : assessment ? (
                <Badge variant="success" dot>
                  {t.common.ready}
                </Badge>
              ) : (
                <Badge variant="outline">IDLE</Badge>
              )}
            </div>

            {assessment ? (
              <div className="space-y-4">
                {/* Book File Header Card */}
                {fileName && (
                  <div className="p-3 rounded-[5px] bg-[var(--paper-subsurface)] border border-[var(--paper-border)] flex items-center justify-between">
                    <div className="flex items-center gap-2.5 truncate">
                      <FileText className="h-4 w-4 text-[var(--ink-secondary)] shrink-0" />
                      <div className="truncate">
                        <div className="text-xs font-semibold text-[var(--ink-primary)] truncate font-mono">
                          {fileName}
                        </div>
                        <div className="text-[11px] text-[var(--ink-secondary)] font-mono">{fileSize}</div>
                      </div>
                    </div>
                    <Badge variant="outline" className="uppercase shrink-0 text-[10px] font-mono">
                      {assessment.document.format_ext.replace('.', '') || 'PDF'}
                    </Badge>
                  </div>
                )}

                {/* Metric Rows — real nested assess fields */}
                <div className="divide-y divide-[var(--paper-border)] text-xs">
                  <div className="py-2.5 flex items-center justify-between">
                    <span className="text-[var(--ink-secondary)]">{t.wizard.pageCount}</span>
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
                    <span className="text-[var(--ink-secondary)]">{t.wizard.sourceChars}</span>
                    <span className="font-mono font-bold text-[var(--ink-primary)]">
                      {assessment.document.source_chars.toLocaleString()}
                    </span>
                  </div>
                  <div className="py-2.5 flex items-center justify-between">
                    <span className="text-[var(--ink-secondary)]">{t.wizard.mathDensity}</span>
                    <span
                      className={`font-mono font-bold ${
                        assessment.document.has_formulas ? 'text-[#b45309]' : 'text-[#15803d]'
                      }`}
                    >
                      {assessment.document.math_density}
                    </span>
                  </div>
                  <div className="py-2.5 flex items-center justify-between">
                    <span className="text-[var(--ink-secondary)]">{t.wizard.scannedRatio}</span>
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
                  <div className="py-2.5 flex items-center justify-between">
                    <span className="text-[var(--ink-secondary)]">{t.wizard.estTokens}</span>
                    <span className="font-mono font-bold text-[var(--ink-primary)]">
                      {assessment.document.estimated_tokens.toLocaleString()}
                    </span>
                  </div>
                  <div className="py-2.5 flex items-center justify-between">
                    <span className="text-[var(--ink-secondary)]">{t.wizard.estCost}</span>
                    <span className="font-mono font-bold">
                      {assessment.cost.total_cost_usd === null ||
                      assessment.cost.total_cost_usd === undefined ? (
                        <span className="text-[var(--ink-muted)]">not priced</span>
                      ) : assessment.cost.total_cost_usd === 0 ? (
                        <span className="text-[#15803d]">
                          $0.00{' '}
                          <span className="text-[10px] font-normal text-[var(--ink-secondary)]">
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
                    <span className="text-[var(--ink-secondary)]">{t.wizard.estTime}</span>
                    <span className="font-mono font-bold text-[var(--ink-primary)]">
                      {formatDuration(assessment.runtime.est_seconds_low)} –{' '}
                      {formatDuration(assessment.runtime.est_seconds_high)}
                    </span>
                  </div>
                  <div className="py-2.5 flex items-start justify-between gap-3">
                    <span className="text-[var(--ink-secondary)] shrink-0">
                      {t.wizard.recommendation}
                    </span>
                    <span className="font-mono text-[#15803d] text-[11px] font-semibold text-right">
                      {assessment.route.recommended_dual_mode} · {assessment.route.recommended_preset}
                      <span className="text-[var(--ink-muted)] font-normal">
                        {' '}
                        ({t.wizard.confidence} {(assessment.route.confidence * 100).toFixed(0)}%)
                      </span>
                    </span>
                  </div>
                </div>

                {assessment.warnings.length > 0 && (
                  <div className="rounded-[5px] border border-[#b45309]/30 bg-[#b45309]/5 p-2.5 space-y-1">
                    <div className="text-[10px] font-mono uppercase tracking-wider text-[#b45309] font-semibold">
                      {t.wizard.probeWarnings} ({assessment.warnings.length})
                    </div>
                    {assessment.warnings.slice(0, 4).map((warning) => (
                      <div key={warning.code} className="text-[11px] text-[var(--ink-secondary)]">
                        <span className="font-mono text-[var(--ink-muted)]">{warning.code}</span> ·{' '}
                        {warning.detail_zh}
                      </div>
                    ))}
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

          {/* Action Footer */}
          <div className="p-4 border-t border-[var(--paper-border)] bg-[var(--paper-subsurface)]">
            {submitError && (
              <div className="mb-3 p-2 rounded-[5px] bg-[#b91c1c]/10 border border-[#b91c1c]/25 text-[#b91c1c] text-xs">
                {submitError}
              </div>
            )}
            <Button
              onClick={handleSubmit}
              disabled={!filePath.trim() || isSubmitting}
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
    </div>
  )
}
