import { useCallback, useEffect, useRef, useState } from 'react'
import { Check, ChevronLeft, Loader2 } from 'lucide-react'

import { getProvider, saveProvider, type ProviderDetail } from '@/api/client'
import { Alert, AlertDescription } from '@/components/ui/Alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n/useI18n'
import { cn } from '@/lib/utils'
import { DoctorField, DoctorPanel } from './components'

interface ProviderEditorProps {
  /** The provider block being edited; ``null`` starts a new one. */
  name: string | null
  /**
   * Whether this deployment may accept a credential over HTTP.
   *
   * ``null`` means the console could not read the deployment configuration, which
   * is NOT a refusal: claiming "this server is reachable beyond loopback" when the
   * panel merely failed to load would be a false statement about the deployment.
   */
  canWriteCredential: boolean | null
  /** Called after a successful save so the parent can refresh its view. */
  onSaved: () => void
  onCancel: () => void
}

/**
 * The capability fields this form edits, and why there are only two.
 *
 * A capability earns a control only when the engine cannot discover it:
 * ``supports_system_prompt`` (a rejected system role is a merge decision) and
 * ``supports_vision`` (a text-only model fails every visual repair). Temperature
 * and reasoning support are recovered by the 400 self-heal and remembered per
 * model, so asking the operator would only save a single round trip.
 */
const CAPABILITY_FIELDS = [
  { field: 'supports_system_prompt', labelKey: 'capabilitySystem', noteKey: 'capabilitySystemNote' },
] as const

type FlagField =
  | 'supports_system_prompt'
  | 'supports_vision'
  | 'supports_reasoning_effort'

/** Every capability field the form can submit. */
const SUBMITTED_FLAGS = [
  'supports_system_prompt',
  'supports_vision',
  'supports_reasoning_effort',
] as const satisfies readonly FlagField[]

const TEXT_FIELDS = [
  { field: 'base_url', labelKey: 'baseUrl', placeholder: 'https://api.example.com/v1' },
  { field: 'draft_model', labelKey: 'draftModel', placeholder: 'model-id' },
  { field: 'repair_model', labelKey: 'repairModel', placeholder: 'model-id' },
] as const

type TextField = (typeof TEXT_FIELDS)[number]['field']

const REASONING_FIELDS = [
  'draft_reasoning_effort',
  'repair_reasoning_effort',
  'reasoning_dialect',
] as const

type ReasoningField = (typeof REASONING_FIELDS)[number]

const API_MODES = ['openai-chat', 'openai-responses', 'anthropic-messages', 'gemini-native']
const REASONING_EFFORTS = ['none', 'low', 'medium', 'high'] as const
const REASONING_DIALECTS = ['nested', 'flat'] as const
const UNDECLARED = '__undeclared__'

/**
 * Edit one `[providers.<name>]` block: endpoint, protocol, models, capability.
 *
 * Credentials are deliberately absent from the API's read path. A block's key
 * may be a literal secret or a ``${VAR}`` reference; this form reports whether
 * one resolves and only offers a write field where the server explicitly allows
 * it. The capability controls are tri-state in the file, so "not declared"
 * remains a distinct choice rather than being coerced to ``false``.
 */
export function ProviderEditor({
  name: initialName,
  canWriteCredential,
  onSaved,
  onCancel,
}: ProviderEditorProps) {
  const { t } = useI18n()
  const [name, setName] = useState(initialName ?? '')
  const [detail, setDetail] = useState<ProviderDetail | null>(null)
  // Declared after ``detail``: reading it before the state exists would throw a
  // temporal dead-zone ReferenceError.
  const isCreate = initialName === null && detail === null
  const loadingDetail = !isCreate && detail === null
  const [credential, setCredential] = useState('')
  const [text, setText] = useState<Partial<Record<TextField, string>>>({})
  const [flags, setFlags] = useState<Partial<Record<FlagField, boolean | undefined>>>({})
  const [apiMode, setApiMode] = useState('')
  const [reasoning, setReasoning] = useState<Record<ReasoningField, string>>({
    draft_reasoning_effort: '',
    repair_reasoning_effort: '',
    reasoning_dialect: '',
  })
  const [isFree, setIsFree] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)
  const [dirty, setDirty] = useState(false)
  const loadedNameRef = useRef<string | null>(null)

  useEffect(() => {
    if (isCreate) return
    if (loadedNameRef.current === name && detail !== null) return
    let cancelled = false
    void getProvider(name)
      .then((loaded) => {
        if (cancelled) return
        loadedNameRef.current = name
        setDetail(loaded)
        setText({
          base_url: loaded.base_url ?? '',
          draft_model: loaded.draft_model ?? '',
          repair_model: loaded.repair_model ?? '',
        })
        const seeded: Partial<Record<FlagField, boolean>> = {}
        for (const field of SUBMITTED_FLAGS) {
          const value = loaded[field]
          if (typeof value === 'boolean') seeded[field] = value
        }
        setFlags(seeded)
        setApiMode(loaded.api_mode ?? '')
        setReasoning({
          draft_reasoning_effort: loaded.draft_reasoning_effort ?? '',
          repair_reasoning_effort: loaded.repair_reasoning_effort ?? '',
          reasoning_dialect: loaded.reasoning_dialect ?? '',
        })
        setIsFree(loaded.is_free ?? false)
        setDirty(false)
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : t.doctor.loadFailed)
      })
    return () => {
      cancelled = true
    }
  }, [detail, isCreate, name, t.doctor.loadFailed])

  const nameIsValid = /^[A-Za-z0-9_.-]{1,64}$/.test(name)

  const markDirty = () => {
    setSaved(false)
    setDirty(true)
  }

  const handleSave = useCallback(async () => {
    if (!nameIsValid) {
      setError(t.doctor.newProviderNameInvalid)
      return
    }
    setIsSaving(true)
    setError(null)
    try {
      // Submit only what changed. An omitted field stays exactly as it is in the
      // file, which is what keeps editing one value from rewriting the block.
      const changes: Record<string, unknown> = {}
      if (credential.trim()) changes.api_key = credential.trim()
      for (const { field } of TEXT_FIELDS) {
        const value = text[field]
        if (value !== undefined && value !== (detail?.[field] ?? '')) {
          changes[field] = value.trim()
        }
      }
      for (const field of SUBMITTED_FLAGS) {
        const value = flags[field]
        const stored = detail?.[field]
        const storedIsSet = typeof stored === 'boolean'
        if (value === undefined && storedIsSet) {
          // Explicit null returns the key to "not declared", which is distinct
          // from declaring that the endpoint does not support it.
          changes[field] = null
        } else if (value !== undefined && value !== stored) {
          changes[field] = value
        }
      }
      if (apiMode && apiMode !== (detail?.api_mode ?? '')) changes.api_mode = apiMode
      for (const field of REASONING_FIELDS) {
        const value = reasoning[field]
        if (value !== (detail?.[field] ?? '')) {
          changes[field] = value === '' ? null : value
        }
      }
      if (isFree !== (detail?.is_free ?? false)) changes.is_free = isFree
      if (Object.keys(changes).length === 0 && !isCreate) {
        onSaved()
        return
      }
      // A new block that nothing points at is still a rehearsal. Creating one
      // therefore makes it the default in the same deliberate action.
      if (isCreate) changes.make_default = true
      const stored = await saveProvider(name, changes)
      setDetail(stored)
      setCredential('')
      setSaved(true)
      setDirty(false)
      onSaved()
    } catch (err) {
      setError(err instanceof Error ? err.message : t.doctor.qeSaveFailed)
    } finally {
      setIsSaving(false)
    }
  }, [
    apiMode,
    credential,
    detail,
    flags,
    isCreate,
    isFree,
    name,
    nameIsValid,
    onSaved,
    reasoning,
    t.doctor.newProviderNameInvalid,
    t.doctor.qeSaveFailed,
    text,
  ])

  const credentialState = (() => {
    if (detail?.credential_issue === 'unresolved_variable') {
      return {
        label: t.doctor.credentialUnresolvedVar.replace(
          '{vars}',
          (detail.unresolved_vars ?? []).join(', '),
        ),
        ok: false,
      }
    }
    if (detail?.has_credential) return { label: t.doctor.credentialConfigured, ok: true }
    if (detail?.authenticates) return { label: t.doctor.credentialEnvSupplied, ok: true }
    if (detail?.credential_issue === null) {
      return { label: t.doctor.credentialLocalEndpoint, ok: true }
    }
    return { label: t.doctor.credentialMissing, ok: false }
  })()

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <Button
          type="button"
          variant="ghost"
          size="sm"
          onClick={onCancel}
          className="-ml-2 self-start text-muted-foreground"
        >
          <ChevronLeft />
          {t.doctor.cancel}
        </Button>
        {dirty ? (
          <Badge variant="outline" className="self-start font-normal text-muted-foreground sm:self-auto">
            {t.doctor.formUnsaved}
          </Badge>
        ) : null}
      </div>

      <DoctorPanel
        title={isCreate ? t.doctor.addProvider : `${t.doctor.editingProvider} · ${name}`}
        description={isCreate ? t.doctor.formCreateHint : t.doctor.formEditHint}
        action={
          <div className="flex items-center gap-2">
            <Badge
              variant="outline"
              className={cn(
                credentialState.ok
                  ? 'border-emerald-600/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
                  : 'border-amber-600/25 bg-amber-500/10 text-amber-700 dark:text-amber-400'
              )}
            >
              {credentialState.label}
            </Badge>
          </div>
        }
      >
        {loadingDetail ? (
          <div className="flex items-center justify-center gap-2 px-4 py-16 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            {t.doctor.loadingChecks}
          </div>
        ) : (
          <div className="space-y-6 p-4 sm:p-5">
            {isCreate ? (
              <section className="space-y-2">
                <h3 className="text-xs font-semibold uppercase tracking-[0.08em] text-muted-foreground">
                  {t.doctor.newProviderName}
                </h3>
                <DoctorField
                  label={t.doctor.newProviderName}
                  hint={
                    name && !nameIsValid
                      ? t.doctor.newProviderNameInvalid
                      : t.doctor.newProviderNameHint
                  }
                >
                  <Input
                    value={name}
                    placeholder="my-endpoint"
                    className="font-mono"
                    aria-invalid={Boolean(name && !nameIsValid)}
                    onChange={(event) => {
                      setName(event.target.value)
                      markDirty()
                    }}
                  />
                </DoctorField>
              </section>
            ) : null}

            <section className="space-y-3">
              <div>
                <h3 className="text-sm font-semibold">{t.doctor.credentialsSection}</h3>
                <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">
                  {t.doctor.providerConfigHint}
                </p>
              </div>
              {detail?.credential_shadowed_by_env ? (
                <Alert variant="warning">
                  <AlertDescription>{t.doctor.credentialShadowedByEnv}</AlertDescription>
                </Alert>
              ) : null}
              {canWriteCredential ? (
                <DoctorField
                  label={t.doctor.credentialField}
                  hint={detail?.has_credential ? t.doctor.credentialKeepHint : undefined}
                >
                  <Input
                    type="password"
                    value={credential}
                    placeholder={detail?.has_credential ? '********' : t.doctor.credentialPlaceholder}
                    autoComplete="off"
                    className="font-mono"
                    onChange={(event) => {
                      setCredential(event.target.value)
                      markDirty()
                    }}
                  />
                </DoctorField>
              ) : (
                <Alert variant="warning">
                  <AlertDescription>
                    {canWriteCredential === null
                      ? t.doctor.credentialUnknownHint
                      : t.doctor.credentialBlockedHint}
                  </AlertDescription>
                </Alert>
              )}
            </section>

            <Separator />

            <section className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold">{t.doctor.connectionSection}</h3>
                <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">
                  {t.doctor.modelFormHelp}
                </p>
              </div>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                {TEXT_FIELDS.map(({ field, labelKey, placeholder }) => (
                  <DoctorField
                    key={field}
                    label={t.doctor[labelKey]}
                    className={field === 'base_url' ? 'md:col-span-2' : undefined}
                  >
                    <Input
                      value={text[field] ?? ''}
                      placeholder={placeholder}
                      className="font-mono"
                      onChange={(event) => {
                        setText((prev) => ({ ...prev, [field]: event.target.value }))
                        markDirty()
                      }}
                    />
                  </DoctorField>
                ))}
                <DoctorField label={t.doctor.apiMode}>
                  <Select
                    value={apiMode || UNDECLARED}
                    onValueChange={(value) => {
                      setApiMode(value === UNDECLARED ? '' : value)
                      markDirty()
                    }}
                  >
                    <SelectTrigger className="font-mono">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value={UNDECLARED}>—</SelectItem>
                      {API_MODES.map((mode) => (
                        <SelectItem key={mode} value={mode} className="font-mono">
                          {mode}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </DoctorField>
              </div>
              <div className="rounded-lg border border-border/70 bg-muted/20 px-3 py-2.5">
                <div className="text-xs font-medium">{t.doctor.extractionStrategy}</div>
                <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                  {t.doctor.extractionAutoNote}
                </p>
              </div>
            </section>

            <Separator />

            <section className="space-y-4">
              <div>
                <h3 className="text-sm font-semibold">{t.doctor.behaviorSection}</h3>
                <p className="mt-0.5 text-xs leading-relaxed text-muted-foreground">
                  {t.doctor.reasoningHelp}
                </p>
              </div>
              <div className="flex items-center justify-between gap-4 rounded-lg border border-border/70 p-3.5">
                <div className="min-w-0">
                  <Label htmlFor="reasoning-supported" className="text-xs font-medium">
                    {t.doctor.reasoningAccepts}
                  </Label>
                  <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                    {t.doctor.reasoningAcceptsNote}
                  </p>
                </div>
                <Switch
                  id="reasoning-supported"
                  checked={flags.supports_reasoning_effort ?? false}
                  onCheckedChange={(checked) => {
                    setFlags((prev) => ({ ...prev, supports_reasoning_effort: checked }))
                    markDirty()
                  }}
                />
              </div>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
                {(
                  [
                    {
                      field: 'draft_reasoning_effort' as const,
                      labelKey: 'reasoningDraftEffort',
                      options: REASONING_EFFORTS,
                    },
                    {
                      field: 'repair_reasoning_effort' as const,
                      labelKey: 'reasoningRepairEffort',
                      options: REASONING_EFFORTS,
                    },
                    {
                      field: 'reasoning_dialect' as const,
                      labelKey: 'reasoningDialect',
                      options: REASONING_DIALECTS,
                    },
                  ] as const
                ).map(({ field, labelKey, options }) => (
                  <DoctorField key={field} label={t.doctor[labelKey]}>
                    <Select
                      value={reasoning[field] || UNDECLARED}
                      onValueChange={(value) => {
                        setReasoning((prev) => ({
                          ...prev,
                          [field]: value === UNDECLARED ? '' : value,
                        }))
                        markDirty()
                      }}
                    >
                      <SelectTrigger className="font-mono">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value={UNDECLARED}>{t.doctor.capabilityUndeclared}</SelectItem>
                        {options.map((option) => (
                          <SelectItem key={option} value={option} className="font-mono">
                            {option}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </DoctorField>
                ))}
              </div>
              <div className="flex items-center justify-between gap-4 rounded-lg border border-border/70 p-3.5">
                <div>
                  <Label htmlFor="provider-free" className="text-xs font-medium">
                    {t.doctor.isFree}
                  </Label>
                  <p className="mt-1 text-xs text-muted-foreground">{t.doctor.freeLabel}</p>
                </div>
                <Switch
                  id="provider-free"
                  checked={isFree}
                  onCheckedChange={(checked) => {
                    setIsFree(checked)
                    markDirty()
                  }}
                />
              </div>
            </section>

            <Accordion type="single" collapsible>
              <AccordionItem value="advanced" className="rounded-lg border border-border/70 px-3.5">
                <AccordionTrigger className="py-3 text-xs font-semibold hover:no-underline">
                  {t.doctor.advancedSection}
                </AccordionTrigger>
                <AccordionContent className="space-y-4">
                  <div className="flex items-start justify-between gap-4">
                    <div className="min-w-0">
                      <Label htmlFor="provider-vision" className="text-xs font-medium">
                        {t.doctor.capabilityVision}
                      </Label>
                      <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                        {t.doctor.capabilityVisionNote}
                      </p>
                    </div>
                    <Switch
                      id="provider-vision"
                      checked={flags.supports_vision ?? false}
                      onCheckedChange={(checked) => {
                        setFlags((prev) => ({ ...prev, supports_vision: checked }))
                        markDirty()
                      }}
                    />
                  </div>

                  <Separator />

                  <div className="space-y-1 text-xs font-medium">
                    {t.doctor.capabilityTroubleshootToggle}
                  </div>
                  {CAPABILITY_FIELDS.map(({ field, labelKey, noteKey }) => (
                    <DoctorField key={field} label={t.doctor[labelKey]} hint={t.doctor[noteKey]}>
                      <Select
                        value={
                          flags[field] === undefined ? UNDECLARED : String(flags[field])
                        }
                        onValueChange={(value) => {
                          setFlags((prev) => {
                            const next = { ...prev }
                            if (value === UNDECLARED) delete next[field]
                            else next[field] = value === 'true'
                            return next
                          })
                          markDirty()
                        }}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value={UNDECLARED}>
                            {t.doctor.capabilityUndeclared}
                          </SelectItem>
                          <SelectItem value="true">{t.doctor.capabilitySystemAccepted}</SelectItem>
                          <SelectItem value="false">{t.doctor.capabilitySystemMerged}</SelectItem>
                        </SelectContent>
                      </Select>
                    </DoctorField>
                  ))}
                </AccordionContent>
              </AccordionItem>
            </Accordion>

            {error ? (
              <Alert variant="destructive">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            ) : null}
            {saved && !error ? (
              <p className="flex items-center gap-1.5 text-xs text-emerald-700 dark:text-emerald-400">
                <Check className="size-3.5" />
                {t.doctor.savedHint}
              </p>
            ) : null}
          </div>
        )}
        <div className="flex flex-col-reverse gap-2 border-t border-border/70 bg-muted/15 px-4 py-3 sm:flex-row sm:justify-end">
          <Button
            type="button"
            variant="ghost"
            onClick={onCancel}
            disabled={isSaving}
            className="sm:px-4"
          >
            {t.doctor.cancel}
          </Button>
          <Button
            type="button"
            onClick={() => void handleSave()}
            disabled={isSaving || loadingDetail || !nameIsValid}
            className="sm:px-5"
          >
            {isSaving ? (
              <>
                <Loader2 className="animate-spin" />
                {t.doctor.saving}
              </>
            ) : isCreate ? (
              t.doctor.createProvider
            ) : (
              t.doctor.save
            )}
          </Button>
        </div>
      </DoctorPanel>
    </div>
  )
}
