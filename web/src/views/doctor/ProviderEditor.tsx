import { useCallback, useEffect, useRef, useState } from 'react'
import { Check, Loader2, X } from 'lucide-react'
import { getProvider, saveProvider, type ProviderDetail } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { InfoTip } from '@/components/ui/InfoTip'
import { useI18n } from '@/i18n/useI18n'

interface ProviderEditorProps {
  /** The provider block being edited; ``null`` starts a new one. */
  name: string | null
  /** Whether this deployment may accept a credential over HTTP.

  ``null`` means the console could not read the deployment configuration, which is
  NOT a refusal: claiming "this server is reachable beyond loopback" when the
  panel merely failed to load would be a false statement about the deployment.
  */
  canWriteCredential: boolean | null
  /** Called after a successful save so the parent can refresh its view. */
  onSaved: () => void
  onCancel: () => void
}

/**
 * The capability fields this form edits — and the reason there are only two.
 *
 * A capability earns a control only when the engine cannot discover it:
 *
 * - ``supports_system_prompt`` — a rejected system role is a 400 no self-heal
 *   rule can fix, because the fix (merge the system text into the user turn) is a
 *   decision, not a dropped parameter. The job fails outright, so it is asked
 *   plainly.
 * - ``supports_vision`` — a model that cannot read images fails every visual
 *   repair attempt, per block. Declaring it avoids a repeated cost.
 *
 * ``supports_temperature`` and ``supports_reasoning_effort`` have no control here
 * on purpose. A rejected one is dropped by the 400 self-heal and then remembered
 * per model (measured: four blocks take five requests, and the parameter never
 * reappears), so a control could only ever save that single round-trip — while
 * asking an operator to answer a question the engine answers itself. The flags
 * remain in ``ModelProfile`` and remain settable in a provider block; they are
 * simply not a form field. See ``ubt/core/config.py`` for the full reasoning.
 */
const CAPABILITY_FIELDS = [
  { field: 'supports_system_prompt', labelKey: 'capabilitySystem', noteKey: 'capabilitySystemNote' },
] as const

/** The system-prompt control renders as a two-way choice with these labels. */
type FlagField = 'supports_system_prompt' | 'supports_vision'

/** Every capability field the form can submit. */
const SUBMITTED_FLAGS = ['supports_system_prompt', 'supports_vision'] as const satisfies
  readonly FlagField[]

/** Which text fields the form edits, and how each is labelled. */
const TEXT_FIELDS = [
  { field: 'base_url', labelKey: 'baseUrl', placeholder: 'https://api.example.com/v1' },
  { field: 'draft_model', labelKey: 'draftModel', placeholder: 'model-id' },
  { field: 'repair_model', labelKey: 'repairModel', placeholder: 'model-id' },
] as const

type TextField = (typeof TEXT_FIELDS)[number]['field']

const API_MODES = ['openai-chat', 'openai-responses', 'anthropic-messages', 'gemini-native']

/**
 * Edit one `[providers.<name>]` block: endpoint, protocol, models, capability.
 *
 * Credentials are deliberately absent from this form and from the API it calls.
 * A block's key may be a literal secret or a `${VAR}` reference, and this server
 * is routinely started on loopback with no API key at all — an endpoint that
 * could write one would turn any local process into a credential-writer. The
 * form therefore reports whether a credential exists (`credential set` /
 * `no credential`) and says where to change it, rather than offering a field
 * that would have to be disabled.
 *
 * The capability checkboxes are tri-state in the file but a checkbox is not, so
 * an untouched box submits nothing and the key is left exactly as it was. That
 * distinction matters: `supports_temperature = false` is a claim that the
 * endpoint rejects the parameter, while an absent key means "not declared, use
 * the safe default". Rendering both as an unchecked box would silently convert
 * one into the other on every save.
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
  // Create mode ends as soon as the block exists. Declared after ``detail``: it
  // reads that state, and reading it before the declaration throws a temporal
  // dead-zone ReferenceError that the error boundary surfaces as a blank screen.
  const isCreate = initialName === null && detail === null
  // Edit mode before the first load lands: the fields stay readable (they are
  // empty) but must not accept typing the landing seed would silently revert.
  const loadingDetail = !isCreate && detail === null
  const [credential, setCredential] = useState('')
  const [text, setText] = useState<Partial<Record<TextField, string>>>({})
  const [flags, setFlags] = useState<Partial<Record<FlagField, boolean | undefined>>>({})
  const [apiMode, setApiMode] = useState('')
  const [isFree, setIsFree] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)

  // The name whose block the current form state was seeded from. Seeding once
  // per name: after a successful create the stored block is already the
  // baseline (``setDetail(stored)`` in the save handler), and refetching on
  // the ``isCreate`` flip would overwrite whatever was typed during the round
  // trip; a name change is the only thing that legitimately reloads.
  const loadedNameRef = useRef<string | null>(null)

  useEffect(() => {
    // A new block starts from the state's initial values (empty), so there is
    // nothing to load and nothing to set — returning here rather than calling
    // setState keeps the effect free of a synchronous state write.
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
        // Seed each flag from the file, keeping "absent" distinguishable: an
        // undefined entry means the key is not in the block at all.
        const seeded: Partial<Record<FlagField, boolean>> = {}
        for (const field of SUBMITTED_FLAGS) {
          const value = loaded[field]
          if (typeof value === 'boolean') seeded[field] = value
        }
        setFlags(seeded)
        setApiMode(loaded.api_mode ?? '')
        setIsFree(loaded.is_free ?? false)
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load provider')
      })
    return () => {
      cancelled = true
    }
  // ``detail`` re-runs the effect after the seed lands; the guard above
  // turns that into a no-op.
  }, [name, isCreate, detail])

  const nameIsValid = /^[A-Za-z0-9_-]{1,64}$/.test(name)

  const handleSave = useCallback(async () => {
    if (!nameIsValid) {
      setError(t.doctor.newProviderNameInvalid)
      return
    }
    setIsSaving(true)
    setError(null)
    try {
      // Only what the operator touched: an omitted key is left untouched in the
      // file, so saving one field cannot rewrite the rest of the block.
      const changes: Record<string, unknown> = {}
      // A credential is submitted only when the operator typed one. An empty box
      // means "keep what is configured", never "clear it" — a form that wiped the
      // key on every save would break the provider it was editing.
      if (credential.trim()) changes.api_key = credential.trim()
      for (const { field } of TEXT_FIELDS) {
        const value = text[field]
        if (value !== undefined && value !== (detail?.[field] ?? '')) {
          changes[field] = value.trim()
        }
      }
      // Every rendered capability control, in one list. ``supports_vision`` is a
      // checkbox rather than a select, and walking only the select group once made
      // it render, toggle and then never reach the request.
      for (const field of SUBMITTED_FLAGS) {
        const value = flags[field]
        const stored = detail?.[field]
        const storedIsSet = typeof stored === 'boolean'
        if (value === undefined && storedIsSet) {
          // Explicit ``null`` clears the key: back to "not declared", which is
          // not the same as declaring ``false``.
          changes[field] = null
        } else if (value !== undefined && value !== stored) {
          changes[field] = value
        }
      }
      if (apiMode && apiMode !== (detail?.api_mode ?? '')) changes.api_mode = apiMode
      if (isFree !== (detail?.is_free ?? false)) changes.is_free = isFree
      if (Object.keys(changes).length === 0 && !isCreate) {
        onSaved()
        return
      }
      // A block nothing points at still runs as a rehearsal: `provider` stays
      // unset, `draft_model` comes out empty, and the wizard announces mock
      // output. On create, "the block I just made" is unambiguously what the
      // operator wants to run, so it becomes the file default in the same save —
      // one action instead of "create, then discover you also have to select it".
      if (isCreate) changes.make_default = true
      const stored = await saveProvider(name, changes)
      // Adopt the stored block as the new baseline. Without this the form stays
      // in "create" mode after a successful create: it keeps offering "Create",
      // keeps showing "no credential" for the credential it just wrote, and a
      // second click would overwrite what the first one made.
      setDetail(stored)
      setCredential('')
      setSaved(true)
      onSaved()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Save failed')
    } finally {
      setIsSaving(false)
    }
  }, [apiMode, credential, detail, flags, isCreate, isFree, name, nameIsValid, onSaved, t, text])

  return (
    <div className="rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] overflow-hidden">
      <div className="px-4 py-2.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] flex items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="type-group">
            {isCreate ? t.doctor.addProvider : t.doctor.editingProvider}
          </div>
          <div className="type-value font-mono font-semibold truncate">{name || '—'}</div>
        </div>
        <button
          type="button"
          onClick={onCancel}
          aria-label={t.doctor.cancel}
          className="p-2 sm:p-1 rounded text-[var(--ink-muted)] hover:text-[var(--ink-primary)] hover:bg-[var(--paper-border)] shrink-0 min-h-[40px] min-w-[40px] sm:min-h-0 sm:min-w-0 flex items-center justify-center cursor-pointer"
        >
          <X size={16} className="sm:w-3.5 sm:h-3.5" />
        </button>
      </div>

      <div className="p-3.5 space-y-3">
        {/* A new block needs a name: it becomes the [providers.<name>] key. */}
        {isCreate ? (
          <label className="block">
            <span className="type-label text-[var(--ink-secondary)]">
              {t.doctor.newProviderName}
            </span>
            <input
              type="text"
              value={name}
              placeholder="my-endpoint"
              onChange={(event) => setName(event.target.value)}
              className="mt-1 w-full rounded-[4px] border border-[var(--paper-border)] bg-[var(--paper-bg)] px-2 py-1 type-value font-mono"
            />
            <span
              className={`block mt-0.5 type-help ${name && !nameIsValid ? 'text-[#b91c1c]' : 'text-[var(--ink-muted)]'}`}
            >
              {name && !nameIsValid
                ? t.doctor.newProviderNameInvalid
                : t.doctor.newProviderNameHint}
            </span>
          </label>
        ) : null}

        {/* Credential status: the state the *loader* will see, not merely whether
            the key exists in the file. A block carrying an unset ``${VAR}``
            reports "unresolved variable" and names it, because "credential set"
            there was a promise the next job could not keep. */}
        <div className="flex items-center gap-2 type-help">
          <span
            className={
              detail?.has_credential || detail?.authenticates || detail?.credential_issue === null
                ? 'text-[#15803d]'
                : 'text-[#b45309]'
            }
          >
            ●{' '}
            {detail?.credential_issue === 'unresolved_variable'
              ? t.doctor.credentialUnresolvedVar.replace(
                  '{vars}',
                  (detail?.unresolved_vars ?? []).join(', '),
                )
              : detail?.has_credential
                ? t.doctor.credentialConfigured
                : detail?.authenticates
                  ? t.doctor.credentialEnvSupplied
                  : detail?.credential_issue === null
                    ? // No issue and no credential: a loopback endpoint, which
                      // accepts an empty bearer. Saying "no credential" here was
                      // the false alarm that sent operators to fix a working
                      // local model.
                      t.doctor.credentialLocalEndpoint
                    : t.doctor.credentialMissing}
          </span>
          <InfoTip label={t.doctor.providerConfigHint}>{t.doctor.providerConfigHint}</InfoTip>
        </div>

        {/* The trap the form used to walk into: it said "saved, effective on the
            next job" while ``UBT_LLM_API_KEY`` outranked the block it had just
            written. A save that cannot take effect has to say so. */}
        {detail?.credential_shadowed_by_env ? (
          <p className="type-help text-[#b45309] leading-relaxed">
            {t.doctor.credentialShadowedByEnv}
          </p>
        ) : null}

        {/* The credential field is offered only where the deployment permits it.
            Rendering it disabled with a reason beats rendering it enabled and
            failing on submit: the operator learns the rule from the form rather
            than from a 400 after typing a secret. */}
        {canWriteCredential ? (
          <label className="block">
            <span className="type-label text-[var(--ink-secondary)]">
              {t.doctor.credentialField}
            </span>
            <input
              type="password"
              value={credential}
              placeholder={detail?.has_credential ? '********' : t.doctor.credentialPlaceholder}
              autoComplete="off"
              disabled={loadingDetail}
              onChange={(event) => setCredential(event.target.value)}
              className="mt-1 w-full rounded-[4px] border border-[var(--paper-border)] bg-[var(--paper-bg)] px-2 py-1 type-value font-mono"
            />
            {detail?.has_credential ? (
              <span className="block mt-0.5 type-help text-[var(--ink-muted)]">
                {t.doctor.credentialKeepHint}
              </span>
            ) : null}
          </label>
        ) : (
          <p className="type-help text-[#b45309] leading-relaxed">
            {canWriteCredential === null
              ? t.doctor.credentialUnknownHint
              : t.doctor.credentialBlockedHint}
          </p>
        )}

        {TEXT_FIELDS.map(({ field, labelKey, placeholder }) => (
          <label key={field} className="block">
            <span className="type-label text-[var(--ink-secondary)]">{t.doctor[labelKey]}</span>
            <input
              type="text"
              value={text[field] ?? ''}
              placeholder={placeholder}
              disabled={loadingDetail}
              onChange={(event) =>
                setText((prev) => ({ ...prev, [field]: event.target.value }))
              }
              className="mt-1 w-full rounded-[4px] border border-[var(--paper-border)] bg-[var(--paper-bg)] px-2 py-1 type-value font-mono"
            />
          </label>
        ))}

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <label className="block">
            <span className="type-label text-[var(--ink-secondary)]">{t.doctor.apiMode}</span>
            <select
              value={apiMode}
              disabled={loadingDetail}
              onChange={(event) => setApiMode(event.target.value)}
              className="mt-1 w-full rounded-[4px] border border-[var(--paper-border)] bg-[var(--paper-bg)] px-2 py-1 type-value font-mono"
            >
              <option value="">—</option>
              {API_MODES.map((mode) => (
                <option key={mode} value={mode}>
                  {mode}
                </option>
              ))}
            </select>
          </label>
          {/* No extraction control. ``auto`` is not one strategy among four — it
              IS "try the XML tag, then strip the conversational wrapper", so it
              already contains ``xml_tag`` and ``cp``. The other two are actively
              worse: ``raw`` skips cleaning (a chatty endpoint's "Here is the
              translation:" and its ``` fence ship into the book) and
              ``xml_tag`` returns *empty* for any reply without a tag, which is
              every reply under the minimal and hybrid prompt strategies.

              Presented as four unexplained choices, an operator had no way to
              pick correctly. The default is what they want; the field stays
              settable in ubt.toml / UBT_EXTRACTION_STRATEGY for the rare endpoint
              that needs otherwise, and this row says so plainly. */}
          <div className="block">
            <span className="flex items-center gap-1">
              <span className="type-label text-[var(--ink-secondary)]">
                {t.doctor.extractionStrategy}
              </span>
              <InfoTip label={t.doctor.extractionStrategy}>
                {t.doctor.extractionStrategyHelp}
              </InfoTip>
            </span>
            <p className="mt-1 type-help text-[var(--ink-secondary)] leading-relaxed">
              {t.doctor.extractionAutoNote}
            </p>
          </div>
        </div>

        <div className="pt-1 border-t border-[var(--paper-border)]">
          <div className="flex items-center gap-1.5 mt-2">
            <span className="type-group">{t.doctor.capabilityTitle}</span>
            <InfoTip label={t.doctor.capabilityHelp}>{t.doctor.capabilityHelp}</InfoTip>
          </div>

          {/* Collapsed, because it is a remedy rather than a choice: the engine
              already assumes a separate system turn is accepted, and that is
              right for essentially every endpoint. This switch exists for the
              one failure mode where it is not — the provider answers the request
              with a 400 complaining about the `system` role — and the operator
              has to tell the engine to fold the instruction into the user turn
              instead. Rendering it as an equal peer of "accepts images" invited
              the reader to make a decision they had no basis for.

              Note the engine cannot detect this: a rejected system role is a
              merge decision, not a capability probe, so the job fails rather than
              degrading — which is why the switch has to stay reachable. */}
          <details className="mt-2">
            <summary className="cursor-pointer type-help text-[var(--ink-secondary)] hover:text-[var(--ink-primary)] select-none">
              {t.doctor.capabilityTroubleshootToggle}
            </summary>
            {CAPABILITY_FIELDS.map(({ field, labelKey, noteKey }) => (
              <div key={field} className="mt-2">
                <div className="flex items-center justify-between gap-3">
                  <span className="type-value">{t.doctor[labelKey]}</span>
                  <select
                    value={flags[field] === undefined ? '' : String(flags[field])}
                    onChange={(event) => {
                      const raw = event.target.value
                      setFlags((prev) => {
                        const next = { ...prev }
                        if (raw === '') delete next[field]
                        else next[field] = raw === 'true'
                        return next
                      })
                    }}
                    className="shrink-0 rounded-[4px] border border-[var(--paper-border)] bg-[var(--paper-bg)] px-1.5 py-0.5 type-help"
                  >
                    <option value="">{t.doctor.capabilityUndeclared}</option>
                    <option value="true">{t.doctor.capabilitySystemAccepted}</option>
                    <option value="false">{t.doctor.capabilitySystemMerged}</option>
                  </select>
                </div>
                <p className="type-help text-[var(--ink-muted)] mt-0.5 leading-relaxed">
                  {t.doctor[noteKey]}
                </p>
              </div>
            ))}
          </details>

          {/* Vision: a plain checkbox. Off is the safe default and costs only a
              degraded repair, never a failed job. */}
          <label className="flex items-start gap-2 cursor-pointer mt-2.5">
            <input
              type="checkbox"
              checked={flags.supports_vision ?? false}
              onChange={(event) =>
                setFlags((prev) => ({ ...prev, supports_vision: event.target.checked }))
              }
              className="mt-0.5 shrink-0"
            />
            <span className="min-w-0">
              <span className="type-value">{t.doctor.capabilityVision}</span>
              <span className="block type-help text-[var(--ink-muted)]">
                {t.doctor.capabilityVisionNote}
              </span>
            </span>
          </label>

          {/* No temperature / reasoning-effort controls, deliberately.

              Both are recovered by the 400 self-heal and then remembered per
              model, so the engine already knows what this form would ask. The
              control that existed here only ever saved one round-trip, and it made
              an operator answer a question the engine answers itself.

              The engine keeps ``ModelProfile.supports_temperature`` and
              ``supports_reasoning_effort``: they decide whether a parameter is
              *sent*, and without them a rejected parameter would be re-sent on
              every block. The flags are still settable in a provider block for the
              operator who wants to skip that first 400 — they are simply no longer
              a form field. */}
        </div>

        <label className="flex items-center gap-2 cursor-pointer pt-1 border-t border-[var(--paper-border)]">
          <input
            type="checkbox"
            checked={isFree}
            onChange={(event) => setIsFree(event.target.checked)}
            className="shrink-0"
          />
          <span className="type-value">{t.doctor.isFree}</span>
        </label>

        {error ? <p className="type-help text-[#b91c1c]">{error}</p> : null}
        {saved && !error ? (
          <p className="type-help text-[#15803d] flex items-center gap-1">
            <Check size={12} /> {t.doctor.savedHint}
          </p>
        ) : null}

        <div className="flex items-center gap-2 pt-1">
          <Button
            size="sm"
            onClick={() => void handleSave()}
            disabled={isSaving || (!isCreate && !detail) || !nameIsValid}
            className="min-h-[40px] sm:min-h-0 px-3"
          >
            {isSaving ? (
              <>
                <Loader2 size={12} className="animate-spin" /> {t.doctor.saving}
              </>
            ) : isCreate ? (
              t.doctor.createProvider
            ) : (
              t.doctor.save
            )}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={onCancel}
            disabled={isSaving}
            className="min-h-[40px] sm:min-h-0 px-3"
          >
            {t.doctor.cancel}
          </Button>
        </div>
      </div>
    </div>
  )
}
