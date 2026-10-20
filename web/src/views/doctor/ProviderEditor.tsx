import { useCallback, useEffect, useState } from 'react'
import { Check, Loader2, X } from 'lucide-react'
import { getProvider, saveProvider, type ProviderDetail } from '@/api/client'
import { Button } from '@/components/ui/Button'
import { InfoTip } from '@/components/ui/InfoTip'
import { useI18n } from '@/i18n/useI18n'

interface ProviderEditorProps {
  /** The provider block being edited. */
  name: string
  /** Called after a successful save so the parent can refresh its view. */
  onSaved: () => void
  onCancel: () => void
}

/** The four capability flags, in the order the form renders them. */
const CAPABILITY_FLAGS = [
  { field: 'supports_temperature', labelKey: 'capabilityTemperature', noteKey: null },
  {
    field: 'supports_reasoning_effort',
    labelKey: 'capabilityReasoning',
    noteKey: 'capabilityReasoningNote',
  },
  { field: 'supports_system_prompt', labelKey: 'capabilitySystem', noteKey: null },
  { field: 'supports_vision', labelKey: 'capabilityVision', noteKey: 'capabilityVisionNote' },
] as const

type FlagField = (typeof CAPABILITY_FLAGS)[number]['field']

/** Which text fields the form edits, and how each is labelled. */
const TEXT_FIELDS = [
  { field: 'base_url', labelKey: 'baseUrl', placeholder: 'https://api.example.com/v1' },
  { field: 'draft_model', labelKey: 'draftModel', placeholder: 'model-id' },
  { field: 'repair_model', labelKey: 'repairModel', placeholder: 'model-id' },
] as const

type TextField = (typeof TEXT_FIELDS)[number]['field']

const API_MODES = ['openai-chat', 'openai-responses', 'anthropic-messages', 'gemini-native']
const EXTRACTION_STRATEGIES = ['auto', 'xml_tag', 'raw', 'cp']

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
export function ProviderEditor({ name, onSaved, onCancel }: ProviderEditorProps) {
  const { t } = useI18n()
  const [detail, setDetail] = useState<ProviderDetail | null>(null)
  const [text, setText] = useState<Partial<Record<TextField, string>>>({})
  const [flags, setFlags] = useState<Partial<Record<FlagField, boolean | undefined>>>({})
  const [apiMode, setApiMode] = useState('')
  const [extraction, setExtraction] = useState('')
  const [isFree, setIsFree] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    let cancelled = false
    void getProvider(name)
      .then((loaded) => {
        if (cancelled) return
        setDetail(loaded)
        setText({
          base_url: loaded.base_url ?? '',
          draft_model: loaded.draft_model ?? '',
          repair_model: loaded.repair_model ?? '',
        })
        // Seed each flag from the file, keeping "absent" distinguishable: an
        // undefined entry means the key is not in the block at all.
        const seeded: Partial<Record<FlagField, boolean>> = {}
        for (const { field } of CAPABILITY_FLAGS) {
          const value = loaded[field]
          if (typeof value === 'boolean') seeded[field] = value
        }
        setFlags(seeded)
        setApiMode(loaded.api_mode ?? '')
        setExtraction(loaded.extraction_strategy ?? '')
        setIsFree(loaded.is_free ?? false)
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : 'Failed to load provider')
      })
    return () => {
      cancelled = true
    }
  }, [name])

  const handleSave = useCallback(async () => {
    setIsSaving(true)
    setError(null)
    try {
      // Only what the operator touched: an omitted key is left untouched in the
      // file, so saving one field cannot rewrite the rest of the block.
      const changes: Record<string, unknown> = {}
      for (const { field } of TEXT_FIELDS) {
        const value = text[field]
        if (value !== undefined && value !== (detail?.[field] ?? '')) {
          changes[field] = value.trim()
        }
      }
      for (const { field } of CAPABILITY_FLAGS) {
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
      if (extraction && extraction !== (detail?.extraction_strategy ?? '')) {
        changes.extraction_strategy = extraction
      }
      if (isFree !== (detail?.is_free ?? false)) changes.is_free = isFree
      if (Object.keys(changes).length === 0) {
        onSaved()
        return
      }
      await saveProvider(name, changes)
      setSaved(true)
      onSaved()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Save failed')
    } finally {
      setIsSaving(false)
    }
  }, [apiMode, detail, extraction, flags, isFree, name, onSaved, text])

  return (
    <div className="rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] overflow-hidden">
      <div className="px-4 py-2.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] flex items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="type-group">{t.doctor.editingProvider}</div>
          <div className="type-value font-mono font-semibold truncate">{name}</div>
        </div>
        <button
          type="button"
          onClick={onCancel}
          aria-label={t.doctor.cancel}
          className="p-1 rounded text-[var(--ink-muted)] hover:text-[var(--ink-primary)] hover:bg-[var(--paper-border)] shrink-0"
        >
          <X size={14} />
        </button>
      </div>

      <div className="p-3.5 space-y-3">
        {/* Credential status: a boolean, because that is all the API reveals. */}
        <div className="flex items-center gap-2 type-help">
          <span
            className={
              detail?.has_credential
                ? 'text-[#15803d]'
                : 'text-[#b45309]'
            }
          >
            ●{' '}
            {detail?.has_credential
              ? t.doctor.credentialConfigured
              : t.doctor.credentialMissing}
          </span>
          <InfoTip label={t.doctor.providerConfigHint}>{t.doctor.providerConfigHint}</InfoTip>
        </div>

        {TEXT_FIELDS.map(({ field, labelKey, placeholder }) => (
          <label key={field} className="block">
            <span className="type-label text-[var(--ink-secondary)]">{t.doctor[labelKey]}</span>
            <input
              type="text"
              value={text[field] ?? ''}
              placeholder={placeholder}
              onChange={(event) =>
                setText((prev) => ({ ...prev, [field]: event.target.value }))
              }
              className="mt-1 w-full rounded-[4px] border border-[var(--paper-border)] bg-[var(--paper-bg)] px-2 py-1 type-value font-mono"
            />
          </label>
        ))}

        <div className="grid grid-cols-2 gap-3">
          <label className="block">
            <span className="type-label text-[var(--ink-secondary)]">{t.doctor.apiMode}</span>
            <select
              value={apiMode}
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
          <label className="block">
            <span className="type-label text-[var(--ink-secondary)]">
              {t.doctor.extractionStrategy}
            </span>
            <select
              value={extraction}
              onChange={(event) => setExtraction(event.target.value)}
              className="mt-1 w-full rounded-[4px] border border-[var(--paper-border)] bg-[var(--paper-bg)] px-2 py-1 type-value font-mono"
            >
              <option value="">—</option>
              {EXTRACTION_STRATEGIES.map((strategy) => (
                <option key={strategy} value={strategy}>
                  {strategy}
                </option>
              ))}
            </select>
          </label>
        </div>

        <div className="pt-1 border-t border-[var(--paper-border)]">
          <div className="flex items-center gap-1.5 mt-2">
            <span className="type-group">{t.doctor.capabilityTitle}</span>
            <InfoTip label={t.doctor.capabilityHelp}>{t.doctor.capabilityHelp}</InfoTip>
          </div>
          <div className="mt-1.5 space-y-2">
            {CAPABILITY_FLAGS.map(({ field, labelKey, noteKey }) => (
              <div key={field}>
                <div className="flex items-center justify-between gap-3">
                  <span className="type-value">{t.doctor[labelKey]}</span>
                  {/* Tri-state, because a checkbox cannot express this field.
                      ``undefined`` means the key is absent from the block and
                      the engine uses the safe default; ``false`` is a real claim
                      that the endpoint REJECTS the parameter (the gpt-5 / o1
                      family rejects temperature). A two-state control collapses
                      "not declared" into "declared false" on the first save,
                      which silently converts an assumption into a statement. */}
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
                    className="shrink-0 rounded-[4px] border border-[var(--paper-border)] bg-[var(--paper-bg)] px-1.5 py-0.5 type-help font-mono"
                  >
                    <option value="">{t.doctor.capabilityUndeclared}</option>
                    <option value="true">{t.doctor.capabilityYes}</option>
                    <option value="false">{t.doctor.capabilityNo}</option>
                  </select>
                </div>
                {noteKey ? (
                  <p className="type-help text-[var(--ink-muted)] mt-0.5">{t.doctor[noteKey]}</p>
                ) : null}
              </div>
            ))}
          </div>
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
          <Button size="sm" onClick={() => void handleSave()} disabled={isSaving || !detail}>
            {isSaving ? (
              <>
                <Loader2 size={12} className="animate-spin" /> {t.doctor.saving}
              </>
            ) : (
              t.doctor.save
            )}
          </Button>
          <Button size="sm" variant="ghost" onClick={onCancel} disabled={isSaving}>
            {t.doctor.cancel}
          </Button>
        </div>
      </div>
    </div>
  )
}
