import { useCallback, useEffect, useState } from 'react'
import { Check, Loader2, Pencil, X } from 'lucide-react'
import { Button } from '@/components/ui/Button'
import { InfoTip } from '@/components/ui/InfoTip'
import { getQeSettings, saveQeSettings, type QESettings, type QESettingsRequest } from '@/api/client'
import { useI18n } from '@/i18n/useI18n'

interface QESettingsCardProps {
  /** Provider names a judge may point at, for the select. */
  providerNames: string[]
}

/**
 * The quality-estimation gate: what scores a run, and the only form that says so.
 *
 * Every finished job used to report "QE not measured (heuristic engine)" and
 * stop there. The remedy lived in `ubt.toml` — `qe_engine = "tiered"` **and**
 * `qe_judge_enabled = true` **and** a judge model — with no control anywhere in
 * the console, so the sentence named a fix the reader could not perform on the
 * screen they were on.
 *
 * **One control, two keys.** The pipeline selects the LLM judge on
 * `qe_judge_enabled` alone; `qe_engine = "tiered"` without it only produces a
 * warning and identical behaviour to the default. Exposing both as checkboxes
 * would let an operator save the documented-but-useless combination, so the
 * selector writes the pair together: "tiered" means both keys on, "heuristic"
 * means both off.
 */
export function QESettingsCard({ providerNames }: QESettingsCardProps) {
  const { t } = useI18n()
  const [settings, setSettings] = useState<QESettings | null>(null)
  const [loadFailed, setLoadFailed] = useState(false)
  const [editing, setEditing] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)

  // Draft state. Numbers are held as strings so a half-typed "0." is not
  // coerced to 0 and saved behind the operator's back.
  const [tiered, setTiered] = useState(false)
  const [judgeProvider, setJudgeProvider] = useState('')
  const [judgeModel, setJudgeModel] = useState('')
  const [threshold, setThreshold] = useState('0.75')
  const [passSample, setPassSample] = useState('0')
  const [sentences, setSentences] = useState(true)

  const adopt = useCallback((next: QESettings) => {
    setSettings(next)
    setLoadFailed(false)
    // The select shows the *effective* choice, not the raw label: a hand-edited
    // `tiered` with the judge off scores nothing, and rendering it as "tiered"
    // would repeat the exact confusion this card exists to remove.
    setTiered(Boolean(next.judge_enabled))
    setJudgeProvider(next.judge_provider ?? '')
    // The raw setting, not the resolved id: seeding from `judge_model` would
    // write an inherited `repair_model` back as an explicit pin on a save that
    // only changed the threshold.
    setJudgeModel(next.judge_model_explicit ?? '')
    setThreshold(String(next.threshold ?? 0.75))
    setPassSample(String(next.judge_pass_sample ?? 0))
    setSentences(next.judge_sentences ?? true)
  }, [])

  useEffect(() => {
    let mounted = true
    void (async () => {
      try {
        const next = await getQeSettings()
        if (mounted) adopt(next)
      } catch {
        if (mounted) setLoadFailed(true)
      }
    })()
    return () => {
      mounted = false
    }
  }, [adopt])

  const startEditing = () => {
    if (settings) adopt(settings)
    setError(null)
    setSaved(false)
    setEditing(true)
  }

  const handleSave = async () => {
    const thresholdValue = Number(threshold)
    const sampleValue = Number(passSample)
    if (!Number.isFinite(thresholdValue) || thresholdValue < 0 || thresholdValue > 1) {
      setError(t.doctor.qeThresholdInvalid)
      return
    }
    if (!Number.isFinite(sampleValue) || sampleValue < 0 || sampleValue > 1) {
      setError(t.doctor.qePassSampleInvalid)
      return
    }
    setSaving(true)
    setError(null)
    try {
      // Both keys travel together — see the component docstring. `null` clears
      // an optional key, which is how "inherit from the provider block" is
      // expressed; an empty string would name a model that does not exist.
      const body: QESettingsRequest = {
        qe_engine: tiered ? 'tiered' : 'heuristic',
        qe_judge_enabled: tiered,
        qe_judge_provider: judgeProvider || null,
        qe_judge_model: judgeModel.trim() || null,
        qe_threshold: thresholdValue,
        qe_judge_pass_sample: sampleValue,
        qe_judge_sentences: sentences,
      }
      const next = await saveQeSettings(body)
      adopt(next)
      setSaved(true)
      setEditing(false)
    } catch (err) {
      setError(err instanceof Error ? err.message : t.doctor.qeSaveFailed)
    } finally {
      setSaving(false)
    }
  }

  const inconsistent =
    settings !== null &&
    ((settings.engine === 'tiered' && !settings.judge_enabled) ||
      (settings.judge_enabled && settings.engine !== 'tiered'))

  return (
    <div className="rounded-lg border border-[var(--paper-border)] bg-[var(--paper-surface)] overflow-hidden shadow-2xs">
      <div className="px-4 py-2.5 border-b border-[var(--paper-border)] bg-[var(--paper-subsurface)] flex items-center justify-between gap-3">
        <div className="min-w-0 flex items-center gap-1.5">
          <span className="type-group">{t.doctor.qeTitle}</span>
          <InfoTip label={t.doctor.qeTitle}>{t.doctor.qeHelp}</InfoTip>
        </div>
        {!editing && settings ? (
          <Button size="sm" variant="ghost" onClick={startEditing} className="!h-7 px-2 type-help">
            <Pencil size={12} /> {t.doctor.qeEdit}
          </Button>
        ) : null}
        {editing ? (
          <button
            type="button"
            onClick={() => setEditing(false)}
            aria-label={t.doctor.qeCancel}
            className="p-2 sm:p-1 rounded text-[var(--ink-muted)] hover:text-[var(--ink-primary)] hover:bg-[var(--paper-border)] shrink-0"
          >
            <X size={16} className="sm:w-3.5 sm:h-3.5" />
          </button>
        ) : null}
      </div>

      {loadFailed ? (
        <div className="p-4 type-help text-[var(--ink-muted)]">{t.doctor.qeLoadFailed}</div>
      ) : !settings ? (
        <div className="p-6 text-center text-[var(--ink-muted)]">
          <Loader2 className="h-4 w-4 animate-spin mx-auto" />
        </div>
      ) : editing ? (
        <div className="p-3.5 space-y-3">
          <label className="block">
            <span className="type-label text-[var(--ink-secondary)]">{t.doctor.qeEngine}</span>
            <select
              value={tiered ? 'tiered' : 'heuristic'}
              onChange={(event) => {
                const next = event.target.value === 'tiered'
                setTiered(next)
                // Turning the judge on with a 0 sample would judge only blocks
                // the deterministic checks call ambiguous — on a clean book
                // that is nothing, so the report still says "not measured".
                // 0.15 is the setup guide's documented starting point, and the
                // field shows the change before it is saved.
                if (next && Number(passSample) === 0) setPassSample('0.15')
              }}
              className="mt-1 w-full rounded-[4px] border border-[var(--paper-border)] bg-[var(--paper-bg)] px-2 py-1 type-value"
            >
              <option value="heuristic">{t.doctor.qeEngineHeuristic}</option>
              <option value="tiered">{t.doctor.qeEngineTiered}</option>
            </select>
            <span className="block mt-0.5 type-help text-[var(--ink-muted)] leading-relaxed">
              {t.doctor.qeEngineHelp}
            </span>
          </label>

          {/* The judge's own block, when it is not the translator's. Optional on
              purpose: an existing single-endpoint config keeps working, and the
              judge then bills through the translation provider. */}
          <label className="block">
            <span className="type-label text-[var(--ink-secondary)]">
              {t.doctor.qeJudgeProvider}
            </span>
            <select
              value={judgeProvider}
              onChange={(event) => setJudgeProvider(event.target.value)}
              className="mt-1 w-full rounded-[4px] border border-[var(--paper-border)] bg-[var(--paper-bg)] px-2 py-1 type-value font-mono"
            >
              <option value="">{t.doctor.qeJudgeProviderInherit}</option>
              {/* The current value may be a shipped protocol (`openai-chat`) that
                  the declared list does not contain. Without its own option the
                  select renders blank and the next save would silently clear a
                  working judge endpoint. */}
              {judgeProvider && !providerNames.includes(judgeProvider) ? (
                <option value={judgeProvider}>{judgeProvider}</option>
              ) : null}
              {providerNames.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>

          <label className="block">
            <span className="type-label text-[var(--ink-secondary)]">{t.doctor.qeJudgeModel}</span>
            <input
              type="text"
              value={judgeModel}
              placeholder={t.doctor.qeJudgeModelInherit}
              onChange={(event) => setJudgeModel(event.target.value)}
              className="mt-1 w-full rounded-[4px] border border-[var(--paper-border)] bg-[var(--paper-bg)] px-2 py-1 type-value font-mono"
            />
          </label>

          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className="flex items-center gap-1">
                <span className="type-label text-[var(--ink-secondary)]">
                  {t.doctor.qeThreshold}
                </span>
                <InfoTip label={t.doctor.qeThreshold}>{t.doctor.qeThresholdHelp}</InfoTip>
              </span>
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={threshold}
                onChange={(event) => setThreshold(event.target.value)}
                className="mt-1 w-full rounded-[4px] border border-[var(--paper-border)] bg-[var(--paper-bg)] px-2 py-1 type-value font-mono"
              />
            </label>
            <label className="block">
              <span className="flex items-center gap-1">
                <span className="type-label text-[var(--ink-secondary)]">
                  {t.doctor.qePassSample}
                </span>
                <InfoTip label={t.doctor.qePassSample}>{t.doctor.qePassSampleHelp}</InfoTip>
              </span>
              <input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={passSample}
                onChange={(event) => setPassSample(event.target.value)}
                className="mt-1 w-full rounded-[4px] border border-[var(--paper-border)] bg-[var(--paper-bg)] px-2 py-1 type-value font-mono"
              />
            </label>
          </div>

          <label className="flex items-start gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={sentences}
              onChange={(event) => setSentences(event.target.checked)}
              className="mt-0.5 shrink-0"
            />
            <span className="min-w-0">
              <span className="type-value">{t.doctor.qeSentences}</span>
              <span className="block type-help text-[var(--ink-muted)] leading-relaxed">
                {t.doctor.qeSentencesHelp}
              </span>
            </span>
          </label>

          {error ? <p className="type-help text-[#b91c1c]">{error}</p> : null}

          <div className="flex items-center gap-2 pt-1">
            <Button
              size="sm"
              onClick={() => void handleSave()}
              disabled={saving}
              className="min-h-[40px] sm:min-h-0 px-3"
            >
              {saving ? (
                <>
                  <Loader2 size={12} className="animate-spin" /> {t.doctor.qeSaving}
                </>
              ) : (
                t.doctor.qeSave
              )}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setEditing(false)}
              disabled={saving}
              className="min-h-[40px] sm:min-h-0 px-3"
            >
              {t.doctor.qeCancel}
            </Button>
          </div>
        </div>
      ) : (
        <div className="p-3.5 space-y-2">
          {/* The headline fact: a run with the heuristic engine has no average
              score, and 0.0 in the report is a placeholder rather than a bad
              result. */}
          <div
            className={`type-value font-semibold ${
              settings.measurable ? 'text-[#15803d]' : 'text-[#b45309]'
            }`}
          >
            {settings.measurable ? t.doctor.qeMeasured : t.doctor.qeNotMeasured}
          </div>
          {!settings.measurable ? (
            <p className="type-help text-[var(--ink-secondary)] leading-relaxed max-w-xl">
              {t.doctor.qeNotMeasuredHelp}
            </p>
          ) : null}

          <dl className="space-y-1.5 type-value font-mono pt-1">
            <div className="flex flex-col sm:flex-row sm:justify-between sm:items-baseline gap-0.5 sm:gap-3">
              <dt className="text-[var(--ink-secondary)] shrink-0">{t.doctor.qeEngine}</dt>
              <dd className="text-[var(--ink-primary)] font-semibold truncate sm:text-right">
                {settings.judge_enabled ? t.doctor.qeEngineTiered : t.doctor.qeEngineHeuristic}
              </dd>
            </div>
            <div className="flex flex-col sm:flex-row sm:justify-between sm:items-baseline gap-0.5 sm:gap-3">
              <dt className="text-[var(--ink-secondary)] shrink-0">{t.doctor.qeJudge}</dt>
              <dd className="text-[var(--ink-primary)] truncate sm:text-right">
                {settings.judge_enabled ? t.doctor.qeJudgeOn : t.doctor.qeJudgeOff}
                {settings.judge_model ? ` · ${settings.judge_model}` : ''}
              </dd>
            </div>
            <div className="flex flex-col sm:flex-row sm:justify-between sm:items-baseline gap-0.5 sm:gap-3">
              <dt className="text-[var(--ink-secondary)] shrink-0">{t.doctor.qeJudgeProvider}</dt>
              <dd className="text-[var(--ink-primary)] truncate sm:text-right">
                {settings.judge_provider ?? t.doctor.qeJudgeProviderInherit}
              </dd>
            </div>
            <div className="flex flex-col sm:flex-row sm:justify-between sm:items-baseline gap-0.5 sm:gap-3">
              <dt className="text-[var(--ink-secondary)] shrink-0">{t.doctor.qeThreshold}</dt>
              <dd className="text-[var(--ink-primary)] sm:text-right">
                {Number(settings.threshold ?? 0).toFixed(2)}
              </dd>
            </div>
            <div className="flex flex-col sm:flex-row sm:justify-between sm:items-baseline gap-0.5 sm:gap-3">
              <dt className="text-[var(--ink-secondary)] shrink-0">{t.doctor.qePassSample}</dt>
              <dd className="text-[var(--ink-primary)] sm:text-right">
                {Math.round(Number(settings.judge_pass_sample ?? 0) * 100)}%
              </dd>
            </div>
          </dl>

          {/* A hand-edited file can hold the state this card refuses to produce.
              Naming it beats silently showing the label as if it worked. */}
          {inconsistent ? (
            <p className="type-help text-[#b45309] leading-relaxed">{t.doctor.qeInconsistent}</p>
          ) : null}

          {/* The judge is on but samples nothing: a book whose every block
              passes still has no average score. This is the one setting that
              makes "measured" false while every switch reads on. */}
          {settings.measurable && Number(settings.judge_pass_sample ?? 0) === 0 ? (
            <p className="type-help text-[#b45309] leading-relaxed">{t.doctor.qePassSampleZero}</p>
          ) : null}

          {(settings.env_overrides ?? []).length > 0 ? (
            <p className="type-help text-[#b45309] leading-relaxed">
              {t.doctor.qeEnvOverride.replace('{vars}', (settings.env_overrides ?? []).join(', '))}
            </p>
          ) : null}

          {saved ? (
            <p className="type-help text-[#15803d] flex items-center gap-1">
              <Check size={12} /> {t.doctor.qeSaved}
            </p>
          ) : null}

          <p className="type-help text-[var(--ink-muted)] leading-relaxed">
            {t.doctor.qeConfigSource}
          </p>
        </div>
      )}
    </div>
  )
}
