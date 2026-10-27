import { useCallback, useEffect, useState } from 'react'
import { Check, Loader2, Pencil, X } from 'lucide-react'

import {
  getQeSettings,
  saveQeSettings,
  type QESettings,
  type QESettingsRequest,
} from '@/api/client'
import { Alert, AlertDescription } from '@/components/ui/Alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { useI18n } from '@/i18n/useI18n'
import { cn } from '@/lib/utils'
import { DoctorField, DoctorMetric } from './components'
import { InfoTip } from '@/components/ui/InfoTip'

interface QESettingsCardProps {
  /** Provider names a judge may point at, for the select. */
  providerNames: string[]
}

const INHERIT = '__inherit__'

/**
 * The quality-estimation gate: what scores a run, and the only form that says so.
 *
 * ``qe_engine`` and ``qe_judge_enabled`` are written together. ``tiered`` with
 * the judge off scores nothing, so exposing them as independent switches would
 * let the console create the exact state this card exists to explain.
 */
export function QESettingsCard({ providerNames }: QESettingsCardProps) {
  const { t } = useI18n()
  const [settings, setSettings] = useState<QESettings | null>(null)
  const [loadFailed, setLoadFailed] = useState(false)
  const [editing, setEditing] = useState(false)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)
  const [tiered, setTiered] = useState(false)
  const [judgeProvider, setJudgeProvider] = useState('')
  const [judgeModel, setJudgeModel] = useState('')
  const [threshold, setThreshold] = useState('0.75')
  const [passSample, setPassSample] = useState('0')
  const [sentences, setSentences] = useState(true)

  const adopt = useCallback((next: QESettings) => {
    setSettings(next)
    setLoadFailed(false)
    // The effective choice, not the raw engine label: a hand-edited "tiered"
    // with the judge off does not score anything.
    setTiered(Boolean(next.judge_enabled))
    setJudgeProvider(next.judge_provider ?? '')
    // The raw setting: seeding the resolved model would pin an inherited value.
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
      const body: QESettingsRequest = {
        qe_engine: tiered ? 'tiered' : 'heuristic',
        qe_judge_enabled: tiered,
        qe_judge_provider: judgeProvider || null,
        qe_judge_model: judgeModel.trim() || null,
        qe_threshold: thresholdValue,
        qe_judge_pass_sample: sampleValue,
        qe_judge_sentences: sentences,
      }
      adopt(await saveQeSettings(body))
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
    <section className="min-w-0 overflow-hidden rounded-xl border border-border/80 bg-card">
      <header className="flex items-start justify-between gap-3 border-b border-border/70 px-4 py-3">
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <h2 className="text-sm font-semibold">{t.doctor.qeTitle}</h2>
            <InfoTip label={t.doctor.qeTitle}>{t.doctor.qeHelp}</InfoTip>
          </div>
        </div>
        {editing ? (
          <Button
            size="icon"
            variant="ghost"
            onClick={() => setEditing(false)}
            aria-label={t.doctor.qeCancel}
            className="size-8 shrink-0"
          >
            <X />
          </Button>
        ) : null}
      </header>

      {loadFailed ? (
        <div className="px-4 py-8 text-center text-sm text-muted-foreground">
          {t.doctor.qeLoadFailed}
        </div>
      ) : !settings ? (
        <div className="flex items-center justify-center px-4 py-10 text-muted-foreground">
          <Loader2 className="size-4 animate-spin" />
        </div>
      ) : editing ? (
        <div className="space-y-4 p-4">
          <DoctorField label={t.doctor.qeEngine} hint={t.doctor.qeEngineHelp}>
            <Select
              value={tiered ? 'tiered' : 'heuristic'}
              onValueChange={(value) => {
                const next = value === 'tiered'
                setTiered(next)
                // A zero pass sample only scores deterministically ambiguous
                // blocks, so a clean book can still report "not measured".
                if (next && Number(passSample) === 0) setPassSample('0.15')
              }}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="heuristic">{t.doctor.qeEngineHeuristic}</SelectItem>
                <SelectItem value="tiered">{t.doctor.qeEngineTiered}</SelectItem>
              </SelectContent>
            </Select>
          </DoctorField>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <DoctorField label={t.doctor.qeJudgeProvider}>
              <Select
                value={judgeProvider || INHERIT}
                onValueChange={(value) => setJudgeProvider(value === INHERIT ? '' : value)}
              >
                <SelectTrigger className="font-mono">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={INHERIT}>{t.doctor.qeJudgeProviderInherit}</SelectItem>
                  {/* Keep a name that resolves outside the declared list; without
                      it a save would silently clear a working judge endpoint. */}
                  {judgeProvider && !providerNames.includes(judgeProvider) ? (
                    <SelectItem value={judgeProvider} className="font-mono">
                      {judgeProvider}
                    </SelectItem>
                  ) : null}
                  {providerNames.map((name) => (
                    <SelectItem key={name} value={name} className="font-mono">
                      {name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </DoctorField>
            <DoctorField label={t.doctor.qeJudgeModel}>
              <Input
                value={judgeModel}
                placeholder={t.doctor.qeJudgeModelInherit}
                onChange={(event) => setJudgeModel(event.target.value)}
                className="font-mono"
              />
            </DoctorField>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <DoctorField
              label={
                <span className="flex items-center gap-1">
                  {t.doctor.qeThreshold}
                  <InfoTip label={t.doctor.qeThreshold}>{t.doctor.qeThresholdHelp}</InfoTip>
                </span>
              }
            >
              <Input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={threshold}
                onChange={(event) => setThreshold(event.target.value)}
                className="font-mono"
              />
            </DoctorField>
            <DoctorField
              label={
                <span className="flex items-center gap-1">
                  {t.doctor.qePassSample}
                  <InfoTip label={t.doctor.qePassSample}>{t.doctor.qePassSampleHelp}</InfoTip>
                </span>
              }
            >
              <Input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={passSample}
                onChange={(event) => setPassSample(event.target.value)}
                className="font-mono"
              />
            </DoctorField>
          </div>

          <div className="flex items-start justify-between gap-4 rounded-lg border border-border/70 p-3">
            <div className="min-w-0">
              <div className="text-xs font-medium">{t.doctor.qeSentences}</div>
              <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                {t.doctor.qeSentencesHelp}
              </p>
            </div>
            <Switch checked={sentences} onCheckedChange={setSentences} />
          </div>

          {error ? (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          ) : null}

          <div className="flex justify-end gap-2">
            <Button
              variant="ghost"
              onClick={() => setEditing(false)}
              disabled={saving}
              className="px-3"
            >
              {t.doctor.qeCancel}
            </Button>
            <Button onClick={() => void handleSave()} disabled={saving} className="px-4">
              {saving ? (
                <>
                  <Loader2 className="animate-spin" />
                  {t.doctor.qeSaving}
                </>
              ) : (
                t.doctor.qeSave
              )}
            </Button>
          </div>
        </div>
      ) : (
        <div className="space-y-3 p-4">
          <div className="flex flex-wrap items-center gap-2">
            <Badge
              variant="outline"
              className={cn(
                settings.measurable
                  ? 'border-emerald-600/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
                  : 'border-amber-600/25 bg-amber-500/10 text-amber-700 dark:text-amber-400'
              )}
            >
              {settings.measurable ? t.doctor.qeMetricMeasured : t.doctor.qeMetricNotMeasured}
            </Badge>
          </div>
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            <DoctorMetric
              label={t.doctor.thresholdMetric}
              value={Number(settings.threshold ?? 0).toFixed(2)}
              mono
            />
            <DoctorMetric
              label={t.doctor.sampleMetric}
              value={`${Math.round(Number(settings.judge_pass_sample ?? 0) * 100)}%`}
              mono
            />
          </div>
          <div className="space-y-1.5 text-xs">
            <div className="flex items-center justify-between gap-3">
              <span className="text-muted-foreground">{t.doctor.qeJudge}</span>
              <span className="truncate font-mono">
                {settings.judge_enabled ? t.doctor.qeJudgeOn : t.doctor.qeJudgeOff}
                {settings.judge_model ? ` · ${settings.judge_model}` : ''}
              </span>
            </div>
            <div className="flex items-center justify-between gap-3">
              <span className="text-muted-foreground">{t.doctor.qeJudgeProvider}</span>
              <span className="truncate font-mono">
                {settings.judge_provider ?? t.doctor.qeJudgeProviderInherit}
              </span>
            </div>
          </div>

          {!settings.measurable ? (
            <p className="text-xs leading-relaxed text-muted-foreground">
              {t.doctor.qeNotMeasuredHelp}
            </p>
          ) : null}
          {inconsistent ? (
            <p className="text-xs leading-relaxed text-amber-700 dark:text-amber-400">
              {t.doctor.qeInconsistent}
            </p>
          ) : null}
          {settings.measurable && Number(settings.judge_pass_sample ?? 0) === 0 ? (
            <p className="text-xs leading-relaxed text-amber-700 dark:text-amber-400">
              {t.doctor.qePassSampleZero}
            </p>
          ) : null}
          {(settings.env_overrides ?? []).length > 0 ? (
            <p className="text-xs leading-relaxed text-amber-700 dark:text-amber-400">
              {t.doctor.qeEnvOverride.replace(
                '{vars}',
                (settings.env_overrides ?? []).join(', '),
              )}
            </p>
          ) : null}
          {saved ? (
            <p className="flex items-center gap-1 text-xs text-emerald-700 dark:text-emerald-400">
              <Check className="size-3.5" />
              {t.doctor.qeSaved}
            </p>
          ) : null}
          <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border/60 pt-3">
            <p className="text-xs leading-relaxed text-muted-foreground">
              {t.doctor.qeConfigSource}
            </p>
            <Button
              size="sm"
              variant="outline"
              onClick={startEditing}
              className="h-8 px-2.5 text-xs"
            >
              <Pencil />
              {t.doctor.qeEdit}
            </Button>
          </div>
        </div>
      )}
    </section>
  )
}
