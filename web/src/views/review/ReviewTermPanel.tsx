import { useCallback, useEffect, useState } from 'react'
import { Loader2, Sparkles, Wand2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  getBlockTerms,
  propagateTerm,
  type Segment,
  type TermViolation,
} from '@/api/client'
import { useI18n } from '@/i18n/useI18n'

type TermScope = 'block' | 'subsequent' | 'all'

interface TermPanelProps {
  jobId: string
  segment: Segment
  onReplaced: (replacements: number) => void
}

export function TermPanel({ jobId, segment, onReplaced }: TermPanelProps) {
  const { t } = useI18n()
  const [violations, setViolations] = useState<TermViolation[] | null>(null)
  const [scopes, setScopes] = useState<Record<string, TermScope>>({})
  const [confirmViolation, setConfirmViolation] = useState<{
    violation: TermViolation
    scope: TermScope
  } | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const reloadTerms = useCallback(() => {
    getBlockTerms(jobId, segment.block_id)
      .then((report) => setViolations(report.violations))
      .catch(() => setViolations([]))
  }, [jobId, segment.block_id])

  useEffect(() => {
    let alive = true
    getBlockTerms(jobId, segment.block_id)
      .then((report) => {
        if (alive) setViolations(report.violations)
      })
      .catch(() => {
        if (alive) setViolations([])
      })
    return () => {
      alive = false
    }
  }, [jobId, segment.block_id, segment.delivered_text])

  const getScope = (v: TermViolation): TermScope => {
    const key = `${v.surface}:${v.expected}`
    if (scopes[key]) return scopes[key]
    if (v.cascade_subsequent > 1) return 'subsequent'
    if (v.cascade_all > 1) return 'all'
    return 'block'
  }

  const setScopeFor = (v: TermViolation, s: TermScope) => {
    const key = `${v.surface}:${v.expected}`
    setScopes((prev) => ({ ...prev, [key]: s }))
    if (confirmViolation?.violation === v) {
      setConfirmViolation({ violation: v, scope: s })
    }
  }

  const handleTriggerReplace = (violation: TermViolation) => {
    const currentScope = getScope(violation)
    if (currentScope === 'block') {
      void executeReplace(violation, 'block')
    } else {
      setConfirmViolation({ violation, scope: currentScope })
    }
  }

  const executeReplace = async (violation: TermViolation, scope: TermScope) => {
    setBusy(true)
    setError(null)
    setConfirmViolation(null)
    try {
      const res = await propagateTerm(jobId, {
        block_id: segment.block_id,
        surface: violation.surface,
        expected: violation.expected,
        scope,
      })
      onReplaced(res.replacements)
      reloadTerms()
    } catch (err) {
      setError(err instanceof Error ? err.message : t.review.propagateFailed)
    } finally {
      setBusy(false)
    }
  }

  if (violations === null || violations.length === 0) return null

  return (
    <div className="mt-3 divide-y divide-amber-600/20 rounded-lg border border-amber-600/35 bg-amber-500/5">
      {violations.map((violation) => {
        const key = `${violation.surface}:${violation.expected}`
        const currentScope = getScope(violation)
        const isConfirming = confirmViolation?.violation === violation
        const countAll = Math.max(1, violation.cascade_all)
        const countSubsequent = Math.max(1, violation.cascade_subsequent)
        const targetCount =
          currentScope === 'all'
            ? countAll
            : currentScope === 'subsequent'
              ? countSubsequent
              : 1

        return (
          <div key={key} className="px-2.5 py-2 flex flex-col gap-2 text-xs">
            <div className="flex flex-wrap items-center gap-2">
              <span className="font-mono text-muted-foreground">{t.review.recommendedTerm}</span>
              <span className="line-through text-amber-700 dark:text-amber-400 font-mono">{violation.surface}</span>
              <span className="text-muted-foreground">→</span>
              <span className="font-semibold text-emerald-700 dark:text-emerald-400 font-mono">{violation.expected}</span>

              {violation.cascade_all > 1 && (
                <div className="flex items-center gap-0.5 bg-muted/40 p-0.5 rounded border border-border ml-auto">
                  <button
                    type="button"
                    onClick={() => setScopeFor(violation, 'block')}
                    className={`px-1.5 py-0.5 rounded text-xs transition-colors cursor-pointer ${
                      currentScope === 'block'
                        ? 'bg-card text-foreground font-bold shadow-xs'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                    title={t.review.scopeBlock}
                  >
                    {t.review.scopeBlock} (1)
                  </button>
                  {violation.cascade_subsequent > 1 && (
                    <button
                      type="button"
                      onClick={() => setScopeFor(violation, 'subsequent')}
                      className={`px-1.5 py-0.5 rounded text-xs transition-colors cursor-pointer ${
                        currentScope === 'subsequent'
                          ? 'bg-card text-foreground font-bold shadow-xs'
                          : 'text-muted-foreground hover:text-foreground'
                      }`}
                      title={t.review.scopeSubsequent}
                    >
                      {t.review.scopeSubsequent} ({countSubsequent})
                    </button>
                  )}
                  <button
                    type="button"
                    onClick={() => setScopeFor(violation, 'all')}
                    className={`px-1.5 py-0.5 rounded text-xs transition-colors cursor-pointer ${
                      currentScope === 'all'
                        ? 'bg-card text-foreground font-bold shadow-xs'
                        : 'text-muted-foreground hover:text-foreground'
                    }`}
                    title={t.review.scopeAll}
                  >
                    {t.review.scopeAll} ({countAll})
                  </button>
                </div>
              )}

              <Button
                onClick={() => handleTriggerReplace(violation)}
                variant="default"
                size="sm"
                disabled={busy}
                className={`text-xs min-h-[28px] md:min-h-0 h-7 md:h-6 px-2.5 font-bold ${violation.cascade_all <= 1 ? 'ml-auto' : ''}`}
              >
                {busy ? (
                  <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                ) : (
                  <Wand2 className="h-3 w-3 mr-1" />
                )}
                {busy
                  ? t.review.cascadeApplying
                  : currentScope === 'block'
                    ? t.review.replaceTerm
                    : `${t.review.replaceTerm} (${targetCount})`}
              </Button>
            </div>

            {isConfirming && (
              <div className="w-full mt-1 p-2.5 rounded-md bg-amber-500/10 border border-amber-600/25 space-y-2 animate-in fade-in duration-150">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-foreground text-xs">
                    {currentScope === 'all'
                      ? t.review.cascadeConfirmTitle
                      : `${t.review.scopeSubsequent} · ${t.review.cascadeConfirmTitle}`}
                  </span>
                  <span className="text-xs font-mono px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-700 dark:text-amber-400 font-bold">
                    {t.review.scopeCount.replace('{n}', String(targetCount))}
                  </span>
                </div>

                <div className="flex items-center gap-2 p-1.5 rounded bg-card border border-border font-mono text-xs">
                  <span className="line-through text-amber-700 dark:text-amber-400 font-medium bg-amber-500/10 px-1.5 py-0.5 rounded">
                    {violation.surface}
                  </span>
                  <span className="text-muted-foreground">➔</span>
                  <span className="text-emerald-700 dark:text-emerald-400 font-bold bg-emerald-500/10 px-1.5 py-0.5 rounded">
                    {violation.expected}
                  </span>
                </div>

                <div className="text-muted-foreground text-xs leading-relaxed">
                  {t.review.cascadeConfirmDesc
                    .replace('{n}', String(targetCount))
                    .replace('{surface}', violation.surface)
                    .replace('{expected}', violation.expected)}
                </div>

                <div className="flex items-center gap-1.5 text-muted-foreground text-xs pt-0.5">
                  <Sparkles className="h-3 w-3 text-amber-500 shrink-0" />
                  <span>{t.review.cascadeTmSync}</span>
                </div>

                <div className="flex items-center gap-2 pt-1 border-t border-amber-600/20">
                  <Button
                    onClick={() => executeReplace(violation, currentScope)}
                    variant="default"
                    size="sm"
                    disabled={busy}
                    className="h-7 min-h-[28px] bg-amber-600 px-2.5 text-xs font-bold text-white hover:bg-amber-700 md:h-6 md:min-h-0"
                  >
                    {busy ? (
                      <Loader2 className="h-3 w-3 animate-spin mr-1" />
                    ) : (
                      <Wand2 className="h-3 w-3 mr-1" />
                    )}
                    {t.review.cascadeConfirmProceed} ({targetCount})
                  </Button>
                  <Button
                    onClick={() => setConfirmViolation(null)}
                    variant="secondary"
                    size="sm"
                    disabled={busy}
                    className="text-xs min-h-[28px] md:min-h-0 h-7 md:h-6 px-2.5"
                  >
                    {t.review.cascadeConfirmCancel}
                  </Button>
                </div>
              </div>
            )}
          </div>
        )
      })}
      {error && <div className="px-2.5 py-1.5 text-xs text-amber-700 dark:text-amber-400">{error}</div>}
    </div>
  )
}
