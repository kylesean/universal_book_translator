import React, { useCallback, useEffect, useState } from 'react'
import { KeyRound, Loader2, ShieldAlert } from 'lucide-react'
import { checkHealth, getStoredApiKey, signIn } from '@/api/client'
import { useI18n } from '@/i18n/useI18n'
import { Button } from '@/components/ui/button'

interface AuthGateProps {
  children: React.ReactNode
}

/**
 * Sign-in gate for a keyed server.
 *
 * A keyless (development) server answers /health without a key and the gate
 * passes straight through, so the loopback console is unchanged. When the
 * server demands a key, every panel's request would 401 and read as a broken
 * UI; the gate asks for the key once and exchanges it for the session cookie
 * that even the headerless requests (SSE, previews, downloads) carry.
 *
 * A rejected *stored* key clears itself and re-renders the form rather than
 * silently retrying, so a rotated key surfaces as a prompt, not a stuck spinner.
 */
export function AuthGate({ children }: AuthGateProps) {
  const { t } = useI18n()
  const [state, setState] = useState<'checking' | 'signed-out' | 'signed-in'>('checking')
  const [key, setKey] = useState('')
  const [busy, setBusy] = useState(false)
  const [failed, setFailed] = useState(false)

  const verify = useCallback(async () => {
    try {
      await checkHealth()
    } catch {
      // A failed health probe is not an auth answer; the panels' own error
      // states handle an unreachable server, so do not block the console.
      setState('signed-in')
      return
    }
    // Health is exempt from the gate, so it proves reachability but not the
    // key. Probe a gated route: the session cookie (when set) makes it pass
    // even without a stored key.
    const stored = getStoredApiKey()
    const res = await fetch('/jobs?limit=1', {
      headers: stored ? { 'X-API-Key': stored } : undefined,
    })
    if (res.ok) {
      setState('signed-in')
      return
    }
    if (res.status === 401) {
      setState('signed-out')
      return
    }
    setState('signed-in')
  }, [])

  useEffect(() => {
    // Mount-time probe: every setState inside `verify` runs after its `await`,
    // so there is no synchronous cascade for the rule to be about.
    // oxlint-disable-next-line react/set-state-in-effect
    void verify()
  }, [verify])

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!key.trim() || busy) return
    setBusy(true)
    setFailed(false)
    let ok = false
    try {
      ok = await signIn(key.trim())
    } catch {
      // A network-layer reject (server unreachable between the probe and this
      // submit) used to skip the reset below: `busy` stayed true forever and
      // the only way out was a full reload. `signIn` has already handled the
      // HTTP-level rejections; this is purely the fetch throwing.
      ok = false
    } finally {
      setBusy(false)
    }
    if (ok) {
      setState('signed-in')
    } else {
      setFailed(true)
    }
  }

  if (state === 'checking') {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="flex items-center gap-3 rounded-xl border border-border/80 bg-card px-4 py-3 text-sm text-muted-foreground shadow-card">
          <Loader2 className="size-4 animate-spin" />
          {t.doctor.auth.checking}
        </div>
      </div>
    )
  }

  if (state === 'signed-out') {
    return (
      <div className="flex h-screen items-center justify-center bg-background p-6">
        <form
          onSubmit={submit}
          className="w-full max-w-md rounded-2xl border border-border/80 bg-card p-8 shadow-floating"
        >
          <div className="mb-5 flex items-center gap-3">
            <span className="flex size-10 items-center justify-center rounded-xl border border-amber-500/25 bg-amber-500/10 text-amber-600 dark:text-amber-400">
              <ShieldAlert className="size-5" />
            </span>
            <h1 className="text-lg font-semibold tracking-[-0.02em] text-foreground">
              {t.doctor.auth.title}
            </h1>
          </div>
          <p className="mb-6 text-sm text-muted-foreground">{t.doctor.auth.subtitle}</p>
          <label
            htmlFor="ubt-api-key"
            className="mb-1.5 block font-mono text-xs font-medium uppercase tracking-[0.12em] text-muted-foreground"
          >
            {t.doctor.auth.keyLabel}
          </label>
          <div className="mb-4 flex items-center gap-2 rounded-lg border border-input bg-background px-3 shadow-2xs transition-colors focus-within:border-ring/50 focus-within:ring-2 focus-within:ring-ring/15">
            <KeyRound className="size-4 shrink-0 text-muted-foreground" />
            <input
              id="ubt-api-key"
              type="password"
              autoFocus
              autoComplete="current-password"
              value={key}
              onChange={(event) => setKey(event.target.value)}
              placeholder={t.doctor.auth.keyPlaceholder}
              className="w-full bg-transparent py-2.5 text-sm text-foreground outline-none"
            />
          </div>
          {failed && <p className="mb-4 text-sm text-destructive">{t.doctor.auth.failed}</p>}
          <Button
            type="submit"
            className="w-full"
            disabled={busy || !key.trim()}
          >
            {busy && <Loader2 className="size-4 animate-spin" />}
            {busy ? t.doctor.auth.checking : t.doctor.auth.signIn}
          </Button>
        </form>
      </div>
    )
  }

  return <>{children}</>
}
