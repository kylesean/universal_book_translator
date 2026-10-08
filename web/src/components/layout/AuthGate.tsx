import React, { useCallback, useEffect, useState } from 'react'
import { KeyRound, Loader2, ShieldAlert } from 'lucide-react'
import { checkHealth, getStoredApiKey, signIn } from '@/api/client'
import { useI18n } from '@/i18n/I18nContext'

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
    void verify()
  }, [verify])

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!key.trim() || busy) return
    setBusy(true)
    setFailed(false)
    const ok = await signIn(key.trim())
    setBusy(false)
    if (ok) {
      setState('signed-in')
    } else {
      setFailed(true)
    }
  }

  if (state === 'checking') {
    return (
      <div className="flex h-screen items-center justify-center bg-[color:var(--paper-bg)]">
        <Loader2 className="h-6 w-6 animate-spin text-[color:var(--ink-muted)]" />
      </div>
    )
  }

  if (state === 'signed-out') {
    return (
      <div className="flex h-screen items-center justify-center bg-[color:var(--paper-bg)] p-6">
        <form
          onSubmit={submit}
          className="w-full max-w-md rounded-xl border border-[color:var(--rule)] bg-[color:var(--paper-raised)] p-8 shadow-sm"
        >
          <div className="mb-4 flex items-center gap-3">
            <ShieldAlert className="h-6 w-6 text-[color:var(--accent)]" />
            <h1 className="font-serif text-lg font-semibold text-[color:var(--ink)]">
              {t.doctor.auth.title}
            </h1>
          </div>
          <p className="mb-6 text-sm text-[color:var(--ink-muted)]">{t.doctor.auth.subtitle}</p>
          <label
            htmlFor="ubt-api-key"
            className="mb-1 block text-xs font-medium uppercase tracking-wide text-[color:var(--ink-muted)]"
          >
            {t.doctor.auth.keyLabel}
          </label>
          <div className="mb-4 flex items-center gap-2 rounded-lg border border-[color:var(--rule)] bg-[color:var(--paper-bg)] px-3">
            <KeyRound className="h-4 w-4 shrink-0 text-[color:var(--ink-muted)]" />
            <input
              id="ubt-api-key"
              type="password"
              autoFocus
              autoComplete="current-password"
              value={key}
              onChange={(event) => setKey(event.target.value)}
              placeholder={t.doctor.auth.keyPlaceholder}
              className="w-full bg-transparent py-2 text-sm text-[color:var(--ink)] outline-none"
            />
          </div>
          {failed && <p className="mb-4 text-sm text-red-600">{t.doctor.auth.failed}</p>}
          <button
            type="submit"
            disabled={busy || !key.trim()}
            className="flex w-full items-center justify-center gap-2 rounded-lg bg-[color:var(--accent)] px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            {busy && <Loader2 className="h-4 w-4 animate-spin" />}
            {busy ? t.doctor.auth.checking : t.doctor.auth.signIn}
          </button>
        </form>
      </div>
    )
  }

  return <>{children}</>
}
