import React, { useEffect, useState } from 'react'
import { NavLink, useLocation, useMatch } from 'react-router-dom'
import {
  PlusCircle,
  Activity,
  ShieldCheck,
  BookOpen,
  Stethoscope,
  SplitSquareVertical,
  Globe,
  Terminal,
  FileSpreadsheet,
  LogOut,
  Menu,
  X,
} from 'lucide-react'
import {
  checkHealth,
  getSystemInfo,
  getStoredApiKey,
  signOut,
  type HealthResponse,
  type SystemInfo,
} from '@/api/client'
import { useI18n } from '@/i18n/useI18n'

interface ConsoleLayoutProps {
  children: React.ReactNode
}

interface NavItem {
  to: string
  label: string
  icon: React.ComponentType<{ className?: string }>
  accent?: boolean
  disabled?: boolean
}

export function ConsoleLayout({ children }: ConsoleLayoutProps) {
  const { t, language, setLanguage } = useI18n()
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [info, setInfo] = useState<SystemInfo | null>(null)
  const [paperTone, setPaperTone] = useState<'cotton' | 'dowling'>(() => {
    return (localStorage.getItem('ubt_paper_tone') as 'cotton' | 'dowling') || 'cotton'
  })
  const [mobileNavOpen, setMobileNavOpen] = useState(false)
  const location = useLocation()

  // Auto-close mobile drawer when navigation occurs
  useEffect(() => {
    // oxlint-disable-next-line react/set-state-in-effect
    setMobileNavOpen(false)
  }, [location.pathname])

  // Close drawer on Escape key
  useEffect(() => {
    if (!mobileNavOpen) return
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setMobileNavOpen(false)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [mobileNavOpen])

  // The Quality / Review screens are job-scoped, so their sidebar links follow
  // the job currently in the URL (parsed here because the layout sits outside
  // the route elements).
  const jobMatch = useMatch('/jobs/:jobId/*')
  const currentJobId = jobMatch?.params.jobId ?? null

  useEffect(() => {
    if (paperTone === 'dowling') {
      document.documentElement.setAttribute('data-paper-tone', 'dowling')
    } else {
      document.documentElement.removeAttribute('data-paper-tone')
    }
    localStorage.setItem('ubt_paper_tone', paperTone)
  }, [paperTone])

  useEffect(() => {
    let isMounted = true
    const probe = async () => {
      try {
        const res = await checkHealth()
        if (isMounted) {
          setHealth(res)
        }
      } catch {
        if (isMounted) {
          setHealth(null)
        }
      }
    }
    probe()
    // The security boundary is read once (it does not change while the server
    // runs) so the status line can show the real port and gate state.
    getSystemInfo()
      .then((res) => {
        if (isMounted) setInfo(res)
      })
      .catch(() => {
        if (isMounted) setInfo(null)
      })
    const timer = setInterval(probe, 10000)
    return () => {
      isMounted = false
      clearInterval(timer)
    }
  }, [])

  const handleSignOut = async () => {
    await signOut()
    // A full reload drops every cached panel and lands back on the gate.
    window.location.reload()
  }

  // Signing out only means something when the server actually gates on a key.
  const canSignOut = Boolean(info?.auth_enabled) && Boolean(getStoredApiKey())

  const navItems: NavItem[] = [
    { to: '/wizard', label: t.nav.newJob, icon: PlusCircle },
    { to: '/jobs', label: t.nav.missionControl, icon: Activity },
    {
      to: currentJobId ? `/jobs/${currentJobId}/quality` : '/jobs',
      label: t.nav.qualityGate,
      icon: ShieldCheck,
      disabled: !currentJobId,
    },
    { to: '/assets', label: t.nav.languageAssets, icon: BookOpen },
    { to: '/system', label: t.nav.systemDoctor, icon: Stethoscope },
  ]

  const workbenchItems: NavItem[] = [
    {
      to: currentJobId ? `/jobs/${currentJobId}/review` : '/jobs',
      label: t.nav.reviewWorkbench,
      icon: SplitSquareVertical,
      accent: true,
      disabled: !currentJobId,
    },
  ]

  const renderNav = (items: NavItem[], onSelect?: () => void) =>
    items.map((item) => {
      const Icon = item.icon
      // Keyed by label, not by target: with no job selected, three entries fall
      // back to `/jobs` (Mission Control, plus the two job-scoped panes in their
      // disabled state), and React reported "two children with the same key,
      // `/jobs`" on every render of that screen. The label is unique per entry;
      // the target is not.
      if (item.disabled) {
        return (
          <div
            key={item.label}
            title={t.mission.noActiveJobTitle}
            className="w-full flex items-center gap-2.5 px-3 py-2.5 md:px-2.5 md:py-2 rounded-[5px] text-xs font-medium text-[var(--ink-muted)] opacity-60 cursor-not-allowed border border-transparent min-h-[40px] md:min-h-0 select-none"
          >
            <Icon className="h-4 w-4 md:h-3.5 md:w-3.5 shrink-0 text-[var(--ink-muted)]" />
            <div className="flex-1 truncate">
              <div className="leading-tight">{item.label}</div>
            </div>
          </div>
        )
      }
      return (
        <NavLink
          key={item.label}
          to={item.to}
          // `/jobs` must NOT be `end`: the job detail screen (`/jobs/:jobId`) is
          // the same MissionControl view (App.tsx), so `end` left no sidebar
          // entry highlighted on it — "you are here" disappeared on the screen
          // the operator spends the most time on.
          end={item.to === '/wizard'}
          onClick={() => onSelect?.()}
          className={({ isActive }) =>
            `w-full flex items-center gap-2.5 px-3 py-2.5 md:px-2.5 md:py-2 rounded-[5px] text-xs font-medium transition-all text-left min-h-[40px] md:min-h-0 ${
              isActive
                ? 'bg-[var(--paper-subsurface)] text-[var(--ink-primary)] font-semibold border border-[var(--paper-border)] shadow-xs'
                : 'text-[var(--ink-secondary)] hover:bg-[var(--paper-subsurface)] hover:text-[var(--ink-primary)] border border-transparent'
            }`
          }
        >
          {({ isActive }) => (
            <>
              <Icon
                className={`h-4 w-4 md:h-3.5 md:w-3.5 shrink-0 ${
                  isActive
                    ? item.accent
                      ? 'text-[#15803d]'
                      : 'text-[var(--ink-primary)]'
                    : 'text-[var(--ink-muted)]'
                }`}
              />
              <div className="flex-1 truncate">
                <div className="leading-tight">{item.label}</div>
              </div>
            </>
          )}
        </NavLink>
      )
    })

  return (
    <div className="flex flex-col md:flex-row h-screen h-[100dvh] w-full max-w-full overflow-hidden bg-[var(--paper-bg)] text-[var(--ink-primary)]">
      {/* Mobile Top Header (< md) */}
      <header className="min-h-14 h-[calc(3.5rem+env(safe-area-inset-top,0px))] px-3.5 border-b border-[var(--paper-border)] bg-[var(--paper-surface)] flex md:hidden items-center justify-between shrink-0 select-none z-30 pt-[env(safe-area-inset-top,0px)]">
        <div className="flex items-center gap-2.5 min-w-0">
          <button
            type="button"
            id="btn-mobile-nav-toggle"
            onClick={() => setMobileNavOpen(true)}
            aria-label={t.nav.toggleNavigation}
            className="w-10 h-10 min-h-[40px] min-w-[40px] flex items-center justify-center rounded-[5px] text-[var(--ink-secondary)] hover:text-[var(--ink-primary)] hover:bg-[var(--paper-subsurface)] border border-transparent active:border-[var(--paper-border)] transition-colors cursor-pointer shrink-0"
          >
            <Menu className="h-5 w-5" />
          </button>

          <div className="flex items-center gap-2 truncate">
            <span className="h-6 w-6 rounded-[4px] bg-[var(--ink-primary)] text-[var(--paper-bg)] flex items-center justify-center font-bold text-xs tracking-tighter shadow-sm font-mono shrink-0">
              U
            </span>
            <div className="truncate">
              <span className="font-bold text-xs tracking-tight text-[var(--ink-primary)] block leading-none truncate">
                {t.common.compilerConsole}
              </span>
              <span className="text-xs text-[var(--ink-muted)] font-mono leading-tight block mt-0.5 uppercase tracking-wider">
                ENGINE 0.4
              </span>
            </div>
          </div>
        </div>

        {/* Mobile Language & Paper Tone Controls */}
        <div className="flex items-center gap-1.5 shrink-0">
          <button
            id="btn-paper-toggle-mobile"
            type="button"
            onClick={() => setPaperTone(paperTone === 'cotton' ? 'dowling' : 'cotton')}
            className="flex items-center gap-1.5 min-h-[40px] h-10 px-2.5 rounded text-xs font-mono text-[var(--ink-secondary)] hover:text-[var(--ink-primary)] bg-[var(--paper-subsurface)] border border-[var(--paper-border)] hover:border-[var(--paper-border-hover)] transition-colors cursor-pointer"
            aria-label={t.common.paperToneSwitchTo.replace(
              '{tone}',
              paperTone === 'cotton' ? t.common.paperToneDowling : t.common.paperToneCotton,
            )}
            title={t.common.paperToneSwitchTo.replace(
              '{tone}',
              paperTone === 'cotton' ? t.common.paperToneDowling : t.common.paperToneCotton,
            )}
          >
            <FileSpreadsheet className="h-3.5 w-3.5 text-[var(--ink-muted)]" />
            <span>{paperTone === 'cotton' ? t.common.paperToneCotton : t.common.paperToneDowling}</span>
          </button>

          <button
            id="btn-lang-toggle-mobile"
            type="button"
            onClick={() => setLanguage(language === 'zh' ? 'en' : 'zh')}
            className="flex items-center gap-1.5 min-h-[40px] h-10 px-2.5 rounded text-xs font-mono text-[var(--ink-secondary)] hover:text-[var(--ink-primary)] bg-[var(--paper-subsurface)] border border-[var(--paper-border)] hover:border-[var(--paper-border-hover)] transition-colors font-medium cursor-pointer"
            aria-label={t.common.switchLanguage}
            title={t.common.switchLanguage}
          >
            <Globe className="h-3.5 w-3.5 text-[var(--ink-muted)]" />
            <span>{language === 'zh' ? '中' : 'EN'}</span>
          </button>
        </div>
      </header>

      {/* Mobile Drawer Backdrop */}
      {mobileNavOpen && (
        <div
          className="fixed inset-0 bg-black/40 z-40 md:hidden backdrop-blur-xs transition-opacity duration-200"
          onClick={() => setMobileNavOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Mobile Drawer Sheet */}
      <div
        className={`fixed inset-y-0 left-0 w-72 max-w-[85vw] bg-[var(--paper-surface)] z-50 md:hidden flex flex-col justify-between shadow-2xl border-r border-[var(--paper-border)] transition-all duration-200 ease-in-out select-none ${
          mobileNavOpen ? 'translate-x-0 visible pointer-events-auto' : '-translate-x-full invisible pointer-events-none'
        }`}
        role="dialog"
        aria-modal="true"
        aria-hidden={!mobileNavOpen}
        aria-label={t.nav.operatorControl}
      >
        <div className="flex flex-col h-full overflow-hidden">
          {/* Top Brand in Drawer */}
          <div className="min-h-14 h-[calc(3.5rem+env(safe-area-inset-top,0px))] px-4 border-b border-[var(--paper-border)] flex items-center justify-between shrink-0 pt-[env(safe-area-inset-top,0px)]">
            <div className="flex items-center gap-2.5">
              <span className="h-6 w-6 rounded-[4px] bg-[var(--ink-primary)] text-[var(--paper-bg)] flex items-center justify-center font-bold text-xs tracking-tighter shadow-sm font-mono">
                U
              </span>
              <div>
                <span className="font-bold text-xs tracking-tight text-[var(--ink-primary)] block leading-none">
                  {t.common.compilerConsole}
                </span>
                <span className="text-xs text-[var(--ink-muted)] font-mono leading-tight block mt-0.5 uppercase tracking-wider">
                  ENGINE 0.4
                </span>
              </div>
            </div>
            <button
              type="button"
              onClick={() => setMobileNavOpen(false)}
              aria-label={t.nav.closeNavigation}
              className="w-10 h-10 min-h-[40px] min-w-[40px] flex items-center justify-center rounded-[5px] text-[var(--ink-muted)] hover:text-[var(--ink-primary)] hover:bg-[var(--paper-subsurface)] transition-colors cursor-pointer"
            >
              <X className="h-5 w-5" />
            </button>
          </div>

          {/* Navigation Sections */}
          <div className="p-3 space-y-5 flex-1 overflow-y-auto">
            <div>
              <div className="px-2.5 py-1 text-xs font-mono uppercase tracking-widest text-[var(--ink-muted)] font-semibold">
                {t.nav.operatorControl}
              </div>
              <nav className="mt-1 space-y-1">{renderNav(navItems, () => setMobileNavOpen(false))}</nav>
            </div>

            <div>
              <div className="px-2.5 py-1 text-xs font-mono uppercase tracking-widest text-[var(--ink-muted)] font-semibold">
                {t.nav.workbenches}
              </div>
              <nav className="mt-1 space-y-1">
                {renderNav(workbenchItems, () => setMobileNavOpen(false))}
              </nav>
            </div>
          </div>

          {/* Engine Status Line (Mobile Drawer) */}
          <div className="min-h-14 h-[calc(3.5rem+env(safe-area-inset-bottom,0px))] px-3.5 border-t border-[var(--paper-border)] bg-[var(--paper-subsurface)] flex items-center justify-between text-xs font-mono shrink-0 pb-[env(safe-area-inset-bottom,0px)]">
            <div className="flex items-center gap-2">
              <span
                className={`h-2 w-2 rounded-full ${health ? 'bg-[#15803d]' : 'bg-[#b91c1c]'}`}
              />
              <span className="text-xs text-[var(--ink-secondary)] font-medium">
                {health ? `Core ${health.version}` : t.common.disconnected}
              </span>
            </div>
            <div className="flex items-center gap-2 text-xs text-[var(--ink-muted)]">
              <span className="flex items-center gap-1">
                <Terminal className="h-3 w-3" />
                <span>{info?.port ? `:${info.port}` : ''}</span>
              </span>
              {canSignOut && (
                <button
                  type="button"
                  onClick={() => void handleSignOut()}
                  title={t.doctor.auth.signOut}
                  aria-label={t.doctor.auth.signOut}
                  className="p-2 min-h-[40px] min-w-[40px] flex items-center justify-center hover:text-[var(--ink-primary)] transition-colors cursor-pointer"
                >
                  <LogOut className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Precision Left Sidebar on Paper Surface (Desktop: md+) */}
      <aside className="hidden md:flex md:w-64 border-r border-[var(--paper-border)] bg-[var(--paper-surface)] flex-col justify-between shrink-0 select-none shadow-[1px_0_4px_rgba(0,0,0,0.02)]">
        <div>
          {/* Top Brand & Language Bar */}
          <div className="h-14 px-4 border-b border-[var(--paper-border)] flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <span className="h-6 w-6 rounded-[4px] bg-[var(--ink-primary)] text-[var(--paper-bg)] flex items-center justify-center font-bold text-xs tracking-tighter shadow-sm font-mono">
                U
              </span>
              <div>
                <span className="font-bold text-xs tracking-tight text-[var(--ink-primary)] block leading-none">
                  {t.common.compilerConsole}
                </span>
                <span className="text-xs text-[var(--ink-muted)] font-mono leading-tight block mt-0.5 uppercase tracking-wider">
                  ENGINE 0.4
                </span>
              </div>
            </div>

            {/* Language & Paper Tone Pills */}
            <div className="flex items-center gap-1.5">
              <button
                id="btn-paper-toggle"
                onClick={() => setPaperTone(paperTone === 'cotton' ? 'dowling' : 'cotton')}
                className="flex items-center gap-1 px-1.5 py-0.5 rounded text-xs font-mono text-[var(--ink-secondary)] hover:text-[var(--ink-primary)] bg-[var(--paper-subsurface)] border border-[var(--paper-border)] hover:border-[var(--paper-border-hover)] transition-colors"
                title={t.common.paperToneSwitchTo.replace(
                  '{tone}',
                  paperTone === 'cotton' ? t.common.paperToneDowling : t.common.paperToneCotton,
                )}
              >
                <FileSpreadsheet className="h-3 w-3 text-[var(--ink-muted)]" />
                <span>
                  {paperTone === 'cotton' ? t.common.paperToneCotton : t.common.paperToneDowling}
                </span>
              </button>

              <button
                id="btn-lang-toggle"
                onClick={() => setLanguage(language === 'zh' ? 'en' : 'zh')}
                className="flex items-center gap-1 px-1.5 py-0.5 rounded text-xs font-mono text-[var(--ink-secondary)] hover:text-[var(--ink-primary)] bg-[var(--paper-subsurface)] border border-[var(--paper-border)] hover:border-[var(--paper-border-hover)] transition-colors font-medium"
                title={t.common.switchLanguage}
              >
                <Globe className="h-3 w-3 text-[var(--ink-muted)]" />
                <span>{language === 'zh' ? '中' : 'EN'}</span>
              </button>
            </div>
          </div>

          {/* Navigation Sections */}
          <div className="p-2.5 space-y-5">
            <div>
              <div className="px-2.5 py-1 text-xs font-mono uppercase tracking-widest text-[var(--ink-muted)] font-semibold">
                {t.nav.operatorControl}
              </div>
              <nav className="mt-1 space-y-0.5">{renderNav(navItems)}</nav>
            </div>

            <div>
              <div className="px-2.5 py-1 text-xs font-mono uppercase tracking-widest text-[var(--ink-muted)] font-semibold">
                {t.nav.workbenches}
              </div>
              <nav className="mt-1 space-y-0.5">{renderNav(workbenchItems)}</nav>
            </div>
          </div>
        </div>

        {/* Engine Status Line */}
        <div className="h-10 px-3.5 border-t border-[var(--paper-border)] bg-[var(--paper-subsurface)] flex items-center justify-between text-xs font-mono">
          <div className="flex items-center gap-2">
            <span
              className={`h-2 w-2 rounded-full ${health ? 'bg-[#15803d]' : 'bg-[#b91c1c]'}`}
            />
            <span className="text-xs text-[var(--ink-secondary)] font-medium">
              {health ? `Core ${health.version}` : t.common.disconnected}
            </span>
          </div>
          <div className="flex items-center gap-2 text-xs text-[var(--ink-muted)]">
            <span className="flex items-center gap-1">
              <Terminal className="h-3 w-3" />
              <span>{info?.port ? `:${info.port}` : ''}</span>
            </span>
            {canSignOut && (
              <button
                type="button"
                onClick={() => void handleSignOut()}
                title={t.doctor.auth.signOut}
                aria-label={t.doctor.auth.signOut}
                className="flex items-center hover:text-[var(--ink-primary)] transition-colors cursor-pointer"
              >
                <LogOut className="h-3 w-3" />
              </button>
            )}
          </div>
        </div>
      </aside>

      {/* Main Work Area */}
      <main className="flex-1 flex flex-col min-w-0 min-h-0 overflow-hidden bg-[var(--paper-bg)] pb-[env(safe-area-inset-bottom,0px)]">
        {children}
      </main>
    </div>
  )
}
