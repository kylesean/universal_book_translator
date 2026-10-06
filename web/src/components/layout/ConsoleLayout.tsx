import React, { useEffect, useState } from 'react'
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
} from 'lucide-react'
import { checkHealth, type HealthResponse } from '@/api/client'
import { useI18n } from '@/i18n/I18nContext'

export type ViewTab = 'wizard' | 'jobs' | 'quality' | 'assets' | 'system' | 'review'

interface ConsoleLayoutProps {
  currentTab: ViewTab
  onSelectTab: (tab: ViewTab) => void
  children: React.ReactNode
}

export function ConsoleLayout({ currentTab, onSelectTab, children }: ConsoleLayoutProps) {
  const { t, language, setLanguage } = useI18n()
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [paperTone, setPaperTone] = useState<'cotton' | 'dowling'>(() => {
    return (localStorage.getItem('ubt_paper_tone') as 'cotton' | 'dowling') || 'cotton'
  })

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
    const timer = setInterval(probe, 10000)
    return () => {
      isMounted = false
      clearInterval(timer)
    }
  }, [])

  const navItems = [
    { id: 'wizard' as ViewTab, label: t.nav.newJob, sub: t.nav.newJobSub, icon: PlusCircle },
    { id: 'jobs' as ViewTab, label: t.nav.missionControl, sub: t.nav.missionControlSub, icon: Activity },
    { id: 'quality' as ViewTab, label: t.nav.qualityGate, sub: t.nav.qualityGateSub, icon: ShieldCheck },
    { id: 'assets' as ViewTab, label: t.nav.languageAssets, sub: t.nav.languageAssetsSub, icon: BookOpen },
    { id: 'system' as ViewTab, label: t.nav.systemDoctor, sub: t.nav.systemDoctorSub, icon: Stethoscope },
  ]

  const workbenchItems = [
    { id: 'review' as ViewTab, label: t.nav.reviewWorkbench, sub: t.nav.reviewWorkbenchSub, icon: SplitSquareVertical },
  ]

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[var(--paper-bg)] text-[var(--ink-primary)]">
      {/* Precision Left Sidebar on Paper Surface */}
      <aside className="w-64 border-r border-[var(--paper-border)] bg-[var(--paper-surface)] flex flex-col justify-between shrink-0 select-none shadow-[1px_0_4px_rgba(0,0,0,0.02)]">
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
                <span className="text-[10px] text-[var(--ink-muted)] font-mono leading-tight block mt-0.5 uppercase tracking-wider">
                  ENGINE 0.4
                </span>
              </div>
            </div>

            {/* Language & Paper Tone Pills */}
            <div className="flex items-center gap-1.5">
              <button
                id="btn-paper-toggle"
                onClick={() => setPaperTone(paperTone === 'cotton' ? 'dowling' : 'cotton')}
                className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono text-[var(--ink-secondary)] hover:text-[var(--ink-primary)] bg-[var(--paper-subsurface)] border border-[var(--paper-border)] hover:border-[var(--paper-border-hover)] transition-colors"
                title={paperTone === 'cotton' ? '切换至道林暖纸 / Switch to Dowling' : '切换至纯质棉白 / Switch to Cotton'}
              >
                <FileSpreadsheet className="h-3 w-3 text-[var(--ink-muted)]" />
                <span>{paperTone === 'cotton' ? '棉白' : '道林'}</span>
              </button>

              <button
                id="btn-lang-toggle"
                onClick={() => setLanguage(language === 'zh' ? 'en' : 'zh')}
                className="flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-mono text-[var(--ink-secondary)] hover:text-[var(--ink-primary)] bg-[var(--paper-subsurface)] border border-[var(--paper-border)] hover:border-[var(--paper-border-hover)] transition-colors font-medium"
                title="Switch language / 切换语言"
              >
                <Globe className="h-3 w-3 text-[var(--ink-muted)]" />
                <span>{language === 'zh' ? '中' : 'EN'}</span>
              </button>
            </div>
          </div>

          {/* Navigation Sections */}
          <div className="p-2.5 space-y-5">
            <div>
              <div className="px-2.5 py-1 text-[10px] font-mono uppercase tracking-widest text-[var(--ink-muted)] font-semibold">
                {t.nav.operatorControl}
              </div>
              <nav className="mt-1 space-y-0.5">
                {navItems.map((item) => {
                  const Icon = item.icon
                  const isActive = currentTab === item.id
                  return (
                    <button
                      key={item.id}
                      onClick={() => onSelectTab(item.id)}
                      className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-[5px] text-xs font-medium transition-all text-left ${
                        isActive
                          ? 'bg-[var(--paper-subsurface)] text-[var(--ink-primary)] font-semibold border border-[var(--paper-border)] shadow-xs'
                          : 'text-[var(--ink-secondary)] hover:bg-[var(--paper-subsurface)] hover:text-[var(--ink-primary)] border border-transparent'
                      }`}
                    >
                      <Icon className={`h-3.5 w-3.5 shrink-0 ${isActive ? 'text-[var(--ink-primary)]' : 'text-[var(--ink-muted)]'}`} />
                      <div className="flex-1 truncate">
                        <div className="leading-tight">{item.label}</div>
                      </div>
                    </button>
                  )
                })}
              </nav>
            </div>

            <div>
              <div className="px-2.5 py-1 text-[10px] font-mono uppercase tracking-widest text-[var(--ink-muted)] font-semibold">
                {t.nav.workbenches}
              </div>
              <nav className="mt-1 space-y-0.5">
                {workbenchItems.map((item) => {
                  const Icon = item.icon
                  const isActive = currentTab === item.id
                  return (
                    <button
                      key={item.id}
                      onClick={() => onSelectTab(item.id)}
                      className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-[5px] text-xs font-medium transition-all text-left ${
                        isActive
                          ? 'bg-[var(--paper-subsurface)] text-[var(--ink-primary)] font-semibold border border-[var(--paper-border)] shadow-xs'
                          : 'text-[var(--ink-secondary)] hover:bg-[var(--paper-subsurface)] hover:text-[var(--ink-primary)] border border-transparent'
                      }`}
                    >
                      <Icon className={`h-3.5 w-3.5 shrink-0 ${isActive ? 'text-[#15803d]' : 'text-[var(--ink-muted)]'}`} />
                      <div className="flex-1 truncate">
                        <div className="leading-tight">{item.label}</div>
                      </div>
                    </button>
                  )
                })}
              </nav>
            </div>
          </div>
        </div>

        {/* Engine Status Line */}
        <div className="h-10 px-3.5 border-t border-[var(--paper-border)] bg-[var(--paper-subsurface)] flex items-center justify-between text-xs font-mono">
          <div className="flex items-center gap-2">
            <span
              className={`h-2 w-2 rounded-full ${health ? 'bg-[#15803d]' : 'bg-[#b91c1c]'}`}
            />
            <span className="text-[11px] text-[var(--ink-secondary)] font-medium">
              {health ? `Core ${health.version}` : t.common.disconnected}
            </span>
          </div>
          <div className="flex items-center gap-1 text-[11px] text-[var(--ink-muted)]">
            <Terminal className="h-3 w-3" />
            <span>:8000</span>
          </div>
        </div>
      </aside>

      {/* Main Work Area */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden bg-[var(--paper-bg)]">
        {children}
      </main>
    </div>
  )
}
