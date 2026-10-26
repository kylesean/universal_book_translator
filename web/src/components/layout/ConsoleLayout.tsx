import React, { useEffect, useState, useCallback } from 'react'
import { NavLink, useLocation, useMatch } from 'react-router-dom'
import {
  PlusCircle,
  Activity,
  ShieldCheck,
  BookOpen,
  Stethoscope,
  SplitSquareVertical,
  LogOut,
  Moon,
  Search,
  Sun,
  FileText,
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
import { LANGUAGE_OPTIONS, languageOption } from '@/i18n/languages'
import type { Language } from '@/i18n/types'
import { applyTheme, detectTheme, rememberTheme, type Theme } from '@/lib/theme'
import { CommandPalette } from './CommandPalette'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { ButtonGroup } from '@/components/ui/button-group'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarRail,
  SidebarSeparator,
  SidebarTrigger,
} from '@/components/ui/sidebar'
import { useSidebar } from '@/components/ui/sidebar-context'

interface ConsoleLayoutProps {
  children: React.ReactNode
}

interface NavItem {
  to: string
  label: string
  icon: React.ComponentType<{ className?: string }>
  badge?: string
  disabled?: boolean
}

function NavRows({ items }: { items: NavItem[] }) {
  const { t } = useI18n()
  const { pathname } = useLocation()

  return (
    <SidebarMenu>
      {items.map((item) => {
        const Icon = item.icon
        if (item.disabled) {
          return (
            <SidebarMenuItem key={item.label}>
              <SidebarMenuButton
                disabled
                tooltip={`${item.label} · ${t.mission.noActiveJobTitle}`}
                className="cursor-not-allowed"
              >
                <Icon />
                <span>{item.label}</span>
              </SidebarMenuButton>
            </SidebarMenuItem>
          )
        }

        return (
          <SidebarMenuItem key={item.label}>
            <SidebarMenuButton
              asChild
              tooltip={item.label}
              isActive={pathname === item.to}
            >
              <NavLink
                to={item.to}
                end
                className={({ isActive }) =>
                  cn(isActive && 'font-medium')
                }
              >
                <Icon />
                <span>{item.label}</span>
                {item.badge && (
                  <span className="ml-auto rounded border border-sidebar-border bg-background/60 px-1.5 py-0.5 font-mono text-xs text-muted-foreground">
                    {item.badge}
                  </span>
                )}
              </NavLink>
            </SidebarMenuButton>
          </SidebarMenuItem>
        )
      })}
    </SidebarMenu>
  )
}

function AppSidebar({
  theme,
  onSetTheme,
  onSetLanguage,
  onOpenPalette,
  health,
  canSignOut,
  onSignOut,
}: {
  theme: Theme
  onSetTheme: (theme: Theme) => void
  onSetLanguage: (language: Language) => void
  onOpenPalette: () => void
  health: HealthResponse | null
  canSignOut: boolean
  onSignOut: () => void
}) {
  const { t, language } = useI18n()
  const currentLanguage = languageOption(language)
  const { isMobile, setOpenMobile } = useSidebar()
  const [languageMenuOpen, setLanguageMenuOpen] = useState(false)
  const jobMatch = useMatch('/jobs/:jobId/*')
  const currentJobId = jobMatch?.params.jobId ?? null

  const closeMobile = () => {
    if (isMobile) setOpenMobile(false)
  }

  const navItems: NavItem[] = [
    { to: '/wizard', label: t.nav.newJob, icon: PlusCircle },
    { to: '/jobs', label: t.nav.missionControl, icon: Activity },
    { to: '/assets', label: t.nav.languageAssets, icon: BookOpen },
    { to: '/system', label: t.nav.systemDoctor, icon: Stethoscope },
  ]

  const workbenchItems: NavItem[] = [
    {
      to: currentJobId ? `/jobs/${currentJobId}` : '/jobs',
      label: t.nav.currentJob,
      icon: FileText,
      disabled: !currentJobId,
    },
    {
      to: currentJobId ? `/jobs/${currentJobId}/review` : '/jobs',
      label: t.nav.reviewWorkbench,
      icon: SplitSquareVertical,
      disabled: !currentJobId,
    },
    {
      to: currentJobId ? `/jobs/${currentJobId}/quality` : '/jobs',
      label: t.nav.qualityGate,
      icon: ShieldCheck,
      disabled: !currentJobId,
    },
  ]

  return (
    <Sidebar variant="inset" collapsible="offcanvas">
      <SidebarRail />
      <SidebarHeader className="gap-3 p-3">
        <SidebarMenu>
          <SidebarMenuItem>
            <div className="flex items-center gap-2.5 px-1">
              <span className="flex size-8 shrink-0 items-center justify-center rounded-lg border border-primary/20 bg-primary font-mono text-sm font-bold text-primary-foreground shadow-xs">
                U
              </span>
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-semibold leading-tight tracking-[-0.015em]">
                  {t.common.compilerConsole}
                </div>
                <div className="mt-0.5 truncate font-mono text-xs uppercase tracking-[0.12em] text-muted-foreground">
                  {t.common.consoleSubtitle}
                </div>
              </div>
            </div>
          </SidebarMenuItem>
        </SidebarMenu>

        <button
          type="button"
          onClick={onOpenPalette}
          className="group flex h-9 w-full items-center gap-2 rounded-lg border border-sidebar-border bg-background/55 px-2.5 text-left text-sm text-muted-foreground shadow-2xs transition-colors hover:border-border-hover hover:bg-background hover:text-foreground"
        >
          <Search className="size-4 shrink-0" />
          <span className="min-w-0 flex-1 truncate text-xs">{t.nav.searchCommands}</span>
          <kbd className="rounded border border-sidebar-border bg-muted/60 px-1.5 py-0.5 font-mono text-xs text-muted-foreground">
            ⌘K
          </kbd>
        </button>
      </SidebarHeader>

      <SidebarSeparator />

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>
            {t.nav.operatorControl}
          </SidebarGroupLabel>
          <SidebarGroupContent onClick={closeMobile}>
            <NavRows items={navItems} />
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel>
            {t.nav.workbenches}
          </SidebarGroupLabel>
          <SidebarGroupContent onClick={closeMobile}>
            <NavRows items={workbenchItems} />
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="gap-2 border-t border-sidebar-border/70 p-3">
        <div className="flex items-center justify-between gap-2 rounded-lg border border-sidebar-border/70 bg-background/50 px-2.5 py-2">
          <div className="flex min-w-0 items-center gap-2">
            <span
              className={cn(
                'size-2 shrink-0 rounded-full',
                health ? 'bg-emerald-500 shadow-[0_0_0_3px_rgba(16,185,129,0.12)]' : 'bg-destructive'
              )}
            />
            <span className="truncate text-xs text-muted-foreground">
              {health ? t.common.serviceConnected : t.common.serviceUnavailable}
            </span>
          </div>
        </div>

        <div className="space-y-1.5">
          <ButtonGroup className="w-full">
            <Button
              type="button"
              size="sm"
              variant={theme === 'light' ? 'secondary' : 'ghost'}
              className="h-8 flex-1 px-2 text-xs"
              aria-pressed={theme === 'light'}
              onClick={() => onSetTheme('light')}
            >
              <Sun />
              {t.common.themeLight}
            </Button>
            <Button
              type="button"
              size="sm"
              variant={theme === 'dark' ? 'secondary' : 'ghost'}
              className="h-8 flex-1 px-2 text-xs"
              aria-pressed={theme === 'dark'}
              onClick={() => onSetTheme('dark')}
            >
              <Moon />
              {t.common.themeDark}
            </Button>
          </ButtonGroup>

          <DropdownMenu open={languageMenuOpen} onOpenChange={setLanguageMenuOpen}>
            <DropdownMenuTrigger asChild>
              <Button
                variant="outline"
                className="h-8 w-full justify-center px-2 text-xs font-normal"
                aria-label={`${t.common.interfaceLanguage}: ${currentLanguage.label}`}
              >
                {currentLanguage.label}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start" side="top" className="w-56">
              <DropdownMenuRadioGroup
                value={language}
                onValueChange={(value) => {
                  onSetLanguage(value as Language)
                  setLanguageMenuOpen(false)
                }}
              >
                {LANGUAGE_OPTIONS.map((option) => (
                  <DropdownMenuRadioItem key={option.code} value={option.code}>
                    {option.label}
                  </DropdownMenuRadioItem>
                ))}
              </DropdownMenuRadioGroup>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>

        {canSignOut ? (
          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-full justify-start text-xs text-muted-foreground"
            onClick={onSignOut}
          >
            <LogOut />
            {t.doctor.auth.signOut}
          </Button>
        ) : null}
      </SidebarFooter>
    </Sidebar>
  )
}

export function ConsoleLayout({ children }: ConsoleLayoutProps) {
  const { t, setLanguage } = useI18n()

  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [info, setInfo] = useState<SystemInfo | null>(null)
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false)

  const [theme, setTheme] = useState<Theme>(detectTheme)

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const active = document.activeElement
      const isInput =
        active instanceof HTMLInputElement ||
        active instanceof HTMLTextAreaElement ||
        (active as HTMLElement | null)?.isContentEditable

      if (!isInput && e.key === '/') {
        e.preventDefault()
        setCommandPaletteOpen(true)
        return
      }

      if ((e.ctrlKey || e.altKey || e.metaKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setCommandPaletteOpen((prev) => !prev)
        return
      }

      if (e.key === 'Escape') {
        setCommandPaletteOpen(false)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  useEffect(() => {
    applyTheme(theme)
    rememberTheme(theme)
  }, [theme])

  useEffect(() => {
    let isMounted = true
    const probe = async () => {
      try {
        const res = await checkHealth()
        if (isMounted) setHealth(res)
      } catch {
        if (isMounted) setHealth(null)
      }
    }
    probe()

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

  const handleSignOut = useCallback(async () => {
    await signOut()
    window.location.reload()
  }, [])

  const canSignOut = Boolean(info?.auth_enabled) && Boolean(getStoredApiKey())

  const jobMatch = useMatch('/jobs/:jobId/*')
  const currentJobId = jobMatch?.params.jobId ?? null

  return (
    <SidebarProvider defaultOpen>
      <CommandPalette
        isOpen={commandPaletteOpen}
        onClose={() => setCommandPaletteOpen(false)}
        currentJobId={currentJobId}
        theme={theme}
        onSetTheme={setTheme}
        onSetLanguage={setLanguage}
      />

      <AppSidebar
        theme={theme}
        onSetTheme={setTheme}
        onSetLanguage={setLanguage}
        onOpenPalette={() => setCommandPaletteOpen(true)}
        health={health}
        canSignOut={canSignOut}
        onSignOut={() => void handleSignOut()}
      />

      <SidebarInset className="min-w-0 overflow-hidden bg-background md:my-2 md:mr-2 md:rounded-xl md:border md:border-border/70 md:shadow-sm">
        <header className="sticky top-0 z-20 flex h-12 shrink-0 items-center gap-2 border-b border-border/70 bg-background/85 px-3 backdrop-blur-md md:hidden">
          <SidebarTrigger className="-ml-1 size-9" aria-label={t.nav.toggleNavigation} />
          <Separator orientation="vertical" className="mx-0.5 h-5" />
          <div className="flex min-w-0 flex-1 items-center gap-2">
            <span className="flex size-6 items-center justify-center rounded-md bg-primary font-mono text-xs font-bold text-primary-foreground">
              U
            </span>
            <span className="truncate text-sm font-semibold">{t.common.compilerConsole}</span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setCommandPaletteOpen(true)}
            aria-label={t.palette.title}
            className="size-9"
          >
            <Search className="size-4" />
          </Button>
          <Badge
            variant="outline"
            className={cn(
              'hidden sm:inline-flex',
              health
                ? 'border-emerald-600/25 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
                : 'border-destructive/25 bg-destructive/10 text-destructive'
            )}
          >
            <span
              className={cn(
                'mr-1.5 size-1.5 rounded-full',
                health ? 'bg-emerald-500' : 'bg-destructive'
              )}
            />
            {health ? t.common.serviceConnected : t.common.serviceUnavailable}
          </Badge>
        </header>

        <main className="flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto bg-background pb-[env(safe-area-inset-bottom,0px)]">
          {children}
        </main>
      </SidebarInset>
    </SidebarProvider>
  )
}
