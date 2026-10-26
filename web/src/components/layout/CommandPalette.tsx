import { useNavigate } from 'react-router-dom'
import { useI18n } from '@/i18n/useI18n'
import {
  PlusCircle,
  Activity,
  SplitSquareVertical,
  ShieldCheck,
  BookOpen,
  Stethoscope,
  Moon,
  Sun,
  Check,
  ArrowRight,
} from 'lucide-react'
import { LANGUAGE_OPTIONS } from '@/i18n/languages'
import type { Language } from '@/i18n/types'
import type { Theme } from '@/lib/theme'
import {
  CommandDialog,
  CommandInput,
  CommandList,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandShortcut,
  CommandSeparator,
} from '@/components/ui/command'

interface CommandPaletteProps {
  isOpen: boolean
  onClose: () => void
  currentJobId?: string | null
  theme: Theme
  onSetTheme: (theme: Theme) => void
  onSetLanguage: (language: Language) => void
}

export function CommandPalette({
  isOpen,
  onClose,
  currentJobId,
  theme,
  onSetTheme,
  onSetLanguage,
}: CommandPaletteProps) {
  const { t, language } = useI18n()
  const navigate = useNavigate()

  const handleSelect = (action: () => void) => {
    action()
    onClose()
  }

  const shortJobId = currentJobId ? currentJobId.slice(0, 8) : ''

  return (
    <CommandDialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <CommandInput placeholder={t.palette.placeholder} />
      <CommandList className="max-h-[360px] p-1">
        <CommandEmpty className="p-8 text-center text-xs text-muted-foreground font-mono">
          {t.palette.noMatch}
        </CommandEmpty>

        {/* Core Controls */}
        <CommandGroup heading={t.nav.operatorControl}>
          <CommandItem
            value={`${t.nav.newJob} ${t.palette.cmdNewJobDesc}`}
            onSelect={() => handleSelect(() => navigate('/wizard'))}
            className="flex items-center justify-between py-2 px-2.5 rounded-lg cursor-pointer"
          >
            <div className="flex items-center gap-2.5 min-w-0">
              <PlusCircle className="size-4 text-muted-foreground shrink-0" />
              <div className="truncate">
                <span className="font-medium text-foreground text-xs">{t.nav.newJob}</span>
                <span className="text-xs text-muted-foreground ml-2 truncate">
                  {t.palette.cmdNewJobDesc}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              <CommandShortcut>G W</CommandShortcut>
              <ArrowRight className="size-3 text-muted-foreground opacity-40 ml-1" />
            </div>
          </CommandItem>

          <CommandItem
            value={`${t.nav.missionControl} ${t.palette.cmdMissionControlDesc}`}
            onSelect={() => handleSelect(() => navigate('/jobs'))}
            className="flex items-center justify-between py-2 px-2.5 rounded-lg cursor-pointer"
          >
            <div className="flex items-center gap-2.5 min-w-0">
              <Activity className="size-4 text-muted-foreground shrink-0" />
              <div className="truncate">
                <span className="font-medium text-foreground text-xs">{t.nav.missionControl}</span>
                <span className="text-xs text-muted-foreground ml-2 truncate">
                  {t.palette.cmdMissionControlDesc}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              <CommandShortcut>G J</CommandShortcut>
              <ArrowRight className="size-3 text-muted-foreground opacity-40 ml-1" />
            </div>
          </CommandItem>

          <CommandItem
            value={`${t.nav.languageAssets} ${t.palette.cmdAssetsDesc}`}
            onSelect={() => handleSelect(() => navigate('/assets'))}
            className="flex items-center justify-between py-2 px-2.5 rounded-lg cursor-pointer"
          >
            <div className="flex items-center gap-2.5 min-w-0">
              <BookOpen className="size-4 text-muted-foreground shrink-0" />
              <div className="truncate">
                <span className="font-medium text-foreground text-xs">{t.nav.languageAssets}</span>
                <span className="text-xs text-muted-foreground ml-2 truncate">
                  {t.palette.cmdAssetsDesc}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              <CommandShortcut>G A</CommandShortcut>
              <ArrowRight className="size-3 text-muted-foreground opacity-40 ml-1" />
            </div>
          </CommandItem>

          <CommandItem
            value={`${t.nav.systemDoctor} ${t.palette.cmdDoctorDesc}`}
            onSelect={() => handleSelect(() => navigate('/system'))}
            className="flex items-center justify-between py-2 px-2.5 rounded-lg cursor-pointer"
          >
            <div className="flex items-center gap-2.5 min-w-0">
              <Stethoscope className="size-4 text-muted-foreground shrink-0" />
              <div className="truncate">
                <span className="font-medium text-foreground text-xs">{t.nav.systemDoctor}</span>
                <span className="text-xs text-muted-foreground ml-2 truncate">
                  {t.palette.cmdDoctorDesc}
                </span>
              </div>
            </div>
            <div className="flex items-center gap-1.5 shrink-0">
              <CommandShortcut>G D</CommandShortcut>
              <ArrowRight className="size-3 text-muted-foreground opacity-40 ml-1" />
            </div>
          </CommandItem>
        </CommandGroup>

        {/* Workbenches & Current Job Navigation */}
        <CommandSeparator />
        <CommandGroup heading={t.nav.workbenches}>
          {currentJobId && (
            <>
              <CommandItem
                value={`${t.nav.reviewWorkbench} ${t.palette.cmdReviewDesc.replace('{id}', shortJobId)}`}
                onSelect={() => handleSelect(() => navigate(`/jobs/${currentJobId}/review`))}
                className="flex items-center justify-between py-2 px-2.5 rounded-lg cursor-pointer"
              >
                <div className="flex items-center gap-2.5 min-w-0">
                  <SplitSquareVertical className="size-4 text-muted-foreground shrink-0" />
                  <div className="truncate">
                    <span className="font-medium text-foreground text-xs">{t.nav.reviewWorkbench}</span>
                    <span className="text-xs text-muted-foreground ml-2 truncate">
                      {t.palette.cmdReviewDesc.replace('{id}', shortJobId)}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  <CommandShortcut>G R</CommandShortcut>
                  <ArrowRight className="size-3 text-muted-foreground opacity-40 ml-1" />
                </div>
              </CommandItem>

              <CommandItem
                value={`${t.nav.qualityGate} ${t.palette.cmdQualityDesc.replace('{id}', shortJobId)}`}
                onSelect={() => handleSelect(() => navigate(`/jobs/${currentJobId}/quality`))}
                className="flex items-center justify-between py-2 px-2.5 rounded-lg cursor-pointer"
              >
                <div className="flex items-center gap-2.5 min-w-0">
                  <ShieldCheck className="size-4 text-muted-foreground shrink-0" />
                  <div className="truncate">
                    <span className="font-medium text-foreground text-xs">{t.nav.qualityGate}</span>
                    <span className="text-xs text-muted-foreground ml-2 truncate">
                      {t.palette.cmdQualityDesc.replace('{id}', shortJobId)}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  <CommandShortcut>G Q</CommandShortcut>
                  <ArrowRight className="size-3 text-muted-foreground opacity-40 ml-1" />
                </div>
              </CommandItem>
            </>
          )}
        </CommandGroup>

        {/* Preferences */}
        <CommandSeparator />
        <CommandGroup heading={t.palette.categoryPreferences}>
          <CommandItem
            value={`${t.common.themeLight} ${t.palette.cmdLightThemeDesc}`}
            onSelect={() => handleSelect(() => onSetTheme('light'))}
            className="flex items-center justify-between py-2 px-2.5 rounded-lg cursor-pointer"
          >
            <div className="flex items-center gap-2.5 min-w-0">
              <Sun className="size-4 text-muted-foreground shrink-0" />
              <div className="truncate">
                <span className="font-medium text-foreground text-xs">{t.common.themeLight}</span>
                <span className="text-xs text-muted-foreground ml-2 truncate">
                  {t.palette.cmdLightThemeDesc}
                </span>
              </div>
            </div>
            {theme === 'light' ? <Check className="size-4 shrink-0" /> : null}
          </CommandItem>
          <CommandItem
            value={`${t.common.themeDark} ${t.palette.cmdDarkThemeDesc}`}
            onSelect={() => handleSelect(() => onSetTheme('dark'))}
            className="flex items-center justify-between py-2 px-2.5 rounded-lg cursor-pointer"
          >
            <div className="flex items-center gap-2.5 min-w-0">
              <Moon className="size-4 text-muted-foreground shrink-0" />
              <div className="truncate">
                <span className="font-medium text-foreground text-xs">{t.common.themeDark}</span>
                <span className="text-xs text-muted-foreground ml-2 truncate">
                  {t.palette.cmdDarkThemeDesc}
                </span>
              </div>
            </div>
            {theme === 'dark' ? <Check className="size-4 shrink-0" /> : null}
          </CommandItem>
        </CommandGroup>

        <CommandSeparator />
        <CommandGroup>
          {LANGUAGE_OPTIONS.map((option) => (
            <CommandItem
              key={option.code}
              value={`${option.label} ${option.code} ${t.palette.cmdLanguageDesc}`}
              onSelect={() => handleSelect(() => onSetLanguage(option.code))}
              className="flex items-center justify-between py-2 px-2.5 rounded-lg cursor-pointer"
            >
              <div className="truncate">
                <span className="font-medium text-foreground text-xs">{option.label}</span>
                <span className="text-xs text-muted-foreground ml-2 truncate">
                  {t.palette.cmdLanguageDesc}
                </span>
              </div>
              {language === option.code ? <Check className="size-4 shrink-0" /> : null}
            </CommandItem>
          ))}
        </CommandGroup>
      </CommandList>

      {/* Footer shortcut hints */}
      <div className="px-4 py-2 border-t border-border bg-muted/40 flex items-center justify-between text-xs font-mono text-muted-foreground">
        <div className="flex items-center gap-3">
          <span>↑↓ {t.palette.navigate}</span>
          <span>↵ {t.palette.execute}</span>
        </div>
        <span className="text-xs uppercase tracking-wider">{t.palette.footerBrand}</span>
      </div>
    </CommandDialog>
  )
}
