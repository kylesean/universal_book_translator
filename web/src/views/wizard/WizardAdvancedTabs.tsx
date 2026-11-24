import { Link } from 'react-router-dom'
import { FileText, Loader2, UploadCloud, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { InfoTip } from '@/components/ui/InfoTip'
import { Input } from '@/components/ui/input'
import { type RuntimeConfig } from '@/api/client'
import { type TranslationDictionary } from '@/i18n/types'
import { ChoiceSelect, FieldLabel } from './WizardPrimitives'

export interface WizardAdvancedTabsProps {
  t: TranslationDictionary
  /** Context flags used to show only options meaningful for the current document/mode. */
  isPdfDoc: boolean
  hasUploadedFile: boolean
  deliveryMode: 'auto' | 'readable' | 'faithful'
  showOcrControl: boolean
  dualMode: 'auto' | 'dual' | 'mono'
  onDualModeChange: (val: 'auto' | 'dual' | 'mono') => void
  emitBoth: 'auto' | 'on' | 'off'
  emitBothHelpText: string
  onEmitBothChange: (val: 'auto' | 'on' | 'off') => void
  layoutColumns: 'auto' | '1' | '2'
  onLayoutColumnsChange: (val: 'auto' | '1' | '2') => void
  translateChrome: 'auto' | 'on' | 'off'
  onTranslateChromeChange: (val: 'auto' | 'on' | 'off') => void
  domainProfile: string
  onDomainProfileChange: (val: string) => void
  domainHint: string
  onDomainHintChange: (val: string) => void
  ocrModeControl: React.ReactNode
  glossaryFile: { name: string; path: string } | null
  glossaryUploading: boolean
  glossaryError: string | null
  onTriggerGlossaryPick: () => void
  onClearGlossary: () => void
  pageRange: string
  onPageRangeChange: (val: string) => void
  chapterStart: string
  onChapterStartChange: (val: string) => void
  chapterCount: string
  onChapterCountChange: (val: string) => void
  chapterStartInvalid: boolean
  chapterCountInvalid: boolean
  declaredProviders: RuntimeConfig['providers']
  repairProvider: string
  onRepairProviderChange: (val: string) => void
  concurrency: string
  onConcurrencyChange: (val: string) => void
  concurrencyInvalid: boolean
}

export function WizardAdvancedTabs(props: WizardAdvancedTabsProps) {
  const { t } = props
  const showLayoutColumns = props.deliveryMode !== 'faithful'
  const showPageRange = !props.hasUploadedFile || props.isPdfDoc
  const showChapterWindow = props.hasUploadedFile && !props.isPdfDoc
  const showRepairProvider = props.declaredProviders.length > 1

  return (
    <div className="space-y-6">
      {/* 1. Layout & Scope */}
      <section className="space-y-3">
        <h3 className="text-xs font-semibold uppercase tracking-[0.08em] text-muted-foreground">
          {t.wizard.groupOutputScope}
        </h3>
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <FieldLabel htmlFor="wizard-dual-mode">{t.wizard.dualMode}</FieldLabel>
              <InfoTip label={t.wizard.dualMode}>{t.wizard.dualModeHelp}</InfoTip>
            </div>
            <ChoiceSelect
              id="wizard-dual-mode"
              value={props.dualMode}
              onValueChange={(value) => props.onDualModeChange(value as typeof props.dualMode)}
              options={[
                { value: 'auto', label: t.wizard.dualModeAuto },
                { value: 'dual', label: t.wizard.dualModeDual },
                { value: 'mono', label: t.wizard.dualModeMono },
              ]}
            />
          </div>

          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <FieldLabel htmlFor="wizard-emit-both">{t.wizard.emitBoth}</FieldLabel>
              <InfoTip label={t.wizard.emitBoth}>{props.emitBothHelpText}</InfoTip>
            </div>
            <ChoiceSelect
              id="wizard-emit-both"
              value={props.emitBoth}
              onValueChange={(value) => props.onEmitBothChange(value as typeof props.emitBoth)}
              options={[
                { value: 'auto', label: t.wizard.emitBothAuto },
                { value: 'on', label: t.wizard.emitBothOn },
                { value: 'off', label: t.wizard.emitBothOff },
              ]}
            />
          </div>

          {showLayoutColumns ? (
            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5">
                <FieldLabel htmlFor="wizard-layout-columns">{t.wizard.layoutColumns}</FieldLabel>
                <InfoTip label={t.wizard.layoutColumns}>{t.wizard.layoutColumnsHelp}</InfoTip>
              </div>
              <ChoiceSelect
                id="wizard-layout-columns"
                value={props.layoutColumns}
                onValueChange={(value) =>
                  props.onLayoutColumnsChange(value as typeof props.layoutColumns)
                }
                options={[
                  { value: 'auto', label: t.wizard.layoutColumnsAuto },
                  { value: '1', label: t.wizard.layoutColumnsOne },
                  { value: '2', label: t.wizard.layoutColumnsTwo },
                ]}
              />
            </div>
          ) : null}

          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <FieldLabel htmlFor="wizard-translate-chrome">{t.wizard.translateChrome}</FieldLabel>
              <InfoTip label={t.wizard.translateChrome}>{t.wizard.translateChromeHelp}</InfoTip>
            </div>
            <ChoiceSelect
              id="wizard-translate-chrome"
              value={props.translateChrome}
              onValueChange={(value) =>
                props.onTranslateChromeChange(value as typeof props.translateChrome)
              }
              options={[
                { value: 'auto', label: t.wizard.translateChromeAuto },
                { value: 'on', label: t.wizard.translateChromeOn },
                { value: 'off', label: t.wizard.translateChromeOff },
              ]}
            />
          </div>

          {showPageRange ? (
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <FieldLabel htmlFor="wizard-page-range">{t.wizard.pageFilter}</FieldLabel>
                  <InfoTip label={t.wizard.pageFilter}>{t.wizard.pageFilterHelp}</InfoTip>
                </div>
                <Button
                  type="button"
                  variant="link"
                  size="sm"
                  onClick={() => props.onPageRangeChange('1-5')}
                  className="h-auto p-0 text-xs"
                >
                  {t.wizard.sliceSampleQuickBtn}
                </Button>
              </div>
              <Input
                id="wizard-page-range"
                value={props.pageRange}
                onChange={(event) => props.onPageRangeChange(event.target.value)}
                placeholder={t.wizard.pageFilterPlaceholder}
              />
            </div>
          ) : null}

          {showChapterWindow ? (
            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5">
                <FieldLabel>{t.wizard.chapterWindow}</FieldLabel>
                <InfoTip label={t.wizard.chapterWindow}>{t.wizard.chapterWindowHelp}</InfoTip>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Input
                  type="number"
                  min={1}
                  value={props.chapterStart}
                  onChange={(event) => props.onChapterStartChange(event.target.value)}
                  placeholder={t.wizard.chapterStart}
                  aria-label={t.wizard.chapterStart}
                />
                <Input
                  type="number"
                  min={1}
                  value={props.chapterCount}
                  onChange={(event) => props.onChapterCountChange(event.target.value)}
                  placeholder={t.wizard.chapterCount}
                  aria-label={t.wizard.chapterCount}
                />
              </div>
              {(props.chapterStartInvalid || props.chapterCountInvalid) && (
                <p className="text-xs text-destructive">{t.wizard.fieldMinOne}</p>
              )}
            </div>
          ) : null}
        </div>
      </section>

      {/* 2. Domain & Glossary */}
      <section className="space-y-3 border-t border-border/60 pt-4">
        <h3 className="text-xs font-semibold uppercase tracking-[0.08em] text-muted-foreground">
          {t.wizard.groupTerminology}
        </h3>
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <FieldLabel htmlFor="wizard-domain-profile">{t.wizard.domainProfile}</FieldLabel>
              <InfoTip label={t.wizard.domainProfile}>{t.wizard.domainProfileHelp}</InfoTip>
            </div>
            <ChoiceSelect
              id="wizard-domain-profile"
              value={props.domainProfile}
              onValueChange={props.onDomainProfileChange}
              options={[
                { value: 'auto', label: t.wizard.domainAuto },
                { value: 'general', label: t.wizard.domainGeneral },
                { value: 'textbook', label: t.wizard.domainTextbook },
                { value: 'paper', label: t.wizard.domainPaper },
                { value: 'fiction', label: t.wizard.domainFiction },
                { value: 'humanities', label: t.wizard.domainHumanities },
              ]}
            />
          </div>

          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <FieldLabel htmlFor="wizard-domain-hint">{t.wizard.domainHint}</FieldLabel>
              <InfoTip label={t.wizard.domainHint}>{t.wizard.domainHintHelp}</InfoTip>
            </div>
            <Input
              id="wizard-domain-hint"
              value={props.domainHint}
              onChange={(event) => props.onDomainHintChange(event.target.value)}
            />
          </div>

          <div className="space-y-1.5 sm:col-span-2">
            <div className="flex items-center gap-1.5">
              <FieldLabel>{t.wizard.glossaryPath}</FieldLabel>
              <InfoTip label={t.wizard.glossaryPath}>{t.wizard.glossaryPathHelp}</InfoTip>
            </div>
            {props.glossaryFile ? (
              <div className="flex items-center gap-2">
                <div className="flex h-9 min-w-0 flex-1 items-center gap-2 rounded-lg border border-border bg-muted/35 px-3 text-xs">
                  <FileText className="size-3.5 shrink-0 text-muted-foreground" />
                  <span className="truncate">{props.glossaryFile.name}</span>
                </div>
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  onClick={props.onClearGlossary}
                  aria-label={t.wizard.glossaryRemove}
                >
                  <X />
                </Button>
              </div>
            ) : (
              <div className="flex flex-wrap items-center gap-3">
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  disabled={props.glossaryUploading}
                  onClick={props.onTriggerGlossaryPick}
                >
                  {props.glossaryUploading ? (
                    <Loader2 className="animate-spin" />
                  ) : (
                    <UploadCloud />
                  )}
                  {t.wizard.glossaryUploadBtn}
                </Button>
                <Link
                  to="/assets"
                  className="text-xs text-muted-foreground underline underline-offset-2 hover:text-foreground"
                >
                  {t.wizard.glossaryManageLink}
                </Link>
              </div>
            )}
            {props.glossaryError && (
              <p className="text-xs text-destructive">{props.glossaryError}</p>
            )}
          </div>
        </div>
      </section>

      {/* 3. Execution & OCR */}
      <section className="space-y-3 border-t border-border/60 pt-4">
        <h3 className="text-xs font-semibold uppercase tracking-[0.08em] text-muted-foreground">
          {t.wizard.groupExecution}
        </h3>
        <div className="grid gap-4 sm:grid-cols-2">
          {props.showOcrControl ? props.ocrModeControl : null}

          {showRepairProvider ? (
            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5">
                <FieldLabel htmlFor="wizard-repair-provider">
                  {t.wizard.repairProvider}
                </FieldLabel>
                <InfoTip label={t.wizard.repairProvider}>{t.wizard.repairProviderHelp}</InfoTip>
              </div>
              <ChoiceSelect
                id="wizard-repair-provider"
                value={props.repairProvider || '__follow__'}
                onValueChange={(value) =>
                  props.onRepairProviderChange(value === '__follow__' ? '' : value)
                }
                options={[
                  {
                    value: '__follow__',
                    label: t.wizard.repairProviderDefault,
                  },
                  ...props.declaredProviders.map((p) => ({
                    value: p.name,
                    label: p.name + (p.draft_model ? ` — ${p.draft_model}` : ''),
                  })),
                ]}
              />
            </div>
          ) : null}

          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <FieldLabel>{t.wizard.concurrency}</FieldLabel>
              <InfoTip label={t.wizard.concurrency}>{t.wizard.concurrencyHelp}</InfoTip>
            </div>
            <Input
              type="number"
              min={1}
              value={props.concurrency}
              onChange={(event) => props.onConcurrencyChange(event.target.value)}
              placeholder={t.wizard.concurrencyAuto}
            />
            {props.concurrencyInvalid && (
              <p className="text-xs text-destructive">{t.wizard.fieldMinOne}</p>
            )}
          </div>
        </div>
      </section>
    </div>
  )
}
