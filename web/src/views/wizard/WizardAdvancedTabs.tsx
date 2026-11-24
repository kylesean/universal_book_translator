import { Link } from 'react-router-dom'
import { AlertTriangle, FileText, Loader2, UploadCloud, X } from 'lucide-react'
import { Alert, AlertDescription } from '@/components/ui/Alert'
import { Button } from '@/components/ui/button'
import { InfoTip } from '@/components/ui/InfoTip'
import { Input } from '@/components/ui/input'
import { Switch } from '@/components/ui/switch'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { type RuntimeConfig } from '@/api/client'
import { type TranslationDictionary } from '@/i18n/types'
import { ChoiceSelect, FieldLabel } from './WizardPrimitives'

export interface WizardAdvancedTabsProps {
  t: TranslationDictionary
  dualMode: 'auto' | 'dual' | 'mono'
  onDualModeChange: (val: 'auto' | 'dual' | 'mono') => void
  emitBoth: 'auto' | 'on' | 'off'
  emitBothHelpText: string
  onEmitBothChange: (val: 'auto' | 'on' | 'off') => void
  layoutColumns: 'auto' | '1' | '2'
  onLayoutColumnsChange: (val: 'auto' | '1' | '2') => void
  translateChrome: 'auto' | 'on' | 'off'
  onTranslateChromeChange: (val: 'auto' | 'on' | 'off') => void
  deliveryMode: 'auto' | 'readable' | 'faithful'
  pdfEngine: 'auto' | 'docling' | 'native' | 'pdfium'
  onPdfEngineChange: (val: 'auto' | 'docling' | 'native' | 'pdfium') => void
  pdfEngineConflict: boolean
  spillPolicy: 'auto' | 'error' | 'warn' | 'appendix'
  onSpillPolicyChange: (val: 'auto' | 'error' | 'warn' | 'appendix') => void
  domainProfile: string
  onDomainProfileChange: (val: string) => void
  domainHint: string
  onDomainHintChange: (val: string) => void
  ocrModeControl: React.ReactNode
  ocrFallback: 'auto' | 'off' | 'missing' | 'weak' | 'all'
  onOcrFallbackChange: (val: 'auto' | 'off' | 'missing' | 'weak' | 'all') => void
  enableDynamicRouting: 'auto' | 'on' | 'off'
  onEnableDynamicRoutingChange: (val: 'auto' | 'on' | 'off') => void
  dynamicRoutingPolicy: 'pareto' | 'quality_first' | 'cost_first'
  onDynamicRoutingPolicyChange: (val: 'pareto' | 'quality_first' | 'cost_first') => void
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
  dryRun: boolean
  onDryRunChange: (val: boolean) => void
  declaredProviders: RuntimeConfig['providers']
  repairProvider: string
  onRepairProviderChange: (val: string) => void
  concurrency: string
  onConcurrencyChange: (val: string) => void
  concurrencyInvalid: boolean
  exportAllowNeedsHuman: 'auto' | 'on' | 'off'
  onExportAllowNeedsHumanChange: (val: 'auto' | 'on' | 'off') => void
}

export function WizardAdvancedTabs(props: WizardAdvancedTabsProps) {
  const { t } = props
  return (
    <Tabs defaultValue="delivery">
      <TabsList className="grid h-auto w-full grid-cols-2 gap-1 p-1 sm:grid-cols-4">
        <TabsTrigger value="delivery">{t.wizard.groupDelivery}</TabsTrigger>
        <TabsTrigger value="content">{t.wizard.groupContent}</TabsTrigger>
        <TabsTrigger value="scope">{t.wizard.groupScope}</TabsTrigger>
        <TabsTrigger value="run">{t.wizard.groupExec}</TabsTrigger>
      </TabsList>

      <TabsContent value="delivery" className="space-y-4 pt-3">
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
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <FieldLabel htmlFor="wizard-layout-columns">{t.wizard.layoutColumns}</FieldLabel>
              <InfoTip label={t.wizard.layoutColumns}>{t.wizard.layoutColumnsHelp}</InfoTip>
            </div>
            <ChoiceSelect
              id="wizard-layout-columns"
              value={props.layoutColumns}
              onValueChange={(value) => props.onLayoutColumnsChange(value as typeof props.layoutColumns)}
              options={[
                { value: 'auto', label: t.wizard.layoutColumnsAuto },
                { value: '1', label: t.wizard.layoutColumnsOne },
                { value: '2', label: t.wizard.layoutColumnsTwo },
              ]}
            />
          </div>
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <FieldLabel htmlFor="wizard-translate-chrome">{t.wizard.translateChrome}</FieldLabel>
              <InfoTip label={t.wizard.translateChrome}>{t.wizard.translateChromeHelp}</InfoTip>
            </div>
            <ChoiceSelect
              id="wizard-translate-chrome"
              value={props.translateChrome}
              onValueChange={(value) => props.onTranslateChromeChange(value as typeof props.translateChrome)}
              options={[
                { value: 'auto', label: t.wizard.translateChromeAuto },
                { value: 'on', label: t.wizard.translateChromeOn },
                { value: 'off', label: t.wizard.translateChromeOff },
              ]}
            />
          </div>
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <FieldLabel htmlFor="wizard-pdf-engine">{t.wizard.pdfEngineLabel}</FieldLabel>
              <InfoTip label={t.wizard.pdfEngineLabel}>{t.wizard.pdfEngineHelp}</InfoTip>
            </div>
            <ChoiceSelect
              id="wizard-pdf-engine"
              value={props.pdfEngine}
              onValueChange={(value) => props.onPdfEngineChange(value as typeof props.pdfEngine)}
              options={[
                { value: 'auto', label: t.wizard.pdfEngineAuto },
                { value: 'docling', label: t.wizard.pdfEngineDocling },
                { value: 'native', label: t.wizard.pdfEngineNative },
                { value: 'pdfium', label: t.wizard.pdfEnginePdfium },
              ]}
            />
            {props.pdfEngineConflict ? (
              <div className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-amber-500/30 bg-amber-500/10 px-2.5 py-2 text-xs text-amber-900 dark:text-amber-200">
                <span className="leading-relaxed">{t.wizard.pdfEngineConflictFaithful}</span>
                <Button
                  type="button"
                  size="sm"
                  variant="link"
                  onClick={() => props.onPdfEngineChange('auto')}
                  className="h-auto p-0 text-xs font-semibold text-amber-900 underline underline-offset-2 dark:text-amber-100"
                >
                  {t.wizard.pdfEngineResetAuto}
                </Button>
              </div>
            ) : null}
          </div>
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <FieldLabel htmlFor="wizard-spill-policy">{t.wizard.spillPolicyLabel}</FieldLabel>
              <InfoTip label={t.wizard.spillPolicyLabel}>{t.wizard.spillPolicyHelp}</InfoTip>
            </div>
            <ChoiceSelect
              id="wizard-spill-policy"
              value={props.spillPolicy}
              onValueChange={(value) => props.onSpillPolicyChange(value as typeof props.spillPolicy)}
              options={[
                { value: 'auto', label: t.wizard.toggleAuto },
                { value: 'error', label: t.wizard.spillError },
                { value: 'warn', label: t.wizard.spillWarn },
                { value: 'appendix', label: t.wizard.spillAppendix },
              ]}
            />
          </div>
        </div>
      </TabsContent>

      <TabsContent value="content" className="space-y-4 pt-3">
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
          {props.ocrModeControl}
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <FieldLabel htmlFor="wizard-ocr-fallback">{t.wizard.ocrFallbackLabel}</FieldLabel>
              <InfoTip label={t.wizard.ocrFallbackLabel}>{t.wizard.ocrFallbackHelp}</InfoTip>
            </div>
            <ChoiceSelect
              id="wizard-ocr-fallback"
              value={props.ocrFallback}
              onValueChange={(value) => props.onOcrFallbackChange(value as typeof props.ocrFallback)}
              options={[
                { value: 'auto', label: t.wizard.toggleAuto },
                { value: 'off', label: t.wizard.ocrFallbackOff },
                { value: 'missing', label: t.wizard.ocrFallbackMissing },
                { value: 'weak', label: t.wizard.ocrFallbackWeak },
                { value: 'all', label: t.wizard.ocrFallbackAll },
              ]}
            />
          </div>
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <FieldLabel htmlFor="wizard-dynamic-routing">{t.wizard.dynamicRoutingLabel}</FieldLabel>
              <InfoTip label={t.wizard.dynamicRoutingLabel}>{t.wizard.dynamicRoutingHelp}</InfoTip>
            </div>
            <ChoiceSelect
              id="wizard-dynamic-routing"
              value={props.enableDynamicRouting}
              onValueChange={(value) =>
                props.onEnableDynamicRoutingChange(value as typeof props.enableDynamicRouting)
              }
              options={[
                { value: 'auto', label: t.wizard.toggleAuto },
                { value: 'on', label: t.wizard.toggleOn },
                { value: 'off', label: t.wizard.toggleOff },
              ]}
            />
          </div>
          {props.enableDynamicRouting !== 'off' && (
            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5">
                <FieldLabel htmlFor="wizard-dynamic-policy">
                  {t.wizard.dynamicRoutingPolicyLabel}
                </FieldLabel>
                <InfoTip label={t.wizard.dynamicRoutingPolicyLabel}>
                  {t.wizard.dynamicRoutingPolicyHelp}
                </InfoTip>
              </div>
              <ChoiceSelect
                id="wizard-dynamic-policy"
                value={props.dynamicRoutingPolicy}
                onValueChange={(value) =>
                  props.onDynamicRoutingPolicyChange(value as typeof props.dynamicRoutingPolicy)
                }
                options={[
                  { value: 'pareto', label: t.wizard.dynamicRoutingPareto },
                  { value: 'quality_first', label: t.wizard.dynamicRoutingQualityFirst },
                  { value: 'cost_first', label: t.wizard.dynamicRoutingCostFirst },
                ]}
              />
            </div>
          )}
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
      </TabsContent>

      <TabsContent value="scope" className="space-y-4 pt-3">
        <div className="grid gap-4 sm:grid-cols-2">
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
                className="h-auto text-xs"
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
        </div>
        <div className="flex items-center justify-between rounded-lg border border-border bg-muted/20 px-3 py-2.5">
          <div className="flex items-center gap-1.5">
            <FieldLabel htmlFor="wizard-dry-run" className="cursor-pointer">
              {t.wizard.rehearsalLabel}
            </FieldLabel>
            <InfoTip label={t.wizard.rehearsalLabel}>{t.wizard.rehearsalHelp}</InfoTip>
          </div>
          <Switch id="wizard-dry-run" checked={props.dryRun} onCheckedChange={props.onDryRunChange} />
        </div>
        {props.dryRun && (
          <Alert variant="warning">
            <AlertTriangle className="size-4" />
            <AlertDescription>{t.wizard.rehearsalActiveWarning}</AlertDescription>
          </Alert>
        )}
      </TabsContent>

      <TabsContent value="run" className="space-y-4 pt-3">
        <div className="grid gap-4 sm:grid-cols-2">
          {props.declaredProviders.length > 0 ? (
            <div className="space-y-1.5">
              <div className="flex items-center gap-1.5">
                <FieldLabel htmlFor="wizard-repair-provider">
                  {t.wizard.repairProvider}
                </FieldLabel>
                <InfoTip label={t.wizard.repairProvider}>
                  {t.wizard.repairProviderHelp}
                </InfoTip>
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
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <FieldLabel htmlFor="wizard-export-needs-human">
                {t.wizard.exportAllowNeedsHumanLabel}
              </FieldLabel>
              <InfoTip label={t.wizard.exportAllowNeedsHumanLabel}>
                {t.wizard.exportAllowNeedsHumanHelp}
              </InfoTip>
            </div>
            <ChoiceSelect
              id="wizard-export-needs-human"
              value={props.exportAllowNeedsHuman}
              onValueChange={(value) =>
                props.onExportAllowNeedsHumanChange(value as typeof props.exportAllowNeedsHuman)
              }
              options={[
                { value: 'auto', label: t.wizard.toggleAuto },
                { value: 'on', label: t.wizard.toggleOn },
                { value: 'off', label: t.wizard.toggleOff },
              ]}
            />
          </div>
        </div>
      </TabsContent>
    </Tabs>
  )
}
