export type Language = 'en' | 'zh'

export interface TranslationDictionary {
  common: {
    compilerConsole: string
    disconnected: string
    cancel: string
    confirm: string
    status: string
    search: string
    ready: string
    running: string
    completed: string
    failed: string
    warning: string
    online: string
    idle: string
    dismiss: string
    paperToneCotton: string
    paperToneDowling: string
    paperToneObsidian: string
    paperToneSwitchTo: string
    switchLanguage: string
    modeReader: string
    modeStudio: string
    switchToReader: string
    switchToStudio: string
  }
  reader: {
    title: string
    subtitle: string
    dropzoneTitle: string
    dropzoneSubtitle: string
    selectFile: string
    orEnterPath: string
    targetLanguage: string
    stylePreset: string
    styleLiterary: string
    styleLiteraryDesc: string
    styleAcademic: string
    styleAcademicDesc: string
    styleTechnical: string
    styleTechnicalDesc: string
    startTranslation: string
    translating: string
    translatingDesc: string
    stepParse: string
    stepTranslate: string
    stepRender: string
    stepReady: string
    recentBooks: string
    noBooksYet: string
    readAndExport: string
    backToShelf: string
    exportDeliverable: string
    exporting: string
    bilingualMode: string
    targetOnlyMode: string
    sourceOnlyMode: string
    editParagraph: string
    saveEdit: string
    cancelEdit: string
    quickEditHint: string
  }
  nav: {
    operatorControl: string
    newJob: string
    missionControl: string
    qualityGate: string
    languageAssets: string
    systemDoctor: string
    workbenches: string
    reviewWorkbench: string
    toggleNavigation: string
    closeNavigation: string
  }
  wizard: {
    title: string
    subtitle: string
    dropzoneTitle: string
    dropzoneSubtitle: string
    uploadAria: string
    dropzoneActive: string
    uploading: string
    orBrowse: string
    manualPathPlaceholder: string
    sourceLang: string
    provider: string
    providerHelp: string
    providerDefault: string
    targetLang: string
    presets: string
    presetHighFidelity: string
    presetHighFidelityDesc: string
    presetStandard: string
    presetStandardDesc: string
    presetVectorMath: string
    presetVectorMathDesc: string
    presetFast: string
    presetFastDesc: string
    presetPreview: string
    presetPreviewDesc: string
    /** Badge on a preset card: the engine parameters that tell it apart. */
    presetBadgeRich: string
    presetBadgeAdaptive: string
    presetBadgeVectorMath: string
    presetBadgeFormulaImage: string
    presetBadgeShortChain: string
    /** Warning badge on the one preset that pays for VLM formula transcription. */
    presetSlowBadge: string
    walLedger: string
    notPriced: string
    formatUnknown: string
    budgetCeiling: string
    budgetUnlimited: string
    budgetUnlimitedToggle: string
    budgetPlaceholder: string
    budgetHelp: string
    budgetInvalid: string
    fieldMinOne: string
    reassessing: string
    assessStale: string
    budgetExceedsQuote: string
    preflightTitle: string
    preflightSubtitle: string
    waitingFile: string
    assessBtn: string
    startCompile: string
    compiling: string
    estTokens: string
    /** Breakdown labels under the total token estimate. */
    tokensSource: string
    tokensPrompt: string
    tokensCompletion: string
    tokensEstimateNote: string
    /** Tag on the duration estimate, which is derived from rate-limit
     *  knobs rather than measured; the engine's basis rides in an InfoTip. */
    heuristicTag: string
    estTimeHeuristicNote: string
    /** Task-order section headers. */
    stepFile: string
    stepQuality: string
    stepBudget: string
    basicsTitle: string
    /** Submit guard: no valid quote for the current form. */
    quoteRiskTitle: string
    quoteRiskBodyNoQuote: string
    quoteRiskBodyStale: string
    quoteRiskUnlimited: string
    quoteRiskConfirm: string
    quoteRiskCancel: string
    budgetSuggest: string
    /** Amber marker on the advanced-options toggle when a field inside
     *  the collapsed drawer is invalid — the signal sits where the fix
     *  happens, not under the disabled button. */
    advancedErrorTag: string
    /** Mechanism detail for each preset, folded behind an InfoTip. */
    presetHFDetail: string
    presetStandardDetail: string
    presetVectorDetail: string
    presetPreviewDetail: string
    presetFastDetail: string
    estCost: string
    pageCount: string
    recommendation: string
    /** Short labels for the route badge's dual-mode / preset enums. */
    routeMono: string
    routeDual: string
    routeHighFidelity: string
    routeStandard: string
    scannedRatio: string
    sourceChars: string
    chapters: string
    mathDensity: string
    /** Distinguishes the text-level density from the layout-level formula count. */
    mathDensityTextOnly: string
    mathDensityWithObjects: string
    estTime: string
    confidence: string
    probeWarnings: string
    /** Pre-flight warning copy keyed by the engine's stable warning code. */
    warningCodes: Record<string, string>
    warningRemedyRescan: string
    /** ``{count}`` is replaced with the at-risk page count. */
    warningAtRiskClause: string
    warningRemedyOcr: string
    /** Separator for a warning param that is a list (e.g. unpriced model names). */
    listSeparator: string
    advancedOptionsTitle: string
    advancedOptionsSubtitle: string
    advancedOptionsTag: string
    domainProfile: string
    domainProfileHelp: string
    domainAuto: string
    domainGeneral: string
    domainTextbook: string
    domainPaper: string
    domainFiction: string
    domainHumanities: string
    dualMode: string
    dualModeHelp: string
    dualModeAuto: string
    dualModeDual: string
    dualModeMono: string
    layoutColumns: string
    layoutColumnsHelp: string
    layoutColumnsAuto: string
    layoutColumnsOne: string
    layoutColumnsTwo: string
    translateChrome: string
    translateChromeHelp: string
    translateChromeAuto: string
    translateChromeOn: string
    translateChromeOff: string
    /** Headings that group the advanced drawer by intent. */
    groupDelivery: string
    groupContent: string
    groupScope: string
    groupExec: string
    ocrMode: string
    ocrModeHelp: string
    ocrModeEgressWarning: string
    ocrModeAuto: string
    ocrModeRapidocr: string
    ocrModeSidecar: string
    ocrModeCloud: string
    ocrModeVlm: string
    ocrModeOff: string
    pageFilter: string
    pageFilterPlaceholder: string
    pageFilterHelp: string
    glossaryPath: string
    glossaryPathHelp: string
    glossaryPathInvalid: string
    /** Cross-link from the per-book glossary field to the asset manager. */
    glossaryManageLink: string
    glossaryUploadBtn: string
    glossaryRemove: string
    rehearsalLabel: string
    rehearsalHelp: string
    rehearsalBannerTitle: string
    rehearsalBannerBody: string
    rehearsalBannerCta: string
    rehearsalBannerAck: string
    /** The page-egress gate: inline enable action at the two controls it blocks. */
    egressEnableBtn: string
    egressEnableFailed: string
    egressConfirmTitle: string
    egressConfirmBody: string
    egressConfirmOn: string
    egressConfirmCancel: string
    rehearsalActiveWarning: string
    engineRecommended: string
    sliceSampleQuickBtn: string
    localFreeLabel: string
    sliceTestingTag: string
    estimateExact: string
    estimateApproximate: string
    /** Shown when a submit matched an existing job id instead of starting one. */
    submitReusedNote: string
    /** Names the provider the quote was priced against. */
    quoteProviderNote: string
    emitBoth: string
    emitBothAuto: string
    emitBothOn: string
    emitBothOff: string
    /** Dynamic: names the concrete complementary deliverable for the chosen dual_mode. */
    emitBothHelpAuto: string
    emitBothHelpDual: string
    emitBothHelpMono: string
    /** The page-egress gate: the visual judge cannot run while it is closed. */
    visualJudge: string
    visualJudgeHelp: string
    visualJudgeBlocked: string
    concurrency: string
    concurrencyHelp: string
    concurrencyAuto: string
    domainHint: string
    domainHintHelp: string
    chapterWindow: string
    chapterWindowHelp: string
    chapterStart: string
    chapterCount: string
    chapterWindowAuto: string
    /** Generic tri-state labels shared by the new delivery switches. */
    toggleAuto: string
    toggleOn: string
    toggleOff: string
  }
  mission: {
    title: string
    noActiveJobTitle: string
    jobMissingTitle: string
    cancelTitle: string
    cancelConfirm: string
    cancelling: string
    cancelFailed: string
    jobMissingDesc: string
    backToQueue: string
    noActiveJobDesc: string
    pipelineGraph: string
    stageExtract: string
    stageSegment: string
    stageTm: string
    stageTranslate: string
    stageQe: string
    stageRepair: string
    stageRender: string
    stageVerify: string
    stagePackage: string
    liveStream: string
    cancelCompile: string
    inspectGate: string
    openWorkbench: string
    progress: string
    completedPages: string
    skippedBlocks: string
    failedBlocks: string
    needsHumanBlocks: string
    qeNotMeasuredHeuristic: string
    queuePosition: string
    totalSpend: string
    executionState: string
    blocksRemaining: string
    stateCertified: string
    blockedDownloadTitle: string
    blockedDownloadBody: string
    blockedDownloadConfirm: string
    stateBlocked: string
    stateNeedsReview: string
    stateDrafting: string
    stageStateExtract: string
    stageStateSegment: string
    stageStateTm: string
    stageStateTranslate: string
    stageStateQe: string
    stageStateRepair: string
    stageStateRender: string
    stageStateVerify: string
    stageStatePackage: string
    engineActiveHeartbeat: string
    logAborted: string
    logResumed: string
    logDeleted: string
    sseStream: string
    bufferedLines: string
    waitingStream: string
    logUnavailableTerminal: string
    queueTitle: string
    queueEmpty: string
    colJob: string
    colFile: string
    colStatus: string
    colProgress: string
    colCost: string
    colUpdated: string
    selectJob: string
    logSearch: string
    autoScroll: string
    noMatch: string
    resumeCompile: string
    failureReason: string
    rowRehearsal: string
    resuming: string
    downloadPrimary: string
    oneClickExport: string
    exportFormats: string
    exportPreparing: string
    exportSuccess: string
    noDeliverables: string
    deleteJob: string
    deleteConfirm: string
    deleteTypePrompt: string
    deleteTypeMismatch: string
    jobIdLabel: string
    collapseLogs: string
    expandLogs: string
    //: Job statuses shown as a badge on the header and in the queue rows.
    //: Both used ``status.toUpperCase()``, so "COMPLETED" appeared in every
    //: locale.
    jobStatuses: Record<string, string>
    //: Pipeline stage names, from the engine's stage identifiers.
    stageNames: Record<string, string>
    stageLine: string
    qeHeuristicHint: string
    avgQe: string
    //: Download is blocked while a job has un-re-rendered edits; the console
    //: re-renders and retries rather than showing the 409.
    reRenderingBeforeDownload: string
    downloadFailed: string
    deliveryStaleBadge: string
    //: Re-parse: the only operation that rebuilds a finished job's parse, and
    //: the only one that discards the ledger. Confirmed by typing the job id.
    reparseAction: string
    reparseTitle: string
    reparseConfirm: string
    reparseHint: string
    reparsing: string
    reparseStarted: string
    reparseFailed: string
    reparseTypePrompt: string
    reparseTypeMismatch: string
  }
  quality: {
    title: string
    subtitle: string
    noJobTitle: string
    noJobDesc: string
    rehearsalBadge: string
    rehearsalBody: string
    decisionReady: string
    decisionBlocked: string
    decisionWarning: string
    allPassedDesc: string
    blockedDesc: string
    warningDesc: string
    costNotPriced: string
    fidelityScore: string
    pillar1Title: string
    pillar1Desc: string
    pillar2Title: string
    pillar2Desc: string
    pillar3Title: string
    pillar3Desc: string
    pillar4Title: string
    pillar4Desc: string
    pillar5Title: string
    pillar5Desc: string
    deliverablesTitle: string
    deliverablesSubtitle: string
    targetPdf: string
    epub: string
    contractJson: string
    secondaryDoc: string
    qualityReportJson: string
    visualReportJson: string
    metricsJson: string
    //: The two companions the export stage writes by default. Their keys were
    //: missing from the label switch, so the button fell back to the backend's
    //: English `label` — the only two deliverables still untranslated.
    htmlView: string
    //: The files are older than the ledger text; the download re-renders first.
    deliveryStaleNote: string
    //: Rendered instead of a verdict when the report could not be loaded.
    visualGateSkipped: string
    visualNoReport: string
    visualNoReportDesc: string
    /** Tier names the visual gate reports in ``tiers_ran`` / ``tiers_skipped``. */
    visualTierNames: Record<string, string>
    /** Stable ``skip_codes`` from the visual report, rendered in this language. */
    visualSkipCodes: Record<string, string>
    visualTiersRan: string
    visualTiersSkipped: string
    visualFullyRan: string
    noVerdictTitle: string
    noVerdictBody: string
    downloadFailed: string
    xliffDoc: string
    //: Quality-gate verdict badge, and the report's pass-rate phrasing.
    verdictPass: string
    verdictWarning: string
    verdictBlocked: string
    passRate: string
    avgQeValue: string
    qeNotMeasured: string
    qeNotMeasuredHint: string
    /** Why no QE average exists, and what turns it on. */
    qeNotMeasuredWhyHeuristic: string
    qeNotMeasuredWhyJudge: string
    qeNotMeasuredHow: string
    qeMeasuredOn: string
    pillarsTitle: string
    pillarStatusPass: string
    pillarStatusWarn: string
    pillarStatusFail: string
    axisMath: string
    axisTerminology: string
    axisVisual: string
    axisCompleteness: string
    axisDelivered: string
    termDriftTitle: string
    colTerm: string
    colExpected: string
    colOccurrences: string
    colDriftRate: string
    noDrift: string
    visualFindingsTitle: string
    noVisualFindings: string
    noDeliverables: string
    pageAlt: string
    flaggedPages: string
    auditDraftModel: string
    auditRepairModel: string
    auditQeEngine: string
    auditQeEngineRuleChecks: string
    auditQeEngineModelScoring: string
    auditQeEngineRehearsal: string
    auditQeEngineUnknown: string
    auditRetries: string
    auditCacheHit: string
    auditTypst: string
    metricMaskedSpans: string
    metricCorruptSpans: string
    metricMaskedBlocks: string
    metricCorruptBlocks: string
    metricFormulaBlocks: string
    //: View density toggle: 简明 for readers, 专业 for operators.
    modeLabel: string
    modeSimple: string
    modePro: string
    switchToPro: string
    //: 简明 verdict card — one sentence per outcome, no engine jargon.
    simpleReadyTitle: string
    simpleReadyBody: string
    simpleWarningTitle: string
    simpleWarningBody: string
    simpleBlockedTitle: string
    simpleBlockedBody: string
    //: Short check labels, shared by the verdict chips and the card titles.
    chipMath: string
    chipTerm: string
    chipLayout: string
    chipComplete: string
    chipCost: string
    //: One-line QE state for readers; the full explanation sits in the InfoTip.
    qeSimpleNote: string
    previewTranslation: string
    //: 简明 per-check sentences. Placeholders interpolate report numbers.
    simpleMathLine: string
    simpleTermLine: string
    simpleVisualCleanLine: string
    simpleVisualFindingsLine: string
    simpleVisualSkippedLine: string
    simpleVisualMissingLine: string
    simpleCompleteLine: string
    simpleCompleteFailLine: string
    simpleCostLine: string
    simpleCostUnpricedLine: string
    //: Honest pro-mode summary when some visual tiers ran and others did not.
    visualPartialRan: string
    radarUnmeasuredLabel: string
    radarUnmeasuredNote: string
    //: Deliverable groups — who each file is for, not what format it is.
    delivGroupReading: string
    delivGroupReadingDesc: string
    delivGroupL10n: string
    delivGroupL10nDesc: string
    delivGroupAudit: string
    delivGroupAuditDesc: string
    //: One plain-language line per deliverable key.
    descPrimary: string
    descSecondary: string
    descEpub: string
    descHtml: string
    descXliff: string
    descContract: string
    descQualityReport: string
    descVisualReport: string
    descMetrics: string
    previewAction: string
    downloadAction: string
    previewLoading: string
    previewDownloadInstead: string
    previewRawJson: string
    previewClose: string
    previewSegmentsTotal: string
    //: Digest labels for the JSON sidecar previews.
    delivCostLabel: string
    delivPagesScanned: string
    delivOverlapPairs: string
    delivTruncations: string
    delivTypeIssues: string
    delivRecall: string
    refreshBtn: string
  }
  assets: {
    colSource: string
    colTarget: string
    colProvenance: string
    colAction: string
    title: string
    subtitle: string
    tabGlossary: string
    tabTm: string
    tabBible: string
    sourceTerm: string
    targetTerm: string
    submitTerm: string
    activeTerms: string
    tmTitle: string
    tmDesc: string
    tmLoadMore: string
    tmLoadingMore: string
    conflictsTitle: string
    conflictTargets: string
    noConflicts: string
    importTitle: string
    importContent: string
    importLangs: string
    importBtn: string
    importing: string
    importDone: string
    fuzzyThreshold: string
    fuzzyThresholdHelp: string
    bibleOverviewTitle: string
    bibleOverviewDesc: string
    bibleStructureTitle: string
    bibleStructureDesc: string
    bibleSourceSeeds: string
    bibleSourceGlossary: string
    bibleSourceChapters: string
    bibleSourceMined: string
    bibleWorkflowTitle: string
    bibleWorkflowDesc: string
    bibleStep1: string
    bibleStep2: string
    bibleStep3: string
    //: The bible workflow's step badges and the empty-glossary row.
    stepBadge: string
    noTerms: string
    glossaryNotConfigured: string
    glossaryImportTitle: string
    glossaryImportHint: string
    glossaryImportBtn: string
    glossaryImportDone: string
    deleteTermTitle: string
    deleteTermBody: string
    evictTmTitle: string
    evictTmBody: string
  }
  doctor: {
    title: string
    subtitle: string
    auth: {
      title: string
      subtitle: string
      keyLabel: string
      keyPlaceholder: string
      signIn: string
      checking: string
      signOut: string
      failed: string
    }
    pingAll: string
    engineDiagnostics: string
    /** "What the engine will actually call" — replaces the old provider matrix. */
    runtimeConfig: string
    activeNow: string
    providerLabel: string
    draftModel: string
    repairModel: string
    endpoint: string
    capabilityProfile: string
    noProfileMatch: string
    noProfileMatchHelp: string
    notConfigured: string
    runtimeConfigUnavailable: string
    declaredProviders: string
    freeLabel: string
    builtinProviders: string
    aliasLabel: string
    protocolLabel: string
    aliasGroupLabel: string
    builtinProvidersHelp: string
    providerConfigHint: string
    credentialConfigured: string
    credentialMissing: string
    credentialLocalEndpoint: string
    credentialUnresolved: string
    credentialUnresolvedVar: string
    credentialShadowedByEnv: string
    credentialEnvSupplied: string
    editProvider: string
    editingProvider: string
    cancel: string
    save: string
    saving: string
    savedHint: string
    qeTitle: string
    qeHelp: string
    qeMeasured: string
    qeNotMeasured: string
    qeNotMeasuredHelp: string
    qeEngine: string
    qeEngineHeuristic: string
    qeEngineTiered: string
    qeEngineHelp: string
    qeJudge: string
    qeJudgeOn: string
    qeJudgeOff: string
    qeJudgeProvider: string
    qeJudgeProviderInherit: string
    qeJudgeModel: string
    qeJudgeModelInherit: string
    qeThreshold: string
    qeThresholdHelp: string
    qeThresholdInvalid: string
    qePassSample: string
    qePassSampleHelp: string
    qePassSampleInvalid: string
    qePassSampleZero: string
    qeSentences: string
    qeSentencesHelp: string
    qeInconsistent: string
    qeEnvOverride: string
    qeConfigSource: string
    qeEdit: string
    qeSave: string
    qeSaving: string
    qeSaveFailed: string
    qeSaved: string
    qeCancel: string
    qeLoadFailed: string
    /** Page-egress card: a runtime toggle for UBTConfig.allow_page_upload. */
    egressTitle: string
    egressStateOn: string
    egressStateOff: string
    egressLabel: string
    egressToggleOn: string
    egressToggleOff: string
    egressDescOff: string
    egressDescOn: string
    egressRuntimeNote: string
    egressConfirmTitle: string
    egressConfirmBody: string
    egressConfirmOn: string
    egressConfirmCancel: string
    egressSaveFailed: string
    capabilityTitle: string
    capabilityHelp: string
    capabilitySystem: string
    capabilitySystemNote: string
    capabilitySystemAccepted: string
    capabilitySystemMerged: string
    capabilityVision: string
    capabilityVisionNote: string
    capabilityUndeclared: string
    capabilityTroubleshootToggle: string
    baseUrl: string
    apiMode: string
    apiTimeout: string
    extractionStrategy: string
    extractionStrategyHelp: string
    extractionAutoNote: string
    deploymentBackend: string
    deploymentBackendHelp: string
    isFree: string
    addProvider: string
    newProviderName: string
    newProviderNameHint: string
    newProviderNameInvalid: string
    activateProvider: string
    activeProviderLabel: string
    defaultProviderLabel: string
    createProvider: string
    credentialField: string
    credentialPlaceholder: string
    credentialKeepHint: string
    credentialBlockedHint: string
    credentialUnknownHint: string
    frozenFieldsWarning: string
    firstRunTitle: string
    firstRunBody: string
    firstRunCta: string
    firstRunDismiss: string
    loadFailed: string
    fixLabel: string
    providerMatrix: string
    securityBoundary: string
    listenInterface: string
    allowedRoots: string
    forbiddenPaths: string
    authGate: string
    jobMode: string
    enabled: string
    disabled: string
    exposedWarning: string
    exposed: string
    loopbackOnly: string
    //: Diagnostic verdict shown as a badge on every check row. The engine's
    //: status codes ("OK" | "WARN" | "FAIL" | "SKIP") are stable, so they are
    //: the translation key.
    verdictCaption: string
    verdictAllOk: string
    verdictNeedsAttention: string
    expandDetail: string
    collapseDetail: string
    fixCopied: string
    securityForbiddenBadge: string
    jobModeHelp: string
    checkStatuses: Record<string, string>
  }
  review: {
    title: string
    modeSegments: string
    modeFlow: string
    modeVisual: string
    sourceSegment: string
    targetSegment: string
    saveFeedback: string
    failedDraftNotice: string
    quarantinedNotice: string
    sourceCanvas: string
    targetCanvas: string
    sourceCanvasAlt: string
    targetCanvasAlt: string
    filterAll: string
    filterIssues: string
    filterNeedsHuman: string
    noSegments: string
    loadMore: string
    saving: string
    saved: string
    //: Marks a block whose edit is restored from a previous session but not saved.
    unsavedDraft: string
    reExportDone: string
    reExportAction: string
    reExportStale: string
    reExporting: string
    reExportHelp: string
    reExportFailed: string
    humanVerified: string
    prevIssue: string
    nextIssue: string
    previewPage: string
    previewFailed: string
    rerenderPage: string
    rerenderHint: string
    recommendedTerm: string
    replaceTerm: string
    cascadeFix: string
    cascadeApplying: string
    cascadeDone: string
    cascadeConfirmTitle: string
    cascadeConfirmDesc: string
    cascadeConfirmProceed: string
    cascadeConfirmCancel: string
    scopeSelectLabel: string
    scopeBlock: string
    scopeSubsequent: string
    scopeAll: string
    scopeCount: string
    cascadeTmSync: string
    clearFilter: string
    filterByIssueKind: string
    diffBlend: string
    inspectorTitle: string
    inspectorQe: string
    inspectorTmHit: string
    inspectorGlossary: string
    inspectorRepairs: string
    inspectorSeverity: string
    inspectorProvenance: string
    editHint: string
    startEditing: string
    cancelEdit: string
    //: Names for the backend's issue kinds (``IssuesReport.counts`` keys).
    //: These were a hardcoded English map in the view, so the ribbon read
    //: "Terminology / Formula / Numeric …" in every locale.
    issueKinds: Record<string, string>
    //: Block types arrive from the engine as snake_case identifiers; the pane
    //: used to render them raw ("narrative", "list_item").
    blockTypes: Record<string, string>
    //: Block statuses, likewise: the card badge used to show the raw enum
    //: upper-cased ("NEEDS_HUMAN").
    blockStatuses: Record<string, string>
    //: MQM triage tiers ("critical" | "major" | "minor").
    mqmSeverities: Record<string, string>
    noIssues: string
    pageLabel: string
    errorFlags: string
    loadFailed: string
    loadMoreFailed: string
    saveFailed: string
    propagateFailed: string
    //: Shown when the ledger is held by another writer (HTTP 409), so the
    //: operator knows to stop the run or retry rather than assume the save lost.
    saveConflict: string
    oneClickExport: string
    exportFormats: string
    exportPreparing: string
    exportRerendering: string
    exportSuccess: string
    exportFailed: string
    exportBlockedTitle: string
    exportBlockedBody: string
    exportBlockedConfirm: string
    exportBlockedCancel: string
    exportPrimaryBadge: string
    exportQuickHelp: string
    editShortcutHint: string
    collapseInspector: string
    expandInspector: string
    exportSavingDrafts: string
  }
  errorBoundary: {
    title: string
    retry: string
  }
}
