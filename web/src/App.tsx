import React, { useState } from 'react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { I18nProvider } from '@/i18n/I18nContext'
import { ConsoleLayout, type ViewTab } from '@/components/layout/ConsoleLayout'
import { NewJobWizard } from '@/views/wizard/NewJobWizard'
import { MissionControl } from '@/views/mission-control/MissionControl'
import { QualityGateView } from '@/views/quality-gate/QualityGateView'
import { LanguageAssetsView } from '@/views/language-assets/LanguageAssetsView'
import { SystemDoctorView } from '@/views/doctor/SystemDoctorView'
import { ReviewWorkbench } from '@/views/review/ReviewWorkbench'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
})

function ConsoleApp() {
  const [currentTab, setCurrentTab] = useState<ViewTab>('wizard')
  const [currentJobId, setCurrentJobId] = useState<string | null>(null)

  const handleJobStarted = (jobId: string) => {
    setCurrentJobId(jobId)
    setCurrentTab('jobs')
  }

  const handleInspectQuality = (jobId: string) => {
    setCurrentJobId(jobId)
    setCurrentTab('quality')
  }

  const handleOpenReview = (jobId: string) => {
    setCurrentJobId(jobId)
    setCurrentTab('review')
  }

  return (
    <ConsoleLayout currentTab={currentTab} onSelectTab={setCurrentTab}>
      {currentTab === 'wizard' && <NewJobWizard onJobStarted={handleJobStarted} />}
      {currentTab === 'jobs' && (
        <MissionControl
          currentJobId={currentJobId}
          onSelectJob={setCurrentJobId}
          onInspectQuality={handleInspectQuality}
          onOpenReview={handleOpenReview}
        />
      )}
      {currentTab === 'quality' && (
        <QualityGateView jobId={currentJobId} onOpenReview={handleOpenReview} />
      )}
      {currentTab === 'assets' && <LanguageAssetsView />}
      {currentTab === 'system' && <SystemDoctorView />}
      {currentTab === 'review' && <ReviewWorkbench jobId={currentJobId} />}
    </ConsoleLayout>
  )
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <I18nProvider>
        <ConsoleApp />
      </I18nProvider>
    </QueryClientProvider>
  )
}

export default App
