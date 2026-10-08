import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom'
import { I18nProvider } from '@/i18n/I18nContext'
import { AuthGate } from '@/components/layout/AuthGate'
import { ConsoleLayout } from '@/components/layout/ConsoleLayout'
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

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <I18nProvider>
        <BrowserRouter>
          <AuthGate>
            <ConsoleLayout>
              <Routes>
                <Route path="/" element={<Navigate to="/wizard" replace />} />
                <Route path="/wizard" element={<NewJobWizard />} />
                <Route path="/jobs" element={<MissionControl />} />
                <Route path="/jobs/:jobId" element={<MissionControl />} />
                <Route path="/jobs/:jobId/quality" element={<QualityGateView />} />
                <Route path="/jobs/:jobId/review" element={<ReviewWorkbench />} />
                <Route path="/assets" element={<LanguageAssetsView />} />
                <Route path="/system" element={<SystemDoctorView />} />
                <Route path="*" element={<Navigate to="/wizard" replace />} />
              </Routes>
            </ConsoleLayout>
          </AuthGate>
        </BrowserRouter>
      </I18nProvider>
    </QueryClientProvider>
  )
}

export default App
