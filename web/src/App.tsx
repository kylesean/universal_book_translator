import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { I18nProvider } from '@/i18n/I18nContext'
import { AuthGate } from '@/components/layout/AuthGate'
import { ConsoleLayout } from '@/components/layout/ConsoleLayout'
import { ErrorBoundary } from '@/components/layout/ErrorBoundary'
import { ToastProvider } from '@/components/ui/Toast'
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

/**
 * Route-scoped error boundary.
 *
 * Keyed by pathname so navigating away from a crashed screen clears the error
 * and the new route mounts fresh, rather than showing the same failure notice
 * over a screen the operator has already left.
 */
function RouteBoundary({ children }: { children: React.ReactNode }) {
  const { pathname } = useLocation()
  return <ErrorBoundary key={pathname}>{children}</ErrorBoundary>
}

export function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <I18nProvider>
        <ToastProvider>
          <BrowserRouter>
            <AuthGate>
              <ConsoleLayout>
                {/* One boundary per route: a render error in one screen is
                    contained there instead of unmounting the whole console
                    (which would also blank a still-running job's dashboard). */}
                <RouteBoundary>
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
                </RouteBoundary>
              </ConsoleLayout>
            </AuthGate>
          </BrowserRouter>
        </ToastProvider>
      </I18nProvider>
    </QueryClientProvider>
  )
}

export default App
