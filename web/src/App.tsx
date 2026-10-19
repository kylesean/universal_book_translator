import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { lazy, Suspense } from 'react'
import { BrowserRouter, Navigate, Route, Routes, useLocation } from 'react-router-dom'
import { I18nProvider } from '@/i18n/I18nContext'
import { AuthGate } from '@/components/layout/AuthGate'
import { ConsoleLayout } from '@/components/layout/ConsoleLayout'
import { ErrorBoundary } from '@/components/layout/ErrorBoundary'
import { ToastProvider } from '@/components/ui/Toast'

// Route-level code splitting. Every screen used to be a static import, so an
// operator downloaded the whole console (wizard, mission control, review
// workbench, quality gate, language assets, doctor) before the first paint of
// any one of them; the single chunk had grown past 500 kB, which is also where
// Vite starts warning. Each screen is now its own chunk, loaded on navigation.
//
// `/` redirects to `/wizard`, so the wizard's chunk still loads immediately for
// the common case — the split does not add a round trip to the default path.
const NewJobWizard = lazy(() =>
  import('@/views/wizard/NewJobWizard').then((m) => ({ default: m.NewJobWizard }))
)
const MissionControl = lazy(() =>
  import('@/views/mission-control/MissionControl').then((m) => ({ default: m.MissionControl }))
)
const QualityGateView = lazy(() =>
  import('@/views/quality-gate/QualityGateView').then((m) => ({ default: m.QualityGateView }))
)
const LanguageAssetsView = lazy(() =>
  import('@/views/language-assets/LanguageAssetsView').then((m) => ({
    default: m.LanguageAssetsView,
  }))
)
const SystemDoctorView = lazy(() =>
  import('@/views/doctor/SystemDoctorView').then((m) => ({ default: m.SystemDoctorView }))
)
const ReviewWorkbench = lazy(() =>
  import('@/views/review/ReviewWorkbench').then((m) => ({ default: m.ReviewWorkbench }))
)

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

/**
 * Placeholder shown while a route's chunk downloads.
 *
 * Deliberately a quiet, layout-preserving block rather than a spinner: the
 * chunk is same-origin and usually resolves in one frame, and a spinner that
 * flashes for 20ms reads as a glitch. `aria-busy` keeps it honest for
 * assistive tech.
 */
function RouteFallback() {
  return (
    <div
      className="flex-1 flex items-center justify-center p-8"
      aria-busy="true"
      aria-live="polite"
    >
      <div className="h-5 w-5 rounded-full border-2 border-[var(--paper-border)] border-t-[var(--ink-primary)] animate-spin" />
    </div>
  )
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
                  {/* Suspense sits inside the boundary so a failed chunk load
                      surfaces through the error boundary rather than leaving a
                      permanent spinner. */}
                  <Suspense fallback={<RouteFallback />}>
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
                  </Suspense>
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
