import React from 'react'
import { AlertTriangle, RotateCcw } from 'lucide-react'
import { Button } from '@/components/ui/Button'

interface ErrorBoundaryProps {
  children: React.ReactNode
  /** Rendered when the subtree throws; defaults to the console's full-page notice. */
  fallback?: React.ReactNode
}

interface ErrorBoundaryState {
  error: Error | null
}

/**
 * Keeps one view's render error from blanking the whole console.
 *
 * React unmounts the entire tree on an uncaught render error, so before this a
 * single bad report payload or a null field in one screen left the operator
 * staring at a white page — with a 30-minute job still running behind it and no
 * way back. The boundary confines the failure to the view and offers a retry
 * that remounts it (``key`` bump) without reloading the app.
 */
export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error }
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    // No telemetry sink in a local-first console; the console is the record.
    console.error('Console view crashed:', error, info.componentStack)
  }

  private reset = () => {
    this.setState({ error: null })
  }

  render(): React.ReactNode {
    if (this.state.error === null) return this.props.children
    if (this.props.fallback !== undefined) return this.props.fallback

    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <div className="max-w-md space-y-3 text-center">
          <AlertTriangle className="h-8 w-8 mx-auto text-[#b45309]" />
          <h2 className="text-sm font-semibold text-[var(--ink-primary)]">
            This screen failed to render
          </h2>
          <p className="text-xs text-[var(--ink-secondary)] leading-relaxed break-words">
            {this.state.error.message}
          </p>
          <Button onClick={this.reset} variant="secondary" size="sm" className="mx-auto">
            <RotateCcw className="h-3.5 w-3.5 mr-1.5" />
            Retry
          </Button>
        </div>
      </div>
    )
  }
}
