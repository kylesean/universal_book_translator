import React from 'react'
import { AlertTriangle, RotateCcw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { en } from '@/i18n/translations/en'
import { zh } from '@/i18n/translations/zh'

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
 *
 * The notice is read from the i18n tables directly rather than through
 * ``useI18n`` (a class component cannot use hooks) and the language is read
 * defensively: the crash being handled may be the provider's own, and a
 * storage refusal must not turn the fallback screen into a second crash.
 */
export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  private dictionary() {
    try {
      const saved = localStorage.getItem('ubt_lang')
      if (saved === 'zh') return zh.errorBoundary
      if (saved === 'en') return en.errorBoundary
    } catch {
      /* refused storage: fall through to the browser language */
    }
    const browserLang =
      typeof navigator !== 'undefined' ? navigator.language || '' : ''
    return browserLang.toLowerCase().startsWith('zh') ? zh.errorBoundary : en.errorBoundary
  }
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
    const text = this.dictionary()

    return (
      <div className="flex flex-1 items-center justify-center p-8">
        <div className="max-w-md space-y-3 rounded-xl border border-border/80 bg-card p-7 text-center shadow-card">
          <span className="mx-auto flex size-10 items-center justify-center rounded-xl border border-amber-600/25 bg-amber-500/10 text-amber-700 dark:text-amber-400">
            <AlertTriangle className="size-5" />
          </span>
          <h2 className="text-sm font-semibold text-foreground">
            {text.title}
          </h2>
          <p className="break-words text-sm leading-relaxed text-muted-foreground">
            {this.state.error.message}
          </p>
          <Button onClick={this.reset} variant="secondary" size="sm" className="mx-auto">
            <RotateCcw className="mr-1.5 size-3.5" />
            {text.retry}
          </Button>
        </div>
      </div>
    )
  }
}
