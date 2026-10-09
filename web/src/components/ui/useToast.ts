import { createContext, useContext } from 'react'

export type ToastVariant = 'info' | 'success' | 'error'

export interface Toast {
  id: number
  message: string
  variant: ToastVariant
}

export interface ToastContextValue {
  push: (message: string, variant?: ToastVariant) => void
}

//: Split from ``Toast.tsx`` for the same reason as the i18n context: a module
//: exporting both a component and a hook cannot be fast-refreshed, which is what
//: ``react/only-export-components`` reports.
export const ToastContext = createContext<ToastContextValue | null>(null)

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext)
  if (!ctx) {
    throw new Error('useToast must be used within a ToastProvider')
  }
  return ctx
}
