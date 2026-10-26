import React, { createContext, useContext, useState, useEffect } from 'react'

export type UiMode = 'reader' | 'studio'

interface UiModeContextType {
  uiMode: UiMode
  setUiMode: (mode: UiMode) => void
  toggleUiMode: () => void
}

const UiModeContext = createContext<UiModeContextType>({
  uiMode: 'reader',
  setUiMode: () => {},
  toggleUiMode: () => {},
})

export function UiModeProvider({ children }: { children: React.ReactNode }) {
  const [uiMode, setUiModeState] = useState<UiMode>(() => {
    try {
      const stored = localStorage.getItem('ubt_ui_mode') as UiMode | null
      return stored && ['reader', 'studio'].includes(stored) ? stored : 'reader'
    } catch {
      return 'reader'
    }
  })

  useEffect(() => {
    try {
      localStorage.setItem('ubt_ui_mode', uiMode)
    } catch {
      // Ignore private browsing quota errors
    }
  }, [uiMode])

  const setUiMode = (mode: UiMode) => {
    setUiModeState(mode)
  }

  const toggleUiMode = () => {
    setUiModeState((prev) => (prev === 'reader' ? 'studio' : 'reader'))
  }

  return (
    <UiModeContext.Provider value={{ uiMode, setUiMode, toggleUiMode }}>
      {children}
    </UiModeContext.Provider>
  )
}

export function useUiMode() {
  return useContext(UiModeContext)
}
